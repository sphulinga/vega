#pragma once

#include <any>
#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>

#include "vega/core/partition.hpp"
#include "vega/serialization/serializer.hpp"

namespace vega {

class SparkContext;  // forward – defined in context.hpp

// ---------------------------------------------------------------------------
// RDDBase – type-erased base for every RDD node in the lineage DAG.
//
// Each node knows how to compute a single partition given its id.  The
// result is returned as std::any wrapping a std::vector<ElementType>.
// ---------------------------------------------------------------------------

class RDDBase : public std::enable_shared_from_this<RDDBase> {
public:
    virtual ~RDDBase() = default;

    virtual size_t num_partitions() const = 0;

    // Recursively compute the data for one partition.  The returned any
    // holds a std::vector<T> where T is the element type of this RDD.
    virtual std::any compute_partition(size_t partition_id) const = 0;

    // Compute the number of records in one partition, potentially avoiding
    // full materialization of the partition output for count().
    virtual size_t compute_partition_count(size_t partition_id) const = 0;

    // Parent RDDs this node depends on (empty for source RDDs).
    virtual std::vector<std::shared_ptr<RDDBase>> dependencies() const = 0;

    // Narrow transforms can be pipelined in a single stage.
    virtual bool is_narrow() const { return true; }

    // Called by the execution engine to feed shuffle data from the
    // completed map-side stage into a wide-dependency RDD.
    virtual void prepare_shuffle(const std::vector<std::any>& /*parent_results*/) {}

    // --- P2P shuffle support (type-erased for cluster layer) ---------------

    // Given one map partition's result (any wrapping vector<pair<K,V>>),
    // hash-partition by key and return serialized blocks keyed by target
    // reduce partition id.
    virtual std::vector<std::pair<size_t, ByteBuffer>>
    shuffle_partition(const std::any& /*map_result*/,
                      size_t /*num_reduce_partitions*/) const { return {}; }

    // Receive a serialized shuffle block from a remote map peer.
    virtual void receive_shuffle_data(size_t /*reduce_pid*/,
                                      const ByteBuffer& /*data*/) {}

    // Mark that all shuffle data has been received (called after the
    // MAP_DONE barrier).
    virtual void finalize_shuffle() {}

    // When true, partition pid covers indexed integers [begin, end) without materializing.
    virtual bool partition_indexed_ints(size_t pid,
                                          int64_t& begin,
                                          int64_t& end) const {
        (void)pid;
        begin = end = 0;
        return false;
    }

    // When true, partition pid covers indexed doubles [begin, end) without materializing.
    virtual bool partition_indexed_doubles(size_t pid,
                                           int64_t& begin,
                                           int64_t& end) const {
        (void)pid;
        begin = end = 0;
        return false;
    }

    void set_context(SparkContext* ctx) { ctx_ = ctx; }
    SparkContext* context() const { return ctx_; }

    // True when partition values contain py::object and need the GIL at
    // actor boundaries.
    virtual bool uses_python_runtime() const { return false; }

    // For distributed execution: the ID under which partition data was
    // stored on worker nodes.
    void set_distribution_id(size_t id) { dist_id_ = id; }
    size_t distribution_id() const { return dist_id_; }

protected:
    SparkContext* ctx_ = nullptr;
    size_t dist_id_ = 0;
};

// ---------------------------------------------------------------------------
// RDD<T> – typed, user-facing handle.
//
// Transformations (map / filter / flatMap) are lazy: they build new RDD
// nodes but perform no computation.  Actions (collect / count / reduce /
// foreach) trigger execution through the actor system via SparkContext.
// ---------------------------------------------------------------------------

template<typename T>
class RDD {
public:
    // ----- Transformations (lazy) ------------------------------------------

    template<typename Func>
    auto map(Func&& func) const
        -> RDD<std::invoke_result_t<Func, const T&>>;

    RDD<T> filter(std::function<bool(const T&)> pred) const;

    template<typename Func>
    auto flatMap(Func&& func) const
        -> RDD<typename std::invoke_result_t<Func, const T&>::value_type>;

    // ----- Key-value transforms (T must be pair<K,V>) ----------------------
    // The extra U=T indirection defers member-access into T until the
    // member function template is actually called (SFINAE-friendly).

    template<typename Func, typename U = T,
             typename K = typename U::first_type,
             typename V = typename U::second_type>
    RDD<std::pair<K, V>> reduceByKey(Func&& func,
                                     size_t num_partitions = 0) const;

    template<typename U = T,
             typename K = typename U::first_type,
             typename V = typename U::second_type>
    RDD<std::pair<K, std::vector<V>>> groupByKey(
        size_t num_partitions = 0) const;

    // ----- Actions (trigger execution) -------------------------------------
    // Defined after SparkContext in context.hpp so they can call execute().

    std::vector<T> collect() const;
    size_t count() const;
    T reduce(std::function<T(const T&, const T&)> func) const;
    void foreach(std::function<void(const T&)> func) const;

    // ----- Accessors -------------------------------------------------------

    size_t num_partitions() const { return impl_->num_partitions(); }
    std::shared_ptr<RDDBase> impl() const { return impl_; }

    // Wrap an existing RDDBase node (used by the session server).
    static RDD<T> from_impl(std::shared_ptr<RDDBase> impl) {
        return RDD<T>(std::move(impl));
    }

private:
    template<typename U> friend class RDD;
    friend class SparkContext;
    friend class Cluster;

    std::shared_ptr<RDDBase> impl_;
    explicit RDD(std::shared_ptr<RDDBase> impl) : impl_(std::move(impl)) {}
};

// ===========================================================================
// Concrete RDD implementations
// ===========================================================================

// Source RDD: holds pre-partitioned data in memory.
template<typename T>
class ParallelCollectionRDDImpl : public RDDBase {
public:
    explicit ParallelCollectionRDDImpl(std::vector<Partition<T>> parts)
        : partitions_(std::move(parts)) {}

    size_t num_partitions() const override { return partitions_.size(); }

    std::any compute_partition(size_t pid) const override {
        if (pid >= partitions_.size())
            throw std::out_of_range("partition id out of range");
        return partitions_[pid].data();
    }

    size_t compute_partition_count(size_t pid) const override {
        if (pid >= partitions_.size())
            throw std::out_of_range("partition id out of range");
        return partitions_[pid].data().size();
    }

    std::vector<std::shared_ptr<RDDBase>> dependencies() const override {
        return {};
    }

private:
    std::vector<Partition<T>> partitions_;
};

// map(f): T -> U
template<typename T, typename U>
class MappedRDDImpl : public RDDBase {
public:
    MappedRDDImpl(std::shared_ptr<RDDBase> parent,
                  std::function<U(const T&)> func)
        : parent_(std::move(parent)), func_(std::move(func)) {}

    size_t num_partitions() const override {
        return parent_->num_partitions();
    }

    std::any compute_partition(size_t pid) const override {
        auto input = std::any_cast<std::vector<T>>(
            parent_->compute_partition(pid));
        std::vector<U> output;
        output.reserve(input.size());
        for (const auto& elem : input)
            output.push_back(func_(elem));
        return output;
    }

    size_t compute_partition_count(size_t pid) const override {
        return parent_->compute_partition_count(pid);
    }

    std::vector<std::shared_ptr<RDDBase>> dependencies() const override {
        return {parent_};
    }

private:
    std::shared_ptr<RDDBase> parent_;
    std::function<U(const T&)> func_;
};

// filter(pred): T -> T (elements passing predicate)
template<typename T>
class FilteredRDDImpl : public RDDBase {
public:
    FilteredRDDImpl(std::shared_ptr<RDDBase> parent,
                    std::function<bool(const T&)> pred)
        : parent_(std::move(parent)), pred_(std::move(pred)) {}

    size_t num_partitions() const override {
        return parent_->num_partitions();
    }

    std::any compute_partition(size_t pid) const override {
        auto input = std::any_cast<std::vector<T>>(
            parent_->compute_partition(pid));
        std::vector<T> output;
        output.reserve(input.size());
        for (auto& elem : input) {
            if (pred_(elem))
                output.push_back(std::move(elem));
        }
        return output;
    }

    size_t compute_partition_count(size_t pid) const override {
        auto input = std::any_cast<std::vector<T>>(
            parent_->compute_partition(pid));
        size_t out = 0;
        for (const auto& elem : input) {
            if (pred_(elem)) ++out;
        }
        return out;
    }

    std::vector<std::shared_ptr<RDDBase>> dependencies() const override {
        return {parent_};
    }

private:
    std::shared_ptr<RDDBase> parent_;
    std::function<bool(const T&)> pred_;
};

// flatMap(f): T -> vector<U>  (each input element produces 0..N outputs)
template<typename T, typename U>
class FlatMappedRDDImpl : public RDDBase {
public:
    FlatMappedRDDImpl(std::shared_ptr<RDDBase> parent,
                      std::function<std::vector<U>(const T&)> func)
        : parent_(std::move(parent)), func_(std::move(func)) {}

    size_t num_partitions() const override {
        return parent_->num_partitions();
    }

    std::any compute_partition(size_t pid) const override {
        auto input = std::any_cast<std::vector<T>>(
            parent_->compute_partition(pid));
        std::vector<U> output;
        output.reserve(input.size());
        for (const auto& elem : input) {
            auto mapped = func_(elem);
            output.insert(output.end(),
                          std::make_move_iterator(mapped.begin()),
                          std::make_move_iterator(mapped.end()));
        }
        return output;
    }

    size_t compute_partition_count(size_t pid) const override {
        auto input = std::any_cast<std::vector<T>>(
            parent_->compute_partition(pid));
        size_t out = 0;
        for (const auto& elem : input) {
            auto mapped = func_(elem);
            out += mapped.size();
        }
        return out;
    }

    std::vector<std::shared_ptr<RDDBase>> dependencies() const override {
        return {parent_};
    }

private:
    std::shared_ptr<RDDBase> parent_;
    std::function<std::vector<U>(const T&)> func_;
};

// ===========================================================================
// RDD<T> transformation implementations (lazy – no SparkContext needed)
// ===========================================================================

template<typename T>
template<typename Func>
auto RDD<T>::map(Func&& func) const
    -> RDD<std::invoke_result_t<Func, const T&>>
{
    using U = std::invoke_result_t<Func, const T&>;
    auto mapped = std::make_shared<MappedRDDImpl<T, U>>(
        impl_, std::function<U(const T&)>(std::forward<Func>(func)));
    mapped->set_context(impl_->context());
    return RDD<U>(std::move(mapped));
}

template<typename T>
RDD<T> RDD<T>::filter(std::function<bool(const T&)> pred) const {
    auto filtered = std::make_shared<FilteredRDDImpl<T>>(
        impl_, std::move(pred));
    filtered->set_context(impl_->context());
    return RDD<T>(std::move(filtered));
}

template<typename T>
template<typename Func>
auto RDD<T>::flatMap(Func&& func) const
    -> RDD<typename std::invoke_result_t<Func, const T&>::value_type>
{
    using Container = std::invoke_result_t<Func, const T&>;
    using U = typename Container::value_type;
    auto fm = std::make_shared<FlatMappedRDDImpl<T, U>>(
        impl_,
        std::function<std::vector<U>(const T&)>(std::forward<Func>(func)));
    fm->set_context(impl_->context());
    return RDD<U>(std::move(fm));
}

// ===========================================================================
// RDD<T> key-value transformation implementations (lazy, need shuffled_rdd.hpp
// which is included from context.hpp after the full types are available).
// The definitions below are declared here and defined in context.hpp.
// ===========================================================================

} // namespace vega
