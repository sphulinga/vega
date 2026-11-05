#pragma once

#include <algorithm>
#include <any>
#include <atomic>
#include <cstddef>
#include <future>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include "vega/core/rdd.hpp"
#include "vega/core/shuffled_rdd.hpp"
#include "vega/actors/actor_system.hpp"
#include "vega/actors/messages.hpp"
#include "vega/actors/worker_actor.hpp"
#include "vega/actors/job_actor.hpp"
#include "vega/transport/local_transport.hpp"

namespace vega {

// ---------------------------------------------------------------------------
// SparkContext – entry point for all Vega operations.
//
// Creates the actor system, spawns WorkerActors and a JobActor, and provides
// the parallelize() factory plus the internal execute() method that bridges
// the synchronous user API with the asynchronous actor pipeline.
// ---------------------------------------------------------------------------

class SparkContext {
public:
    explicit SparkContext(size_t num_workers = 0)
        : num_workers_(num_workers > 0
                       ? num_workers
                       : std::max(1u, std::thread::hardware_concurrency()))
        , system_(std::make_unique<actor::ActorSystem>(num_workers_))
        , transport_(std::make_unique<LocalTransport>())
    {
        // Spawn worker actors.
        worker_ids_.reserve(num_workers_);
        for (size_t i = 0; i < num_workers_; ++i)
            worker_ids_.push_back(system_->spawn<WorkerActor>());

        // Spawn the single job actor that knows about all workers.
        job_actor_id_ = system_->spawn<JobActor>(worker_ids_);

        system_->start();
    }

    ~SparkContext() { stop(); }

    // No copy / move.
    SparkContext(const SparkContext&) = delete;
    SparkContext& operator=(const SparkContext&) = delete;

    // Create an RDD from an in-memory collection.
    template<typename T>
    RDD<T> parallelize(const std::vector<T>& data,
                       size_t num_partitions = 0);

    // Submit an RDD for execution and return per-partition results.
    // Blocks until the job completes.
    std::vector<std::any> execute(std::shared_ptr<RDDBase> rdd) {
        return execute_partitions(std::move(rdd), {});
    }

    // Execute only the given final-stage partitions (empty = all). Upstream
    // shuffles are still fully materialized (or reused if already ready).
    // Results are returned in the order of `partitions`.
    std::vector<std::any> execute_partitions(std::shared_ptr<RDDBase> rdd,
                                             std::vector<size_t> partitions) {
        return submit(std::move(rdd), msg::JobAction::COMPUTE_PARTITION,
                      std::move(partitions));
    }

    size_t execute_count(std::shared_ptr<RDDBase> rdd) {
        auto partition_results =
            submit(std::move(rdd), msg::JobAction::COUNT_PARTITION, {});
        size_t total = 0;
        for (auto& pr : partition_results) {
            total += std::any_cast<size_t>(std::move(pr));
        }
        return total;
    }

    // Per-partition record counts for a subset of partitions.
    std::vector<size_t> execute_count_partitions(std::shared_ptr<RDDBase> rdd,
                                                 std::vector<size_t> partitions) {
        auto partition_results = submit(std::move(rdd),
                                        msg::JobAction::COUNT_PARTITION,
                                        std::move(partitions));
        std::vector<size_t> out;
        out.reserve(partition_results.size());
        for (auto& pr : partition_results)
            out.push_back(std::any_cast<size_t>(std::move(pr)));
        return out;
    }

    void stop() {
        if (system_) system_->stop();
    }

    size_t num_workers() const { return num_workers_; }

private:
    std::vector<std::any> submit(std::shared_ptr<RDDBase> rdd,
                                 msg::JobAction action,
                                 std::vector<size_t> partitions) {
        auto promise =
            std::make_shared<std::promise<std::vector<std::any>>>();
        auto future = promise->get_future();

        size_t jid = next_job_id_.fetch_add(1, std::memory_order_relaxed);

        msg::SubmitJob job{jid, std::move(rdd), std::move(promise), action};
        job.partitions = std::move(partitions);
        system_->send<msg::SubmitJob>(job_actor_id_, std::move(job));

        return future.get();
    }

    size_t num_workers_;
    std::unique_ptr<actor::ActorSystem> system_;
    std::unique_ptr<LocalTransport> transport_;

    std::vector<actor::ActorId> worker_ids_;
    actor::ActorId job_actor_id_ = actor::INVALID_ACTOR;
    std::atomic<size_t> next_job_id_{1};
};

// ===========================================================================
// SparkContext template implementations
// ===========================================================================

template<typename T>
RDD<T> SparkContext::parallelize(const std::vector<T>& data,
                                 size_t num_partitions) {
    if (num_partitions == 0)
        num_partitions = num_workers_;

    std::vector<Partition<T>> parts;
    size_t total = data.size();
    size_t chunk = (total + num_partitions - 1) / num_partitions;

    for (size_t i = 0; i < num_partitions; ++i) {
        size_t begin = i * chunk;
        size_t end   = std::min(begin + chunk, total);
        if (begin >= total) break;
        parts.emplace_back(i,
            std::vector<T>(data.begin() + begin, data.begin() + end));
    }

    auto impl = std::make_shared<ParallelCollectionRDDImpl<T>>(
        std::move(parts));
    impl->set_context(this);
    return RDD<T>(std::move(impl));
}

// ===========================================================================
// RDD<T> action implementations (need SparkContext)
// ===========================================================================

template<typename T>
std::vector<T> RDD<T>::collect() const {
    auto* ctx = impl_->context();
    if (!ctx) throw std::runtime_error("RDD has no SparkContext");

    auto partition_results = ctx->execute(impl_);

    std::vector<T> output;
    for (auto& pr : partition_results) {
        auto vec = std::any_cast<std::vector<T>>(std::move(pr));
        output.insert(output.end(),
                      std::make_move_iterator(vec.begin()),
                      std::make_move_iterator(vec.end()));
    }
    return output;
}

template<typename T>
size_t RDD<T>::count() const {
    auto* ctx = impl_->context();
    if (!ctx) throw std::runtime_error("RDD has no SparkContext");
    return ctx->execute_count(impl_);
}

template<typename T>
T RDD<T>::reduce(std::function<T(const T&, const T&)> func) const {
    auto* ctx = impl_->context();
    if (!ctx) throw std::runtime_error("RDD has no SparkContext");

    auto partition_results = ctx->execute(impl_);

    std::optional<T> acc;
    for (auto& pr : partition_results) {
        auto vec = std::any_cast<std::vector<T>>(std::move(pr));
        for (auto& elem : vec) {
            if (!acc.has_value())
                acc = std::move(elem);
            else
                acc = func(*acc, elem);
        }
    }
    if (!acc.has_value())
        throw std::runtime_error("cannot reduce an empty RDD");
    return std::move(*acc);
}

template<typename T>
void RDD<T>::foreach(std::function<void(const T&)> func) const {
    auto* ctx = impl_->context();
    if (!ctx) throw std::runtime_error("RDD has no SparkContext");

    auto partition_results = ctx->execute(impl_);

    for (auto& pr : partition_results) {
        auto vec = std::any_cast<std::vector<T>>(std::move(pr));
        for (const auto& elem : vec)
            func(elem);
    }
}

// ===========================================================================
// RDD<T> key-value transformation implementations (need ShuffledRDD types)
// ===========================================================================

template<typename T>
template<typename Func, typename U, typename K, typename V>
RDD<std::pair<K, V>> RDD<T>::reduceByKey(Func&& func,
                                          size_t num_partitions) const {
    if (num_partitions == 0)
        num_partitions = impl_->num_partitions();
    auto rdd = std::make_shared<ReduceByKeyRDDImpl<K, V>>(
        impl_,
        std::function<V(const V&, const V&)>(std::forward<Func>(func)),
        num_partitions);
    rdd->set_context(impl_->context());
    return RDD<std::pair<K, V>>(std::move(rdd));
}

template<typename T>
template<typename U, typename K, typename V>
RDD<std::pair<K, std::vector<V>>> RDD<T>::groupByKey(
        size_t num_partitions) const {
    if (num_partitions == 0)
        num_partitions = impl_->num_partitions();
    auto rdd = std::make_shared<GroupByKeyRDDImpl<K, V>>(
        impl_, num_partitions);
    rdd->set_context(impl_->context());
    return RDD<std::pair<K, std::vector<V>>>(std::move(rdd));
}

} // namespace vega
