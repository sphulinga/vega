#pragma once

#include <any>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "vega/core/rdd.hpp"
#include "vega/serialization/serializer.hpp"

namespace vega {

// ---------------------------------------------------------------------------
// ReduceByKeyRDDImpl<K, V>
//
// Wide (shuffle) dependency: the parent produces vector<pair<K,V>>.
// Supports two modes:
//   1. Single-node: prepare_shuffle() with in-process map results.
//   2. P2P cluster: shuffle_partition() + receive_shuffle_data() +
//      finalize_shuffle() for decentralized execution.
// ---------------------------------------------------------------------------

template<typename K, typename V>
class ReduceByKeyRDDImpl : public RDDBase {
public:
    ReduceByKeyRDDImpl(std::shared_ptr<RDDBase> parent,
                       std::function<V(const V&, const V&)> combine,
                       size_t num_partitions)
        : parent_(std::move(parent))
        , combine_(std::move(combine))
        , num_partitions_(num_partitions)
    {
        buckets_.resize(num_partitions_);
    }

    size_t num_partitions() const override { return num_partitions_; }
    bool is_narrow() const override { return false; }

    std::vector<std::shared_ptr<RDDBase>> dependencies() const override {
        return {parent_};
    }

    // --- Single-node path (JobActor) ---------------------------------------

    void prepare_shuffle(const std::vector<std::any>& parent_results) override {
        std::lock_guard<std::mutex> lock(mu_);
        for (auto& bucket : buckets_)
            bucket.clear();
        shuffle_ready_ = false;
        for (auto& pr : parent_results) {
            auto data =
                std::any_cast<std::vector<std::pair<K, V>>>(pr);
            for (auto& kv : data) {
                size_t bucket = std::hash<K>{}(kv.first) % num_partitions_;
                buckets_[bucket].push_back(std::move(kv));
            }
        }
        shuffle_ready_ = true;
    }

    // --- P2P cluster path --------------------------------------------------

    // Hash-partition one map partition's output and serialize each bucket.
    std::vector<std::pair<size_t, ByteBuffer>>
    shuffle_partition(const std::any& map_result,
                      size_t num_reduce) const override {
        auto data =
            std::any_cast<std::vector<std::pair<K, V>>>(map_result);

        std::unordered_map<size_t, std::vector<std::pair<K, V>>> tmp;
        for (auto& kv : data) {
            size_t bucket = std::hash<K>{}(kv.first) % num_reduce;
            tmp[bucket].push_back(std::move(kv));
        }

        std::vector<std::pair<size_t, ByteBuffer>> out;
        out.reserve(tmp.size());
        for (auto& [pid, pairs] : tmp) {
            out.emplace_back(pid,
                vega::serialize<std::vector<std::pair<K, V>>>(pairs));
        }
        return out;
    }

    // Receive a serialized shuffle block from a remote peer.
    void receive_shuffle_data(size_t reduce_pid,
                              const ByteBuffer& data) override {
        auto pairs =
            vega::deserialize<std::vector<std::pair<K, V>>>(data);
        std::lock_guard<std::mutex> lock(mu_);
        for (auto& p : pairs)
            buckets_[reduce_pid].push_back(std::move(p));
    }

    void finalize_shuffle() override {
        std::lock_guard<std::mutex> lock(mu_);
        shuffle_ready_ = true;
    }

    // --- Compute -----------------------------------------------------------

    std::any compute_partition(size_t pid) const override {
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (!shuffle_ready_)
                throw std::runtime_error(
                    "reduceByKey: shuffle data not prepared");
        }

        std::unordered_map<K, V> combined;
        for (auto& [k, v] : buckets_[pid]) {
            auto it = combined.find(k);
            if (it != combined.end())
                it->second = combine_(it->second, v);
            else
                combined.emplace(k, v);
        }

        std::vector<std::pair<K, V>> result;
        result.reserve(combined.size());
        for (auto& [k, v] : combined)
            result.emplace_back(k, std::move(v));
        return result;
    }

    size_t compute_partition_count(size_t pid) const override {
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (!shuffle_ready_)
                throw std::runtime_error(
                    "reduceByKey: shuffle data not prepared");
        }

        std::unordered_set<K> keys;
        keys.reserve(buckets_[pid].size());
        for (const auto& [k, _] : buckets_[pid])
            keys.insert(k);
        return keys.size();
    }

private:
    std::shared_ptr<RDDBase> parent_;
    std::function<V(const V&, const V&)> combine_;
    size_t num_partitions_;

    mutable std::mutex mu_;
    mutable bool shuffle_ready_ = false;
    mutable std::vector<std::vector<std::pair<K, V>>> buckets_;
};

// ---------------------------------------------------------------------------
// GroupByKeyRDDImpl<K, V>
// ---------------------------------------------------------------------------

template<typename K, typename V>
class GroupByKeyRDDImpl : public RDDBase {
public:
    GroupByKeyRDDImpl(std::shared_ptr<RDDBase> parent,
                      size_t num_partitions)
        : parent_(std::move(parent))
        , num_partitions_(num_partitions)
    {
        buckets_.resize(num_partitions_);
    }

    size_t num_partitions() const override { return num_partitions_; }
    bool is_narrow() const override { return false; }

    std::vector<std::shared_ptr<RDDBase>> dependencies() const override {
        return {parent_};
    }

    void prepare_shuffle(const std::vector<std::any>& parent_results) override {
        std::lock_guard<std::mutex> lock(mu_);
        for (auto& bucket : buckets_)
            bucket.clear();
        shuffle_ready_ = false;
        for (auto& pr : parent_results) {
            auto data =
                std::any_cast<std::vector<std::pair<K, V>>>(pr);
            for (auto& kv : data) {
                size_t bucket = std::hash<K>{}(kv.first) % num_partitions_;
                buckets_[bucket].push_back(std::move(kv));
            }
        }
        shuffle_ready_ = true;
    }

    std::vector<std::pair<size_t, ByteBuffer>>
    shuffle_partition(const std::any& map_result,
                      size_t num_reduce) const override {
        auto data =
            std::any_cast<std::vector<std::pair<K, V>>>(map_result);

        std::unordered_map<size_t, std::vector<std::pair<K, V>>> tmp;
        for (auto& kv : data) {
            size_t bucket = std::hash<K>{}(kv.first) % num_reduce;
            tmp[bucket].push_back(std::move(kv));
        }

        std::vector<std::pair<size_t, ByteBuffer>> out;
        out.reserve(tmp.size());
        for (auto& [pid, pairs] : tmp) {
            out.emplace_back(pid,
                vega::serialize<std::vector<std::pair<K, V>>>(pairs));
        }
        return out;
    }

    void receive_shuffle_data(size_t reduce_pid,
                              const ByteBuffer& data) override {
        auto pairs =
            vega::deserialize<std::vector<std::pair<K, V>>>(data);
        std::lock_guard<std::mutex> lock(mu_);
        for (auto& p : pairs)
            buckets_[reduce_pid].push_back(std::move(p));
    }

    void finalize_shuffle() override {
        std::lock_guard<std::mutex> lock(mu_);
        shuffle_ready_ = true;
    }

    std::any compute_partition(size_t pid) const override {
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (!shuffle_ready_)
                throw std::runtime_error(
                    "groupByKey: shuffle data not prepared");
        }

        std::unordered_map<K, std::vector<V>> grouped;
        for (const auto& [k, v] : buckets_[pid])
            grouped[k].push_back(v);

        std::vector<std::pair<K, std::vector<V>>> result;
        result.reserve(grouped.size());
        for (auto& [k, vs] : grouped)
            result.emplace_back(k, std::move(vs));
        return result;
    }

    size_t compute_partition_count(size_t pid) const override {
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (!shuffle_ready_)
                throw std::runtime_error(
                    "groupByKey: shuffle data not prepared");
        }

        std::unordered_map<K, size_t> seen;
        for (const auto& [k, _] : buckets_[pid])
            seen[k] = 1;
        return seen.size();
    }

private:
    std::shared_ptr<RDDBase> parent_;
    size_t num_partitions_;

    mutable std::mutex mu_;
    mutable bool shuffle_ready_ = false;
    mutable std::vector<std::vector<std::pair<K, V>>> buckets_;
};

} // namespace vega
