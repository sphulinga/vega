#pragma once

#include <algorithm>
#include <any>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "vega/cluster/gossip.hpp"
#include "vega/cluster/node.hpp"
#include "vega/core/rdd.hpp"
#include "vega/core/partition.hpp"
#include "vega/core/shuffled_rdd.hpp"
#include "vega/execution/dag.hpp"
#include "vega/network/endpoint.hpp"
#include "vega/network/message_codec.hpp"
#include "vega/network/message_router.hpp"
#include "vega/serialization/serializer.hpp"

namespace vega {

using cluster::NodeId;
using cluster::NodeInfo;
using cluster::NodeRole;
using cluster::GossipProtocol;

// ---------------------------------------------------------------------------
// ClusterConfig
// ---------------------------------------------------------------------------

struct ClusterConfig {
    std::string bind_host = "0.0.0.0";
    uint16_t bind_port    = 9000;
    std::string node_id;
    std::vector<net::Endpoint> seed_nodes;
    size_t expected_peers = 1;

    static ClusterConfig from_args(int argc, char** argv) {
        ClusterConfig cfg;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--bind" && i + 1 < argc) {
                auto ep = net::Endpoint::parse(argv[++i]);
                cfg.bind_host = ep.host;
                cfg.bind_port = ep.port;
            }
            else if (arg == "--seeds" && i + 1 < argc) {
                std::string seeds = argv[++i];
                size_t pos = 0;
                while (pos < seeds.size()) {
                    auto comma = seeds.find(',', pos);
                    auto token = seeds.substr(
                        pos, comma == std::string::npos
                                 ? std::string::npos : comma - pos);
                    if (!token.empty())
                        cfg.seed_nodes.push_back(
                            net::Endpoint::parse(token));
                    if (comma == std::string::npos) break;
                    pos = comma + 1;
                }
            }
            else if (arg == "--id" && i + 1 < argc)
                cfg.node_id = argv[++i];
            else if (arg == "--expect-peers" && i + 1 < argc)
                cfg.expected_peers =
                    static_cast<size_t>(std::stoi(argv[++i]));
            // Legacy compat: ignore --driver/--worker/--expect-workers
            else if (arg == "--local") { /* handled by caller */ }
        }
        if (cfg.node_id.empty()) {
            cfg.node_id = "peer-" + std::to_string(cfg.bind_port);
        }
        return cfg;
    }
};

// ---------------------------------------------------------------------------
// Cluster -- fully decentralized, peer-to-peer execution.
//
// Every node is an equal peer.  There is no driver or central scheduler.
// Peers discover each other via gossip.  Execution uses a BSP model:
// all peers participate in every job.
//
// Partition ownership: partition P → peer[ P % num_peers ].
// Peers are sorted by node_id to ensure consistent ordering across nodes.
//
// Execution flow for collect():
//   1. Each peer determines its owned partitions.
//   2. Narrow transforms: computed locally.
//   3. Shuffle: map results hash-partitioned and sent directly to target
//      peers (P2P).  MAP_DONE barrier ensures all blocks have arrived.
//   4. Reduce: computed locally from received shuffle blocks.
//   5. Results gathered to the requesting peer via allgather.
// ---------------------------------------------------------------------------

class Cluster {
public:
    explicit Cluster(const ClusterConfig& cfg)
        : cfg_(cfg)
        , router_(cfg.bind_host, cfg.bind_port)
    {
        NodeInfo self;
        self.id = cfg_.node_id;
        self.endpoint = {
            (cfg_.bind_host == "0.0.0.0" || cfg_.bind_host.empty())
                ? "127.0.0.1" : cfg_.bind_host,
            cfg_.bind_port};
        self.role = NodeRole::WORKER;   // all peers are equal

        GossipProtocol::Config gcfg;
        gcfg.seed_nodes = cfg_.seed_nodes;
        gcfg.gossip_interval = std::chrono::milliseconds(500);
        gcfg.dead_timeout = std::chrono::milliseconds(10000);

        gossip_ = std::make_unique<GossipProtocol>(
            self, gcfg,
            [this](const NodeInfo& ni) { on_node_join(ni); },
            [this](const NodeId& id) { on_node_leave(id); });

        register_handlers();
        router_.start();
        gossip_->start();
    }

    static Cluster init(int argc, char** argv) {
        return Cluster(ClusterConfig::from_args(argc, argv));
    }

    ~Cluster() { stop(); }

    const ClusterConfig& config() const { return cfg_; }
    const NodeId& node_id() const { return cfg_.node_id; }

    bool wait_for_peers(
            std::chrono::seconds timeout = std::chrono::seconds(30)) {
        auto deadline = std::chrono::steady_clock::now() + timeout;
        while (std::chrono::steady_clock::now() < deadline) {
            if (gossip_->live_count() >= cfg_.expected_peers)
                return true;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        return gossip_->live_count() >= cfg_.expected_peers;
    }

    // Sorted list of all live peers (deterministic across all nodes).
    std::vector<NodeInfo> peers() const {
        auto nodes = gossip_->live_nodes();
        std::sort(nodes.begin(), nodes.end(),
                  [](auto& a, auto& b) { return a.id < b.id; });
        return nodes;
    }

    size_t num_peers() const { return gossip_->live_count(); }

    // My index in the sorted peer list.
    size_t my_index() const {
        auto p = peers();
        for (size_t i = 0; i < p.size(); ++i)
            if (p[i].id == cfg_.node_id) return i;
        return 0;
    }

    // Partitions owned by this peer.
    std::vector<size_t> my_partitions(size_t total_partitions) const {
        auto idx = my_index();
        auto np = num_peers();
        std::vector<size_t> out;
        for (size_t p = 0; p < total_partitions; ++p)
            if (p % np == idx) out.push_back(p);
        return out;
    }

    // Which peer owns a given partition.
    NodeInfo owner_of(size_t partition_id) const {
        auto p = peers();
        return p[partition_id % p.size()];
    }

    // -----------------------------------------------------------------------
    // scatter: distribute data from one peer to all peers.
    // Only the peer that HAS data calls this with non-empty data.
    // -----------------------------------------------------------------------

    template<typename T>
    RDD<T> scatter(const std::vector<T>& data,
                   size_t num_partitions = 0);

    // -----------------------------------------------------------------------
    // collect: all peers participate in distributed execution.
    // Returns the complete result on this peer.
    // -----------------------------------------------------------------------

    template<typename T>
    std::vector<T> collect(const RDD<T>& rdd);

    void stop() {
        stopped_.store(true);
        gossip_->stop();
        router_.stop();
    }

private:
    ClusterConfig cfg_;
    net::MessageRouter router_;
    std::unique_ptr<GossipProtocol> gossip_;
    std::atomic<bool> stopped_{false};
    std::atomic<size_t> next_job_id_{1};

    // --- Shuffle state (per job) -------------------------------------------

    struct ShuffleBarrier {
        std::mutex mu;
        std::condition_variable cv;
        std::unordered_set<std::string> done_peers;
        size_t expected = 0;
    };
    std::mutex barriers_mu_;
    std::unordered_map<size_t, std::shared_ptr<ShuffleBarrier>> barriers_;

    // --- Result gathering state (per job) ----------------------------------

    struct GatherState {
        std::mutex mu;
        std::condition_variable cv;
        std::unordered_map<size_t, ByteBuffer> blocks; // pid → data
        size_t expected = 0;
        size_t received = 0;
    };
    std::mutex gather_mu_;
    std::unordered_map<size_t, std::shared_ptr<GatherState>> gathers_;

    // --- Message handlers --------------------------------------------------

    void register_handlers() {
        router_.on(net::NetMsgType::GOSSIP_PING,
            [this](const std::string&, const ByteBuffer& body) {
                gossip_->handle_gossip_ping(body);
            });

        router_.on(net::NetMsgType::PARTITION_DATA,
            [this](const std::string& s, const ByteBuffer& b) {
                handle_partition_data(s, b);
            });

        router_.on(net::NetMsgType::SHUFFLE_BLOCK,
            [this](const std::string& s, const ByteBuffer& b) {
                handle_shuffle_block(s, b);
            });

        router_.on(net::NetMsgType::SHUFFLE_MAP_DONE,
            [this](const std::string& s, const ByteBuffer& b) {
                handle_map_done(s, b);
            });

        router_.on(net::NetMsgType::RESULT_BLOCK,
            [this](const std::string& s, const ByteBuffer& b) {
                handle_result_block(s, b);
            });
    }

    void on_node_join(const NodeInfo& ni) {
        std::cerr << "[cluster] peer joined: " << ni.id
                  << " at " << ni.endpoint.to_string() << "\n";
    }
    void on_node_leave(const NodeId& id) {
        std::cerr << "[cluster] peer left: " << id << "\n";
    }

    // -- Partition data receive (for scatter) --------------------------------

    std::mutex local_data_mu_;
    std::unordered_map<std::string, ByteBuffer> local_partitions_;

    void handle_partition_data(const std::string&, const ByteBuffer& body) {
        size_t off = 0;
        auto key = Serializer<std::string>::deserialize(body, off);
        ByteBuffer data(body.begin() + off, body.end());
        std::lock_guard<std::mutex> lock(local_data_mu_);
        local_partitions_[key] = std::move(data);
    }

    // -- Shuffle block receive (P2P) ----------------------------------------

    // Active shuffle RDD for the current job (set before shuffle starts).
    std::mutex active_shuffle_mu_;
    std::unordered_map<size_t, std::shared_ptr<RDDBase>> active_shuffle_rdds_;

    void handle_shuffle_block(const std::string&, const ByteBuffer& body) {
        size_t off = 0;
        auto job_id = read_pod<uint64_t>(body, off);
        auto reduce_pid = read_pod<uint64_t>(body, off);
        ByteBuffer block(body.begin() + off, body.end());

        std::shared_ptr<RDDBase> shuffle_rdd;
        {
            std::lock_guard<std::mutex> lock(active_shuffle_mu_);
            auto it = active_shuffle_rdds_.find(job_id);
            if (it != active_shuffle_rdds_.end())
                shuffle_rdd = it->second;
        }
        if (shuffle_rdd) {
            shuffle_rdd->receive_shuffle_data(
                static_cast<size_t>(reduce_pid), block);
        }
    }

    // -- MAP_DONE barrier ----------------------------------------------------

    void handle_map_done(const std::string& sender, const ByteBuffer& body) {
        size_t off = 0;
        auto job_id = read_pod<uint64_t>(body, off);

        std::shared_ptr<ShuffleBarrier> barrier;
        {
            std::lock_guard<std::mutex> lock(barriers_mu_);
            auto it = barriers_.find(job_id);
            if (it == barriers_.end()) return;
            barrier = it->second;
        }
        {
            std::lock_guard<std::mutex> lock(barrier->mu);
            barrier->done_peers.insert(sender);
            barrier->cv.notify_all();
        }
    }

    void broadcast_map_done(size_t job_id) {
        ByteBuffer body;
        write_pod<uint64_t>(body, static_cast<uint64_t>(job_id));
        for (auto& peer : peers()) {
            if (peer.id == cfg_.node_id) continue;
            router_.send(peer.endpoint,
                         net::NetMsgType::SHUFFLE_MAP_DONE,
                         cfg_.node_id, body);
        }
        // Also mark self as done.
        {
            std::lock_guard<std::mutex> lock(barriers_mu_);
            auto it = barriers_.find(job_id);
            if (it != barriers_.end()) {
                std::lock_guard<std::mutex> bl(it->second->mu);
                it->second->done_peers.insert(cfg_.node_id);
                it->second->cv.notify_all();
            }
        }
    }

    void wait_for_map_done(size_t job_id, size_t expected) {
        std::shared_ptr<ShuffleBarrier> barrier;
        {
            std::lock_guard<std::mutex> lock(barriers_mu_);
            barrier = barriers_[job_id];
        }
        barrier->expected = expected;

        std::unique_lock<std::mutex> lock(barrier->mu);
        barrier->cv.wait(lock, [&] {
            return barrier->done_peers.size() >= expected;
        });
    }

    // -- Result gather -------------------------------------------------------

    void handle_result_block(const std::string&, const ByteBuffer& body) {
        size_t off = 0;
        auto job_id = read_pod<uint64_t>(body, off);
        auto partition_id = read_pod<uint64_t>(body, off);
        ByteBuffer data(body.begin() + off, body.end());

        std::shared_ptr<GatherState> gs;
        {
            std::lock_guard<std::mutex> lock(gather_mu_);
            auto it = gathers_.find(job_id);
            if (it == gathers_.end()) return;
            gs = it->second;
        }
        {
            std::lock_guard<std::mutex> lock(gs->mu);
            gs->blocks[static_cast<size_t>(partition_id)] = std::move(data);
            gs->received++;
            gs->cv.notify_all();
        }
    }

    void send_result_to_all_peers(
            size_t job_id, size_t partition_id, const ByteBuffer& data) {
        ByteBuffer body;
        write_pod<uint64_t>(body, static_cast<uint64_t>(job_id));
        write_pod<uint64_t>(body, static_cast<uint64_t>(partition_id));
        body.insert(body.end(), data.begin(), data.end());

        for (auto& peer : peers()) {
            if (peer.id == cfg_.node_id) continue;
            router_.send(peer.endpoint, net::NetMsgType::RESULT_BLOCK,
                         cfg_.node_id, body);
        }
    }

    // -----------------------------------------------------------------------
    // P2P execution engine
    // -----------------------------------------------------------------------

    template<typename T>
    std::vector<T> execute_narrow(
            const ExecutionPlan& plan, const RDD<T>& rdd, size_t job_id);

    template<typename T>
    std::vector<T> execute_with_shuffle(
            const ExecutionPlan& plan, const RDD<T>& rdd, size_t job_id);
};

// ===========================================================================
// Template implementations
// ===========================================================================

template<typename T>
RDD<T> Cluster::scatter(const std::vector<T>& data,
                         size_t num_partitions) {
    auto np = num_peers();
    if (num_partitions == 0)
        num_partitions = np;

    // Partition the data.
    std::vector<Partition<T>> all_parts(num_partitions, Partition<T>(0, {}));
    size_t total = data.size();
    size_t chunk = total > 0
                   ? (total + num_partitions - 1) / num_partitions : 0;

    for (size_t i = 0; i < num_partitions && i * chunk < total; ++i) {
        size_t begin = i * chunk;
        size_t end   = std::min(begin + chunk, total);
        all_parts[i] = Partition<T>(i,
            std::vector<T>(data.begin() + begin, data.begin() + end));
    }

    // Send each partition directly to the owning peer (P2P).
    auto sorted_peers = peers();
    size_t dist_id = next_job_id_.fetch_add(1, std::memory_order_relaxed);

    for (size_t p = 0; p < num_partitions; ++p) {
        auto& owner = sorted_peers[p % sorted_peers.size()];
        std::string key = std::to_string(dist_id) + ":"
                        + std::to_string(p);

        if (owner.id == cfg_.node_id) {
            // Local partition -- store directly.
            ByteBuffer buf =
                vega::serialize<std::vector<T>>(all_parts[p].data());
            std::lock_guard<std::mutex> lock(local_data_mu_);
            local_partitions_[key] = std::move(buf);
        } else if (!all_parts[p].data().empty()) {
            ByteBuffer body;
            Serializer<std::string>::serialize(key, body);
            Serializer<std::vector<T>>::serialize(
                all_parts[p].data(), body);
            router_.send(owner.endpoint,
                         net::NetMsgType::PARTITION_DATA,
                         cfg_.node_id, body);
        }
    }

    // Build the local RDD with correct global partition layout.
    auto impl = std::make_shared<ParallelCollectionRDDImpl<T>>(
        std::move(all_parts));
    impl->set_distribution_id(dist_id);
    return RDD<T>(std::move(impl));
}

template<typename T>
std::vector<T> Cluster::collect(const RDD<T>& rdd) {
    size_t job_id = next_job_id_.fetch_add(1, std::memory_order_relaxed);
    auto plan = DAGBuilder::build(job_id, rdd.impl());

    if (plan.stages.size() <= 1)
        return execute_narrow<T>(plan, rdd, job_id);
    else
        return execute_with_shuffle<T>(plan, rdd, job_id);
}

template<typename T>
std::vector<T> Cluster::execute_narrow(
        const ExecutionPlan& plan, const RDD<T>& rdd, size_t job_id) {
    size_t total_parts = rdd.num_partitions();
    auto my_pids = my_partitions(total_parts);
    size_t np = num_peers();

    // Set up gather state for results from other peers.
    auto gs = std::make_shared<GatherState>();
    gs->expected = total_parts;
    gs->received = 0;
    {
        std::lock_guard<std::mutex> lock(gather_mu_);
        gathers_[job_id] = gs;
    }

    // Compute my local partitions and broadcast results.
    for (auto pid : my_pids) {
        auto result = rdd.impl()->compute_partition(pid);
        auto vec = std::any_cast<std::vector<T>>(result);
        auto buf = vega::serialize<std::vector<T>>(vec);

        // Store locally.
        {
            std::lock_guard<std::mutex> lock(gs->mu);
            gs->blocks[pid] = buf;
            gs->received++;
        }

        // Send to all other peers (allgather).
        send_result_to_all_peers(job_id, pid, buf);
    }

    // Wait for results from all other peers.
    {
        std::unique_lock<std::mutex> lock(gs->mu);
        gs->cv.wait(lock, [&] {
            return gs->received >= gs->expected;
        });
    }

    // Merge all partition results in order.
    std::vector<T> output;
    {
        std::lock_guard<std::mutex> lock(gs->mu);
        for (size_t p = 0; p < total_parts; ++p) {
            auto it = gs->blocks.find(p);
            if (it != gs->blocks.end() && !it->second.empty()) {
                auto vec = vega::deserialize<std::vector<T>>(it->second);
                output.insert(output.end(),
                              std::make_move_iterator(vec.begin()),
                              std::make_move_iterator(vec.end()));
            }
        }
    }

    // Cleanup.
    {
        std::lock_guard<std::mutex> lock(gather_mu_);
        gathers_.erase(job_id);
    }
    return output;
}

template<typename T>
std::vector<T> Cluster::execute_with_shuffle(
        const ExecutionPlan& plan, const RDD<T>& rdd, size_t job_id) {
    auto& map_stage = plan.stages[0];
    auto& reduce_stage = plan.stages[1];
    auto shuffle_rdd = reduce_stage.shuffle_rdd;

    size_t map_parts = map_stage.final_rdd->num_partitions();
    size_t reduce_parts = rdd.num_partitions();
    size_t np = num_peers();

    // Register the shuffle RDD so incoming blocks can be routed to it.
    {
        std::lock_guard<std::mutex> lock(active_shuffle_mu_);
        active_shuffle_rdds_[job_id] = shuffle_rdd;
    }

    // Create the MAP_DONE barrier.
    auto barrier = std::make_shared<ShuffleBarrier>();
    {
        std::lock_guard<std::mutex> lock(barriers_mu_);
        barriers_[job_id] = barrier;
    }

    // --- MAP PHASE: compute local map partitions, send shuffle blocks P2P ---

    auto my_map_pids = my_partitions(map_parts);
    for (auto pid : my_map_pids) {
        auto result = map_stage.final_rdd->compute_partition(pid);

        // Hash-partition into reduce buckets and send each block
        // directly to the target peer.
        auto blocks = shuffle_rdd->shuffle_partition(result, reduce_parts);
        for (auto& [reduce_pid, block] : blocks) {
            auto target = owner_of(reduce_pid);
            if (target.id == cfg_.node_id) {
                // Local delivery.
                shuffle_rdd->receive_shuffle_data(reduce_pid, block);
            } else {
                ByteBuffer body;
                write_pod<uint64_t>(body,
                    static_cast<uint64_t>(job_id));
                write_pod<uint64_t>(body,
                    static_cast<uint64_t>(reduce_pid));
                body.insert(body.end(), block.begin(), block.end());
                router_.send(target.endpoint,
                             net::NetMsgType::SHUFFLE_BLOCK,
                             cfg_.node_id, body);
            }
        }
    }

    // --- BARRIER: signal MAP_DONE to all peers, wait for all peers ----------

    broadcast_map_done(job_id);
    wait_for_map_done(job_id, np);

    // All shuffle blocks have arrived (TCP ordering + barrier).
    shuffle_rdd->finalize_shuffle();

    // --- REDUCE PHASE: compute local reduce partitions ----------------------

    auto my_reduce_pids = my_partitions(reduce_parts);

    // Set up gather state.
    auto gs = std::make_shared<GatherState>();
    gs->expected = reduce_parts;
    gs->received = 0;
    {
        std::lock_guard<std::mutex> lock(gather_mu_);
        gathers_[job_id] = gs;
    }

    for (auto pid : my_reduce_pids) {
        auto result = rdd.impl()->compute_partition(pid);
        auto vec = std::any_cast<std::vector<T>>(result);
        auto buf = vega::serialize<std::vector<T>>(vec);

        {
            std::lock_guard<std::mutex> lock(gs->mu);
            gs->blocks[pid] = buf;
            gs->received++;
        }
        send_result_to_all_peers(job_id, pid, buf);
    }

    // Wait for all reduce results (allgather).
    {
        std::unique_lock<std::mutex> lock(gs->mu);
        gs->cv.wait(lock, [&] {
            return gs->received >= gs->expected;
        });
    }

    std::vector<T> output;
    {
        std::lock_guard<std::mutex> lock(gs->mu);
        for (size_t p = 0; p < reduce_parts; ++p) {
            auto it = gs->blocks.find(p);
            if (it != gs->blocks.end() && !it->second.empty()) {
                auto vec = vega::deserialize<std::vector<T>>(it->second);
                output.insert(output.end(),
                              std::make_move_iterator(vec.begin()),
                              std::make_move_iterator(vec.end()));
            }
        }
    }

    // Cleanup.
    {
        std::lock_guard<std::mutex> lock(gather_mu_);
        gathers_.erase(job_id);
    }
    {
        std::lock_guard<std::mutex> lock(barriers_mu_);
        barriers_.erase(job_id);
    }
    {
        std::lock_guard<std::mutex> lock(active_shuffle_mu_);
        active_shuffle_rdds_.erase(job_id);
    }
    return output;
}

} // namespace vega
