#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>
#include <unordered_map>

#include "vega/cluster/node.hpp"
#include "vega/network/endpoint.hpp"
#include "vega/network/message_codec.hpp"
#include "vega/network/tcp_connection.hpp"
#include "vega/network/tcp_server.hpp"

namespace vega::cluster {

// ---------------------------------------------------------------------------
// GossipProtocol -- decentralized peer discovery via push-gossip.
//
// Periodically sends the full membership list to a random peer.  Each node
// merges incoming lists, automatically discovering new peers.  Nodes that
// haven't sent a heartbeat within `dead_timeout` are marked dead.
// ---------------------------------------------------------------------------

class GossipProtocol {
public:
    using OnNodeJoin  = std::function<void(const NodeInfo&)>;
    using OnNodeLeave = std::function<void(const NodeId&)>;

    struct Config {
        std::chrono::milliseconds gossip_interval{1000};
        std::chrono::milliseconds dead_timeout{10000};
        std::vector<net::Endpoint> seed_nodes;
    };

    GossipProtocol(NodeInfo self, Config cfg,
                   OnNodeJoin on_join = nullptr,
                   OnNodeLeave on_leave = nullptr)
        : self_(std::move(self))
        , cfg_(std::move(cfg))
        , on_join_(std::move(on_join))
        , on_leave_(std::move(on_leave))
    {
        {
            std::lock_guard<std::mutex> lock(members_mu_);
            members_[self_.id] = self_;
        }
    }

    void start() {
        if (running_.exchange(true)) return;
        gossip_thread_ = std::thread(&GossipProtocol::gossip_loop, this);
    }

    void stop() {
        if (!running_.exchange(false)) return;
        if (gossip_thread_.joinable()) gossip_thread_.join();
    }

    // Handle an incoming gossip message (called from the TcpServer handler).
    void handle_gossip_ping(const ByteBuffer& body) {
        size_t off = 0;
        auto count = read_pod<uint32_t>(body, off);
        for (uint32_t i = 0; i < count; ++i) {
            auto ni = NodeInfo::deserialize(body, off);
            merge_node(ni);
        }
    }

    // Returns the list of known live nodes.
    std::vector<NodeInfo> live_nodes() const {
        std::lock_guard<std::mutex> lock(members_mu_);
        std::vector<NodeInfo> out;
        for (auto& [_, ni] : members_)
            if (ni.alive) out.push_back(ni);
        return out;
    }

    // Returns live worker nodes only.
    std::vector<NodeInfo> live_workers() const {
        std::lock_guard<std::mutex> lock(members_mu_);
        std::vector<NodeInfo> out;
        for (auto& [_, ni] : members_)
            if (ni.alive && ni.role == NodeRole::WORKER) out.push_back(ni);
        return out;
    }

    size_t live_count() const {
        std::lock_guard<std::mutex> lock(members_mu_);
        size_t n = 0;
        for (auto& [_, ni] : members_)
            if (ni.alive) ++n;
        return n;
    }

    const NodeInfo& self() const { return self_; }

private:
    NodeInfo self_;
    Config cfg_;
    OnNodeJoin on_join_;
    OnNodeLeave on_leave_;

    mutable std::mutex members_mu_;
    std::unordered_map<NodeId, NodeInfo> members_;

    std::atomic<bool> running_{false};
    std::thread gossip_thread_;

    void merge_node(const NodeInfo& incoming) {
        std::lock_guard<std::mutex> lock(members_mu_);
        auto it = members_.find(incoming.id);
        if (it == members_.end()) {
            auto copy = incoming;
            copy.last_heartbeat = Clock::now();
            copy.alive = true;
            members_[copy.id] = copy;
            if (on_join_) on_join_(copy);
        } else {
            it->second.last_heartbeat = Clock::now();
            if (!it->second.alive) {
                it->second.alive = true;
                if (on_join_) on_join_(it->second);
            }
        }
    }

    // Build the membership list as a ByteBuffer.
    ByteBuffer build_ping_body() const {
        std::lock_guard<std::mutex> lock(members_mu_);
        ByteBuffer body;
        uint32_t count = 0;
        for (auto& [_, ni] : members_)
            if (ni.alive) ++count;
        write_pod<uint32_t>(body, count);
        for (auto& [_, ni] : members_)
            if (ni.alive) ni.serialize(body);
        return body;
    }

    void gossip_loop() {
        std::mt19937 rng(std::random_device{}());

        // Seed: immediately contact seed nodes to bootstrap.
        for (auto& seed : cfg_.seed_nodes)
            send_gossip_to(seed);

        while (running_.load()) {
            std::this_thread::sleep_for(cfg_.gossip_interval);
            if (!running_.load()) break;

            reap_dead_nodes();

            // Pick a random live peer and push our membership list.
            auto peers = live_nodes();
            if (peers.size() <= 1) {
                // No peers yet; retry seeds.
                for (auto& seed : cfg_.seed_nodes)
                    send_gossip_to(seed);
                continue;
            }

            std::uniform_int_distribution<size_t> dist(0, peers.size() - 1);
            size_t idx = dist(rng);
            if (peers[idx].id == self_.id) {
                idx = (idx + 1) % peers.size();
                if (peers[idx].id == self_.id) continue;
            }
            send_gossip_to(peers[idx].endpoint);
        }
    }

    void send_gossip_to(const net::Endpoint& ep) {
        try {
            auto conn = net::TcpConnection::connect(ep.host, ep.port);
            auto body = build_ping_body();
            auto frame = net::encode_message(
                net::NetMsgType::GOSSIP_PING, self_.id, body);
            conn->send_frame(frame);
        } catch (...) {
            // Connection failure is fine -- the peer might not be up yet.
        }
    }

    void reap_dead_nodes() {
        auto now = Clock::now();
        std::lock_guard<std::mutex> lock(members_mu_);
        for (auto& [id, ni] : members_) {
            if (id == self_.id) continue;
            if (ni.alive &&
                now - ni.last_heartbeat > cfg_.dead_timeout) {
                ni.alive = false;
                if (on_leave_) on_leave_(id);
            }
        }
    }
};

} // namespace vega::cluster
