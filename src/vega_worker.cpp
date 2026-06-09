#include <csignal>
#include <iostream>

#include "vega/cluster/cluster.hpp"

static std::atomic<bool> g_running{true};

static void signal_handler(int) { g_running.store(false); }

// Generic Vega peer node.
// All nodes are equal -- there is no driver/worker distinction.
// Any peer can submit jobs; other peers participate in execution.
int main(int argc, char** argv) {
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    auto cfg = vega::ClusterConfig::from_args(argc, argv);

    std::cerr << "[peer] starting " << cfg.node_id
              << " on " << cfg.bind_host << ":" << cfg.bind_port << "\n";

    if (!cfg.seed_nodes.empty()) {
        std::cerr << "[peer] seeds:";
        for (auto& s : cfg.seed_nodes)
            std::cerr << " " << s.to_string();
        std::cerr << "\n";
    }

    vega::Cluster cluster(cfg);
    std::cerr << "[peer] ready, listening on port "
              << cfg.bind_port << "\n";

    while (g_running.load())
        std::this_thread::sleep_for(std::chrono::milliseconds(200));

    std::cerr << "[peer] shutting down\n";
    cluster.stop();
    return 0;
}
