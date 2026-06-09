// Distributed word count using Vega's decentralized P2P cluster.
//
// Every node is an equal peer -- there is no driver or central scheduler.
// Peers discover each other via gossip, then coordinate execution directly.
//
// Usage (same binary on all machines):
//
//   # Peer 1 (submitter -- but NOT a special role):
//   ./cluster_word_count --bind 0.0.0.0:9001 --id peer-1 --expect-peers 3
//
//   # Peer 2:
//   ./cluster_word_count --bind 0.0.0.0:9002 --id peer-2 \
//       --seeds 127.0.0.1:9001 --expect-peers 3
//
//   # Peer 3:
//   ./cluster_word_count --bind 0.0.0.0:9003 --id peer-3 \
//       --seeds 127.0.0.1:9001 --expect-peers 3
//
// For a single-node demo (no cluster required):
//   ./cluster_word_count --local

#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <vega/vega.hpp>

static std::vector<std::string> split_words(const std::string& line) {
    std::vector<std::string> words;
    std::istringstream iss(line);
    std::string w;
    while (iss >> w) words.push_back(w);
    return words;
}

// Single-node demo using the local SparkContext + reduceByKey.
static void run_local() {
    std::cout << "=== Single-node word count (reduceByKey) ===\n\n";

    vega::SparkContext ctx(4);

    std::vector<std::string> lines = {
        "hello world hello",
        "world of vega",
        "hello vega cluster",
        "distributed word count example",
    };

    auto counts = ctx.parallelize(lines, 4)
        .flatMap([](const std::string& line) { return split_words(line); })
        .map([](const std::string& word) -> std::pair<std::string, int> {
            return {word, 1};
        })
        .reduceByKey([](int a, int b) { return a + b; })
        .collect();

    std::sort(counts.begin(), counts.end(),
              [](auto& a, auto& b) { return a.second > b.second; });

    for (auto& [word, count] : counts)
        std::cout << "  " << word << ": " << count << "\n";

    std::cout << "\n=== groupByKey demo ===\n\n";

    std::vector<std::pair<std::string, int>> pairs = {
        {"a", 1}, {"b", 2}, {"a", 3}, {"b", 4}, {"c", 5},
    };

    auto grouped = ctx.parallelize(pairs, 2)
        .groupByKey()
        .collect();

    for (auto& [key, values] : grouped) {
        std::cout << "  " << key << ": [";
        for (size_t i = 0; i < values.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << values[i];
        }
        std::cout << "]\n";
    }
}

// Decentralized cluster mode: all peers are equal.
static void run_cluster(int argc, char** argv) {
    auto cfg = vega::ClusterConfig::from_args(argc, argv);
    vega::Cluster cluster(cfg);

    std::cerr << "[" << cluster.node_id()
              << "] waiting for " << cfg.expected_peers << " peers...\n";

    if (!cluster.wait_for_peers(std::chrono::seconds(30))) {
        std::cerr << "[" << cluster.node_id()
                  << "] timed out waiting for peers\n";
        cluster.stop();
        return;
    }

    auto p = cluster.peers();
    std::cerr << "[" << cluster.node_id() << "] cluster ready with "
              << p.size() << " peers:";
    for (auto& peer : p)
        std::cerr << " " << peer.id;
    std::cerr << "\n";

    // ALL peers create the same pipeline (SPMD).
    std::vector<std::string> lines = {
        "hello world hello",
        "world of vega",
        "hello vega cluster",
        "distributed word count example",
    };

    // Scatter data from peer-0 (sorted first) to all peers.
    // Every peer calls scatter, but only one has the data.
    std::vector<std::string> my_data;
    if (cluster.my_index() == 0)
        my_data = lines;

    auto rdd = cluster.scatter(my_data, p.size());

    // Collect: each peer computes its own partitions, no central driver.
    auto result = cluster.collect(rdd);

    // Only peer-0 prints the result (arbitrary -- any peer could).
    if (cluster.my_index() == 0) {
        std::cout << "\n=== Distributed word count (" << p.size()
                  << " peers, P2P) ===\n\n";
        for (auto& word : result)
            std::cout << "  " << word << "\n";
    }

    cluster.stop();
}

int main(int argc, char** argv) {
    bool local_mode = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--local") {
            local_mode = true;
            break;
        }
    }

    if (local_mode)
        run_local();
    else
        run_cluster(argc, argv);

    return 0;
}
