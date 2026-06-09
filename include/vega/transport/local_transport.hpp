#pragma once

#include <any>
#include <mutex>
#include <string>
#include <unordered_map>

#include "vega/transport/transport.hpp"

namespace vega {

// In-process transport for single-node execution.
// Data is stored in a thread-safe hash map keyed by "node:key".
class LocalTransport : public Transport {
public:
    void send_data(const std::string& target_node,
                   const std::string& key,
                   std::any data) override {
        std::lock_guard<std::mutex> lock(mu_);
        store_[make_key(target_node, key)] = std::move(data);
    }

    std::any receive_data(const std::string& source_node,
                          const std::string& key) override {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = store_.find(make_key(source_node, key));
        if (it == store_.end()) return {};
        return it->second;
    }

private:
    std::mutex mu_;
    std::unordered_map<std::string, std::any> store_;

    static std::string make_key(const std::string& node,
                                const std::string& key) {
        return node + ":" + key;
    }
};

} // namespace vega
