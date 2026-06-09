#pragma once

#include <chrono>
#include <string>

#include "vega/network/endpoint.hpp"
#include "vega/serialization/serializer.hpp"

namespace vega::cluster {

using NodeId = std::string;
using Clock = std::chrono::steady_clock;

enum class NodeRole : uint8_t {
    DRIVER  = 0,
    WORKER  = 1,
};

struct NodeInfo {
    NodeId id;
    net::Endpoint endpoint;
    NodeRole role = NodeRole::WORKER;
    Clock::time_point last_heartbeat = Clock::now();
    bool alive = true;

    void serialize(ByteBuffer& buf) const {
        Serializer<std::string>::serialize(id, buf);
        Serializer<std::string>::serialize(endpoint.host, buf);
        write_pod<uint16_t>(buf, endpoint.port);
        write_pod<uint8_t>(buf, static_cast<uint8_t>(role));
    }

    static NodeInfo deserialize(const ByteBuffer& buf, size_t& off) {
        NodeInfo ni;
        ni.id = Serializer<std::string>::deserialize(buf, off);
        ni.endpoint.host = Serializer<std::string>::deserialize(buf, off);
        ni.endpoint.port = read_pod<uint16_t>(buf, off);
        ni.role = static_cast<NodeRole>(read_pod<uint8_t>(buf, off));
        ni.last_heartbeat = Clock::now();
        ni.alive = true;
        return ni;
    }
};

} // namespace vega::cluster
