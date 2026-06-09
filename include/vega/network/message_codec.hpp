#pragma once

#include <cstdint>
#include <string>

#include "vega/serialization/serializer.hpp"

namespace vega::net {

// Network message types that travel between nodes.
enum class NetMsgType : uint8_t {
    // Gossip
    GOSSIP_PING         = 0x01,
    GOSSIP_ACK          = 0x02,

    // Data distribution (driver → workers)
    PARTITION_DATA      = 0x10,
    PARTITION_ACK       = 0x11,

    // Task control
    ASSIGN_TASK         = 0x20,
    TASK_RESULT         = 0x21,

    // Shuffle (P2P)
    SHUFFLE_BLOCK       = 0x30,
    SHUFFLE_MAP_DONE    = 0x31,

    // P2P job coordination
    RESULT_BLOCK        = 0x40,
    JOB_BARRIER         = 0x41,
};

// Header prepended to every network frame's payload.
struct NetMsgHeader {
    NetMsgType type;
    std::string sender_node_id;

    void write(ByteBuffer& buf) const {
        write_pod<uint8_t>(buf, static_cast<uint8_t>(type));
        Serializer<std::string>::serialize(sender_node_id, buf);
    }

    static NetMsgHeader read(const ByteBuffer& buf, size_t& off) {
        NetMsgHeader h;
        h.type = static_cast<NetMsgType>(read_pod<uint8_t>(buf, off));
        h.sender_node_id = Serializer<std::string>::deserialize(buf, off);
        return h;
    }
};

// Builds a complete wire frame: header + body.
inline ByteBuffer encode_message(NetMsgType type,
                                  const std::string& sender_node_id,
                                  const ByteBuffer& body) {
    ByteBuffer frame;
    NetMsgHeader{type, sender_node_id}.write(frame);
    frame.insert(frame.end(), body.begin(), body.end());
    return frame;
}

} // namespace vega::net
