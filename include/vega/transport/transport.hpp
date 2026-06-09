#pragma once

#include <any>
#include <string>

namespace vega {

// Abstract transport for cross-node data movement.
//
// Phase 1 (single node): all communication goes through the actor system;
// this interface exists to define the contract for Phase 3's
// NetworkTransport (gRPC / TCP).
class Transport {
public:
    virtual ~Transport() = default;

    // Ship a data block to a remote node.
    virtual void send_data(const std::string& target_node,
                           const std::string& key,
                           std::any data) = 0;

    // Retrieve a previously sent data block.
    virtual std::any receive_data(const std::string& source_node,
                                  const std::string& key) = 0;
};

} // namespace vega
