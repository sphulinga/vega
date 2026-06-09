#pragma once

#include <memory>

#include "vega/network/tcp_connection.hpp"

namespace vega::client {

void serve_client(std::unique_ptr<net::TcpConnection> conn);

} // namespace vega::client
