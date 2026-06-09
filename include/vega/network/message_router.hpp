#pragma once

#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "vega/cluster/node.hpp"
#include "vega/network/endpoint.hpp"
#include "vega/network/message_codec.hpp"
#include "vega/network/tcp_connection.hpp"
#include "vega/network/tcp_server.hpp"

namespace vega::net {

// ---------------------------------------------------------------------------
// MessageRouter -- routes wire-format messages between nodes.
//
// Runs a TcpServer to receive messages, and maintains a connection pool
// for outgoing messages.  Incoming messages are dispatched to registered
// handlers keyed by NetMsgType.
// ---------------------------------------------------------------------------

class MessageRouter {
public:
    using MessageHandler = std::function<void(const std::string& sender_id,
                                              const ByteBuffer& body)>;

    MessageRouter(const std::string& bind_host, uint16_t port)
        : server_(bind_host, port,
                  [this](std::unique_ptr<TcpConnection> conn) {
                      handle_connection(std::move(conn));
                  })
        , port_(server_.port())
    {}

    uint16_t port() const { return port_; }

    void start() { server_.start(); }
    void stop()  { server_.stop(); }

    // Register a handler for a specific message type.
    void on(NetMsgType type, MessageHandler handler) {
        std::lock_guard<std::mutex> lock(handlers_mu_);
        handlers_[type] = std::move(handler);
    }

    // Send a message to a remote node.
    void send(const net::Endpoint& target,
              NetMsgType type,
              const std::string& sender_node_id,
              const ByteBuffer& body) {
        try {
            std::lock_guard<std::mutex> lock(pool_mu_);
            auto& conn = get_or_connect_locked(target);
            auto frame = encode_message(type, sender_node_id, body);
            conn->send_frame(frame);
        } catch (const std::exception& e) {
            std::cerr << "[router] send to " << target.to_string()
                      << " failed: " << e.what() << "\n";
        }
    }

private:
    TcpServer server_;
    uint16_t port_;

    std::mutex handlers_mu_;
    std::unordered_map<NetMsgType, MessageHandler> handlers_;

    std::mutex pool_mu_;
    std::unordered_map<std::string, std::unique_ptr<TcpConnection>> pool_;

    void handle_connection(std::unique_ptr<TcpConnection> conn) {
        try {
            // Each connection may deliver multiple frames.
            while (conn->valid()) {
                auto frame = conn->recv_frame();
                size_t off = 0;
                auto header = NetMsgHeader::read(frame, off);

                ByteBuffer body(frame.begin() + off, frame.end());

                MessageHandler handler;
                {
                    std::lock_guard<std::mutex> lock(handlers_mu_);
                    auto it = handlers_.find(header.type);
                    if (it == handlers_.end()) continue;
                    handler = it->second;
                }
                handler(header.sender_node_id, body);
            }
        } catch (...) {
            // Connection closed or error -- normal during shutdown.
        }
    }

    // Must be called with pool_mu_ held.
    std::unique_ptr<TcpConnection>& get_or_connect_locked(
            const Endpoint& ep) {
        auto key = ep.to_string();
        auto it = pool_.find(key);
        if (it != pool_.end() && it->second->valid())
            return it->second;
        pool_[key] = TcpConnection::connect(ep.host, ep.port);
        return pool_[key];
    }
};

} // namespace vega::net
