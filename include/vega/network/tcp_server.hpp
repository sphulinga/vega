#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include "vega/network/tcp_connection.hpp"

namespace vega::net {

// Simple TCP server: listens on a port, accepts connections, and dispatches
// each to a user-provided handler on its own thread.
class TcpServer {
public:
    using Handler = std::function<void(std::unique_ptr<TcpConnection>)>;

    TcpServer(const std::string& bind_host, uint16_t port, Handler handler)
        : handler_(std::move(handler))
    {
        listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (listen_fd_ < 0)
            throw std::runtime_error("socket() failed");

        int opt = 1;
        setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        struct sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        if (bind_host == "0.0.0.0" || bind_host.empty())
            addr.sin_addr.s_addr = INADDR_ANY;
        else
            inet_pton(AF_INET, bind_host.c_str(), &addr.sin_addr);

        if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr),
                   sizeof(addr)) < 0) {
            ::close(listen_fd_);
            throw std::runtime_error("bind() failed on port " +
                                     std::to_string(port));
        }
        if (::listen(listen_fd_, 64) < 0) {
            ::close(listen_fd_);
            throw std::runtime_error("listen() failed");
        }

        // Retrieve the actual port (useful when binding to port 0).
        struct sockaddr_in bound{};
        socklen_t blen = sizeof(bound);
        getsockname(listen_fd_, reinterpret_cast<sockaddr*>(&bound), &blen);
        port_ = ntohs(bound.sin_port);
    }

    ~TcpServer() { stop(); }

    uint16_t port() const { return port_; }

    void start() {
        if (running_.exchange(true)) return;
        accept_thread_ = std::thread(&TcpServer::accept_loop, this);
    }

    void stop() {
        if (!running_.exchange(false)) return;
        ::shutdown(listen_fd_, SHUT_RDWR);
        ::close(listen_fd_);
        listen_fd_ = -1;
        if (accept_thread_.joinable()) accept_thread_.join();
        for (auto& t : conn_threads_)
            if (t.joinable()) t.join();
        conn_threads_.clear();
    }

private:
    Handler handler_;
    int listen_fd_ = -1;
    uint16_t port_ = 0;
    std::atomic<bool> running_{false};
    std::thread accept_thread_;
    std::vector<std::thread> conn_threads_;

    void accept_loop() {
        while (running_.load()) {
            struct sockaddr_in peer{};
            socklen_t plen = sizeof(peer);
            int fd = ::accept(listen_fd_,
                              reinterpret_cast<sockaddr*>(&peer), &plen);
            if (fd < 0) break;
            auto conn = std::make_unique<TcpConnection>(fd);
            conn_threads_.emplace_back(
                [this, c = std::move(conn)]() mutable {
                    try { handler_(std::move(c)); }
                    catch (...) {}
                });
        }
    }
};

} // namespace vega::net
