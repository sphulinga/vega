#pragma once

#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <unistd.h>

#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0  // macOS: SO_NOSIGPIPE is set on the socket instead
#endif

#include "vega/serialization/serializer.hpp"

namespace vega::net {

// RAII wrapper for a TCP socket with length-prefix framing.
//
// Wire format per frame:
//   [4 bytes big-endian: payload length N]
//   [N bytes: payload]
class TcpConnection {
public:
    explicit TcpConnection(int fd) : fd_(fd) {
        if (fd_ >= 0) {
            // Request/response framing: disable Nagle so small replies are not
            // held back waiting for the peer's delayed ACK (~40 ms per RPC).
            int one = 1;
            ::setsockopt(fd_, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
#ifdef SO_NOSIGPIPE
            ::setsockopt(fd_, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
#endif
        }
    }

    // Connect to a remote endpoint.
    static std::unique_ptr<TcpConnection> connect(const std::string& host,
                                                   uint16_t port) {
        struct addrinfo hints{}, *res = nullptr;
        hints.ai_family = AF_INET;
        hints.ai_socktype = SOCK_STREAM;
        auto port_str = std::to_string(port);

        if (getaddrinfo(host.c_str(), port_str.c_str(), &hints, &res) != 0)
            throw std::runtime_error("getaddrinfo failed for " + host);

        int fd = ::socket(res->ai_family, res->ai_socktype, res->ai_protocol);
        if (fd < 0) {
            freeaddrinfo(res);
            throw std::runtime_error("socket() failed");
        }
        if (::connect(fd, res->ai_addr, res->ai_addrlen) < 0) {
            ::close(fd);
            freeaddrinfo(res);
            throw std::runtime_error("connect() failed to " + host + ":" +
                                     port_str);
        }
        freeaddrinfo(res);
        return std::make_unique<TcpConnection>(fd);
    }

    ~TcpConnection() {
        if (fd_ >= 0) ::close(fd_);
    }

    TcpConnection(const TcpConnection&) = delete;
    TcpConnection& operator=(const TcpConnection&) = delete;
    TcpConnection(TcpConnection&& o) noexcept : fd_(o.fd_) { o.fd_ = -1; }

    int fd() const { return fd_; }
    bool valid() const { return fd_ >= 0; }

    // ---- framed send/recv -------------------------------------------------

    void send_frame(const ByteBuffer& data) const {
        uint32_t len = htonl(static_cast<uint32_t>(data.size()));
        // Header and payload in one syscall (one TCP segment for small frames).
        struct iovec iov[2];
        iov[0].iov_base = &len;
        iov[0].iov_len = sizeof(len);
        iov[1].iov_base = const_cast<uint8_t*>(data.data());
        iov[1].iov_len = data.size();
        size_t total = sizeof(len) + data.size();
        size_t sent = 0;
        int idx = 0;
        while (sent < total) {
            struct msghdr msg{};
            msg.msg_iov = iov + idx;
            msg.msg_iovlen = static_cast<decltype(msg.msg_iovlen)>(2 - idx);
            auto n = ::sendmsg(fd_, &msg, MSG_NOSIGNAL);
            if (n <= 0)
                throw std::runtime_error("send failed");
            sent += static_cast<size_t>(n);
            size_t adv = static_cast<size_t>(n);
            while (idx < 2 && adv >= iov[idx].iov_len) {
                adv -= iov[idx].iov_len;
                iov[idx].iov_len = 0;
                ++idx;
            }
            if (idx < 2) {
                iov[idx].iov_base = static_cast<uint8_t*>(iov[idx].iov_base) + adv;
                iov[idx].iov_len -= adv;
            }
        }
    }

    ByteBuffer recv_frame() const {
        uint32_t net_len = 0;
        recv_all(&net_len, sizeof(net_len));
        uint32_t len = ntohl(net_len);
        ByteBuffer buf(len);
        if (len > 0)
            recv_all(buf.data(), len);
        return buf;
    }

private:
    int fd_;

    void send_all(const void* data, size_t len) const {
        auto* p = static_cast<const uint8_t*>(data);
        size_t sent = 0;
        while (sent < len) {
            auto n = ::send(fd_, p + sent, len - sent, MSG_NOSIGNAL);
            if (n <= 0)
                throw std::runtime_error("send failed");
            sent += static_cast<size_t>(n);
        }
    }

    void recv_all(void* data, size_t len) const {
        auto* p = static_cast<uint8_t*>(data);
        size_t got = 0;
        while (got < len) {
            auto n = ::recv(fd_, p + got, len - got, 0);
            if (n <= 0)
                throw std::runtime_error("recv failed (connection closed)");
            got += static_cast<size_t>(n);
        }
    }
};

} // namespace vega::net
