#pragma once

#include <cstdint>
#include <functional>
#include <string>

namespace vega::net {

struct Endpoint {
    std::string host;
    uint16_t port = 0;

    std::string to_string() const { return host + ":" + std::to_string(port); }

    bool operator==(const Endpoint& o) const {
        return host == o.host && port == o.port;
    }
    bool operator!=(const Endpoint& o) const { return !(*this == o); }

    static Endpoint parse(const std::string& s) {
        auto pos = s.rfind(':');
        if (pos == std::string::npos)
            return {s, 0};
        return {s.substr(0, pos),
                static_cast<uint16_t>(std::stoi(s.substr(pos + 1)))};
    }
};

} // namespace vega::net

namespace std {
template<> struct hash<vega::net::Endpoint> {
    size_t operator()(const vega::net::Endpoint& e) const {
        return hash<string>()(e.to_string());
    }
};
} // namespace std
