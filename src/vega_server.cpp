#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

#include <pybind11/embed.h>

#include "vega/client/session_server.hpp"
#include "vega/network/tcp_server.hpp"

namespace {

std::atomic<bool> g_running{true};

void signal_handler(int) { g_running.store(false); }

struct Config {
    std::string bind_host = "127.0.0.1";
    uint16_t port = 9500;
    bool print_listening = false;
};

Config parse_args(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--bind" && i + 1 < argc) {
            auto ep = argv[++i];
            auto colon = std::string(ep).find(':');
            if (colon != std::string::npos) {
                cfg.bind_host = std::string(ep).substr(0, colon);
                cfg.port = static_cast<uint16_t>(
                    std::strtoul(std::string(ep).substr(colon + 1).c_str(), nullptr, 10));
            } else {
                cfg.bind_host = ep;
            }
        } else if (a == "--port" && i + 1 < argc) {
            cfg.port = static_cast<uint16_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (a == "--print-listening") {
            cfg.print_listening = true;
        }
    }
    return cfg;
}

} // namespace

int main(int argc, char** argv) {
    pybind11::scoped_interpreter guard{};
    pybind11::gil_scoped_release release_main_gil{};

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    auto cfg = parse_args(argc, argv);

    vega::net::TcpServer server(cfg.bind_host, cfg.port,
        [](std::unique_ptr<vega::net::TcpConnection> conn) {
            vega::client::serve_client(std::move(conn));
        });
    server.start();

    if (cfg.print_listening)
        std::cout << "LISTENING " << server.port() << std::endl << std::flush;

    while (g_running.load())
        std::this_thread::sleep_for(std::chrono::milliseconds(200));

    server.stop();
    return 0;
}
