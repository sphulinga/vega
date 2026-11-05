// pyvega._vega_server -- the Vega session server as a CPython extension.
//
// Wheels on PyPI ship this module instead of a standalone vega_server binary.
// The standalone binary embeds libpython, which manylinux interpreters do not
// provide as a shared library; an extension module instead runs inside the
// user's own interpreter, which also guarantees that client and server use
// the same Python version (a cloudpickle requirement).
//
//     python -m pyvega._server_cli --bind 127.0.0.1 --port 9500

#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

#include <pybind11/pybind11.h>

#include "vega/client/session_server.hpp"
#include "vega/network/tcp_server.hpp"

namespace py = pybind11;

namespace {

std::atomic<bool> g_stop{false};

void serve(const std::string& host, uint16_t port, bool print_listening) {
    g_stop.store(false);
    vega::net::TcpServer server(host, port,
        [](std::unique_ptr<vega::net::TcpConnection> conn) {
            vega::client::serve_client(std::move(conn));
        });
    server.start();
    if (print_listening)
        std::cout << "LISTENING " << server.port() << std::endl << std::flush;

    // Wait with the GIL released (session threads need it), waking up
    // periodically to deliver Python signals such as KeyboardInterrupt.
    while (!g_stop.load()) {
        {
            py::gil_scoped_release release;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        if (PyErr_CheckSignals() != 0) {
            // Leave in-flight sessions running; the process is exiting.
            throw py::error_already_set();
        }
    }
}

}  // namespace

PYBIND11_MODULE(_vega_server, m) {
    m.doc() = "Vega C++ session server (in-process extension)";
    m.def("serve", &serve, py::arg("host") = "127.0.0.1", py::arg("port") = 9500,
          py::arg("print_listening") = false,
          "Run the session server until interrupted (blocks).");
    m.def("request_stop", []() { g_stop.store(true); },
          "Ask a running serve() call to return.");
}
