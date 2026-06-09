#pragma once

#include <functional>

namespace vega {

// Optional hook for embedded-Python sessions.  When set, actor threads
// run partition work and shuffle handoff while holding the GIL so py::object
// values can cross thread boundaries safely.
inline std::function<void(std::function<void()>)>& python_gil_runner() {
    static std::function<void(std::function<void()>)> runner;
    return runner;
}

inline void run_with_python_gil(const std::function<void()>& fn) {
    if (auto& gil = python_gil_runner(); gil)
        gil(fn);
    else
        fn();
}

} // namespace vega
