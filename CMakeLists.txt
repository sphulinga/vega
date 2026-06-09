cmake_minimum_required(VERSION 3.16)
project(Vega VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(VEGA_BUILD_PYTHON "Build in-process Python bindings (legacy)" OFF)
option(VEGA_BUILD_SERVER "Build Vega C++ session server for PyVega" ON)

include(cmake/FetchDependencies.cmake)

# Header-only library
add_library(vega INTERFACE)
target_include_directories(vega INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(vega INTERFACE Threads::Threads)

# The public headers (e.g. actors/worker_actor.hpp) include pybind11 and use
# the CPython API, so every consumer of vega needs the pybind11/Python headers
# and must link libpython. Propagate that through the interface target.
if(TARGET pybind11::embed)
    target_link_libraries(vega INTERFACE pybind11::embed)
endif()

# Worker daemon (P2P cluster peer)
add_executable(vega_worker src/vega_worker.cpp)
target_link_libraries(vega_worker PRIVATE vega)

# Session server for PyVega clients
if(VEGA_BUILD_SERVER)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    add_executable(vega_server src/vega_server.cpp src/server_session.cpp)
    target_link_libraries(vega_server PRIVATE vega pybind11::embed Python3::Python)
    target_include_directories(vega_server PRIVATE ${Python3_INCLUDE_DIRS})
endif()

add_subdirectory(examples)

enable_testing()
add_subdirectory(tests)

if(VEGA_BUILD_PYTHON)
    add_subdirectory(python)
endif()
