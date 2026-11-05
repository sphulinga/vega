cmake_minimum_required(VERSION 3.16)

# The version is single-sourced from python/pyvega/__init__.py; scikit-build-core
# passes it as SKBUILD_PROJECT_VERSION when building wheels.
if(DEFINED SKBUILD_PROJECT_VERSION)
    set(_vega_version ${SKBUILD_PROJECT_VERSION})
else()
    file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/python/pyvega/__init__.py" _vega_version_line
         REGEX "^__version__ = ")
    string(REGEX REPLACE "^__version__ = \"([0-9]+\\.[0-9]+\\.[0-9]+).*\"$" "\\1"
           _vega_version "${_vega_version_line}")
endif()
project(Vega VERSION ${_vega_version} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Wheel builds (scikit-build-core sets SKBUILD) only need the extension module.
if(SKBUILD)
    set(_vega_full OFF)
else()
    set(_vega_full ON)
endif()

option(VEGA_BUILD_PYMODULE "Build the pyvega._vega_server Python extension" ${SKBUILD})
option(VEGA_BUILD_SERVER "Build the standalone vega_server binary (embeds Python)" ${_vega_full})
option(VEGA_BUILD_TESTS "Build C++ tests (fetches GoogleTest)" ${_vega_full})
option(VEGA_BUILD_EXAMPLES "Build C++ examples and the vega_worker daemon" ${_vega_full})
option(VEGA_BUILD_PYTHON "Build in-process Python bindings (legacy)" OFF)

if(NOT SKBUILD AND NOT VEGA_BUILD_PYMODULE AND NOT DEFINED CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

include(cmake/FetchDependencies.cmake)

# Header-only library (no Python linkage): used by the extension module,
# which must not link libpython.
add_library(vega_core INTERFACE)
target_include_directories(vega_core INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(vega_core INTERFACE Threads::Threads)

# The public headers (e.g. actors/worker_actor.hpp) include pybind11 and use
# the CPython API, so every *executable* consumer needs the pybind11/Python
# headers and must link libpython (pybind11::embed).
add_library(vega INTERFACE)
target_link_libraries(vega INTERFACE vega_core)
if(TARGET pybind11::embed)
    target_link_libraries(vega INTERFACE pybind11::embed)
endif()

if(VEGA_BUILD_EXAMPLES)
    # Worker daemon (P2P cluster peer)
    add_executable(vega_worker src/vega_worker.cpp)
    target_link_libraries(vega_worker PRIVATE vega)
    add_subdirectory(examples)
endif()

# Standalone session server for PyVega clients (source checkouts).
if(VEGA_BUILD_SERVER)
    add_executable(vega_server src/vega_server.cpp src/server_session.cpp)
    target_link_libraries(vega_server PRIVATE vega pybind11::embed Python3::Python)
    target_include_directories(vega_server PRIVATE ${Python3_INCLUDE_DIRS})
endif()

# In-process server extension shipped in wheels: pyvega/_vega_server.*.so
if(VEGA_BUILD_PYMODULE)
    pybind11_add_module(_vega_server MODULE src/server_module.cpp src/server_session.cpp)
    target_link_libraries(_vega_server PRIVATE vega_core)
    if(SKBUILD)
        install(TARGETS _vega_server LIBRARY DESTINATION pyvega)
    else()
        # Developer build: drop the module next to the sources for imports.
        set_target_properties(_vega_server PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/pyvega)
    endif()
endif()

if(VEGA_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(VEGA_BUILD_PYTHON)
    add_subdirectory(python)
endif()
