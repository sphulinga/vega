include(FetchContent)

find_package(Threads REQUIRED)

if(VEGA_BUILD_TESTS)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

if(VEGA_BUILD_PYTHON OR VEGA_BUILD_SERVER OR VEGA_BUILD_PYMODULE OR VEGA_BUILD_TESTS
   OR VEGA_BUILD_EXAMPLES)
    # Embedding (executables) needs the Development.Embed component; the
    # extension module only needs Development.Module (no libpython).
    if(VEGA_BUILD_SERVER OR VEGA_BUILD_TESTS OR VEGA_BUILD_EXAMPLES OR VEGA_BUILD_PYTHON)
        find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    else()
        find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
    endif()

    # Force pybind11 onto the modern FindPython module and pin it to the SAME
    # interpreter we just resolved. Otherwise pybind11 falls back to the legacy
    # FindPythonLibs, which may pick a different (and possibly library-less)
    # Python (e.g. the system /usr/bin/python3) and break linking.
    set(PYBIND11_FINDPYTHON ON CACHE BOOL "" FORCE)
    set(Python_EXECUTABLE "${Python3_EXECUTABLE}" CACHE FILEPATH "" FORCE)

    # Prefer an installed pybind11 (pip's build requirement in wheel builds);
    # fall back to fetching it for plain CMake builds.
    find_package(pybind11 2.13 CONFIG QUIET)
    if(NOT pybind11_FOUND)
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG v2.13.6
        )
        FetchContent_MakeAvailable(pybind11)
    endif()
endif()
