include(FetchContent)

find_package(Threads REQUIRED)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

if(VEGA_BUILD_PYTHON OR VEGA_BUILD_SERVER)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

    # Force pybind11 onto the modern FindPython module and pin it to the SAME
    # interpreter we just resolved. Otherwise pybind11 falls back to the legacy
    # FindPythonLibs, which may pick a different (and possibly library-less)
    # Python (e.g. the system /usr/bin/python3) and break linking.
    set(PYBIND11_FINDPYTHON ON CACHE BOOL "" FORCE)
    set(Python_EXECUTABLE "${Python3_EXECUTABLE}" CACHE FILEPATH "" FORCE)

    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.13.6
    )
    FetchContent_MakeAvailable(pybind11)
endif()
