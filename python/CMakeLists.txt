pybind11_add_module(_vega_core bindings.cpp)
target_link_libraries(_vega_core PRIVATE vega)

# Copy the pure-Python package next to the .so for easy import
add_custom_command(TARGET _vega_core POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/pyvega
        $<TARGET_FILE_DIR:_vega_core>/pyvega
)
