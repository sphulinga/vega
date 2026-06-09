include(GoogleTest)

function(add_vega_test test_name)
    add_executable(${test_name} ${test_name}.cpp)
    target_link_libraries(${test_name} PRIVATE vega GTest::gtest_main)
    gtest_discover_tests(${test_name})
endfunction()

add_vega_test(test_partition)
add_vega_test(test_rdd)
add_vega_test(test_dag)
add_vega_test(test_actors)
add_vega_test(test_serialization)
add_vega_test(test_shuffle)
add_vega_test(test_cluster)

add_executable(test_spark_rdd_suite spark_compat/rdd_suite.cpp)
target_link_libraries(test_spark_rdd_suite PRIVATE vega GTest::gtest_main)
gtest_discover_tests(test_spark_rdd_suite)
