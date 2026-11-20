add_executable(word_count word_count.cpp)
target_link_libraries(word_count PRIVATE vega)

add_executable(cluster_word_count cluster_word_count.cpp)
target_link_libraries(cluster_word_count PRIVATE vega)

add_executable(core_benchmark core_benchmark.cpp)
target_link_libraries(core_benchmark PRIVATE vega)

add_executable(shuffle_benchmark shuffle_benchmark.cpp)
target_link_libraries(shuffle_benchmark PRIVATE vega)

# Like-for-like counterpart of benchmarks/spark_scala_ops.scala (C++ Vega vs
# Scala Spark); run through benchmarks/bench_vega_cpp_ops.py or directly.
add_executable(vega_cpp_ops ${PROJECT_SOURCE_DIR}/benchmarks/vega_cpp_ops.cpp)
target_link_libraries(vega_cpp_ops PRIVATE vega)
