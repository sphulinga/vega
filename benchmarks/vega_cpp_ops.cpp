// C++ Vega counterpart of benchmarks/spark_scala_ops.scala.
//
// Times the RDD operations that Vega's C++ API offers, on the same datasets
// and partitioning as the Scala Spark benchmark, so the two native APIs can
// be compared like for like. Built as the `vega_cpp_ops` target:
//
//   cmake --build build-release --target vega_cpp_ops
//   build-release/vega_cpp_ops --n 1000000 --workers 4 --repeat 3
//
// Prints one "RESULT\t<operation>\t<best ms>\t<result>" line per operation;
// the result (a count or sum) lets the runner check that both engines did the
// same work.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <vega/vega.hpp>

namespace {

struct Config {
    std::size_t n = 1'000'000;
    std::size_t workers = 4;
    std::size_t partitions = 8;
    std::size_t repeat = 3;
    std::size_t warmup = 1;
};

Config parse_args(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](std::size_t& dst) {
            if (i + 1 < argc) dst = static_cast<std::size_t>(std::strtoull(argv[++i], nullptr, 10));
        };
        if (a == "--n") next(cfg.n);
        else if (a == "--workers") next(cfg.workers);
        else if (a == "--partitions") next(cfg.partitions);
        else if (a == "--repeat") next(cfg.repeat);
        else if (a == "--warmup") next(cfg.warmup);
    }
    return cfg;
}

using Case = std::function<int64_t()>;

// Best wall time over `repeat` runs, in milliseconds; `result` receives the
// operation's result (a count or sum) from the last run.
double best_ms(const Case& fn, std::size_t repeat, int64_t& result) {
    double best = std::numeric_limits<double>::infinity();
    for (std::size_t r = 0; r < repeat; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        result = fn();
        const auto t1 = std::chrono::steady_clock::now();
        best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    return best;
}

}  // namespace

int main(int argc, char** argv) {
    const Config cfg = parse_args(argc, argv);
    using Pair = std::pair<int, double>;

    // Same datasets as spark_scala_ops.scala: (i % 1000, i) pairs and ints.
    std::vector<Pair> objs(cfg.n);
    std::vector<int64_t> nums(cfg.n);
    for (std::size_t i = 0; i < cfg.n; ++i) {
        objs[i] = {static_cast<int>(i % 1000), static_cast<double>(i)};
        nums[i] = static_cast<int64_t>(i);
    }

    vega::SparkContext sc(cfg.workers);
    const std::size_t parts = cfg.partitions;
    auto pairs = sc.parallelize(objs, parts);
    auto nrdd = sc.parallelize(nums, parts);

    auto n = [](std::size_t v) { return static_cast<int64_t>(v); };
    const std::vector<std::pair<std::string, Case>> cases = {
        {"parallelize+collect (pairs)", [&]() -> int64_t {
             return n(sc.parallelize(objs, parts).collect().size());
         }},
        {"map+filter (pairs)", [&]() -> int64_t {
             auto rdd = pairs.map([](const Pair& kv) { return Pair{kv.first, kv.second * 2}; })
                            .filter([](const Pair& kv) { return kv.first % 2 == 0; });
             return n(rdd.count());
         }},
        {"flatMap x3 (ints)", [&]() -> int64_t {
             auto rdd = nrdd.flatMap([](const int64_t& x) {
                 return std::vector<int64_t>{x, x + 1, x + 2};
             });
             return n(rdd.count());
         }},
        {"reduce sum (ints)", [&]() -> int64_t {
             return nrdd.reduce([](const int64_t& a, const int64_t& b) { return a + b; });
         }},
        {"reduceByKey (pairs)", [&]() -> int64_t {
             return n(pairs.reduceByKey([](double a, double b) { return a + b; }).count());
         }},
        {"groupByKey (pairs)", [&]() -> int64_t {
             return n(pairs.groupByKey().count());
         }},
        {"chained shuffle", [&]() -> int64_t {
             auto rdd = pairs.reduceByKey([](double a, double b) { return a + b; })
                            .map([](const Pair& kv) { return Pair{kv.first % 10, kv.second}; })
                            .groupByKey();
             return n(rdd.count());
         }},
        {"narrow pipeline (ints)", [&]() -> int64_t {
             auto rdd = nrdd.map([](const int64_t& x) { return (x * 31 + 7) % 1'000'003; })
                            .filter([](const int64_t& x) { return x % 7 == 0; })
                            .flatMap([](const int64_t& x) {
                                return std::vector<int64_t>{x, x + 1, x + 2};
                            });
             return n(rdd.count());
         }},
    };

    // Warm-up pass, matching the Spark benchmark.
    for (std::size_t w = 0; w < cfg.warmup; ++w)
        for (const auto& [name, fn] : cases) fn();

    for (const auto& [name, fn] : cases) {
        int64_t result = 0;
        const double ms = best_ms(fn, cfg.repeat, result);
        std::printf("RESULT\t%s\t%.3f\t%lld\n", name.c_str(), ms, static_cast<long long>(result));
    }
    return 0;
}
