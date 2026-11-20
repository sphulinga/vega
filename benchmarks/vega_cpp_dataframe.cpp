// Vega C++ DataFrame benchmark: the same queries as
// benchmarks/spark_scala_dataframe.scala (Spark SQL) and
// benchmarks/sail_dataframe.py (Sail), over the same generated tables.
//
//   cmake --build build-release --target vega_cpp_dataframe
//   build-release/examples/vega_cpp_dataframe --n 10000000 --workers 4 --partitions 8
//
// Run through benchmarks/bench_dataframe.py to compare engines. Prints one
// "RESULT\t<query>\t<best ms>\t<value>" line per query, where value is a
// '|'-separated list of numbers the runner compares across engines.

#include <vega/vega.hpp>
#include <vega/dataframe.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

using namespace vega::df;

namespace {

std::string num(double v) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.17g", v);
    return buf;
}
std::string num(int64_t v) { return std::to_string(v); }

template<typename T>
T at(const Batch& b, size_t c, size_t r = 0) {
    return b.column(c).values<T>()[r];
}

}  // namespace

int main(int argc, char** argv) {
    size_t n = 10'000'000, workers = 4, parts = 8, repeat = 5, warmup = 1;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        const size_t v = std::strtoull(argv[i + 1], nullptr, 10);
        if (a == "--n") n = v;
        else if (a == "--workers") workers = v;
        else if (a == "--partitions") parts = v;
        else if (a == "--repeat") repeat = v;
        else if (a == "--warmup") warmup = v;
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }

    vega::SparkContext ctx(workers);

    // sales(id, store, product, qty, price, category) and stores(store, region);
    // the same formulas as the Spark and Sail scripts.
    std::vector<int64_t> id(n), store(n), product(n), qty(n);
    std::vector<double> price(n);
    std::vector<std::string> category(n);
    for (size_t k = 0; k < n; ++k) {
        const auto i = static_cast<int64_t>(k);
        id[k] = i;
        store[k] = i % 1000;
        product[k] = (i * 7919) % 100000;
        qty[k] = i % 10 + 1;
        price[k] = static_cast<double>((i * 13) % 1000) / 10.0 + 1.0;
        category[k] = "cat" + std::to_string(i % 8);
    }
    auto sales = DataFrame::from_columns(ctx,
                                         {{"id", Column(std::move(id))},
                                          {"store", Column(std::move(store))},
                                          {"product", Column(std::move(product))},
                                          {"qty", Column(std::move(qty))},
                                          {"price", Column(std::move(price))},
                                          {"category", Column(std::move(category))}},
                                         parts);
    std::vector<int64_t> sid(1000);
    std::vector<std::string> region(1000);
    for (int64_t s = 0; s < 1000; ++s) {
        sid[static_cast<size_t>(s)] = s;
        region[static_cast<size_t>(s)] = "r" + std::to_string(s % 5);
    }
    auto stores = DataFrame::from_columns(ctx, {{"store", Column(sid)}, {"region", Column(region)}}, 1);

    using Query = std::function<std::string()>;
    const std::vector<std::pair<std::string, Query>> queries = {
        {"filter+count", [&] {
             return num(static_cast<int64_t>(
                 sales.filter(col("qty") > 5 && col("price") < 50.0).count()));
         }},
        {"sum(qty*price)", [&] {
             return num(at<double>(sales.agg({sum(col("qty") * col("price"))}).collect(), 0));
         }},
        {"groupBy store (1k keys)", [&] {
             Batch b = sales.group_by({col("store")})
                           .agg({sum(col("qty")).alias("u"), avg(col("price")).alias("a")})
                           .agg({count_all(), sum(col("u")), sum(col("a"))})
                           .collect();
             return num(at<int64_t>(b, 0)) + "|" + num(at<int64_t>(b, 1)) + "|" + num(at<double>(b, 2));
         }},
        {"groupBy category (string, 8 keys)", [&] {
             Batch b = sales.group_by({col("category")})
                           .agg({sum(col("qty") * col("price")).alias("rev"), count_all().alias("c")})
                           .agg({count_all(), sum(col("rev")), sum(col("c"))})
                           .collect();
             return num(at<int64_t>(b, 0)) + "|" + num(at<double>(b, 1)) + "|" + num(at<int64_t>(b, 2));
         }},
        {"groupBy product (100k keys)", [&] {
             Batch b = sales.group_by({col("product")})
                           .agg({count_all().alias("c"), max(col("price")).alias("m")})
                           .agg({count_all(), sum(col("c")), sum(col("m"))})
                           .collect();
             return num(at<int64_t>(b, 0)) + "|" + num(at<int64_t>(b, 1)) + "|" + num(at<double>(b, 2));
         }},
        {"groupBy category, qty (40 keys)", [&] {
             Batch b = sales.group_by({col("category"), col("qty")})
                           .agg({count_all().alias("c"), sum(col("price")).alias("p")})
                           .agg({count_all(), sum(col("c")), sum(col("p"))})
                           .collect();
             return num(at<int64_t>(b, 0)) + "|" + num(at<int64_t>(b, 1)) + "|" + num(at<double>(b, 2));
         }},
        {"join stores + groupBy region", [&] {
             Batch b = sales.join(stores, {"store"})
                           .group_by({col("region")})
                           .agg({sum(col("qty")).alias("u")})
                           .agg({count_all(), sum(col("u"))})
                           .collect();
             return num(at<int64_t>(b, 0)) + "|" + num(at<int64_t>(b, 1));
         }},
        {"distinct (product, qty)", [&] {
             return num(static_cast<int64_t>(sales.select({col("product"), col("qty")}).distinct().count()));
         }},
        {"orderBy price desc limit 10", [&] {
             Batch b = sales.order_by({desc(col("price")), asc(col("id"))}).limit(10).collect();
             int64_t ids = 0;
             for (size_t r = 0; r < b.num_rows(); ++r) ids += at<int64_t>(b, 0, r);
             return num(static_cast<int64_t>(b.num_rows())) + "|" + num(ids) + "|" + num(at<double>(b, 4));
         }},
    };

    for (size_t w = 0; w < warmup; ++w)
        for (const auto& [name, q] : queries) q();

    for (const auto& [name, q] : queries) {
        double best = std::numeric_limits<double>::max();
        std::string value;
        for (size_t r = 0; r < repeat; ++r) {
            const auto t0 = std::chrono::steady_clock::now();
            value = q();
            const std::chrono::duration<double, std::milli> dt = std::chrono::steady_clock::now() - t0;
            best = std::min(best, dt.count());
        }
        std::printf("RESULT\t%s\t%.3f\t%s\n", name.c_str(), best, value.c_str());
        std::fflush(stdout);
    }
    return 0;
}
