#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

#include <vega/vega.hpp>

namespace {

struct Config {
    std::size_t n = 20'000'000;
    std::size_t partitions = 64;
    std::size_t workers = 8;
    std::size_t repeats = 3;
    std::size_t num_keys = 1'000'000;
};

Config parse_args(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto read_next = [&](std::size_t& dst) {
            if (i + 1 < argc) dst = static_cast<std::size_t>(std::strtoull(argv[++i], nullptr, 10));
        };
        if (a == "--n") read_next(cfg.n);
        else if (a == "--partitions") read_next(cfg.partitions);
        else if (a == "--workers") read_next(cfg.workers);
        else if (a == "--repeats") read_next(cfg.repeats);
        else if (a == "--num-keys") read_next(cfg.num_keys);
    }
    if (cfg.num_keys == 0) cfg.num_keys = 1;
    return cfg;
}

double median(std::vector<double> vals) {
    std::sort(vals.begin(), vals.end());
    const std::size_t m = vals.size() / 2;
    if (vals.size() % 2 == 1) return vals[m];
    return (vals[m - 1] + vals[m]) / 2.0;
}

double mean(const std::vector<double>& vals) {
    const double sum = std::accumulate(vals.begin(), vals.end(), 0.0);
    return vals.empty() ? 0.0 : (sum / static_cast<double>(vals.size()));
}

} // namespace

int main(int argc, char** argv) {
    const Config cfg = parse_args(argc, argv);
    std::vector<int64_t> data(cfg.n);
    for (std::size_t i = 0; i < cfg.n; ++i) data[i] = static_cast<int64_t>(i);

    vega::SparkContext ctx(cfg.workers);

    auto run_once = [&]() -> std::size_t {
        auto agg = ctx.parallelize(data, cfg.partitions)
            .map([num_keys = static_cast<int64_t>(cfg.num_keys)](const int64_t& x) {
                const int64_t k = (x * 1315423911LL + 7LL) % num_keys;
                const int64_t v = (x * 31LL + 7LL) % 97LL;
                return std::make_pair(k, v);
            })
            .reduceByKey([](const int64_t& a, const int64_t& b) { return a + b; }, cfg.partitions);
        return agg.count();
    };

    (void)run_once(); // warmup

    std::vector<double> times;
    times.reserve(cfg.repeats);
    std::size_t last_count = 0;
    for (std::size_t i = 0; i < cfg.repeats; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        last_count = run_once();
        const auto t1 = std::chrono::steady_clock::now();
        times.push_back(std::chrono::duration<double>(t1 - t0).count());
    }

    std::cout << "RESULT"
              << " count=" << last_count
              << " median_s=" << median(times)
              << " mean_s=" << mean(times)
              << " times_s=";
    for (std::size_t i = 0; i < times.size(); ++i) {
        if (i) std::cout << ",";
        std::cout << times[i];
    }
    std::cout << "\n";
    return 0;
}
