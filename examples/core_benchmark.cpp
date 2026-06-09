#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <vega/vega.hpp>

namespace {

struct Config {
    std::size_t n = 5'000'000;
    std::size_t partitions = 32;
    std::size_t workers = 8;
    std::size_t repeats = 5;
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
    }
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

}  // namespace

int main(int argc, char** argv) {
    const Config cfg = parse_args(argc, argv);
    std::vector<int64_t> data(cfg.n);
    for (std::size_t i = 0; i < cfg.n; ++i) data[i] = static_cast<int64_t>(i);

    vega::SparkContext ctx(cfg.workers);

    auto run_once = [&]() -> std::size_t {
        return ctx.parallelize(data, cfg.partitions)
            .map([](const int64_t& x) { return (x * 31 + 7) % 1'000'003; })
            .filter([](const int64_t& x) { return x % 7 == 0; })
            .flatMap([](const int64_t& x) { return std::vector<int64_t>{x, x + 1, x + 2}; })
            .count();
    };

    // Warmup.
    (void)run_once();

    std::vector<double> times;
    times.reserve(cfg.repeats);
    std::size_t last_count = 0;

    for (std::size_t i = 0; i < cfg.repeats; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        last_count = run_once();
        const auto t1 = std::chrono::steady_clock::now();
        const double sec = std::chrono::duration<double>(t1 - t0).count();
        times.push_back(sec);
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
