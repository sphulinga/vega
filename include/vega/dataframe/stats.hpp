#pragma once

// Column statistics for the optimizer: row counts are exact; distinct-value
// counts (NDV) come from HyperLogLog sketches built when a DataFrame is
// created or cached.
//
// A sketch has 2^12 one-byte registers (4 KB per column), so the estimate is
// within about 1.6% (one standard error) whatever the column size. Sketches
// of different batches merge by taking register maxima, so each partition is
// sketched in parallel and the results combined. Dictionary-encoded columns
// are counted exactly instead: the (hashes of the) dictionary values are
// collected and unioned across batches, at O(distinct values) cost.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "vega/dataframe/batch.hpp"

namespace vega::df {

struct ColumnStats {
    double ndv = -1;    // estimated distinct non-null values; < 0 = unknown
    size_t nulls = 0;   // exact null count
};

class HyperLogLog {
public:
    static constexpr int kBits = 12;
    static constexpr size_t kRegisters = size_t{1} << kBits;

    HyperLogLog() : reg_(kRegisters, 0) {}
    explicit HyperLogLog(std::vector<uint8_t> registers) : reg_(std::move(registers)) {
        if (reg_.size() != kRegisters) reg_.assign(kRegisters, 0);
    }

    // `h` must be a well-mixed 64-bit hash (e.g. KeyHasher output).
    void add(uint64_t h) {
        const size_t idx = static_cast<size_t>(h >> (64 - kBits));
        const uint64_t rest = (h << kBits) | (uint64_t{1} << (kBits - 1));  // sentinel bounds the rank
        const auto rank = static_cast<uint8_t>(leading_zeros(rest) + 1);
        if (rank > reg_[idx]) reg_[idx] = rank;
    }

    void merge(const HyperLogLog& o) {
        for (size_t i = 0; i < kRegisters; ++i) reg_[i] = std::max(reg_[i], o.reg_[i]);
    }

    double estimate() const {
        const double m = static_cast<double>(kRegisters);
        double sum = 0;
        size_t zeros = 0;
        for (uint8_t r : reg_) {
            sum += std::ldexp(1.0, -static_cast<int>(r));
            zeros += r == 0;
        }
        const double alpha = 0.7213 / (1.0 + 1.079 / m);
        const double e = alpha * m * m / sum;
        if (e <= 2.5 * m && zeros > 0) return m * std::log(m / static_cast<double>(zeros));  // linear counting
        return e;
    }

    const std::vector<uint8_t>& registers() const { return reg_; }

private:
    static int leading_zeros(uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
        return x == 0 ? 64 : __builtin_clzll(x);
#else
        int n = 0;
        for (uint64_t bit = uint64_t{1} << 63; bit && !(x & bit); bit >>= 1) ++n;
        return n;
#endif
    }
    std::vector<uint8_t> reg_;
};

// Per-column sketches and null counts of one batch.
struct BatchSketch {
    std::vector<std::vector<uint8_t>> registers;  // one HLL per column
    std::vector<int64_t> nulls;
    std::vector<int64_t> exact;                   // 1: `values` lists the column's distinct hashes
    std::vector<std::vector<int64_t>> values;  // hashes, stored signed for the wire format
};

inline BatchSketch sketch_batch(const Batch& b) {
    BatchSketch s;
    for (const auto& c : b.columns()) {
        HyperLogLog hll;
        std::vector<int64_t> distinct;
        const bool exact = c.is_dict();
        if (exact) {
            // the dictionary holds every value the column can take; only the
            // codes in use count
            const Column dict = c.dict_values();
            const KeyHasher h({dict});
            std::vector<uint8_t> used(dict.size(), 0);
            const uint8_t* valid = c.validity();
            const auto& codes = c.codes();
            for (size_t r = 0; r < codes.size(); ++r)
                if (!valid || valid[r]) used[codes[r]] = 1;
            for (size_t i = 0; i < dict.size(); ++i)
                if (used[i]) distinct.push_back(static_cast<int64_t>(h(i)));
            std::sort(distinct.begin(), distinct.end());
            distinct.erase(std::unique(distinct.begin(), distinct.end()), distinct.end());
        } else {
            const KeyHasher h({c});
            const uint8_t* valid = c.validity();
            for (size_t r = 0; r < c.size(); ++r)
                if (!valid || valid[r]) hll.add(h(r));
        }
        s.registers.push_back(hll.registers());
        s.nulls.push_back(static_cast<int64_t>(c.null_count()));
        s.exact.push_back(exact ? 1 : 0);
        s.values.push_back(std::move(distinct));
    }
    return s;
}

inline std::vector<ColumnStats> merge_sketches(const std::vector<BatchSketch>& parts, size_t ncols) {
    std::vector<HyperLogLog> hll(ncols);
    std::vector<ColumnStats> out(ncols);
    std::vector<bool> exact(ncols, true);
    std::vector<std::vector<int64_t>> values(ncols);
    for (const auto& p : parts) {
        for (size_t c = 0; c < ncols && c < p.registers.size(); ++c) {
            out[c].nulls += static_cast<size_t>(p.nulls[c]);
            if (p.exact[c]) {
                values[c].insert(values[c].end(), p.values[c].begin(), p.values[c].end());
            } else {
                exact[c] = false;
                hll[c].merge(HyperLogLog(p.registers[c]));
            }
        }
    }
    for (size_t c = 0; c < ncols; ++c) {
        auto& v = values[c];
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
        if (exact[c]) {
            out[c].ndv = static_cast<double>(v.size());
        } else {
            for (int64_t h : v) hll[c].add(static_cast<uint64_t>(h));  // batches that were exact
            out[c].ndv = std::round(hll[c].estimate());
        }
    }
    return out;
}

}  // namespace vega::df

namespace vega {

template<>
struct Serializer<df::BatchSketch> {
    static void serialize(const df::BatchSketch& s, ByteBuffer& buf) {
        Serializer<std::vector<std::vector<uint8_t>>>::serialize(s.registers, buf);
        Serializer<std::vector<int64_t>>::serialize(s.nulls, buf);
        Serializer<std::vector<int64_t>>::serialize(s.exact, buf);
        Serializer<std::vector<std::vector<int64_t>>>::serialize(s.values, buf);
    }
    static df::BatchSketch deserialize(const ByteBuffer& buf, size_t& off) {
        df::BatchSketch s;
        s.registers = Serializer<std::vector<std::vector<uint8_t>>>::deserialize(buf, off);
        s.nulls = Serializer<std::vector<int64_t>>::deserialize(buf, off);
        s.exact = Serializer<std::vector<int64_t>>::deserialize(buf, off);
        s.values = Serializer<std::vector<std::vector<int64_t>>>::deserialize(buf, off);
        return s;
    }
};

}  // namespace vega
