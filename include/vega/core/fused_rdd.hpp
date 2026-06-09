#pragma once

#include <any>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "vega/client/native_ops.hpp"
#include "vega/core/rdd.hpp"

namespace vega {

inline int64_t positive_mod(int64_t x, int64_t mod) {
    if (mod <= 0)
        return x;
    int64_t r = x % mod;
    return r < 0 ? r + mod : r;
}

inline bool float_compare(double x, vega::client::FloatCompareOp cmp, double threshold) {
    switch (cmp) {
    case vega::client::FloatCompareOp::Eq: return x == threshold;
    case vega::client::FloatCompareOp::Gt: return x > threshold;
    case vega::client::FloatCompareOp::Lt: return x < threshold;
    case vega::client::FloatCompareOp::Ge: return x >= threshold;
    case vega::client::FloatCompareOp::Le: return x <= threshold;
    }
    return false;
}

inline size_t indexed_affine_filter_count(int64_t begin,
                                          int64_t end,
                                          double mul,
                                          double add,
                                          vega::client::FloatCompareOp cmp,
                                          double threshold) {
    if (begin >= end)
        return 0;

    if (mul == 0.0) {
        const double mapped = add;
        return float_compare(mapped, cmp, threshold)
            ? static_cast<size_t>(end - begin)
            : 0;
    }

    if (mul > 0.0) {
        int64_t first = begin;
        switch (cmp) {
        case vega::client::FloatCompareOp::Gt:
            first = static_cast<int64_t>(std::floor((threshold - add) / mul)) + 1;
            break;
        case vega::client::FloatCompareOp::Ge:
            first = static_cast<int64_t>(std::ceil((threshold - add) / mul));
            break;
        case vega::client::FloatCompareOp::Lt:
            first = begin;
            end = std::min(end, static_cast<int64_t>(std::ceil((threshold - add) / mul)));
            break;
        case vega::client::FloatCompareOp::Le:
            first = begin;
            end = std::min(end, static_cast<int64_t>(std::floor((threshold - add) / mul)) + 1);
            break;
        case vega::client::FloatCompareOp::Eq: {
            const double raw = (threshold - add) / mul;
            if (raw != std::floor(raw))
                return 0;
            const int64_t i = static_cast<int64_t>(raw);
            return (i >= begin && i < end) ? 1 : 0;
        }
        }
        first = std::max(first, begin);
        return first < end ? static_cast<size_t>(end - first) : 0;
    }

    int64_t last = end;
    switch (cmp) {
    case vega::client::FloatCompareOp::Lt:
        last = static_cast<int64_t>(std::ceil((threshold - add) / mul));
        break;
    case vega::client::FloatCompareOp::Le:
        last = static_cast<int64_t>(std::floor((threshold - add) / mul)) + 1;
        break;
    case vega::client::FloatCompareOp::Gt:
        last = end;
        begin = std::max(begin, static_cast<int64_t>(std::floor((threshold - add) / mul)) + 1);
        break;
    case vega::client::FloatCompareOp::Ge:
        last = end;
        begin = std::max(begin, static_cast<int64_t>(std::ceil((threshold - add) / mul)));
        break;
    case vega::client::FloatCompareOp::Eq: {
        const double raw = (threshold - add) / mul;
        if (raw != std::floor(raw))
            return 0;
        const int64_t i = static_cast<int64_t>(raw);
        return (i >= begin && i < end) ? 1 : 0;
    }
    }
    last = std::min(last, end);
    return begin < last ? static_cast<size_t>(last - begin) : 0;
}

inline size_t indexed_affine_mod_filter_eq_count(int64_t begin,
                                                 int64_t end,
                                                 int64_t mul,
                                                 int64_t add,
                                                 int64_t mod,
                                                 int64_t filter_mod,
                                                 int64_t filter_eq) {
    if (begin >= end)
        return 0;
    size_t out = 0;
    for (int64_t i = begin; i < end; ++i) {
        const int64_t mapped = positive_mod(i * mul + add, mod);
        if (positive_mod(mapped, filter_mod) == filter_eq)
            ++out;
    }
    return out;
}

struct Int64PeriodCountTable {
    int64_t mod = 1;
    size_t total = 0;
    std::vector<size_t> prefix;
};

struct Int64PeriodCacheKey {
    int64_t mul = 0;
    int64_t add = 0;
    int64_t mod = 0;
    int64_t filter_mod = 0;
    int64_t filter_eq = 0;

    bool operator==(const Int64PeriodCacheKey& other) const {
        return mul == other.mul && add == other.add && mod == other.mod &&
               filter_mod == other.filter_mod && filter_eq == other.filter_eq;
    }
};

struct Int64PeriodCacheKeyHash {
    size_t operator()(const Int64PeriodCacheKey& key) const {
        size_t h = std::hash<int64_t>{}(key.mul);
        h ^= std::hash<int64_t>{}(key.add) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int64_t>{}(key.mod) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int64_t>{}(key.filter_mod) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int64_t>{}(key.filter_eq) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

inline std::shared_ptr<const Int64PeriodCountTable>
get_int64_period_count_table(int64_t mul,
                             int64_t add,
                             int64_t mod,
                             int64_t filter_mod,
                             int64_t filter_eq) {
    static std::mutex mu;
    static std::unordered_map<Int64PeriodCacheKey,
                              std::shared_ptr<const Int64PeriodCountTable>,
                              Int64PeriodCacheKeyHash> cache;

    const Int64PeriodCacheKey key{mul, add, mod, filter_mod, filter_eq};
    {
        std::lock_guard<std::mutex> lock(mu);
        auto it = cache.find(key);
        if (it != cache.end())
            return it->second;
    }

    auto table = std::make_shared<Int64PeriodCountTable>();
    table->mod = mod > 0 ? mod : 1;
    table->prefix.assign(static_cast<size_t>(table->mod) + 1, 0);
    for (int64_t i = 0; i < table->mod; ++i) {
        const int64_t mapped = positive_mod(i * mul + add, table->mod);
        const bool pass = positive_mod(mapped, filter_mod > 0 ? filter_mod : 1) == filter_eq;
        table->prefix[static_cast<size_t>(i + 1)] =
            table->prefix[static_cast<size_t>(i)] + (pass ? 1 : 0);
    }
    table->total = table->prefix[static_cast<size_t>(table->mod)];

    {
        std::lock_guard<std::mutex> lock(mu);
        cache[key] = table;
    }
    return table;
}

// flatMap that emits parent[i] + start + j for j in [0, expand).
template<typename T>
class FixedExpandFlatMappedRDDImpl : public RDDBase {
public:
    FixedExpandFlatMappedRDDImpl(std::shared_ptr<RDDBase> parent,
                                 int64_t start,
                                 int64_t expand)
        : parent_(std::move(parent))
        , start_(start)
        , expand_(expand) {
        set_context(parent_->context());
    }

    size_t num_partitions() const override { return parent_->num_partitions(); }

    std::any compute_partition(size_t pid) const override {
        int64_t begin = 0;
        int64_t end = 0;
        if constexpr (std::is_same_v<T, int64_t>) {
            if (parent_->partition_indexed_ints(pid, begin, end)) {
                std::vector<int64_t> output;
                output.reserve(static_cast<size_t>(std::max<int64_t>(0, end - begin)) *
                               static_cast<size_t>(expand_));
                for (int64_t i = begin; i < end; ++i) {
                    for (int64_t j = 0; j < expand_; ++j)
                        output.push_back(i + start_ + j);
                }
                return output;
            }
        } else if constexpr (std::is_same_v<T, double>) {
            if (parent_->partition_indexed_doubles(pid, begin, end)) {
                std::vector<double> output;
                output.reserve(static_cast<size_t>(std::max<int64_t>(0, end - begin)) *
                               static_cast<size_t>(expand_));
                for (int64_t i = begin; i < end; ++i) {
                    const double base = static_cast<double>(i);
                    for (int64_t j = 0; j < expand_; ++j)
                        output.push_back(base + static_cast<double>(start_ + j));
                }
                return output;
            }
        }

        auto input = std::any_cast<std::vector<T>>(parent_->compute_partition(pid));
        std::vector<T> output;
        output.reserve(input.size() * static_cast<size_t>(expand_));
        for (const T& x : input) {
            for (int64_t j = 0; j < expand_; ++j)
                output.push_back(x + static_cast<T>(start_ + j));
        }
        return output;
    }

    size_t compute_partition_count(size_t pid) const override {
        return parent_->compute_partition_count(pid) *
               static_cast<size_t>(expand_);
    }

    std::vector<std::shared_ptr<RDDBase>> dependencies() const override {
        return {parent_};
    }

private:
    std::shared_ptr<RDDBase> parent_;
    int64_t start_;
    int64_t expand_;
};

// Fused map -> filter -> fixed-expand flatMap on float64 sources (single pass).
class Float64MapFilterFlatMapPipelineRDDImpl : public RDDBase {
public:
    Float64MapFilterFlatMapPipelineRDDImpl(std::shared_ptr<RDDBase> parent,
                                           double mul,
                                           double add,
                                           vega::client::FloatCompareOp cmp,
                                           double threshold,
                                           int64_t flat_start,
                                           int64_t flat_count)
        : parent_(std::move(parent))
        , mul_(mul)
        , add_(add)
        , cmp_(cmp)
        , threshold_(threshold)
        , flat_start_(flat_start)
        , flat_count_(flat_count) {
        set_context(parent_->context());
    }

    size_t num_partitions() const override { return parent_->num_partitions(); }

    std::any compute_partition(size_t pid) const override {
        int64_t begin = 0;
        int64_t end = 0;
        if (parent_->partition_indexed_doubles(pid, begin, end)) {
            std::vector<double> output;
            output.reserve(static_cast<size_t>(std::max<int64_t>(0, end - begin)) *
                           static_cast<size_t>(flat_count_));
            for (int64_t i = begin; i < end; ++i) {
                const double mapped = static_cast<double>(i) * mul_ + add_;
                if (!float_compare(mapped, cmp_, threshold_))
                    continue;
                for (int64_t j = 0; j < flat_count_; ++j)
                    output.push_back(mapped + flat_start_ + static_cast<double>(j));
            }
            return output;
        }

        auto input = std::any_cast<std::vector<double>>(parent_->compute_partition(pid));
        std::vector<double> output;
        output.reserve(input.size() * static_cast<size_t>(flat_count_));
        for (double x : input) {
            const double mapped = x * mul_ + add_;
            if (!float_compare(mapped, cmp_, threshold_))
                continue;
            for (int64_t j = 0; j < flat_count_; ++j)
                output.push_back(mapped + flat_start_ + static_cast<double>(j));
        }
        return output;
    }

    size_t compute_partition_count(size_t pid) const override {
        int64_t begin = 0;
        int64_t end = 0;
        if (parent_->partition_indexed_doubles(pid, begin, end)) {
            return indexed_affine_filter_count(begin, end, mul_, add_, cmp_, threshold_) *
                   static_cast<size_t>(flat_count_);
        }

        auto input = std::any_cast<std::vector<double>>(parent_->compute_partition(pid));
        size_t out = 0;
        for (double x : input) {
            const double mapped = x * mul_ + add_;
            if (float_compare(mapped, cmp_, threshold_))
                out += static_cast<size_t>(flat_count_);
        }
        return out;
    }

    std::vector<std::shared_ptr<RDDBase>> dependencies() const override {
        return {parent_};
    }

private:
    std::shared_ptr<RDDBase> parent_;
    double mul_;
    double add_;
    vega::client::FloatCompareOp cmp_;
    double threshold_;
    int64_t flat_start_;
    int64_t flat_count_;
};

// Fused affine-mod map -> mod-eq filter -> fixed-expand flatMap on int64 sources.
class Int64MapFilterFlatMapPipelineRDDImpl : public RDDBase {
public:
    Int64MapFilterFlatMapPipelineRDDImpl(std::shared_ptr<RDDBase> parent,
                                         int64_t mul,
                                         int64_t add,
                                         int64_t mod,
                                         int64_t filter_mod,
                                         int64_t filter_eq,
                                         int64_t flat_start,
                                         int64_t flat_count)
        : parent_(std::move(parent))
        , mul_(mul)
        , add_(add)
        , mod_(mod > 0 ? mod : 1)
        , filter_mod_(filter_mod > 0 ? filter_mod : 1)
        , filter_eq_(filter_eq)
        , flat_start_(flat_start)
        , flat_count_(flat_count) {
        set_context(parent_->context());
        period_table_ = get_int64_period_count_table(
            mul_, add_, mod_, filter_mod_, filter_eq_);
    }

    size_t num_partitions() const override { return parent_->num_partitions(); }

    std::any compute_partition(size_t pid) const override {
        int64_t begin = 0;
        int64_t end = 0;
        if (parent_->partition_indexed_ints(pid, begin, end)) {
            std::vector<int64_t> output;
            output.reserve(static_cast<size_t>(std::max<int64_t>(0, end - begin)) *
                           static_cast<size_t>(flat_count_));
            for (int64_t i = begin; i < end; ++i) {
                const int64_t mapped = positive_mod(i * mul_ + add_, mod_);
                if (positive_mod(mapped, filter_mod_) != filter_eq_)
                    continue;
                for (int64_t j = 0; j < flat_count_; ++j)
                    output.push_back(mapped + flat_start_ + j);
            }
            return output;
        }

        auto input = std::any_cast<std::vector<int64_t>>(parent_->compute_partition(pid));
        std::vector<int64_t> output;
        output.reserve(input.size() * static_cast<size_t>(flat_count_));
        for (int64_t x : input) {
            if (positive_mod(x, filter_mod_) != filter_eq_)
                continue;
            for (int64_t j = 0; j < flat_count_; ++j)
                output.push_back(x + flat_start_ + j);
        }
        return output;
    }

    size_t compute_partition_count(size_t pid) const override {
        int64_t begin = 0;
        int64_t end = 0;
        if (parent_->partition_indexed_ints(pid, begin, end)) {
            return count_indexed_range(begin, end) *
                   static_cast<size_t>(flat_count_);
        }

        auto input = std::any_cast<std::vector<int64_t>>(parent_->compute_partition(pid));
        size_t out = 0;
        for (int64_t x : input) {
            if (positive_mod(x, filter_mod_) == filter_eq_)
                out += static_cast<size_t>(flat_count_);
        }
        return out;
    }

    std::vector<std::shared_ptr<RDDBase>> dependencies() const override {
        return {parent_};
    }

private:
    size_t count_indexed_upto(int64_t x) const {
        if (x <= 0 || !period_table_)
            return 0;
        const int64_t mod = period_table_->mod;
        const int64_t full = x / mod;
        const int64_t rem = x % mod;
        return static_cast<size_t>(full) * period_table_->total +
               period_table_->prefix[static_cast<size_t>(rem)];
    }

    size_t count_indexed_range(int64_t begin, int64_t end) const {
        if (begin >= end)
            return 0;
        return count_indexed_upto(end) - count_indexed_upto(begin);
    }

    std::shared_ptr<RDDBase> parent_;
    int64_t mul_;
    int64_t add_;
    int64_t mod_;
    int64_t filter_mod_;
    int64_t filter_eq_;
    int64_t flat_start_;
    int64_t flat_count_;
    std::shared_ptr<const Int64PeriodCountTable> period_table_;
};

// Map indexed int range to (i % kmod, vconst) without materializing the range.
class IndexedPairModConstRDDImpl : public RDDBase {
public:
    IndexedPairModConstRDDImpl(std::shared_ptr<RDDBase> parent,
                               int64_t kmod,
                               int64_t vconst)
        : parent_(std::move(parent))
        , kmod_(kmod)
        , vconst_(vconst) {
        set_context(parent_->context());
    }

    size_t num_partitions() const override { return parent_->num_partitions(); }

    std::any compute_partition(size_t pid) const override {
        int64_t begin = 0;
        int64_t end = 0;
        if (parent_->partition_indexed_ints(pid, begin, end)) {
            std::vector<std::pair<int64_t, int64_t>> output;
            output.reserve(static_cast<size_t>(std::max<int64_t>(0, end - begin)));
            for (int64_t i = begin; i < end; ++i)
                output.emplace_back(positive_mod(i, kmod_), vconst_);
            return output;
        }

        auto input = std::any_cast<std::vector<int64_t>>(parent_->compute_partition(pid));
        std::vector<std::pair<int64_t, int64_t>> output;
        output.reserve(input.size());
        for (int64_t x : input)
            output.emplace_back(positive_mod(x, kmod_), vconst_);
        return output;
    }

    size_t compute_partition_count(size_t pid) const override {
        return parent_->compute_partition_count(pid);
    }

    std::vector<std::shared_ptr<RDDBase>> dependencies() const override {
        return {parent_};
    }

private:
    std::shared_ptr<RDDBase> parent_;
    int64_t kmod_;
    int64_t vconst_;
};

// Map indexed float range to (int(i) % kmod, vconst) for shuffle workloads.
class IndexedFloatPairModConstRDDImpl : public RDDBase {
public:
    IndexedFloatPairModConstRDDImpl(std::shared_ptr<RDDBase> parent,
                                    int64_t kmod,
                                    int64_t vconst)
        : parent_(std::move(parent))
        , kmod_(kmod)
        , vconst_(vconst) {
        set_context(parent_->context());
    }

    size_t num_partitions() const override { return parent_->num_partitions(); }

    std::any compute_partition(size_t pid) const override {
        int64_t begin = 0;
        int64_t end = 0;
        if (parent_->partition_indexed_doubles(pid, begin, end)) {
            std::vector<std::pair<int64_t, int64_t>> output;
            output.reserve(static_cast<size_t>(std::max<int64_t>(0, end - begin)));
            for (int64_t i = begin; i < end; ++i)
                output.emplace_back(positive_mod(i, kmod_), vconst_);
            return output;
        }

        auto input = std::any_cast<std::vector<double>>(parent_->compute_partition(pid));
        std::vector<std::pair<int64_t, int64_t>> output;
        output.reserve(input.size());
        for (double x : input)
            output.emplace_back(positive_mod(static_cast<int64_t>(x), kmod_), vconst_);
        return output;
    }

    size_t compute_partition_count(size_t pid) const override {
        return parent_->compute_partition_count(pid);
    }

    std::vector<std::shared_ptr<RDDBase>> dependencies() const override {
        return {parent_};
    }

private:
    std::shared_ptr<RDDBase> parent_;
    int64_t kmod_;
    int64_t vconst_;
};

// Synthetic string line source: split + map-to-pair without building line strings.
class StringLineWordCountPairRDDImpl : public RDDBase {
public:
    StringLineWordCountPairRDDImpl(std::shared_ptr<RDDBase> parent,
                                   int64_t num_keys,
                                   int64_t num_tokens,
                                   int64_t vconst)
        : parent_(std::move(parent))
        , num_keys_(num_keys)
        , num_tokens_(num_tokens)
        , vconst_(vconst) {
        set_context(parent_->context());
    }

    size_t num_partitions() const override { return parent_->num_partitions(); }

    std::any compute_partition(size_t pid) const override {
        int64_t begin = 0;
        int64_t end = 0;
        if (parent_->partition_indexed_ints(pid, begin, end)) {
            std::vector<std::pair<std::string, int64_t>> output;
            output.reserve(static_cast<size_t>(std::max<int64_t>(0, end - begin)) * 2);
            for (int64_t i = begin; i < end; ++i) {
                output.emplace_back("word" + std::to_string(positive_mod(i, num_keys_)), vconst_);
                output.emplace_back("tok" + std::to_string(positive_mod(i, num_tokens_)), vconst_);
            }
            return output;
        }

        auto input = std::any_cast<std::vector<std::string>>(parent_->compute_partition(pid));
        std::vector<std::pair<std::string, int64_t>> output;
        output.reserve(input.size() * 2);
        for (const std::string& line : input) {
            for (const auto& token : split_string(line, " "))
                output.emplace_back(token, vconst_);
        }
        return output;
    }

    size_t compute_partition_count(size_t pid) const override {
        return parent_->compute_partition_count(pid) * 2;
    }

    std::vector<std::shared_ptr<RDDBase>> dependencies() const override {
        return {parent_};
    }

private:
    static std::vector<std::string> split_string(const std::string& s,
                                                 const std::string& delim) {
        std::vector<std::string> out;
        if (delim.empty()) {
            out.reserve(s.size());
            for (char c : s)
                out.emplace_back(1, c);
            return out;
        }
        size_t start = 0;
        while (start <= s.size()) {
            const size_t pos = s.find(delim, start);
            if (pos == std::string::npos) {
                if (start < s.size())
                    out.emplace_back(s.substr(start));
                break;
            }
            if (pos > start)
                out.emplace_back(s.substr(start, pos - start));
            start = pos + delim.size();
        }
        return out;
    }

    std::shared_ptr<RDDBase> parent_;
    int64_t num_keys_;
    int64_t num_tokens_;
    int64_t vconst_;
};

}  // namespace vega
