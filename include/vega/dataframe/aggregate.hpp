#pragma once

// Aggregate functions and the two-phase hash aggregation behind
// DataFrame::group_by(...).agg(...):
//
//   1. partial:  each partition groups its rows and keeps per-group state
//                (sum, count, sum+count for avg, min, max);
//   2. shuffle:  partial states move to the partition that owns their key;
//   3. final:    states for the same key are merged and finalized.
//
// Result types follow Spark SQL: sum(bigint) -> bigint, sum(double) ->
// double, count -> bigint, avg -> double, min/max keep the input type.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "vega/dataframe/expr.hpp"

namespace vega::df {

enum class AggFn { Sum, Count, CountAll, Avg, Min, Max };

class AggExpr {
public:
    AggExpr(AggFn fn, std::optional<Expr> input) : fn_(fn), input_(std::move(input)) {}

    AggExpr alias(std::string name) const {
        AggExpr a = *this;
        a.alias_ = std::move(name);
        return a;
    }

    AggFn fn() const { return fn_; }
    const std::optional<Expr>& input() const { return input_; }

    // Spark-style output names: "sum(qty)", "count(1)", "avg((a * b))".
    std::string name() const {
        if (!alias_.empty()) return alias_;
        switch (fn_) {
        case AggFn::Sum: return "sum(" + input_->name() + ")";
        case AggFn::Count: return "count(" + input_->name() + ")";
        case AggFn::CountAll: return "count(1)";
        case AggFn::Avg: return "avg(" + input_->name() + ")";
        case AggFn::Min: return "min(" + input_->name() + ")";
        case AggFn::Max: return "max(" + input_->name() + ")";
        }
        return "?";
    }

private:
    AggFn fn_;
    std::optional<Expr> input_;
    std::string alias_;
};

inline AggExpr sum(Expr e) { return AggExpr(AggFn::Sum, std::move(e)); }
inline AggExpr count(Expr e) { return AggExpr(AggFn::Count, std::move(e)); }
inline AggExpr count_all() { return AggExpr(AggFn::CountAll, std::nullopt); }
inline AggExpr avg(Expr e) { return AggExpr(AggFn::Avg, std::move(e)); }
inline AggExpr min(Expr e) { return AggExpr(AggFn::Min, std::move(e)); }
inline AggExpr max(Expr e) { return AggExpr(AggFn::Max, std::move(e)); }

namespace detail {

// Row -> group id assignment for one batch.
struct Groups {
    std::vector<uint32_t> gid;    // group of each row
    std::vector<uint32_t> first;  // a representative row of each group
    size_t count() const { return first.size(); }
};

inline Groups group_rows(const std::vector<Column>& keys, size_t rows) {
    Groups g;
    g.gid.resize(rows);
    if (keys.empty()) {  // global aggregate: one group
        if (rows > 0) g.first.push_back(0);
        return g;
    }
    if (keys.size() == 1 && keys[0].type() == DType::Int64) {  // fast path
        const auto& v = keys[0].values<int64_t>();
        Int64GroupTable table(std::min<size_t>(rows, 4096));
        for (size_t r = 0; r < rows; ++r) {
            auto [id, inserted] = table.find_or_insert(v[r]);
            if (inserted) g.first.push_back(static_cast<uint32_t>(r));
            g.gid[r] = id;
        }
        return g;
    }
    GroupTable table(std::min<size_t>(rows, 4096));
    const KeyHasher hash(keys);
    const KeyEq eq(keys, keys);
    for (size_t r = 0; r < rows; ++r) {
        auto [id, inserted] = table.find_or_insert(hash(r), [&](uint32_t c) { return eq(r, g.first[c]); });
        if (inserted) g.first.push_back(static_cast<uint32_t>(r));
        g.gid[r] = id;
    }
    return g;
}

template<typename T, typename Op>
Column fold(const Column& in, const Groups& g, T init, Op op) {
    std::vector<T> st(g.count(), init);
    const auto& v = in.values<T>();
    for (size_t r = 0; r < v.size(); ++r) st[g.gid[r]] = op(st[g.gid[r]], v[r]);
    return Column(std::move(st));
}

// min/max state over any orderable type; groups are never empty, so each
// starts from its representative row.
template<bool IsMin>
Column extreme(const Column& in, const Groups& g) {
    return in.visit([&](const auto& v) -> Column {
        using T = typename std::decay_t<decltype(v)>::value_type;
        std::vector<T> st;
        st.reserve(g.count());
        for (uint32_t r : g.first) st.push_back(v[r]);
        for (size_t r = 0; r < v.size(); ++r) {
            T& s = st[g.gid[r]];
            if (IsMin ? v[r] < s : s < v[r]) s = v[r];
        }
        return Column(std::move(st));
    });
}

inline Column count_rows(const Groups& g) {
    std::vector<int64_t> st(g.count(), 0);
    for (uint32_t id : g.gid) ++st[id];
    return Column(std::move(st));
}

inline Column sum_column(const Column& in, const Groups& g) {
    if (in.type() == DType::Int64)
        return fold<int64_t>(in, g, 0, [](int64_t a, int64_t b) { return a + b; });
    return fold<double>(in, g, 0.0, [](double a, double b) { return a + b; });
}

}  // namespace detail

// Everything needed to run one aggregation, resolved against the input schema.
struct AggPlan {
    std::vector<Expr> keys;
    std::vector<AggExpr> aggs;
    Schema key_schema;      // grouping columns
    std::vector<DType> input_types;
    Schema partial_schema;  // keys + per-aggregate state columns
    Schema output_schema;   // keys + results

    AggPlan(std::vector<Expr> k, std::vector<AggExpr> a, const Schema& in)
        : keys(std::move(k)), aggs(std::move(a)) {
        for (const auto& e : keys) key_schema.push_back({e.name(), e.type(in)});
        partial_schema = key_schema;
        output_schema = key_schema;
        for (size_t i = 0; i < aggs.size(); ++i) {
            const auto& ag = aggs[i];
            DType t = DType::Int64;
            if (ag.input()) t = ag.input()->type(in);
            input_types.push_back(t);
            const std::string s = "__s" + std::to_string(i);
            switch (ag.fn()) {
            case AggFn::Sum:
                if (t != DType::Int64 && t != DType::Float64)
                    throw std::invalid_argument("sum needs a number: " + ag.name());
                partial_schema.push_back({s, t});
                output_schema.push_back({ag.name(), t});
                break;
            case AggFn::Count:
            case AggFn::CountAll:
                partial_schema.push_back({s, DType::Int64});
                output_schema.push_back({ag.name(), DType::Int64});
                break;
            case AggFn::Avg:
                if (t != DType::Int64 && t != DType::Float64)
                    throw std::invalid_argument("avg needs a number: " + ag.name());
                partial_schema.push_back({s, DType::Float64});
                partial_schema.push_back({"__c" + std::to_string(i), DType::Int64});
                output_schema.push_back({ag.name(), DType::Float64});
                break;
            case AggFn::Min:
            case AggFn::Max:
                if (t == DType::Bool)
                    throw std::invalid_argument("min/max need an orderable type: " + ag.name());
                partial_schema.push_back({s, t});
                output_schema.push_back({ag.name(), t});
                break;
            }
        }
    }

    // Phase 1: group one input batch and compute partial states.
    Batch partial(const Batch& in) const {
        std::vector<Column> key_cols;
        for (const auto& e : keys) key_cols.push_back(e.eval(in));
        const detail::Groups g = detail::group_rows(key_cols, in.num_rows());
        std::vector<Column> out;
        for (const auto& c : key_cols) out.push_back(c.take(g.first));
        for (size_t i = 0; i < aggs.size(); ++i) {
            const auto& ag = aggs[i];
            switch (ag.fn()) {
            case AggFn::Sum: out.push_back(detail::sum_column(ag.input()->eval(in), g)); break;
            case AggFn::Count:
            case AggFn::CountAll: out.push_back(detail::count_rows(g)); break;
            case AggFn::Avg: {
                Column v = ag.input()->eval(in);
                if (v.type() == DType::Int64) {  // accumulate as double
                    const auto& iv = v.values<int64_t>();
                    v = Column(std::vector<double>(iv.begin(), iv.end()));
                }
                out.push_back(detail::sum_column(v, g));
                out.push_back(detail::count_rows(g));
                break;
            }
            case AggFn::Min: out.push_back(detail::extreme<true>(ag.input()->eval(in), g)); break;
            case AggFn::Max: out.push_back(detail::extreme<false>(ag.input()->eval(in), g)); break;
            }
        }
        return Batch(partial_schema, std::move(out));
    }

    // Phases 2+3: merge partial states (all rows for a key are in `parts`)
    // and produce final results.
    Batch merge_final(const Batch& parts) const {
        const size_t nk = key_schema.size();
        std::vector<Column> key_cols(parts.columns().begin(),
                                     parts.columns().begin() + static_cast<std::ptrdiff_t>(nk));
        detail::Groups g = detail::group_rows(key_cols, parts.num_rows());
        const bool empty_global = nk == 0 && g.count() == 0;
        if (empty_global) {  // global aggregate over no rows: one row of zeros
            g.first.clear();
        }
        std::vector<Column> out;
        for (const auto& c : key_cols) out.push_back(c.take(g.first));
        size_t col = nk;
        for (size_t i = 0; i < aggs.size(); ++i) {
            const auto& ag = aggs[i];
            const Column& s = parts.column(col++);
            switch (ag.fn()) {
            case AggFn::Sum:
            case AggFn::Count:
            case AggFn::CountAll:
                out.push_back(empty_global ? zero(output_schema[nk + i].type)
                                           : detail::sum_column(s, g));
                break;
            case AggFn::Avg: {
                const Column& c = parts.column(col++);
                if (empty_global) {
                    out.push_back(Column(std::vector<double>{std::nan("")}));
                    break;
                }
                const auto sums = detail::sum_column(s, g).values<double>();
                const auto counts = detail::sum_column(c, g).values<int64_t>();
                std::vector<double> res(sums.size());
                for (size_t k = 0; k < res.size(); ++k)
                    res[k] = sums[k] / static_cast<double>(counts[k]);
                out.push_back(Column(std::move(res)));
                break;
            }
            case AggFn::Min:
            case AggFn::Max:
                if (empty_global)
                    throw std::runtime_error(ag.name() + " of an empty input (nulls are not supported)");
                out.push_back(ag.fn() == AggFn::Min ? detail::extreme<true>(s, g)
                                                    : detail::extreme<false>(s, g));
                break;
            }
        }
        return Batch(output_schema, std::move(out));
    }

private:
    static Column zero(DType t) {
        if (t == DType::Float64) return Column(std::vector<double>{0.0});
        return Column(std::vector<int64_t>{0});
    }
};

}  // namespace vega::df
