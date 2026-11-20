#pragma once

// Aggregate functions and the two-phase hash aggregation behind
// DataFrame::group_by(...).agg(...):
//
//   1. partial:  each partition groups its rows and keeps per-group state
//                (sum, count, sum+count for avg, min, max, count+mean+M2 for
//                variance);
//   2. shuffle:  partial states move to the partition that owns their key;
//   3. final:    states for the same key are merged and finalized.
//
// Semantics follow Spark SQL: aggregates skip nulls; sum/avg/min/max/
// variance of no (non-null) values is null; count never is. Result types:
// sum(bigint) -> bigint, sum(double) -> double, count -> bigint, avg and
// variance/stddev -> double, min/max keep the input type. Null grouping keys
// form their own group.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vega/dataframe/expr.hpp"

namespace vega::df {

enum class AggFn { Sum, Count, CountAll, Avg, Min, Max, VarSamp, VarPop, StddevSamp, StddevPop, CountDistinct };

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
    const std::string& alias_name() const { return alias_; }

    // Spark-style output names: "sum(qty)", "count(1)", "avg((a * b))".
    std::string name() const {
        if (!alias_.empty()) return alias_;
        if (fn_ == AggFn::CountAll) return "count(1)";
        if (fn_ == AggFn::CountDistinct) return "count(DISTINCT " + input_->name() + ")";
        return std::string(fn_name(fn_)) + "(" + input_->name() + ")";
    }

    static const char* fn_name(AggFn f) {
        switch (f) {
        case AggFn::Sum: return "sum";
        case AggFn::Count: return "count";
        case AggFn::CountAll: return "count_all";
        case AggFn::Avg: return "avg";
        case AggFn::Min: return "min";
        case AggFn::Max: return "max";
        case AggFn::VarSamp: return "var_samp";
        case AggFn::VarPop: return "var_pop";
        case AggFn::StddevSamp: return "stddev_samp";
        case AggFn::StddevPop: return "stddev_pop";
        case AggFn::CountDistinct: return "count_distinct";
        }
        return "?";
    }

    // As a window function: sum(col("x")).over(Window::partition_by(...)).
    Expr over(const WindowSpec& spec) const {
        switch (fn_) {
        case AggFn::Sum:
        case AggFn::Count:
        case AggFn::CountAll:
        case AggFn::Avg:
        case AggFn::Min:
        case AggFn::Max: break;
        default: throw std::invalid_argument(name() + " is not supported as a window function");
        }
        std::vector<Expr> args;
        if (input_) args.push_back(*input_);
        Expr w = Expr::window(fn_name(fn_), std::move(args), std::make_shared<const WindowSpec>(spec));
        return alias_.empty() ? w : w.alias(alias_);
    }

    std::string debug_string() const {
        return std::string(fn_name(fn_)) + "(" + (input_ ? input_->debug_string() : "") + ")" +
               (alias_.empty() ? "" : " AS " + alias_);
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
inline AggExpr mean(Expr e) { return avg(std::move(e)); }
inline AggExpr min(Expr e) { return AggExpr(AggFn::Min, std::move(e)); }
inline AggExpr max(Expr e) { return AggExpr(AggFn::Max, std::move(e)); }
inline AggExpr variance(Expr e) { return AggExpr(AggFn::VarSamp, std::move(e)); }
inline AggExpr var_samp(Expr e) { return AggExpr(AggFn::VarSamp, std::move(e)); }
inline AggExpr var_pop(Expr e) { return AggExpr(AggFn::VarPop, std::move(e)); }
inline AggExpr stddev(Expr e) { return AggExpr(AggFn::StddevSamp, std::move(e)); }
inline AggExpr stddev_samp(Expr e) { return AggExpr(AggFn::StddevSamp, std::move(e)); }
inline AggExpr stddev_pop(Expr e) { return AggExpr(AggFn::StddevPop, std::move(e)); }
// Exact distinct count; planned as a two-level aggregation (see DataFrame).
inline AggExpr count_distinct(Expr e) { return AggExpr(AggFn::CountDistinct, std::move(e)); }

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
    if (keys.size() == 1 && keys[0].is_dict()) {  // dense code -> group array
        const Column& k = keys[0];
        const auto& codes = k.codes();
        const uint8_t* valid = k.validity();
        const size_t d = k.dict_size();
        std::vector<uint32_t> map(d + 1, GroupTable::kEmpty);  // last slot: null
        for (size_t r = 0; r < rows; ++r) {
            const size_t slot = valid && !valid[r] ? d : codes[r];
            uint32_t& id = map[slot];
            if (id == GroupTable::kEmpty) {
                id = static_cast<uint32_t>(g.first.size());
                g.first.push_back(static_cast<uint32_t>(r));
            }
            g.gid[r] = id;
        }
        // Distinct codes can carry equal values (dictionaries need not be
        // unique); regroup the representatives by value if that happens.
        if (g.count() > 1) {
            const Column reps = k.take(g.first);
            const KeyHasher hash({reps});
            const KeyEq eq({reps}, {reps});
            GroupTable t(g.count());
            std::vector<uint32_t> remap(g.count());
            std::vector<uint32_t> first, rep;  // per merged group: row, index into reps
            for (size_t i = 0; i < g.count(); ++i) {
                auto [id, inserted] = t.find_or_insert(hash(i), [&](uint32_t c) { return eq(i, rep[c]); });
                if (inserted) {
                    first.push_back(g.first[i]);
                    rep.push_back(static_cast<uint32_t>(i));
                }
                remap[i] = id;
            }
            if (first.size() != g.count()) {
                for (auto& id : g.gid) id = remap[id];
                g.first = std::move(first);
            }
        }
        return g;
    }
    if (keys.size() == 1 && keys[0].type() == DType::Int64 && !keys[0].has_nulls()) {  // fast path
        const auto& v = keys[0].values<int64_t>();
        Int64GroupTable table(std::min<size_t>(rows, 4096));
        for (size_t r = 0; r < rows; ++r) {
            auto [id, inserted] = table.find_or_insert(v[r]);
            if (inserted) g.first.push_back(static_cast<uint32_t>(r));
            g.gid[r] = id;
        }
        return g;
    }
    auto plain_int = [](const Column& c) { return c.type() == DType::Int64 && !c.has_nulls() && !c.is_dict(); };
    if (keys.size() == 2 && plain_int(keys[0]) && plain_int(keys[1])) {  // two integer keys
        const auto& a = keys[0].values<int64_t>();
        const auto& b = keys[1].values<int64_t>();
        Int64PairGroupTable table(std::min<size_t>(rows, 4096));
        for (size_t r = 0; r < rows; ++r) {
            auto [id, inserted] = table.find_or_insert(a[r], b[r]);
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

// Sum of the valid values per group; null where a group has none.
inline Column sum_column(const Column& in, const Groups& g) {
    const uint8_t* valid = in.validity();
    BoolVec seen(g.count(), valid ? 0 : 1);
    Column out = in.visit([&](const auto& v) -> Column {
        using T = typename std::decay_t<decltype(v)>::value_type;
        if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, double>) {
            std::vector<T> st(g.count(), T{});
            if (!valid) {
                for (size_t r = 0; r < v.size(); ++r) st[g.gid[r]] += v[r];
            } else {
                for (size_t r = 0; r < v.size(); ++r)
                    if (valid[r]) {
                        st[g.gid[r]] += v[r];
                        seen[g.gid[r]] = 1;
                    }
            }
            return Column(std::move(st));
        } else {
            throw std::invalid_argument("sum needs a number");
        }
    });
    return valid ? out.with_validity(std::make_shared<const BoolVec>(std::move(seen))) : out;
}

// Count of valid values (or of rows) per group.
inline Column count_column(const Column* in, const Groups& g) {
    std::vector<int64_t> st(g.count(), 0);
    const uint8_t* valid = in ? in->validity() : nullptr;
    if (!valid) {
        for (uint32_t id : g.gid) ++st[id];
    } else {
        for (size_t r = 0; r < g.gid.size(); ++r) st[g.gid[r]] += valid[r] ? 1 : 0;
    }
    return Column(std::move(st));
}

// Sum of int64 states per group (merging counts).
inline Column sum_counts(const Column& in, const Groups& g) {
    std::vector<int64_t> st(g.count(), 0);
    const auto& v = in.values<int64_t>();
    for (size_t r = 0; r < v.size(); ++r) st[g.gid[r]] += v[r];
    return Column(std::move(st));
}

// min/max of the valid values per group; null where a group has none.
template<bool IsMin>
Column extreme(const Column& in, const Groups& g) {
    const uint8_t* valid = in.validity();
    BoolVec seen(g.count(), 0);
    Column out = in.visit([&](const auto& v) -> Column {
        using T = typename std::decay_t<decltype(v)>::value_type;
        std::vector<T> st(g.count());
        for (size_t r = 0; r < v.size(); ++r) {
            if (valid && !valid[r]) continue;
            const uint32_t id = g.gid[r];
            T& s = st[id];
            if (!seen[id]) {
                s = v[r];
                seen[id] = 1;
            } else if (IsMin ? v[r] < s : s < v[r]) {
                s = v[r];
            }
        }
        return Column(std::move(st));
    });
    return out.with_validity(std::make_shared<const BoolVec>(std::move(seen)));
}

// Welford states (n, mean, M2) per group from raw values.
inline void moments(const Column& in, const Groups& g, Column& n_out, Column& mean_out, Column& m2_out) {
    const Column d = cast_column(in, DType::Float64);
    const auto& v = d.values<double>();
    const uint8_t* valid = d.validity();
    std::vector<int64_t> n(g.count(), 0);
    std::vector<double> mean(g.count(), 0.0), m2(g.count(), 0.0);
    for (size_t r = 0; r < v.size(); ++r) {
        if (valid && !valid[r]) continue;
        const uint32_t id = g.gid[r];
        const double x = v[r];
        const double delta = x - mean[id];
        ++n[id];
        mean[id] += delta / static_cast<double>(n[id]);
        m2[id] += delta * (x - mean[id]);
    }
    n_out = Column(std::move(n));
    mean_out = Column(std::move(mean));
    m2_out = Column(std::move(m2));
}

// Merge Welford states per group (Chan et al.).
inline void merge_moments(const Column& nc, const Column& meanc, const Column& m2c, const Groups& g,
                          std::vector<int64_t>& n, std::vector<double>& mean, std::vector<double>& m2) {
    n.assign(g.count(), 0);
    mean.assign(g.count(), 0.0);
    m2.assign(g.count(), 0.0);
    const auto& nv = nc.values<int64_t>();
    const auto& mv = meanc.values<double>();
    const auto& qv = m2c.values<double>();
    for (size_t r = 0; r < nv.size(); ++r) {
        if (nv[r] == 0) continue;
        const uint32_t id = g.gid[r];
        const double na = static_cast<double>(n[id]), nb = static_cast<double>(nv[r]);
        const double delta = mv[r] - mean[id];
        const double nab = na + nb;
        mean[id] += delta * nb / nab;
        m2[id] += qv[r] + delta * delta * na * nb / nab;
        n[id] += nv[r];
    }
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
        for (const auto& e : keys) {
            if (e.contains_window()) throw std::invalid_argument("window function in a grouping key");
            key_schema.push_back({e.name(), e.type(in)});
        }
        partial_schema = key_schema;
        output_schema = key_schema;
        for (size_t i = 0; i < aggs.size(); ++i) {
            const auto& ag = aggs[i];
            DType t = DType::Int64;
            if (ag.input()) {
                if (ag.input()->contains_window())
                    throw std::invalid_argument("window function inside an aggregate: " + ag.name());
                t = ag.input()->type(in);
            }
            input_types.push_back(t);
            const std::string s = "__s" + std::to_string(i);
            const bool num = Expr::numeric(t);
            switch (ag.fn()) {
            case AggFn::Sum:
                if (!num) throw std::invalid_argument("sum needs a number: " + ag.name());
                partial_schema.push_back({s, t});
                output_schema.push_back({ag.name(), t});
                break;
            case AggFn::Count:
            case AggFn::CountAll:
                partial_schema.push_back({s, DType::Int64});
                output_schema.push_back({ag.name(), DType::Int64});
                break;
            case AggFn::Avg:
                if (!num) throw std::invalid_argument("avg needs a number: " + ag.name());
                partial_schema.push_back({s, DType::Float64});
                partial_schema.push_back({"__c" + std::to_string(i), DType::Int64});
                output_schema.push_back({ag.name(), DType::Float64});
                break;
            case AggFn::Min:
            case AggFn::Max:
                if (t == DType::Bool) throw std::invalid_argument("min/max need an orderable type: " + ag.name());
                partial_schema.push_back({s, t});
                output_schema.push_back({ag.name(), t});
                break;
            case AggFn::VarSamp:
            case AggFn::VarPop:
            case AggFn::StddevSamp:
            case AggFn::StddevPop:
                if (!num) throw std::invalid_argument(ag.name() + " needs a number");
                partial_schema.push_back({"__n" + std::to_string(i), DType::Int64});
                partial_schema.push_back({"__m" + std::to_string(i), DType::Float64});
                partial_schema.push_back({"__q" + std::to_string(i), DType::Float64});
                output_schema.push_back({ag.name(), DType::Float64});
                break;
            case AggFn::CountDistinct:
                throw std::invalid_argument("count_distinct must be planned by the DataFrame");
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
        for (const auto& ag : aggs) {
            switch (ag.fn()) {
            case AggFn::Sum: out.push_back(detail::sum_column(ag.input()->eval(in), g)); break;
            case AggFn::Count: {
                const Column v = ag.input()->eval(in);
                out.push_back(detail::count_column(&v, g));
                break;
            }
            case AggFn::CountAll: out.push_back(detail::count_column(nullptr, g)); break;
            case AggFn::Avg: {
                const Column v = cast_column(ag.input()->eval(in), DType::Float64);
                out.push_back(detail::sum_column(v, g).with_validity(nullptr));
                out.push_back(detail::count_column(&v, g));
                break;
            }
            case AggFn::Min: out.push_back(detail::extreme<true>(ag.input()->eval(in), g)); break;
            case AggFn::Max: out.push_back(detail::extreme<false>(ag.input()->eval(in), g)); break;
            case AggFn::VarSamp:
            case AggFn::VarPop:
            case AggFn::StddevSamp:
            case AggFn::StddevPop: {
                Column n, m, q;
                detail::moments(ag.input()->eval(in), g, n, m, q);
                out.push_back(n);
                out.push_back(m);
                out.push_back(q);
                break;
            }
            case AggFn::CountDistinct: throw std::logic_error("count_distinct");
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
        if (empty_global) {
            // A global aggregate over no rows still yields one row: counts
            // are 0, everything else is null.
            std::vector<Column> out;
            for (size_t i = 0; i < aggs.size(); ++i) {
                const auto f = aggs[i].fn();
                const DType t = output_schema[i].type;
                if (f == AggFn::Count || f == AggFn::CountAll) out.push_back(Column(std::vector<int64_t>{0}));
                else out.push_back(Column::nulls(t, 1));
            }
            return Batch(output_schema, std::move(out), 1);
        }
        std::vector<Column> out;
        for (const auto& c : key_cols) out.push_back(c.take(g.first));
        size_t col = nk;
        for (const auto& ag : aggs) {
            const Column& s = parts.column(col++);
            switch (ag.fn()) {
            case AggFn::Sum: out.push_back(detail::sum_column(s, g)); break;
            case AggFn::Count:
            case AggFn::CountAll: out.push_back(detail::sum_counts(s, g)); break;
            case AggFn::Avg: {
                const Column& c = parts.column(col++);
                const auto sums = detail::sum_column(s, g).values<double>();
                const auto counts = detail::sum_counts(c, g).values<int64_t>();
                std::vector<double> res(sums.size());
                BoolVec valid(sums.size());
                for (size_t k = 0; k < res.size(); ++k) {
                    valid[k] = counts[k] > 0;
                    res[k] = counts[k] > 0 ? sums[k] / static_cast<double>(counts[k]) : 0.0;
                }
                out.push_back(Column(std::move(res), std::move(valid)));
                break;
            }
            case AggFn::Min: out.push_back(detail::extreme<true>(s, g)); break;
            case AggFn::Max: out.push_back(detail::extreme<false>(s, g)); break;
            case AggFn::VarSamp:
            case AggFn::VarPop:
            case AggFn::StddevSamp:
            case AggFn::StddevPop: {
                const Column& m = parts.column(col++);
                const Column& q = parts.column(col++);
                std::vector<int64_t> n;
                std::vector<double> mean, m2;
                detail::merge_moments(s, m, q, g, n, mean, m2);
                const bool samp = ag.fn() == AggFn::VarSamp || ag.fn() == AggFn::StddevSamp;
                const bool sd = ag.fn() == AggFn::StddevSamp || ag.fn() == AggFn::StddevPop;
                std::vector<double> res(n.size());
                BoolVec valid(n.size());
                for (size_t k = 0; k < n.size(); ++k) {
                    const int64_t denom = samp ? n[k] - 1 : n[k];
                    valid[k] = denom > 0;
                    if (!valid[k]) continue;
                    res[k] = m2[k] / static_cast<double>(denom);
                    if (sd) res[k] = std::sqrt(res[k]);
                }
                out.push_back(Column(std::move(res), std::move(valid)));
                break;
            }
            case AggFn::CountDistinct: throw std::logic_error("count_distinct");
            }
        }
        return Batch(output_schema, std::move(out));
    }
};

}  // namespace vega::df
