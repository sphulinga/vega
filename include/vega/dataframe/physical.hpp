#pragma once

// Physical execution of optimized DataFrame plans on the RDD engine.
//
//   * Narrow operators (scan column selection, projections, filters and the
//     probe side of broadcast joins) are fused into a single per-batch stage.
//   * Joins are broadcast hash joins when one side is estimated to be small
//     (broadcast_threshold(), 32 MB by default, like Spark's
//     autoBroadcastJoinThreshold), otherwise shuffle hash joins: both sides
//     are hash-partitioned by key through the engine shuffle and joined per
//     partition. Build-side columns are emitted dictionary-encoded (codes into
//     the build batch), so a join copies no strings.
//   * Aggregates are two-phase hash aggregates around a shuffle.
//   * Windows shuffle by partition key, sort each partition and evaluate.
//   * order_by + limit runs as a parallel top-N; a full sort sorts partitions
//     in parallel and k-way merges them.

#include <algorithm>
#include <atomic>
#include <functional>
#include <memory>
#include <optional>
#include <queue>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "vega/dataframe/optimizer.hpp"

namespace vega::df {

// Size (estimated bytes) under which a join side is broadcast. 0 disables
// broadcast joins (except for cross joins).
inline std::atomic<double>& broadcast_threshold() {
    static std::atomic<double> bytes{32.0 * 1024 * 1024};
    return bytes;
}

enum class JoinStrategy { BroadcastRight, BroadcastLeft, Shuffle };

inline JoinStrategy choose_join(const PlanNode& j) {
    const double t = broadcast_threshold().load();
    const double lb = plan::estimate_bytes(j.input(0)), rb = plan::estimate_bytes(j.input(1));
    const bool bl = lb <= t, br = rb <= t;
    switch (j.join_type) {
    case JoinType::Inner:
        if (br && (!bl || rb <= lb)) return JoinStrategy::BroadcastRight;
        if (bl) return JoinStrategy::BroadcastLeft;
        return JoinStrategy::Shuffle;
    case JoinType::Left:
    case JoinType::LeftSemi:
    case JoinType::LeftAnti: return br ? JoinStrategy::BroadcastRight : JoinStrategy::Shuffle;
    case JoinType::Right: return bl ? JoinStrategy::BroadcastLeft : JoinStrategy::Shuffle;
    case JoinType::Full: return JoinStrategy::Shuffle;
    case JoinType::Cross: return (!br && bl) ? JoinStrategy::BroadcastLeft : JoinStrategy::BroadcastRight;
    }
    return JoinStrategy::Shuffle;
}

namespace exec {

// ---------------------------------------------------------------------------
// Hash join kernel
// ---------------------------------------------------------------------------

struct JoinIndex {
    Batch build;
    std::vector<Column> keys;
    GroupTable table;
    std::vector<uint32_t> head, tail;  // per distinct key: first/last build row
    std::vector<uint32_t> next;        // per build row: next row with the same key

    JoinIndex(Batch b, const std::vector<std::string>& key_names) : build(std::move(b)), table(build.num_rows()) {
        for (const auto& k : key_names) keys.push_back(build.column(k));
        const size_t n = build.num_rows();
        next.assign(n, kNullRow);
        if (keys.empty()) return;
        const KeyHasher hash(keys);
        const KeyEq eq(keys, keys);
        for (size_t r = 0; r < n; ++r) {
            if (any_null(keys, r)) continue;  // null keys never match
            auto [slot, inserted] = table.find_or_insert(hash(r), [&](uint32_t s) { return eq(r, head[s]); });
            if (inserted) {
                head.push_back(static_cast<uint32_t>(r));
                tail.push_back(static_cast<uint32_t>(r));
            } else {
                next[tail[slot]] = static_cast<uint32_t>(r);
                tail[slot] = static_cast<uint32_t>(r);
            }
        }
    }
};

enum class ProbeMode { Inner, Outer, Semi, Anti, Cross };

// Matches of probe rows against the index: (probe row, build row) pairs;
// build row kNullRow marks an unmatched probe row (Outer mode).
inline void probe(const Batch& p, const std::vector<std::string>& key_names, const JoinIndex& idx, ProbeMode mode,
                  std::vector<uint32_t>& pi, std::vector<uint32_t>& bi, std::vector<uint8_t>* matched) {
    const size_t n = p.num_rows();
    pi.reserve(n);
    if (mode != ProbeMode::Semi && mode != ProbeMode::Anti) bi.reserve(n);
    if (mode == ProbeMode::Cross) {
        const size_t m = idx.build.num_rows();
        for (size_t r = 0; r < n; ++r)
            for (size_t b = 0; b < m; ++b) {
                pi.push_back(static_cast<uint32_t>(r));
                bi.push_back(static_cast<uint32_t>(b));
            }
        return;
    }
    std::vector<Column> keys;
    for (const auto& k : key_names) keys.push_back(p.column(k));
    const KeyHasher hash(keys);
    const KeyEq eq(keys, idx.keys);
    for (size_t r = 0; r < n; ++r) {
        uint32_t slot = GroupTable::kEmpty;
        if (!idx.head.empty() && !any_null(keys, r))
            slot = idx.table.find(hash(r), [&](uint32_t s) { return eq(r, idx.head[s]); });
        const auto row = static_cast<uint32_t>(r);
        switch (mode) {
        case ProbeMode::Semi:
            if (slot != GroupTable::kEmpty) pi.push_back(row);
            break;
        case ProbeMode::Anti:
            if (slot == GroupTable::kEmpty) pi.push_back(row);
            break;
        default:
            if (slot == GroupTable::kEmpty) {
                if (mode == ProbeMode::Outer) {
                    pi.push_back(row);
                    bi.push_back(kNullRow);
                }
                break;
            }
            for (uint32_t b = idx.head[slot]; b != kNullRow; b = idx.next[b]) {
                pi.push_back(row);
                bi.push_back(b);
                if (matched) (*matched)[b] = 1;
            }
        }
    }
}

// Build-side rows for the output, without copying: a dictionary column of
// codes into the build column (kNullRow -> null).
inline Column gather_build(const Column& c, const std::vector<uint32_t>& rows) {
    const size_t n = rows.size();
    if (c.size() == 0) return Column::nulls(c.type(), n);
    std::vector<uint32_t> codes(n);
    BoolVec valid;
    bool nulls = false;
    for (uint32_t r : rows)
        if (r == kNullRow || !c.is_valid(r)) {
            nulls = true;
            break;
        }
    if (nulls) valid.resize(n);
    const uint32_t* base = c.is_dict() ? c.codes().data() : nullptr;
    for (size_t i = 0; i < n; ++i) {
        const uint32_t r = rows[i];
        if (r == kNullRow) {
            codes[i] = 0;
            continue;
        }
        codes[i] = base ? base[r] : r;
        if (nulls) valid[i] = c.is_valid(r);
    }
    const Column dict = c.is_dict() ? c.dict_values() : c.with_validity(nullptr);
    return Column::dictionary(dict, std::make_shared<const std::vector<uint32_t>>(std::move(codes)),
                              nulls ? std::make_shared<const BoolVec>(std::move(valid)) : nullptr);
}

inline Column gather_probe(const Column& c, const std::vector<uint32_t>& rows, bool identity) {
    return identity ? c : c.take(rows);
}

inline bool is_identity(const std::vector<uint32_t>& rows, size_t n) {
    if (rows.size() != n) return false;
    for (size_t i = 0; i < n; ++i)
        if (rows[i] != i) return false;
    return true;
}

// Assemble join output columns in `out` schema order. Columns named in
// `probe_names` come from the probe batch, the others from the build batch.
inline Batch assemble(const Schema& out, const Batch& p, const std::vector<uint32_t>& pi, const Batch& build,
                      const std::vector<uint32_t>& bi) {
    const bool ident = is_identity(pi, p.num_rows());
    std::vector<Column> cols;
    for (const auto& f : out) {
        if (schema_index(p.schema(), f.name) >= 0) cols.push_back(gather_probe(p.column(f.name), pi, ident));
        else cols.push_back(gather_build(build.column(f.name), bi));
    }
    return Batch(out, std::move(cols), pi.size());
}

// Rows of the build side no probe row matched, with the probe side null.
inline Batch unmatched_build(const Schema& out, const Schema& probe_schema, const JoinIndex& idx,
                             const std::vector<uint8_t>& matched) {
    std::vector<uint32_t> rows;
    for (size_t r = 0; r < matched.size(); ++r)
        if (!matched[r]) rows.push_back(static_cast<uint32_t>(r));
    std::vector<Column> cols;
    for (const auto& f : out) {
        if (schema_index(probe_schema, f.name) >= 0) cols.push_back(Column::nulls(f.type, rows.size()));
        else cols.push_back(idx.build.column(f.name).take(rows));
    }
    return Batch(out, std::move(cols), rows.size());
}

// ---------------------------------------------------------------------------
// Sorting
// ---------------------------------------------------------------------------

// Compare row ra of key columns a with row rb of key columns b.
inline int compare_rows(const std::vector<Column>& a, size_t ra, const std::vector<Column>& b, size_t rb,
                        const std::vector<SortKey>& keys) {
    for (size_t k = 0; k < keys.size(); ++k) {
        const bool na = !a[k].is_valid(ra), nb = !b[k].is_valid(rb);
        if (na || nb) {
            if (na && nb) continue;
            const int c = na ? -1 : 1;
            return keys[k].nulls_first ? c : -c;
        }
        const int c = a[k].visit([&](const auto& va) {
            using T = typename std::decay_t<decltype(va)>::value_type;
            const auto& vb = b[k].values<T>();
            return va[ra] < vb[rb] ? -1 : (vb[rb] < va[ra] ? 1 : 0);
        });
        if (c != 0) return keys[k].ascending ? c : -c;
    }
    return 0;
}

// Row order of a batch under `keys`; keeps only the first `limit` rows.
inline std::vector<uint32_t> sort_order(const Batch& b, const std::vector<SortKey>& keys,
                                        std::optional<size_t> limit) {
    std::vector<Column> kc;
    for (const auto& k : keys) kc.push_back(k.expr.eval(b).flat());
    std::vector<uint32_t> idx(b.num_rows());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = static_cast<uint32_t>(i);
    auto less = [&](uint32_t x, uint32_t y) {
        const int c = compare_rows(kc, x, kc, y, keys);
        return c != 0 ? c < 0 : x < y;
    };
    if (limit && *limit < idx.size()) {
        std::partial_sort(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(*limit), idx.end(), less);
        idx.resize(*limit);
    } else {
        std::sort(idx.begin(), idx.end(), less);
    }
    return idx;
}

// K-way merge of sorted runs.
inline Batch merge_runs(const std::vector<Batch>& runs, const Schema& schema, const std::vector<SortKey>& keys,
                        std::optional<size_t> limit) {
    std::vector<std::vector<Column>> run_keys(runs.size());
    std::vector<size_t> offset(runs.size());
    size_t total = 0;
    for (size_t r = 0; r < runs.size(); ++r) {
        for (const auto& k : keys) run_keys[r].push_back(k.expr.eval(runs[r]).flat());
        offset[r] = total;
        total += runs[r].num_rows();
    }
    const size_t want = limit ? std::min(*limit, total) : total;
    auto greater = [&](const std::pair<size_t, size_t>& x, const std::pair<size_t, size_t>& y) {
        const int c = compare_rows(run_keys[x.first], x.second, run_keys[y.first], y.second, keys);
        return c != 0 ? c > 0 : x.first > y.first;
    };
    std::priority_queue<std::pair<size_t, size_t>, std::vector<std::pair<size_t, size_t>>, decltype(greater)> heap(
        greater);
    for (size_t r = 0; r < runs.size(); ++r)
        if (runs[r].num_rows() > 0) heap.emplace(r, 0);
    std::vector<uint32_t> order;
    order.reserve(want);
    while (!heap.empty() && order.size() < want) {
        auto [r, i] = heap.top();
        heap.pop();
        order.push_back(static_cast<uint32_t>(offset[r] + i));
        if (i + 1 < runs[r].num_rows()) heap.emplace(r, i + 1);
    }
    return Batch::concat(runs, schema).take(order);
}

// ---------------------------------------------------------------------------
// Window functions
// ---------------------------------------------------------------------------

// Pick a[i] where use_b[i] is 0, else b[i].
inline Column choose(const std::vector<uint8_t>& use_b, const Column& a, const Column& b) {
    const size_t n = use_b.size();
    BoolVec valid(n);
    Column out = a.visit([&](const auto& av) -> Column {
        using T = typename std::decay_t<decltype(av)>::value_type;
        const auto& bv = b.values<T>();
        std::vector<T> o(n);
        for (size_t i = 0; i < n; ++i) {
            o[i] = use_b[i] ? bv[i] : av[i];
            valid[i] = use_b[i] ? b.is_valid(i) : a.is_valid(i);
        }
        return Column(std::move(o));
    });
    return out.with_validity(std::make_shared<const BoolVec>(std::move(valid)));
}

// Evaluate the window expressions over one shuffle partition. All
// expressions share one window specification.
inline Batch compute_window(const Batch& in, const std::vector<Expr>& exprs, const Schema& out_schema) {
    const auto& spec = *exprs.at(0).strip_alias().window_spec();
    const size_t n = in.num_rows();
    std::vector<Column> pk;
    for (const auto& e : spec.partition) pk.push_back(e.eval(in).flat());
    std::vector<SortKey> keys;
    for (const auto& e : spec.partition) keys.push_back(asc(e));
    for (const auto& k : spec.order) keys.push_back(k);
    const std::vector<uint32_t> idx = sort_order(in, keys, std::nullopt);
    std::vector<Column> ok;
    for (const auto& k : spec.order) ok.push_back(k.expr.eval(in).flat());
    const KeyEq peq(pk, pk);

    // Partition and peer-group boundaries in sorted order.
    std::vector<size_t> part_start(n), peer_start(n), peer_end(n);
    for (size_t i = 0; i < n; ++i) {
        const bool new_part = i == 0 || !peq(idx[i], idx[i - 1]);
        part_start[i] = new_part ? i : part_start[i - 1];
        const bool new_peer = new_part || spec.order.empty() ||
                              compare_rows(ok, idx[i], ok, idx[i - 1], spec.order) != 0;
        peer_start[i] = new_peer ? i : peer_start[i - 1];
    }
    for (size_t i = n; i-- > 0;) {
        const bool last = i + 1 == n || peer_start[i + 1] != peer_start[i] || part_start[i + 1] != part_start[i];
        peer_end[i] = last ? i + 1 : peer_end[i + 1];
    }
    std::vector<size_t> part_end(n);
    for (size_t i = n; i-- > 0;) part_end[i] = (i + 1 == n || part_start[i + 1] != part_start[i]) ? i + 1 : part_end[i + 1];

    std::vector<Column> cols;
    for (const auto& c : in.columns()) cols.push_back(c.take(idx));
    const Schema& in_schema = in.schema();

    for (const auto& we : exprs) {
        const Expr w = we.strip_alias();
        const std::string& f = w.func_name();
        const DType t = w.type(in_schema);
        if (f == "row_number" || f == "rank" || f == "dense_rank") {
            std::vector<int64_t> o(n);
            int64_t dense = 0;
            for (size_t i = 0; i < n; ++i) {
                if (i == part_start[i]) dense = 0;
                if (i == peer_start[i]) ++dense;
                if (f == "row_number") o[i] = static_cast<int64_t>(i - part_start[i] + 1);
                else if (f == "rank") o[i] = static_cast<int64_t>(peer_start[i] - part_start[i] + 1);
                else o[i] = dense;
            }
            cols.push_back(Column(std::move(o)));
            continue;
        }
        if (f == "lag" || f == "lead") {
            const Column arg = cast_column(w.children()[0].eval(in), t);
            const int64_t off = f == "lag" ? -w.window_offset() : w.window_offset();
            std::vector<uint32_t> src(n);
            std::vector<uint8_t> outside(n, 0);
            for (size_t i = 0; i < n; ++i) {
                const int64_t j = static_cast<int64_t>(i) + off;
                if (j < static_cast<int64_t>(part_start[i]) || j >= static_cast<int64_t>(part_end[i])) {
                    src[i] = kNullRow;
                    outside[i] = 1;
                } else {
                    src[i] = idx[static_cast<size_t>(j)];
                }
            }
            Column r = arg.take(src);
            if (w.children().size() > 1) {
                const Column def = cast_column(w.children()[1].eval(in), t).take(idx);
                r = choose(outside, r, def);
            }
            cols.push_back(r);
            continue;
        }
        // Aggregates: whole partition without ORDER BY, else a running
        // aggregate over peer groups (RANGE UNBOUNDED PRECEDING .. CURRENT ROW).
        const bool counting = f == "count" || f == "count_all";
        Column arg;
        if (f != "count_all") {
            arg = w.children()[0].eval(in);
            if (f == "avg") arg = cast_column(arg, DType::Float64);
            arg = arg.flat();
        }
        const uint8_t* av = f == "count_all" ? nullptr : arg.validity();
        auto valid_at = [&](size_t i) { return f == "count_all" || !av || av[idx[i]]; };
        if (counting) {
            std::vector<int64_t> o(n);
            int64_t cnt = 0;
            size_t i = 0;
            while (i < n) {
                if (i == part_start[i]) cnt = 0;
                const size_t end = spec.order.empty() ? part_end[i] : peer_end[i];
                for (size_t k = i; k < end; ++k) cnt += valid_at(k) ? 1 : 0;
                for (size_t k = i; k < end; ++k) o[k] = cnt;
                i = end;
            }
            cols.push_back(Column(std::move(o)));
            continue;
        }
        Column res = arg.visit([&](const auto& v) -> Column {
            using T = typename std::decay_t<decltype(v)>::value_type;
            std::vector<T> o(n);
            BoolVec valid(n);
            T acc{};
            int64_t cnt = 0;
            size_t i = 0;
            while (i < n) {
                if (i == part_start[i]) {
                    acc = T{};
                    cnt = 0;
                }
                const size_t end = spec.order.empty() ? part_end[i] : peer_end[i];
                for (size_t k = i; k < end; ++k) {
                    if (!valid_at(k)) continue;
                    const T& x = v[idx[k]];
                    if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, double>) {
                        if (f == "sum" || f == "avg") acc = cnt ? acc + x : x;
                        else if (f == "min") acc = (cnt == 0 || x < acc) ? x : acc;
                        else if (f == "max") acc = (cnt == 0 || acc < x) ? x : acc;
                    } else {
                        if (f == "min") acc = (cnt == 0 || x < acc) ? x : acc;
                        else if (f == "max") acc = (cnt == 0 || acc < x) ? x : acc;
                    }
                    ++cnt;
                }
                for (size_t k = i; k < end; ++k) {
                    valid[k] = cnt > 0;
                    if constexpr (std::is_same_v<T, double>)
                        o[k] = (f == "avg" && cnt > 0) ? acc / static_cast<double>(cnt) : acc;
                    else
                        o[k] = acc;
                }
                i = end;
            }
            return Column(std::move(o), std::move(valid));
        });
        cols.push_back(cast_column(res, t));
    }
    return Batch(out_schema, std::move(cols), n);
}

// ---------------------------------------------------------------------------
// Planner / executor
// ---------------------------------------------------------------------------

using Rdd = vega::RDD<Batch>;
using Step = std::function<Batch(const Batch&)>;

struct Phys {
    Rdd rdd;
    size_t parts;
};

struct Pipeline {
    Phys base;
    std::vector<Step> steps;
};

class Executor {
public:
    explicit Executor(vega::SparkContext* ctx) : ctx_(ctx) {}

    Phys run(const PlanPtr& n) { return materialize(pipe(n)); }

    // Execute and collect all batches.
    std::vector<Batch> collect(const PlanPtr& n) { return run(n).rdd.collect(); }

    Phys single(const Batch& b) {
        return {ctx_->parallelize(std::vector<Batch>{b}, 1), 1};
    }

private:
    using K = PlanNode::Kind;

    static Phys materialize(const Pipeline& p) {
        if (p.steps.empty()) return p.base;
        auto steps = std::make_shared<const std::vector<Step>>(p.steps);
        auto rdd = p.base.rdd.map([steps](const Batch& b) {
            Batch x = b;
            for (const auto& s : *steps) x = s(x);
            return x;
        });
        return {rdd, p.base.parts};
    }

    Pipeline pipe(const PlanPtr& n) {
        switch (n->kind) {
        case K::Scan: {
            Pipeline p{{n->source->rdd, n->source->partitions}, {}};
            if (n->scan_columns) {
                const auto cols = *n->scan_columns;
                p.steps.push_back([cols](const Batch& b) { return b.project(cols); });
            }
            return p;
        }
        case K::Empty: return {single(Batch::empty(n->schema)), {}};
        case K::Project: {
            Pipeline p = pipe(n->input());
            const auto exprs = n->exprs;
            const Schema schema = n->schema;
            p.steps.push_back([exprs, schema](const Batch& b) {
                std::vector<Column> cols;
                cols.reserve(exprs.size());
                for (const auto& e : exprs) cols.push_back(e.eval(b));
                return Batch(schema, std::move(cols), b.num_rows());
            });
            return p;
        }
        case K::Filter: {
            Pipeline p = pipe(n->input());
            const Expr pred = n->exprs[0];
            p.steps.push_back([pred](const Batch& b) {
                const Column mask = pred.eval(b);
                const auto& m = mask.values<uint8_t>();
                const uint8_t* v = mask.validity();
                std::vector<uint32_t> keep;
                keep.reserve(m.size());
                for (size_t i = 0; i < m.size(); ++i)
                    if (m[i] && (!v || v[i])) keep.push_back(static_cast<uint32_t>(i));
                return keep.size() == b.num_rows() ? b : b.take(keep);
            });
            return p;
        }
        case K::Join: {
            const JoinStrategy s = choose_join(*n);
            if (s == JoinStrategy::Shuffle) return {shuffle_join(n), {}};
            return broadcast_join(n, s == JoinStrategy::BroadcastLeft);
        }
        case K::Aggregate: return {aggregate(n), {}};
        case K::Sort: return {sort(n->input(), n->sort_keys, std::nullopt), {}};
        case K::Limit: {
            if (n->input()->kind == K::Sort)
                return {sort(n->input()->input(), n->input()->sort_keys, n->limit), {}};
            const size_t lim = n->limit;
            auto parts = run(n->input()).rdd.map([lim](const Batch& b) { return b.slice(0, lim); }).collect();
            return {single(Batch::concat(parts, n->schema).slice(0, lim)), {}};
        }
        case K::Union: {
            std::optional<Phys> acc;
            for (const auto& in : n->inputs) {
                Phys p = run(in);
                bool same = true;
                for (size_t i = 0; i < n->schema.size(); ++i) same = same && in->schema[i].name == n->schema[i].name;
                if (!same) {
                    std::vector<std::string> names;
                    for (const auto& f : n->schema) names.push_back(f.name);
                    p.rdd = p.rdd.map([names](const Batch& b) { return b.renamed(names); });
                }
                if (!acc) acc = p;
                else acc = Phys{acc->rdd.union_with(p.rdd), acc->parts + p.parts};
            }
            return {*acc, {}};
        }
        case K::Window: return {window(n), {}};
        case K::Repartition: return {repartition(n), {}};
        }
        throw std::logic_error("unknown plan node");
    }

    Pipeline broadcast_join(const PlanPtr& n, bool build_left) {
        const PlanPtr& build_plan = n->input(build_left ? 0 : 1);
        const PlanPtr& probe_plan = n->input(build_left ? 1 : 0);
        const auto& build_keys = build_left ? n->left_keys : n->right_keys;
        const auto probe_keys = build_left ? n->right_keys : n->left_keys;
        Batch build = Batch::concat(collect(build_plan), build_plan->schema);
        auto idx = std::make_shared<const JoinIndex>(std::move(build), build_keys);
        ProbeMode mode = ProbeMode::Inner;
        switch (n->join_type) {
        case JoinType::Inner: mode = ProbeMode::Inner; break;
        case JoinType::Left:
        case JoinType::Right: mode = ProbeMode::Outer; break;  // probe side is the preserved side
        case JoinType::LeftSemi: mode = ProbeMode::Semi; break;
        case JoinType::LeftAnti: mode = ProbeMode::Anti; break;
        case JoinType::Cross: mode = ProbeMode::Cross; break;
        case JoinType::Full: throw std::logic_error("full outer join cannot be broadcast");
        }
        Pipeline p = pipe(probe_plan);
        const Schema out = n->schema;
        p.steps.push_back([idx, probe_keys, mode, out](const Batch& b) {
            std::vector<uint32_t> pi, bi;
            probe(b, probe_keys, *idx, mode, pi, bi, nullptr);
            if (mode == ProbeMode::Semi || mode == ProbeMode::Anti) return b.take(pi);
            return assemble(out, b, pi, idx->build, bi);
        });
        return p;
    }

    // Hash-partition a side's batches by key into `parts` buckets, tagged
    // with the side (0 = left, 1 = right).
    static vega::RDD<std::pair<int, std::pair<int, Batch>>> bucketize(const Phys& in, const std::vector<std::string>& keys,
                                                                       size_t parts, int side) {
        return in.rdd.flatMap([keys, parts, side](const Batch& b) {
            std::vector<std::pair<int, std::pair<int, Batch>>> res;
            std::vector<Column> kc;
            for (const auto& k : keys) kc.push_back(b.column(k));
            const KeyHasher h(kc);
            std::vector<std::vector<uint32_t>> buckets(parts);
            for (size_t r = 0; r < b.num_rows(); ++r) buckets[(h(r) >> 32) % parts].push_back(static_cast<uint32_t>(r));
            for (size_t k = 0; k < parts; ++k)
                if (!buckets[k].empty()) res.push_back({static_cast<int>(k), {side, b.take(buckets[k])}});
            return res;
        });
    }

    Phys shuffle_join(const PlanPtr& n) {
        const Phys l = run(n->input(0)), r = run(n->input(1));
        const size_t parts = std::max(l.parts, r.parts);
        auto lt = bucketize(l, n->left_keys, parts, 0);
        auto rt = bucketize(r, n->right_keys, parts, 1);
        const Schema ls = n->input(0)->schema, rs = n->input(1)->schema, out = n->schema;
        const auto lk = n->left_keys, rk = n->right_keys;
        const JoinType jt = n->join_type;
        auto joined = lt.union_with(rt).groupByKey(parts).map(
            [ls, rs, out, lk, rk, jt](const std::pair<int, std::vector<std::pair<int, Batch>>>& kv) {
                std::vector<Batch> lb, rb;
                for (const auto& [side, b] : kv.second) (side == 0 ? lb : rb).push_back(b);
                const Batch left = Batch::concat(lb, ls), right = Batch::concat(rb, rs);
                const JoinIndex idx(right, rk);
                std::vector<uint32_t> pi, bi;
                std::vector<uint8_t> matched(right.num_rows(), 0);
                const bool track = jt == JoinType::Right || jt == JoinType::Full;
                ProbeMode mode = ProbeMode::Inner;
                if (jt == JoinType::Left || jt == JoinType::Full) mode = ProbeMode::Outer;
                if (jt == JoinType::LeftSemi) mode = ProbeMode::Semi;
                if (jt == JoinType::LeftAnti) mode = ProbeMode::Anti;
                probe(left, lk, idx, mode, pi, bi, track ? &matched : nullptr);
                if (mode == ProbeMode::Semi || mode == ProbeMode::Anti) return left.take(pi);
                Batch res = assemble(out, left, pi, idx.build, bi);
                if (track) res = Batch::concat({res, unmatched_build(out, ls, idx, matched)}, out);
                return res;
            });
        return {joined, parts};
    }

    Phys aggregate(const PlanPtr& n) {
        const Phys in = run(n->input());
        auto plan = std::make_shared<const AggPlan>(n->keys, n->aggs, n->input()->schema);
        const size_t nk = plan->key_schema.size();
        if (nk == 0) {
            // Global aggregate: one partial row per batch, merged on the
            // driver (always exactly one output row, even for no input).
            auto partials = in.rdd.map([plan](const Batch& b) { return plan->partial(b); }).collect();
            return single(plan->merge_final(Batch::concat(partials, plan->partial_schema)));
        }
        const size_t parts = in.parts;
        auto staged = in.rdd.flatMap([plan, nk, parts](const Batch& b) {
            Batch p = plan->partial(b);
            std::vector<std::pair<int, Batch>> out;
            std::vector<Column> keys(p.columns().begin(), p.columns().begin() + static_cast<std::ptrdiff_t>(nk));
            const KeyHasher h(keys);
            std::vector<std::vector<uint32_t>> buckets(parts);
            for (size_t r = 0; r < p.num_rows(); ++r) buckets[(h(r) >> 32) % parts].push_back(static_cast<uint32_t>(r));
            for (size_t k = 0; k < parts; ++k)
                if (!buckets[k].empty()) out.emplace_back(static_cast<int>(k), p.take(buckets[k]));
            return out;
        });
        auto merged = staged.groupByKey(parts).map([plan](const std::pair<int, std::vector<Batch>>& kv) {
            return plan->merge_final(Batch::concat(kv.second, plan->partial_schema));
        });
        return {merged, parts};
    }

    Phys sort(const PlanPtr& child, const std::vector<SortKey>& keys, std::optional<size_t> limit) {
        const auto k = std::make_shared<const std::vector<SortKey>>(keys);
        auto runs = run(child).rdd.map([k, limit](const Batch& b) { return b.take(sort_order(b, *k, limit)); }).collect();
        return single(merge_runs(runs, child->schema, keys, limit));
    }

    Phys window(const PlanPtr& n) {
        const Phys in = run(n->input());
        const auto& spec = *n->exprs.at(0).strip_alias().window_spec();
        const size_t parts = spec.partition.empty() ? 1 : in.parts;
        const auto part_exprs = spec.partition;
        auto staged = in.rdd.flatMap([part_exprs, parts](const Batch& b) {
            std::vector<std::pair<int, Batch>> out;
            if (parts == 1) {
                if (b.num_rows() > 0) out.emplace_back(0, b);
                return out;
            }
            std::vector<Column> kc;
            for (const auto& e : part_exprs) kc.push_back(e.eval(b));
            const KeyHasher h(kc);
            std::vector<std::vector<uint32_t>> buckets(parts);
            for (size_t r = 0; r < b.num_rows(); ++r) buckets[(h(r) >> 32) % parts].push_back(static_cast<uint32_t>(r));
            for (size_t k = 0; k < parts; ++k)
                if (!buckets[k].empty()) out.emplace_back(static_cast<int>(k), b.take(buckets[k]));
            return out;
        });
        const Schema in_schema = n->input()->schema, out_schema = n->schema;
        const auto exprs = n->exprs;
        auto res = staged.groupByKey(parts).map(
            [in_schema, out_schema, exprs](const std::pair<int, std::vector<Batch>>& kv) {
                return compute_window(Batch::concat(kv.second, in_schema), exprs, out_schema);
            });
        return {res, parts};
    }

    Phys repartition(const PlanPtr& n) {
        const Phys in = run(n->input());
        const size_t parts = n->partitions;
        const auto keys = n->keys;
        auto staged = in.rdd.flatMap([keys, parts](const Batch& b) {
            std::vector<std::pair<int, Batch>> out;
            std::vector<std::vector<uint32_t>> buckets(parts);
            if (keys.empty()) {
                for (size_t r = 0; r < b.num_rows(); ++r) buckets[r % parts].push_back(static_cast<uint32_t>(r));
            } else {
                std::vector<Column> kc;
                for (const auto& e : keys) kc.push_back(e.eval(b));
                const KeyHasher h(kc);
                for (size_t r = 0; r < b.num_rows(); ++r) buckets[(h(r) >> 32) % parts].push_back(static_cast<uint32_t>(r));
            }
            for (size_t k = 0; k < parts; ++k)
                if (!buckets[k].empty()) out.emplace_back(static_cast<int>(k), b.take(buckets[k]));
            return out;
        });
        const Schema s = n->schema;
        auto res = staged.groupByKey(parts).map(
            [s](const std::pair<int, std::vector<Batch>>& kv) { return Batch::concat(kv.second, s); });
        return {res, parts};
    }

    vega::SparkContext* ctx_;
};

// Physical plan rendering (for explain()).
inline void render_physical(const PlanPtr& n, std::ostringstream& os, int depth) {
    using K = PlanNode::Kind;
    std::string label = plan::node_label(*n);
    const std::string pad = std::string(static_cast<size_t>(depth) * 2, ' ') + (depth ? "+- " : "");
    switch (n->kind) {
    case K::Join: {
        const JoinStrategy s = choose_join(*n);
        const char* name = s == JoinStrategy::Shuffle ? "ShuffleHashJoin"
                           : s == JoinStrategy::BroadcastLeft ? "BroadcastHashJoin (build left)"
                                                              : "BroadcastHashJoin (build right)";
        label = std::string(name) + label.substr(4);
        break;
    }
    case K::Aggregate: label = "HashAggregate" + label.substr(9); break;
    case K::Limit:
        if (n->input()->kind == K::Sort) {
            os << pad << "TakeOrdered " << n->limit << " " << plan::node_label(*n->input()).substr(5) << "\n";
            for (const auto& in : n->input()->inputs) render_physical(in, os, depth + 1);
            return;
        }
        break;
    default: break;
    }
    os << pad << label << "\n";
    for (const auto& in : n->inputs) render_physical(in, os, depth + 1);
}

}  // namespace exec
}  // namespace vega::df
