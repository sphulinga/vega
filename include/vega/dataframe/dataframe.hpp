#pragma once

// Vega DataFrames: columnar, typed tables on top of the RDD engine.
//
//   using namespace vega::df;
//   auto sales = DataFrame::from_columns(ctx, {{"store", stores}, {"qty", qtys}}, 8);
//   auto top = sales.filter(col("qty") > 2)
//                   .group_by({col("store")})
//                   .agg({sum(col("qty")).alias("units"), count_all()})
//                   .order_by({desc(col("units"))})
//                   .limit(10)
//                   .collect();
//
// A DataFrame is an RDD of columnar Batches (one or more per partition) plus
// a schema. Operations compile to RDD stages and run in parallel on the actor
// system:
//
//   * select / with_column / filter / join are narrow operations. They are
//     queued and fused into a single per-batch stage, and only the columns a
//     later step reads are carried through it (column pruning), so e.g.
//     `df.filter(p).count()` never copies a column it does not test;
//   * group_by(...).agg(...) and distinct() run a two-phase hash aggregate
//     with an engine shuffle between the phases;
//   * order_by sorts partitions in parallel and k-way merges them; a
//     following limit(n) keeps only n rows per partition (top-N).
//
// Current limits: values are never null; joins are inner equi-joins using a
// broadcast hash join (the right side is collected once and shared, as Spark
// does for small tables); there is no SQL parser.

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <memory>
#include <optional>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vega/core/context.hpp"
#include "vega/dataframe/aggregate.hpp"

namespace vega::df {

struct SortKey {
    Expr expr;
    bool ascending = true;
};
inline SortKey asc(Expr e) { return {std::move(e), true}; }
inline SortKey desc(Expr e) { return {std::move(e), false}; }

class GroupedData;

namespace detail {

// The build side of a broadcast hash join.
struct JoinIndex {
    Batch build;
    std::vector<std::string> keys;        // key column names (both sides)
    std::vector<Column> key_cols;         // build-side key columns
    std::vector<std::string> right_cols;  // build-side non-key columns, in output order
    GroupTable table;                     // distinct build keys -> slot
    std::vector<std::vector<uint32_t>> rows;  // build rows of each distinct key
};

// One queued narrow operation.
struct NarrowOp {
    enum class Kind { Project, Filter, Join } kind;
    std::vector<Expr> exprs;  // Project: output expressions; Filter: {predicate}
    std::shared_ptr<const JoinIndex> join;
};

using NameSet = std::set<std::string>;

inline void add_refs(const Expr& e, NameSet& out) {
    std::vector<std::string> refs;
    e.referenced_columns(refs);
    out.insert(refs.begin(), refs.end());
}

// A narrow op specialized to the columns its consumers need.
struct PrunedOp {
    NarrowOp::Kind kind = NarrowOp::Kind::Project;
    std::vector<Expr> exprs;
    std::shared_ptr<const JoinIndex> join;
    std::vector<std::string> join_right;  // build columns to emit
};

inline Batch apply_filter(const Batch& b, const Expr& predicate) {
    const Column mask = predicate.eval(b);
    const auto& m = mask.values<uint8_t>();
    std::vector<uint32_t> keep;
    keep.reserve(m.size());
    for (size_t i = 0; i < m.size(); ++i)
        if (m[i]) keep.push_back(static_cast<uint32_t>(i));
    return keep.size() == b.num_rows() ? b : b.take(keep);
}

inline Batch apply_join(const Batch& b, const JoinIndex& idx, const std::vector<std::string>& right) {
    std::vector<Column> keys;
    for (const auto& k : idx.keys) keys.push_back(b.column(k));
    const KeyHasher hash(keys);
    const KeyEq eq(keys, idx.key_cols);
    std::vector<uint32_t> li, ri;
    li.reserve(b.num_rows());
    ri.reserve(b.num_rows());
    for (size_t r = 0; r < b.num_rows(); ++r) {
        const uint32_t slot =
            idx.table.find(hash(r), [&](uint32_t s) { return eq(r, idx.rows[s].front()); });
        if (slot == GroupTable::kEmpty) continue;
        for (uint32_t rr : idx.rows[slot]) {
            li.push_back(static_cast<uint32_t>(r));
            ri.push_back(rr);
        }
    }
    // Every left row matched exactly once (the common foreign-key join):
    // the left columns pass through untouched.
    bool identity = li.size() == b.num_rows();
    for (size_t i = 0; identity && i < li.size(); ++i) identity = li[i] == i;
    Schema schema = b.schema();
    std::vector<Column> cols;
    for (const auto& c : b.columns()) cols.push_back(identity ? c : c.take(li));
    for (const auto& name : right) {
        cols.push_back(idx.build.column(name).take(ri));
        schema.push_back({name, cols.back().type()});
    }
    return Batch(std::move(schema), std::move(cols), li.size());
}

}  // namespace detail

class DataFrame {
public:
    using Rdd = vega::RDD<Batch>;

    DataFrame(vega::SparkContext* ctx, Rdd rdd, Schema schema, size_t partitions)
        : ctx_(ctx), base_(std::move(rdd)), base_schema_(schema), schema_(std::move(schema)),
          ops_(std::make_shared<const std::vector<detail::NarrowOp>>()),
          parts_(std::max<size_t>(1, partitions)) {}

    // Split a batch into `partitions` contiguous slices.
    static DataFrame from_batch(vega::SparkContext& ctx, const Batch& batch, size_t partitions) {
        partitions = std::max<size_t>(1, partitions);
        const size_t rows = batch.num_rows();
        std::vector<Batch> slices;
        slices.reserve(partitions);
        for (size_t p = 0; p < partitions; ++p)
            slices.push_back(batch.slice(rows * p / partitions, rows * (p + 1) / partitions));
        auto rdd = ctx.parallelize(slices, partitions);
        return DataFrame(&ctx, std::move(rdd), batch.schema(), partitions);
    }

    static DataFrame from_columns(vega::SparkContext& ctx,
                                  std::vector<std::pair<std::string, Column>> columns,
                                  size_t partitions) {
        Schema schema;
        std::vector<Column> cols;
        for (auto& [name, c] : columns) {
            schema.push_back({name, c.type()});
            cols.push_back(std::move(c));
        }
        check_unique(schema);
        return from_batch(ctx, Batch(std::move(schema), std::move(cols)), partitions);
    }

    const Schema& schema() const { return schema_; }
    size_t num_partitions() const { return parts_; }

    // ----- transformations (lazy) -------------------------------------------

    DataFrame select(std::vector<Expr> exprs) const {
        const DataFrame& in = resolved();
        Schema out;
        for (const auto& e : exprs) out.push_back({e.name(), e.type(in.schema_)});
        check_unique(out);
        return in.with_op({detail::NarrowOp::Kind::Project, std::move(exprs), nullptr}, std::move(out));
    }

    // Add a column, or replace the one with the same name.
    DataFrame with_column(const std::string& name, const Expr& e) const {
        std::vector<Expr> exprs;
        bool replaced = false;
        for (const auto& f : resolved().schema_) {
            if (f.name == name) {
                exprs.push_back(e.alias(name));
                replaced = true;
            } else {
                exprs.push_back(col(f.name));
            }
        }
        if (!replaced) exprs.push_back(e.alias(name));
        return select(std::move(exprs));
    }

    DataFrame filter(const Expr& predicate) const {
        const DataFrame& in = resolved();
        if (predicate.type(in.schema_) != DType::Bool)
            throw std::invalid_argument("filter needs a boolean expression: " + predicate.name());
        return in.with_op({detail::NarrowOp::Kind::Filter, {predicate}, nullptr}, in.schema_);
    }
    DataFrame where(const Expr& predicate) const { return filter(predicate); }

    GroupedData group_by(std::vector<Expr> keys) const;

    // Aggregate the whole DataFrame into one row.
    DataFrame agg(std::vector<AggExpr> aggs) const { return aggregate({}, std::move(aggs)); }

    // Inner equi-join on columns with the same names on both sides
    // (like Spark's df.join(other, Seq("k"))). The right side is collected
    // and broadcast to every partition of the left side; output columns are
    // the left side's, then the right side's non-key columns.
    DataFrame join(const DataFrame& right, const std::vector<std::string>& on) const {
        const DataFrame& left = resolved();
        if (on.empty()) throw std::invalid_argument("join needs at least one key column");
        auto idx = std::make_shared<detail::JoinIndex>();
        idx->build = right.collect();
        idx->keys = on;
        for (const auto& k : on) {
            const int li = schema_index(left.schema_, k), ri = schema_index(idx->build.schema(), k);
            if (li < 0 || ri < 0) throw std::invalid_argument("join key '" + k + "' missing");
            if (left.schema_[static_cast<size_t>(li)].type !=
                idx->build.schema()[static_cast<size_t>(ri)].type)
                throw std::invalid_argument("join key '" + k + "' has different types");
            idx->key_cols.push_back(idx->build.column(static_cast<size_t>(ri)));
        }
        Schema out = left.schema_;
        for (const auto& f : idx->build.schema()) {
            if (std::find(on.begin(), on.end(), f.name) != on.end()) continue;
            out.push_back(f);
            idx->right_cols.push_back(f.name);
        }
        check_unique(out);

        const size_t rows = idx->build.num_rows();
        idx->table = GroupTable(rows);
        const std::vector<uint64_t> h = hash_rows(idx->key_cols, rows);
        const KeyEq eq(idx->key_cols, idx->key_cols);
        for (size_t r = 0; r < rows; ++r) {
            auto [slot, inserted] =
                idx->table.find_or_insert(h[r], [&](uint32_t s) { return eq(r, idx->rows[s].front()); });
            if (inserted) idx->rows.emplace_back();
            idx->rows[slot].push_back(static_cast<uint32_t>(r));
        }
        return left.with_op({detail::NarrowOp::Kind::Join, {}, std::move(idx)}, std::move(out));
    }

    DataFrame distinct() const {
        std::vector<Expr> keys;
        for (const auto& f : resolved().schema_) keys.push_back(col(f.name));
        return aggregate(std::move(keys), {});
    }

    // Sort. Lazy: a following limit(n) keeps only n rows per partition before
    // the merge (Spark's TakeOrdered); other operations materialize the sort.
    DataFrame order_by(std::vector<SortKey> keys) const {
        if (keys.empty()) throw std::invalid_argument("order_by needs at least one key");
        const DataFrame& in = resolved();
        for (const auto& k : keys) {
            if (k.expr.type(in.schema_) == DType::Bool)
                throw std::invalid_argument("cannot order by a boolean: " + k.expr.name());
        }
        DataFrame out = in;
        out.pending_sort_ = std::make_shared<const std::vector<SortKey>>(std::move(keys));
        return out;
    }

    // First n rows (of the sort order, after order_by).
    DataFrame limit(size_t n) const {
        if (pending_sort_) return sorted(n);
        auto parts = rdd().map([n](const Batch& b) { return b.slice(0, n); }).collect();
        Batch all = Batch::concat(parts, schema_);
        return from_batch(*ctx_, all.slice(0, n), 1);
    }

    // ----- actions ------------------------------------------------------------

    size_t count() const {
        auto counts = rdd(detail::NameSet{})
                          .map([](const Batch& b) { return static_cast<int64_t>(b.num_rows()); })
                          .collect();
        size_t total = 0;
        for (int64_t c : counts) total += static_cast<size_t>(c);
        return total;
    }

    Batch collect() const {
        if (pending_sort_) return sorted(std::nullopt).collect();
        return Batch::concat(rdd().collect(), schema_);
    }

    // Render the first n rows as a text table (like Spark's show()).
    std::string show(size_t n = 20) const {
        const Batch b = limit(n).collect();
        std::vector<std::vector<std::string>> cells(b.num_columns());
        std::vector<size_t> width(b.num_columns());
        for (size_t c = 0; c < b.num_columns(); ++c) {
            width[c] = b.schema()[c].name.size();
            b.column(c).visit([&](const auto& v) {
                for (const auto& x : v) {
                    std::ostringstream os;
                    using T = std::decay_t<decltype(x)>;
                    if constexpr (std::is_same_v<T, uint8_t>) os << (x ? "true" : "false");
                    else os << x;
                    cells[c].push_back(os.str());
                    width[c] = std::max(width[c], cells[c].back().size());
                }
            });
        }
        std::ostringstream os;
        auto rule = [&] {
            os << '+';
            for (size_t w : width) os << std::string(w + 2, '-') << '+';
            os << '\n';
        };
        rule();
        os << '|';
        for (size_t c = 0; c < width.size(); ++c)
            os << ' ' << std::setw(static_cast<int>(width[c])) << b.schema()[c].name << " |";
        os << '\n';
        rule();
        for (size_t r = 0; r < b.num_rows(); ++r) {
            os << '|';
            for (size_t c = 0; c < width.size(); ++c)
                os << ' ' << std::setw(static_cast<int>(width[c])) << cells[c][r] << " |";
            os << '\n';
        }
        rule();
        return os.str();
    }

private:
    friend class GroupedData;

    DataFrame with_op(detail::NarrowOp op, Schema out) const {
        DataFrame d = *this;
        auto ops = std::make_shared<std::vector<detail::NarrowOp>>(*ops_);
        ops->push_back(std::move(op));
        d.ops_ = std::move(ops);
        d.schema_ = std::move(out);
        d.sorted_cache_.reset();
        return d;
    }

    // The RDD of this DataFrame's batches with the queued narrow operations
    // fused into one stage. With `needed`, only those output columns are
    // produced, and every column no later step reads is dropped as early as
    // possible (all columns when omitted).
    Rdd rdd(std::optional<detail::NameSet> needed = std::nullopt) const {
        using detail::NarrowOp;
        detail::NameSet need;
        if (needed) {
            need = *needed;
        } else {
            for (const auto& f : schema_) need.insert(f.name);
        }
        const bool all_out = !needed || need.size() == schema_.size();
        if (ops_->empty() && all_out) return base_;

        // Walk the ops backwards, keeping only what consumers need.
        std::vector<detail::PrunedOp> plan(ops_->size());
        for (size_t i = ops_->size(); i-- > 0;) {
            const NarrowOp& op = (*ops_)[i];
            detail::PrunedOp& p = plan[i];
            p.kind = op.kind;
            switch (op.kind) {
            case NarrowOp::Kind::Project: {
                detail::NameSet in;
                for (const auto& e : op.exprs) {
                    if (!need.count(e.name())) continue;
                    p.exprs.push_back(e);
                    detail::add_refs(e, in);
                }
                need = std::move(in);
                break;
            }
            case NarrowOp::Kind::Filter:
                p.exprs = op.exprs;
                detail::add_refs(op.exprs[0], need);
                break;
            case NarrowOp::Kind::Join:
                p.join = op.join;
                for (const auto& c : op.join->right_cols)
                    if (need.erase(c)) p.join_right.push_back(c);
                need.insert(op.join->keys.begin(), op.join->keys.end());
                break;
            }
        }
        std::vector<std::string> base_cols;
        for (const auto& f : base_schema_)
            if (need.count(f.name)) base_cols.push_back(f.name);

        // Output columns, in schema order.
        std::vector<std::string> out_cols;
        for (const auto& f : schema_)
            if (!needed || needed->count(f.name)) out_cols.push_back(f.name);

        auto steps = std::make_shared<const std::vector<detail::PrunedOp>>(std::move(plan));
        return base_.map([steps, base_cols, out_cols](const Batch& in) {
            Batch b = in.project(base_cols);
            for (const auto& p : *steps) {
                switch (p.kind) {
                case NarrowOp::Kind::Project: {
                    Schema s;
                    std::vector<Column> cols;
                    for (const auto& e : p.exprs) {
                        cols.push_back(e.eval(b));
                        s.push_back({e.name(), cols.back().type()});
                    }
                    b = Batch(std::move(s), std::move(cols), b.num_rows());
                    break;
                }
                case NarrowOp::Kind::Filter: b = detail::apply_filter(b, p.exprs[0]); break;
                case NarrowOp::Kind::Join: b = detail::apply_join(b, *p.join, p.join_right); break;
                }
            }
            return b.project(out_cols);
        });
    }

    // This DataFrame with any pending sort applied.
    const DataFrame& resolved() const {
        if (!pending_sort_) return *this;
        if (!sorted_cache_) sorted_cache_ = std::make_shared<DataFrame>(sorted(std::nullopt));
        return *sorted_cache_;
    }

    DataFrame aggregate(std::vector<Expr> keys, std::vector<AggExpr> aggs) const {
        const DataFrame& in = resolved();
        auto plan = std::make_shared<const AggPlan>(std::move(keys), std::move(aggs), in.schema_);
        check_unique(plan->output_schema);
        detail::NameSet needed;
        for (const auto& k : plan->keys) detail::add_refs(k, needed);
        for (const auto& a : plan->aggs)
            if (a.input()) detail::add_refs(*a.input(), needed);
        const size_t nk = plan->key_schema.size();
        const size_t parts = nk == 0 ? 1 : in.parts_;  // shuffle partitions

        // Stage 1: partial aggregate, then split by key hash into buckets.
        auto staged = in.rdd(needed).flatMap([plan, nk, parts](const Batch& b) {
            Batch p = plan->partial(b);
            std::vector<std::pair<int, Batch>> out;
            if (parts == 1) {
                out.emplace_back(0, std::move(p));  // always emit: global aggs need a row
                return out;
            }
            std::vector<Column> keys(p.columns().begin(),
                                     p.columns().begin() + static_cast<std::ptrdiff_t>(nk));
            const std::vector<uint64_t> h = hash_rows(keys, p.num_rows());
            std::vector<std::vector<uint32_t>> buckets(parts);
            for (size_t r = 0; r < p.num_rows(); ++r)
                buckets[(h[r] >> 32) % parts].push_back(static_cast<uint32_t>(r));
            for (size_t k = 0; k < parts; ++k)
                if (!buckets[k].empty()) out.emplace_back(static_cast<int>(k), p.take(buckets[k]));
            return out;
        });
        // Stage 2: shuffle by bucket, then merge partial states and finalize.
        auto merged = staged.groupByKey(parts).map(
            [plan](const std::pair<int, std::vector<Batch>>& kv) {
                return plan->merge_final(Batch::concat(kv.second, plan->partial_schema));
            });
        return DataFrame(ctx_, std::move(merged), plan->output_schema, parts);
    }

    // Row comparator over evaluated sort-key columns of (possibly different)
    // batches: negative / zero / positive.
    static int compare_rows(const std::vector<Column>& a, size_t ra, const std::vector<Column>& b,
                            size_t rb, const std::vector<SortKey>& keys) {
        for (size_t k = 0; k < keys.size(); ++k) {
            const int c = a[k].visit([&](const auto& va) {
                using T = typename std::decay_t<decltype(va)>::value_type;
                const auto& vb = b[k].values<T>();
                return va[ra] < vb[rb] ? -1 : (vb[rb] < va[ra] ? 1 : 0);
            });
            if (c != 0) return keys[k].ascending ? c : -c;
        }
        return 0;
    }

    // Sort every partition in parallel (keeping at most `limit` rows each),
    // then k-way merge the sorted runs into one partition.
    DataFrame sorted(std::optional<size_t> limit) const {
        auto keys = pending_sort_;
        auto runs_rdd = rdd().map([keys, limit](const Batch& b) {
            std::vector<Column> kc;
            for (const auto& k : *keys) kc.push_back(k.expr.eval(b));
            std::vector<uint32_t> idx(b.num_rows());
            for (size_t i = 0; i < idx.size(); ++i) idx[i] = static_cast<uint32_t>(i);
            auto less = [&](uint32_t x, uint32_t y) {
                const int c = compare_rows(kc, x, kc, y, *keys);
                return c != 0 ? c < 0 : x < y;  // stable
            };
            if (limit && *limit < idx.size()) {
                std::partial_sort(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(*limit),
                                  idx.end(), less);
                idx.resize(*limit);
            } else {
                std::sort(idx.begin(), idx.end(), less);
            }
            return b.take(idx);
        });
        std::vector<Batch> runs = runs_rdd.collect();

        std::vector<std::vector<Column>> run_keys(runs.size());
        size_t total = 0;
        std::vector<size_t> offset(runs.size());
        for (size_t r = 0; r < runs.size(); ++r) {
            for (const auto& k : *keys) run_keys[r].push_back(k.expr.eval(runs[r]));
            offset[r] = total;
            total += runs[r].num_rows();
        }
        const size_t want = limit ? std::min(*limit, total) : total;
        // Min-heap of (run, row); ties broken by run index for determinism.
        auto greater = [&](const std::pair<size_t, size_t>& x, const std::pair<size_t, size_t>& y) {
            const int c = compare_rows(run_keys[x.first], x.second, run_keys[y.first], y.second, *keys);
            return c != 0 ? c > 0 : x.first > y.first;
        };
        std::priority_queue<std::pair<size_t, size_t>, std::vector<std::pair<size_t, size_t>>,
                            decltype(greater)>
            heap(greater);
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
        const Batch merged = Batch::concat(runs, schema_).take(order);
        return from_batch(*ctx_, merged, 1);
    }

    static void check_unique(const Schema& s) {
        for (size_t i = 0; i < s.size(); ++i)
            for (size_t j = i + 1; j < s.size(); ++j)
                if (s[i].name == s[j].name)
                    throw std::invalid_argument("duplicate column name '" + s[i].name +
                                                "' (use alias())");
    }

    vega::SparkContext* ctx_;
    Rdd base_;            // materialized input
    Schema base_schema_;  // its schema
    Schema schema_;       // schema after the queued narrow ops
    std::shared_ptr<const std::vector<detail::NarrowOp>> ops_;
    size_t parts_;
    std::shared_ptr<const std::vector<SortKey>> pending_sort_;
    mutable std::shared_ptr<DataFrame> sorted_cache_;
};

class GroupedData {
public:
    GroupedData(DataFrame df, std::vector<Expr> keys) : df_(std::move(df)), keys_(std::move(keys)) {}
    DataFrame agg(std::vector<AggExpr> aggs) const { return df_.aggregate(keys_, std::move(aggs)); }
    DataFrame count() const { return agg({count_all().alias("count")}); }

private:
    DataFrame df_;
    std::vector<Expr> keys_;
};

inline GroupedData DataFrame::group_by(std::vector<Expr> keys) const {
    if (keys.empty()) throw std::invalid_argument("group_by needs at least one key");
    return GroupedData(resolved(), std::move(keys));
}

}  // namespace vega::df
