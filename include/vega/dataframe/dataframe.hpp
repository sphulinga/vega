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
// system: column-wise operations are narrow per-batch stages; group_by uses
// a two-phase hash aggregate with an engine shuffle between the phases.
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
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
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

class DataFrame {
public:
    using Rdd = vega::RDD<Batch>;

    DataFrame(vega::SparkContext* ctx, Rdd rdd, Schema schema, size_t partitions)
        : ctx_(ctx), rdd_(std::move(rdd)), schema_(std::move(schema)),
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
        return from_batch(ctx, Batch(std::move(schema), std::move(cols)), partitions);
    }

    const Schema& schema() const { return schema_; }
    size_t num_partitions() const { return parts_; }

    // ----- transformations (lazy) -------------------------------------------

    DataFrame select(std::vector<Expr> exprs) const {
        const DataFrame& in = resolved();
        auto plan = std::make_shared<const std::vector<Expr>>(std::move(exprs));
        Schema out;
        for (const auto& e : *plan) out.push_back({e.name(), e.type(in.schema_)});
        check_unique(out);
        auto rdd = in.rdd_.map([plan, out](const Batch& b) {
            std::vector<Column> cols;
            cols.reserve(plan->size());
            for (const auto& e : *plan) cols.push_back(e.eval(b));
            return Batch(out, std::move(cols));
        });
        return DataFrame(ctx_, std::move(rdd), std::move(out), in.parts_);
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
        auto rdd = in.rdd_.map([predicate](const Batch& b) {
            const Column mask = predicate.eval(b);
            const auto& m = mask.values<uint8_t>();
            std::vector<uint32_t> keep;
            keep.reserve(m.size());
            for (size_t i = 0; i < m.size(); ++i)
                if (m[i]) keep.push_back(static_cast<uint32_t>(i));
            return keep.size() == b.num_rows() ? b : b.take(keep);
        });
        return DataFrame(ctx_, std::move(rdd), in.schema_, in.parts_);
    }
    DataFrame where(const Expr& predicate) const { return filter(predicate); }

    GroupedData group_by(std::vector<Expr> keys) const;

    // Aggregate the whole DataFrame into one row.
    DataFrame agg(std::vector<AggExpr> aggs) const { return aggregate({}, std::move(aggs)); }

    // Inner equi-join on columns with the same names on both sides
    // (like Spark's df.join(other, Seq("k"))). The right side is collected
    // and broadcast to every partition of the left side.
    DataFrame join(const DataFrame& right, const std::vector<std::string>& on) const {
        const DataFrame& left = resolved();
        if (on.empty()) throw std::invalid_argument("join needs at least one key column");
        const Batch build = right.collect();
        std::vector<int> lk, rk;
        for (const auto& k : on) {
            const int li = schema_index(left.schema_, k), ri = schema_index(build.schema(), k);
            if (li < 0 || ri < 0) throw std::invalid_argument("join key '" + k + "' missing");
            if (left.schema_[li].type != build.schema()[ri].type)
                throw std::invalid_argument("join key '" + k + "' has different types");
            lk.push_back(li);
            rk.push_back(ri);
        }
        // Output: all left columns, then the right side's non-key columns.
        Schema out = left.schema_;
        std::vector<size_t> right_cols;
        for (size_t c = 0; c < build.num_columns(); ++c) {
            if (std::find(rk.begin(), rk.end(), static_cast<int>(c)) != rk.end()) continue;
            out.push_back(build.schema()[c]);
            right_cols.push_back(c);
        }
        check_unique(out);

        struct Index {
            Batch build;
            std::vector<Column> keys;
            std::unordered_map<uint64_t, std::vector<uint32_t>> rows;
        };
        auto index = std::make_shared<Index>();
        index->build = build;
        for (int c : rk) index->keys.push_back(build.column(static_cast<size_t>(c)));
        index->rows.reserve(build.num_rows());
        for (size_t r = 0; r < build.num_rows(); ++r)
            index->rows[row_hash(index->keys, r)].push_back(static_cast<uint32_t>(r));
        std::shared_ptr<const Index> idx = index;

        auto rdd = left.rdd_.map([idx, lk, right_cols, out](const Batch& b) {
            std::vector<Column> keys;
            for (int c : lk) keys.push_back(b.column(static_cast<size_t>(c)));
            std::vector<uint32_t> li, ri;
            for (size_t r = 0; r < b.num_rows(); ++r) {
                auto it = idx->rows.find(row_hash(keys, r));
                if (it == idx->rows.end()) continue;
                for (uint32_t rr : it->second) {
                    if (rows_equal(keys, r, idx->keys, rr)) {
                        li.push_back(static_cast<uint32_t>(r));
                        ri.push_back(rr);
                    }
                }
            }
            std::vector<Column> cols;
            for (const auto& c : b.columns()) cols.push_back(c.take(li));
            for (size_t c : right_cols) cols.push_back(idx->build.column(c).take(ri));
            return Batch(out, std::move(cols));
        });
        return DataFrame(ctx_, std::move(rdd), std::move(out), left.parts_);
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
        auto rdd = rdd_.map([n](const Batch& b) { return b.slice(0, n); });
        std::vector<Batch> parts = rdd.collect();
        Batch all = Batch::concat(parts, schema_);
        return from_batch(*ctx_, all.slice(0, n), 1);
    }

    // ----- actions ------------------------------------------------------------

    size_t count() const {
        auto counts = rdd_.map([](const Batch& b) { return static_cast<int64_t>(b.num_rows()); })
                          .collect();
        size_t total = 0;
        for (int64_t c : counts) total += static_cast<size_t>(c);
        return total;
    }

    Batch collect() const {
        if (pending_sort_) return sorted(std::nullopt).collect();
        return Batch::concat(rdd_.collect(), schema_);
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
        const size_t nk = plan->key_schema.size();
        const size_t parts = nk == 0 ? 1 : in.parts_;  // shuffle partitions

        // Stage 1: partial aggregate, then split by key hash into buckets.
        auto staged = in.rdd_.flatMap([plan, nk, parts](const Batch& b) {
            Batch p = plan->partial(b);
            std::vector<std::pair<int, Batch>> out;
            if (parts == 1) {
                out.emplace_back(0, std::move(p));  // always emit: global aggs need a row
                return out;
            }
            std::vector<Column> keys(p.columns().begin(),
                                     p.columns().begin() + static_cast<std::ptrdiff_t>(nk));
            std::vector<std::vector<uint32_t>> buckets(parts);
            for (size_t r = 0; r < p.num_rows(); ++r)
                buckets[row_hash(keys, r) % parts].push_back(static_cast<uint32_t>(r));
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
        auto runs_rdd = rdd_.map([keys, limit](const Batch& b) {
            std::vector<Column> kc;
            for (const auto& k : *keys) kc.push_back(k.expr.eval(b));
            std::vector<uint32_t> idx(b.num_rows());
            for (size_t i = 0; i < idx.size(); ++i) idx[i] = static_cast<uint32_t>(i);
            auto less = [&](uint32_t x, uint32_t y) { return compare_rows(kc, x, kc, y, *keys) < 0; };
            if (limit && *limit < idx.size()) {
                std::partial_sort(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(*limit),
                                  idx.end(), less);
                idx.resize(*limit);
            } else {
                std::stable_sort(idx.begin(), idx.end(), less);
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
    Rdd rdd_;
    Schema schema_;
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
