#pragma once

// Vega DataFrames: columnar, typed tables on top of the RDD engine.
//
//   using namespace vega::df;
//   auto sales = DataFrame::from_columns(ctx, {{"store", stores}, {"qty", qtys}}, 8);
//   auto top = sales.filter(col("qty") > 2)
//                   .join(regions, {"store"}, JoinType::Left)
//                   .group_by({col("region")})
//                   .agg({sum(col("qty")).alias("units"), count_all()})
//                   .order_by({desc(col("units"))})
//                   .limit(10);
//   std::cout << top.show() << top.explain();
//
// A DataFrame is an immutable logical plan (plan.hpp). Transformations
// validate and extend the plan; actions (count, collect, show, ...) run the
// optimizer (optimizer.hpp) and execute the optimized plan as parallel RDD
// stages on the actor system (physical.hpp).

#include <cctype>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vega/core/context.hpp"
#include "vega/dataframe/physical.hpp"

namespace vega::df {

class GroupedData;

struct CsvOptions {
    bool header = true;          // first line holds column names
    char delimiter = ',';
    char quote = '"';
    bool infer_schema = true;    // else every column is a string
    std::string null_value;      // field text read/written as null
};

class DataFrame {
public:
    DataFrame(vega::SparkContext* ctx, PlanPtr plan) : ctx_(ctx), plan_(std::move(plan)) {}

    // ----- sources ---------------------------------------------------------

    // Split a batch into `partitions` contiguous slices. Low-cardinality
    // string columns are dictionary-encoded (as Spark's columnar cache does).
    static DataFrame from_batch(vega::SparkContext& ctx, const Batch& batch, size_t partitions) {
        partitions = std::max<size_t>(1, partitions);
        std::vector<Column> cols;
        for (const auto& c : batch.columns()) cols.push_back(c.encode());
        const Batch enc(batch.schema(), std::move(cols), batch.num_rows());
        const size_t rows = enc.num_rows();
        std::vector<Batch> slices;
        slices.reserve(partitions);
        for (size_t p = 0; p < partitions; ++p) slices.push_back(enc.slice(rows * p / partitions, rows * (p + 1) / partitions));
        return from_batches(ctx, std::move(slices), batch.schema(), partitions);
    }

    static DataFrame from_columns(vega::SparkContext& ctx, std::vector<std::pair<std::string, Column>> columns,
                                  size_t partitions) {
        Schema schema;
        std::vector<Column> cols;
        for (auto& [name, c] : columns) {
            schema.push_back({name, c.type()});
            cols.push_back(std::move(c));
        }
        plan::check_unique(schema);
        return from_batch(ctx, Batch(std::move(schema), std::move(cols)), partitions);
    }

    static DataFrame from_batches(vega::SparkContext& ctx, std::vector<Batch> batches, const Schema& schema,
                                  size_t partitions) {
        size_t rows = 0;
        for (const auto& b : batches) rows += b.num_rows();
        if (batches.empty()) batches.push_back(Batch::empty(schema));
        partitions = std::max<size_t>(1, std::min(partitions, batches.size()));
        auto src = std::make_shared<Source>(ctx.parallelize(batches, partitions));
        src->schema = schema;
        src->partitions = partitions;
        src->rows = rows;
        // Column statistics: one HyperLogLog sketch per column and batch,
        // built in parallel on the engine and merged here.
        const auto sketches = src->rdd.map([](const Batch& b) { return sketch_batch(b); }).collect();
        src->stats = merge_sketches(sketches, schema.size());
        return DataFrame(&ctx, plan::scan(src));
    }

    static DataFrame empty(vega::SparkContext& ctx, const Schema& schema) {
        return DataFrame(&ctx, plan::empty(schema));
    }

    static DataFrame read_csv(vega::SparkContext& ctx, const std::string& path, const CsvOptions& opt = {},
                              size_t partitions = 8);

    // ----- metadata ------------------------------------------------------------

    const Schema& schema() const { return plan_->schema; }
    std::vector<std::string> columns() const {
        std::vector<std::string> out;
        for (const auto& f : schema()) out.push_back(f.name);
        return out;
    }
    const PlanPtr& plan() const { return plan_; }

    // Optimizer estimates for this DataFrame: rows, and distinct values of
    // a column (nullopt when unknown, e.g. for computed columns).
    double estimated_rows() const { return plan::estimate_rows(plan_); }
    std::optional<double> estimated_distinct(const std::string& column) const {
        return plan::estimate_ndv(plan_, column);
    }

    // ----- transformations (lazy) ----------------------------------------------

    // Expressions may contain window functions (row_number().over(w), ...).
    DataFrame select(std::vector<Expr> exprs) const {
        std::vector<Expr> windows;
        std::vector<Expr> out;
        for (const auto& e : exprs) {
            Expr r = extract_windows(e, windows);
            if (r.name() != e.name()) r = r.strip_alias().alias(e.name());
            out.push_back(r);
        }
        PlanPtr p = plan_;
        // one Window node per distinct specification
        std::vector<std::pair<std::string, std::vector<Expr>>> groups;
        for (const auto& w : windows) {
            const std::string key = spec_key(*w.strip_alias().window_spec());
            auto it = std::find_if(groups.begin(), groups.end(), [&](const auto& g) { return g.first == key; });
            if (it == groups.end()) groups.push_back({key, {w}});
            else it->second.push_back(w);
        }
        for (const auto& g : groups) p = plan::window(p, g.second);
        return wrap(plan::project(p, out));
    }

    DataFrame select_columns(const std::vector<std::string>& names) const {
        std::vector<Expr> es;
        for (const auto& n : names) es.push_back(col(n));
        return select(es);
    }

    // Add a column, or replace the one with the same name.
    DataFrame with_column(const std::string& name, const Expr& e) const {
        std::vector<Expr> exprs;
        bool replaced = false;
        for (const auto& f : schema()) {
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

    DataFrame with_column_renamed(const std::string& from, const std::string& to) const {
        std::vector<Expr> exprs;
        for (const auto& f : schema()) exprs.push_back(f.name == from ? col(f.name).alias(to) : col(f.name));
        return select(std::move(exprs));
    }

    DataFrame drop(const std::vector<std::string>& names) const {
        std::vector<Expr> exprs;
        for (const auto& f : schema())
            if (std::find(names.begin(), names.end(), f.name) == names.end()) exprs.push_back(col(f.name));
        return select(std::move(exprs));
    }

    DataFrame filter(const Expr& predicate) const { return wrap(plan::filter(plan_, predicate)); }
    DataFrame where(const Expr& predicate) const { return filter(predicate); }

    GroupedData group_by(std::vector<Expr> keys) const;

    // Aggregate the whole DataFrame into one row.
    DataFrame agg(std::vector<AggExpr> aggs) const;

    // Equi-join on columns with the same names on both sides, like Spark's
    // df.join(other, Seq("k"), how): the key columns appear once (taken from
    // the left side, the right side for right joins, and coalesced for full
    // joins), followed by the other left and then right columns. Semi and anti
    // joins return the left columns only.
    DataFrame join(const DataFrame& right, const std::vector<std::string>& on, JoinType how = JoinType::Inner) const {
        if (on.empty()) throw std::invalid_argument("join needs at least one key column (or use cross_join)");
        std::vector<Expr> rproj;
        std::vector<std::string> rkeys;
        for (const auto& f : right.schema()) {
            if (std::find(on.begin(), on.end(), f.name) != on.end()) rproj.push_back(col(f.name).alias("__r_" + f.name));
            else rproj.push_back(col(f.name));
        }
        for (const auto& k : on) {
            if (schema_index(right.schema(), k) < 0) throw std::invalid_argument("join key '" + k + "' not in right side");
            rkeys.push_back("__r_" + k);
        }
        const PlanPtr j = plan::join(plan_, plan::project(right.plan_, rproj), how, on, rkeys);
        std::vector<Expr> out;
        for (const auto& k : on) {
            if (how == JoinType::Right) out.push_back(col("__r_" + k).alias(k));
            else if (how == JoinType::Full) out.push_back(coalesce({col(k), col("__r_" + k)}).alias(k));
            else out.push_back(col(k));
        }
        for (const auto& f : schema())
            if (std::find(on.begin(), on.end(), f.name) == on.end()) out.push_back(col(f.name));
        if (how != JoinType::LeftSemi && how != JoinType::LeftAnti)
            for (const auto& f : right.schema())
                if (std::find(on.begin(), on.end(), f.name) == on.end()) out.push_back(col(f.name));
        return wrap(plan::project(j, out));
    }

    // Equi-join on (left column, right column) pairs; the output has all
    // left columns, then all right columns (names must not collide).
    DataFrame join_on(const DataFrame& right, const std::vector<std::pair<std::string, std::string>>& on,
                      JoinType how = JoinType::Inner) const {
        std::vector<std::string> lk, rk;
        for (const auto& [l, r] : on) {
            lk.push_back(l);
            rk.push_back(r);
        }
        return wrap(plan::join(plan_, right.plan_, how, lk, rk));
    }

    DataFrame cross_join(const DataFrame& right) const {
        return wrap(plan::join(plan_, right.plan_, JoinType::Cross, {}, {}));
    }

    // Union by position (Spark's union / unionAll: duplicates are kept).
    DataFrame union_all(const DataFrame& other) const { return wrap(plan::union_all({plan_, other.plan_})); }
    // Union matching columns by name.
    DataFrame union_by_name(const DataFrame& other) const {
        std::vector<Expr> es;
        for (const auto& f : schema()) es.push_back(col(f.name));
        return wrap(plan::union_all({plan_, plan::project(other.plan_, es)}));
    }

    DataFrame distinct() const {
        std::vector<Expr> keys;
        for (const auto& f : schema()) keys.push_back(col(f.name));
        return wrap(plan::aggregate(plan_, keys, {}));
    }

    // One row per distinct value of `subset` (the first in input order).
    DataFrame drop_duplicates(const std::vector<std::string>& subset) const {
        if (subset.empty()) return distinct();
        std::vector<Expr> part;
        for (const auto& c : subset) part.push_back(col(c));
        const DataFrame numbered =
            with_column("__dedup_rn", row_number().over(Window::partition_by(part)));
        return numbered.filter(col("__dedup_rn") == 1).drop({"__dedup_rn"});
    }

    DataFrame order_by(std::vector<SortKey> keys) const { return wrap(plan::sort(plan_, std::move(keys))); }
    DataFrame sort(std::vector<SortKey> keys) const { return order_by(std::move(keys)); }

    DataFrame limit(size_t n) const { return wrap(plan::limit(plan_, n)); }

    // Replace nulls in the given (default: all type-compatible) columns.
    // A numeric value fills numeric columns, a string fills string columns,
    // a boolean fills boolean columns.
    DataFrame fill_null(const Expr& value, const std::vector<std::string>& cols = {}) const {
        const DType vt = value.type({});
        std::vector<Expr> es;
        for (const auto& f : schema()) {
            const bool chosen = cols.empty() || std::find(cols.begin(), cols.end(), f.name) != cols.end();
            const bool compatible = f.type == vt || (Expr::numeric(f.type) && Expr::numeric(vt));
            if (chosen && compatible) es.push_back(coalesce({col(f.name), cast(value, f.type)}).alias(f.name));
            else es.push_back(col(f.name));
        }
        return select(es);
    }

    // Drop rows with a null in any of `cols` (default: any column).
    DataFrame drop_nulls(const std::vector<std::string>& cols = {}) const {
        std::vector<Expr> preds;
        for (const auto& f : schema())
            if (cols.empty() || std::find(cols.begin(), cols.end(), f.name) != cols.end())
                preds.push_back(col(f.name).is_not_null());
        if (preds.empty()) return *this;
        return filter(opt::conjoin(preds));
    }

    // Redistribute rows into n partitions (by key hash, or round robin).
    DataFrame repartition(size_t n, std::vector<Expr> keys = {}) const {
        return wrap(plan::repartition(plan_, n, std::move(keys)));
    }

    // Compute now and keep the result in memory (string columns
    // dictionary-encoded), so later queries start from it.
    DataFrame cache() const {
        exec::Executor ex(ctx_);
        const PlanPtr p = optimized();
        const exec::Phys phys = ex.run(p);
        std::vector<Batch> batches = phys.rdd.collect();
        for (auto& b : batches) {
            std::vector<Column> cols;
            for (const auto& c : b.columns()) cols.push_back(c.flat().encode());
            b = Batch(b.schema(), std::move(cols), b.num_rows());
        }
        return from_batches(*ctx_, std::move(batches), schema(), std::max<size_t>(phys.parts, 1));
    }

    // ----- actions ---------------------------------------------------------------

    size_t count() const {
        const DataFrame c(ctx_, plan::aggregate(plan_, {}, {count_all().alias("count")}));
        const Batch b = c.collect();
        return static_cast<size_t>(b.column(0).values<int64_t>()[0]);
    }

    bool is_empty() const { return limit(1).count() == 0; }

    Batch collect() const {
        exec::Executor ex(ctx_);
        return Batch::concat(ex.collect(optimized()), schema()).flat();
    }

    Batch head(size_t n) const { return limit(n).collect(); }

    // Render the first n rows as a text table (like Spark's show()).
    std::string show(size_t n = 20) const {
        const Batch b = limit(n).collect();
        std::vector<std::vector<std::string>> cells(b.num_columns());
        std::vector<size_t> width(b.num_columns());
        for (size_t c = 0; c < b.num_columns(); ++c) {
            width[c] = b.schema()[c].name.size();
            const Column& column = b.column(c);
            column.visit([&](const auto& v) {
                for (size_t r = 0; r < v.size(); ++r) {
                    std::string s;
                    using T = typename std::decay_t<decltype(v)>::value_type;
                    if (!column.is_valid(r)) s = "NULL";
                    else if constexpr (std::is_same_v<T, uint8_t>) s = v[r] ? "true" : "false";
                    else if constexpr (std::is_same_v<T, double>) s = detail::format_double(v[r]);
                    else if constexpr (std::is_same_v<T, int64_t>) s = std::to_string(v[r]);
                    else s = v[r];
                    cells[c].push_back(s);
                    width[c] = std::max(width[c], s.size());
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

    // Logical, optimized and physical plans (like Spark's explain(true)).
    std::string explain() const {
        const PlanPtr o = optimized();
        std::ostringstream os;
        os << "== Logical Plan ==\n" << plan::to_string(plan_);
        os << "== Optimized Logical Plan ==\n" << plan::to_string(o);
        os << "== Physical Plan ==\n";
        exec::render_physical(o, os, 0);
        return os.str();
    }

    PlanPtr optimized() const { return opt::optimize(plan_); }

    void write_csv(const std::string& path, const CsvOptions& opt = {}) const;

private:
    friend class GroupedData;

    DataFrame wrap(PlanPtr p) const { return DataFrame(ctx_, std::move(p)); }

    static std::string spec_key(const WindowSpec& s) {
        std::string k;
        for (const auto& p : s.partition) k += p.debug_string() + ",";
        k += "|";
        for (const auto& o : s.order) k += o.expr.debug_string() + (o.ascending ? "+" : "-") + (o.nulls_first ? "f" : "l");
        return k;
    }

    // Replace window calls by references to columns a Window node computes.
    static Expr extract_windows(const Expr& e, std::vector<Expr>& windows) {
        if (e.kind() == Expr::Kind::Window) {
            const std::string name = "__w" + std::to_string(windows.size());
            windows.push_back(e.alias(name));
            return col(name);
        }
        if (!e.contains_window()) return e;
        std::vector<Expr> kids;
        for (const auto& c : e.children()) kids.push_back(extract_windows(c, windows));
        return e.with_children(kids);
    }

    vega::SparkContext* ctx_;
    PlanPtr plan_;
};

class GroupedData {
public:
    GroupedData(DataFrame df, std::vector<Expr> keys) : df_(std::move(df)), keys_(std::move(keys)) {}

    DataFrame agg(std::vector<AggExpr> aggs) const {
        bool distinct = false;
        for (const auto& a : aggs) distinct |= a.fn() == AggFn::CountDistinct;
        if (!distinct) return df_.wrap(plan::aggregate(df_.plan_, keys_, std::move(aggs)));
        // count(DISTINCT x): group by (keys, x), then count per key.
        for (const auto& a : aggs)
            if (a.fn() != AggFn::CountDistinct || !a.input()->same_as(*aggs[0].input()))
                throw std::invalid_argument(
                    "count_distinct can only be combined with count_distinct of the same expression");
        std::vector<Expr> inner_keys = keys_;
        inner_keys.push_back(aggs[0].input()->alias("__distinct"));
        const PlanPtr inner = plan::aggregate(df_.plan_, inner_keys, {});
        std::vector<Expr> outer_keys;
        for (const auto& k : keys_) outer_keys.push_back(col(k.name()));
        std::vector<AggExpr> outer;
        for (const auto& a : aggs) outer.push_back(vega::df::count(col("__distinct")).alias(a.name()));
        return df_.wrap(plan::aggregate(inner, outer_keys, outer));
    }
    DataFrame count() const { return agg({count_all().alias("count")}); }

private:
    DataFrame df_;
    std::vector<Expr> keys_;
};

inline GroupedData DataFrame::group_by(std::vector<Expr> keys) const {
    if (keys.empty()) throw std::invalid_argument("group_by needs at least one key");
    return GroupedData(*this, std::move(keys));
}

inline DataFrame DataFrame::agg(std::vector<AggExpr> aggs) const {
    bool distinct = false;
    for (const auto& a : aggs) distinct |= a.fn() == AggFn::CountDistinct;
    if (distinct) return GroupedData(*this, {}).agg(std::move(aggs));
    return wrap(plan::aggregate(plan_, {}, std::move(aggs)));
}

// ---------------------------------------------------------------------------
// CSV
// ---------------------------------------------------------------------------

namespace detail {

inline std::vector<std::vector<std::optional<std::string>>> parse_csv(const std::string& text, const CsvOptions& o) {
    std::vector<std::vector<std::optional<std::string>>> rows;
    std::vector<std::optional<std::string>> row;
    std::string field;
    bool in_quotes = false, quoted = false, any = false;
    auto end_field = [&] {
        if (!quoted && field == o.null_value) row.push_back(std::nullopt);
        else row.push_back(field);
        field.clear();
        quoted = false;
    };
    for (size_t i = 0; i < text.size(); ++i) {
        const char c = text[i];
        any = true;
        if (in_quotes) {
            if (c == o.quote) {
                if (i + 1 < text.size() && text[i + 1] == o.quote) {
                    field += c;
                    ++i;
                } else {
                    in_quotes = false;
                }
            } else {
                field += c;
            }
        } else if (c == o.quote) {
            in_quotes = true;
            quoted = true;
        } else if (c == o.delimiter) {
            end_field();
        } else if (c == '\n' || c == '\r') {
            if (c == '\r' && i + 1 < text.size() && text[i + 1] == '\n') ++i;
            end_field();
            rows.push_back(std::move(row));
            row.clear();
            any = false;
        } else {
            field += c;
        }
    }
    if (any) {
        end_field();
        rows.push_back(std::move(row));
    }
    return rows;
}

inline bool parse_int(const std::string& s, int64_t& out) {
    if (s.empty()) return false;
    char* end = nullptr;
    errno = 0;
    out = std::strtoll(s.c_str(), &end, 10);
    return *end == 0 && errno != ERANGE;
}

inline bool parse_double(const std::string& s, double& out) {
    if (s.empty()) return false;
    char* end = nullptr;
    out = std::strtod(s.c_str(), &end);
    return *end == 0;
}

inline std::string lower_copy(std::string s) {
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

}  // namespace detail

inline DataFrame DataFrame::read_csv(vega::SparkContext& ctx, const std::string& path, const CsvOptions& opt,
                                     size_t partitions) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::stringstream ss;
    ss << in.rdbuf();
    auto rows = detail::parse_csv(ss.str(), opt);
    std::vector<std::string> names;
    size_t ncols = 0;
    for (const auto& r : rows) ncols = std::max(ncols, r.size());
    size_t first = 0;
    if (opt.header && !rows.empty()) {
        for (const auto& f : rows[0]) names.push_back(f.value_or(""));
        ncols = std::max(ncols, names.size());
        first = 1;
    }
    for (size_t c = names.size(); c < ncols; ++c) names.push_back("_c" + std::to_string(c));
    const size_t n = rows.size() - first;
    std::vector<std::pair<std::string, Column>> cols;
    for (size_t c = 0; c < ncols; ++c) {
        auto cell = [&](size_t r) -> const std::optional<std::string>& {
            static const std::optional<std::string> none;
            const auto& row = rows[first + r];
            return c < row.size() ? row[c] : none;
        };
        bool is_int = opt.infer_schema, is_double = opt.infer_schema, is_bool = opt.infer_schema;
        for (size_t r = 0; r < n && (is_int || is_double || is_bool); ++r) {
            const auto& v = cell(r);
            if (!v) continue;
            int64_t i;
            double d;
            if (is_int && !detail::parse_int(*v, i)) is_int = false;
            if (is_double && !detail::parse_double(*v, d)) is_double = false;
            const std::string l = detail::lower_copy(*v);
            if (is_bool && l != "true" && l != "false") is_bool = false;
        }
        BoolVec valid(n, 1);
        for (size_t r = 0; r < n; ++r) valid[r] = cell(r).has_value();
        if (is_int) {
            std::vector<int64_t> v(n);
            for (size_t r = 0; r < n; ++r)
                if (cell(r)) detail::parse_int(*cell(r), v[r]);
            cols.emplace_back(names[c], Column(std::move(v), std::move(valid)));
        } else if (is_double) {
            std::vector<double> v(n);
            for (size_t r = 0; r < n; ++r)
                if (cell(r)) detail::parse_double(*cell(r), v[r]);
            cols.emplace_back(names[c], Column(std::move(v), std::move(valid)));
        } else if (is_bool) {
            BoolVec v(n);
            for (size_t r = 0; r < n; ++r) v[r] = cell(r) && detail::lower_copy(*cell(r)) == "true";
            cols.emplace_back(names[c], Column(std::move(v), std::move(valid)));
        } else {
            std::vector<std::string> v(n);
            for (size_t r = 0; r < n; ++r)
                if (cell(r)) v[r] = *cell(r);
            cols.emplace_back(names[c], Column(std::move(v), std::move(valid)));
        }
    }
    auto df = from_columns(ctx, std::move(cols), partitions);
    std::const_pointer_cast<Source>(df.plan_->source)->label = "csv " + path;
    return df;
}

inline void DataFrame::write_csv(const std::string& path, const CsvOptions& opt) const {
    const Batch b = collect();
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot write " + path);
    auto quote = [&](const std::string& s) {
        if (s.find(opt.delimiter) == std::string::npos && s.find(opt.quote) == std::string::npos &&
            s.find('\n') == std::string::npos && s.find('\r') == std::string::npos && !(s == opt.null_value && !s.empty()))
            return s.empty() && opt.null_value.empty() ? std::string(2, opt.quote) : s;
        std::string q(1, opt.quote);
        for (char c : s) {
            if (c == opt.quote) q += opt.quote;
            q += c;
        }
        return q + opt.quote;
    };
    if (opt.header) {
        for (size_t c = 0; c < b.num_columns(); ++c) out << (c ? std::string(1, opt.delimiter) : "") << quote(b.schema()[c].name);
        out << '\n';
    }
    for (size_t r = 0; r < b.num_rows(); ++r) {
        for (size_t c = 0; c < b.num_columns(); ++c) {
            if (c) out << opt.delimiter;
            const Column& column = b.column(c);
            if (!column.is_valid(r)) {
                out << opt.null_value;
                continue;
            }
            column.visit([&](const auto& v) {
                using T = typename std::decay_t<decltype(v)>::value_type;
                if constexpr (std::is_same_v<T, uint8_t>) out << (v[r] ? "true" : "false");
                else if constexpr (std::is_same_v<T, double>) out << detail::format_double(v[r]);
                else if constexpr (std::is_same_v<T, int64_t>) out << v[r];
                else out << quote(v[r]);
            });
        }
        out << '\n';
    }
}

}  // namespace vega::df
