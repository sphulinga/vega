#pragma once

// Logical query plans for Vega DataFrames.
//
// Every DataFrame is an immutable tree of PlanNodes. Building a DataFrame
// only validates and records the operation (schemas are derived eagerly, so
// errors surface at the call that caused them); actions hand the tree to the
// optimizer (optimizer.hpp) and then to the physical planner (physical.hpp),
// which turns it into RDD stages.

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vega/core/context.hpp"
#include "vega/dataframe/aggregate.hpp"
#include "vega/dataframe/stats.hpp"

namespace vega::df {

enum class JoinType { Inner, Left, Right, Full, LeftSemi, LeftAnti, Cross };

inline const char* join_type_name(JoinType t) {
    switch (t) {
    case JoinType::Inner: return "Inner";
    case JoinType::Left: return "LeftOuter";
    case JoinType::Right: return "RightOuter";
    case JoinType::Full: return "FullOuter";
    case JoinType::LeftSemi: return "LeftSemi";
    case JoinType::LeftAnti: return "LeftAnti";
    case JoinType::Cross: return "Cross";
    }
    return "?";
}

// In-memory input of a plan: an RDD of batches plus what we know about it.
struct Source {
    explicit Source(vega::RDD<Batch> r) : rdd(std::move(r)) {}
    vega::RDD<Batch> rdd;
    Schema schema;
    size_t partitions = 1;
    size_t rows = 0;  // exact row count
    std::vector<ColumnStats> stats;  // per schema column (may be empty: unknown)
    std::string label = "LocalRelation";
};

struct PlanNode;
using PlanPtr = std::shared_ptr<const PlanNode>;

struct PlanNode {
    enum class Kind { Scan, Empty, Project, Filter, Join, Aggregate, Sort, Limit, Union, Window, Repartition };

    Kind kind = Kind::Scan;
    Schema schema;
    std::vector<PlanPtr> inputs;

    // Scan
    std::shared_ptr<const Source> source;
    std::optional<std::vector<std::string>> scan_columns;  // pruned column list
    // Project (exprs), Filter (exprs[0]), Window (exprs: window calls, each
    // aliased to its output column)
    std::vector<Expr> exprs;
    // Join: output = left columns ++ right columns (left only for semi/anti)
    JoinType join_type = JoinType::Inner;
    std::vector<std::string> left_keys, right_keys;
    // Aggregate
    std::vector<Expr> keys;
    std::vector<AggExpr> aggs;
    // Sort
    std::vector<SortKey> sort_keys;
    // Limit
    size_t limit = 0;
    // Repartition (keys: hash partitioning, empty: round robin)
    size_t partitions = 0;
    // Window: only rows ranked <= group_limit per partition are needed
    // (set by the optimizer when a rank filter follows; 0 = all)
    size_t group_limit = 0;

    const PlanPtr& input(size_t i = 0) const { return inputs.at(i); }
};

namespace plan {

inline void check_unique(const Schema& s) {
    for (size_t i = 0; i < s.size(); ++i)
        for (size_t j = i + 1; j < s.size(); ++j)
            if (s[i].name == s[j].name)
                throw std::invalid_argument("duplicate column name '" + s[i].name + "' (use alias())");
}

inline std::set<std::string> names_of(const Schema& s) {
    std::set<std::string> out;
    for (const auto& f : s) out.insert(f.name);
    return out;
}

inline std::vector<std::string> refs(const Expr& e) {
    std::vector<std::string> out;
    e.referenced_columns(out);
    return out;
}

inline void add_refs(const Expr& e, std::set<std::string>& out) {
    for (auto& r : refs(e)) out.insert(std::move(r));
}

// ----- constructors (derive and validate schemas) ---------------------------

inline PlanPtr scan(std::shared_ptr<const Source> src, std::optional<std::vector<std::string>> cols = std::nullopt) {
    auto n = std::make_shared<PlanNode>();
    n->kind = PlanNode::Kind::Scan;
    n->source = std::move(src);
    if (cols) {
        for (const auto& c : *cols) {
            const int i = schema_index(n->source->schema, c);
            if (i < 0) throw std::invalid_argument("no column " + c);
            n->schema.push_back(n->source->schema[static_cast<size_t>(i)]);
        }
        n->scan_columns = std::move(cols);
    } else {
        n->schema = n->source->schema;
    }
    return n;
}

inline PlanPtr empty(Schema schema) {
    auto n = std::make_shared<PlanNode>();
    n->kind = PlanNode::Kind::Empty;
    n->schema = std::move(schema);
    return n;
}

inline PlanPtr project(PlanPtr child, std::vector<Expr> exprs) {
    auto n = std::make_shared<PlanNode>();
    n->kind = PlanNode::Kind::Project;
    for (const auto& e : exprs) {
        if (e.contains_window()) throw std::logic_error("window expression in a projection");
        n->schema.push_back({e.name(), e.type(child->schema)});
    }
    check_unique(n->schema);
    n->exprs = std::move(exprs);
    n->inputs = {std::move(child)};
    return n;
}

inline PlanPtr filter(PlanPtr child, Expr predicate) {
    if (predicate.contains_window()) throw std::invalid_argument("window functions are not allowed in filter()");
    if (predicate.type(child->schema) != DType::Bool)
        throw std::invalid_argument("filter needs a boolean expression: " + predicate.name());
    auto n = std::make_shared<PlanNode>();
    n->kind = PlanNode::Kind::Filter;
    n->schema = child->schema;
    n->exprs = {std::move(predicate)};
    n->inputs = {std::move(child)};
    return n;
}

inline PlanPtr join(PlanPtr left, PlanPtr right, JoinType type, std::vector<std::string> lk,
                    std::vector<std::string> rk) {
    if (lk.size() != rk.size()) throw std::invalid_argument("join key lists differ in length");
    if (type != JoinType::Cross && lk.empty()) throw std::invalid_argument("join needs at least one key");
    for (size_t i = 0; i < lk.size(); ++i) {
        const int li = schema_index(left->schema, lk[i]), ri = schema_index(right->schema, rk[i]);
        if (li < 0) throw std::invalid_argument("join key '" + lk[i] + "' not in left side " + schema_string(left->schema));
        if (ri < 0) throw std::invalid_argument("join key '" + rk[i] + "' not in right side " + schema_string(right->schema));
        const DType lt = left->schema[static_cast<size_t>(li)].type, rt = right->schema[static_cast<size_t>(ri)].type;
        if (lt != rt) throw std::invalid_argument("join key '" + lk[i] + "' has different types on each side");
    }
    auto n = std::make_shared<PlanNode>();
    n->kind = PlanNode::Kind::Join;
    n->schema = left->schema;
    if (type != JoinType::LeftSemi && type != JoinType::LeftAnti)
        n->schema.insert(n->schema.end(), right->schema.begin(), right->schema.end());
    check_unique(n->schema);
    n->join_type = type;
    n->left_keys = std::move(lk);
    n->right_keys = std::move(rk);
    n->inputs = {std::move(left), std::move(right)};
    return n;
}

inline PlanPtr aggregate(PlanPtr child, std::vector<Expr> keys, std::vector<AggExpr> aggs) {
    AggPlan p(keys, aggs, child->schema);  // validates
    check_unique(p.output_schema);
    auto n = std::make_shared<PlanNode>();
    n->kind = PlanNode::Kind::Aggregate;
    n->schema = p.output_schema;
    n->keys = std::move(keys);
    n->aggs = std::move(aggs);
    n->inputs = {std::move(child)};
    return n;
}

inline PlanPtr sort(PlanPtr child, std::vector<SortKey> keys) {
    if (keys.empty()) throw std::invalid_argument("order_by needs at least one key");
    for (const auto& k : keys) {
        if (k.expr.contains_window()) throw std::invalid_argument("window function in a sort key");
        if (k.expr.type(child->schema) == DType::Bool)
            throw std::invalid_argument("cannot order by a boolean: " + k.expr.name());
    }
    auto n = std::make_shared<PlanNode>();
    n->kind = PlanNode::Kind::Sort;
    n->schema = child->schema;
    n->sort_keys = std::move(keys);
    n->inputs = {std::move(child)};
    return n;
}

inline PlanPtr limit(PlanPtr child, size_t count) {
    auto n = std::make_shared<PlanNode>();
    n->kind = PlanNode::Kind::Limit;
    n->schema = child->schema;
    n->limit = count;
    n->inputs = {std::move(child)};
    return n;
}

// Union by position: column names come from the first input.
inline PlanPtr union_all(std::vector<PlanPtr> inputs) {
    if (inputs.empty()) throw std::invalid_argument("union needs inputs");
    for (const auto& in : inputs) {
        if (in->schema.size() != inputs[0]->schema.size())
            throw std::invalid_argument("union inputs have different numbers of columns");
        for (size_t i = 0; i < in->schema.size(); ++i)
            if (in->schema[i].type != inputs[0]->schema[i].type)
                throw std::invalid_argument("union column " + std::to_string(i + 1) + " has different types");
    }
    auto n = std::make_shared<PlanNode>();
    n->kind = PlanNode::Kind::Union;
    n->schema = inputs[0]->schema;
    n->inputs = std::move(inputs);
    return n;
}

// Window: appends one column per window expression (each aliased); all
// expressions share one window specification.
inline PlanPtr window(PlanPtr child, std::vector<Expr> exprs, size_t group_limit = 0) {
    auto n = std::make_shared<PlanNode>();
    n->group_limit = group_limit;
    n->kind = PlanNode::Kind::Window;
    n->schema = child->schema;
    for (const auto& e : exprs) {
        if (!e.is_window()) throw std::logic_error("not a window expression");
        n->schema.push_back({e.name(), e.type(child->schema)});
    }
    check_unique(n->schema);
    n->exprs = std::move(exprs);
    n->inputs = {std::move(child)};
    return n;
}

inline PlanPtr repartition(PlanPtr child, size_t parts, std::vector<Expr> keys) {
    for (const auto& k : keys) k.type(child->schema);
    auto n = std::make_shared<PlanNode>();
    n->kind = PlanNode::Kind::Repartition;
    n->schema = child->schema;
    n->partitions = std::max<size_t>(1, parts);
    n->keys = std::move(keys);
    n->inputs = {std::move(child)};
    return n;
}

// Same node with new inputs (schema re-derived).
inline PlanPtr with_inputs(const PlanPtr& n, std::vector<PlanPtr> inputs) {
    using K = PlanNode::Kind;
    switch (n->kind) {
    case K::Scan:
    case K::Empty: return n;
    case K::Project: return project(inputs[0], n->exprs);
    case K::Filter: return filter(inputs[0], n->exprs[0]);
    case K::Join: return join(inputs[0], inputs[1], n->join_type, n->left_keys, n->right_keys);
    case K::Aggregate: return aggregate(inputs[0], n->keys, n->aggs);
    case K::Sort: return sort(inputs[0], n->sort_keys);
    case K::Limit: return limit(inputs[0], n->limit);
    case K::Union: return union_all(std::move(inputs));
    case K::Window: return window(inputs[0], n->exprs, n->group_limit);
    case K::Repartition: return repartition(inputs[0], n->partitions, n->keys);
    }
    return n;
}

// ----- estimates (used for join strategy and optimizer heuristics) ----------

double estimate_rows(const PlanPtr& n);

// Estimated number of distinct values (null counted as one value) of an
// output column, or nullopt when unknown (e.g. computed expressions).
inline std::optional<double> estimate_ndv(const PlanPtr& n, const std::string& column) {
    using K = PlanNode::Kind;
    auto capped = [&](std::optional<double> v) -> std::optional<double> {
        if (!v) return v;
        return std::min(*v, std::max(1.0, estimate_rows(n)));
    };
    switch (n->kind) {
    case K::Scan: {
        const int i = schema_index(n->source->schema, column);
        if (i < 0 || static_cast<size_t>(i) >= n->source->stats.size()) return std::nullopt;
        const ColumnStats& st = n->source->stats[static_cast<size_t>(i)];
        if (st.ndv < 0) return std::nullopt;
        return capped(st.ndv + (st.nulls ? 1 : 0));
    }
    case K::Empty: return 0.0;
    case K::Project: {
        for (const auto& e : n->exprs) {
            if (e.name() != column) continue;
            const Expr s = e.strip_alias();
            if (s.is_column()) return capped(estimate_ndv(n->input(), s.column_name()));
            if (s.is_literal()) return 1.0;
            return std::nullopt;
        }
        return std::nullopt;
    }
    case K::Join: {
        const bool left = schema_index(n->input(0)->schema, column) >= 0;
        return capped(estimate_ndv(n->input(left ? 0 : 1), column));
    }
    case K::Aggregate: {
        for (const auto& k : n->keys)
            if (k.name() == column) {
                const Expr s = k.strip_alias();
                return s.is_column() ? capped(estimate_ndv(n->input(), s.column_name())) : std::nullopt;
            }
        return std::nullopt;
    }
    case K::Window:
        if (schema_index(n->input()->schema, column) < 0) return std::nullopt;
        return capped(estimate_ndv(n->input(), column));
    case K::Union: {
        double total = 0;
        for (size_t i = 0; i < n->inputs.size(); ++i) {
            const int pos = schema_index(n->schema, column);
            auto v = estimate_ndv(n->inputs[i], n->inputs[i]->schema[static_cast<size_t>(pos)].name);
            if (!v) return std::nullopt;
            total += *v;
        }
        return capped(total);
    }
    default: return capped(estimate_ndv(n->input(), column));
    }
}

// Estimated number of groups of `keys` over `n` (product of the key NDVs,
// capped by the row count), or nullopt when a key's NDV is unknown.
inline std::optional<double> estimate_groups(const PlanPtr& n, const std::vector<Expr>& keys) {
    double groups = 1;
    for (const auto& k : keys) {
        const Expr s = k.strip_alias();
        if (s.is_literal()) continue;
        if (!s.is_column()) return std::nullopt;
        const auto v = estimate_ndv(n, s.column_name());
        if (!v) return std::nullopt;
        groups *= std::max(1.0, *v);
    }
    return std::min(groups, std::max(1.0, estimate_rows(n)));
}

// Selectivity of a predicate: column = literal -> 1 / NDV, IS NULL -> null
// fraction (when known), AND multiplies, otherwise a fixed 0.5.
inline double estimate_selectivity(const Expr& e, const PlanPtr& input) {
    const Expr p = e.strip_alias();
    if (p.kind() == Expr::Kind::Binary && p.op() == Expr::Op::And)
        return estimate_selectivity(p.children()[0], input) * estimate_selectivity(p.children()[1], input);
    if (p.kind() == Expr::Kind::Binary && p.op() == Expr::Op::Eq) {
        const Expr& a = p.children()[0];
        const Expr& b = p.children()[1];
        const Expr* c = a.is_column() && b.is_literal() ? &a : (b.is_column() && a.is_literal() ? &b : nullptr);
        if (c)
            if (auto v = estimate_ndv(input, c->column_name())) return 1.0 / std::max(1.0, *v);
    }
    return 0.5;
}

inline double estimate_rows(const PlanPtr& n) {
    using K = PlanNode::Kind;
    switch (n->kind) {
    case K::Scan: return static_cast<double>(n->source->rows);
    case K::Empty: return 0;
    case K::Filter: return estimate_rows(n->input()) * estimate_selectivity(n->exprs[0], n->input());
    case K::Aggregate: {
        if (n->keys.empty()) return 1;
        if (auto g = estimate_groups(n->input(), n->keys)) return *g;
        return estimate_rows(n->input()) * 0.5;
    }
    case K::Limit: return std::min(static_cast<double>(n->limit), estimate_rows(n->input()));
    case K::Join: {
        const double l = estimate_rows(n->input(0)), r = estimate_rows(n->input(1));
        switch (n->join_type) {
        case JoinType::Inner: return std::max(l, r);
        case JoinType::Left:
        case JoinType::LeftSemi:
        case JoinType::LeftAnti: return l;
        case JoinType::Right: return r;
        case JoinType::Full: return l + r;
        case JoinType::Cross: return l * r;
        }
        return l;
    }
    case K::Union: {
        double s = 0;
        for (const auto& in : n->inputs) s += estimate_rows(in);
        return s;
    }
    default: return estimate_rows(n->input());
    }
}

inline double estimate_bytes(const PlanPtr& n) {
    double width = 8;
    for (const auto& f : n->schema) width += f.type == DType::String ? 32 : 8;
    return estimate_rows(n) * width;
}

// ----- rendering -------------------------------------------------------------

inline std::string node_label(const PlanNode& n) {
    using K = PlanNode::Kind;
    auto list = [](const std::vector<Expr>& es) {
        std::string s;
        for (size_t i = 0; i < es.size(); ++i) {
            s += (i ? ", " : "");
            const Expr& e = es[i];
            const std::string inner = e.strip_alias().name();
            s += e.is_alias() && inner != e.name() ? inner + " AS " + e.name() : e.name();
        }
        return s;
    };
    std::ostringstream os;
    switch (n.kind) {
    case K::Scan: {
        os << "Scan " << n.source->label << " [";
        for (size_t i = 0; i < n.schema.size(); ++i) os << (i ? ", " : "") << n.schema[i].name;
        os << "] rows=" << n.source->rows;
        break;
    }
    case K::Empty: os << "Empty " << schema_string(n.schema); break;
    case K::Project: os << "Project [" << list(n.exprs) << "]"; break;
    case K::Filter: os << "Filter " << n.exprs[0].name(); break;
    case K::Join: {
        os << "Join " << join_type_name(n.join_type);
        if (!n.left_keys.empty()) {
            os << " on ";
            for (size_t i = 0; i < n.left_keys.size(); ++i)
                os << (i ? " AND " : "") << n.left_keys[i] << " = " << n.right_keys[i];
        }
        break;
    }
    case K::Aggregate: {
        os << "Aggregate keys=[" << list(n.keys) << "] aggs=[";
        for (size_t i = 0; i < n.aggs.size(); ++i) os << (i ? ", " : "") << n.aggs[i].name();
        os << "]";
        break;
    }
    case K::Sort: {
        os << "Sort [";
        for (size_t i = 0; i < n.sort_keys.size(); ++i)
            os << (i ? ", " : "") << n.sort_keys[i].expr.name() << (n.sort_keys[i].ascending ? " ASC" : " DESC")
               << (n.sort_keys[i].nulls_first ? " NULLS FIRST" : " NULLS LAST");
        os << "]";
        break;
    }
    case K::Limit: os << "Limit " << n.limit; break;
    case K::Union: os << "Union"; break;
    case K::Window:
        os << "Window [" << list(n.exprs) << "]";
        if (n.group_limit) os << " limit " << n.group_limit << " per partition";
        break;
    case K::Repartition: os << "Repartition " << n.partitions << (n.keys.empty() ? "" : " by [" + list(n.keys) + "]"); break;
    }
    return os.str();
}

inline void render(const PlanPtr& n, std::ostringstream& os, int depth) {
    os << std::string(static_cast<size_t>(depth) * 2, ' ') << (depth ? "+- " : "") << node_label(*n) << "\n";
    for (const auto& in : n->inputs) render(in, os, depth + 1);
}

inline std::string to_string(const PlanPtr& n) {
    std::ostringstream os;
    render(n, os, 0);
    return os.str();
}

}  // namespace plan
}  // namespace vega::df
