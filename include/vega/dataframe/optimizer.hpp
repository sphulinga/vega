#pragma once

// Rule-based optimizer for DataFrame plans. Rules (applied in this order,
// with the pushdown/simplify pair iterated to a fixed point):
//
//   ConstantFolding        literal subexpressions are evaluated once;
//                          x AND true -> x, x OR true -> true, NOT NOT x -> x
//   PruneFilters           filter(true) disappears, filter(false|null) -> empty
//   PushDownPredicates     filters move below projections (aliases are
//                          substituted), sorts, repartitions, unions, windows
//                          (partition-key predicates), aggregates (grouping-key
//                          predicates) and joins (per side, see below)
//   EliminateOuterJoin     a null-rejecting filter on the null-supplying side
//                          turns an outer join into an inner/left/right join
//   InferJoinPredicates    a predicate on join keys is copied to the other side
//   CombineFilters / CollapseProject / RemoveIdentityProject
//   EliminateSorts         a sort feeding an aggregate, join, window or another
//                          sort is dropped (it cannot affect the result)
//   LimitPushDown          through projections and into union inputs;
//                          limit(limit(x)) keeps the smaller one
//   PropagateEmpty         operations on empty inputs become empty
//   PushAggregateThroughJoin  (eager aggregation) group-by over an inner join
//                          whose aggregates read one side only: that side is
//                          pre-aggregated by its join keys before the join,
//                          then partial results are combined above it. Applied
//                          only when the keys' distinct-value estimates show
//                          the pre-aggregation shrinks that side >= 4x
//   ColumnPruning          every node produces only the columns used above it,
//                          down to the scans

#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "vega/dataframe/plan.hpp"

namespace vega::df::opt {

using K = PlanNode::Kind;

// ----- expression helpers -----------------------------------------------------

inline void split_conjuncts(const Expr& e, std::vector<Expr>& out) {
    const Expr s = e.strip_alias();
    if (s.kind() == Expr::Kind::Binary && s.op() == Expr::Op::And) {
        split_conjuncts(s.children()[0], out);
        split_conjuncts(s.children()[1], out);
    } else {
        out.push_back(s);
    }
}

inline std::vector<Expr> conjuncts(const Expr& e) {
    std::vector<Expr> out;
    split_conjuncts(e, out);
    return out;
}

inline Expr conjoin(const std::vector<Expr>& es) {
    Expr r = es.at(0);
    for (size_t i = 1; i < es.size(); ++i) r = r && es[i];
    return r;
}

inline bool refs_within(const Expr& e, const std::set<std::string>& cols) {
    for (const auto& r : plan::refs(e))
        if (!cols.count(r)) return false;
    return true;
}

inline Expr literal_from(const Column& c, size_t row) {
    if (!c.is_valid(row)) return Expr::null_of(c.type());
    return c.visit([&](const auto& v) -> Expr {
        using T = typename std::decay_t<decltype(v)>::value_type;
        if constexpr (std::is_same_v<T, uint8_t>) return Expr(v[row] != 0);
        else return Expr::literal(Expr::Scalar(v[row]));
    });
}

inline bool is_bool_literal(const Expr& e, bool value) {
    return e.is_literal() && !e.is_null_literal() && e.literal_value().index() == 3 &&
           std::get<bool>(e.literal_value()) == value;
}

// Constant folding and boolean simplification.
inline Expr fold(const Expr& e) {
    switch (e.kind()) {
    case Expr::Kind::Column:
    case Expr::Kind::Literal:
    case Expr::Kind::Window: return e;
    case Expr::Kind::Alias: {
        const Expr c = fold(e.children()[0]);
        return c.same_as(e.children()[0]) ? e : c.alias(e.name());
    }
    default: break;
    }
    std::vector<Expr> kids;
    bool changed = false;
    for (const auto& c : e.children()) {
        kids.push_back(fold(c));
        changed |= !kids.back().same_as(c);
    }
    const Expr r = changed ? e.with_children(kids) : e;
    if (r.kind() == Expr::Kind::Binary && (r.op() == Expr::Op::And || r.op() == Expr::Op::Or)) {
        const Expr& a = kids[0];
        const Expr& b = kids[1];
        if (r.op() == Expr::Op::And) {
            if (is_bool_literal(a, false) || is_bool_literal(b, false)) return Expr(false);
            if (is_bool_literal(a, true)) return b;
            if (is_bool_literal(b, true)) return a;
        } else {
            if (is_bool_literal(a, true) || is_bool_literal(b, true)) return Expr(true);
            if (is_bool_literal(a, false)) return b;
            if (is_bool_literal(b, false)) return a;
        }
    }
    if (r.kind() == Expr::Kind::Not && kids[0].kind() == Expr::Kind::Not) return kids[0].children()[0];
    if (plan::refs(r).empty() && !r.contains_window()) {
        try {
            const Batch one({}, {}, 1);
            return literal_from(r.eval(one), 0);
        } catch (...) {
            // leave errors (e.g. 1 / 0) to execution time
        }
    }
    return r;
}

// Fold, keeping the output name.
inline Expr fold_named(const Expr& e) {
    const Expr f = fold(e);
    if (f.same_as(e)) return e;
    return f.name() == e.name() ? f : f.strip_alias().alias(e.name());
}

// Does the predicate reject rows where all its columns are null?
inline bool null_rejecting(const Expr& p, const Schema& side) {
    try {
        std::set<std::string> cols;
        plan::add_refs(p, cols);
        if (cols.empty()) return false;
        Schema s;
        std::vector<Column> c;
        for (const auto& name : cols) {
            const int i = schema_index(side, name);
            if (i < 0) return false;
            s.push_back(side[static_cast<size_t>(i)]);
            c.push_back(Column::nulls(s.back().type, 1));
        }
        const Column r = p.eval(Batch(s, c));
        return !r.is_valid(0) || r.values<uint8_t>()[0] == 0;
    } catch (...) {
        return false;
    }
}

// ----- ConstantFolding over a plan ---------------------------------------------

inline PlanPtr fold_plan(const PlanPtr& n) {
    std::vector<PlanPtr> in;
    for (const auto& c : n->inputs) in.push_back(fold_plan(c));
    switch (n->kind) {
    case K::Project: {
        std::vector<Expr> es;
        for (const auto& e : n->exprs) es.push_back(fold_named(e));
        return plan::project(in[0], es);
    }
    case K::Filter: return plan::filter(in[0], fold(n->exprs[0]));
    case K::Aggregate: {
        std::vector<Expr> keys;
        for (const auto& k : n->keys) keys.push_back(fold_named(k));
        std::vector<AggExpr> aggs;
        for (const auto& a : n->aggs) {
            if (!a.input()) {
                aggs.push_back(a);
                continue;
            }
            const Expr f = fold(*a.input());
            if (f.same_as(*a.input())) {
                aggs.push_back(a);
                continue;
            }
            AggExpr na(a.fn(), f);
            aggs.push_back(na.alias(a.name()));
        }
        return plan::aggregate(in[0], keys, aggs);
    }
    case K::Sort: {
        std::vector<SortKey> keys = n->sort_keys;
        for (auto& k : keys) k.expr = fold(k.expr);
        return plan::sort(in[0], keys);
    }
    default: return in.empty() ? n : plan::with_inputs(n, in);
    }
}

// ----- PushDownPredicates / EliminateOuterJoin / InferJoinPredicates -----------

PlanPtr push_filters(const PlanPtr& n);

inline PlanPtr push_into_join(const PlanPtr& j, const std::vector<Expr>& preds) {
    const PlanPtr& L = j->input(0);
    const PlanPtr& R = j->input(1);
    const auto lcols = plan::names_of(L->schema), rcols = plan::names_of(R->schema);
    JoinType jt = j->join_type;

    bool left_nonnull = false, right_nonnull = false;
    for (const auto& c : preds) {
        if (refs_within(c, rcols) && !refs_within(c, lcols) && null_rejecting(c, R->schema)) right_nonnull = true;
        if (refs_within(c, lcols) && !refs_within(c, rcols) && null_rejecting(c, L->schema)) left_nonnull = true;
    }
    if (jt == JoinType::Left && right_nonnull) jt = JoinType::Inner;
    else if (jt == JoinType::Right && left_nonnull) jt = JoinType::Inner;
    else if (jt == JoinType::Full) {
        if (left_nonnull && right_nonnull) jt = JoinType::Inner;
        else if (left_nonnull) jt = JoinType::Left;
        else if (right_nonnull) jt = JoinType::Right;
    }

    const bool left_ok = jt == JoinType::Inner || jt == JoinType::Left || jt == JoinType::LeftSemi ||
                         jt == JoinType::LeftAnti || jt == JoinType::Cross;
    const bool right_ok = jt == JoinType::Inner || jt == JoinType::Right || jt == JoinType::Cross;
    std::vector<Expr> lp, rp, rest;
    for (const auto& c : preds) {
        const bool lonly = refs_within(c, lcols), ronly = refs_within(c, rcols);
        if (lonly && left_ok) lp.push_back(c);
        else if (ronly && !lonly && right_ok) rp.push_back(c);
        else rest.push_back(c);
    }
    // Predicates on join keys hold on the other side too.
    std::map<std::string, Expr> l2r, r2l;
    for (size_t i = 0; i < j->left_keys.size(); ++i) {
        l2r.emplace(j->left_keys[i], col(j->right_keys[i]));
        r2l.emplace(j->right_keys[i], col(j->left_keys[i]));
    }
    const std::set<std::string> lkeys(j->left_keys.begin(), j->left_keys.end());
    const std::set<std::string> rkeys(j->right_keys.begin(), j->right_keys.end());
    const size_t nl = lp.size(), nr = rp.size();
    if (jt == JoinType::Inner || jt == JoinType::Left || jt == JoinType::LeftSemi || jt == JoinType::LeftAnti)
        for (size_t i = 0; i < nl; ++i)
            if (!plan::refs(lp[i]).empty() && refs_within(lp[i], lkeys)) rp.push_back(lp[i].substitute(l2r));
    if (jt == JoinType::Inner || jt == JoinType::Right)
        for (size_t i = 0; i < nr; ++i)
            if (!plan::refs(rp[i]).empty() && refs_within(rp[i], rkeys)) lp.push_back(rp[i].substitute(r2l));

    const PlanPtr nl_plan = push_filters(lp.empty() ? L : plan::filter(L, conjoin(lp)));
    const PlanPtr nr_plan = push_filters(rp.empty() ? R : plan::filter(R, conjoin(rp)));
    PlanPtr out = plan::join(nl_plan, nr_plan, jt, j->left_keys, j->right_keys);
    return rest.empty() ? out : plan::filter(out, conjoin(rest));
}

inline PlanPtr push_filter_node(const PlanPtr& f) {
    const PlanPtr& c = f->input();
    const Expr& p = f->exprs[0];
    const std::vector<Expr> preds = conjuncts(p);
    switch (c->kind) {
    case K::Filter: {
        std::vector<Expr> all = preds;
        for (auto& e : conjuncts(c->exprs[0])) all.push_back(e);
        return push_filters(plan::filter(c->input(), conjoin(all)));
    }
    case K::Project: {
        std::map<std::string, Expr> by_name;
        for (const auto& e : c->exprs) by_name.emplace(e.name(), e);
        return plan::project(push_filters(plan::filter(c->input(), p.substitute(by_name))), c->exprs);
    }
    case K::Sort: return plan::sort(push_filters(plan::filter(c->input(), p)), c->sort_keys);
    case K::Repartition:
        return plan::repartition(push_filters(plan::filter(c->input(), p)), c->partitions, c->keys);
    case K::Join: return push_into_join(c, preds);
    case K::Aggregate: {
        std::map<std::string, Expr> key_by_name;
        std::set<std::string> key_names;
        for (const auto& k : c->keys) {
            key_by_name.emplace(k.name(), k);
            key_names.insert(k.name());
        }
        std::vector<Expr> down, up;
        for (const auto& e : preds) {
            if (!c->keys.empty() && !plan::refs(e).empty() && refs_within(e, key_names))
                down.push_back(e.substitute(key_by_name));
            else
                up.push_back(e);
        }
        const PlanPtr in = push_filters(down.empty() ? c->input() : plan::filter(c->input(), conjoin(down)));
        PlanPtr out = plan::aggregate(in, c->keys, c->aggs);
        return up.empty() ? out : plan::filter(out, conjoin(up));
    }
    case K::Union: {
        std::vector<PlanPtr> ins;
        for (const auto& in : c->inputs) {
            std::map<std::string, Expr> by_pos;
            for (size_t i = 0; i < c->schema.size(); ++i) by_pos.emplace(c->schema[i].name, col(in->schema[i].name));
            ins.push_back(push_filters(plan::filter(in, p.substitute(by_pos))));
        }
        return plan::union_all(ins);
    }
    case K::Window: {
        const auto& spec = c->exprs.at(0).strip_alias().window_spec();
        std::set<std::string> part_cols;
        for (const auto& e : spec->partition)
            if (e.is_column()) part_cols.insert(e.column_name());
        std::vector<Expr> down, up;
        for (const auto& e : preds) {
            if (!plan::refs(e).empty() && refs_within(e, part_cols)) down.push_back(e);
            else up.push_back(e);
        }
        const PlanPtr in = push_filters(down.empty() ? c->input() : plan::filter(c->input(), conjoin(down)));
        PlanPtr out = plan::window(in, c->exprs, c->group_limit);
        return up.empty() ? out : plan::filter(out, conjoin(up));
    }
    default: return plan::filter(push_filters(c), p);
    }
}

inline PlanPtr push_filters(const PlanPtr& n) {
    if (n->kind == K::Filter) return push_filter_node(n);
    if (n->inputs.empty()) return n;
    std::vector<PlanPtr> in;
    for (const auto& c : n->inputs) in.push_back(push_filters(c));
    return plan::with_inputs(n, in);
}

// ----- simplification rules ----------------------------------------------------

inline PlanPtr strip_sorts(const PlanPtr& n) {
    if (n->kind == K::Sort) return strip_sorts(n->input());
    if (n->kind == K::Project || n->kind == K::Filter) return plan::with_inputs(n, {strip_sorts(n->input())});
    return n;
}

inline bool is_identity_project(const PlanNode& p) {
    const Schema& in = p.input()->schema;
    if (p.exprs.size() != in.size()) return false;
    for (size_t i = 0; i < in.size(); ++i)
        if (!p.exprs[i].is_column() || p.exprs[i].column_name() != in[i].name) return false;
    return true;
}

inline std::optional<PlanPtr> collapse_projects(const PlanNode& outer) {
    const PlanNode& inner = *outer.input();
    std::map<std::string, Expr> by_name;
    std::map<std::string, int> uses;
    for (const auto& e : inner.exprs) by_name.emplace(e.name(), e);
    for (const auto& e : outer.exprs)
        for (const auto& r : plan::refs(e)) ++uses[r];
    for (const auto& e : inner.exprs) {
        const Expr s = e.strip_alias();
        if (!s.is_column() && !s.is_literal() && uses[e.name()] > 1) return std::nullopt;
    }
    std::vector<Expr> es;
    for (const auto& e : outer.exprs) {
        const Expr sub = e.substitute(by_name);
        es.push_back(sub.name() == e.name() ? sub : sub.strip_alias().alias(e.name()));
    }
    return plan::project(inner.input(), es);
}

inline PlanPtr simplify(const PlanPtr& node) {
    std::vector<PlanPtr> in;
    for (const auto& c : node->inputs) in.push_back(simplify(c));
    PlanPtr n = in.empty() ? node : plan::with_inputs(node, in);
    auto is_empty = [](const PlanPtr& p) { return p->kind == K::Empty; };

    switch (n->kind) {
    case K::Filter: {
        const Expr& p = n->exprs[0];
        if (is_bool_literal(p, true)) return n->input();
        if (is_bool_literal(p, false) || p.is_null_literal() || is_empty(n->input())) return plan::empty(n->schema);
        if (n->input()->kind == K::Filter)
            return plan::filter(n->input()->input(), n->input()->exprs[0] && p);
        return n;
    }
    case K::Project: {
        if (is_empty(n->input())) return plan::empty(n->schema);
        if (is_identity_project(*n)) return n->input();
        if (n->input()->kind == K::Project) {
            if (auto c = collapse_projects(*n)) return simplify(*c);
        }
        if (n->input()->kind == K::Scan) {
            std::vector<std::string> cols;
            for (const auto& e : n->exprs) {
                if (!e.is_column()) return n;
                cols.push_back(e.column_name());
            }
            return plan::scan(n->input()->source, cols);
        }
        return n;
    }
    case K::Sort: {
        if (is_empty(n->input())) return plan::empty(n->schema);
        const PlanPtr c = strip_sorts(n->input());
        return c == n->input() ? n : plan::sort(c, n->sort_keys);
    }
    case K::Aggregate: {
        if (is_empty(n->input()) && !n->keys.empty()) return plan::empty(n->schema);
        const PlanPtr c = strip_sorts(n->input());
        return c == n->input() ? n : plan::aggregate(c, n->keys, n->aggs);
    }
    case K::Window:
    case K::Repartition: {
        if (is_empty(n->input())) return plan::empty(n->schema);
        const PlanPtr c = strip_sorts(n->input());
        return c == n->input() ? n : plan::with_inputs(n, {c});
    }
    case K::Join: {
        const bool le = is_empty(n->input(0)), re = is_empty(n->input(1));
        const JoinType t = n->join_type;
        if ((le && t != JoinType::Right && t != JoinType::Full) ||
            (re && (t == JoinType::Inner || t == JoinType::Right || t == JoinType::LeftSemi || t == JoinType::Cross)))
            return plan::empty(n->schema);
        const PlanPtr l = strip_sorts(n->input(0)), r = strip_sorts(n->input(1));
        if (l == n->input(0) && r == n->input(1)) return n;
        return plan::join(l, r, t, n->left_keys, n->right_keys);
    }
    case K::Limit: {
        const PlanPtr& c = n->input();
        if (n->limit == 0 || is_empty(c)) return plan::empty(n->schema);
        if (c->kind == K::Limit) return plan::limit(c->input(), std::min(n->limit, c->limit));
        if (c->kind == K::Project) return plan::project(simplify(plan::limit(c->input(), n->limit)), c->exprs);
        if (c->kind == K::Union) {
            bool changed = false;
            std::vector<PlanPtr> ins;
            for (const auto& u : c->inputs) {
                if (u->kind == K::Limit && u->limit <= n->limit) {
                    ins.push_back(u);
                } else {
                    ins.push_back(plan::limit(u, n->limit));
                    changed = true;
                }
            }
            if (changed) return plan::limit(plan::union_all(ins), n->limit);
        }
        return n;
    }
    case K::Union: {
        std::vector<PlanPtr> keep;
        for (const auto& u : n->inputs)
            if (!is_empty(u)) keep.push_back(u);
        if (keep.empty()) return plan::empty(n->schema);
        if (keep.size() == n->inputs.size()) return n;
        if (keep.size() == 1) {
            // rename to the union's column names
            std::vector<Expr> es;
            for (size_t i = 0; i < n->schema.size(); ++i) {
                const std::string& from = keep[0]->schema[i].name;
                es.push_back(from == n->schema[i].name ? col(from) : col(from).alias(n->schema[i].name));
            }
            return plan::project(keep[0], es);
        }
        return plan::union_all(keep);
    }
    default: return n;
    }
}

// ----- PushAggregateThroughJoin --------------------------------------------------

inline bool decomposable(const AggExpr& a) {
    switch (a.fn()) {
    case AggFn::Sum:
    case AggFn::Count:
    case AggFn::CountAll:
    case AggFn::Min:
    case AggFn::Max:
    case AggFn::Avg: return true;
    default: return false;
    }
}

PlanPtr push_aggregates(const PlanPtr& n);

inline std::optional<PlanPtr> try_push_aggregate(const PlanPtr& agg) {
    PlanPtr j = agg->input();
    std::map<std::string, std::string> rename;  // agg-input name -> join output name
    if (j->kind == K::Project) {
        for (const auto& e : j->exprs) {
            const Expr s = e.strip_alias();
            if (!s.is_column()) return std::nullopt;
            rename[e.name()] = s.column_name();
        }
        j = j->input();
    }
    if (j->kind != K::Join || j->join_type != JoinType::Inner) return std::nullopt;
    auto resolve = [&](const std::string& name) {
        auto it = rename.find(name);
        return it == rename.end() ? name : it->second;
    };
    std::map<std::string, Expr> subst;
    for (const auto& [from, to] : rename) subst.emplace(from, col(to));

    for (const auto& k : agg->keys)
        if (!k.strip_alias().is_column()) return std::nullopt;
    std::set<std::string> input_refs;
    for (const auto& a : agg->aggs) {
        if (!decomposable(a)) return std::nullopt;
        if (a.input()) plan::add_refs(a.input()->substitute(subst), input_refs);
    }
    const PlanPtr& L = j->input(0);
    const PlanPtr& R = j->input(1);
    const auto lcols = plan::names_of(L->schema), rcols = plan::names_of(R->schema);
    bool s_left;
    if (input_refs.empty()) s_left = plan::estimate_rows(L) >= plan::estimate_rows(R);
    else if (std::all_of(input_refs.begin(), input_refs.end(), [&](const std::string& c) { return lcols.count(c); }))
        s_left = true;
    else if (std::all_of(input_refs.begin(), input_refs.end(), [&](const std::string& c) { return rcols.count(c); }))
        s_left = false;
    else
        return std::nullopt;
    const PlanPtr& S = s_left ? L : R;
    const PlanPtr& O = s_left ? R : L;
    const auto& s_keys = s_left ? j->left_keys : j->right_keys;
    const auto& s_cols = s_left ? lcols : rcols;
    // Grouping keys from S must be join keys, so S is pre-aggregated by its
    // join keys only (the reduction this rule relies on).
    const std::set<std::string> s_key_set(s_keys.begin(), s_keys.end());
    for (const auto& k : agg->keys) {
        const std::string c = resolve(k.strip_alias().column_name());
        if (s_cols.count(c) && !s_key_set.count(c)) return std::nullopt;
    }
    // Only worth it when pre-aggregating S by its join keys shrinks it a lot:
    // S must have at least 4x more rows than (estimated) distinct join keys.
    // Without statistics for the keys, do not risk it.
    std::vector<Expr> key_exprs;
    for (const auto& k : s_keys) key_exprs.push_back(col(k));
    const auto groups = plan::estimate_groups(S, key_exprs);
    if (!groups || *groups * 4 > plan::estimate_rows(S)) return std::nullopt;

    // Partial aggregates on S.
    std::vector<AggExpr> partial, final_aggs;
    bool has_avg = false;
    for (size_t i = 0; i < agg->aggs.size(); ++i) {
        const AggExpr& a = agg->aggs[i];
        const std::string p = "__pa" + std::to_string(i), q = "__pc" + std::to_string(i);
        std::optional<Expr> x;
        if (a.input()) x = a.input()->substitute(subst);
        switch (a.fn()) {
        case AggFn::Sum:
            partial.push_back(sum(*x).alias(p));
            final_aggs.push_back(sum(col(p)).alias(a.name()));
            break;
        case AggFn::Count:
            partial.push_back(count(*x).alias(p));
            final_aggs.push_back(sum(col(p)).alias(a.name()));
            break;
        case AggFn::CountAll:
            partial.push_back(count_all().alias(p));
            final_aggs.push_back(sum(col(p)).alias(a.name()));
            break;
        case AggFn::Min:
            partial.push_back(min(*x).alias(p));
            final_aggs.push_back(min(col(p)).alias(a.name()));
            break;
        case AggFn::Max:
            partial.push_back(max(*x).alias(p));
            final_aggs.push_back(max(col(p)).alias(a.name()));
            break;
        case AggFn::Avg:
            has_avg = true;
            partial.push_back(sum(cast(*x, DType::Float64)).alias(p));
            partial.push_back(count(*x).alias(q));
            final_aggs.push_back(sum(col(p)).alias(p));
            final_aggs.push_back(sum(col(q)).alias(q));
            break;
        default: return std::nullopt;
        }
    }
    std::vector<Expr> s_group;
    for (const auto& k : s_keys) s_group.push_back(col(k));
    const PlanPtr s_agg = push_aggregates(plan::aggregate(S, s_group, partial));
    const PlanPtr nj = s_left ? plan::join(s_agg, O, JoinType::Inner, j->left_keys, j->right_keys)
                              : plan::join(O, s_agg, JoinType::Inner, j->left_keys, j->right_keys);
    std::vector<Expr> keys;
    for (const auto& k : agg->keys) {
        const std::string c = resolve(k.strip_alias().column_name());
        keys.push_back(c == k.name() ? col(c) : col(c).alias(k.name()));
    }
    PlanPtr out = plan::aggregate(nj, keys, final_aggs);
    if (has_avg) {
        std::vector<Expr> es;
        for (const auto& k : agg->keys) es.push_back(col(k.name()));
        for (size_t i = 0; i < agg->aggs.size(); ++i) {
            const AggExpr& a = agg->aggs[i];
            if (a.fn() == AggFn::Avg) {
                const Expr s = col("__pa" + std::to_string(i)), c = col("__pc" + std::to_string(i));
                es.push_back(when(c > 0, s / c).alias(a.name()));
            } else {
                es.push_back(col(a.name()));
            }
        }
        out = plan::project(out, es);
    }
    return out;
}

inline PlanPtr push_aggregates(const PlanPtr& n) {
    std::vector<PlanPtr> in;
    for (const auto& c : n->inputs) in.push_back(push_aggregates(c));
    PlanPtr m = in.empty() ? n : plan::with_inputs(n, in);
    if (m->kind == K::Aggregate)
        if (auto r = try_push_aggregate(m)) return *r;
    return m;
}

// ----- ColumnPruning -------------------------------------------------------------

inline PlanPtr prune(const PlanPtr& n, const std::set<std::string>& req) {
    switch (n->kind) {
    case K::Scan: {
        std::vector<std::string> cols;
        for (const auto& f : n->schema)
            if (req.count(f.name)) cols.push_back(f.name);
        if (cols.size() == n->schema.size()) return n;
        return plan::scan(n->source, cols);
    }
    case K::Empty: {
        Schema s;
        for (const auto& f : n->schema)
            if (req.count(f.name)) s.push_back(f);
        return plan::empty(s);
    }
    case K::Project: {
        std::vector<Expr> kept;
        std::set<std::string> child_req;
        for (const auto& e : n->exprs)
            if (req.count(e.name())) {
                kept.push_back(e);
                plan::add_refs(e, child_req);
            }
        return plan::project(prune(n->input(), child_req), kept);
    }
    case K::Filter: {
        std::set<std::string> r = req;
        plan::add_refs(n->exprs[0], r);
        return plan::filter(prune(n->input(), r), n->exprs[0]);
    }
    case K::Sort: {
        std::set<std::string> r = req;
        for (const auto& k : n->sort_keys) plan::add_refs(k.expr, r);
        return plan::sort(prune(n->input(), r), n->sort_keys);
    }
    case K::Limit: return plan::limit(prune(n->input(), req), n->limit);
    case K::Repartition: {
        std::set<std::string> r = req;
        for (const auto& k : n->keys) plan::add_refs(k, r);
        return plan::repartition(prune(n->input(), r), n->partitions, n->keys);
    }
    case K::Join: {
        const auto lcols = plan::names_of(n->input(0)->schema), rcols = plan::names_of(n->input(1)->schema);
        std::set<std::string> lr(n->left_keys.begin(), n->left_keys.end());
        std::set<std::string> rr(n->right_keys.begin(), n->right_keys.end());
        const bool semi = n->join_type == JoinType::LeftSemi || n->join_type == JoinType::LeftAnti;
        for (const auto& c : req) {
            if (lcols.count(c)) lr.insert(c);
            else if (!semi && rcols.count(c)) rr.insert(c);
        }
        return plan::join(prune(n->input(0), lr), prune(n->input(1), rr), n->join_type, n->left_keys,
                          n->right_keys);
    }
    case K::Aggregate: {
        std::vector<AggExpr> kept;
        std::set<std::string> r;
        for (const auto& k : n->keys) plan::add_refs(k, r);
        for (const auto& a : n->aggs)
            if (req.count(a.name())) {
                kept.push_back(a);
                if (a.input()) plan::add_refs(*a.input(), r);
            }
        return plan::aggregate(prune(n->input(), r), n->keys, kept);
    }
    case K::Window: {
        std::vector<Expr> kept;
        std::set<std::string> r;
        std::set<std::string> outs;
        for (const auto& e : n->exprs) outs.insert(e.name());
        for (const auto& c : req)
            if (!outs.count(c)) r.insert(c);
        for (const auto& e : n->exprs)
            if (req.count(e.name())) {
                kept.push_back(e);
                plan::add_refs(e, r);
            }
        if (kept.empty()) return prune(n->input(), r);
        return plan::window(prune(n->input(), r), kept, n->group_limit);
    }
    case K::Union: {
        std::vector<size_t> pos;
        for (size_t i = 0; i < n->schema.size(); ++i)
            if (req.count(n->schema[i].name)) pos.push_back(i);
        std::vector<PlanPtr> ins;
        for (const auto& in : n->inputs) {
            std::vector<std::string> names;
            for (size_t p : pos) names.push_back(in->schema[p].name);
            PlanPtr c = prune(in, std::set<std::string>(names.begin(), names.end()));
            bool exact = c->schema.size() == names.size();
            for (size_t i = 0; exact && i < names.size(); ++i) exact = c->schema[i].name == names[i];
            if (!exact) {
                std::vector<Expr> es;
                for (const auto& nm : names) es.push_back(col(nm));
                c = plan::project(c, es);
            }
            ins.push_back(c);
        }
        return plan::union_all(ins);
    }
    }
    return n;
}

// ----- WindowGroupLimit ------------------------------------------------------------
// filter(rn <= k) over a window computing only row_number/rank (with an
// ORDER BY): each input partition can drop all but its top k rows per window
// partition before the shuffle (Spark 3.5's WindowGroupLimit).

inline PlanPtr window_group_limit(const PlanPtr& n) {
    std::vector<PlanPtr> in;
    for (const auto& c : n->inputs) in.push_back(window_group_limit(c));
    PlanPtr m = in.empty() ? n : plan::with_inputs(n, in);
    if (m->kind != K::Filter || m->input()->kind != K::Window) return m;
    const PlanPtr& w = m->input();
    if (w->exprs.at(0).strip_alias().window_spec()->order.empty()) return m;
    std::set<std::string> ranks;
    for (const auto& e : w->exprs) {
        const std::string& f = e.strip_alias().func_name();
        if (f != "row_number" && f != "rank") return m;
        ranks.insert(e.name());
    }
    size_t limit = 0;
    for (const auto& c : conjuncts(m->exprs[0])) {
        if (c.kind() != Expr::Kind::Binary || c.children().size() != 2) continue;
        const Expr& a = c.children()[0];
        const Expr& b = c.children()[1];
        if (!a.is_column() || !ranks.count(a.column_name()) || !b.is_literal() || b.is_null_literal() ||
            b.literal_value().index() != 0)
            continue;
        const int64_t k = std::get<int64_t>(b.literal_value());
        int64_t lim = -1;
        if (c.op() == Expr::Op::Le) lim = k;
        else if (c.op() == Expr::Op::Lt) lim = k - 1;
        else if (c.op() == Expr::Op::Eq) lim = k;
        if (lim > 0 && (limit == 0 || static_cast<size_t>(lim) < limit)) limit = static_cast<size_t>(lim);
    }
    if (limit == 0 || (w->group_limit && w->group_limit <= limit)) return m;
    return plan::filter(plan::window(w->input(), w->exprs, limit), m->exprs[0]);
}

// ----- driver ----------------------------------------------------------------------

inline PlanPtr optimize(const PlanPtr& root) {
    PlanPtr p = fold_plan(root);
    for (int i = 0; i < 3; ++i) {
        p = push_filters(p);
        p = simplify(p);
    }
    p = push_aggregates(p);
    p = push_filters(p);
    p = simplify(p);
    p = prune(p, plan::names_of(root->schema));
    p = simplify(p);
    p = window_group_limit(p);
    bool same = p->schema.size() == root->schema.size();
    for (size_t i = 0; same && i < p->schema.size(); ++i) same = p->schema[i].name == root->schema[i].name;
    if (!same) {
        std::vector<Expr> es;
        for (const auto& f : root->schema) es.push_back(col(f.name));
        p = plan::project(p, es);
    }
    return p;
}

}  // namespace vega::df::opt
