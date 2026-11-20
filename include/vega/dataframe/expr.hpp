#pragma once

// Column expressions for Vega DataFrames:
//
//   col("price") * col("qty") > 100.0 && col("category") == "c7"
//   when(col("qty") > 5, "bulk").otherwise("retail")
//   coalesce(col("discount"), lit(0.0))
//
// Expressions are immutable trees, evaluated a whole column at a time
// (vectorized): one tight loop per operator, with literals applied as scalars
// rather than expanded into columns. Semantics follow Spark SQL (ANSI mode,
// the Spark 4 default):
//
//   * typing: integer arithmetic stays bigint, `/` always yields double, and
//     mixed int/double operands promote to double;
//   * nulls: arithmetic, comparisons and most functions return null when an
//     input is null; AND/OR use three-valued logic (false AND null = false,
//     true OR null = true); filters keep rows whose predicate is true;
//   * errors: division or modulo by zero, invalid casts and cast overflow
//     raise errors instead of returning null.
//
// Expressions that read a single dictionary-encoded column are evaluated
// once per distinct value and then gathered by code.

#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "vega/dataframe/batch.hpp"

namespace vega::df {

class Expr;
struct WindowSpec;

// A sort key: ascending sorts put nulls first, descending ones put them
// last (Spark's defaults); asc_nulls_last / desc_nulls_first override.
struct SortKey;

namespace detail {
inline std::string format_double(double d) {
    if (std::isnan(d)) return "NaN";
    if (std::isinf(d)) return d > 0 ? "Infinity" : "-Infinity";
    char buf[64];
    for (int prec = 1; prec <= 17; ++prec) {
        std::snprintf(buf, sizeof buf, "%.*g", prec, d);
        if (std::strtod(buf, nullptr) == d) break;
    }
    std::string s = buf;
    if (s.find_first_of(".eEn") == std::string::npos) s += ".0";
    return s;
}
inline size_t utf8_length(const std::string& s) {
    size_t n = 0;
    for (unsigned char c : s) n += (c & 0xC0) != 0x80;
    return n;
}
// Byte offset of code point `cp` (clamped to the end).
inline size_t utf8_offset(const std::string& s, size_t cp) {
    size_t i = 0;
    while (i < s.size() && cp > 0) {
        ++i;
        while (i < s.size() && (static_cast<unsigned char>(s[i]) & 0xC0) == 0x80) ++i;
        --cp;
    }
    return i;
}
// SQL LIKE: % matches any run, _ one character, \ escapes.
inline bool like_match(const std::string& s, const std::string& p) {
    size_t si = 0, pi = 0, star = std::string::npos, mark = 0;
    while (si < s.size()) {
        if (pi < p.size() && p[pi] == '\\' && pi + 1 < p.size() && p[pi + 1] == s[si]) {
            pi += 2;
            ++si;
        } else if (pi < p.size() && (p[pi] == '_' || (p[pi] == s[si] && p[pi] != '%' && p[pi] != '\\'))) {
            ++pi;
            ++si;
        } else if (pi < p.size() && p[pi] == '%') {
            star = pi++;
            mark = si;
        } else if (star != std::string::npos) {
            pi = star + 1;
            si = ++mark;
        } else {
            return false;
        }
    }
    while (pi < p.size() && p[pi] == '%') ++pi;
    return pi == p.size();
}
}  // namespace detail

class Expr {
public:
    enum class Op { Add, Sub, Mul, Div, Mod, Eq, Ne, Lt, Le, Gt, Ge, And, Or, EqNullSafe };
    enum class Kind { Column, Literal, Binary, Not, Neg, Alias, Func, Case, Cast, Window };
    using Scalar = std::variant<int64_t, double, std::string, bool>;

    // Literals convert implicitly, so `col("x") > 3` and `col("s") == "a"` work.
    Expr(int v) : Expr(Scalar(static_cast<int64_t>(v))) {}
    Expr(long v) : Expr(Scalar(static_cast<int64_t>(v))) {}
    Expr(long long v) : Expr(Scalar(static_cast<int64_t>(v))) {}
    Expr(double v) : Expr(Scalar(v)) {}
    Expr(const char* v) : Expr(Scalar(std::string(v))) {}
    Expr(std::string v) : Expr(Scalar(std::move(v))) {}
    Expr(bool v) : Expr(Scalar(v)) {}

    static Expr column(std::string name) {
        auto n = std::make_shared<Node>();
        n->kind = Kind::Column;
        n->name = std::move(name);
        return Expr(std::move(n));
    }
    // A typed null literal.
    static Expr null_of(DType t) {
        auto n = std::make_shared<Node>();
        n->kind = Kind::Literal;
        n->null = true;
        n->type = t;
        n->literal = default_scalar(t);
        return Expr(std::move(n));
    }
    static Expr literal(const Scalar& s) { return Expr(s); }
    static Expr func(std::string name, std::vector<Expr> args) {
        auto n = std::make_shared<Node>();
        n->kind = Kind::Func;
        n->name = std::move(name);
        n->children = std::move(args);
        return Expr(std::move(n));
    }
    static Expr cast(const Expr& e, DType t) {
        auto n = std::make_shared<Node>();
        n->kind = Kind::Cast;
        n->type = t;
        n->children = {e};
        return Expr(std::move(n));
    }
    // CASE WHEN c1 THEN v1 ... [ELSE e] END; `branches` alternates cond, value.
    static Expr case_when(std::vector<Expr> branches, std::optional<Expr> otherwise) {
        auto n = std::make_shared<Node>();
        n->kind = Kind::Case;
        n->children = std::move(branches);
        if (otherwise) {
            n->children.push_back(*otherwise);
            n->has_else = true;
        }
        return Expr(std::move(n));
    }
    static Expr window(std::string fn, std::vector<Expr> args, std::shared_ptr<const WindowSpec> spec,
                       int64_t offset = 0) {
        auto n = std::make_shared<Node>();
        n->kind = Kind::Window;
        n->name = std::move(fn);
        n->children = std::move(args);
        n->window = std::move(spec);
        n->offset = offset;
        return Expr(std::move(n));
    }

    // Rename the expression's output column.
    Expr alias(std::string name) const {
        auto n = std::make_shared<Node>();
        n->kind = Kind::Alias;
        n->name = std::move(name);
        n->children = {*this};
        return Expr(std::move(n));
    }

    // ----- builders for CASE WHEN: when(c, v).when(c2, v2).otherwise(e) -----
    Expr when(const Expr& cond, const Expr& value) const {
        if (n_->kind != Kind::Case || n_->has_else)
            throw std::invalid_argument("when() must follow when(...)");
        std::vector<Expr> b = n_->children;
        b.push_back(cond);
        b.push_back(value);
        return case_when(std::move(b), std::nullopt);
    }
    Expr otherwise(const Expr& value) const {
        if (n_->kind != Kind::Case || n_->has_else)
            throw std::invalid_argument("otherwise() must follow when(...)");
        return case_when(n_->children, value);
    }

    // ----- method-style helpers (Spark Column API) --------------------------
    Expr is_null() const { return func("isnull", {*this}); }
    Expr is_not_null() const { return func("isnotnull", {*this}); }
    Expr isin(std::vector<Expr> values) const {
        values.insert(values.begin(), *this);
        return func("in", std::move(values));
    }
    Expr between(const Expr& lo, const Expr& hi) const;
    Expr cast(DType t) const { return cast(*this, t); }
    Expr eq_null_safe(const Expr& o) const;
    Expr like(const std::string& pattern) const { return func("like", {*this, Expr(pattern)}); }
    Expr contains(const Expr& s) const { return func("contains", {*this, s}); }
    Expr starts_with(const Expr& s) const { return func("startswith", {*this, s}); }
    Expr ends_with(const Expr& s) const { return func("endswith", {*this, s}); }
    Expr substr(int64_t pos, int64_t len) const { return func("substring", {*this, Expr(pos), Expr(len)}); }

    // ----- introspection (used by the optimizer) ----------------------------
    Kind kind() const { return n_->kind; }
    Op op() const { return n_->op; }
    const std::vector<Expr>& children() const { return n_->children; }
    const std::string& func_name() const { return n_->name; }
    bool is_column() const { return n_->kind == Kind::Column; }
    bool is_literal() const { return n_->kind == Kind::Literal; }
    bool is_null_literal() const { return n_->kind == Kind::Literal && n_->null; }
    const Scalar& literal_value() const { return n_->literal; }
    const std::string& column_name() const { return n_->name; }
    bool is_alias() const { return n_->kind == Kind::Alias; }
    Expr strip_alias() const { return n_->kind == Kind::Alias ? n_->children[0].strip_alias() : *this; }
    DType cast_type() const { return n_->type; }
    bool has_else() const { return n_->has_else; }
    const std::shared_ptr<const WindowSpec>& window_spec() const { return n_->window; }
    int64_t window_offset() const { return n_->offset; }
    bool is_window() const { return strip_alias().kind() == Kind::Window; }
    bool contains_window() const {
        if (n_->kind == Kind::Window) return true;
        for (const auto& c : n_->children)
            if (c.contains_window()) return true;
        return false;
    }

    // Names of the input columns the expression reads (appended to `out`).
    void referenced_columns(std::vector<std::string>& out) const;

    // Replace column references by expressions (aliases are kept).
    Expr substitute(const std::map<std::string, Expr>& by_name) const {
        if (n_->kind == Kind::Column) {
            auto it = by_name.find(n_->name);
            return it == by_name.end() ? *this : it->second.strip_alias();
        }
        if (n_->children.empty()) return *this;
        auto n = std::make_shared<Node>(*n_);
        for (auto& c : n->children) c = c.substitute(by_name);
        return Expr(std::move(n));
    }
    Expr with_children(std::vector<Expr> kids) const {
        auto n = std::make_shared<Node>(*n_);
        n->children = std::move(kids);
        return Expr(std::move(n));
    }

    // Output column name, following Spark's conventions, e.g. "(qty * price)".
    std::string name() const;
    // Unambiguous rendering (string literals quoted) for plans and equality.
    std::string debug_string() const;
    bool same_as(const Expr& o) const { return n_ == o.n_ || debug_string() == o.debug_string(); }

    // Result type against a schema; also validates the expression.
    DType type(const Schema& schema) const;

    // Evaluate against a batch, producing a column with one value per row.
    Column eval(const Batch& batch) const;

    // ----- builders -----------------------------------------------------------
    friend Expr make_binary(Op op, const Expr& a, const Expr& b) {
        auto n = std::make_shared<Node>();
        n->kind = Kind::Binary;
        n->op = op;
        n->children = {a, b};
        return Expr(std::move(n));
    }
    friend Expr operator!(const Expr& a) {
        auto n = std::make_shared<Node>();
        n->kind = Kind::Not;
        n->children = {a};
        return Expr(std::move(n));
    }
    friend Expr operator-(const Expr& a) {
        auto n = std::make_shared<Node>();
        n->kind = Kind::Neg;
        n->children = {a};
        return Expr(std::move(n));
    }

    static DType scalar_type(const Scalar& s) {
        switch (s.index()) {
        case 0: return DType::Int64;
        case 1: return DType::Float64;
        case 2: return DType::String;
        default: return DType::Bool;
        }
    }
    static bool numeric(DType t) { return t == DType::Int64 || t == DType::Float64; }
    // Common type of branches/arguments (int + double -> double).
    static DType unify(DType a, DType b, const std::string& what) {
        if (a == b) return a;
        if (numeric(a) && numeric(b)) return DType::Float64;
        throw std::invalid_argument("incompatible types " + std::string(dtype_name(a)) + " and " +
                                    dtype_name(b) + " in " + what);
    }

private:
    struct Node {
        Kind kind = Kind::Literal;
        Op op = Op::Add;
        std::string name;
        Scalar literal;
        bool null = false;        // null literal
        DType type = DType::Int64;  // null literal / cast target
        bool has_else = false;
        int64_t offset = 0;       // lag / lead
        std::vector<Expr> children;
        std::shared_ptr<const WindowSpec> window;
    };

    // A column, or a scalar standing for a column of identical values.
    struct Value {
        bool is_scalar = false;
        bool null = false;  // scalar null
        Scalar scalar;
        Column col;
        DType type = DType::Int64;
    };

    explicit Expr(Scalar s) {
        auto n = std::make_shared<Node>();
        n->kind = Kind::Literal;
        n->literal = std::move(s);
        n->type = scalar_type(n->literal);
        n_ = std::move(n);
    }
    explicit Expr(std::shared_ptr<const Node> n) : n_(std::move(n)) {}

    static Scalar default_scalar(DType t) {
        switch (t) {
        case DType::Int64: return int64_t{0};
        case DType::Float64: return 0.0;
        case DType::String: return std::string();
        case DType::Bool: return false;
        }
        return int64_t{0};
    }

    static void require(bool ok, const std::string& msg) {
        if (!ok) throw std::invalid_argument(msg);
    }

    static const char* op_symbol(Op op) {
        switch (op) {
        case Op::Add: return "+";
        case Op::Sub: return "-";
        case Op::Mul: return "*";
        case Op::Div: return "/";
        case Op::Mod: return "%";
        case Op::Eq: return "=";
        case Op::Ne: return "!=";
        case Op::Lt: return "<";
        case Op::Le: return "<=";
        case Op::Gt: return ">";
        case Op::Ge: return ">=";
        case Op::And: return "AND";
        case Op::Or: return "OR";
        case Op::EqNullSafe: return "<=>";
        }
        return "?";
    }

    static std::string literal_text(const Scalar& s) {
        switch (s.index()) {
        case 0: return std::to_string(std::get<int64_t>(s));
        case 1: return detail::format_double(std::get<double>(s));
        case 2: return std::get<std::string>(s);
        default: return std::get<bool>(s) ? "true" : "false";
        }
    }

    static DType binary_type(Op op, DType a, DType b, const std::string& what) {
        switch (op) {
        case Op::Add:
        case Op::Sub:
        case Op::Mul:
        case Op::Mod:
            require(numeric(a) && numeric(b), "arithmetic needs numbers: " + what);
            return a == DType::Int64 && b == DType::Int64 ? DType::Int64 : DType::Float64;
        case Op::Div:
            require(numeric(a) && numeric(b), "arithmetic needs numbers: " + what);
            return DType::Float64;
        case Op::And:
        case Op::Or:
            require(a == DType::Bool && b == DType::Bool, "AND/OR need booleans: " + what);
            return DType::Bool;
        case Op::Eq:
        case Op::Ne:
        case Op::EqNullSafe:
            require((numeric(a) && numeric(b)) || a == b,
                    "cannot compare " + std::string(dtype_name(a)) + " with " + dtype_name(b) + ": " + what);
            return DType::Bool;
        default:  // ordering comparisons
            require((numeric(a) && numeric(b)) || (a == b && a != DType::Bool),
                    "cannot order " + std::string(dtype_name(a)) + " and " + dtype_name(b) + ": " + what);
            return DType::Bool;
        }
    }

    DType func_type(const Schema& schema) const;

    static Column broadcast(const Value& v, size_t n) {
        if (!v.is_scalar) return v.col;
        if (v.null) return Column::nulls(v.type, n);
        switch (v.scalar.index()) {
        case 0: return Column(std::vector<int64_t>(n, std::get<int64_t>(v.scalar)));
        case 1: return Column(std::vector<double>(n, std::get<double>(v.scalar)));
        case 2: return Column(std::vector<std::string>(n, std::get<std::string>(v.scalar)));
        default: return Column(BoolVec(n, std::get<bool>(v.scalar) ? 1 : 0));
        }
    }

    // Validity of a op b: null where either is null.
    static Column::Validity combine_validity(const Value& a, const Value& b, size_t n) {
        const uint8_t* va = a.is_scalar ? nullptr : a.col.validity();
        const uint8_t* vb = b.is_scalar ? nullptr : b.col.validity();
        if (!va && !vb) return nullptr;
        BoolVec m(n);
        for (size_t i = 0; i < n; ++i) m[i] = (!va || va[i]) && (!vb || vb[i]);
        return std::make_shared<const BoolVec>(std::move(m));
    }

    // Call f(getter) where getter(i) yields row i as int64_t or double.
    template<typename F>
    static decltype(auto) with_number(const Value& v, F&& f) {
        if (v.type == DType::Int64) {
            if (v.is_scalar) {
                const int64_t s = std::get<int64_t>(v.scalar);
                return f([s](size_t) { return s; });
            }
            const int64_t* p = v.col.values<int64_t>().data();
            return f([p](size_t i) { return p[i]; });
        }
        if (v.type == DType::Float64) {
            if (v.is_scalar) {
                const double s = std::get<double>(v.scalar);
                return f([s](size_t) { return s; });
            }
            const double* p = v.col.values<double>().data();
            return f([p](size_t i) { return p[i]; });
        }
        throw std::invalid_argument("expected a number");
    }

    template<typename F>
    static decltype(auto) with_string(const Value& v, F&& f) {
        if (v.is_scalar) {
            const std::string& s = std::get<std::string>(v.scalar);
            return f([&s](size_t) -> const std::string& { return s; });
        }
        const std::string* p = v.col.values<std::string>().data();
        return f([p](size_t i) -> const std::string& { return p[i]; });
    }

    template<typename F>
    static decltype(auto) with_bool(const Value& v, F&& f) {
        if (v.is_scalar) {
            const uint8_t s = std::get<bool>(v.scalar) ? 1 : 0;
            return f([s](size_t) { return s; });
        }
        const uint8_t* p = v.col.values<uint8_t>().data();
        return f([p](size_t i) { return p[i]; });
    }

    template<typename Out, typename GA, typename GB>
    static Column arith(Op op, GA ga, GB gb, size_t n, const uint8_t* valid) {
        std::vector<Out> out(n);
        switch (op) {
        case Op::Add: for (size_t i = 0; i < n; ++i) out[i] = Out(ga(i)) + Out(gb(i)); break;
        case Op::Sub: for (size_t i = 0; i < n; ++i) out[i] = Out(ga(i)) - Out(gb(i)); break;
        case Op::Mul: for (size_t i = 0; i < n; ++i) out[i] = Out(ga(i)) * Out(gb(i)); break;
        case Op::Div:
            for (size_t i = 0; i < n; ++i) {
                if (gb(i) == 0) {
                    if (!valid || valid[i]) throw std::runtime_error("[DIVIDE_BY_ZERO] Division by zero");
                    continue;
                }
                out[i] = Out(ga(i)) / Out(gb(i));
            }
            break;
        case Op::Mod:
            for (size_t i = 0; i < n; ++i) {
                if (gb(i) == 0) {
                    if (!valid || valid[i]) throw std::runtime_error("[DIVIDE_BY_ZERO] Division by zero");
                    continue;
                }
                if constexpr (std::is_same_v<Out, double>) out[i] = std::fmod(Out(ga(i)), Out(gb(i)));
                else out[i] = Out(ga(i)) % Out(gb(i));
            }
            break;
        default: throw std::logic_error("not arithmetic");
        }
        return Column(std::move(out));
    }

    template<typename GA, typename GB>
    static Column compare(Op op, GA ga, GB gb, size_t n) {
        BoolVec out(n);
        switch (op) {
        case Op::Eq:
        case Op::EqNullSafe: for (size_t i = 0; i < n; ++i) out[i] = ga(i) == gb(i); break;
        case Op::Ne: for (size_t i = 0; i < n; ++i) out[i] = ga(i) != gb(i); break;
        case Op::Lt: for (size_t i = 0; i < n; ++i) out[i] = ga(i) < gb(i); break;
        case Op::Le: for (size_t i = 0; i < n; ++i) out[i] = ga(i) <= gb(i); break;
        case Op::Gt: for (size_t i = 0; i < n; ++i) out[i] = ga(i) > gb(i); break;
        case Op::Ge: for (size_t i = 0; i < n; ++i) out[i] = ga(i) >= gb(i); break;
        default: throw std::logic_error("not a comparison");
        }
        return Column(std::move(out));
    }

    Value eval_value(const Batch& batch) const;
    Value eval_binary(const Batch& batch) const;
    Column eval_func(const Batch& batch) const;
    Column eval_case(const Batch& batch) const;
    bool dictionary_eval(const Batch& batch, Column& out) const;

    std::shared_ptr<const Node> n_;

    friend Column cast_column(const Column& c, DType t);
};

// ---------------------------------------------------------------------------
// Sort keys and window specifications
// ---------------------------------------------------------------------------

struct SortKey {
    Expr expr;
    bool ascending = true;
    bool nulls_first = true;
};
inline SortKey asc(Expr e) { return {std::move(e), true, true}; }
inline SortKey desc(Expr e) { return {std::move(e), false, false}; }
inline SortKey asc_nulls_last(Expr e) { return {std::move(e), true, false}; }
inline SortKey desc_nulls_first(Expr e) { return {std::move(e), false, true}; }

// Window::partition_by({...}).order_by({...}). With an ORDER BY, aggregate
// window functions use Spark's default frame (RANGE BETWEEN UNBOUNDED
// PRECEDING AND CURRENT ROW: a running total including ties); without one,
// they cover the whole partition.
struct WindowSpec {
    std::vector<Expr> partition;
    std::vector<SortKey> order;

    WindowSpec partition_by(std::vector<Expr> keys) const {
        WindowSpec w = *this;
        w.partition = std::move(keys);
        return w;
    }
    WindowSpec order_by(std::vector<SortKey> keys) const {
        WindowSpec w = *this;
        w.order = std::move(keys);
        return w;
    }
    std::string text() const {
        std::string s;
        if (!partition.empty()) {
            s += "PARTITION BY ";
            for (size_t i = 0; i < partition.size(); ++i) s += (i ? ", " : "") + partition[i].name();
        }
        if (!order.empty()) {
            if (!s.empty()) s += " ";
            s += "ORDER BY ";
            for (size_t i = 0; i < order.size(); ++i)
                s += (i ? ", " : "") + order[i].expr.name() + (order[i].ascending ? " ASC" : " DESC");
        }
        return s;
    }
};

struct Window {
    static WindowSpec partition_by(std::vector<Expr> keys) { return WindowSpec{}.partition_by(std::move(keys)); }
    static WindowSpec order_by(std::vector<SortKey> keys) { return WindowSpec{}.order_by(std::move(keys)); }
};

// ---------------------------------------------------------------------------
// Builders
// ---------------------------------------------------------------------------

inline Expr col(std::string name) { return Expr::column(std::move(name)); }
inline Expr lit(Expr e) { return e; }
inline Expr lit_null(DType t) { return Expr::null_of(t); }

#define VEGA_DF_BINOP(sym, op)                                                   \
    inline Expr operator sym(const Expr& a, const Expr& b) {                     \
        return make_binary(Expr::Op::op, a, b);                                  \
    }
VEGA_DF_BINOP(+, Add)
VEGA_DF_BINOP(-, Sub)
VEGA_DF_BINOP(*, Mul)
VEGA_DF_BINOP(/, Div)
VEGA_DF_BINOP(%, Mod)
VEGA_DF_BINOP(==, Eq)
VEGA_DF_BINOP(!=, Ne)
VEGA_DF_BINOP(<, Lt)
VEGA_DF_BINOP(<=, Le)
VEGA_DF_BINOP(>, Gt)
VEGA_DF_BINOP(>=, Ge)
VEGA_DF_BINOP(&&, And)
VEGA_DF_BINOP(||, Or)
#undef VEGA_DF_BINOP

inline Expr Expr::between(const Expr& lo, const Expr& hi) const { return (*this >= lo) && (*this <= hi); }
inline Expr Expr::eq_null_safe(const Expr& o) const { return make_binary(Op::EqNullSafe, *this, o); }
inline Expr when(const Expr& cond, const Expr& value) { return Expr::case_when({cond, value}, std::nullopt); }
inline Expr coalesce(std::vector<Expr> args) { return Expr::func("coalesce", std::move(args)); }
inline Expr is_null(const Expr& e) { return e.is_null(); }
inline Expr is_not_null(const Expr& e) { return e.is_not_null(); }
inline Expr cast(const Expr& e, DType t) { return Expr::cast(e, t); }
inline Expr upper(const Expr& e) { return Expr::func("upper", {e}); }
inline Expr lower(const Expr& e) { return Expr::func("lower", {e}); }
inline Expr trim(const Expr& e) { return Expr::func("trim", {e}); }
inline Expr length(const Expr& e) { return Expr::func("length", {e}); }
inline Expr substring(const Expr& e, int64_t pos, int64_t len) { return e.substr(pos, len); }
inline Expr concat(std::vector<Expr> args) { return Expr::func("concat", std::move(args)); }
inline Expr abs(const Expr& e) { return Expr::func("abs", {e}); }
inline Expr round(const Expr& e, int64_t scale = 0) { return Expr::func("round", {e, Expr(scale)}); }
inline Expr floor(const Expr& e) { return Expr::func("floor", {e}); }
inline Expr ceil(const Expr& e) { return Expr::func("ceil", {e}); }
inline Expr sqrt(const Expr& e) { return Expr::func("sqrt", {e}); }

// Ranking and offset window functions: row_number().over(w), lag(col("x"), 1).over(w).
struct WindowFunction {
    std::string fn;
    std::vector<Expr> args;
    int64_t offset = 0;
    Expr over(const WindowSpec& spec) const {
        return Expr::window(fn, args, std::make_shared<const WindowSpec>(spec), offset);
    }
};
inline WindowFunction row_number() { return {"row_number", {}, 0}; }
inline WindowFunction rank() { return {"rank", {}, 0}; }
inline WindowFunction dense_rank() { return {"dense_rank", {}, 0}; }
inline WindowFunction lag(const Expr& e, int64_t offset = 1) { return {"lag", {e}, offset}; }
inline WindowFunction lead(const Expr& e, int64_t offset = 1) { return {"lead", {e}, offset}; }
// With a default for rows that have no such neighbour.
inline WindowFunction lag(const Expr& e, int64_t offset, const Expr& def) { return {"lag", {e, def}, offset}; }
inline WindowFunction lead(const Expr& e, int64_t offset, const Expr& def) { return {"lead", {e, def}, offset}; }

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

inline void Expr::referenced_columns(std::vector<std::string>& out) const {
    if (n_->kind == Kind::Column) {
        out.push_back(n_->name);
        return;
    }
    for (const auto& c : n_->children) c.referenced_columns(out);
    if (n_->kind == Kind::Window && n_->window) {
        for (const auto& p : n_->window->partition) p.referenced_columns(out);
        for (const auto& k : n_->window->order) k.expr.referenced_columns(out);
    }
}

inline std::string Expr::name() const {
    const auto& c = n_->children;
    auto list = [&](size_t from, size_t to) {
        std::string s;
        for (size_t i = from; i < to; ++i) s += (i > from ? ", " : "") + c[i].name();
        return s;
    };
    switch (n_->kind) {
    case Kind::Column:
    case Kind::Alias: return n_->name;
    case Kind::Literal: return n_->null ? "NULL" : literal_text(n_->literal);
    case Kind::Not: return "(NOT " + c[0].name() + ")";
    case Kind::Neg: return "(- " + c[0].name() + ")";
    case Kind::Binary: return "(" + c[0].name() + " " + op_symbol(n_->op) + " " + c[1].name() + ")";
    case Kind::Cast: {
        std::string t = dtype_name(n_->type);
        for (auto& ch : t) ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
        return "CAST(" + c[0].name() + " AS " + t + ")";
    }
    case Kind::Case: {
        std::string s = "CASE";
        const size_t pairs = (c.size() - (n_->has_else ? 1 : 0)) / 2;
        for (size_t i = 0; i < pairs; ++i) s += " WHEN " + c[2 * i].name() + " THEN " + c[2 * i + 1].name();
        if (n_->has_else) s += " ELSE " + c.back().name();
        return s + " END";
    }
    case Kind::Func: {
        const std::string& f = n_->name;
        if (f == "isnull") return "(" + c[0].name() + " IS NULL)";
        if (f == "isnotnull") return "(" + c[0].name() + " IS NOT NULL)";
        if (f == "in") return "(" + c[0].name() + " IN (" + list(1, c.size()) + "))";
        if (f == "like") return c[0].name() + " LIKE " + c[1].name();
        if (f == "floor" || f == "ceil" || f == "sqrt") {
            std::string u = f;
            for (auto& ch : u) ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
            return u + "(" + c[0].name() + ")";
        }
        return f + "(" + list(0, c.size()) + ")";
    }
    case Kind::Window: {
        std::string args = list(0, c.size());
        if ((n_->name == "lag" || n_->name == "lead")) args += ", " + std::to_string(n_->offset);
        if (n_->name == "count_all") return "count(1) OVER (" + n_->window->text() + ")";
        return n_->name + "(" + args + ") OVER (" + n_->window->text() + ")";
    }
    }
    return "?";
}

inline std::string Expr::debug_string() const {
    const auto& c = n_->children;
    switch (n_->kind) {
    case Kind::Column: return n_->name;
    case Kind::Literal:
        if (n_->null) return std::string("null:") + dtype_name(n_->type);
        if (n_->literal.index() == 2) return "'" + std::get<std::string>(n_->literal) + "'";
        if (n_->literal.index() == 1) return literal_text(n_->literal) + "d";
        return literal_text(n_->literal);
    case Kind::Alias: return c[0].debug_string() + " AS " + n_->name;
    default: break;
    }
    std::string s = std::to_string(static_cast<int>(n_->kind)) + ":" + std::to_string(static_cast<int>(n_->op)) +
                    ":" + n_->name + ":" + std::to_string(static_cast<int>(n_->type)) + (n_->has_else ? "e" : "") +
                    ":" + std::to_string(n_->offset) + "(";
    for (size_t i = 0; i < c.size(); ++i) s += (i ? "," : "") + c[i].debug_string();
    s += ")";
    if (n_->kind == Kind::Window) {
        s += "[";
        for (const auto& p : n_->window->partition) s += p.debug_string() + ",";
        s += "|";
        for (const auto& k : n_->window->order)
            s += k.expr.debug_string() + (k.ascending ? "+" : "-") + (k.nulls_first ? "f" : "l") + ",";
        s += "]";
    }
    return s;
}

inline DType Expr::type(const Schema& schema) const {
    const auto& c = n_->children;
    switch (n_->kind) {
    case Kind::Column: {
        const int i = schema_index(schema, n_->name);
        if (i < 0)
            throw std::invalid_argument("cannot resolve column '" + n_->name + "' in " + schema_string(schema));
        return schema[static_cast<size_t>(i)].type;
    }
    case Kind::Literal: return n_->type;
    case Kind::Alias: return c[0].type(schema);
    case Kind::Not:
        require(c[0].type(schema) == DType::Bool, "NOT needs a boolean: " + name());
        return DType::Bool;
    case Kind::Neg: {
        const DType t = c[0].type(schema);
        require(numeric(t), "negation needs a number: " + name());
        return t;
    }
    case Kind::Binary: return binary_type(n_->op, c[0].type(schema), c[1].type(schema), name());
    case Kind::Cast: c[0].type(schema); return n_->type;
    case Kind::Case: {
        const size_t pairs = (c.size() - (n_->has_else ? 1 : 0)) / 2;
        require(pairs > 0, "CASE needs a WHEN");
        std::optional<DType> t;
        auto add = [&](const Expr& v) {
            const DType vt = v.type(schema);
            if (v.is_null_literal()) return;
            t = t ? unify(*t, vt, name()) : vt;
        };
        for (size_t i = 0; i < pairs; ++i) {
            require(c[2 * i].type(schema) == DType::Bool, "WHEN needs a boolean: " + c[2 * i].name());
            add(c[2 * i + 1]);
        }
        if (n_->has_else) add(c.back());
        return t ? *t : c[1].type(schema);
    }
    case Kind::Func: return func_type(schema);
    case Kind::Window: {
        const std::string& f = n_->name;
        for (const auto& p : n_->window->partition) p.type(schema);
        for (const auto& k : n_->window->order) k.expr.type(schema);
        if (f == "row_number" || f == "rank" || f == "dense_rank" || f == "count" || f == "count_all")
            return DType::Int64;
        require(!c.empty(), f + " needs an argument");
        const DType t = c[0].type(schema);
        if (f == "lag" || f == "lead") {
            if (c.size() > 1) return unify(t, c[1].type(schema), name());
            return t;
        }
        if (f == "avg") {
            require(numeric(t), "avg needs a number: " + name());
            return DType::Float64;
        }
        if (f == "sum") {
            require(numeric(t), "sum needs a number: " + name());
            return t;
        }
        if (f == "min" || f == "max") {
            require(t != DType::Bool, "min/max need an orderable type: " + name());
            return t;
        }
        throw std::invalid_argument("unknown window function " + f);
    }
    }
    throw std::logic_error("bad expression");
}

inline DType Expr::func_type(const Schema& schema) const {
    const auto& c = n_->children;
    const std::string& f = n_->name;
    auto arity = [&](size_t lo, size_t hi) {
        require(c.size() >= lo && c.size() <= hi, f + ": wrong number of arguments");
    };
    auto str_arg = [&](size_t i) {
        require(c[i].type(schema) == DType::String, f + " needs a string: " + c[i].name());
    };
    if (f == "isnull" || f == "isnotnull") {
        arity(1, 1);
        c[0].type(schema);
        return DType::Bool;
    }
    if (f == "coalesce") {
        require(!c.empty(), "coalesce needs arguments");
        std::optional<DType> t;
        for (const auto& a : c) {
            const DType at = a.type(schema);
            if (!a.is_null_literal()) t = t ? unify(*t, at, name()) : at;
        }
        return t ? *t : c[0].type(schema);
    }
    if (f == "in") {
        require(c.size() >= 2, "isin needs values");
        const DType t = c[0].type(schema);
        for (size_t i = 1; i < c.size(); ++i) binary_type(Op::Eq, t, c[i].type(schema), name());
        return DType::Bool;
    }
    if (f == "upper" || f == "lower" || f == "trim") {
        arity(1, 1);
        str_arg(0);
        return DType::String;
    }
    if (f == "length") {
        arity(1, 1);
        str_arg(0);
        return DType::Int64;
    }
    if (f == "substring") {
        arity(3, 3);
        str_arg(0);
        require(c[1].type(schema) == DType::Int64 && c[2].type(schema) == DType::Int64,
                "substring needs integer positions");
        return DType::String;
    }
    if (f == "concat") {
        require(!c.empty(), "concat needs arguments");
        for (size_t i = 0; i < c.size(); ++i) str_arg(i);
        return DType::String;
    }
    if (f == "contains" || f == "startswith" || f == "endswith" || f == "like") {
        arity(2, 2);
        str_arg(0);
        str_arg(1);
        if (f == "like") require(c[1].is_literal(), "LIKE needs a literal pattern");
        return DType::Bool;
    }
    if (f == "abs") {
        arity(1, 1);
        const DType t = c[0].type(schema);
        require(numeric(t), "abs needs a number");
        return t;
    }
    if (f == "round") {
        arity(2, 2);
        const DType t = c[0].type(schema);
        require(numeric(t) && c[1].is_literal(), "round needs a number and a literal scale");
        return t;
    }
    if (f == "floor" || f == "ceil") {
        arity(1, 1);
        require(numeric(c[0].type(schema)), f + " needs a number");
        return DType::Int64;
    }
    if (f == "sqrt") {
        arity(1, 1);
        require(numeric(c[0].type(schema)), "sqrt needs a number");
        return DType::Float64;
    }
    throw std::invalid_argument("unknown function " + f);
}

// ANSI cast between types; null stays null.
inline Column cast_column(const Column& in, DType t) {
    const DType from = in.type();
    if (from == t) return in;
    const Column c = in.flat();
    const uint8_t* valid = c.validity();
    const size_t n = c.size();
    auto ok = [&](size_t i) { return !valid || valid[i]; };
    Column out;
    auto fail = [](const std::string& what) {
        throw std::runtime_error("[CAST_INVALID_INPUT] cannot cast '" + what + "'");
    };
    switch (t) {
    case DType::Int64: {
        std::vector<int64_t> o(n);
        if (from == DType::Float64) {
            const auto& v = c.values<double>();
            for (size_t i = 0; i < n; ++i) {
                if (!ok(i)) continue;
                if (!std::isfinite(v[i]) || v[i] >= 9.2233720368547758e18 || v[i] < -9.2233720368547758e18)
                    throw std::runtime_error("[CAST_OVERFLOW] " + detail::format_double(v[i]) +
                                             " does not fit in bigint");
                o[i] = static_cast<int64_t>(v[i]);
            }
        } else if (from == DType::Bool) {
            const auto& v = c.values<uint8_t>();
            for (size_t i = 0; i < n; ++i) o[i] = v[i] ? 1 : 0;
        } else {
            const auto& v = c.values<std::string>();
            for (size_t i = 0; i < n; ++i) {
                if (!ok(i)) continue;
                const char* s = v[i].c_str();
                while (std::isspace(static_cast<unsigned char>(*s))) ++s;
                char* end = nullptr;
                errno = 0;
                const long long x = std::strtoll(s, &end, 10);
                while (end && std::isspace(static_cast<unsigned char>(*end))) ++end;
                if (end == s || (end && *end) || errno == ERANGE) fail(v[i]);
                o[i] = x;
            }
        }
        out = Column(std::move(o));
        break;
    }
    case DType::Float64: {
        std::vector<double> o(n);
        if (from == DType::Int64) {
            const auto& v = c.values<int64_t>();
            for (size_t i = 0; i < n; ++i) o[i] = static_cast<double>(v[i]);
        } else if (from == DType::Bool) {
            const auto& v = c.values<uint8_t>();
            for (size_t i = 0; i < n; ++i) o[i] = v[i] ? 1.0 : 0.0;
        } else {
            const auto& v = c.values<std::string>();
            for (size_t i = 0; i < n; ++i) {
                if (!ok(i)) continue;
                std::string s = v[i];
                while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
                size_t b = 0;
                while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
                s = s.substr(b);
                if (s == "NaN") o[i] = std::nan("");
                else if (s == "Infinity" || s == "+Infinity") o[i] = std::numeric_limits<double>::infinity();
                else if (s == "-Infinity") o[i] = -std::numeric_limits<double>::infinity();
                else {
                    char* end = nullptr;
                    o[i] = std::strtod(s.c_str(), &end);
                    if (s.empty() || *end) fail(v[i]);
                }
            }
        }
        out = Column(std::move(o));
        break;
    }
    case DType::String: {
        std::vector<std::string> o(n);
        c.visit([&](const auto& v) {
            using T = typename std::decay_t<decltype(v)>::value_type;
            for (size_t i = 0; i < n; ++i) {
                if (!ok(i)) continue;
                if constexpr (std::is_same_v<T, double>) o[i] = detail::format_double(v[i]);
                else if constexpr (std::is_same_v<T, uint8_t>) o[i] = v[i] ? "true" : "false";
                else if constexpr (std::is_same_v<T, int64_t>) o[i] = std::to_string(v[i]);
            }
        });
        out = Column(std::move(o));
        break;
    }
    case DType::Bool: {
        BoolVec o(n);
        if (from == DType::String) {
            const auto& v = c.values<std::string>();
            for (size_t i = 0; i < n; ++i) {
                if (!ok(i)) continue;
                std::string s;
                for (char ch : v[i])
                    if (!std::isspace(static_cast<unsigned char>(ch)))
                        s += static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
                if (s == "true" || s == "t" || s == "yes" || s == "y" || s == "1") o[i] = 1;
                else if (s == "false" || s == "f" || s == "no" || s == "n" || s == "0") o[i] = 0;
                else fail(v[i]);
            }
        } else {
            c.visit([&](const auto& v) {
                using T = typename std::decay_t<decltype(v)>::value_type;
                if constexpr (!std::is_same_v<T, std::string>)
                    for (size_t i = 0; i < n; ++i) o[i] = v[i] != T{};
            });
        }
        out = Column(std::move(o));
        break;
    }
    }
    return out.with_validity(c.validity_ptr());
}

inline Column Expr::eval(const Batch& batch) const {
    if (n_->kind == Kind::Window)
        throw std::logic_error("window function evaluated outside a window: " + name());
    Column dict_out;
    if (dictionary_eval(batch, dict_out)) return dict_out;
    Value v = eval_value(batch);
    return broadcast(v, batch.num_rows());
}

// Expressions over exactly one dictionary-encoded column: evaluate over the
// dictionary (plus one null entry when needed) and gather by code.
inline bool Expr::dictionary_eval(const Batch& batch, Column& out) const {
    if (n_->kind == Kind::Column || n_->kind == Kind::Literal || n_->kind == Kind::Alias) {
        if (n_->kind != Kind::Alias) return false;
        return n_->children[0].dictionary_eval(batch, out);
    }
    std::vector<std::string> refs;
    referenced_columns(refs);
    if (refs.empty()) return false;
    for (const auto& r : refs)
        if (r != refs[0]) return false;
    const Column& c = batch.column(refs[0]);
    if (!c.is_dict() || c.dict_size() * 2 > c.size()) return false;
    Column dict = c.dict_values();
    const size_t d = dict.size();
    std::vector<uint32_t> codes = c.codes();
    if (c.has_nulls()) {
        // append one null entry for null rows
        dict = Column::concat({dict, Column::nulls(dict.type(), 1)}, dict.type());
        const uint8_t* v = c.validity();
        for (size_t i = 0; i < codes.size(); ++i)
            if (!v[i]) codes[i] = static_cast<uint32_t>(d);
    }
    const Batch small({{refs[0], dict.type()}}, {dict});
    Value v = eval_value(small);
    Column r = broadcast(v, small.num_rows());
    if (!r.has_nulls() && !r.is_dict()) {
        out = Column::dictionary(r, std::make_shared<const std::vector<uint32_t>>(std::move(codes)));
    } else {
        out = r.take(codes);
    }
    return true;
}

inline Expr::Value Expr::eval_value(const Batch& batch) const {
    const size_t n = batch.num_rows();
    switch (n_->kind) {
    case Kind::Column: {
        Value v;
        v.col = batch.column(n_->name);
        v.type = v.col.type();
        return v;
    }
    case Kind::Literal: {
        Value v;
        v.is_scalar = true;
        v.null = n_->null;
        v.scalar = n_->literal;
        v.type = n_->type;
        return v;
    }
    case Kind::Alias: return n_->children[0].eval_value(batch);
    case Kind::Not: {
        Value a = n_->children[0].eval_value(batch);
        require(a.type == DType::Bool, "NOT needs a boolean");
        Value v;
        v.type = DType::Bool;
        if (a.is_scalar) {
            v = a;
            if (!a.null) v.scalar = !std::get<bool>(a.scalar);
            return v;
        }
        const auto& av = a.col.values<uint8_t>();
        BoolVec out(n);
        for (size_t i = 0; i < n; ++i) out[i] = !av[i];
        v.col = Column(std::move(out)).with_validity(a.col.validity_ptr());
        return v;
    }
    case Kind::Neg: {
        Value a = n_->children[0].eval_value(batch);
        Value v;
        v.type = a.type;
        if (a.is_scalar) {
            v = a;
            if (!a.null) {
                if (a.type == DType::Int64) v.scalar = -std::get<int64_t>(a.scalar);
                else v.scalar = -std::get<double>(a.scalar);
            }
            return v;
        }
        v.col = a.col.visit([&](const auto& av) -> Column {
            using T = typename std::decay_t<decltype(av)>::value_type;
            if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, double>) {
                std::vector<T> out(n);
                for (size_t i = 0; i < n; ++i) out[i] = -av[i];
                return Column(std::move(out));
            } else {
                throw std::invalid_argument("negation needs a number");
            }
        }).with_validity(a.col.validity_ptr());
        return v;
    }
    case Kind::Binary: return eval_binary(batch);
    case Kind::Cast: {
        Value a = n_->children[0].eval_value(batch);
        Value v;
        v.type = n_->type;
        if (a.is_scalar) {
            // cast a one-row column, keep it scalar
            const Column one = cast_column(broadcast(a, 1), n_->type);
            v.is_scalar = true;
            v.null = !one.is_valid(0);
            v.scalar = default_scalar(n_->type);
            if (!v.null) {
                one.visit([&](const auto& vec) {
                    using T = typename std::decay_t<decltype(vec)>::value_type;
                    if constexpr (std::is_same_v<T, uint8_t>) v.scalar = vec[0] != 0;
                    else v.scalar = vec[0];
                });
            }
            return v;
        }
        v.col = cast_column(a.col, n_->type);
        return v;
    }
    case Kind::Case: {
        Value v;
        v.type = type(batch.schema());
        v.col = eval_case(batch);
        return v;
    }
    case Kind::Func: {
        Value v;
        v.type = type(batch.schema());
        v.col = eval_func(batch);
        return v;
    }
    case Kind::Window: throw std::logic_error("window function outside a window");
    }
    throw std::logic_error("bad expression");
}

inline Expr::Value Expr::eval_binary(const Batch& batch) const {
    const Op op = n_->op;
    Value a = n_->children[0].eval_value(batch);
    Value b = n_->children[1].eval_value(batch);
    const DType out_type = binary_type(op, a.type, b.type, name());
    const size_t n = batch.num_rows();
    Value v;
    v.type = out_type;

    if (op == Op::And || op == Op::Or) {
        const bool is_and = op == Op::And;
        // Scalar shortcuts: x AND false = false, x OR true = true.
        for (const Value* s : {&a, &b}) {
            if (s->is_scalar && !s->null && std::get<bool>(s->scalar) != is_and) {
                v.is_scalar = true;
                v.scalar = !is_and;
                return v;
            }
        }
        const Column ca = broadcast(a, n), cb = broadcast(b, n);
        const auto& av = ca.values<uint8_t>();
        const auto& bv = cb.values<uint8_t>();
        const uint8_t* am = ca.validity();
        const uint8_t* bm = cb.validity();
        BoolVec out(n);
        if (!am && !bm) {
            if (is_and)
                for (size_t i = 0; i < n; ++i) out[i] = av[i] && bv[i];
            else
                for (size_t i = 0; i < n; ++i) out[i] = av[i] || bv[i];
            v.col = Column(std::move(out));
            return v;
        }
        BoolVec valid(n);
        for (size_t i = 0; i < n; ++i) {
            const bool ak = !am || am[i], bk = !bm || bm[i];
            if (is_and) {
                if ((ak && !av[i]) || (bk && !bv[i])) {
                    out[i] = 0;
                    valid[i] = 1;
                } else {
                    out[i] = 1;
                    valid[i] = ak && bk;
                }
            } else {
                if ((ak && av[i]) || (bk && bv[i])) {
                    out[i] = 1;
                    valid[i] = 1;
                } else {
                    out[i] = 0;
                    valid[i] = ak && bk;
                }
            }
        }
        v.col = Column(std::move(out), std::move(valid));
        return v;
    }

    if (op == Op::EqNullSafe) {
        // null <=> null is true, null <=> x is false.
        const Column ca = broadcast(a, n), cb = broadcast(b, n);
        const Batch tmp({{"a", ca.type()}, {"b", cb.type()}}, {ca.with_validity(nullptr), cb.with_validity(nullptr)});
        const Column e = make_binary(Op::Eq, Expr::column("a"), Expr::column("b")).eval(tmp);
        const auto& ev = e.values<uint8_t>();
        BoolVec out(n);
        for (size_t i = 0; i < n; ++i) {
            const bool na = !ca.is_valid(i), nb = !cb.is_valid(i);
            out[i] = (na || nb) ? (na && nb) : ev[i];
        }
        v.col = Column(std::move(out));
        return v;
    }

    if ((a.is_scalar && a.null) || (b.is_scalar && b.null)) {
        v.is_scalar = true;
        v.null = true;
        v.scalar = default_scalar(out_type);
        return v;
    }
    const Column::Validity valid = combine_validity(a, b, n);
    const uint8_t* vm = valid ? valid->data() : nullptr;
    const bool is_cmp = out_type == DType::Bool;
    if (is_cmp && a.type == DType::String) {
        v.col = with_string(a, [&](auto ga) { return with_string(b, [&](auto gb) { return compare(op, ga, gb, n); }); });
    } else if (is_cmp && a.type == DType::Bool) {
        v.col = with_bool(a, [&](auto ga) { return with_bool(b, [&](auto gb) { return compare(op, ga, gb, n); }); });
    } else {
        v.col = with_number(a, [&](auto ga) {
            return with_number(b, [&](auto gb) -> Column {
                if (is_cmp) {
                    if (a.type == DType::Int64 && b.type == DType::Int64) return compare(op, ga, gb, n);
                    return compare(op, [&](size_t i) { return double(ga(i)); }, [&](size_t i) { return double(gb(i)); },
                                   n);
                }
                if (out_type == DType::Int64) return arith<int64_t>(op, ga, gb, n, vm);
                return arith<double>(op, ga, gb, n, vm);
            });
        });
    }
    if (valid) v.col = v.col.with_validity(valid);
    return v;
}

inline Column Expr::eval_case(const Batch& batch) const {
    const auto& c = n_->children;
    const size_t n = batch.num_rows();
    const DType t = type(batch.schema());
    const size_t pairs = (c.size() - (n_->has_else ? 1 : 0)) / 2;
    // branch chosen per row (pairs = else / null)
    std::vector<uint32_t> pick(n, static_cast<uint32_t>(pairs));
    std::vector<uint8_t> done(n, 0);
    for (size_t p = 0; p < pairs; ++p) {
        const Column cond = c[2 * p].eval(batch);
        const auto& cv = cond.values<uint8_t>();
        for (size_t i = 0; i < n; ++i)
            if (!done[i] && cond.is_valid(i) && cv[i]) {
                pick[i] = static_cast<uint32_t>(p);
                done[i] = 1;
            }
    }
    std::vector<Column> vals;
    for (size_t p = 0; p < pairs; ++p) vals.push_back(cast_column(c[2 * p + 1].eval(batch), t));
    vals.push_back(n_->has_else ? cast_column(c.back().eval(batch), t) : Column::nulls(t, n));
    BoolVec valid(n);
    Column out = Column::empty(t).visit([&](const auto& proto) -> Column {
        using T = typename std::decay_t<decltype(proto)>::value_type;
        std::vector<T> o(n);
        std::vector<const std::vector<T>*> src;
        for (const auto& v : vals) src.push_back(&v.template values<T>());
        for (size_t i = 0; i < n; ++i) {
            o[i] = (*src[pick[i]])[i];
            valid[i] = vals[pick[i]].is_valid(i);
        }
        return Column(std::move(o));
    });
    return Column(out).with_validity(std::make_shared<const BoolVec>(std::move(valid)));
}

inline Column Expr::eval_func(const Batch& batch) const {
    const auto& c = n_->children;
    const std::string& f = n_->name;
    const size_t n = batch.num_rows();
    const DType t = type(batch.schema());

    if (f == "isnull" || f == "isnotnull") {
        const Column a = c[0].eval(batch);
        BoolVec out(n);
        const bool want_null = f == "isnull";
        for (size_t i = 0; i < n; ++i) out[i] = (!a.is_valid(i)) == want_null;
        return Column(std::move(out));
    }
    if (f == "coalesce") {
        std::vector<Column> args;
        for (const auto& a : c) args.push_back(cast_column(a.eval(batch), t));
        BoolVec valid(n, 0);
        std::vector<uint32_t> which(n, 0);
        for (size_t i = 0; i < n; ++i)
            for (size_t k = 0; k < args.size(); ++k)
                if (args[k].is_valid(i)) {
                    which[i] = static_cast<uint32_t>(k);
                    valid[i] = 1;
                    break;
                }
        Column out = Column::empty(t).visit([&](const auto& proto) -> Column {
            using T = typename std::decay_t<decltype(proto)>::value_type;
            std::vector<T> o(n);
            std::vector<const std::vector<T>*> src;
            for (const auto& a : args) src.push_back(&a.template values<T>());
            for (size_t i = 0; i < n; ++i) o[i] = (*src[which[i]])[i];
            return Column(std::move(o));
        });
        return out.with_validity(std::make_shared<const BoolVec>(std::move(valid)));
    }
    if (f == "in") {
        // e IN (a, b, ...) == (e = a) OR (e = b) OR ... (SQL null semantics)
        Expr chain = make_binary(Op::Eq, c[0], c[1]);
        for (size_t i = 2; i < c.size(); ++i) chain = make_binary(Op::Or, chain, make_binary(Op::Eq, c[0], c[i]));
        return chain.eval(batch);
    }

    // Functions that are null when any argument is null.
    std::vector<Column> args;
    for (const auto& a : c) args.push_back(a.eval(batch));
    Column::Validity valid;
    bool any_null = false;
    for (const auto& a : args) any_null |= a.has_nulls();
    if (any_null) {
        BoolVec m(n, 1);
        for (const auto& a : args)
            if (a.has_nulls())
                for (size_t i = 0; i < n; ++i) m[i] &= a.validity()[i];
        valid = std::make_shared<const BoolVec>(std::move(m));
    }
    auto ok = [&](size_t i) { return !valid || (*valid)[i]; };
    Column out;
    if (f == "upper" || f == "lower" || f == "trim") {
        const auto& v = args[0].values<std::string>();
        std::vector<std::string> o(n);
        for (size_t i = 0; i < n; ++i) {
            if (!ok(i)) continue;
            if (f == "trim") {
                size_t b = v[i].find_first_not_of(' ');
                size_t e = v[i].find_last_not_of(' ');
                o[i] = b == std::string::npos ? "" : v[i].substr(b, e - b + 1);
            } else {
                o[i] = v[i];
                for (auto& ch : o[i])
                    ch = static_cast<char>(f == "upper" ? std::toupper(static_cast<unsigned char>(ch))
                                                        : std::tolower(static_cast<unsigned char>(ch)));
            }
        }
        out = Column(std::move(o));
    } else if (f == "length") {
        const auto& v = args[0].values<std::string>();
        std::vector<int64_t> o(n);
        for (size_t i = 0; i < n; ++i) o[i] = static_cast<int64_t>(detail::utf8_length(v[i]));
        out = Column(std::move(o));
    } else if (f == "substring") {
        const auto& v = args[0].values<std::string>();
        const auto& pv = args[1].values<int64_t>();
        const auto& lv = args[2].values<int64_t>();
        std::vector<std::string> o(n);
        for (size_t i = 0; i < n; ++i) {
            if (!ok(i)) continue;
            const auto len = static_cast<int64_t>(detail::utf8_length(v[i]));
            int64_t start = pv[i] > 0 ? pv[i] - 1 : (pv[i] < 0 ? len + pv[i] : 0);
            int64_t end = lv[i] < 0 ? start : start + lv[i];
            start = std::max<int64_t>(0, start);
            end = std::min(len, end);
            if (end <= start) continue;
            const size_t b = detail::utf8_offset(v[i], static_cast<size_t>(start));
            const size_t e = detail::utf8_offset(v[i], static_cast<size_t>(end));
            o[i] = v[i].substr(b, e - b);
        }
        out = Column(std::move(o));
    } else if (f == "concat") {
        std::vector<std::string> o(n);
        for (const auto& a : args) {
            const auto& v = a.values<std::string>();
            for (size_t i = 0; i < n; ++i) o[i] += v[i];
        }
        out = Column(std::move(o));
    } else if (f == "contains" || f == "startswith" || f == "endswith" || f == "like") {
        const auto& s = args[0].values<std::string>();
        const auto& p = args[1].values<std::string>();
        BoolVec o(n);
        for (size_t i = 0; i < n; ++i) {
            if (!ok(i)) continue;
            if (f == "contains") o[i] = s[i].find(p[i]) != std::string::npos;
            else if (f == "startswith") o[i] = s[i].compare(0, p[i].size(), p[i]) == 0;
            else if (f == "endswith")
                o[i] = s[i].size() >= p[i].size() && s[i].compare(s[i].size() - p[i].size(), p[i].size(), p[i]) == 0;
            else o[i] = detail::like_match(s[i], p[i]);
        }
        out = Column(std::move(o));
    } else if (f == "abs") {
        out = args[0].visit([&](const auto& v) -> Column {
            using T = typename std::decay_t<decltype(v)>::value_type;
            if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, double>) {
                std::vector<T> o(n);
                for (size_t i = 0; i < n; ++i) o[i] = v[i] < 0 ? -v[i] : v[i];
                return Column(std::move(o));
            } else {
                throw std::invalid_argument("abs needs a number");
            }
        });
    } else if (f == "round") {
        const int64_t scale = std::get<int64_t>(c[1].literal_value());
        if (args[0].type() == DType::Int64) {
            out = args[0];
        } else {
            const auto& v = args[0].values<double>();
            const double m = std::pow(10.0, static_cast<double>(scale));
            std::vector<double> o(n);
            for (size_t i = 0; i < n; ++i) o[i] = std::round(v[i] * m) / m;
            out = Column(std::move(o));
        }
    } else if (f == "floor" || f == "ceil") {
        if (args[0].type() == DType::Int64) {
            out = args[0];
        } else {
            const auto& v = args[0].values<double>();
            std::vector<int64_t> o(n);
            for (size_t i = 0; i < n; ++i)
                if (ok(i)) o[i] = static_cast<int64_t>(f == "floor" ? std::floor(v[i]) : std::ceil(v[i]));
            out = Column(std::move(o));
        }
    } else if (f == "sqrt") {
        const Column d = cast_column(args[0], DType::Float64);
        const auto& v = d.values<double>();
        std::vector<double> o(n);
        for (size_t i = 0; i < n; ++i) o[i] = std::sqrt(v[i]);
        out = Column(std::move(o));
    } else {
        throw std::invalid_argument("unknown function " + f);
    }
    return out.flat().with_validity(valid);
}

}  // namespace vega::df
