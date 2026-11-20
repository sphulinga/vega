#pragma once

// Column expressions for Vega DataFrames:
//
//   col("price") * col("qty") > 100.0 && col("category") == "c7"
//
// Expressions are immutable trees, evaluated a whole column at a time
// (vectorized): one tight loop per operator, with literals applied as scalars
// rather than expanded into columns. Typing follows Spark SQL: integer
// arithmetic stays bigint, `/` always yields double, mixed int/double
// operands promote to double, and division or modulo by zero raises an error
// (Spark 4's default ANSI mode).

#include <cmath>
#include <cstdint>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "vega/dataframe/batch.hpp"

namespace vega::df {

class Expr {
public:
    enum class Op { Add, Sub, Mul, Div, Mod, Eq, Ne, Lt, Le, Gt, Ge, And, Or };
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

    // Rename the expression's output column.
    Expr alias(std::string name) const {
        auto n = std::make_shared<Node>();
        n->kind = Kind::Alias;
        n->name = std::move(name);
        n->children = {*this};
        return Expr(std::move(n));
    }

    // Output column name, following Spark's conventions, e.g. "(qty * price)".
    std::string name() const {
        switch (n_->kind) {
        case Kind::Column:
        case Kind::Alias: return n_->name;
        case Kind::Literal: return literal_text(n_->literal);
        case Kind::Not: return "(NOT " + n_->children[0].name() + ")";
        case Kind::Binary:
            return "(" + n_->children[0].name() + " " + op_symbol(n_->op) + " " +
                   n_->children[1].name() + ")";
        }
        return "?";
    }

    bool is_column() const { return n_->kind == Kind::Column; }

    // Result type against a schema; also validates the expression.
    DType type(const Schema& schema) const {
        switch (n_->kind) {
        case Kind::Column: {
            const int i = schema_index(schema, n_->name);
            if (i < 0)
                throw std::invalid_argument("cannot resolve column '" + n_->name + "' in " +
                                            schema_string(schema));
            return schema[static_cast<size_t>(i)].type;
        }
        case Kind::Literal: return scalar_type(n_->literal);
        case Kind::Alias: return n_->children[0].type(schema);
        case Kind::Not:
            require(n_->children[0].type(schema) == DType::Bool, "NOT needs a boolean");
            return DType::Bool;
        case Kind::Binary: {
            const DType a = n_->children[0].type(schema);
            const DType b = n_->children[1].type(schema);
            return binary_type(n_->op, a, b, name());
        }
        }
        throw std::logic_error("bad expression");
    }

    // Evaluate against a batch, producing a column with one value per row.
    Column eval(const Batch& batch) const {
        Value v = eval_value(batch);
        return v.is_scalar ? broadcast(v.scalar, batch.num_rows()) : std::move(v.col);
    }

    // ----- builders ---------------------------------------------------------
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

private:
    enum class Kind { Column, Literal, Binary, Not, Alias };
    struct Node {
        Kind kind = Kind::Literal;
        Op op = Op::Add;
        std::string name;
        Scalar literal;
        std::vector<Expr> children;
    };

    // A column, or a scalar standing for a column of identical values.
    struct Value {
        bool is_scalar = false;
        Column col;
        Scalar scalar;
        DType type = DType::Int64;
    };

    explicit Expr(Scalar s) {
        auto n = std::make_shared<Node>();
        n->kind = Kind::Literal;
        n->literal = std::move(s);
        n_ = std::move(n);
    }
    explicit Expr(std::shared_ptr<const Node> n) : n_(std::move(n)) {}

    static void require(bool ok, const std::string& msg) {
        if (!ok) throw std::invalid_argument(msg);
    }
    static bool numeric(DType t) { return t == DType::Int64 || t == DType::Float64; }

    static DType scalar_type(const Scalar& s) {
        switch (s.index()) {
        case 0: return DType::Int64;
        case 1: return DType::Float64;
        case 2: return DType::String;
        default: return DType::Bool;
        }
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
        }
        return "?";
    }

    static std::string literal_text(const Scalar& s) {
        switch (s.index()) {
        case 0: return std::to_string(std::get<int64_t>(s));
        case 1: {
            std::ostringstream os;
            os << std::get<double>(s);
            return os.str();
        }
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
            require((numeric(a) && numeric(b)) || a == b, "cannot compare " +
                    std::string(dtype_name(a)) + " with " + dtype_name(b) + ": " + what);
            return DType::Bool;
        default:  // ordering comparisons
            require((numeric(a) && numeric(b)) || (a == b && a != DType::Bool),
                    "cannot order " + std::string(dtype_name(a)) + " and " + dtype_name(b) +
                        ": " + what);
            return DType::Bool;
        }
    }

    static Column broadcast(const Scalar& s, size_t n) {
        switch (s.index()) {
        case 0: return Column(std::vector<int64_t>(n, std::get<int64_t>(s)));
        case 1: return Column(std::vector<double>(n, std::get<double>(s)));
        case 2: return Column(std::vector<std::string>(n, std::get<std::string>(s)));
        default: return Column(BoolVec(n, std::get<bool>(s) ? 1 : 0));
        }
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
    static Column arith(Op op, GA ga, GB gb, size_t n) {
        std::vector<Out> out(n);
        switch (op) {
        case Op::Add: for (size_t i = 0; i < n; ++i) out[i] = Out(ga(i)) + Out(gb(i)); break;
        case Op::Sub: for (size_t i = 0; i < n; ++i) out[i] = Out(ga(i)) - Out(gb(i)); break;
        case Op::Mul: for (size_t i = 0; i < n; ++i) out[i] = Out(ga(i)) * Out(gb(i)); break;
        case Op::Div:
            for (size_t i = 0; i < n; ++i) {
                if (gb(i) == 0) throw std::runtime_error("[DIVIDE_BY_ZERO] Division by zero");
                out[i] = Out(ga(i)) / Out(gb(i));
            }
            break;
        case Op::Mod:
            for (size_t i = 0; i < n; ++i) {
                if (gb(i) == 0) throw std::runtime_error("[DIVIDE_BY_ZERO] Division by zero");
                if constexpr (std::is_same_v<Out, double>)
                    out[i] = std::fmod(Out(ga(i)), Out(gb(i)));
                else
                    out[i] = Out(ga(i)) % Out(gb(i));
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
        case Op::Eq: for (size_t i = 0; i < n; ++i) out[i] = ga(i) == gb(i); break;
        case Op::Ne: for (size_t i = 0; i < n; ++i) out[i] = ga(i) != gb(i); break;
        case Op::Lt: for (size_t i = 0; i < n; ++i) out[i] = ga(i) < gb(i); break;
        case Op::Le: for (size_t i = 0; i < n; ++i) out[i] = ga(i) <= gb(i); break;
        case Op::Gt: for (size_t i = 0; i < n; ++i) out[i] = ga(i) > gb(i); break;
        case Op::Ge: for (size_t i = 0; i < n; ++i) out[i] = ga(i) >= gb(i); break;
        default: throw std::logic_error("not a comparison");
        }
        return Column(std::move(out));
    }

    Value eval_value(const Batch& batch) const {
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
            v.scalar = n_->literal;
            v.type = scalar_type(n_->literal);
            return v;
        }
        case Kind::Alias: return n_->children[0].eval_value(batch);
        case Kind::Not: {
            Value a = n_->children[0].eval_value(batch);
            require(a.type == DType::Bool, "NOT needs a boolean");
            const size_t n = batch.num_rows();
            Value v;
            v.type = DType::Bool;
            v.col = with_bool(a, [&](auto ga) {
                BoolVec out(n);
                for (size_t i = 0; i < n; ++i) out[i] = !ga(i);
                return Column(std::move(out));
            });
            return v;
        }
        case Kind::Binary: break;
        }

        const Op op = n_->op;
        Value a = n_->children[0].eval_value(batch);
        Value b = n_->children[1].eval_value(batch);
        const DType out_type = binary_type(op, a.type, b.type, name());
        const size_t n = batch.num_rows();
        Value v;
        v.type = out_type;

        if (op == Op::And || op == Op::Or) {
            v.col = with_bool(a, [&](auto ga) {
                return with_bool(b, [&](auto gb) {
                    BoolVec out(n);
                    if (op == Op::And)
                        for (size_t i = 0; i < n; ++i) out[i] = ga(i) && gb(i);
                    else
                        for (size_t i = 0; i < n; ++i) out[i] = ga(i) || gb(i);
                    return Column(std::move(out));
                });
            });
            return v;
        }
        const bool is_cmp = out_type == DType::Bool;
        if (is_cmp && a.type == DType::String) {
            v.col = with_string(a, [&](auto ga) {
                return with_string(b, [&](auto gb) { return compare(op, ga, gb, n); });
            });
            return v;
        }
        if (is_cmp && a.type == DType::Bool) {
            v.col = with_bool(a, [&](auto ga) {
                return with_bool(b, [&](auto gb) { return compare(op, ga, gb, n); });
            });
            return v;
        }
        v.col = with_number(a, [&](auto ga) {
            return with_number(b, [&](auto gb) -> Column {
                if (is_cmp) {
                    if (a.type == DType::Int64 && b.type == DType::Int64)
                        return compare(op, ga, gb, n);
                    return compare(
                        op, [&](size_t i) { return double(ga(i)); },
                        [&](size_t i) { return double(gb(i)); }, n);
                }
                if (out_type == DType::Int64) return arith<int64_t>(op, ga, gb, n);
                return arith<double>(op, ga, gb, n);
            });
        });
        return v;
    }

    std::shared_ptr<const Node> n_;
};

inline Expr col(std::string name) { return Expr::column(std::move(name)); }
inline Expr lit(Expr e) { return e; }

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

}  // namespace vega::df
