#pragma once

// Columnar storage for Vega DataFrames.
//
// A Column is an immutable, typed vector shared by reference: copying a
// Column (or a Batch of them) copies pointers, never the data, so batches can
// move through the engine's stages and shuffles cheaply.
//
// Two optional refinements:
//
//   * a validity mask (one byte per row, 1 = valid). Columns without nulls
//     carry no mask at all, so null support costs nothing when unused.
//     Values at null positions are unspecified (zero / empty);
//   * dictionary encoding: the column stores distinct `dictionary` values
//     plus one 32-bit code per row. Joins emit columns from the small side
//     this way (no copying), low-cardinality strings are encoded when a
//     DataFrame is created, and grouping, hashing and expressions work on
//     the codes. values<T>() decodes on first use (once, thread-safely).

#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace vega::df {

enum class DType : uint8_t { Int64 = 0, Float64 = 1, String = 2, Bool = 3 };

inline const char* dtype_name(DType t) {
    switch (t) {
    case DType::Int64: return "bigint";
    case DType::Float64: return "double";
    case DType::String: return "string";
    case DType::Bool: return "boolean";
    }
    return "?";
}

// Booleans are stored one per byte (not std::vector<bool>) so they can be
// read with plain pointers in tight loops.
using BoolVec = std::vector<uint8_t>;

// Row index meaning "produce a null" in Column::take (used by outer joins).
constexpr uint32_t kNullRow = 0xFFFFFFFFu;

template<typename T> struct dtype_of;
template<> struct dtype_of<int64_t> { static constexpr DType value = DType::Int64; };
template<> struct dtype_of<double> { static constexpr DType value = DType::Float64; };
template<> struct dtype_of<std::string> { static constexpr DType value = DType::String; };
template<> struct dtype_of<uint8_t> { static constexpr DType value = DType::Bool; };

class Column {
public:
    using Data = std::variant<std::shared_ptr<const std::vector<int64_t>>,
                              std::shared_ptr<const std::vector<double>>,
                              std::shared_ptr<const std::vector<std::string>>,
                              std::shared_ptr<const BoolVec>>;
    using Validity = std::shared_ptr<const BoolVec>;
    using Codes = std::shared_ptr<const std::vector<uint32_t>>;

    Column() : Column(std::vector<int64_t>{}) {}
    Column(std::vector<int64_t> v) : data_(make(std::move(v))) {}
    Column(std::vector<double> v) : data_(make(std::move(v))) {}
    Column(std::vector<std::string> v) : data_(make(std::move(v))) {}
    Column(BoolVec v) : data_(make(std::move(v))) {}

    // Values plus a validity mask (1 = valid); an all-valid mask is dropped.
    template<typename T>
    Column(std::vector<T> v, BoolVec valid) : data_(make(std::move(v))) {
        if (valid.size() != size()) throw std::invalid_argument("validity length mismatch");
        set_validity(std::move(valid));
    }

    // From optionals: std::nullopt becomes null.
    template<typename T>
    Column(const std::vector<std::optional<T>>& v) {
        using S = std::conditional_t<std::is_same_v<T, bool>, uint8_t, T>;
        std::vector<S> vals(v.size());
        BoolVec valid(v.size(), 1);
        for (size_t i = 0; i < v.size(); ++i) {
            if (v[i]) vals[i] = static_cast<S>(*v[i]);
            else valid[i] = 0;
        }
        data_ = make(std::move(vals));
        set_validity(std::move(valid));
    }

    static Column empty(DType t) {
        switch (t) {
        case DType::Int64: return Column(std::vector<int64_t>{});
        case DType::Float64: return Column(std::vector<double>{});
        case DType::String: return Column(std::vector<std::string>{});
        case DType::Bool: return Column(BoolVec{});
        }
        throw std::logic_error("unknown dtype");
    }

    // n nulls of type t.
    static Column nulls(DType t, size_t n) {
        Column c;
        switch (t) {
        case DType::Int64: c = Column(std::vector<int64_t>(n)); break;
        case DType::Float64: c = Column(std::vector<double>(n)); break;
        case DType::String: c = Column(std::vector<std::string>(n)); break;
        case DType::Bool: c = Column(BoolVec(n)); break;
        }
        c.valid_ = std::make_shared<const BoolVec>(n, 0);
        return c;
    }

    // A dictionary-encoded column: row i is dictionary[codes[i]] (or null
    // where `valid` says so). The dictionary itself must have no nulls.
    static Column dictionary(const Column& dict, Codes codes, Validity valid = nullptr) {
        if (dict.is_dict() || dict.has_nulls())
            throw std::invalid_argument("dictionary must be flat and null-free");
        Column c;
        c.data_ = dict.data_;
        c.codes_ = std::move(codes);
        c.cache_ = std::make_shared<Cache>();
        c.valid_ = std::move(valid);
        if (c.valid_ && c.valid_->size() != c.codes_->size())
            throw std::invalid_argument("validity length mismatch");
        return c;
    }

    DType type() const { return static_cast<DType>(data_.index()); }

    size_t size() const {
        if (codes_) return codes_->size();
        return std::visit([](const auto& p) { return p->size(); }, data_);
    }

    // ----- nulls ----------------------------------------------------------
    bool has_nulls() const { return valid_ != nullptr; }
    const uint8_t* validity() const { return valid_ ? valid_->data() : nullptr; }
    const Validity& validity_ptr() const { return valid_; }
    bool is_valid(size_t i) const { return !valid_ || (*valid_)[i]; }
    size_t null_count() const {
        if (!valid_) return 0;
        size_t n = 0;
        for (uint8_t v : *valid_) n += v == 0;
        return n;
    }
    // The same values with another validity mask (nullptr: no nulls).
    Column with_validity(Validity valid) const {
        Column c = *this;
        c.valid_ = std::move(valid);
        if (c.valid_ && c.valid_->size() != size()) throw std::invalid_argument("validity length mismatch");
        if (c.valid_ && std::find(c.valid_->begin(), c.valid_->end(), 0) == c.valid_->end())
            c.valid_.reset();
        return c;
    }

    // ----- dictionary encoding -----------------------------------------------
    bool is_dict() const { return codes_ != nullptr; }
    const std::vector<uint32_t>& codes() const { return *codes_; }
    const Codes& codes_ptr() const { return codes_; }
    // The dictionary values as a flat column (requires is_dict()).
    Column dict_values() const {
        Column c;
        c.data_ = data_;
        return c;
    }
    size_t dict_size() const {
        return std::visit([](const auto& p) { return p->size(); }, data_);
    }
    // A flat (decoded) copy; cheap for columns that are already flat.
    Column flat() const {
        if (!codes_) return *this;
        Column c;
        c.data_ = decoded();
        c.valid_ = valid_;
        return c;
    }

    // Dictionary-encode a flat column when it has at most `max_ratio` of
    // its rows distinct (strings only; other types are returned as-is).
    Column encode(double max_ratio = 0.5) const {
        if (codes_ || type() != DType::String || size() < 64) return *this;
        const auto& v = std::get<2>(data_);
        std::unordered_map<std::string_view, uint32_t> index;
        std::vector<std::string> dict;
        std::vector<uint32_t> codes(v->size());
        const size_t limit = static_cast<size_t>(static_cast<double>(v->size()) * max_ratio);
        for (size_t i = 0; i < v->size(); ++i) {
            if (valid_ && !(*valid_)[i]) {
                codes[i] = 0;
                continue;
            }
            auto [it, inserted] = index.try_emplace((*v)[i], static_cast<uint32_t>(dict.size()));
            if (inserted) {
                dict.push_back((*v)[i]);
                if (dict.size() > limit) return *this;
            }
            codes[i] = it->second;
        }
        if (dict.empty()) dict.emplace_back();  // codes of all-null rows point here
        return dictionary(Column(std::move(dict)), std::make_shared<const std::vector<uint32_t>>(std::move(codes)),
                          valid_);
    }

    // ----- values -------------------------------------------------------------
    template<typename T>
    const std::vector<T>& values() const {
        using Ptr = std::shared_ptr<const std::vector<T>>;
        const Data& d = codes_ ? decoded() : data_;
        if (const auto* p = std::get_if<Ptr>(&d)) return **p;
        throw std::runtime_error(std::string("column is ") + dtype_name(type()) + ", expected " +
                                 dtype_name(dtype_of<T>::value));
    }

    // Visit the (decoded) typed vector: f(const std::vector<T>&).
    template<typename F>
    decltype(auto) visit(F&& f) const {
        const Data& d = codes_ ? decoded() : data_;
        return std::visit([&](const auto& p) -> decltype(auto) { return f(*p); }, d);
    }

    // Visit the stored vector: the dictionary for dictionary columns.
    template<typename F>
    decltype(auto) visit_stored(F&& f) const {
        return std::visit([&](const auto& p) -> decltype(auto) { return f(*p); }, data_);
    }

    // ----- row selection ----------------------------------------------------

    // Rows at the given indices, in order; kNullRow yields a null.
    Column take(const std::vector<uint32_t>& idx) const {
        const bool null_rows = std::find(idx.begin(), idx.end(), kNullRow) != idx.end();
        Validity valid;
        if (valid_ || null_rows) {
            BoolVec m(idx.size());
            for (size_t i = 0; i < idx.size(); ++i)
                m[i] = idx[i] != kNullRow && (!valid_ || (*valid_)[idx[i]]);
            valid = std::make_shared<const BoolVec>(std::move(m));
        }
        if (codes_) {
            std::vector<uint32_t> c(idx.size());
            const auto& src = *codes_;
            for (size_t i = 0; i < idx.size(); ++i) c[i] = idx[i] == kNullRow ? 0 : src[idx[i]];
            Column out = *this;
            out.codes_ = std::make_shared<const std::vector<uint32_t>>(std::move(c));
            out.cache_ = std::make_shared<Cache>();
            out.valid_ = std::move(valid);
            return out;
        }
        Column out = visit_stored([&](const auto& v) -> Column {
            using T = typename std::decay_t<decltype(v)>::value_type;
            std::vector<T> o(idx.size());
            if (null_rows) {
                for (size_t i = 0; i < idx.size(); ++i)
                    if (idx[i] != kNullRow) o[i] = v[idx[i]];
            } else {
                for (size_t i = 0; i < idx.size(); ++i) o[i] = v[idx[i]];
            }
            return Column(std::move(o));
        });
        out.valid_ = std::move(valid);
        return out;
    }

    Column slice(size_t begin, size_t end) const {
        end = std::min(end, size());
        begin = std::min(begin, end);
        Validity valid;
        if (valid_)
            valid = std::make_shared<const BoolVec>(valid_->begin() + static_cast<std::ptrdiff_t>(begin),
                                                    valid_->begin() + static_cast<std::ptrdiff_t>(end));
        if (codes_) {
            Column out = *this;
            out.codes_ = std::make_shared<const std::vector<uint32_t>>(
                codes_->begin() + static_cast<std::ptrdiff_t>(begin),
                codes_->begin() + static_cast<std::ptrdiff_t>(end));
            out.cache_ = std::make_shared<Cache>();
            out.valid_ = std::move(valid);
            return out;
        }
        Column out = visit_stored([&](const auto& v) -> Column {
            using T = typename std::decay_t<decltype(v)>::value_type;
            return Column(std::vector<T>(v.begin() + static_cast<std::ptrdiff_t>(begin),
                                         v.begin() + static_cast<std::ptrdiff_t>(end)));
        });
        out.valid_ = std::move(valid);
        return out;
    }

    // Concatenate columns of type t. Dictionary columns sharing one
    // dictionary stay encoded; anything else is decoded.
    static Column concat(const std::vector<Column>& parts, DType t) {
        if (parts.size() == 1) return parts.front();
        if (parts.empty()) return empty(t);
        size_t total = 0;
        bool any_nulls = false, same_dict = parts.front().is_dict();
        for (const auto& c : parts) {
            total += c.size();
            any_nulls |= c.has_nulls();
            same_dict = same_dict && c.is_dict() && c.data_ == parts.front().data_;
        }
        Validity valid;
        if (any_nulls) {
            BoolVec m;
            m.reserve(total);
            for (const auto& c : parts) {
                if (c.valid_) m.insert(m.end(), c.valid_->begin(), c.valid_->end());
                else m.insert(m.end(), c.size(), 1);
            }
            valid = std::make_shared<const BoolVec>(std::move(m));
        }
        if (same_dict) {
            std::vector<uint32_t> codes;
            codes.reserve(total);
            for (const auto& c : parts) codes.insert(codes.end(), c.codes_->begin(), c.codes_->end());
            return dictionary(parts.front().dict_values(),
                              std::make_shared<const std::vector<uint32_t>>(std::move(codes)), valid);
        }
        Column out = empty(t).visit_stored([&](const auto& proto) -> Column {
            using T = typename std::decay_t<decltype(proto)>::value_type;
            std::vector<T> o;
            o.reserve(total);
            for (const auto& c : parts) {
                const auto& v = c.values<T>();
                o.insert(o.end(), v.begin(), v.end());
            }
            return Column(std::move(o));
        });
        out.valid_ = std::move(valid);
        return out;
    }

private:
    struct Cache {
        std::once_flag once;
        Data flat;
    };

    template<typename T>
    static Data make(std::vector<T> v) {
        return Data(std::make_shared<const std::vector<T>>(std::move(v)));
    }

    void set_validity(BoolVec valid) {
        if (std::find(valid.begin(), valid.end(), 0) != valid.end())
            valid_ = std::make_shared<const BoolVec>(std::move(valid));
    }

    const Data& decoded() const {
        std::call_once(cache_->once, [this] {
            cache_->flat = std::visit(
                [this](const auto& p) -> Data {
                    using T = typename std::decay_t<decltype(*p)>::value_type;
                    std::vector<T> o(codes_->size());
                    const auto& d = *p;
                    const auto& c = *codes_;
                    for (size_t i = 0; i < o.size(); ++i) o[i] = d[c[i]];
                    return make(std::move(o));
                },
                data_);
        });
        return cache_->flat;
    }

    Data data_;          // values, or the dictionary when codes_ is set
    Validity valid_;     // nullptr: no nulls
    Codes codes_;        // dictionary codes (nullptr: flat)
    std::shared_ptr<Cache> cache_;  // decoded values of a dictionary column
};

}  // namespace vega::df
