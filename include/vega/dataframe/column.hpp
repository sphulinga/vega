#pragma once

// Columnar storage for Vega DataFrames.
//
// A Column is an immutable, typed vector shared by reference: copying a
// Column (or a Batch of them) copies a pointer, never the data, so batches can
// move through the engine's stages and shuffles cheaply.

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
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

    Column() : Column(std::vector<int64_t>{}) {}
    Column(std::vector<int64_t> v) : data_(make(std::move(v))) {}
    Column(std::vector<double> v) : data_(make(std::move(v))) {}
    Column(std::vector<std::string> v) : data_(make(std::move(v))) {}
    Column(BoolVec v) : data_(make(std::move(v))) {}

    static Column empty(DType t) {
        switch (t) {
        case DType::Int64: return Column(std::vector<int64_t>{});
        case DType::Float64: return Column(std::vector<double>{});
        case DType::String: return Column(std::vector<std::string>{});
        case DType::Bool: return Column(BoolVec{});
        }
        throw std::logic_error("unknown dtype");
    }

    DType type() const { return static_cast<DType>(data_.index()); }

    size_t size() const {
        return std::visit([](const auto& p) { return p->size(); }, data_);
    }

    template<typename T>
    const std::vector<T>& values() const {
        using Ptr = std::shared_ptr<const std::vector<T>>;
        if (const auto* p = std::get_if<Ptr>(&data_)) return **p;
        throw std::runtime_error(std::string("column is ") + dtype_name(type()) +
                                 ", expected " + dtype_name(dtype_of<T>::value));
    }

    // Visit the underlying typed vector: f(const std::vector<T>&).
    template<typename F>
    decltype(auto) visit(F&& f) const {
        return std::visit([&](const auto& p) -> decltype(auto) { return f(*p); }, data_);
    }

    // Rows at the given indices, in order.
    Column take(const std::vector<uint32_t>& idx) const {
        return visit([&](const auto& v) -> Column {
            using T = typename std::decay_t<decltype(v)>::value_type;
            std::vector<T> out;
            out.reserve(idx.size());
            for (uint32_t i : idx) out.push_back(v[i]);
            return Column(std::move(out));
        });
    }

    Column slice(size_t begin, size_t end) const {
        return visit([&](const auto& v) -> Column {
            using T = typename std::decay_t<decltype(v)>::value_type;
            end = std::min(end, v.size());
            begin = std::min(begin, end);
            return Column(std::vector<T>(v.begin() + static_cast<std::ptrdiff_t>(begin),
                                         v.begin() + static_cast<std::ptrdiff_t>(end)));
        });
    }

    static Column concat(const std::vector<Column>& parts, DType t) {
        if (parts.size() == 1) return parts.front();
        return empty(t).visit([&](const auto& proto) -> Column {
            using T = typename std::decay_t<decltype(proto)>::value_type;
            size_t total = 0;
            for (const auto& c : parts) total += c.size();
            std::vector<T> out;
            out.reserve(total);
            for (const auto& c : parts) {
                const auto& v = c.values<T>();
                out.insert(out.end(), v.begin(), v.end());
            }
            return Column(std::move(out));
        });
    }

    const Data& data() const { return data_; }

private:
    template<typename T>
    static Data make(std::vector<T> v) {
        return Data(std::make_shared<const std::vector<T>>(std::move(v)));
    }

    Data data_;
};

}  // namespace vega::df
