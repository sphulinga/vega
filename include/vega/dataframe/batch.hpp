#pragma once

// Batch: a set of equal-length named columns -- one partition of a DataFrame.

#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vega/dataframe/column.hpp"
#include "vega/serialization/serializer.hpp"

namespace vega::df {

struct Field {
    std::string name;
    DType type;
    bool operator==(const Field& o) const { return name == o.name && type == o.type; }
};

using Schema = std::vector<Field>;

inline int schema_index(const Schema& s, const std::string& name) {
    for (size_t i = 0; i < s.size(); ++i)
        if (s[i].name == name) return static_cast<int>(i);
    return -1;
}

inline std::string schema_string(const Schema& s) {
    std::string out = "[";
    for (size_t i = 0; i < s.size(); ++i) {
        if (i) out += ", ";
        out += s[i].name + ": " + dtype_name(s[i].type);
    }
    return out + "]";
}

class Batch {
public:
    Batch() = default;

    Batch(Schema schema, std::vector<Column> columns)
        : schema_(std::move(schema)), columns_(std::move(columns)) {
        if (schema_.size() != columns_.size())
            throw std::invalid_argument("Batch: schema/column count mismatch");
        rows_ = columns_.empty() ? 0 : columns_.front().size();
        for (size_t i = 0; i < columns_.size(); ++i) {
            if (columns_[i].size() != rows_)
                throw std::invalid_argument("Batch: columns have different lengths");
            if (columns_[i].type() != schema_[i].type)
                throw std::invalid_argument("Batch: column '" + schema_[i].name +
                                            "' does not match its schema type");
        }
    }

    static Batch empty(const Schema& schema) {
        std::vector<Column> cols;
        cols.reserve(schema.size());
        for (const auto& f : schema) cols.push_back(Column::empty(f.type));
        return Batch(schema, std::move(cols));
    }

    const Schema& schema() const { return schema_; }
    size_t num_rows() const { return rows_; }
    size_t num_columns() const { return columns_.size(); }
    const Column& column(size_t i) const { return columns_.at(i); }
    const std::vector<Column>& columns() const { return columns_; }

    const Column& column(const std::string& name) const {
        const int i = schema_index(schema_, name);
        if (i < 0)
            throw std::invalid_argument("no column '" + name + "' in " + schema_string(schema_));
        return columns_[static_cast<size_t>(i)];
    }

    Batch take(const std::vector<uint32_t>& idx) const {
        std::vector<Column> cols;
        cols.reserve(columns_.size());
        for (const auto& c : columns_) cols.push_back(c.take(idx));
        return Batch(schema_, std::move(cols));
    }

    Batch slice(size_t begin, size_t end) const {
        std::vector<Column> cols;
        cols.reserve(columns_.size());
        for (const auto& c : columns_) cols.push_back(c.slice(begin, end));
        return Batch(schema_, std::move(cols));
    }

    // Concatenate batches with the same schema (empty input -> empty batch).
    static Batch concat(const std::vector<Batch>& parts, const Schema& schema) {
        std::vector<const Batch*> nonempty;
        for (const auto& b : parts)
            if (b.num_rows() > 0) nonempty.push_back(&b);
        if (nonempty.empty()) return empty(schema);
        if (nonempty.size() == 1) return *nonempty.front();
        std::vector<Column> cols;
        cols.reserve(schema.size());
        for (size_t c = 0; c < schema.size(); ++c) {
            std::vector<Column> pieces;
            pieces.reserve(nonempty.size());
            for (const Batch* b : nonempty) pieces.push_back(b->column(c));
            cols.push_back(Column::concat(pieces, schema[c].type));
        }
        return Batch(schema, std::move(cols));
    }

private:
    Schema schema_;
    std::vector<Column> columns_;
    size_t rows_ = 0;
};

// ---------------------------------------------------------------------------
// Row hashing and equality over a set of key columns. Used by grouping,
// joins and distinct; keys may span several columns of any type.
// ---------------------------------------------------------------------------

inline uint64_t mix_hash(uint64_t h, uint64_t v) {
    // Boost-style combine with a 64-bit odd constant.
    return h ^ (v + 0x9E3779B97F4A7C15ULL + (h << 6) + (h >> 2));
}

inline uint64_t value_hash(const Column& c, size_t row) {
    return c.visit([&](const auto& v) -> uint64_t {
        using T = typename std::decay_t<decltype(v)>::value_type;
        if constexpr (std::is_same_v<T, double>) {
            const double d = v[row] == 0.0 ? 0.0 : v[row];  // -0.0 == 0.0
            return std::hash<double>{}(d);
        } else {
            return std::hash<T>{}(v[row]);
        }
    });
}

inline uint64_t row_hash(const std::vector<Column>& keys, size_t row) {
    uint64_t h = 0x84222325CBF29CE4ULL;
    for (const auto& c : keys) h = mix_hash(h, value_hash(c, row));
    return h;
}

// keys_a[row_a] == keys_b[row_b], column by column (types must match).
inline bool rows_equal(const std::vector<Column>& a, size_t ra,
                       const std::vector<Column>& b, size_t rb) {
    for (size_t k = 0; k < a.size(); ++k) {
        const bool eq = a[k].visit([&](const auto& va) {
            using T = typename std::decay_t<decltype(va)>::value_type;
            return va[ra] == b[k].values<T>()[rb];
        });
        if (!eq) return false;
    }
    return true;
}

}  // namespace vega::df

// ---------------------------------------------------------------------------
// Wire format, so batches can flow through the engine's shuffle (and, later,
// between cluster peers).
// ---------------------------------------------------------------------------

namespace vega {

template<>
struct Serializer<df::Column> {
    static void serialize(const df::Column& c, ByteBuffer& buf) {
        write_pod<uint8_t>(buf, static_cast<uint8_t>(c.type()));
        c.visit([&](const auto& v) {
            using T = typename std::decay_t<decltype(v)>::value_type;
            write_pod<uint64_t>(buf, v.size());
            if constexpr (std::is_same_v<T, std::string>) {
                for (const auto& s : v) {
                    write_pod<uint32_t>(buf, static_cast<uint32_t>(s.size()));
                    write_raw(buf, s.data(), s.size());
                }
            } else {
                write_raw(buf, v.data(), v.size() * sizeof(T));
            }
        });
    }
    static df::Column deserialize(const ByteBuffer& buf, size_t& off) {
        const auto type = static_cast<df::DType>(read_pod<uint8_t>(buf, off));
        const auto n = static_cast<size_t>(read_pod<uint64_t>(buf, off));
        return df::Column::empty(type).visit([&](const auto& proto) -> df::Column {
            using T = typename std::decay_t<decltype(proto)>::value_type;
            std::vector<T> v(n);
            if constexpr (std::is_same_v<T, std::string>) {
                for (auto& s : v) {
                    const auto len = read_pod<uint32_t>(buf, off);
                    s.resize(len);
                    read_raw(buf, off, s.data(), len);
                }
            } else {
                read_raw(buf, off, v.data(), n * sizeof(T));
            }
            return df::Column(std::move(v));
        });
    }
};

template<>
struct Serializer<df::Batch> {
    static void serialize(const df::Batch& b, ByteBuffer& buf) {
        write_pod<uint32_t>(buf, static_cast<uint32_t>(b.num_columns()));
        for (size_t i = 0; i < b.num_columns(); ++i) {
            Serializer<std::string>::serialize(b.schema()[i].name, buf);
            Serializer<df::Column>::serialize(b.column(i), buf);
        }
    }
    static df::Batch deserialize(const ByteBuffer& buf, size_t& off) {
        const auto ncols = read_pod<uint32_t>(buf, off);
        df::Schema schema;
        std::vector<df::Column> cols;
        for (uint32_t i = 0; i < ncols; ++i) {
            auto name = Serializer<std::string>::deserialize(buf, off);
            auto col = Serializer<df::Column>::deserialize(buf, off);
            schema.push_back({std::move(name), col.type()});
            cols.push_back(std::move(col));
        }
        return df::Batch(std::move(schema), std::move(cols));
    }
};

}  // namespace vega
