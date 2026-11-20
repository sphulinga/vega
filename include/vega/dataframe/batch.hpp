#pragma once

// Batch: a set of equal-length named columns -- one partition of a DataFrame.

#include <algorithm>
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

    // `rows` only matters for a batch with no columns (e.g. after pruning
    // every column away before a count); otherwise it comes from the columns.
    Batch(Schema schema, std::vector<Column> columns, size_t rows = 0)
        : schema_(std::move(schema)), columns_(std::move(columns)) {
        if (schema_.size() != columns_.size())
            throw std::invalid_argument("Batch: schema/column count mismatch");
        rows_ = columns_.empty() ? rows : columns_.front().size();
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

    // The named columns only, in the given order.
    Batch project(const std::vector<std::string>& names) const {
        Schema s;
        std::vector<Column> cols;
        for (const auto& n : names) {
            cols.push_back(column(n));
            s.push_back({n, cols.back().type()});
        }
        return Batch(std::move(s), std::move(cols), rows_);
    }
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
        return Batch(schema_, std::move(cols), idx.size());
    }

    Batch slice(size_t begin, size_t end) const {
        std::vector<Column> cols;
        cols.reserve(columns_.size());
        for (const auto& c : columns_) cols.push_back(c.slice(begin, end));
        end = std::min(end, rows_);
        return Batch(schema_, std::move(cols), end > begin ? end - begin : 0);
    }

    // Concatenate batches with the same schema (empty input -> empty batch).
    static Batch concat(const std::vector<Batch>& parts, const Schema& schema) {
        std::vector<const Batch*> nonempty;
        size_t total = 0;
        for (const auto& b : parts)
            if (b.num_rows() > 0) {
                nonempty.push_back(&b);
                total += b.num_rows();
            }
        if (nonempty.empty()) return empty(schema);
        if (schema.empty()) return Batch(schema, {}, total);
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

// splitmix64 finalizer: spreads identity-hashed integers over all 64 bits.
inline uint64_t finalize_hash(uint64_t x) {
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

template<typename T>
inline uint64_t scalar_hash(const T& v) {
    if constexpr (std::is_same_v<T, double>) {
        return std::hash<double>{}(v == 0.0 ? 0.0 : v);  // -0.0 == 0.0
    } else {
        return std::hash<T>{}(v);
    }
}

inline uint64_t value_hash(const Column& c, size_t row) {
    return c.visit([&](const auto& v) -> uint64_t { return scalar_hash(v[row]); });
}

constexpr uint64_t kHashSeed = 0x84222325CBF29CE4ULL;

inline uint64_t row_hash(const std::vector<Column>& keys, size_t row) {
    uint64_t h = kHashSeed;
    for (const auto& c : keys) h = mix_hash(h, value_hash(c, row));
    return finalize_hash(h);
}

// row_hash of every row, computed column by column (one type dispatch per
// column instead of per value).
inline std::vector<uint64_t> hash_rows(const std::vector<Column>& keys, size_t rows) {
    std::vector<uint64_t> h(rows, kHashSeed);
    for (const auto& c : keys) {
        c.visit([&](const auto& v) {
            for (size_t r = 0; r < rows; ++r) h[r] = mix_hash(h[r], scalar_hash(v[r]));
        });
    }
    for (auto& x : h) x = finalize_hash(x);
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

// Row equality over key columns with the type dispatch done up front.
class KeyEq {
public:
    KeyEq(const std::vector<Column>& a, const std::vector<Column>& b) {
        for (size_t k = 0; k < a.size(); ++k) {
            if (a[k].type() != b[k].type()) throw std::invalid_argument("key types differ");
            cols_.push_back({a[k].type(), raw(a[k]), raw(b[k])});
        }
    }
    bool operator()(size_t ra, size_t rb) const {
        for (const auto& c : cols_) {
            switch (c.type) {
            case DType::Int64:
                if (static_cast<const int64_t*>(c.a)[ra] != static_cast<const int64_t*>(c.b)[rb]) return false;
                break;
            case DType::Float64:
                if (static_cast<const double*>(c.a)[ra] != static_cast<const double*>(c.b)[rb]) return false;
                break;
            case DType::String:
                if (static_cast<const std::string*>(c.a)[ra] != static_cast<const std::string*>(c.b)[rb]) return false;
                break;
            case DType::Bool:
                if (static_cast<const uint8_t*>(c.a)[ra] != static_cast<const uint8_t*>(c.b)[rb]) return false;
                break;
            }
        }
        return true;
    }

private:
    struct Col {
        DType type;
        const void* a;
        const void* b;
    };
    static const void* raw(const Column& c) {
        return c.visit([](const auto& v) -> const void* { return v.data(); });
    }
    std::vector<Col> cols_;
};

// row_hash with the type dispatch done up front: hashes one row at a time
// without a per-row visit or a temporary hash array.
class KeyHasher {
public:
    explicit KeyHasher(const std::vector<Column>& keys) {
        for (const auto& c : keys)
            cols_.push_back({c.type(), c.visit([](const auto& v) -> const void* { return v.data(); })});
    }
    uint64_t operator()(size_t r) const {
        uint64_t h = kHashSeed;
        for (const auto& c : cols_) {
            switch (c.type) {
            case DType::Int64: h = mix_hash(h, scalar_hash(static_cast<const int64_t*>(c.p)[r])); break;
            case DType::Float64: h = mix_hash(h, scalar_hash(static_cast<const double*>(c.p)[r])); break;
            case DType::String: h = mix_hash(h, scalar_hash(static_cast<const std::string*>(c.p)[r])); break;
            case DType::Bool: h = mix_hash(h, scalar_hash(static_cast<const uint8_t*>(c.p)[r])); break;
            }
        }
        return finalize_hash(h);
    }

private:
    struct Col {
        DType type;
        const void* p;
    };
    std::vector<Col> cols_;
};

// Open-addressing map from int64 key to dense id, with keys stored inline
// (the single-integer-key fast path for grouping).
class Int64GroupTable {
public:
    explicit Int64GroupTable(size_t expected = 16) { reset(expected); }

    std::pair<uint32_t, bool> find_or_insert(int64_t key) {
        if ((size_ + 1) * 2 > slots_.size()) grow();
        for (size_t i = finalize_hash(static_cast<uint64_t>(key)) & mask_;; i = (i + 1) & mask_) {
            Slot& s = slots_[i];
            if (s.id == kEmpty) {
                s = {key, static_cast<uint32_t>(size_++)};
                return {s.id, true};
            }
            if (s.key == key) return {s.id, false};
        }
    }

private:
    static constexpr uint32_t kEmpty = 0xFFFFFFFFu;
    struct Slot {
        int64_t key;
        uint32_t id;
    };
    void reset(size_t expected) {
        size_t cap = 16;
        while (cap < expected * 2) cap <<= 1;
        slots_.assign(cap, Slot{0, kEmpty});
        mask_ = cap - 1;
    }
    void grow() {
        std::vector<Slot> old = std::move(slots_);
        reset(old.size());
        for (const Slot& s : old) {
            if (s.id == kEmpty) continue;
            size_t i = finalize_hash(static_cast<uint64_t>(s.key)) & mask_;
            while (slots_[i].id != kEmpty) i = (i + 1) & mask_;
            slots_[i] = s;
        }
    }
    std::vector<Slot> slots_;
    size_t mask_ = 0;
    size_t size_ = 0;
};

// Open-addressing hash table from row hash to group id (ids are dense,
// assigned in insertion order). The caller resolves hash collisions with an
// equality callback against each candidate group's representative row.
class GroupTable {
public:
    explicit GroupTable(size_t expected = 16) {
        size_t cap = 16;
        while (cap < expected * 2) cap <<= 1;
        slots_.assign(cap, kEmpty);
        mask_ = cap - 1;
    }

    size_t size() const { return hashes_.size(); }

    // Returns {group id, inserted}. `eq(gid)` says whether the row being
    // inserted equals group `gid`'s key.
    template<typename Eq>
    std::pair<uint32_t, bool> find_or_insert(uint64_t h, Eq&& eq) {
        if ((hashes_.size() + 1) * 2 > slots_.size()) grow();
        for (size_t i = h & mask_;; i = (i + 1) & mask_) {
            const uint32_t g = slots_[i];
            if (g == kEmpty) {
                const auto id = static_cast<uint32_t>(hashes_.size());
                slots_[i] = id;
                hashes_.push_back(h);
                return {id, true};
            }
            if (hashes_[g] == h && eq(g)) return {g, false};
        }
    }

    // Lookup only; returns kEmpty when absent.
    template<typename Eq>
    uint32_t find(uint64_t h, Eq&& eq) const {
        for (size_t i = h & mask_;; i = (i + 1) & mask_) {
            const uint32_t g = slots_[i];
            if (g == kEmpty) return kEmpty;
            if (hashes_[g] == h && eq(g)) return g;
        }
    }

    static constexpr uint32_t kEmpty = 0xFFFFFFFFu;

private:
    void grow() {
        std::vector<uint32_t> next(slots_.size() * 2, kEmpty);
        const size_t mask = next.size() - 1;
        for (uint32_t g = 0; g < hashes_.size(); ++g) {
            size_t i = hashes_[g] & mask;
            while (next[i] != kEmpty) i = (i + 1) & mask;
            next[i] = g;
        }
        slots_ = std::move(next);
        mask_ = mask;
    }

    std::vector<uint32_t> slots_;
    std::vector<uint64_t> hashes_;
    size_t mask_ = 0;
};

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
        write_pod<uint64_t>(buf, b.num_rows());
        write_pod<uint32_t>(buf, static_cast<uint32_t>(b.num_columns()));
        for (size_t i = 0; i < b.num_columns(); ++i) {
            Serializer<std::string>::serialize(b.schema()[i].name, buf);
            Serializer<df::Column>::serialize(b.column(i), buf);
        }
    }
    static df::Batch deserialize(const ByteBuffer& buf, size_t& off) {
        const auto rows = static_cast<size_t>(read_pod<uint64_t>(buf, off));
        const auto ncols = read_pod<uint32_t>(buf, off);
        df::Schema schema;
        std::vector<df::Column> cols;
        for (uint32_t i = 0; i < ncols; ++i) {
            auto name = Serializer<std::string>::deserialize(buf, off);
            auto col = Serializer<df::Column>::deserialize(buf, off);
            schema.push_back({std::move(name), col.type()});
            cols.push_back(std::move(col));
        }
        return df::Batch(std::move(schema), std::move(cols), rows);
    }
};

}  // namespace vega
