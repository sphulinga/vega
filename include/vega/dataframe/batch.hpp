#pragma once

// Batch: a set of equal-length named columns -- one partition of a DataFrame.
// Also the row hashing / equality / hash-table machinery shared by grouping,
// joins and distinct, and the wire format for shuffles.

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
    bool operator!=(const Field& o) const { return !(*this == o); }
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
    const std::vector<Column>& columns() const { return columns_; }

    const Column& column(const std::string& name) const {
        const int i = schema_index(schema_, name);
        if (i < 0)
            throw std::invalid_argument("no column '" + name + "' in " + schema_string(schema_));
        return columns_[static_cast<size_t>(i)];
    }

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

    // Same columns under new names.
    Batch renamed(const std::vector<std::string>& names) const {
        Schema s = schema_;
        for (size_t i = 0; i < s.size(); ++i) s[i].name = names.at(i);
        return Batch(std::move(s), columns_, rows_);
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

    // Decode dictionary columns (e.g. before handing results to a user).
    Batch flat() const {
        std::vector<Column> cols;
        for (const auto& c : columns_) cols.push_back(c.flat());
        return Batch(schema_, std::move(cols), rows_);
    }

    // Concatenate batches with the same schema (empty input -> empty batch).
    static Batch concat(const std::vector<Batch>& parts, const Schema& schema) {
        std::vector<const Batch*> nonempty;
        size_t total = 0;
        for (const auto& b : parts) {
            if (b.num_rows() > 0) {
                nonempty.push_back(&b);
                total += b.num_rows();
            }
        }
        if (nonempty.empty()) return empty(schema);
        if (schema.empty()) return Batch(schema, {}, total);
        if (nonempty.size() == 1) return Batch(schema, nonempty.front()->columns_, total);
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
// Row hashing and equality over a set of key columns. Nulls hash to one
// value and compare equal to each other (grouping semantics; joins drop
// null keys before probing). Dictionary columns hash and compare by value,
// so encoded and plain columns with equal contents agree.
// ---------------------------------------------------------------------------

inline uint64_t mix_hash(uint64_t h, uint64_t v) {
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

constexpr uint64_t kHashSeed = 0x84222325CBF29CE4ULL;
constexpr uint64_t kNullHash = 0x5BD1E9955BD1E995ULL;

namespace detail {

// One key column, flattened to raw pointers for tight loops.
struct KeyView {
    DType type;
    const void* values;      // flat values, or the dictionary
    const uint32_t* codes;   // dictionary codes (nullptr: flat)
    const uint8_t* valid;    // nullptr: no nulls
    std::vector<uint64_t> dict_hash;  // per dictionary entry

    explicit KeyView(const Column& c, bool hashes) {
        type = c.type();
        values = c.visit_stored([](const auto& v) -> const void* { return v.data(); });
        codes = c.is_dict() ? c.codes().data() : nullptr;
        valid = c.validity();
        if (codes && hashes) {
            c.visit_stored([&](const auto& v) {
                dict_hash.resize(v.size());
                for (size_t i = 0; i < v.size(); ++i) dict_hash[i] = scalar_hash(v[i]);
            });
        }
    }
    size_t index(size_t r) const { return codes ? codes[r] : r; }
    bool is_null(size_t r) const { return valid && !valid[r]; }
};

}  // namespace detail

// Row hash with the type dispatch done once per batch.
class KeyHasher {
public:
    explicit KeyHasher(const std::vector<Column>& keys) {
        for (const auto& c : keys) cols_.emplace_back(c, true);
    }
    uint64_t operator()(size_t r) const {
        uint64_t h = kHashSeed;
        for (const auto& c : cols_) h = mix_hash(h, value(c, r));
        return finalize_hash(h);
    }

private:
    static uint64_t value(const detail::KeyView& c, size_t r) {
        if (c.is_null(r)) return kNullHash;
        if (c.codes) return c.dict_hash[c.codes[r]];
        switch (c.type) {
        case DType::Int64: return scalar_hash(static_cast<const int64_t*>(c.values)[r]);
        case DType::Float64: return scalar_hash(static_cast<const double*>(c.values)[r]);
        case DType::String: return scalar_hash(static_cast<const std::string*>(c.values)[r]);
        case DType::Bool: return scalar_hash(static_cast<const uint8_t*>(c.values)[r]);
        }
        return 0;
    }
    std::vector<detail::KeyView> cols_;
};

inline uint64_t row_hash(const std::vector<Column>& keys, size_t row) {
    return KeyHasher(keys)(row);
}

inline std::vector<uint64_t> hash_rows(const std::vector<Column>& keys, size_t rows) {
    const KeyHasher h(keys);
    std::vector<uint64_t> out(rows);
    for (size_t r = 0; r < rows; ++r) out[r] = h(r);
    return out;
}

// keys_a[row_a] == keys_b[row_b] column by column (types must match; nulls
// equal nulls).
class KeyEq {
public:
    KeyEq(const std::vector<Column>& a, const std::vector<Column>& b) {
        if (a.size() != b.size()) throw std::invalid_argument("key count differs");
        for (size_t k = 0; k < a.size(); ++k) {
            if (a[k].type() != b[k].type()) throw std::invalid_argument("key types differ");
            pairs_.push_back({detail::KeyView(a[k], false), detail::KeyView(b[k], false)});
        }
    }
    bool operator()(size_t ra, size_t rb) const {
        for (const auto& p : pairs_) {
            const bool na = p.a.is_null(ra), nb = p.b.is_null(rb);
            if (na || nb) {
                if (na != nb) return false;
                continue;
            }
            // Same dictionary and code: equal. (Different codes may still hold
            // equal values: join outputs use the build column as dictionary.)
            if (p.a.codes && p.b.codes && p.a.values == p.b.values && p.a.codes[ra] == p.b.codes[rb]) continue;
            const size_t ia = p.a.index(ra), ib = p.b.index(rb);
            switch (p.a.type) {
            case DType::Int64:
                if (static_cast<const int64_t*>(p.a.values)[ia] != static_cast<const int64_t*>(p.b.values)[ib])
                    return false;
                break;
            case DType::Float64:
                if (static_cast<const double*>(p.a.values)[ia] != static_cast<const double*>(p.b.values)[ib])
                    return false;
                break;
            case DType::String:
                if (static_cast<const std::string*>(p.a.values)[ia] !=
                    static_cast<const std::string*>(p.b.values)[ib])
                    return false;
                break;
            case DType::Bool:
                if (static_cast<const uint8_t*>(p.a.values)[ia] != static_cast<const uint8_t*>(p.b.values)[ib])
                    return false;
                break;
            }
        }
        return true;
    }

private:
    struct Pair {
        detail::KeyView a, b;
    };
    std::vector<Pair> pairs_;
};

inline bool rows_equal(const std::vector<Column>& a, size_t ra, const std::vector<Column>& b, size_t rb) {
    return KeyEq(a, b)(ra, rb);
}

// True when any key column is null at row r.
inline bool any_null(const std::vector<Column>& keys, size_t r) {
    for (const auto& c : keys)
        if (!c.is_valid(r)) return true;
    return false;
}

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
// Wire format, so batches can flow through the engine's shuffle (and between
// cluster peers). Dictionary columns are sent decoded.
// ---------------------------------------------------------------------------

namespace vega {

template<>
struct Serializer<df::Column> {
    static void serialize(const df::Column& in, ByteBuffer& buf) {
        const df::Column c = in.flat();
        write_pod<uint8_t>(buf, static_cast<uint8_t>(c.type()));
        write_pod<uint8_t>(buf, c.has_nulls() ? 1 : 0);
        c.visit([&](const auto& v) {
            using T = typename std::decay_t<decltype(v)>::value_type;
            write_pod<uint64_t>(buf, v.size());
            if (c.has_nulls()) write_raw(buf, c.validity(), v.size());
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
        const bool nulls = read_pod<uint8_t>(buf, off) != 0;
        const auto n = static_cast<size_t>(read_pod<uint64_t>(buf, off));
        df::BoolVec valid;
        if (nulls) {
            valid.resize(n);
            read_raw(buf, off, valid.data(), n);
        }
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
            if (nulls) return df::Column(std::move(v), std::move(valid));
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
