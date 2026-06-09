#pragma once

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace vega {

// ---------------------------------------------------------------------------
// ByteBuffer -- growable byte vector used as the wire format.
// ---------------------------------------------------------------------------

using ByteBuffer = std::vector<uint8_t>;

// ---------------------------------------------------------------------------
// Type tags for the wire format.
// ---------------------------------------------------------------------------

enum class TypeTag : uint8_t {
    INT32   = 0x01,
    INT64   = 0x02,
    FLOAT32 = 0x03,
    FLOAT64 = 0x04,
    STRING  = 0x05,
    BOOL    = 0x06,
    VECTOR  = 0x07,
    PAIR    = 0x08,
    BYTES   = 0x09,
};

// ---------------------------------------------------------------------------
// Low-level write/read helpers (little-endian, no alignment requirements).
// ---------------------------------------------------------------------------

inline void write_raw(ByteBuffer& buf, const void* src, size_t n) {
    const auto* p = static_cast<const uint8_t*>(src);
    buf.insert(buf.end(), p, p + n);
}

inline void read_raw(const ByteBuffer& buf, size_t& off,
                     void* dst, size_t n) {
    if (off + n > buf.size())
        throw std::runtime_error("deserialize: buffer underflow");
    std::memcpy(dst, buf.data() + off, n);
    off += n;
}

template<typename T>
void write_pod(ByteBuffer& buf, T val) {
    static_assert(std::is_trivially_copyable_v<T>);
    write_raw(buf, &val, sizeof(T));
}

template<typename T>
T read_pod(const ByteBuffer& buf, size_t& off) {
    static_assert(std::is_trivially_copyable_v<T>);
    T val;
    read_raw(buf, off, &val, sizeof(T));
    return val;
}

inline void write_tag(ByteBuffer& buf, TypeTag t) {
    buf.push_back(static_cast<uint8_t>(t));
}

inline TypeTag read_tag(const ByteBuffer& buf, size_t& off) {
    return static_cast<TypeTag>(read_pod<uint8_t>(buf, off));
}

// ---------------------------------------------------------------------------
// Serializer<T> -- primary template; specializations below.
//
// Every specialization provides:
//   static void serialize(const T& val, ByteBuffer& buf);
//   static T    deserialize(const ByteBuffer& buf, size_t& off);
// ---------------------------------------------------------------------------

template<typename T, typename Enable = void>
struct Serializer;

// -- int32 ------------------------------------------------------------------

template<>
struct Serializer<int32_t> {
    static void serialize(int32_t v, ByteBuffer& buf) {
        write_tag(buf, TypeTag::INT32);
        write_pod(buf, v);
    }
    static int32_t deserialize(const ByteBuffer& buf, size_t& off) {
        auto tag = read_tag(buf, off);
        if (tag != TypeTag::INT32)
            throw std::runtime_error("expected INT32 tag");
        return read_pod<int32_t>(buf, off);
    }
};

// -- int / int64 ------------------------------------------------------------

template<>
struct Serializer<int64_t> {
    static void serialize(int64_t v, ByteBuffer& buf) {
        write_tag(buf, TypeTag::INT64);
        write_pod(buf, v);
    }
    static int64_t deserialize(const ByteBuffer& buf, size_t& off) {
        auto tag = read_tag(buf, off);
        if (tag != TypeTag::INT64)
            throw std::runtime_error("expected INT64 tag");
        return read_pod<int64_t>(buf, off);
    }
};

// If `int` is not the same type as int32_t (rare), delegate to int32_t.
// On most platforms int == int32_t so this is unused.

// -- float / double ---------------------------------------------------------

template<>
struct Serializer<float> {
    static void serialize(float v, ByteBuffer& buf) {
        write_tag(buf, TypeTag::FLOAT32);
        write_pod(buf, v);
    }
    static float deserialize(const ByteBuffer& buf, size_t& off) {
        auto tag = read_tag(buf, off);
        if (tag != TypeTag::FLOAT32)
            throw std::runtime_error("expected FLOAT32 tag");
        return read_pod<float>(buf, off);
    }
};

template<>
struct Serializer<double> {
    static void serialize(double v, ByteBuffer& buf) {
        write_tag(buf, TypeTag::FLOAT64);
        write_pod(buf, v);
    }
    static double deserialize(const ByteBuffer& buf, size_t& off) {
        auto tag = read_tag(buf, off);
        if (tag != TypeTag::FLOAT64)
            throw std::runtime_error("expected FLOAT64 tag");
        return read_pod<double>(buf, off);
    }
};

// -- bool -------------------------------------------------------------------

template<>
struct Serializer<bool> {
    static void serialize(bool v, ByteBuffer& buf) {
        write_tag(buf, TypeTag::BOOL);
        write_pod<uint8_t>(buf, v ? 1 : 0);
    }
    static bool deserialize(const ByteBuffer& buf, size_t& off) {
        auto tag = read_tag(buf, off);
        if (tag != TypeTag::BOOL)
            throw std::runtime_error("expected BOOL tag");
        return read_pod<uint8_t>(buf, off) != 0;
    }
};

// -- std::string ------------------------------------------------------------

template<>
struct Serializer<std::string> {
    static void serialize(const std::string& v, ByteBuffer& buf) {
        write_tag(buf, TypeTag::STRING);
        write_pod<uint32_t>(buf, static_cast<uint32_t>(v.size()));
        write_raw(buf, v.data(), v.size());
    }
    static std::string deserialize(const ByteBuffer& buf, size_t& off) {
        auto tag = read_tag(buf, off);
        if (tag != TypeTag::STRING)
            throw std::runtime_error("expected STRING tag");
        auto len = read_pod<uint32_t>(buf, off);
        if (off + len > buf.size())
            throw std::runtime_error("deserialize: string overflow");
        std::string s(reinterpret_cast<const char*>(buf.data() + off), len);
        off += len;
        return s;
    }
};

// -- std::vector<T> ---------------------------------------------------------

template<typename T>
struct Serializer<std::vector<T>> {
    static void serialize(const std::vector<T>& v, ByteBuffer& buf) {
        write_tag(buf, TypeTag::VECTOR);
        write_pod<uint32_t>(buf, static_cast<uint32_t>(v.size()));
        for (const auto& elem : v)
            Serializer<T>::serialize(elem, buf);
    }
    static std::vector<T> deserialize(const ByteBuffer& buf, size_t& off) {
        auto tag = read_tag(buf, off);
        if (tag != TypeTag::VECTOR)
            throw std::runtime_error("expected VECTOR tag");
        auto len = read_pod<uint32_t>(buf, off);
        std::vector<T> v;
        v.reserve(len);
        for (uint32_t i = 0; i < len; ++i)
            v.push_back(Serializer<T>::deserialize(buf, off));
        return v;
    }
};

// -- std::pair<K, V> --------------------------------------------------------

template<typename K, typename V>
struct Serializer<std::pair<K, V>> {
    static void serialize(const std::pair<K, V>& p, ByteBuffer& buf) {
        write_tag(buf, TypeTag::PAIR);
        Serializer<K>::serialize(p.first, buf);
        Serializer<V>::serialize(p.second, buf);
    }
    static std::pair<K, V> deserialize(const ByteBuffer& buf, size_t& off) {
        auto tag = read_tag(buf, off);
        if (tag != TypeTag::PAIR)
            throw std::runtime_error("expected PAIR tag");
        auto k = Serializer<K>::deserialize(buf, off);
        auto v = Serializer<V>::deserialize(buf, off);
        return {std::move(k), std::move(v)};
    }
};

// -- raw bytes (ByteBuffer) -------------------------------------------------

template<>
struct Serializer<ByteBuffer> {
    static void serialize(const ByteBuffer& v, ByteBuffer& buf) {
        write_tag(buf, TypeTag::BYTES);
        write_pod<uint32_t>(buf, static_cast<uint32_t>(v.size()));
        write_raw(buf, v.data(), v.size());
    }
    static ByteBuffer deserialize(const ByteBuffer& buf, size_t& off) {
        auto tag = read_tag(buf, off);
        if (tag != TypeTag::BYTES)
            throw std::runtime_error("expected BYTES tag");
        auto len = read_pod<uint32_t>(buf, off);
        if (off + len > buf.size())
            throw std::runtime_error("deserialize: bytes overflow");
        ByteBuffer v(buf.begin() + off, buf.begin() + off + len);
        off += len;
        return v;
    }
};

// ---------------------------------------------------------------------------
// Convenience helpers
// ---------------------------------------------------------------------------

template<typename T>
ByteBuffer serialize(const T& val) {
    ByteBuffer buf;
    Serializer<T>::serialize(val, buf);
    return buf;
}

template<typename T>
T deserialize(const ByteBuffer& buf) {
    size_t off = 0;
    return Serializer<T>::deserialize(buf, off);
}

} // namespace vega
