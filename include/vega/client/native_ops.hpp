#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "vega/serialization/serializer.hpp"

namespace vega::client {

enum class NativeOp : uint8_t {
    MapAffine = 1,
    FilterModEq = 2,
    FlatMapRange = 3,
    MapToPairAffine = 4,
    ReduceByKeySum = 5,
    MapToPairModConst = 6,
    MapToPairConst = 7,
    MapAffineF64 = 8,
    FilterCompareF64 = 9,
    FlatMapRangeF64 = 10,
    MapToPairConstF64 = 11,
    FlatMapSplit = 12,
    MapToPairConstStr = 13,
    FilterLengthGt = 14,
    Float64NarrowPipeline = 15,
    Int64NarrowPipeline = 16,
    StringLineWordCount = 17,
    MapToPairModConstF64 = 18,
    MapToPairModIdentity = 19,
};

enum class NativeParamKind : uint8_t {
    I64 = 0,
    F64 = 1,
    Str = 2,
};

enum class FloatCompareOp : uint8_t {
    Eq = 0,
    Gt = 1,
    Lt = 2,
    Ge = 3,
    Le = 4,
};

enum class TransformMode : uint8_t {
    Python = 0,
    Native = 1,
};

struct NativeParam {
    NativeParamKind kind = NativeParamKind::I64;
    int64_t i64 = 0;
    double f64 = 0.0;
    std::string str;
};

inline void write_i64(ByteBuffer& buf, int64_t value) {
    write_pod<int64_t>(buf, value);
}

inline int64_t read_i64(const ByteBuffer& buf, size_t& off) {
    return read_pod<int64_t>(buf, off);
}

inline void write_native_params(ByteBuffer& buf,
                                NativeOp op,
                                const std::vector<NativeParam>& params) {
    write_pod<uint8_t>(buf, static_cast<uint8_t>(op));
    write_pod<uint8_t>(buf, static_cast<uint8_t>(params.size()));
    for (const auto& p : params) {
        write_pod<uint8_t>(buf, static_cast<uint8_t>(p.kind));
        switch (p.kind) {
        case NativeParamKind::I64:
            write_i64(buf, p.i64);
            break;
        case NativeParamKind::F64:
            write_pod<double>(buf, p.f64);
            break;
        case NativeParamKind::Str:
            write_bytes(buf, ByteBuffer(
                reinterpret_cast<const uint8_t*>(p.str.data()),
                reinterpret_cast<const uint8_t*>(p.str.data()) + p.str.size()));
            break;
        }
    }
}

inline std::vector<NativeParam> read_native_params(const ByteBuffer& buf,
                                                   size_t& off,
                                                   NativeOp& op) {
    op = static_cast<NativeOp>(read_pod<uint8_t>(buf, off));
    const uint8_t count = read_pod<uint8_t>(buf, off);
    std::vector<NativeParam> params;
    params.reserve(count);
    for (uint8_t i = 0; i < count; ++i) {
        NativeParam p;
        p.kind = static_cast<NativeParamKind>(read_pod<uint8_t>(buf, off));
        switch (p.kind) {
        case NativeParamKind::I64:
            p.i64 = read_i64(buf, off);
            break;
        case NativeParamKind::F64:
            p.f64 = read_pod<double>(buf, off);
            break;
        case NativeParamKind::Str: {
            auto bytes = read_bytes(buf, off);
            p.str.assign(reinterpret_cast<const char*>(bytes.data()), bytes.size());
            break;
        }
        }
        params.push_back(std::move(p));
    }
    return params;
}

inline size_t expected_native_param_count(NativeOp op) {
    switch (op) {
    case NativeOp::MapAffine: return 3;
    case NativeOp::FilterModEq: return 2;
    case NativeOp::FlatMapRange: return 2;
    case NativeOp::MapToPairAffine: return 6;
    case NativeOp::ReduceByKeySum: return 0;
    case NativeOp::MapToPairModConst: return 2;
    case NativeOp::MapToPairConst: return 1;
    case NativeOp::MapAffineF64: return 2;
    case NativeOp::FilterCompareF64: return 2;
    case NativeOp::FlatMapRangeF64: return 2;
    case NativeOp::MapToPairConstF64: return 1;
    case NativeOp::FlatMapSplit: return 1;
    case NativeOp::MapToPairConstStr: return 1;
    case NativeOp::FilterLengthGt: return 1;
    case NativeOp::Float64NarrowPipeline: return 6;
    case NativeOp::Int64NarrowPipeline: return 7;
    case NativeOp::StringLineWordCount: return 3;
    case NativeOp::MapToPairModConstF64: return 2;
    case NativeOp::MapToPairModIdentity: return 1;
    }
    throw std::runtime_error("unknown native op");
}

}  // namespace vega::client
