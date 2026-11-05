#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

#include "vega/serialization/serializer.hpp"

namespace vega::client {

enum class MsgType : uint8_t {
    Hello         = 0x80,
    HelloAck      = 0x81,
    Parallelize   = 0x82,
    ParallelizeAck = 0x83,
    Map           = 0x84,
    Filter        = 0x85,
    FlatMap       = 0x86,
    TransformAck  = 0x87,
    Count         = 0x88,
    CountAck      = 0x89,
    Collect       = 0x8a,
    CollectAck    = 0x8b,
    Reduce        = 0x8c,
    ReduceAck     = 0x8d,
    Foreach        = 0x8e,
    ForeachAck     = 0x8f,
    ParallelizeRange = 0x90,
    ParallelizeRangeAck = 0x91,
    ParallelizeFloatRange = 0x96,
    ParallelizeFloatRangeAck = 0x97,
    ParallelizeStringLines = 0x98,
    ParallelizeStringLinesAck = 0x99,
    ReduceByKey    = 0x94,
    ReduceByKeyAck = 0x95,
    GroupByKey     = 0x9A,
    GroupByKeyAck  = 0x9B,
    CollectPartitions = 0x9C,
    CollectPartitionsAck = 0x9D,
    CollectPartition = 0x9E,
    CollectPartitionAck = 0x9F,
    ParallelizePartitions = 0xA0,
    ParallelizePartitionsAck = 0xA1,
    Execute        = 0x92,
    ExecuteAck     = 0x93,
    // --- v0.6 server-side operators --------------------------------------
    ApplyStages    = 0xA2,   // base id + stages -> new id (one round trip)
    ApplyStagesAck = 0xA3,
    Union          = 0xA4,   // ids -> concatenated RDD
    UnionAck       = 0xA5,
    PartitionBy    = 0xA6,   // id, partitions, mode, partitioner -> shuffled RDD
    PartitionByAck = 0xA7,
    Persist        = 0xA8,   // id -> cached RDD id
    PersistAck     = 0xA9,
    Release        = 0xAA,   // ids -> drop server-side handles
    ReleaseAck     = 0xAB,
    RunPython      = 0xAC,   // pickled (func, args) -> pickled result
    RunPythonAck   = 0xAD,
    Coalesce       = 0xAE,   // id + partition groups -> narrow coalesced RDD
    CoalesceAck    = 0xAF,
    Cartesian      = 0xB0,
    CartesianAck   = 0xB1,
    ZipPartitions  = 0xB2,
    ZipPartitionsAck = 0xB3,
    RunJob         = 0xB4,   // id + partition subset + mode -> per-partition results
    RunJobAck      = 0xB5,
    NativeStats    = 0xB6,   // id -> (count, sum, min, max, m2) or None
    NativeStatsAck = 0xB7,
    Describe       = 0xB8,   // id -> storage kind, partitions, cached partitions
    DescribeAck    = 0xB9,
    Error         = 0xff,
};

inline void write_bytes(ByteBuffer& buf, const ByteBuffer& data) {
    write_pod<uint32_t>(buf, static_cast<uint32_t>(data.size()));
    buf.insert(buf.end(), data.begin(), data.end());
}

inline ByteBuffer read_bytes(const ByteBuffer& buf, size_t& off) {
    uint32_t len = read_pod<uint32_t>(buf, off);
    if (off + len > buf.size())
        throw std::out_of_range("client protocol: truncated byte field");
    ByteBuffer out(buf.begin() + static_cast<ptrdiff_t>(off),
                   buf.begin() + static_cast<ptrdiff_t>(off + len));
    off += len;
    return out;
}

inline ByteBuffer encode(MsgType type, const ByteBuffer& body) {
    ByteBuffer frame;
    write_pod<uint8_t>(frame, static_cast<uint8_t>(type));
    frame.insert(frame.end(), body.begin(), body.end());
    return frame;
}

inline MsgType decode_type(const ByteBuffer& frame) {
    if (frame.empty())
        throw std::runtime_error("client protocol: empty frame");
    return static_cast<MsgType>(frame[0]);
}

inline ByteBuffer decode_body(const ByteBuffer& frame) {
    if (frame.size() <= 1) return {};
    return ByteBuffer(frame.begin() + 1, frame.end());
}

} // namespace vega::client
