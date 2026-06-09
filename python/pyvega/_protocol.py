"""Wire protocol helpers for PyVega <-> Vega server."""

from __future__ import annotations

import struct
from enum import IntEnum


class MsgType(IntEnum):
    HELLO = 0x80
    HELLO_ACK = 0x81
    PARALLELIZE = 0x82
    PARALLELIZE_ACK = 0x83
    MAP = 0x84
    FILTER = 0x85
    FLAT_MAP = 0x86
    TRANSFORM_ACK = 0x87
    COUNT = 0x88
    COUNT_ACK = 0x89
    COLLECT = 0x8A
    COLLECT_ACK = 0x8B
    REDUCE = 0x8C
    REDUCE_ACK = 0x8D
    FOREACH = 0x8E
    FOREACH_ACK = 0x8F
    PARALLELIZE_RANGE = 0x90
    PARALLELIZE_RANGE_ACK = 0x91
    PARALLELIZE_FLOAT_RANGE = 0x96
    PARALLELIZE_FLOAT_RANGE_ACK = 0x97
    PARALLELIZE_STRING_LINES = 0x98
    PARALLELIZE_STRING_LINES_ACK = 0x99
    REDUCE_BY_KEY = 0x94
    REDUCE_BY_KEY_ACK = 0x95
    GROUP_BY_KEY = 0x9A
    GROUP_BY_KEY_ACK = 0x9B
    COLLECT_PARTITIONS = 0x9C
    COLLECT_PARTITIONS_ACK = 0x9D
    COLLECT_PARTITION = 0x9E
    COLLECT_PARTITION_ACK = 0x9F
    PARALLELIZE_PARTITIONS = 0xA0
    PARALLELIZE_PARTITIONS_ACK = 0xA1
    EXECUTE = 0x92
    EXECUTE_ACK = 0x93
    ERROR = 0xFF


class StageOp(IntEnum):
    MAP = 1
    FILTER = 2
    FLAT_MAP = 3
    REDUCE_BY_KEY = 4
    GROUP_BY_KEY = 5


class NativeOp(IntEnum):
    MAP_AFFINE = 1
    FILTER_MOD_EQ = 2
    FLAT_MAP_RANGE = 3
    MAP_TO_PAIR_AFFINE = 4
    REDUCE_BY_KEY_SUM = 5
    MAP_TO_PAIR_MOD_CONST = 6
    MAP_TO_PAIR_CONST = 7
    MAP_AFFINE_F64 = 8
    FILTER_COMPARE_F64 = 9
    FLAT_MAP_RANGE_F64 = 10
    MAP_TO_PAIR_CONST_F64 = 11
    FLAT_MAP_SPLIT = 12
    MAP_TO_PAIR_CONST_STR = 13
    FILTER_LENGTH_GT = 14
    FLOAT64_NARROW_PIPELINE = 15
    INT64_NARROW_PIPELINE = 16
    STRING_LINE_WORD_COUNT = 17
    MAP_TO_PAIR_MOD_CONST_F64 = 18
    MAP_TO_PAIR_MOD_IDENTITY = 19


class NativeParamKind(IntEnum):
    I64 = 0
    F64 = 1
    STR = 2


class FloatCompareOp(IntEnum):
    EQ = 0
    GT = 1
    LT = 2
    GE = 3
    LE = 4


def pack_native_param(value: int | float | str) -> bytes:
    if isinstance(value, float):
        return pack_u8(NativeParamKind.F64) + struct.pack("<d", float(value))
    if isinstance(value, str):
        return pack_u8(NativeParamKind.STR) + pack_bytes(value.encode("utf-8"))
    return pack_u8(NativeParamKind.I64) + struct.pack("<q", int(value))


def pack_native_tail(
    native_op: NativeOp, native_params: tuple[int | float | str, ...]
) -> bytes:
    body = pack_u8(int(TransformMode.NATIVE))
    body += pack_u8(int(native_op))
    body += pack_u8(len(native_params))
    for param in native_params:
        body += pack_native_param(param)
    return body


class TransformMode(IntEnum):
    PYTHON = 0
    NATIVE = 1


class Action(IntEnum):
    COUNT = 1
    COLLECT = 2
    REDUCE = 3
    FOREACH = 4


def pack_frame(msg_type: MsgType, body: bytes = b"") -> bytes:
    payload = bytes([int(msg_type)]) + body
    return struct.pack(">I", len(payload)) + payload


def unpack_frame(data: bytes) -> tuple[MsgType, bytes]:
    if len(data) < 5:
        raise RuntimeError("truncated frame")
    (length,) = struct.unpack(">I", data[:4])
    payload = data[4 : 4 + length]
    if len(payload) != length:
        raise RuntimeError("incomplete frame payload")
    return MsgType(payload[0]), payload[1:]


def pack_u32(value: int) -> bytes:
    return struct.pack("<I", value)


def pack_u64(value: int) -> bytes:
    return struct.pack("<Q", value)


def pack_u8(value: int) -> bytes:
    return struct.pack("<B", value)


def unpack_u32(body: bytes, offset: int = 0) -> tuple[int, int]:
    (value,) = struct.unpack("<I", body[offset : offset + 4])
    return value, offset + 4


def unpack_u64(body: bytes, offset: int = 0) -> tuple[int, int]:
    (value,) = struct.unpack("<Q", body[offset : offset + 8])
    return value, offset + 8


def unpack_u8(body: bytes, offset: int = 0) -> tuple[int, int]:
    (value,) = struct.unpack("<B", body[offset : offset + 1])
    return value, offset + 1


def pack_bytes(blob: bytes) -> bytes:
    return pack_u32(len(blob)) + blob


def unpack_bytes(body: bytes, offset: int = 0) -> tuple[bytes, int]:
    length, offset = unpack_u32(body, offset)
    end = offset + length
    return body[offset:end], end
