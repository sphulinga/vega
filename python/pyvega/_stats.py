"""RDD statistics helpers."""

from __future__ import annotations

import bisect
from math import isinf, isnan
from typing import Any, Callable, Iterable, Sequence


def portable_hash(x: Any) -> int:
    if x is None:
        return 0
    if isinstance(x, tuple):
        h = 0
        for item in x:
            h = (31 * h + portable_hash(item)) & 0xFFFFFFFF
        return h
    return hash(x) & 0xFFFFFFFF


def count_approx_distinct(values: Iterable[Any], relative_sd: float = 0.05) -> int:
    if relative_sd < 0.000017:
        raise ValueError("relativeSD should be greater than 0.000017")
    return len({portable_hash(v) for v in values})


def histogram_values(
    values: Iterable[Any],
    buckets: int | list[Any] | tuple[Any, ...],
) -> tuple[Sequence[Any], list[int]]:
    if isinstance(buckets, int):
        if buckets < 1:
            raise ValueError("number of buckets must be >= 1")

        def comparable(x: Any) -> bool:
            if x is None:
                return False
            if isinstance(x, float) and isnan(x):
                return False
            return True

        filtered = [v for v in values if comparable(v)]
        if not filtered:
            raise ValueError("can not generate buckets from empty RDD")

        minv = min(filtered)
        maxv = max(filtered)
        if minv == maxv or buckets == 1:
            return [minv, maxv], [len(filtered)]

        try:
            inc = (maxv - minv) / buckets
        except TypeError as exc:
            raise TypeError("Can not generate buckets with non-number in RDD") from exc

        if isinf(inc):
            raise ValueError("Can not generate buckets with infinite value")

        inc_val: int | float = int(inc)
        if inc_val * buckets != maxv - minv:
            inc_val = (maxv - minv) * 1.0 / buckets

        bucket_list = [i * inc_val + minv for i in range(buckets)]
        bucket_list.append(maxv)
        even = True
    elif isinstance(buckets, (list, tuple)):
        if len(buckets) < 2:
            raise ValueError("buckets should have more than one value")
        if any(i is None or isinstance(i, float) and isnan(i) for i in buckets):
            raise ValueError("can not have None or NaN in buckets")
        if sorted(buckets) != list(buckets):
            raise ValueError("buckets should be sorted")
        if len(set(buckets)) != len(buckets):
            raise ValueError("buckets should not contain duplicated values")
        bucket_list = list(buckets)
        minv = bucket_list[0]
        maxv = bucket_list[-1]
        even = False
        inc_val = None
        try:
            steps = [bucket_list[i + 1] - bucket_list[i] for i in range(len(bucket_list) - 1)]
        except TypeError:
            pass
        else:
            if max(steps) - min(steps) < 1e-10:
                even = True
                inc_val = (maxv - minv) / (len(bucket_list) - 1)
    else:
        raise TypeError("buckets should be a list or tuple or number(int or long)")

    counters = [0] * len(bucket_list)

    for item in values:
        if item is None or (isinstance(item, float) and isnan(item)) or item > maxv or item < minv:
            continue
        if even:
            assert inc_val is not None
            idx = int((item - minv) / inc_val)
        else:
            idx = bisect.bisect_right(bucket_list, item) - 1
        counters[idx] += 1

    last = counters.pop()
    counters[-1] += last
    return bucket_list, counters


def histogram_partition(
    iterator: Iterable[Any],
    buckets: int | list[Any] | tuple[Any, ...],
) -> list[int]:
    _, counts = histogram_values(iterator, buckets)
    return counts


# ---------------------------------------------------------------------------
# HyperLogLog (used by countApproxDistinct)
# ---------------------------------------------------------------------------

_MASK64 = (1 << 64) - 1


def _mix64(x: int) -> int:
    """SplitMix64 finalizer: spreads Python's (often identity) hashes."""
    x = (x + 0x9E3779B97F4A7C15) & _MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK64
    return x ^ (x >> 31)


# Minimum register precision. Like HyperLogLog++'s high-precision sparse mode,
# a floor of 2**14 registers (16 KiB per partition) keeps small cardinalities
# effectively exact; a requested relativeSD only ever raises the precision.
HLL_MIN_PRECISION = 14
HLL_MAX_PRECISION = 18


def hll_precision(relative_sd: float) -> int:
    if relative_sd < 0.000017:
        raise ValueError("relativeSD should be greater than 0.000017")
    if relative_sd > 0.37:
        raise ValueError("relativeSD should be smaller than 0.37")
    import math

    p = int(math.ceil(2.0 * math.log(1.054 / relative_sd) / math.log(2)))
    return max(HLL_MIN_PRECISION, min(p, HLL_MAX_PRECISION))


def _type_salt(value: Any) -> int:
    """Deterministic per-type salt (str hashes are randomized per process)."""
    import zlib

    return zlib.crc32(type(value).__name__.encode()) << 32


def hll_partition(iterator: Iterable[Any], p: int) -> Iterable[bytearray]:
    """Build HyperLogLog registers for one partition."""
    m = 1 << p
    registers = bytearray(m)
    shift = 64 - p
    for value in iterator:
        h = _mix64(portable_hash(value) ^ _type_salt(value))
        idx = h >> shift
        rest = (h << p) & _MASK64
        rank = (64 - p + 1) if rest == 0 else (65 - rest.bit_length())
        if rank > registers[idx]:
            registers[idx] = rank
    yield registers


def hll_merge(a: bytearray, b: bytearray) -> bytearray:
    return bytearray(map(max, a, b))


def hll_estimate(registers: bytearray) -> int:
    import math

    m = len(registers)
    alpha = {16: 0.673, 32: 0.697, 64: 0.709}.get(m, 0.7213 / (1.0 + 1.079 / m))
    estimate = alpha * m * m / sum(2.0 ** -r for r in registers)
    zeros = registers.count(0)
    if estimate <= 2.5 * m and zeros:
        estimate = m * math.log(m / zeros)  # linear counting (small range)
    return int(round(estimate))
