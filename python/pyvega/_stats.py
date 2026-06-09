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
