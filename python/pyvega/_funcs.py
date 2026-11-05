"""Partition-level helper functions executed inside the Vega server.

They live in an importable module so closures that use them pickle by
reference (small payloads) and are shared by every task.
"""

from __future__ import annotations

import bisect
import copy
import heapq
import itertools
import shlex
import threading
from functools import reduce as _reduce
from subprocess import PIPE, Popen
from typing import Any, Callable, Iterable, Iterator

_STOP_MSG = "Caught StopIteration thrown from user's code; failing the task"


def fail_on_stopiteration(f: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap ``f`` so a StopIteration it raises becomes a RuntimeError."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return f(*args, **kwargs)
        except StopIteration as exc:
            raise RuntimeError(_STOP_MSG) from exc

    return wrapper


def make_zero(zero_value: Any) -> Any:
    return zero_value() if callable(zero_value) else copy.deepcopy(zero_value)


# ----- simple partition functions -------------------------------------------


def apply_to_iterator(func: Callable[[Iterator[Any]], Iterable[Any]], _index: int, it):
    return func(it)


def glom(_index: int, it: Iterable[Any]) -> Iterator[list[Any]]:
    yield list(it)


def islice_partition(limit: int, _index: int, it: Iterable[Any]) -> Iterator[Any]:
    return itertools.islice(it, limit)


def partial_reduce(func, _index: int, it: Iterable[Any]) -> Iterator[Any]:
    iterator = iter(it)
    try:
        initial = next(iterator)
    except StopIteration:
        return
    yield _reduce(func, iterator, initial)


def partial_fold(zero_value, func, _index: int, it: Iterable[Any]) -> Iterator[Any]:
    acc = make_zero(zero_value)
    for obj in it:
        acc = func(acc, obj)
    yield acc


def partial_sum(_index: int, it: Iterable[Any]) -> Iterator[Any]:
    yield sum(it)


def consume_foreach(func, _index: int, it: Iterable[Any]) -> Iterator[Any]:
    for obj in it:
        func(obj)
    return iter(())


def consume_foreach_partition(func, _index: int, it: Iterable[Any]) -> Iterator[Any]:
    func(it)
    return iter(())


def stat_partition(_index: int, it: Iterable[Any]):
    from pyvega.statcounter import StatCounter

    yield StatCounter(it)


def top_partition(num: int, key, _index: int, it: Iterable[Any]):
    yield heapq.nlargest(num, it, key=key)


def bottom_partition(num: int, key, _index: int, it: Iterable[Any]):
    yield heapq.nsmallest(num, it, key=key)


def minmax_partition(_index: int, it: Iterable[Any]):
    from math import isnan

    lo = hi = None
    seen = False
    for x in it:
        if x is None or (isinstance(x, float) and isnan(x)):
            continue
        if not seen:
            lo = hi = x
            seen = True
        else:
            if x < lo:
                lo = x
            if x > hi:
                hi = x
    if seen:
        yield (lo, hi)


def histogram_partition(buckets, _index: int, it: Iterable[Any]):
    from pyvega._stats import histogram_partition as _hp

    yield _hp(it, buckets)


def hll_partition(p: int, _index: int, it: Iterable[Any]):
    from pyvega._stats import hll_partition as _hll

    return _hll(it, p)


def count_partition_items(_index: int, it: Iterable[Any]):
    count = 0
    for _ in it:
        count += 1
    yield count


# ----- key/value helpers ------------------------------------------------------


def combine_locally(create, merge_value, _index: int, it: Iterable[Any]):
    combiners: dict[Any, Any] = {}
    for k, v in it:
        if k in combiners:
            combiners[k] = merge_value(combiners[k], v)
        else:
            combiners[k] = create(v)
    return iter(combiners.items())


def count_values_partition(_index: int, it: Iterable[Any]):
    counts: dict[Any, int] = {}
    for x in it:
        counts[x] = counts.get(x, 0) + 1
    return iter(counts.items())


def tag_values(tag: int, kv):
    return (kv[0], (tag, kv[1]))


def split_tagged(num: int, kv):
    from pyvega.resultiterable import ResultIterable

    groups: list[list[Any]] = [[] for _ in range(num)]
    for tag, value in kv[1]:
        groups[tag].append(value)
    return (kv[0], tuple(ResultIterable(g) for g in groups))


def join_values(how: str, kv):
    key, (left, right) = kv
    if not left and how in ("right", "full"):
        left = [None]
    if not right and how in ("left", "full"):
        right = [None]
    return [(key, (v, w)) for v in left for w in right]


def first_of_pair(kv):
    return kv[0]


def second_of_pair(kv):
    return kv[1]


def with_none(x):
    return (x, None)


def keep_first(a, _b):
    return a


def add(a, b):
    return a + b


def map_values(func, kv):
    return (kv[0], func(kv[1]))


def flat_map_values(func, kv):
    key = kv[0]
    return [(key, x) for x in func(kv[1])]


def key_by(func, x):
    return (func(x), x)


def lookup_filter(key, kv):
    return kv[0] == key


def subtract_filter(kv):
    left, right = kv[1]
    return bool(left) and not right


def intersection_filter(kv):
    left, right = kv[1]
    return bool(left) and bool(right)


def flatten_left(kv):
    return [(kv[0], v) for v in kv[1][0]]


# ----- partitioning & sorting --------------------------------------------------


def range_partition(bounds, key_fn, ascending: bool, num_partitions: int, key) -> int:
    pos = bisect.bisect_left(bounds, key_fn(key))
    return pos if ascending else num_partitions - 1 - pos


def sort_partition(key_fn, ascending: bool, _index: int, it: Iterable[Any]):
    return iter(sorted(it, key=lambda kv: key_fn(kv[0]), reverse=not ascending))


def identity(x):
    return x


# ----- zipping & indexing ------------------------------------------------------

_SENTINEL = object()


def zip_iterators(left: Iterable[Any], right: Iterable[Any]):
    for x, y in itertools.zip_longest(left, right, fillvalue=_SENTINEL):
        if x is _SENTINEL or y is _SENTINEL:
            raise ValueError(
                "Can not zip RDDs with different number of elements in partition"
            )
        yield (x, y)


def zip_with_index(starts: list[int], index: int, it: Iterable[Any]):
    for i, x in enumerate(it, starts[index]):
        yield (x, i)


def zip_with_unique_id(num_partitions: int, index: int, it: Iterable[Any]):
    for i, x in enumerate(it):
        yield (x, i * num_partitions + index)


# ----- pipe ---------------------------------------------------------------------


def pipe_partition(command: str, env: dict[str, str], check_code: bool, _index: int, it):
    try:
        proc = Popen(shlex.split(command), env=env or None, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    except (FileNotFoundError, OSError) as exc:
        if check_code:
            raise RuntimeError(f"Pipe command {command!r} failed to start: {exc}") from exc
        return

    def write_input() -> None:
        try:
            assert proc.stdin is not None
            for obj in it:
                line = str(obj).rstrip("\n") + "\n"
                proc.stdin.write(line.encode("utf-8"))
        except BrokenPipeError:
            pass
        finally:
            if proc.stdin is not None:
                try:
                    proc.stdin.close()
                except BrokenPipeError:
                    pass

    threading.Thread(target=write_input, daemon=True).start()
    assert proc.stdout is not None
    for raw in iter(proc.stdout.readline, b""):
        yield raw.rstrip(b"\n").decode("utf-8")
    proc.wait()
    if check_code and proc.returncode:
        raise RuntimeError(f"Pipe command {command!r} exited with code {proc.returncode}")


# ----- Vega extras ------------------------------------------------------------


def batches(func, batch_size: int, _index: int, it: Iterable[Any]):
    iterator = iter(it)
    while True:
        batch = list(itertools.islice(iterator, batch_size))
        if not batch:
            return
        result = func(batch)
        if result is not None:
            yield from result


def distinct_by_partition(key_fn, _index: int, it: Iterable[Any]):
    seen: dict[Any, Any] = {}
    for x in it:
        k = key_fn(x)
        if k not in seen:
            seen[k] = x
    return iter(seen.items())


def top_by_key_partition(num: int, key, _index: int, it: Iterable[Any]):
    heaps: dict[Any, list[Any]] = {}
    for k, v in it:
        heaps.setdefault(k, []).append(v)
    for k, values in heaps.items():
        yield (k, heapq.nlargest(num, values, key=key))


def merge_top(num: int, key, a, b):
    return heapq.nlargest(num, itertools.chain(a, b), key=key)


def misra_gries_partition(capacity: int, _index: int, it: Iterable[Any]):
    counters: dict[Any, int] = {}
    for x in it:
        if x in counters:
            counters[x] += 1
        elif len(counters) < capacity:
            counters[x] = 1
        else:
            for k in list(counters):
                counters[k] -= 1
                if counters[k] == 0:
                    del counters[k]
    yield counters


def merge_misra_gries(capacity: int, a: dict, b: dict) -> dict:
    merged = dict(a)
    for k, v in b.items():
        merged[k] = merged.get(k, 0) + v
    if len(merged) > capacity:
        cut = sorted(merged.values(), reverse=True)[capacity]
        merged = {k: v - cut for k, v in merged.items() if v > cut}
    return merged


def member_filter(candidates: frozenset, x) -> bool:
    return x in candidates


def window_starts(window: int, step: int, pair):
    x, i = pair
    first = max(0, i - window + 1)
    first += (-first) % step
    return [(s, (i, x)) for s in range(first, i + 1, step)]


def assemble_window(window: int, kv):
    items = sorted(kv[1], key=lambda t: t[0])
    if len(items) < window:
        return []
    return [(kv[0], [x for _, x in items])]


def reservoir_partition(size: int, seed: int, key, index: int, it: Iterable[Any]):
    import random

    rng = random.Random(seed ^ index)
    sample: list[Any] = []
    n = 0
    for x in it:
        n += 1
        v = key(x) if key is not None else x
        if len(sample) < size:
            sample.append(v)
        else:
            j = rng.randrange(n)
            if j < size:
                sample[j] = v
    yield (n, sample)
