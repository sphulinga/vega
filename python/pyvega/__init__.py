"""
PyVega -- Python client for the Vega C++ engine.

Connects to a Vega session server over TCP. Computation and the actor
system run in the C++ server process; this package is a thin RPC client.

    from pyvega import SparkContext

    with SparkContext(4) as sc:
        result = sc.range(0, 100).map(lambda x: x * 2).collect()
"""

from __future__ import annotations

import bisect
import copy
import math
import os
import shlex
import subprocess
import threading
from subprocess import PIPE, Popen
from typing import Any, Callable, Iterable, Iterator, Optional

from pyvega._client import PendingStage, SessionClient, parse_remote, spawn_server
from pyvega._io import (
    read_pickle_file,
    read_text_file,
    save_as_pickle_file,
    save_as_text_file,
)
from pyvega._native import NativeKind, try_compile
from pyvega._protocol import Action, StageOp
from pyvega._sampler import RDDSampler
from pyvega._stats import count_approx_distinct, histogram_partition, histogram_values
from pyvega.broadcast import Broadcast, BroadcastPickleRegistry

__version__ = "0.5.0"
__all__ = ["SparkContext", "RDD"]

_next_rdd_id = 1


class RDD:
    """Remote RDD handle backed by the Vega C++ server.

    Transformations are recorded client-side and sent as one pipeline when an
    action runs, avoiding per-transform RPC round-trips.
    """

    def __init__(
        self,
        client: SessionClient,
        rdd_id: int,
        num_partitions: int,
        pending: Optional[list[PendingStage]] = None,
        native: bool = False,
        native_kind: NativeKind = "int64",
        line_meta: Optional[tuple[int, int]] = None,
        context: Optional["SparkContext"] = None,
        rdd_id_counter: Optional[int] = None,
        text_file_path: Optional[str] = None,
        text_file_parts: int = 0,
    ):
        global _next_rdd_id
        self._client = client
        self._rdd_id = rdd_id
        self._num_partitions = num_partitions
        self._pending = list(pending or [])
        self._native = native
        self._native_kind: NativeKind = native_kind
        self._line_meta = line_meta
        self._context = context
        self._text_file_path = text_file_path
        self._text_file_parts = text_file_parts
        if rdd_id_counter is None:
            rdd_id_counter = _next_rdd_id
            _next_rdd_id += 1
        self._logical_id = rdd_id_counter

    def _derive(self, stage: PendingStage, native: bool, native_kind: NativeKind) -> RDD:
        new_id = self._logical_id + (0 if self._pending else 1)
        return RDD(
            self._client,
            self._rdd_id,
            self._num_partitions,
            self._pending + [stage],
            native=native,
            native_kind=native_kind,
            line_meta=self._line_meta,
            context=self._context,
            rdd_id_counter=new_id,
            text_file_path=self._text_file_path,
            text_file_parts=self._text_file_parts,
        )

    def _refresh_text_file_source(self) -> None:
        if not self._text_file_path:
            return
        with open(self._text_file_path, encoding="utf-8") as handle:
            lines = [line.rstrip("\n") for line in handle]
        parts = self._text_file_parts or self._num_partitions or 1
        native, native_kind = SparkContext._detect_native_kind(lines)
        self._rdd_id, self._num_partitions = self._client.parallelize(lines, parts)
        self._native = native
        self._native_kind = native_kind

    def _stage(self, op: StageOp, func, num_partitions: int = 0) -> RDD:
        native = self._native
        native_kind = self._native_kind
        native_op = None
        native_params: tuple[int | float | str, ...] = ()
        if native and op != StageOp.GROUP_BY_KEY:
            compiled = try_compile(op, func, native_kind)
            if compiled is not None:
                native_op = compiled.native_op
                native_params = compiled.params
            else:
                native = False
        return self._derive(
            PendingStage(op, func, num_partitions, native_op, native_params),
            native,
            native_kind,
        )

    @staticmethod
    def _normalize_func(func: Callable[..., Any]) -> Callable[..., Any]:
        if func is range:
            return lambda x: list(range(x))
        if isinstance(func, type) or func in (str, int, float, bool, list, tuple, set, dict):
            return lambda x, f=func: f(x)
        return func

    def _materialize(self) -> None:
        self._refresh_text_file_source()
        if not self._pending:
            return
        rdd_id, parts = self._client.materialize(
            self._rdd_id,
            self._num_partitions,
            self._pending,
            self._line_meta,
        )
        self._rdd_id = rdd_id
        self._num_partitions = parts
        self._pending = []

    def _collect_partitions(self) -> list[list[Any]]:
        self._materialize()
        return self._client.collect_partitions(self._rdd_id)

    def _collect_partition(self, partition: int) -> list[Any]:
        self._materialize()
        return self._client.collect_partition(self._rdd_id, partition)

    def _from_partitions(
        self,
        partitions: list[list[Any]],
        native: bool = False,
        native_kind: NativeKind = "int64",
    ) -> RDD:
        ctx = self._require_context()
        rdd_id, parts = self._client.parallelize_partitions(partitions)
        return RDD(
            self._client,
            rdd_id,
            parts,
            native=native,
            native_kind=native_kind,
            context=ctx,
        )

    def map_partitions(self, func, preserves_partitioning: bool = False) -> RDD:
        parts = self._collect_partitions()
        mapped = [list(func(iter(part))) for part in parts]
        if preserves_partitioning:
            return self._from_partitions(mapped)
        flat = [item for part in mapped for item in part]
        return self._require_context().parallelize(flat, self._num_partitions)

    mapPartitions = map_partitions

    def map_partitions_with_index(
        self, func, preserves_partitioning: bool = False
    ) -> RDD:
        parts = self._collect_partitions()
        mapped = [list(func(index, iter(part))) for index, part in enumerate(parts)]
        if preserves_partitioning:
            return self._from_partitions(mapped)
        flat = [item for part in mapped for item in part]
        return self._require_context().parallelize(flat, self._num_partitions)

    mapPartitionsWithIndex = map_partitions_with_index

    def glom(self) -> RDD:
        def func(iterator: Iterable[Any]) -> Iterable[list[Any]]:
            yield list(iterator)

        return self.map_partitions(func, preserves_partitioning=True)

    def partition_by(self, num_partitions: int, partition_func) -> RDD:
        buckets: list[list[Any]] = [[] for _ in range(num_partitions)]
        for part in self._collect_partitions():
            for item in part:
                if isinstance(item, tuple) and len(item) >= 1:
                    key = item[0]
                else:
                    key = item
                bucket = int(partition_func(key)) % num_partitions
                buckets[bucket].append(item)
        return self._from_partitions(buckets)

    partitionBy = partition_by

    def map(self, func) -> RDD:
        return self._stage(StageOp.MAP, self._normalize_func(func))

    def filter(self, func) -> RDD:
        return self._stage(StageOp.FILTER, func)

    def flat_map(self, func) -> RDD:
        return self._stage(StageOp.FLAT_MAP, self._normalize_func(func))

    flatMap = flat_map

    def reduce_by_key(self, func, num_partitions: int = 0) -> RDD:
        if num_partitions == 0:
            num_partitions = self._num_partitions
        return self._stage(StageOp.REDUCE_BY_KEY, func, num_partitions)

    reduceByKey = reduce_by_key

    def group_by_key(self, num_partitions: int = 0) -> RDD:
        if num_partitions == 0:
            num_partitions = self._num_partitions
        return self._stage(StageOp.GROUP_BY_KEY, None, num_partitions)

    groupByKey = group_by_key

    def _resolve_pending(self, action: Action, action_func=None) -> Any:
        self._refresh_text_file_source()
        fused = self._client._fuse_stages(self._pending, self._line_meta)
        has_shuffle = any(
            stage.op in (StageOp.REDUCE_BY_KEY, StageOp.GROUP_BY_KEY) for stage in fused
        )
        if (
            action == Action.COUNT
            and not has_shuffle
            and self._client.stages_are_native(fused)
        ):
            result = self._client.execute(
                self._rdd_id,
                self._num_partitions,
                self._pending,
                action,
                action_func,
                self._line_meta,
            )
            self._pending = []
            return result
        rdd_id, parts = self._client.materialize(
            self._rdd_id,
            self._num_partitions,
            self._pending,
            self._line_meta,
        )
        self._rdd_id = rdd_id
        self._num_partitions = parts
        self._pending = []
        if action == Action.COUNT:
            return self._client.count(rdd_id)
        if action == Action.COLLECT:
            return self._client.collect(rdd_id)
        if action == Action.REDUCE:
            return self._client.reduce(rdd_id, action_func)
        if action == Action.FOREACH:
            self._client.foreach(rdd_id, action_func)
            return None
        raise RuntimeError(f"unsupported action: {action}")

    def _run(self, action: Action, action_func=None) -> Any:
        if self._pending:
            return self._resolve_pending(action, action_func)
        self._refresh_text_file_source()
        if action == Action.COUNT:
            return self._client.count(self._rdd_id)
        if action == Action.COLLECT:
            return self._client.collect(self._rdd_id)
        if action == Action.REDUCE:
            return self._client.reduce(self._rdd_id, action_func)
        if action == Action.FOREACH:
            self._client.foreach(self._rdd_id, action_func)
            return None
        raise RuntimeError(f"unsupported action: {action}")

    def collect(self) -> list:
        return self._run(Action.COLLECT)

    def count(self) -> int:
        return self._run(Action.COUNT)

    def reduce(self, func):
        return self._run(Action.REDUCE, func)

    def foreach(self, func) -> None:
        self._run(Action.FOREACH, func)

    def first(self):
        values = self.collect()
        if not values:
            raise ValueError("Cannot retrieve first element from an empty RDD")
        return values[0]

    def take(self, n: int) -> list:
        return self.collect()[:n]

    def sum(self):
        if self.isEmpty():
            return 0
        return self.reduce(lambda a, b: a + b)

    def isEmpty(self) -> bool:
        return self.count() == 0

    def distinct(self, num_partitions: int = 0) -> RDD:
        seen: list[Any] = []
        for item in self.collect():
            if item not in seen:
                seen.append(item)
        parts = num_partitions if num_partitions > 0 else self._num_partitions
        return self._require_context().parallelize(seen, parts)

    def coalesce(self, num_partitions: int, shuffle: bool = False) -> RDD:
        if shuffle:
            data = self.collect()
            return self._require_context().parallelize(data, num_partitions)
        parts = self._collect_partitions()
        if num_partitions >= len(parts):
            while len(parts) < num_partitions:
                parts.append([])
            return self._from_partitions(parts[:num_partitions])
        merged: list[list[Any]] = [[] for _ in range(num_partitions)]
        for index, part in enumerate(parts):
            merged[index % num_partitions].extend(part)
        return self._from_partitions(merged)

    def repartition(self, num_partitions: int) -> RDD:
        return self.coalesce(num_partitions, shuffle=True)

    def sort_by(self, keyfunc, ascending: bool = True, num_partitions: int = 0) -> RDD:
        keyed = self.map(lambda x: (keyfunc(x), x))
        sorted_pairs = keyed.sortByKey(ascending, num_partitions or None)
        return sorted_pairs.map(lambda kv: kv[1])

    def sortByKey(
        self,
        ascending: bool = True,
        numPartitions: int | None = None,
        keyfunc: Callable[[Any], Any] | None = None,
    ) -> RDD:
        key_fn = keyfunc or (lambda x: x)
        num_partitions = numPartitions or self._num_partitions
        if num_partitions == 1:
            data = sorted(self.collect(), key=lambda kv: key_fn(kv[0]), reverse=not ascending)
            return self._require_context().parallelize(data, 1)

        rdd_size = self.count()
        if rdd_size == 0:
            return self

        max_sample_size = num_partitions * 20.0
        fraction = min(max_sample_size / max(rdd_size, 1), 1.0)
        samples = sorted(
            (kv[0] for kv in self.sample(False, fraction, 1).collect()),
            key=key_fn,
        )
        bounds = [
            samples[int(len(samples) * (i + 1) / num_partitions)]
            for i in range(num_partitions - 1)
        ]

        def range_partitioner(key: Any) -> int:
            pos = bisect.bisect_left(bounds, key_fn(key))
            if ascending:
                return pos
            return num_partitions - 1 - pos

        def sort_partition(iterator: Iterable[Any]) -> Iterable[Any]:
            return iter(
                sorted(
                    iterator,
                    key=lambda kv: key_fn(kv[0]),
                    reverse=not ascending,
                )
            )

        return self.partition_by(num_partitions, range_partitioner).map_partitions(
            sort_partition,
            preserves_partitioning=True,
        )

    def sample(
        self,
        withReplacement: bool,
        fraction: float,
        seed: int | None = None,
    ) -> RDD:
        if fraction < 0:
            raise ValueError("Fraction must be nonnegative.")
        sampler = RDDSampler(withReplacement, fraction, seed)
        return self.map_partitions_with_index(sampler.func, preserves_partitioning=True)

    def takeSample(
        self, withReplacement: bool, num: int, seed: int | None = None
    ) -> list[Any]:
        import random

        fraction = min(1.0, (num * 10) / max(self.count(), 1))
        sampled = self.sample(withReplacement, fraction, seed).collect()
        rng = random.Random(seed)
        if withReplacement:
            return [rng.choice(sampled) for _ in range(num)] if sampled else []
        return rng.sample(sampled, min(num, len(sampled)))

    take_sample = takeSample

    def _pipe_impl(self, command: str, env: dict[str, str], checkCode: bool) -> RDD:
        def func(iterator: Iterable[Any]) -> Iterable[str]:
            try:
                proc = Popen(shlex.split(command), env=env, stdin=PIPE, stdout=PIPE, stderr=PIPE)
            except (FileNotFoundError, OSError) as exc:
                if checkCode:
                    raise RuntimeError(str(exc)) from exc
                return

            def write_input() -> None:
                try:
                    assert proc.stdin is not None
                    for obj in iterator:
                        line = str(obj).rstrip("\n") + "\n"
                        proc.stdin.write(line.encode("utf-8"))
                finally:
                    if proc.stdin is not None:
                        proc.stdin.close()

            threading.Thread(target=write_input, daemon=True).start()
            assert proc.stdout is not None
            for raw in iter(proc.stdout.readline, b""):
                yield raw.rstrip(b"\n").decode("utf-8")
            proc.wait()
            if checkCode and proc.returncode:
                raise RuntimeError(
                    f"Pipe command {command!r} exited with code {proc.returncode}"
                )

        return self.map_partitions(func)

    def pipe(self, command: str, env: dict[str, str] | None = None, checkCode: bool = False) -> RDD:
        return _PipeRDD(self, command, dict(env or {}), checkCode, self._require_context())

    def save_as_text_file(self, path: str, compressionCodecClass: str | None = None) -> None:
        del compressionCodecClass
        save_as_text_file(self._collect_partitions(), path)

    saveAsTextFile = save_as_text_file

    def save_as_pickle_file(self, path: str) -> None:
        save_as_pickle_file(self._collect_partitions(), path)

    saveAsPickleFile = save_as_pickle_file

    def histogram(self, buckets: int | list[Any] | tuple[Any, ...]):
        data = self.collect()
        if not data:
            if isinstance(buckets, int):
                if buckets < 1:
                    raise ValueError("number of buckets must be >= 1")
                raise ValueError("can not generate buckets from empty RDD")
            if isinstance(buckets, (list, tuple)):
                if len(buckets) < 2:
                    raise ValueError("buckets should have more than one value")
                return list(buckets), [0] * (len(buckets) - 1)
        if isinstance(buckets, int):
            return histogram_values(data, buckets)

        def partition_hist(iterator: Iterable[Any]) -> Iterable[list[int]]:
            yield histogram_partition(iterator, buckets)

        def merge(a: list[int], b: list[int]) -> list[int]:
            return [x + y for x, y in zip(a, b)]

        return buckets, self.map_partitions(partition_hist).reduce(merge)

    def countApproxDistinct(self, relativeSD: float = 0.05) -> int:
        return count_approx_distinct(self.collect(), relativeSD)

    count_approx_distinct = countApproxDistinct

    def treeAggregate(
        self,
        zeroValue,
        seqOp,
        combOp,
        depth: int = 2,
    ):
        if depth < 1:
            raise ValueError(f"Depth cannot be smaller than 1 but got {depth}.")
        if self.getNumPartitions() == 0:
            return zeroValue() if callable(zeroValue) else copy.deepcopy(zeroValue)

        def aggregate_partition(iterator: Iterable[Any]) -> Iterable[Any]:
            acc = zeroValue() if callable(zeroValue) else copy.deepcopy(zeroValue)
            for obj in iterator:
                acc = seqOp(acc, obj)
            yield acc

        partially = self.map_partitions(aggregate_partition, preserves_partitioning=True)
        num_partitions = partially.getNumPartitions()
        scale = max(int(math.ceil(pow(num_partitions, 1.0 / depth))), 2)
        while num_partitions > scale + num_partitions / scale:
            num_partitions = num_partitions / scale
            cur = int(num_partitions)

            def map_partition(index: int, iterator: Iterable[Any]) -> Iterable[tuple[int, Any]]:
                for obj in iterator:
                    yield (index % cur, obj)

            partially = (
                partially.map_partitions_with_index(map_partition, preserves_partitioning=False)
                .reduce_by_key(combOp, cur)
                .map(lambda kv: kv[1])
            )
            num_partitions = partially.getNumPartitions()
        return partially.reduce(combOp)

    tree_aggregate = treeAggregate

    def repartition_and_sort_within_partitions(
        self,
        num_partitions: int,
        partition_func,
        ascending: bool = True,
        keyfunc: Callable[[Any], Any] | None = None,
    ) -> RDD:
        key_fn = keyfunc or (lambda x: x)

        def sort_partition(iterator: Iterable[Any]) -> Iterable[Any]:
            return iter(
                sorted(
                    iterator,
                    key=lambda kv: key_fn(kv[0]),
                    reverse=not ascending,
                )
            )

        return self.partition_by(num_partitions, partition_func).map_partitions(
            sort_partition,
            preserves_partitioning=True,
        )

    repartitionAndSortWithinPartitions = repartition_and_sort_within_partitions

    def to_local_iterator(self, prefetch_partitions: bool = False) -> Iterator[Any]:
        self._materialize()
        total = self._num_partitions
        cache: dict[int, list[Any]] = {}
        lock = threading.Lock()

        def prefetch(pid: int) -> None:
            if pid >= total:
                return
            data = self._collect_partition(pid)
            with lock:
                cache[pid] = data

        for pid in range(total):
            if prefetch_partitions and pid + 1 < total:
                threading.Thread(target=prefetch, args=(pid + 1,), daemon=True).start()
            with lock:
                part = cache.pop(pid, None)
            if part is None:
                part = self._collect_partition(pid)
            yield from part

    toLocalIterator = to_local_iterator

    def cartesian(self, other: RDD) -> RDD:
        left = self.collect()
        right = other.collect()
        return self._require_context().parallelize(
            [(a, b) for a in left for b in right],
            self._num_partitions,
        )

    def zip(self, other: RDD) -> RDD:
        if self._num_partitions != other._num_partitions:
            raise ValueError(
                "Cannot zip RDDs with different number of partitions: "
                f"{self._num_partitions} != {other._num_partitions}"
            )
        return _ZipRDD(self, other, self._require_context())

    def aggregate(self, zeroValue, seqOp, combOp):
        def apply_seq(acc, value):
            try:
                return seqOp(acc, value)
            except StopIteration as exc:
                raise RuntimeError(
                    "Caught StopIteration thrown from user's code; failing the task"
                ) from exc

        def apply_comb(a, b):
            try:
                return combOp(a, b)
            except StopIteration as exc:
                raise RuntimeError(
                    "Caught StopIteration thrown from user's code; failing the task"
                ) from exc

        partials: list[Any] = []
        for part in self._collect_partitions():
            acc = zeroValue() if callable(zeroValue) else copy.deepcopy(zeroValue)
            for value in part:
                acc = apply_seq(acc, value)
            partials.append(acc)
        if not partials:
            return zeroValue() if callable(zeroValue) else copy.deepcopy(zeroValue)
        acc = partials[0]
        for partial in partials[1:]:
            acc = apply_comb(acc, partial)
        return acc

    def aggregate_by_key(self, zeroValue, seqOp, combOp, num_partitions: int = 0):
        parts = num_partitions or self._num_partitions

        def merge_values(kv):
            key, values = kv
            acc = zeroValue() if callable(zeroValue) else copy.deepcopy(zeroValue)
            for value in values:
                acc = seqOp(acc, value)
            return (key, acc)

        return self.group_by_key(parts).map(merge_values)

    aggregateByKey = aggregate_by_key

    def fold(self, zeroValue, op):
        return self.aggregate(zeroValue, op, op)

    def fold_by_key(self, zeroValue, op, num_partitions: int = 0):
        return self.aggregate_by_key(zeroValue, op, op, num_partitions)

    foldByKey = fold_by_key

    def cache(self) -> RDD:
        return self

    def id(self) -> int:
        return self._logical_id

    def _require_context(self) -> "SparkContext":
        if self._context is None:
            raise RuntimeError("RDD has no SparkContext for client-side operation")
        return self._context

    @property
    def num_partitions(self) -> int:
        return self._num_partitions

    def getNumPartitions(self) -> int:
        return self._num_partitions

    def __repr__(self) -> str:
        pending = len(self._pending)
        suffix = f", pending_ops={pending}" if pending else ""
        return f"RDD(num_partitions={self.num_partitions}{suffix})"


class _PipeRDD(RDD):
    def __init__(
        self,
        parent: RDD,
        command: str,
        env: dict[str, str],
        checkCode: bool,
        context: SparkContext,
    ):
        self._pipe_parent = parent
        self._pipe_command = command
        self._pipe_env = env
        self._pipe_check_code = checkCode
        super().__init__(
            parent._client,
            parent._rdd_id,
            parent._num_partitions,
            pending=list(parent._pending),
            native=parent._native,
            native_kind=parent._native_kind,
            line_meta=parent._line_meta,
            context=context,
            text_file_path=parent._text_file_path,
            text_file_parts=parent._text_file_parts,
        )

    def _materialize_pipe(self) -> None:
        if self._pipe_parent is None:
            return
        piped = self._pipe_parent._pipe_impl(
            self._pipe_command, self._pipe_env, self._pipe_check_code
        )
        self._rdd_id = piped._rdd_id
        self._num_partitions = piped._num_partitions
        self._pending = list(piped._pending)
        self._native = piped._native
        self._native_kind = piped._native_kind
        self._pipe_parent = None

    def _resolve_pending(self, action: Action, action_func=None) -> Any:
        self._materialize_pipe()
        return super()._resolve_pending(action, action_func)

    def _run(self, action: Action, action_func=None) -> Any:
        if self._pipe_parent is not None:
            self._materialize_pipe()
        return super()._run(action, action_func)


class _ZipRDD(RDD):
    def __init__(self, left: RDD, right: RDD, context: SparkContext):
        self._zip_left = left
        self._zip_right = right
        super().__init__(
            left._client,
            left._rdd_id,
            left._num_partitions,
            pending=list(left._pending),
            native=left._native,
            native_kind=left._native_kind,
            line_meta=left._line_meta,
            context=context,
            text_file_path=left._text_file_path,
            text_file_parts=left._text_file_parts,
        )

    def _materialize_zipped(self) -> None:
        left_parts = self._zip_left._collect_partitions()
        right_parts = self._zip_right._collect_partitions()
        if len(left_parts) != len(right_parts):
            raise RuntimeError("Cannot zip RDDs with different number of partitions")
        out: list[list[Any]] = []
        for left, right in zip(left_parts, right_parts):
            if len(left) != len(right):
                raise ValueError(
                    "Cannot zip RDDs with different number of elements in partition"
                )
            out.append(list(zip(left, right)))
        rdd_id, parts = self._client.parallelize_partitions(out)
        self._rdd_id = rdd_id
        self._num_partitions = parts
        self._pending = []
        self._native = False
        self._zip_left = None
        self._zip_right = None

    def _collect_partitions(self) -> list[list[Any]]:
        if self._zip_left is not None:
            self._materialize_zipped()
        return super()._collect_partitions()

    def _resolve_pending(self, action: Action, action_func=None) -> Any:
        self._materialize_zipped()
        return super()._resolve_pending(action, action_func)

    def _run(self, action: Action, action_func=None) -> Any:
        if self._zip_left is not None:
            self._materialize_zipped()
        return super()._run(action, action_func)


class SparkContext:
    """Entry point for PyVega. Talks to a Vega C++ session server."""

    def __init__(self, num_workers: int = 0, remote: Optional[str] = None):
        self._server_proc = None
        self._num_workers = num_workers
        self._pickled_broadcast_vars = BroadcastPickleRegistry()

        if remote:
            host, port = parse_remote(remote)
        else:
            self._server_proc, host, port = spawn_server(num_workers)

        self._client = SessionClient(host, port, num_workers)
        self._num_workers = self._client.num_workers

    def range(self, start: int, end: int, num_partitions: int = 0) -> RDD:
        """Create an RDD of integers [start, end) generated on the server."""
        if num_partitions == 0:
            num_partitions = self._num_workers
        rdd_id, parts = self._client.parallelize_range(start, end, num_partitions)
        return RDD(
            self._client, rdd_id, parts, native=True, native_kind="int64", context=self
        )

    def float_range(self, start: int, end: int, num_partitions: int = 0) -> RDD:
        """Create a native float64 RDD [start, end) generated on the server."""
        if num_partitions == 0:
            num_partitions = self._num_workers
        rdd_id, parts = self._client.parallelize_float_range(start, end, num_partitions)
        return RDD(
            self._client, rdd_id, parts, native=True, native_kind="float64", context=self
        )

    def string_lines(
        self,
        start: int,
        end: int,
        num_keys: int,
        num_tokens: int = 97,
        num_partitions: int = 0,
    ) -> RDD:
        """Create a native string RDD of synthetic lines generated on the server."""
        if num_partitions == 0:
            num_partitions = self._num_workers
        rdd_id, parts = self._client.parallelize_string_lines(
            start, end, num_keys, num_tokens, num_partitions
        )
        return RDD(
            self._client,
            rdd_id,
            parts,
            native=True,
            native_kind="str",
            line_meta=(num_keys, num_tokens),
            context=self,
        )

    @staticmethod
    def _detect_native_kind(data_list: list) -> tuple[bool, NativeKind]:
        if not data_list:
            return False, "int64"
        if all(isinstance(x, int) and not isinstance(x, bool) for x in data_list):
            return True, "int64"
        if all(
            isinstance(x, (int, float)) and not isinstance(x, bool) for x in data_list
        ):
            return True, "float64"
        if all(isinstance(x, str) for x in data_list):
            return True, "str"
        return False, "int64"

    def parallelize(self, data, num_partitions: int = 0) -> RDD:
        if num_partitions == 0:
            num_partitions = self._num_workers
        data_list = list(data)
        native, native_kind = self._detect_native_kind(data_list)
        rdd_id, parts = self._client.parallelize(data_list, num_partitions)
        return RDD(
            self._client, rdd_id, parts, native=native, native_kind=native_kind, context=self
        )

    def empty_rdd(self) -> RDD:
        return self.parallelize([], 1)

    emptyRDD = empty_rdd

    def text_file(self, path: str, min_partitions: int = 0) -> RDD:
        parts = min_partitions or self._num_workers
        if os.path.isdir(path):
            return self.parallelize(read_text_file(path), parts)
        return RDD(
            self._client,
            0,
            parts,
            context=self,
            text_file_path=path,
            text_file_parts=parts,
        )

    textFile = text_file

    def pickle_file(self, path: str, min_partitions: int = 0) -> RDD:
        data = read_pickle_file(path)
        parts = min_partitions or self._num_workers
        return self.parallelize(data, parts)

    pickleFile = pickle_file

    def broadcast(self, value: Any) -> Broadcast[Any]:
        return Broadcast(self, value, self._pickled_broadcast_vars)

    def union(self, rdds: list[RDD]) -> RDD:
        data: list[Any] = []
        for rdd in rdds:
            data.extend(rdd.collect())
        return self.parallelize(data, max(1, len(rdds)))

    @property
    def num_workers(self) -> int:
        return self._num_workers

    def stop(self) -> None:
        self._client.close()
        if self._server_proc is not None:
            self._server_proc.terminate()
            try:
                self._server_proc.wait(timeout=5)
            except Exception:
                self._server_proc.kill()
            self._server_proc = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stop()

    def __repr__(self) -> str:
        return f"SparkContext(num_workers={self.num_workers})"
