"""The PyVega RDD: a lazy, server-side resilient distributed dataset.

An :class:`RDD` is a handle to a server-side RDD (``_rdd_id``) plus a list of
pending narrow stages that are sent to the server in one round trip when the
RDD is next used. Every operation -- including shuffles, unions, joins,
sorting, sampling and partition-level functions -- executes inside the
server; only action results travel back to the client.
"""

from __future__ import annotations

import contextlib
import itertools
import math
import operator
import random
import sys
import time
from collections import defaultdict
from functools import partial
from functools import reduce as _reduce
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator, Optional

from pyvega import _funcs as F
from pyvega._client import PendingStage, SessionClient
from pyvega._native import NativeKind, try_compile
from pyvega._protocol import Action, StageOp
from pyvega._sampler import (
    RDDRangeSampler,
    RDDSampler,
    RDDStratifiedSampler,
    compute_fraction_for_sample_size,
)
from pyvega._stats import (
    hll_estimate,
    hll_merge,
    hll_precision,
    histogram_values,
    portable_hash,
)
from pyvega.statcounter import StatCounter
from pyvega.storagelevel import StorageLevel

if TYPE_CHECKING:  # pragma: no cover
    from pyvega.context import SparkContext

__all__ = ["RDD", "BoundedFloat"]

_next_rdd_id = 1


class BoundedFloat(float):
    """A float with an attached confidence interval (approximate results)."""

    def __new__(cls, mean: float, confidence: float, low: float, high: float):
        obj = float.__new__(cls, mean)
        obj.confidence = confidence
        obj.low = low
        obj.high = high
        return obj


def _check_stop(f: Callable[..., Any]) -> Callable[..., Any]:
    return F.fail_on_stopiteration(f)


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
        name: Optional[str] = None,
        lineage: Optional[str] = None,
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
        self._name = name
        self._lineage = lineage or "ParallelCollectionRDD"
        self._storage_level = StorageLevel.NONE
        self._uncached_id = 0
        self._checkpointed = False
        if rdd_id_counter is None:
            rdd_id_counter = _next_rdd_id
            _next_rdd_id += 1
        self._logical_id = rdd_id_counter

    # ----- server handle lifetime --------------------------------------------

    @property
    def _rdd_id(self) -> int:
        return self.__dict__.get("_server_rdd_id", 0)

    @_rdd_id.setter
    def _rdd_id(self, value: int) -> None:
        old = self.__dict__.get("_server_rdd_id", 0)
        if value == old:
            return
        self._client.retain(value)
        self.__dict__["_server_rdd_id"] = value
        if old:
            self._client.release(old)

    def __del__(self) -> None:
        try:
            client = self.__dict__.get("_client")
            if client is not None:
                client.release(self.__dict__.get("_server_rdd_id", 0))
                uncached = self.__dict__.get("_uncached_id", 0)
                if uncached:
                    client.release(uncached)
        except Exception:  # noqa: BLE001 - never raise from a finalizer
            pass

    # ----- internal construction ---------------------------------------------

    def _derive(
        self, stage: PendingStage, native: bool, native_kind: NativeKind, label: str = ""
    ) -> "RDD":
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
            lineage=self._child_lineage(label or stage.op.name.title().replace("_", "")),
        )

    def _child_lineage(self, label: str, *others: "RDD") -> str:
        parents = [self._lineage] + [o._lineage for o in others]
        body = "\n".join("  " + line for p in parents for line in p.splitlines())
        return f"{label}RDD\n{body}"

    def _wrap(self, ack: tuple[int, int], label: str, *others: "RDD") -> "RDD":
        rdd_id, parts = ack
        return RDD(
            self._client,
            rdd_id,
            parts,
            context=self._context,
            lineage=self._child_lineage(label, *others),
        )

    def _stage(self, op: StageOp, func, num_partitions: int = 0, label: str = "") -> "RDD":
        native = self._native
        native_kind = self._native_kind
        native_op = None
        native_params: tuple[int | float | str, ...] = ()
        if native and op in (StageOp.MAP, StageOp.FILTER, StageOp.FLAT_MAP, StageOp.REDUCE_BY_KEY):
            compiled = None
            if hasattr(func, "__code__"):
                try:
                    compiled = try_compile(op, func, native_kind)
                except Exception:  # noqa: BLE001 - unsupported shape: run in Python
                    compiled = None
            if compiled is not None:
                native_op = compiled.native_op
                native_params = compiled.params
            else:
                native = False
        else:
            native = False
        return self._derive(
            PendingStage(op, func, num_partitions, native_op, native_params),
            native,
            native_kind,
            label,
        )

    def _part_stage(self, func: Callable[[int, Iterator[Any]], Iterable[Any]], label: str) -> "RDD":
        """Append a server-side mapPartitionsWithIndex stage (lazy)."""
        return self._stage(StageOp.MAP_PARTITIONS, func, label=label)

    @staticmethod
    def _normalize_func(func: Callable[..., Any]) -> Callable[..., Any]:
        if func is range:
            return lambda x: list(range(x))
        if isinstance(func, type) or func in (str, int, float, bool, list, tuple, set, dict):
            return lambda x, f=func: f(x)
        return func

    def _materialize(self) -> None:
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

    def _sid(self) -> int:
        """Materialize pending stages and return the server-side RDD id."""
        self._materialize()
        return self._rdd_id

    def _collect_partitions(self) -> list[list[Any]]:
        return self._client.collect_partitions(self._sid())

    def _collect_partition(self, partition: int) -> list[Any]:
        return self._client.collect_partition(self._sid(), partition)

    def _run_job(self, partitions: list[int], counts: bool = False) -> list[Any]:
        return self._client.run_job(self._sid(), partitions, counts)

    def _from_partitions(
        self,
        partitions: list[list[Any]],
        native: bool = False,
        native_kind: NativeKind = "int64",
    ) -> "RDD":
        ctx = self._require_context()
        rdd_id, parts = self._client.parallelize_partitions(partitions)
        return RDD(self._client, rdd_id, parts, native=native, native_kind=native_kind, context=ctx)

    def _native_stats(self):
        """(count, sum, min, max, m2) computed natively, or None."""
        if not self._native or self._client.protocol_version < 2:
            return None
        return self._client.native_stats(self._sid())

    # ----- basic transformations -----------------------------------------

    def map(self, func, preservesPartitioning: bool = False) -> "RDD":
        return self._stage(StageOp.MAP, self._normalize_func(func), label="PythonMapped")

    def filter(self, func) -> "RDD":
        return self._stage(StageOp.FILTER, func, label="PythonFiltered")

    def flat_map(self, func, preservesPartitioning: bool = False) -> "RDD":
        return self._stage(StageOp.FLAT_MAP, self._normalize_func(func), label="PythonFlatMapped")

    flatMap = flat_map

    def map_partitions_with_index(self, func, preserves_partitioning: bool = False) -> "RDD":
        """Apply ``func(partition_index, iterator)`` to every partition, server-side."""
        return self._part_stage(func, "MapPartitions")

    mapPartitionsWithIndex = map_partitions_with_index

    def map_partitions(self, func, preserves_partitioning: bool = False) -> "RDD":
        return self._part_stage(partial(F.apply_to_iterator, func), "MapPartitions")

    mapPartitions = map_partitions

    def mapPartitionsWithSplit(self, f, preservesPartitioning: bool = False) -> "RDD":
        return self.map_partitions_with_index(f, preservesPartitioning)

    def glom(self) -> "RDD":
        return self._part_stage(F.glom, "Glom")

    def keys(self) -> "RDD":
        return self.map(F.first_of_pair)

    def values(self) -> "RDD":
        return self.map(F.second_of_pair)

    def keyBy(self, f) -> "RDD":
        return self.map(partial(F.key_by, f))

    key_by = keyBy

    def mapValues(self, f) -> "RDD":
        return self.map(partial(F.map_values, f))

    map_values = mapValues

    def flatMapValues(self, f) -> "RDD":
        return self.flat_map(partial(F.flat_map_values, f))

    flat_map_values = flatMapValues

    # ----- shuffles ------------------------------------------------------------

    def _default_parts(self, num_partitions: Optional[int]) -> int:
        return num_partitions or self.getNumPartitions() or 1

    def reduce_by_key(self, func, num_partitions: Optional[int] = None, partitionFunc=None) -> "RDD":
        parts = self._default_parts(num_partitions)
        return self._stage(StageOp.REDUCE_BY_KEY, func, parts, label="ShuffledReduceByKey")

    reduceByKey = reduce_by_key

    def group_by_key(self, num_partitions: Optional[int] = None, partitionFunc=None) -> "RDD":
        parts = self._default_parts(num_partitions)
        return self._stage(StageOp.GROUP_BY_KEY, None, parts, label="ShuffledGroupByKey")

    groupByKey = group_by_key

    def combineByKey(
        self,
        createCombiner,
        mergeValue,
        mergeCombiners,
        numPartitions: Optional[int] = None,
        partitionFunc=portable_hash,
    ) -> "RDD":
        """Generic per-key aggregation with map-side combining.

        Values are combined within each partition first (``createCombiner`` /
        ``mergeValue``) so only one combiner per key per partition is shuffled.
        """
        parts = self._default_parts(numPartitions)
        local = self._part_stage(
            partial(F.combine_locally, createCombiner, mergeValue), "CombineLocally"
        )
        return local.reduce_by_key(mergeCombiners, parts)

    combine_by_key = combineByKey

    def aggregate_by_key(
        self, zeroValue, seqOp, combOp, num_partitions: Optional[int] = None, partitionFunc=None
    ) -> "RDD":
        def create(v, _z=zeroValue, _s=seqOp):
            return _s(F.make_zero(_z), v)

        return self.combineByKey(create, seqOp, combOp, num_partitions)

    aggregateByKey = aggregate_by_key

    def fold_by_key(self, zeroValue, op, num_partitions: Optional[int] = None, partitionFunc=None):
        return self.aggregate_by_key(zeroValue, op, op, num_partitions)

    foldByKey = fold_by_key

    def groupBy(self, f, numPartitions: Optional[int] = None, partitionFunc=None) -> "RDD":
        return self.map(partial(F.key_by, f)).group_by_key(numPartitions)

    group_by = groupBy

    def distinct(self, num_partitions: Optional[int] = None) -> "RDD":
        parts = self._default_parts(num_partitions)
        return (
            self.map(F.with_none)
            .reduce_by_key(F.keep_first, parts)
            .map(F.first_of_pair)
        )

    def partition_by(self, num_partitions: Optional[int], partition_func=portable_hash) -> "RDD":
        """Hash- (or custom-) partition a (key, value) RDD into N partitions."""
        parts = self._default_parts(num_partitions)
        func = None if partition_func is portable_hash else partition_func
        return self._wrap(self._client.partition_by(self._sid(), parts, func), "ShuffledPartitionBy")

    partitionBy = partition_by

    def repartition(self, num_partitions: int) -> "RDD":
        return self.coalesce(num_partitions, shuffle=True)

    def coalesce(self, num_partitions: int, shuffle: bool = False) -> "RDD":
        if num_partitions <= 0:
            raise ValueError("Number of partitions must be positive.")
        sid = self._sid()
        if shuffle:
            ack = self._client.partition_by(sid, num_partitions, None, round_robin=True)
            return self._wrap(ack, "ShuffledRepartition")
        current = self._num_partitions
        if num_partitions >= current:
            return self
        groups = [
            list(range(i * current // num_partitions, (i + 1) * current // num_partitions))
            for i in range(num_partitions)
        ]
        return self._wrap(self._client.coalesce(sid, groups), "Coalesced")

    def sortByKey(
        self,
        ascending: bool = True,
        numPartitions: Optional[int] = None,
        keyfunc: Optional[Callable[[Any], Any]] = None,
    ) -> "RDD":
        key_fn = keyfunc or F.identity
        num_partitions = self._default_parts(numPartitions)
        sorter = partial(F.sort_partition, key_fn, ascending)
        if num_partitions == 1:
            return self.coalesce(1)._part_stage(sorter, "Sorted")

        rdd_size = self.count()
        if rdd_size == 0:
            return self

        max_sample_size = num_partitions * 20.0
        fraction = min(max_sample_size / max(rdd_size, 1), 1.0)
        samples = sorted(
            (key_fn(kv[0]) for kv in self.sample(False, fraction, 1).collect()),
        )
        if not samples:
            samples = sorted(key_fn(kv[0]) for kv in self.take(int(max_sample_size)))
        bounds = [
            samples[int(len(samples) * (i + 1) / num_partitions)]
            for i in range(num_partitions - 1)
        ]
        partitioner = partial(F.range_partition, bounds, key_fn, ascending, num_partitions)
        return self.partition_by(num_partitions, partitioner)._part_stage(sorter, "Sorted")

    sort_by_key = sortByKey

    def sort_by(self, keyfunc, ascending: bool = True, num_partitions: Optional[int] = None) -> "RDD":
        return self.keyBy(keyfunc).sortByKey(ascending, num_partitions).values()

    sortBy = sort_by

    def repartition_and_sort_within_partitions(
        self,
        num_partitions: Optional[int] = None,
        partition_func=portable_hash,
        ascending: bool = True,
        keyfunc: Optional[Callable[[Any], Any]] = None,
    ) -> "RDD":
        key_fn = keyfunc or F.identity
        return self.partition_by(num_partitions, partition_func)._part_stage(
            partial(F.sort_partition, key_fn, ascending), "SortedWithinPartitions"
        )

    repartitionAndSortWithinPartitions = repartition_and_sort_within_partitions

    # ----- multi-RDD operations ---------------------------------------------

    def union(self, other: "RDD") -> "RDD":
        return self._wrap(self._client.union([self._sid(), other._sid()]), "Union", other)

    def __add__(self, other: "RDD") -> "RDD":
        if not isinstance(other, RDD):
            raise TypeError
        return self.union(other)

    def cartesian(self, other: "RDD") -> "RDD":
        return self._wrap(self._client.cartesian(self._sid(), other._sid()), "Cartesian", other)

    def zip(self, other: "RDD") -> "RDD":
        if self.getNumPartitions() != other.getNumPartitions():
            raise ValueError(
                "Can only zip with RDD which has the same number of partitions: "
                f"{self.getNumPartitions()} != {other.getNumPartitions()}"
            )
        ack = self._client.zip_partitions(self._sid(), other._sid(), F.zip_iterators)
        return self._wrap(ack, "ZippedPartitions", other)

    def zipWithIndex(self) -> "RDD":
        n = self.getNumPartitions()
        starts = [0]
        if n > 1:
            counts = self._run_job(list(range(n - 1)), counts=True)
            for c in counts:
                starts.append(starts[-1] + c)
        return self._part_stage(partial(F.zip_with_index, starts), "ZippedWithIndex")

    zip_with_index = zipWithIndex

    def zipWithUniqueId(self) -> "RDD":
        return self._part_stage(
            partial(F.zip_with_unique_id, self.getNumPartitions()), "ZippedWithUniqueId"
        )

    zip_with_unique_id = zipWithUniqueId

    def cogroup(self, other: "RDD", numPartitions: Optional[int] = None) -> "RDD":
        return self.groupWith(other, numPartitions=numPartitions)

    def groupWith(self, other: "RDD", *others: "RDD", numPartitions: Optional[int] = None) -> "RDD":
        rdds = [self, other, *others]
        parts = numPartitions or max(r.getNumPartitions() for r in rdds)
        tagged = [r.map(partial(F.tag_values, i)) for i, r in enumerate(rdds)]
        unioned = self._require_context().union(tagged)
        return unioned.group_by_key(parts).map(partial(F.split_tagged, len(rdds)))

    group_with = groupWith

    def _join(self, other: "RDD", how: str, numPartitions: Optional[int]) -> "RDD":
        return self.cogroup(other, numPartitions).flat_map(partial(F.join_values, how))

    def join(self, other: "RDD", numPartitions: Optional[int] = None) -> "RDD":
        return self._join(other, "inner", numPartitions)

    def leftOuterJoin(self, other: "RDD", numPartitions: Optional[int] = None) -> "RDD":
        return self._join(other, "left", numPartitions)

    def rightOuterJoin(self, other: "RDD", numPartitions: Optional[int] = None) -> "RDD":
        return self._join(other, "right", numPartitions)

    def fullOuterJoin(self, other: "RDD", numPartitions: Optional[int] = None) -> "RDD":
        return self._join(other, "full", numPartitions)

    left_outer_join = leftOuterJoin
    right_outer_join = rightOuterJoin
    full_outer_join = fullOuterJoin

    def subtractByKey(self, other: "RDD", numPartitions: Optional[int] = None) -> "RDD":
        return self.cogroup(other, numPartitions).filter(F.subtract_filter).flat_map(F.flatten_left)

    subtract_by_key = subtractByKey

    def subtract(self, other: "RDD", numPartitions: Optional[int] = None) -> "RDD":
        return self.map(F.with_none).subtractByKey(other.map(F.with_none), numPartitions).keys()

    def intersection(self, other: "RDD") -> "RDD":
        return (
            self.map(F.with_none)
            .cogroup(other.map(F.with_none))
            .filter(F.intersection_filter)
            .keys()
        )

    # ----- sampling ------------------------------------------------------------

    def sample(self, withReplacement: bool, fraction: float, seed: Optional[int] = None) -> "RDD":
        if fraction < 0:
            raise ValueError("Fraction must be nonnegative.")
        sampler = RDDSampler(withReplacement, fraction, seed)
        return self._part_stage(sampler.func, "PartitionwiseSampled")

    def sampleByKey(self, withReplacement: bool, fractions: dict, seed: Optional[int] = None) -> "RDD":
        for fraction in fractions.values():
            if fraction < 0:
                raise ValueError("Negative fraction value: %s" % fraction)
        sampler = RDDStratifiedSampler(withReplacement, fractions, seed)
        return self._part_stage(sampler.func, "PartitionwiseSampled")

    sample_by_key = sampleByKey

    def randomSplit(self, weights: list[float], seed: Optional[int] = None) -> list["RDD"]:
        if not all(w >= 0 for w in weights):
            raise ValueError("Weights must be nonnegative")
        s = float(sum(weights))
        if s <= 0:
            raise ValueError("Sum of weights must be positive")
        cweights = [0.0]
        for w in weights:
            cweights.append(cweights[-1] + w / s)
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        return [
            self._part_stage(RDDRangeSampler(lb, ub, seed).func, "PartitionwiseSampled")
            for lb, ub in zip(cweights, cweights[1:])
        ]

    random_split = randomSplit

    def takeSample(self, withReplacement: bool, num: int, seed: Optional[int] = None) -> list[Any]:
        num_st_dev = 10.0
        max_sample_size = sys.maxsize - int(num_st_dev * math.sqrt(sys.maxsize))
        if num < 0:
            raise ValueError("Sample size cannot be negative.")
        if num > max_sample_size:
            raise ValueError("Sample size cannot be greater than %d." % max_sample_size)
        if num == 0 or self.getNumPartitions() == 0:
            return []
        initial_count = self.count()
        if initial_count == 0:
            return []
        rand = random.Random(seed)
        if not withReplacement and num >= initial_count:
            samples = self.collect()
            rand.shuffle(samples)
            return samples
        fraction = compute_fraction_for_sample_size(num, initial_count, withReplacement)
        samples = self.sample(withReplacement, fraction, seed).collect()
        while len(samples) < num:
            seed = rand.randint(0, sys.maxsize)
            samples = self.sample(withReplacement, fraction, seed).collect()
        rand.shuffle(samples)
        return samples[0:num]

    take_sample = takeSample

    # ----- actions ---------------------------------------------------------------

    def _resolve_pending(self, action: Action, action_func=None) -> Any:
        fused = self._client._fuse_stages(self._pending, self._line_meta)
        has_shuffle = any(
            stage.op in (StageOp.REDUCE_BY_KEY, StageOp.GROUP_BY_KEY) for stage in fused
        )
        if action == Action.COUNT and not has_shuffle and self._client.stages_are_native(fused):
            # Fully native narrow pipeline: count without materializing the
            # chain. The pending stages are kept, so later actions still see
            # the transformations.
            return self._client.execute(
                self._rdd_id,
                self._num_partitions,
                self._pending,
                action,
                action_func,
                self._line_meta,
            )
        self._materialize()
        return self._run(action, action_func)

    def _run(self, action: Action, action_func=None) -> Any:
        if self._pending:
            return self._resolve_pending(action, action_func)
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

    def collectAsMap(self) -> dict:
        return dict(self.collect())

    collect_as_map = collectAsMap

    def count(self) -> int:
        return self._run(Action.COUNT)

    def countByKey(self) -> dict:
        return self.map(F.first_of_pair).countByValue()

    count_by_key = countByKey

    def countByValue(self) -> dict:
        counts = (
            self._part_stage(F.count_values_partition, "CountByValue")
            .reduce_by_key(F.add)
            .collect()
        )
        out: defaultdict = defaultdict(int)
        out.update(counts)
        return out

    count_by_value = countByValue

    def reduce(self, func):
        f = _check_stop(func)
        vals = self._part_stage(partial(F.partial_reduce, f), "Reduce").collect()
        if not vals:
            raise ValueError("Can not reduce() empty RDD")
        return _reduce(f, vals)

    def treeReduce(self, f, depth: int = 2):
        if depth < 1:
            raise ValueError("Depth cannot be smaller than 1 but got %d." % depth)
        zero = (None, True)

        def op(x, y):
            if x[1]:
                return y
            if y[1]:
                return x
            return (f(x[0], y[0]), False)

        reduced = self.map(lambda x: (x, False)).treeAggregate(zero, op, op, depth)
        if reduced[1]:
            raise ValueError("Cannot reduce empty RDD.")
        return reduced[0]

    tree_reduce = treeReduce

    def fold(self, zeroValue, op):
        f = _check_stop(op)
        vals = self._part_stage(partial(F.partial_fold, zeroValue, f), "Fold").collect()
        return _reduce(f, vals, F.make_zero(zeroValue))

    def aggregate(self, zeroValue, seqOp, combOp):
        seq = _check_stop(seqOp)
        comb = _check_stop(combOp)
        vals = self._part_stage(partial(F.partial_fold, zeroValue, seq), "Aggregate").collect()
        return _reduce(comb, vals, F.make_zero(zeroValue))

    def treeAggregate(self, zeroValue, seqOp, combOp, depth: int = 2):
        if depth < 1:
            raise ValueError(f"Depth cannot be smaller than 1 but got {depth}.")
        if self.getNumPartitions() == 0:
            return F.make_zero(zeroValue)
        seq = _check_stop(seqOp)
        comb = _check_stop(combOp)
        partially = self._part_stage(partial(F.partial_fold, zeroValue, seq), "Aggregate")
        num_partitions = partially.getNumPartitions()
        scale = max(int(math.ceil(pow(num_partitions, 1.0 / depth))), 2)
        while num_partitions > scale + num_partitions / scale:
            num_partitions = num_partitions / scale
            cur = int(num_partitions)

            def map_partition(index: int, iterator: Iterable[Any], _cur=cur):
                for obj in iterator:
                    yield (index % _cur, obj)

            partially = (
                partially.map_partitions_with_index(map_partition)
                .reduce_by_key(comb, cur)
                .map(F.second_of_pair)
            )
            num_partitions = partially.getNumPartitions()
        return _reduce(comb, partially.collect(), F.make_zero(zeroValue))

    tree_aggregate = treeAggregate

    def foreach(self, func) -> None:
        self._part_stage(partial(F.consume_foreach, _check_stop(func)), "Foreach").count()

    def foreachPartition(self, func) -> None:
        self._part_stage(partial(F.consume_foreach_partition, func), "ForeachPartition").count()

    foreach_partition = foreachPartition

    def take(self, num: int) -> list:
        """First ``num`` elements, scanning as few partitions as possible."""
        items: list[Any] = []
        total_parts = self.getNumPartitions()
        parts_scanned = 0
        while len(items) < num and parts_scanned < total_parts:
            num_parts_to_try = 1
            if parts_scanned > 0:
                if len(items) == 0:
                    num_parts_to_try = parts_scanned * 4
                else:
                    num_parts_to_try = int(1.5 * num * parts_scanned / len(items)) - parts_scanned
                    num_parts_to_try = min(max(num_parts_to_try, 1), parts_scanned * 4)
            left = num - len(items)
            limited = self._part_stage(partial(F.islice_partition, left), "Take")
            pids = list(range(parts_scanned, min(parts_scanned + num_parts_to_try, total_parts)))
            for part in limited._run_job(pids):
                items.extend(part)
            parts_scanned += num_parts_to_try
        return items[:num]

    def first(self):
        values = self.take(1)
        if not values:
            raise ValueError("RDD is empty")
        return values[0]

    def isEmpty(self) -> bool:
        return self.getNumPartitions() == 0 or len(self.take(1)) == 0

    is_empty = isEmpty

    def top(self, num: int, key=None) -> list:
        parts = self._part_stage(partial(F.top_partition, num, key), "Top").collect()
        return _reduce(lambda a, b: __import__("heapq").nlargest(num, a + b, key=key), parts, [])

    def takeOrdered(self, num: int, key=None) -> list:
        import heapq

        parts = self._part_stage(partial(F.bottom_partition, num, key), "TakeOrdered").collect()
        return _reduce(lambda a, b: heapq.nsmallest(num, a + b, key=key), parts, [])

    take_ordered = takeOrdered

    def lookup(self, key) -> list:
        return self.filter(partial(F.lookup_filter, key)).values().collect()

    def reduceByKeyLocally(self, func) -> dict:
        f = _check_stop(func)
        merged: dict[Any, Any] = {}
        for key, value in self.reduce_by_key(f).collect():
            if key in merged:
                merged[key] = f(merged[key], value)
            else:
                merged[key] = value
        return merged

    reduce_by_key_locally = reduceByKeyLocally

    def to_local_iterator(self, prefetch_partitions: bool = False) -> Iterator[Any]:
        import threading

        sid = self._sid()
        total = self._num_partitions
        cache: dict[int, list[Any]] = {}
        errors: dict[int, BaseException] = {}
        lock = threading.Lock()

        def prefetch(pid: int) -> None:
            try:
                data = self._client.collect_partition(sid, pid)
            except BaseException as exc:  # noqa: BLE001 - re-raised on the consumer
                with lock:
                    errors[pid] = exc
                return
            with lock:
                cache[pid] = data

        threads: dict[int, threading.Thread] = {}
        for pid in range(total):
            if prefetch_partitions and pid + 1 < total:
                t = threading.Thread(target=prefetch, args=(pid + 1,), daemon=True)
                threads[pid + 1] = t
                t.start()
            if pid in threads:
                threads.pop(pid).join()
            with lock:
                part = cache.pop(pid, None)
                err = errors.pop(pid, None)
            if err is not None:
                raise err
            if part is None:
                part = self._client.collect_partition(sid, pid)
            yield from part

    toLocalIterator = to_local_iterator

    # ----- numeric actions ---------------------------------------------------

    def stats(self) -> StatCounter:
        native = self._native_stats()
        if native is not None:
            n, total, mn, mx, m2 = native
            return StatCounter.from_moments(n, (total / n) if n else 0.0, m2, mn, mx)
        parts = self._part_stage(F.stat_partition, "Stats").collect()
        return _reduce(lambda a, b: a.mergeStats(b), parts, StatCounter())

    def sum(self):
        native = self._native_stats()
        if native is not None:
            return native[1]
        return _reduce(operator.add, self._part_stage(F.partial_sum, "Sum").collect(), 0)

    def mean(self) -> float:
        return self.stats().mean()

    def variance(self) -> float:
        return self.stats().variance()

    def stdev(self) -> float:
        return self.stats().stdev()

    def sampleStdev(self) -> float:
        return self.stats().sampleStdev()

    def sampleVariance(self) -> float:
        return self.stats().sampleVariance()

    sample_stdev = sampleStdev
    sample_variance = sampleVariance

    def max(self, key=None):
        if key is None:
            native = self._native_stats()
            if native is not None:
                if not native[0]:
                    raise ValueError("Can not reduce() empty RDD")
                return native[3]
            return self.reduce(max)
        return self.reduce(lambda a, b: max(a, b, key=key))

    def min(self, key=None):
        if key is None:
            native = self._native_stats()
            if native is not None:
                if not native[0]:
                    raise ValueError("Can not reduce() empty RDD")
                return native[2]
            return self.reduce(min)
        return self.reduce(lambda a, b: min(a, b, key=key))

    def histogram(self, buckets):
        if isinstance(buckets, int):
            if buckets < 1:
                raise ValueError("number of buckets must be >= 1")
            bounds = self._part_stage(F.minmax_partition, "MinMax").collect()
            if not bounds:
                raise ValueError("can not generate buckets from empty RDD")
            minv = min(b[0] for b in bounds)
            maxv = max(b[1] for b in bounds)
            if minv == maxv or buckets == 1:
                comparable = self.filter(
                    lambda x: x is not None and not (isinstance(x, float) and x != x)
                ).count()
                return [minv, maxv], [comparable]
            bucket_list, _ = histogram_values([minv, maxv], buckets)
        elif isinstance(buckets, (list, tuple)):
            if len(buckets) < 2:
                raise ValueError("buckets should have more than one value")
            histogram_values([], list(buckets))  # validates the bucket list
            bucket_list = list(buckets)
        else:
            raise TypeError("buckets should be a list or tuple or number(int or long)")
        parts = self._part_stage(partial(F.histogram_partition, bucket_list), "Histogram").collect()
        counts = [0] * (len(bucket_list) - 1)
        for part in parts:
            counts = [a + b for a, b in zip(counts, part)]
        return bucket_list, counts

    def countApproxDistinct(self, relativeSD: float = 0.05) -> int:
        """Approximate distinct count using HyperLogLog (computed server-side)."""
        p = hll_precision(relativeSD)
        regs = self._part_stage(partial(F.hll_partition, p), "HyperLogLog").collect()
        if not regs:
            return 0
        return hll_estimate(_reduce(hll_merge, regs))

    count_approx_distinct = countApproxDistinct

    def _batched_partition_results(self, rdd: "RDD", counts: bool, timeout_ms: int):
        """Run partitions in waves until done or the timeout passes."""
        deadline = time.monotonic() + max(0, timeout_ms) / 1000.0
        total = rdd.getNumPartitions()
        wave = max(1, self._client.num_workers)
        results: list[Any] = []
        done = 0
        while done < total:
            pids = list(range(done, min(total, done + wave)))
            results.extend(rdd._run_job(pids, counts=counts))
            done += len(pids)
            if time.monotonic() >= deadline:
                break
        return results, done, total

    def countApprox(self, timeout: int, confidence: float = 0.95) -> int:
        """Count within ``timeout`` ms; extrapolates if not all partitions finished."""
        counts, done, total = self._batched_partition_results(self, True, timeout)
        if done == total or done == 0:
            return int(sum(counts)) if done else 0
        return int(sum(counts) * total / done)

    count_approx = countApprox

    def meanApprox(self, timeout: int, confidence: float = 0.95) -> BoundedFloat:
        stats_rdd = self._part_stage(F.stat_partition, "Stats")
        parts, done, total = self._batched_partition_results(stats_rdd, False, timeout)
        stats = _reduce(lambda a, b: a.mergeStats(b), [p[0] for p in parts if p], StatCounter())
        mean = stats.mean() if stats.count() else float("nan")
        if stats.count() > 1:
            z = _z_score(confidence)
            err = z * stats.sampleStdev() / math.sqrt(stats.count())
        else:
            err = float("inf") if done < total else 0.0
        return BoundedFloat(mean, confidence, mean - err, mean + err)

    mean_approx = meanApprox

    def sumApprox(self, timeout: int, confidence: float = 0.95) -> BoundedFloat:
        stats_rdd = self._part_stage(F.stat_partition, "Stats")
        parts, done, total = self._batched_partition_results(stats_rdd, False, timeout)
        stats = _reduce(lambda a, b: a.mergeStats(b), [p[0] for p in parts if p], StatCounter())
        scale = total / done if done else 0.0
        estimate = stats.sum() * scale
        return BoundedFloat(estimate, confidence, estimate, estimate)

    sum_approx = sumApprox

    # ----- persistence ---------------------------------------------------------

    def persist(self, storageLevel: StorageLevel = StorageLevel.MEMORY_ONLY) -> "RDD":
        """Keep computed partitions in server memory (lazy, on first action)."""
        if self.is_cached:
            self._storage_level = storageLevel
            return self
        sid = self._sid()
        cached_id, _ = self._client.persist(sid)
        self._client.retain(sid)
        self._uncached_id = sid
        self._rdd_id = cached_id
        self._storage_level = storageLevel
        return self

    def cache(self) -> "RDD":
        return self.persist(StorageLevel.MEMORY_ONLY)

    def unpersist(self, blocking: bool = False) -> "RDD":
        if self._uncached_id:
            original = self._uncached_id
            self._uncached_id = 0
            self._rdd_id = original
            self._client.release(original)
        self._storage_level = StorageLevel.NONE
        return self

    @property
    def is_cached(self) -> bool:
        return bool(self._uncached_id)

    def getStorageLevel(self) -> StorageLevel:
        return self._storage_level

    def checkpoint(self) -> None:
        """Truncate lineage by materializing into server memory on next action."""
        self.persist()
        self._checkpointed = True

    def localCheckpoint(self) -> None:
        self.checkpoint()

    def isCheckpointed(self) -> bool:
        return self._checkpointed

    def isLocallyCheckpointed(self) -> bool:
        return self._checkpointed

    def getCheckpointFile(self) -> Optional[str]:
        return None

    @contextlib.contextmanager
    def persisted(self, storageLevel: StorageLevel = StorageLevel.MEMORY_ONLY):
        """Context manager: persist for the duration of a ``with`` block.

        >>> with rdd.persisted() as cached:
        ...     a = cached.count(); b = cached.sum()
        """
        was_cached = self.is_cached
        self.persist(storageLevel)
        try:
            yield self
        finally:
            if not was_cached:
                self.unpersist()

    # ----- I/O ---------------------------------------------------------------

    def pipe(self, command: str, env: Optional[dict[str, str]] = None, checkCode: bool = False) -> "RDD":
        return self._part_stage(
            partial(F.pipe_partition, command, dict(env or {}), checkCode), "Piped"
        )

    def _write(self, path: str, writer) -> None:
        from pyvega import _io

        path = str(path)
        self._client.run_python(_io.prepare_output_dir, path)
        self._part_stage(writer, "Write").collect()
        self._client.run_python(_io.mark_success, path)

    def save_as_text_file(self, path: str, compressionCodecClass: Optional[str] = None) -> None:
        from pyvega import _io

        codec = None
        if compressionCodecClass:
            codec = compressionCodecClass.rsplit(".", 1)[-1] if "." in compressionCodecClass else compressionCodecClass
        self._write(path, _io.text_writer(str(path), codec))

    saveAsTextFile = save_as_text_file

    def save_as_pickle_file(self, path: str, batchSize: int = 10) -> None:
        from pyvega import _io

        self._write(path, _io.pickle_writer(str(path), max(1, batchSize or 1024)))

    saveAsPickleFile = save_as_pickle_file

    def save_as_jsonl(self, path: str) -> None:
        """Write one JSON document per line (``part-NNNNN.jsonl`` files)."""
        from pyvega import _io

        self._write(path, _io.jsonl_writer(str(path)))

    saveAsJsonLines = save_as_jsonl

    # ----- Vega extras (not in Spark's RDD API) ----------------------------

    def map_batches(self, func: Callable[[list[Any]], Iterable[Any]], batch_size: int = 1024) -> "RDD":
        """Vectorized UDF: ``func`` receives lists of up to ``batch_size`` elements.

        Amortizes per-element Python call overhead (and lets ``func`` use
        NumPy/pandas on a whole batch). ``func`` returns an iterable of
        output elements (any length).
        """
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        return self._part_stage(partial(F.batches, func, batch_size), "MapBatches")

    def distinct_by(self, keyfunc: Callable[[Any], Any], num_partitions: Optional[int] = None) -> "RDD":
        """Keep one element per ``keyfunc(element)`` (first seen per partition)."""
        return (
            self._part_stage(partial(F.distinct_by_partition, keyfunc), "DistinctBy")
            .reduce_by_key(F.keep_first, num_partitions)
            .values()
        )

    def top_by_key(self, num: int, key: Optional[Callable[[Any], Any]] = None,
                   num_partitions: Optional[int] = None) -> "RDD":
        """Per-key top-``num`` values (descending), combined map-side.

        Returns an RDD of ``(key, [values...])``.
        """
        return self._part_stage(
            partial(F.top_by_key_partition, num, key), "TopByKey"
        ).reduce_by_key(partial(F.merge_top, num, key), num_partitions)

    def quantiles(
        self,
        probabilities: list[float],
        relative_error: float = 0.0,
        key: Optional[Callable[[Any], Any]] = None,
        seed: int = 17,
    ) -> list[Any]:
        """Quantiles of a numeric (or ``key``-mapped) RDD.

        ``relative_error == 0`` computes exact quantiles with a distributed
        sort; otherwise a per-partition reservoir sample of
        ``~1/relative_error**2`` elements gives approximate results in a
        single pass.
        """
        for p in probabilities:
            if not 0.0 <= p <= 1.0:
                raise ValueError("probabilities must be in [0, 1]")
        if relative_error < 0:
            raise ValueError("relative_error must be >= 0")
        if relative_error > 0:
            size = int(min(1e6, max(100, 1.0 / (relative_error * relative_error))))
            parts = self._part_stage(
                partial(F.reservoir_partition, size, seed, key), "Reservoir"
            ).collect()
            total = sum(n for n, _ in parts)
            if total == 0:
                return [None for _ in probabilities]
            # Weight each partition's sample by its share of the data.
            weighted: list[tuple[Any, float]] = []
            for n, sample in parts:
                if sample:
                    w = n / len(sample)
                    weighted.extend((v, w) for v in sample)
            weighted.sort(key=lambda t: t[0])
            cumulative = list(itertools.accumulate(w for _, w in weighted))
            out = []
            for p in probabilities:
                target = p * total
                idx = min(len(weighted) - 1, _bisect_left(cumulative, target))
                out.append(weighted[idx][0])
            return out
        values = self if key is None else self.map(key)
        n = values.count()
        if n == 0:
            return [None for _ in probabilities]
        ranks = sorted({min(n - 1, int(math.floor(p * (n - 1) + 0.5))) for p in probabilities})
        wanted = frozenset(ranks)
        indexed = values.sortBy(F.identity).zipWithIndex()
        found = dict(
            (i, v) for v, i in indexed.filter(lambda vi, _w=wanted: vi[1] in _w).collect()
        )
        return [found[min(n - 1, int(math.floor(p * (n - 1) + 0.5)))] for p in probabilities]

    def heavy_hitters(self, k: int, exact_counts: bool = True) -> list[tuple[Any, int]]:
        """The ``k`` most frequent elements, via mergeable Misra-Gries summaries.

        One pass finds candidates in O(k) memory per partition; with
        ``exact_counts`` a second pass counts the candidates exactly.
        Returns ``[(element, count), ...]`` sorted by count descending.
        """
        if k < 1:
            raise ValueError("k must be >= 1")
        capacity = max(k * 4, 16)
        summaries = self._part_stage(
            partial(F.misra_gries_partition, capacity), "MisraGries"
        ).collect()
        merged = _reduce(partial(F.merge_misra_gries, capacity), summaries, {})
        if not merged:
            return []
        if exact_counts:
            candidates = frozenset(merged)
            merged = dict(self.filter(partial(F.member_filter, candidates)).countByValue())
        return sorted(merged.items(), key=lambda kv: kv[1], reverse=True)[:k]

    def sliding(self, window: int, step: int = 1) -> "RDD":
        """Sliding windows over the RDD's order (like Spark MLlib's ``sliding``).

        Returns an RDD of lists ``[x_i, ..., x_{i+window-1}]`` for every
        ``i`` that is a multiple of ``step``; incomplete trailing windows are
        dropped.
        """
        if window < 1 or step < 1:
            raise ValueError("window and step must be >= 1")
        return (
            self.zipWithIndex()
            .flat_map(partial(F.window_starts, window, step))
            .group_by_key()
            .flat_map(partial(F.assemble_window, window))
            .sortByKey()
            .values()
        )

    def explain(self) -> str:
        """Describe how this RDD will execute: pending stages, native fusion
        (which lambdas were compiled to C++), shuffles, and caching."""
        lines = [f"RDD[{self.id()}] {self._name or ''}".rstrip()]
        fused = self._client._fuse_stages(self._pending, self._line_meta)
        if self._rdd_id and self._client.protocol_version >= 2:
            kind, parts, cached = self._client.describe(self._rdd_id)
            extra = f", cached partitions {cached}/{parts}" if cached >= 0 else ""
            lines.append(f"  source: server RDD #{self._rdd_id} ({kind}, {parts} partitions{extra})")
        if not fused:
            lines.append("  pending stages: none (materialized)")
        for i, stage in enumerate(fused, 1):
            if stage.native_op is not None:
                how = f"native C++ {stage.native_op.name.lower()}{tuple(stage.native_params)}"
            else:
                how = "Python UDF"
            shuffle = " [shuffle]" if stage.op in (StageOp.REDUCE_BY_KEY, StageOp.GROUP_BY_KEY) else ""
            lines.append(f"  {i}. {stage.op.name.lower()}{shuffle}: {how}")
        lines.append(f"  storage: {self._storage_level}")
        return "\n".join(lines)

    # ----- metadata --------------------------------------------------------

    def cache_info(self) -> dict[str, Any]:
        kind, parts, cached = self._client.describe(self._sid())
        return {"kind": kind, "partitions": parts, "cached_partitions": max(cached, 0)}

    def setName(self, name: str) -> "RDD":
        self._name = name
        return self

    def name(self) -> Optional[str]:
        return self._name

    def toDebugString(self) -> bytes:
        header = f"({self.getNumPartitions()}) "
        return (header + self._lineage).encode("utf-8")

    def id(self) -> int:
        return self._logical_id

    @property
    def context(self) -> "SparkContext":
        return self._require_context()

    @property
    def partitioner(self):
        return None

    def _require_context(self) -> "SparkContext":
        if self._context is None:
            raise RuntimeError("RDD has no SparkContext for client-side operation")
        return self._context

    def _logical_partitions(self) -> int:
        parts = self._num_partitions
        for stage in self._pending:
            if stage.op in (StageOp.REDUCE_BY_KEY, StageOp.GROUP_BY_KEY) and stage.num_partitions:
                parts = stage.num_partitions
        return parts

    @property
    def num_partitions(self) -> int:
        return self._logical_partitions()

    def getNumPartitions(self) -> int:
        return self._logical_partitions()

    def withResources(self, profile) -> "RDD":
        self._resource_profile = profile
        return self

    def getResourceProfile(self):
        return getattr(self, "_resource_profile", None)

    def __repr__(self) -> str:
        pending = len(self._pending)
        suffix = f", pending_ops={pending}" if pending else ""
        name = f"{self._name}, " if self._name else ""
        return f"RDD({name}num_partitions={self.num_partitions}{suffix})"


def _bisect_left(seq: list[float], value: float) -> int:
    import bisect

    return bisect.bisect_left(seq, value)


def _z_score(confidence: float) -> float:
    from statistics import NormalDist

    return NormalDist().inv_cdf(0.5 + confidence / 2.0)
