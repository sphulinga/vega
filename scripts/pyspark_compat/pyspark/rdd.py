"""RDD wrapper with PySpark method names over PyVega."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Iterator

from py4j.protocol import Py4JJavaError
from pyvega import RDD as _PyVegaRDD

if TYPE_CHECKING:
    from pyspark.context import SparkContext


def _translate_spark_error(exc: Exception) -> Exception:
    message = str(exc)
    if (
        "StopIteration" in message
        or "Caught StopIteration" in message
        or "Pipe command" in message
    ):
        return Py4JJavaError(message)
    return exc


class RDD:
    def __init__(
        self,
        inner: _PyVegaRDD,
        sc: "SparkContext",
        resource_profile=None,
        external_group: bool = False,
    ):
        self._inner = inner
        self.ctx = sc
        self._resource_profile = resource_profile
        self._external_group = external_group

    def _wrap(self, inner: _PyVegaRDD, external_group: bool | None = None) -> "RDD":
        if external_group is None:
            external_group = self._external_group
        return RDD(inner, self.ctx, self._resource_profile, external_group)

    def _run_action(self, func: Callable[[], Any]) -> Any:
        try:
            return func()
        except Exception as exc:
            raise _translate_spark_error(exc) from exc

    def _collect_respecting_job_group(self, group: str) -> list[Any]:
        if self.ctx.isJobGroupCancelled(group):
            raise RuntimeError(f"Job group {group} cancelled")
        inner = self._inner
        inner._materialize()
        out: list[Any] = []
        for pid in range(inner._num_partitions):
            if self.ctx.isJobGroupCancelled(group):
                raise RuntimeError(f"Job group {group} cancelled")
            out.extend(inner._collect_partition(pid))
        return out

    def map(self, func: Callable[..., Any]) -> "RDD":
        return self._wrap(self._inner.map(func))

    def filter(self, func: Callable[..., Any]) -> "RDD":
        return self._wrap(self._inner.filter(func))

    def flatMap(self, func: Callable[..., Any]) -> "RDD":
        return self._wrap(self._inner.flatMap(func))

    flat_map = flatMap

    def mapPartitions(self, func: Callable[..., Any], preservesPartitioning: bool = False) -> "RDD":
        return self._wrap(self._inner.mapPartitions(func, preservesPartitioning))

    def mapPartitionsWithIndex(
        self, func: Callable[..., Any], preservesPartitioning: bool = False
    ) -> "RDD":
        return self._wrap(self._inner.mapPartitionsWithIndex(func, preservesPartitioning))

    def reduceByKey(
        self,
        func: Callable[..., Any],
        numPartitions: int | None = None,
    ) -> "RDD":
        parts = numPartitions or self._inner.num_partitions
        return self._wrap(self._inner.reduceByKey(func, parts))

    reduce_by_key = reduceByKey

    def groupByKey(self, numPartitions: int | None = None) -> "RDD":
        parts = numPartitions or self._inner.num_partitions
        external = bool(self.ctx._conf.get("spark.python.worker.memory"))
        return self._wrap(self._inner.groupByKey(parts), external_group=external)

    group_by_key = groupByKey

    def reduceByKeyLocally(self, func: Callable[..., Any]) -> dict[Any, Any]:
        merged: dict[Any, Any] = {}
        for key, value in self.collect():
            try:
                if key in merged:
                    merged[key] = func(merged[key], value)
                else:
                    merged[key] = value
            except StopIteration as exc:
                raise RuntimeError(
                    "Caught StopIteration thrown from user's code; failing the task"
                ) from exc
        return merged

    reduce_by_key_locally = reduceByKeyLocally

    def mapValues(self, func: Callable[..., Any]) -> "RDD":
        if self._external_group:
            rows = [(key, func(values)) for key, values in self.collect()]
            return self.ctx.parallelize(rows, self.getNumPartitions())
        return self._wrap(
            self._inner.map(lambda kv: (kv[0], func(kv[1]))),
            external_group=False,
        )

    map_values = mapValues

    def values(self) -> "RDD":
        return self._wrap(self._inner.map(lambda kv: kv[1]), external_group=False)

    def aggregateByKey(self, zeroValue, seqFunc, combFunc, numPartitions: int | None = None):
        parts = numPartitions or self._inner.num_partitions
        return self._wrap(
            self._inner.aggregateByKey(zeroValue, seqFunc, combFunc, parts)
        )

    aggregate_by_key = aggregateByKey

    def aggregate(self, zeroValue, seqOp, combOp):
        return self._run_action(lambda: self._inner.aggregate(zeroValue, seqOp, combOp))

    def treeAggregate(self, zeroValue, seqOp, combOp, depth: int = 2):
        return self._run_action(
            lambda: self._inner.treeAggregate(zeroValue, seqOp, combOp, depth)
        )

    tree_aggregate = treeAggregate

    def fold(self, zeroValue, op):
        return self._run_action(lambda: self._inner.fold(zeroValue, op))

    def foldByKey(self, zeroValue, op, numPartitions: int | None = None):
        parts = numPartitions or self._inner.num_partitions
        return self._wrap(self._inner.foldByKey(zeroValue, op, parts))

    fold_by_key = foldByKey

    def collect(self) -> list[Any]:
        group = getattr(self.ctx._job_group, "group", None)

        def run() -> list[Any]:
            if group is not None:
                rows = self._collect_respecting_job_group(group)
            else:
                rows = list(self._inner.collect())
            if not self._external_group:
                return rows
            from pyspark.resultiterable import ResultIterable
            from pyspark import shuffle

            wrapped: list[Any] = []
            for key, values in rows:
                ext = shuffle.ExternalListOfList([list(values)])
                wrapped.append((key, ResultIterable(ext)))
            return wrapped

        return self._run_action(run)

    def count(self) -> int:
        return int(self._run_action(self._inner.count))

    def reduce(self, func: Callable[..., Any]) -> Any:
        return self._run_action(lambda: self._inner.reduce(func))

    def foreach(self, func: Callable[..., Any]) -> None:
        self._run_action(lambda: self._inner.foreach(func))

    def sum(self) -> Any:
        return self._run_action(self._inner.sum)

    def first(self) -> Any:
        return self._run_action(self._inner.first)

    def take(self, n: int) -> list[Any]:
        return self._run_action(lambda: self._inner.take(n))

    def isEmpty(self) -> bool:
        return bool(self._run_action(self._inner.isEmpty))

    def id(self) -> int:
        return self._inner.id()

    def getNumPartitions(self) -> int:
        return self._inner.getNumPartitions()

    def toLocalIterator(self, prefetchPartitions: bool = False) -> Iterator[Any]:
        def iterator() -> Iterator[Any]:
            try:
                yield from self._inner.toLocalIterator(prefetchPartitions)
            except Exception as exc:
                raise _translate_spark_error(exc) from exc

        return iterator()

    def collectWithJobGroup(self, jobGroup: str, description: str, interruptOnCancel: bool = True):
        del interruptOnCancel
        self.ctx.setJobGroup(jobGroup, description)
        try:
            return self.collect()
        finally:
            self.ctx.clearJobGroup()

    def repartition(self, numPartitions: int) -> "RDD":
        return self._wrap(self._inner.repartition(numPartitions))

    def coalesce(self, numPartitions: int, shuffle: bool = False) -> "RDD":
        return self._wrap(self._inner.coalesce(numPartitions, shuffle))

    def sortBy(self, keyfunc: Callable[..., Any], ascending: bool = True) -> "RDD":
        return self._wrap(self._inner.sort_by(keyfunc, ascending))

    def sortByKey(
        self,
        ascending: bool = True,
        numPartitions: int | None = None,
        keyfunc: Callable[..., Any] | None = None,
    ) -> "RDD":
        return self._wrap(self._inner.sortByKey(ascending, numPartitions, keyfunc))

    def repartitionAndSortWithinPartitions(
        self,
        numPartitions: int,
        partitionFunc: Callable[..., Any],
        ascending: bool = True,
        keyfunc: Callable[..., Any] | None = None,
    ) -> "RDD":
        return self._wrap(
            self._inner.repartitionAndSortWithinPartitions(
                numPartitions, partitionFunc, ascending, keyfunc
            )
        )

    def zip(self, other: "RDD") -> "RDD":
        return self._wrap(self._inner.zip(other._inner))

    def cartesian(self, other: "RDD") -> "RDD":
        return self._wrap(self._inner.cartesian(other._inner))

    def distinct(self, numPartitions: int | None = None) -> "RDD":
        parts = numPartitions if numPartitions is not None else self.getNumPartitions()
        return self._wrap(self._inner.distinct(parts))

    def sample(
        self,
        withReplacement: bool,
        fraction: float,
        seed: int | None = None,
    ) -> "RDD":
        return self._wrap(self._inner.sample(withReplacement, fraction, seed))

    def takeSample(self, withReplacement: bool, num: int, seed: int | None = None) -> list[Any]:
        return self._inner.takeSample(withReplacement, num, seed)

    def pipe(
        self,
        command: str,
        env: dict[str, str] | None = None,
        checkCode: bool = False,
    ) -> "RDD":
        try:
            return self._wrap(self._inner.pipe(command, env, checkCode))
        except RuntimeError as exc:
            if checkCode:
                raise Py4JJavaError(str(exc)) from exc
            raise

    def glom(self) -> "RDD":
        return self._wrap(self._inner.glom())

    def saveAsTextFile(self, path: str, compressionCodecClass: str | None = None) -> None:
        self._inner.saveAsTextFile(path, compressionCodecClass)

    def saveAsPickleFile(self, path: str) -> None:
        self._inner.saveAsPickleFile(path)

    def histogram(self, buckets):
        return self._inner.histogram(buckets)

    def countApproxDistinct(self, relativeSD: float = 0.05) -> int:
        return self._inner.countApproxDistinct(relativeSD)

    def cache(self) -> "RDD":
        self._inner.cache()
        return self

    def persist(self, storageLevel=None) -> "RDD":
        if storageLevel is None:
            self._inner.persist()
        else:
            self._inner.persist(storageLevel)
        return self

    def unpersist(self, blocking: bool = False) -> "RDD":
        self._inner.unpersist(blocking)
        return self

    def _reserialize(self, serializer) -> "RDD":
        del serializer
        return self

    def withResources(self, profile) -> "RDD":
        return RDD(self._inner, self.ctx, profile, self._external_group)

    def getResourceProfile(self):
        return self._resource_profile

    def __getattr__(self, name: str) -> Any:
        # Delegate everything not wrapped explicitly to the PyVega RDD,
        # re-wrapping RDD results (and lists of RDDs, e.g. randomSplit).
        if name.startswith("__"):
            raise AttributeError(name)
        attr = getattr(self._inner, name)
        if not callable(attr):
            return attr

        def call(*args: Any, **kwargs: Any) -> Any:
            args = tuple(a._inner if isinstance(a, RDD) else a for a in args)
            kwargs = {k: (v._inner if isinstance(v, RDD) else v) for k, v in kwargs.items()}
            result = self._run_action(lambda: attr(*args, **kwargs))
            if isinstance(result, _PyVegaRDD):
                return self._wrap(result, external_group=False)
            if isinstance(result, list) and result and all(
                isinstance(r, _PyVegaRDD) for r in result
            ):
                return [self._wrap(r, external_group=False) for r in result]
            return result

        return call

    def __add__(self, other: "RDD") -> "RDD":
        return self._wrap(self._inner.union(other._inner))

    def __repr__(self) -> str:
        return f"PySparkRDD({self._inner!r})"
