"""SparkContext backed by PyVega."""

from __future__ import annotations

import threading
from typing import Any, Iterable

from pyvega import SparkContext as _PyVegaSparkContext
from pyvega.broadcast import BroadcastPickleRegistry

from pyspark.conf import SparkConf
from pyspark.rdd import RDD


class SparkContext:
    _active_context: "SparkContext | None" = None

    def __init__(
        self,
        master: str | None = None,
        appName: str | None = None,
        sparkHome: str | None = None,
        conf: Any = None,
        gateway: Any = None,
        jsc: Any = None,
        profiler_cls: Any = None,
        udf_profiler_cls: Any = None,
        memory_profiler_cls: Any = None,
    ):
        del master, appName, sparkHome, gateway, jsc
        del profiler_cls, udf_profiler_cls, memory_profiler_cls
        self._conf = conf if conf is not None else SparkConf()
        self._ctx = _PyVegaSparkContext()
        self._pickled_broadcast_vars = self._ctx._pickled_broadcast_vars
        self.serializer = None
        self._cancelled_groups: set[str] = set()
        self._job_group = threading.local()
        SparkContext._active_context = self

    @classmethod
    def getOrCreate(cls, conf: Any = None) -> "SparkContext":
        if cls._active_context is None:
            cls._active_context = cls(conf=conf)
        return cls._active_context

    @classmethod
    def getActive(cls) -> "SparkContext | None":
        return cls._active_context

    def parallelize(self, c: Iterable[Any], numSlices: int | None = None) -> RDD:
        parts = numSlices or self._ctx.num_workers
        return RDD(self._ctx.parallelize(list(c), parts), self)

    def range(
        self,
        start: int,
        end: int | None = None,
        step: int = 1,
        numSlices: int | None = None,
    ) -> RDD:
        if end is None:
            start, end = 0, start
        if step != 1:
            data = list(range(start, end, step))
            return self.parallelize(data, numSlices)
        parts = numSlices or self._ctx.num_workers
        return RDD(self._ctx.range(start, end, parts), self)

    def emptyRDD(self) -> RDD:
        return RDD(self._ctx.emptyRDD(), self)

    def union(self, rdds: list[RDD]) -> RDD:
        return RDD(self._ctx.union([r._inner for r in rdds]), self)

    def textFile(self, path: str, minPartitions: int | None = None) -> RDD:
        parts = minPartitions or self._ctx.num_workers
        return RDD(self._ctx.textFile(path, parts), self)

    def pickleFile(self, path: str, minPartitions: int | None = None) -> RDD:
        parts = minPartitions or self._ctx.num_workers
        return RDD(self._ctx.pickleFile(path, parts), self)

    def broadcast(self, value: Any) -> Any:
        return self._ctx.broadcast(value)

    def setJobGroup(self, group: str, description: str, interruptOnCancel: bool = True) -> None:
        del description, interruptOnCancel
        self._job_group.group = group

    def clearJobGroup(self) -> None:
        if hasattr(self._job_group, "group"):
            del self._job_group.group

    def cancelJobGroup(self, group: str) -> None:
        self._cancelled_groups.add(group)

    def isJobGroupCancelled(self, group: str) -> bool:
        return group in self._cancelled_groups

    def stop(self) -> None:
        self._ctx.stop()
        if SparkContext._active_context is self:
            SparkContext._active_context = None

    def setLogLevel(self, level: str) -> None:
        del level

    @property
    def defaultParallelism(self) -> int:
        return self._ctx.num_workers

    def __enter__(self) -> "SparkContext":
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
