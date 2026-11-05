"""SparkContext: the PyVega entry point."""

from __future__ import annotations

import collections
import threading
import time
from functools import partial
from typing import Any, Callable, Iterable, Optional

from pyvega import _io
from pyvega._client import SessionClient, parse_remote, spawn_server
from pyvega._native import NativeKind
from pyvega.accumulators import (
    COMPLEX_ACCUMULATOR_PARAM,
    FLOAT_ACCUMULATOR_PARAM,
    INT_ACCUMULATOR_PARAM,
    Accumulator,
    AccumulatorParam,
    _drain,
    _forget,
)
from pyvega.broadcast import Broadcast, BroadcastPickleRegistry
from pyvega.rdd import RDD

__all__ = ["SparkContext", "JobRecord"]


JobRecord = collections.namedtuple("JobRecord", ["kind", "seconds", "finished_at"])


def _read_split(use_unicode: bool, split):
    return _io.read_text_split(split, use_unicode)


def _read_whole(use_unicode: bool, path: str):
    return _io.read_whole_file(path, use_unicode)


def _read_jsonl_line(line: str):
    import json

    return json.loads(line)


class SparkContext:
    """Entry point for PyVega. Talks to a Vega C++ session server.

    ``SparkContext(4)`` spawns a private local server with 4 worker threads;
    ``SparkContext(remote="host:port")`` connects to a running server.
    """

    _active: Optional["SparkContext"] = None
    _lock = threading.Lock()

    def __init__(
        self,
        num_workers: int = 0,
        remote: Optional[str] = None,
        appName: str = "pyvega",
        conf: Optional[dict[str, Any]] = None,
    ):
        self._server_proc = None
        self._num_workers = num_workers
        self._pickled_broadcast_vars = BroadcastPickleRegistry()
        self._accumulators: dict[str, Accumulator[Any]] = {}
        self._app_name = appName
        self._conf: dict[str, Any] = dict(conf or {})
        self._local_properties = threading.local()
        self._cancelled_groups: set[str] = set()
        self._start_time = int(time.time() * 1000)
        self.job_history: collections.deque[JobRecord] = collections.deque(maxlen=256)

        if remote:
            host, port = parse_remote(remote)
            self._master = f"sc://{host}:{port}"
        else:
            self._server_proc, host, port = spawn_server(num_workers)
            self._master = f"local[{num_workers or '*'}]"

        self._client = SessionClient(host, port, num_workers)
        self._client.on_job_done = self._after_job
        self._num_workers = self._client.num_workers
        with SparkContext._lock:
            SparkContext._active = self

    # ----- lifecycle / metadata ---------------------------------------------

    @classmethod
    def getOrCreate(cls, num_workers: int = 0) -> "SparkContext":
        with cls._lock:
            active = cls._active
        return active if active is not None else cls(num_workers)

    @property
    def version(self) -> str:
        from pyvega import __version__

        return __version__

    @property
    def appName(self) -> str:
        return self._app_name

    @property
    def master(self) -> str:
        return self._master

    @property
    def applicationId(self) -> str:
        return f"vega-{self._start_time}"

    @property
    def startTime(self) -> int:
        return self._start_time

    @property
    def defaultParallelism(self) -> int:
        return self._num_workers

    @property
    def defaultMinPartitions(self) -> int:
        return min(self.defaultParallelism, 2)

    @property
    def num_workers(self) -> int:
        return self._num_workers

    def sparkUser(self) -> str:
        import getpass

        return getpass.getuser()

    def getConf(self) -> dict[str, Any]:
        return dict(self._conf)

    def setLogLevel(self, logLevel: str) -> None:
        self._conf["vega.logLevel"] = logLevel

    def setLocalProperty(self, key: str, value: Optional[str]) -> None:
        props = self._local_props()
        if value is None:
            props.pop(key, None)
        else:
            props[key] = value

    def getLocalProperty(self, key: str) -> Optional[str]:
        return self._local_props().get(key)

    def _local_props(self) -> dict[str, str]:
        if not hasattr(self._local_properties, "props"):
            self._local_properties.props = {}
        return self._local_properties.props

    def setJobGroup(self, groupId: str, description: str, interruptOnCancel: bool = False) -> None:
        self.setLocalProperty("spark.jobGroup.id", groupId)
        self.setLocalProperty("spark.job.description", description)

    def setJobDescription(self, value: str) -> None:
        self.setLocalProperty("spark.job.description", value)

    def clearJobGroup(self) -> None:
        self.setLocalProperty("spark.jobGroup.id", None)

    def cancelJobGroup(self, groupId: str) -> None:
        self._cancelled_groups.add(groupId)

    def cancelAllJobs(self) -> None:
        """Best-effort: jobs already running on the server finish normally."""
        group = self.getLocalProperty("spark.jobGroup.id")
        if group:
            self._cancelled_groups.add(group)

    def stop(self) -> None:
        if self._accumulators:
            try:
                self._client.run_python(_forget, list(self._accumulators))
            except Exception:  # noqa: BLE001 - server may already be gone
                pass
        self._client.close()
        if self._server_proc is not None:
            self._server_proc.terminate()
            try:
                self._server_proc.wait(timeout=5)
            except Exception:  # noqa: BLE001
                self._server_proc.kill()
            self._server_proc = None
        with SparkContext._lock:
            if SparkContext._active is self:
                SparkContext._active = None

    def __enter__(self) -> "SparkContext":
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def __repr__(self) -> str:
        return f"SparkContext(num_workers={self.num_workers})"

    # ----- jobs & accumulators -----------------------------------------------

    def _after_job(self) -> None:
        last = self._client.last_job
        if last is not None:
            self.job_history.append(JobRecord(last[0], last[1], time.time()))
        if self._accumulators:
            updates = self._client.run_python(_drain, list(self._accumulators))
            for aid, partial_value in updates.items():
                acc = self._accumulators.get(aid)
                if acc is not None:
                    acc._merge_remote(partial_value)

    def accumulator(self, value: Any, accum_param: Optional[AccumulatorParam[Any]] = None) -> Accumulator[Any]:
        """Create an accumulator; tasks may only ``add`` to it."""
        if accum_param is None:
            if isinstance(value, bool):
                raise TypeError("No default accumulator param for type bool")
            if isinstance(value, int):
                accum_param = INT_ACCUMULATOR_PARAM
            elif isinstance(value, float):
                accum_param = FLOAT_ACCUMULATOR_PARAM
            elif isinstance(value, complex):
                accum_param = COMPLEX_ACCUMULATOR_PARAM
            else:
                raise TypeError("No default accumulator param for type %s" % type(value))
        acc = Accumulator(None, value, accum_param)
        self._accumulators[acc.aid] = acc
        return acc

    def broadcast(self, value: Any) -> Broadcast[Any]:
        return Broadcast(self, value, self._pickled_broadcast_vars)

    def runJob(
        self,
        rdd: RDD,
        partitionFunc: Callable[[Iterable[Any]], Iterable[Any]],
        partitions: Optional[list[int]] = None,
        allowLocal: bool = False,
    ) -> list[Any]:
        """Run ``partitionFunc`` on selected partitions and return the flattened results."""
        if partitions is None:
            partitions = list(range(rdd.getNumPartitions()))
        mapped = rdd.mapPartitions(partitionFunc)
        return [x for part in mapped._run_job(list(partitions)) for x in part]

    def status(self) -> dict[str, Any]:
        """Client-side job metrics: count and wall time of recent jobs."""
        records = list(self.job_history)
        return {
            "jobs": len(records),
            "total_seconds": round(sum(r.seconds for r in records), 6),
            "last": records[-1]._asdict() if records else None,
            "server_protocol": self._client.protocol_version,
            "workers": self._num_workers,
        }

    # ----- RDD creation ------------------------------------------------------

    def _rdd(self, ack: tuple[int, int], native: bool = False, kind: NativeKind = "int64",
             line_meta=None, lineage: str = "ParallelCollectionRDD") -> RDD:
        rdd_id, parts = ack
        return RDD(
            self._client,
            rdd_id,
            parts,
            native=native,
            native_kind=kind,
            line_meta=line_meta,
            context=self,
            lineage=lineage,
        )

    def range(
        self,
        start: int,
        end: Optional[int] = None,
        num_partitions: int = 0,
        *,
        step: int = 1,
        numSlices: Optional[int] = None,
    ) -> RDD:
        """RDD of integers ``range(start, end, step)`` generated on the server.

        The third positional argument is ``num_partitions`` (PyVega's
        historical signature); pass ``step=`` by keyword.
        """
        if end is None:
            start, end = 0, start
        parts = numSlices or num_partitions or self._num_workers
        if step == 1:
            return self._rdd(self._client.parallelize_range(start, end, parts), True, "int64",
                             lineage="RangeRDD")
        if step == 0:
            raise ValueError("step must not be zero")
        length = len(range(start, end, step))
        base = self._rdd(self._client.parallelize_range(0, length, parts), True, "int64",
                         lineage="RangeRDD")
        return base.map(lambda i, _s=start, _t=step: _s + i * _t)

    def float_range(self, start: int, end: int, num_partitions: int = 0) -> RDD:
        """Create a native float64 RDD [start, end) generated on the server."""
        parts = num_partitions or self._num_workers
        return self._rdd(self._client.parallelize_float_range(start, end, parts), True, "float64")

    def string_lines(
        self,
        start: int,
        end: int,
        num_keys: int,
        num_tokens: int = 97,
        num_partitions: int = 0,
    ) -> RDD:
        """Create a native string RDD of synthetic lines generated on the server."""
        parts = num_partitions or self._num_workers
        ack = self._client.parallelize_string_lines(start, end, num_keys, num_tokens, parts)
        return self._rdd(ack, True, "str", line_meta=(num_keys, num_tokens))

    @staticmethod
    def _detect_native_kind(data_list: list) -> tuple[bool, NativeKind]:
        if not data_list:
            return False, "int64"
        if all(type(x) is int for x in data_list):
            return True, "int64"
        if all(type(x) in (int, float) for x in data_list):
            return True, "float64"
        if all(type(x) is str for x in data_list):
            return True, "str"
        return False, "int64"

    def parallelize(self, data: Iterable[Any], num_partitions: int = 0,
                    numSlices: Optional[int] = None) -> RDD:
        parts = numSlices or num_partitions or self._num_workers
        if isinstance(data, range) and data.step == 1 and len(data) > 0:
            # Generated on the server: nothing is pickled or transferred.
            return self._rdd(self._client.parallelize_range(data.start, data.stop, parts), True,
                             "int64")
        data_list = data if isinstance(data, list) else list(data)
        native, native_kind = self._detect_native_kind(data_list)
        return self._rdd(self._client.parallelize(data_list, parts), native, native_kind)

    def empty_rdd(self) -> RDD:
        return self.parallelize([], 1)

    emptyRDD = empty_rdd

    def union(self, rdds: list[RDD]) -> RDD:
        rdds = list(rdds)
        if not rdds:
            return self.empty_rdd()
        if len(rdds) == 1:
            return rdds[0]
        ack = self._client.union([r._sid() for r in rdds])
        return self._rdd(ack, lineage=rdds[0]._child_lineage("Union", *rdds[1:]))

    # ----- file input (read on the server, like Spark executors) ----------

    def text_file(self, path: str, min_partitions: Optional[int] = None,
                  use_unicode: bool = True, minPartitions: Optional[int] = None) -> RDD:
        """Lines of text file(s): a file, directory, glob or comma-separated list.

        Files are split into byte ranges read in parallel by server tasks;
        ``.gz``/``.bz2``/``.xz`` files are decompressed transparently.
        """
        parts = minPartitions or min_partitions or self.defaultMinPartitions
        splits = self._client.run_python(_io.plan_text_splits, str(path), parts)
        source = self._rdd(self._client.parallelize(splits, max(1, len(splits))),
                           lineage=f"HadoopRDD[{path}]")
        return source.flat_map(partial(_read_split, use_unicode))

    textFile = text_file

    def wholeTextFiles(self, path: str, minPartitions: Optional[int] = None,
                       use_unicode: bool = True) -> RDD:
        """``(path, content)`` pairs, one per file."""
        files = self._client.run_python(_io.list_input_files, str(path))
        parts = max(1, min(len(files), minPartitions or self._num_workers))
        return self.parallelize(files, parts).map(partial(_read_whole, use_unicode))

    whole_text_files = wholeTextFiles

    def binaryFiles(self, path: str, minPartitions: Optional[int] = None) -> RDD:
        files = self._client.run_python(_io.list_input_files, str(path))
        parts = max(1, min(len(files), minPartitions or self._num_workers))
        return self.parallelize(files, parts).map(_io.read_binary_file)

    binary_files = binaryFiles

    def pickle_file(self, path: str, min_partitions: Optional[int] = None,
                    minPartitions: Optional[int] = None) -> RDD:
        files = self._client.run_python(_io.list_input_files, str(path))
        parts = max(1, minPartitions or min_partitions or len(files) or 1)
        return self.parallelize(files, parts).flat_map(_io.read_pickle_part)

    pickleFile = pickle_file

    def jsonl_file(self, path: str, min_partitions: Optional[int] = None) -> RDD:
        """Parse JSON-lines file(s) into Python objects (Vega extra)."""
        return self.text_file(path, min_partitions).filter(bool).map(_read_jsonl_line)

    jsonLinesFile = jsonl_file
