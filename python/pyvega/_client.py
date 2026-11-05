"""TCP client for the Vega C++ session server."""

from __future__ import annotations

import builtins
import os
import pickle
import shutil
import socket
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pyvega._protocol import (
    PROTOCOL_VERSION,
    Action,
    MsgType,
    NativeOp,
    StageOp,
    TransformMode,
    pack_bytes,
    pack_frame,
    pack_native_tail,
    pack_u32,
    pack_u64,
    pack_u8,
    unpack_bytes,
    unpack_u32,
    unpack_u64,
    unpack_u8,
)

try:
    import cloudpickle as _pickle
except ImportError:  # pragma: no cover
    import pickle as _pickle


@dataclass(frozen=True)
class PendingStage:
    op: StageOp
    func: Any = None
    num_partitions: int = 0
    native_op: NativeOp | None = None
    native_params: tuple[int | float | str, ...] = ()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def find_server_binary() -> str:
    env = os.environ.get("VEGA_SERVER")
    if env and os.path.isfile(env):
        return env
    which = shutil.which("vega_server")
    if which:
        return which
    for candidate in (
        _repo_root() / "build-release" / "vega_server",
        _repo_root() / "build" / "vega_server",
    ):
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(
        "vega_server not found. Build with: cmake -S . -B build-release "
        "-DVEGA_BUILD_SERVER=ON && cmake --build build-release --target vega_server"
    )


def _server_command() -> tuple[list[str], dict[str, str]]:
    """Command used to launch a local session server.

    Prefers the compiled ``pyvega._vega_server`` extension shipped in wheels:
    it runs the server inside *this* interpreter build, which guarantees the
    client and server agree on the Python version (required by cloudpickle).
    Falls back to a standalone ``vega_server`` binary (source checkouts).
    """
    env = os.environ.copy()
    force_binary = env.get("VEGA_SERVER")
    if not force_binary:
        try:
            import importlib.util

            if importlib.util.find_spec("pyvega._vega_server") is not None:
                pkg_root = str(Path(__file__).resolve().parents[1])
                env["PYTHONPATH"] = os.pathsep.join(
                    p for p in (pkg_root, env.get("PYTHONPATH", "")) if p
                )
                return [sys.executable, "-m", "pyvega._server_cli", "--embedded"], env
        except (ImportError, ValueError):
            pass

    binary = find_server_binary()
    env["PYTHONHOME"] = sys.base_prefix
    py_lib = os.path.join(
        sys.base_prefix,
        "lib",
        f"python{sys.version_info.major}.{sys.version_info.minor}",
    )
    env["PYTHONPATH"] = os.pathsep.join(
        p
        for p in (
            str(_repo_root() / "python"),
            env.get("PYTHONPATH", ""),
            py_lib,
            os.path.join(
                sys.prefix,
                "lib",
                f"python{sys.version_info.major}.{sys.version_info.minor}",
                "site-packages",
            ),
        )
        if p
    )
    return [binary], env


def spawn_server(num_workers: int = 0) -> tuple[subprocess.Popen, str, int]:
    del num_workers
    cmd, env = _server_command()
    proc = subprocess.Popen(
        cmd + ["--bind", "127.0.0.1", "--port", "0", "--print-listening"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    line = proc.stdout.readline() if proc.stdout else ""
    if not line.startswith("LISTENING"):
        err = proc.stderr.read() if proc.stderr else ""
        proc.kill()
        raise RuntimeError(f"failed to start vega_server: {line!r} {err}")
    port = int(line.strip().split()[1])
    # Drain stderr in the background so a chatty server cannot block on a
    # full pipe.
    if proc.stderr is not None:
        import threading as _threading

        def _drain(stream) -> None:
            for _ in stream:
                pass

        _threading.Thread(target=_drain, args=(proc.stderr,), daemon=True).start()
    return proc, "127.0.0.1", port


def parse_remote(remote: str) -> tuple[str, int]:
    if remote.startswith("sc://"):
        remote = remote[len("sc://") :]
    host, _, port_str = remote.partition(":")
    if not port_str:
        raise ValueError(f"invalid remote endpoint: {remote!r}")
    return host, int(port_str)


def _dumps_data(obj: Any) -> bytes:
    """Pickle data with the fast C pickler, falling back to cloudpickle."""
    try:
        return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:  # noqa: BLE001 - lambdas, local classes, ...
        return _pickle.dumps(obj)


class RemoteError(RuntimeError):
    """An exception raised on the Vega server that has no local equivalent."""


def remote_exception(message: str) -> Exception:
    """Rebuild a server-side Python exception with its original type.

    Server errors arrive as text such as ``"ValueError: bad input"``. When the
    prefix names a builtin exception type, an instance of that type is
    returned so ``except ValueError`` works across the RPC boundary.
    """
    head = message.split("\n", 1)[0]
    name, sep, _ = head.partition(":")
    exc_type = getattr(builtins, name.strip(), None) if sep else None
    if (
        isinstance(exc_type, type)
        and issubclass(exc_type, Exception)
        and exc_type not in (StopIteration,)
    ):
        try:
            return exc_type(message)
        except Exception:  # noqa: BLE001 - exotic constructor signature
            pass
    return RuntimeError(message)


# Message types whose reply means a job ran (used for accumulator updates).
_JOB_MESSAGES = frozenset(
    {
        MsgType.EXECUTE,
        MsgType.COUNT,
        MsgType.COLLECT,
        MsgType.REDUCE,
        MsgType.FOREACH,
        MsgType.COLLECT_PARTITIONS,
        MsgType.COLLECT_PARTITION,
        MsgType.RUN_JOB,
        MsgType.NATIVE_STATS,
    }
)


class SessionClient:
    def __init__(self, host: str, port: int, num_workers: int = 0):
        self._sock = socket.create_connection((host, port))
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._lock = threading.RLock()
        self._refs: dict[int, int] = {}
        self._to_release: list[int] = []
        self._closed = False
        # Called (outside the lock) after every RPC that executed a job.
        self.on_job_done = None
        self.last_job: tuple[str, float] | None = None
        self.protocol_version = 0
        self._workers = self._hello(num_workers)

    @property
    def num_workers(self) -> int:
        return self._workers

    def close(self) -> None:
        self._closed = True
        try:
            self._sock.close()
        except OSError:
            pass

    # ----- server-side handle lifetime ------------------------------------

    def retain(self, rdd_id: int) -> None:
        if rdd_id:
            self._refs[rdd_id] = self._refs.get(rdd_id, 0) + 1

    def release(self, rdd_id: int) -> None:
        """Drop one reference; the server handle is freed at the next RPC."""
        if not rdd_id or self._closed:
            return
        count = self._refs.get(rdd_id, 0) - 1
        if count > 0:
            self._refs[rdd_id] = count
            return
        self._refs.pop(rdd_id, None)
        self._to_release.append(rdd_id)

    def _flush_releases(self) -> None:
        if not self._to_release or self.protocol_version < 2:
            self._to_release.clear()
            return
        ids, self._to_release = self._to_release, []
        body = pack_u32(len(ids)) + b"".join(pack_u64(i) for i in ids)
        self._roundtrip(MsgType.RELEASE, body)

    # ----- framing ---------------------------------------------------------

    def _recv_exact(self, n: int) -> bytes:
        buf = bytearray(n)
        view = memoryview(buf)
        got = 0
        while got < n:
            read = self._sock.recv_into(view[got:], n - got)
            if not read:
                raise ConnectionError("vega_server closed connection")
            got += read
        return bytes(buf)

    def _roundtrip(self, msg_type: MsgType, body: bytes) -> tuple[MsgType, bytes]:
        frame = pack_frame(msg_type, body)
        self._sock.sendall(frame)
        header = self._recv_exact(4)
        (length,) = struct.unpack(">I", header)
        payload = self._recv_exact(length)
        msg = MsgType(payload[0])
        reply_body = payload[1:]
        if msg == MsgType.ERROR:
            err_bytes, _ = unpack_bytes(reply_body)
            raise remote_exception(err_bytes.decode("utf-8", errors="replace"))
        return msg, reply_body

    def _send_recv(self, msg_type: MsgType, body: bytes = b"") -> tuple[MsgType, bytes]:
        is_job = msg_type in _JOB_MESSAGES
        with self._lock:
            if self._to_release:
                self._flush_releases()
            start = time.perf_counter() if is_job else 0.0
            result = self._roundtrip(msg_type, body)
            if is_job:
                self.last_job = (msg_type.name.lower(), time.perf_counter() - start)
        if is_job and self.on_job_done is not None:
            self.on_job_done()
        return result

    def _hello(self, num_workers: int) -> int:
        _, body = self._send_recv(MsgType.HELLO, pack_u32(num_workers))
        workers, off = unpack_u32(body)
        if len(body) >= off + 4:
            self.protocol_version, _ = unpack_u32(body, off)
        return workers

    def require_protocol(self, feature: str) -> None:
        if self.protocol_version < PROTOCOL_VERSION:
            raise RuntimeError(
                f"{feature} requires a vega_server with protocol v{PROTOCOL_VERSION} "
                f"(connected server speaks v{self.protocol_version}); rebuild or "
                "upgrade the server"
            )

    def parallelize_range(self, start: int, end: int, num_partitions: int) -> tuple[int, int]:
        body = pack_u64(start) + pack_u64(end) + pack_u32(num_partitions)
        _, reply = self._send_recv(MsgType.PARALLELIZE_RANGE, body)
        rdd_id, off = unpack_u64(reply)
        parts, _ = unpack_u32(reply, off)
        return rdd_id, parts

    def parallelize_float_range(
        self, start: int, end: int, num_partitions: int
    ) -> tuple[int, int]:
        body = pack_u64(start) + pack_u64(end) + pack_u32(num_partitions)
        _, reply = self._send_recv(MsgType.PARALLELIZE_FLOAT_RANGE, body)
        rdd_id, off = unpack_u64(reply)
        parts, _ = unpack_u32(reply, off)
        return rdd_id, parts

    def parallelize_string_lines(
        self,
        start: int,
        end: int,
        num_keys: int,
        num_tokens: int,
        num_partitions: int,
    ) -> tuple[int, int]:
        body = (
            pack_u64(start)
            + pack_u64(end)
            + pack_u64(num_keys)
            + pack_u64(num_tokens)
            + pack_u32(num_partitions)
        )
        _, reply = self._send_recv(MsgType.PARALLELIZE_STRING_LINES, body)
        rdd_id, off = unpack_u64(reply)
        parts, _ = unpack_u32(reply, off)
        return rdd_id, parts

    def parallelize(self, data: list[Any], num_partitions: int) -> tuple[int, int]:
        pickled = _dumps_data(data if isinstance(data, list) else list(data))
        body = pack_u32(num_partitions) + pack_bytes(pickled)
        _, reply = self._send_recv(MsgType.PARALLELIZE, body)
        rdd_id, off = unpack_u64(reply)
        parts, _ = unpack_u32(reply, off)
        return rdd_id, parts

    def _transform_ack(self, reply: bytes) -> tuple[int, int]:
        rdd_id, off = unpack_u64(reply)
        parts, _ = unpack_u32(reply, off)
        return rdd_id, parts

    def _transform_body(
        self,
        rdd_id: int,
        func,
        native_op: NativeOp | None,
        native_params: tuple[int | float | str, ...],
    ) -> bytes:
        if native_op is not None:
            return pack_u64(rdd_id) + pack_native_tail(native_op, native_params)
        return pack_u64(rdd_id) + pack_u8(TransformMode.PYTHON) + pack_bytes(
            _pickle.dumps(func)
        )

    def map(
        self,
        rdd_id: int,
        func,
        native_op: NativeOp | None = None,
        native_params: tuple[int | float | str, ...] = (),
    ) -> tuple[int, int]:
        body = self._transform_body(rdd_id, func, native_op, native_params)
        _, reply = self._send_recv(MsgType.MAP, body)
        return self._transform_ack(reply)

    def filter(
        self,
        rdd_id: int,
        func,
        native_op: NativeOp | None = None,
        native_params: tuple[int | float | str, ...] = (),
    ) -> tuple[int, int]:
        body = self._transform_body(rdd_id, func, native_op, native_params)
        _, reply = self._send_recv(MsgType.FILTER, body)
        return self._transform_ack(reply)

    def flat_map(
        self,
        rdd_id: int,
        func,
        native_op: NativeOp | None = None,
        native_params: tuple[int | float | str, ...] = (),
    ) -> tuple[int, int]:
        body = self._transform_body(rdd_id, func, native_op, native_params)
        _, reply = self._send_recv(MsgType.FLAT_MAP, body)
        return self._transform_ack(reply)

    def reduce_by_key(
        self,
        rdd_id: int,
        func,
        num_partitions: int,
        native_op: NativeOp | None = None,
        native_params: tuple[int | float | str, ...] = (),
    ) -> tuple[int, int]:
        if native_op is not None:
            tail = pack_native_tail(native_op, native_params)
        else:
            tail = pack_u8(TransformMode.PYTHON) + pack_bytes(_pickle.dumps(func))
        body = pack_u64(rdd_id) + pack_u32(num_partitions) + tail
        _, reply = self._send_recv(MsgType.REDUCE_BY_KEY, body)
        return self._transform_ack(reply)

    def group_by_key(
        self,
        rdd_id: int,
        num_partitions: int,
    ) -> tuple[int, int]:
        body = pack_u64(rdd_id) + pack_u32(num_partitions)
        _, reply = self._send_recv(MsgType.GROUP_BY_KEY, body)
        return self._transform_ack(reply)

    @staticmethod
    def _pack_stage(stage: PendingStage, default_partitions: int) -> bytes:
        body = pack_u8(int(stage.op))
        body += pack_u32(stage.num_partitions or default_partitions)
        if stage.native_op is not None:
            body += pack_native_tail(stage.native_op, stage.native_params)
        else:
            body += pack_u8(TransformMode.PYTHON) + pack_bytes(_pickle.dumps(stage.func))
        return body

    @staticmethod
    def _fuse_stages(
        stages: list[PendingStage],
        line_meta: tuple[int, int] | None = None,
    ) -> list[PendingStage]:
        """Fuse consecutive native stages into single server-side pipelines."""
        out = list(stages)
        i = 0
        while i < len(out):
            if i + 2 < len(out):
                a, b, c = out[i], out[i + 1], out[i + 2]
                if (
                    a.op == StageOp.MAP
                    and a.native_op == NativeOp.MAP_AFFINE_F64
                    and b.op == StageOp.FILTER
                    and b.native_op == NativeOp.FILTER_COMPARE_F64
                    and c.op == StageOp.FLAT_MAP
                    and c.native_op == NativeOp.FLAT_MAP_RANGE_F64
                    and len(a.native_params) == 2
                    and len(b.native_params) == 2
                    and len(c.native_params) == 2
                ):
                    mul, add = a.native_params
                    cmp_op, threshold = b.native_params
                    flat_start, flat_count = c.native_params
                    out[i : i + 3] = [
                        PendingStage(
                            StageOp.MAP,
                            a.func,
                            0,
                            NativeOp.FLOAT64_NARROW_PIPELINE,
                            (mul, add, cmp_op, threshold, flat_start, flat_count),
                        )
                    ]
                    continue
                if (
                    a.op == StageOp.MAP
                    and a.native_op == NativeOp.MAP_AFFINE
                    and b.op == StageOp.FILTER
                    and b.native_op == NativeOp.FILTER_MOD_EQ
                    and c.op == StageOp.FLAT_MAP
                    and c.native_op == NativeOp.FLAT_MAP_RANGE
                    and len(a.native_params) == 3
                    and len(b.native_params) == 2
                    and len(c.native_params) == 2
                ):
                    mul, add, mod = a.native_params
                    filter_mod, filter_eq = b.native_params
                    flat_start, flat_count = c.native_params
                    out[i : i + 3] = [
                        PendingStage(
                            StageOp.MAP,
                            a.func,
                            0,
                            NativeOp.INT64_NARROW_PIPELINE,
                            (
                                mul,
                                add,
                                mod,
                                filter_mod,
                                filter_eq,
                                flat_start,
                                flat_count,
                            ),
                        )
                    ]
                    continue
            if i + 1 < len(out) and line_meta is not None:
                a, b = out[i], out[i + 1]
                if (
                    a.op == StageOp.FLAT_MAP
                    and a.native_op == NativeOp.FLAT_MAP_SPLIT
                    and a.native_params == ("",)
                    and b.op == StageOp.MAP
                    and b.native_op == NativeOp.MAP_TO_PAIR_CONST_STR
                    and len(b.native_params) == 1
                ):
                    num_keys, num_tokens = line_meta
                    vconst = b.native_params[0]
                    out[i : i + 2] = [
                        PendingStage(
                            StageOp.MAP,
                            a.func,
                            0,
                            NativeOp.STRING_LINE_WORD_COUNT,
                            (num_keys, num_tokens, vconst),
                        )
                    ]
                    continue
            i += 1
        return out

    @staticmethod
    def stages_are_native(stages: list[PendingStage]) -> bool:
        return bool(stages) and all(stage.native_op is not None for stage in stages)

    def materialize(
        self,
        rdd_id: int,
        num_partitions: int,
        stages: list[PendingStage],
        line_meta: tuple[int, int] | None = None,
    ) -> tuple[int, int]:
        fused = self._fuse_stages(stages, line_meta)
        if self.protocol_version >= 2:
            if not fused:
                return rdd_id, num_partitions
            body = pack_u64(rdd_id) + pack_u32(len(fused))
            for stage in fused:
                body += self._pack_stage(stage, num_partitions)
            _, reply = self._send_recv(MsgType.APPLY_STAGES, body)
            return self._transform_ack(reply)
        cur_id, parts = rdd_id, num_partitions
        for stage in fused:
            native_op = stage.native_op
            native_params = stage.native_params
            if stage.op == StageOp.MAP:
                cur_id, parts = self.map(cur_id, stage.func, native_op, native_params)
            elif stage.op == StageOp.FILTER:
                cur_id, parts = self.filter(cur_id, stage.func, native_op, native_params)
            elif stage.op == StageOp.FLAT_MAP:
                cur_id, parts = self.flat_map(cur_id, stage.func, native_op, native_params)
            elif stage.op == StageOp.REDUCE_BY_KEY:
                cur_id, parts = self.reduce_by_key(
                    cur_id,
                    stage.func,
                    stage.num_partitions or parts,
                    native_op,
                    native_params,
                )
            elif stage.op == StageOp.GROUP_BY_KEY:
                cur_id, parts = self.group_by_key(
                    cur_id,
                    stage.num_partitions or parts,
                )
        return cur_id, parts

    def execute(
        self,
        rdd_id: int,
        num_partitions: int,
        stages: list[PendingStage],
        action: Action,
        action_func=None,
        line_meta: tuple[int, int] | None = None,
    ) -> Any:
        fused = self._fuse_stages(stages, line_meta)
        body = pack_u64(rdd_id)
        body += pack_u32(len(fused))
        for stage in fused:
            body += self._pack_stage(stage, num_partitions)
        body += pack_u8(int(action))
        body += pack_bytes(_pickle.dumps(action_func) if action_func is not None else b"")
        _, reply = self._send_recv(MsgType.EXECUTE, body)
        action_code, off = unpack_u8(reply)
        if Action(action_code) == Action.COUNT:
            value, _ = unpack_u64(reply, off)
            return int(value)
        if Action(action_code) == Action.COLLECT:
            data, _ = unpack_bytes(reply, off)
            return pickle.loads(data)
        if Action(action_code) == Action.REDUCE:
            data, _ = unpack_bytes(reply, off)
            return _pickle.loads(data)
        if Action(action_code) == Action.FOREACH:
            return None
        raise RuntimeError(f"unsupported execute action: {action_code}")

    def count(self, rdd_id: int) -> int:
        _, reply = self._send_recv(MsgType.COUNT, pack_u64(rdd_id))
        value, _ = unpack_u64(reply)
        return int(value)

    def collect(self, rdd_id: int) -> list[Any]:
        _, reply = self._send_recv(MsgType.COLLECT, pack_u64(rdd_id))
        return pickle.loads(reply)

    def collect_partitions(self, rdd_id: int) -> list[list[Any]]:
        _, reply = self._send_recv(MsgType.COLLECT_PARTITIONS, pack_u64(rdd_id))
        data = _pickle.loads(reply)
        return [list(part) for part in data]

    def collect_partition(self, rdd_id: int, partition: int) -> list[Any]:
        body = pack_u64(rdd_id) + pack_u32(partition)
        _, reply = self._send_recv(MsgType.COLLECT_PARTITION, body)
        return list(_pickle.loads(reply))

    def parallelize_partitions(self, partitions: list[list[Any]]) -> tuple[int, int]:
        body = pack_bytes(_dumps_data(partitions))
        _, reply = self._send_recv(MsgType.PARALLELIZE_PARTITIONS, body)
        rdd_id, off = unpack_u64(reply)
        parts, _ = unpack_u32(reply, off)
        return rdd_id, parts

    def reduce(self, rdd_id: int, func) -> Any:
        body = pack_u64(rdd_id) + pack_bytes(_pickle.dumps(func))
        _, reply = self._send_recv(MsgType.REDUCE, body)
        return _pickle.loads(reply)

    def foreach(self, rdd_id: int, func) -> None:
        body = pack_u64(rdd_id) + pack_bytes(_pickle.dumps(func))
        self._send_recv(MsgType.FOREACH, body)

    # ----- v0.6 server-side operators (protocol v2) -------------------------

    def union(self, rdd_ids: list[int]) -> tuple[int, int]:
        body = pack_u32(len(rdd_ids)) + b"".join(pack_u64(i) for i in rdd_ids)
        _, reply = self._send_recv(MsgType.UNION, body)
        return self._transform_ack(reply)

    def partition_by(
        self, rdd_id: int, num_partitions: int, func=None, round_robin: bool = False
    ) -> tuple[int, int]:
        blob = _pickle.dumps(func) if func is not None else b""
        body = (
            pack_u64(rdd_id)
            + pack_u32(num_partitions)
            + pack_u8(1 if round_robin else 0)
            + pack_bytes(blob)
        )
        _, reply = self._send_recv(MsgType.PARTITION_BY, body)
        return self._transform_ack(reply)

    def coalesce(self, rdd_id: int, groups: list[list[int]]) -> tuple[int, int]:
        body = pack_u64(rdd_id) + pack_u32(len(groups))
        for group in groups:
            body += pack_u32(len(group)) + b"".join(pack_u32(p) for p in group)
        _, reply = self._send_recv(MsgType.COALESCE, body)
        return self._transform_ack(reply)

    def cartesian(self, left: int, right: int) -> tuple[int, int]:
        _, reply = self._send_recv(MsgType.CARTESIAN, pack_u64(left) + pack_u64(right))
        return self._transform_ack(reply)

    def zip_partitions(self, left: int, right: int, func) -> tuple[int, int]:
        body = pack_u64(left) + pack_u64(right) + pack_bytes(_pickle.dumps(func))
        _, reply = self._send_recv(MsgType.ZIP_PARTITIONS, body)
        return self._transform_ack(reply)

    def persist(self, rdd_id: int) -> tuple[int, int]:
        _, reply = self._send_recv(MsgType.PERSIST, pack_u64(rdd_id))
        return self._transform_ack(reply)

    def run_python(self, func, *args) -> Any:
        """Run ``func(*args)`` inside the server interpreter."""
        body = pack_bytes(_pickle.dumps((func, args)))
        _, reply = self._send_recv(MsgType.RUN_PYTHON, body)
        return pickle.loads(reply)

    def run_job(self, rdd_id: int, partitions: list[int], counts: bool = False) -> list[Any]:
        """Compute a subset of partitions (lists of elements, or counts)."""
        body = (
            pack_u64(rdd_id)
            + pack_u32(len(partitions))
            + b"".join(pack_u32(p) for p in partitions)
            + pack_u8(1 if counts else 0)
        )
        _, reply = self._send_recv(MsgType.RUN_JOB, body)
        return pickle.loads(reply)

    def native_stats(self, rdd_id: int):
        _, reply = self._send_recv(MsgType.NATIVE_STATS, pack_u64(rdd_id))
        return pickle.loads(reply)

    def describe(self, rdd_id: int) -> tuple[str, int, int]:
        _, reply = self._send_recv(MsgType.DESCRIBE, pack_u64(rdd_id))
        return tuple(pickle.loads(reply))
