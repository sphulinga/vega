"""Accumulators: write-only shared variables aggregated back to the driver.

Tasks run inside the Vega server process. When a closure referencing an
:class:`Accumulator` is shipped to the server it unpickles to a server-side
instance (one per accumulator id) that collects ``add()`` calls. After every
job the client drains those partial values over RPC and merges them into the
driver-side value, giving Spark's semantics: updates made in tasks are only
visible on the driver, via ``.value``.
"""

from __future__ import annotations

import threading
import uuid
from typing import Any, Generic, TypeVar

T = TypeVar("T")

__all__ = [
    "Accumulator",
    "AccumulatorParam",
    "INT_ACCUMULATOR_PARAM",
    "FLOAT_ACCUMULATOR_PARAM",
    "COMPLEX_ACCUMULATOR_PARAM",
]


class AccumulatorParam(Generic[T]):
    """Defines how to accumulate values of a given type."""

    def zero(self, value: T) -> T:
        raise NotImplementedError

    def addInPlace(self, value1: T, value2: T) -> T:
        raise NotImplementedError


class AddingAccumulatorParam(AccumulatorParam[T]):
    def __init__(self, zero_value: T):
        self.zero_value = zero_value

    def zero(self, value: T) -> T:
        return self.zero_value

    def addInPlace(self, value1: T, value2: T) -> T:
        value1 += value2  # type: ignore[operator]
        return value1


INT_ACCUMULATOR_PARAM = AddingAccumulatorParam(0)
FLOAT_ACCUMULATOR_PARAM = AddingAccumulatorParam(0.0)
COMPLEX_ACCUMULATOR_PARAM = AddingAccumulatorParam(0.0j)

# Server-side registry: accumulator id -> server instance. Lives in the
# server's interpreter; shared by all sessions, keyed by random ids.
_server_accumulators: dict[str, "Accumulator[Any]"] = {}
_server_lock = threading.Lock()


def _server_instance(aid: str, zero: Any, param: AccumulatorParam[Any]) -> "Accumulator[Any]":
    with _server_lock:
        acc = _server_accumulators.get(aid)
        if acc is None:
            acc = Accumulator(aid, zero, param, _on_server=True)
            _server_accumulators[aid] = acc
        return acc


def _drain(aids: list[str]) -> dict[str, Any]:
    """Return and reset the partial values collected on the server."""
    out: dict[str, Any] = {}
    with _server_lock:
        for aid in aids:
            acc = _server_accumulators.get(aid)
            if acc is not None and acc._dirty:
                out[aid] = acc._value
                acc._value = acc.accum_param.zero(acc._value)
                acc._dirty = False
    return out


def _forget(aids: list[str]) -> None:
    with _server_lock:
        for aid in aids:
            _server_accumulators.pop(aid, None)


class Accumulator(Generic[T]):
    """A shared variable that tasks can only add to.

    >>> acc = sc.accumulator(0)
    >>> sc.parallelize(range(10)).foreach(lambda x: acc.add(x))
    >>> acc.value
    45
    """

    def __init__(
        self,
        aid: str | int | None,
        value: T,
        accum_param: AccumulatorParam[T],
        _on_server: bool = False,
    ):
        self.aid = str(aid) if aid is not None else uuid.uuid4().hex
        self.accum_param = accum_param
        self._value = value
        self._on_server = _on_server
        self._dirty = False
        self._lock = threading.Lock()

    def __reduce__(self):
        zero = self.accum_param.zero(self._value)
        return (_server_instance, (self.aid, zero, self.accum_param))

    @property
    def value(self) -> T:
        if self._on_server:
            raise RuntimeError("Accumulator.value cannot be accessed inside tasks")
        return self._value

    @value.setter
    def value(self, value: T) -> None:
        if self._on_server:
            raise RuntimeError("Accumulator.value cannot be accessed inside tasks")
        self._value = value

    def add(self, term: T) -> None:
        with self._lock:
            self._value = self.accum_param.addInPlace(self._value, term)
            self._dirty = True

    def __iadd__(self, term: T) -> "Accumulator[T]":
        self.add(term)
        return self

    def _merge_remote(self, partial: T) -> None:
        with self._lock:
            self._value = self.accum_param.addInPlace(self._value, partial)

    def __str__(self) -> str:
        return str(self._value)

    def __repr__(self) -> str:
        return "Accumulator<id=%s, value=%s>" % (self.aid, self._value)
