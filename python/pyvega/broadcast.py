"""Broadcast variables for PyVega."""

from __future__ import annotations

import copy
import threading
from typing import Any, Generic, Iterator, TypeVar

T = TypeVar("T")


class _BroadcastValue(Generic[T]):
    def __init__(self, value: T):
        self._value = value

    @property
    def value(self) -> T:
        return self._value


class BroadcastPickleRegistry(threading.local):
    """Thread-local registry for broadcast variables encountered during pickling."""

    def __init__(self) -> None:
        self.__dict__.setdefault("_registry", set())

    def __iter__(self) -> Iterator["Broadcast[Any]"]:
        for bcast in self._registry:
            yield bcast

    def add(self, bcast: "Broadcast[Any]") -> None:
        self._registry.add(bcast)

    def clear(self) -> None:
        self._registry.clear()


class Broadcast(Generic[T]):
    _next_id = 1

    def __init__(self, sc: Any, value: T, registry: BroadcastPickleRegistry):
        self._sc = sc
        self._value: T | object = value
        self._destroyed = False
        self._registry = registry
        self._id = Broadcast._next_id
        Broadcast._next_id += 1
        _broadcastRegistry[self._id] = self

    @property
    def value(self) -> T:
        if self._destroyed:
            raise RuntimeError("Broadcast has been destroyed")
        return self._value  # type: ignore[return-value]

    def unpersist(self, blocking: bool = False) -> None:
        del blocking

    def destroy(self, blocking: bool = False) -> None:
        del blocking
        self._destroyed = True
        self._value = None  # type: ignore[assignment]
        _broadcastRegistry.pop(self._id, None)

    def __reduce__(self) -> tuple[Any, tuple[Any]]:
        if self._destroyed:
            raise RuntimeError("Broadcast has been destroyed")
        self._registry.add(self)
        return _BroadcastValue, (copy.deepcopy(self._value),)


_broadcastRegistry: dict[int, Broadcast[Any]] = {}


def _from_id(bid: int) -> Broadcast[Any]:
    if bid not in _broadcastRegistry:
        raise RuntimeError(f"Broadcast variable {bid} is not available")
    return _broadcastRegistry[bid]
