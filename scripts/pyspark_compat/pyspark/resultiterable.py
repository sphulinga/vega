"""ResultIterable stub for upstream PySpark tests."""

from __future__ import annotations

from typing import Generic, Iterator, TypeVar

from pyspark.shuffle import ExternalListOfList

T = TypeVar("T")


class ResultIterable(Generic[T]):
    def __init__(self, data: ExternalListOfList[T]):
        self.data = data
        self.index = 0
        self.maxindex = len(data)

    def __iter__(self) -> Iterator[T]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)
