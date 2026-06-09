"""Shuffle module stub for upstream PySpark tests."""

from __future__ import annotations

from typing import Generic, Iterator, TypeVar

T = TypeVar("T")


class ExternalList(Generic[T]):
    def __init__(self, data: list[T]):
        self.data = data
        self.values = data
        self.count = len(data)

    def __iter__(self) -> Iterator[T]:
        return iter(self.data)

    def __len__(self) -> int:
        return self.count

    def append(self, value: T) -> None:
        self.data.append(value)
        self.count += 1


class ExternalListOfList(ExternalList[list[T]]):
    def __init__(self, values: list[list[T]]):
        super().__init__(values)
        self.count = sum(len(items) for items in values)

    def append(self, value: list[T]) -> None:
        super().append(value)
        self.count += len(value) - 1

    def __iter__(self) -> Iterator[T]:
        for values in self.data:
            yield from values


class ResultIterable(Generic[T]):
    def __init__(self, data: ExternalListOfList[T]):
        self.data = data
        self.index = 0
        self.maxindex = len(data)

    def __iter__(self) -> Iterator[T]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)
