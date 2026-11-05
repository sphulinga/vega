"""Iterable wrapper for grouped values (cogroup / groupWith results)."""

from __future__ import annotations

from typing import Any


class ResultIterable(list):
    """A list of grouped values; ``.data`` mirrors PySpark's attribute."""

    @property
    def data(self) -> list[Any]:
        return self

    @property
    def maxindex(self) -> int:
        return len(self)

    def __reduce__(self):
        return (ResultIterable, (list(self),))
