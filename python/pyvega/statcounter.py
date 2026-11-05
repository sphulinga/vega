"""Mergeable running statistics (count, mean, variance, min, max)."""

from __future__ import annotations

import math
from typing import Any, Iterable


class StatCounter:
    """Numerically stable running statistics, mergeable across partitions.

    Uses Welford's online algorithm and Chan et al.'s parallel merge, so the
    per-partition counters computed on the server combine exactly.
    """

    __slots__ = ("n", "mu", "m2", "maxValue", "minValue")

    def __init__(self, values: Iterable[Any] | None = None):
        self.n = 0
        self.mu = 0.0
        self.m2 = 0.0
        self.maxValue = float("-inf")
        self.minValue = float("inf")
        if values is not None:
            for value in values:
                self.merge(value)

    @classmethod
    def from_moments(cls, n: int, mean: float, m2: float, min_value, max_value) -> "StatCounter":
        out = cls()
        out.n = int(n)
        out.mu = float(mean)
        out.m2 = float(m2)
        if n:
            out.minValue = min_value
            out.maxValue = max_value
        return out

    def merge(self, value: Any) -> "StatCounter":
        delta = value - self.mu
        self.n += 1
        self.mu += delta / self.n
        self.m2 += delta * (value - self.mu)
        if value > self.maxValue:
            self.maxValue = value
        if value < self.minValue:
            self.minValue = value
        return self

    def mergeStats(self, other: "StatCounter") -> "StatCounter":
        if not isinstance(other, StatCounter):
            raise TypeError("Can only merge StatCounter but got %s" % type(other))
        if other is self:
            return self.mergeStats(other.copy())
        if other.n == 0:
            return self
        if self.n == 0:
            self.n, self.mu, self.m2 = other.n, other.mu, other.m2
            self.maxValue, self.minValue = other.maxValue, other.minValue
            return self
        delta = other.mu - self.mu
        total = self.n + other.n
        if other.n * 10 < self.n:
            self.mu += delta * other.n / total
        elif self.n * 10 < other.n:
            self.mu = other.mu - delta * self.n / total
        else:
            self.mu = (self.mu * self.n + other.mu * other.n) / total
        self.m2 += other.m2 + delta * delta * self.n * other.n / total
        self.n = total
        self.maxValue = max(self.maxValue, other.maxValue)
        self.minValue = min(self.minValue, other.minValue)
        return self

    def copy(self) -> "StatCounter":
        out = StatCounter()
        out.n, out.mu, out.m2 = self.n, self.mu, self.m2
        out.maxValue, out.minValue = self.maxValue, self.minValue
        return out

    def count(self) -> int:
        return int(self.n)

    def mean(self) -> float:
        return float(self.mu)

    def sum(self) -> float:
        return float(self.n * self.mu)

    def min(self):
        return self.minValue

    def max(self):
        return self.maxValue

    def variance(self) -> float:
        return float("nan") if self.n == 0 else float(self.m2 / self.n)

    def sampleVariance(self) -> float:
        return float("nan") if self.n <= 1 else float(self.m2 / (self.n - 1))

    def stdev(self) -> float:
        return float(math.sqrt(self.variance()))

    def sampleStdev(self) -> float:
        return float(math.sqrt(self.sampleVariance()))

    def asDict(self, sample: bool = False) -> dict[str, Any]:
        return {
            "count": self.count(),
            "mean": self.mean(),
            "sum": self.sum(),
            "min": self.min(),
            "max": self.max(),
            "stdev": self.stdev() if not sample else self.sampleStdev(),
            "variance": self.variance() if not sample else self.sampleVariance(),
        }

    def __getstate__(self):
        return (self.n, self.mu, self.m2, self.maxValue, self.minValue)

    def __setstate__(self, state):
        self.n, self.mu, self.m2, self.maxValue, self.minValue = state

    def __repr__(self) -> str:
        return "(count: %s, mean: %s, stdev: %s, max: %s, min: %s)" % (
            self.count(),
            self.mean(),
            self.stdev(),
            self.max(),
            self.min(),
        )
