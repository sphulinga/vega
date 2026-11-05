"""RDD sampling utilities (ported from Apache Spark rddsampler.py)."""

from __future__ import annotations

import math
import random
import sys
from typing import Generic, Hashable, Iterable, Iterator, Optional, TypeVar

T = TypeVar("T")
K = TypeVar("K", bound=Hashable)


class RDDSamplerBase:
    def __init__(self, withReplacement: bool, seed: Optional[int] = None) -> None:
        self._seed = seed if seed is not None else random.randint(0, sys.maxsize)
        self._withReplacement = withReplacement
        self._random: Optional[random.Random] = None

    def initRandomGenerator(self, split: int) -> None:
        self._random = random.Random(self._seed ^ split)
        for _ in range(10):
            self._random.randint(0, 1)

    def getUniformSample(self) -> float:
        assert self._random is not None
        return self._random.random()

    def getPoissonSample(self, mean: float) -> int:
        assert self._random is not None
        if mean < 20.0:
            lda = math.exp(-mean)
            p = self._random.random()
            k = 0
            while p > lda:
                k += 1
                p *= self._random.random()
        else:
            p = self._random.expovariate(mean)
            k = 0
            while p < 1.0:
                k += 1
                p += self._random.expovariate(mean)
        return k

    def func(self, split: int, iterator: Iterable[T]) -> Iterator[T]:
        raise NotImplementedError


class RDDSampler(RDDSamplerBase):
    def __init__(
        self, withReplacement: bool, fraction: float, seed: Optional[int] = None
    ) -> None:
        super().__init__(withReplacement, seed)
        self._fraction = fraction

    def func(self, split: int, iterator: Iterable[T]) -> Iterator[T]:
        self.initRandomGenerator(split)
        if self._withReplacement:
            for obj in iterator:
                count = self.getPoissonSample(self._fraction)
                for _ in range(count):
                    yield obj
        else:
            for obj in iterator:
                if self.getUniformSample() < self._fraction:
                    yield obj


class RDDRangeSampler(RDDSamplerBase):
    """Keeps elements whose uniform draw falls in [lower, upper) (randomSplit)."""

    def __init__(self, lowerBound: float, upperBound: float, seed: Optional[int] = None) -> None:
        super().__init__(False, seed)
        self._lowerBound = lowerBound
        self._upperBound = upperBound

    def func(self, split: int, iterator: Iterable[T]) -> Iterator[T]:
        self.initRandomGenerator(split)
        for obj in iterator:
            if self._lowerBound <= self.getUniformSample() < self._upperBound:
                yield obj


class RDDStratifiedSampler(RDDSamplerBase):
    """Per-key sampling fractions for (key, value) RDDs (sampleByKey)."""

    def __init__(
        self, withReplacement: bool, fractions: dict, seed: Optional[int] = None
    ) -> None:
        super().__init__(withReplacement, seed)
        self._fractions = fractions

    def func(self, split: int, iterator: Iterable[tuple]) -> Iterator[tuple]:
        self.initRandomGenerator(split)
        if self._withReplacement:
            for key, val in iterator:
                count = self.getPoissonSample(self._fractions[key])
                for _ in range(count):
                    yield key, val
        else:
            for key, val in iterator:
                if self.getUniformSample() < self._fractions[key]:
                    yield key, val


def compute_fraction_for_sample_size(sample_size_lower_bound: int, total: int, with_replacement: bool) -> float:
    """Sampling rate that yields >= the requested size with high probability."""
    fraction = float(sample_size_lower_bound) / total
    if with_replacement:
        num_st_dev = 5
        if sample_size_lower_bound < 12:
            num_st_dev = 9
        return fraction + num_st_dev * math.sqrt(fraction / total)
    delta = 0.00005
    gamma = -math.log(delta) / total
    return min(1, fraction + gamma + math.sqrt(gamma * gamma + 2 * gamma * fraction))
