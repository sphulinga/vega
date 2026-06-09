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
