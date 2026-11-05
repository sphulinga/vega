"""
PyVega -- Python client for the Vega C++ engine.

Connects to a Vega session server over TCP. Computation and the actor
system run in the C++ server process; this package is a thin RPC client.

    from pyvega import SparkContext

    with SparkContext(4) as sc:
        result = sc.range(0, 100).map(lambda x: x * 2).collect()
"""

from __future__ import annotations

__version__ = "0.6.0"

from pyvega.accumulators import Accumulator, AccumulatorParam
from pyvega.broadcast import Broadcast
from pyvega.context import JobRecord, SparkContext
from pyvega.rdd import RDD, BoundedFloat
from pyvega.resultiterable import ResultIterable
from pyvega.statcounter import StatCounter
from pyvega.storagelevel import StorageLevel

__all__ = [
    "SparkContext",
    "RDD",
    "Accumulator",
    "AccumulatorParam",
    "Broadcast",
    "BoundedFloat",
    "JobRecord",
    "ResultIterable",
    "StatCounter",
    "StorageLevel",
    "__version__",
]
