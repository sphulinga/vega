#!/usr/bin/env python3
"""Micro-benchmarks for common RDD operations through a Python API.

Runs the *same* Python code on either engine, so PyVega and PySpark are
compared like for like:

    python benchmarks/bench_python_ops.py --engine pyvega  --json pyvega.json
    python benchmarks/bench_python_ops.py --engine pyspark --json pyspark.json

PyVega needs a server (``pip install pyvega``, or VEGA_SERVER pointing at a
built ``vega_server``); PySpark needs ``pip install pyspark`` and Java 17+.

Each operation's time is its best of --repeat runs, after --warmup untimed
passes over all operations. The JSON report holds the times (ms) and each
operation's result summary (a count, sum or length), so runs on different
engines can be checked for doing the same work.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Callable


def make_context(engine: str, workers: int, driver_memory: str):
    """Return (SparkContext-like object, engine label)."""
    if engine == "pyvega":
        import pyvega
        from pyvega import SparkContext

        return SparkContext(workers), f"pyvega-{pyvega.__version__}"
    if engine == "pyspark":
        import pyspark
        from pyspark import SparkConf, SparkContext

        # Python workers must run the same interpreter as the driver.
        os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
        conf = (
            SparkConf()
            .setMaster(f"local[{workers}]")
            .setAppName("bench_python_ops")
            .set("spark.ui.enabled", "false")
            .set("spark.driver.memory", driver_memory)
        )
        sc = SparkContext(conf=conf)
        sc.setLogLevel("WARN")
        return sc, f"pyspark-{pyspark.__version__}"
    raise ValueError(f"unknown engine {engine!r}")


def summarize(value: Any) -> Any:
    """A comparable summary of an operation's result."""
    if isinstance(value, (list, tuple)) and not (isinstance(value, tuple) and len(value) == 2):
        return len(value)
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, tuple):  # first(): a (key, value) pair
        return [value[0], value[1]]
    return value


def build_cases(sc, n: int) -> dict[str, Callable[[], Any]]:
    objs = [(i % 1000, float(i)) for i in range(n)]
    words = [f"w{i % 5000}" for i in range(n)]
    nums = list(range(n))
    pairs = sc.parallelize(objs, 8)
    wrdd = sc.parallelize(words, 8)
    nrdd = sc.parallelize(nums, 8)

    return {
        "parallelize+collect (tuples)": lambda: sc.parallelize(objs, 8).collect(),
        "map+filter python (tuples)": lambda: pairs.map(lambda kv: (kv[0], kv[1] * 2))
        .filter(lambda kv: kv[0] % 2 == 0)
        .count(),
        "reduceByKey (tuples)": lambda: pairs.reduceByKey(lambda a, b: a + b).count(),
        "groupByKey (tuples)": lambda: pairs.groupByKey().count(),
        "distinct (strings)": lambda: wrdd.distinct().count(),
        "sortByKey (tuples)": lambda: pairs.sortByKey().count(),
        "mapPartitions": lambda: pairs.mapPartitions(lambda it: (x for x in it)).count(),
        "union x2": lambda: sc.union([pairs, pairs]).count(),
        "repartition(16)": lambda: pairs.repartition(16).count(),
        "aggregateByKey": lambda: pairs.aggregateByKey(
            0.0, lambda a, v: a + v, lambda a, b: a + b
        ).count(),
        "take(10)": lambda: pairs.map(lambda kv: kv).take(10),
        "first()": lambda: pairs.map(lambda kv: kv).first(),
        "sum (ints)": lambda: nrdd.sum(),
        "aggregate": lambda: pairs.aggregate(0.0, lambda a, kv: a + kv[1], lambda a, b: a + b),
        "chained shuffle": lambda: pairs.reduceByKey(lambda a, b: a + b)
        .map(lambda kv: (kv[0] % 10, kv[1]))
        .groupByKey()
        .count(),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine", choices=["pyvega", "pyspark"], default="pyvega")
    parser.add_argument("--n", type=int, default=200_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=0,
                        help="untimed passes over all operations first (0 = none)")
    parser.add_argument("--driver-memory", default="4g", help="PySpark only")
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    sc, label = make_context(args.engine, args.workers, args.driver_memory)
    times: dict[str, float | None] = {}
    values: dict[str, Any] = {}
    try:
        cases = build_cases(sc, args.n)
        for _ in range(args.warmup):
            for fn in cases.values():
                try:
                    fn()
                except Exception:  # noqa: BLE001 - reported in the timed pass
                    pass
        for name, fn in cases.items():
            try:
                best = float("inf")
                for _ in range(args.repeat):
                    start = time.perf_counter()
                    result = fn()
                    best = min(best, time.perf_counter() - start)
                times[name] = round(best * 1000.0, 2)
                values[name] = summarize(result)
                print(f"{name:34s} {best * 1000.0:10.2f} ms")
            except Exception as exc:  # noqa: BLE001 - report and continue
                times[name] = None
                print(f"{name:34s}     FAILED ({str(exc)[:60]})")
    finally:
        sc.stop()
    if args.json:
        with open(args.json, "w") as handle:
            json.dump({"engine": label, "n": args.n, "workers": args.workers,
                       "warmup": args.warmup, "results": times, "values": values},
                      handle, indent=2)


if __name__ == "__main__":
    main()
