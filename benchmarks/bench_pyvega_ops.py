#!/usr/bin/env python3
"""Micro-benchmarks for common PyVega RDD operations.

Usage:
    VEGA_SERVER=build-release/vega_server python benchmarks/bench_pyvega_ops.py [--n 200000]

Prints one line per operation with the best-of-N wall time in milliseconds,
and optionally writes a JSON report (``--json out.json``) so two builds can
be compared side by side.
"""

from __future__ import annotations

import argparse
import json
import time

from pyvega import SparkContext


def _best(fn, repeat: int) -> float:
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=200_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    n = args.n

    with SparkContext(args.workers) as sc:
        objs = [(i % 1000, float(i)) for i in range(n)]
        words = [f"w{i % 5000}" for i in range(n)]
        nums = list(range(n))
        pairs = sc.parallelize(objs, 8)
        wrdd = sc.parallelize(words, 8)
        nrdd = sc.parallelize(nums, 8)

        cases = {
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
        report = {}
        for name, fn in cases.items():
            try:
                ms = _best(fn, args.repeat)
                report[name] = round(ms, 2)
                print(f"{name:34s} {ms:10.2f} ms")
            except Exception as exc:  # noqa: BLE001 - report and continue
                report[name] = None
                print(f"{name:34s}     FAILED ({str(exc)[:60]})")
    if args.json:
        with open(args.json, "w") as handle:
            json.dump({"n": n, "workers": args.workers, "results": report}, handle, indent=2)


if __name__ == "__main__":
    main()
