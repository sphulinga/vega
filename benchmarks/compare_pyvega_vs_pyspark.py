#!/usr/bin/env python3
"""Compare Vega Python API vs Apache Spark (PySpark) on one workload.

This benchmark intentionally runs both engines from Python so the API surface is
comparable for this repository's current Python bindings.
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from dataclasses import dataclass
from typing import Callable


def configure_java_env() -> None:
    """Set Java runtime paths for PySpark on Homebrew macOS installs."""
    java_home = os.environ.get("JAVA_HOME", "").strip()
    if java_home:
        return

    for candidate in (
        "/opt/homebrew/opt/openjdk@21",
        "/opt/homebrew/opt/openjdk",
    ):
        bin_java = os.path.join(candidate, "bin", "java")
        if os.path.exists(bin_java):
            os.environ["JAVA_HOME"] = candidate
            os.environ["PATH"] = f"{os.path.join(candidate, 'bin')}:{os.environ.get('PATH', '')}"
            return


@dataclass
class RunStats:
    name: str
    times: list[float]
    result: int

    @property
    def median(self) -> float:
        return statistics.median(self.times)

    @property
    def mean(self) -> float:
        return statistics.mean(self.times)


def timed_runs(label: str, repeats: int, fn: Callable[[], int]) -> RunStats:
    # Warmup run to reduce one-time initialization effects.
    warm = fn()
    del warm

    times: list[float] = []
    last_result = 0
    for _ in range(repeats):
        t0 = time.perf_counter()
        last_result = fn()
        times.append(time.perf_counter() - t0)
    return RunStats(name=label, times=times, result=last_result)


def pyvega_benchmark(n: int, partitions: int, workers: int, repeats: int) -> RunStats:
    from pyvega import SparkContext

    data = list(range(n))
    with SparkContext(workers) as sc:
        def run_once() -> int:
            rdd = (
                sc.parallelize(data, partitions)
                .map(lambda x: (x * 31 + 7) % 1_000_003)
                .filter(lambda x: x % 7 == 0)
                .flat_map(lambda x: (x, x + 1, x + 2))
            )
            return rdd.count()

        return timed_runs("pyvega", repeats, run_once)


def pyspark_benchmark(n: int, partitions: int, workers: int, repeats: int) -> RunStats:
    configure_java_env()
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.master(f"local[{workers}]")
        .appName("pyvega-vs-pyspark-benchmark")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    data = list(range(n))
    sc = spark.sparkContext

    def run_once() -> int:
        rdd = (
            sc.parallelize(data, partitions)
            .map(lambda x: (x * 31 + 7) % 1_000_003)
            .filter(lambda x: x % 7 == 0)
            .flatMap(lambda x: (x, x + 1, x + 2))
        )
        return int(rdd.count())

    try:
        return timed_runs("pyspark", repeats, run_once)
    finally:
        spark.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare PyVega vs Apache Spark (PySpark).")
    parser.add_argument("--n", type=int, default=2_000_000, help="Input size")
    parser.add_argument("--partitions", type=int, default=16, help="RDD partitions")
    parser.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) // 2), help="Workers/threads")
    parser.add_argument("--repeats", type=int, default=3, help="Measured repeats (after warmup)")
    args = parser.parse_args()

    print(f"Benchmark config: n={args.n}, partitions={args.partitions}, workers={args.workers}, repeats={args.repeats}")

    my_stats = pyvega_benchmark(args.n, args.partitions, args.workers, args.repeats)
    sp_stats = pyspark_benchmark(args.n, args.partitions, args.workers, args.repeats)

    if my_stats.result != sp_stats.result:
        raise RuntimeError(f"Result mismatch: vega={my_stats.result}, pyspark={sp_stats.result}")

    speedup = sp_stats.median / my_stats.median if my_stats.median > 0 else float("inf")

    print("")
    print("Engine timings (seconds):")
    print(f"- pyvega times: {', '.join(f'{t:.4f}' for t in my_stats.times)}")
    print(f"- pyspark times: {', '.join(f'{t:.4f}' for t in sp_stats.times)}")
    print("")
    print("| engine | median_s | mean_s | result_count |")
    print("|---|---:|---:|---:|")
    print(f"| pyvega | {my_stats.median:.4f} | {my_stats.mean:.4f} | {my_stats.result} |")
    print(f"| pyspark | {sp_stats.median:.4f} | {sp_stats.mean:.4f} | {sp_stats.result} |")
    print("")
    print(f"Median speedup (pyspark / pyvega): {speedup:.2f}x")


if __name__ == "__main__":
    main()
