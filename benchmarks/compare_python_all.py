#!/usr/bin/env python3
"""Compare Vega (Python), PySpark, and Sail (PySpark Connect) from Python.

Workloads:
  - narrow: int map / filter / flatMap / count
  - shuffle: int keyed aggregate + count
  - float: float64 map / filter / flatMap / count (native C++ path)
  - string: string split / word-count shuffle + count (native C++ path)
"""

from __future__ import annotations

import argparse
import os
import re
import statistics
import time
from dataclasses import dataclass
from typing import Callable

RESULT_RE = re.compile(
    r"RESULT\s+count=(?P<count>\d+)\s+median_s=(?P<median>[0-9.]+)\s+mean_s=(?P<mean>[0-9.]+)\s+times_s=(?P<times>[0-9.,]+)"
)


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


def configure_java_env() -> None:
    if os.environ.get("JAVA_HOME", "").strip():
        return
    for candidate in ("/opt/homebrew/opt/openjdk@21", "/opt/homebrew/opt/openjdk"):
        if os.path.exists(os.path.join(candidate, "bin", "java")):
            os.environ["JAVA_HOME"] = candidate
            os.environ["PATH"] = f"{os.path.join(candidate, 'bin')}:{os.environ.get('PATH', '')}"
            return


def local_pyspark_session(app_name: str, workers: int):
    """Build a local JVM SparkSession (not Spark Connect)."""
    configure_java_env()
    os.environ.pop("SPARK_REMOTE", None)
    from pyspark.sql import SparkSession

    active = SparkSession.getActiveSession()
    if active is not None:
        active.stop()

    spark = (
        SparkSession.builder.master(f"local[{workers}]")
        .appName(app_name)
        .config("spark.ui.enabled", "false")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def timed_runs(label: str, repeats: int, fn: Callable[[], int]) -> RunStats:
    _ = fn()  # warmup
    times: list[float] = []
    last = 0
    for _ in range(repeats):
        t0 = time.perf_counter()
        last = fn()
        times.append(time.perf_counter() - t0)
    return RunStats(name=label, times=times, result=last)


def parse_cpp_result(text: str, name: str) -> RunStats:
    m = RESULT_RE.search(text)
    if not m:
        raise RuntimeError(f"Could not parse RESULT for {name}.\n{text}")
    return RunStats(
        name=name,
        times=[float(x) for x in m.group("times").split(",") if x],
        result=int(m.group("count")),
    )


# ---------------------------------------------------------------------------
# Narrow: map -> filter -> flatMap -> count
# ---------------------------------------------------------------------------

def pyvega_narrow(n: int, partitions: int, workers: int, repeats: int) -> RunStats:
    from pyvega import SparkContext

    def run_once(sc: SparkContext) -> int:
        return (
            sc.range(0, n, partitions)
            .map(lambda x: (x * 31 + 7) % 1_000_003)
            .filter(lambda x: x % 7 == 0)
            .flat_map(lambda x: (x, x + 1, x + 2))
            .count()
        )

    with SparkContext(workers) as sc:
        _ = run_once(sc)
        times: list[float] = []
        last = 0
        for _ in range(repeats):
            t0 = time.perf_counter()
            last = run_once(sc)
            times.append(time.perf_counter() - t0)
    return RunStats("pyvega", times, last)


def pyspark_narrow(n: int, partitions: int, workers: int, repeats: int) -> RunStats:
    spark = local_pyspark_session("python-all-narrow", workers)
    data = list(range(n))
    sc = spark.sparkContext

    def run_once() -> int:
        return int(
            sc.parallelize(data, partitions)
            .map(lambda x: (x * 31 + 7) % 1_000_003)
            .filter(lambda x: x % 7 == 0)
            .flatMap(lambda x: (x, x + 1, x + 2))
            .count()
        )

    try:
        return timed_runs("pyspark", repeats, run_once)
    finally:
        spark.stop()


def sail_narrow(n: int, partitions: int, repeats: int) -> RunStats:
    from pysail.spark import SparkConnectServer
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    server = SparkConnectServer("127.0.0.1", 0)
    server.start(background=True)
    _, port = server.listening_address
    spark = SparkSession.builder.remote(f"sc://localhost:{port}").getOrCreate()

    def run_once() -> int:
        df = (
            spark.range(0, n, 1, partitions)
            .select(((F.col("id") * F.lit(31) + F.lit(7)) % F.lit(1_000_003)).alias("x"))
            .filter((F.col("x") % F.lit(7)) == F.lit(0))
            .select(
                F.explode(F.array(F.col("x"), F.col("x") + F.lit(1), F.col("x") + F.lit(2))).alias("y")
            )
        )
        return int(df.count())

    try:
        return timed_runs("sail", repeats, run_once)
    finally:
        spark.stop()
        server.stop()


# ---------------------------------------------------------------------------
# Shuffle: keyed sum + count(distinct keys)
# ---------------------------------------------------------------------------

def pyvega_shuffle(
    n: int, partitions: int, workers: int, repeats: int, num_keys: int
) -> RunStats:
    from pyvega import SparkContext

    def run_once(sc: SparkContext) -> int:
        return (
            sc.range(0, n, partitions)
            .map(lambda x: (x % num_keys, 1))
            .reduceByKey(lambda a, b: a + b, partitions)
            .count()
        )

    with SparkContext(workers) as sc:
        _ = run_once(sc)
        times: list[float] = []
        last = 0
        for _ in range(repeats):
            t0 = time.perf_counter()
            last = run_once(sc)
            times.append(time.perf_counter() - t0)
    return RunStats("pyvega", times, last)


def pyspark_shuffle(
    n: int, partitions: int, workers: int, repeats: int, num_keys: int
) -> RunStats:
    spark = local_pyspark_session("python-all-shuffle", workers)

    def run_once() -> int:
        df = (
            spark.range(0, n, 1, partitions)
            .selectExpr(
                f"((id * 1315423911 + 7) % {num_keys}) as k",
                "((id * 31 + 7) % 97) as v",
            )
            .repartition(partitions, "k")
            .groupBy("k")
            .sum("v")
        )
        return int(df.count())

    try:
        return timed_runs("pyspark", repeats, run_once)
    finally:
        spark.stop()


def sail_shuffle(n: int, partitions: int, repeats: int, num_keys: int) -> RunStats:
    from pysail.spark import SparkConnectServer
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    server = SparkConnectServer("127.0.0.1", 0)
    server.start(background=True)
    _, port = server.listening_address
    spark = SparkSession.builder.remote(f"sc://localhost:{port}").getOrCreate()

    def run_once() -> int:
        df = (
            spark.range(0, n, 1, partitions)
            .select(
                (((F.col("id") * F.lit(1315423911) + F.lit(7)) % F.lit(num_keys))).alias("k"),
                (((F.col("id") * F.lit(31) + F.lit(7)) % F.lit(97))).alias("v"),
            )
            .repartition(partitions, "k")
            .groupBy("k")
            .agg(F.sum("v").alias("sv"))
        )
        return int(df.count())

    try:
        return timed_runs("sail", repeats, run_once)
    finally:
        spark.stop()
        server.stop()


# ---------------------------------------------------------------------------
# Float shuffle: float range -> keyed sum + count
# ---------------------------------------------------------------------------

def pyvega_float_shuffle(
    n: int, partitions: int, workers: int, repeats: int, num_keys: int
) -> RunStats:
    from pyvega import SparkContext

    def run_once(sc: SparkContext) -> int:
        return (
            sc.float_range(0, n, partitions)
            .map(lambda x: (int(x) % num_keys, 1))
            .reduceByKey(lambda a, b: a + b, partitions)
            .count()
        )

    with SparkContext(workers) as sc:
        _ = run_once(sc)
        times: list[float] = []
        last = 0
        for _ in range(repeats):
            t0 = time.perf_counter()
            last = run_once(sc)
            times.append(time.perf_counter() - t0)
    return RunStats("pyvega", times, last)


def pyspark_float_shuffle(
    n: int, partitions: int, workers: int, repeats: int, num_keys: int
) -> RunStats:
    from pyspark.sql import functions as F

    spark = local_pyspark_session("python-all-float-shuffle", workers)

    def run_once() -> int:
        df = (
            spark.range(0, n, 1, partitions)
            .select(
                (F.col("id").cast("double").alias("x")),
            )
            .select(
                (F.col("x").cast("long") % F.lit(num_keys)).alias("k"),
                F.lit(1).alias("v"),
            )
            .repartition(partitions, "k")
            .groupBy("k")
            .sum("v")
        )
        return int(df.count())

    try:
        return timed_runs("pyspark", repeats, run_once)
    finally:
        spark.stop()


def sail_float_shuffle(n: int, partitions: int, repeats: int, num_keys: int) -> RunStats:
    from pysail.spark import SparkConnectServer
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    server = SparkConnectServer("127.0.0.1", 0)
    server.start(background=True)
    _, port = server.listening_address
    spark = SparkSession.builder.remote(f"sc://localhost:{port}").getOrCreate()

    def run_once() -> int:
        df = (
            spark.range(0, n, 1, partitions)
            .select(F.col("id").cast("double").alias("x"))
            .select(
                (F.col("x").cast("long") % F.lit(num_keys)).alias("k"),
                F.lit(1).alias("v"),
            )
            .repartition(partitions, "k")
            .groupBy("k")
            .agg(F.sum("v").alias("sv"))
        )
        return int(df.count())

    try:
        return timed_runs("sail", repeats, run_once)
    finally:
        spark.stop()
        server.stop()


# ---------------------------------------------------------------------------
# Float: map -> filter -> flatMap -> count (native float64 path)
# ---------------------------------------------------------------------------

def pyvega_float(n: int, partitions: int, workers: int, repeats: int) -> RunStats:
    from pyvega import SparkContext

    def run_once(sc: SparkContext) -> int:
        return (
            sc.float_range(0, n, partitions)
            .map(lambda x: x * 31.0 + 7.0)
            .filter(lambda x: x > 100.0)
            .flat_map(lambda x: (x, x + 1.0, x + 2.0))
            .count()
        )

    with SparkContext(workers) as sc:
        _ = run_once(sc)
        times: list[float] = []
        last = 0
        for _ in range(repeats):
            t0 = time.perf_counter()
            last = run_once(sc)
            times.append(time.perf_counter() - t0)
    return RunStats("pyvega", times, last)


def pyspark_float(n: int, partitions: int, workers: int, repeats: int) -> RunStats:
    from pyspark.sql import functions as F

    spark = local_pyspark_session("python-all-float", workers)

    def run_once() -> int:
        df = (
            spark.range(0, n, 1, partitions)
            .select((F.col("id").cast("double") * F.lit(31.0) + F.lit(7.0)).alias("x"))
            .filter(F.col("x") > F.lit(100.0))
            .select(
                F.explode(
                    F.array(F.col("x"), F.col("x") + F.lit(1.0), F.col("x") + F.lit(2.0))
                ).alias("y")
            )
        )
        return int(df.count())

    try:
        return timed_runs("pyspark", repeats, run_once)
    finally:
        spark.stop()


def sail_float(n: int, partitions: int, repeats: int) -> RunStats:
    from pysail.spark import SparkConnectServer
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    server = SparkConnectServer("127.0.0.1", 0)
    server.start(background=True)
    _, port = server.listening_address
    spark = SparkSession.builder.remote(f"sc://localhost:{port}").getOrCreate()

    def run_once() -> int:
        df = (
            spark.range(0, n, 1, partitions)
            .select((F.col("id").cast("double") * F.lit(31.0) + F.lit(7.0)).alias("x"))
            .filter(F.col("x") > F.lit(100.0))
            .select(
                F.explode(
                    F.array(F.col("x"), F.col("x") + F.lit(1.0), F.col("x") + F.lit(2.0))
                ).alias("y")
            )
        )
        return int(df.count())

    try:
        return timed_runs("sail", repeats, run_once)
    finally:
        spark.stop()
        server.stop()


# ---------------------------------------------------------------------------
# String: split lines -> word count shuffle + count (native string path)
# ---------------------------------------------------------------------------

def pyvega_string(
    n: int, partitions: int, workers: int, repeats: int, num_keys: int, num_tokens: int
) -> RunStats:
    from pyvega import SparkContext

    def run_once(sc: SparkContext) -> int:
        return (
            sc.string_lines(0, n, num_keys, num_tokens, partitions)
            .flat_map(lambda s: s.split())
            .map(lambda w: (w, 1))
            .reduceByKey(lambda a, b: a + b, partitions)
            .count()
        )

    with SparkContext(workers) as sc:
        _ = run_once(sc)
        times: list[float] = []
        last = 0
        for _ in range(repeats):
            t0 = time.perf_counter()
            last = run_once(sc)
            times.append(time.perf_counter() - t0)
    return RunStats("pyvega", times, last)


def pyspark_string(
    n: int, partitions: int, workers: int, repeats: int, num_keys: int, num_tokens: int
) -> RunStats:
    from pyspark.sql import functions as F

    spark = local_pyspark_session("python-all-string", workers)

    def run_once() -> int:
        df = (
            spark.range(0, n, 1, partitions)
            .select(
                F.concat(
                    F.lit("word"),
                    (F.col("id") % F.lit(num_keys)).cast("string"),
                    F.lit(" tok"),
                    (F.col("id") % F.lit(num_tokens)).cast("string"),
                ).alias("line")
            )
            .select(F.explode(F.split(F.col("line"), F.lit(" "))).alias("w"))
            .groupBy("w")
            .count()
        )
        return int(df.count())

    try:
        return timed_runs("pyspark", repeats, run_once)
    finally:
        spark.stop()


def sail_string(
    n: int, partitions: int, repeats: int, num_keys: int, num_tokens: int
) -> RunStats:
    from pysail.spark import SparkConnectServer
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    server = SparkConnectServer("127.0.0.1", 0)
    server.start(background=True)
    _, port = server.listening_address
    spark = SparkSession.builder.remote(f"sc://localhost:{port}").getOrCreate()

    def run_once() -> int:
        df = (
            spark.range(0, n, 1, partitions)
            .select(
                F.concat(
                    F.lit("word"),
                    (F.col("id") % F.lit(num_keys)).cast("string"),
                    F.lit(" tok"),
                    (F.col("id") % F.lit(num_tokens)).cast("string"),
                ).alias("line")
            )
            .select(F.explode(F.split(F.col("line"), F.lit(" "))).alias("w"))
            .groupBy("w")
            .count()
        )
        return int(df.count())

    try:
        return timed_runs("sail", repeats, run_once)
    finally:
        spark.stop()
        server.stop()


def print_table(title: str, rows: list[RunStats], baseline: str = "pyvega") -> None:
    order = {"pyvega": 0, "pyspark": 1, "sail": 2}
    rows = sorted(rows, key=lambda r: order.get(r.name, 99))
    base = next(r for r in rows if r.name == baseline)
    print(f"\n## {title}")
    print("")
    print("| engine | median_s | mean_s | count | vs_vega |")
    print("|---|---:|---:|---:|---:|")
    for r in rows:
        ratio = r.median / base.median if base.median > 0 else float("inf")
        vs = "1.00x" if r.name == baseline else f"{ratio:.2f}x"
        print(f"| {r.name} | {r.median:.4f} | {r.mean:.4f} | {r.result} | {vs} |")
    for r in rows:
        print(f"{r.name} times: {', '.join(f'{t:.4f}' for t in r.times)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Python comparison: Vega vs PySpark vs Sail.")
    parser.add_argument(
        "--workload",
        choices=("narrow", "shuffle", "float", "string", "both", "all"),
        default="all",
    )
    parser.add_argument("--n", type=int, default=5_000_000)
    parser.add_argument("--partitions", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--num-keys", type=int, default=1_000_000)
    parser.add_argument("--num-tokens", type=int, default=97)
    parser.add_argument("--build-dir", default="build-release")
    args = parser.parse_args()

    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    build_dir = os.path.join(repo, args.build_dir)

    workloads = set()
    if args.workload == "both":
        workloads = {"narrow", "shuffle"}
    elif args.workload == "all":
        workloads = {"narrow", "shuffle", "float", "string"}
    else:
        workloads = {args.workload}

    print(
        f"Config: workload={args.workload}, n={args.n}, partitions={args.partitions}, "
        f"workers={args.workers}, repeats={args.repeats}, num_keys={args.num_keys}, "
        f"num_tokens={args.num_tokens}"
    )

    rows_narrow: list[RunStats] = []
    rows_shuffle: list[RunStats] = []
    rows_float: list[RunStats] = []
    rows_float_shuffle: list[RunStats] = []
    rows_string: list[RunStats] = []

    if "shuffle" in workloads or "string" in workloads:
        print(
            "\nNote: Vega shuffle/string workloads use native reduceByKey over RPC."
        )

    # PySpark local JVM runs before Sail (Spark Connect) to avoid session conflicts.
    if "narrow" in workloads:
        rows_narrow.append(pyvega_narrow(args.n, args.partitions, args.workers, args.repeats))
        rows_narrow.append(pyspark_narrow(args.n, args.partitions, args.workers, args.repeats))

    if "shuffle" in workloads:
        rows_shuffle.append(
            pyspark_shuffle(args.n, args.partitions, args.workers, args.repeats, args.num_keys)
        )
        rows_shuffle.append(
            pyvega_shuffle(args.n, args.partitions, args.workers, args.repeats, args.num_keys)
        )

    if "float" in workloads:
        rows_float.append(pyvega_float(args.n, args.partitions, args.workers, args.repeats))
        rows_float.append(pyspark_float(args.n, args.partitions, args.workers, args.repeats))
        rows_float_shuffle.append(
            pyvega_float_shuffle(
                args.n, args.partitions, args.workers, args.repeats, args.num_keys
            )
        )
        rows_float_shuffle.append(
            pyspark_float_shuffle(
                args.n, args.partitions, args.workers, args.repeats, args.num_keys
            )
        )

    if "string" in workloads:
        rows_string.append(
            pyspark_string(
                args.n,
                args.partitions,
                args.workers,
                args.repeats,
                args.num_keys,
                args.num_tokens,
            )
        )
        rows_string.append(
            pyvega_string(
                args.n,
                args.partitions,
                args.workers,
                args.repeats,
                args.num_keys,
                args.num_tokens,
            )
        )

    if "narrow" in workloads:
        rows_narrow.append(sail_narrow(args.n, args.partitions, args.repeats))

    if "shuffle" in workloads:
        rows_shuffle.append(sail_shuffle(args.n, args.partitions, args.repeats, args.num_keys))

    if "float" in workloads:
        rows_float.append(sail_float(args.n, args.partitions, args.repeats))
        rows_float_shuffle.append(
            sail_float_shuffle(args.n, args.partitions, args.repeats, args.num_keys)
        )

    if "string" in workloads:
        rows_string.append(
            sail_string(
                args.n, args.partitions, args.repeats, args.num_keys, args.num_tokens
            )
        )

    if rows_narrow:
        if not all(r.result == rows_narrow[0].result for r in rows_narrow):
            raise RuntimeError(f"Narrow count mismatch: {[(r.name, r.result) for r in rows_narrow]}")
        print_table("Narrow int (map / filter / flatMap / count)", rows_narrow)

    if rows_shuffle:
        if not all(r.result == rows_shuffle[0].result for r in rows_shuffle):
            raise RuntimeError(f"Shuffle count mismatch: {[(r.name, r.result) for r in rows_shuffle]}")
        print_table("Shuffle int (keyed aggregate + count)", rows_shuffle)

    if rows_float:
        if not all(r.result == rows_float[0].result for r in rows_float):
            raise RuntimeError(f"Float count mismatch: {[(r.name, r.result) for r in rows_float]}")
        print_table("Float (map / filter / flatMap / count)", rows_float)

    if rows_float_shuffle:
        if not all(r.result == rows_float_shuffle[0].result for r in rows_float_shuffle):
            raise RuntimeError(
                f"Float shuffle count mismatch: {[(r.name, r.result) for r in rows_float_shuffle]}"
            )
        print_table("Float shuffle (keyed aggregate + count)", rows_float_shuffle)

    if rows_string:
        if not all(r.result == rows_string[0].result for r in rows_string):
            raise RuntimeError(f"String count mismatch: {[(r.name, r.result) for r in rows_string]}")
        print_table("String (split / word-count shuffle + count)", rows_string)


if __name__ == "__main__":
    main()
