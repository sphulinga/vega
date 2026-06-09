#!/usr/bin/env python3
from __future__ import annotations

import os
import statistics
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pysail.spark import SparkConnectServer


def run_once(spark: SparkSession, n: int, partitions: int) -> int:
    df = (
        spark.range(0, n, 1, partitions)
        .select(((F.col("id") * F.lit(31) + F.lit(7)) % F.lit(1_000_003)).alias("x"))
        .filter((F.col("x") % F.lit(7)) == F.lit(0))
        .select(F.explode(F.array(F.col("x"), F.col("x") + F.lit(1), F.col("x") + F.lit(2))).alias("y"))
    )
    return int(df.count())


def main() -> None:
    n = int(os.environ.get("BENCH_N", "5000000"))
    partitions = int(os.environ.get("BENCH_PARTITIONS", "32"))
    repeats = int(os.environ.get("BENCH_REPEATS", "5"))

    server = SparkConnectServer("127.0.0.1", 0)
    server.start(background=True)
    _, port = server.listening_address
    remote = f"sc://localhost:{port}"

    spark = SparkSession.builder.remote(remote).getOrCreate()
    try:
        # Warmup.
        run_once(spark, n, partitions)

        times: list[float] = []
        last = 0
        for _ in range(repeats):
            t0 = time.perf_counter()
            last = run_once(spark, n, partitions)
            times.append(time.perf_counter() - t0)

        print(
            "RESULT "
            f"count={last} "
            f"median_s={statistics.median(times)} "
            f"mean_s={statistics.mean(times)} "
            f"times_s={','.join(str(t) for t in times)}"
        )
    finally:
        spark.stop()
        server.stop()


if __name__ == "__main__":
    main()
