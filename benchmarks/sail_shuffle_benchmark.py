#!/usr/bin/env python3
from __future__ import annotations

import os
import statistics
import time

from pysail.spark import SparkConnectServer
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def run_once(spark: SparkSession, n: int, partitions: int, num_keys: int) -> int:
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


def main() -> None:
    n = int(os.environ.get("BENCH_N", "20000000"))
    partitions = int(os.environ.get("BENCH_PARTITIONS", "64"))
    repeats = int(os.environ.get("BENCH_REPEATS", "3"))
    num_keys = int(os.environ.get("BENCH_NUM_KEYS", "1000000"))

    server = SparkConnectServer("127.0.0.1", 0)
    server.start(background=True)
    _, port = server.listening_address
    spark = SparkSession.builder.remote(f"sc://localhost:{port}").getOrCreate()

    try:
        run_once(spark, n, partitions, num_keys)  # warmup

        times: list[float] = []
        last_count = 0
        for _ in range(repeats):
            t0 = time.perf_counter()
            last_count = run_once(spark, n, partitions, num_keys)
            times.append(time.perf_counter() - t0)

        print(
            "RESULT "
            f"count={last_count} "
            f"median_s={statistics.median(times)} "
            f"mean_s={statistics.mean(times)} "
            f"times_s={','.join(str(t) for t in times)}"
        )
    finally:
        spark.stop()
        server.stop()


if __name__ == "__main__":
    main()
