#!/usr/bin/env python3
"""Sail counterpart of benchmarks/vega_cpp_dataframe.cpp.

Sail (https://github.com/lakehq/sail) is a Spark Connect server written in
Rust on Apache DataFusion; its DataFrame API is PySpark's Spark Connect
client. Sail does not cache DataFrames in memory yet (``cache()`` is a
no-op), so the tables are written once as uncompressed Parquet -- Sail's
native input, served from the OS page cache -- and every query scans them.

Needs ``pip install pysail pyspark-client pyarrow`` (pysail 0.7 accepts
PySpark Connect clients up to 4.0). Run through benchmarks/bench_dataframe.py,
or directly:

    BENCH_N=10000000 python benchmarks/sail_dataframe.py

Prints one "RESULT\\t<query>\\t<best ms>\\t<value>" line per query.
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pysail.spark import SparkConnectServer
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def write_tables(directory: Path, n: int, parts: int) -> tuple[str, str]:
    i = np.arange(n, dtype=np.int64)
    sales = pa.table({
        "id": i,
        "store": i % 1000,
        "product": (i * 7919) % 100000,
        "qty": i % 10 + 1,
        "price": ((i * 13) % 1000) / 10.0 + 1.0,
        "category": pa.array(np.char.add("cat", (i % 8).astype(str)).astype(object)),
    })
    s = np.arange(1000, dtype=np.int64)
    stores = pa.table({"store": s, "region": np.char.add("r", (s % 5).astype(str)).astype(object)})
    sales_path, stores_path = directory / "sales.parquet", directory / "stores.parquet"
    # One row group per partition, like the other engines' partitioning.
    pq.write_table(sales, sales_path, row_group_size=-(-n // parts), compression="none")
    pq.write_table(stores, stores_path, compression="none")
    return str(sales_path), str(stores_path)


def fmt(v) -> str:
    return repr(float(v)) if isinstance(v, float) else str(v)


def row(df) -> str:
    return "|".join(fmt(v) for v in df.collect()[0])


def main() -> None:
    n = int(os.environ.get("BENCH_N", "10000000"))
    repeat = int(os.environ.get("BENCH_REPEAT", "5"))
    warmup = int(os.environ.get("BENCH_WARMUP", "1"))
    parts = int(os.environ.get("BENCH_PARTITIONS", "8"))

    with tempfile.TemporaryDirectory() as tmp:
        sales_path, stores_path = write_tables(Path(tmp), n, parts)
        server = SparkConnectServer("127.0.0.1", 0)
        server.start(background=True)
        _, port = server.listening_address
        spark = SparkSession.builder.remote(f"sc://localhost:{port}").getOrCreate()
        spark.conf.set("spark.sql.shuffle.partitions", str(parts))
        sales = spark.read.parquet(sales_path)
        stores = spark.read.parquet(stores_path)
        col = F.col

        def top10() -> str:
            rows = sales.orderBy(col("price").desc(), col("id").asc()).limit(10).collect()
            return f"{len(rows)}|{sum(r['id'] for r in rows)}|{fmt(rows[0]['price'])}"

        queries = [
            ("filter+count", lambda: str(sales.filter((col("qty") > 5) & (col("price") < 50.0)).count())),
            ("sum(qty*price)", lambda: row(sales.agg(F.sum(col("qty") * col("price"))))),
            ("groupBy store (1k keys)", lambda: row(
                sales.groupBy("store").agg(F.sum("qty").alias("u"), F.avg("price").alias("a"))
                .agg(F.count(F.lit(1)), F.sum("u"), F.sum("a")))),
            ("groupBy category (string, 8 keys)", lambda: row(
                sales.groupBy("category")
                .agg(F.sum(col("qty") * col("price")).alias("rev"), F.count(F.lit(1)).alias("c"))
                .agg(F.count(F.lit(1)), F.sum("rev"), F.sum("c")))),
            ("groupBy product (100k keys)", lambda: row(
                sales.groupBy("product").agg(F.count(F.lit(1)).alias("c"), F.max("price").alias("m"))
                .agg(F.count(F.lit(1)), F.sum("c"), F.sum("m")))),
            ("groupBy category, qty (40 keys)", lambda: row(
                sales.groupBy("category", "qty").agg(F.count(F.lit(1)).alias("c"), F.sum("price").alias("p"))
                .agg(F.count(F.lit(1)), F.sum("c"), F.sum("p")))),
            ("join stores + groupBy region", lambda: row(
                sales.join(stores, ["store"]).groupBy("region").agg(F.sum("qty").alias("u"))
                .agg(F.count(F.lit(1)), F.sum("u")))),
            ("distinct (product, qty)", lambda: str(sales.select("product", "qty").distinct().count())),
            ("orderBy price desc limit 10", top10),
            ("left join + filter region", lambda: row(
                sales.join(stores, ["store"], "left").filter(col("region") == "r1")
                .agg(F.count(F.lit(1)), F.sum("qty")))),
            ("window top-3 per store", lambda: row(
                sales.withColumn("rn", F.row_number().over(
                    Window.partitionBy("store").orderBy(col("price").desc(), col("id").asc())))
                .filter(col("rn") <= 3).agg(F.count(F.lit(1)), F.sum("id")))),
            ("countDistinct product by category", lambda: row(
                sales.groupBy("category").agg(F.countDistinct("product").alias("d"))
                .agg(F.count(F.lit(1)), F.sum("d")))),
        ]

        for _ in range(warmup):
            for _, q in queries:
                q()
        for name, q in queries:
            best, value = float("inf"), ""
            for _ in range(repeat):
                t0 = time.perf_counter()
                value = q()
                best = min(best, time.perf_counter() - t0)
            print(f"RESULT\t{name}\t{best * 1000:.3f}\t{value}", flush=True)
        spark.stop()
        server.stop()


if __name__ == "__main__":
    main()
