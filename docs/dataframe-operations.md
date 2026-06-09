# DataFrame-style operations

PyVega today is an **RDD engine**. There is no Spark SQL catalog, no Catalyst optimizer,
and no `SparkSession.createDataFrame` in the core client.

This guide shows how to express common **DataFrame / SQL** patterns with RDDs, and
clarifies what is and is not available.

---

## Status summary

| Feature | PyVega | PySpark compat shim |
|---------|--------|---------------------|
| `SparkContext` / RDD | Yes | Yes |
| `SparkSession` | No | Stub only |
| `DataFrame` | No | No |
| SQL (`spark.sql(...)`) | No | No |
| Column expressions (`F.col`) | No | No |
| Spark Connect server | No | No (separate `pysail` benchmarks) |

For SQL-heavy workloads you still need Apache Spark (JVM) or another engine. For
distributed Python data processing with UDFs, use the RDD API below.

---

## Mental model: DataFrame → RDD

| SQL / DataFrame concept | RDD equivalent |
|-------------------------|----------------|
| `spark.range(n)` | `sc.range(0, n)` |
| `spark.createDataFrame(rows, schema)` | `sc.parallelize(rows)` (tuples or dicts) |
| `df.select("col")` | `rdd.map(lambda row: row["col"])` or `rdd.map(lambda r: r[0])` |
| `df.filter(df.x > 0)` | `rdd.filter(lambda row: row["x"] > 0)` |
| `df.withColumn("y", expr)` | `rdd.map(lambda row: {**row, "y": f(row)})` |
| `df.groupBy("k").sum("v")` | See [grouped aggregation](#grouped-aggregation) |
| `df.orderBy("k")` | `rdd.sort_by(lambda row: row["k"])` |
| `df.repartition(n)` | `rdd.repartition(n)` |
| `df.count()` | `rdd.count()` |
| `df.collect()` | `rdd.collect()` |
| `df.show()` | `print(rdd.take(20))` |

---

## Example: range and projection

**DataFrame (Spark SQL — not available in PyVega):**

```python
df = spark.range(0, 100).select((col("id") * 2).alias("x"))
df.show()
```

**RDD (PyVega):**

```python
from pyvega import SparkContext

with SparkContext(4) as sc:
    result = (
        sc.range(0, 100)
        .map(lambda x: (x, x * 2))      # (id, x)
        .take(20)
    )
    for row in result:
        print(row)
```

---

## Example: rows as records

Represent table rows as **dicts** or **named tuples**:

```python
from pyvega import SparkContext

rows = [
    {"name": "Alice", "dept": "eng", "salary": 120_000},
    {"name": "Bob",   "dept": "eng", "salary": 110_000},
    {"name": "Carol", "dept": "sales", "salary": 95_000},
]

with SparkContext(4) as sc:
    eng = (
        sc.parallelize(rows)
        .filter(lambda r: r["dept"] == "eng")
        .map(lambda r: r["name"])
        .collect()
    )
    print(eng)   # ['Alice', 'Bob']
```

---

## Grouped aggregation

**DataFrame:**

```python
df.groupBy("dept").agg(F.sum("salary").alias("total"))
```

**RDD — key-value then reduceByKey:**

```python
with SparkContext(4) as sc:
    totals = (
        sc.parallelize(rows)
        .map(lambda r: (r["dept"], r["salary"]))
        .reduce_by_key(lambda a, b: a + b)
        .collect()
    )
    print(totals)   # [('eng', 230000), ('sales', 95000)]
```

**RDD — groupByKey (all values per key):**

```python
by_dept = (
    sc.parallelize(rows)
    .map(lambda r: (r["dept"], r["salary"]))
    .group_by_key()
    .map(lambda kv: (kv[0], sum(kv[1])))
    .collect()
)
```

---

## Example: shuffle benchmark pattern

The repo's `benchmarks/sail_shuffle_benchmark.py` uses **Spark Connect + DataFrames**
against an external Sail server. The equivalent **PyVega RDD** version:

```python
from pyvega import SparkContext

def run_shuffle(sc, n=1_000_000, partitions=8, num_keys=100_000):
    return (
        sc.range(0, n, num_partitions=partitions)
        .map(lambda x: ((x * 1315423911 + 7) % num_keys, (x * 31 + 7) % 97))
        .reduce_by_key(lambda a, b: a + b)
        .count()
    )

with SparkContext(4) as sc:
    print(run_shuffle(sc))
```

This performs keyed shuffle + sum in one action, similar to
`groupBy("k").agg(sum("v"))` on a derived key column.

---

## Example: word count (SQL vs RDD)

**SQL:**

```sql
SELECT word, COUNT(*) AS c
FROM lines LATERAL VIEW explode(split(line, ' ')) t AS word
GROUP BY word
ORDER BY c DESC
```

**RDD:**

```python
from pyvega import SparkContext

with SparkContext(4) as sc:
    counts = (
        sc.text_file("README.md")
        .flat_map(lambda line: line.split())
        .map(lambda w: (w.strip(".,()"), 1))
        .filter(lambda kv: kv[0])
        .reduce_by_key(lambda a, b: a + b)
        .sort_by(lambda kv: -kv[1])
        .take(10)
    )
    for word, count in counts:
        print(f"{word}: {count}")
```

---

## Example: join two tables

**DataFrame:**

```python
a.join(b, on="id")
```

**RDD (keyed join via cartesian + filter — fine for small data):**

```python
def join_by_key(a_rdd, b_rdd, key_fn=lambda x: x["id"]):
    a = a_rdd.map(lambda r: (key_fn(r), r))
    b = b_rdd.map(lambda r: (key_fn(r), r))
    return (
        a.cartesian(b)
        .filter(lambda ab: ab[0][0] == ab[1][0])
        .map(lambda ab: (ab[0][1], ab[1][1]))
    )
```

For large joins, prefer **reduceByKey** / cogroup-style patterns after partitioning
both sides by key (`partition_by`, `reduce_by_key`).

---

## Using PySpark RDD syntax

If you prefer `pyspark` imports, enable the compat layer:

```python
import sys, os
sys.path[:0] = ["python", "scripts/pyspark_compat"]

from pyspark.context import SparkContext

sc = SparkContext()
try:
    df_like = sc.parallelize([("a", 1), ("b", 2)])
    print(df_like.map(lambda kv: (kv[0], kv[1] * 10)).collect())
finally:
    sc.stop()
```

This does **not** add DataFrame support — only RDD methods.

---

## Roadmap note

Full DataFrame support would require:

- A query planner (or Spark Connect protocol handler)
- Columnar batch execution or JVM delegation
- Schema inference and encoders

Today, **PyVega + vega_server** focus on high-performance **RDD** execution with native
fast paths and embedded Python UDFs. Use the patterns above until SQL/DataFrame APIs
land.

For benchmarks that compare against Spark DataFrames, see `benchmarks/compare_python_all.py`
and `benchmarks/sail_shuffle_benchmark.py` (external Spark Connect / JVM Spark).
