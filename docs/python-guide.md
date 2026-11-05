# Python RDD guide (PyVega)

PyVega exposes a Spark-like **RDD** API. Transformations are lazy; actions trigger
execution on the C++ server.

```python
from pyvega import SparkContext, RDD
```

Both camelCase (`flatMap`) and snake_case (`flat_map`) names are available on most methods.

---

## SparkContext

### Creating a context

```python
# Local: spawn server with 4 worker threads
sc = SparkContext(4)

# Remote: connect to running vega_server
sc = SparkContext(remote="127.0.0.1:9500")

# Context manager (recommended)
with SparkContext(4) as sc:
    ...
```

### Creating RDDs

| Method | Description |
|--------|-------------|
| `sc.range(start, end, num_partitions=0, step=1)` | Native int64 range on server |
| `sc.float_range(start, end, num_partitions=0)` | Native float64 range |
| `sc.parallelize(data, num_partitions=0)` | Python (or native) collection; `range(...)` is generated server-side |
| `sc.empty_rdd()` | Empty RDD |
| `sc.text_file(path, min_partitions=None)` | Lines of files/dirs/globs (split, read by server tasks) |
| `sc.wholeTextFiles(path)` / `sc.binaryFiles(path)` | `(path, content)` per file |
| `sc.pickle_file(path, min_partitions=None)` | Load pickled Python objects |
| `sc.union([rdd1, rdd2, ...])` | Union of RDDs (server-side, no data movement) |
| `sc.broadcast(value)` | Broadcast variable for UDFs |
| `sc.accumulator(value)` | Add-only shared variable (see below) |
| `sc.runJob(rdd, func, partitions)` | Run a partition function on selected partitions |

`num_partitions=0` defaults to `sc.num_workers`.

```python
with SparkContext(4) as sc:
    ints = sc.range(0, 1_000_000)           # fast native path
    words = sc.parallelize(["a", "b", "c"])
    lines = sc.text_file("/path/to/log.txt")
```

---

## Transformations

### Element-wise

```python
rdd.map(lambda x: x * 2)
rdd.filter(lambda x: x > 0)
rdd.flat_map(lambda x: x.split())
rdd.distinct(num_partitions=0)
```

### Key-value (shuffle)

```python
pairs = sc.parallelize([("a", 1), ("b", 2), ("a", 3)])

pairs.reduce_by_key(lambda a, b: a + b)   # shuffle + combine
pairs.group_by_key()                       # shuffle + group into lists
pairs.combineByKey(create, merge_value, merge_combiners)  # map-side combine
pairs.aggregate_by_key(0, seqOp, combOp)   # built on combineByKey
pairs.fold_by_key(0, op)
pairs.mapValues(f); pairs.flatMapValues(f); pairs.keys(); pairs.values()
rdd.keyBy(f); rdd.groupBy(f)
```

Shuffles can be chained freely (`reduceByKey(...).map(...).groupByKey()` runs as a
three-stage job), and a shuffle's output is reused by later actions on the same
RDD (Spark's "skipped stages").

### Partition operations

```python
rdd.map_partitions(lambda it: [sum(it)])
rdd.map_partitions_with_index(lambda idx, it: [(idx, len(list(it)))])
rdd.glom()                    # one list per partition
rdd.coalesce(n, shuffle=False)
rdd.repartition(n)            # shuffle=True coalesce
rdd.partition_by(n, partition_func)
```

### Sorting and sampling

```python
rdd.sort_by(lambda x: x, ascending=True)
pairs.sort_by_key(ascending=True, numPartitions=4)
rdd.sample(withReplacement=False, fraction=0.1, seed=42)
rdd.take_sample(withReplacement=False, num=10, seed=42)
pairs.sampleByKey(False, {"a": 0.1, "b": 0.5}, seed=1)
train, test = rdd.randomSplit([0.8, 0.2], seed=7)
```

### Joins, set operations and products

```python
a.join(b); a.leftOuterJoin(b); a.rightOuterJoin(b); a.fullOuterJoin(b)
a.cogroup(b); a.groupWith(b, c)            # (key, (ResultIterable, ...))
a.subtractByKey(b)
rdd.union(other)  # or rdd + other, or sc.union([...]) -- no data movement
rdd.intersection(other); rdd.subtract(other); rdd.distinct()
rdd.zip(other)                # same partition count required
rdd.zipWithIndex(); rdd.zipWithUniqueId()
rdd.cartesian(other)
```

### I/O

Files are read and written **by server tasks**, in parallel, like Spark executors.

```python
sc.textFile("data/*.txt,more.log.gz", minPartitions=8)  # globs, dirs, .gz/.bz2/.xz
sc.wholeTextFiles("docs/")            # (path, content) per file
sc.binaryFiles("images/")             # (path, bytes) per file
sc.pickleFile("/tmp/out.pkl")
rdd.save_as_text_file("/tmp/out")     # part-NNNNN files + _SUCCESS
rdd.saveAsTextFile("/tmp/gz", "org.apache.hadoop.io.compress.GzipCodec")
rdd.save_as_pickle_file("/tmp/out.pkl")
rdd.save_as_jsonl("/tmp/json"); sc.jsonl_file("/tmp/json")   # Vega extra
```

Large text files are split into byte ranges (Hadoop line-record semantics), so
one file is read by many tasks.

### Pipes

```python
rdd.pipe("cat", env={"LC_ALL": "C"}, checkCode=False)   # runs on the server
```

---

## Actions

| Action | Returns | Notes |
|--------|---------|-------|
| `collect()` | `list` | All elements to driver |
| `count()` | `int` | Element count |
| `first()` | element | First element |
| `take(n)` | `list` | First n elements |
| `reduce(func)` | aggregated | Binary op must be associative |
| `fold(zero, op)` | aggregated | Like reduce with zero |
| `aggregate(z, seqOp, combOp)` | aggregated | Per-partition seq + combine |
| `tree_aggregate(z, seqOp, combOp, depth=2)` | aggregated | Tree combine |
| `tree_reduce(func, depth=2)` | aggregated | Tree combine |
| `foreach(func)` / `foreachPartition(func)` | `None` | Side effects (run on the server) |
| `sum()`, `mean()`, `max()`, `min()` | number | Native C++ for int/float RDDs |
| `stats()` | `StatCounter` | count, mean, stdev, variance, min, max in one pass |
| `variance()`, `stdev()`, `sampleVariance()`, `sampleStdev()` | `float` | |
| `top(n, key=None)` / `takeOrdered(n, key=None)` | `list` | Per-partition heaps |
| `countByKey()` / `countByValue()` | `dict` | |
| `collectAsMap()` / `lookup(key)` | `dict` / `list` | |
| `isEmpty()` | `bool` | Scans one partition |
| `histogram(buckets)` | `(buckets, counts)` | |
| `count_approx_distinct(relativeSD=0.05)` | `int` | HyperLogLog, server-side |
| `countApprox(timeout)`, `meanApprox`, `sumApprox` | number | Stops after `timeout` ms and extrapolates |
| `toLocalIterator(prefetchPartitions=False)` | iterator | One partition at a time |

`take(n)` / `first()` scan as few partitions as possible (1, then 4x more...),
like Spark, instead of computing the whole RDD.

```python
with SparkContext(4) as sc:
    total = sc.range(0, 100).sum()
    mx = sc.range(0, 100).reduce(max)
    sc.range(0, 100).foreach(lambda x: print(x, end=" "))
```

---

## Lazy evaluation and staging

Transformations append **pending stages** on the client. Nothing runs on the server
until an action:

```python
rdd = sc.range(0, 10).map(lambda x: x + 1).filter(lambda x: x % 2 == 0)
# No server work yet

evens = rdd.collect()   # single fused pipeline sent to server
```

Native lambdas (simple arithmetic, `(x % k, x)` pairs, etc.) are compiled to C++
ops client-side for speed. Other functions are cloudpickled and executed in embedded
Python on the server.

---

## Persistence

```python
rdd = sc.textFile("big.txt").map(parse).cache()   # or .persist(StorageLevel.MEMORY_ONLY)
rdd.count()          # computes and stores partitions in server memory
rdd.filter(f).count()  # served from the cache, upstream stages skipped
rdd.cache_info()     # {'kind': 'python', 'partitions': 8, 'cached_partitions': 8}
rdd.unpersist()

with rdd.persisted() as cached:   # Vega extra: auto-unpersist
    cached.count(); cached.sum()
```

Server-side RDD handles are reference-counted: when the last Python `RDD` object
for a server RDD is garbage-collected, the server frees it (and its cached data).

---

## Accumulators

```python
errors = sc.accumulator(0)
rdd.foreach(lambda line: errors.add(1) if "ERROR" in line else None)
print(errors.value)
```

Tasks can only `add`; reading `.value` inside a task raises. Updates are merged
into the driver after every job. Custom types: pass an `AccumulatorParam`.

---

## Broadcast variables

```python
with SparkContext(4) as sc:
    lookup = sc.broadcast({"a": 1, "b": 2})

    result = sc.parallelize(["a", "b", "a"]).map(
        lambda k: lookup.value[k]
    ).collect()
```

Broadcasts are registered in the session and unpickled on the server when UDFs run.

---

## Example: reduce-by-key aggregation

```python
from pyvega import SparkContext

with SparkContext(4) as sc:
    sales = sc.parallelize([
        ("east", 100),
        ("west", 200),
        ("east", 50),
        ("west", 75),
    ])

    by_region = (
        sales
        .reduce_by_key(lambda a, b: a + b)
        .sort_by_key()
        .collect()
    )
    print(by_region)
    # [('east', 150), ('west', 275)]
```

---

## Example: group-by-key

```python
with SparkContext(4) as sc:
    grouped = (
        sc.range(0, 12)
        .map(lambda x: (x % 3, x))
        .group_by_key()
        .map(lambda kv: (kv[0], sum(kv[1])))
        .collect()
    )
    print(grouped)
```

---

## Example: read, transform, write

```python
with SparkContext(4) as sc:
    (
        sc.text_file("input.txt")
        .flat_map(lambda line: line.split())
        .map(lambda w: (w.lower(), 1))
        .reduce_by_key(lambda a, b: a + b)
        .filter(lambda kv: kv[1] > 5)
        .save_as_text_file("output/")
    )
```

Text output writes one file per partition under `output/part-00000`, etc.

---

## Beyond Spark

PyVega adds a few RDD operations Spark does not have -- `explain()`,
`map_batches`, `distinct_by`, `top_by_key`, `quantiles`, `heavy_hitters`,
`sliding`, `persisted()`, JSON-lines I/O and `sc.status()`. See
[Vega extras](vega-extras.md).

---

## Limitations (current)

- **No Spark SQL / DataFrame API** in PyVega — see [DataFrame-style operations](dataframe-operations.md)
- **No SparkSession** in core PyVega — use RDD API or the PySpark compat shim for RDDs
- **Cluster mode** (`vega_worker`) is not wired to PyVega clients yet
- **`cancelJobGroup` / `cancelAllJobs`** are best-effort: running jobs finish
- Python UDFs share one interpreter in the server, so they run one at a time
  (GIL); native-compiled lambdas and I/O-bound UDFs run in parallel

---

## API entry points

| Import | Use when |
|--------|----------|
| `from pyvega import SparkContext, RDD` | Native PyVega API |
| `from pyspark import SparkContext` (with compat on `PYTHONPATH`) | PySpark-compatible RDD code |

See [PySpark compatibility](pyspark-compat.md) for running upstream tests.
