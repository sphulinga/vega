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
| `sc.range(start, end, num_partitions=0)` | Native int64 range `[start, end)` on server |
| `sc.float_range(start, end, num_partitions=0)` | Native float64 range |
| `sc.parallelize(data, num_partitions=0)` | Python (or native) collection |
| `sc.empty_rdd()` | Empty RDD |
| `sc.text_file(path, min_partitions=0)` | Read lines from a file |
| `sc.pickle_file(path, min_partitions=0)` | Load pickled Python objects |
| `sc.union([rdd1, rdd2, ...])` | Union of RDDs (collects each side) |
| `sc.broadcast(value)` | Broadcast variable for UDFs |

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
pairs.aggregate_by_key(0, seqOp, combOp)
pairs.fold_by_key(0, op)
```

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
```

### Joins and products

```python
rdd.zip(other)                # same partition count required
rdd.cartesian(other)
```

### I/O (actions that write files)

```python
rdd.save_as_text_file("/tmp/out")
rdd.save_as_pickle_file("/tmp/out.pkl")
```

### Pipes

```python
rdd.pipe("cat", env={"LC_ALL": "C"}, check_code=False)
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
| `foreach(func)` | `None` | Side effect per element |
| `sum()` | number | Numeric sum |
| `isEmpty()` | `bool` | |
| `histogram(buckets)` | `(counts, buckets)` | |
| `count_approx_distinct(relativeSD=0.05)` | `int` | HyperLogLog-style estimate |

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

## Limitations (current)

- **No Spark SQL / DataFrame API** in PyVega — see [DataFrame-style operations](dataframe-operations.md)
- **No SparkSession** in core PyVega — use RDD API or the PySpark compat shim for RDDs
- **`union`** collects each RDD to the driver (not a streaming union)
- **Cluster mode** (`vega_worker`) is not wired to PyVega clients yet
- Some PySpark RDD methods are only available through `scripts/pyspark_compat`

---

## API entry points

| Import | Use when |
|--------|----------|
| `from pyvega import SparkContext, RDD` | Native PyVega API |
| `from pyspark import SparkContext` (with compat on `PYTHONPATH`) | PySpark-compatible RDD code |

See [PySpark compatibility](pyspark-compat.md) for running upstream tests.
