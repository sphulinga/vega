# Vega extras: beyond Spark's RDD API

PyVega implements Spark's RDD API and adds a few operations that Spark users
commonly hand-roll. All of them run on the server like any other RDD operation.

| Operation | What it does | Why it's not just Spark |
|-----------|--------------|-------------------------|
| `rdd.explain()` | Shows pending stages, which lambdas were **compiled to native C++** vs run as Python UDFs, shuffle boundaries, and cache status | Spark's `toDebugString` shows lineage but not *how* each step runs |
| `rdd.map_batches(func, batch_size=1024)` | Vectorized UDF: `func(list_of_elements) -> iterable` | Spark has vectorized UDFs only for DataFrames (pandas UDFs) |
| `rdd.distinct_by(keyfunc)` | One element per key, deduplicated map-side | Spark needs `keyBy + reduceByKey + values` |
| `pairs.top_by_key(n, key=None)` | Per-key top-N with map-side heaps | No per-key top-N on Spark RDDs |
| `rdd.quantiles(probs, relative_error=0.0)` | Exact (distributed sort) or approximate (single-pass reservoir) quantiles | `approxQuantile` exists only on DataFrames |
| `rdd.heavy_hitters(k)` | Most frequent elements using mergeable Misra-Gries summaries, O(k) memory per partition, then an exact recount of candidates | `countByValue` returns *every* value to the driver |
| `rdd.sliding(window, step=1)` | Sliding windows over the RDD's order | Only in Scala MLlib (`RDDFunctions.sliding`) |
| `with rdd.persisted(): ...` | Persist for the duration of a block, then unpersist automatically | Manual `cache()`/`unpersist()` is easy to leak |
| `rdd.cache_info()` | How many partitions are actually cached | Spark exposes this only in the web UI |
| `rdd.save_as_jsonl(path)` / `sc.jsonl_file(path)` | JSON-lines I/O | RDDs have no JSON I/O in Spark |
| `sc.status()` / `sc.job_history` | Count and wall time of recent jobs | Spark exposes this via the UI/REST API |

## Examples

```python
from pyvega import SparkContext

with SparkContext(4) as sc:
    events = sc.textFile("events/*.jsonl.gz").map(json.loads)

    # Which steps run natively?
    print(sc.range(0, 10**6).filter(lambda x: x % 3 == 0).map(str).explain())
    # RDD[2]
    #   source: server RDD #1 (int64, 4 partitions)
    #   1. filter: native C++ filter_mod_eq(3, 0)
    #   2. map: Python UDF
    #   storage: Serialized 1x Replicated

    # Vectorized scoring with NumPy, one call per 4096 rows.
    scores = events.map_batches(lambda rows: model.predict(featurize(rows)), batch_size=4096)

    # Latest event per user, top-3 purchases per user.
    latest = events.distinct_by(lambda e: e["user"])
    top3 = events.map(lambda e: (e["user"], e["amount"])).top_by_key(3)

    # p50 / p99 latency in a single pass, ~1% rank error.
    p50, p99 = events.map(lambda e: e["latency_ms"]).quantiles([0.5, 0.99], relative_error=0.01)

    # The 10 noisiest endpoints without shipping every count to the driver.
    hot = events.map(lambda e: e["path"]).heavy_hitters(10)

    # 3-step moving windows (in RDD order).
    windows = sc.parallelize(range(10), 3).sliding(3).collect()

    with events.persisted() as cached:      # unpersisted on exit
        n = cached.count()
        errors = cached.filter(lambda e: e["status"] >= 500).count()

    print(sc.status())
```
