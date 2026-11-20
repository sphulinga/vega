# Vega

An in-memory distributed data processing engine written in C++17. Vega
provides a Spark-like RDD API with lazy evaluation, but replaces Spark's
centralized driver/scheduler with a **decentralized actor model** where
workers coordinate through message passing.

**PyVega** is the Python client: it connects to a C++ **session server** over TCP,
records transformations lazily, and executes them on an embedded actor runtime with
Python UDF support.

## Documentation

| Guide | Description |
|-------|-------------|
| [**Docs index**](docs/README.md) | Start here |
| [Getting started](docs/getting-started.md) | Build, install, first program |
| [Server, workers, and clients](docs/server-and-workers.md) | `vega_server`, worker threads, cluster peers |
| [Python RDD guide](docs/python-guide.md) | PyVega API reference with examples |
| [C++ DataFrames](docs/cpp-dataframes.md) | Native columnar DataFrames: expressions, group-by, joins, top-N |
| [DataFrame-style operations](docs/dataframe-operations.md) | SQL-like patterns using RDDs from Python |
| [PySpark compatibility](docs/pyspark-compat.md) | Run PySpark RDD code and upstream tests |
| [Vega extras](docs/vega-extras.md) | Operations beyond Spark's RDD API |
| [Releasing](docs/releasing.md) | Building wheels and publishing to PyPI |

## Quick start (Python)

```bash
pip install pyvega
```

Wheels for Linux (x86_64, aarch64) and macOS (arm64, x86_64), CPython 3.9-3.13,
include the compiled C++ server, so there is nothing else to build. From a source
checkout, `pip install .` compiles it (needs CMake >= 3.16 and a C++17 compiler).

```python
from pyvega import SparkContext

with SparkContext(4) as sc:
    print(sc.range(0, 10).map(lambda x: x * 2).collect())
    # [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
```

**Manual server:** `vega-server --bind 127.0.0.1:9500` then
`SparkContext(remote="127.0.0.1:9500")`. See [server and clients](docs/server-and-workers.md).

## Performance

Measured with [`benchmarks/bench_pyvega_ops.py`](benchmarks/bench_pyvega_ops.py)
on one machine (Linux, 4 vCPU Intel Xeon @ 2.3 GHz, 4 workers). Each version
was built from its release tag and run with its own client and server; runs
were interleaved across versions, and each cell is the best time over 8 runs.

| Operation (200k elements, 4 workers) | v0.5.0 (ms) | v1.0.0 (ms) | v1.1.0 (ms) | v1.1.0 vs v0.5.0 |
|---|---:|---:|---:|---:|
| parallelize+collect (tuples) | 117.4 | 108.7 | 106.4 | 1.1x |
| map+filter python (tuples) | 188.0 | 30.6 | 30.4 | 6.2x |
| reduceByKey (tuples) | 103.8 | 11.8 | 11.7 | 8.9x |
| groupByKey (tuples) | 103.3 | 8.7 | 10.0 | 10x |
| distinct (strings) | 2844.0 | 42.4 | 42.6 | 67x |
| sortByKey (tuples) | 568.1 | 52.3 | 55.4 | 10x |
| mapPartitions | 180.1 | 5.9 | 5.7 | 32x |
| union x2 | 307.9 | 0.3 | 0.3 | 906x |
| repartition(16) | 171.8 | 2.8 | 3.0 | 57x |
| aggregateByKey | 156.0 | 20.1 | 22.9 | 6.8x |
| take(10) | 119.3 | 1.4 | 1.2 | 97x |
| first() | 114.8 | 1.2 | 1.1 | 102x |
| sum (ints) | 107.2 | 1.3 | 1.3 | 81x |
| aggregate | 55.2 | 20.3 | 20.9 | 2.6x |
| chained shuffle | failed | 12.0 | 11.9 | fixed |

v1.1.0 only changed the shutdown path (`ActorSystem::stop()`), so it performs
the same as v1.0.0. Differences of 10-15% between them in this table are
measurement noise: for example, 240 isolated `groupByKey` runs per version
gave a fastest time of 8.5 ms for both, and a median of 11.2 ms (v1.0.0)
versus 10.4 ms (v1.1.0).

### Compared with Apache Spark

Two like-for-like comparisons with Apache Spark 4.3.0, on the same machine
(Spark: Scala 2.13, Java 21, `local[4]`; Vega: 4 worker threads; 8 partitions
on both). Each pair runs the same operations on the same data, and every
operation reports its result (a count or sum): the results matched exactly
between the two engines in every run, so both sides did identical work.
Both sides get one untimed warm-up pass (the JVM's JIT needs it), then each
cell is the best of 3 repetitions over 5 runs (3 at the larger size), with
runs alternating between the two engines.

**Native APIs: Vega's C++ API vs Spark's Scala RDD API.** Only the operations
Vega's C++ API offers ([`benchmarks/vega_cpp_ops.cpp`](benchmarks/vega_cpp_ops.cpp)
and [`benchmarks/spark_scala_ops.scala`](benchmarks/spark_scala_ops.scala);
data: `(i % 1000, i)` pairs and integers):

| Operation | Vega C++, 1M (ms) | Spark Scala, 1M (ms) | Vega C++ vs Spark Scala | Vega C++, 10M (ms) | Spark Scala, 10M (ms) | Vega C++ vs Spark Scala |
|---|---:|---:|---:|---:|---:|---:|
| parallelize+collect (pairs) | 5.2 | 340.5 | 65× faster | 170.5 | 5059.0 | 30× faster |
| map+filter (pairs) | 1.1 | 205.2 | 188× faster | 42.3 | 3592.4 | 85× faster |
| flatMap x3 (ints) | 2.9 | 41.6 | 14× faster | 30.6 | 272.0 | 8.9× faster |
| reduce sum (ints) | 2.0 | 19.9 | 10× faster | 24.7 | 116.8 | 4.7× faster |
| reduceByKey (pairs) | 8.0 | 249.1 | 31× faster | 215.9 | 3384.5 | 16× faster |
| groupByKey (pairs) | 7.8 | 313.1 | 40× faster | 212.1 | 4043.5 | 19× faster |
| chained shuffle | 8.3 | 268.4 | 32× faster | 211.1 | 3362.9 | 16× faster |
| narrow pipeline (ints) | 1.5 | 32.3 | 22× faster | 19.8 | 190.3 | 9.6× faster |

**Python APIs: PyVega vs PySpark.** The same Python code runs on both
([`benchmarks/bench_python_ops.py`](benchmarks/bench_python_ops.py)
`--engine pyvega|pyspark`):

| Operation | PyVega, 200k (ms) | PySpark, 200k (ms) | PyVega vs PySpark | PyVega, 2M (ms) | PySpark, 2M (ms) | PyVega vs PySpark |
|---|---:|---:|---:|---:|---:|---:|
| parallelize+collect (tuples) | 111.9 | 63.9 | 1.8× slower | 1546.4 | 461.2 | 3.4× slower |
| map+filter python (tuples) | 30.5 | 65.7 | 2.2× faster | 376.6 | 177.5 | 2.1× slower |
| reduceByKey (tuples) | 11.7 | 210.9 | 18× faster | 121.9 | 330.8 | 2.7× faster |
| groupByKey (tuples) | 8.6 | 173.7 | 20× faster | 101.1 | 363.9 | 3.6× faster |
| distinct (strings) | 42.5 | 234.2 | 5.5× faster | 438.0 | 361.7 | 1.2× slower |
| sortByKey (tuples) | 53.0 | 305.9 | 5.8× faster | 648.9 | 874.8 | 1.3× faster |
| mapPartitions | 5.9 | 55.5 | 9.4× faster | 67.1 | 105.2 | 1.6× faster |
| union x2 † | 0.38 | 82.8 | 218× faster | 0.42 | 164.7 | 392× faster |
| repartition(16) | 3.5 | 227.2 | 65× faster | 47.5 | 555.5 | 12× faster |
| aggregateByKey | 20.6 | 204.8 | 9.9× faster | 213.9 | 330.0 | 1.5× faster |
| take(10) | 1.3 | 12.2 | 9.3× faster | 10.6 | 16.4 | 1.5× faster |
| first() | 1.2 | 12.6 | 10× faster | 9.6 | 17.1 | 1.8× faster |
| sum (ints) | 1.4 | 41.2 | 29× faster | 12.1 | 55.1 | 4.5× faster |
| aggregate | 20.5 | 50.9 | 2.5× faster | 224.2 | 126.1 | 1.8× slower |
| chained shuffle | 12.1 | 324.0 | 27× faster | 124.5 | 451.1 | 3.6× faster |

How to read these:

- **Single machine, in-memory data.** Spark is built for fault-tolerant
  clusters: it schedules every job as distributed tasks, serializes records,
  and always writes shuffle data to local disk. Vega runs in one process and
  shuffles in memory, with no serialization or disk I/O. Spark scales out to a cluster; Vega's
  Python API does not yet. The gap narrows as data grows, as Spark's fixed
  per-job costs are spread over more records.
- **Native gaps are largest on tuples.** Spark's JVM stores each `(Int,
  Double)` pair as a boxed object; Vega's C++ API stores a plain 16-byte
  struct.
- **PySpark wins Python-heavy work at larger sizes.** PySpark runs Python
  functions in one worker process per core, while PyVega runs them in a
  single interpreter, effectively on one core because of the GIL. At 2M
  elements that makes PySpark faster on `map+filter`, `distinct` and
  `aggregate`. `parallelize+collect` favours PySpark at both sizes because
  PyVega converts every element in Python on the client.
- **† `union x2` is not the same work.** PyVega answers `count()` on a union
  of in-memory collections from stored partition sizes without touching any
  element; PySpark iterates them.

To reproduce (`pip install pyspark` provides Spark, which needs Java 17+):

```bash
cmake --build build-release --target vega_cpp_ops
python benchmarks/bench_native_ops.py --engine vega-cpp --json vega_cpp.json
python benchmarks/bench_native_ops.py --engine spark-scala --json spark_scala.json
python benchmarks/bench_python_ops.py --engine pyvega --warmup 1 --json pyvega.json
python benchmarks/bench_python_ops.py --engine pyspark --warmup 1 --json pyspark.json
```

Where the gains come from:

- **Everything runs on the server.** `distinct`, `union`, `partitionBy`,
  `repartition`, `coalesce`, `sortByKey`, `mapPartitions`, `zip`, `cartesian`,
  joins, sampling and file I/O used to pull data back into the client and
  re-upload it. They are now server-side operators.
- **No 40 ms stall per RPC.** Server sockets use `TCP_NODELAY` and send each
  frame with one `sendmsg` call. Before, Nagle's algorithm plus delayed ACKs
  added about 44 ms to every request.
- **Partial jobs.** `take`/`first`/`isEmpty` scan only as many partitions as
  needed, and `toLocalIterator` computes one partition at a time.
- **Multi-stage DAG with shuffle reuse.** Chained shuffles work, and a
  materialized shuffle is reused by later actions (Spark's skipped stages).
- **Map-side combine** for `reduceByKey`/`combineByKey`/`aggregateByKey`.
- **Cheaper UDF calls.** Python functions are invoked through the C API
  (vectorcall) without a wrapper frame, and data is pickled with the C pickler.
- **Native numeric actions.** `sum`/`mean`/`stats`/`max`/`min` on int/float RDDs
  run in C++ (integer sums are exact, using 128-bit accumulators).

### DataFrames: Vega C++ vs Spark SQL vs Sail

This compares Vega's native C++ DataFrames ([C++ DataFrames](docs/cpp-dataframes.md))
with two other engines:

- [Spark SQL](https://spark.apache.org/sql/) 4.3.0, through the Scala DataFrame API.
- [Sail](https://github.com/lakehq/sail) 0.7.3, a Rust Spark Connect server built
  on DataFusion, driven through the PySpark Connect client.

All three run the same twelve queries over the same generated 10M-row `sales`
table (`id`, `store`, `product`, `qty`, `price`, `category` string) and a
1,000-row `stores` table.

The setup:

- Every engine gets 4 cores and 8 partitions, also used as the shuffle
  partition count (`spark.sql.shuffle.partitions=8`).
- Each cell is the best of 5 repetitions after a warm-up pass, over 2 rounds
  with the engines run in rotating order.
- Every query returns numbers that the runner checks across engines, and they
  matched: integers exactly, floating-point sums to within 1e-9 relative.

Lower is better; the fastest engine for each query is in bold.

| Query (10M rows) | Vega C++ | Spark SQL | Sail |
|------------------|---------:|----------:|-----:|
| `filter(qty > 5 && price < 50).count()` | **20 ms** | 146 ms | 22 ms |
| `agg(sum(qty * price))` | **13 ms** | 133 ms | 16 ms |
| `groupBy(store)` (1k keys): `sum`, `avg` | **19 ms** | 230 ms | 34 ms |
| `groupBy(category)` (string, 8 keys): `sum(qty * price)`, `count` | **14 ms** | 435 ms | 48 ms |
| `groupBy(product)` (100k keys): `count`, `max` | **57 ms** | 392 ms | 87 ms |
| `groupBy(category, qty)` (40 keys): `count`, `sum` | **36 ms** | 462 ms | 64 ms |
| `join(stores, "store").groupBy(region).sum(qty)` | **14 ms** | 399 ms | 52 ms |
| `select(product, qty).distinct().count()` | **65 ms** | 397 ms | 90 ms |
| `orderBy(price desc, id).limit(10)` | **11 ms** | 247 ms | 35 ms |
| `join(stores, "store", "left").filter(region == "r1")`: `count`, `sum` | **15 ms** | 173 ms | 18 ms |
| `row_number() over (partition by store order by price desc)`, keep `<= 3` | **52 ms** | 834 ms | 757 ms |
| `groupBy(category).agg(countDistinct(product))` | **94 ms** | 669 ms | 119 ms |

How to read these numbers:

- **Data placement differs by engine.**
  - Vega holds the table as in-memory columns, with low-cardinality strings
    dictionary-encoded.
  - Spark reads it from its in-memory columnar cache (`df.cache()`).
  - Sail cannot cache DataFrames yet (`cache()` is a no-op), so it scans the
    table on every query from uncompressed Parquet in the OS page cache.
    That is Sail's native input, and its scan is cheap: a full `count()`
    takes about 4 ms.
- **Optimizer rules help in some queries:**
  - **The join + groupBy query** pre-aggregates sales by store before joining
    the 1,000-row dimension (eager aggregation). Neither Spark nor Sail does
    this, and the query went from 125 ms to 14 ms.
  - **The left join + filter query** becomes an inner join because the filter
    rejects nulls (all three engines do this).
  - **The window query** keeps only each partition's top 3 rows per store
    before the shuffle, as Spark's `WindowGroupLimit` does. It went from 2 s to
    52 ms.
- **Vega vs Sail.** Without those rules, the gaps come from Vega's execution
  itself:
  - dictionary-encoded string grouping (3.4×)
  - dedicated hash tables for one or two integer keys
  - a parallel top-N
  - fused, column-pruned per-batch stages
- **Spark SQL** is the slowest on every query on this single machine: 6–31×
  slower than Vega and 1.1–10× slower than Sail. Part of that is fixed
  per-query overhead: query planning, whole-stage code
  generation, and scheduling two stages of tasks. Part is decoding its
  compressed columnar cache.
- **Scope.** These are single-machine, in-memory analytics queries. They don't
  measure scale-out, spilling to disk, or file formats (Vega reads CSV, not
  Parquet).

To reproduce:

```bash
cmake --build build-release --target vega_cpp_dataframe
pip install pyspark                                   # Spark SQL (Java 17+)
python -m venv sail && sail/bin/pip install pysail "pyspark-client==4.0.1" pyarrow
python benchmarks/bench_dataframe.py --n 10000000 --sail-python sail/bin/python
```

> **Note:** DataFrames are native in C++ ([C++ DataFrames](docs/cpp-dataframes.md)).
> PyVega provides **RDD** APIs only for now. See
> [DataFrame-style operations](docs/dataframe-operations.md) for RDD equivalents.

## Architecture

```
                      User Code (Python or C++)
                                |
                    SparkContext / Cluster
                                |
                          +-----------+
                          |  JobActor |      (builds DAG, distributes tasks)
                          +-----------+
                           /    |     \
                   +------+ +------+ +------+
                   |Worker| |Worker| |Worker|   (execute partitions in parallel)
                   +------+ +------+ +------+
                       \       |       /
                    +-------------------------+
                    |   Actor System          |   (thread pool, message routing)
                    +-------------------------+
                    |   Transport Layer       |   (Local / TCP + Gossip)
                    +-------------------------+
```

For PyVega, the actor system runs **inside `vega_server`**. The Python process is a thin
RPC client; worker threads are not separate OS processes.

**Key difference from Spark:** there is no central scheduler in cluster mode. Peers
discover each other via gossip and coordinate execution directly. In single-node /
session-server mode, the JobActor still orchestrates stages but all workers are local
threads.

### Actor framework

Vega includes a lightweight actor framework inspired by
[rotor](https://github.com/basiliscos/cpp-rotor):

- **Message-based communication** — typed messages dispatched to registered handlers
- **Thread pool execution** — actors process messages in parallel across worker threads
- **Sequential per-actor** — each actor processes one message at a time
- **Python GIL integration** — worker threads acquire the GIL when executing Python UDFs

## Build (C++)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

Targets include `vega_server`, `vega_worker`, examples, and C++ tests.

## Python API (PyVega)

```python
from pyvega import SparkContext

with SparkContext(4) as sc:
    rdd = sc.parallelize([1, 2, 3, 4, 5], num_partitions=2)

    result = (
        rdd.filter(lambda x: x > 2)
        .map(lambda x: x * 10)
        .collect()
    )
    # [30, 40, 50]

    # Word count
    lines = ["hello world hello", "hello vega"]
    counts = (
        sc.parallelize(lines)
        .flat_map(lambda line: line.split())
        .map(lambda word: (word, 1))
        .reduce_by_key(lambda a, b: a + b)
        .collect()
    )
```

Full API: [Python RDD guide](docs/python-guide.md).

## Running on a cluster (C++)

`vega_worker` runs **equal peers** (no special driver role). PyVega does not connect
to cluster peers yet; use this for C++ multi-node examples.

```bash
# Terminal 1
./build/vega_worker --bind 127.0.0.1:9001 --id peer-1

# Terminal 2
./build/vega_worker --bind 127.0.0.1:9002 --id peer-2 --seeds 127.0.0.1:9001

# Terminal 3 — submitter (same binary, not a "driver")
./build/examples/cluster_word_count --bind 127.0.0.1:9003 --id peer-3 \
    --seeds 127.0.0.1:9001 --expect-peers 3
```

Single-node demo: `./build/examples/cluster_word_count --local`

| Flag | Description |
|------|-------------|
| `--bind HOST:PORT` | Listen address |
| `--seeds H1:P1,H2:P2` | Gossip seed peers |
| `--id NAME` | Node identifier |
| `--expect-peers N` | Wait for N peers before running (examples) |

Details: [Server, workers, and clients](docs/server-and-workers.md).

## C++ API

```cpp
#include <vega/vega.hpp>

int main() {
    vega::SparkContext ctx(4);
    auto rdd = ctx.parallelize(std::vector<int>{1, 2, 3, 4, 5}, 2);
    auto result = rdd
        .filter([](const int& x) { return x > 2; })
        .map([](const int& x) { return x * 10; })
        .collect();
}
```

### Available operations

The C++ RDD API covers `map`, `filter`, `flatMap`, `reduceByKey`, `groupByKey`
and the `collect` / `count` / `reduce` / `foreach` actions.

C++ also has columnar DataFrames (`#include <vega/dataframe.hpp>`), with Spark
SQL semantics:

- **Nulls** throughout, with three-valued logic.
- **Expressions:** Spark-style column expressions, including `when`, `coalesce`,
  `cast`, and string and math functions.
- **Operations:**
  - `select`, `filter`, `group_by().agg()`; aggregates include `variance`,
    `stddev` and `count_distinct`
  - inner, outer, semi, anti and cross joins
  - unions, `distinct`, `drop_duplicates`
  - window functions: `row_number`, `rank`, `lag`/`lead`, and windowed
    aggregates
  - `order_by`, `limit`
  - CSV read and write
- **A rule-based optimizer:**
  - predicate pushdown
  - column pruning
  - outer-join elimination
  - eager aggregation through joins
  - top-N and window group limits
  - broadcast or shuffle join selection
- **`explain()`** shows the plans.

```cpp
using namespace vega::df;
auto top = sales.filter(col("qty") > 1)
                .join(stores, {"store"})
                .group_by({col("region")})
                .agg({sum(col("qty") * col("price")).alias("revenue")})
                .order_by({desc(col("revenue"))})
                .limit(10);
std::cout << top.show();
```

See [C++ DataFrames](docs/cpp-dataframes.md).

PyVega implements the PySpark RDD API: element-wise and partition
transformations, all key/value operations (`reduceByKey`, `combineByKey`,
`aggregateByKey`, `groupByKey`, `join` and the outer joins, `cogroup`,
`subtractByKey`), set operations, `sortBy`/`sortByKey`, sampling, numeric
actions (`stats`, `mean`, `stdev`, `histogram`, ...), approximate actions,
`cache`/`persist`, accumulators, broadcast variables and file I/O. See the
[Python RDD guide](docs/python-guide.md), and [Vega extras](docs/vega-extras.md)
for operations Spark does not have (`explain`, `map_batches`, `quantiles`,
`heavy_hitters`, `sliding`, ...).

## Project structure

```
vega/
  docs/                      User guides (start at docs/README.md)
  include/vega/              C++ header-only library
    core/                    RDD, SparkContext, shuffle
    actors/                  JobActor, WorkerActor, messages
    execution/               DAG builder, stages
    cluster/                 P2P cluster, gossip
    network/                 TCP framing, routing
  src/
    vega_server.cpp          TCP session server (PyVega backend)
    server_session.cpp       Session RPC + embedded Python UDFs
    vega_worker.cpp          Cluster peer daemon
  src/server_module.cpp      Server as a Python extension (shipped in wheels)
  python/pyvega/             PyVega client package
  scripts/
    pyspark_compat/          PySpark RDD API shim
    run_pyspark_tests.py     Upstream Spark RDD tests vs PyVega
  third_party/spark/         Vendored Spark test sources
  examples/                  word_count, cluster_word_count, benchmarks
  tests/                     C++ unit tests + Spark RDD suite ports
```

## Spark compatibility testing

```bash
# C++ Spark RDD logic ports
cmake --build build --target test_spark_rdd_suite
cd build && ctest -R SparkRDDSuite --output-on-failure

# Upstream PySpark RDD tests (53/57 pass; 4 require a JVM)
VEGA_SERVER=build/vega_server \
  python scripts/run_pyspark_tests.py --filter pyspark.tests.test_rdd
```

See [PySpark compatibility](docs/pyspark-compat.md).

## Execution flow

### PyVega (session server)

1. Client builds an RDD chain lazily (`map`, `filter`, …)
2. First action sends fused stages over TCP
3. Server materializes lineage and submits a job to JobActor
4. WorkerActors compute partitions (native C++ or embedded Python)
5. Shuffles run as multi-stage DAGs (any number of chained shuffles), and a
   shuffle's materialized output is reused by later actions
6. Results return to the Python client

### Multi-node C++ cluster

1. Peers start and discover each other via gossip
2. Submitter distributes partition data and task assignments over TCP
3. Peers execute tasks and return serialized results

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for build/test instructions and PR
guidelines, and our [Code of Conduct](CODE_OF_CONDUCT.md). Report security
issues privately per [SECURITY.md](SECURITY.md).

## License

Licensed under the [MIT License](LICENSE).

Vega is **not affiliated with the Apache Software Foundation**. "Apache Spark"
and "PySpark" are trademarks of the ASF; Vega offers a Spark-like API for
familiarity and compatibility testing only. See [NOTICE](NOTICE) for
third-party attributions (Apache Spark test sources, pybind11, GoogleTest,
cloudpickle).
