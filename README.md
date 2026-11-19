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
| [DataFrame-style operations](docs/dataframe-operations.md) | SQL-like patterns using RDDs |
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

### Compared with Apache Spark (Scala)

[`benchmarks/spark_scala_ops.scala`](benchmarks/spark_scala_ops.scala) runs the
same 15 operations, datasets and partition counts with Spark's Scala RDD API
(row names match the PyVega benchmark; on Spark the functions are Scala
lambdas);
[`benchmarks/bench_spark_scala_ops.py`](benchmarks/bench_spark_scala_ops.py)
runs it (`pip install pyspark` provides a full Spark, including `spark-shell`).
Measured on the same machine as above with Spark 4.3.0 (Scala 2.13, Java 21,
`local[4]`, 8 partitions, web UI off). Spark gets one untimed warm-up pass over
every operation first, so its JIT-compiled code is measured at full speed.
Runs alternated between the two engines; each cell is the best of 8 runs at
200k elements and of 2 runs at 2M elements.

| Operation | PyVega 1.1.0, 200k (ms) | Spark, 200k (ms) | PyVega vs Spark | PyVega 1.1.0, 2M (ms) | Spark, 2M (ms) | PyVega vs Spark |
|---|---:|---:|---:|---:|---:|---:|
| parallelize+collect (tuples) | 104.6 | 70.7 | 1.5× slower | 1553.5 | 661.2 | 2.3× slower |
| map+filter python (tuples) | 29.6 | 48.1 | 1.6× faster | 399.2 | 493.1 | 1.2× faster |
| reduceByKey (tuples) | 11.4 | 85.3 | 7.5× faster | 120.4 | 514.8 | 4.3× faster |
| groupByKey (tuples) | 8.0 | 109.3 | 14× faster | 94.4 | 621.0 | 6.6× faster |
| distinct (strings) | 40.2 | 91.8 | 2.3× faster | 407.1 | 382.5 | 1.1× slower |
| sortByKey (tuples) | 49.8 | 189.2 | 3.8× faster | 601.7 | 1350.9 | 2.2× faster |
| mapPartitions | 5.7 | 42.5 | 7.5× faster | 59.0 | 443.2 | 7.5× faster |
| union x2 † | 0.3 | 67.1 | 216× faster | 0.4 | 806.7 | 2180× faster |
| repartition(16) | 3.0 | 111.7 | 38× faster | 38.5 | 794.1 | 21× faster |
| aggregateByKey | 19.9 | 91.5 | 4.6× faster | 198.9 | 505.3 | 2.5× faster |
| take(10) | 1.1 | 14.6 | 13× faster | 9.3 | 89.4 | 9.6× faster |
| first() | 1.1 | 14.2 | 13× faster | 9.2 | 86.3 | 9.4× faster |
| sum (ints) | 1.3 | 12.2 | 9.7× faster | 10.8 | 45.1 | 4.2× faster |
| aggregate | 20.2 | 40.6 | 2.0× faster | 219.9 | 452.5 | 2.1× faster |
| chained shuffle | 11.5 | 106.9 | 9.3× faster | 121.4 | 593.4 | 4.9× faster |

How to read this:

- **This is single-machine, in-memory data.** At this size Spark's fixed
  per-job costs dominate: it schedules every job and stage as distributed
  tasks and always writes shuffle data to local disk files, because it is built
  for fault-tolerant clusters. PyVega runs in one process and shuffles in
  memory by passing pointers. Spark scales out to a cluster; PyVega does not
  yet.
- **The gap narrows as data grows.** From 200k to 2M elements PyVega's lead
  shrinks on most operations (e.g. `reduceByKey` 7.5× to 4.3×). Operations that
  call a Python function per element (`map+filter`, `distinct`) run on
  effectively one core in PyVega (the GIL), while Spark's JIT-compiled Scala
  uses every core, so expect Spark to pull ahead on larger CPU-heavy jobs.
- **`parallelize+collect` favours Spark** because PyVega pickles every element
  in Python on the client.
- **† `union x2` is not the same work on both sides.** PyVega answers `count()`
  on a union of in-memory collections from the stored partition sizes without
  touching any element; Spark iterates all of them.

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

> **Note:** PyVega provides **RDD** APIs, not Spark SQL DataFrames. See
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

The C++ API covers `map`, `filter`, `flatMap`, `reduceByKey`, `groupByKey`
and the `collect` / `count` / `reduce` / `foreach` actions.

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
