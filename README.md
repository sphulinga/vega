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

## Quick start (Python)

```bash
# Build session server
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DVEGA_BUILD_SERVER=ON
cmake --build build --target vega_server -j$(nproc)

# Install client
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
export VEGA_SERVER="$PWD/build/vega_server"
```

```python
from pyvega import SparkContext

with SparkContext(4) as sc:
    print(sc.range(0, 10).map(lambda x: x * 2).collect())
    # [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
```

**Manual server:** `vega-server --bind 127.0.0.1:9500` then
`SparkContext(remote="127.0.0.1:9500")`. See [server and clients](docs/server-and-workers.md).

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

| Transformation | Python | C++ |
|----------------|--------|-----|
| `map` | `rdd.map(func)` | `rdd.map(func)` |
| `filter` | `rdd.filter(func)` | `rdd.filter(pred)` |
| `flat_map` | `rdd.flat_map(func)` | `rdd.flatMap(func)` |
| `reduceByKey` | `rdd.reduce_by_key(func)` | `rdd.reduceByKey(func)` |
| `groupByKey` | `rdd.group_by_key()` | `rdd.groupByKey()` |

| Action | Python | C++ |
|--------|--------|-----|
| `collect` | `rdd.collect()` | `rdd.collect()` |
| `count` | `rdd.count()` | `rdd.count()` |
| `reduce` | `rdd.reduce(func)` | `rdd.reduce(func)` |
| `foreach` | `rdd.foreach(func)` | `rdd.foreach(func)` |

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

# Upstream PySpark RDD tests (50/54 pass; 4 require JVM)
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
5. Shuffle stages (`reduceByKey`, `groupByKey`) run as multi-stage DAGs
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
