# Vega documentation

Vega is a C++17 distributed data engine with a Spark-like RDD API. **PyVega** is the
Python client: it talks to a C++ **session server** over TCP and records transformations
lazily until an action runs.

## Guides

| Document | What you'll learn |
|----------|-------------------|
| [Getting started](getting-started.md) | Prerequisites, build, install, first program |
| [Server, workers, and clients](server-and-workers.md) | `vega_server`, actor workers, cluster peers, connection modes |
| [Python RDD guide](python-guide.md) | PyVega API, transformations, actions, I/O |
| [PySpark compatibility](pyspark-compat.md) | Run PySpark-style code and upstream RDD tests |
| [DataFrame-style operations](dataframe-operations.md) | RDD equivalents for common SQL/DataFrame patterns |
| [Vega extras](vega-extras.md) | Operations beyond Spark's RDD API |
| [Releasing](releasing.md) | Wheels, PyPI trusted publishing, release checklist |

## Quick reference

```bash
# Install PyVega (prebuilt wheel with the C++ server inside)
pip install pyvega

# ...or from a source checkout (compiles the server extension)
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"

# Run a program (auto-starts vega_server)
python -c "from pyvega import SparkContext; \
  print(SparkContext(4).range(0,10).map(lambda x: x*2).collect())"
```

```bash
# Or start the server manually
vega-server --bind 127.0.0.1:9500 --print-listening

# Connect from Python
python -c "from pyvega import SparkContext; \
  sc=SparkContext(remote='127.0.0.1:9500'); \
  print(sc.range(0,5).count()); sc.stop()"
```

## Architecture at a glance

```
  Python process                         vega_server process
 ┌─────────────────┐                   ┌──────────────────────────┐
 │  SparkContext   │  TCP (lazy RPC)   │  Session + SparkContext  │
 │  RDD handles    │ ────────────────► │  JobActor + WorkerActors │
 │  pending stages │                   │  embedded Python (UDFs)  │
 └─────────────────┘                   └──────────────────────────┘
```

- **Client workers** (`SparkContext(4)`) → worker *threads* inside `vega_server`, not separate OS processes.
- **`vega_worker`** → separate cluster *peers* for multi-node C++ jobs (not used by PyVega today).
- **Transformations** are batched client-side and sent on the first action to minimize RPC round-trips.

## C++ and cluster docs

The root [README](../README.md) covers the C++ API, P2P cluster mode, gossip, and project layout.
