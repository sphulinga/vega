# Getting started

This guide walks through building Vega, installing PyVega, and running your first
distributed RDD program.

## Prerequisites

| Component | Version | Notes |
|-----------|---------|-------|
| C++ compiler | C++17 (GCC 8+, Clang 7+) | Required to build `vega_server` |
| CMake | 3.16+ | |
| Python | 3.8+ | Client and embedded interpreter in the server |
| POSIX sockets | — | Linux or macOS |

Optional:

- **cloudpickle** — installed automatically with PyVega; serializes Python UDFs
- **pytest** — for running Python tests (`pip install -e ".[test]"`)

## 1. Clone and create a virtual environment

```bash
git clone <repo-url> vega && cd vega

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

## 2. Build the session server

PyVega requires the `vega_server` binary. It embeds Python to execute user-defined
functions (UDFs) on worker threads.

```bash
cmake -S . -B build-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DVEGA_BUILD_SERVER=ON

cmake --build build-release --target vega_server -j$(nproc)
```

Verify the binary exists:

```bash
ls -l build-release/vega_server
```

### Build everything (optional)

```bash
cmake --build build-release -j$(nproc)   # examples, C++ tests, vega_worker
cd build-release && ctest --output-on-failure
```

## 3. Install PyVega

From the repository root:

```bash
pip install -e ".[test]"
```

This installs the `pyvega` package and the `vega-server` console script (a thin wrapper
around the `vega_server` binary).

Tell PyVega where to find the server if it is not on `PATH`:

```bash
export VEGA_SERVER="$PWD/build-release/vega_server"
```

## 4. Hello, Vega

### Automatic server (recommended for development)

`SparkContext(N)` spawns a local `vega_server` subprocess, connects over TCP, and
negotiates `N` worker threads (default: server CPU count if `N=0`).

```python
from pyvega import SparkContext

with SparkContext(4) as sc:
    doubled = sc.range(0, 10).map(lambda x: x * 2).collect()
    print(doubled)
    # [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
```

The `with` block ensures the server subprocess is stopped when you exit.

### Manual server (production-style)

**Terminal 1** — start the server:

```bash
export VEGA_SERVER="$PWD/build-release/vega_server"
export PYTHONHOME="$(python -c 'import sys; print(sys.base_prefix)')"
vega-server --bind 127.0.0.1:9500 --print-listening
# prints: LISTENING 9500
```

**Terminal 2** — connect a client:

```python
from pyvega import SparkContext

sc = SparkContext(remote="127.0.0.1:9500")
try:
    print(sc.parallelize(["a", "bb", "ccc"]).map(len).collect())
finally:
    sc.stop()   # does not kill a remote server
```

## 5. Word count

```python
from pyvega import SparkContext

lines = [
    "hello world hello",
    "world of vega",
    "hello vega cluster",
]

with SparkContext(4) as sc:
    counts = (
        sc.parallelize(lines, num_partitions=2)
        .flat_map(lambda line: line.split())
        .map(lambda word: (word, 1))
        .reduce_by_key(lambda a, b: a + b)
        .collect()
    )
    for word, count in sorted(counts, key=lambda kv: -kv[1]):
        print(f"{word}: {count}")
```

## 6. Run tests

```bash
# PyVega unit tests
VEGA_SERVER=build-release/vega_server PYTHONPATH=python \
  python -m pytest python/tests/test_pyvega.py -q

# Upstream PySpark RDD tests (via compatibility shim)
VEGA_SERVER=build-release/vega_server \
  python scripts/run_pyspark_tests.py --filter pyspark.tests.test_rdd
```

## Troubleshooting

### `vega_server not found`

Set `VEGA_SERVER` to the absolute path of the binary, or add `build-release/` to
`PATH`.

### `Failed to import encodings module` / server exits immediately

The server embeds Python and needs a valid runtime. When PyVega spawns the server it
sets `PYTHONHOME` and `PYTHONPATH` automatically. If you start `vega_server` manually,
use the same Python environment:

```bash
export PYTHONHOME="$(python -c 'import sys; print(sys.base_prefix)')"
export PYTHONPATH="$(python -c 'import sys; print(":".join(sys.path))')"
build-release/vega_server --bind 127.0.0.1:9500
```

### `ConnectionError: vega_server closed connection`

The server process crashed. Run it in the foreground (without `--print-listening`) to
see stderr, or check for GIL / UDF errors in your Python functions.

## Next steps

- [Server, workers, and clients](server-and-workers.md) — connection modes and worker model
- [Python RDD guide](python-guide.md) — full API reference with examples
- [DataFrame-style operations](dataframe-operations.md) — SQL-like patterns on RDDs
