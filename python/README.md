# PyVega

Python client for the [Vega](../README.md) C++ engine.

## Install

```bash
# From repo root (after building vega_server)
pip install -e ".[test]"
export VEGA_SERVER=build-release/vega_server
```

## Minimal example

```python
from pyvega import SparkContext

with SparkContext(4) as sc:
    print(sc.range(0, 10).map(lambda x: x ** 2).collect())
```

## Documentation

Full guides live in [`docs/`](../docs/README.md):

- [Getting started](../docs/getting-started.md)
- [Server, workers, and clients](../docs/server-and-workers.md)
- [Python RDD guide](../docs/python-guide.md)
- [DataFrame-style operations](../docs/dataframe-operations.md)

## Package layout

```
python/pyvega/
  __init__.py      SparkContext, RDD
  _client.py       TCP session client, stage fusion, spawn_server
  _protocol.py     Message types and binary codec
  _native.py       Lambda → native C++ op compiler
  broadcast.py     Broadcast variables
  _io.py           text/pickle file helpers
```

## Tests

```bash
VEGA_SERVER=build-release/vega_server PYTHONPATH=python \
  python -m pytest python/tests/test_pyvega.py -q
```
