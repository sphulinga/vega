# Contributing to Vega

Thanks for your interest in contributing! Vega is an early-stage distributed
data engine, so the API and internals may change. Bug reports, docs, tests, and
focused features are all welcome.

By contributing, you agree that your contributions are licensed under the
project's [MIT License](LICENSE).

## Project layout

See the [docs index](docs/README.md) for an overview. In short:

- `include/vega/` — header-only C++ engine (RDD, actors, execution, cluster)
- `src/` — `vega_server` (PyVega backend), `server_session.cpp`, `vega_worker`
- `python/pyvega/` — Python client package
- `scripts/pyspark_compat/` — PySpark-compatible RDD shim
- `tests/` — C++ unit tests and Spark RDD suite ports
- `python/tests/` — PyVega Python tests

## Prerequisites

- C++17 compiler (GCC 8+, Clang 7+)
- CMake 3.16+
- Python 3.8+
- Linux or macOS (POSIX sockets)

## Build

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release -DVEGA_BUILD_SERVER=ON
cmake --build build-release -j$(nproc)
```

## Install the Python client

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
export VEGA_SERVER="$PWD/build-release/vega_server"
```

## Run the tests

```bash
# C++ unit tests + Spark RDD suite
cd build-release && ctest --output-on-failure && cd ..

# PyVega Python tests
VEGA_SERVER=build-release/vega_server PYTHONPATH=python \
  python -m pytest python/tests/test_pyvega.py -q

# Upstream PySpark RDD compatibility tests (optional; fetches Spark sources)
./scripts/fetch_spark_tests.sh
VEGA_SERVER=build-release/vega_server \
  python scripts/run_pyspark_tests.py --filter pyspark.tests.test_rdd
```

> **Note:** `third_party/spark/` is fetched on demand by
> `scripts/fetch_spark_tests.sh` and is intentionally git-ignored. You do not
> need it to build or use Vega — only to run the PySpark compatibility suite.

## Development guidelines

- **Keep changes focused.** Small, reviewable PRs merge faster.
- **Add or update tests** for behavior changes. C++ logic goes in `tests/`,
  Python in `python/tests/`, compatibility coverage via the PySpark suite.
- **Match existing style.** C++17 with the conventions already in `include/vega/`;
  Python with type hints as in `python/pyvega/`.
- **Document user-facing changes** in `docs/` and the relevant README.
- **No unrelated reformatting** in feature/bugfix PRs.

## Pull request checklist

- [ ] Builds cleanly: `cmake --build build-release`
- [ ] C++ tests pass: `ctest --output-on-failure`
- [ ] Python tests pass: `pytest python/tests/`
- [ ] PySpark compat suite not regressed (if you touched RDD/shuffle code)
- [ ] Docs updated for user-facing changes
- [ ] Commits have clear messages

## Reporting bugs

Open an issue with:

- What you ran (commands, code snippet)
- Expected vs actual behavior
- Environment (OS, compiler, Python version)
- Server stderr if `vega_server` crashed (run it in the foreground)

## Security issues

Please do not file public issues for vulnerabilities. See [SECURITY.md](SECURITY.md).
