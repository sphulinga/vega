# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project aims
to follow [Semantic Versioning](https://semver.org/) once it reaches 1.0.

## [1.0.0] - 2026-11-05

### Added
- **PyPI packaging.** The session server now also builds as a CPython extension
  (`pyvega._vega_server`) via scikit-build-core, so wheels are self-contained:
  `pip install pyvega` needs no CMake or separate binary. The extension does not
  link libpython and runs inside the user's interpreter (so client and server
  always agree on the Python version). `vega-server` uses it when available.
- Release workflow (`.github/workflows/release.yml`): cibuildwheel wheels for
  CPython 3.9-3.13 on manylinux_2_28 x86_64/aarch64 and macOS arm64/x86_64, each
  smoke-tested; sdist; PyPI/TestPyPI Trusted Publishing; GitHub Release assets.
  See `docs/releasing.md`. CI gained a `pip install .` job.
- Spark RDD API parity: `join`, `leftOuterJoin`, `rightOuterJoin`,
  `fullOuterJoin`, `cogroup`/`groupWith`, `combineByKey`, `subtractByKey`,
  `subtract`, `intersection`, `union`/`+`, `keys`, `values`, `keyBy`,
  `mapValues`, `flatMapValues`, `groupBy`, `zipWithIndex`, `zipWithUniqueId`,
  `sampleByKey`, `randomSplit`, `countByKey`, `countByValue`, `collectAsMap`,
  `lookup`, `top`, `takeOrdered`, `treeReduce`, `foreachPartition`, `stats`
  (`StatCounter`), `mean`, `variance`, `stdev`, `sampleStdev`,
  `sampleVariance`, `max`, `min`, `countApprox`, `meanApprox`, `sumApprox`,
  `persist`/`unpersist`/`getStorageLevel` (`StorageLevel`), `checkpoint`,
  `setName`/`name`, `toDebugString`, `context`, `mapPartitionsWithSplit`.
- `SparkContext`: accumulators (`sc.accumulator`, `AccumulatorParam`),
  `wholeTextFiles`, `binaryFiles`, `runJob`, `range(..., step=)`,
  `defaultParallelism`, `defaultMinPartitions`, `version`, `getOrCreate`, job
  group/description/local-property APIs, `cancelJobGroup`/`cancelAllJobs`
  (best effort).
- File I/O runs in server tasks: `textFile` accepts directories, globs and
  comma-separated lists, splits large files into byte ranges, and reads
  `.gz`/`.bz2`/`.xz`; `saveAsTextFile` supports Gzip/BZip2 codecs and writes
  `_SUCCESS`.
- Vega extras beyond Spark (`docs/vega-extras.md`): `explain()`, `map_batches`,
  `distinct_by`, `top_by_key`, `quantiles` (exact/approximate),
  `heavy_hitters`, `sliding`, `persisted()` context manager, `cache_info()`,
  JSON-lines I/O, `sc.status()` / `sc.job_history`.
- Server-side operators (protocol v2): apply-stages, union, partitionBy /
  round-robin repartition, coalesce, cartesian, zipPartitions,
  mapPartitionsWithIndex stages, persist, release, run-job on a partition
  subset, native numeric stats, describe, run-python.
- `benchmarks/bench_pyvega_ops.py` micro-benchmarks; `python/tests/test_spark_api.py`.

### Changed
- Operations that used to collect data into the client and re-upload it
  (`distinct`, `union`, `partitionBy`, `repartition`, `coalesce`, `sortByKey`,
  `mapPartitions*`, `glom`, `zip`, `cartesian`, `sample`, `aggregate`,
  `histogram`, `pipe`, saves) now run on the server. Large speedups: see the
  README performance table.
- The DAG scheduler handles any number of shuffles (post-order over all
  dependencies) and reuses materialized shuffle output (skipped stages).
  Jobs can target a subset of partitions; `take`/`first`/`isEmpty` scan
  partitions incrementally and `toLocalIterator` computes one at a time.
- Python shuffles combine values while bucketing (map-side combine);
  `aggregateByKey`/`foldByKey` use `combineByKey` instead of `groupByKey`.
- UDFs are called through the CPython vectorcall API with C++ StopIteration
  translation instead of a Python wrapper frame; data payloads use the C
  pickler (cloudpickle only for code / unpicklable objects).
- Server errors keep their Python exception type on the client (for example a
  `ValueError` raised in a UDF is raised as `ValueError`).
- Server-side RDD handles are reference-counted and freed when the last client
  `RDD` object is garbage-collected.
- `countApproxDistinct` uses HyperLogLog computed in server tasks.
- `requires-python` is now `>=3.9`. The version is single-sourced from
  `python/pyvega/__init__.py`. `setup.py` was removed (scikit-build-core).
- CMake options `VEGA_BUILD_PYMODULE`, `VEGA_BUILD_TESTS`,
  `VEGA_BUILD_EXAMPLES`; an installed pybind11 is preferred over fetching it.

### Fixed
- CI: the PySpark compatibility job tracked Spark `master` and gated on an
  error *count*, so new upstream tests broke the build. Spark test sources are
  now pinned (`scripts/fetch_spark_tests.sh`) and results are checked against
  an explicit list of JVM-only tests (`scripts/check_pyspark_report.py`,
  `scripts/pyspark_known_failures.txt`); the report is uploaded as an artifact.
- Release workflow also runs (build + test only, no publishing) on pull
  requests that touch packaging or C++ sources, so the wheel matrix is
  validated before a tag is pushed.
- `countApproxDistinct` was flaky (upstream `test_count_approx_distinct`
  failed for ~13% of hash seeds): register precision now has a floor of 2^14
  (HyperLogLog++-style high precision for small cardinalities) and the
  per-type hash salt is deterministic.
- Chained shuffles (e.g. `reduceByKey(...).map(...).groupByKey()`) failed with
  "shuffle data not prepared".
- `count()` on a fully native pipeline dropped the pending transformations, so
  a later `collect()` returned untransformed data.
- ~44 ms latency on every RPC (Nagle's algorithm + delayed ACK): server sockets
  now set `TCP_NODELAY` and send each frame in one `sendmsg` call;
  `MSG_NOSIGNAL` prevents SIGPIPE on disconnected clients.
- Server crash ("PyThreadState_Get: GIL is released") when a job failed after
  other partitions produced Python results, or when actor messages held the
  last reference to a Python-backed RDD.
- `getNumPartitions()` on an RDD with a pending shuffle returned the parent's
  partition count.
- `sortByKey(keyfunc=...)` compared range bounds against transformed keys
  inconsistently.

## [0.5.0] - 2026-06-09

First public open-source release.

### Added
- Open-source project files: `LICENSE` (MIT), `NOTICE`, `CONTRIBUTING.md`,
  `SECURITY.md`, `CODE_OF_CONDUCT.md`, `CHANGELOG.md`.
- Repository hygiene: root `.gitignore` (ignores build dirs, venv, fetched
  `third_party/spark`, caches, reports).
- GitHub Actions CI: build + C++/Python tests on Linux and macOS, plus a
  PySpark RDD compatibility job.
- Issue and pull request templates under `.github/`.
- Expanded documentation under `docs/` (getting started, server/workers/clients,
  Python RDD guide, PySpark compatibility, DataFrame-style operations).
- RDD engine with native fast paths, embedded-Python UDFs, and shuffle support.
- PySpark `test_rdd` compatibility at 50/54 (4 remaining require a JVM).

### Changed
- Synced package version to `0.5.0` and enriched `pyproject.toml` metadata
  (authors, classifiers, URLs).

### Fixed
- `test_external_group_by_key`: native shuffle buckets are now cleared on
  re-execution, so grouped values no longer accumulate across actions.
- `test_large_closure`: Python shuffle partition results are released under the
  GIL, fixing a server crash after `groupByKey`.
- `test_stopiteration_in_user_code`: `StopIteration` raised in user code during
  `aggregate`/`fold`/`reduceByKeyLocally` is converted to a Spark-like error.
