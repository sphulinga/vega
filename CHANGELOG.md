# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project aims
to follow [Semantic Versioning](https://semver.org/) once it reaches 1.0.

## [2.0.1] - 2026-11-20

### Fixed
- **Multi-process C++ cluster mode (`vega::Cluster`) works end to end.** A job
  across several processes used to hang or return incomplete results:
  - Peers other than the root ignored scattered data and computed empty
    partitions.
  - A slower peer dropped shuffle blocks, map-done signals and results that
    faster peers sent before it reached the job.
  - `stop()` waited for remote peers to close their connections.

  Early messages are now buffered or recorded until the job reaches them.
  `scatter` sends every partition to its owner, and owners wait for theirs.
  The router unblocks its connection threads on `stop()`. A new test runs
  scatter, a narrow job and a shuffle job on three peers over localhost TCP.
- `TcpServer` no longer races its accept thread on the listening socket, and
  `stop()` works on macOS, where `shutdown()` doesn't wake a blocked `accept()`.
  The accept loop now polls with a timeout.
- `cluster_word_count` now counts words in cluster mode; it used to only
  collect the input lines.

### Changed
- `Cluster::scatter` takes the root peer's index (default 0) and a timeout.
- README and `docs/server-and-workers.md`: cluster mode is documented as SPMD
  (every peer runs the same program), with its limits. `vega_worker` only joins
  gossip and cannot run jobs, and must not join a cluster that does.

## [2.0.0] - 2026-11-20

A major release for the new native C++ DataFrame API. No existing C++ RDD or
PyVega API changed, so code written for 1.x keeps working.

### Added
- **C++ DataFrames** (`#include <vega/dataframe.hpp>`, namespace `vega::df`):
  columnar, typed DataFrames that run on the actor engine, with Spark SQL
  semantics. See `docs/cpp-dataframes.md`.
  - Types `bigint`, `double`, `string` and `boolean`. Every column can hold
    nulls, stored as an optional validity mask.
  - Low-cardinality strings are dictionary-encoded automatically.
  - Expressions:
    - arithmetic and comparisons
    - three-valued `AND`/`OR`/`NOT`, `<=>`, `isin`, `between`
    - `when`/`otherwise`, `coalesce`, `isnull`/`isnotnull`
    - ANSI `cast`
    - string functions: `upper`, `lower`, `trim`, `length`, `substring`,
      `concat`, `contains`, `startswith`, `endswith`, `like`
    - math functions: `abs`, `round`, `floor`, `ceil`, `sqrt`
  - Operations:
    - `select`, `with_column`, `filter`
    - `group_by().agg()` with `sum`, `count`, `avg`, `min`, `max`, `variance`,
      `stddev` and `count_distinct`
    - inner, left, right, full, semi, anti and cross joins
    - `union_all`, `union_by_name`, `distinct`, `drop_duplicates`
    - `order_by`, `limit`, `fill_null`, `drop_nulls`, `repartition`, `cache`
    - `count`, `collect`, `show`, `explain`
    - CSV read and write
  - Window functions: `row_number`, `rank`, `dense_rank`, `lag`, `lead`, and
    windowed `sum`/`avg`/`min`/`max`/`count`.
- **Query optimizer** for DataFrames:
  - constant folding and boolean simplification
  - predicate pushdown through projections, sorts, unions, windows,
    aggregates and joins
  - outer-join elimination, and join-key predicates copied to the other side
  - column pruning down to the scans
  - project collapsing, sort elimination, limit pushdown
  - eager aggregation through inner joins
  - window group limits, and top-N for `order_by().limit()`
- **Physical planning:**
  - broadcast or shuffle hash joins, chosen from size estimates
  - fused per-batch stages
  - two-phase hash aggregation, with dedicated hash tables for one or two
    integer keys and for dictionary-encoded keys
- **Column statistics.** Creating or caching a DataFrame records exact row
  and null counts, plus distinct-value counts: exact for dictionary-encoded
  columns, HyperLogLog estimates otherwise. They guide group-size and filter
  estimates, the join strategy, and eager aggregation.
  `df.estimated_rows()` and `df.estimated_distinct(col)` expose them.
- `RDD::union_with` in the C++ core.
- Benchmarks:
  - `benchmarks/bench_dataframe.py` runs 12 DataFrame queries on Vega C++,
    Spark SQL and Sail, and checks that all three return the same answers.
  - `benchmarks/bench_native_ops.py` compares Vega's C++ RDD API with Spark's
    Scala RDD API.
  - `benchmarks/bench_python_ops.py` compares PyVega with PySpark.
  - Results are in the README.

### Changed
- README performance section: v0.5.0, v1.0.0 and v1.1.0 re-measured on one
  machine, plus like-for-like comparisons with Apache Spark.

## [1.1.0] - 2026-11-17

### Fixed
- **Intermittent hang on shutdown.** `ActorSystem::stop()` could block forever
  in `join()`: it cleared its running flag without holding the ready-queue
  mutex, so a worker could miss the shutdown wakeup and sleep forever. Any
  `SparkContext` (C++ or the PyVega session server) could hit this when
  stopping; it hung a CI run for 6 hours on macOS. The flag is now changed
  under the mutex, with a regression test that stops right after a handler
  unblocks the caller.

### Changed
- Release workflow accepts tags with or without a leading `v` (`v1.2.3` or
  `1.2.3`, including pre-releases such as `1.2.3rc1`); the 1.0.0 release was
  tagged `1.0.0`, which did not trigger publishing.
- CI: every job has a time limit and each C++ test a 3-minute limit, so a hang
  fails fast and names the stuck test; the `build & test` job now runs the
  whole Python suite (`test_spark_api.py` was skipped there). Wheel jobs in
  the release workflow are capped at 60 minutes.
- `docs/releasing.md`: trusted-publisher setup for the existing `pyvega` PyPI
  project and how to recover from an `invalid-publisher` error.

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
