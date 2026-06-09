# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project aims
to follow [Semantic Versioning](https://semver.org/) once it reaches 1.0.

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
