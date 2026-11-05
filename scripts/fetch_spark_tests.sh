#!/usr/bin/env bash
# Fetch Apache Spark test sources used by PyVega compatibility runs.
#
# The sources are pinned to a fixed commit so CI results are reproducible:
# tracking `master` meant every new upstream test could break the build.
# Override with SPARK_REF=<sha|tag|branch> to test against another revision,
# and bump DEFAULT_SPARK_REF (plus the expected-results list in
# scripts/pyspark_known_failures.txt) when moving the pin.
set -euo pipefail

DEFAULT_SPARK_REF="8410fcffa919b682c66bf4886858ceb9134a0c5a"  # apache/spark master, 2026-11-05
SPARK_REF="${SPARK_REF:-$DEFAULT_SPARK_REF}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/third_party/spark"

if [[ ! -d "$DEST/.git" ]]; then
  rm -rf "$DEST"
  git init -q "$DEST"
  git -C "$DEST" remote add origin https://github.com/apache/spark.git
  git -C "$DEST" config remote.origin.promisor true
  git -C "$DEST" config remote.origin.partialclonefilter blob:none
fi

git -C "$DEST" sparse-checkout set \
  python/pyspark/tests \
  python/pyspark/sql/tests \
  core/src/test/scala/org/apache/spark/rdd \
  python/test_support

git -C "$DEST" fetch -q --depth 1 --filter=blob:none origin "$SPARK_REF"
git -C "$DEST" checkout -q --detach FETCH_HEAD

echo "Spark tests at $(git -C "$DEST" rev-parse --short HEAD) available under $DEST"
