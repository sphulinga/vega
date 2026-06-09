#!/usr/bin/env bash
# Fetch/update Apache Spark test sources used by PyVega compatibility runs.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/third_party/spark"

if [[ -d "$DEST/.git" ]]; then
  git -C "$DEST" fetch --depth 1 origin master
  git -C "$DEST" checkout master
  git -C "$DEST" pull --ff-only origin master
else
  rm -rf "$DEST"
  git clone --depth 1 --filter=blob:none --sparse https://github.com/apache/spark.git "$DEST"
  git -C "$DEST" sparse-checkout set \
    python/pyspark/tests \
    python/pyspark/sql/tests \
    core/src/test/scala/org/apache/spark/rdd \
    python/test_support
fi

echo "Spark tests available under $DEST"
