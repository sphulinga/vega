# PySpark compatibility

Vega ships a **PySpark compatibility shim** so upstream Apache Spark RDD tests and
PySpark-style scripts can run against PyVega without a JVM.

Location: `scripts/pyspark_compat/`

## What works

- **RDD API** — `map`, `filter`, `flatMap`, `reduceByKey`, `groupByKey`, `aggregate`,
  `fold`, `sortByKey`, `zip`, `cartesian`, `pipe`, `broadcast`, `cache`, partition ops,
  and most actions used in `pyspark.tests.test_rdd`
- **SparkContext** — `parallelize`, `range`, `textFile`, job groups (stubs), resource
  profile stubs
- **50/54** upstream `test_rdd` cases pass (remaining 4 require JVM / Java RDD APIs)

## What does not work

- **Spark SQL / DataFrame** — `SparkSession.createDataFrame`, SQL, Catalyst, etc.
- **JVM integration** — `_jvm`, `_jrdd`, `_to_java_object_rdd`
- **Streaming, MLlib, structured APIs**

The shim's `SparkSession` is a minimal stub for tests that import it but do not call
DataFrame methods.

---

## Running with the compat layer

```bash
export VEGA_SERVER=build-release/vega_server
export PYTHONPATH="$PWD/python:$PWD/scripts/pyspark_compat"
export SPARK_HOME="$PWD/third_party/spark"   # optional; for test resources

python scripts/run_pyspark_tests.py --filter pyspark.tests.test_rdd
```

Reports are written to `test-results/pyspark_vs_pyvega.json`.

### Fetch upstream Spark tests

```bash
./scripts/fetch_spark_tests.sh
```

Tests live under `third_party/spark/python/pyspark/tests/`.

---

## Writing PySpark-style scripts

```python
import sys, os
sys.path[:0] = [
    os.path.abspath("python"),
    os.path.abspath("scripts/pyspark_compat"),
]

from pyspark.context import SparkContext

sc = SparkContext()
try:
    data = sc.parallelize(range(100), 4)
    print(data.map(lambda x: x * x).reduce(lambda a, b: a + b))
finally:
    sc.stop()
```

Or set `PYTHONPATH` once in your shell / IDE.

### Differences from real PySpark

| Topic | PySpark (JVM) | PyVega compat |
|-------|---------------|---------------|
| Execution | JVM + Python workers | C++ server + embedded Python |
| Lazy stages | JVM DAG scheduler | Client-side stage fusion + RPC |
| Errors | `Py4JJavaError` | Translated from server `RuntimeError` |
| `SparkContext(master=...)` | Honored | Ignored (always Vega backend) |
| DataFrames | Full SQL engine | Not implemented |

StopIteration in user code is converted to a Spark-like error message and surfaced as
`Py4JJavaError` where the upstream tests expect it.

---

## C++ Spark RDD tests

Logic ports of Spark's RDDSuite also exist in C++:

```bash
cmake --build build-release --target test_spark_rdd_suite
cd build-release && ctest -R SparkRDDSuite --output-on-failure
```

These validate native shuffle and RDD semantics without Python.

---

## When to use which API

| Use case | API |
|----------|-----|
| New Vega projects | `from pyvega import SparkContext` |
| Porting PySpark RDD jobs | Compat shim + `pyspark.*` imports |
| Validating compatibility | `scripts/run_pyspark_tests.py` |
| SQL / DataFrame workloads | Not supported yet — see [DataFrame-style operations](dataframe-operations.md) |
