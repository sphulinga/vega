# C++ DataFrames

Vega has a native, columnar DataFrame API for C++ (`#include <vega/dataframe.hpp>`,
namespace `vega::df`). It runs on the same actor engine as RDDs: every
DataFrame is an `RDD<Batch>` of typed columnar batches plus a schema, and its
operations compile to parallel RDD stages, shuffles included.

PyVega does not expose DataFrames yet; from Python, use the RDD patterns in
[DataFrame-style operations](dataframe-operations.md).

```cpp
#include <vega/vega.hpp>
#include <vega/dataframe.hpp>

using namespace vega::df;

int main() {
    vega::SparkContext ctx(4);

    auto sales = DataFrame::from_columns(ctx, {
        {"store",    Column(std::vector<int64_t>{1, 2, 1, 3})},
        {"qty",      Column(std::vector<int64_t>{5, 1, 2, 7})},
        {"price",    Column(std::vector<double>{9.5, 3.0, 9.5, 1.25})},
        {"category", Column(std::vector<std::string>{"books", "toys", "books", "food"})},
    }, /*partitions=*/8);

    auto stores = DataFrame::from_columns(ctx, {
        {"store",  Column(std::vector<int64_t>{1, 2, 3})},
        {"region", Column(std::vector<std::string>{"east", "west", "east"})},
    }, 1);

    auto report = sales
        .filter(col("qty") > 1)
        .with_column("revenue", col("qty") * col("price"))
        .join(stores, {"store"})
        .group_by({col("region")})
        .agg({sum(col("revenue")).alias("revenue"), count_all().alias("orders")})
        .order_by({desc(col("revenue"))});

    std::cout << report.show();
    // +--------+---------+--------+
    // | region | revenue | orders |
    // +--------+---------+--------+
    // |   east |   75.25 |      3 |
    // +--------+---------+--------+

    vega::df::Batch rows = report.collect();         // columnar result
    const auto& revenue = rows.column("revenue").values<double>();
}
```

## Data model

| Type (`DType`) | C++ storage | Spark SQL type |
|----------------|-------------|----------------|
| `Int64` | `std::vector<int64_t>` | `bigint` |
| `Float64` | `std::vector<double>` | `double` |
| `String` | `std::vector<std::string>` | `string` |
| `Bool` | `std::vector<uint8_t>` (`BoolVec`) | `boolean` |

- A `Column` is an immutable, shared, typed vector; copying one is cheap.
- A `Batch` is a set of equal-length named columns. `collect()` returns one.
- Values are never null.

## Expressions

`col("x")` refers to a column. Literals convert implicitly, so `col("qty") > 3`
and `col("category") == "books"` work.

| Operators | Notes |
|-----------|-------|
| `+ - * %` | Integer operands stay integers; mixing with a double gives a double |
| `/` | Always a double, as in Spark SQL |
| `== != < <= > >=` | Numbers compare with numbers, strings with strings |
| `&& \|\| !` | On booleans |

- Integer division or modulo by zero throws `[DIVIDE_BY_ZERO]`, as Spark does in ANSI mode (the default in Spark 4).
- Output columns get Spark's names (`(qty * price)`, `sum(qty)`, `count(1)`) unless renamed with `.alias("name")`.
- Unknown columns and type errors throw when the DataFrame is built, not when it runs.

## Operations

| Operation | Spark equivalent | How it runs |
|-----------|------------------|-------------|
| `select({exprs})`, `with_column(name, e)` | `select`, `withColumn` | Narrow, fused |
| `filter(e)` / `where(e)` | `filter` / `where` | Narrow, fused |
| `join(right, {"k", ...})` | `join(right, Seq("k"))` (inner) | Broadcast hash join, fused |
| `group_by({keys}).agg({aggs})`, `.count()` | `groupBy(...).agg(...)` | Two-phase hash aggregate with a shuffle |
| `agg({aggs})` | `agg(...)` | Global aggregate |
| `distinct()` | `distinct()` | Hash aggregate on all columns |
| `order_by({asc(e), desc(e)})` | `orderBy` | Parallel sort, then k-way merge |
| `limit(n)` | `limit` | After `order_by`, keeps only n rows per partition (top-N) |
| `count()`, `collect()`, `show(n)` | `count`, `collect`, `show` | Actions |

Aggregates: `sum`, `count`, `count_all()` (`count(1)`), `avg`, `min`, `max`.
Their result types follow Spark: `sum(bigint)` gives `bigint`, `avg` gives
`double`, and `min`/`max` keep their input's type. A global aggregate over no
rows returns one row, with 0 for sums and counts and NaN for `avg`; `min` and
`max` throw. Spark returns nulls in this case.

## Execution

- **Fused narrow stages with column pruning.** `select`, `with_column`, `filter`
  and `join` are queued, then run as a single per-batch stage. Before running,
  Vega works out which columns later steps read and drops the rest as early as
  possible. So `df.filter(p).count()` evaluates only the predicate's columns,
  and a join followed by an aggregate carries only the columns the aggregate
  needs.
- **Vectorized expressions.** Each operator is one tight loop over a column,
  and literals are applied as scalars.
- **Hash aggregation:**
  1. Each partition groups its rows with an open-addressing hash table, using a
     dedicated table for a single integer key.
  2. It keeps per-group partial states.
  3. It splits them by key hash.
  4. The engine shuffle (`groupByKey`) brings each key's partial states together.
  5. The states are merged and finalized.
- **Broadcast hash join.** The right side is collected once and indexed when
  `join` is called; every left partition probes it. This suits the usual case
  of a large fact table joined with small dimension tables. A partitioned
  (shuffle) join for two large sides is not implemented yet.
- **Batches are serializable** (`vega::Serializer<df::Batch>`), so they can move
  through shuffles.

## Current limits

- No nulls, no outer, semi or anti joins, and no window functions.
- No SQL parser, and no readers for files such as Parquet or CSV. Build
  columns in C++, or create them from an RDD.
- No cost-based optimizer. Beyond column pruning and top-N, operations run in
  the order written, so filter before you join.
- No Python bindings yet.

## Performance

See [the README](../README.md#dataframes-vega-c-vs-spark-sql-vs-sail) for a
comparison with Spark SQL and Sail on the same queries, and
`benchmarks/bench_dataframe.py` to reproduce it.
