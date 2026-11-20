# C++ DataFrames

Vega has a native, columnar DataFrame API for C++ (`#include <vega/dataframe.hpp>`,
namespace `vega::df`), with Spark SQL semantics.

Each DataFrame is an immutable **logical plan**. Actions run a rule-based
**optimizer** over it, then execute the optimized plan as parallel RDD stages on
Vega's actor engine, shuffles included.

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

    std::cout << report.explain();                   // logical, optimized and physical plans
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

- **Columns.** A `Column` is an immutable, shared, typed vector, so copying one is cheap.
- **Nulls.** Any column can hold nulls, stored as an optional validity mask. A column without nulls carries no mask, so nulls cost nothing when unused. To create null values:
  - `Column(std::vector<std::optional<int64_t>>{1, std::nullopt})`, or
  - `Column(values, validity)`, or
  - `lit_null(DType::Int64)`.
- **Dictionary encoding.** Strings with few distinct values are dictionary-encoded automatically when a DataFrame is created or cached. Join outputs reference the build side the same way, with no copying. Encoding is transparent, and `values<T>()` decodes on first use.
- **Batches.** A `Batch` is a set of equal-length named columns. `collect()` returns one.
- **Inspecting values.** Read nullable values with `column.is_valid(i)` and `column.values<T>()[i]`.

## Expressions

`col("x")` refers to a column. Literals convert implicitly, so
`col("qty") > 3` and `col("category") == "books"` work.

| Kind | Expressions |
|------|-------------|
| Arithmetic | `+ - * / %`, unary `-` |
| Comparison | `== != < <= > >=`, `a.eq_null_safe(b)` (`<=>`), `a.between(lo, hi)`, `a.isin({...})` |
| Logic | `&& \|\| !` (three-valued) |
| Nulls | `is_null(e)`, `is_not_null(e)` (or `e.is_null()`), `coalesce({a, b, ...})` |
| Conditionals | `when(c, v).when(c2, v2).otherwise(e)` |
| Casts | `cast(e, DType::String)` / `e.cast(...)` (ANSI) |
| Strings | `upper`, `lower`, `trim`, `length`, `substring(e, pos, len)` (1-based), `concat({...})`, `e.contains(s)`, `e.starts_with(s)`, `e.ends_with(s)`, `e.like("a%_")` |
| Math | `abs`, `round(e, scale)`, `floor`, `ceil`, `sqrt` |

Semantics follow Spark SQL in ANSI mode, the default in Spark 4:

- **Typing.** Integer arithmetic stays `bigint`, `/` always gives a double, and mixed operands promote to double.
- **Null propagation.** A null input gives a null result, except for `is_null`, `coalesce`, `<=>`, and short-circuiting `AND`/`OR` (`false AND null` is `false`, `true OR null` is `true`).
- **Filters** keep rows whose predicate is `true`. Rows where it is `false` or null are dropped.
- **Errors.** Division by zero, invalid casts (`cast("x" AS BIGINT)`) and cast overflow raise errors. Null rows never raise.
- **Output names** follow Spark (`(qty * price)`, `sum(qty)`, `count(1)`, `CASE WHEN ... END`) unless renamed with `.alias("name")`.
- **Validation.** Unknown columns and type errors throw when the DataFrame is built, not when it runs.

## Operations

| Operation | Spark equivalent |
|-----------|------------------|
| `select({exprs})`, `select_columns({"a", "b"})`, `with_column(name, e)`, `with_column_renamed(a, b)`, `drop({...})` | `select`, `withColumn`, `withColumnRenamed`, `drop` |
| `filter(e)` / `where(e)` | `filter` / `where` |
| `group_by({keys}).agg({aggs})`, `.count()`, `agg({aggs})` | `groupBy(...).agg(...)`, `agg` |
| `join(right, {"k"}, JoinType::Left)` | `join(right, Seq("k"), "left")` |
| `join_on(right, {{"a", "b"}}, how)` | `join(right, $"a" === $"b", how)` |
| `cross_join(right)` | `crossJoin` |
| `union_all(other)`, `union_by_name(other)` | `union`, `unionByName` |
| `distinct()`, `drop_duplicates({"k"})` | `distinct`, `dropDuplicates` |
| `order_by({asc(e), desc(e), asc_nulls_last(e), desc_nulls_first(e)})` / `sort` | `orderBy` / `sort` |
| `limit(n)` | `limit` |
| `fill_null(value, {cols})`, `drop_nulls({cols})` | `na.fill`, `na.drop` |
| `repartition(n, {keys})` | `repartition` |
| `cache()` | `cache()` + an action (Vega materializes immediately) |
| `explain()` | `explain(true)` |
| `count()`, `collect()`, `head(n)`, `show(n)`, `is_empty()` | `count`, `collect`, `head`, `show`, `isEmpty` |
| `DataFrame::read_csv(ctx, path)`, `write_csv(path)` | `spark.read.csv` (header, schema inference), `df.write.csv` |

- **Join types:** `Inner`, `Left`, `Right`, `Full`, `LeftSemi`, `LeftAnti`, `Cross`.
- **Join output columns** follow Spark's using-column join:
  - the keys appear once: taken from the left side, from the right side for right joins, and coalesced for full joins;
  - then the other left columns, then the other right columns.
  - Semi and anti joins return only the left columns.
  - Null keys never match.
- **Aggregates:** `sum`, `count`, `count_all()` (`count(1)`), `avg`/`mean`, `min`, `max`, `variance`/`var_samp`, `var_pop`, `stddev`/`stddev_samp`, `stddev_pop`, `count_distinct`.
  - Aggregates skip nulls. `sum`, `avg`, `min`, `max` and the variances of no values return null; `count` returns 0.
  - Null grouping keys form their own group.
  - `count_distinct` can be combined only with other `count_distinct` calls on the same expression.
- **Window functions:** `row_number()`, `rank()`, `dense_rank()`, `lag(e, n[, default])`, `lead(e, n[, default])`, and `sum`, `avg`, `count`, `count_all`, `min`, `max` via `.over(spec)`.
  - Use them in `select` or `with_column`, even inside larger expressions:

    ```cpp
    auto w = Window::partition_by({col("store")}).order_by({desc(col("qty"))});
    sales.with_column("rank", rank().over(w))
         .with_column("share", col("qty") / sum(col("qty")).over(Window::partition_by({col("store")})));
    ```

  - With an `order_by`, aggregates use Spark's default frame: a running total, with ties included. Without one, they cover the whole partition.

## The optimizer

`explain()` prints the plan before and after optimization, plus the physical
strategy. The optimizer applies these rules:

| Rule | Effect |
|------|--------|
| Constant folding | Literal subexpressions are evaluated once. `x AND true` → `x`; `NOT NOT x` → `x` |
| Filter pruning | `filter(true)` disappears. `filter(false)` makes the subtree empty, and the emptiness propagates |
| Predicate pushdown | Filters move below projections (with aliases substituted), sorts, unions and repartitions. They also move below windows (partition-key predicates), aggregates (grouping-key predicates) and joins (each conjunct to the side it reads) |
| Outer-join elimination | A null-rejecting filter on the null-supplying side turns an outer join into an inner, left or right join |
| Join predicate inference | A filter on a join key is copied to the other side |
| Column pruning | Every node produces only the columns used above it, down to the scans |
| Project collapsing | Adjacent projections merge (unless that would repeat an expensive expression), and identity projections disappear |
| Sort elimination | A sort that feeds an aggregate, join, window or another sort is dropped |
| Limit pushdown | Limits move through projections and into union inputs. `order_by(...).limit(n)` runs as a parallel top-N |
| Eager aggregation | A group-by over an inner join whose aggregates read one side is pre-aggregated by that side's join keys before the join (for example, sales grouped by store before joining stores), then combined. It applies to `sum`, `count`, `min`, `max` and `avg`. It fires only when the statistics show the pre-aggregation shrinks that side at least 4×: the join keys' estimated distinct-value count must be at most a quarter of the side's estimated row count. So a fact table with few distinct keys is pre-aggregated, while line items joined to orders (about 2 items per order) are not |

**Statistics.** Creating or caching a DataFrame (`from_columns`,
`from_batch`, `read_csv`, `cache()`) also records statistics for every
column:

- **Row counts** are exact.
- **Null counts** are exact.
- **Distinct-value counts** are exact for dictionary-encoded columns and
  estimated with a HyperLogLog sketch for the others (2^12 registers, 4 KB
  per column, about 1.6% error).
- **Cost.** Partitions are sketched in parallel on the engine and merged. For
  a 10M-row, 6-column table this adds about 150 ms on 4 cores.

The estimates flow through the plan: projections pass them through, and
joins, filters and limits cap them. They drive three decisions:

- **Group-by sizes:** the product of the key NDVs, capped by the input rows.
- **Filter selectivity:** `col = literal` is `1 / NDV(col)`; other predicates use a fixed 0.5.
- **Optimizer choices:** whether eager aggregation fires, and whether a join is broadcast or shuffled.

`df.estimated_rows()` and `df.estimated_distinct("col")` expose the estimates.

**Physical planning:**

- **Join strategy.**
  - A join side estimated under `broadcast_threshold()` (32 MB by default) is broadcast: collected once, hashed, and probed by every partition.
  - Otherwise both sides are hash-partitioned through the engine shuffle and joined per partition (a shuffle hash join). Full outer joins always use this.
  - Set `vega::df::broadcast_threshold() = bytes` to tune it; 0 disables broadcasting.
- **Fused stages.** Projections, filters and broadcast-join probes run as one fused per-batch stage.
- **Aggregates** are two-phase hash aggregates around a shuffle, with dedicated hash tables for one or two integer keys and for dictionary-encoded keys.

## Execution details

- **Vectorized expressions.** Each operator is one tight loop over a column, and literals are applied as scalars.
- **Dictionary-aware evaluation.** An expression over a single dictionary-encoded column, such as `upper(col("cat")) == "X"`, is evaluated once per distinct value and then mapped to rows.
- **Shuffle format.** Batches are serializable (`vega::Serializer<df::Batch>`), validity masks included, so they can move through shuffles.

## Current limits

- Joins are equi-joins. Non-equi conditions go in a `filter` after a join or cross join.
- No SQL parser and no Parquet reader. Sources are in-memory columns, CSV, and RDDs.
- No date or decimal types.
- Window frames are the two Spark defaults only: a running frame with `ORDER BY`, the whole partition without.
- String functions are byte-oriented, except `length` and `substring`, which count UTF-8 code points.
- Statistics cover sources and cached data only. Computed columns have no NDV estimate. Predicates other than `col = literal` get a fixed selectivity, and there are no histograms.
- No Python bindings yet.

## Performance

See [the README](../README.md#dataframes-vega-c-vs-spark-sql-vs-sail) for a
comparison with Spark SQL and Sail on the same queries, and
`benchmarks/bench_dataframe.py` to reproduce it.
