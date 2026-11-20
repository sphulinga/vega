// Spark SQL (Scala DataFrame API) counterpart of benchmarks/vega_cpp_dataframe.cpp.
//
// The same tables (generated from the same formulas, cached in memory) and the
// same queries. Run through benchmarks/bench_dataframe.py, or directly:
//
//   BENCH_N=10000000 spark-shell --master "local[4]" --conf spark.ui.enabled=false \
//     --conf spark.sql.shuffle.partitions=8 -i benchmarks/spark_scala_dataframe.scala
//
// Prints one "RESULT\t<query>\t<best ms>\t<value>" line per query.

import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.functions._

val n = sys.env.getOrElse("BENCH_N", "10000000").toLong
val repeat = sys.env.getOrElse("BENCH_REPEAT", "5").toInt
val warmup = sys.env.getOrElse("BENCH_WARMUP", "1").toInt
val parts = sys.env.getOrElse("BENCH_PARTITIONS", "8").toInt

spark.sparkContext.setLogLevel("WARN")

val sales = spark.range(0, n, 1, parts).select(
  col("id"),
  (col("id") % 1000).as("store"),
  ((col("id") * 7919) % 100000).as("product"),
  (col("id") % 10 + 1).as("qty"),
  ((col("id") * 13) % 1000 / 10.0 + 1.0).as("price"),
  concat(lit("cat"), (col("id") % 8).cast("string")).as("category")
).cache()
sales.count()
val stores = spark.range(0, 1000, 1, 1).select(
  col("id").as("store"),
  concat(lit("r"), (col("id") % 5).cast("string")).as("region")
).cache()
stores.count()

def fmt(v: Any): String = v match {
  case d: Double => java.lang.Double.toString(d)
  case x => x.toString
}
def row(df: DataFrame): String = df.collect()(0).toSeq.map(fmt).mkString("|")

val queries: Seq[(String, () => String)] = Seq(
  "filter+count" -> (() => sales.filter(col("qty") > 5 && col("price") < 50.0).count().toString),
  "sum(qty*price)" -> (() => row(sales.agg(sum(col("qty") * col("price"))))),
  "groupBy store (1k keys)" -> (() => row(
    sales.groupBy("store").agg(sum("qty").as("u"), avg("price").as("a"))
      .agg(count(lit(1)), sum("u"), sum("a")))),
  "groupBy category (string, 8 keys)" -> (() => row(
    sales.groupBy("category").agg(sum(col("qty") * col("price")).as("rev"), count(lit(1)).as("c"))
      .agg(count(lit(1)), sum("rev"), sum("c")))),
  "groupBy product (100k keys)" -> (() => row(
    sales.groupBy("product").agg(count(lit(1)).as("c"), max("price").as("m"))
      .agg(count(lit(1)), sum("c"), sum("m")))),
  "groupBy category, qty (40 keys)" -> (() => row(
    sales.groupBy("category", "qty").agg(count(lit(1)).as("c"), sum("price").as("p"))
      .agg(count(lit(1)), sum("c"), sum("p")))),
  "join stores + groupBy region" -> (() => row(
    sales.join(stores, Seq("store")).groupBy("region").agg(sum("qty").as("u"))
      .agg(count(lit(1)), sum("u")))),
  "distinct (product, qty)" -> (() => sales.select("product", "qty").distinct().count().toString),
  "orderBy price desc limit 10" -> (() => {
    val rows = sales.orderBy(col("price").desc, col("id").asc).limit(10).collect()
    s"${rows.length}|${rows.map(_.getLong(0)).sum}|${fmt(rows(0).getDouble(4))}"
  })
)

for (_ <- 0 until warmup; (_, q) <- queries) q()

for ((name, q) <- queries) {
  var best = Long.MaxValue
  var value = ""
  for (_ <- 0 until repeat) {
    val t0 = System.nanoTime()
    value = q()
    best = math.min(best, System.nanoTime() - t0)
  }
  println(s"RESULT\t$name\t${best / 1e6}\t$value")
}

System.exit(0)
