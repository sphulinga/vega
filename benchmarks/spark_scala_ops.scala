// Apache Spark (Scala RDD API) counterpart of benchmarks/bench_pyvega_ops.py.
//
// Same datasets, sizes, partition counts and operations, so the results can be
// put side by side with PyVega's. Run it through bench_spark_scala_ops.py, or
// directly with:
//
//   BENCH_N=200000 BENCH_REPEAT=3 spark-shell --master "local[4]" \
//     --conf spark.ui.enabled=false -i benchmarks/spark_scala_ops.scala
//
// Prints one "RESULT\t<operation>\t<best ms>" line per operation.

val n = sys.env.getOrElse("BENCH_N", "200000").toInt
val repeat = sys.env.getOrElse("BENCH_REPEAT", "3").toInt
val parts = 8

sc.setLogLevel("WARN")

val objs: Seq[(Int, Double)] = (0 until n).map(i => (i % 1000, i.toDouble))
val words: Seq[String] = (0 until n).map(i => s"w${i % 5000}")
val nums: Seq[Long] = (0 until n).map(_.toLong)
val pairs = sc.parallelize(objs, parts)
val wrdd = sc.parallelize(words, parts)
val nrdd = sc.parallelize(nums, parts)

// Best wall time over `repeat` runs, in milliseconds (same as _best() in the
// Python benchmark: the first run is not discarded separately).
def best(f: () => Any): Double = {
  var bestNs = Long.MaxValue
  for (_ <- 0 until repeat) {
    val t0 = System.nanoTime()
    f()
    bestNs = math.min(bestNs, System.nanoTime() - t0)
  }
  bestNs / 1e6
}

val cases: Seq[(String, () => Any)] = Seq(
  "parallelize+collect (tuples)" -> (() => sc.parallelize(objs, parts).collect()),
  "map+filter python (tuples)" -> (() =>
    pairs.map(kv => (kv._1, kv._2 * 2)).filter(kv => kv._1 % 2 == 0).count()),
  "reduceByKey (tuples)" -> (() => pairs.reduceByKey(_ + _).count()),
  "groupByKey (tuples)" -> (() => pairs.groupByKey().count()),
  "distinct (strings)" -> (() => wrdd.distinct().count()),
  "sortByKey (tuples)" -> (() => pairs.sortByKey().count()),
  "mapPartitions" -> (() => pairs.mapPartitions(it => it).count()),
  "union x2" -> (() => sc.union(pairs, pairs).count()),
  "repartition(16)" -> (() => pairs.repartition(16).count()),
  "aggregateByKey" -> (() => pairs.aggregateByKey(0.0)(_ + _, _ + _).count()),
  "take(10)" -> (() => pairs.map(kv => kv).take(10)),
  "first()" -> (() => pairs.map(kv => kv).first()),
  "sum (ints)" -> (() => nrdd.sum()),
  "aggregate" -> (() => pairs.aggregate(0.0)((a, kv) => a + kv._2, _ + _)),
  "chained shuffle" -> (() =>
    pairs.reduceByKey(_ + _).map(kv => (kv._1 % 10, kv._2)).groupByKey().count())
)

// Warm-up pass: the JVM's JIT compiler speeds code up over its first
// executions, so run every operation untimed first and report Spark at its
// best. (PyVega needs no equivalent: CPython has no JIT warm-up.)
val warmup = sys.env.getOrElse("BENCH_WARMUP", "1").toInt
for (_ <- 0 until warmup; (_, f) <- cases) f()

for ((name, f) <- cases) {
  val ms = best(f)
  println(f"RESULT\t$name\t$ms%.3f")
}
System.exit(0)
