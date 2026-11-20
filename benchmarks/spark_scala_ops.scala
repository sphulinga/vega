// Apache Spark (Scala RDD API) counterpart of benchmarks/vega_cpp_ops.cpp.
//
// The same operations, datasets, element types and partitioning as Vega's C++
// benchmark, so the two native APIs are compared like for like. Run it
// through bench_spark_scala_ops.py, or directly with:
//
//   BENCH_N=1000000 BENCH_REPEAT=3 spark-shell --master "local[4]" \
//     --conf spark.ui.enabled=false -i benchmarks/spark_scala_ops.scala
//
// Prints one "RESULT\t<operation>\t<best ms>\t<result>" line per operation;
// the result (a count or sum) lets the runner check that both engines did the
// same work.

val n = sys.env.getOrElse("BENCH_N", "1000000").toInt
val repeat = sys.env.getOrElse("BENCH_REPEAT", "3").toInt
val warmup = sys.env.getOrElse("BENCH_WARMUP", "1").toInt
val parts = sys.env.getOrElse("BENCH_PARTITIONS", "8").toInt

sc.setLogLevel("WARN")

// Same datasets as vega_cpp_ops.cpp: (i % 1000, i) pairs and ints.
val objs: Seq[(Int, Double)] = (0 until n).map(i => (i % 1000, i.toDouble))
val nums: Seq[Long] = (0 until n).map(_.toLong)
val pairs = sc.parallelize(objs, parts)
val nrdd = sc.parallelize(nums, parts)

// Best wall time over `repeat` runs, in milliseconds, plus the last result.
def best(f: () => Long): (Double, Long) = {
  var bestNs = Long.MaxValue
  var result = 0L
  for (_ <- 0 until repeat) {
    val t0 = System.nanoTime()
    result = f()
    bestNs = math.min(bestNs, System.nanoTime() - t0)
  }
  (bestNs / 1e6, result)
}

val cases: Seq[(String, () => Long)] = Seq(
  "parallelize+collect (pairs)" -> (() => sc.parallelize(objs, parts).collect().length.toLong),
  "map+filter (pairs)" -> (() =>
    pairs.map(kv => (kv._1, kv._2 * 2)).filter(kv => kv._1 % 2 == 0).count()),
  "flatMap x3 (ints)" -> (() => nrdd.flatMap(x => Seq(x, x + 1, x + 2)).count()),
  "reduce sum (ints)" -> (() => nrdd.reduce(_ + _)),
  "reduceByKey (pairs)" -> (() => pairs.reduceByKey(_ + _).count()),
  "groupByKey (pairs)" -> (() => pairs.groupByKey().count()),
  "chained shuffle" -> (() =>
    pairs.reduceByKey(_ + _).map(kv => (kv._1 % 10, kv._2)).groupByKey().count()),
  "narrow pipeline (ints)" -> (() =>
    nrdd.map(x => (x * 31L + 7L) % 1000003L)
      .filter(x => x % 7L == 0L)
      .flatMap(x => Seq(x, x + 1L, x + 2L))
      .count())
)

// Warm-up pass: the JVM's JIT compiler speeds code up over its first
// executions, so run every operation untimed first (vega_cpp_ops does the
// same).
for (_ <- 0 until warmup; (_, f) <- cases) f()

for ((name, f) <- cases) {
  val (ms, result) = best(f)
  println(f"RESULT\t$name\t$ms%.3f\t$result")
}
System.exit(0)
