val n = sys.env.getOrElse("BENCH_N", "20000000").toLong
val partitions = sys.env.getOrElse("BENCH_PARTITIONS", "64").toInt
val repeats = sys.env.getOrElse("BENCH_REPEATS", "3").toInt
val numKeys = sys.env.getOrElse("BENCH_NUM_KEYS", "1000000").toLong

def runOnce(): Long = {
  import spark.implicits._
  val df = spark.range(0L, n, 1L, partitions)
    .selectExpr(
      s"((id * 1315423911 + 7) % $numKeys) as k",
      "((id * 31 + 7) % 97) as v"
    )
    .repartition(partitions, $"k")
    .groupBy("k")
    .sum("v")
  df.count()
}

def median(vals: scala.collection.Seq[Double]): Double = {
  val sorted = vals.sorted
  val m = sorted.length / 2
  if (sorted.length % 2 == 1) sorted(m) else (sorted(m - 1) + sorted(m)) / 2.0
}

def mean(vals: scala.collection.Seq[Double]): Double = if (vals.isEmpty) 0.0 else vals.sum / vals.length

runOnce() // warmup

val times = scala.collection.mutable.ArrayBuffer[Double]()
var lastCount = 0L
for (_ <- 0 until repeats) {
  val t0 = System.nanoTime()
  lastCount = runOnce()
  val t1 = System.nanoTime()
  times += (t1 - t0).toDouble / 1e9
}

println(
  s"RESULT count=$lastCount median_s=${median(times)} mean_s=${mean(times)} times_s=${times.mkString(",")}"
)
System.exit(0)
