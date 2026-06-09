import scala.collection.mutable.ArrayBuffer

val n = sys.env.getOrElse("BENCH_N", "5000000").toInt
val partitions = sys.env.getOrElse("BENCH_PARTITIONS", "32").toInt
val repeats = sys.env.getOrElse("BENCH_REPEATS", "5").toInt

def runOnce(): Long = {
  sc.parallelize(0L until n.toLong, partitions)
    .map(x => (x * 31L + 7L) % 1000003L)
    .filter(x => x % 7L == 0L)
    .flatMap(x => Seq(x, x + 1L, x + 2L))
    .count()
}

def median(vals: scala.collection.Seq[Double]): Double = {
  val sorted = vals.sorted
  val m = sorted.length / 2
  if (sorted.length % 2 == 1) sorted(m) else (sorted(m - 1) + sorted(m)) / 2.0
}

def mean(vals: scala.collection.Seq[Double]): Double = if (vals.isEmpty) 0.0 else vals.sum / vals.length

// Warmup.
runOnce()

val times = ArrayBuffer[Double]()
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
