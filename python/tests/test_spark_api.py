"""Tests for the Spark RDD API surface, server-side operators and Vega extras."""

from __future__ import annotations

import gzip
import os
import statistics

import pytest

from pyvega import SparkContext, StatCounter, StorageLevel


@pytest.fixture(scope="module")
def sc():
    ctx = SparkContext(4)
    yield ctx
    ctx.stop()


# ----- regressions --------------------------------------------------------


class TestRegressions:
    def test_native_count_keeps_pending_transformations(self, sc):
        rdd = sc.range(0, 6).filter(lambda x: x % 2 == 0)
        assert rdd.count() == 3
        assert rdd.collect() == [0, 2, 4]

    def test_chained_shuffles(self, sc):
        pairs = sc.parallelize([(i % 5, i) for i in range(100)], 4)
        out = (
            pairs.reduceByKey(lambda a, b: a + b)
            .map(lambda kv: (kv[1] % 3, kv[0]))
            .groupByKey()
            .mapValues(sorted)
            .collect()
        )
        expected = {}
        for k in range(5):
            total = sum(i for i in range(100) if i % 5 == k)
            expected.setdefault(total % 3, []).append(k)
        assert dict(out) == {k: sorted(v) for k, v in expected.items()}

    def test_take_and_iterate_over_shuffled_rdd(self, sc):
        rdd = sc.parallelize([(i % 7, 1) for i in range(70)], 3).reduceByKey(lambda a, b: a + b)
        assert len(rdd.take(3)) == 3
        assert sorted(rdd.toLocalIterator()) == [(k, 10) for k in range(7)]

    def test_server_survives_task_errors(self, sc):
        with pytest.raises(ZeroDivisionError):
            sc.parallelize([1, 0, 2], 3).map(lambda x: 1 / x).collect()
        assert sc.parallelize([1, 2]).map(lambda x: x + 1).collect() == [2, 3]

    def test_pending_shuffle_partition_count(self, sc):
        rdd = sc.parallelize(range(30), 10).distinct(5)
        assert rdd.getNumPartitions() == 5
        assert rdd.count() == 30


# ----- pair RDD operations ------------------------------------------------


class TestPairOps:
    def test_join_family(self, sc):
        a = sc.parallelize([(1, "a"), (2, "b"), (3, "c")], 2)
        b = sc.parallelize([(1, "x"), (1, "y"), (4, "z")], 2)
        assert sorted(a.join(b).collect()) == [(1, ("a", "x")), (1, ("a", "y"))]
        assert sorted(a.leftOuterJoin(b).collect()) == [
            (1, ("a", "x")),
            (1, ("a", "y")),
            (2, ("b", None)),
            (3, ("c", None)),
        ]
        assert sorted(a.rightOuterJoin(b).collect()) == [
            (1, ("a", "x")),
            (1, ("a", "y")),
            (4, (None, "z")),
        ]
        assert len(a.fullOuterJoin(b).collect()) == 5

    def test_cogroup_and_group_with(self, sc):
        a = sc.parallelize([(1, 1), (2, 2)])
        b = sc.parallelize([(1, 10)])
        c = sc.parallelize([(2, 20), (2, 21)])
        got = {k: tuple(sorted(v) for v in vs) for k, vs in a.groupWith(b, c).collect()}
        assert got == {1: ([1], [10], []), 2: ([2], [], [20, 21])}

    def test_subtract_intersection(self, sc):
        a = sc.parallelize([1, 2, 3, 4, 4])
        b = sc.parallelize([3, 4, 5])
        assert sorted(a.subtract(b).collect()) == [1, 2]
        assert sorted(a.intersection(b).collect()) == [3, 4]
        kv = sc.parallelize([(1, "a"), (2, "b")])
        assert kv.subtractByKey(sc.parallelize([(2, None)])).collect() == [(1, "a")]

    def test_combine_aggregate_fold_by_key(self, sc):
        rdd = sc.parallelize([("a", 1), ("b", 2), ("a", 3)], 3)
        comb = rdd.combineByKey(lambda v: [v], lambda c, v: c + [v], lambda x, y: x + y)
        assert {k: sorted(v) for k, v in comb.collect()} == {"a": [1, 3], "b": [2]}
        assert dict(rdd.foldByKey(0, lambda a, b: a + b).collect()) == {"a": 4, "b": 2}
        agg = rdd.aggregateByKey((0, 0), lambda acc, v: (acc[0] + v, acc[1] + 1),
                                 lambda x, y: (x[0] + y[0], x[1] + y[1]))
        assert dict(agg.collect()) == {"a": (4, 2), "b": (2, 1)}

    def test_key_value_helpers(self, sc):
        rdd = sc.parallelize([(1, 2), (3, 4), (1, 6)])
        assert sorted(rdd.keys().collect()) == [1, 1, 3]
        assert sorted(rdd.values().collect()) == [2, 4, 6]
        assert sorted(rdd.mapValues(lambda v: v * 10).collect()) == [(1, 20), (1, 60), (3, 40)]
        assert sorted(rdd.flatMapValues(lambda v: [v, v]).collect())[:2] == [(1, 2), (1, 2)]
        assert dict(rdd.countByKey()) == {1: 2, 3: 1}
        assert sorted(rdd.lookup(1)) == [2, 6]
        assert sc.parallelize([("a", 1)]).collectAsMap() == {"a": 1}
        assert sc.parallelize(["x", "yy"]).keyBy(len).collect() == [(1, "x"), (2, "yy")]
        assert rdd.reduceByKeyLocally(lambda a, b: a + b) == {1: 8, 3: 4}

    def test_group_by_and_count_by_value(self, sc):
        grouped = sc.parallelize(range(10)).groupBy(lambda x: x % 2).mapValues(sorted)
        assert dict(grouped.collect()) == {0: [0, 2, 4, 6, 8], 1: [1, 3, 5, 7, 9]}
        assert dict(sc.parallelize("abracadabra").countByValue()) == {
            "a": 5, "b": 2, "r": 2, "c": 1, "d": 1
        }

    def test_partition_by_custom(self, sc):
        rdd = sc.parallelize([(i, i) for i in range(10)], 2).partitionBy(3, lambda k: k % 3)
        parts = rdd.glom().collect()
        assert len(parts) == 3
        for i, part in enumerate(parts):
            assert all(k % 3 == i for k, _ in part)


# ----- set / structural operations ---------------------------------------


class TestStructural:
    def test_distinct_union_cartesian(self, sc):
        assert sorted(sc.parallelize([1, 1, 2, 3, 3]).distinct().collect()) == [1, 2, 3]
        u = sc.parallelize([1, 2], 2).union(sc.parallelize([3], 1))
        assert u.getNumPartitions() == 3 and sorted(u.collect()) == [1, 2, 3]
        assert sorted((sc.parallelize([1]) + sc.parallelize([2])).collect()) == [1, 2]
        cart = sc.parallelize([1, 2]).cartesian(sc.parallelize(["a", "b"]))
        assert sorted(cart.collect()) == [(1, "a"), (1, "b"), (2, "a"), (2, "b")]
        assert cart.count() == 4

    def test_union_of_shuffled_rdds(self, sc):
        a = sc.parallelize([(1, 1), (1, 1)]).reduceByKey(lambda x, y: x + y)
        b = sc.parallelize([(2, 5)]).groupByKey().mapValues(list)
        assert sorted(sc.union([a, b]).collect()) == [(1, 2), (2, [5])]

    def test_zip_and_indexes(self, sc):
        a = sc.parallelize(range(6), 3)
        assert a.zip(a.map(lambda x: x * 10)).collect()[:2] == [(0, 0), (1, 10)]
        assert a.zipWithIndex().collect() == [(i, i) for i in range(6)]
        ids = a.zipWithUniqueId().map(lambda t: t[1]).collect()
        assert len(set(ids)) == 6

    def test_coalesce_repartition(self, sc):
        rdd = sc.parallelize(range(100), 8)
        assert rdd.coalesce(3).getNumPartitions() == 3
        assert sorted(rdd.coalesce(3).collect()) == list(range(100))
        assert rdd.coalesce(20).getNumPartitions() == 8
        rep = rdd.repartition(5)
        sizes = rep.glom().map(len).collect()
        assert len(sizes) == 5 and sum(sizes) == 100 and min(sizes) > 0

    def test_sorting(self, sc):
        data = [(5, "e"), (1, "a"), (3, "c"), (2, "b"), (4, "d")]
        rdd = sc.parallelize(data, 3)
        assert rdd.sortByKey().collect() == sorted(data)
        assert rdd.sortByKey(ascending=False).collect() == sorted(data, reverse=True)
        assert rdd.sortBy(lambda kv: kv[1], ascending=False).keys().collect() == [5, 4, 3, 2, 1]
        by_len = sc.parallelize([("bb", 1), ("a", 2), ("ccc", 3)], 2).sortByKey(keyfunc=len)
        assert by_len.keys().collect() == ["a", "bb", "ccc"]

    def test_glom_map_partitions_with_index(self, sc):
        rdd = sc.parallelize(range(4), 2)
        assert rdd.glom().collect() == [[0, 1], [2, 3]]
        out = rdd.mapPartitionsWithIndex(lambda i, it: [(i, sum(it))]).collect()
        assert out == [(0, 1), (1, 5)]


# ----- actions --------------------------------------------------------------


class TestActions:
    def test_numeric_actions_native_and_python(self, sc):
        values = [1.5, 2.5, 3.0, 10.0, -4.0]
        for rdd in (sc.parallelize(values, 2), sc.parallelize(values, 2).map(lambda x: x)):
            st = rdd.stats()
            assert st.count() == 5
            assert st.mean() == pytest.approx(statistics.fmean(values))
            assert rdd.variance() == pytest.approx(statistics.pvariance(values))
            assert rdd.sampleStdev() == pytest.approx(statistics.stdev(values))
            assert rdd.max() == 10.0 and rdd.min() == -4.0
            assert rdd.sum() == pytest.approx(sum(values))

    def test_int_sum_is_exact(self, sc):
        big = [2**62, 2**62, 2**62]
        assert sc.parallelize(big).sum() == 3 * 2**62
        assert sc.range(0, 1001).sum() == 500500

    def test_top_take_ordered(self, sc):
        rdd = sc.parallelize([5, 1, 9, 3, 7], 3)
        assert rdd.top(2) == [9, 7]
        assert rdd.takeOrdered(2) == [1, 3]
        assert rdd.takeOrdered(2, key=lambda x: -x) == [9, 7]

    def test_take_first_is_empty(self, sc):
        rdd = sc.parallelize(range(100), 10).map(lambda x: x * 2)
        assert rdd.take(5) == [0, 2, 4, 6, 8]
        assert rdd.first() == 0
        assert not rdd.isEmpty()
        assert sc.emptyRDD().isEmpty()
        with pytest.raises(ValueError):
            sc.emptyRDD().first()

    def test_reduce_fold_aggregate_tree(self, sc):
        rdd = sc.parallelize(range(1, 11), 3)
        assert rdd.reduce(lambda a, b: a + b) == 55
        assert rdd.fold(0, lambda a, b: a + b) == 55
        assert rdd.aggregate((0, 0), lambda a, x: (a[0] + x, a[1] + 1),
                             lambda a, b: (a[0] + b[0], a[1] + b[1])) == (55, 10)
        assert rdd.treeReduce(lambda a, b: a + b) == 55
        assert rdd.treeAggregate(0, lambda a, b: a + b, lambda a, b: a + b, depth=3) == 55
        with pytest.raises(ValueError):
            sc.emptyRDD().reduce(lambda a, b: a + b)

    def test_histogram(self, sc):
        rdd = sc.parallelize(range(51))
        assert rdd.histogram(2) == ([0, 25, 50], [25, 26])
        assert rdd.histogram([0, 10, 20, 50]) == ([0, 10, 20, 50], [10, 10, 31])

    def test_count_approx_distinct(self, sc):
        rdd = sc.parallelize(range(1000), 4)
        assert 950 < rdd.countApproxDistinct(0.03) < 1050
        assert 950 < rdd.map(str).countApproxDistinct(0.03) < 1050
        with pytest.raises(ValueError):
            rdd.countApproxDistinct(0.00000001)

    def test_approximate_actions(self, sc):
        rdd = sc.parallelize(range(1000), 8)
        assert rdd.countApprox(timeout=10_000) == 1000
        assert rdd.meanApprox(timeout=10_000) == pytest.approx(499.5)
        assert rdd.sumApprox(timeout=10_000) == pytest.approx(499500)

    def test_foreach_partition_and_pipe(self, sc):
        acc = sc.accumulator(0)
        sc.parallelize(range(10), 3).foreachPartition(lambda it: acc.add(sum(it)))
        assert acc.value == 45
        assert sc.parallelize(["b", "a"], 1).pipe("sort").collect() == ["a", "b"]

    def test_sampling(self, sc):
        rdd = sc.parallelize(range(1000), 4)
        assert set(rdd.sample(False, 0.1, 3).collect()) == set(rdd.sample(False, 0.1, 3).collect())
        a, b = rdd.randomSplit([0.3, 0.7], seed=1)
        assert sorted(a.collect() + b.collect()) == list(range(1000))
        assert len(rdd.takeSample(False, 20, 7)) == 20
        by_key = sc.parallelize([(i % 2, i) for i in range(1000)]).sampleByKey(
            False, {0: 0.0, 1: 1.0}, 5
        )
        assert all(k == 1 for k, _ in by_key.collect())


# ----- persistence, accumulators, I/O ----------------------------------


class TestPersistence:
    def test_cache_computes_once(self, sc):
        acc = sc.accumulator(0)

        def tick(x):
            acc.add(1)
            return x

        rdd = sc.parallelize(range(20), 4).map(tick).cache()
        assert rdd.is_cached and rdd.getStorageLevel() == StorageLevel.MEMORY_ONLY
        assert rdd.count() == 20
        assert rdd.sum() == 190
        assert acc.value == 20  # second action served from the cache
        assert rdd.cache_info()["cached_partitions"] == 4
        rdd.unpersist()
        assert not rdd.is_cached
        rdd.count()
        assert acc.value == 40

    def test_persisted_context_manager(self, sc):
        rdd = sc.parallelize(range(5))
        with rdd.persisted() as cached:
            assert cached.is_cached
            assert cached.count() == 5
        assert not rdd.is_cached

    def test_server_handles_are_released(self, sc):
        client = sc._client
        rdd = sc.parallelize(range(10)).map(lambda x: x + 1)
        rdd.count()
        sid = rdd._rdd_id
        del rdd
        sc.parallelize([1]).count()  # next RPC flushes releases
        with pytest.raises(RuntimeError, match="unknown RDD id"):
            client.describe(sid)


class TestAccumulators:
    def test_int_and_float(self, sc):
        acc = sc.accumulator(0)
        facc = sc.accumulator(0.0)
        sc.parallelize(range(10), 3).foreach(lambda x: (acc.add(x), facc.add(0.5)))
        assert acc.value == 45 and facc.value == 5.0
        acc += 5  # driver-side add
        assert acc.value == 50

    def test_value_not_readable_in_tasks(self, sc):
        acc = sc.accumulator(1)
        with pytest.raises(RuntimeError, match="cannot be accessed inside tasks"):
            sc.parallelize([1]).map(lambda _: acc.value).collect()


class TestIO:
    def test_text_file_splits_and_formats(self, sc, tmp_path):
        lines = [f"line {i} " + "x" * (i % 13) for i in range(500)]
        plain = tmp_path / "data.txt"
        plain.write_text("\n".join(lines) + "\n")
        rdd = sc.textFile(str(plain), minPartitions=7)
        assert rdd.getNumPartitions() >= 7
        assert rdd.collect() == lines
        gz = tmp_path / "data.txt.gz"
        with gzip.open(gz, "wt") as fh:
            fh.write("\n".join(lines))
        assert sc.textFile(str(gz)).collect() == lines
        assert sc.textFile(f"{plain},{gz}").count() == 1000
        assert sc.textFile(str(tmp_path / "*.txt")).count() == 500

    def test_text_file_is_lazy(self, sc, tmp_path):
        f = tmp_path / "gone.txt"
        f.write_text("Hello World!")
        rdd = sc.textFile(str(f)).filter(lambda x: True)
        assert rdd.count() == 1
        os.unlink(f)
        with pytest.raises(Exception):
            rdd.count()

    def test_save_and_reload(self, sc, tmp_path):
        rdd = sc.parallelize(range(100), 4)
        out = tmp_path / "txt"
        rdd.saveAsTextFile(str(out))
        assert (out / "_SUCCESS").exists()
        assert sorted(sc.textFile(str(out)).map(int).collect()) == list(range(100))
        gz = tmp_path / "gz"
        rdd.saveAsTextFile(str(gz), "org.apache.hadoop.io.compress.GzipCodec")
        assert any(p.endswith(".gz") for p in os.listdir(gz))
        assert sorted(sc.textFile(str(gz)).map(int).collect()) == list(range(100))
        pk = tmp_path / "pk"
        sc.parallelize([{"a": i} for i in range(30)], 3).saveAsPickleFile(str(pk))
        assert sorted(d["a"] for d in sc.pickleFile(str(pk)).collect()) == list(range(30))
        js = tmp_path / "js"
        sc.parallelize([{"k": 1}, {"k": 2}]).save_as_jsonl(str(js))
        assert sorted(d["k"] for d in sc.jsonl_file(str(js)).collect()) == [1, 2]

    def test_whole_text_files(self, sc, tmp_path):
        (tmp_path / "a.txt").write_text("alpha")
        (tmp_path / "b.txt").write_text("beta")
        got = dict(sc.wholeTextFiles(str(tmp_path)).collect())
        assert sorted(got.values()) == ["alpha", "beta"]


class TestContext:
    def test_range_and_metadata(self, sc):
        assert sc.range(1, 0, step=-1).count() == 1
        assert sc.range(0, 1 << 40, step=1 << 39).count() == 2
        assert sc.range(0, 10, step=3).collect() == [0, 3, 6, 9]
        assert sc.range(5).collect() == [0, 1, 2, 3, 4]
        assert sc.defaultParallelism == 4 and sc.defaultMinPartitions == 2
        assert sc.version
        assert sc.runJob(sc.parallelize(range(6), 3), lambda it: [sum(it)], [0, 2]) == [1, 9]
        assert sc.parallelize(range(10)).count() == 10  # native range fast path
        assert sc.status()["jobs"] > 0

    def test_stat_counter_merge(self):
        a = StatCounter([1.0, 2.0])
        b = StatCounter([3.0, 4.0, 5.0])
        merged = a.copy().mergeStats(b)
        assert merged.count() == 5
        assert merged.mean() == pytest.approx(3.0)
        assert merged.sampleVariance() == pytest.approx(2.5)


# ----- Vega extras (not in Spark) ---------------------------------------


class TestVegaExtras:
    def test_map_batches(self, sc):
        rdd = sc.parallelize(range(10), 2).map_batches(lambda batch: [sum(batch)], batch_size=3)
        assert sorted(rdd.collect()) == sorted([0 + 1 + 2, 3 + 4, 5 + 6 + 7, 8 + 9])

    def test_distinct_by_and_top_by_key(self, sc):
        people = sc.parallelize([("ann", 3), ("bob", 3), ("cy", 5)])
        assert sorted(x[1] for x in people.distinct_by(lambda p: p[1]).collect()) == [3, 5]
        scores = sc.parallelize([("a", 1), ("a", 9), ("a", 5), ("b", 2)], 2)
        assert dict(scores.top_by_key(2).collect()) == {"a": [9, 5], "b": [2]}

    def test_quantiles(self, sc):
        rdd = sc.parallelize(range(101), 4)
        assert rdd.quantiles([0.0, 0.5, 1.0]) == [0, 50, 100]
        approx = sc.parallelize(range(10_000), 4).quantiles([0.5], relative_error=0.01)
        assert abs(approx[0] - 5000) < 300

    def test_heavy_hitters(self, sc):
        data = ["a"] * 50 + ["b"] * 30 + list("cdefghijklmnop") * 2
        assert sc.parallelize(data, 3).heavy_hitters(2) == [("a", 50), ("b", 30)]

    def test_sliding(self, sc):
        windows = sc.parallelize(range(6), 3).sliding(3).collect()
        assert windows == [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]]
        assert sc.parallelize(range(6), 2).sliding(2, step=2).collect() == [[0, 1], [2, 3], [4, 5]]

    def test_explain_reports_native_fusion(self, sc):
        text = sc.range(0, 100).filter(lambda x: x % 2 == 0).map(lambda x: str(x)).explain()
        assert "native C++" in text and "Python UDF" in text


def test_count_approx_distinct_stable_across_hash_seeds(monkeypatch):
    """Regression: string hashing is randomized per server process, and with
    too few registers small cardinalities were estimated as 18 for 20 items
    under some seeds (upstream test_count_approx_distinct was flaky)."""
    data = [i % 20 for i in range(1000)]
    for seed in range(6):
        monkeypatch.setenv("PYTHONHASHSEED", str(seed))
        with SparkContext(2) as ctx:
            rdd = ctx.parallelize(data, 7)
            for variant in (rdd, rdd.map(str), rdd.map(lambda x: (x, -x))):
                assert 18 < variant.countApproxDistinct() < 22
