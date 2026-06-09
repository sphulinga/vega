"""Tests for the PyVega client (Vega C++ server)."""

import pytest
from pyvega import SparkContext, RDD


@pytest.fixture(scope="module")
def sc():
    ctx = SparkContext(2)
    yield ctx
    ctx.stop()


class TestParallelizeAndCollect:
    def test_ints(self, sc):
        result = sc.parallelize([1, 2, 3, 4, 5], 2).collect()
        assert sorted(result) == [1, 2, 3, 4, 5]

    def test_strings(self, sc):
        result = sc.parallelize(["a", "b", "c"]).collect()
        assert sorted(result) == ["a", "b", "c"]

    def test_from_generator(self, sc):
        result = sc.parallelize(range(5)).collect()
        assert sorted(result) == [0, 1, 2, 3, 4]


class TestMap:
    def test_double(self, sc):
        result = sc.parallelize([1, 2, 3], 2).map(lambda x: x * 2).collect()
        assert sorted(result) == [2, 4, 6]


class TestFilter:
    def test_evens(self, sc):
        result = sc.parallelize(range(1, 7), 2).filter(lambda x: x % 2 == 0).collect()
        assert sorted(result) == [2, 4, 6]


class TestFlatMap:
    def test_split_words(self, sc):
        lines = ["hello world", "foo bar baz"]
        result = sc.parallelize(lines, 2).flat_map(lambda s: s.split()).collect()
        assert sorted(result) == ["bar", "baz", "foo", "hello", "world"]


class TestCount:
    def test_basic(self, sc):
        assert sc.parallelize(range(10)).count() == 10


class TestReduce:
    def test_sum(self, sc):
        assert sc.parallelize([1, 2, 3, 4, 5]).reduce(lambda a, b: a + b) == 15


class TestReduceByKey:
    def test_sum_by_key(self, sc):
        pairs = [(1, 10), (2, 20), (1, 5), (2, 3)]
        result = (
            sc.parallelize(pairs, 2)
            .reduceByKey(lambda a, b: a + b)
            .collect()
        )
        assert sorted(result) == [(1, 15), (2, 23)]


class TestMapBeforeShuffle:
    def test_parallelize_map_reduce_by_key_collect(self, sc):
        result = (
            sc.parallelize([1, 2, 3, 1], 2)
            .map(lambda x: (x, 1))
            .reduceByKey(lambda a, b: a + b)
            .collect()
        )
        assert sorted(result) == [(1, 2), (2, 1), (3, 1)]

    def test_parallelize_map_reduce_by_key_count(self, sc):
        count = (
            sc.parallelize([1, 2, 3, 1], 2)
            .map(lambda x: (x, 1))
            .reduceByKey(lambda a, b: a + b)
            .count()
        )
        assert count == 3

    def test_range_map_reduce_by_key(self, sc):
        result = (
            sc.range(0, 6)
            .map(lambda x: (x % 2, 1))
            .reduceByKey(lambda a, b: a + b)
            .collect()
        )
        assert sorted(result) == [(0, 3), (1, 3)]

    def test_range_map_reduce_by_key_count(self, sc):
        rows = (
            sc.range(0, 10)
            .map(lambda x: (x % 10, 1))
            .reduceByKey(lambda a, b: a + b)
            .collect()
        )
        assert len(rows) == 10

    def test_range_map_reduce_by_key_count_many_partitions(self, sc):
        """Chained lambdas on one line must compile both map and reduce natively."""
        count = (
            sc.range(0, 10_000, 32)
            .map(lambda x: (x % 1000, 1))
            .reduceByKey(lambda a, b: a + b, 32)
            .count()
        )
        assert count == 1000


class TestRange:
    def test_range_count(self, sc):
        assert sc.range(0, 10).count() == 10


class TestNativeFloat:
    def test_parallelize_map_collect(self, sc):
        result = sc.parallelize([1.0, 2.0, 3.0, 4.0], 2).map(lambda x: x * 2.0 + 1.0).collect()
        assert sorted(result) == [3.0, 5.0, 7.0, 9.0]

    def test_parallelize_filter_count(self, sc):
        data = [0.5, 1.5, 2.5, 3.5, 4.5]
        count = sc.parallelize(data, 2).filter(lambda x: x > 2.0).count()
        assert count == 3

    def test_parallelize_flat_map_count(self, sc):
        count = (
            sc.parallelize([1.0, 2.0], 2)
            .flat_map(lambda x: (x, x + 1.0, x + 2.0))
            .count()
        )
        assert count == 6


class TestNativeString:
    def test_parallelize_collect(self, sc):
        result = sc.parallelize(["a", "b", "c"], 2).collect()
        assert sorted(result) == ["a", "b", "c"]

    def test_split_words(self, sc):
        lines = ["hello world", "foo bar baz"]
        result = sc.parallelize(lines, 2).flat_map(lambda s: s.split()).collect()
        assert sorted(result) == ["bar", "baz", "foo", "hello", "world"]

    def test_word_count(self, sc):
        lines = ["hello world", "hello spark"]
        result = (
            sc.parallelize(lines, 2)
            .flat_map(lambda s: s.split())
            .map(lambda w: (w, 1))
            .reduceByKey(lambda a, b: a + b)
            .collect()
        )
        assert sorted(result) == [("hello", 2), ("spark", 1), ("world", 1)]

    def test_filter_length(self, sc):
        result = (
            sc.parallelize(["a", "bb", "ccc", "dddd"], 2)
            .filter(lambda s: len(s) > 2)
            .collect()
        )
        assert sorted(result) == ["ccc", "dddd"]


    def test_string_lines_word_count(self, sc):
        count = (
            sc.string_lines(0, 100, 10, 5, 2)
            .flat_map(lambda s: s.split())
            .map(lambda w: (w, 1))
            .reduceByKey(lambda a, b: a + b, 2)
            .count()
        )
        assert count == 15  # 10 distinct word* keys + 5 distinct tok* keys

    def test_float_shuffle_count(self, sc):
        count = (
            sc.float_range(0, 100, 2)
            .map(lambda x: (int(x) % 10, 1))
            .reduceByKey(lambda a, b: a + b, 2)
            .count()
        )
        assert count == 10

    def test_narrow_int_pipeline(self, sc):
        count = (
            sc.range(0, 1000, 4)
            .map(lambda x: (x * 31 + 7) % 1_000_003)
            .filter(lambda x: x % 7 == 0)
            .flat_map(lambda x: (x, x + 1, x + 2))
            .count()
        )
        assert count > 0

    def test_bytecode_fallback_narrow(self, sc):
        """Lambdas without source (e.g. REPL) still compile via bytecode patterns."""
        ns: dict[str, object] = {}
        exec(
            "run = lambda sc: sc.range(0, 500, 4)"
            ".map(lambda x: (x * 31 + 7) % 1_000_003)"
            ".filter(lambda x: x % 7 == 0)"
            ".flat_map(lambda x: (x, x + 1, x + 2))"
            ".count()",
            ns,
        )
        assert ns["run"](sc) > 0


class TestPyVegaFeatures:
    def test_empty_rdd_sum(self, sc):
        assert sc.empty_rdd().sum() == 0

    def test_first_and_take(self, sc):
        rdd = sc.range(0, 10, 2)
        assert rdd.first() == 0
        assert rdd.take(3) == [0, 1, 2]

    def test_group_by_key(self, sc):
        pairs = sc.parallelize([(1, 1), (1, 2), (2, 3)], 2)
        grouped = {k: sorted(v) for k, v in pairs.group_by_key().collect()}
        assert grouped == {1: [1, 2], 2: [3]}

    def test_aggregate_by_key(self, sc):
        pairs = sc.parallelize([(1, 1), (1, 2), (2, 3)], 2)
        result = dict(
            pairs.aggregate_by_key(0, lambda a, b: a + b, lambda a, b: a + b).collect()
        )
        assert result == {1: 3, 2: 3}

    def test_zip_validation(self, sc):
        a = sc.parallelize([1, 2, 3], 2)
        b = sc.parallelize([4, 5], 2)
        with pytest.raises(ValueError):
            a.zip(b).collect()

    def test_sample_deterministic(self, sc):
        rdd = sc.parallelize(range(100), 4)
        a = set(rdd.sample(False, 0.1, 2).collect())
        b = set(rdd.sample(False, 0.1, 2).collect())
        assert a == b

    def test_sort_by_key_partitions(self, sc):
        seq = [(i * 59 % 101, i) for i in range(101)]
        rdd = sc.parallelize(seq, 4)
        sort = rdd.sortByKey(numPartitions=5)
        assert sort.collect() == sorted(seq)
        sizes = sort.glom().map(len).collect()
        assert all(size > 0 for size in sizes)

    def test_save_and_read_text_file(self, sc, tmp_path):
        path = tmp_path / "text_out"
        sc.parallelize(["a", "b"]).saveAsTextFile(str(path))
        assert sc.textFile(str(path)).collect() == ["a", "b"]

    def test_broadcast(self, sc):
        b = sc.broadcast([1, 2, 3])
        assert sc.parallelize([0]).map(lambda _: sum(b.value)).collect() == [6]


class TestContextManager:
    def test_with_statement(self):
        with SparkContext(2) as sc:
            result = sc.parallelize([1, 2, 3]).collect()
            assert sorted(result) == [1, 2, 3]
