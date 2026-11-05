#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <vega/vega.hpp>

using namespace vega;

class ShuffleTest : public ::testing::Test {
protected:
    SparkContext ctx{4};
};

TEST_F(ShuffleTest, ReduceByKeyBasic) {
    std::vector<std::pair<std::string, int>> data = {
        {"a", 1}, {"b", 2}, {"a", 3}, {"b", 4}, {"c", 5},
    };

    auto result = ctx.parallelize(data, 2)
        .reduceByKey([](int a, int b) { return a + b; })
        .collect();

    // Sort for deterministic comparison.
    std::sort(result.begin(), result.end());
    ASSERT_EQ(result.size(), 3u);
    EXPECT_EQ(result[0], (std::pair<std::string, int>{"a", 4}));
    EXPECT_EQ(result[1], (std::pair<std::string, int>{"b", 6}));
    EXPECT_EQ(result[2], (std::pair<std::string, int>{"c", 5}));
}

TEST_F(ShuffleTest, ReduceByKeySinglePartition) {
    std::vector<std::pair<std::string, int>> data = {
        {"x", 10}, {"y", 20}, {"x", 30},
    };

    auto result = ctx.parallelize(data, 1)
        .reduceByKey([](int a, int b) { return a + b; }, 1)
        .collect();

    std::sort(result.begin(), result.end());
    ASSERT_EQ(result.size(), 2u);
    EXPECT_EQ(result[0], (std::pair<std::string, int>{"x", 40}));
    EXPECT_EQ(result[1], (std::pair<std::string, int>{"y", 20}));
}

TEST_F(ShuffleTest, ReduceByKeyWordCount) {
    std::vector<std::string> lines = {
        "hello world hello",
        "world of vega",
    };

    auto result = ctx.parallelize(lines, 2)
        .flatMap([](const std::string& line) {
            std::vector<std::string> words;
            std::string word;
            for (char c : line) {
                if (c == ' ') {
                    if (!word.empty()) words.push_back(word);
                    word.clear();
                } else {
                    word += c;
                }
            }
            if (!word.empty()) words.push_back(word);
            return words;
        })
        .map([](const std::string& w) -> std::pair<std::string, int> {
            return {w, 1};
        })
        .reduceByKey([](int a, int b) { return a + b; })
        .collect();

    std::sort(result.begin(), result.end());

    std::vector<std::pair<std::string, int>> expected = {
        {"hello", 2}, {"of", 1}, {"vega", 1}, {"world", 2},
    };
    EXPECT_EQ(result, expected);
}

TEST_F(ShuffleTest, GroupByKeyBasic) {
    std::vector<std::pair<std::string, int>> data = {
        {"a", 1}, {"b", 2}, {"a", 3}, {"b", 4}, {"c", 5},
    };

    auto result = ctx.parallelize(data, 2)
        .groupByKey()
        .collect();

    std::sort(result.begin(), result.end(),
              [](auto& x, auto& y) { return x.first < y.first; });

    ASSERT_EQ(result.size(), 3u);
    EXPECT_EQ(result[0].first, "a");
    EXPECT_EQ(result[1].first, "b");
    EXPECT_EQ(result[2].first, "c");

    auto sorted_a = result[0].second;
    std::sort(sorted_a.begin(), sorted_a.end());
    EXPECT_EQ(sorted_a, (std::vector<int>{1, 3}));

    auto sorted_b = result[1].second;
    std::sort(sorted_b.begin(), sorted_b.end());
    EXPECT_EQ(sorted_b, (std::vector<int>{2, 4}));

    EXPECT_EQ(result[2].second, (std::vector<int>{5}));
}

TEST_F(ShuffleTest, ReduceByKeyThenFilter) {
    std::vector<std::pair<std::string, int>> data = {
        {"a", 1}, {"b", 10}, {"a", 2}, {"c", 20},
    };

    auto result = ctx.parallelize(data, 2)
        .reduceByKey([](int a, int b) { return a + b; })
        .filter([](const std::pair<std::string, int>& p) {
            return p.second >= 10;
        })
        .collect();

    std::sort(result.begin(), result.end());
    ASSERT_EQ(result.size(), 2u);
    EXPECT_EQ(result[0], (std::pair<std::string, int>{"b", 10}));
    EXPECT_EQ(result[1], (std::pair<std::string, int>{"c", 20}));
}

TEST_F(ShuffleTest, DAGBuilderDetectsShuffleBoundary) {
    std::vector<std::pair<std::string, int>> data = {{"a", 1}};

    auto rdd = ctx.parallelize(data, 2)
        .reduceByKey([](int a, int b) { return a + b; });

    auto plan = DAGBuilder::build(0, rdd.impl());
    ASSERT_EQ(plan.stages.size(), 2u);
    EXPECT_FALSE(plan.stages[0].is_shuffle_boundary);
    EXPECT_TRUE(plan.stages[1].is_shuffle_boundary);
    EXPECT_NE(plan.stages[1].shuffle_rdd, nullptr);
}

TEST_F(ShuffleTest, ChainedShufflesExecuteAsMultipleStages) {
    std::vector<std::pair<int, int>> data;
    for (int i = 0; i < 100; ++i) data.emplace_back(i % 10, 1);

    // reduceByKey -> re-key -> reduceByKey: two shuffle boundaries.
    auto rdd = ctx.parallelize(data, 4)
        .reduceByKey([](int a, int b) { return a + b; })
        .map([](const std::pair<int, int>& kv) {
            return std::pair<int, int>{kv.first % 2, kv.second};
        })
        .reduceByKey([](int a, int b) { return a + b; }, 3);

    auto plan = DAGBuilder::build(0, rdd.impl());
    ASSERT_EQ(plan.stages.size(), 3u);
    EXPECT_FALSE(plan.stages[0].is_shuffle_boundary);
    EXPECT_TRUE(plan.stages[1].is_shuffle_boundary);
    EXPECT_TRUE(plan.stages[2].is_shuffle_boundary);

    auto result = rdd.collect();
    std::sort(result.begin(), result.end());
    ASSERT_EQ(result.size(), 2u);
    EXPECT_EQ(result[0], (std::pair<int, int>{0, 50}));
    EXPECT_EQ(result[1], (std::pair<int, int>{1, 50}));
    EXPECT_EQ(rdd.count(), 2u);
}

TEST_F(ShuffleTest, MaterializedShuffleIsReused) {
    std::vector<std::pair<int, int>> data = {{1, 1}, {2, 2}, {1, 3}};
    auto rdd = ctx.parallelize(data, 2)
        .reduceByKey([](int a, int b) { return a + b; });

    EXPECT_EQ(DAGBuilder::build(0, rdd.impl()).stages.size(), 2u);
    (void)rdd.collect();
    // Shuffle output is ready: the map stage is skipped.
    auto plan = DAGBuilder::build(0, rdd.impl());
    EXPECT_EQ(plan.stages.size(), 1u);
    // ...unless reuse is disabled.
    EXPECT_EQ(DAGBuilder::build(0, rdd.impl(), {}, false).stages.size(), 2u);

    auto again = rdd.collect();
    std::sort(again.begin(), again.end());
    ASSERT_EQ(again.size(), 2u);
    EXPECT_EQ(again[0], (std::pair<int, int>{1, 4}));
}

TEST_F(ShuffleTest, ExecutePartitionSubset) {
    std::vector<int> data = {0, 1, 2, 3, 4, 5, 6, 7};
    auto rdd = ctx.parallelize(data, 4);
    auto parts = ctx.execute_partitions(rdd.impl(), {3, 1});
    ASSERT_EQ(parts.size(), 2u);
    EXPECT_EQ(std::any_cast<std::vector<int>>(parts[0]), (std::vector<int>{6, 7}));
    EXPECT_EQ(std::any_cast<std::vector<int>>(parts[1]), (std::vector<int>{2, 3}));

    auto counts = ctx.execute_count_partitions(rdd.impl(), {0, 2});
    EXPECT_EQ(counts, (std::vector<size_t>{2u, 2u}));

    // Subset of a shuffled RDD still runs the full map stage first.
    std::vector<std::pair<int, int>> kv = {{1, 1}, {2, 2}, {3, 3}, {1, 1}};
    auto reduced = ctx.parallelize(kv, 2)
        .reduceByKey([](int a, int b) { return a + b; }, 2);
    auto sub = ctx.execute_partitions(reduced.impl(), {0});
    ASSERT_EQ(sub.size(), 1u);
    auto total = ctx.execute_count(reduced.impl());
    EXPECT_EQ(total, 3u);
}
