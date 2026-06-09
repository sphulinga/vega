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
