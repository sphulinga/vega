#include <gtest/gtest.h>
#include <vega/vega.hpp>

#include <algorithm>
#include <numeric>
#include <string>

class RDDTest : public ::testing::Test {
protected:
    vega::SparkContext ctx{2};
};

TEST_F(RDDTest, ParallelizeAndCollect) {
    auto rdd = ctx.parallelize(std::vector<int>{1, 2, 3, 4, 5}, 2);
    auto result = rdd.collect();
    std::sort(result.begin(), result.end());
    EXPECT_EQ(result, (std::vector<int>{1, 2, 3, 4, 5}));
}

TEST_F(RDDTest, MapTransform) {
    auto rdd = ctx.parallelize(std::vector<int>{1, 2, 3}, 2);
    auto doubled = rdd.map([](const int& x) { return x * 2; });

    auto result = doubled.collect();
    std::sort(result.begin(), result.end());
    EXPECT_EQ(result, (std::vector<int>{2, 4, 6}));
}

TEST_F(RDDTest, FilterTransform) {
    auto rdd = ctx.parallelize(std::vector<int>{1, 2, 3, 4, 5, 6}, 3);
    auto evens = rdd.filter([](const int& x) { return x % 2 == 0; });

    auto result = evens.collect();
    std::sort(result.begin(), result.end());
    EXPECT_EQ(result, (std::vector<int>{2, 4, 6}));
}

TEST_F(RDDTest, FlatMapTransform) {
    auto rdd = ctx.parallelize(
        std::vector<std::string>{"a b", "c d e"}, 2);

    auto words = rdd.flatMap([](const std::string& line) {
        std::vector<std::string> out;
        std::istringstream iss(line);
        std::string w;
        while (iss >> w) out.push_back(w);
        return out;
    });

    auto result = words.collect();
    std::sort(result.begin(), result.end());
    EXPECT_EQ(result, (std::vector<std::string>{"a", "b", "c", "d", "e"}));
}

TEST_F(RDDTest, ChainedTransforms) {
    auto result = ctx.parallelize(std::vector<int>{1, 2, 3, 4, 5, 6}, 2)
        .filter([](const int& x) { return x > 2; })
        .map([](const int& x) { return x * 10; })
        .collect();

    std::sort(result.begin(), result.end());
    EXPECT_EQ(result, (std::vector<int>{30, 40, 50, 60}));
}

TEST_F(RDDTest, Count) {
    auto rdd = ctx.parallelize(std::vector<int>{1, 2, 3, 4, 5}, 3);
    EXPECT_EQ(rdd.count(), 5u);

    auto filtered = rdd.filter([](const int& x) { return x > 3; });
    EXPECT_EQ(filtered.count(), 2u);
}

TEST_F(RDDTest, Reduce) {
    auto rdd = ctx.parallelize(std::vector<int>{1, 2, 3, 4, 5}, 2);
    int sum = rdd.reduce([](const int& a, const int& b) { return a + b; });
    EXPECT_EQ(sum, 15);
}

TEST_F(RDDTest, ReduceStrings) {
    auto rdd = ctx.parallelize(
        std::vector<std::string>{"hello", " ", "world"}, 2);
    auto result = rdd.reduce(
        [](const std::string& a, const std::string& b) { return a + b; });
    EXPECT_EQ(result, "hello world");
}

TEST_F(RDDTest, Foreach) {
    auto rdd = ctx.parallelize(std::vector<int>{1, 2, 3}, 2);
    std::mutex mu;
    int total = 0;
    rdd.foreach([&](const int& x) {
        std::lock_guard<std::mutex> lock(mu);
        total += x;
    });
    EXPECT_EQ(total, 6);
}

TEST_F(RDDTest, SinglePartition) {
    auto rdd = ctx.parallelize(std::vector<int>{10, 20, 30}, 1);
    EXPECT_EQ(rdd.num_partitions(), 1u);

    auto result = rdd.collect();
    EXPECT_EQ(result, (std::vector<int>{10, 20, 30}));
}

TEST_F(RDDTest, ManyPartitions) {
    std::vector<int> data(100);
    std::iota(data.begin(), data.end(), 0);

    auto rdd = ctx.parallelize(data, 10);
    EXPECT_EQ(rdd.num_partitions(), 10u);

    auto result = rdd.collect();
    std::sort(result.begin(), result.end());
    EXPECT_EQ(result, data);
}

TEST_F(RDDTest, MapChangesType) {
    auto rdd = ctx.parallelize(std::vector<int>{1, 2, 3}, 2);
    auto strs = rdd.map([](const int& x) { return std::to_string(x); });

    auto result = strs.collect();
    std::sort(result.begin(), result.end());
    EXPECT_EQ(result, (std::vector<std::string>{"1", "2", "3"}));
}
