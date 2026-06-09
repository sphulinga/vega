// Ported from Apache Spark RDDSuite.scala and PairRDDFunctionsSuite.scala.
// Test logic mirrors upstream Scala assertions for operations Vega implements.

#include <gtest/gtest.h>
#include <vega/vega.hpp>

#include <algorithm>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {

using vega::SparkContext;
using vega::RDD;

class SparkRDDSuite : public ::testing::Test {
protected:
    SparkContext sc{2};
};

std::vector<int> sorted_vec(std::vector<int> v) {
    std::sort(v.begin(), v.end());
    return v;
}

}  // namespace

// RDDSuite: "basic operations"
TEST_F(SparkRDDSuite, BasicOperations) {
    auto nums = sc.parallelize(std::vector<int>{1, 2, 3, 4}, 2);
    ASSERT_EQ(nums.num_partitions(), 2u);

    auto collected = nums.collect();
    std::sort(collected.begin(), collected.end());
    ASSERT_EQ(collected, (std::vector<int>{1, 2, 3, 4}));

    auto mapped = nums.map([](const int& x) { return x * 2; }).collect();
    std::sort(mapped.begin(), mapped.end());
    ASSERT_EQ(mapped, (std::vector<int>{2, 4, 6, 8}));

    auto filtered = nums.filter([](const int& x) { return x > 2; }).collect();
    std::sort(filtered.begin(), filtered.end());
    ASSERT_EQ(filtered, (std::vector<int>{3, 4}));

    auto flat = nums.flatMap([](const int& x) {
        std::vector<int> out;
        for (int i = 1; i <= x; ++i) out.push_back(i);
        return out;
    }).collect();
    ASSERT_EQ(flat, (std::vector<int>{1, 1, 2, 1, 2, 3, 1, 2, 3, 4}));

    ASSERT_EQ(nums.reduce([](const int& a, const int& b) { return a + b; }), 10);

    auto as_str = nums.map([](const int& x) { return std::to_string(x); }).collect();
    std::sort(as_str.begin(), as_str.end());
    ASSERT_EQ(as_str, (std::vector<std::string>{"1", "2", "3", "4"}));
}

// RDDSuite: count/filter paths
TEST_F(SparkRDDSuite, CountAndReduce) {
    auto rdd = sc.parallelize(std::vector<int>{1, 2, 3, 4, 5}, 3);
    ASSERT_EQ(rdd.count(), 5u);

    auto filtered = rdd.filter([](const int& x) { return x > 3; });
    ASSERT_EQ(filtered.count(), 2u);
    ASSERT_EQ(filtered.reduce([](const int& a, const int& b) { return a + b; }), 9);
}

// PairRDDFunctionsSuite: "reduceByKey"
TEST_F(SparkRDDSuite, ReduceByKey) {
    using Pair = std::pair<int, int>;
    auto pairs = sc.parallelize(
        std::vector<Pair>{{1, 1}, {1, 2}, {1, 3}, {1, 1}, {2, 1}}, 2);

    auto sums = pairs.reduceByKey([](const int& a, const int& b) { return a + b; }).collect();
    std::set<Pair> got(sums.begin(), sums.end());
    ASSERT_EQ(got, (std::set<Pair>{{1, 7}, {2, 1}}));
}

// PairRDDFunctionsSuite: "reduceByKey with many output partitions"
TEST_F(SparkRDDSuite, ReduceByKeyManyPartitions) {
    using Pair = std::pair<int, int>;
    auto pairs = sc.parallelize(
        std::vector<Pair>{{1, 1}, {1, 2}, {1, 3}, {1, 1}, {2, 1}}, 2);

    auto sums = pairs.reduceByKey([](const int& a, const int& b) { return a + b; }, 10)
                    .collect();
    std::set<Pair> got(sums.begin(), sums.end());
    ASSERT_EQ(got, (std::set<Pair>{{1, 7}, {2, 1}}));
}

// PairRDDFunctionsSuite: "groupByKey"
TEST_F(SparkRDDSuite, GroupByKey) {
    using Pair = std::pair<int, int>;
    using Group = std::pair<int, std::vector<int>>;
    auto pairs = sc.parallelize(
        std::vector<Pair>{{1, 1}, {1, 2}, {1, 3}, {2, 1}}, 2);

    auto groups = pairs.groupByKey().collect();
    ASSERT_EQ(groups.size(), 2u);

    std::sort(groups.begin(), groups.end(),
              [](const Group& a, const Group& b) { return a.first < b.first; });

    auto values_for_1 = groups[0].first == 1 ? groups[0].second : groups[1].second;
    auto values_for_2 = groups[0].first == 2 ? groups[0].second : groups[1].second;
    std::sort(values_for_1.begin(), values_for_1.end());
    std::sort(values_for_2.begin(), values_for_2.end());
    ASSERT_EQ(values_for_1, (std::vector<int>{1, 2, 3}));
    ASSERT_EQ(values_for_2, (std::vector<int>{1}));
}

// RDDSuite: chained narrow transforms (subset)
TEST_F(SparkRDDSuite, ChainedNarrowTransforms) {
    auto result = sc.parallelize(std::vector<int>{1, 2, 3, 4, 5, 6}, 2)
                      .filter([](const int& x) { return x > 2; })
                      .map([](const int& x) { return x * 10; })
                      .collect();
    std::sort(result.begin(), result.end());
    ASSERT_EQ(result, (std::vector<int>{30, 40, 50, 60}));
}
