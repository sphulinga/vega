#include <gtest/gtest.h>
#include <vega/vega.hpp>

TEST(DAGBuilderTest, SingleStageNarrowPipeline) {
    vega::SparkContext ctx(2);

    auto rdd = ctx.parallelize(std::vector<int>{1, 2, 3, 4}, 2)
        .map([](const int& x) { return x + 1; })
        .filter([](const int& x) { return x > 2; });

    auto plan = vega::DAGBuilder::build(/*job_id=*/1, rdd.impl());

    EXPECT_EQ(plan.stages.size(), 1u);
    EXPECT_EQ(plan.stages[0].tasks.size(), 2u);  // 2 partitions
    EXPECT_EQ(plan.stages[0].id, 0u);
    EXPECT_FALSE(plan.stages[0].is_shuffle_boundary);

    for (auto& task : plan.stages[0].tasks) {
        EXPECT_EQ(task.job_id, 1u);
        EXPECT_EQ(task.stage_id, 0u);
    }
}

TEST(DAGBuilderTest, TaskPartitionIds) {
    vega::SparkContext ctx(2);
    auto rdd = ctx.parallelize(std::vector<int>{1, 2, 3, 4, 5, 6}, 3);

    auto plan = vega::DAGBuilder::build(42, rdd.impl());
    ASSERT_EQ(plan.stages[0].tasks.size(), 3u);

    for (size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(plan.stages[0].tasks[i].partition_id, i);
        EXPECT_EQ(plan.stages[0].tasks[i].job_id, 42u);
    }
}

TEST(DAGBuilderTest, AllTransformsAreNarrow) {
    vega::SparkContext ctx(2);
    auto rdd = ctx.parallelize(std::vector<int>{1}, 1);

    EXPECT_TRUE(rdd.impl()->is_narrow());

    auto mapped = rdd.map([](const int& x) { return x; });
    EXPECT_TRUE(mapped.impl()->is_narrow());

    auto filtered = mapped.filter([](const int&) { return true; });
    EXPECT_TRUE(filtered.impl()->is_narrow());
}
