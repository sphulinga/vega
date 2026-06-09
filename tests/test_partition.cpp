#include <gtest/gtest.h>
#include <vega/core/partition.hpp>

using vega::Partition;

TEST(PartitionTest, ConstructAndAccess) {
    Partition<int> p(0, {1, 2, 3});
    EXPECT_EQ(p.id(), 0u);
    EXPECT_EQ(p.size(), 3u);
    EXPECT_FALSE(p.empty());
    EXPECT_EQ(p.data()[0], 1);
    EXPECT_EQ(p.data()[2], 3);
}

TEST(PartitionTest, EmptyPartition) {
    Partition<int> p(5, {});
    EXPECT_EQ(p.id(), 5u);
    EXPECT_EQ(p.size(), 0u);
    EXPECT_TRUE(p.empty());
}

TEST(PartitionTest, Iteration) {
    Partition<std::string> p(0, {"a", "b", "c"});
    std::vector<std::string> collected(p.begin(), p.end());
    EXPECT_EQ(collected, (std::vector<std::string>{"a", "b", "c"}));
}

TEST(PartitionTest, MutableAccess) {
    Partition<int> p(0, {10, 20});
    p.data()[0] = 99;
    EXPECT_EQ(p.data()[0], 99);
}
