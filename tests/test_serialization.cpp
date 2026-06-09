#include <gtest/gtest.h>
#include <vega/serialization/serializer.hpp>

using namespace vega;

TEST(SerializerTest, Int32RoundTrip) {
    int32_t val = -42;
    auto buf = serialize(val);
    EXPECT_EQ(deserialize<int32_t>(buf), val);
}

TEST(SerializerTest, Int64RoundTrip) {
    int64_t val = 1234567890123LL;
    auto buf = serialize(val);
    EXPECT_EQ(deserialize<int64_t>(buf), val);
}

TEST(SerializerTest, FloatRoundTrip) {
    float val = 3.14f;
    auto buf = serialize(val);
    EXPECT_FLOAT_EQ(deserialize<float>(buf), val);
}

TEST(SerializerTest, DoubleRoundTrip) {
    double val = 2.718281828;
    auto buf = serialize(val);
    EXPECT_DOUBLE_EQ(deserialize<double>(buf), val);
}

TEST(SerializerTest, BoolRoundTrip) {
    EXPECT_EQ(deserialize<bool>(serialize(true)), true);
    EXPECT_EQ(deserialize<bool>(serialize(false)), false);
}

TEST(SerializerTest, StringRoundTrip) {
    std::string val = "hello distributed world";
    auto buf = serialize(val);
    EXPECT_EQ(deserialize<std::string>(buf), val);
}

TEST(SerializerTest, EmptyString) {
    std::string val;
    auto buf = serialize(val);
    EXPECT_EQ(deserialize<std::string>(buf), val);
}

TEST(SerializerTest, VectorInt) {
    std::vector<int32_t> val = {1, 2, 3, 4, 5};
    auto buf = serialize(val);
    EXPECT_EQ(deserialize<std::vector<int32_t>>(buf), val);
}

TEST(SerializerTest, VectorString) {
    std::vector<std::string> val = {"hello", "world", ""};
    auto buf = serialize(val);
    EXPECT_EQ(deserialize<std::vector<std::string>>(buf), val);
}

TEST(SerializerTest, EmptyVector) {
    std::vector<int32_t> val;
    auto buf = serialize(val);
    EXPECT_EQ(deserialize<std::vector<int32_t>>(buf), val);
}

TEST(SerializerTest, PairStringInt) {
    std::pair<std::string, int32_t> val = {"count", 42};
    auto buf = serialize(val);
    auto result = deserialize<std::pair<std::string, int32_t>>(buf);
    EXPECT_EQ(result.first, "count");
    EXPECT_EQ(result.second, 42);
}

TEST(SerializerTest, VectorOfPairs) {
    using P = std::pair<std::string, int32_t>;
    std::vector<P> val = {{"hello", 3}, {"world", 2}};
    auto buf = serialize(val);
    auto result = deserialize<std::vector<P>>(buf);
    ASSERT_EQ(result.size(), 2u);
    EXPECT_EQ(result[0].first, "hello");
    EXPECT_EQ(result[0].second, 3);
    EXPECT_EQ(result[1].first, "world");
    EXPECT_EQ(result[1].second, 2);
}

TEST(SerializerTest, NestedVectors) {
    std::vector<std::vector<int32_t>> val = {{1, 2}, {3}, {}};
    auto buf = serialize(val);
    EXPECT_EQ(deserialize<std::vector<std::vector<int32_t>>>(buf), val);
}

TEST(SerializerTest, ByteBufferRoundTrip) {
    ByteBuffer val = {0x01, 0xFF, 0x00, 0xAB};
    auto buf = serialize(val);
    EXPECT_EQ(deserialize<ByteBuffer>(buf), val);
}
