#include <gtest/gtest.h>
#include <vega/vega.hpp>
#include <vega/dataframe.hpp>

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <vector>

using namespace vega::df;

namespace {

// A small sales table with deterministic contents.
struct Sales {
    std::vector<int64_t> id, store, qty;
    std::vector<double> price;
    std::vector<std::string> category;
};

Sales make_sales(size_t n) {
    Sales s;
    const std::vector<std::string> cats = {"books", "games", "music", "tools"};
    for (size_t i = 0; i < n; ++i) {
        s.id.push_back(static_cast<int64_t>(i));
        s.store.push_back(static_cast<int64_t>((i * 7) % 13));
        s.qty.push_back(static_cast<int64_t>(i % 5 + 1));
        s.price.push_back(static_cast<double>(i % 17) + 0.25);
        s.category.push_back(cats[(i * 3) % cats.size()]);
    }
    return s;
}

DataFrame sales_df(vega::SparkContext& ctx, const Sales& s, size_t parts) {
    return DataFrame::from_columns(ctx,
                                   {{"id", Column(s.id)},
                                    {"store", Column(s.store)},
                                    {"qty", Column(s.qty)},
                                    {"price", Column(s.price)},
                                    {"category", Column(s.category)}},
                                   parts);
}

template<typename T>
const std::vector<T>& vals(const Batch& b, const std::string& name) {
    return b.column(name).values<T>();
}

}  // namespace

class DataFrameTest : public ::testing::Test {
protected:
    vega::SparkContext ctx{4};
    Sales s = make_sales(1000);
};

TEST_F(DataFrameTest, CollectRoundTrip) {
    auto df = sales_df(ctx, s, 4);
    EXPECT_EQ(df.count(), 1000u);
    Batch b = df.collect();
    EXPECT_EQ(b.num_rows(), 1000u);
    EXPECT_EQ(vals<int64_t>(b, "id"), s.id);
    EXPECT_EQ(vals<std::string>(b, "category"), s.category);
    EXPECT_EQ(schema_string(df.schema()),
              "[id: bigint, store: bigint, qty: bigint, price: double, category: string]");
}

TEST_F(DataFrameTest, SelectArithmeticAndNames) {
    auto df = sales_df(ctx, s, 3).select({col("id"), col("qty") * col("price"), (col("qty") + 1).alias("q1"),
                                          col("qty") / 2});
    EXPECT_EQ(df.schema()[1].name, "(qty * price)");
    EXPECT_EQ(df.schema()[1].type, DType::Float64);
    EXPECT_EQ(df.schema()[2].type, DType::Int64);
    EXPECT_EQ(df.schema()[3].name, "(qty / 2)");
    EXPECT_EQ(df.schema()[3].type, DType::Float64);  // Spark: / is double division
    Batch b = df.collect();
    for (size_t i = 0; i < s.id.size(); ++i) {
        EXPECT_DOUBLE_EQ(b.column(1).values<double>()[i], s.qty[i] * s.price[i]);
        EXPECT_EQ(vals<int64_t>(b, "q1")[i], s.qty[i] + 1);
        EXPECT_DOUBLE_EQ(b.column(3).values<double>()[i], s.qty[i] / 2.0);
    }
}

TEST_F(DataFrameTest, DivisionByZeroErrors) {
    auto df = sales_df(ctx, s, 2).select({col("qty") % 0});
    EXPECT_ANY_THROW(df.collect());
}

TEST_F(DataFrameTest, FilterAndWithColumn) {
    auto df = sales_df(ctx, s, 4)
                  .filter(col("qty") >= 3 && col("category") == "games")
                  .with_column("revenue", col("qty") * col("price"))
                  .with_column("qty", col("qty") * 10);
    Batch b = df.collect();
    std::vector<int64_t> want_ids;
    for (size_t i = 0; i < s.id.size(); ++i)
        if (s.qty[i] >= 3 && s.category[i] == "games") want_ids.push_back(s.id[i]);
    EXPECT_EQ(vals<int64_t>(b, "id"), want_ids);
    EXPECT_EQ(df.count(), want_ids.size());
    EXPECT_EQ(b.schema().back().name, "revenue");
    for (size_t r = 0; r < b.num_rows(); ++r) {
        const size_t i = static_cast<size_t>(vals<int64_t>(b, "id")[r]);
        EXPECT_EQ(vals<int64_t>(b, "qty")[r], s.qty[i] * 10);
        EXPECT_DOUBLE_EQ(vals<double>(b, "revenue")[r], s.qty[i] * s.price[i]);
    }
    EXPECT_ANY_THROW(sales_df(ctx, s, 2).filter(col("qty")));          // not boolean
    EXPECT_ANY_THROW(sales_df(ctx, s, 2).filter(col("nope") > 1));     // unknown column
    EXPECT_ANY_THROW(sales_df(ctx, s, 2).filter(col("category") > 1)); // type mismatch
}

TEST_F(DataFrameTest, GroupByIntKey) {
    for (size_t parts : {1u, 3u, 8u}) {
        auto df = sales_df(ctx, s, parts)
                      .group_by({col("store")})
                      .agg({sum(col("qty")), count_all(), avg(col("price")), min(col("id")),
                            max(col("category"))});
        EXPECT_EQ(df.schema()[1].name, "sum(qty)");
        EXPECT_EQ(df.schema()[2].name, "count(1)");
        Batch b = df.collect();
        std::map<int64_t, std::tuple<int64_t, int64_t, double, int64_t, std::string>> want;
        for (size_t i = 0; i < s.id.size(); ++i) {
            auto [it, fresh] = want.try_emplace(s.store[i], 0, 0, 0.0, s.id[i], s.category[i]);
            auto& [sq, c, sp, mn, mx] = it->second;
            sq += s.qty[i];
            ++c;
            sp += s.price[i];
            mn = std::min(mn, s.id[i]);
            mx = std::max(mx, s.category[i]);
        }
        ASSERT_EQ(b.num_rows(), want.size()) << parts;
        for (size_t r = 0; r < b.num_rows(); ++r) {
            const auto& [sq, c, sp, mn, mx] = want.at(b.column(0).values<int64_t>()[r]);
            EXPECT_EQ(b.column(1).values<int64_t>()[r], sq);
            EXPECT_EQ(b.column(2).values<int64_t>()[r], c);
            EXPECT_NEAR(b.column(3).values<double>()[r], sp / c, 1e-9);
            EXPECT_EQ(b.column(4).values<int64_t>()[r], mn);
            EXPECT_EQ(b.column(5).values<std::string>()[r], mx);
        }
    }
}

TEST_F(DataFrameTest, GroupByMultiKeyAndExpression) {
    auto df = sales_df(ctx, s, 4)
                  .group_by({col("category"), (col("store") % 3).alias("bucket")})
                  .agg({sum(col("qty") * col("price")).alias("revenue")});
    Batch b = df.collect();
    std::map<std::pair<std::string, int64_t>, double> want;
    for (size_t i = 0; i < s.id.size(); ++i)
        want[{s.category[i], s.store[i] % 3}] += s.qty[i] * s.price[i];
    ASSERT_EQ(b.num_rows(), want.size());
    for (size_t r = 0; r < b.num_rows(); ++r) {
        const auto key = std::make_pair(vals<std::string>(b, "category")[r], vals<int64_t>(b, "bucket")[r]);
        EXPECT_NEAR(vals<double>(b, "revenue")[r], want.at(key), 1e-6);
    }
}

TEST_F(DataFrameTest, GlobalAggregate) {
    Batch b = sales_df(ctx, s, 4).agg({sum(col("qty")), count_all(), max(col("price"))}).collect();
    ASSERT_EQ(b.num_rows(), 1u);
    int64_t total = 0;
    for (auto q : s.qty) total += q;
    EXPECT_EQ(b.column(0).values<int64_t>()[0], total);
    EXPECT_EQ(b.column(1).values<int64_t>()[0], 1000);
    EXPECT_DOUBLE_EQ(b.column(2).values<double>()[0], 16.25);

    // Over no rows: one row with zero sum/count (Spark returns null for sum).
    Batch e = sales_df(ctx, s, 4).filter(col("qty") > 100).agg({sum(col("qty")), count_all()}).collect();
    ASSERT_EQ(e.num_rows(), 1u);
    EXPECT_EQ(e.column(0).values<int64_t>()[0], 0);
    EXPECT_EQ(e.column(1).values<int64_t>()[0], 0);
}

TEST_F(DataFrameTest, EmptyGroupBy) {
    auto df = sales_df(ctx, s, 4).filter(col("qty") > 100).group_by({col("store")}).count();
    EXPECT_EQ(df.count(), 0u);
    EXPECT_EQ(df.collect().num_rows(), 0u);
}

TEST_F(DataFrameTest, Distinct) {
    Batch b = sales_df(ctx, s, 4).select({col("category"), col("qty")}).distinct().collect();
    std::set<std::pair<std::string, int64_t>> want;
    for (size_t i = 0; i < s.id.size(); ++i) want.insert({s.category[i], s.qty[i]});
    std::set<std::pair<std::string, int64_t>> got;
    for (size_t r = 0; r < b.num_rows(); ++r)
        got.insert({vals<std::string>(b, "category")[r], vals<int64_t>(b, "qty")[r]});
    EXPECT_EQ(b.num_rows(), want.size());
    EXPECT_EQ(got, want);
}

TEST_F(DataFrameTest, Join) {
    std::vector<int64_t> sid;
    std::vector<std::string> region;
    for (int64_t k = 0; k < 13; k += 2) {  // only even stores have a region
        sid.push_back(k);
        region.push_back(k < 6 ? "east" : "west");
    }
    sid.push_back(4);  // a duplicate key produces two matches
    region.push_back("north");
    auto stores = DataFrame::from_columns(ctx, {{"store", Column(sid)}, {"region", Column(region)}}, 2);
    auto joined = sales_df(ctx, s, 4).join(stores, {"store"});
    EXPECT_EQ(schema_string(joined.schema()),
              "[store: bigint, id: bigint, qty: bigint, price: double, category: string, region: string]");
    Batch b = joined.collect();
    std::multiset<std::pair<int64_t, std::string>> want, got;
    for (size_t i = 0; i < s.id.size(); ++i)
        for (size_t k = 0; k < sid.size(); ++k)
            if (sid[k] == s.store[i]) want.insert({s.id[i], region[k]});
    for (size_t r = 0; r < b.num_rows(); ++r)
        got.insert({vals<int64_t>(b, "id")[r], vals<std::string>(b, "region")[r]});
    EXPECT_EQ(got, want);

    // join then aggregate
    Batch g = joined.group_by({col("region")}).agg({sum(col("qty"))}).collect();
    std::map<std::string, int64_t> wsum;
    for (size_t i = 0; i < s.id.size(); ++i)
        for (size_t k = 0; k < sid.size(); ++k)
            if (sid[k] == s.store[i]) wsum[region[k]] += s.qty[i];
    ASSERT_EQ(g.num_rows(), wsum.size());
    for (size_t r = 0; r < g.num_rows(); ++r)
        EXPECT_EQ(g.column(1).values<int64_t>()[r], wsum.at(g.column(0).values<std::string>()[r]));
}

TEST_F(DataFrameTest, OrderByAndLimit) {
    auto df = sales_df(ctx, s, 4);
    // Full sort: price desc, then id asc.
    Batch b = df.order_by({desc(col("price")), asc(col("id"))}).collect();
    std::vector<size_t> idx(s.id.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](size_t x, size_t y) {
        if (s.price[x] != s.price[y]) return s.price[x] > s.price[y];
        return s.id[x] < s.id[y];
    });
    ASSERT_EQ(b.num_rows(), idx.size());
    for (size_t r = 0; r < idx.size(); ++r) EXPECT_EQ(vals<int64_t>(b, "id")[r], s.id[idx[r]]);

    // Top-N.
    Batch t = df.order_by({desc(col("price")), asc(col("id"))}).limit(7).collect();
    ASSERT_EQ(t.num_rows(), 7u);
    for (size_t r = 0; r < 7; ++r) EXPECT_EQ(vals<int64_t>(t, "id")[r], s.id[idx[r]]);

    // Operations after a sort see sorted data.
    Batch f = df.order_by({desc(col("id"))}).filter(col("id") < 5).collect();
    EXPECT_EQ(vals<int64_t>(f, "id"), (std::vector<int64_t>{4, 3, 2, 1, 0}));

    // String sort and plain limit.
    Batch c = df.order_by({asc(col("category")), desc(col("id"))}).limit(1).collect();
    EXPECT_EQ(vals<std::string>(c, "category")[0], "books");
    EXPECT_EQ(df.limit(5).count(), 5u);
    EXPECT_EQ(df.limit(5000).count(), 1000u);
}

TEST_F(DataFrameTest, GroupByThenOrderBy) {
    Batch b = sales_df(ctx, s, 4)
                  .group_by({col("store")})
                  .agg({sum(col("qty")).alias("units")})
                  .order_by({desc(col("units")), asc(col("store"))})
                  .limit(3)
                  .collect();
    std::map<int64_t, int64_t> units;
    for (size_t i = 0; i < s.id.size(); ++i) units[s.store[i]] += s.qty[i];
    std::vector<std::pair<int64_t, int64_t>> v;
    for (auto [k, u] : units) v.push_back({-u, k});
    std::sort(v.begin(), v.end());
    ASSERT_EQ(b.num_rows(), 3u);
    for (size_t r = 0; r < 3; ++r) {
        EXPECT_EQ(vals<int64_t>(b, "store")[r], v[r].second);
        EXPECT_EQ(vals<int64_t>(b, "units")[r], -v[r].first);
    }
}

TEST_F(DataFrameTest, Show) {
    std::string out = sales_df(ctx, s, 2).select({col("id"), col("category")}).show(2);
    EXPECT_NE(out.find("| id | category |"), std::string::npos) << out;
    EXPECT_NE(out.find("|  1 |    tools |"), std::string::npos) << out;
}

TEST_F(DataFrameTest, DuplicateNamesRejected) {
    EXPECT_ANY_THROW(sales_df(ctx, s, 2).select({col("id"), col("id")}));
}

TEST(DataFrameSerialization, BatchRoundTrip) {
    Batch b({{"a", DType::Int64}, {"b", DType::Float64}, {"c", DType::String}, {"d", DType::Bool}},
            {Column(std::vector<int64_t>{1, -2}), Column(std::vector<double>{0.5, -1.5}),
             Column(std::vector<std::string>{"x", ""}), Column(BoolVec{1, 0})});
    vega::ByteBuffer buf;
    vega::Serializer<Batch>::serialize(b, buf);
    size_t off = 0;
    Batch r = vega::Serializer<Batch>::deserialize(buf, off);
    EXPECT_EQ(off, buf.size());
    EXPECT_EQ(r.schema(), b.schema());
    EXPECT_EQ(vals<int64_t>(r, "a"), vals<int64_t>(b, "a"));
    EXPECT_EQ(vals<double>(r, "b"), vals<double>(b, "b"));
    EXPECT_EQ(vals<std::string>(r, "c"), vals<std::string>(b, "c"));
    EXPECT_EQ(vals<uint8_t>(r, "d"), vals<uint8_t>(b, "d"));
}

TEST_F(DataFrameTest, FusedPipelineWithPruning) {
    // Project -> filter -> join -> filter -> project, consumed by count,
    // an aggregate that reads one column, and a full collect.
    std::vector<int64_t> sid;
    std::vector<std::string> region;
    for (int64_t k = 0; k < 13; ++k) {
        sid.push_back(k);
        region.push_back(k % 2 ? "odd" : "even");
    }
    auto stores = DataFrame::from_columns(ctx, {{"store", Column(sid)}, {"region", Column(region)}}, 1);
    auto df = sales_df(ctx, s, 4)
                  .with_column("rev", col("qty") * col("price"))
                  .filter(col("qty") > 1)
                  .join(stores, {"store"})
                  .filter(col("region") == "odd")
                  .select({col("id"), col("rev"), lit(7).alias("seven")});
    size_t want = 0;
    double want_rev = 0;
    for (size_t i = 0; i < s.id.size(); ++i)
        if (s.qty[i] > 1 && s.store[i] % 2 == 1) {
            ++want;
            want_rev += s.qty[i] * s.price[i];
        }
    EXPECT_EQ(df.count(), want);
    Batch a = df.agg({sum(col("rev")), sum(col("seven"))}).collect();
    EXPECT_NEAR(a.column(0).values<double>()[0], want_rev, 1e-6);
    EXPECT_EQ(a.column(1).values<int64_t>()[0], static_cast<int64_t>(7 * want));
    Batch b = df.collect();
    EXPECT_EQ(b.num_rows(), want);
    EXPECT_EQ(schema_string(b.schema()), "[id: bigint, rev: double, seven: bigint]");
    // count of a frame whose columns are all pruned away
    EXPECT_EQ(sales_df(ctx, s, 3).select({lit(1).alias("one")}).count(), 1000u);
    EXPECT_EQ(sales_df(ctx, s, 3).select({col("id")}).distinct().count(), 1000u);
}
