// Spark-parity features of Vega DataFrames: nulls, outer/semi/anti/cross
// joins (broadcast and shuffle), unions, window functions, expression
// functions, extra aggregates, CSV, and the query optimizer.

#include <gtest/gtest.h>
#include <vega/vega.hpp>
#include <vega/dataframe.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <tuple>
#include <vector>

using namespace vega::df;

namespace {

using OptI = std::optional<int64_t>;
using OptS = std::optional<std::string>;
using OptD = std::optional<double>;

template<typename T>
std::vector<std::optional<T>> opt_values(const Batch& b, const std::string& name) {
    const Column& c = b.column(name);
    const auto& v = c.values<T>();
    std::vector<std::optional<T>> out;
    for (size_t i = 0; i < v.size(); ++i) out.push_back(c.is_valid(i) ? std::optional<T>(v[i]) : std::nullopt);
    return out;
}

std::vector<OptI> ints(const Batch& b, const std::string& n) { return opt_values<int64_t>(b, n); }
std::vector<OptS> strs(const Batch& b, const std::string& n) { return opt_values<std::string>(b, n); }
std::vector<OptD> dbls(const Batch& b, const std::string& n) { return opt_values<double>(b, n); }
std::vector<std::optional<uint8_t>> bools(const Batch& b, const std::string& n) { return opt_values<uint8_t>(b, n); }

struct ThresholdGuard {
    double saved = broadcast_threshold().load();
    explicit ThresholdGuard(double v) { broadcast_threshold() = v; }
    ~ThresholdGuard() { broadcast_threshold() = saved; }
};

}  // namespace

class DataFrameSqlTest : public ::testing::Test {
protected:
    vega::SparkContext ctx{4};

    // people(id, name, dept, salary) with nulls
    DataFrame people() {
        return DataFrame::from_columns(
            ctx,
            {{"id", Column(std::vector<int64_t>{1, 2, 3, 4, 5, 6})},
             {"name", Column(std::vector<OptS>{"ann", "bob", std::nullopt, "dan", "eve", "fay"})},
             {"dept", Column(std::vector<OptI>{10, 20, 10, std::nullopt, 30, 20})},
             {"salary", Column(std::vector<OptD>{100.0, std::nullopt, 50.0, 70.0, 90.0, 60.0})}},
            3);
    }
    DataFrame depts() {
        return DataFrame::from_columns(ctx,
                                       {{"dept", Column(std::vector<OptI>{10, 20, 40, std::nullopt})},
                                        {"dname", Column(std::vector<std::string>{"eng", "ops", "hr", "nil"})}},
                                       2);
    }
};

// ----- nulls in expressions ----------------------------------------------------

TEST_F(DataFrameSqlTest, NullArithmeticAndComparison) {
    Batch b = people()
                  .select({col("id"), (col("salary") * 2).alias("s2"), (col("dept") + 1).alias("d1"),
                           (col("salary") > 60.0).alias("rich"), col("salary").is_null().alias("nosal"),
                           (col("dept") == lit_null(DType::Int64)).alias("eqnull"),
                           col("dept").eq_null_safe(lit_null(DType::Int64)).alias("nseq")})
                  .collect();
    EXPECT_EQ(dbls(b, "s2"), (std::vector<OptD>{200.0, std::nullopt, 100.0, 140.0, 180.0, 120.0}));
    EXPECT_EQ(ints(b, "d1"), (std::vector<OptI>{11, 21, 11, std::nullopt, 31, 21}));
    EXPECT_EQ(bools(b, "rich"), (std::vector<std::optional<uint8_t>>{1, std::nullopt, 0, 1, 1, 0}));
    EXPECT_EQ(bools(b, "nosal"), (std::vector<std::optional<uint8_t>>{0, 1, 0, 0, 0, 0}));
    for (auto v : bools(b, "eqnull")) EXPECT_FALSE(v.has_value());  // x = NULL is NULL
    EXPECT_EQ(bools(b, "nseq"), (std::vector<std::optional<uint8_t>>{0, 0, 0, 1, 0, 0}));
}

TEST_F(DataFrameSqlTest, ThreeValuedLogicAndFilter) {
    const Expr t = lit(true), f = lit(false), n = lit_null(DType::Bool);
    Batch b = people()
                  .limit(1)
                  .select({(f && n).alias("fn"), (t && n).alias("tn"), (t || n).alias("to"), (f || n).alias("fo"),
                           (!n).alias("nn")})
                  .collect();
    EXPECT_EQ(bools(b, "fn")[0], std::optional<uint8_t>(0));
    EXPECT_FALSE(bools(b, "tn")[0].has_value());
    EXPECT_EQ(bools(b, "to")[0], std::optional<uint8_t>(1));
    EXPECT_FALSE(bools(b, "fo")[0].has_value());
    EXPECT_FALSE(bools(b, "nn")[0].has_value());
    // filter keeps only rows where the predicate is true (not null)
    EXPECT_EQ(people().filter(col("salary") > 60.0).count(), 3u);
    EXPECT_EQ(people().filter(!(col("salary") > 60.0)).count(), 2u);
    EXPECT_EQ(people().filter(col("salary") > 60.0 || col("dept") == 20).count(), 5u);
}

TEST_F(DataFrameSqlTest, Functions) {
    Batch b = people()
                  .select({col("id"),
                           coalesce({col("salary"), lit(0.0)}).alias("sal"),
                           when(col("salary") >= 90.0, "high").when(col("salary") >= 60.0, "mid").otherwise("low").alias("band"),
                           when(col("dept") == 10, 1).alias("is10"),
                           upper(col("name")).alias("up"),
                           length(col("name")).alias("len"),
                           substring(col("name"), 2, 2).alias("sub"),
                           concat({col("name"), lit("!")}).alias("cc"),
                           col("name").like("%a%").alias("has_a"),
                           col("name").starts_with("d").alias("sw"),
                           col("dept").isin({10, 30}).alias("in"),
                           cast(col("dept"), DType::String).alias("ds"),
                           cast(col("salary"), DType::Int64).alias("si"),
                           round(col("salary") / 7.0, 2).alias("r"),
                           abs(-col("id")).alias("a"),
                           floor(col("salary") / 3.0).alias("fl"),
                           col("id").between(2, 4).alias("btw")})
                  .collect();
    EXPECT_EQ(dbls(b, "sal"), (std::vector<OptD>{100.0, 0.0, 50.0, 70.0, 90.0, 60.0}));
    EXPECT_EQ(strs(b, "band"), (std::vector<OptS>{"high", "low", "low", "mid", "high", "mid"}));
    EXPECT_EQ(ints(b, "is10"), (std::vector<OptI>{1, std::nullopt, 1, std::nullopt, std::nullopt, std::nullopt}));
    EXPECT_EQ(strs(b, "up"), (std::vector<OptS>{"ANN", "BOB", std::nullopt, "DAN", "EVE", "FAY"}));
    EXPECT_EQ(ints(b, "len"), (std::vector<OptI>{3, 3, std::nullopt, 3, 3, 3}));
    EXPECT_EQ(strs(b, "sub"), (std::vector<OptS>{"nn", "ob", std::nullopt, "an", "ve", "ay"}));
    EXPECT_EQ(strs(b, "cc")[0], OptS("ann!"));
    EXPECT_FALSE(strs(b, "cc")[2].has_value());
    EXPECT_EQ(bools(b, "has_a"), (std::vector<std::optional<uint8_t>>{1, 0, std::nullopt, 1, 0, 1}));
    EXPECT_EQ(bools(b, "sw"), (std::vector<std::optional<uint8_t>>{0, 0, std::nullopt, 1, 0, 0}));
    EXPECT_EQ(bools(b, "in"), (std::vector<std::optional<uint8_t>>{1, 0, 1, std::nullopt, 1, 0}));
    EXPECT_EQ(strs(b, "ds"), (std::vector<OptS>{"10", "20", "10", std::nullopt, "30", "20"}));
    EXPECT_EQ(ints(b, "si"), (std::vector<OptI>{100, std::nullopt, 50, 70, 90, 60}));
    EXPECT_NEAR(*dbls(b, "r")[0], 14.29, 1e-12);
    EXPECT_EQ(ints(b, "a"), (std::vector<OptI>{1, 2, 3, 4, 5, 6}));
    EXPECT_EQ(ints(b, "fl")[0], OptI(33));
    EXPECT_EQ(bools(b, "btw"), (std::vector<std::optional<uint8_t>>{0, 1, 1, 1, 0, 0}));
    // ANSI errors
    EXPECT_ANY_THROW(people().select({cast(col("name"), DType::Int64)}).collect());
    EXPECT_ANY_THROW(people().select({col("id") / 0}).collect());
    // nulls never trigger divide-by-zero
    EXPECT_NO_THROW(people().select({col("id") / (col("dept") - col("dept"))}).filter(lit(false)).collect());
}

TEST_F(DataFrameSqlTest, NullAwareAggregates) {
    Batch b = people()
                  .group_by({col("dept")})
                  .agg({sum(col("salary")).alias("s"), count(col("salary")).alias("c"), count_all().alias("n"),
                        avg(col("salary")).alias("a"), min(col("name")).alias("mn"), max(col("salary")).alias("mx")})
                  .order_by({asc(col("dept"))})
                  .collect();
    // nulls sort first ascending: groups NULL, 10, 20, 30
    EXPECT_EQ(ints(b, "dept"), (std::vector<OptI>{std::nullopt, 10, 20, 30}));
    EXPECT_EQ(dbls(b, "s"), (std::vector<OptD>{70.0, 150.0, 60.0, 90.0}));
    EXPECT_EQ(ints(b, "c"), (std::vector<OptI>{1, 2, 1, 1}));
    EXPECT_EQ(ints(b, "n"), (std::vector<OptI>{1, 2, 2, 1}));
    EXPECT_EQ(dbls(b, "a"), (std::vector<OptD>{70.0, 75.0, 60.0, 90.0}));
    EXPECT_EQ(strs(b, "mn"), (std::vector<OptS>{"dan", "ann", "bob", "eve"}));
    // Spark: sum/avg/min/max of no rows (or only nulls) are null, count is 0.
    Batch e = people().filter(col("id") > 100).agg({sum(col("salary")), avg(col("salary")), min(col("id")),
                                                    count(col("id")), count_all()}).collect();
    ASSERT_EQ(e.num_rows(), 1u);
    EXPECT_FALSE(e.column(0).is_valid(0));
    EXPECT_FALSE(e.column(1).is_valid(0));
    EXPECT_FALSE(e.column(2).is_valid(0));
    EXPECT_EQ(e.column(3).values<int64_t>()[0], 0);
    EXPECT_EQ(e.column(4).values<int64_t>()[0], 0);
    Batch only_null = people().filter(col("id") == 2).agg({sum(col("salary")), max(col("salary"))}).collect();
    EXPECT_FALSE(only_null.column(0).is_valid(0));
    EXPECT_FALSE(only_null.column(1).is_valid(0));
}

TEST_F(DataFrameSqlTest, VarianceStddevCountDistinct) {
    std::vector<double> xs;
    std::vector<int64_t> g;
    for (int i = 0; i < 500; ++i) {
        xs.push_back(std::sin(i) * 10 + i % 7);
        g.push_back(i % 3);
    }
    auto df = DataFrame::from_columns(ctx, {{"g", Column(g)}, {"x", Column(xs)}}, 5);
    Batch b = df.group_by({col("g")})
                  .agg({variance(col("x")).alias("v"), stddev(col("x")).alias("sd"), var_pop(col("x")).alias("vp")})
                  .order_by({asc(col("g"))})
                  .collect();
    for (int k = 0; k < 3; ++k) {
        double m = 0, n = 0;
        for (size_t i = 0; i < xs.size(); ++i)
            if (g[i] == k) {
                m += xs[i];
                ++n;
            }
        m /= n;
        double ss = 0;
        for (size_t i = 0; i < xs.size(); ++i)
            if (g[i] == k) ss += (xs[i] - m) * (xs[i] - m);
        EXPECT_NEAR(*dbls(b, "v")[k], ss / (n - 1), 1e-9);
        EXPECT_NEAR(*dbls(b, "sd")[k], std::sqrt(ss / (n - 1)), 1e-9);
        EXPECT_NEAR(*dbls(b, "vp")[k], ss / n, 1e-9);
    }
    // stddev_samp of one value is null
    Batch one = df.filter(col("x") == xs[0]).agg({stddev(col("x"))}).collect();
    EXPECT_FALSE(one.column(0).is_valid(0));

    Batch cd = people().agg({count_distinct(col("dept"))}).collect();
    EXPECT_EQ(cd.column(0).values<int64_t>()[0], 3);  // 10, 20, 30 (nulls ignored)
    EXPECT_EQ(cd.schema()[0].name, "count(DISTINCT dept)");
    Batch cdg = people().group_by({col("dept")}).agg({count_distinct(col("name")).alias("u")})
                    .order_by({asc(col("dept"))}).collect();
    EXPECT_EQ(ints(cdg, "u"), (std::vector<OptI>{1, 1, 2, 1}));
}

// ----- joins -------------------------------------------------------------------

namespace {

// Reference join of (k, lv) with (k, rv) by nested loops, SQL semantics.
using Row = std::tuple<OptI, OptI, OptI, OptI>;  // left k, lv, right k, rv

std::multiset<Row> reference_join(const std::vector<OptI>& lk, const std::vector<OptI>& lv,
                                  const std::vector<OptI>& rk, const std::vector<OptI>& rv, JoinType t) {
    std::multiset<Row> out;
    std::vector<bool> rmatched(rk.size(), false);
    for (size_t i = 0; i < lk.size(); ++i) {
        bool m = false;
        for (size_t j = 0; j < rk.size(); ++j) {
            if (t != JoinType::Cross && !(lk[i] && rk[j] && *lk[i] == *rk[j])) continue;
            m = true;
            rmatched[j] = true;
            if (t != JoinType::LeftSemi && t != JoinType::LeftAnti) out.insert({lk[i], lv[i], rk[j], rv[j]});
        }
        if (t == JoinType::LeftSemi && m) out.insert({lk[i], lv[i], std::nullopt, std::nullopt});
        if (t == JoinType::LeftAnti && !m) out.insert({lk[i], lv[i], std::nullopt, std::nullopt});
        if (!m && (t == JoinType::Left || t == JoinType::Full)) out.insert({lk[i], lv[i], std::nullopt, std::nullopt});
    }
    if (t == JoinType::Right || t == JoinType::Full)
        for (size_t j = 0; j < rk.size(); ++j)
            if (!rmatched[j]) out.insert({std::nullopt, std::nullopt, rk[j], rv[j]});
    return out;
}

}  // namespace

TEST_F(DataFrameSqlTest, AllJoinTypesMatchReference) {
    std::mt19937 rng(7);
    auto gen = [&](size_t n, int keys, std::vector<OptI>& k, std::vector<OptI>& v, int base) {
        for (size_t i = 0; i < n; ++i) {
            const int x = static_cast<int>(rng() % (keys + 1));
            k.push_back(x == keys ? OptI() : OptI(x));  // some null keys
            v.push_back(base + static_cast<int64_t>(i));
        }
    };
    std::vector<OptI> lk, lv, rk, rv;
    gen(300, 40, lk, lv, 0);
    gen(120, 50, rk, rv, 10000);
    auto L = DataFrame::from_columns(ctx, {{"lk", Column(lk)}, {"lv", Column(lv)}}, 4);
    auto R = DataFrame::from_columns(ctx, {{"rk", Column(rk)}, {"rv", Column(rv)}}, 3);
    for (double threshold : {32.0 * 1024 * 1024, 0.0}) {  // broadcast, then shuffle
        ThresholdGuard guard(threshold);
        for (JoinType t : {JoinType::Inner, JoinType::Left, JoinType::Right, JoinType::Full, JoinType::LeftSemi,
                           JoinType::LeftAnti, JoinType::Cross}) {
            const DataFrame j = t == JoinType::Cross ? L.cross_join(R) : L.join_on(R, {{"lk", "rk"}}, t);
            const Batch b = j.collect();
            std::multiset<Row> got;
            const auto glk = ints(b, "lk"), glv = ints(b, "lv");
            const bool semi = t == JoinType::LeftSemi || t == JoinType::LeftAnti;
            const auto grk = semi ? glk : ints(b, "rk"), grv = semi ? glv : ints(b, "rv");
            for (size_t r = 0; r < b.num_rows(); ++r) {
                if (semi) got.insert({glk[r], glv[r], std::nullopt, std::nullopt});
                else got.insert({glk[r], glv[r], grk[r], grv[r]});
            }
            EXPECT_EQ(got, reference_join(lk, lv, rk, rv, t))
                << join_type_name(t) << " threshold=" << threshold << "\n" << j.explain();
            EXPECT_EQ(j.count(), got.size()) << join_type_name(t);
        }
    }
}

TEST_F(DataFrameSqlTest, UsingJoinColumns) {
    // Spark: df.join(other, Seq("dept"), how) -- the key appears once.
    Batch inner = people().join(depts(), {"dept"}).order_by({asc(col("id"))}).collect();
    EXPECT_EQ(schema_string(inner.schema()), "[dept: bigint, id: bigint, name: string, salary: double, dname: string]");
    EXPECT_EQ(ints(inner, "id"), (std::vector<OptI>{1, 2, 3, 6}));  // null dept never matches
    Batch full = people().join(depts(), {"dept"}, JoinType::Full).order_by({asc(col("dept")), asc(col("id"))}).collect();
    // keys coalesced from both sides; unmatched null-key rows on both sides
    EXPECT_EQ(full.num_rows(), 8u);
    EXPECT_EQ(ints(full, "dept"), (std::vector<OptI>{std::nullopt, std::nullopt, 10, 10, 20, 20, 30, 40}));
    Batch right = people().join(depts(), {"dept"}, JoinType::Right).order_by({asc(col("dname"))}).collect();
    EXPECT_EQ(strs(right, "dname"), (std::vector<OptS>{"eng", "eng", "hr", "nil", "ops", "ops"}));
    EXPECT_EQ(ints(right, "dept"), (std::vector<OptI>{10, 10, 40, std::nullopt, 20, 20}));
    Batch anti = people().join(depts(), {"dept"}, JoinType::LeftAnti).order_by({asc(col("id"))}).collect();
    EXPECT_EQ(ints(anti, "id"), (std::vector<OptI>{4, 5}));
    EXPECT_EQ(anti.num_columns(), 4u);
}

// ----- unions, dedup, null handling helpers --------------------------------------

TEST_F(DataFrameSqlTest, UnionsAndDedup) {
    auto a = people().select_columns({"id", "name"});
    auto b = people().filter(col("id") <= 2).select({col("name"), (col("id") + 100).alias("id")});
    EXPECT_EQ(a.union_all(a).count(), 12u);
    Batch u = a.union_by_name(b).order_by({desc(col("id"))}).limit(2).collect();
    EXPECT_EQ(ints(u, "id"), (std::vector<OptI>{102, 101}));
    EXPECT_EQ(a.union_all(a).distinct().count(), 6u);
    Batch d = people().drop_duplicates({"dept"}).order_by({asc(col("dept"))}).collect();
    EXPECT_EQ(ints(d, "dept"), (std::vector<OptI>{std::nullopt, 10, 20, 30}));
    EXPECT_EQ(d.num_columns(), 4u);
    EXPECT_EQ(people().drop_nulls().count(), 3u);
    EXPECT_EQ(people().drop_nulls({"dept"}).count(), 5u);
    Batch f = people().fill_null(0).fill_null("?").collect();
    EXPECT_EQ(ints(f, "dept")[3], OptI(0));
    EXPECT_EQ(dbls(f, "salary")[1], OptD(0.0));
    EXPECT_EQ(strs(f, "name")[2], OptS("?"));
    Batch r = people().with_column_renamed("name", "who").drop({"salary", "dept"}).collect();
    EXPECT_EQ(schema_string(r.schema()), "[id: bigint, who: string]");
}

TEST_F(DataFrameSqlTest, SortNullOrdering) {
    Batch a = people().order_by({asc(col("salary"))}).collect();
    EXPECT_EQ(dbls(a, "salary"), (std::vector<OptD>{std::nullopt, 50.0, 60.0, 70.0, 90.0, 100.0}));
    Batch d = people().order_by({desc(col("salary"))}).collect();
    EXPECT_EQ(dbls(d, "salary"), (std::vector<OptD>{100.0, 90.0, 70.0, 60.0, 50.0, std::nullopt}));
    Batch al = people().order_by({asc_nulls_last(col("name"))}).collect();
    EXPECT_FALSE(strs(al, "name").back().has_value());
    Batch top = people().order_by({desc_nulls_first(col("dept")), asc(col("id"))}).limit(2).collect();
    EXPECT_EQ(ints(top, "id"), (std::vector<OptI>{4, 5}));
}

// ----- window functions -----------------------------------------------------------

TEST_F(DataFrameSqlTest, WindowFunctions) {
    auto df = DataFrame::from_columns(ctx,
                                      {{"g", Column(std::vector<std::string>{"a", "a", "a", "b", "b", "a"})},
                                       {"t", Column(std::vector<int64_t>{1, 2, 2, 1, 5, 3})},
                                       {"v", Column(std::vector<OptI>{10, 20, 30, 5, std::nullopt, 40})}},
                                      3);
    const WindowSpec w = Window::partition_by({col("g")}).order_by({asc(col("t"))});
    Batch b = df.select({col("g"), col("t"), col("v"), row_number().over(w).alias("rn"), rank().over(w).alias("rk"),
                         dense_rank().over(w).alias("dr"), sum(col("v")).over(w).alias("run"),
                         sum(col("v")).over(Window::partition_by({col("g")})).alias("tot"),
                         lag(col("v"), 1).over(w).alias("prev"), lead(col("v"), 1, lit(-1)).over(w).alias("next"),
                         (col("v") - avg(col("v")).over(Window::partition_by({col("g")}))).alias("dev"),
                         count_all().over(w).alias("cnt")})
                  .order_by({asc(col("g")), asc(col("t")), asc(col("v"))})
                  .collect();
    EXPECT_EQ(ints(b, "rn"), (std::vector<OptI>{1, 2, 3, 4, 1, 2}));
    EXPECT_EQ(ints(b, "rk"), (std::vector<OptI>{1, 2, 2, 4, 1, 2}));
    EXPECT_EQ(ints(b, "dr"), (std::vector<OptI>{1, 2, 2, 3, 1, 2}));
    // RANGE frame: peers (t=2) share the running total
    EXPECT_EQ(ints(b, "run"), (std::vector<OptI>{10, 60, 60, 100, 5, 5}));
    EXPECT_EQ(ints(b, "tot"), (std::vector<OptI>{100, 100, 100, 100, 5, 5}));
    EXPECT_EQ(ints(b, "cnt"), (std::vector<OptI>{1, 3, 3, 4, 1, 2}));
    const auto prev = ints(b, "prev");
    EXPECT_FALSE(prev[0].has_value());
    EXPECT_EQ(prev[3], OptI(30));
    EXPECT_FALSE(prev[4].has_value());
    EXPECT_EQ(ints(b, "next")[3], OptI(-1));
    EXPECT_EQ(dbls(b, "dev")[0], OptD(-15.0));
    EXPECT_FALSE(dbls(b, "dev")[5].has_value());
    // a window without partitioning, filtered on its output
    Batch top = df.with_column("r", row_number().over(Window::order_by({desc(col("t"))})))
                    .filter(col("r") <= 2)
                    .collect();
    EXPECT_EQ(top.num_rows(), 2u);
}

// ----- CSV -----------------------------------------------------------------------

TEST_F(DataFrameSqlTest, CsvRoundTrip) {
    const std::string path = ::testing::TempDir() + "vega_df_test.csv";
    people().write_csv(path);
    auto back = DataFrame::read_csv(ctx, path);
    EXPECT_EQ(schema_string(back.schema()), "[id: bigint, name: string, dept: bigint, salary: double]");
    Batch a = people().collect(), b = back.order_by({asc(col("id"))}).collect();
    EXPECT_EQ(strs(a, "name"), strs(b, "name"));
    EXPECT_EQ(ints(a, "dept"), ints(b, "dept"));
    EXPECT_EQ(dbls(a, "salary"), dbls(b, "salary"));
    {
        std::FILE* f = std::fopen(path.c_str(), "w");
        std::fputs("a,b,c\n1,\"x, \"\"y\"\"\",true\n2,,false\n", f);
        std::fclose(f);
    }
    Batch q = DataFrame::read_csv(ctx, path).collect();
    EXPECT_EQ(strs(q, "b"), (std::vector<OptS>{"x, \"y\"", std::nullopt}));
    EXPECT_EQ(q.schema()[2].type, DType::Bool);
    std::remove(path.c_str());
}

// ----- optimizer -------------------------------------------------------------------

namespace {
std::string optimized(const DataFrame& df) { return plan::to_string(df.optimized()); }
}

TEST_F(DataFrameSqlTest, OptimizerPushesFiltersAndPrunes) {
    auto q = people()
                 .join(depts(), {"dept"})
                 .with_column("bonus", col("salary") * 0.1)
                 .filter(col("dname") == "eng" && col("bonus") > 1.0 && col("dept") > 5)
                 .select({col("name"), col("bonus")});
    const std::string p = optimized(q);
    // the filter is split and pushed below the join on each side
    const auto join_pos = p.find("Join Inner");
    ASSERT_NE(join_pos, std::string::npos) << p;
    EXPECT_NE(p.find("Filter", join_pos), std::string::npos) << p;
    EXPECT_EQ(p.substr(0, join_pos).find("Filter"), std::string::npos) << p;
    // the key predicate dept > 5 was inferred for the other side (and moved
    // below its renaming projection)
    EXPECT_NE(p.find("Filter ((dname = eng) AND (dept > 5))"), std::string::npos) << p;
    // only needed columns are scanned (id is dropped)
    EXPECT_NE(p.find("Scan LocalRelation [name, dept, salary]"), std::string::npos) << p;
    Batch b = q.collect();
    EXPECT_EQ(strs(b, "name"), (std::vector<OptS>{"ann", std::nullopt}));
}

TEST_F(DataFrameSqlTest, OptimizerOuterJoinElimination) {
    auto q = people().join(depts(), {"dept"}, JoinType::Left).filter(col("dname") == "ops");
    EXPECT_NE(optimized(q).find("Join Inner"), std::string::npos) << optimized(q);
    EXPECT_EQ(q.count(), 2u);
    // is_null does not reject nulls: the join must stay an outer join
    auto keep = people().join(depts(), {"dept"}, JoinType::Left).filter(col("dname").is_null());
    EXPECT_NE(optimized(keep).find("LeftOuter"), std::string::npos) << optimized(keep);
    EXPECT_EQ(keep.count(), 2u);  // dan (null dept), eve (dept 30)
}

TEST_F(DataFrameSqlTest, OptimizerFoldsConstantsAndPrunesFilters) {
    auto q = people().filter(lit(1) + 1 == 2 && col("id") > 3);
    const std::string p = optimized(q);
    EXPECT_NE(p.find("Filter (id > 3)"), std::string::npos) << p;
    auto none = people().filter(lit(1) > 2).select({col("id")});
    EXPECT_NE(optimized(none).find("Empty"), std::string::npos) << optimized(none);
    EXPECT_EQ(none.count(), 0u);
    EXPECT_EQ(people().filter(lit(true)).count(), 6u);
}

TEST_F(DataFrameSqlTest, OptimizerEliminatesSortsAndCollapsesProjects) {
    auto q = people()
                 .order_by({asc(col("salary"))})
                 .with_column("x", col("id") + 1)
                 .with_column("y", col("x") * 2)
                 .group_by({col("dept")})
                 .agg({sum(col("y"))});
    const std::string p = optimized(q);
    EXPECT_EQ(p.find("Sort"), std::string::npos) << p;
    EXPECT_EQ(p.find("Project"), p.rfind("Project")) << p;  // at most one projection
    Batch b = q.order_by({asc(col("dept"))}).collect();
    EXPECT_EQ(ints(b, "sum(y)"), (std::vector<OptI>{10, 12, 20, 12}));
}

TEST_F(DataFrameSqlTest, OptimizerPushesAggregatesThroughJoins) {
    std::vector<int64_t> sk, qty;
    for (int i = 0; i < 2000; ++i) {
        sk.push_back(i % 50);
        qty.push_back(i % 7);
    }
    std::vector<int64_t> dk;
    std::vector<std::string> region;
    for (int k = 0; k < 50; ++k) {
        dk.push_back(k);
        region.push_back("r" + std::to_string(k % 4));
    }
    auto sales = DataFrame::from_columns(ctx, {{"k", Column(sk)}, {"qty", Column(qty)}}, 4);
    auto dim = DataFrame::from_columns(ctx, {{"k", Column(dk)}, {"region", Column(region)}}, 1);
    auto q = sales.join(dim, {"k"})
                 .group_by({col("region")})
                 .agg({sum(col("qty")).alias("s"), avg(col("qty")).alias("a"), count_all().alias("n"),
                       max(col("qty")).alias("m")})
                 .order_by({asc(col("region"))});
    const std::string p = optimized(q);
    EXPECT_NE(p.find("Aggregate keys=[k]"), std::string::npos) << p;  // pre-aggregated below the join
    Batch b = q.collect();
    std::map<std::string, std::tuple<int64_t, int64_t, int64_t>> want;
    for (int i = 0; i < 2000; ++i) {
        auto& [s, n, m] = want[region[static_cast<size_t>(sk[i])]];
        s += qty[i];
        ++n;
        m = std::max(m, qty[i]);
    }
    ASSERT_EQ(b.num_rows(), want.size());
    size_t r = 0;
    for (const auto& [reg, v] : want) {
        EXPECT_EQ(strs(b, "region")[r], OptS(reg));
        EXPECT_EQ(ints(b, "s")[r], OptI(std::get<0>(v)));
        EXPECT_EQ(ints(b, "n")[r], OptI(std::get<1>(v)));
        EXPECT_EQ(ints(b, "m")[r], OptI(std::get<2>(v)));
        EXPECT_NEAR(*dbls(b, "a")[r], double(std::get<0>(v)) / std::get<1>(v), 1e-12);
        ++r;
    }
    EXPECT_EQ(schema_string(b.schema()), "[region: string, s: bigint, a: double, n: bigint, m: bigint]");
}

TEST_F(DataFrameSqlTest, ExplainAndCache) {
    auto q = people().join(depts(), {"dept"}).order_by({desc(col("salary"))}).limit(2);
    const std::string e = q.explain();
    EXPECT_NE(e.find("== Optimized Logical Plan =="), std::string::npos);
    EXPECT_NE(e.find("BroadcastHashJoin"), std::string::npos) << e;
    EXPECT_NE(e.find("TakeOrdered 2"), std::string::npos) << e;
    auto cached = people().filter(col("salary") > 55.0).cache();
    EXPECT_EQ(cached.count(), 4u);
    EXPECT_EQ(cached.agg({sum(col("salary"))}).collect().column(0).values<double>()[0], 320.0);
    EXPECT_EQ(people().repartition(5, {col("dept")}).count(), 6u);
    EXPECT_EQ(people().repartition(2).group_by({col("dept")}).count().count(), 4u);
}

TEST_F(DataFrameSqlTest, DictionaryEncodedStrings) {
    std::vector<std::string> cat;
    std::vector<int64_t> v;
    for (int i = 0; i < 5000; ++i) {
        cat.push_back("c" + std::to_string(i % 9));
        v.push_back(i);
    }
    auto df = DataFrame::from_columns(ctx, {{"cat", Column(cat)}, {"v", Column(v)}}, 4);
    // predicates, functions and grouping on encoded strings
    EXPECT_EQ(df.filter(col("cat") == "c3").count(), 556u);
    EXPECT_EQ(df.filter(upper(col("cat")) == "C3" || col("cat").like("%8")).count(), 1111u);
    Batch g = df.group_by({col("cat")}).count().order_by({asc(col("cat"))}).collect();
    EXPECT_EQ(g.num_rows(), 9u);
    EXPECT_EQ(strs(g, "cat")[0], OptS("c0"));
    EXPECT_EQ(ints(g, "count")[0], OptI(556));
    Batch s = df.select({concat({col("cat"), lit("x")}).alias("y")}).distinct().collect();
    EXPECT_EQ(s.num_rows(), 9u);
}

TEST_F(DataFrameSqlTest, WindowGroupLimitMatchesReference) {
    std::mt19937 rng(3);
    std::vector<int64_t> g, s, id;
    for (int i = 0; i < 3000; ++i) {
        g.push_back(rng() % 17);
        s.push_back(rng() % 9);  // many ties
        id.push_back(i);
    }
    auto df = DataFrame::from_columns(ctx, {{"g", Column(g)}, {"s", Column(s)}, {"id", Column(id)}}, 4);
    const auto w = Window::partition_by({col("g")}).order_by({desc(col("s"))});
    for (const bool use_rank : {false, true}) {
        const WindowFunction f = use_rank ? rank() : row_number();
        auto q = df.with_column("r", f.over(w)).filter(col("r") <= 3);
        EXPECT_NE(plan::to_string(q.optimized()).find("limit 3 per partition"), std::string::npos);
        const Batch b = q.collect();
        // reference: per group, count rows whose rank is <= 3
        std::map<int64_t, std::vector<int64_t>> by_g;
        for (size_t i = 0; i < g.size(); ++i) by_g[g[i]].push_back(s[i]);
        size_t want = 0;
        for (auto& [k, v] : by_g) {
            std::sort(v.rbegin(), v.rend());
            if (!use_rank) {
                want += std::min<size_t>(3, v.size());
            } else {
                for (size_t i = 0; i < v.size(); ++i) {
                    const size_t rk = static_cast<size_t>(std::find(v.begin(), v.end(), v[i]) - v.begin()) + 1;
                    want += rk <= 3;
                }
            }
        }
        EXPECT_EQ(b.num_rows(), want) << (use_rank ? "rank" : "row_number");
        for (auto r : ints(b, "r")) EXPECT_LE(*r, 3);
    }
}

TEST_F(DataFrameSqlTest, DistinctValueEstimates) {
    std::vector<int64_t> id, k, small;
    std::vector<std::string> s;
    std::vector<OptI> sparse;
    for (int i = 0; i < 200000; ++i) {
        id.push_back(i);
        k.push_back(i % 50000);
        small.push_back(i % 7);
        s.push_back("v" + std::to_string(i % 300));
        sparse.push_back(i % 4 == 0 ? OptI(i % 10) : std::nullopt);
    }
    auto df = DataFrame::from_columns(ctx,
                                      {{"id", Column(id)}, {"k", Column(k)}, {"small", Column(small)},
                                       {"s", Column(s)}, {"sparse", Column(sparse)}},
                                      6);
    auto near = [](std::optional<double> v, double want, double tol) {
        return v && std::abs(*v - want) <= want * tol;
    };
    EXPECT_TRUE(near(df.estimated_distinct("id"), 200000, 0.05)) << *df.estimated_distinct("id");
    EXPECT_TRUE(near(df.estimated_distinct("k"), 50000, 0.05)) << *df.estimated_distinct("k");
    EXPECT_EQ(df.estimated_distinct("small"), 7.0);
    EXPECT_EQ(df.estimated_distinct("s"), 300.0);        // dictionary-encoded: exact
    EXPECT_EQ(df.estimated_distinct("sparse"), 6.0);     // 0, 2, 4, 6, 8 + null
    // propagation through plans
    EXPECT_EQ(df.select({col("s").alias("t")}).estimated_distinct("t"), 300.0);
    EXPECT_FALSE(df.select({(col("k") + 1).alias("k1")}).estimated_distinct("k1").has_value());
    EXPECT_EQ(df.group_by({col("small")}).count().estimated_rows(), 7.0);
    EXPECT_TRUE(near(df.group_by({col("small"), col("s")}).count().estimated_rows(), 2100, 0.01));
    EXPECT_NEAR(df.filter(col("s") == "v1").estimated_rows(), 200000.0 / 300, 1);
    EXPECT_EQ(df.limit(5).estimated_distinct("k"), 5.0);
    // cache() re-derives statistics from the cached data
    auto c = df.filter(col("small") == 3).cache();
    EXPECT_EQ(c.estimated_distinct("small"), 1.0);
    EXPECT_EQ(c.estimated_rows(), static_cast<double>(c.count()));
}

TEST_F(DataFrameSqlTest, EagerAggregationNeedsAReduction) {
    // Line items x orders: ~2 items per order, so pre-aggregating items by
    // order_id would not shrink them -- the rule must not fire.
    std::vector<int64_t> item_order, qty, order_id, customer;
    for (int i = 0; i < 20000; ++i) {
        item_order.push_back(i / 2);
        qty.push_back(i % 5);
    }
    for (int o = 0; o < 10000; ++o) {
        order_id.push_back(o);
        customer.push_back(o % 300);
    }
    auto items = DataFrame::from_columns(ctx, {{"order_id", Column(item_order)}, {"qty", Column(qty)}}, 4);
    auto orders = DataFrame::from_columns(ctx, {{"order_id", Column(order_id)}, {"customer", Column(customer)}}, 2);
    auto q = items.join(orders, {"order_id"}).group_by({col("customer")}).agg({sum(col("qty")).alias("q")});
    const std::string p = plan::to_string(q.optimized());
    EXPECT_EQ(p.find("Aggregate keys=[order_id]"), std::string::npos) << p;
    int64_t total = 0;
    for (auto v : ints(q.collect(), "q")) total += *v;
    EXPECT_EQ(total, 2 * 20000);
    EXPECT_EQ(q.count(), 300u);
    // ...while a fact table with few distinct keys is pre-aggregated
    std::vector<int64_t> few;
    for (int i = 0; i < 20000; ++i) few.push_back(i % 100);
    auto facts = DataFrame::from_columns(ctx, {{"order_id", Column(few)}, {"qty", Column(qty)}}, 4);
    auto q2 = facts.join(orders, {"order_id"}).group_by({col("customer")}).agg({sum(col("qty")).alias("q")});
    EXPECT_NE(plan::to_string(q2.optimized()).find("Aggregate keys=[order_id]"), std::string::npos);
}
