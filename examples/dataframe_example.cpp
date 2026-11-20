// The example from docs/cpp-dataframes.md.
#include <vega/vega.hpp>
#include <vega/dataframe.hpp>

#include <iostream>

using namespace vega::df;

int main() {
    vega::SparkContext ctx(4);

    auto sales = DataFrame::from_columns(ctx, {
        {"store",    Column(std::vector<int64_t>{1, 2, 1, 3})},
        {"qty",      Column(std::vector<int64_t>{5, 1, 2, 7})},
        {"price",    Column(std::vector<double>{9.5, 3.0, 9.5, 1.25})},
        {"category", Column(std::vector<std::string>{"books", "toys", "books", "food"})},
    }, /*partitions=*/8);

    auto stores = DataFrame::from_columns(ctx, {
        {"store",  Column(std::vector<int64_t>{1, 2, 3})},
        {"region", Column(std::vector<std::string>{"east", "west", "east"})},
    }, 1);

    auto report = sales
        .filter(col("qty") > 1)
        .with_column("revenue", col("qty") * col("price"))
        .join(stores, {"store"})
        .group_by({col("region")})
        .agg({sum(col("revenue")).alias("revenue"), count_all().alias("orders")})
        .order_by({desc(col("revenue"))});

    std::cout << report.show();
    return 0;
}
