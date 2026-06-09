#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <vega/vega.hpp>

int main() {
    // Create a context with 4 worker actors.
    vega::SparkContext ctx(4);

    std::vector<std::string> lines = {
        "hello world hello",
        "hello spark world",
        "vega is fast",
        "actors are awesome",
        "hello actors hello world",
    };

    // ── Build the lazy RDD pipeline ──────────────────────────────────────
    //
    //  parallelize  →  flatMap (split into words)
    //               →  map    (word → (word, 1))
    //               →  collect
    //
    // reduceByKey requires shuffle (Phase 2), so we aggregate locally.

    auto word_pairs = ctx.parallelize(lines, 2)
        .flatMap([](const std::string& line) {
            std::vector<std::string> words;
            std::istringstream iss(line);
            std::string w;
            while (iss >> w) words.push_back(w);
            return words;
        })
        .map([](const std::string& word) -> std::pair<std::string, int> {
            return {word, 1};
        })
        .collect();

    // Local aggregation (will be replaced by reduceByKey in Phase 2).
    std::map<std::string, int> counts;
    for (auto& [word, one] : word_pairs)
        counts[word] += one;

    std::cout << "=== Word Count Results ===\n";
    for (auto& [word, count] : counts)
        std::cout << "  " << word << ": " << count << "\n";

    // ── Demonstrate other actions ────────────────────────────────────────

    auto numbers = ctx.parallelize(std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, 3);

    auto evens = numbers.filter([](const int& n) { return n % 2 == 0; });

    std::cout << "\nEven numbers: ";
    for (int n : evens.collect()) std::cout << n << " ";
    std::cout << "\n";

    auto doubled = numbers.map([](const int& n) { return n * 2; });

    int sum = doubled.reduce([](const int& a, const int& b) { return a + b; });
    std::cout << "Sum of doubled: " << sum << "\n";

    std::cout << "Count of numbers: " << numbers.count() << "\n";

    return 0;
}
