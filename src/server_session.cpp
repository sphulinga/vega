#include <atomic>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "vega/client/protocol.hpp"
#include "vega/client/native_ops.hpp"
#include "vega/client/session_server.hpp"
#include "vega/core/context.hpp"
#include "vega/core/fused_rdd.hpp"
#include "vega/core/shuffled_rdd.hpp"
#include "vega/network/tcp_connection.hpp"
#include "vega/python/gil_guard.hpp"

namespace py = pybind11;

namespace {

enum class StageOp : uint8_t {
    Map = 1,
    Filter = 2,
    FlatMap = 3,
    ReduceByKey = 4,
    GroupByKey = 5,
};

enum class ActionOp : uint8_t {
    Count = 1,
    Collect = 2,
    Reduce = 3,
    Foreach = 4,
};

struct Stage {
    StageOp op;
    py::object func;
    size_t num_partitions = 0;
    bool is_native = false;
    vega::client::NativeOp native_op = vega::client::NativeOp::MapAffine;
    std::vector<vega::client::NativeParam> native_params;
};

static size_t py_hash(const py::object& obj) {
    const ssize_t h = PyObject_Hash(obj.ptr());
    if (h == -1) throw py::error_already_set();
    return static_cast<size_t>(h);
}

using PyAggBucket = std::vector<std::pair<py::object, py::object>>;

static py::object* py_find_agg(PyAggBucket& bucket, const py::object& key) {
    for (auto& [existing_key, value] : bucket) {
        if (PyObject_RichCompareBool(key.ptr(), existing_key.ptr(), Py_EQ) == 1)
            return &value;
    }
    return nullptr;
}

static void py_accumulate(std::unordered_map<size_t, PyAggBucket>& groups,
                          py::object key,
                          py::object value,
                          const py::object& combine) {
    PyAggBucket& bucket = groups[py_hash(key)];
    if (py::object* acc = py_find_agg(bucket, key))
        *acc = combine(*acc, value);
    else
        bucket.emplace_back(std::move(key), std::move(value));
}

static size_t py_distinct_keys(const std::unordered_map<size_t, PyAggBucket>& groups) {
    size_t count = 0;
    for (const auto& [_, bucket] : groups)
        count += bucket.size();
    return count;
}

// Source RDD: integer range [start, end) as native int64 (no Python objects).
class NativeInt64RangeRDDImpl : public vega::RDDBase {
public:
    NativeInt64RangeRDDImpl(int64_t start, int64_t end, size_t num_partitions)
        : start_(start), end_(end), num_partitions_(num_partitions) {}

    size_t num_partitions() const override { return num_partitions_; }

    std::any compute_partition(size_t pid) const override {
        std::vector<int64_t> out;
        for (int64_t i = partition_begin(pid); i < partition_end(pid); ++i)
            out.push_back(i);
        return out;
    }

    size_t compute_partition_count(size_t pid) const override {
        return static_cast<size_t>(std::max<int64_t>(0, partition_end(pid) - partition_begin(pid)));
    }

    std::vector<std::shared_ptr<vega::RDDBase>> dependencies() const override {
        return {};
    }

    bool partition_indexed_ints(size_t pid, int64_t& begin, int64_t& end) const override {
        begin = partition_begin(pid);
        end = partition_end(pid);
        return true;
    }

private:
    int64_t start_, end_;
    size_t num_partitions_;

    int64_t partition_begin(size_t pid) const {
        const int64_t total = end_ - start_;
        const int64_t chunk = (total + static_cast<int64_t>(num_partitions_) - 1)
                              / static_cast<int64_t>(num_partitions_);
        return start_ + static_cast<int64_t>(pid) * chunk;
    }

    int64_t partition_end(size_t pid) const {
        const int64_t total = end_ - start_;
        const int64_t chunk = (total + static_cast<int64_t>(num_partitions_) - 1)
                              / static_cast<int64_t>(num_partitions_);
        return std::min(start_ + static_cast<int64_t>(pid + 1) * chunk, end_);
    }
};

class NativeFloat64RangeRDDImpl : public vega::RDDBase {
public:
    NativeFloat64RangeRDDImpl(int64_t start, int64_t end, size_t num_partitions)
        : start_(start), end_(end), num_partitions_(num_partitions) {}

    size_t num_partitions() const override { return num_partitions_; }

    std::any compute_partition(size_t pid) const override {
        std::vector<double> out;
        for (int64_t i = partition_begin(pid); i < partition_end(pid); ++i)
            out.push_back(static_cast<double>(i));
        return out;
    }

    size_t compute_partition_count(size_t pid) const override {
        return static_cast<size_t>(std::max<int64_t>(0, partition_end(pid) - partition_begin(pid)));
    }

    std::vector<std::shared_ptr<vega::RDDBase>> dependencies() const override {
        return {};
    }

    bool partition_indexed_doubles(size_t pid, int64_t& begin, int64_t& end) const override {
        begin = partition_begin(pid);
        end = partition_end(pid);
        return true;
    }

private:
    int64_t start_, end_;
    size_t num_partitions_;

    int64_t partition_begin(size_t pid) const {
        const int64_t total = end_ - start_;
        const int64_t chunk = (total + static_cast<int64_t>(num_partitions_) - 1)
                              / static_cast<int64_t>(num_partitions_);
        return start_ + static_cast<int64_t>(pid) * chunk;
    }

    int64_t partition_end(size_t pid) const {
        const int64_t total = end_ - start_;
        const int64_t chunk = (total + static_cast<int64_t>(num_partitions_) - 1)
                              / static_cast<int64_t>(num_partitions_);
        return std::min(start_ + static_cast<int64_t>(pid + 1) * chunk, end_);
    }
};

class NativeStringLineRDDImpl : public vega::RDDBase {
public:
    NativeStringLineRDDImpl(int64_t start, int64_t end, int64_t num_keys,
                            int64_t num_tokens, size_t num_partitions)
        : start_(start)
        , end_(end)
        , num_keys_(num_keys)
        , num_tokens_(num_tokens)
        , num_partitions_(num_partitions) {}

    size_t num_partitions() const override { return num_partitions_; }

    std::any compute_partition(size_t pid) const override {
        std::vector<std::string> out;
        for (int64_t i = partition_begin(pid); i < partition_end(pid); ++i) {
            out.push_back(
                "word" + std::to_string(i % num_keys_) +
                " tok" + std::to_string(i % num_tokens_));
        }
        return out;
    }

    size_t compute_partition_count(size_t pid) const override {
        return static_cast<size_t>(std::max<int64_t>(0, partition_end(pid) - partition_begin(pid)));
    }

    std::vector<std::shared_ptr<vega::RDDBase>> dependencies() const override {
        return {};
    }

    bool partition_indexed_ints(size_t pid, int64_t& begin, int64_t& end) const override {
        begin = partition_begin(pid);
        end = partition_end(pid);
        return true;
    }

private:
    int64_t start_, end_, num_keys_, num_tokens_;
    size_t num_partitions_;

    int64_t partition_begin(size_t pid) const {
        const int64_t total = end_ - start_;
        const int64_t chunk = (total + static_cast<int64_t>(num_partitions_) - 1)
                              / static_cast<int64_t>(num_partitions_);
        return start_ + static_cast<int64_t>(pid) * chunk;
    }

    int64_t partition_end(size_t pid) const {
        const int64_t total = end_ - start_;
        const int64_t chunk = (total + static_cast<int64_t>(num_partitions_) - 1)
                              / static_cast<int64_t>(num_partitions_);
        return std::min(start_ + static_cast<int64_t>(pid + 1) * chunk, end_);
    }
};

class NativeInt64ParallelCollectionRDDImpl : public vega::RDDBase {
public:
    explicit NativeInt64ParallelCollectionRDDImpl(
            std::vector<std::vector<int64_t>> partitions)
        : partitions_(std::move(partitions)) {}

    size_t num_partitions() const override { return partitions_.size(); }

    std::any compute_partition(size_t pid) const override {
        if (pid >= partitions_.size())
            throw std::out_of_range("partition id out of range");
        return partitions_[pid];
    }

    size_t compute_partition_count(size_t pid) const override {
        if (pid >= partitions_.size())
            throw std::out_of_range("partition id out of range");
        return partitions_[pid].size();
    }

    std::vector<std::shared_ptr<vega::RDDBase>> dependencies() const override {
        return {};
    }

private:
    std::vector<std::vector<int64_t>> partitions_;
};

class NativeFloat64ParallelCollectionRDDImpl : public vega::RDDBase {
public:
    explicit NativeFloat64ParallelCollectionRDDImpl(
            std::vector<std::vector<double>> partitions)
        : partitions_(std::move(partitions)) {}

    size_t num_partitions() const override { return partitions_.size(); }

    std::any compute_partition(size_t pid) const override {
        if (pid >= partitions_.size())
            throw std::out_of_range("partition id out of range");
        return partitions_[pid];
    }

    size_t compute_partition_count(size_t pid) const override {
        if (pid >= partitions_.size())
            throw std::out_of_range("partition id out of range");
        return partitions_[pid].size();
    }

    std::vector<std::shared_ptr<vega::RDDBase>> dependencies() const override {
        return {};
    }

private:
    std::vector<std::vector<double>> partitions_;
};

class NativeStringParallelCollectionRDDImpl : public vega::RDDBase {
public:
    explicit NativeStringParallelCollectionRDDImpl(
            std::vector<std::vector<std::string>> partitions)
        : partitions_(std::move(partitions)) {}

    size_t num_partitions() const override { return partitions_.size(); }

    std::any compute_partition(size_t pid) const override {
        if (pid >= partitions_.size())
            throw std::out_of_range("partition id out of range");
        return partitions_[pid];
    }

    size_t compute_partition_count(size_t pid) const override {
        if (pid >= partitions_.size())
            throw std::out_of_range("partition id out of range");
        return partitions_[pid].size();
    }

    std::vector<std::shared_ptr<vega::RDDBase>> dependencies() const override {
        return {};
    }

private:
    std::vector<std::vector<std::string>> partitions_;
};

static std::vector<std::string> split_string(const std::string& s,
                                             const std::string& delim) {
    std::vector<std::string> out;
    if (delim.empty()) {
        size_t i = 0;
        while (i < s.size()) {
            while (i < s.size() &&
                   std::isspace(static_cast<unsigned char>(s[i])))
                ++i;
            if (i >= s.size())
                break;
            size_t j = i;
            while (j < s.size() &&
                   !std::isspace(static_cast<unsigned char>(s[j])))
                ++j;
            out.emplace_back(s.substr(i, j - i));
            i = j;
        }
        return out;
    }
    if (delim.size() == 1) {
        const char ch = delim.front();
        size_t start = 0;
        while (start <= s.size()) {
            const size_t pos = s.find(ch, start);
            if (pos == std::string::npos) {
                out.emplace_back(s.substr(start));
                break;
            }
            out.emplace_back(s.substr(start, pos - start));
            start = pos + 1;
        }
        return out;
    }
    size_t start = 0;
    while (start <= s.size()) {
        const size_t pos = s.find(delim, start);
        if (pos == std::string::npos) {
            out.emplace_back(s.substr(start));
            break;
        }
        out.emplace_back(s.substr(start, pos - start));
        start = pos + delim.size();
    }
    return out;
}

// Legacy Python-object range (parallelize fallback paths).
class RangeRDDImpl : public vega::RDDBase {
public:
    RangeRDDImpl(int64_t start, int64_t end, size_t num_partitions)
        : start_(start), end_(end), num_partitions_(num_partitions) {}

    size_t num_partitions() const override { return num_partitions_; }

    std::any compute_partition(size_t pid) const override {
        py::gil_scoped_acquire acquire;
        std::vector<py::object> out;
        for (int64_t i = partition_begin(pid); i < partition_end(pid); ++i)
            out.push_back(py::int_(i));
        return out;
    }

    size_t compute_partition_count(size_t pid) const override {
        return static_cast<size_t>(std::max<int64_t>(0, partition_end(pid) - partition_begin(pid)));
    }

    std::vector<std::shared_ptr<vega::RDDBase>> dependencies() const override {
        return {};
    }

private:
    int64_t start_, end_;
    size_t num_partitions_;

    int64_t partition_begin(size_t pid) const {
        const int64_t total = end_ - start_;
        const int64_t chunk = (total + static_cast<int64_t>(num_partitions_) - 1)
                              / static_cast<int64_t>(num_partitions_);
        return start_ + static_cast<int64_t>(pid) * chunk;
    }

    int64_t partition_end(size_t pid) const {
        const int64_t total = end_ - start_;
        const int64_t chunk = (total + static_cast<int64_t>(num_partitions_) - 1)
                              / static_cast<int64_t>(num_partitions_);
        return std::min(start_ + static_cast<int64_t>(pid + 1) * chunk, end_);
    }
};

static std::vector<py::object> py_parent_partition(
        const std::shared_ptr<vega::RDDBase>& parent, size_t pid) {
    return std::any_cast<std::vector<py::object>>(parent->compute_partition(pid));
}

class Int64ToPyObjectRDDImpl : public vega::RDDBase {
public:
    explicit Int64ToPyObjectRDDImpl(std::shared_ptr<vega::RDDBase> inner)
        : inner_(std::move(inner))
    {
        set_context(inner_->context());
    }

    bool uses_python_runtime() const override { return true; }

    size_t num_partitions() const override { return inner_->num_partitions(); }

    std::any compute_partition(size_t pid) const override {
        py::gil_scoped_acquire acquire;
        auto input = std::any_cast<std::vector<int64_t>>(inner_->compute_partition(pid));
        std::vector<py::object> output;
        output.reserve(input.size());
        for (int64_t v : input)
            output.push_back(py::int_(v));
        return output;
    }

    size_t compute_partition_count(size_t pid) const override {
        return inner_->compute_partition_count(pid);
    }

    std::vector<std::shared_ptr<vega::RDDBase>> dependencies() const override {
        return {inner_};
    }

private:
    std::shared_ptr<vega::RDDBase> inner_;
};

class PairInt64ToPyObjectRDDImpl : public vega::RDDBase {
public:
    explicit PairInt64ToPyObjectRDDImpl(std::shared_ptr<vega::RDDBase> inner)
        : inner_(std::move(inner))
    {
        set_context(inner_->context());
    }

    bool uses_python_runtime() const override { return true; }

    size_t num_partitions() const override { return inner_->num_partitions(); }

    std::any compute_partition(size_t pid) const override {
        py::gil_scoped_acquire acquire;
        auto input = std::any_cast<std::vector<std::pair<int64_t, int64_t>>>(
            inner_->compute_partition(pid));
        std::vector<py::object> output;
        output.reserve(input.size());
        for (auto& [k, v] : input)
            output.push_back(py::make_tuple(k, v));
        return output;
    }

    size_t compute_partition_count(size_t pid) const override {
        return inner_->compute_partition_count(pid);
    }

    std::vector<std::shared_ptr<vega::RDDBase>> dependencies() const override {
        return {inner_};
    }

private:
    std::shared_ptr<vega::RDDBase> inner_;
};

class GroupedPairInt64ToPyObjectRDDImpl : public vega::RDDBase {
public:
    explicit GroupedPairInt64ToPyObjectRDDImpl(std::shared_ptr<vega::RDDBase> inner)
        : inner_(std::move(inner))
    {
        set_context(inner_->context());
    }

    bool uses_python_runtime() const override { return true; }

    size_t num_partitions() const override { return inner_->num_partitions(); }

    std::any compute_partition(size_t pid) const override {
        py::gil_scoped_acquire acquire;
        auto input = std::any_cast<
            std::vector<std::pair<int64_t, std::vector<int64_t>>>>(
            inner_->compute_partition(pid));
        std::vector<py::object> output;
        output.reserve(input.size());
        for (auto& [k, vs] : input) {
            py::list lst;
            for (int64_t v : vs)
                lst.append(v);
            output.push_back(py::make_tuple(k, std::move(lst)));
        }
        return output;
    }

    size_t compute_partition_count(size_t pid) const override {
        return inner_->compute_partition_count(pid);
    }

    std::vector<std::shared_ptr<vega::RDDBase>> dependencies() const override {
        return {inner_};
    }

private:
    std::shared_ptr<vega::RDDBase> inner_;
};

class Float64ToPyObjectRDDImpl : public vega::RDDBase {
public:
    explicit Float64ToPyObjectRDDImpl(std::shared_ptr<vega::RDDBase> inner)
        : inner_(std::move(inner))
    {
        set_context(inner_->context());
    }

    bool uses_python_runtime() const override { return true; }

    size_t num_partitions() const override { return inner_->num_partitions(); }

    std::any compute_partition(size_t pid) const override {
        py::gil_scoped_acquire acquire;
        auto input = std::any_cast<std::vector<double>>(inner_->compute_partition(pid));
        std::vector<py::object> output;
        output.reserve(input.size());
        for (double v : input)
            output.push_back(py::float_(v));
        return output;
    }

    size_t compute_partition_count(size_t pid) const override {
        return inner_->compute_partition_count(pid);
    }

    std::vector<std::shared_ptr<vega::RDDBase>> dependencies() const override {
        return {inner_};
    }

private:
    std::shared_ptr<vega::RDDBase> inner_;
};

class StringToPyObjectRDDImpl : public vega::RDDBase {
public:
    explicit StringToPyObjectRDDImpl(std::shared_ptr<vega::RDDBase> inner)
        : inner_(std::move(inner))
    {
        set_context(inner_->context());
    }

    bool uses_python_runtime() const override { return true; }

    size_t num_partitions() const override { return inner_->num_partitions(); }

    std::any compute_partition(size_t pid) const override {
        py::gil_scoped_acquire acquire;
        auto input = std::any_cast<std::vector<std::string>>(inner_->compute_partition(pid));
        std::vector<py::object> output;
        output.reserve(input.size());
        for (const auto& s : input)
            output.push_back(py::str(s));
        return output;
    }

    size_t compute_partition_count(size_t pid) const override {
        return inner_->compute_partition_count(pid);
    }

    std::vector<std::shared_ptr<vega::RDDBase>> dependencies() const override {
        return {inner_};
    }

private:
    std::shared_ptr<vega::RDDBase> inner_;
};

class PairDoubleInt64ToPyObjectRDDImpl : public vega::RDDBase {
public:
    explicit PairDoubleInt64ToPyObjectRDDImpl(std::shared_ptr<vega::RDDBase> inner)
        : inner_(std::move(inner))
    {
        set_context(inner_->context());
    }

    bool uses_python_runtime() const override { return true; }

    size_t num_partitions() const override { return inner_->num_partitions(); }

    std::any compute_partition(size_t pid) const override {
        py::gil_scoped_acquire acquire;
        auto input = std::any_cast<std::vector<std::pair<double, int64_t>>>(
            inner_->compute_partition(pid));
        std::vector<py::object> output;
        output.reserve(input.size());
        for (auto& [k, v] : input)
            output.push_back(py::make_tuple(k, v));
        return output;
    }

    size_t compute_partition_count(size_t pid) const override {
        return inner_->compute_partition_count(pid);
    }

    std::vector<std::shared_ptr<vega::RDDBase>> dependencies() const override {
        return {inner_};
    }

private:
    std::shared_ptr<vega::RDDBase> inner_;
};

class PairStringInt64ToPyObjectRDDImpl : public vega::RDDBase {
public:
    explicit PairStringInt64ToPyObjectRDDImpl(std::shared_ptr<vega::RDDBase> inner)
        : inner_(std::move(inner))
    {
        set_context(inner_->context());
    }

    bool uses_python_runtime() const override { return true; }

    size_t num_partitions() const override { return inner_->num_partitions(); }

    std::any compute_partition(size_t pid) const override {
        py::gil_scoped_acquire acquire;
        auto input = std::any_cast<std::vector<std::pair<std::string, int64_t>>>(
            inner_->compute_partition(pid));
        std::vector<py::object> output;
        output.reserve(input.size());
        for (auto& [k, v] : input)
            output.push_back(py::make_tuple(py::str(k), v));
        return output;
    }

    size_t compute_partition_count(size_t pid) const override {
        return inner_->compute_partition_count(pid);
    }

    std::vector<std::shared_ptr<vega::RDDBase>> dependencies() const override {
        return {inner_};
    }

private:
    std::shared_ptr<vega::RDDBase> inner_;
};

// Narrow Python transforms: acquire the GIL once per partition so worker
// threads can execute UDFs safely during shuffle map stages.
class PyMappedRDDImpl : public vega::RDDBase {
public:
    PyMappedRDDImpl(std::shared_ptr<vega::RDDBase> parent, py::object func)
        : parent_(std::move(parent)), func_(std::move(func))
    {
        set_context(parent_->context());
    }

    bool uses_python_runtime() const override { return true; }

    size_t num_partitions() const override { return parent_->num_partitions(); }

    std::any compute_partition(size_t pid) const override {
        py::gil_scoped_acquire acquire;
        auto input = py_parent_partition(parent_, pid);
        std::vector<py::object> output;
        output.reserve(input.size());
        for (auto& elem : input)
            output.push_back(func_(std::move(elem)));
        return output;
    }

    size_t compute_partition_count(size_t pid) const override {
        py::gil_scoped_acquire acquire;
        auto input = py_parent_partition(parent_, pid);
        for (const auto& elem : input)
            (void)func_(elem);
        return input.size();
    }

    std::vector<std::shared_ptr<vega::RDDBase>> dependencies() const override {
        return {parent_};
    }

private:
    std::shared_ptr<vega::RDDBase> parent_;
    py::object func_;
};

class PyFilteredRDDImpl : public vega::RDDBase {
public:
    PyFilteredRDDImpl(std::shared_ptr<vega::RDDBase> parent, py::object pred)
        : parent_(std::move(parent)), pred_(std::move(pred))
    {
        set_context(parent_->context());
    }

    bool uses_python_runtime() const override { return true; }

    size_t num_partitions() const override { return parent_->num_partitions(); }

    std::any compute_partition(size_t pid) const override {
        py::gil_scoped_acquire acquire;
        auto input = py_parent_partition(parent_, pid);
        std::vector<py::object> output;
        output.reserve(input.size());
        for (auto& elem : input) {
            if (py::cast<bool>(pred_(elem)))
                output.push_back(std::move(elem));
        }
        return output;
    }

    size_t compute_partition_count(size_t pid) const override {
        py::gil_scoped_acquire acquire;
        auto input = py_parent_partition(parent_, pid);
        size_t out = 0;
        for (const auto& elem : input) {
            if (py::cast<bool>(pred_(elem))) ++out;
        }
        return out;
    }

    std::vector<std::shared_ptr<vega::RDDBase>> dependencies() const override {
        return {parent_};
    }

private:
    std::shared_ptr<vega::RDDBase> parent_;
    py::object pred_;
};

class PyFlatMappedRDDImpl : public vega::RDDBase {
public:
    PyFlatMappedRDDImpl(std::shared_ptr<vega::RDDBase> parent, py::object func)
        : parent_(std::move(parent)), func_(std::move(func))
    {
        set_context(parent_->context());
    }

    bool uses_python_runtime() const override { return true; }

    size_t num_partitions() const override { return parent_->num_partitions(); }

    std::any compute_partition(size_t pid) const override {
        py::gil_scoped_acquire acquire;
        auto input = py_parent_partition(parent_, pid);
        std::vector<py::object> output;
        output.reserve(input.size());
        for (auto& elem : input) {
            py::object mapped = func_(std::move(elem));
            for (auto item : mapped)
                output.push_back(py::reinterpret_borrow<py::object>(item));
        }
        return output;
    }

    size_t compute_partition_count(size_t pid) const override {
        py::gil_scoped_acquire acquire;
        auto input = py_parent_partition(parent_, pid);
        size_t out = 0;
        for (auto& elem : input) {
            py::object mapped = func_(std::move(elem));
            out += static_cast<size_t>(py::len(mapped));
        }
        return out;
    }

    std::vector<std::shared_ptr<vega::RDDBase>> dependencies() const override {
        return {parent_};
    }

private:
    std::shared_ptr<vega::RDDBase> parent_;
    py::object func_;
};

// reduceByKey for Python (k, v) tuples stored as py::object.
class PyReduceByKeyRDDImpl : public vega::RDDBase {
public:
    PyReduceByKeyRDDImpl(std::shared_ptr<vega::RDDBase> parent,
                         py::object combine,
                         size_t num_partitions)
        : parent_(std::move(parent))
        , combine_(std::move(combine))
        , num_partitions_(num_partitions)
    {
        buckets_.resize(num_partitions_);
    }

    size_t num_partitions() const override { return num_partitions_; }
    bool is_narrow() const override { return false; }

    bool uses_python_runtime() const override { return true; }

    std::vector<std::shared_ptr<vega::RDDBase>> dependencies() const override {
        return {parent_};
    }

    void prepare_shuffle(const std::vector<std::any>& parent_results) override {
        py::gil_scoped_acquire acquire;
        for (auto& bucket : buckets_)
            bucket.clear();
        shuffle_ready_ = false;
        for (const auto& pr : parent_results) {
            if (!pr.has_value())
                throw std::runtime_error("reduceByKey: empty map partition result");
            auto data = std::any_cast<std::vector<py::object>>(pr);
            for (auto& item : data) {
                if (!py::isinstance<py::tuple>(item))
                    throw std::runtime_error("reduceByKey: map output is not tuple");
                py::tuple t = py::reinterpret_borrow<py::tuple>(item);
                py::object k = t[0];
                py::object v = t[1];
                const size_t bucket = py_hash(k) % num_partitions_;
                buckets_[bucket].emplace_back(std::move(k), std::move(v));
            }
        }
        shuffle_ready_ = true;
    }

    std::any compute_partition(size_t pid) const override {
        py::gil_scoped_acquire acquire;
        if (!shuffle_ready_)
            throw std::runtime_error("reduceByKey: shuffle data not prepared");

        std::unordered_map<size_t, PyAggBucket> groups;
        groups.reserve(buckets_[pid].size());
        for (const auto& [k, v] : buckets_[pid])
            py_accumulate(groups, k, v, combine_);

        std::vector<py::object> result;
        result.reserve(py_distinct_keys(groups));
        for (const auto& [_, bucket] : groups) {
            for (const auto& [k, v] : bucket)
                result.push_back(py::make_tuple(k, v));
        }
        return result;
    }

    size_t compute_partition_count(size_t pid) const override {
        py::gil_scoped_acquire acquire;
        if (!shuffle_ready_)
            throw std::runtime_error("reduceByKey: shuffle data not prepared");
        std::unordered_map<size_t, PyAggBucket> groups;
        groups.reserve(buckets_[pid].size());
        for (const auto& [k, v] : buckets_[pid]) {
            (void)v;
            PyAggBucket& bucket = groups[py_hash(k)];
            if (!py_find_agg(bucket, k))
                bucket.emplace_back(k, py::object());
        }
        return py_distinct_keys(groups);
    }

private:
    std::shared_ptr<vega::RDDBase> parent_;
    py::object combine_;
    size_t num_partitions_;
    bool shuffle_ready_ = false;
    std::vector<std::vector<std::pair<py::object, py::object>>> buckets_;
};

static void py_group_append(std::unordered_map<size_t, PyAggBucket>& groups,
                            py::object key,
                            py::object value) {
    PyAggBucket& bucket = groups[py_hash(key)];
    if (py::object* acc = py_find_agg(bucket, key)) {
        py::list lst = py::reinterpret_borrow<py::list>(*acc);
        lst.append(std::move(value));
    } else {
        py::list lst;
        lst.append(std::move(value));
        bucket.emplace_back(std::move(key), std::move(lst));
    }
}

class PyGroupByKeyRDDImpl : public vega::RDDBase {
public:
    PyGroupByKeyRDDImpl(std::shared_ptr<vega::RDDBase> parent, size_t num_partitions)
        : parent_(std::move(parent)), num_partitions_(num_partitions) {
        buckets_.resize(num_partitions_);
    }

    size_t num_partitions() const override { return num_partitions_; }
    bool is_narrow() const override { return false; }
    bool uses_python_runtime() const override { return true; }

    std::vector<std::shared_ptr<vega::RDDBase>> dependencies() const override {
        return {parent_};
    }

    void prepare_shuffle(const std::vector<std::any>& parent_results) override {
        py::gil_scoped_acquire acquire;
        for (auto& bucket : buckets_)
            bucket.clear();
        shuffle_ready_ = false;
        for (const auto& pr : parent_results) {
            if (!pr.has_value())
                throw std::runtime_error("groupByKey: empty map partition result");
            auto data = std::any_cast<std::vector<py::object>>(pr);
            for (auto& item : data) {
                if (!py::isinstance<py::tuple>(item))
                    throw std::runtime_error("groupByKey: map output is not tuple");
                py::tuple t = py::reinterpret_borrow<py::tuple>(item);
                py::object k = t[0];
                py::object v = t[1];
                const size_t bucket = py_hash(k) % num_partitions_;
                buckets_[bucket].emplace_back(std::move(k), std::move(v));
            }
        }
        shuffle_ready_ = true;
    }

    std::any compute_partition(size_t pid) const override {
        py::gil_scoped_acquire acquire;
        if (!shuffle_ready_)
            throw std::runtime_error("groupByKey: shuffle data not prepared");

        std::unordered_map<size_t, PyAggBucket> groups;
        groups.reserve(buckets_[pid].size());
        for (const auto& [k, v] : buckets_[pid])
            py_group_append(groups, k, v);

        std::vector<py::object> result;
        result.reserve(py_distinct_keys(groups));
        for (const auto& [_, bucket] : groups) {
            for (const auto& [k, values] : bucket)
                result.push_back(py::make_tuple(k, values));
        }
        return result;
    }

    size_t compute_partition_count(size_t pid) const override {
        py::gil_scoped_acquire acquire;
        if (!shuffle_ready_)
            throw std::runtime_error("groupByKey: shuffle data not prepared");
        std::unordered_map<size_t, PyAggBucket> groups;
        groups.reserve(buckets_[pid].size());
        for (const auto& [k, v] : buckets_[pid]) {
            (void)v;
            PyAggBucket& bucket = groups[py_hash(k)];
            if (!py_find_agg(bucket, k))
                bucket.emplace_back(k, py::object());
        }
        return py_distinct_keys(groups);
    }

private:
    std::shared_ptr<vega::RDDBase> parent_;
    size_t num_partitions_;
    bool shuffle_ready_ = false;
    std::vector<std::vector<std::pair<py::object, py::object>>> buckets_;
};

class PythonMarkedRDDImpl : public vega::RDDBase {
public:
    explicit PythonMarkedRDDImpl(std::shared_ptr<vega::RDDBase> inner)
        : inner_(std::move(inner))
    {
        set_context(inner_->context());
    }

    bool uses_python_runtime() const override { return true; }

    size_t num_partitions() const override { return inner_->num_partitions(); }
    std::any compute_partition(size_t pid) const override {
        return inner_->compute_partition(pid);
    }
    size_t compute_partition_count(size_t pid) const override {
        return inner_->compute_partition_count(pid);
    }
    std::vector<std::shared_ptr<vega::RDDBase>> dependencies() const override {
        return inner_->dependencies();
    }
    bool is_narrow() const override { return inner_->is_narrow(); }
    void prepare_shuffle(const std::vector<std::any>& parent_results) override {
        inner_->prepare_shuffle(parent_results);
    }
    std::vector<std::pair<size_t, vega::ByteBuffer>>
    shuffle_partition(const std::any& map_result,
                      size_t num_reduce_partitions) const override {
        return inner_->shuffle_partition(map_result, num_reduce_partitions);
    }
    void receive_shuffle_data(size_t reduce_pid,
                              const vega::ByteBuffer& data) override {
        inner_->receive_shuffle_data(reduce_pid, data);
    }
    void finalize_shuffle() override { inner_->finalize_shuffle(); }

private:
    std::shared_ptr<vega::RDDBase> inner_;
};

class PyObjectParallelCollectionRDDImpl : public vega::RDDBase {
public:
    explicit PyObjectParallelCollectionRDDImpl(
        std::vector<std::vector<py::object>> partitions)
        : partitions_(std::move(partitions)) {}

    bool uses_python_runtime() const override { return true; }

    size_t num_partitions() const override { return partitions_.size(); }

    std::any compute_partition(size_t pid) const override {
        if (pid >= partitions_.size())
            throw std::out_of_range("partition id out of range");
        return partitions_[pid];
    }

    size_t compute_partition_count(size_t pid) const override {
        if (pid >= partitions_.size())
            throw std::out_of_range("partition id out of range");
        return partitions_[pid].size();
    }

    std::vector<std::shared_ptr<vega::RDDBase>> dependencies() const override {
        return {};
    }

    bool is_narrow() const override { return true; }

private:
    std::vector<std::vector<py::object>> partitions_;
};

class GILRDDImpl : public vega::RDDBase {
public:
    explicit GILRDDImpl(std::shared_ptr<vega::RDDBase> inner)
        : inner_(std::move(inner))
    {
        set_context(inner_->context());
    }

    size_t num_partitions() const override { return inner_->num_partitions(); }

    std::any compute_partition(size_t pid) const override {
        py::gil_scoped_acquire acquire;
        return inner_->compute_partition(pid);
    }

    size_t compute_partition_count(size_t pid) const override {
        py::gil_scoped_acquire acquire;
        return inner_->compute_partition_count(pid);
    }

    std::vector<std::shared_ptr<vega::RDDBase>> dependencies() const override {
        std::vector<std::shared_ptr<vega::RDDBase>> wrapped;
        for (auto& dep : inner_->dependencies())
            wrapped.push_back(std::make_shared<GILRDDImpl>(dep));
        return wrapped;
    }

    bool is_narrow() const override { return inner_->is_narrow(); }

    bool uses_python_runtime() const override {
        return inner_->uses_python_runtime();
    }

    void prepare_shuffle(const std::vector<std::any>& parent_results) override {
        py::gil_scoped_acquire acquire;
        inner_->prepare_shuffle(parent_results);
    }

    std::vector<std::pair<size_t, vega::ByteBuffer>>
    shuffle_partition(const std::any& map_result,
                      size_t num_reduce_partitions) const override {
        py::gil_scoped_acquire acquire;
        return inner_->shuffle_partition(map_result, num_reduce_partitions);
    }

    void receive_shuffle_data(size_t reduce_pid,
                              const vega::ByteBuffer& data) override {
        inner_->receive_shuffle_data(reduce_pid, data);
    }

    void finalize_shuffle() override { inner_->finalize_shuffle(); }

private:
    std::shared_ptr<vega::RDDBase> inner_;
};

static std::vector<std::any> execute_with_gil(vega::SparkContext* ctx,
                                              std::shared_ptr<vega::RDDBase> rdd) {
    py::gil_scoped_acquire ensure;
    std::vector<std::any> results;
    {
        py::gil_scoped_release release;
        results = ctx->execute(std::move(rdd));
    }
    return results;
}

static size_t execute_count_with_gil(vega::SparkContext* ctx,
                                     std::shared_ptr<vega::RDDBase> rdd) {
    py::gil_scoped_acquire ensure;
    size_t total = 0;
    {
        py::gil_scoped_release release;
        total = ctx->execute_count(std::move(rdd));
    }
    return total;
}

static py::object unpickle_bytes(const vega::ByteBuffer& buf) {
    py::bytes b(std::string(reinterpret_cast<const char*>(buf.data()), buf.size()));
    try {
        return py::module_::import("cloudpickle").attr("loads")(b);
    } catch (const py::error_already_set&) {
        PyErr_Clear();
        return py::module_::import("pickle").attr("loads")(b);
    }
}

static py::bytes pickle_dumps(const py::object& obj) {
    try {
        return py::module_::import("cloudpickle").attr("dumps")(obj);
    } catch (const py::error_already_set&) {
        PyErr_Clear();
        return py::module_::import("pickle").attr("dumps")(obj);
    }
}

static py::object wrap_python_udf(py::object func) {
    py::dict ns;
    ns["func"] = std::move(func);
    py::exec(
        "def _vega_udf(*args, **kwargs):\n"
        "    try:\n"
        "        return func(*args, **kwargs)\n"
        "    except StopIteration as exc:\n"
        "        raise RuntimeError(\n"
        "            \"Caught StopIteration thrown from user's code; failing the task\"\n"
        "        ) from exc\n",
        ns,
        ns);
    return ns["_vega_udf"];
}

class ServerSession {
public:
    explicit ServerSession(size_t num_workers)
        : ctx_(std::make_unique<vega::SparkContext>(num_workers))
    {
        vega::python_gil_runner() = [](const std::function<void()>& fn) {
            py::gil_scoped_acquire acquire;
            fn();
        };
    }

    size_t num_workers() const { return ctx_->num_workers(); }

    uint64_t parallelize_range(int64_t start, int64_t end, size_t partitions) {
        auto impl = std::make_shared<NativeInt64RangeRDDImpl>(start, end, partitions);
        impl->set_context(ctx_.get());
        return store_int64(vega::RDD<int64_t>::from_impl(std::move(impl)));
    }

    uint64_t parallelize_float_range(int64_t start, int64_t end, size_t partitions) {
        auto impl = std::make_shared<NativeFloat64RangeRDDImpl>(start, end, partitions);
        impl->set_context(ctx_.get());
        return store_float64(vega::RDD<double>::from_impl(std::move(impl)));
    }

    uint64_t parallelize_string_lines(int64_t start, int64_t end, int64_t num_keys,
                                      int64_t num_tokens, size_t partitions) {
        auto impl = std::make_shared<NativeStringLineRDDImpl>(
            start, end, num_keys, num_tokens, partitions);
        impl->set_context(ctx_.get());
        return store_string(vega::RDD<std::string>::from_impl(std::move(impl)));
    }

    uint64_t parallelize(const vega::ByteBuffer& pickled, size_t partitions) {
        py::gil_scoped_acquire acquire;
        py::object data = unpickle(pickled);
        const size_t total = static_cast<size_t>(py::len(data));
        if (partitions == 0)
            partitions = ctx_->num_workers();
        if (partitions == 0)
            partitions = 1;

        bool all_int = total > 0;
        std::vector<int64_t> flat;
        flat.reserve(total);
        for (auto item : data) {
            if (!py::isinstance<py::int_>(item)) {
                all_int = false;
                break;
            }
            flat.push_back(py::cast<int64_t>(item));
        }

        if (all_int) {
            const size_t chunk = (total + partitions - 1) / partitions;
            std::vector<std::vector<int64_t>> parts;
            for (size_t i = 0; i < partitions; ++i) {
                const size_t begin = i * chunk;
                const size_t end = std::min(begin + chunk, total);
                if (begin >= total)
                    parts.emplace_back();
                else
                    parts.emplace_back(flat.begin() + static_cast<std::ptrdiff_t>(begin),
                                       flat.begin() + static_cast<std::ptrdiff_t>(end));
            }
            auto impl = std::make_shared<NativeInt64ParallelCollectionRDDImpl>(
                std::move(parts));
            impl->set_context(ctx_.get());
            return store_int64(vega::RDD<int64_t>::from_impl(std::move(impl)));
        }

        bool all_float = total > 0;
        std::vector<double> flat_f;
        flat_f.reserve(total);
        for (auto item : data) {
            if (py::isinstance<py::bool_>(item)) {
                all_float = false;
                break;
            }
            if (py::isinstance<py::float_>(item)) {
                flat_f.push_back(py::cast<double>(item));
                continue;
            }
            if (py::isinstance<py::int_>(item)) {
                flat_f.push_back(static_cast<double>(py::cast<int64_t>(item)));
                continue;
            }
            all_float = false;
            break;
        }

        if (all_float) {
            const size_t chunk = (total + partitions - 1) / partitions;
            std::vector<std::vector<double>> parts;
            for (size_t i = 0; i < partitions; ++i) {
                const size_t begin = i * chunk;
                const size_t end = std::min(begin + chunk, total);
                if (begin >= total)
                    parts.emplace_back();
                else
                    parts.emplace_back(flat_f.begin() + static_cast<std::ptrdiff_t>(begin),
                                       flat_f.begin() + static_cast<std::ptrdiff_t>(end));
            }
            auto impl = std::make_shared<NativeFloat64ParallelCollectionRDDImpl>(
                std::move(parts));
            impl->set_context(ctx_.get());
            return store_float64(vega::RDD<double>::from_impl(std::move(impl)));
        }

        bool all_str = total > 0;
        std::vector<std::string> flat_s;
        flat_s.reserve(total);
        for (auto item : data) {
            if (!py::isinstance<py::str>(item)) {
                all_str = false;
                break;
            }
            flat_s.push_back(py::cast<std::string>(item));
        }

        if (all_str) {
            const size_t chunk = (total + partitions - 1) / partitions;
            std::vector<std::vector<std::string>> parts;
            for (size_t i = 0; i < partitions; ++i) {
                const size_t begin = i * chunk;
                const size_t end = std::min(begin + chunk, total);
                if (begin >= total)
                    parts.emplace_back();
                else
                    parts.emplace_back(flat_s.begin() + static_cast<std::ptrdiff_t>(begin),
                                       flat_s.begin() + static_cast<std::ptrdiff_t>(end));
            }
            auto impl = std::make_shared<NativeStringParallelCollectionRDDImpl>(
                std::move(parts));
            impl->set_context(ctx_.get());
            return store_string(vega::RDD<std::string>::from_impl(std::move(impl)));
        }

        std::vector<py::object> vec;
        vec.reserve(total);
        for (auto item : data)
            vec.push_back(py::reinterpret_borrow<py::object>(item));

        const size_t chunk = (total + partitions - 1) / partitions;
        std::vector<std::vector<py::object>> parts;
        parts.reserve(partitions);
        for (size_t i = 0; i < partitions; ++i) {
            const size_t begin = i * chunk;
            const size_t end = std::min(begin + chunk, total);
            if (begin >= total)
                parts.emplace_back();
            else
                parts.emplace_back(vec.begin() + static_cast<std::ptrdiff_t>(begin),
                                   vec.begin() + static_cast<std::ptrdiff_t>(end));
        }
        auto impl = std::make_shared<PyObjectParallelCollectionRDDImpl>(
            std::move(parts));
        impl->set_context(ctx_.get());
        return store(RddStorageKind::Python, std::move(impl));
    }

    uint64_t apply_stages_id(uint64_t base_id, const std::vector<Stage>& stages) {
        uint64_t cur = base_id;
        for (const auto& stage : stages) {
            if (stage.is_native) {
                cur = apply_native_transform(
                    cur, stage.op, stage.native_op, stage.native_params, stage.num_partitions);
                continue;
            }
            py::gil_scoped_acquire acquire;
            switch (stage.op) {
            case StageOp::Map:
                cur = map_with_func(cur, stage.func);
                break;
            case StageOp::Filter:
                cur = filter_with_func(cur, stage.func);
                break;
            case StageOp::FlatMap:
                cur = flat_map_with_func(cur, stage.func);
                break;
            case StageOp::ReduceByKey:
                cur = reduce_by_key_with_func(cur, stage.func, stage.num_partitions);
                break;
            case StageOp::GroupByKey:
                cur = group_by_key_with_func(cur, stage.num_partitions);
                break;
            }
        }
        return cur;
    }

    vega::ByteBuffer execute(uint64_t base_id,
                                const std::vector<Stage>& stages,
                                ActionOp action,
                                const vega::ByteBuffer& action_func_bytes) {
        const uint64_t rdd_id = apply_stages_id(base_id, stages);

        switch (action) {
        case ActionOp::Count: {
            const size_t n = count(rdd_id);
            vega::ByteBuffer body;
            vega::write_pod<uint8_t>(body, static_cast<uint8_t>(action));
            vega::write_pod<uint64_t>(body, static_cast<uint64_t>(n));
            return vega::client::encode(vega::client::MsgType::ExecuteAck, body);
        }
        case ActionOp::Collect: {
            py::list data = collect(rdd_id);
            py::gil_scoped_acquire acquire;
            py::bytes pickled = pickle_dumps(data);
            std::string s = pickled;
            vega::ByteBuffer payload(
                reinterpret_cast<const uint8_t*>(s.data()),
                reinterpret_cast<const uint8_t*>(s.data()) + s.size());
            vega::ByteBuffer body;
            vega::write_pod<uint8_t>(body, static_cast<uint8_t>(action));
            vega::client::write_bytes(body, payload);
            return vega::client::encode(vega::client::MsgType::ExecuteAck, body);
        }
        case ActionOp::Reduce: {
            py::object value = reduce(rdd_id, action_func_bytes);
            py::gil_scoped_acquire acquire;
            py::bytes out = pickle_dumps(value);
            std::string s = out;
            vega::ByteBuffer payload(
                reinterpret_cast<const uint8_t*>(s.data()),
                reinterpret_cast<const uint8_t*>(s.data()) + s.size());
            vega::ByteBuffer body;
            vega::write_pod<uint8_t>(body, static_cast<uint8_t>(action));
            vega::client::write_bytes(body, payload);
            return vega::client::encode(vega::client::MsgType::ExecuteAck, body);
        }
        case ActionOp::Foreach: {
            foreach(rdd_id, action_func_bytes);
            vega::ByteBuffer body;
            vega::write_pod<uint8_t>(body, static_cast<uint8_t>(action));
            return vega::client::encode(vega::client::MsgType::ExecuteAck, body);
        }
        }
        throw std::runtime_error("unsupported execute action");
    }

    uint64_t map(uint64_t parent, const vega::ByteBuffer& body, size_t off) {
        return apply_transform(parent, body, off, StageOp::Map);
    }

    uint64_t filter(uint64_t parent, const vega::ByteBuffer& body, size_t off) {
        return apply_transform(parent, body, off, StageOp::Filter);
    }

    uint64_t flat_map(uint64_t parent, const vega::ByteBuffer& body, size_t off) {
        return apply_transform(parent, body, off, StageOp::FlatMap);
    }

    uint64_t reduce_by_key(uint64_t parent, size_t num_partitions,
                           const vega::ByteBuffer& body, size_t off) {
        return apply_transform(parent, body, off, StageOp::ReduceByKey, num_partitions);
    }

    uint64_t group_by_key(uint64_t parent, size_t num_partitions) {
        return group_by_key_with_func(parent, num_partitions);
    }

    size_t count(uint64_t rdd_id) {
        auto& entry = load_entry(rdd_id);
        if (entry.kind == RddStorageKind::Python
            || entry.kind == RddStorageKind::GroupedPairInt64) {
            py::gil_scoped_acquire acquire;
            return execute_count_with_gil(entry.impl->context(), entry.impl);
        }
        return entry.impl->context()->execute_count(entry.impl);
    }

    py::list collect(uint64_t rdd_id, bool caller_holds_gil = false) {
        auto& entry = load_entry(rdd_id);
        if (entry.kind == RddStorageKind::Python) {
            auto results = execute_with_gil(entry.impl->context(), entry.impl);
            py::list output;
            for (auto& r : results) {
                auto vec = std::any_cast<std::vector<py::object>>(std::move(r));
                for (auto& obj : vec)
                    output.append(std::move(obj));
            }
            return output;
        }

        std::vector<std::any> results;
        if (caller_holds_gil) {
            py::gil_scoped_release release;
            results = entry.impl->context()->execute(entry.impl);
        } else {
            results = entry.impl->context()->execute(entry.impl);
        }
        py::gil_scoped_acquire acquire;
        py::list output;
        if (entry.kind == RddStorageKind::Int64) {
            for (auto& r : results) {
                auto vec = std::any_cast<std::vector<int64_t>>(std::move(r));
                for (int64_t v : vec)
                    output.append(v);
            }
            return output;
        }
        if (entry.kind == RddStorageKind::Float64) {
            for (auto& r : results) {
                auto vec = std::any_cast<std::vector<double>>(std::move(r));
                for (double v : vec)
                    output.append(v);
            }
            return output;
        }
        if (entry.kind == RddStorageKind::String) {
            for (auto& r : results) {
                auto vec = std::any_cast<std::vector<std::string>>(std::move(r));
                for (const auto& s : vec)
                    output.append(py::str(s));
            }
            return output;
        }
        if (entry.kind == RddStorageKind::PairInt64) {
            for (auto& r : results) {
                auto vec = std::any_cast<std::vector<std::pair<int64_t, int64_t>>>(
                    std::move(r));
                for (auto& [k, v] : vec)
                    output.append(py::make_tuple(k, v));
            }
            return output;
        }
        if (entry.kind == RddStorageKind::PairDoubleInt64) {
            for (auto& r : results) {
                auto vec = std::any_cast<std::vector<std::pair<double, int64_t>>>(
                    std::move(r));
                for (auto& [k, v] : vec)
                    output.append(py::make_tuple(k, v));
            }
            return output;
        }
        if (entry.kind == RddStorageKind::GroupedPairInt64) {
            for (auto& r : results) {
                auto vec = std::any_cast<
                    std::vector<std::pair<int64_t, std::vector<int64_t>>>>(std::move(r));
                for (auto& [k, vs] : vec) {
                    py::list lst;
                    for (int64_t v : vs)
                        lst.append(v);
                    output.append(py::make_tuple(k, lst));
                }
            }
            return output;
        }
        for (auto& r : results) {
            auto vec = std::any_cast<std::vector<std::pair<std::string, int64_t>>>(
                std::move(r));
            for (auto& [k, v] : vec)
                output.append(py::make_tuple(py::str(k), v));
        }
        return output;
    }

    py::list collect_partition(uint64_t rdd_id, size_t pid) {
        auto& entry = load_entry(rdd_id);
        py::list output;
        if (entry.kind == RddStorageKind::Python) {
            py::gil_scoped_acquire acquire;
            auto vec = py_parent_partition(entry.impl, pid);
            for (auto& obj : vec)
                output.append(std::move(obj));
            return output;
        }

        std::any part;
        if (entry.kind == RddStorageKind::Int64
            || entry.kind == RddStorageKind::Float64
            || entry.kind == RddStorageKind::String
            || entry.kind == RddStorageKind::PairInt64
            || entry.kind == RddStorageKind::PairDoubleInt64
            || entry.kind == RddStorageKind::GroupedPairInt64) {
            part = entry.impl->compute_partition(pid);
        } else {
            py::gil_scoped_release release;
            part = entry.impl->compute_partition(pid);
        }

        py::gil_scoped_acquire acquire;
        if (entry.kind == RddStorageKind::Int64) {
            auto vec = std::any_cast<std::vector<int64_t>>(part);
            for (int64_t v : vec)
                output.append(v);
            return output;
        }
        if (entry.kind == RddStorageKind::Float64) {
            auto vec = std::any_cast<std::vector<double>>(part);
            for (double v : vec)
                output.append(v);
            return output;
        }
        if (entry.kind == RddStorageKind::String) {
            auto vec = std::any_cast<std::vector<std::string>>(part);
            for (const auto& s : vec)
                output.append(py::str(s));
            return output;
        }
        if (entry.kind == RddStorageKind::PairInt64) {
            auto vec = std::any_cast<std::vector<std::pair<int64_t, int64_t>>>(part);
            for (auto& [k, v] : vec)
                output.append(py::make_tuple(k, v));
            return output;
        }
        if (entry.kind == RddStorageKind::PairDoubleInt64) {
            auto vec = std::any_cast<std::vector<std::pair<double, int64_t>>>(part);
            for (auto& [k, v] : vec)
                output.append(py::make_tuple(k, v));
            return output;
        }
        if (entry.kind == RddStorageKind::GroupedPairInt64) {
            auto vec = std::any_cast<
                std::vector<std::pair<int64_t, std::vector<int64_t>>>>(part);
            for (auto& [k, vs] : vec) {
                py::list lst;
                for (int64_t v : vs)
                    lst.append(v);
                output.append(py::make_tuple(k, lst));
            }
            return output;
        }
        auto vec = std::any_cast<std::vector<std::pair<std::string, int64_t>>>(part);
        for (auto& [k, v] : vec)
            output.append(py::make_tuple(py::str(k), v));
        return output;
    }

    py::list collect_partitions(uint64_t rdd_id) {
        auto& entry = load_entry(rdd_id);
        const size_t parts = entry.impl->num_partitions();
        py::list output;
        for (size_t i = 0; i < parts; ++i)
            output.append(collect_partition(rdd_id, i));
        return output;
    }

    uint64_t parallelize_partitions(const vega::ByteBuffer& pickled) {
        py::gil_scoped_acquire acquire;
        py::object data = unpickle(pickled);
        std::vector<std::vector<py::object>> partitions;
        for (auto sub : data) {
            std::vector<py::object> vec;
            for (auto item : sub)
                vec.push_back(py::reinterpret_borrow<py::object>(item));
            partitions.push_back(std::move(vec));
        }
        if (partitions.empty())
            partitions.emplace_back();
        auto impl = std::make_shared<PyObjectParallelCollectionRDDImpl>(
            std::move(partitions));
        impl->set_context(ctx_.get());
        return store(RddStorageKind::Python, std::move(impl));
    }

    py::object reduce(uint64_t rdd_id, const vega::ByteBuffer& pickled_func) {
        py::gil_scoped_acquire acquire;
        py::object func = wrap_python_udf(unpickle(pickled_func));
        auto& entry = load_entry(rdd_id);
        if (entry.kind == RddStorageKind::Python) {
            auto results = execute_with_gil(entry.impl->context(), entry.impl);
            py::object acc = py::none();
            bool first = true;
            for (auto& r : results) {
                auto vec = std::any_cast<std::vector<py::object>>(std::move(r));
                for (auto& obj : vec) {
                    if (first) {
                        acc = std::move(obj);
                        first = false;
                    } else {
                        acc = func(acc, obj);
                    }
                }
            }
            if (first)
                throw std::runtime_error("cannot reduce an empty RDD");
            return acc;
        }

        std::vector<std::any> results;
        {
            py::gil_scoped_release release;
            results = entry.impl->context()->execute(entry.impl);
        }
        py::object acc = py::none();
        bool first = true;
        if (entry.kind == RddStorageKind::Int64) {
            for (auto& r : results) {
                auto vec = std::any_cast<std::vector<int64_t>>(std::move(r));
                for (int64_t v : vec) {
                    py::object obj = py::int_(v);
                    if (first) {
                        acc = std::move(obj);
                        first = false;
                    } else {
                        acc = func(acc, obj);
                    }
                }
            }
        } else if (entry.kind == RddStorageKind::Float64) {
            for (auto& r : results) {
                auto vec = std::any_cast<std::vector<double>>(std::move(r));
                for (double v : vec) {
                    py::object obj = py::float_(v);
                    if (first) {
                        acc = std::move(obj);
                        first = false;
                    } else {
                        acc = func(acc, obj);
                    }
                }
            }
        } else if (entry.kind == RddStorageKind::String) {
            for (auto& r : results) {
                auto vec = std::any_cast<std::vector<std::string>>(std::move(r));
                for (const auto& s : vec) {
                    py::object obj = py::str(s);
                    if (first) {
                        acc = std::move(obj);
                        first = false;
                    } else {
                        acc = func(acc, obj);
                    }
                }
            }
        } else if (entry.kind == RddStorageKind::PairInt64) {
            for (auto& r : results) {
                auto vec = std::any_cast<std::vector<std::pair<int64_t, int64_t>>>(
                    std::move(r));
                for (auto& [k, v] : vec) {
                    py::object obj = py::make_tuple(k, v);
                    if (first) {
                        acc = std::move(obj);
                        first = false;
                    } else {
                        acc = func(acc, obj);
                    }
                }
            }
        } else if (entry.kind == RddStorageKind::PairDoubleInt64) {
            for (auto& r : results) {
                auto vec = std::any_cast<std::vector<std::pair<double, int64_t>>>(
                    std::move(r));
                for (auto& [k, v] : vec) {
                    py::object obj = py::make_tuple(k, v);
                    if (first) {
                        acc = std::move(obj);
                        first = false;
                    } else {
                        acc = func(acc, obj);
                    }
                }
            }
        } else {
            for (auto& r : results) {
                auto vec = std::any_cast<std::vector<std::pair<std::string, int64_t>>>(
                    std::move(r));
                for (auto& [k, v] : vec) {
                    py::object obj = py::make_tuple(py::str(k), v);
                    if (first) {
                        acc = std::move(obj);
                        first = false;
                    } else {
                        acc = func(acc, obj);
                    }
                }
            }
        }
        if (first)
            throw std::runtime_error("cannot reduce an empty RDD");
        return acc;
    }

    void foreach(uint64_t rdd_id, const vega::ByteBuffer& pickled_func) {
        py::gil_scoped_acquire acquire;
        py::object func = wrap_python_udf(unpickle(pickled_func));
        auto& entry = load_entry(rdd_id);
        if (entry.kind == RddStorageKind::Python) {
            auto results = execute_with_gil(entry.impl->context(), entry.impl);
            for (auto& r : results) {
                auto vec = std::any_cast<std::vector<py::object>>(std::move(r));
                for (auto& obj : vec)
                    func(std::move(obj));
            }
            return;
        }

        py::list data = collect(rdd_id, true);
        for (auto item : data)
            func(item);
    }

    size_t num_partitions(uint64_t rdd_id) const {
        return load_entry(rdd_id).impl->num_partitions();
    }

private:
    enum class RddStorageKind : uint8_t {
        Python,
        Int64,
        Float64,
        String,
        PairInt64,
        PairDoubleInt64,
        PairStringInt64,
        GroupedPairInt64,
    };

    struct StoredRdd {
        RddStorageKind kind;
        std::shared_ptr<vega::RDDBase> impl;
    };

    std::unique_ptr<vega::SparkContext> ctx_;
    std::atomic<uint64_t> next_id_{1};
    std::unordered_map<uint64_t, StoredRdd> rdds_;

    static void require_params(vega::client::NativeOp op,
                               const std::vector<vega::client::NativeParam>& params) {
        if (params.size() != vega::client::expected_native_param_count(op))
            throw std::runtime_error("invalid native transform parameters");
    }

    static int64_t param_i64(const vega::client::NativeParam& p) {
        if (p.kind != vega::client::NativeParamKind::I64)
            throw std::runtime_error("expected i64 native param");
        return p.i64;
    }

    static double param_f64(const vega::client::NativeParam& p) {
        if (p.kind != vega::client::NativeParamKind::F64)
            throw std::runtime_error("expected f64 native param");
        return p.f64;
    }

    static const std::string& param_str(const vega::client::NativeParam& p) {
        if (p.kind != vega::client::NativeParamKind::Str)
            throw std::runtime_error("expected string native param");
        return p.str;
    }

    vega::RDD<int64_t> load_int64(uint64_t id) {
        auto& entry = load_entry(id);
        if (entry.kind != RddStorageKind::Int64)
            throw std::runtime_error("RDD is not a native int64 pipeline");
        return vega::RDD<int64_t>::from_impl(entry.impl);
    }

    vega::RDD<std::pair<int64_t, int64_t>> load_pair(uint64_t id) {
        auto& entry = load_entry(id);
        if (entry.kind != RddStorageKind::PairInt64)
            throw std::runtime_error("RDD is not a native pair pipeline");
        return vega::RDD<std::pair<int64_t, int64_t>>::from_impl(entry.impl);
    }

    vega::RDD<double> load_float64(uint64_t id) {
        auto& entry = load_entry(id);
        if (entry.kind != RddStorageKind::Float64)
            throw std::runtime_error("RDD is not a native float64 pipeline");
        return vega::RDD<double>::from_impl(entry.impl);
    }

    vega::RDD<std::string> load_string(uint64_t id) {
        auto& entry = load_entry(id);
        if (entry.kind != RddStorageKind::String)
            throw std::runtime_error("RDD is not a native string pipeline");
        return vega::RDD<std::string>::from_impl(entry.impl);
    }

    vega::RDD<std::pair<double, int64_t>> load_pair_double_int64(uint64_t id) {
        auto& entry = load_entry(id);
        if (entry.kind != RddStorageKind::PairDoubleInt64)
            throw std::runtime_error("RDD is not a native double pair pipeline");
        return vega::RDD<std::pair<double, int64_t>>::from_impl(entry.impl);
    }

    vega::RDD<std::pair<std::string, int64_t>> load_pair_string_int64(uint64_t id) {
        auto& entry = load_entry(id);
        if (entry.kind != RddStorageKind::PairStringInt64)
            throw std::runtime_error("RDD is not a native string pair pipeline");
        return vega::RDD<std::pair<std::string, int64_t>>::from_impl(entry.impl);
    }

    uint64_t apply_transform(uint64_t parent,
                             const vega::ByteBuffer& body,
                             size_t& off,
                             StageOp stage,
                             size_t num_partitions = 0) {
        const auto mode = static_cast<vega::client::TransformMode>(
            vega::read_pod<uint8_t>(body, off));
        if (mode == vega::client::TransformMode::Native) {
            vega::client::NativeOp native_op;
            auto params = vega::client::read_native_params(body, off, native_op);
            return apply_native_transform(parent, stage, native_op, params, num_partitions);
        }

        py::gil_scoped_acquire acquire;
        auto pickled = vega::client::read_bytes(body, off);
        switch (stage) {
        case StageOp::Map:
            return map_with_func(parent, unpickle(pickled));
        case StageOp::Filter:
            return filter_with_func(parent, unpickle(pickled));
        case StageOp::FlatMap:
            return flat_map_with_func(parent, unpickle(pickled));
        case StageOp::ReduceByKey:
            return reduce_by_key_with_func(parent, unpickle(pickled), num_partitions);
        case StageOp::GroupByKey:
            return group_by_key_with_func(parent, num_partitions);
        }
        throw std::runtime_error("unsupported transform stage");
    }

    uint64_t apply_native_transform(uint64_t parent,
                                    StageOp stage,
                                    vega::client::NativeOp op,
                                    const std::vector<vega::client::NativeParam>& params,
                                    size_t num_partitions) {
        require_params(op, params);
        switch (stage) {
        case StageOp::Map:
            if (op == vega::client::NativeOp::MapAffine) {
                auto rdd = load_int64(parent).map(
                    [mul = param_i64(params[0]), add = param_i64(params[1]),
                     mod = param_i64(params[2])](const int64_t& x) {
                        return (x * mul + add) % mod;
                    });
                return store_int64(std::move(rdd));
            }
            if (op == vega::client::NativeOp::MapToPairAffine) {
                auto rdd = load_int64(parent).map(
                    [km = param_i64(params[0]), ka = param_i64(params[1]),
                     kmod = param_i64(params[2]), vm = param_i64(params[3]),
                     va = param_i64(params[4]), vmod = param_i64(params[5])](const int64_t& x) {
                        return std::make_pair((x * km + ka) % kmod, (x * vm + va) % vmod);
                    });
                return store_pair(std::move(rdd));
            }
            if (op == vega::client::NativeOp::MapToPairConst) {
                auto rdd = load_int64(parent).map(
                    [vconst = param_i64(params[0])](const int64_t& x) {
                        return std::make_pair(x, vconst);
                    });
                return store_pair(std::move(rdd));
            }
            if (op == vega::client::NativeOp::MapToPairModConst) {
                const int64_t kmod = param_i64(params[0]);
                const int64_t vconst = param_i64(params[1]);
                auto& entry = load_entry(parent);
                int64_t begin = 0;
                int64_t end = 0;
                if (entry.kind == RddStorageKind::Int64 &&
                    entry.impl->partition_indexed_ints(0, begin, end)) {
                    auto impl = std::make_shared<vega::IndexedPairModConstRDDImpl>(
                        entry.impl, kmod, vconst);
                    impl->set_context(ctx_.get());
                    return store(RddStorageKind::PairInt64, std::move(impl));
                }
                auto rdd = load_int64(parent).map(
                    [kmod, vconst](const int64_t& x) {
                        return std::make_pair(x % kmod, vconst);
                    });
                return store_pair(std::move(rdd));
            }
            if (op == vega::client::NativeOp::MapToPairModIdentity) {
                const int64_t kmod = param_i64(params[0]);
                auto rdd = load_int64(parent).map(
                    [kmod](const int64_t& x) {
                        return std::make_pair(x % kmod, x);
                    });
                return store_pair(std::move(rdd));
            }
            if (op == vega::client::NativeOp::MapToPairModConstF64) {
                const int64_t kmod = param_i64(params[0]);
                const int64_t vconst = param_i64(params[1]);
                auto& entry = load_entry(parent);
                int64_t begin = 0;
                int64_t end = 0;
                if (entry.kind == RddStorageKind::Float64 &&
                    entry.impl->partition_indexed_doubles(0, begin, end)) {
                    auto impl = std::make_shared<vega::IndexedFloatPairModConstRDDImpl>(
                        entry.impl, kmod, vconst);
                    impl->set_context(ctx_.get());
                    return store(RddStorageKind::PairInt64, std::move(impl));
                }
                auto rdd = load_float64(parent).map(
                    [kmod, vconst](const double& x) {
                        return std::make_pair(
                            static_cast<int64_t>(x) % kmod, vconst);
                    });
                return store_pair(std::move(rdd));
            }
            if (op == vega::client::NativeOp::MapAffineF64) {
                auto rdd = load_float64(parent).map(
                    [mul = param_f64(params[0]), add = param_f64(params[1])](const double& x) {
                        return x * mul + add;
                    });
                return store_float64(std::move(rdd));
            }
            if (op == vega::client::NativeOp::Float64NarrowPipeline) {
                auto& entry = load_entry(parent);
                if (entry.kind != RddStorageKind::Float64)
                    throw std::runtime_error("Float64NarrowPipeline requires float64 parent");
                auto impl = std::make_shared<vega::Float64MapFilterFlatMapPipelineRDDImpl>(
                    entry.impl,
                    param_f64(params[0]),
                    param_f64(params[1]),
                    static_cast<vega::client::FloatCompareOp>(param_i64(params[2])),
                    param_f64(params[3]),
                    param_i64(params[4]),
                    param_i64(params[5]));
                impl->set_context(ctx_.get());
                return store(RddStorageKind::Float64, std::move(impl));
            }
            if (op == vega::client::NativeOp::Int64NarrowPipeline) {
                auto& entry = load_entry(parent);
                if (entry.kind != RddStorageKind::Int64)
                    throw std::runtime_error("Int64NarrowPipeline requires int64 parent");
                auto impl = std::make_shared<vega::Int64MapFilterFlatMapPipelineRDDImpl>(
                    entry.impl,
                    param_i64(params[0]),
                    param_i64(params[1]),
                    param_i64(params[2]),
                    param_i64(params[3]),
                    param_i64(params[4]),
                    param_i64(params[5]),
                    param_i64(params[6]));
                impl->set_context(ctx_.get());
                return store(RddStorageKind::Int64, std::move(impl));
            }
            if (op == vega::client::NativeOp::StringLineWordCount) {
                auto& entry = load_entry(parent);
                if (entry.kind != RddStorageKind::String)
                    throw std::runtime_error("StringLineWordCount requires string parent");
                auto impl = std::make_shared<vega::StringLineWordCountPairRDDImpl>(
                    entry.impl,
                    param_i64(params[0]),
                    param_i64(params[1]),
                    param_i64(params[2]));
                impl->set_context(ctx_.get());
                return store(RddStorageKind::PairStringInt64, std::move(impl));
            }
            if (op == vega::client::NativeOp::MapToPairConstF64) {
                auto rdd = load_float64(parent).map(
                    [vconst = param_i64(params[0])](const double& x) {
                        return std::make_pair(x, vconst);
                    });
                return store_pair_double_int64(std::move(rdd));
            }
            if (op == vega::client::NativeOp::MapToPairConstStr) {
                auto rdd = load_string(parent).map(
                    [vconst = param_i64(params[0])](const std::string& x) {
                        return std::make_pair(x, vconst);
                    });
                return store_pair_string_int64(std::move(rdd));
            }
            break;
        case StageOp::Filter:
            if (op == vega::client::NativeOp::FilterModEq) {
                auto rdd = load_int64(parent).filter(
                    [mod = param_i64(params[0]), eq = param_i64(params[1])](const int64_t& x) {
                        return x % mod == eq;
                    });
                return store_int64(std::move(rdd));
            }
            if (op == vega::client::NativeOp::FilterCompareF64) {
                const auto cmp = static_cast<vega::client::FloatCompareOp>(
                    param_i64(params[0]));
                const double value = param_f64(params[1]);
                auto rdd = load_float64(parent).filter(
                    [cmp, value](const double& x) {
                        switch (cmp) {
                        case vega::client::FloatCompareOp::Eq: return x == value;
                        case vega::client::FloatCompareOp::Gt: return x > value;
                        case vega::client::FloatCompareOp::Lt: return x < value;
                        case vega::client::FloatCompareOp::Ge: return x >= value;
                        case vega::client::FloatCompareOp::Le: return x <= value;
                        }
                        return false;
                    });
                return store_float64(std::move(rdd));
            }
            if (op == vega::client::NativeOp::FilterLengthGt) {
                const int64_t min_len = param_i64(params[0]);
                auto rdd = load_string(parent).filter(
                    [min_len](const std::string& s) {
                        return static_cast<int64_t>(s.size()) > min_len;
                    });
                return store_string(std::move(rdd));
            }
            break;
        case StageOp::FlatMap:
            if (op == vega::client::NativeOp::FlatMapRange) {
                const int64_t start = param_i64(params[0]);
                const int64_t count = param_i64(params[1]);
                auto& entry = load_entry(parent);
                if (entry.kind != RddStorageKind::Int64)
                    throw std::runtime_error("FlatMapRange requires int64 parent");
                auto impl = std::make_shared<vega::FixedExpandFlatMappedRDDImpl<int64_t>>(
                    entry.impl, start, count);
                impl->set_context(ctx_.get());
                return store(RddStorageKind::Int64, std::move(impl));
            }
            if (op == vega::client::NativeOp::FlatMapRangeF64) {
                const int64_t start = param_i64(params[0]);
                const int64_t count = param_i64(params[1]);
                auto& entry = load_entry(parent);
                if (entry.kind != RddStorageKind::Float64)
                    throw std::runtime_error("FlatMapRangeF64 requires float64 parent");
                auto impl = std::make_shared<vega::FixedExpandFlatMappedRDDImpl<double>>(
                    entry.impl, start, count);
                impl->set_context(ctx_.get());
                return store(RddStorageKind::Float64, std::move(impl));
            }
            if (op == vega::client::NativeOp::FlatMapSplit) {
                const std::string delim = param_str(params[0]);
                auto rdd = load_string(parent).flatMap(
                    [delim](const std::string& s) {
                        return split_string(s, delim);
                    });
                return store_string(std::move(rdd));
            }
            break;
        case StageOp::ReduceByKey:
            if (op == vega::client::NativeOp::ReduceByKeySum) {
                if (num_partitions == 0)
                    num_partitions = load_entry(parent).impl->num_partitions();
                const auto kind = load_entry(parent).kind;
                if (kind == RddStorageKind::PairInt64) {
                    auto rdd = load_pair(parent).reduceByKey(
                        [](const int64_t& a, const int64_t& b) { return a + b; },
                        num_partitions);
                    return store_pair(std::move(rdd));
                }
                if (kind == RddStorageKind::PairDoubleInt64) {
                    auto rdd = load_pair_double_int64(parent).reduceByKey(
                        [](const int64_t& a, const int64_t& b) { return a + b; },
                        num_partitions);
                    return store_pair_double_int64(std::move(rdd));
                }
                if (kind == RddStorageKind::PairStringInt64) {
                    auto rdd = load_pair_string_int64(parent).reduceByKey(
                        [](const int64_t& a, const int64_t& b) { return a + b; },
                        num_partitions);
                    return store_pair_string_int64(std::move(rdd));
                }
            }
            break;
        }
        throw std::runtime_error("unsupported native transform");
    }

    std::shared_ptr<vega::RDDBase> python_parent(uint64_t parent) {
        auto& entry = load_entry(parent);
        if (entry.kind == RddStorageKind::Int64)
            return std::make_shared<Int64ToPyObjectRDDImpl>(entry.impl);
        if (entry.kind == RddStorageKind::Float64)
            return std::make_shared<Float64ToPyObjectRDDImpl>(entry.impl);
        if (entry.kind == RddStorageKind::String)
            return std::make_shared<StringToPyObjectRDDImpl>(entry.impl);
        if (entry.kind == RddStorageKind::PairInt64)
            return std::make_shared<PairInt64ToPyObjectRDDImpl>(entry.impl);
        if (entry.kind == RddStorageKind::GroupedPairInt64)
            return std::make_shared<GroupedPairInt64ToPyObjectRDDImpl>(entry.impl);
        if (entry.kind == RddStorageKind::PairDoubleInt64)
            return std::make_shared<PairDoubleInt64ToPyObjectRDDImpl>(entry.impl);
        if (entry.kind == RddStorageKind::PairStringInt64)
            return std::make_shared<PairStringInt64ToPyObjectRDDImpl>(entry.impl);
        return entry.impl;
    }

    uint64_t map_with_func(uint64_t parent, const py::object& func) {
        auto impl = std::make_shared<PyMappedRDDImpl>(
            python_parent(parent), wrap_python_udf(func));
        impl->set_context(ctx_.get());
        return store_python(std::move(impl));
    }

    uint64_t filter_with_func(uint64_t parent, const py::object& func) {
        auto impl = std::make_shared<PyFilteredRDDImpl>(
            python_parent(parent), wrap_python_udf(func));
        impl->set_context(ctx_.get());
        return store_python(std::move(impl));
    }

    uint64_t flat_map_with_func(uint64_t parent, const py::object& func) {
        auto impl = std::make_shared<PyFlatMappedRDDImpl>(
            python_parent(parent), wrap_python_udf(func));
        impl->set_context(ctx_.get());
        return store_python(std::move(impl));
    }

    uint64_t reduce_by_key_with_func(uint64_t parent, const py::object& func,
                                     size_t num_partitions) {
        auto impl = std::make_shared<PyReduceByKeyRDDImpl>(
            python_parent(parent), wrap_python_udf(func), num_partitions);
        impl->set_context(ctx_.get());
        return store_python(std::move(impl));
    }

    uint64_t group_by_key_with_func(uint64_t parent, size_t num_partitions) {
        auto& entry = load_entry(parent);
        if (num_partitions == 0)
            num_partitions = entry.impl->num_partitions();
        if (entry.kind == RddStorageKind::PairInt64) {
            auto rdd = load_pair(parent).groupByKey(num_partitions);
            return store(RddStorageKind::GroupedPairInt64, rdd.impl());
        }
        auto impl = std::make_shared<PyGroupByKeyRDDImpl>(
            python_parent(parent), num_partitions);
        impl->set_context(ctx_.get());
        return store_python(std::move(impl));
    }

    static py::object unpickle(const vega::ByteBuffer& buf) {
        return unpickle_bytes(buf);
    }

    uint64_t store_python(std::shared_ptr<vega::RDDBase> impl) {
        return store(RddStorageKind::Python, std::move(impl));
    }

    uint64_t store_int64(vega::RDD<int64_t> rdd) {
        return store(RddStorageKind::Int64, rdd.impl());
    }

    uint64_t store_pair(vega::RDD<std::pair<int64_t, int64_t>> rdd) {
        return store(RddStorageKind::PairInt64, rdd.impl());
    }

    uint64_t store_float64(vega::RDD<double> rdd) {
        return store(RddStorageKind::Float64, rdd.impl());
    }

    uint64_t store_string(vega::RDD<std::string> rdd) {
        return store(RddStorageKind::String, rdd.impl());
    }

    uint64_t store_pair_double_int64(vega::RDD<std::pair<double, int64_t>> rdd) {
        return store(RddStorageKind::PairDoubleInt64, rdd.impl());
    }

    uint64_t store_pair_string_int64(vega::RDD<std::pair<std::string, int64_t>> rdd) {
        return store(RddStorageKind::PairStringInt64, rdd.impl());
    }

    uint64_t store(vega::RDD<py::object> rdd) {
        return store_python(rdd.impl());
    }

    uint64_t store(RddStorageKind kind, std::shared_ptr<vega::RDDBase> impl) {
        uint64_t id = next_id_.fetch_add(1, std::memory_order_relaxed);
        impl->set_context(ctx_.get());
        rdds_.emplace(id, StoredRdd{kind, std::move(impl)});
        return id;
    }

    StoredRdd& load_entry(uint64_t id) {
        auto it = rdds_.find(id);
        if (it == rdds_.end())
            throw std::runtime_error("unknown RDD id: " + std::to_string(id));
        return it->second;
    }

    const StoredRdd& load_entry(uint64_t id) const {
        auto it = rdds_.find(id);
        if (it == rdds_.end())
            throw std::runtime_error("unknown RDD id: " + std::to_string(id));
        return it->second;
    }
};

static std::vector<Stage> parse_stages(const vega::ByteBuffer& body, size_t& off) {
    const uint32_t n = vega::read_pod<uint32_t>(body, off);
    std::vector<Stage> stages;
    stages.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        Stage stage;
        stage.op = static_cast<StageOp>(vega::read_pod<uint8_t>(body, off));
        stage.num_partitions = vega::read_pod<uint32_t>(body, off);
        const auto mode = static_cast<vega::client::TransformMode>(
            vega::read_pod<uint8_t>(body, off));
        if (mode == vega::client::TransformMode::Native) {
            stage.is_native = true;
            stage.native_params = vega::client::read_native_params(body, off, stage.native_op);
        } else {
            py::gil_scoped_acquire acquire;
            auto func_bytes = vega::client::read_bytes(body, off);
            if (!func_bytes.empty())
                stage.func = unpickle_bytes(func_bytes);
        }
        stages.push_back(std::move(stage));
    }
    return stages;
}

static vega::ByteBuffer send_error(const std::string& msg) {
    vega::ByteBuffer body;
    vega::client::write_bytes(
        body, vega::ByteBuffer(reinterpret_cast<const uint8_t*>(msg.data()),
                                  reinterpret_cast<const uint8_t*>(msg.data()) + msg.size()));
    return vega::client::encode(vega::client::MsgType::Error, body);
}

static void handle_client(std::unique_ptr<vega::net::TcpConnection> conn) {
    std::unique_ptr<ServerSession> session;

    while (conn->valid()) {
        vega::ByteBuffer frame = conn->recv_frame();
        if (frame.empty()) break;

        vega::client::MsgType type = vega::client::decode_type(frame);
        vega::ByteBuffer body = vega::client::decode_body(frame);
        vega::ByteBuffer reply;

        try {
            size_t off = 0;
            switch (type) {
            case vega::client::MsgType::Hello: {
                uint32_t requested = vega::read_pod<uint32_t>(body, off);
                session = std::make_unique<ServerSession>(requested);
                vega::ByteBuffer ack;
                vega::write_pod<uint32_t>(
                    ack, static_cast<uint32_t>(session->num_workers()));
                reply = vega::client::encode(vega::client::MsgType::HelloAck, ack);
                break;
            }
            case vega::client::MsgType::Parallelize: {
                if (!session) throw std::runtime_error("session not initialized");
                uint32_t parts = vega::read_pod<uint32_t>(body, off);
                auto pickled = vega::client::read_bytes(body, off);
                uint64_t id = session->parallelize(pickled, parts);
                vega::ByteBuffer ack;
                vega::write_pod<uint64_t>(ack, id);
                vega::write_pod<uint32_t>(
                    ack, static_cast<uint32_t>(session->num_partitions(id)));
                reply = vega::client::encode(
                    vega::client::MsgType::ParallelizeAck, ack);
                break;
            }
            case vega::client::MsgType::Map:
            case vega::client::MsgType::Filter:
            case vega::client::MsgType::FlatMap: {
                if (!session) throw std::runtime_error("session not initialized");
                uint64_t parent = vega::read_pod<uint64_t>(body, off);
                uint64_t id = 0;
                if (type == vega::client::MsgType::Map)
                    id = session->map(parent, body, off);
                else if (type == vega::client::MsgType::Filter)
                    id = session->filter(parent, body, off);
                else
                    id = session->flat_map(parent, body, off);
                vega::ByteBuffer ack;
                vega::write_pod<uint64_t>(ack, id);
                vega::write_pod<uint32_t>(
                    ack, static_cast<uint32_t>(session->num_partitions(id)));
                reply = vega::client::encode(
                    vega::client::MsgType::TransformAck, ack);
                break;
            }
            case vega::client::MsgType::ReduceByKey: {
                if (!session) throw std::runtime_error("session not initialized");
                uint64_t parent = vega::read_pod<uint64_t>(body, off);
                uint32_t parts = vega::read_pod<uint32_t>(body, off);
                uint64_t id = session->reduce_by_key(parent, parts, body, off);
                vega::ByteBuffer ack;
                vega::write_pod<uint64_t>(ack, id);
                vega::write_pod<uint32_t>(
                    ack, static_cast<uint32_t>(session->num_partitions(id)));
                reply = vega::client::encode(
                    vega::client::MsgType::ReduceByKeyAck, ack);
                break;
            }
            case vega::client::MsgType::GroupByKey: {
                if (!session) throw std::runtime_error("session not initialized");
                uint64_t parent = vega::read_pod<uint64_t>(body, off);
                uint32_t parts = vega::read_pod<uint32_t>(body, off);
                uint64_t id = session->group_by_key(parent, parts);
                vega::ByteBuffer ack;
                vega::write_pod<uint64_t>(ack, id);
                vega::write_pod<uint32_t>(
                    ack, static_cast<uint32_t>(session->num_partitions(id)));
                reply = vega::client::encode(
                    vega::client::MsgType::GroupByKeyAck, ack);
                break;
            }
            case vega::client::MsgType::Count: {
                if (!session) throw std::runtime_error("session not initialized");
                uint64_t id = vega::read_pod<uint64_t>(body, off);
                uint64_t n = session->count(id);
                vega::ByteBuffer ack;
                vega::write_pod<uint64_t>(ack, n);
                reply = vega::client::encode(vega::client::MsgType::CountAck, ack);
                break;
            }
            case vega::client::MsgType::Collect: {
                if (!session) throw std::runtime_error("session not initialized");
                uint64_t id = vega::read_pod<uint64_t>(body, off);
                py::gil_scoped_acquire acquire;
                py::list data = session->collect(id, true);
                py::bytes pickled = pickle_dumps(data);
                std::string s = pickled;
                vega::ByteBuffer ack(
                    reinterpret_cast<const uint8_t*>(s.data()),
                    reinterpret_cast<const uint8_t*>(s.data()) + s.size());
                reply = vega::client::encode(vega::client::MsgType::CollectAck, ack);
                break;
            }
            case vega::client::MsgType::CollectPartitions: {
                if (!session) throw std::runtime_error("session not initialized");
                uint64_t id = vega::read_pod<uint64_t>(body, off);
                py::gil_scoped_acquire acquire;
                py::list data = session->collect_partitions(id);
                py::bytes pickled = pickle_dumps(data);
                std::string s = pickled;
                vega::ByteBuffer ack(
                    reinterpret_cast<const uint8_t*>(s.data()),
                    reinterpret_cast<const uint8_t*>(s.data()) + s.size());
                reply = vega::client::encode(
                    vega::client::MsgType::CollectPartitionsAck, ack);
                break;
            }
            case vega::client::MsgType::CollectPartition: {
                if (!session) throw std::runtime_error("session not initialized");
                uint64_t id = vega::read_pod<uint64_t>(body, off);
                uint32_t pid = vega::read_pod<uint32_t>(body, off);
                py::gil_scoped_acquire acquire;
                py::list data = session->collect_partition(id, pid);
                py::bytes pickled = pickle_dumps(data);
                std::string s = pickled;
                vega::ByteBuffer ack(
                    reinterpret_cast<const uint8_t*>(s.data()),
                    reinterpret_cast<const uint8_t*>(s.data()) + s.size());
                reply = vega::client::encode(
                    vega::client::MsgType::CollectPartitionAck, ack);
                break;
            }
            case vega::client::MsgType::ParallelizePartitions: {
                if (!session) throw std::runtime_error("session not initialized");
                auto pickled = vega::client::read_bytes(body, off);
                uint64_t id = session->parallelize_partitions(pickled);
                vega::ByteBuffer ack;
                vega::write_pod<uint64_t>(ack, id);
                vega::write_pod<uint32_t>(
                    ack, static_cast<uint32_t>(session->num_partitions(id)));
                reply = vega::client::encode(
                    vega::client::MsgType::ParallelizePartitionsAck, ack);
                break;
            }
            case vega::client::MsgType::Reduce: {
                if (!session) throw std::runtime_error("session not initialized");
                uint64_t id = vega::read_pod<uint64_t>(body, off);
                auto pickled = vega::client::read_bytes(body, off);
                py::gil_scoped_acquire acquire;
                py::object value = session->reduce(id, pickled);
                py::bytes out = pickle_dumps(value);
                std::string s = out;
                vega::ByteBuffer ack(
                    reinterpret_cast<const uint8_t*>(s.data()),
                    reinterpret_cast<const uint8_t*>(s.data()) + s.size());
                reply = vega::client::encode(vega::client::MsgType::ReduceAck, ack);
                break;
            }
            case vega::client::MsgType::ParallelizeRange: {
                if (!session) throw std::runtime_error("session not initialized");
                int64_t start = static_cast<int64_t>(vega::read_pod<uint64_t>(body, off));
                int64_t end = static_cast<int64_t>(vega::read_pod<uint64_t>(body, off));
                uint32_t parts = vega::read_pod<uint32_t>(body, off);
                uint64_t id = session->parallelize_range(start, end, parts);
                vega::ByteBuffer ack;
                vega::write_pod<uint64_t>(ack, id);
                vega::write_pod<uint32_t>(
                    ack, static_cast<uint32_t>(session->num_partitions(id)));
                reply = vega::client::encode(
                    vega::client::MsgType::ParallelizeRangeAck, ack);
                break;
            }
            case vega::client::MsgType::ParallelizeFloatRange: {
                if (!session) throw std::runtime_error("session not initialized");
                int64_t start = static_cast<int64_t>(vega::read_pod<uint64_t>(body, off));
                int64_t end = static_cast<int64_t>(vega::read_pod<uint64_t>(body, off));
                uint32_t parts = vega::read_pod<uint32_t>(body, off);
                uint64_t id = session->parallelize_float_range(start, end, parts);
                vega::ByteBuffer ack;
                vega::write_pod<uint64_t>(ack, id);
                vega::write_pod<uint32_t>(
                    ack, static_cast<uint32_t>(session->num_partitions(id)));
                reply = vega::client::encode(
                    vega::client::MsgType::ParallelizeFloatRangeAck, ack);
                break;
            }
            case vega::client::MsgType::ParallelizeStringLines: {
                if (!session) throw std::runtime_error("session not initialized");
                int64_t start = static_cast<int64_t>(vega::read_pod<uint64_t>(body, off));
                int64_t end = static_cast<int64_t>(vega::read_pod<uint64_t>(body, off));
                int64_t num_keys = static_cast<int64_t>(vega::read_pod<uint64_t>(body, off));
                int64_t num_tokens = static_cast<int64_t>(vega::read_pod<uint64_t>(body, off));
                uint32_t parts = vega::read_pod<uint32_t>(body, off);
                uint64_t id = session->parallelize_string_lines(
                    start, end, num_keys, num_tokens, parts);
                vega::ByteBuffer ack;
                vega::write_pod<uint64_t>(ack, id);
                vega::write_pod<uint32_t>(
                    ack, static_cast<uint32_t>(session->num_partitions(id)));
                reply = vega::client::encode(
                    vega::client::MsgType::ParallelizeStringLinesAck, ack);
                break;
            }
            case vega::client::MsgType::Execute: {
                if (!session) throw std::runtime_error("session not initialized");
                uint64_t base = vega::read_pod<uint64_t>(body, off);
                auto stages = parse_stages(body, off);
                auto action = static_cast<ActionOp>(vega::read_pod<uint8_t>(body, off));
                auto action_func = vega::client::read_bytes(body, off);
                reply = session->execute(base, stages, action, action_func);
                {
                    py::gil_scoped_acquire acquire;
                    stages.clear();
                }
                break;
            }
            case vega::client::MsgType::Foreach: {
                if (!session) throw std::runtime_error("session not initialized");
                uint64_t id = vega::read_pod<uint64_t>(body, off);
                auto pickled = vega::client::read_bytes(body, off);
                session->foreach(id, pickled);
                reply = vega::client::encode(vega::client::MsgType::ForeachAck, {});
                break;
            }
            default:
                reply = send_error("unsupported message type");
                break;
            }
        } catch (const py::error_already_set& e) {
            reply = send_error(e.what());
        } catch (const std::exception& e) {
            reply = send_error(e.what());
        }

        conn->send_frame(reply);
    }
}

} // namespace

namespace vega::client {

void serve_client(std::unique_ptr<net::TcpConnection> conn) {
    handle_client(std::move(conn));
}

} // namespace vega::client
