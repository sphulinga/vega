#include <atomic>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
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

// Bumped whenever new message types are added. v2 = server-side operators.
constexpr uint32_t kProtocolVersion = 2;

enum class StageOp : uint8_t {
    Map = 1,
    Filter = 2,
    FlatMap = 3,
    ReduceByKey = 4,
    GroupByKey = 5,
    MapPartitions = 6,  // func(partition_index, iterator) -> iterable
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

static size_t py_distinct_keys(const std::unordered_map<size_t, PyAggBucket>& groups) {
    size_t count = 0;
    for (const auto& [_, bucket] : groups)
        count += bucket.size();
    return count;
}


// ---------------------------------------------------------------------------
// Fast UDF invocation helpers.
//
// User functions are called straight through the CPython C API instead of
// via a Python-level try/except wrapper, which removes one interpreted frame
// per element. A StopIteration escaping user code is converted to a
// RuntimeError (Spark semantics) here in C++.
// ---------------------------------------------------------------------------

[[noreturn]] static void raise_udf_error() {
    if (PyErr_ExceptionMatches(PyExc_StopIteration)) {
        PyObject *type = nullptr, *value = nullptr, *tb = nullptr;
        PyErr_Fetch(&type, &value, &tb);
        PyErr_NormalizeException(&type, &value, &tb);
        PyErr_SetString(PyExc_RuntimeError,
                        "Caught StopIteration thrown from user's code; failing the task");
        PyObject *ntype = nullptr, *nvalue = nullptr, *ntb = nullptr;
        PyErr_Fetch(&ntype, &nvalue, &ntb);
        PyErr_NormalizeException(&ntype, &nvalue, &ntb);
        if (value) {
            PyException_SetCause(nvalue, value);  // steals value
        }
        Py_XDECREF(type);
        Py_XDECREF(tb);
        PyErr_Restore(ntype, nvalue, ntb);
    }
    throw py::error_already_set();
}

static inline py::object udf_call1(const py::object& func, const py::object& arg) {
    PyObject* r = PyObject_CallOneArg(func.ptr(), arg.ptr());
    if (!r) raise_udf_error();
    return py::reinterpret_steal<py::object>(r);
}

static inline py::object udf_call2(const py::object& func,
                                   const py::object& a,
                                   const py::object& b) {
    PyObject* args[2] = {a.ptr(), b.ptr()};
    PyObject* r = PyObject_Vectorcall(func.ptr(), args, 2, nullptr);
    if (!r) raise_udf_error();
    return py::reinterpret_steal<py::object>(r);
}

static inline bool udf_truthy(const py::object& obj) {
    const int t = PyObject_IsTrue(obj.ptr());
    if (t < 0) raise_udf_error();
    return t == 1;
}

// Iterate any Python iterable, invoking `fn` with each item (new reference).
template<typename Fn>
static void udf_iterate(const py::object& iterable, Fn&& fn) {
    PyObject* it = PyObject_GetIter(iterable.ptr());
    if (!it) raise_udf_error();
    py::object iter = py::reinterpret_steal<py::object>(it);
    while (PyObject* item = PyIter_Next(iter.ptr()))
        fn(py::reinterpret_steal<py::object>(item));
    if (PyErr_Occurred()) raise_udf_error();
}

// Build a Python list from a partition vector (moves the references).
static py::list py_list_from(std::vector<py::object>&& items) {
    PyObject* lst = PyList_New(static_cast<Py_ssize_t>(items.size()));
    if (!lst) throw py::error_already_set();
    for (size_t i = 0; i < items.size(); ++i)
        PyList_SET_ITEM(lst, static_cast<Py_ssize_t>(i), items[i].release().ptr());
    items.clear();
    return py::reinterpret_steal<py::list>(lst);
}

static inline std::vector<py::object>& py_vec_ref(std::any& a) {
    auto* p = std::any_cast<std::vector<py::object>>(&a);
    if (!p) throw std::runtime_error("expected a Python partition");
    return *p;
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
            output.push_back(udf_call1(func_, elem));
        return output;
    }

    size_t compute_partition_count(size_t pid) const override {
        py::gil_scoped_acquire acquire;
        auto input = py_parent_partition(parent_, pid);
        for (const auto& elem : input)
            (void)udf_call1(func_, elem);
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
            if (udf_truthy(udf_call1(pred_, elem)))
                output.push_back(std::move(elem));
        }
        return output;
    }

    size_t compute_partition_count(size_t pid) const override {
        py::gil_scoped_acquire acquire;
        auto input = py_parent_partition(parent_, pid);
        size_t out = 0;
        for (const auto& elem : input) {
            if (udf_truthy(udf_call1(pred_, elem))) ++out;
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
            py::object mapped = udf_call1(func_, elem);
            udf_iterate(mapped, [&](py::object item) {
                output.push_back(std::move(item));
            });
        }
        return output;
    }

    size_t compute_partition_count(size_t pid) const override {
        py::gil_scoped_acquire acquire;
        auto input = py_parent_partition(parent_, pid);
        size_t out = 0;
        for (auto& elem : input) {
            py::object mapped = udf_call1(func_, elem);
            const Py_ssize_t n = PyObject_Length(mapped.ptr());
            if (n >= 0) {
                out += static_cast<size_t>(n);
            } else {
                PyErr_Clear();
                udf_iterate(mapped, [&](py::object) { ++out; });
            }
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

// Shared machinery for Python key/value shuffles. Map outputs are bucketed
// by hash(key) % N and combined *while bucketing* (map-side combine), so the
// reduce side only emits pre-aggregated groups. The materialized buckets are
// kept so later actions on the same RDD skip the map stage (shuffle reuse).
using PyGroups = std::unordered_map<size_t, PyAggBucket>;

static py::tuple py_kv(const py::object& item, const char* op) {
    if (!PyTuple_Check(item.ptr()) || PyTuple_GET_SIZE(item.ptr()) < 2) {
        if (PyList_Check(item.ptr()) && PyList_GET_SIZE(item.ptr()) == 2)
            return py::tuple(py::reinterpret_borrow<py::list>(item));
        throw std::runtime_error(std::string(op) + ": map output is not a (key, value) tuple");
    }
    return py::reinterpret_borrow<py::tuple>(item);
}

class PyKeyedShuffleRDDBase : public vega::RDDBase {
public:
    PyKeyedShuffleRDDBase(std::shared_ptr<vega::RDDBase> parent, size_t num_partitions)
        : parent_(std::move(parent)), num_partitions_(std::max<size_t>(1, num_partitions)) {
        groups_.resize(num_partitions_);
    }

    size_t num_partitions() const override { return num_partitions_; }
    bool is_narrow() const override { return false; }
    bool uses_python_runtime() const override { return true; }
    bool shuffle_ready() const override { return shuffle_ready_; }

    std::vector<std::shared_ptr<vega::RDDBase>> dependencies() const override {
        return {parent_};
    }

    void prepare_shuffle(const std::vector<std::any>& parent_results) override {
        py::gil_scoped_acquire acquire;
        for (auto& g : groups_)
            g.clear();
        shuffle_ready_ = false;
        for (const auto& pr : parent_results) {
            if (!pr.has_value())
                throw std::runtime_error(std::string(op_name()) + ": empty map partition result");
            const auto* data = std::any_cast<std::vector<py::object>>(&pr);
            if (!data)
                throw std::runtime_error(std::string(op_name()) + ": unexpected map output type");
            for (const auto& item : *data) {
                py::tuple t = py_kv(item, op_name());
                py::object k = t[0];
                py::object v = t[1];
                const size_t h = py_hash(k);
                add(groups_[h % num_partitions_][h], std::move(k), std::move(v));
            }
        }
        shuffle_ready_ = true;
    }

    std::any compute_partition(size_t pid) const override {
        py::gil_scoped_acquire acquire;
        check_ready();
        std::vector<py::object> result;
        result.reserve(py_distinct_keys(groups_[pid]));
        for (const auto& [_, bucket] : groups_[pid]) {
            for (const auto& [k, v] : bucket)
                result.push_back(emit(k, v));
        }
        return result;
    }

    size_t compute_partition_count(size_t pid) const override {
        check_ready();
        return py_distinct_keys(groups_[pid]);
    }

protected:
    virtual const char* op_name() const = 0;
    virtual void add(PyAggBucket& bucket, py::object key, py::object value) = 0;
    virtual py::object emit(const py::object& key, const py::object& value) const {
        return py::make_tuple(key, value);
    }

private:
    void check_ready() const {
        if (!shuffle_ready_)
            throw std::runtime_error(std::string(op_name()) + ": shuffle data not prepared");
    }

    std::shared_ptr<vega::RDDBase> parent_;
    size_t num_partitions_;
    bool shuffle_ready_ = false;
    std::vector<PyGroups> groups_;
};

// reduceByKey for Python (k, v) tuples stored as py::object.
class PyReduceByKeyRDDImpl : public PyKeyedShuffleRDDBase {
public:
    PyReduceByKeyRDDImpl(std::shared_ptr<vega::RDDBase> parent,
                         py::object combine,
                         size_t num_partitions)
        : PyKeyedShuffleRDDBase(std::move(parent), num_partitions)
        , combine_(std::move(combine)) {}

protected:
    const char* op_name() const override { return "reduceByKey"; }
    void add(PyAggBucket& bucket, py::object key, py::object value) override {
        if (py::object* acc = py_find_agg(bucket, key))
            *acc = udf_call2(combine_, *acc, value);
        else
            bucket.emplace_back(std::move(key), std::move(value));
    }

private:
    py::object combine_;
};

class PyGroupByKeyRDDImpl : public PyKeyedShuffleRDDBase {
public:
    PyGroupByKeyRDDImpl(std::shared_ptr<vega::RDDBase> parent, size_t num_partitions)
        : PyKeyedShuffleRDDBase(std::move(parent), num_partitions) {}

protected:
    const char* op_name() const override { return "groupByKey"; }
    void add(PyAggBucket& bucket, py::object key, py::object value) override {
        if (py::object* acc = py_find_agg(bucket, key)) {
            if (PyList_Append(acc->ptr(), value.ptr()) != 0)
                throw py::error_already_set();
        } else {
            py::list lst;
            lst.append(std::move(value));
            bucket.emplace_back(std::move(key), std::move(lst));
        }
    }
    // Hand out a copy so callers mutating the group cannot corrupt the
    // reusable shuffle output.
    py::object emit(const py::object& key, const py::object& value) const override {
        PyObject* copy = PyList_GetSlice(value.ptr(), 0, PyList_GET_SIZE(value.ptr()));
        if (!copy) throw py::error_already_set();
        return py::make_tuple(key, py::reinterpret_steal<py::object>(copy));
    }
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
    bool shuffle_ready() const override { return inner_->shuffle_ready(); }
    const vega::RDDBase* identity() const override { return inner_->identity(); }

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
    bool shuffle_ready() const override { return inner_->shuffle_ready(); }
    const vega::RDDBase* identity() const override { return inner_->identity(); }

private:
    std::shared_ptr<vega::RDDBase> inner_;
};


// ---------------------------------------------------------------------------
// Server-side Python RDD operators. These keep data inside the server instead
// of round-tripping partitions through the client.
// ---------------------------------------------------------------------------

// mapPartitionsWithIndex: func(index, iterator) -> iterable.
class PyMapPartitionsRDDImpl : public vega::RDDBase {
public:
    PyMapPartitionsRDDImpl(std::shared_ptr<vega::RDDBase> parent, py::object func)
        : parent_(std::move(parent)), func_(std::move(func)) {
        set_context(parent_->context());
    }

    bool uses_python_runtime() const override { return true; }
    size_t num_partitions() const override { return parent_->num_partitions(); }
    std::vector<std::shared_ptr<vega::RDDBase>> dependencies() const override {
        return {parent_};
    }

    std::any compute_partition(size_t pid) const override {
        py::gil_scoped_acquire acquire;
        std::vector<py::object> output;
        run(pid, [&](py::object item) { output.push_back(std::move(item)); });
        return output;
    }

    size_t compute_partition_count(size_t pid) const override {
        py::gil_scoped_acquire acquire;
        size_t n = 0;
        run(pid, [&](py::object) { ++n; });
        return n;
    }

private:
    template<typename Fn>
    void run(size_t pid, Fn&& fn) const {
        auto input = py_parent_partition(parent_, pid);
        py::list lst = py_list_from(std::move(input));
        py::object it = py::reinterpret_steal<py::object>(PyObject_GetIter(lst.ptr()));
        if (!it) throw py::error_already_set();
        py::object result = udf_call2(func_, py::int_(pid), it);
        if (result.is_none()) return;
        udf_iterate(result, std::forward<Fn>(fn));
    }

    std::shared_ptr<vega::RDDBase> parent_;
    py::object func_;
};

// union: concatenates the partitions of several parents (narrow).
class PyUnionRDDImpl : public vega::RDDBase {
public:
    explicit PyUnionRDDImpl(std::vector<std::shared_ptr<vega::RDDBase>> parents)
        : parents_(std::move(parents)) {
        for (size_t i = 0; i < parents_.size(); ++i)
            for (size_t p = 0; p < parents_[i]->num_partitions(); ++p)
                index_.emplace_back(i, p);
    }

    bool uses_python_runtime() const override { return true; }
    size_t num_partitions() const override { return index_.size(); }
    std::vector<std::shared_ptr<vega::RDDBase>> dependencies() const override {
        return parents_;
    }

    std::any compute_partition(size_t pid) const override {
        const auto& [i, p] = index_.at(pid);
        return parents_[i]->compute_partition(p);
    }
    size_t compute_partition_count(size_t pid) const override {
        const auto& [i, p] = index_.at(pid);
        return parents_[i]->compute_partition_count(p);
    }

private:
    std::vector<std::shared_ptr<vega::RDDBase>> parents_;
    std::vector<std::pair<size_t, size_t>> index_;
};

// coalesce without shuffle: output partition i concatenates a group of
// parent partitions (narrow).
class PyCoalescedRDDImpl : public vega::RDDBase {
public:
    PyCoalescedRDDImpl(std::shared_ptr<vega::RDDBase> parent,
                       std::vector<std::vector<size_t>> groups)
        : parent_(std::move(parent)), groups_(std::move(groups)) {}

    bool uses_python_runtime() const override { return true; }
    size_t num_partitions() const override { return groups_.size(); }
    std::vector<std::shared_ptr<vega::RDDBase>> dependencies() const override {
        return {parent_};
    }

    std::any compute_partition(size_t pid) const override {
        py::gil_scoped_acquire acquire;
        std::vector<py::object> out;
        for (size_t p : groups_.at(pid)) {
            auto part = py_parent_partition(parent_, p);
            out.insert(out.end(), std::make_move_iterator(part.begin()),
                       std::make_move_iterator(part.end()));
        }
        return out;
    }
    size_t compute_partition_count(size_t pid) const override {
        size_t n = 0;
        for (size_t p : groups_.at(pid))
            n += parent_->compute_partition_count(p);
        return n;
    }

private:
    std::shared_ptr<vega::RDDBase> parent_;
    std::vector<std::vector<size_t>> groups_;
};

// cartesian: partition (i, j) = left[i] x right[j].
class PyCartesianRDDImpl : public vega::RDDBase {
public:
    PyCartesianRDDImpl(std::shared_ptr<vega::RDDBase> left,
                       std::shared_ptr<vega::RDDBase> right)
        : left_(std::move(left)), right_(std::move(right)) {}

    bool uses_python_runtime() const override { return true; }
    size_t num_partitions() const override {
        return left_->num_partitions() * right_->num_partitions();
    }
    std::vector<std::shared_ptr<vega::RDDBase>> dependencies() const override {
        return {left_, right_};
    }

    std::any compute_partition(size_t pid) const override {
        py::gil_scoped_acquire acquire;
        const size_t m = right_->num_partitions();
        auto a = py_parent_partition(left_, pid / m);
        auto b = py_parent_partition(right_, pid % m);
        std::vector<py::object> out;
        out.reserve(a.size() * b.size());
        for (const auto& x : a)
            for (const auto& y : b)
                out.push_back(py::make_tuple(x, y));
        return out;
    }
    size_t compute_partition_count(size_t pid) const override {
        const size_t m = right_->num_partitions();
        return left_->compute_partition_count(pid / m)
               * right_->compute_partition_count(pid % m);
    }

private:
    std::shared_ptr<vega::RDDBase> left_, right_;
};

// zipPartitions: func(iter_left, iter_right) -> iterable, per partition.
class PyZipPartitionsRDDImpl : public vega::RDDBase {
public:
    PyZipPartitionsRDDImpl(std::shared_ptr<vega::RDDBase> left,
                           std::shared_ptr<vega::RDDBase> right,
                           py::object func)
        : left_(std::move(left)), right_(std::move(right)), func_(std::move(func)) {
        if (left_->num_partitions() != right_->num_partitions())
            throw std::runtime_error(
                "Can only zip RDDs with the same number of partitions");
    }

    bool uses_python_runtime() const override { return true; }
    size_t num_partitions() const override { return left_->num_partitions(); }
    std::vector<std::shared_ptr<vega::RDDBase>> dependencies() const override {
        return {left_, right_};
    }

    std::any compute_partition(size_t pid) const override {
        py::gil_scoped_acquire acquire;
        std::vector<py::object> out;
        run(pid, [&](py::object item) { out.push_back(std::move(item)); });
        return out;
    }
    size_t compute_partition_count(size_t pid) const override {
        py::gil_scoped_acquire acquire;
        size_t n = 0;
        run(pid, [&](py::object) { ++n; });
        return n;
    }

private:
    template<typename Fn>
    void run(size_t pid, Fn&& fn) const {
        py::list a = py_list_from(py_parent_partition(left_, pid));
        py::list b = py_list_from(py_parent_partition(right_, pid));
        py::object result = udf_call2(func_, a.attr("__iter__")(), b.attr("__iter__")());
        udf_iterate(result, std::forward<Fn>(fn));
    }

    std::shared_ptr<vega::RDDBase> left_, right_;
    py::object func_;
};

// partitionBy / repartition shuffle.
//   mode 0: bucket = partitioner(key) % N for (key, value) items
//           (partitioner None => hash(key))
//   mode 1: round-robin by element position (repartition of any data)
class PyPartitionByRDDImpl : public vega::RDDBase {
public:
    PyPartitionByRDDImpl(std::shared_ptr<vega::RDDBase> parent,
                         size_t num_partitions,
                         py::object partitioner,
                         uint8_t mode)
        : parent_(std::move(parent))
        , num_partitions_(std::max<size_t>(1, num_partitions))
        , partitioner_(std::move(partitioner))
        , mode_(mode) {
        buckets_.resize(num_partitions_);
    }

    size_t num_partitions() const override { return num_partitions_; }
    bool is_narrow() const override { return false; }
    bool uses_python_runtime() const override { return true; }
    bool shuffle_ready() const override { return shuffle_ready_; }
    std::vector<std::shared_ptr<vega::RDDBase>> dependencies() const override {
        return {parent_};
    }

    void prepare_shuffle(const std::vector<std::any>& parent_results) override {
        py::gil_scoped_acquire acquire;
        for (auto& b : buckets_) b.clear();
        shuffle_ready_ = false;
        const int64_t n = static_cast<int64_t>(num_partitions_);
        for (size_t map_pid = 0; map_pid < parent_results.size(); ++map_pid) {
            const auto* data = std::any_cast<std::vector<py::object>>(&parent_results[map_pid]);
            if (!data)
                throw std::runtime_error("partitionBy: unexpected map output type");
            size_t pos = map_pid;  // round-robin start differs per map partition
            for (const auto& item : *data) {
                size_t bucket = 0;
                if (mode_ == 1) {
                    bucket = pos++ % num_partitions_;
                } else {
                    py::object key = item;
                    if (PyTuple_Check(item.ptr()) && PyTuple_GET_SIZE(item.ptr()) >= 1)
                        key = py::reinterpret_borrow<py::object>(PyTuple_GET_ITEM(item.ptr(), 0));
                    int64_t h;
                    if (partitioner_.is_none()) {
                        h = static_cast<int64_t>(py_hash(key) & 0x7fffffffffffffffULL);
                    } else {
                        py::object r = udf_call1(partitioner_, key);
                        h = PyLong_AsLongLong(r.ptr());
                        if (h == -1 && PyErr_Occurred()) raise_udf_error();
                    }
                    bucket = static_cast<size_t>(((h % n) + n) % n);
                }
                buckets_[bucket].push_back(item);
            }
        }
        shuffle_ready_ = true;
    }

    std::any compute_partition(size_t pid) const override {
        py::gil_scoped_acquire acquire;
        if (!shuffle_ready_)
            throw std::runtime_error("partitionBy: shuffle data not prepared");
        return buckets_.at(pid);
    }
    size_t compute_partition_count(size_t pid) const override {
        if (!shuffle_ready_)
            throw std::runtime_error("partitionBy: shuffle data not prepared");
        return buckets_.at(pid).size();
    }

private:
    std::shared_ptr<vega::RDDBase> parent_;
    size_t num_partitions_;
    py::object partitioner_;
    uint8_t mode_;
    bool shuffle_ready_ = false;
    std::vector<std::vector<py::object>> buckets_;
};

// persist()/cache(): memoizes computed partitions in server memory. Lazy --
// partitions are stored the first time an action computes them. Once every
// partition is cached the lineage is cut (no dependencies), so upstream
// stages are skipped entirely.
class CachedRDDImpl : public vega::RDDBase {
public:
    using Sizer = std::function<size_t(const std::any&)>;

    CachedRDDImpl(std::shared_ptr<vega::RDDBase> parent, Sizer sizer)
        : parent_(std::move(parent)), sizer_(std::move(sizer)),
          python_(parent_->uses_python_runtime()) {
        cache_.resize(parent_->num_partitions());
        set_context(parent_->context());
    }

    ~CachedRDDImpl() override {
        if (python_ && Py_IsInitialized()) {
            py::gil_scoped_acquire acquire;
            cache_.clear();
        }
    }

    bool uses_python_runtime() const override { return python_; }
    size_t num_partitions() const override { return cache_.size(); }
    bool is_narrow() const override { return true; }

    std::vector<std::shared_ptr<vega::RDDBase>> dependencies() const override {
        std::lock_guard<std::mutex> lock(mu_);
        if (cached_count_ == cache_.size()) return {};
        return {parent_};
    }

    std::any compute_partition(size_t pid) const override {
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (cache_.at(pid).has_value()) return copy(*cache_[pid]);
        }
        std::any value = parent_->compute_partition(pid);
        std::lock_guard<std::mutex> lock(mu_);
        if (!cache_[pid].has_value()) {
            cache_[pid] = copy(value);
            ++cached_count_;
        }
        return value;
    }

    size_t compute_partition_count(size_t pid) const override {
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (cache_.at(pid).has_value()) return sizer_(*cache_[pid]);
        }
        if (python_) {
            py::gil_scoped_acquire acquire;
            return sizer_(compute_partition(pid));
        }
        return sizer_(compute_partition(pid));
    }

    size_t cached_partitions() const {
        std::lock_guard<std::mutex> lock(mu_);
        return cached_count_;
    }

private:
    std::any copy(const std::any& v) const {
        if (python_) {
            py::gil_scoped_acquire acquire;
            return std::any(v);
        }
        return std::any(v);
    }

    std::shared_ptr<vega::RDDBase> parent_;
    Sizer sizer_;
    bool python_;
    mutable std::mutex mu_;
    mutable std::vector<std::optional<std::any>> cache_;
    mutable size_t cached_count_ = 0;
};

template<typename T>
static size_t any_vec_size(const std::any& a) {
    const auto* v = std::any_cast<std::vector<T>>(&a);
    return v ? v->size() : 0;
}

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

static PyObject* cached_attr(const char* module, const char* attr);

// cloudpickle output is a standard pickle stream, so the C unpickler loads
// both stdlib- and cloudpickle-produced payloads.
static py::object unpickle_bytes(const vega::ByteBuffer& buf) {
    static PyObject* loads = cached_attr("pickle", "loads");
    PyObject* b = PyBytes_FromStringAndSize(reinterpret_cast<const char*>(buf.data()),
                                            static_cast<Py_ssize_t>(buf.size()));
    if (!b) throw py::error_already_set();
    PyObject* r = PyObject_CallOneArg(loads, b);
    Py_DECREF(b);
    if (!r) throw py::error_already_set();
    return py::reinterpret_steal<py::object>(r);
}

static py::bytes pickle_dumps(const py::object& obj) {
    try {
        return py::module_::import("cloudpickle").attr("dumps")(obj);
    } catch (const py::error_already_set&) {
        PyErr_Clear();
        return py::module_::import("pickle").attr("dumps")(obj);
    }
}


// Data (not code) is pickled with the C pickler first -- considerably faster
// than cloudpickle for plain lists/tuples -- falling back to cloudpickle for
// objects the stdlib pickler cannot handle (lambdas, local classes, ...).
static PyObject* cached_attr(const char* module, const char* attr) {
    PyObject* mod = PyImport_ImportModule(module);
    if (!mod) throw py::error_already_set();
    PyObject* fn = PyObject_GetAttrString(mod, attr);
    Py_DECREF(mod);
    if (!fn) throw py::error_already_set();
    return fn;  // intentionally leaked: lives for the interpreter lifetime
}

static py::bytes pickle_data(const py::object& obj) {
    static PyObject* dumps = cached_attr("pickle", "dumps");
    static PyObject* protocol = PyLong_FromLong(5);
    PyObject* args[2] = {obj.ptr(), protocol};
    PyObject* r = PyObject_Vectorcall(dumps, args, 2, nullptr);
    if (r) return py::reinterpret_steal<py::bytes>(r);
    PyErr_Clear();
    return pickle_dumps(obj);
}

static vega::ByteBuffer pickled_payload(const py::object& obj) {
    py::bytes pickled = pickle_data(obj);
    char* data = nullptr;
    Py_ssize_t len = 0;
    if (PyBytes_AsStringAndSize(pickled.ptr(), &data, &len) != 0)
        throw py::error_already_set();
    return vega::ByteBuffer(reinterpret_cast<const uint8_t*>(data),
                            reinterpret_cast<const uint8_t*>(data) + len);
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

    explicit ServerSession(size_t num_workers)
        : ctx_(std::make_unique<vega::SparkContext>(num_workers))
    {
        vega::python_gil_runner() = [](const std::function<void()>& fn) {
            py::gil_scoped_acquire acquire;
            fn();
        };
    }

    ~ServerSession() {
        // Stored RDDs hold py::object values: drop them with the GIL held,
        // then stop the actor system (whose threads may need the GIL).
        if (Py_IsInitialized()) {
            py::gil_scoped_acquire acquire;
            rdds_.clear();
        }
        ctx_.reset();  // called without the GIL: actor threads may need it
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
            case StageOp::MapPartitions:
                cur = map_partitions_with_func(cur, stage.func);
                break;
            }
        }
        return cur;
    }

    // ----- v0.6 server-side operators ---------------------------------------

    uint64_t map_partitions_with_func(uint64_t parent, const py::object& func) {
        auto impl = std::make_shared<PyMapPartitionsRDDImpl>(python_parent(parent), func);
        impl->set_context(ctx_.get());
        return store_python(std::move(impl));
    }

    uint64_t union_rdds(const std::vector<uint64_t>& ids) {
        py::gil_scoped_acquire acquire;
        std::vector<std::shared_ptr<vega::RDDBase>> parents;
        parents.reserve(ids.size());
        for (uint64_t id : ids)
            parents.push_back(python_parent(id));
        auto impl = std::make_shared<PyUnionRDDImpl>(std::move(parents));
        return store_python(std::move(impl));
    }

    uint64_t partition_by(uint64_t parent, size_t parts, uint8_t mode,
                          const vega::ByteBuffer& func_bytes) {
        py::gil_scoped_acquire acquire;
        py::object func = func_bytes.empty() ? py::none() : unpickle(func_bytes);
        if (parts == 0)
            parts = load_entry(parent).impl->num_partitions();
        auto impl = std::make_shared<PyPartitionByRDDImpl>(
            python_parent(parent), parts, std::move(func), mode);
        return store_python(std::move(impl));
    }

    uint64_t coalesce(uint64_t parent, std::vector<std::vector<size_t>> groups) {
        py::gil_scoped_acquire acquire;
        auto base = python_parent(parent);
        for (const auto& g : groups)
            for (size_t p : g)
                if (p >= base->num_partitions())
                    throw std::runtime_error("coalesce: partition index out of range");
        auto impl = std::make_shared<PyCoalescedRDDImpl>(std::move(base), std::move(groups));
        return store_python(std::move(impl));
    }

    uint64_t cartesian(uint64_t left, uint64_t right) {
        py::gil_scoped_acquire acquire;
        auto impl = std::make_shared<PyCartesianRDDImpl>(python_parent(left),
                                                         python_parent(right));
        return store_python(std::move(impl));
    }

    uint64_t zip_partitions(uint64_t left, uint64_t right, const vega::ByteBuffer& func_bytes) {
        py::gil_scoped_acquire acquire;
        auto impl = std::make_shared<PyZipPartitionsRDDImpl>(
            python_parent(left), python_parent(right), unpickle(func_bytes));
        return store_python(std::move(impl));
    }

    uint64_t persist(uint64_t id) {
        auto& entry = load_entry(id);
        CachedRDDImpl::Sizer sizer;
        switch (entry.kind) {
        case RddStorageKind::Python: sizer = any_vec_size<py::object>; break;
        case RddStorageKind::Int64: sizer = any_vec_size<int64_t>; break;
        case RddStorageKind::Float64: sizer = any_vec_size<double>; break;
        case RddStorageKind::String: sizer = any_vec_size<std::string>; break;
        case RddStorageKind::PairInt64:
            sizer = any_vec_size<std::pair<int64_t, int64_t>>; break;
        case RddStorageKind::PairDoubleInt64:
            sizer = any_vec_size<std::pair<double, int64_t>>; break;
        case RddStorageKind::PairStringInt64:
            sizer = any_vec_size<std::pair<std::string, int64_t>>; break;
        case RddStorageKind::GroupedPairInt64:
            sizer = any_vec_size<std::pair<int64_t, std::vector<int64_t>>>; break;
        }
        const auto kind = entry.kind;
        auto impl = std::make_shared<CachedRDDImpl>(entry.impl, std::move(sizer));
        return store(kind, std::move(impl));
    }

    void release(const std::vector<uint64_t>& ids) {
        py::gil_scoped_acquire acquire;
        for (uint64_t id : ids)
            rdds_.erase(id);
    }

    // Run per-partition jobs over a subset of partitions.
    //   mode 0: list of per-partition element lists
    //   mode 1: list of per-partition counts
    py::list run_job(uint64_t id, std::vector<size_t> pids, uint8_t mode) {
        auto& entry = load_entry(id);
        const size_t total = entry.impl->num_partitions();
        for (size_t pid : pids)
            if (pid >= total)
                throw std::runtime_error("runJob: partition index out of range");
        py::gil_scoped_acquire acquire;
        py::list out;
        if (pids.empty()) return out;
        if (mode == 1) {
            std::vector<size_t> counts;
            {
                py::gil_scoped_release release;
                counts = ctx_->execute_count_partitions(entry.impl, std::move(pids));
            }
            for (size_t c : counts) out.append(c);
            return out;
        }
        std::vector<std::any> results;
        {
            py::gil_scoped_release release;
            results = ctx_->execute_partitions(entry.impl, std::move(pids));
        }
        for (auto& part : results)
            out.append(partition_to_list(entry.kind, part));
        results.clear();
        return out;
    }

    // Exact numeric statistics computed natively for int64/float64 RDDs.
    // Returns None for other storage kinds so the client can fall back.
    py::object native_stats(uint64_t id) {
        auto& entry = load_entry(id);
        py::gil_scoped_acquire acquire;
        if (entry.kind != RddStorageKind::Int64 && entry.kind != RddStorageKind::Float64)
            return py::none();
        std::vector<std::any> results;
        {
            py::gil_scoped_release release;
            results = ctx_->execute(entry.impl);
        }
        uint64_t n = 0;
        double mean = 0.0, m2 = 0.0;
        auto welford = [&](double x) {
            ++n;
            const double delta = x - mean;
            mean += delta / static_cast<double>(n);
            m2 += delta * (x - mean);
        };
        if (entry.kind == RddStorageKind::Int64) {
            __int128 sum = 0;
            int64_t mn = 0, mx = 0;
            for (auto& r : results) {
                const auto& vec = *std::any_cast<std::vector<int64_t>>(&r);
                for (int64_t v : vec) {
                    if (n == 0) { mn = mx = v; }
                    mn = std::min(mn, v);
                    mx = std::max(mx, v);
                    sum += v;
                    welford(static_cast<double>(v));
                }
            }
            // __int128 -> Python int via decimal string.
            const bool neg = sum < 0;
            unsigned __int128 mag = neg ? static_cast<unsigned __int128>(-sum)
                                        : static_cast<unsigned __int128>(sum);
            std::string digits;
            do { digits.push_back(static_cast<char>('0' + static_cast<int>(mag % 10))); mag /= 10; } while (mag);
            if (neg) digits.push_back('-');
            std::reverse(digits.begin(), digits.end());
            py::object py_sum = py::reinterpret_steal<py::object>(
                PyLong_FromString(digits.c_str(), nullptr, 10));
            if (n == 0) return py::make_tuple(0, 0, py::none(), py::none(), 0.0);
            return py::make_tuple(n, py_sum, mn, mx, m2);
        }
        double sum = 0.0;
        double mn = 0.0, mx = 0.0;
        for (auto& r : results) {
            const auto& vec = *std::any_cast<std::vector<double>>(&r);
            for (double v : vec) {
                if (n == 0) { mn = mx = v; }
                mn = std::min(mn, v);
                mx = std::max(mx, v);
                sum += v;
                welford(v);
            }
        }
        if (n == 0) return py::make_tuple(0, 0.0, py::none(), py::none(), 0.0);
        return py::make_tuple(n, sum, mn, mx, m2);
    }

    py::tuple describe(uint64_t id) {
        auto& entry = load_entry(id);
        py::gil_scoped_acquire acquire;
        static const char* names[] = {"python", "int64", "float64", "str",
                                      "pair<int64,int64>", "pair<float64,int64>",
                                      "pair<str,int64>", "pair<int64,list<int64>>"};
        long cached = -1;
        if (auto* c = dynamic_cast<CachedRDDImpl*>(entry.impl.get()))
            cached = static_cast<long>(c->cached_partitions());
        return py::make_tuple(names[static_cast<int>(entry.kind)],
                              entry.impl->num_partitions(), cached);
    }

    // Convert one computed partition of the given storage kind to a list.
    // Caller must hold the GIL.
    static py::list partition_to_list(RddStorageKind kind, std::any& part) {
        py::list output;
        switch (kind) {
        case RddStorageKind::Python:
            return py_list_from(std::move(py_vec_ref(part)));
        case RddStorageKind::Int64:
            for (int64_t v : *std::any_cast<std::vector<int64_t>>(&part)) output.append(v);
            break;
        case RddStorageKind::Float64:
            for (double v : *std::any_cast<std::vector<double>>(&part)) output.append(v);
            break;
        case RddStorageKind::String:
            for (const auto& v : *std::any_cast<std::vector<std::string>>(&part))
                output.append(py::str(v));
            break;
        case RddStorageKind::PairInt64:
            for (auto& [k, v] : *std::any_cast<std::vector<std::pair<int64_t, int64_t>>>(&part))
                output.append(py::make_tuple(k, v));
            break;
        case RddStorageKind::PairDoubleInt64:
            for (auto& [k, v] : *std::any_cast<std::vector<std::pair<double, int64_t>>>(&part))
                output.append(py::make_tuple(k, v));
            break;
        case RddStorageKind::PairStringInt64:
            for (auto& [k, v] : *std::any_cast<std::vector<std::pair<std::string, int64_t>>>(&part))
                output.append(py::make_tuple(py::str(k), v));
            break;
        case RddStorageKind::GroupedPairInt64:
            for (auto& [k, vs] : *std::any_cast<
                     std::vector<std::pair<int64_t, std::vector<int64_t>>>>(&part)) {
                py::list lst;
                for (int64_t v : vs) lst.append(v);
                output.append(py::make_tuple(k, lst));
            }
            break;
        }
        return output;
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
            py::bytes pickled = pickle_data(data);
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
            py::bytes out = pickle_data(value);
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
        py::gil_scoped_acquire acquire;
        py::list parts = run_job(rdd_id, {pid}, 0);
        return py::reinterpret_borrow<py::list>(parts[0]);
    }

    py::list collect_partitions(uint64_t rdd_id) {
        const size_t parts = load_entry(rdd_id).impl->num_partitions();
        std::vector<size_t> pids(parts);
        for (size_t i = 0; i < parts; ++i) pids[i] = i;
        return run_job(rdd_id, std::move(pids), 0);
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
        auto impl = std::make_shared<PyMappedRDDImpl>(python_parent(parent), func);
        impl->set_context(ctx_.get());
        return store_python(std::move(impl));
    }

    uint64_t filter_with_func(uint64_t parent, const py::object& func) {
        auto impl = std::make_shared<PyFilteredRDDImpl>(python_parent(parent), func);
        impl->set_context(ctx_.get());
        return store_python(std::move(impl));
    }

    uint64_t flat_map_with_func(uint64_t parent, const py::object& func) {
        auto impl = std::make_shared<PyFlatMappedRDDImpl>(python_parent(parent), func);
        impl->set_context(ctx_.get());
        return store_python(std::move(impl));
    }

    uint64_t reduce_by_key_with_func(uint64_t parent, const py::object& func,
                                     size_t num_partitions) {
        auto impl = std::make_shared<PyReduceByKeyRDDImpl>(
            python_parent(parent), func, num_partitions);
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

static vega::ByteBuffer id_ack(const ServerSession& session,
                               vega::client::MsgType type, uint64_t id) {
    vega::ByteBuffer ack;
    vega::write_pod<uint64_t>(ack, id);
    vega::write_pod<uint32_t>(ack, static_cast<uint32_t>(session.num_partitions(id)));
    return vega::client::encode(type, ack);
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
                // Protocol revision (older clients ignore trailing bytes).
                vega::write_pod<uint32_t>(ack, kProtocolVersion);
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
                py::bytes pickled = pickle_data(data);
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
                py::bytes pickled = pickle_data(data);
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
                py::bytes pickled = pickle_data(data);
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
                py::bytes out = pickle_data(value);
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
            case vega::client::MsgType::ApplyStages: {
                if (!session) throw std::runtime_error("session not initialized");
                uint64_t base = vega::read_pod<uint64_t>(body, off);
                auto stages = parse_stages(body, off);
                uint64_t id = session->apply_stages_id(base, stages);
                {
                    py::gil_scoped_acquire acquire;
                    stages.clear();
                }
                reply = id_ack(*session, vega::client::MsgType::ApplyStagesAck, id);
                break;
            }
            case vega::client::MsgType::Union: {
                if (!session) throw std::runtime_error("session not initialized");
                const uint32_t n = vega::read_pod<uint32_t>(body, off);
                std::vector<uint64_t> ids(n);
                for (auto& id : ids) id = vega::read_pod<uint64_t>(body, off);
                reply = id_ack(*session, vega::client::MsgType::UnionAck,
                               session->union_rdds(ids));
                break;
            }
            case vega::client::MsgType::PartitionBy: {
                if (!session) throw std::runtime_error("session not initialized");
                uint64_t parent = vega::read_pod<uint64_t>(body, off);
                uint32_t parts = vega::read_pod<uint32_t>(body, off);
                uint8_t mode = vega::read_pod<uint8_t>(body, off);
                auto func = vega::client::read_bytes(body, off);
                reply = id_ack(*session, vega::client::MsgType::PartitionByAck,
                               session->partition_by(parent, parts, mode, func));
                break;
            }
            case vega::client::MsgType::Coalesce: {
                if (!session) throw std::runtime_error("session not initialized");
                uint64_t parent = vega::read_pod<uint64_t>(body, off);
                const uint32_t ngroups = vega::read_pod<uint32_t>(body, off);
                std::vector<std::vector<size_t>> groups(ngroups);
                for (auto& g : groups) {
                    const uint32_t k = vega::read_pod<uint32_t>(body, off);
                    g.resize(k);
                    for (auto& p : g) p = vega::read_pod<uint32_t>(body, off);
                }
                reply = id_ack(*session, vega::client::MsgType::CoalesceAck,
                               session->coalesce(parent, std::move(groups)));
                break;
            }
            case vega::client::MsgType::Cartesian: {
                if (!session) throw std::runtime_error("session not initialized");
                uint64_t left = vega::read_pod<uint64_t>(body, off);
                uint64_t right = vega::read_pod<uint64_t>(body, off);
                reply = id_ack(*session, vega::client::MsgType::CartesianAck,
                               session->cartesian(left, right));
                break;
            }
            case vega::client::MsgType::ZipPartitions: {
                if (!session) throw std::runtime_error("session not initialized");
                uint64_t left = vega::read_pod<uint64_t>(body, off);
                uint64_t right = vega::read_pod<uint64_t>(body, off);
                auto func = vega::client::read_bytes(body, off);
                reply = id_ack(*session, vega::client::MsgType::ZipPartitionsAck,
                               session->zip_partitions(left, right, func));
                break;
            }
            case vega::client::MsgType::Persist: {
                if (!session) throw std::runtime_error("session not initialized");
                uint64_t id = vega::read_pod<uint64_t>(body, off);
                reply = id_ack(*session, vega::client::MsgType::PersistAck,
                               session->persist(id));
                break;
            }
            case vega::client::MsgType::Release: {
                if (!session) throw std::runtime_error("session not initialized");
                const uint32_t n = vega::read_pod<uint32_t>(body, off);
                std::vector<uint64_t> ids(n);
                for (auto& id : ids) id = vega::read_pod<uint64_t>(body, off);
                session->release(ids);
                reply = vega::client::encode(vega::client::MsgType::ReleaseAck, {});
                break;
            }
            case vega::client::MsgType::RunPython: {
                if (!session) throw std::runtime_error("session not initialized");
                auto pickled = vega::client::read_bytes(body, off);
                py::gil_scoped_acquire acquire;
                py::tuple call = py::reinterpret_borrow<py::tuple>(unpickle_bytes(pickled));
                py::object func = call[0];
                py::tuple args = py::reinterpret_borrow<py::tuple>(call[1]);
                py::object result = func(*args);
                reply = vega::client::encode(vega::client::MsgType::RunPythonAck,
                                             pickled_payload(result));
                break;
            }
            case vega::client::MsgType::RunJob: {
                if (!session) throw std::runtime_error("session not initialized");
                uint64_t id = vega::read_pod<uint64_t>(body, off);
                const uint32_t n = vega::read_pod<uint32_t>(body, off);
                std::vector<size_t> pids(n);
                for (auto& p : pids) p = vega::read_pod<uint32_t>(body, off);
                uint8_t mode = vega::read_pod<uint8_t>(body, off);
                py::gil_scoped_acquire acquire;
                py::list data = session->run_job(id, std::move(pids), mode);
                reply = vega::client::encode(vega::client::MsgType::RunJobAck,
                                             pickled_payload(data));
                break;
            }
            case vega::client::MsgType::NativeStats: {
                if (!session) throw std::runtime_error("session not initialized");
                uint64_t id = vega::read_pod<uint64_t>(body, off);
                py::gil_scoped_acquire acquire;
                py::object stats = session->native_stats(id);
                reply = vega::client::encode(vega::client::MsgType::NativeStatsAck,
                                             pickled_payload(stats));
                break;
            }
            case vega::client::MsgType::Describe: {
                if (!session) throw std::runtime_error("session not initialized");
                uint64_t id = vega::read_pod<uint64_t>(body, off);
                py::gil_scoped_acquire acquire;
                py::tuple info = session->describe(id);
                reply = vega::client::encode(vega::client::MsgType::DescribeAck,
                                             pickled_payload(info));
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
