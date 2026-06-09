#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include <vega/vega.hpp>

#include <any>
#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

// ---------------------------------------------------------------------------
// GILRDDImpl -- wraps any RDDBase so that compute_partition acquires the GIL.
//
// Python objects (py::object) that flow through the RDD pipeline require the
// GIL for any reference-count manipulation.  Worker threads in the actor
// system do not hold the GIL by default, so we wrap the outermost RDD with
// this class before handing it to execute().  Because compute_partition is
// recursive, a single acquire at the top covers the entire chain.
// ---------------------------------------------------------------------------

class GILRDDImpl : public vega::RDDBase {
public:
    explicit GILRDDImpl(std::shared_ptr<vega::RDDBase> inner)
        : inner_(std::move(inner))
    {
        set_context(inner_->context());
    }

    size_t num_partitions() const override {
        return inner_->num_partitions();
    }

    std::any compute_partition(size_t pid) const override {
        py::gil_scoped_acquire acquire;
        return inner_->compute_partition(pid);
    }

    std::vector<std::shared_ptr<vega::RDDBase>> dependencies() const override {
        return inner_->dependencies();
    }

    bool is_narrow() const override { return inner_->is_narrow(); }

private:
    std::shared_ptr<vega::RDDBase> inner_;
};

// ---------------------------------------------------------------------------
// Helper: execute an RDD through the actor system with correct GIL handling.
//
// 1. Wrap the RDD in GILRDDImpl so workers acquire the GIL.
// 2. Release the GIL on the calling (Python) thread so workers can obtain it.
// 3. Block on the future.
// 4. Re-acquire the GIL when the future resolves.
// ---------------------------------------------------------------------------

static std::vector<std::any>
execute_with_gil(vega::SparkContext* ctx,
                 std::shared_ptr<vega::RDDBase> rdd)
{
    auto gil_rdd = std::make_shared<GILRDDImpl>(std::move(rdd));
    std::vector<std::any> results;
    {
        py::gil_scoped_release release;
        results = ctx->execute(std::move(gil_rdd));
    }
    return results;
}

// ---------------------------------------------------------------------------
// PyRDD -- Python-visible RDD wrapping RDD<py::object>.
// ---------------------------------------------------------------------------

class PyRDD {
public:
    explicit PyRDD(vega::RDD<py::object> rdd) : rdd_(std::move(rdd)) {}

    PyRDD py_map(py::object func) const {
        py::object f = func;   // copy under GIL (held by caller)
        return PyRDD(rdd_.map(
            [f](const py::object& obj) -> py::object {
                py::gil_scoped_acquire acquire;
                return f(obj);
            }));
    }

    PyRDD py_filter(py::object func) const {
        py::object f = func;
        return PyRDD(rdd_.filter(
            [f](const py::object& obj) -> bool {
                py::gil_scoped_acquire acquire;
                return py::cast<bool>(f(obj));
            }));
    }

    PyRDD py_flat_map(py::object func) const {
        py::object f = func;
        return PyRDD(rdd_.flatMap(
            [f](const py::object& obj) -> std::vector<py::object> {
                py::gil_scoped_acquire acquire;
                py::object result = f(obj);
                std::vector<py::object> vec;
                for (auto item : result)
                    vec.push_back(py::reinterpret_borrow<py::object>(item));
                return vec;
            }));
    }

    // --- actions (trigger execution) ----------------------------------------

    py::list py_collect() const {
        auto results = execute_with_gil(context(), rdd_.impl());

        py::list output;
        for (auto& r : results) {
            auto vec = std::any_cast<std::vector<py::object>>(std::move(r));
            for (auto& obj : vec)
                output.append(std::move(obj));
        }
        return output;
    }

    size_t py_count() const {
        auto results = execute_with_gil(context(), rdd_.impl());

        size_t total = 0;
        for (auto& r : results) {
            auto vec = std::any_cast<std::vector<py::object>>(std::move(r));
            total += vec.size();
        }
        return total;
    }

    py::object py_reduce(py::object func) const {
        auto results = execute_with_gil(context(), rdd_.impl());

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

    void py_foreach(py::object func) const {
        auto results = execute_with_gil(context(), rdd_.impl());

        for (auto& r : results) {
            auto vec = std::any_cast<std::vector<py::object>>(std::move(r));
            for (auto& obj : vec)
                func(obj);
        }
    }

    size_t py_num_partitions() const { return rdd_.num_partitions(); }

private:
    vega::RDD<py::object> rdd_;

    vega::SparkContext* context() const {
        auto* ctx = rdd_.impl()->context();
        if (!ctx) throw std::runtime_error("RDD has no SparkContext");
        return ctx;
    }
};

// ---------------------------------------------------------------------------
// PySparkContext -- Python-visible SparkContext.
// ---------------------------------------------------------------------------

class PySparkContext {
public:
    explicit PySparkContext(size_t num_workers = 0)
        : ctx_(std::make_unique<vega::SparkContext>(num_workers)) {}

    PyRDD parallelize(py::list data, size_t num_partitions) {
        std::vector<py::object> vec;
        vec.reserve(py::len(data));
        for (auto item : data)
            vec.push_back(py::reinterpret_borrow<py::object>(item));

        auto rdd = ctx_->parallelize(vec, num_partitions);
        return PyRDD(std::move(rdd));
    }

    void stop() {
        if (ctx_) ctx_->stop();
    }

    size_t num_workers() const {
        return ctx_ ? ctx_->num_workers() : 0;
    }

    ~PySparkContext() { stop(); }

private:
    std::unique_ptr<vega::SparkContext> ctx_;
};

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

PYBIND11_MODULE(_vega_core, m) {
    m.doc() = "Vega C++ core: actor-based distributed data processing";

    py::class_<PySparkContext>(m, "SparkContext")
        .def(py::init<size_t>(), py::arg("num_workers") = 0)
        .def("parallelize", &PySparkContext::parallelize,
             py::arg("data"), py::arg("num_partitions") = 0)
        .def("stop", &PySparkContext::stop)
        .def_property_readonly("num_workers", &PySparkContext::num_workers)
        .def("__enter__",
             [](PySparkContext& self) -> PySparkContext& { return self; },
             py::return_value_policy::reference_internal)
        .def("__exit__",
             [](PySparkContext& self, py::args) { self.stop(); });

    py::class_<PyRDD>(m, "RDD")
        .def("map", &PyRDD::py_map, py::arg("func"))
        .def("filter", &PyRDD::py_filter, py::arg("func"))
        .def("flat_map", &PyRDD::py_flat_map, py::arg("func"))
        .def("collect", &PyRDD::py_collect)
        .def("count", &PyRDD::py_count)
        .def("reduce", &PyRDD::py_reduce, py::arg("func"))
        .def("foreach", &PyRDD::py_foreach, py::arg("func"))
        .def_property_readonly("num_partitions", &PyRDD::py_num_partitions);
}
