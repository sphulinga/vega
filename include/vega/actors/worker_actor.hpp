#pragma once

#include <any>
#include <stdexcept>
#include <string>

#include "vega/actors/actor_system.hpp"
#include "vega/actors/messages.hpp"
#include "vega/core/rdd.hpp"
#include "vega/python/gil_guard.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace vega {

// WorkerActor: owns a slice of the partition space and executes tasks.
//
// When it receives an AssignTask, it computes the RDD partition and sends
// the result back to the JobActor identified by reply_to.  In a multi-node
// deployment each node would run one or more WorkerActors; for now they
// all live in the same process.
class WorkerActor : public actor::Actor {
public:
    WorkerActor() = default;

protected:
    void configure() override {
        on<msg::AssignTask>([this](msg::AssignTask& task) {
            handle_task(task);
        });
    }

private:
    void send_task_error(const msg::AssignTask& task, const std::string& error) {
        send<msg::TaskResult>(task.reply_to, msg::TaskResult{
            task.job_id,
            task.partition_id,
            std::any{},
            error,
        });
    }

    void handle_task(msg::AssignTask& task) {
        try {
            if (task.rdd->uses_python_runtime()) {
                run_with_python_gil([&]() {
                    std::any result;
                    if (task.action == msg::JobAction::COUNT_PARTITION) {
                        result = task.rdd->compute_partition_count(task.partition_id);
                    } else {
                        result = task.rdd->compute_partition(task.partition_id);
                    }
                    send<msg::TaskResult>(task.reply_to, msg::TaskResult{
                        task.job_id,
                        task.partition_id,
                        std::move(result),
                        {},
                    });
                });
                return;
            }

            std::any result;
            if (task.action == msg::JobAction::COUNT_PARTITION) {
                result = task.rdd->compute_partition_count(task.partition_id);
            } else {
                result = task.rdd->compute_partition(task.partition_id);
            }
            send<msg::TaskResult>(task.reply_to, msg::TaskResult{
                task.job_id,
                task.partition_id,
                std::move(result),
                {},
            });
        } catch (const py::error_already_set& e) {
            send_task_error(task, e.what());
        } catch (const std::exception& e) {
            send_task_error(task, e.what());
        }
    }
};

} // namespace vega
