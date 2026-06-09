#pragma once

#include <any>
#include <future>
#include <unordered_map>
#include <vector>

#include "vega/actors/actor_system.hpp"
#include "vega/actors/messages.hpp"
#include "vega/core/rdd.hpp"
#include "vega/execution/dag.hpp"
#include "vega/python/gil_guard.hpp"

namespace vega {

// JobActor: orchestrates a single RDD action.
//
// Builds a DAG of stages from the RDD lineage, executes them in
// dependency order, performs shuffle between stages, and fulfills the
// promise so the calling thread (SparkContext::execute) can return.
class JobActor : public actor::Actor {
public:
    explicit JobActor(std::vector<actor::ActorId> worker_ids)
        : workers_(std::move(worker_ids)) {}

protected:
    void configure() override {
        on<msg::SubmitJob>([this](msg::SubmitJob& job) {
            handle_submit(job);
        });
        on<msg::TaskResult>([this](msg::TaskResult& res) {
            handle_result(res);
        });
    }

private:
    std::vector<actor::ActorId> workers_;

    struct JobState {
        ExecutionPlan plan;
        size_t current_stage = 0;
        size_t expected = 0;
        size_t received = 0;
        std::vector<std::any> results;
        std::shared_ptr<std::promise<std::vector<std::any>>> promise;
        msg::JobAction action = msg::JobAction::COMPUTE_PARTITION;
        msg::JobAction current_task_action = msg::JobAction::COMPUTE_PARTITION;
    };
    std::unordered_map<size_t, JobState> jobs_;

    static bool compute_results_need_python_gil(const JobState& state) {
        if (state.action != msg::JobAction::COMPUTE_PARTITION)
            return false;
        if (state.plan.stages.empty())
            return false;
        const auto& stage = state.plan.stages[state.current_stage];
        if (stage.tasks.empty())
            return false;
        return stage.tasks.front().rdd->uses_python_runtime();
    }

    static bool stage_results_need_python_gil(const JobState& state) {
        if (state.plan.stages.empty())
            return false;
        const auto& stage = state.plan.stages[state.current_stage];
        if (stage.tasks.empty())
            return false;
        return stage.tasks.front().rdd->uses_python_runtime()
               && state.current_task_action == msg::JobAction::COMPUTE_PARTITION;
    }

    static void clear_stage_results(JobState& state) {
        if (stage_results_need_python_gil(state)) {
            run_with_python_gil([&]() { state.results.clear(); });
        } else {
            state.results.clear();
        }
    }

    static void fulfill_promise(JobState& state) {
        if (compute_results_need_python_gil(state)) {
            run_with_python_gil([&]() {
                state.promise->set_value(std::move(state.results));
            });
        } else {
            state.promise->set_value(std::move(state.results));
        }
    }

    void handle_submit(msg::SubmitJob& job) {
        auto plan = DAGBuilder::build(job.job_id, job.rdd);

        JobState state;
        state.plan = std::move(plan);
        state.promise = std::move(job.promise);
        state.current_stage = 0;
        state.action = job.action;

        if (state.plan.stages.empty()) {
            state.promise->set_value({});
            return;
        }

        size_t jid = job.job_id;
        jobs_[jid] = std::move(state);
        launch_current_stage(jid);
    }

    void handle_result(msg::TaskResult& res) {
        auto it = jobs_.find(res.job_id);
        if (it == jobs_.end()) return;
        auto& state = it->second;

        if (!res.error.empty()) {
            state.promise->set_exception(
                std::make_exception_ptr(std::runtime_error(res.error)));
            jobs_.erase(it);
            return;
        }

        if (res.partition_id < state.results.size()) {
            auto& stage = state.plan.stages[state.current_stage];
            const bool uses_py =
                state.action == msg::JobAction::COMPUTE_PARTITION &&
                !stage.tasks.empty() &&
                stage.tasks.front().rdd->uses_python_runtime();
            try {
                if (uses_py) {
                    run_with_python_gil([&]() {
                        state.results[res.partition_id] = std::move(res.data);
                    });
                } else {
                    state.results[res.partition_id] = std::move(res.data);
                }
            } catch (const std::exception& e) {
                state.promise->set_exception(std::current_exception());
                jobs_.erase(it);
                return;
            }
        }

        if (++state.received < state.expected)
            return;

        // Current stage complete.
        size_t next = state.current_stage + 1;
        if (next < state.plan.stages.size()) {
            auto& next_stage = state.plan.stages[next];
            if (next_stage.is_shuffle_boundary && next_stage.shuffle_rdd) {
                try {
                    if (next_stage.shuffle_rdd->uses_python_runtime()) {
                        run_with_python_gil([&]() {
                            next_stage.shuffle_rdd->prepare_shuffle(state.results);
                        });
                    } else {
                        next_stage.shuffle_rdd->prepare_shuffle(state.results);
                    }
                } catch (const std::exception& e) {
                    state.promise->set_exception(std::current_exception());
                    jobs_.erase(it);
                    return;
                }
            }
            state.current_stage = next;
            state.received = 0;
            clear_stage_results(state);
            launch_current_stage(res.job_id);
        } else {
            fulfill_promise(state);
            jobs_.erase(it);
        }
    }

    void launch_current_stage(size_t job_id) {
        auto& state = jobs_[job_id];
        auto& stage = state.plan.stages[state.current_stage];
        auto task_action = state.action;
        // Shuffle-producing stages must materialize partition outputs for
        // prepare_shuffle(); count-optimized execution only applies when the
        // stage output is consumed directly by the final action.
        if (state.current_stage + 1 < state.plan.stages.size()) {
            const auto& next_stage = state.plan.stages[state.current_stage + 1];
            if (next_stage.is_shuffle_boundary) {
                task_action = msg::JobAction::COMPUTE_PARTITION;
            }
        }

        state.expected = stage.tasks.size();
        state.results.resize(state.expected);
        state.current_task_action = task_action;

        if (state.expected == 0) {
            // Empty stage -- advance immediately.
            size_t next = state.current_stage + 1;
            if (next < state.plan.stages.size()) {
                auto& ns = state.plan.stages[next];
                if (ns.is_shuffle_boundary && ns.shuffle_rdd) {
                    if (ns.shuffle_rdd->uses_python_runtime()) {
                        run_with_python_gil([&]() {
                            ns.shuffle_rdd->prepare_shuffle(state.results);
                        });
                    } else {
                        ns.shuffle_rdd->prepare_shuffle(state.results);
                    }
                }
                state.current_stage = next;
                launch_current_stage(job_id);
            } else {
                fulfill_promise(state);
                jobs_.erase(job_id);
            }
            return;
        }

        for (auto& task : stage.tasks) {
            actor::ActorId worker =
                workers_[task.partition_id % workers_.size()];
            send<msg::AssignTask>(worker, msg::AssignTask{
                task.job_id, task.partition_id, task.rdd, id(), task_action
            });
        }
    }
};

} // namespace vega
