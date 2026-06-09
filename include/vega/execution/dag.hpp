#pragma once

#include <memory>
#include <vector>

#include "vega/core/rdd.hpp"
#include "vega/execution/stage.hpp"

namespace vega {

struct ExecutionPlan {
    std::vector<Stage> stages;
};

// Builds a DAG of stages from an RDD lineage graph.
//
// Walks the lineage backwards from the target RDD.  Each non-narrow
// dependency (shuffle) introduces a stage boundary.  Stages are ordered
// so that all parent stages come before child stages.
class DAGBuilder {
public:
    static ExecutionPlan build(size_t job_id,
                               std::shared_ptr<RDDBase> target_rdd) {
        ExecutionPlan plan;
        size_t stage_id = 0;

        // Find the first shuffle boundary by walking the lineage.
        auto shuffle_rdd = find_shuffle(target_rdd);

        if (!shuffle_rdd) {
            // No shuffle -- single stage (narrow transforms only).
            plan.stages.push_back(
                make_stage(stage_id, job_id, target_rdd));
            return plan;
        }

        // Two stages: map-side (parent of shuffle) then reduce-side.
        auto parent = shuffle_rdd->dependencies().at(0);

        Stage map_stage = make_stage(stage_id++, job_id, parent);
        plan.stages.push_back(std::move(map_stage));

        Stage reduce_stage;
        reduce_stage.id = stage_id++;
        reduce_stage.final_rdd = target_rdd;
        reduce_stage.is_shuffle_boundary = true;
        reduce_stage.shuffle_rdd = shuffle_rdd;
        reduce_stage.parent_stage_ids.push_back(map_stage.id);

        size_t n = target_rdd->num_partitions();
        reduce_stage.tasks.reserve(n);
        for (size_t p = 0; p < n; ++p)
            reduce_stage.tasks.push_back(
                TaskDescriptor{job_id, reduce_stage.id, p, target_rdd});

        plan.stages.push_back(std::move(reduce_stage));
        return plan;
    }

private:
    static Stage make_stage(size_t stage_id, size_t job_id,
                            std::shared_ptr<RDDBase> rdd) {
        Stage s;
        s.id = stage_id;
        s.final_rdd = rdd;
        size_t n = rdd->num_partitions();
        s.tasks.reserve(n);
        for (size_t p = 0; p < n; ++p)
            s.tasks.push_back(TaskDescriptor{job_id, stage_id, p, rdd});
        return s;
    }

    // Walk lineage and return the first non-narrow RDD, or nullptr.
    static std::shared_ptr<RDDBase> find_shuffle(
            std::shared_ptr<RDDBase> rdd) {
        if (!rdd->is_narrow())
            return rdd;
        for (auto& dep : rdd->dependencies()) {
            auto result = find_shuffle(dep);
            if (result) return result;
        }
        return nullptr;
    }
};

} // namespace vega
