#pragma once

#include <memory>
#include <unordered_set>
#include <vector>

#include "vega/core/rdd.hpp"
#include "vega/execution/stage.hpp"

namespace vega {

struct ExecutionPlan {
    std::vector<Stage> stages;
};

// Builds a DAG of stages from an RDD lineage graph.
//
// Walks the lineage backwards from the target RDD (post-order, across *all*
// dependencies, so unions/joins of several shuffled parents work). Every
// wide (shuffle) RDD found introduces one map-side stage that computes the
// shuffle's parent; the stage that follows it carries the shuffle RDD so the
// JobActor can call prepare_shuffle() with the map outputs before launching
// it. Nested shuffles are ordered innermost-first, so chains like
// reduceByKey -> map -> groupByKey execute as three stages.
//
// Shuffle reuse: when `reuse_shuffles` is true, a shuffle whose output is
// already materialized (shuffle_ready()) is not recomputed and its lineage is
// not walked -- the equivalent of Spark's "skipped stages". This makes
// repeated actions on the same shuffled RDD (e.g. toLocalIterator, take)
// cheap.
//
// `partitions` restricts the final stage to a subset of partition ids
// (empty = all partitions). Map stages always cover all partitions because a
// shuffle needs every map output.
class DAGBuilder {
public:
    static ExecutionPlan build(size_t job_id,
                               std::shared_ptr<RDDBase> target_rdd,
                               const std::vector<size_t>& partitions = {},
                               bool reuse_shuffles = true) {
        ExecutionPlan plan;
        size_t stage_id = 0;

        std::vector<std::shared_ptr<RDDBase>> shuffles;
        std::unordered_set<const RDDBase*> seen;
        collect_shuffles(target_rdd, reuse_shuffles, seen, shuffles);

        std::shared_ptr<RDDBase> pending_shuffle;
        for (auto& shuffle : shuffles) {
            auto deps = shuffle->dependencies();
            if (deps.empty())
                continue;
            Stage map_stage = make_stage(stage_id++, job_id, deps.at(0), {});
            if (pending_shuffle) {
                map_stage.is_shuffle_boundary = true;
                map_stage.shuffle_rdd = pending_shuffle;
                map_stage.parent_stage_ids.push_back(map_stage.id - 1);
            }
            plan.stages.push_back(std::move(map_stage));
            pending_shuffle = shuffle;
        }

        Stage final_stage = make_stage(stage_id++, job_id, target_rdd, partitions);
        if (pending_shuffle) {
            final_stage.is_shuffle_boundary = true;
            final_stage.shuffle_rdd = pending_shuffle;
            final_stage.parent_stage_ids.push_back(final_stage.id - 1);
        }
        plan.stages.push_back(std::move(final_stage));
        return plan;
    }

private:
    static Stage make_stage(size_t stage_id, size_t job_id,
                            std::shared_ptr<RDDBase> rdd,
                            const std::vector<size_t>& partitions) {
        Stage s;
        s.id = stage_id;
        s.final_rdd = rdd;
        if (partitions.empty()) {
            size_t n = rdd->num_partitions();
            s.tasks.reserve(n);
            for (size_t p = 0; p < n; ++p)
                s.tasks.push_back(TaskDescriptor{job_id, stage_id, p, rdd});
        } else {
            s.tasks.reserve(partitions.size());
            for (size_t p : partitions)
                s.tasks.push_back(TaskDescriptor{job_id, stage_id, p, rdd});
        }
        return s;
    }

    // Post-order walk: inner shuffles are appended before the shuffles that
    // consume them.
    static void collect_shuffles(const std::shared_ptr<RDDBase>& rdd,
                                 bool reuse_shuffles,
                                 std::unordered_set<const RDDBase*>& seen,
                                 std::vector<std::shared_ptr<RDDBase>>& out) {
        if (!rdd) return;
        const RDDBase* key = rdd->identity();
        if (!seen.insert(key).second) return;

        const bool wide = !rdd->is_narrow();
        if (wide && reuse_shuffles && rdd->shuffle_ready())
            return;  // materialized shuffle output: skip its lineage

        for (auto& dep : rdd->dependencies())
            collect_shuffles(dep, reuse_shuffles, seen, out);

        if (wide)
            out.push_back(rdd);
    }
};

} // namespace vega
