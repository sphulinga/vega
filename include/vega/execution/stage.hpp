#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "vega/execution/task.hpp"

namespace vega {

// A stage is a group of tasks that can be pipelined (all narrow transforms).
// Stage boundaries occur at shuffle points (wide transforms – Phase 2).
struct Stage {
    size_t id = 0;
    std::shared_ptr<RDDBase> final_rdd;   // output RDD of this stage
    std::vector<TaskDescriptor> tasks;     // one task per partition
    std::vector<size_t> parent_stage_ids;  // stages this depends on
    bool is_shuffle_boundary = false;      // true if preceded by a shuffle

    // For shuffle stages: the RDD that needs prepare_shuffle() called
    // before its compute_partition() will work.
    std::shared_ptr<RDDBase> shuffle_rdd;
};

} // namespace vega
