#pragma once

#include <cstddef>
#include <memory>

namespace vega {

class RDDBase;

// Describes one unit of work: compute a single partition of an RDD.
struct TaskDescriptor {
    size_t job_id        = 0;
    size_t stage_id      = 0;
    size_t partition_id  = 0;
    std::shared_ptr<RDDBase> rdd;  // the final RDD whose partition we compute
};

} // namespace vega
