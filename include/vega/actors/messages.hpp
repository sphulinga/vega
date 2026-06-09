#pragma once

#include <any>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <vector>

#include "vega/actors/actor_system.hpp"

namespace vega {

class RDDBase;  // forward declare

namespace msg {

enum class JobAction : uint8_t {
    COMPUTE_PARTITION = 0,
    COUNT_PARTITION   = 1,
};

// Context -> JobActor: execute an RDD action.
struct SubmitJob {
    size_t job_id;
    std::shared_ptr<RDDBase> rdd;
    std::shared_ptr<std::promise<std::vector<std::any>>> promise;
    JobAction action = JobAction::COMPUTE_PARTITION;
};

// JobActor -> WorkerActor: execute one partition.
struct AssignTask {
    size_t job_id;
    size_t partition_id;
    std::shared_ptr<RDDBase> rdd;
    actor::ActorId reply_to;   // JobActor id
    JobAction action = JobAction::COMPUTE_PARTITION;
};

// WorkerActor -> JobActor: result of one partition.
struct TaskResult {
    size_t job_id;
    size_t partition_id;
    std::any data;             // vector<T> wrapped in any
    std::string error;
};

// TaskActor -> downstream TaskActors (Phase 2: multi-stage peer coordination).
struct TaskComplete {
    size_t job_id;
    size_t stage_id;
    size_t partition_id;
    actor::ActorId source;
};

// Shuffle data between stages (Phase 2).
struct ShuffleBlock {
    size_t job_id;
    size_t source_partition;
    size_t target_partition;
    std::any data;
};

// JobActor -> Context: final job result (used internally via promise).
struct JobComplete {
    size_t job_id;
    bool success;
    std::string error;
};

} // namespace msg
} // namespace vega
