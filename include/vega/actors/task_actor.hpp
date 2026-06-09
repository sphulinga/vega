#pragma once

#include "vega/actors/actor_system.hpp"
#include "vega/actors/messages.hpp"
#include "vega/core/rdd.hpp"

namespace vega {

// TaskActor: fine-grained, ephemeral actor for one task.
//
// Phase 1 uses WorkerActors directly.  TaskActor is the building block
// for Phase 2's multi-stage peer-to-peer coordination where each task
// independently tracks its upstream dependencies and self-activates
// when all inputs are ready.
class TaskActor : public actor::Actor {
public:
    TaskActor(size_t job_id, size_t stage_id, size_t partition_id,
              std::shared_ptr<RDDBase> rdd,
              actor::ActorId reply_to,
              std::vector<actor::ActorId> downstream = {})
        : job_id_(job_id)
        , stage_id_(stage_id)
        , partition_id_(partition_id)
        , rdd_(std::move(rdd))
        , reply_to_(reply_to)
        , downstream_(std::move(downstream)) {}

protected:
    void configure() override {
        on<msg::TaskComplete>([this](msg::TaskComplete& tc) {
            on_upstream_complete(tc);
        });
    }

    void on_start() override {
        if (pending_deps_ == 0) {
            execute();
        }
    }

private:
    size_t job_id_;
    size_t stage_id_;
    size_t partition_id_;
    std::shared_ptr<RDDBase> rdd_;
    actor::ActorId reply_to_;
    std::vector<actor::ActorId> downstream_;

    size_t pending_deps_ = 0;

    void on_upstream_complete(msg::TaskComplete& /*tc*/) {
        if (--pending_deps_ == 0)
            execute();
    }

    void execute() {
        try {
            std::any result = rdd_->compute_partition(partition_id_);

            send<msg::TaskResult>(reply_to_, msg::TaskResult{
                job_id_, partition_id_, std::move(result), {}
            });

            // Notify downstream task actors that this partition is done.
            for (auto dst : downstream_) {
                send<msg::TaskComplete>(dst, msg::TaskComplete{
                    job_id_, stage_id_, partition_id_, id()
                });
            }
        } catch (const std::exception& e) {
            send<msg::TaskResult>(reply_to_, msg::TaskResult{
                job_id_, partition_id_, std::any{}, e.what()
            });
        }
    }
};

} // namespace vega
