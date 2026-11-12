#pragma once

#include <any>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <typeindex>
#include <unordered_map>
#include <vector>
#include <iostream>

namespace vega::actor {

using ActorId = uint64_t;
constexpr ActorId INVALID_ACTOR = 0;

// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------

struct MessageBase {
    virtual ~MessageBase() = default;
    ActorId sender_id = INVALID_ACTOR;
};

template<typename Payload>
struct TypedMessage : MessageBase {
    Payload data;
    explicit TypedMessage(Payload d) : data(std::move(d)) {}
};

// ---------------------------------------------------------------------------
// Forward declarations
// ---------------------------------------------------------------------------

class ActorSystem;

// ---------------------------------------------------------------------------
// Actor base class
//
// Actors process messages sequentially from their mailbox.  Multiple actors
// execute in parallel across the ActorSystem thread pool.  Derived classes
// override configure() to register handlers via on<PayloadT>(...) and
// optionally override on_start() / on_stop().
// ---------------------------------------------------------------------------

class Actor {
public:
    virtual ~Actor() = default;

    ActorId id() const { return id_; }
    ActorSystem* system() const { return system_; }

    // --- messaging (available to subclasses) --------------------------------

    template<typename PayloadT>
    void send(ActorId target, PayloadT payload);

protected:
    // Register a handler for messages carrying PayloadT.
    template<typename PayloadT>
    void on(std::function<void(PayloadT&)> handler) {
        handlers_[std::type_index(typeid(TypedMessage<PayloadT>))] =
            [h = std::move(handler)](MessageBase& raw) {
                h(static_cast<TypedMessage<PayloadT>&>(raw).data);
            };
    }

    // Called once after construction & registration; override to subscribe.
    virtual void configure() {}

    // Called after configure(); actor is live and can send messages.
    virtual void on_start() {}

    // Called when the actor system is shutting down.
    virtual void on_stop() {}

private:
    friend class ActorSystem;

    ActorId id_ = INVALID_ACTOR;
    ActorSystem* system_ = nullptr;

    // Mailbox -----------------------------------------------------------------
    std::mutex mailbox_mutex_;
    std::queue<std::unique_ptr<MessageBase>> mailbox_;
    bool scheduled_ = false;  // guarded by mailbox_mutex_

    // Handler table -----------------------------------------------------------
    std::unordered_map<std::type_index,
                       std::function<void(MessageBase&)>> handlers_;

    // Enqueue a message into this actor's mailbox.
    // Returns true if the actor was not already scheduled (caller must then
    // add it to the system's ready-queue).
    bool enqueue(std::unique_ptr<MessageBase> msg) {
        std::lock_guard<std::mutex> lock(mailbox_mutex_);
        mailbox_.push(std::move(msg));
        if (!scheduled_) {
            scheduled_ = true;
            return true;  // caller must schedule
        }
        return false;
    }

    // Drain the mailbox, dispatching each message to the matching handler.
    void process_messages() {
        while (true) {
            std::unique_ptr<MessageBase> msg;
            {
                std::lock_guard<std::mutex> lock(mailbox_mutex_);
                if (mailbox_.empty()) {
                    scheduled_ = false;
                    return;
                }
                msg = std::move(mailbox_.front());
                mailbox_.pop();
            }
            dispatch(*msg);
        }
    }

    void dispatch(MessageBase& msg) {
        auto it = handlers_.find(std::type_index(typeid(msg)));
        if (it != handlers_.end()) {
            try {
                it->second(msg);
            } catch (const std::exception& e) {
                std::cerr << "[actor " << id_
                          << "] handler exception: " << e.what() << "\n";
            }
        }
    }
};

// ---------------------------------------------------------------------------
// ActorSystem
//
// Manages actor lifecycles and a thread pool for message processing.
// ---------------------------------------------------------------------------

class ActorSystem {
public:
    explicit ActorSystem(size_t num_threads = 0)
        : num_threads_(num_threads > 0
                       ? num_threads
                       : std::max(1u, std::thread::hardware_concurrency())) {}

    ~ActorSystem() { stop(); }

    // Spawn a new actor and return its id.
    template<typename ActorT, typename... Args>
    ActorId spawn(Args&&... args) {
        auto actor = std::make_unique<ActorT>(std::forward<Args>(args)...);
        ActorId id = next_id_.fetch_add(1, std::memory_order_relaxed);

        // Access protected members through Actor* so that the
        // friend declaration in Actor applies.
        Actor* base = actor.get();
        base->id_ = id;
        base->system_ = this;
        base->configure();

        {
            std::lock_guard<std::mutex> lock(actors_mutex_);
            actors_[id] = std::move(actor);
        }
        base->on_start();
        return id;
    }

    // Deliver a message from outside the actor system (e.g. SparkContext).
    template<typename PayloadT>
    void send(ActorId target, PayloadT payload) {
        auto msg = std::make_unique<TypedMessage<PayloadT>>(std::move(payload));
        deliver(target, std::move(msg));
    }

    void deliver(ActorId target, std::unique_ptr<MessageBase> msg) {
        Actor* actor = nullptr;
        {
            std::lock_guard<std::mutex> lock(actors_mutex_);
            auto it = actors_.find(target);
            if (it == actors_.end()) return;
            actor = it->second.get();
        }
        if (actor->enqueue(std::move(msg))) {
            schedule(target);
        }
    }

    void start() {
        if (running_.exchange(true)) return;  // already running
        for (size_t i = 0; i < num_threads_; ++i) {
            threads_.emplace_back(&ActorSystem::worker_loop, this);
        }
    }

    void stop() {
        {
            // Flip `running_` while holding the ready-queue mutex. Workers
            // evaluate the wait predicate under this mutex, so each one either
            // sees running_ == false or is already blocked in wait() and gets
            // the notify below. Without the lock, a worker that had just found
            // the predicate false could miss the notify and sleep forever,
            // hanging join() (seen as a 6-hour CI hang on macOS).
            std::lock_guard<std::mutex> lock(queue_mutex_);
            if (!running_.exchange(false)) return;
        }
        queue_cv_.notify_all();
        for (auto& t : threads_) {
            if (t.joinable()) t.join();
        }
        threads_.clear();

        std::lock_guard<std::mutex> lock(actors_mutex_);
        for (auto& [id, actor] : actors_) {
            actor->on_stop();
        }
    }

    bool running() const { return running_.load(); }

private:
    size_t num_threads_;
    std::atomic<ActorId> next_id_{1};
    std::atomic<bool> running_{false};

    std::unordered_map<ActorId, std::unique_ptr<Actor>> actors_;
    std::mutex actors_mutex_;

    std::queue<ActorId> ready_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;

    std::vector<std::thread> threads_;

    void schedule(ActorId id) {
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            ready_queue_.push(id);
        }
        queue_cv_.notify_one();
    }

    void worker_loop() {
        while (true) {
            ActorId actor_id;
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                queue_cv_.wait(lock, [&] {
                    return !ready_queue_.empty() || !running_.load();
                });
                if (!running_.load() && ready_queue_.empty()) return;
                if (ready_queue_.empty()) continue;
                actor_id = ready_queue_.front();
                ready_queue_.pop();
            }

            Actor* actor = nullptr;
            {
                std::lock_guard<std::mutex> lock(actors_mutex_);
                auto it = actors_.find(actor_id);
                if (it == actors_.end()) continue;
                actor = it->second.get();
            }
            actor->process_messages();
        }
    }
};

// --- Actor template method implementations (need ActorSystem definition) ----

template<typename PayloadT>
void Actor::send(ActorId target, PayloadT payload) {
    auto msg = std::make_unique<TypedMessage<PayloadT>>(std::move(payload));
    msg->sender_id = id_;
    system_->deliver(target, std::move(msg));
}

} // namespace vega::actor
