#include <gtest/gtest.h>
#include <vega/vega.hpp>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <numeric>
#include <sstream>
#include <thread>

using namespace vega::actor;

// --- Ping/Pong smoke test for the actor system ---

namespace {

struct Ping { int value; };
struct Pong { int value; };

class PingActor : public Actor {
public:
    PingActor(ActorId pong_id, std::shared_ptr<std::promise<int>> p)
        : pong_(pong_id), promise_(std::move(p)) {}

protected:
    void configure() override {
        on<Pong>([this](Pong& msg) {
            promise_->set_value(msg.value);
        });
    }

    void on_start() override {
        send<Ping>(pong_, Ping{42});
    }

private:
    ActorId pong_;
    std::shared_ptr<std::promise<int>> promise_;
};

class PongActor : public Actor {
protected:
    void configure() override {
        on<Ping>([this](Ping& msg) {
            // Reply with value doubled. sender_id is on MessageBase;
            // we get the sender from the raw message via the actor system.
            // For simplicity, PingActor passes its own id embedded in the
            // message in a real system; here we use a workaround.
        });
    }
};

} // namespace

TEST(ActorSystemTest, StartAndStop) {
    ActorSystem sys(2);
    sys.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    sys.stop();
}

// Regression: stop() used to clear `running_` without holding the ready-queue
// mutex, so a worker that had just checked the wait predicate could miss the
// shutdown notification and sleep forever, hanging stop() in join(). This
// hung CI for 6 hours on macOS. The window is a few instructions wide and is
// hit when stop() runs just as a worker returns to wait() -- e.g. right after
// that worker's handler unblocked the caller -- so each cycle does exactly
// that. It is probabilistic (a few thousand cycles); a watchdog turns a hang
// into a failure instead of a stuck test run.
namespace {
struct Signal { int value; };
class SignalActor : public Actor {
public:
    explicit SignalActor(std::shared_ptr<std::promise<void>> p) : p_(std::move(p)) {}
protected:
    void configure() override {
        on<Signal>([this](Signal&) { p_->set_value(); });
    }
private:
    std::shared_ptr<std::promise<void>> p_;
};
}  // namespace

TEST(ActorSystemTest, StopRightAfterHandlerDoesNotHang) {
    auto cycles = std::async(std::launch::async, [] {
        for (int i = 0; i < 4000; ++i) {
            ActorSystem sys(2);
            auto done = std::make_shared<std::promise<void>>();
            auto fut = done->get_future();
            auto id = sys.spawn<SignalActor>(done);
            sys.start();
            sys.send<Signal>(id, Signal{i});
            fut.wait();  // the worker is now returning to wait()...
            sys.stop();  // ...which is where the shutdown wakeup got lost
        }
    });
    if (cycles.wait_for(std::chrono::seconds(60)) != std::future_status::ready) {
        // The hung thread can never be joined (the future's destructor would
        // block too), so report and exit instead of hanging the test run.
        std::fprintf(stderr, "FAILED: ActorSystem::stop() hung (lost shutdown wakeup)\n");
        std::fflush(stderr);
        std::_Exit(1);
    }
    cycles.get();
}

TEST(ActorSystemTest, SpawnActors) {
    ActorSystem sys(2);
    auto id1 = sys.spawn<PongActor>();
    auto id2 = sys.spawn<PongActor>();
    EXPECT_NE(id1, id2);
    EXPECT_NE(id1, INVALID_ACTOR);
    sys.start();
    sys.stop();
}

TEST(ActorSystemTest, MessageDelivery) {
    ActorSystem sys(2);
    sys.start();

    struct Notify { int val; };
    struct Ack { int val; };

    auto promise = std::make_shared<std::promise<int>>();
    auto future = promise->get_future();

    // Receiver echoes back via promise.
    class Receiver : public Actor {
    public:
        explicit Receiver(std::shared_ptr<std::promise<int>> p)
            : p_(std::move(p)) {}
    protected:
        void configure() override {
            on<Notify>([this](Notify& n) { p_->set_value(n.val); });
        }
    private:
        std::shared_ptr<std::promise<int>> p_;
    };

    auto rid = sys.spawn<Receiver>(std::move(promise));
    sys.send<Notify>(rid, Notify{123});

    auto status = future.wait_for(std::chrono::seconds(5));
    ASSERT_EQ(status, std::future_status::ready);
    EXPECT_EQ(future.get(), 123);

    sys.stop();
}

// --- End-to-end: actor-based RDD execution ---

TEST(ActorEndToEnd, ParallelizeMapCollect) {
    vega::SparkContext ctx(2);
    auto result = ctx.parallelize(std::vector<int>{10, 20, 30}, 2)
        .map([](const int& x) { return x + 1; })
        .collect();
    std::sort(result.begin(), result.end());
    EXPECT_EQ(result, (std::vector<int>{11, 21, 31}));
}

TEST(ActorEndToEnd, ParallelReduce) {
    vega::SparkContext ctx(4);
    std::vector<int> data(1000);
    std::iota(data.begin(), data.end(), 1);

    int sum = ctx.parallelize(data, 8)
        .reduce([](const int& a, const int& b) { return a + b; });
    EXPECT_EQ(sum, 500500);  // 1+2+...+1000
}

TEST(ActorEndToEnd, FlatMapFilterCollect) {
    vega::SparkContext ctx(2);

    auto result = ctx.parallelize(
            std::vector<std::string>{"1 2 3", "4 5 6"}, 2)
        .flatMap([](const std::string& s) {
            std::vector<int> nums;
            std::istringstream iss(s);
            int n;
            while (iss >> n) nums.push_back(n);
            return nums;
        })
        .filter([](const int& n) { return n > 3; })
        .collect();

    std::sort(result.begin(), result.end());
    EXPECT_EQ(result, (std::vector<int>{4, 5, 6}));
}
