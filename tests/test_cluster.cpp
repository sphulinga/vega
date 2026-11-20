#include <atomic>
#include <string>
#include <thread>

#include <gtest/gtest.h>
#include <vega/serialization/serializer.hpp>
#include <vega/network/endpoint.hpp>
#include <vega/network/tcp_server.hpp>
#include <vega/network/tcp_connection.hpp>
#include <vega/network/message_codec.hpp>
#include <vega/cluster/node.hpp>
#include <vega/cluster/gossip.hpp>
#include <vega/vega.hpp>
#include <vega/cluster/cluster.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <future>
#include <map>
#include <unistd.h>

using namespace vega;

TEST(EndpointTest, ParseHostPort) {
    auto ep = net::Endpoint::parse("10.0.0.1:8080");
    EXPECT_EQ(ep.host, "10.0.0.1");
    EXPECT_EQ(ep.port, 8080);
}

TEST(EndpointTest, ToString) {
    net::Endpoint ep{"localhost", 9000};
    EXPECT_EQ(ep.to_string(), "localhost:9000");
}

TEST(TcpTest, SendReceiveFrame) {
    ByteBuffer received;
    std::atomic<bool> got_data{false};

    net::TcpServer server("127.0.0.1", 0,
        [&](std::unique_ptr<net::TcpConnection> conn) {
            received = conn->recv_frame();
            got_data = true;
        });
    server.start();

    auto conn = net::TcpConnection::connect("127.0.0.1", server.port());
    ByteBuffer payload = {1, 2, 3, 4, 5};
    conn->send_frame(payload);
    conn.reset();  // close connection

    // Wait for server to process.
    for (int i = 0; i < 50 && !got_data; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

    server.stop();
    EXPECT_TRUE(got_data);
    EXPECT_EQ(received, payload);
}

TEST(TcpTest, MultipleFrames) {
    std::vector<ByteBuffer> received;
    std::mutex mu;

    net::TcpServer server("127.0.0.1", 0,
        [&](std::unique_ptr<net::TcpConnection> conn) {
            try {
                while (true) {
                    auto frame = conn->recv_frame();
                    std::lock_guard<std::mutex> lock(mu);
                    received.push_back(std::move(frame));
                }
            } catch (...) {}
        });
    server.start();

    auto conn = net::TcpConnection::connect("127.0.0.1", server.port());
    conn->send_frame({10});
    conn->send_frame({20, 30});
    conn->send_frame({40, 50, 60});
    conn.reset();

    for (int i = 0; i < 50; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        std::lock_guard<std::mutex> lock(mu);
        if (received.size() >= 3) break;
    }

    server.stop();
    ASSERT_EQ(received.size(), 3u);
    EXPECT_EQ(received[0], (ByteBuffer{10}));
    EXPECT_EQ(received[1], (ByteBuffer{20, 30}));
    EXPECT_EQ(received[2], (ByteBuffer{40, 50, 60}));
}

TEST(MessageCodecTest, EncodeDecodeHeader) {
    ByteBuffer body = {0xAA, 0xBB};
    auto frame = net::encode_message(
        net::NetMsgType::ASSIGN_TASK, "node-1", body);

    size_t off = 0;
    auto header = net::NetMsgHeader::read(frame, off);
    EXPECT_EQ(header.type, net::NetMsgType::ASSIGN_TASK);
    EXPECT_EQ(header.sender_node_id, "node-1");

    ByteBuffer remaining(frame.begin() + off, frame.end());
    EXPECT_EQ(remaining, body);
}

TEST(NodeInfoTest, SerializeDeserialize) {
    cluster::NodeInfo ni;
    ni.id = "worker-42";
    ni.endpoint = {"192.168.1.10", 9001};
    ni.role = cluster::NodeRole::WORKER;

    ByteBuffer buf;
    ni.serialize(buf);

    size_t off = 0;
    auto result = cluster::NodeInfo::deserialize(buf, off);
    EXPECT_EQ(result.id, "worker-42");
    EXPECT_EQ(result.endpoint.host, "192.168.1.10");
    EXPECT_EQ(result.endpoint.port, 9001);
    EXPECT_EQ(result.role, cluster::NodeRole::WORKER);
}

TEST(GossipTest, SelfDiscovery) {
    cluster::NodeInfo self;
    self.id = "test-node";
    self.endpoint = {"127.0.0.1", 0};
    self.role = cluster::NodeRole::WORKER;

    cluster::GossipProtocol::Config cfg;
    cluster::GossipProtocol gossip(self, cfg);

    auto nodes = gossip.live_nodes();
    ASSERT_EQ(nodes.size(), 1u);
    EXPECT_EQ(nodes[0].id, "test-node");
}

TEST(GossipTest, HandlePingMergesNodes) {
    cluster::NodeInfo self;
    self.id = "node-a";
    self.endpoint = {"127.0.0.1", 0};
    self.role = cluster::NodeRole::WORKER;

    bool joined = false;
    cluster::GossipProtocol::Config cfg;
    cluster::GossipProtocol gossip(self, cfg,
        [&](const cluster::NodeInfo&) { joined = true; });

    // Simulate receiving a gossip ping with another node.
    cluster::NodeInfo remote;
    remote.id = "node-b";
    remote.endpoint = {"10.0.0.1", 9001};
    remote.role = cluster::NodeRole::WORKER;

    ByteBuffer body;
    write_pod<uint32_t>(body, 1);  // count
    remote.serialize(body);

    gossip.handle_gossip_ping(body);

    EXPECT_TRUE(joined);
    EXPECT_EQ(gossip.live_count(), 2u);
    EXPECT_EQ(gossip.live_workers().size(), 2u);
}

// End-to-end P2P cluster: three peers in one process, talking over real
// localhost TCP, run the same program (SPMD) -- scatter from peer 0, a
// narrow job and a shuffle job -- then all stop. Each step used to hang:
// non-root peers ignored scattered data, messages from faster peers that
// arrived before a slower peer reached the job were dropped, and stop()
// waited for remote peers to close their connections.
TEST(ClusterTest, ThreePeersScatterCollectAndShuffle) {
    const uint16_t base = static_cast<uint16_t>(20000 + (::getpid() % 4000) * 3);
    struct Out {
        std::vector<int> doubled;
        std::map<int, int> counts;
    };
    auto peer = [base](int i) {
        ClusterConfig cfg;
        cfg.bind_host = "127.0.0.1";
        cfg.bind_port = static_cast<uint16_t>(base + i);
        cfg.node_id = "peer-" + std::to_string(i);
        cfg.expected_peers = 3;
        if (i > 0) cfg.seed_nodes.push_back(net::Endpoint{"127.0.0.1", base});
        Cluster c(cfg);
        if (!c.wait_for_peers(std::chrono::seconds(30))) throw std::runtime_error("peers did not join");
        std::vector<int> data;
        if (c.my_index() == 0)
            for (int x = 0; x < 1000; ++x) data.push_back(x);
        auto rdd = c.scatter(data, 6);
        Out out;
        out.doubled = c.collect(rdd.map([](const int& x) { return x * 2; }));
        auto pairs = c.collect(rdd.map([](const int& x) { return std::make_pair(x % 10, 1); })
                                   .reduceByKey([](const int& a, const int& b) { return a + b; }));
        for (auto& [k, v] : pairs) out.counts[k] = v;
        c.stop();
        return out;
    };
    std::vector<std::future<Out>> futures;
    for (int i = 0; i < 3; ++i) {
        futures.push_back(std::async(std::launch::async, peer, i));
        std::this_thread::sleep_for(std::chrono::milliseconds(i == 0 ? 200 : 50));
    }
    for (auto& f : futures) {
        if (f.wait_for(std::chrono::seconds(90)) != std::future_status::ready) {
            ADD_FAILURE() << "cluster job hung";
            std::_Exit(1);  // threads are stuck; do not wait for them
        }
    }
    std::vector<int> want;
    for (int x = 0; x < 1000; ++x) want.push_back(2 * x);
    for (auto& f : futures) {
        Out o = f.get();
        std::sort(o.doubled.begin(), o.doubled.end());
        EXPECT_EQ(o.doubled, want);  // every peer gets the whole result
        ASSERT_EQ(o.counts.size(), 10u);
        for (auto& [k, v] : o.counts) EXPECT_EQ(v, 100) << "key " << k;
    }
}
