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
    bool got_data = false;

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
