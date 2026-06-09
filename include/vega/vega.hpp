#pragma once

// Single-header convenience include for Vega.
//
// Users only need:
//   #include <vega/vega.hpp>

#include "vega/core/partition.hpp"
#include "vega/core/rdd.hpp"
#include "vega/core/shuffled_rdd.hpp"
#include "vega/core/context.hpp"

#include "vega/actors/actor_system.hpp"
#include "vega/actors/messages.hpp"
#include "vega/actors/worker_actor.hpp"
#include "vega/actors/task_actor.hpp"
#include "vega/actors/job_actor.hpp"

#include "vega/execution/task.hpp"
#include "vega/execution/stage.hpp"
#include "vega/execution/dag.hpp"

#include "vega/transport/transport.hpp"
#include "vega/transport/local_transport.hpp"

#include "vega/serialization/serializer.hpp"
#include "vega/network/endpoint.hpp"
#include "vega/network/tcp_connection.hpp"
#include "vega/network/tcp_server.hpp"
#include "vega/network/message_codec.hpp"
#include "vega/network/message_router.hpp"
#include "vega/cluster/node.hpp"
#include "vega/cluster/gossip.hpp"
#include "vega/cluster/cluster.hpp"
