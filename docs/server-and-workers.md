# Server, workers, and clients

Vega has two execution surfaces that are easy to confuse:

1. **`vega_server`** — TCP session server used by **PyVega** (Python client)
2. **`vega_worker`** — P2P cluster **peer** for multi-node **C++** programs

They solve different problems. PyVega always talks to `vega_server`; it does not
connect to `vega_worker` nodes today.

---

## vega_server (PyVega backend)

### What it does

`vega_server` is a long-lived process that:

- Listens on a TCP port for PyVega clients
- Hosts an in-process `vega::SparkContext` with a **JobActor** and **WorkerActors**
- Embeds CPython to run Python UDFs (`map`, `filter`, etc.)
- Stores RDD lineage and executes jobs when clients trigger actions

Each connected client gets a **session** with its own RDD namespace.

### Starting the server

#### Option A — PyVega spawns it (development)

```python
from pyvega import SparkContext
sc = SparkContext(4)   # spawns vega_server on an ephemeral port
```

PyVega sets `PYTHONHOME`, `PYTHONPATH` (including `python/` for broadcast support),
and passes `--port 0 --print-listening` so the client learns the bound port.

#### Option B — `vega-server` CLI

After `pip install -e .`:

```bash
export VEGA_SERVER=build-release/vega_server
vega-server --bind 127.0.0.1:9500 --print-listening
```

#### Option C — run the binary directly

```bash
build-release/vega_server --bind 127.0.0.1 --port 9500
```

### Server flags

| Flag | Default | Description |
|------|---------|-------------|
| `--bind HOST` or `HOST:PORT` | `127.0.0.1` | Bind address |
| `--port PORT` | `9500` | Port (when `--bind` is host-only) |
| `--print-listening` | off | Print `LISTENING <port>` on stdout (used by PyVega spawn) |

The server runs until it receives `SIGINT` / `SIGTERM`.

### Worker threads inside the server

When a client connects, it sends a **HELLO** message with a requested worker count.
The server responds with the actual number of **WorkerActor** threads (capped by
hardware / configuration).

```python
sc = SparkContext(8)
print(sc.num_workers)   # actual worker count after handshake
```

These workers:

- Execute partition tasks in parallel inside the **same OS process** as the server
- Share memory for shuffle buckets and RDD lineage
- Are **not** separate `vega_worker` processes

**Partition count** (`num_partitions` on `parallelize`, `range`, etc.) controls how
many slices data is split into. It can differ from `num_workers`; tasks are assigned
round-robin across workers.

### Environment variables

| Variable | Purpose |
|----------|---------|
| `VEGA_SERVER` | Path to `vega_server` binary (used by PyVega spawn and `vega-server` CLI) |
| `PYTHONHOME` | Required when starting the server manually; base Python prefix |
| `PYTHONPATH` | Must include repo `python/` if using broadcast or server-side PyVega imports |

### Client connection modes

```python
# Auto-spawn local server
sc = SparkContext(4)

# Connect to existing server
sc = SparkContext(remote="127.0.0.1:9500")
sc = SparkContext(remote="sc://127.0.0.1:9500")   # Spark Connect-style URI also accepted

# Always call stop() when done (kills auto-spawned server; closes socket for remote)
sc.stop()
```

### Client ↔ server protocol (overview)

PyVega batches lazy transformations client-side. On the first **action**
(`collect`, `count`, `reduce`, `foreach`):

1. Pending stages are **fused** and sent via RPC (`MAP`, `FILTER`, `GROUP_BY_KEY`, …)
2. The server materializes RDD ids in the session store
3. The action runs through the actor system (`COUNT`, `COLLECT`, …)

Native fast paths (e.g. `range` + integer `map`) skip Python when the client recognizes
the lambda pattern (`pyvega._native.try_compile`).

---

## vega_worker (C++ cluster peers)

### What it does

`vega_worker` runs a **decentralized peer** that participates in multi-node C++
execution. Peers discover each other via **gossip** and coordinate jobs without a
central Spark-style driver.

PyVega does **not** submit jobs to cluster peers yet; use this for C++ cluster
examples and experiments.

### Starting peers locally (three terminals)

```bash
# Terminal 1
./build-release/vega_worker --bind 127.0.0.1:9001 --id peer-1

# Terminal 2
./build-release/vega_worker --bind 127.0.0.1:9002 --id peer-2 \
  --seeds 127.0.0.1:9001

# Terminal 3 — run the cluster word-count example
./build-release/examples/cluster_word_count \
  --bind 127.0.0.1:9003 --id peer-3 \
  --seeds 127.0.0.1:9001 --expect-peers 3
```

All nodes run the **same binary**; the submitter is not a special "driver" role.

### Peer flags

| Flag | Description |
|------|-------------|
| `--bind HOST:PORT` | Listen address for this peer |
| `--seeds H1:P1,H2:P2` | Seed peers for gossip discovery |
| `--id NAME` | Node id (default: `peer-<port>`) |
| `--expect-peers N` | Block until N peers are visible (submitter examples) |

Legacy flags `--driver`, `--worker`, and `--expect-workers` are accepted but ignored;
see `ClusterConfig::from_args` in `include/vega/cluster/cluster.hpp`.

### Single-node C++ without cluster

```bash
./build-release/examples/cluster_word_count --local
./build-release/examples/word_count
```

---

## Choosing a setup

| Goal | Use |
|------|-----|
| Python / PySpark RDD scripts | `vega_server` + PyVega |
| Upstream PySpark RDD test suite | `vega_server` + `scripts/pyspark_compat` |
| Multi-node C++ benchmarks | `vega_worker` peers + cluster examples |
| Production Python deployment | Long-running `vega-server`, clients use `remote=` |

---

## Process diagram

```
Development (auto-spawn)
========================
 python -c "SparkContext(4)..."
        │
        ├─ subprocess: vega_server --port 0
        │       └─ WorkerActor × 4 (threads)
        └─ TCP localhost:random_port


Manual server + client
======================
 Terminal A: vega-server --bind 127.0.0.1:9500
 Terminal B: SparkContext(remote="127.0.0.1:9500")


C++ cluster (no Python)
=======================
 peer-1 ◄──gossip──► peer-2 ◄──gossip──► peer-3
   │                    │                  │
   └──────── partition tasks over TCP ─────┘
```
