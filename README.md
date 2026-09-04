# kvtier

A KV cache tier for LLM inference engines. It sits underneath vLLM or SGLang
rather than replacing them: the engine asks how much of a prompt's KV cache
already exists somewhere, fetches what does, and hands back what it had to
compute. Written in Rust.

The point is that prefill is expensive and redundant. Two requests sharing a
4k-token system prompt compute byte-identical KV for those tokens, on every
replica, every time. A shared tier means it gets computed once.

## Status

Working: single-node store, binary wire protocol, cost-aware eviction, and a
RAM to NVMe tier. Not yet done: the vLLM connector, so **the tier has never
seen real KV, and time-to-first-token is unmeasured** — the metric the whole
design is justified by.

Measured so far, on synthetic traces:

| | |
|---|---|
| Dedup, 8 conversations sharing a prompt | 3.06 GiB served from 0.81 GiB stored |
| Hit rate at 5% of working set in RAM | 35.9% flat, 90.7% with the disk tier |
| Transfer throughput | 8.2–9.2 GB/s single client, ~15 GB/s aggregate (loopback, not a NIC) |
| Block naming | 108 ns/block; index probe 0.9 ns/block |

## How it works

A block is 16 tokens' worth of KV. Its name is a hash of *its entire token
history*, not just its own tokens, so two sequences share a name exactly when
their KV bytes are identical. Prefix matching becomes a hash table walk, and
deduplication falls out of the hash function.

```
block.rs   naming: prefix-chained BLAKE3, namespaced per model and layout
slab.rs    RAM tier: fixed slots over an anonymous mmap, pinnable
index.rs   name -> location, tree invariants, pins, eviction priorities
evict.rs   GreedyDual-Size: recency and recompute cost in one number
tier.rs    NVMe tier: the same slot discipline over a file
store.rs   lookup / admit / fetch / writeback
proto.rs   16-byte framed binary protocol
server.rs  tokio server; reads go slab -> socket with no staging buffer
client.rs  connection with a layout handshake
```

Two invariants shape most of the design, both from the fact that a block is
useless without every block before it: a block is only admitted when its
parent is resident, and only a leaf may be removed outright. Demotion to disk
is exempt from the second, since a demoted parent is still there for its
children.

## Build and test

Needs Rust 1.85+ (edition 2024).

```bash
cargo test
cargo bench          # lookup path
cargo run --release --bin kvtier       # workload demo, hit rate vs cache size
cargo run --release --bin kvtier-load  # throughput, cold fetch, writeback
```

## Running the daemon

```bash
cargo run --release --bin kvtierd
```

| Variable | Default | Meaning |
|---|---|---|
| `KVTIER_ADDR` | `127.0.0.1:7431` | listen address |
| `KVTIER_MODEL` | `llama-3-8b` | model id, part of the block namespace |
| `KVTIER_LAYOUT` | `llama3-8b` | block shape: `llama3-8b` or `tiny` |
| `KVTIER_BLOCKS` | `4096` | RAM tier capacity, in blocks |
| `KVTIER_DISK_BLOCKS` | `0` | disk tier capacity; 0 disables it |

The cache is in memory and the disk tier is unlinked at startup, so nothing
survives a restart of the daemon itself.

## Python bindings

The vLLM connector is Python, so the Rust client is exposed through PyO3
rather than reimplemented. That keeps one implementation of block naming —
a Python version that hashed even slightly differently would not error, it
would just never hit.

```bash
python3 -m venv .venv
.venv/bin/pip install maturin
VIRTUAL_ENV=$PWD/.venv .venv/bin/maturin develop -m kvtier-py/Cargo.toml --release
.venv/bin/python python/test_client.py
```

```python
import kvtier

client = kvtier.Client("127.0.0.1:7431")
hasher = client.hasher()          # built from the layout the server reported

names = hasher.chain(token_ids)   # one 16-byte name per full block
matched = client.match_prefix(names)
blocks = client.get_blocks(names[:matched])
client.put_blocks(parent, [(name, depth), ...], payload)
```
