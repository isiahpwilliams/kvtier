# kvtier

A KV cache tier for LLM inference engines. It sits underneath vLLM or SGLang
rather than replacing them: the engine asks how much of a prompt's KV cache
already exists somewhere, fetches what does, and hands back what it had to
compute. Written in Rust.

The point is that prefill is expensive and redundant. Two requests sharing a
4k-token system prompt compute byte-identical KV for those tokens, on every
replica, every time. A shared tier means it gets computed once.

## Status

Working: single-node store, binary wire protocol, cost-aware eviction, a RAM
to NVMe tier, and a vLLM connector. TTFT is measured.

### Time to first token

Qwen2.5-7B-Instruct on one A100-40GB, vLLM 0.28.0. 16 conversations of 4
turns each, interleaved round-robin so a conversation's blocks are pushed out
of GPU memory before it comes back; median prompt 2660 tokens. Every arm runs
with vLLM's own prefix caching on, which is the only honest baseline. Median
of 64 requests, three repeats, each on a cold engine in its own process.

`cold` fills and reads in one pass. `warm` is a tier another pass already
filled, read by a fresh engine: another replica's traffic, or yesterday's.

| GPU KV cache | baseline | tier, cold | tier, warm |
|---|---|---|---|
| 256 MiB (18% of working set) | 88.8 ms | **-10.0%** | **-36.6%** |
| 1024 MiB (73%) | 48.0 ms | +35.0% | **-14.4%** |
| 4096 MiB (293%) | 48.0 ms | +30.7% | **-13.8%** |

By turn, at 256 MiB -- the baseline's cost climbs with the conversation, a
warm tier's barely does:

| | turn 0 | turn 1 | turn 2 | turn 3 |
|---|---|---|---|---|
| baseline | 31.3 | 55.0 | 89.1 | 112.5 |
| tier, warm | 33.5 | 40.8 | 48.1 | 67.2 |

A warm tier is the result, and it is the advertised one: computed once,
reused everywhere. It wins wherever it is asked, and wins biggest where the
GPU cannot hold the working set.

A cold tier is the caveat. It pays to store blocks before anything comes back
for them, so it only comes out ahead when the GPU cache is too small to
compete -- 10% better at 256 MiB, 31% worse at 4096 MiB. Switching the tier on
is not free, and on a workload that fits in GPU memory it is a loss until
something reuses what it stored.

Read the spreads before quoting any of this. The baseline repeats within 0.4%
and the tier arms do not: cold at 1024 MiB measured 64.8/50.2/65.0 ms across
three runs, so its +35% is the median of a distribution wide enough to contain
a much smaller number. The likeliest cause is the save queue -- when every
staging slab is in flight the next gather blocks, and the cost lands back on
the forward pass -- but that is a hypothesis, not a measurement.

Async saves are what make the cold column survivable at all. Writing blocks
inside the forward pass instead costs another 10-18%; at 256 MiB the same
trace measured 89.1/89.0 ms with no tier, 98.7/94.5 with synchronous saves,
and 80.7/85.6 with asynchronous ones.

### Two engines, one tier

The premise is that KV computed on one replica is worth something on another,
and until now every measurement here was a single engine talking to a single
daemon, where the tier is really just a bigger cache for that one engine.

Engine A runs, exits, and engine B starts against the same daemon with
*different* users behind the same system prompt. A and B share nothing else,
so a block B fetches for that prompt was computed by A and by no one else.

| | median | first request |
|---|---|---|
| engine A, empty tier | 38.6 ms | 163.3 ms |
| engine B, A's tier | 43.4 ms | **69.1 ms** |
| control, no tier | 31.2 ms | 158.1 ms |

B's first request costs 69 ms against the control's 158, on a prompt it had
never seen, from KV it never computed: 1921 block hits against the 112 blocks
of shared prompt, and B added only 96 blocks of its own. So the premise holds.

B's *median* is worse than the control's, and that matters more than it looks.
After the first request every arm's own GPU cache holds the shared prompt, so
the rest are local hits for everybody and B is still paying a lookup per
request and storing its new blocks. The median is measuring overhead.

Which sharpens the claim rather than supporting it: the tier turns a cold start
into a warm one. It is worth having for a replica that has just started, a
working set larger than GPU memory, or traffic that moves between replicas. It
is not worth having in front of one engine whose cache already fits its
workload.

### Transfer against recompute

Where fetching a prefix beats recomputing it, same model and machine.
Recompute is prefill minus the fixed per-request engine cost, which both arms
pay.

| prefix | fetch | recompute | |
|---|---|---|---|
| 256 tokens | 5.9 ms | 2.1 ms | recompute |
| 512 tokens | 10.0 ms | 17.4 ms | tier, 1.7x |
| 2048 tokens | 47.1 ms | 109.3 ms | tier, 2.3x |
| 4096 tokens | 89.8 ms | ~230 ms | tier, 2.6x |

**The crossover is around 350-400 tokens**, or 22-25 blocks. Below it, asking
is more expensive than recomputing.

The gap widens only slowly above it: 1.7x at 512 tokens to 2.6x at 4096. The
original design note expected prefill's superlinearity to open that gap much
faster. At these lengths, for a 7B model, attention is still a small part of
prefill and recompute is nearly linear too, so a ~2x win is what is on the
table rather than an order of magnitude.

### Other measurements

| | |
|---|---|
| Dedup, 8 conversations sharing a prompt | 3.06 GiB served from 0.81 GiB stored |
| Hit rate at 5% of working set in RAM | 35.9% flat, 90.7% with the disk tier |
| Fetch into a page-locked slab | 2.5–3.4 GiB/s over loopback |
| Host to device, once page-locked | 22 GiB/s |
| Block naming | 108 ns/block; index probe 0.9 ns/block |

The transfer number is close to this machine's ceiling: a bare loopback socket
here carries 2.36 GiB/s on one stream. An earlier revision of this file quoted
8.2–9.2 GB/s single client; that is not reachable here and was measured
somewhere else.

## How it works

A block is 16 tokens' worth of KV. Its name is a hash of *its entire token
history*, not just its own tokens, so two sequences share a name exactly when
their KV bytes are identical. Prefix matching becomes a hash table walk, and
deduplication falls out of the hash function.

```
block.rs   naming: prefix-chained BLAKE3, namespaced per model, layout,
           byte order and TP shard
slab.rs    RAM tier: fixed slots over an anonymous mmap, pinnable
index.rs   name -> location, tree invariants, pins, eviction priorities
evict.rs   GreedyDual-Size: recency and recompute cost in one number
tier.rs    NVMe tier: the same slot discipline over a file
store.rs   lookup / admit / fetch / writeback
proto.rs   16-byte framed binary protocol
server.rs  tokio server; reads go slab -> socket with no staging buffer
client.rs  connection with a layout handshake
```

```
python/kvtier_connector.py   the vLLM KV connector
```

A block's name covers its byte order and its tensor-parallel shard as well as
its dimensions. Two connectors can agree on every dimension and still write
the bytes in a different order; without the order in the digest they would
share a name and quietly serve each other the wrong KV. With it they simply
miss. Shards are per-rank: a rank stores only its own heads, so the cache is
reusable at the same TP degree and not across degrees.

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
| `KVTIER_LAYOUT` | `llama3-8b` | block shape: `llama3-8b`, `tiny`, or `custom` |
| `KVTIER_LAYERS` etc | — | with `custom`: `KVTIER_TOKENS_PER_BLOCK`, `KVTIER_LAYERS`, `KVTIER_KV_HEADS`, `KVTIER_HEAD_DIM`, `KVTIER_DTYPE` |
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

`fetch_into` and `put_from` move KV through a buffer the caller owns, rather
than through Python `bytes`. The connector page-locks that buffer, so the
copy to the GPU is a DMA.

```python
slab = torch.empty(n * client.block_bytes, dtype=torch.uint8)
torch.cuda.cudart().cudaHostRegister(slab.data_ptr(), slab.numel(), 3)

got = client.fetch_into(names, slab.numpy())      # blocks written into slab
client.put_from(parent, [(name, depth), ...], slab.numpy())
```

## The vLLM connector

```bash
# One daemon, sized for the model. The layout must match the engine's
# exactly, or the connect handshake refuses the client.
KVTIER_LAYOUT=custom KVTIER_LAYERS=28 KVTIER_KV_HEADS=4 KVTIER_HEAD_DIM=128 \
  KVTIER_DTYPE=bf16 KVTIER_MODEL=Qwen/Qwen2.5-7B-Instruct KVTIER_BLOCKS=8192 \
  cargo run --release --bin kvtierd
```

```python
from vllm import LLM
from vllm.config import KVTransferConfig

llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    kv_transfer_config=KVTransferConfig(
        kv_connector="KvtierConnector",
        kv_connector_module_path="kvtier_connector",   # PYTHONPATH=python/
        kv_role="kv_both",
        kv_connector_extra_config={"kvtier_address": "127.0.0.1:7431"},
    ),
    disable_hybrid_kv_cache_manager=True,   # the connector is not HMA-aware
)
```

The connector forces vLLM's KV layout to NHD, because the byte order inside a
block is part of the namespace digest and must not depend on which backend the
engine picked. It handles one full-attention KV cache group; hybrid models are
out of scope.

`KVTIER_VERIFY=1` turns on save- and load-side self checks that compare the
staging slab against the paged buffer directly. `KVTIER_INERT=1` keeps the
connector installed but makes it serve and store nothing, which is the control
for telling "the tier moved KV" apart from "a connector was present".

## Benchmarks

```bash
.venv/bin/python python/test_layout.py        # KV layout translation
.venv/bin/python python/test_connector.py     # tier KV == reused KV, end to end
.venv/bin/python python/run_ttft.py           # TTFT, all arms, cache sweep
.venv/bin/python python/report_ttft.py bench_results
.venv/bin/python python/bench_transfer.py --layers 28 --kv-heads 4 --head-dim 128
.venv/bin/python python/bench_prefill.py      # the recompute side of the crossover
```

A warning worth keeping. `test_connector.py` compares KV from the tier against
KV that vLLM *reused*, not against KV it *recomputed*. Those two are not
bitwise equal — attention over a cached prefix runs a different kernel shape
than attention that builds it — and on a near-tied logit the argmax flips. On
that prompt set it flips one request in four, with no connector present at
all. Checking a connector against recomputed tokens makes a correct one look
broken.
