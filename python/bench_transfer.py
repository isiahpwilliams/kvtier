"""Where fetching KV from the tier becomes cheaper than recomputing it.

Two curves, both measured rather than modelled:

  transfer(n)  fetch n blocks from a live kvtierd into page-locked host memory
               and scatter them into a paged KV buffer on the GPU -- exactly
               what the connector's load path does, minus vLLM.

  prefill(n)   an engine computing n * block_size tokens of prompt with an
               empty cache, which is what the tier saves you.

The crossover is the prefix length below which the tier is not worth asking.
Transfer is linear in tokens; prefill is superlinear, so above the crossover
the gap widens.

The paged buffer here is allocated with vLLM's own shape and stride order, so
the scatter costs what it costs in the engine. It is not an engine, though:
there is no attention, no sampling, no scheduler.

Run with: .venv/bin/python python/bench_transfer.py --help
"""

import argparse
import json
import statistics
import sys
import time

import torch

import kvtier


def paged_buffers(num_blocks, block_size, num_kv_heads, head_size, layers, dtype):
    """One tensor per layer, shaped and strided the way vLLM allocates them."""
    from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
    from vllm.v1.attention.backends.utils import set_kv_cache_layout

    set_kv_cache_layout("NHD")
    shape = FlashAttentionBackend.get_kv_cache_shape(
        num_blocks, block_size, num_kv_heads, head_size
    )
    order = FlashAttentionBackend.get_kv_cache_stride_order()
    inverse = [order.index(i) for i in range(len(order))]

    tensors = []
    for _ in range(layers):
        physical = torch.zeros(
            [shape[i] for i in order], dtype=dtype, device="cuda"
        )
        tensors.append(physical.permute(*inverse))
    return tensors


def block_views(tensors):
    views = []
    for t in tensors:
        per_block = t.stride(0)
        assert per_block == t[0].numel() == max(t.stride())
        views.append(t.as_strided((t.shape[0], per_block), (per_block, 1)))
    return views


def time_it(fn, repeats, warmup=3):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - start)
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--addr", default="127.0.0.1:7431")
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--kv-heads", type=int, required=True)
    parser.add_argument("--head-dim", type=int, required=True)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--pool-blocks", type=int, default=4096)
    parser.add_argument("--slab-blocks", type=int, default=512)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--pageable",
        action="store_true",
        help="skip cudaHostRegister, to show what the page-lock is worth",
    )
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    client = kvtier.Client(args.addr)
    hasher = client.hasher()
    block_bytes = client.block_bytes

    expected = (
        2 * args.layers * args.block_size * args.kv_heads * args.head_dim
        * dtype.itemsize
    )
    if block_bytes != expected:
        raise SystemExit(
            f"kvtierd holds {block_bytes} B blocks, this layout needs {expected} B"
        )

    tensors = paged_buffers(
        args.pool_blocks, args.block_size, args.kv_heads, args.head_dim,
        args.layers, dtype,
    )
    views = block_views(tensors)
    elems = views[0].shape[1]

    # Page-locked staging, the same slab the connector uses.
    slab = torch.empty(args.slab_blocks * block_bytes, dtype=torch.uint8)
    locked = False
    if not args.pageable:
        status = torch.cuda.cudart().cudaHostRegister(slab.data_ptr(), slab.numel(), 3)
        locked = int(status) == 0
    print(f"slab {slab.numel() / 2**20:.0f} MiB, page-locked={locked}")

    # Fill the tier with a chain long enough for the largest fetch.
    tokens = list(range(3, 3 + args.slab_blocks * args.block_size))
    names = hasher.chain(tokens)
    for view in views:
        view.random_(0, 100)
    depths = [(n, (i + 1) * args.block_size) for i, n in enumerate(names)]
    ids = torch.arange(len(names), device="cuda")
    staged_all = (
        slab[: len(names) * block_bytes]
        .view(dtype)
        .view(len(names), len(views), elems)
    )
    staged_all.copy_(torch.stack([v.index_select(0, ids) for v in views], dim=1))
    torch.cuda.synchronize()
    inserted, deduped, dropped = client.put_from(None, depths, slab.numpy())
    print(f"seeded the tier: {inserted} inserted, {deduped} deduped, {dropped} dropped")
    if inserted + deduped < len(names):
        raise SystemExit("could not seed the tier")

    counts = [n for n in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512) if n <= len(names)]
    results = []
    for n in counts:
        want = names[:n]
        target = torch.arange(n, device="cuda")
        staged = slab[: n * block_bytes].view(dtype).view(n, len(views), elems)

        def wire():
            got = client.fetch_into(want, slab.numpy())
            assert got == n, got

        def scatter():
            resident = staged.to("cuda", non_blocking=True)
            for layer, view in enumerate(views):
                view.index_copy_(0, target, resident[:, layer, :])

        def load():
            wire()
            scatter()

        moved = n * block_bytes
        on_wire = statistics.median(time_it(wire, args.repeats))
        on_gpu = statistics.median(time_it(scatter, args.repeats))
        samples = time_it(load, args.repeats)
        median = statistics.median(samples)
        results.append(
            {
                "blocks": n,
                "tokens": n * args.block_size,
                "bytes": moved,
                "median_s": median,
                "min_s": min(samples),
                "max_s": max(samples),
                "spread": (max(samples) - min(samples)) / median,
                "wire_s": on_wire,
                "scatter_s": on_gpu,
                "gib_per_s": moved / median / 2**30,
                "wire_gib_per_s": moved / on_wire / 2**30,
                "scatter_gib_per_s": moved / on_gpu / 2**30,
            }
        )
        print(
            f"  {n:4d} blocks ({n * args.block_size:6d} tok, {moved / 2**20:7.1f} MiB): "
            f"total {median * 1000:7.2f} ms = wire {on_wire * 1000:7.2f} "
            f"({moved / on_wire / 2**30:5.2f} GiB/s) + gpu {on_gpu * 1000:6.2f} "
            f"({moved / on_gpu / 2**30:5.2f} GiB/s)"
        )

    out = {
        "config": vars(args),
        "page_locked": locked,
        "block_bytes": block_bytes,
        "transfer": results,
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
    print("TRANSFER_DONE")


if __name__ == "__main__":
    sys.exit(main() or 0)
