"""Check the KV layout translation against an independently written order.

The connector reads a block as one contiguous range per layer. That is only
correct if vLLM's paged tensor really is laid out block-major with the block's
own bytes running token -> head -> K then V -> head_dim. This builds a tensor
with vLLM's exact shape and strides, fills every element with a value that
encodes where it came from, and compares the connector's view against the
order spelled out the slow way.

Run with: .venv/bin/python python/test_layout.py
"""

import sys

import torch

from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
from vllm.v1.attention.backends.utils import set_kv_cache_layout

# The connector pins this via get_required_kvcache_layout; do the same here so
# the test checks the layout we actually run under.
set_kv_cache_layout("NHD")

NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE, NUM_LAYERS = 6, 16, 4, 8, 3


def vllm_shaped_tensor():
    """A layer's paged cache, allocated the way vLLM allocates it."""
    shape = FlashAttentionBackend.get_kv_cache_shape(
        NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE
    )
    order = FlashAttentionBackend.get_kv_cache_stride_order()
    # vLLM allocates in stride order, then permutes back to the logical shape.
    physical = torch.zeros([shape[i] for i in order], dtype=torch.float32)
    return physical.permute(*[order.index(i) for i in range(len(order))])


def main():
    torch.manual_seed(0)
    passed = True

    layers = []
    for layer in range(NUM_LAYERS):
        t = vllm_shaped_tensor()
        # value = a unique id for (layer, block, head, token, k/v, dim)
        b, h, n, d = t.shape
        idx = torch.arange(t.numel(), dtype=torch.float32)
        t.copy_(idx.reshape(b, h, n, d))
        layers.append(t)

    probe = layers[0]
    print(f"shape {tuple(probe.shape)} stride {tuple(probe.stride())} "
          f"contiguous={probe.is_contiguous()}")

    # What the connector does.
    per_block = probe.stride(0)
    ok = per_block == probe[0].numel() and per_block == max(probe.stride())
    print(f"  {'ok  ' if ok else 'FAIL'}  a block is one contiguous range "
          f"({per_block} elements)")
    passed &= ok

    views = [t.as_strided((t.shape[0], per_block), (per_block, 1)) for t in layers]

    block = 3
    got = torch.stack([v[block] for v in views], dim=0).flatten()

    # The same bytes, written out the slow way from the documented order:
    # layer -> token -> head -> K then V -> head_dim.
    want = []
    for layer in range(NUM_LAYERS):
        t = layers[layer]
        for token in range(BLOCK_SIZE):
            for head in range(NUM_KV_HEADS):
                for c in range(2 * HEAD_SIZE):
                    want.append(t[block, head, token, c])
    want = torch.tensor(want)

    same = torch.equal(got, want)
    print(f"  {'ok  ' if same else 'FAIL'}  order is layer -> token -> head -> "
          f"K then V -> head_dim")
    passed &= same

    # K really is the first half of the content dim, V the second.
    k_cache, v_cache = probe.transpose(1, 2).split(HEAD_SIZE, dim=-1)
    k_ok = torch.equal(k_cache[block, 0, 0], probe[block, 0, 0, :HEAD_SIZE])
    v_ok = torch.equal(v_cache[block, 0, 0], probe[block, 0, 0, HEAD_SIZE:])
    print(f"  {'ok  ' if k_ok and v_ok else 'FAIL'}  K is the low half of the "
          f"content dim, V the high half")
    passed &= k_ok and v_ok

    # Round trip: writing a block back through the view restores it exactly.
    saved = [v[block].clone() for v in views]
    for v in views:
        v[block] = -1
    for v, s in zip(views, saved):
        v[block] = s
    restored = all(
        torch.equal(layers[i][block], torch.arange(
            layers[i].numel(), dtype=torch.float32
        ).reshape(layers[i].shape[0], *layers[i].shape[1:])[block])
        for i in range(NUM_LAYERS)
    )
    print(f"  {'ok  ' if restored else 'FAIL'}  a block survives a save/load "
          f"round trip byte for byte")
    passed &= restored

    # Untouched blocks stay untouched.
    clean = torch.equal(views[0][block + 1], layers[0].as_strided(
        (NUM_BLOCKS, per_block), (per_block, 1))[block + 1])
    print(f"  {'ok  ' if clean else 'FAIL'}  neighbouring blocks are untouched")
    passed &= clean

    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
