"""Time-to-first-token on a multi-turn trace, with and without the tier.

The trace is the point. A single conversation replayed into a warm engine
proves nothing: vLLM's own prefix cache already holds the prefix, and the tier
can only add a lookup. The tier earns its keep when the GPU cache does *not*
have the prefix, which on a real server means the working set is larger than
GPU memory. So the trace interleaves many conversations round-robin by turn:
by the time a conversation comes back for its next turn, the others have
pushed its blocks out.

Assistant turns are synthetic rather than generated, so both arms replay a
byte-identical trace. A trace that differed between arms would not be a
comparison.

TTFT is measured as wall time around a single request with max_tokens=1, one
request at a time. That is prefill plus one decode step plus a fixed amount of
engine overhead. The overhead is the same in both arms, so the difference is
real even though the absolute number is a slight overestimate.

Run with: .venv/bin/python python/bench_ttft.py --help
"""

import argparse
import json
import math
import os
import statistics
import sys
import time


def build_trace(num_conversations, turns, system_tokens, turn_tokens, tokenizer):
    """Round-robin multi-turn prompts sharing one long system prompt."""
    word = " context"

    def filler(n, salt):
        # Distinct text per conversation, so only the system prompt is shared.
        return f" {salt}" + word * n

    system = "You are a meticulous assistant." + filler(system_tokens, "sys")
    histories = [system for _ in range(num_conversations)]
    trace = []
    for turn in range(turns):
        for conv in range(num_conversations):
            histories[conv] += (
                f"\n\nUser (conversation {conv}, turn {turn}):"
                + filler(turn_tokens // 2, f"u{conv}x{turn}")
                + f"\n\nAssistant:"
            )
            trace.append((conv, turn, histories[conv]))
            histories[conv] += filler(turn_tokens // 2, f"a{conv}x{turn}")

    lengths = [len(tokenizer(p).input_ids) for _, _, p in trace]
    return trace, lengths


def make_engine(args, use_tier, addr):
    from vllm import LLM
    from vllm.config import KVTransferConfig

    kwargs = {}
    if use_tier:
        kwargs["kv_transfer_config"] = KVTransferConfig(
            kv_connector="KvtierConnector",
            kv_connector_module_path="kvtier_connector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "kvtier_address": addr,
                "kvtier_slab_blocks": args.slab_blocks,
            },
        )
    return LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        kv_cache_memory_bytes=args.kv_cache_bytes,
        gpu_memory_utilization=args.gpu_fraction,
        enforce_eager=True,
        disable_hybrid_kv_cache_manager=True,
        enable_prefix_caching=not args.no_prefix_cache,
        **kwargs,
    )


def run_arm(args, use_tier, addr):
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    trace, lengths = build_trace(
        args.conversations, args.turns, args.system_tokens, args.turn_tokens, tokenizer
    )

    llm = make_engine(args, use_tier, addr)
    sampling = SamplingParams(max_tokens=1, temperature=0)

    # The first request in a process pays for lazy init; do not time it.
    llm.generate(["warm up the engine please"], sampling)

    records = []
    for (conv, turn, prompt), n_tokens in zip(trace, lengths):
        start = time.perf_counter()
        llm.generate([prompt], sampling, use_tqdm=False)
        elapsed = time.perf_counter() - start
        records.append(
            {"conv": conv, "turn": turn, "tokens": n_tokens, "ttft": elapsed}
        )
    return records


def quantile(values, q):
    """Nearest-rank quantile. Small samples make interpolation a lie."""
    ordered = sorted(values)
    rank = max(1, math.ceil(q * len(ordered)))
    return ordered[rank - 1]


def summarize(records):
    by_turn = {}
    for r in records:
        by_turn.setdefault(r["turn"], []).append(r["ttft"])
    everything = [r["ttft"] for r in records]
    return {
        "n": len(records),
        "median": statistics.median(everything),
        "mean": statistics.fmean(everything),
        "p25": quantile(everything, 0.25),
        "p75": quantile(everything, 0.75),
        "p90": quantile(everything, 0.90),
        "by_turn": {
            turn: {
                "median": statistics.median(v),
                "p25": quantile(v, 0.25),
                "p75": quantile(v, 0.75),
                "p90": quantile(v, 0.90),
                "n": len(v),
            }
            for turn, v in sorted(by_turn.items())
        },
        "median_tokens": statistics.median(r["tokens"] for r in records),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--arm", choices=["baseline", "tier"], required=True)
    parser.add_argument("--addr", default="127.0.0.1:7431")
    parser.add_argument("--out", required=True)
    parser.add_argument("--conversations", type=int, default=16)
    parser.add_argument("--turns", type=int, default=4)
    parser.add_argument("--system-tokens", type=int, default=1800)
    parser.add_argument("--turn-tokens", type=int, default=400)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--kv-cache-bytes", type=int, default=None)
    parser.add_argument("--gpu-fraction", type=float, default=0.90)
    parser.add_argument("--slab-blocks", type=int, default=256)
    parser.add_argument("--no-prefix-cache", action="store_true")
    args = parser.parse_args()

    records = run_arm(args, args.arm == "tier", args.addr)
    result = {
        "arm": args.arm,
        "config": vars(args),
        "records": records,
        "summary": summarize(records),
    }
    if args.arm == "tier":
        import kvtier

        result["tier_stats"] = kvtier.Client(args.addr).stats()

    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)

    s = result["summary"]
    print(f"{args.arm}: median TTFT {s['median'] * 1000:.1f} ms over {s['n']} requests")
    for turn, stats in s["by_turn"].items():
        print(f"  turn {turn}: median {stats['median'] * 1000:7.1f} ms  "
              f"iqr {stats['p25'] * 1000:.1f}-{stats['p75'] * 1000:.1f}  "
              f"p90 {stats['p90'] * 1000:7.1f} ms  n={stats['n']}")
    print("BENCH_DONE")


if __name__ == "__main__":
    sys.exit(main() or 0)
