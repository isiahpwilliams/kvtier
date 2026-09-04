"""Driver: run the TTFT benchmark for both arms, across GPU cache sizes.

Starts a kvtierd sized for the model, then runs bench_ttft.py once per arm per
GPU KV cache size, each in its own process so every arm gets a cold engine.

The GPU cache sweep is the point. A tier cannot beat a GPU cache that already
holds the prefix, so a single large-cache run would only measure the tier's
overhead. Sweeping down through the working set shows where it starts paying,
and the top of the sweep is kept precisely so the overhead is visible too.

Run with: .venv/bin/python python/run_ttft.py --help
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time

import torch


def model_layout(model, dtype_name):
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model)
    heads = config.num_attention_heads
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // heads
    return {
        "layers": config.num_hidden_layers,
        "kv_heads": getattr(config, "num_key_value_heads", heads),
        "head_dim": head_dim,
        "dtype": dtype_name,
    }


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def start_daemon(port, model, layout, blocks, log):
    env = {
        **os.environ,
        "KVTIER_ADDR": f"127.0.0.1:{port}",
        "KVTIER_MODEL": model,
        "KVTIER_LAYOUT": "custom",
        "KVTIER_TOKENS_PER_BLOCK": "16",
        "KVTIER_LAYERS": str(layout["layers"]),
        "KVTIER_KV_HEADS": str(layout["kv_heads"]),
        "KVTIER_HEAD_DIM": str(layout["head_dim"]),
        "KVTIER_DTYPE": layout["dtype"],
        "KVTIER_BLOCKS": str(blocks),
    }
    handle = open(log, "w")
    daemon = subprocess.Popen(
        ["target/release/kvtierd"], env=env, stdout=handle, stderr=handle
    )
    for _ in range(400):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.1).close()
            return daemon
        except OSError:
            time.sleep(0.05)
    daemon.kill()
    raise RuntimeError("kvtierd did not come up")


def run_arm(args, arm, addr, kv_bytes, out_path, log_path):
    cmd = [
        sys.executable, "python/bench_ttft.py",
        "--model", args.model,
        "--arm", arm,
        "--addr", addr,
        "--out", out_path,
        "--conversations", str(args.conversations),
        "--turns", str(args.turns),
        "--system-tokens", str(args.system_tokens),
        "--turn-tokens", str(args.turn_tokens),
        "--max-model-len", str(args.max_model_len),
        "--kv-cache-bytes", str(kv_bytes),
    ]
    env = {**os.environ, "PYTHONPATH": os.path.abspath("python")}
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(f"  {arm} FAILED, see {log_path}")
        return None
    with open(out_path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--conversations", type=int, default=16)
    parser.add_argument("--turns", type=int, default=4)
    parser.add_argument("--system-tokens", type=int, default=1800)
    parser.add_argument("--turn-tokens", type=int, default=400)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--tier-blocks", type=int, default=32768)
    parser.add_argument(
        "--kv-cache-mib", type=int, nargs="+", default=[256, 512, 1024, 4096]
    )
    parser.add_argument("--results", default="bench_results")
    args = parser.parse_args()

    os.makedirs(args.results, exist_ok=True)
    layout = model_layout(args.model, args.dtype)
    block_bytes = (
        2 * layout["layers"] * 16 * layout["kv_heads"] * layout["head_dim"]
        * (2 if args.dtype in ("f16", "bf16") else 4)
    )
    print(f"model {args.model}")
    print(f"layout {layout}, block {block_bytes / 1024:.0f} KiB")
    print(f"tier capacity {args.tier_blocks} blocks = "
          f"{args.tier_blocks * block_bytes / 2**30:.1f} GiB")

    port = free_port()
    addr = f"127.0.0.1:{port}"
    daemon = start_daemon(
        port, args.model, layout, args.tier_blocks,
        os.path.join(args.results, "kvtierd.log"),
    )
    print(f"kvtierd on {addr}")

    summary = {}
    try:
        for mib in args.kv_cache_mib:
            kv_bytes = mib * 2**20
            print(f"\n=== GPU KV cache {mib} MiB "
                  f"({kv_bytes // block_bytes} blocks) ===")
            for arm in ("baseline", "tier"):
                tag = f"{arm}_{mib}mib"
                result = run_arm(
                    args, arm, addr, kv_bytes,
                    os.path.join(args.results, f"{tag}.json"),
                    os.path.join(args.results, f"{tag}.log"),
                )
                if result is None:
                    continue
                s = result["summary"]
                summary[tag] = s
                print(f"  {arm:9s} median {s['median'] * 1000:8.1f} ms   " +
                      "  ".join(
                          f"t{t}:{v['median'] * 1000:.0f}"
                          for t, v in s["by_turn"].items()
                      ))
    finally:
        daemon.terminate()
        daemon.wait(timeout=10)

    with open(os.path.join(args.results, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)

    print("\n=== TTFT median, ms ===")
    print(f"{'GPU KV cache':>14}  {'baseline':>9}  {'tier':>9}  {'change':>9}")
    for mib in args.kv_cache_mib:
        b = summary.get(f"baseline_{mib}mib")
        t = summary.get(f"tier_{mib}mib")
        if not b or not t:
            continue
        change = (t["median"] - b["median"]) / b["median"] * 100
        print(f"{mib:>10} MiB  {b['median'] * 1000:9.1f}  "
              f"{t['median'] * 1000:9.1f}  {change:+8.1f}%")
    print("RUN_TTFT_DONE")


if __name__ == "__main__":
    sys.exit(main() or 0)
