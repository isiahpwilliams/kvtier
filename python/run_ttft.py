"""Driver: TTFT for each arm, across GPU KV cache sizes.

Three arms, each on a cold engine in its own process:

  baseline    no connector. vLLM's own prefix cache, which is the only
              honest thing to compare against.
  tier-cold   fresh kvtierd, one pass. The tier fills and reads in the same
              pass, so a conversation's turn t+1 can hit what its turn t
              stored, and nothing else is there.
  tier-warm   fresh kvtierd, filled by an untimed pass, then a *new* engine
              times a second pass. The GPU cache starts empty and the tier
              starts full: another replica's traffic, or yesterday's.

Each tier arm gets its own daemon, started empty and killed afterwards. An
earlier version of this script started one daemon for the whole sweep, which
quietly turned every run after the first into a warm-tier run and made the
sweep points incomparable.

The GPU cache sweep is the point. A tier cannot beat a GPU cache that already
holds the prefix, so the top of the sweep measures the tier's overhead and the
bottom measures what it buys.

Run with: .venv/bin/python python/run_ttft.py --help
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time


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


def start_daemon(model, layout, blocks, log):
    port = free_port()
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
        ["target/release/kvtierd"],
        env=env,
        stdout=handle,
        stderr=handle,
        # Its own session, so a shutting-down engine cannot take the daemon
        # with it if it signals its process group on the way out.
        start_new_session=True,
    )
    for _ in range(400):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.1).close()
            return daemon, f"127.0.0.1:{port}"
        except OSError:
            time.sleep(0.05)
    daemon.kill()
    raise RuntimeError("kvtierd did not come up")


def gpu_free_mib():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    return int(out.stdout.strip().splitlines()[0])


def wait_for_gpu(limit_mib=1024, timeout=120):
    """Block until the last engine has actually given the GPU back.

    subprocess.run returns when the driver process exits, but vLLM's EngineCore
    is a child of that and can still be holding several GiB. Starting the next
    engine then fails its own free-memory check, which looks like a flaky
    benchmark and is really just a race with teardown.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        used = gpu_free_mib()
        if used <= limit_mib:
            return True
        time.sleep(1)
    print(f"  GPU still holding {gpu_free_mib()} MiB after {timeout}s")
    return False


def run_engine(args, arm, addr, kv_bytes, out_path, log_path):
    cmd = [
        sys.executable, "python/bench_ttft.py",
        "--model", args.model,
        "--arm", "baseline" if arm == "baseline" else "tier",
        "--addr", addr,
        "--out", out_path,
        "--conversations", str(args.conversations),
        "--turns", str(args.turns),
        "--system-tokens", str(args.system_tokens),
        "--turn-tokens", str(args.turn_tokens),
        "--max-model-len", str(args.max_model_len),
        "--kv-cache-bytes", str(kv_bytes),
        "--gpu-fraction", str(args.gpu_fraction),
    ]
    env = {**os.environ, "PYTHONPATH": os.path.abspath("python")}
    # Two attempts. An engine can fail its own free-memory check because the
    # previous one has not finished handing the GPU back -- EngineCore is a
    # child of the process we waited on, so its teardown outlives that wait.
    # Retrying is cheaper and more honest than pretending the arm failed.
    for attempt in range(2):
        wait_for_gpu()
        with open(log_path, "w") as log:
            proc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        wait_for_gpu()
        if proc.returncode == 0:
            with open(out_path) as f:
                return json.load(f)
        if attempt == 0:
            print(f"  {arm} failed once, retrying")
            time.sleep(20)
    print(f"  {arm} FAILED, see {log_path}")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--conversations", type=int, default=16)
    parser.add_argument("--turns", type=int, default=4)
    parser.add_argument("--system-tokens", type=int, default=1800)
    parser.add_argument("--turn-tokens", type=int, default=400)
    parser.add_argument("--max-model-len", type=int, default=3584)
    parser.add_argument("--tier-blocks", type=int, default=8192)
    parser.add_argument("--kv-cache-mib", type=int, nargs="+",
                        default=[256, 1024, 4096])
    parser.add_argument("--results", default="bench_results")
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="leave arms that already have a result, to top up a partial sweep",
    )
    parser.add_argument(
        "--gpu-fraction", type=float, default=0.55,
        help="only a startup sanity check here, since kv_cache_memory_bytes "
             "is set explicitly; kept low so a slow teardown is not fatal",
    )
    args = parser.parse_args()

    os.makedirs(args.results, exist_ok=True)
    layout = model_layout(args.model, args.dtype)
    block_bytes = (
        2 * layout["layers"] * 16 * layout["kv_heads"] * layout["head_dim"] * 2
    )
    print(f"model {args.model}")
    print(f"layout {layout}, block {block_bytes / 1024:.0f} KiB")

    arms = ("baseline", "tier-cold", "tier-warm")
    summary = {}
    for rep in range(args.repeat):
        for mib in args.kv_cache_mib:
            kv_bytes = mib * 2**20
            print(f"\n=== repeat {rep}, GPU KV cache {mib} MiB "
                  f"({kv_bytes // block_bytes} blocks) ===")
            for arm in arms:
                tag = f"{arm}_{mib}mib_r{rep}"
                done_path = os.path.join(args.results, f"{tag}.json")
                if args.skip_existing and os.path.exists(done_path):
                    with open(done_path) as f:
                        summary[tag] = json.load(f)["summary"]
                    print(f"  {arm:10s} already have it")
                    continue
                daemon, addr = None, "127.0.0.1:1"
                try:
                    if arm != "baseline":
                        daemon, addr = start_daemon(
                            args.model, layout, args.tier_blocks,
                            os.path.join(args.results, f"{tag}_daemon.log"),
                        )
                    if arm == "tier-warm":
                        # Untimed pass to fill the tier, then a cold engine.
                        run_engine(
                            args, arm, addr, kv_bytes,
                            os.path.join(args.results, f"{tag}_fill.json"),
                            os.path.join(args.results, f"{tag}_fill.log"),
                        )
                        # The timed pass needs the daemon the fill pass filled.
                        # If it is gone, say so: a restarted daemon would be
                        # empty and this would quietly become a cold-tier run.
                        code = daemon.poll()
                        if code is not None:
                            print(f"  {arm} daemon died after the fill pass, "
                                  f"exit {code}")
                    result = run_engine(
                        args, arm, addr, kv_bytes,
                        os.path.join(args.results, f"{tag}.json"),
                        os.path.join(args.results, f"{tag}.log"),
                    )
                finally:
                    if daemon is not None:
                        daemon.terminate()
                        daemon.wait(timeout=10)
                if result is None:
                    continue
                s = result["summary"]
                summary[tag] = s
                hits = ""
                if "tier_stats" in result:
                    st = result["tier_stats"]
                    rate = st["hit_blocks"] / max(1, st["queried_blocks"]) * 100
                    hits = f"  [tier {rate:.0f}% of {st['queried_blocks']} queried]"
                print(f"  {arm:10s} median {s['median'] * 1000:8.1f} ms   " +
                      "  ".join(f"t{t}:{v['median'] * 1000:.0f}"
                                for t, v in s["by_turn"].items()) + hits)

    with open(os.path.join(args.results, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)

    print("\n=== TTFT median, ms (repeats listed; spread is the noise floor) ===")
    header = f"{'GPU KV cache':>13}"
    for arm in arms:
        header += f"  {arm:>16}"
    print(header + f"  {'cold vs base':>13}  {'warm vs base':>13}")
    for mib in args.kv_cache_mib:
        cells, best = [], {}
        for arm in arms:
            vals = [summary[f"{arm}_{mib}mib_r{r}"]["median"] * 1000
                    for r in range(args.repeat)
                    if f"{arm}_{mib}mib_r{r}" in summary]
            if not vals:
                cells.append(f"{'-':>16}")
                continue
            best[arm] = min(vals)
            cells.append(f"{'/'.join(f'{v:.1f}' for v in vals):>16}")
        line = f"{mib:>9} MiB" + "  ".join([""] + cells)
        for arm in ("tier-cold", "tier-warm"):
            if arm in best and "baseline" in best:
                delta = (best[arm] - best["baseline"]) / best["baseline"] * 100
                line += f"  {delta:+12.1f}%"
        print(line)
    print("RUN_TTFT_DONE")


if __name__ == "__main__":
    sys.exit(main() or 0)
