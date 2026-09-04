"""Two engines, one tier: does the second one reuse KV it never computed?

This is the claim the whole design rests on -- "two requests sharing a 4k
system prompt compute byte-identical KV on every replica, every time; a shared
tier means it gets computed once" -- and nothing so far has tested it. Every
other benchmark is one engine talking to one daemon, where the tier is really
just a bigger cache for that engine.

Three runs, each a separate process with its own cold GPU:

  A        engine A, tier on, prompts  S + a_i
  B        engine B, tier on, prompts  S + b_i     (different users, same S)
  control  no tier at all, prompts     S + b_i

A and B share only the system prompt S. So any block B fetches for S was
computed by A and by nobody else: B never sees a_i, and its own b_i are
distinct. If B's hits come back at zero, the premise does not hold.

The engines run one after another rather than at once. Two engines on one GPU
would contend for compute and memory, and this is a question about sharing
blocks, not about scheduling.

Run with: .venv/bin/python python/test_replicas.py
"""

import json
import os
import socket
import statistics
import subprocess
import sys
import time

MODEL = os.environ.get("KVTIER_TEST_MODEL", "Qwen/Qwen2.5-7B-Instruct")
LAYOUT = dict(layers=28, kv_heads=4, head_dim=128, dtype="bf16")
SYSTEM_TOKENS = 1800
USER_TOKENS = 200
CONVERSATIONS = 8

WORKER = r'''
import json, os, sys, time
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

role, addr, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
system_tokens = int(os.environ["SYSTEM_TOKENS"])
user_tokens = int(os.environ["USER_TOKENS"])
conversations = int(os.environ["CONVERSATIONS"])

kw = {}
if role != "control":
    kw["kv_transfer_config"] = KVTransferConfig(
        kv_connector="KvtierConnector",
        kv_connector_module_path="kvtier_connector",
        kv_role="kv_both",
        kv_connector_extra_config={"kvtier_address": addr},
    )
llm = LLM(
    model=os.environ["KVTIER_TEST_MODEL"],
    max_model_len=3584,
    kv_cache_memory_bytes=512 * 2**20,
    enforce_eager=True,
    disable_hybrid_kv_cache_manager=True,
    enable_prefix_caching=True,
    **kw,
)

# The shared system prompt, identical for every engine.
system = "You are a meticulous assistant." + " context" * system_tokens
# Per-engine users. "a" for engine A, "b" for B and the control.
who = "a" if role == "A" else "b"
prompts = [
    system + f"\n\nUser {who}{i}:" + f" {who}{i}" + " detail" * user_tokens
    for i in range(conversations)
]

sampling = SamplingParams(max_tokens=1, temperature=0)
llm.generate(["warm up the engine please"], sampling)

records = []
for i, prompt in enumerate(prompts):
    start = time.perf_counter()
    llm.generate([prompt], sampling, use_tqdm=False)
    records.append(time.perf_counter() - start)
json.dump(records, open(out_path, "w"))
'''


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def start_daemon(port, log):
    env = {
        **os.environ,
        "KVTIER_ADDR": f"127.0.0.1:{port}",
        "KVTIER_MODEL": MODEL,
        "KVTIER_LAYOUT": "custom",
        "KVTIER_TOKENS_PER_BLOCK": "16",
        "KVTIER_LAYERS": str(LAYOUT["layers"]),
        "KVTIER_KV_HEADS": str(LAYOUT["kv_heads"]),
        "KVTIER_HEAD_DIM": str(LAYOUT["head_dim"]),
        "KVTIER_DTYPE": LAYOUT["dtype"],
        "KVTIER_BLOCKS": "8192",
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


def run(role, addr, script_path, tag):
    out = f"/tmp/kvtier_replica_{tag}.json"
    env = {
        **os.environ,
        "PYTHONPATH": os.path.abspath("python"),
        "KVTIER_TEST_MODEL": MODEL,
        "SYSTEM_TOKENS": str(SYSTEM_TOKENS),
        "USER_TOKENS": str(USER_TOKENS),
        "CONVERSATIONS": str(CONVERSATIONS),
    }
    log = f"/tmp/kvtier_replica_{tag}.log"
    with open(log, "w") as handle:
        proc = subprocess.run(
            [sys.executable, script_path, role, addr, out],
            env=env, stdout=handle, stderr=subprocess.STDOUT, timeout=3600,
        )
    if proc.returncode != 0:
        print(f"    {tag} engine failed, see {log}")
        return None
    return json.load(open(out))


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    return ok


def main():
    import kvtier

    script_path = "/tmp/kvtier_replica_worker.py"
    with open(script_path, "w") as f:
        f.write(WORKER)

    port = free_port()
    addr = f"127.0.0.1:{port}"
    daemon = start_daemon(port, "/tmp/kvtier_replica_daemon.log")
    passed = True
    try:
        client = kvtier.Client(addr)

        print("engine A, cold tier")
        a_times = run("A", addr, script_path, "a")
        after_a = client.stats()
        passed &= check("A stored blocks", after_a["inserted_blocks"] > 0,
                        f"{after_a['inserted_blocks']} blocks")

        print("engine B, same tier, different users")
        b_times = run("B", addr, script_path, "b")
        after_b = client.stats()
        b_hits = after_b["hit_blocks"] - after_a["hit_blocks"]
        b_inserted = after_b["inserted_blocks"] - after_a["inserted_blocks"]
        b_deduped = after_b["deduped_blocks"] - after_a["deduped_blocks"]

        print("control, no tier")
        c_times = run("control", addr, script_path, "control")

        if not (a_times and b_times and c_times):
            print("FAIL")
            return 1

        # The system prompt is the only thing A and B share.
        shared_blocks = SYSTEM_TOKENS // 16
        passed &= check(
            "B reused blocks it never computed",
            b_hits >= shared_blocks,
            f"{b_hits} block hits, system prompt is ~{shared_blocks} blocks",
        )
        passed &= check(
            "B stored only its own new blocks",
            b_inserted < after_a["inserted_blocks"],
            f"A stored {after_a['inserted_blocks']}, B added {b_inserted}, "
            f"{b_deduped} deduped",
        )

        first_b, first_c = b_times[0], c_times[0]
        passed &= check(
            "B's first request beats the no-tier control",
            first_b < first_c,
            f"{first_b * 1000:.1f} ms vs {first_c * 1000:.1f} ms",
        )

        print()
        print(f"  engine A  median {statistics.median(a_times) * 1000:6.1f} ms   "
              f"first {a_times[0] * 1000:6.1f} ms")
        print(f"  engine B  median {statistics.median(b_times) * 1000:6.1f} ms   "
              f"first {b_times[0] * 1000:6.1f} ms")
        print(f"  control   median {statistics.median(c_times) * 1000:6.1f} ms   "
              f"first {c_times[0] * 1000:6.1f} ms")
    finally:
        daemon.terminate()
        daemon.wait(timeout=10)

    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
