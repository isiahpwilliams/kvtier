"""Correctness of the vLLM connector: KV fetched from the tier must produce
the same tokens as KV that vLLM reused from its own cache.

Four engines, one after another in separate processes:

  1. no connector, prefix caching on   -> the reference: vLLM reusing KV
  2. no connector, prefix caching off  -> the same prompts, recomputed
  3. connector, prefix caching on      -> populates the tier
  4. connector, prefix caching off     -> the tier is the only source of KV

The check that matters is 4 against 1. Both reuse KV rather than recompute
it; the only difference is where the KV came from.

2 is here because comparing against it is a trap, and this project has been
caught by it. Reused KV and recomputed KV are not bitwise equal -- attention
over a cached prefix runs a different kernel shape than attention that builds
the prefix -- and on a near-tied logit that flips the argmax. On this prompt
set it flips exactly one request in four. A connector checked against
recomputed tokens therefore looks broken when it is fine. The run is kept and
reported so the difference stays visible instead of being rediscovered.

Run with: .venv/bin/python python/test_connector.py
"""

import json
import os
import socket
import subprocess
import sys
import time

MODEL = os.environ.get("KVTIER_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
LAYOUT = dict(layers=24, kv_heads=2, head_dim=64, dtype="bf16", block=16)

WORKER = r'''
import json, os, sys
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

mode, addr, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
kw = {}
if mode.startswith("tier"):
    kw["kv_transfer_config"] = KVTransferConfig(
        kv_connector="KvtierConnector",
        kv_connector_module_path="kvtier_connector",
        kv_role="kv_both",
        kv_connector_extra_config={"kvtier_address": addr},
    )
llm = LLM(
    model=os.environ["KVTIER_TEST_MODEL"],
    gpu_memory_utilization=0.30,
    max_model_len=4096,
    enforce_eager=True,
    disable_hybrid_kv_cache_manager=True,
    enable_prefix_caching=mode.endswith("reuse"),
    **kw,
)
shared = "The following is a transcript of a technical discussion. " * 40
prompts = [shared + f"\nSpeaker {i} said: the answer is" for i in range(4)]
outs = llm.generate(prompts, SamplingParams(max_tokens=24, temperature=0))
json.dump([list(o.outputs[0].token_ids) for o in outs], open(out_path, "w"))
'''


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def start_daemon(port):
    env = {
        **os.environ,
        "KVTIER_ADDR": f"127.0.0.1:{port}",
        "KVTIER_MODEL": MODEL,
        "KVTIER_LAYOUT": "custom",
        "KVTIER_TOKENS_PER_BLOCK": str(LAYOUT["block"]),
        "KVTIER_LAYERS": str(LAYOUT["layers"]),
        "KVTIER_KV_HEADS": str(LAYOUT["kv_heads"]),
        "KVTIER_HEAD_DIM": str(LAYOUT["head_dim"]),
        "KVTIER_DTYPE": LAYOUT["dtype"],
        "KVTIER_BLOCKS": "8192",
    }
    daemon = subprocess.Popen(
        ["target/release/kvtierd"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(200):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.1).close()
            return daemon
        except OSError:
            time.sleep(0.05)
    daemon.kill()
    raise RuntimeError("kvtierd did not come up")


def run(mode, addr, tag, script_path):
    out = f"/tmp/kvtier_test_{tag}.json"
    env = {
        **os.environ,
        "PYTHONPATH": os.path.abspath("python"),
        "KVTIER_TEST_MODEL": MODEL,
    }
    proc = subprocess.run(
        [sys.executable, script_path, mode, addr, out],
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode != 0:
        print(proc.stdout[-3000:])
        print(proc.stderr[-3000:])
        raise RuntimeError(f"{tag} engine failed")
    return json.load(open(out))


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    return ok


def main():
    import kvtier

    script_path = "/tmp/kvtier_test_worker.py"
    with open(script_path, "w") as f:
        f.write(WORKER)

    port = free_port()
    addr = f"127.0.0.1:{port}"
    daemon = start_daemon(port)
    passed = True
    try:
        client = kvtier.Client(addr)

        reference = run("plain-reuse", addr, "reference", script_path)
        passed &= check(
            "reference engine produced tokens", all(len(t) > 0 for t in reference)
        )

        recomputed = run("plain-fresh", addr, "recomputed", script_path)
        flipped = sum(a != b for a, b in zip(reference, recomputed))
        print(f"  note  recompute vs reuse differs on {flipped} of "
              f"{len(reference)} requests with no connector involved; "
              f"that is vLLM, not the tier")

        before = client.stats()
        run("tier-reuse", addr, "populate", script_path)
        after_populate = client.stats()
        stored = after_populate["inserted_blocks"] - before["inserted_blocks"]
        passed &= check("the tier was populated", stored > 0, f"{stored} blocks")

        hits_before = after_populate["hit_blocks"]
        warm = run("tier-fresh", addr, "warm", script_path)
        after_warm = client.stats()
        hits = after_warm["hit_blocks"] - hits_before
        passed &= check(
            "the warm engine actually hit the tier", hits > 0, f"{hits} block hits"
        )

        passed &= check(
            "KV from the tier gives the same tokens as KV vLLM reused",
            warm == reference,
        )
        if warm != reference:
            for i, (a, b) in enumerate(zip(reference, warm)):
                if a != b:
                    print(f"     request {i}: reference {a[:8]} vs warm {b[:8]}")
                    break
    finally:
        daemon.terminate()
        daemon.wait(timeout=5)

    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
