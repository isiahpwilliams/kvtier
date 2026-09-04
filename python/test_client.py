"""End-to-end check of the Python bindings against a live kvtierd.

Run with: .venv/bin/python python/test_client.py
"""

import os
import socket
import subprocess
import sys
import time

import kvtier

# A tiny layout, so the test moves kilobytes rather than gigabytes. Must match
# what the daemon is started with.
LAYOUT = dict(
    tokens_per_block=16,
    num_layers=2,
    num_kv_heads=2,
    head_dim=8,
    dtype="f16",
)
MODEL = "py-test"


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def start_daemon(port):
    env = {
        **os.environ,
        "KVTIER_ADDR": f"127.0.0.1:{port}",
        "KVTIER_MODEL": MODEL,
        "KVTIER_BLOCKS": "256",
        "KVTIER_LAYOUT": "tiny",
    }
    daemon = subprocess.Popen(
        ["target/release/kvtierd"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(100):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                return daemon
        except OSError:
            time.sleep(0.05)
    daemon.kill()
    raise RuntimeError("daemon did not come up")


def check(name, condition, detail=""):
    status = "ok  " if condition else "FAIL"
    print(f"  {status}  {name}{'  ' + detail if detail else ''}")
    return condition


def main():
    port = free_port()
    daemon = start_daemon(port)
    passed = True
    try:
        client = kvtier.Client(f"127.0.0.1:{port}")
        block_bytes = client.block_bytes
        print(f"connected: {client.model_id}, {block_bytes} B blocks")

        # The server tells us its layout, and the hasher it hands back agrees
        # with it by construction. Nothing here reimplements the chain rule.
        hasher = client.hasher()
        passed &= check("hasher matches server layout", hasher.block_bytes == block_bytes)

        system_prompt = list(range(1000, 1048))          # 48 tokens, 3 blocks
        first = system_prompt + list(range(2000, 2032))  # + 2 private blocks
        second = system_prompt + list(range(3000, 3032))

        names = hasher.chain(first)
        passed &= check("names one per full block", len(names) == 5, f"{len(names)}")
        passed &= check("a name is 16 bytes", len(names[0]) == 16)
        passed &= check(
            "shared prefix shares names",
            hasher.chain(second)[:3] == names[:3],
        )

        passed &= check("cold lookup misses", client.match_prefix(names) == 0)

        # Deterministic stand-in for real KV.
        payloads = [bytes([(i * 37 + j) % 251 for j in range(block_bytes)]) for i in range(5)]
        depths = [(name, (i + 1) * LAYOUT["tokens_per_block"]) for i, name in enumerate(names)]

        inserted, deduped, dropped = client.put_blocks(None, depths, b"".join(payloads))
        passed &= check("put lands", (inserted, deduped, dropped) == (5, 0, 0),
                        f"{inserted}/{deduped}/{dropped}")

        passed &= check("warm lookup hits", client.match_prefix(names) == 5)
        passed &= check("round trip is byte exact", client.get_blocks(names) == payloads)

        other = hasher.chain(second)
        passed &= check("a second sequence hits the shared prefix",
                        client.match_prefix(other) == 3)
        passed &= check("and fetches only what is there",
                        client.get_blocks(other) == payloads[:3])

        stats = client.stats()
        passed &= check("stats come back", stats["resident_blocks"] == 5, str(stats["resident_blocks"]))

        # A name of the wrong width is a caller bug, not a wire error.
        try:
            client.match_prefix([b"too short"])
            passed &= check("rejects a malformed name", False)
        except ValueError:
            passed &= check("rejects a malformed name", True)
    finally:
        daemon.terminate()
        daemon.wait(timeout=5)

    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
