"""Cost of recomputing a prefix, as a function of its length.

The other half of the crossover. Prefix caching is off and every prompt is
salted differently, so nothing is reused and each measurement is a real
prefill of that many tokens.

Reported alongside bench_transfer.py, the two curves say how long a prefix has
to be before fetching it beats computing it.

Run with: .venv/bin/python python/bench_prefill.py --help
"""

import argparse
import json
import math
import statistics
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--kv-cache-bytes", type=int, default=2 * 2**30)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--tokens", type=int, nargs="+",
        default=[128, 256, 512, 1024, 1536, 2048, 3072],
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        kv_cache_memory_bytes=args.kv_cache_bytes,
        enforce_eager=True,
        disable_hybrid_kv_cache_manager=True,
        enable_prefix_caching=False,
    )
    sampling = SamplingParams(max_tokens=1, temperature=0)
    llm.generate(["warm up the engine please"], sampling)

    def prompt_of(target, salt):
        # Salted so no two measurements can share a prefix.
        text = f"session {salt} begins." + " context" * target
        ids = tokenizer(text).input_ids[:target]
        return tokenizer.decode(ids)

    results = []
    for target in args.tokens:
        samples = []
        for r in range(args.repeats):
            text = prompt_of(target, f"{target}x{r}")
            actual = len(tokenizer(text).input_ids)
            start = time.perf_counter()
            llm.generate([text], sampling, use_tqdm=False)
            samples.append(time.perf_counter() - start)
        median = statistics.median(samples)
        results.append(
            {
                "tokens": target,
                "actual_tokens": actual,
                "median_s": median,
                "min_s": min(samples),
                "max_s": max(samples),
                "spread": (max(samples) - min(samples)) / median,
            }
        )
        print(f"  {target:5d} tokens: median {median * 1000:8.2f} ms  "
              f"[{min(samples) * 1000:.2f}-{max(samples) * 1000:.2f}]  "
              f"spread {(max(samples) - min(samples)) / median * 100:.0f}%")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"config": vars(args), "prefill": results}, f, indent=1)
    print("PREFILL_DONE")


if __name__ == "__main__":
    sys.exit(main() or 0)
