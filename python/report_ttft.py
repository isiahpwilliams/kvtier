"""Build the TTFT table from whatever run_ttft.py left in a results directory.

Separate from the driver on purpose. A run writes its JSON before it exits,
so a result is complete whether or not the process that produced it exited
cleanly, and reading the directory afterwards keeps the numbers independent
of that.

Run with: .venv/bin/python python/report_ttft.py bench_results
"""

import glob
import json
import os
import re
import statistics
import sys

ARMS = ("baseline", "tier-cold", "tier-warm")


def load(results_dir):
    runs = {}
    pattern = re.compile(r"^(baseline|tier-cold|tier-warm)_(\d+)mib_r(\d+)\.json$")
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        name = os.path.basename(path)
        match = pattern.match(name)
        if not match:
            continue  # summary.json, and the untimed *_fill.json passes
        arm, mib, rep = match.group(1), int(match.group(2)), int(match.group(3))
        with open(path) as f:
            runs.setdefault((arm, mib), {})[rep] = json.load(f)
    return runs


def main(results_dir):
    runs = load(results_dir)
    if not runs:
        print(f"no runs in {results_dir}")
        return 1
    sizes = sorted({mib for _, mib in runs})

    any_run = next(iter(runs.values()))[min(next(iter(runs.values())))]
    cfg = any_run["config"]
    print(f"model {cfg['model']}, {cfg['conversations']} conversations x "
          f"{cfg['turns']} turns, {any_run['summary']['n']} requests per run")
    print(f"median prompt {any_run['summary']['median_tokens']:.0f} tokens\n")

    print("TTFT median, ms.  Repeats separated by /; the spread between them")
    print("is the noise floor, and nothing smaller than it is a result.\n")
    head = f"{'GPU KV cache':>13}"
    for arm in ARMS:
        head += f"  {arm:>17}"
    print(head + f"  {'cold':>7}  {'warm':>7}")

    for mib in sizes:
        cells, best = [], {}
        for arm in ARMS:
            reps = runs.get((arm, mib), {})
            vals = [reps[r]["summary"]["median"] * 1000 for r in sorted(reps)]
            if not vals:
                cells.append(f"{'-':>17}")
                continue
            # Median across repeats, not the best one: picking the best
            # repeat quietly reports the noise as a result.
            best[arm] = statistics.median(vals)
            cells.append(f"{'/'.join(f'{v:.1f}' for v in vals):>17}")
        line = f"{mib:>9} MiB" + "  ".join([""] + cells)
        for arm in ("tier-cold", "tier-warm"):
            if arm in best and "baseline" in best:
                delta = (best[arm] - best["baseline"]) / best["baseline"] * 100
                line += f"  {delta:+6.1f}%"
            else:
                line += f"  {'-':>7}"
        print(line)

    print("\nBy turn (median ms), lowest repeat:")
    for mib in sizes:
        print(f"\n  GPU KV cache {mib} MiB")
        for arm in ARMS:
            reps = runs.get((arm, mib), {})
            if not reps:
                continue
            pick = min(reps, key=lambda r: reps[r]["summary"]["median"])
            summary = reps[pick]["summary"]
            turns = "  ".join(
                f"t{t}:{v['median'] * 1000:6.1f}"
                for t, v in summary["by_turn"].items()
            )
            hits = ""
            stats = reps[pick].get("tier_stats")
            if stats:
                rate = stats["hit_blocks"] / max(1, stats["queried_blocks"]) * 100
                hits = (f"   tier served {rate:.0f}% of "
                        f"{stats['queried_blocks']:,} blocks queried")
            print(f"    {arm:10s} {turns}{hits}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "bench_results"))
