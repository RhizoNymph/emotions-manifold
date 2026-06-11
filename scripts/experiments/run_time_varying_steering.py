"""Time-varying steering along the trajectory (Goodfire's temporal claim).

Thin CLI over ``manifold_emotions.experiments.time_varying``. Defaults
reproduce the original n=12 design (8 segments × 12 tokens, hard-switch
schedule, sequential judge, results/time_varying). The n=40 four-condition
design uses one out-dir per condition:

    # replication (tv8) — generate only, judge later in one batched pass
    uv run python scripts/experiments/run_time_varying_steering.py \\
        --pairs experiments/pairs/alift_n40.json \\
        --out-dir results/time_varying_n40/tv8 --judge none

    # smoothed variant (tv16): same 96 tokens, half-size switches
    ... --segments 16 --tokens-per-segment 6 --out-dir .../tv16 --judge none

    # segmentation controls: constant midpoint vector, same call structure
    ... --schedule constant --out-dir .../cv8 --judge none
    ... --schedule constant --segments 16 --tokens-per-segment 6 \\
        --out-dir .../cv16 --judge none

    # then judge each condition in a single batched pass
    ... --out-dir .../tv8 --judge batched

Set VLLM_BASE_URL to point a process at a specific host (e.g.
``VLLM_BASE_URL=http://node1:8000/v1``) and split the pair file to run
several hosts in parallel.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from manifold_emotions.config import load_config
from manifold_emotions.experiments.time_varying import TVRunConfig, run_tv_pairs


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("start", nargs="?", default=None)
    parser.add_argument("end", nargs="?", default=None)
    parser.add_argument("--pairs", type=Path, default=None,
                        help="JSON pair file ([[start, end], ...]) to run "
                             "instead of a single pair")
    parser.add_argument("--manifold", type=Path, default=Path("data/manifold_h.npz"),
                        help="FittedManifold npz (default: production 8-D)")
    parser.add_argument("--schedule", choices=["varying", "constant"],
                        default="varying",
                        help="varying = step waypoints across segments (TV); "
                             "constant = path-midpoint waypoint in every segment "
                             "(segmentation control)")
    parser.add_argument("--judge", choices=["sequential", "batched", "none"],
                        default="sequential",
                        help="none = generate+save completions only; judge later "
                             "by re-running with sequential or batched")
    parser.add_argument("--segments", type=int, default=8,
                        help="number of steering segments per generation")
    parser.add_argument("--tokens-per-segment", type=int, default=12)
    parser.add_argument("--out-dir", type=Path, default=Path("results/time_varying"),
                        help="condition output directory (use a fresh one per "
                             "condition; never overwrite earlier results)")
    parser.add_argument("--results-suffix", default="")
    parser.add_argument("--concurrency", type=int, default=8,
                        help="concurrent generations per process (each one is "
                             "K sequential segment calls)")
    parser.add_argument("--force", action="store_true",
                        help="re-run pairs whose outputs already exist")
    args = parser.parse_args()

    if args.pairs is not None:
        pairs = [(a, b) for a, b in json.loads(args.pairs.read_text())]
    elif args.start and args.end:
        pairs = [(args.start, args.end)]
    else:
        parser.error("provide either start+end labels or --pairs")

    config = load_config()
    run = TVRunConfig(
        out_dir=args.out_dir,
        manifold_path=args.manifold,
        schedule=args.schedule,
        judge=args.judge,
        num_segments=args.segments,
        tokens_per_segment=args.tokens_per_segment,
        results_suffix=args.results_suffix,
        concurrency=args.concurrency,
    )
    report = await run_tv_pairs(config, run, pairs, force=args.force)

    print(f"\ngenerated={len(report.generated)} judged={len(report.judged)} "
          f"skipped={len(report.skipped)} failed={len(report.failed)}")
    if report.failed:
        for start, end, err in report.failed:
            print(f"  FAILED {start}->{end}: {err}")
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
