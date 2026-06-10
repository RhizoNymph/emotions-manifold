"""Run a chord-experiment chain: all pairs of a variant across vLLM hosts.

Replaces the per-experiment shell chains (alift_*_chain.sh,
pullback_171_*.sh, node1/overnight/recovery/resume chains). Resume is
built in: already-complete pairs are skipped, so rerunning the same
command continues an interrupted chain.

    # n=40 chord at 4-D across two vLLM hosts
    uv run python scripts/orchestration/run_chain.py \
        --config experiments/chord_4d.yaml \
        --pairs experiments/pairs/alift_n40.json \
        --hosts http://localhost:8000/v1,http://node1:8000/v1

    # single-host (config default), sigma override, fresh rerun
    uv run python scripts/orchestration/run_chain.py \
        --config experiments/chord_8d.yaml --pairs experiments/pairs/alift_n40.json \
        --sigma knn:5 --results-suffix _knn5 --force

Long runs: launch under nohup/tmux and tee stdout to a log file.
For ``judge: none`` variants, follow with the phase-2 batch judging:
    uv run python scripts/orchestration/run_chain_batched_judge.py --chain <name>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from manifold_emotions.config import load_config
from manifold_emotions.experiments.chain import run_chain
from manifold_emotions.experiments.chord import ChordRunConfig


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True,
                        help="experiment variant YAML (see experiments/)")
    parser.add_argument("--pairs", type=Path, required=True,
                        help="JSON pair file: [[start, end], ...]")
    parser.add_argument(
        "--hosts", default=None,
        help="comma-separated vLLM base URLs (default: the configured server)",
    )
    parser.add_argument("--k", type=int, default=None, help="num_waypoints override")
    parser.add_argument("--n", type=int, default=None, help="num_prompts override")
    parser.add_argument("--sigma", default=None,
                        help="pullback σ override: float or 'knn:K'")
    parser.add_argument("--results-suffix", default="")
    parser.add_argument("--force", action="store_true",
                        help="re-run pairs even if their outputs exist")
    args = parser.parse_args()

    sigma: float | str | None
    if args.sigma is None:
        sigma = None
    elif args.sigma.startswith("knn:"):
        sigma = args.sigma
    else:
        sigma = float(args.sigma)

    run = ChordRunConfig.from_yaml(args.config)
    pairs = [(a, b) for a, b in json.loads(args.pairs.read_text())]
    hosts = args.hosts.split(",") if args.hosts else None
    config = load_config()

    report = await run_chain(
        config, run, pairs, hosts,
        num_waypoints=args.k, num_prompts=args.n,
        sigma=sigma, results_suffix=args.results_suffix,
        force=args.force,
    )

    print()
    print(f"chain {run.name}: {len(report.completed)} completed, "
          f"{len(report.skipped)} skipped (already done), "
          f"{len(report.failed)} failed")
    for start, end, error in report.failed:
        print(f"  FAILED {start} → {end}: {error}")
    if report.failed:
        print("re-run the same command to retry failed pairs (resume skips the rest)")
    if run.judge == "none" and report.completed:
        print(f"next: uv run python scripts/orchestration/run_chain_batched_judge.py "
              f"--chain {run.name}")
    sys.exit(0 if report.ok else 1)


if __name__ == "__main__":
    asyncio.run(main())
