"""8-D Silverman-bandwidth variant of run_pullback_experiment.py.

Loads data/manifold_h_8d_silverman.npz instead of the production
8-D manifold (which uses clustered_NN bandwidth). Writes results to
results/pullback_8d_silverman/ and data/pullback_8d_silverman/ so
production artifacts are not overwritten.

Motivated by the manifold_alternatives geometric finding (Silverman
gives +0.062 edge vs clustered_NN +0.050 on the same 800 pairs).
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np

from manifold_emotions.behavior.manifold import BehaviorManifold
from manifold_emotions.config import load_config
from manifold_emotions.manifold.fit import FittedManifold
from manifold_emotions.steering.pullback_experiment import (
    PullbackExperimentReport,
    run_pullback_experiment,
)

NEUTRAL_PROMPTS: tuple[str, ...] = (
    "Tell me about your day in a few sentences.",
    "What's on your mind right now?",
    "Describe what you see out the window.",
    "Tell me a short story.",
    "What did you do yesterday?",
    "Describe a simple meal.",
    "What's the weather like in your imagination?",
    "Share a memory from childhood.",
    "Talk about a hobby you enjoy.",
    "Describe a walk through a park.",
)

DEFAULT_PAIRS: tuple[tuple[str, str], ...] = (
    ("excited", "weary"),
    ("terrified", "serene"),
)

MANIFOLD_PATH = Path("data/manifold_h_8d_silverman.npz")
OUT_DATA_DIR = Path("data/pullback_8d_silverman")
OUT_RESULTS_DIR = Path("results/pullback_8d_silverman")


def _summary_dict(report: PullbackExperimentReport) -> dict:
    g = report.geometry
    return {
        "pair": [report.start_label, report.end_label],
        "manifold_dim": 8,
        "bandwidth_heuristic": "silverman",
        "num_waypoints": int(g.num_waypoints),
        "sigma": g.sigma,
        "sigma_spec": g.sigma_spec,
        "sigma_per_waypoint": g.sigma_per_waypoint.tolist(),
        "geometry": {
            "pullback_length": g.pullback_length,
            "geodesic_length": g.geodesic_length,
            "linear_length": g.linear_length,
            "mean_dist_pullback_to_geodesic": g.mean_dist_to_geodesic,
            "mean_dist_pullback_to_linear": g.mean_dist_to_linear,
            "closer_to": g.closer_to,
            "per_waypoint_dist_pullback_to_geodesic": g.dist_pullback_to_geodesic.tolist(),
            "per_waypoint_dist_pullback_to_linear": g.dist_pullback_to_linear.tolist(),
            "my_path_valence": g.my_path[:, 0].tolist(),
            "my_path_arousal": g.my_path[:, 1].tolist(),
        },
        "trajectories": {
            name: {
                "off_manifold_energy": traj.off_manifold_energy,
                "my_geodesic_distance": traj.my_geodesic_distance,
                "waypoint_valence": traj.waypoint_behavior_mean[:, 0].tolist(),
                "waypoint_arousal": traj.waypoint_behavior_mean[:, 1].tolist(),
            }
            for name, traj in [
                ("pullback", report.pullback),
                ("geodesic", report.geodesic),
                ("linear", report.linear),
            ]
        },
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("start", nargs="?", default=None)
    parser.add_argument("end", nargs="?", default=None)
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--sigma", default=None)
    parser.add_argument("--results-suffix", default="")
    args = parser.parse_args()

    sigma_arg: float | str | None
    if args.sigma is None:
        sigma_arg = None
    elif isinstance(args.sigma, str) and args.sigma.startswith("knn:"):
        sigma_arg = args.sigma
    else:
        sigma_arg = float(args.sigma)

    config = load_config()
    if not MANIFOLD_PATH.exists():
        raise SystemExit(
            f"Missing {MANIFOLD_PATH}. Run scripts/setup_8d_silverman_pipeline.py first."
        )
    manifold = FittedManifold.load(MANIFOLD_PATH)
    behavior = BehaviorManifold.load(config.paths.manifold_y)

    if args.start is not None and args.end is not None:
        pairs = [(args.start, args.end)]
    else:
        pairs = list(DEFAULT_PAIRS)

    OUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for start, end in pairs:
        print()
        print(f"=== 8-D Silverman pullback: {start} → {end}  (K={args.k}, N={args.n}) ===")
        if start not in manifold.labels or end not in manifold.labels:
            print(f"  skipping — missing centroid in M_h")
            continue
        if start not in behavior.labels or end not in behavior.labels:
            print(f"  skipping — missing centroid in M_y")
            continue

        judge_cache = OUT_DATA_DIR / f"ratings_{start}_{end}{args.results_suffix}.json"

        report = await run_pullback_experiment(
            config=config,
            manifold=manifold,
            behavior=behavior,
            start_label=start,
            end_label=end,
            num_waypoints=args.k,
            num_prompts=args.n,
            prompts=NEUTRAL_PROMPTS,
            max_tokens=96,
            concurrency=16,
            judge_cache_path=judge_cache,
            sigma=sigma_arg,
        )

        g = report.geometry
        print(f"  G_E lengths  pullback={g.pullback_length:.3f}  "
              f"geodesic={g.geodesic_length:.3f}  linear={g.linear_length:.3f}")
        print(f"  off-M_y E   pullback={report.pullback.off_manifold_energy:.3f}  "
              f"geodesic={report.geodesic.off_manifold_energy:.3f}  "
              f"linear={report.linear.off_manifold_energy:.3f}")
        print(f"  M_y-line    pullback={report.pullback.my_geodesic_distance:.3f}  "
              f"geodesic={report.geodesic.my_geodesic_distance:.3f}  "
              f"linear={report.linear.my_geodesic_distance:.3f}")

        summary = _summary_dict(report)
        out = OUT_RESULTS_DIR / f"{start}_{end}{args.results_suffix}.json"
        out.write_text(json.dumps(summary, indent=2))
        print(f"  saved {out}")

        np.savez_compressed(
            OUT_DATA_DIR / f"paths_{start}_{end}{args.results_suffix}.npz",
            my_path=g.my_path,
            pullback_sub=g.pullback_sub,
            geodesic_sub=g.geodesic_sub,
            linear_sub=g.linear_sub,
            pullback_full=g.pullback_full,
            geodesic_full=g.geodesic_full,
            linear_full=g.linear_full,
        )


if __name__ == "__main__":
    asyncio.run(main())
