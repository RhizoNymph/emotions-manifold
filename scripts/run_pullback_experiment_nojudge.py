"""Phase-1 variant of run_pullback_experiment for the batched-judge pipeline.

Generates trajectories and steered continuations EXACTLY like
``run_pullback_experiment.py``, but skips the Claude V/A judge step.
Writes raw completions to ``data/<chain>/completions_<pair>.json`` for
later batch judging via ``run_chain_batched_judge.py``.

Also writes a skeleton ``results/<chain>/<pair>.json`` with NaN for
``off_manifold_energy`` and ``my_geodesic_distance`` so the Phase 2
script has a file to mutate.

The sequential ``run_pullback_experiment.py`` is untouched — this script
is purely additive. Sequential and batched pipelines can coexist.

Usage:
    PYTHONUNBUFFERED=1 uv run python scripts/run_pullback_experiment_nojudge.py <start> <end> \\
        [--manifold path/to/manifold.npz] [--chain pullback] [--k 30] [--n 10]
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from manifold_emotions.behavior.manifold import BehaviorManifold
from manifold_emotions.config import load_config
from manifold_emotions.manifold.fit import FittedManifold
from manifold_emotions.manifold.pullback import compute_pullback
from manifold_emotions.steering.trajectory import generate_along_path


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

SCALE = 8.0


@dataclass(frozen=True)
class CompletionRecord:
    """Per-(method, waypoint, prompt) raw text record.

    ``text_id`` is the stable identifier that ties the eventual judge
    rating back to the (pair, method, waypoint, prompt) tuple.
    """
    text_id: str
    method: str       # "pullback" | "geodesic" | "linear"
    waypoint: int
    prompt: int
    text: str


def _build_text_id(method: str, pair_a: str, pair_b: str, waypoint: int, prompt: int) -> str:
    """Match the sequential pipeline's text_id format exactly.

    Sequential builds: f"{method}_{start_label}_{end_label}_wp{N:03d}_p{N:02d}"
    where start_label/end_label preserve literal spaces in multi-word labels
    (e.g. 'pullback_at ease_disdainful_wp000_p00'). Keeping the same format
    means batched ratings caches can be swapped in for sequential ones
    without re-keying.
    """
    return f"{method}_{pair_a}_{pair_b}_wp{waypoint:03d}_p{prompt:02d}"


async def run_pair(
    config,
    manifold: FittedManifold,
    behavior: BehaviorManifold,
    start: str,
    end: str,
    chain: str,
    num_waypoints: int,
    num_prompts: int,
) -> None:
    out_data_dir = Path(f"data/{chain}")
    out_results_dir = Path(f"results/{chain}")
    out_data_dir.mkdir(parents=True, exist_ok=True)
    out_results_dir.mkdir(parents=True, exist_ok=True)

    if start not in manifold.labels or end not in manifold.labels:
        print(f"  skip {start}→{end}: missing centroid in M_h"); return
    if start not in behavior.labels or end not in behavior.labels:
        print(f"  skip {start}→{end}: missing centroid in M_y"); return

    print(f"=== nojudge generation: {start} → {end}  (K={num_waypoints}, N={num_prompts}) ===")

    # Compute trajectories (subspace + full residual)
    g = compute_pullback(
        manifold=manifold, behavior=behavior,
        start_label=start, end_label=end,
        num_waypoints=num_waypoints, sigma=None,
    )

    pullback_full = np.asarray(g.pullback_full, dtype=np.float32) * SCALE
    geodesic_full = np.asarray(g.geodesic_full, dtype=np.float32) * SCALE
    linear_full   = np.asarray(g.linear_full,   dtype=np.float32) * SCALE

    prompts_used = list(NEUTRAL_PROMPTS[:num_prompts])
    methods = {
        "pullback": pullback_full,
        "geodesic": geodesic_full,
        "linear":   linear_full,
    }

    records: list[CompletionRecord] = []
    for method, waypoints in methods.items():
        # generate_along_path returns a flat list of SteeredContinuation
        continuations = await generate_along_path(
            config, waypoints, prompts_used,
            max_tokens=96, concurrency=16,
        )
        for c in continuations:
            records.append(CompletionRecord(
                text_id=_build_text_id(method, start, end, c.waypoint_index, c.prompt_index),
                method=method,
                waypoint=c.waypoint_index,
                prompt=c.prompt_index,
                text=c.text,
            ))

    # Persist raw completions for Phase 2
    completions_path = out_data_dir / f"completions_{start}_{end}.json"
    completions_path.write_text(json.dumps(
        [asdict(r) for r in records], indent=2,
    ))
    print(f"  wrote {len(records)} completions → {completions_path}")

    # Persist subspace + full residual paths (Phase 2 reads my_path from here)
    paths_path = out_data_dir / f"paths_{start}_{end}.npz"
    np.savez_compressed(
        paths_path,
        my_path=g.my_path,
        pullback_sub=g.pullback_sub,
        geodesic_sub=g.geodesic_sub,
        linear_sub=g.linear_sub,
        pullback_full=g.pullback_full,
        geodesic_full=g.geodesic_full,
        linear_full=g.linear_full,
    )

    # Skeleton summary with NaN metrics — Phase 2 mutates this in place
    skeleton = {
        "pair": [start, end],
        "manifold_dim": int(manifold.num_components),
        "num_waypoints": int(num_waypoints),
        "sigma": float(g.sigma),
        "sigma_spec": g.sigma_spec,
        "sigma_per_waypoint": g.sigma_per_waypoint.tolist(),
        "geometry": {
            "pullback_length": float(g.pullback_length),
            "geodesic_length": float(g.geodesic_length),
            "linear_length":   float(g.linear_length),
            "mean_dist_pullback_to_geodesic": float(g.mean_dist_to_geodesic),
            "mean_dist_pullback_to_linear":   float(g.mean_dist_to_linear),
            "closer_to": g.closer_to,
            "per_waypoint_dist_pullback_to_geodesic": g.dist_pullback_to_geodesic.tolist(),
            "per_waypoint_dist_pullback_to_linear":   g.dist_pullback_to_linear.tolist(),
            "my_path_valence": g.my_path[:, 0].tolist(),
            "my_path_arousal": g.my_path[:, 1].tolist(),
        },
        "trajectories": {
            method: {
                # NaN — Phase 2 fills these in after batch judging
                "off_manifold_energy": float("nan"),
                "my_geodesic_distance": float("nan"),
                "waypoint_valence": [float("nan")] * num_waypoints,
                "waypoint_arousal": [float("nan")] * num_waypoints,
            }
            for method in ("pullback", "geodesic", "linear")
        },
        "phase": "nojudge",  # marker so Phase 2 / consumers know judging is pending
    }
    summary_path = out_results_dir / f"{start}_{end}.json"
    summary_path.write_text(json.dumps(skeleton, indent=2))
    print(f"  wrote skeleton summary → {summary_path} (judging pending)")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("start")
    parser.add_argument("end")
    parser.add_argument("--manifold", default="data/manifold_h.npz",
                        help="Path to the FittedManifold npz to use.")
    parser.add_argument("--chain", default="pullback",
                        help="Chain subdirectory under data/ and results/ "
                             "(e.g. pullback_8d_silverman, pullback_6d).")
    parser.add_argument("--k", type=int, default=30, help="num_waypoints")
    parser.add_argument("--n", type=int, default=10, help="num_prompts")
    args = parser.parse_args()

    config = load_config()
    manifold = FittedManifold.load(Path(args.manifold))
    behavior = BehaviorManifold.load(Path(config.paths.manifold_y))

    await run_pair(
        config, manifold, behavior,
        start=args.start, end=args.end,
        chain=args.chain,
        num_waypoints=args.k, num_prompts=args.n,
    )


if __name__ == "__main__":
    asyncio.run(main())
