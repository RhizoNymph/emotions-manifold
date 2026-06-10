"""σ-sweep for the pullback kernel bandwidth on a single pair.

The pullback construction's bandwidth σ controls how strongly the
kernel-barycenter weights drop off with M_y distance:
- Small σ → pullback approaches nearest-neighbor snapping (jagged)
- Large σ → pullback approaches the convex hull centroid (washed out)

Day 2 used σ = median NN distance among M_y centroids = 0.132 for all
pullback experiments. This script sweeps σ ∈ {0.05, 0.13, 0.30, 0.50,
1.00} on a single pair (default: excited→weary, our cleanest
manifold-favored pullback win) to test whether the M_y-line tracking
finding is robust to bandwidth or sensitive to the default.

Only the pullback arm is rerun at each σ — geodesic and linear are
σ-independent, so we reuse their results from the existing
``results/pullback/<start>_<end>.json``. Wall-clock per σ: ~6 min
generation + ~1 min judge ≈ 7 min.

Run with:
    uv run python scripts/run_pullback_sigma_sweep.py
    uv run python scripts/run_pullback_sigma_sweep.py --start depressed --end energized
    VLLM_BASE_URL=http://node1:8000/v1 uv run python scripts/run_pullback_sigma_sweep.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import numpy as np

from manifold_emotions.behavior.judge_text import judge_texts
from manifold_emotions.behavior.manifold import BehaviorManifold
from manifold_emotions.config import load_config
from manifold_emotions.manifold.fit import FittedManifold
from manifold_emotions.manifold.pullback import construct_pullback_path
from manifold_emotions.steering.experiment import (
    _aggregate_waypoint_behavior,
    _off_manifold_energy,
    _text_id,
)
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

DEFAULT_SIGMAS: tuple[float, ...] = (0.05, 0.13, 0.30, 0.50, 1.00)
NUM_WAYPOINTS = 30
NUM_PROMPTS = 10
STEERING_SCALE = 8.0

OUT_DATA_DIR = Path("data/pullback_sigma")
OUT_RESULTS_DIR = Path("results/pullback_sigma")


def _my_line_distance(
    waypoint_behavior: np.ndarray,
    y_start: np.ndarray,
    y_end: np.ndarray,
) -> float:
    """Mean distance from each waypoint's behavior to the M_y straight line.

    Matches ``steering.pullback_experiment._distance_to_my_line``:
    distance is from ``waypoint_behavior[k]`` to ``my_path[k]`` (the k-th
    point along the straight line from y_start to y_end), not the
    perpendicular distance to the infinite chord. The point-to-point
    formulation captures both perpendicular offset and parametric
    progress mismatch — a path that runs parallel to the chord at a
    fixed offset gets credit; a path that hits the line but at the
    wrong t doesn't.
    """
    finite = np.all(np.isfinite(waypoint_behavior), axis=1)
    if not finite.any():
        return float("nan")
    n = waypoint_behavior.shape[0]
    ts = np.linspace(0.0, 1.0, n)
    my_path = (1.0 - ts)[:, None] * y_start[None, :] + ts[:, None] * y_end[None, :]
    diffs = waypoint_behavior[finite] - my_path[finite]
    return float(np.sqrt(np.sum(diffs * diffs, axis=1)).mean())


async def run_one_sigma(
    config,
    manifold: FittedManifold,
    behavior: BehaviorManifold,
    start: str,
    end: str,
    sigma: float,
) -> dict:
    OUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{start}_{end}_sigma{int(round(sigma*1000)):04d}"
    cache_file = OUT_DATA_DIR / f"ratings_{tag}.json"
    result_file = OUT_RESULTS_DIR / f"{tag}.json"

    if result_file.exists():
        existing = json.loads(result_file.read_text())
        print(
            f"  σ={sigma}: already done "
            f"(off-M_y E={existing['off_manifold_energy']:.3f}, "
            f"M_y-line={existing['my_geodesic_distance']:.3f}), skipping",
            flush=True,
        )
        return existing

    print()
    print(f"=== {start} → {end}  σ={sigma} ===", flush=True)

    my_path, pullback_sub, sigma_used, *_ = construct_pullback_path(
        manifold, behavior, start, end,
        num_waypoints=NUM_WAYPOINTS, sigma=sigma,
    )

    # Snap endpoints so the path starts and ends at the named emotions.
    start_idx = manifold.labels.index(start)
    end_idx = manifold.labels.index(end)
    pullback_sub = pullback_sub.copy()
    pullback_sub[0] = manifold.centroids_subspace[start_idx]
    pullback_sub[-1] = manifold.centroids_subspace[end_idx]
    pullback_full = manifold.unproject(pullback_sub).astype(np.float32)
    pullback_steer = pullback_full * STEERING_SCALE

    prompts_used = list(NEUTRAL_PROMPTS[:NUM_PROMPTS])
    conts = await generate_along_path(
        config, pullback_steer, prompts_used,
        max_tokens=96, concurrency=16,
    )

    # Namespace by sigma_used so cache survives σ changes without collision.
    prefix = f"pullback_{start}_{end}_sigma{int(round(sigma_used*1000)):04d}_"
    passages = [(f"{prefix}{_text_id(c)}", c.text) for c in conts]
    ratings = await judge_texts(config, passages, cache_path=cache_file)

    rating_map: dict[str, tuple[float, float]] = {}
    for text_id, rating in ratings.items():
        if text_id.startswith(prefix):
            rating_map[text_id[len(prefix):]] = (rating.valence, rating.arousal)

    waypoint_mean, _ = _aggregate_waypoint_behavior(conts, rating_map, NUM_WAYPOINTS)

    y_start = behavior.centroids[behavior.labels.index(start)]
    y_end = behavior.centroids[behavior.labels.index(end)]
    off_e = _off_manifold_energy(waypoint_mean, behavior.centroids)
    my_line = _my_line_distance(waypoint_mean, y_start, y_end)

    summary = {
        "pair": [start, end],
        "sigma": float(sigma_used),
        "steering_scale": STEERING_SCALE,
        "num_waypoints": NUM_WAYPOINTS,
        "num_prompts": NUM_PROMPTS,
        "off_manifold_energy": off_e,
        "my_geodesic_distance": my_line,
        "pullback_waypoint_valence": waypoint_mean[:, 0].tolist(),
        "pullback_waypoint_arousal": waypoint_mean[:, 1].tolist(),
        "pullback_subspace_waypoints": pullback_sub.tolist(),
    }
    result_file.write_text(json.dumps(summary, indent=2))
    return summary


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", default="excited")
    parser.add_argument("--end", default="weary")
    parser.add_argument("--sigmas", nargs="*", type=float, default=list(DEFAULT_SIGMAS))
    args = parser.parse_args()

    config = load_config()
    manifold = FittedManifold.load(config.paths.manifold_h)
    behavior = BehaviorManifold.load(config.paths.manifold_y)

    print(f"vLLM at {config.vllm_server.base_url}")
    print(f"pair: {args.start} → {args.end}, σ values: {args.sigmas}")

    # Pull the geodesic + linear baselines from the existing pullback run
    # for context — they don't depend on σ.
    baseline_path = Path(f"results/pullback/{args.start}_{args.end}.json")
    if baseline_path.exists():
        base = json.loads(baseline_path.read_text())
        g = base["trajectories"]["geodesic"]
        l = base["trajectories"]["linear"]
        print(
            f"baseline (from existing pullback run):  "
            f"geodesic off-M_y E={g['off_manifold_energy']:.3f}, "
            f"M_y-line={g['my_geodesic_distance']:.3f}  |  "
            f"linear off-M_y E={l['off_manifold_energy']:.3f}, "
            f"M_y-line={l['my_geodesic_distance']:.3f}"
        )

    overall_start = time.monotonic()
    rows: list[dict] = []
    for sigma in args.sigmas:
        row = await run_one_sigma(
            config, manifold, behavior, args.start, args.end, sigma
        )
        rows.append(row)
        print(
            f"  σ={row['sigma']:.3f}  "
            f"off-M_y E={row['off_manifold_energy']:.3f}  "
            f"M_y-line={row['my_geodesic_distance']:.3f}",
            flush=True,
        )

    elapsed = time.monotonic() - overall_start
    print()
    print(f"=== σ sweep complete in {elapsed/60:.1f} min ===")
    print()
    print(f"  {'σ':>6s}  {'off_M_y_E':>10s}  {'M_y_line':>10s}")
    for row in rows:
        print(
            f"  {row['sigma']:>6.3f}  "
            f"{row['off_manifold_energy']:>10.3f}  "
            f"{row['my_geodesic_distance']:>10.3f}"
        )


if __name__ == "__main__":
    asyncio.run(main())
