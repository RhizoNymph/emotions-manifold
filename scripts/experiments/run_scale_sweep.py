"""Manifold-vs-linear comparison swept across steering scales.

Day 2 left an open question: at scale=8 every trajectory plateaus well
short of the M_y endpoints (~80% of the chord), and the manifold-vs-
linear gap may grow at higher scale where the linear path actually
hits low-density regions while the manifold path stays in-distribution.

This script runs K=30, N=10 at scale ∈ {8, 10, 12, 15} on a single pair
(default: depressed → energized, our strongest manifold-favored pair at
scale=8 with Δ=+0.219). Fresh judge cache per scale so each measurement
is independent.

Run with:
    uv run python scripts/run_scale_sweep.py
    uv run python scripts/run_scale_sweep.py --start excited --end weary
    uv run python scripts/run_scale_sweep.py --scales 8 12 15
    VLLM_BASE_URL=http://node1:8000/v1 uv run python scripts/run_scale_sweep.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import numpy as np

from manifold_emotions.behavior.manifold import BehaviorManifold
from manifold_emotions.config import load_config
from manifold_emotions.manifold.fit import FittedManifold
from manifold_emotions.steering.experiment import compare_steering
from manifold_emotions.vectors.diff_in_means import EmotionVectors

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

DEFAULT_SCALES: tuple[float, ...] = (8.0, 10.0, 12.0, 15.0)
NUM_WAYPOINTS = 30
NUM_PROMPTS = 10

OUT_DATA_DIR = Path("data/scale_sweep")
OUT_RESULTS_DIR = Path("results/scale_sweep")


async def run_one(
    config,
    emotion_vectors: EmotionVectors,
    manifold: FittedManifold,
    behavior: BehaviorManifold,
    start: str,
    end: str,
    scale: float,
) -> dict:
    tag = f"{start}_{end}_scale{int(round(scale*10)):03d}"
    cache_file = OUT_DATA_DIR / f"ratings_{tag}.json"
    results_file = OUT_RESULTS_DIR / f"{tag}.json"

    if results_file.exists():
        existing = json.loads(results_file.read_text())
        print(
            f"  {start}→{end} @ scale={scale} already done "
            f"(Δ={existing['delta_linear_minus_manifold']:+.3f}), skipping",
            flush=True,
        )
        return existing

    print()
    print(f"=== {start} → {end}  @ steering_scale={scale} ===", flush=True)
    report = await compare_steering(
        config=config,
        emotion_vectors=emotion_vectors,
        manifold=manifold,
        behavior=behavior,
        start_label=start,
        end_label=end,
        num_waypoints=NUM_WAYPOINTS,
        num_prompts=NUM_PROMPTS,
        prompts=NEUTRAL_PROMPTS,
        max_tokens=96,
        concurrency=16,
        judge_cache_path=cache_file,
        steering_scale=scale,
    )

    m_e = report.manifold.cumulative_off_manifold
    l_e = report.linear.cumulative_off_manifold
    delta = l_e - m_e

    my_start = behavior.centroids[behavior.labels.index(start)]
    my_end = behavior.centroids[behavior.labels.index(end)]
    chord = float(np.linalg.norm(my_end - my_start))

    summary = {
        "pair": [start, end],
        "steering_scale": scale,
        "num_waypoints": NUM_WAYPOINTS,
        "num_prompts": NUM_PROMPTS,
        "my_chord_length": chord,
        "my_start_v": float(my_start[0]),
        "my_start_a": float(my_start[1]),
        "my_end_v": float(my_end[0]),
        "my_end_a": float(my_end[1]),
        "manifold_off_manifold_energy": m_e,
        "linear_off_manifold_energy": l_e,
        "delta_linear_minus_manifold": delta,
        "manifold_waypoint_valence": report.manifold.waypoint_behavior_mean[:, 0].tolist(),
        "manifold_waypoint_arousal": report.manifold.waypoint_behavior_mean[:, 1].tolist(),
        "linear_waypoint_valence": report.linear.waypoint_behavior_mean[:, 0].tolist(),
        "linear_waypoint_arousal": report.linear.waypoint_behavior_mean[:, 1].tolist(),
    }
    OUT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_file.write_text(json.dumps(summary, indent=2))
    return summary


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", default="depressed")
    parser.add_argument("--end", default="energized")
    parser.add_argument("--scales", nargs="*", type=float, default=list(DEFAULT_SCALES))
    args = parser.parse_args()

    config = load_config()
    ev = EmotionVectors.load(config.paths.emotion_vectors)
    mh = FittedManifold.load(config.paths.manifold_h)
    behavior = BehaviorManifold.load(config.paths.manifold_y)

    OUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"vLLM at {config.vllm_server.base_url}")
    print(f"pair: {args.start} → {args.end}, scales: {args.scales}")

    overall_start = time.monotonic()
    rows: list[dict] = []
    for scale in args.scales:
        row = await run_one(config, ev, mh, behavior, args.start, args.end, scale)
        rows.append(row)
        print(
            f"  manifold E={row['manifold_off_manifold_energy']:.3f}  "
            f"linear E={row['linear_off_manifold_energy']:.3f}  "
            f"Δ={row['delta_linear_minus_manifold']:+.3f}",
            flush=True,
        )

    elapsed = time.monotonic() - overall_start
    print()
    print(f"=== scale sweep complete in {elapsed/60:.1f} min ===")
    print()
    print(
        f"  {'scale':>5s}  {'manifold E':>10s}  {'linear E':>10s}  {'Δ':>7s}"
    )
    for row in rows:
        print(
            f"  {row['steering_scale']:>5.1f}  "
            f"{row['manifold_off_manifold_energy']:>10.3f}  "
            f"{row['linear_off_manifold_energy']:>10.3f}  "
            f"{row['delta_linear_minus_manifold']:>+7.3f}"
        )


if __name__ == "__main__":
    asyncio.run(main())
