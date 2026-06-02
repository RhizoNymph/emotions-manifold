"""Phase 8: scaled run on one pair with tighter statistics.

Default pair is excited↔weary (biggest manifold win in the multi-pair sweep,
Δ=+0.118). Bumped to K=30 waypoints and N=10 prompts: 600 generations per
trajectory, 1200 total + 1200 judge calls; runs in roughly 15 minutes.

Goal: check whether smoke-scale Δ values hold up under tighter statistics.
For a smoke-favored-manifold pair, we want Δ to remain positive; for a
smoke-favored-linear pair (terrified↔serene, Δ=-0.043), this is the
falsifiable test of whether the linear preference was noise.

Cache and results files are named by pair so multiple runs accumulate
distinct artifacts under ``data/`` and ``results/``.

Run with:
    uv run python scripts/steering_scaled.py                    # excited↔weary default
    uv run python scripts/steering_scaled.py terrified serene   # any pair
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
from manifold_emotions.steering.experiment import compare_steering
from manifold_emotions.vectors.diff_in_means import EmotionVectors

# Neutral prompts kept emotionally bland so the steering vector dominates
# the conditional. Mostly first-person open-ended.
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

NUM_WAYPOINTS = 30
NUM_PROMPTS = 10


def _summary_row(
    name: str,
    energy: float,
    waypoint_v: np.ndarray,
    waypoint_a: np.ndarray,
) -> str:
    # Compress the K waypoint trace to 5 sample points for the printed line.
    k = len(waypoint_v)
    idx = np.linspace(0, k - 1, 5).astype(int)
    sampled = ", ".join(f"({waypoint_v[i]:.2f},{waypoint_a[i]:.2f})" for i in idx)
    return f"  {name:>8s} E={energy:.3f}  trace[5]={sampled}"


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("start", nargs="?", default="excited")
    parser.add_argument("end", nargs="?", default="weary")
    args = parser.parse_args()

    config = load_config()
    ev = EmotionVectors.load(config.paths.emotion_vectors)
    mh = FittedManifold.load(config.paths.manifold_h)
    my = BehaviorManifold.load(config.paths.manifold_y)

    start_label, end_label = args.start, args.end
    if start_label not in mh.labels or end_label not in mh.labels:
        raise SystemExit(f"unknown labels: {start_label!r}, {end_label!r}")

    judge_cache = Path(f"data/steering_scaled_{start_label}_{end_label}_ratings.json")
    results_out = Path(f"results/steering_scaled_{start_label}_{end_label}.json")

    print(
        f"=== {start_label} → {end_label}, K={NUM_WAYPOINTS}, N={NUM_PROMPTS} ==="
    )
    print(
        f"running 2 trajectories × {NUM_WAYPOINTS} waypoints × {NUM_PROMPTS} prompts "
        f"= {2 * NUM_WAYPOINTS * NUM_PROMPTS} generations + judge"
    )
    print()

    report = await compare_steering(
        config=config,
        emotion_vectors=ev,
        manifold=mh,
        behavior=my,
        start_label=start_label,
        end_label=end_label,
        num_waypoints=NUM_WAYPOINTS,
        num_prompts=NUM_PROMPTS,
        prompts=NEUTRAL_PROMPTS,
        max_tokens=96,
        concurrency=16,
        judge_cache_path=judge_cache,
    )

    m_e = report.manifold.cumulative_off_manifold
    l_e = report.linear.cumulative_off_manifold
    delta = l_e - m_e
    winner = "manifold" if delta > 0 else "linear" if delta < 0 else "tie"
    print(f"manifold E={m_e:.3f}  linear E={l_e:.3f}  Δ={delta:+.3f}  winner={winner}")
    print()

    my_start = my.centroids[my.labels.index(start_label)]
    my_end = my.centroids[my.labels.index(end_label)]
    chord = float(np.linalg.norm(my_end - my_start))
    print(
        f"M_y endpoints: start=({my_start[0]:.2f}, {my_start[1]:.2f})  "
        f"end=({my_end[0]:.2f}, {my_end[1]:.2f})  chord={chord:.2f}"
    )
    print()

    # Per-waypoint behavior valence/arousal trace, manifold then linear.
    print(f"  {'wp':>3} {'man V':>6} {'man A':>6}    {'lin V':>6} {'lin A':>6}")
    m_v = report.manifold.waypoint_behavior_mean[:, 0]
    m_a = report.manifold.waypoint_behavior_mean[:, 1]
    l_v = report.linear.waypoint_behavior_mean[:, 0]
    l_a = report.linear.waypoint_behavior_mean[:, 1]
    for k in range(NUM_WAYPOINTS):
        print(
            f"  {k:>3} {m_v[k]:>6.2f} {m_a[k]:>6.2f}    {l_v[k]:>6.2f} {l_a[k]:>6.2f}"
        )

    results_out.parent.mkdir(parents=True, exist_ok=True)
    results_out.write_text(
        json.dumps(
            {
                "pair": [start_label, end_label],
                "num_waypoints": NUM_WAYPOINTS,
                "num_prompts": NUM_PROMPTS,
                "my_chord_length": chord,
                "manifold_off_manifold_energy": m_e,
                "linear_off_manifold_energy": l_e,
                "delta_linear_minus_manifold": delta,
                "manifold_waypoint_valence": m_v.tolist(),
                "manifold_waypoint_arousal": m_a.tolist(),
                "linear_waypoint_valence": l_v.tolist(),
                "linear_waypoint_arousal": l_a.tolist(),
            },
            indent=2,
        )
    )
    print()
    print(f"saved scaled results to {results_out}")


if __name__ == "__main__":
    asyncio.run(main())
