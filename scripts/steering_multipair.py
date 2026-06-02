"""Phase 8: run the manifold-vs-linear steering comparison across multiple emotion pairs.

Same per-pair protocol as ``steering_smoke.py`` (K=10 waypoints, N=3 prompts,
strength ~0.5), but across a set of pairs chosen to span different geometric
relationships in (valence, arousal) space:

- intra-quadrant short jumps (happy↔sad is essentially along the valence axis)
- cross-quadrant pairs that should require the path to curve through some
  intermediate region (e.g. ecstatic↔melancholy, terrified↔serene)
- pairs along the arousal axis only (calm↔desperate)

Per-pair manifold-vs-linear deltas accumulate into a summary table. Writes
the full per-pair report to ``results/steering_multipair.json`` for later
plotting / analysis.

Run with:
    uv run python scripts/steering_multipair.py
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np

from manifold_emotions.behavior.manifold import BehaviorManifold
from manifold_emotions.config import load_config
from manifold_emotions.manifold.fit import FittedManifold
from manifold_emotions.steering.experiment import compare_steering
from manifold_emotions.vectors.diff_in_means import EmotionVectors

PAIRS: tuple[tuple[str, str], ...] = (
    ("happy", "sad"),  # valence axis, short
    ("calm", "desperate"),  # cross-quadrant via arousal
    ("excited", "weary"),  # cross-quadrant, intra-arousal-pair
    ("terrified", "serene"),  # cross-quadrant, long
    ("ecstatic", "melancholy"),  # max cross-quadrant
)

JUDGE_CACHE = Path("data/steering_multipair_ratings.json")
RESULTS_OUT = Path("results/steering_multipair.json")


async def main() -> None:
    config = load_config()
    ev = EmotionVectors.load(config.paths.emotion_vectors)
    mh = FittedManifold.load(config.paths.manifold_h)
    my = BehaviorManifold.load(config.paths.manifold_y)

    print(
        f"running {len(PAIRS)} pairs × 2 trajectories × 10 waypoints × 3 prompts "
        f"= {len(PAIRS) * 60} generations"
    )
    print()

    per_pair_summary: list[dict] = []

    for start_label, end_label in PAIRS:
        if start_label not in mh.labels or end_label not in mh.labels:
            print(f"skipping {start_label}↔{end_label} (missing label)")
            continue

        print(f"=== {start_label} → {end_label} ===")
        report = await compare_steering(
            config=config,
            emotion_vectors=ev,
            manifold=mh,
            behavior=my,
            start_label=start_label,
            end_label=end_label,
            num_waypoints=10,
            num_prompts=3,
            max_tokens=96,
            concurrency=8,
            judge_cache_path=JUDGE_CACHE,
        )

        m_e = report.manifold.cumulative_off_manifold
        l_e = report.linear.cumulative_off_manifold
        delta = l_e - m_e
        winner = "manifold" if delta > 0 else "linear" if delta < 0 else "tie"
        print(
            f"  manifold E={m_e:.3f}  linear E={l_e:.3f}  Δ={delta:+.3f}  winner={winner}"
        )

        my_start = my.centroids[my.labels.index(start_label)]
        my_end = my.centroids[my.labels.index(end_label)]
        chord_len = float(np.linalg.norm(my_end - my_start))
        print(
            f"  M_y endpoints: start=({my_start[0]:.2f}, {my_start[1]:.2f})  "
            f"end=({my_end[0]:.2f}, {my_end[1]:.2f})  chord={chord_len:.2f}"
        )

        # Compact per-waypoint behavior trace (just valence so the table fits).
        m_v = report.manifold.waypoint_behavior_mean[:, 0]
        l_v = report.linear.waypoint_behavior_mean[:, 0]
        print(
            "  manifold V: " + ", ".join(f"{v:.2f}" for v in m_v)
        )
        print(
            "  linear   V: " + ", ".join(f"{v:.2f}" for v in l_v)
        )
        print()

        per_pair_summary.append(
            {
                "start": start_label,
                "end": end_label,
                "my_chord_length": chord_len,
                "manifold_off_manifold_energy": m_e,
                "linear_off_manifold_energy": l_e,
                "delta_linear_minus_manifold": delta,
                "manifold_waypoint_valence": report.manifold.waypoint_behavior_mean[:, 0].tolist(),
                "manifold_waypoint_arousal": report.manifold.waypoint_behavior_mean[:, 1].tolist(),
                "linear_waypoint_valence": report.linear.waypoint_behavior_mean[:, 0].tolist(),
                "linear_waypoint_arousal": report.linear.waypoint_behavior_mean[:, 1].tolist(),
            }
        )

    # Aggregate summary.
    print("=" * 60)
    deltas = np.array([s["delta_linear_minus_manifold"] for s in per_pair_summary])
    wins = int((deltas > 0).sum())
    losses = int((deltas < 0).sum())
    ties = len(deltas) - wins - losses
    print(
        f"Aggregate: manifold wins {wins} / loses {losses} / ties {ties} of {len(deltas)} pairs"
    )
    print(f"Mean Δ (linear − manifold): {deltas.mean():+.3f}")
    print(f"Per-pair Δ: {[f'{d:+.3f}' for d in deltas]}")

    RESULTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_OUT.write_text(
        json.dumps(
            {
                "pairs": per_pair_summary,
                "aggregate": {
                    "wins": wins,
                    "losses": losses,
                    "ties": ties,
                    "mean_delta": float(deltas.mean()),
                    "per_pair_delta": deltas.tolist(),
                },
            },
            indent=2,
        )
    )
    print(f"saved per-pair results to {RESULTS_OUT}")


if __name__ == "__main__":
    asyncio.run(main())
