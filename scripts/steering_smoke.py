"""Phase 8 smoke: run the manifold-vs-linear steering comparison on one emotion pair.

Cheap end-to-end shakeout (1 pair × 10 waypoints × 3 prompts = 60 generations
per trajectory, 120 total + 120 judge calls) before committing GPU time to
the full Phase 8 sweep across many emotion pairs.

Run with:
    uv run python scripts/steering_smoke.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from manifold_emotions.behavior.manifold import BehaviorManifold
from manifold_emotions.config import load_config
from manifold_emotions.manifold.fit import FittedManifold
from manifold_emotions.steering.experiment import compare_steering
from manifold_emotions.vectors.diff_in_means import EmotionVectors

JUDGE_CACHE = Path("data/steering_ratings.json")


async def main() -> None:
    config = load_config()
    ev = EmotionVectors.load(config.paths.emotion_vectors)
    mh = FittedManifold.load(config.paths.manifold_h)
    my = BehaviorManifold.load(config.paths.manifold_y)

    pair = ("happy", "sad")
    print(f"Steering comparison: {pair[0]} → {pair[1]}")
    report = await compare_steering(
        config=config,
        emotion_vectors=ev,
        manifold=mh,
        behavior=my,
        start_label=pair[0],
        end_label=pair[1],
        num_waypoints=10,
        num_prompts=3,
        max_tokens=96,
        concurrency=8,
        judge_cache_path=JUDGE_CACHE,
    )

    print()
    print(f"  Manifold off-manifold energy: {report.manifold.cumulative_off_manifold:.3f}")
    print(f"  Linear   off-manifold energy: {report.linear.cumulative_off_manifold:.3f}")
    delta = report.linear.cumulative_off_manifold - report.manifold.cumulative_off_manifold
    print(f"  Δ (linear − manifold): {delta:+.3f}  (positive = manifold beats linear)")

    print()
    print("Per-waypoint behavior (valence, arousal):")
    print(f"  {'wp':>3} {'manifold V':>11} {'manifold A':>11}    {'linear V':>9} {'linear A':>9}")
    for k in range(report.manifold.waypoint_behavior_mean.shape[0]):
        mv, ma = report.manifold.waypoint_behavior_mean[k]
        lv, la = report.linear.waypoint_behavior_mean[k]
        print(f"  {k:>3} {mv:>11.2f} {ma:>11.2f}    {lv:>9.2f} {la:>9.2f}")


if __name__ == "__main__":
    asyncio.run(main())
