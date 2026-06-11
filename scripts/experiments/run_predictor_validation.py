"""Validate the G_E-gap predictor on 3 untested pairs at the 8-D operating point.

Day 2 closed with five measured pairs whose Δ values plausibly track
the G_E length gap but don't separate cleanly within the ±0.05 judge-
noise floor. The afternoon follow-up flagged three untested pairs with
G_E gaps substantially larger (4.2–8.2 vs ≤2.5 for any measured pair),
predicting large manifold wins.

This script runs K=30, N=10 manifold-vs-linear comparisons on those
three pairs and writes fresh per-pair judge caches to avoid stale-hit
poisoning. Output schema matches ``run_subspace_sweep.py`` so plots
can be regenerated with the existing tooling.

Pairs:
- excited → relaxed     (G_E gap = 4.74) — moderate-gap test
- calm → energized      (G_E gap = 4.41) — moderate-gap test
- calm → enthusiastic   (G_E gap = 8.20) — extreme-gap test, doubles
                                            as "is `enthusiastic` a real
                                            outlier or metric artifact?"

Run with:
    uv run python scripts/run_predictor_validation.py
"""

from __future__ import annotations

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

# (start, end, predicted_ge_gap) — predictions from results/pair_alignment.json.
# The first three are the original round; the next three are a refinement
# pass to fill the gap ∈ [3, 4] band where calm→energized won big but
# excited→relaxed underperformed. Resumes idempotently — pairs whose
# result JSON exists are skipped (delete to force re-run).
PAIRS: tuple[tuple[str, str, float], ...] = (
    ("excited", "relaxed", 4.74),
    ("calm", "energized", 4.41),
    ("calm", "enthusiastic", 8.20),
    ("depressed", "excited", 3.31),
    ("calm", "excited", 3.83),
    ("depressed", "energized", 3.96),
)

NUM_WAYPOINTS = 30
NUM_PROMPTS = 10

OUT_DATA_DIR = Path("data/pair_validation")
OUT_RESULTS_DIR = Path("results/pair_validation")


async def run_pair(
    config,
    emotion_vectors: EmotionVectors,
    manifold: FittedManifold,
    behavior: BehaviorManifold,
    start: str,
    end: str,
    predicted_gap: float,
) -> dict:
    cache_file = OUT_DATA_DIR / f"ratings_{start}_{end}.json"
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
    )

    m_e = report.manifold.cumulative_off_manifold
    l_e = report.linear.cumulative_off_manifold
    delta = l_e - m_e

    my_start = behavior.centroids[behavior.labels.index(start)]
    my_end = behavior.centroids[behavior.labels.index(end)]
    chord = float(np.linalg.norm(my_end - my_start))

    summary = {
        "pair": [start, end],
        "predicted_ge_gap": predicted_gap,
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
    out_path = OUT_RESULTS_DIR / f"{start}_{end}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    return summary


async def main() -> None:
    config = load_config()
    ev = EmotionVectors.load(config.paths.emotion_vectors)
    mh = FittedManifold.load(config.paths.manifold_h)
    behavior = BehaviorManifold.load(config.paths.manifold_y)

    OUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    overall_start = time.monotonic()
    summary_rows: list[dict] = []

    for start, end, predicted_gap in PAIRS:
        if start not in mh.labels or end not in mh.labels:
            print(f"  skipping {start}→{end}: missing from M_h labels", flush=True)
            continue
        result_path = OUT_RESULTS_DIR / f"{start}_{end}.json"
        if result_path.exists():
            existing = json.loads(result_path.read_text())
            print(
                f"  {start}→{end} already done "
                f"(Δ={existing['delta_linear_minus_manifold']:+.3f}), skipping",
                flush=True,
            )
            summary_rows.append(existing)
            continue
        print()
        print(f"=== {start} → {end}  (predicted G_E gap = {predicted_gap}) ===", flush=True)
        pair_start = time.monotonic()
        summary = await run_pair(
            config=config,
            emotion_vectors=ev,
            manifold=mh,
            behavior=behavior,
            start=start,
            end=end,
            predicted_gap=predicted_gap,
        )
        pair_elapsed = time.monotonic() - pair_start
        print(
            f"  manifold E={summary['manifold_off_manifold_energy']:.3f}  "
            f"linear E={summary['linear_off_manifold_energy']:.3f}  "
            f"Δ={summary['delta_linear_minus_manifold']:+.3f}  "
            f"({pair_elapsed:.0f}s)",
            flush=True,
        )
        summary_rows.append(summary)

    overall_elapsed = time.monotonic() - overall_start
    print()
    print(f"=== validation complete in {overall_elapsed/60:.1f} min ===", flush=True)
    print()
    print(
        f"  {'pair':>22s}  {'pred_gap':>8s}  {'manifold E':>10s}  "
        f"{'linear E':>10s}  {'Δ':>7s}"
    )
    for row in summary_rows:
        print(
            f"  {row['pair'][0]+'→'+row['pair'][1]:>22s}  "
            f"{row['predicted_ge_gap']:>8.2f}  "
            f"{row['manifold_off_manifold_energy']:>10.3f}  "
            f"{row['linear_off_manifold_energy']:>10.3f}  "
            f"{row['delta_linear_minus_manifold']:>+7.3f}"
        )


if __name__ == "__main__":
    asyncio.run(main())
