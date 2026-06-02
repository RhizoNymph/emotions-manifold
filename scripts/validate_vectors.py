"""Validate extracted emotion vectors against LLM-judged valence/arousal.

Loads emotion vectors from ``config.paths.emotion_vectors``, calls
Claude on each emotion label to get valence/arousal ratings (cached to
``data/emotion_ratings.json``), runs PCA, and reports which PCA axes
best match each affective dimension.

Anthropic's reference numbers from Sonnet 4.5:
    valence ↔ PC1: r ≈ 0.81
    arousal ↔ PC2 or PC3: r ≈ 0.66

For Gemma 4 31B, anything above ~0.5 is encouraging.

Run with:
    uv run python scripts/validate_vectors.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from manifold_emotions.config import load_config
from manifold_emotions.vectors.diff_in_means import EmotionVectors
from manifold_emotions.vectors.judge import judge_emotions
from manifold_emotions.vectors.validate import validate_emotion_vectors

RATINGS_CACHE = Path("data/emotion_ratings.json")


async def main() -> None:
    config = load_config()
    ev = EmotionVectors.load(config.paths.emotion_vectors)

    ratings = await judge_emotions(
        config=config,
        emotions=list(ev.labels),
        cache_path=RATINGS_CACHE,
    )

    report = validate_emotion_vectors(
        vectors=ev.vectors,
        labels=ev.labels,
        ratings=ratings,
        candidate_axis_count=5,
    )

    print("Top 5 PCA axes — explained variance ratios:")
    for i, r in enumerate(report.pca.explained_variance_ratio):
        print(f"  PC{i+1}: {r:.3f}")
    print()
    for dim, axis_report in report.best_axis.items():
        print(
            f"Best {dim} axis: PC{axis_report.axis_index + 1} "
            f"(r={axis_report.pearson_r:+.3f}, "
            f"explains {axis_report.explained_variance_ratio:.1%} of variance)"
        )
    print()
    print("All axis × dimension correlations:")
    print(f"  {'axis':<6} {'valence_r':>10} {'arousal_r':>10}")
    for i, (v_r, a_r) in enumerate(
        zip(
            report.all_correlations["valence"],
            report.all_correlations["arousal"],
            strict=True,
        )
    ):
        print(f"  PC{i+1:<4} {v_r:>+10.3f} {a_r:>+10.3f}")


if __name__ == "__main__":
    asyncio.run(main())
