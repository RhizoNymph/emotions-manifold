"""Compute emotion vectors from captured activations and save to disk.

Walks the capture root, computes per-emotion centroids (skipping the
first 50 tokens per Anthropic), diff-in-means against the global mean,
and writes the result to ``config.paths.emotion_vectors``.

Run with:
    uv run python scripts/compute_vectors.py
"""

from __future__ import annotations

from manifold_emotions.config import load_config
from manifold_emotions.vectors.diff_in_means import compute_emotion_vectors


def main() -> None:
    config = load_config()
    result = compute_emotion_vectors(config)
    result.save(config.paths.emotion_vectors)
    print(
        f"saved {result.vectors.shape[0]} emotion vectors "
        f"(hidden_size={result.hidden_size}) to {config.paths.emotion_vectors}"
    )
    print(
        f"story counts per emotion: "
        f"min={result.story_counts.min()} "
        f"max={result.story_counts.max()} "
        f"mean={result.story_counts.mean():.1f}"
    )


if __name__ == "__main__":
    main()
