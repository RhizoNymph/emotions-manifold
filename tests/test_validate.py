"""End-to-end validation: synthetic vectors with embedded valence/arousal signal."""

from __future__ import annotations

import numpy as np

from manifold_emotions.vectors.judge import EmotionRating
from manifold_emotions.vectors.validate import validate_emotion_vectors


def test_validation_recovers_embedded_valence_arousal() -> None:
    """Build a synthetic emotion-vector matrix where dim 0 encodes valence
    and dim 1 encodes arousal, plus noise. PCA should rediscover both axes.
    """
    rng = np.random.default_rng(seed=7)
    labels = tuple(f"emotion_{i:02d}" for i in range(40))

    valence_scores = rng.uniform(1.0, 7.0, size=40)
    arousal_scores = rng.uniform(1.0, 7.0, size=40)

    # Hidden size 64, signal in first two dims, noise everywhere.
    vectors = rng.normal(0, 0.1, size=(40, 64))
    vectors[:, 0] = (valence_scores - 4.0) * 3.0  # strong valence signal in dim 0
    vectors[:, 1] = (arousal_scores - 4.0) * 3.0  # strong arousal signal in dim 1

    ratings = {
        label: EmotionRating(emotion=label, valence=float(v), arousal=float(a))
        for label, v, a in zip(labels, valence_scores, arousal_scores, strict=True)
    }

    report = validate_emotion_vectors(
        vectors=vectors.astype(np.float32),
        labels=labels,
        ratings=ratings,
        candidate_axis_count=5,
    )

    # Embedded valence/arousal should be recovered with high correlation.
    assert abs(report.best_axis["valence"].pearson_r) > 0.9
    assert abs(report.best_axis["arousal"].pearson_r) > 0.9
    # And they should land on different axes.
    assert (
        report.best_axis["valence"].axis_index
        != report.best_axis["arousal"].axis_index
    )
