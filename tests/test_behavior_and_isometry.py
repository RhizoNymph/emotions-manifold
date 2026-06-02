"""Behavior aggregation + isometry check on synthetic data."""

from __future__ import annotations

import numpy as np
import pytest

from manifold_emotions.behavior.judge_text import TextRating
from manifold_emotions.behavior.manifold import (
    BehaviorManifold,
    aggregate_behavior_manifold,
)
from manifold_emotions.isometry import check_isometry


def test_aggregate_behavior_manifold_groups_by_emotion() -> None:
    text_id_to_emotion = {
        "happy_0": "happy",
        "happy_1": "happy",
        "sad_0": "sad",
        "sad_1": "sad",
    }
    ratings = {
        "happy_0": TextRating(text_id="happy_0", valence=6.0, arousal=5.0),
        "happy_1": TextRating(text_id="happy_1", valence=7.0, arousal=6.0),
        "sad_0": TextRating(text_id="sad_0", valence=2.0, arousal=2.0),
        "sad_1": TextRating(text_id="sad_1", valence=1.5, arousal=2.5),
    }

    bm = aggregate_behavior_manifold(text_id_to_emotion, ratings)
    assert bm.labels == ("happy", "sad")
    np.testing.assert_allclose(bm.centroids[bm.labels.index("happy")], [6.5, 5.5])
    np.testing.assert_allclose(bm.centroids[bm.labels.index("sad")], [1.75, 2.25])
    assert bm.story_counts.tolist() == [2, 2]


def test_aggregate_behavior_manifold_round_trip(tmp_path) -> None:
    bm = BehaviorManifold(
        labels=("happy", "sad"),
        centroids=np.array([[6.5, 5.5], [1.75, 2.25]], dtype=np.float32),
        stds=np.array([[0.5, 0.5], [0.25, 0.25]], dtype=np.float32),
        story_counts=np.array([2, 2], dtype=np.int64),
    )
    path = tmp_path / "my.npz"
    bm.save(path)
    loaded = BehaviorManifold.load(path)
    assert loaded.labels == bm.labels
    np.testing.assert_array_equal(loaded.centroids, bm.centroids)
    np.testing.assert_array_equal(loaded.stds, bm.stds)
    np.testing.assert_array_equal(loaded.story_counts, bm.story_counts)


def test_isometry_perfect_when_subspace_matches_behavior() -> None:
    """Construct M_h_subspace and M_y as the same 2-D point set. Pairwise
    distances should match exactly, yielding Pearson = 1.0. The full-space
    centroids are the same points padded with noise that dwarfs the signal
    in absolute scale; that drives the linear-baseline correlation down
    relative to the subspace one.
    """
    rng = np.random.default_rng(7)
    n = 12
    labels = tuple(f"e{i:02d}" for i in range(n))

    # 2-D point cloud — used identically for subspace and behavior so the
    # pairwise distance vectors match exactly.
    points_2d = rng.normal(0, 1.0, size=(n, 2))
    subspace = points_2d
    behavior = points_2d

    # Full-space centroids: same 2-D signal in dims 0-1 plus 100 dims of
    # high-variance noise. The noise dominates absolute pairwise distances
    # but the pattern (which pairs are closer) is randomized.
    noise = rng.normal(0, 10.0, size=(n, 100))
    full = np.hstack([points_2d, noise])

    report = check_isometry(
        labels=labels,
        m_h_subspace_centroids=subspace,
        m_h_full_centroids=full,
        m_y_centroids=behavior,
    )

    # Subspace and behavior are identical → perfect Pearson.
    assert report.pearson_subspace_vs_behavior == pytest.approx(1.0, abs=1e-10)
    # The noise scrambles the linear-space distance pattern, so the linear
    # baseline correlation must be strictly worse than the subspace one.
    assert report.pearson_linear_vs_behavior < report.pearson_subspace_vs_behavior


def test_isometry_zero_when_behavior_is_random() -> None:
    rng = np.random.default_rng(3)
    n = 10
    labels = tuple(f"e{i:02d}" for i in range(n))
    subspace = rng.normal(0, 1.0, size=(n, 4))
    behavior = rng.normal(0, 1.0, size=(n, 2))  # totally unrelated
    full = subspace  # doesn't matter

    report = check_isometry(
        labels=labels,
        m_h_subspace_centroids=subspace,
        m_h_full_centroids=full,
        m_y_centroids=behavior,
    )
    # No structural relationship → correlation near zero.
    assert abs(report.pearson_subspace_vs_behavior) < 0.5


def test_isometry_rejects_mismatched_shapes() -> None:
    labels = ("a", "b", "c")
    with pytest.raises(ValueError, match="centroid row counts"):
        check_isometry(
            labels=labels,
            m_h_subspace_centroids=np.zeros((3, 4)),
            m_h_full_centroids=np.zeros((2, 4)),  # wrong row count
            m_y_centroids=np.zeros((3, 2)),
        )
