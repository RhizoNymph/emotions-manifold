"""Manifold-fitting integration test on synthetic emotion vectors."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from manifold_emotions.manifold.fit import FittedManifold, fit_manifold
from manifold_emotions.types import EmotionLabel
from manifold_emotions.vectors.diff_in_means import EmotionVectors


def _synthetic_vectors(num_emotions: int = 8, hidden: int = 32) -> EmotionVectors:
    """Build a small EmotionVectors whose signal lives in a 2-D subspace."""
    rng = np.random.default_rng(seed=11)
    # 2-D ring layout for the signal (mimics circumplex-like structure).
    thetas = np.linspace(0, 2 * np.pi, num_emotions, endpoint=False)
    signal = np.stack(
        [np.cos(thetas), np.sin(thetas), np.zeros_like(thetas)], axis=1
    )
    # Embed in `hidden` dims with the signal in dims 0-2 and noise elsewhere.
    vectors = rng.normal(0, 0.05, size=(num_emotions, hidden)).astype(np.float32)
    vectors[:, :3] += signal.astype(np.float32)
    return EmotionVectors(
        labels=tuple(EmotionLabel(f"e{i:02d}") for i in range(num_emotions)),
        vectors=vectors,
        centroids=vectors,
        global_mean=np.zeros(hidden, dtype=np.float32),
        story_counts=np.full(num_emotions, 10, dtype=np.int64),
        hidden_size=hidden,
        skip_tokens_before=50,
    )


def test_fit_manifold_captures_subspace_variance() -> None:
    ev = _synthetic_vectors()
    manifold, pca = fit_manifold(ev, num_components=4)
    # Three signal dims + noise — top 2 PCs should dominate.
    assert pca.explained_variance_ratio[0] + pca.explained_variance_ratio[1] > 0.8
    assert manifold.num_components == 4
    assert manifold.centroids_subspace.shape == (8, 4)


def test_fit_manifold_round_trip(tmp_path: Path) -> None:
    ev = _synthetic_vectors()
    manifold, _ = fit_manifold(ev, num_components=4)
    out = tmp_path / "m.npz"
    manifold.save(out)
    loaded = FittedManifold.load(out)
    assert loaded.labels == manifold.labels
    np.testing.assert_array_equal(loaded.centroids_subspace, manifold.centroids_subspace)
    np.testing.assert_array_equal(loaded.pca_components, manifold.pca_components)
    np.testing.assert_array_equal(loaded.pca_mean, manifold.pca_mean)
    assert loaded.kde_bandwidth == pytest.approx(manifold.kde_bandwidth)
    assert loaded.alpha == manifold.alpha
    assert loaded.beta == manifold.beta


def test_project_and_unproject_are_inverses_for_subspace_points() -> None:
    """Points already in the manifold subspace should round-trip through
    project(unproject(x)) without distortion (up to PCA-truncation error).
    """
    ev = _synthetic_vectors()
    manifold, _ = fit_manifold(ev, num_components=4)

    # Take the centroids themselves: they live in the subspace by construction.
    subspace_points = manifold.centroids_subspace
    full = manifold.unproject(subspace_points)
    back = manifold.project(full)
    np.testing.assert_allclose(back, subspace_points, atol=1e-4)


def test_fit_manifold_rejects_too_many_components() -> None:
    ev = _synthetic_vectors(num_emotions=4, hidden=8)
    with pytest.raises(ValueError, match="num_components"):
        fit_manifold(ev, num_components=20)
