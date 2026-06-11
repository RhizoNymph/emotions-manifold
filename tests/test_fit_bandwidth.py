"""Bandwidth-spec resolution in manifold fitting."""

from __future__ import annotations

import numpy as np
import pytest

from manifold_emotions.manifold.density import (
    clustered_bandwidth,
    silverman_bandwidth,
)
from manifold_emotions.manifold.fit import fit_manifold, resolve_bandwidth
from manifold_emotions.types import EmotionLabel
from manifold_emotions.vectors.diff_in_means import EmotionVectors


def _synthetic_vectors(num_emotions: int = 8, hidden: int = 32) -> EmotionVectors:
    rng = np.random.default_rng(seed=11)
    thetas = np.linspace(0, 2 * np.pi, num_emotions, endpoint=False)
    signal = np.stack(
        [np.cos(thetas), np.sin(thetas), np.zeros_like(thetas)], axis=1
    )
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


def test_default_bandwidth_is_clustered_nn() -> None:
    ev = _synthetic_vectors()
    default_manifold, _ = fit_manifold(ev, num_components=4)
    explicit_manifold, _ = fit_manifold(ev, num_components=4, bandwidth="clustered_nn")
    expected = clustered_bandwidth(default_manifold.centroids_subspace, multiplier=1.0)
    assert default_manifold.kde_bandwidth == pytest.approx(expected)
    assert explicit_manifold.kde_bandwidth == default_manifold.kde_bandwidth


def test_silverman_bandwidth_spec() -> None:
    ev = _synthetic_vectors()
    manifold, _ = fit_manifold(ev, num_components=4, bandwidth="silverman")
    expected = silverman_bandwidth(manifold.centroids_subspace)
    assert manifold.kde_bandwidth == pytest.approx(float(expected))
    # On this data the two heuristics must genuinely differ, or the test
    # proves nothing.
    clustered = clustered_bandwidth(manifold.centroids_subspace, multiplier=1.0)
    assert manifold.kde_bandwidth != pytest.approx(clustered)


def test_float_bandwidth_passthrough() -> None:
    ev = _synthetic_vectors()
    manifold, _ = fit_manifold(ev, num_components=4, bandwidth=0.123)
    assert manifold.kde_bandwidth == 0.123


def test_resolve_bandwidth_rejects_unknown_spec() -> None:
    centroids = np.zeros((4, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="unknown bandwidth spec"):
        resolve_bandwidth("median_pairwise", centroids)  # type: ignore[arg-type]
