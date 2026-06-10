"""All-pairs geodesic cache: build, lookup, round-trip."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from manifold_emotions.manifold.fit import fit_manifold
from manifold_emotions.manifold.geodesic_cache import (
    GeodesicCache,
    build_geodesic_cache,
)
from manifold_emotions.types import EmotionLabel
from manifold_emotions.vectors.diff_in_means import EmotionVectors


def _tiny_manifold(num_emotions: int = 4):
    rng = np.random.default_rng(seed=23)
    thetas = np.linspace(0, 2 * np.pi, num_emotions, endpoint=False)
    signal = np.stack(
        [np.cos(thetas), np.sin(thetas), np.zeros_like(thetas)], axis=1
    )
    vectors = rng.normal(0, 0.05, size=(num_emotions, 16)).astype(np.float32)
    vectors[:, :3] += signal.astype(np.float32)
    ev = EmotionVectors(
        labels=tuple(EmotionLabel(f"e{i:02d}") for i in range(num_emotions)),
        vectors=vectors,
        centroids=vectors,
        global_mean=np.zeros(16, dtype=np.float32),
        story_counts=np.full(num_emotions, 10, dtype=np.int64),
        hidden_size=16,
        skip_tokens_before=50,
    )
    manifold, _ = fit_manifold(ev, num_components=2)
    return manifold


def test_build_covers_all_pairs_with_fixed_endpoints() -> None:
    manifold = _tiny_manifold(num_emotions=4)
    cache = build_geodesic_cache(manifold, num_waypoints=8, max_iter=50)

    assert cache.waypoints.shape == (6, 8, 2)  # C(4,2) pairs
    assert cache.num_waypoints == 8
    assert cache.labels == manifold.labels
    # Endpoints of each path must be the centroids of its pair.
    for k, (i, j) in enumerate(cache.pair_indices):
        np.testing.assert_allclose(
            cache.waypoints[k, 0], manifold.centroids_subspace[i], atol=1e-4
        )
        np.testing.assert_allclose(
            cache.waypoints[k, -1], manifold.centroids_subspace[j], atol=1e-4
        )


def test_lookup_orients_path_by_argument_order() -> None:
    manifold = _tiny_manifold(num_emotions=4)
    cache = build_geodesic_cache(manifold, num_waypoints=8, max_iter=50)

    forward = cache.lookup(0, 2)
    backward = cache.lookup(2, 0)
    np.testing.assert_array_equal(forward, backward[::-1])
    np.testing.assert_allclose(forward[0], manifold.centroids_subspace[0], atol=1e-4)
    np.testing.assert_allclose(forward[-1], manifold.centroids_subspace[2], atol=1e-4)

    with pytest.raises(KeyError, match="not in geodesic cache"):
        cache.lookup(0, 99)


def test_round_trip_preserves_arrays(tmp_path: Path) -> None:
    manifold = _tiny_manifold(num_emotions=3)
    cache = build_geodesic_cache(manifold, num_waypoints=6, max_iter=50)

    out = tmp_path / "cache.npz"
    cache.save(out)
    loaded = GeodesicCache.load(out)

    assert loaded.labels == cache.labels
    assert loaded.num_waypoints == cache.num_waypoints
    np.testing.assert_array_equal(loaded.waypoints, cache.waypoints)
    np.testing.assert_array_equal(loaded.pair_indices, cache.pair_indices)
