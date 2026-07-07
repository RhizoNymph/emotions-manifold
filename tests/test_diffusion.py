"""Diffusion-map coordinate construction (bijective spline parameterization)."""

from __future__ import annotations

import numpy as np

from manifold_emotions.manifold.diffusion import diffusion_embed


def test_shape_and_finite() -> None:
    rng = np.random.default_rng(0)
    centroids = rng.normal(size=(40, 8))
    u = diffusion_embed(centroids, 2)
    assert u.shape == (40, 2)
    assert u.dtype == np.float64
    assert np.isfinite(u).all()


def test_distinct_points_for_distinct_centroids() -> None:
    """Bijectivity: distinct centroids map to distinct diffusion coordinates."""
    rng = np.random.default_rng(1)
    centroids = rng.normal(size=(30, 6))
    u = diffusion_embed(centroids, 2)
    # No two rows coincide (pairwise min distance strictly positive).
    d = np.linalg.norm(u[:, None, :] - u[None, :, :], axis=2)
    d[np.diag_indices_from(d)] = np.inf
    assert d.min() > 0.0


def test_deterministic() -> None:
    rng = np.random.default_rng(2)
    centroids = rng.normal(size=(25, 5))
    assert np.array_equal(diffusion_embed(centroids, 2), diffusion_embed(centroids, 2))
