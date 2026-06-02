"""Manifold-fitting tests on synthetic 2-D point clouds."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from manifold_emotions.manifold.density import GaussianKDE, silverman_bandwidth
from manifold_emotions.manifold.geodesic import fit_geodesic, linear_interpolation
from manifold_emotions.manifold.metric import DensityGeometry


def test_kde_density_higher_near_centroids() -> None:
    """A KDE built on centroids at +/- 1 should peak at +/- 1, not at the origin."""
    centroids = np.array([[-1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    kde = GaussianKDE.fit(centroids, bandwidth=0.2)

    near_centroid = float(kde.log_density(jnp.array([1.0, 0.0])))
    at_origin = float(kde.log_density(jnp.array([0.0, 0.0])))
    far = float(kde.log_density(jnp.array([10.0, 10.0])))

    assert near_centroid > at_origin > far


def test_silverman_bandwidth_positive_and_scale_invariant() -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(0, 1.0, size=(50, 4)).astype(np.float32)
    h1 = silverman_bandwidth(jnp.asarray(data))
    h2 = silverman_bandwidth(jnp.asarray(data * 5.0))
    # Bandwidth scales linearly with data scale.
    assert h2 == pytest.approx(h1 * 5.0, rel=0.05)
    assert h1 > 0


def test_path_length_zero_when_endpoints_coincide() -> None:
    centroids = np.array([[0.0, 0.0]], dtype=np.float32)
    kde = GaussianKDE.fit(centroids, bandwidth=0.5)
    geom = DensityGeometry(energy_fn=kde.energy, alpha=1.0, beta=0.01)

    waypoints = jnp.zeros((5, 2), dtype=jnp.float32)
    length = float(geom.path_length(waypoints))
    assert length == pytest.approx(0.0, abs=1e-6)


def test_geodesic_bends_around_off_manifold_region() -> None:
    """Two centroids at (-1, 0) and (1, 0). The linear path runs straight
    through the origin where density is moderate. With strong alpha and
    bandwidth, the optimized geodesic should follow the high-density
    region — staying close to y=0 since density is concentrated there.

    This test mainly checks that:
    1. The optimizer converges to something shorter than the linear path
       in the metric we care about (not Euclidean — under G_E).
    2. The optimized path still goes between the endpoints.
    """
    centroids = np.array([[-1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    kde = GaussianKDE.fit(centroids, bandwidth=0.4)
    geom = DensityGeometry(energy_fn=kde.energy, alpha=1.0, beta=0.01)

    start = centroids[0]
    end = centroids[1]
    result = fit_geodesic(geom, start, end, num_waypoints=20, max_iter=100)

    # Endpoints must match exactly.
    np.testing.assert_allclose(result.waypoints[0], start)
    np.testing.assert_allclose(result.waypoints[-1], end)
    # Number of waypoints matches request.
    assert result.num_waypoints == 20
    # Final length under G_E should be no worse than initial (which was
    # the linear interpolation). The metric makes near-centroid regions
    # cheap; the optimizer should at least not make things worse.
    assert result.final_length <= result.initial_length + 1e-3


def test_geodesic_prefers_high_density_route_in_curved_manifold() -> None:
    """Place centroids around a semicircle in the upper half plane. The
    linear path from leftmost to rightmost cuts through y=0, where
    density is low. The geodesic under G_E should bend upward through
    the high-density arc.
    """
    # Semicircle of radius 1 centered at origin, upper half, 9 centroids.
    thetas = np.linspace(0, np.pi, 9)
    centroids = np.stack(
        [np.cos(thetas), np.sin(thetas)], axis=1
    ).astype(np.float32)
    kde = GaussianKDE.fit(centroids, bandwidth=0.2)
    geom = DensityGeometry(energy_fn=kde.energy, alpha=10.0, beta=0.001)

    start = centroids[0]  # (-1, 0)
    end = centroids[-1]  # (1, 0)

    result = fit_geodesic(geom, start, end, num_waypoints=30, max_iter=200)

    # The optimized path should achieve lower G_E length than the linear
    # baseline (which travels along y=0 through low-density territory).
    assert result.final_length < result.initial_length

    # And the interior midpoint should have moved upward — toward the
    # high-density arc — relative to the linear interpolation (which
    # places it at the origin (0, 0)).
    optimized_midpoint = result.waypoints[len(result.waypoints) // 2]
    assert optimized_midpoint[1] > 0.3


def test_linear_interpolation_endpoints_and_count() -> None:
    start = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    end = np.array([1.0, 2.0, -1.0], dtype=np.float32)
    way = linear_interpolation(start, end, 10)
    assert way.shape == (10, 3)
    np.testing.assert_allclose(way[0], start)
    np.testing.assert_allclose(way[-1], end)
    # Middle should be the average.
    np.testing.assert_allclose(way[len(way) // 2], (start + end) / 2.0, atol=0.15)
