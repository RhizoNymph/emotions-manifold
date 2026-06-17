"""Geodesics on a thin-plate-spline surface."""

from __future__ import annotations

import numpy as np

from manifold_emotions.manifold.spline import SplineManifold
from manifold_emotions.manifold.spline_geodesic import fit_spline_geodesic


def _grid_coords(n_side: int = 5, lo: float = -2.0, hi: float = 2.0) -> np.ndarray:
    xs = np.linspace(lo, hi, n_side)
    gx, gy = np.meshgrid(xs, xs)
    return np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float64)


def _make_spline(coords: np.ndarray, targets: np.ndarray, bw: float = 1.0) -> SplineManifold:
    n, d = targets.shape
    return SplineManifold.fit(
        labels=tuple(f"e{i}" for i in range(n)),
        control_coords=coords,
        centroids_subspace=targets,
        pca_components=np.eye(d, 16, dtype=np.float64),
        pca_mean=np.zeros(16, dtype=np.float64),
        kde_bandwidth=bw,
        smoothing=0.0,
    )


def _linear(a: np.ndarray, b: np.ndarray, k: int) -> np.ndarray:
    ts = np.linspace(0.0, 1.0, k)
    return (1 - ts)[:, None] * a[None, :] + ts[:, None] * b[None, :]


def test_flat_surface_geodesic_is_the_ambient_straight_line() -> None:
    # Affine embedding -> surface is a plane -> induced geodesic is straight.
    coords = _grid_coords()
    rng = np.random.default_rng(2)
    m = rng.normal(size=(2, 4))
    b = rng.normal(size=(4,))
    targets = coords @ m + b
    spline = _make_spline(coords, targets)

    start, end = np.array([-1.5, -1.0]), np.array([1.0, 1.5])
    res = fit_spline_geodesic(spline, start, end, metric="induced", num_waypoints=12)
    ambient_chord = _linear(spline.embed_np(start.astype(np.float32)),
                            spline.embed_np(end.astype(np.float32)), 12)
    assert np.allclose(res.waypoints, ambient_chord, atol=1e-2)


def test_curved_surface_geodesic_bends_and_shortens() -> None:
    # Paraboloid: a genuinely curved surface. The surface geodesic must be
    # shorter than the embedded straight-coordinate path and must deviate
    # from the ambient chord (which would leave the surface).
    coords = _grid_coords()
    z = 0.5 * (coords[:, 0] ** 2 + coords[:, 1] ** 2)
    targets = np.stack([coords[:, 0], coords[:, 1], z], axis=1)
    spline = _make_spline(coords, targets)

    start, end = np.array([-1.5, -1.5]), np.array([1.5, 1.5])
    res = fit_spline_geodesic(spline, start, end, metric="induced", num_waypoints=20)

    assert res.final_length <= res.initial_length + 1e-4
    # genuine bend: parameter path departs from the straight coord line
    coord_chord = _linear(start, end, 20)
    assert np.max(np.linalg.norm(res.coords - coord_chord, axis=1)) > 1e-2
    # and the embedded geodesic departs from the ambient straight chord
    ambient_chord = _linear(res.waypoints[0], res.waypoints[-1], 20)
    assert np.max(np.linalg.norm(res.waypoints - ambient_chord, axis=1)) > 1e-2


def test_endpoints_are_snapped_when_requested() -> None:
    coords = _grid_coords()
    z = 0.5 * (coords[:, 0] ** 2 + coords[:, 1] ** 2)
    targets = np.stack([coords[:, 0], coords[:, 1], z], axis=1)
    spline = _make_spline(coords, targets)
    snap_a = np.array([9.0, 9.0, 9.0], dtype=np.float32)
    snap_b = np.array([-9.0, -9.0, -9.0], dtype=np.float32)
    res = fit_spline_geodesic(
        spline, np.array([-1.0, -1.0]), np.array([1.0, 1.0]),
        metric="induced", num_waypoints=10, snap_start=snap_a, snap_end=snap_b,
    )
    assert np.allclose(res.waypoints[0], snap_a)
    assert np.allclose(res.waypoints[-1], snap_b)


def test_density_metric_differs_from_induced_on_curved_surface() -> None:
    coords = _grid_coords()
    z = 0.5 * (coords[:, 0] ** 2 + coords[:, 1] ** 2)
    targets = np.stack([coords[:, 0], coords[:, 1], z], axis=1)
    spline = _make_spline(coords, targets, bw=0.8)

    start, end = np.array([-1.5, 0.0]), np.array([1.5, 0.0])
    induced = fit_spline_geodesic(spline, start, end, metric="induced", num_waypoints=16)
    density = fit_spline_geodesic(spline, start, end, metric="density", num_waypoints=16)
    # The two metrics route the surface path differently.
    assert np.max(np.linalg.norm(induced.coords - density.coords, axis=1)) > 1e-3
