"""Thin-plate spline manifold: interpolation, Jacobian, metric, persistence."""

from __future__ import annotations

import numpy as np

from manifold_emotions.manifold.spline import SplineManifold


def _grid_coords(n_side: int = 5, lo: float = -2.0, hi: float = 2.0) -> np.ndarray:
    xs = np.linspace(lo, hi, n_side)
    gx, gy = np.meshgrid(xs, xs)
    return np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float64)


def _affine_targets(coords: np.ndarray, d: int = 4) -> np.ndarray:
    rng = np.random.default_rng(1)
    m = rng.normal(size=(2, d))
    b = rng.normal(size=(d,))
    return (coords @ m + b).astype(np.float64)


def _paraboloid_targets(coords: np.ndarray) -> np.ndarray:
    """A genuinely curved 3-D surface: [x, y, 0.5(x^2+y^2)]."""
    z = 0.5 * (coords[:, 0] ** 2 + coords[:, 1] ** 2)
    return np.stack([coords[:, 0], coords[:, 1], z], axis=1).astype(np.float64)


def _make_spline(coords: np.ndarray, targets: np.ndarray, smoothing: float = 0.0) -> SplineManifold:
    n, d = targets.shape
    return SplineManifold.fit(
        labels=tuple(f"e{i}" for i in range(n)),
        control_coords=coords,
        centroids_subspace=targets,
        pca_components=np.eye(d, 16, dtype=np.float64),  # arbitrary valid lift
        pca_mean=np.zeros(16, dtype=np.float64),
        kde_bandwidth=1.0,
        smoothing=smoothing,
    )


def test_interpolates_control_points_exactly_at_zero_smoothing() -> None:
    coords = _grid_coords()
    targets = _paraboloid_targets(coords)
    spline = _make_spline(coords, targets, smoothing=0.0)
    embedded = spline.embed_np(coords.astype(np.float32))
    assert np.allclose(embedded, targets, atol=1e-3)


def test_affine_data_is_reproduced_offgrid() -> None:
    coords = _grid_coords()
    targets = _affine_targets(coords, d=4)
    spline = _make_spline(coords, targets, smoothing=0.0)
    # Off-grid query: TPS of affine data must equal the affine map (no warp).
    rng = np.random.default_rng(7)
    q = rng.uniform(-2, 2, size=(10, 2)).astype(np.float64)
    # Recover the affine map from the data and compare.
    a, *_ = np.linalg.lstsq(np.concatenate([np.ones((coords.shape[0], 1)), coords], axis=1), targets, rcond=None)
    expected = np.concatenate([np.ones((10, 1)), q], axis=1) @ a
    assert np.allclose(spline.embed_np(q.astype(np.float32)), expected, atol=1e-2)


def test_jacobian_matches_finite_difference() -> None:
    coords = _grid_coords()
    targets = _paraboloid_targets(coords)
    spline = _make_spline(coords, targets, smoothing=0.0)
    u = np.array([0.3, -0.4], dtype=np.float32)
    j_analytic = spline.jacobian(u)
    eps = 1e-3
    j_fd = np.zeros_like(j_analytic)
    for k in range(2):
        du = np.zeros(2, dtype=np.float32)
        du[k] = eps
        j_fd[:, k] = (spline.embed_np(u + du) - spline.embed_np(u - du)) / (2 * eps)
    assert np.allclose(j_analytic, j_fd, atol=5e-2)


def test_induced_metric_is_symmetric_psd() -> None:
    coords = _grid_coords()
    targets = _paraboloid_targets(coords)
    spline = _make_spline(coords, targets, smoothing=0.0)
    g = spline.induced_metric(np.array([0.2, 0.5], dtype=np.float32))
    assert np.allclose(g, g.T, atol=1e-5)
    eigvals = np.linalg.eigvalsh(g)
    assert (eigvals >= -1e-6).all()


def test_unproject_matches_pca_lift() -> None:
    coords = _grid_coords()
    targets = _paraboloid_targets(coords)
    spline = _make_spline(coords, targets, smoothing=0.0)
    sub = np.array([[1.0, -2.0, 0.5]], dtype=np.float32)
    expected = sub @ spline.pca_components + spline.pca_mean[None, :]
    assert np.allclose(spline.unproject(sub), expected, atol=1e-5)


def test_save_load_roundtrip(tmp_path) -> None:
    coords = _grid_coords()
    targets = _paraboloid_targets(coords)
    spline = _make_spline(coords, targets, smoothing=0.25)
    path = tmp_path / "spline.npz"
    spline.save(path)
    loaded = SplineManifold.load(path)
    q = np.array([[0.1, 0.2], [-0.5, 0.8]], dtype=np.float32)
    assert np.allclose(spline.embed_np(q), loaded.embed_np(q), atol=1e-5)
    assert loaded.smoothing == spline.smoothing
    assert loaded.labels == spline.labels
