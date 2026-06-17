"""Spline integration into the chord experiment: config parsing + steering paths."""

from __future__ import annotations

import numpy as np

from manifold_emotions.experiments.chord import ChordRunConfig, _spline_steer_paths
from manifold_emotions.manifold.spline import SplineManifold


def _toy_spline(d: int = 4, hidden: int = 16) -> SplineManifold:
    rng = np.random.default_rng(3)
    n = 16
    coords = rng.uniform(-3, 3, size=(n, 2))
    targets = rng.normal(size=(n, d))
    return SplineManifold.fit(
        labels=tuple(f"e{i}" for i in range(n)),
        control_coords=coords,
        centroids_subspace=targets,
        pca_components=rng.normal(size=(d, hidden)),
        pca_mean=rng.normal(size=(hidden,)),
        kde_bandwidth=1.0,
        smoothing=0.1,
    )


def test_config_parses_spline_key(tmp_path) -> None:
    cfg = tmp_path / "chord_spline.yaml"
    cfg.write_text(
        "name: spline_8d\n"
        "manifold: data/manifold_h.npz\n"
        "judge: none\n"
        "spline: data/manifold_spline_8d.npz\n"
    )
    run = ChordRunConfig.from_yaml(cfg)
    assert run.spline_path is not None
    assert run.spline_path.name == "manifold_spline_8d.npz"
    assert run.judge == "none"


def test_config_spline_defaults_none(tmp_path) -> None:
    cfg = tmp_path / "chord.yaml"
    cfg.write_text("name: pullback_8d\nmanifold: data/manifold_h.npz\njudge: none\n")
    assert ChordRunConfig.from_yaml(cfg).spline_path is None


def test_spline_steer_paths_shapes_and_endpoints() -> None:
    spline = _toy_spline(d=4, hidden=16)
    k, scale = 30, 8.0
    steer, sub = _spline_steer_paths(
        spline, "e0", "e7", num_waypoints=k, steering_scale=scale
    )
    assert set(steer) == {"spline_induced", "spline_density"}
    for name in steer:
        assert steer[name].shape == (k, 16)
        assert sub[name].shape == (k, 4)
        assert np.isfinite(steer[name]).all()
    # endpoints snapped to the exact centroids, lifted and scaled
    idx = {lab: i for i, lab in enumerate(spline.labels)}
    c0 = spline.centroids_subspace[idx["e0"]]
    expected0 = spline.unproject(c0[None])[0] * scale
    assert np.allclose(steer["spline_induced"][0], expected0, atol=1e-3)
