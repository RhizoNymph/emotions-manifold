"""Pullback construction tests on synthetic emotion + behavior centroids."""

from __future__ import annotations

import numpy as np

from manifold_emotions.behavior.manifold import BehaviorManifold
from manifold_emotions.manifold.fit import fit_manifold
from manifold_emotions.manifold.pullback import (
    _kernel_weights,
    _knn_distance_from_point,
    _median_nn_distance,
    compute_pullback,
    construct_pullback_path,
)
from manifold_emotions.types import EmotionLabel
from manifold_emotions.vectors.diff_in_means import EmotionVectors


def _synthetic_circumplex(
    num_emotions: int = 8, hidden: int = 32
) -> tuple[EmotionVectors, BehaviorManifold]:
    """Eight emotions arranged in a ring, aligned 1:1 in M_h and M_y.

    Lifting the M_y ring back via barycentric pullback should reproduce
    the M_h ring exactly (modulo PCA discretization) — the cleanest
    sanity check for the kernel-barycenter inverse.
    """
    rng = np.random.default_rng(seed=37)
    thetas = np.linspace(0, 2 * np.pi, num_emotions, endpoint=False)
    # M_h signal: cos/sin ring in first two dims, noise elsewhere
    h_signal = np.stack([np.cos(thetas), np.sin(thetas), np.zeros_like(thetas)], axis=1)
    h_vectors = rng.normal(0.0, 0.02, size=(num_emotions, hidden)).astype(np.float32)
    h_vectors[:, :3] += h_signal.astype(np.float32)
    labels = tuple(EmotionLabel(f"e{i:02d}") for i in range(num_emotions))
    ev = EmotionVectors(
        labels=labels,
        vectors=h_vectors,
        centroids=h_vectors,
        global_mean=np.zeros(hidden, dtype=np.float32),
        story_counts=np.full(num_emotions, 10, dtype=np.int64),
        hidden_size=hidden,
        skip_tokens_before=50,
    )
    # M_y: identical ring in 2-D, with valence=cos and arousal=sin.
    my_centroids = np.stack([np.cos(thetas), np.sin(thetas)], axis=1).astype(np.float32)
    my = BehaviorManifold(
        labels=labels,
        centroids=my_centroids,
        stds=np.zeros_like(my_centroids),
        story_counts=np.full(num_emotions, 10, dtype=np.int64),
    )
    return ev, my


def test_kernel_weights_sum_to_one() -> None:
    rng = np.random.default_rng(seed=0)
    anchors = rng.normal(size=(5, 2)).astype(np.float32)
    target = rng.normal(size=(2,)).astype(np.float32)
    w = _kernel_weights(target, anchors, sigma=0.5)
    assert w.shape == (5,)
    assert np.isclose(w.sum(), 1.0, atol=1e-6)
    assert (w >= 0.0).all()


def test_median_nn_distance_on_unit_square() -> None:
    """Four corners of a unit square — NN distance = 1.0 for every point."""
    pts = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    assert _median_nn_distance(pts) == 1.0


def test_pullback_path_starts_and_ends_at_centroids() -> None:
    """Endpoints of the pullback must equal the centroids' M_h positions
    (the constructor snaps them so all three paths share endpoints)."""
    ev, my = _synthetic_circumplex()
    manifold, _ = fit_manifold(ev, num_components=4)

    result = compute_pullback(
        manifold, my, "e00", "e04", num_waypoints=11, geodesic_max_iter=20,
    )
    start_h = manifold.centroids_subspace[manifold.labels.index("e00")]
    end_h = manifold.centroids_subspace[manifold.labels.index("e04")]
    np.testing.assert_allclose(result.pullback_sub[0], start_h, atol=1e-5)
    np.testing.assert_allclose(result.pullback_sub[-1], end_h, atol=1e-5)
    np.testing.assert_allclose(result.geodesic_sub[0], start_h, atol=1e-5)
    np.testing.assert_allclose(result.linear_sub[0], start_h, atol=1e-5)


def test_pullback_interior_lies_in_subspace_hull() -> None:
    """Kernel barycenter weights are convex (sum to 1, all positive), so
    every interior pullback point must lie in the convex hull of the
    M_h centroids."""
    ev, my = _synthetic_circumplex()
    manifold, _ = fit_manifold(ev, num_components=4)
    _, pullback_sub, *_ = construct_pullback_path(
        manifold, my, "e00", "e04", num_waypoints=15,
    )
    # Convex-hull membership test: every interior pullback point should
    # be expressible as a convex combination of centroids. Easier proxy:
    # its norm should be <= max centroid norm.
    max_centroid_norm = np.linalg.norm(manifold.centroids_subspace, axis=1).max()
    interior_norms = np.linalg.norm(pullback_sub[1:-1], axis=1)
    assert (interior_norms <= max_centroid_norm + 1e-4).all()


def test_knn_distance_from_point_on_unit_square() -> None:
    """K-NN distance from the origin to corners of a unit square.

    The four corners are at distances {0, 1, 1, √2} from (0,0). K=1 is 0
    (origin itself counts as nearest if included; here we query a point
    not in the anchor set so K=1 is the closest non-origin corner = 1)."""
    anchors = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    target = np.array([0.0, 0.0], dtype=np.float32)
    assert _knn_distance_from_point(target, anchors, k=1) == 0.0
    assert _knn_distance_from_point(target, anchors, k=2) == 1.0
    assert np.isclose(_knn_distance_from_point(target, anchors, k=4), np.sqrt(2.0))
    # K clipped to [1, N]
    assert _knn_distance_from_point(target, anchors, k=10) == _knn_distance_from_point(target, anchors, k=4)


def test_adaptive_sigma_per_waypoint_widens_in_sparse_regions() -> None:
    """K-NN-adaptive σ should be wider where the M_y point set is sparser.

    Build M_y with a tight cluster on one side and an isolated centroid
    on the other. A waypoint near the isolated point should get a
    wider σ than a waypoint inside the cluster."""
    rng = np.random.default_rng(seed=7)
    # 6 emotions: 5 tight cluster around (0, 0), 1 isolated at (3, 0)
    h_signal = rng.normal(0.0, 1.0, size=(6, 3)).astype(np.float32)
    my_centroids = np.array(
        [[0.0, 0.0], [0.1, 0.1], [-0.1, 0.05], [0.05, -0.1], [-0.05, -0.05], [3.0, 0.0]],
        dtype=np.float32,
    )
    labels = tuple(EmotionLabel(f"e{i:02d}") for i in range(6))
    ev = EmotionVectors(
        labels=labels,
        vectors=h_signal,
        centroids=h_signal,
        global_mean=np.zeros(3, dtype=np.float32),
        story_counts=np.full(6, 10, dtype=np.int64),
        hidden_size=3,
        skip_tokens_before=50,
    )
    my = BehaviorManifold(
        labels=labels,
        centroids=my_centroids,
        stds=np.zeros_like(my_centroids),
        story_counts=np.full(6, 10, dtype=np.int64),
    )
    manifold, _ = fit_manifold(ev, num_components=2)
    # Chord from cluster center (e00) to the isolated point (e05) — early
    # waypoints sit in dense region, late waypoints sit in sparse region.
    _, _, _, sigma_per, spec = construct_pullback_path(
        manifold, my, "e00", "e05", num_waypoints=10, sigma="knn:3",
    )
    assert spec == "knn:3"
    assert sigma_per.shape == (10,)
    # σ near the isolated point should be much wider than σ inside cluster
    assert sigma_per[-1] > 2.0 * sigma_per[0]


def test_knn_scaled_sigma_preserves_shape_but_shifts_magnitude() -> None:
    """`knn:K*scale` should give per-waypoint σ exactly equal to the
    unscaled K-NN σ multiplied by the scale, on the same chord."""
    rng = np.random.default_rng(seed=11)
    h = rng.normal(0.0, 1.0, size=(8, 3)).astype(np.float32)
    my = np.stack(
        [np.cos(np.linspace(0, 2 * np.pi, 8, endpoint=False)),
         np.sin(np.linspace(0, 2 * np.pi, 8, endpoint=False))],
        axis=1,
    ).astype(np.float32)
    labels = tuple(EmotionLabel(f"e{i:02d}") for i in range(8))
    ev = EmotionVectors(
        labels=labels, vectors=h, centroids=h,
        global_mean=np.zeros(3, dtype=np.float32),
        story_counts=np.full(8, 10, dtype=np.int64),
        hidden_size=3, skip_tokens_before=50,
    )
    beh = BehaviorManifold(
        labels=labels, centroids=my, stds=np.zeros_like(my),
        story_counts=np.full(8, 10, dtype=np.int64),
    )
    manifold, _ = fit_manifold(ev, num_components=2)
    _, _, _, sigma_unscaled, spec_un = construct_pullback_path(
        manifold, beh, "e00", "e04", num_waypoints=12, sigma="knn:3",
    )
    _, _, _, sigma_scaled, spec_sc = construct_pullback_path(
        manifold, beh, "e00", "e04", num_waypoints=12, sigma="knn:3*0.5",
    )
    assert spec_un == "knn:3"
    assert spec_sc == "knn:3*0.5"
    np.testing.assert_allclose(sigma_scaled, 0.5 * sigma_unscaled, atol=1e-9)


def test_pullback_on_ring_lands_near_geodesic_not_linear() -> None:
    """On a perfectly-aligned ring (M_h ring matches M_y ring), pulling
    back an M_y straight line between two non-antipodal points should
    produce a curve that mirrors the M_h geodesic — both bend along the
    ring, not through the center.

    Antipodal endpoints (e.g. e00→e06 on a 12-ring) are degenerate: the
    chord passes through the M_y origin which is equidistant from every
    centroid, so the pullback flattens to roughly a straight line itself
    and the prediction doesn't apply.
    """
    ev, my = _synthetic_circumplex(num_emotions=12)
    manifold, _ = fit_manifold(ev, num_components=4)
    # e00 -> e03 is a 90° arc — the chord stays inside the ring and the
    # geodesic should bow outward toward higher-density centroids.
    result = compute_pullback(
        manifold, my, "e00", "e03", num_waypoints=20, geodesic_max_iter=300,
    )
    assert result.mean_dist_to_geodesic < result.mean_dist_to_linear
