"""Structural-alignment metric tests."""

from __future__ import annotations

import numpy as np

from manifold_emotions.behavior.manifold import BehaviorManifold
from manifold_emotions.manifold.alignment import (
    max_chord_deflection,
    near_chord_centroid_count,
    pair_alignment,
    participation_ratio,
    predicted_off_my_energy,
)
from manifold_emotions.manifold.fit import fit_manifold
from manifold_emotions.types import EmotionLabel
from manifold_emotions.vectors.diff_in_means import EmotionVectors


def test_participation_ratio_one_hot_direction_is_one() -> None:
    """A direction lying exactly on one PCA axis should have PR = 1."""
    for k in range(5):
        d = np.zeros(5, dtype=np.float64)
        d[k] = 3.0
        assert participation_ratio(d) == 1.0


def test_participation_ratio_uniform_direction_equals_dim() -> None:
    """A direction with equal magnitude on every axis has PR = num_axes."""
    for dim in (2, 4, 8, 16):
        d = np.ones(dim, dtype=np.float64)
        assert abs(participation_ratio(d) - dim) < 1e-10


def test_participation_ratio_is_scale_invariant() -> None:
    rng = np.random.default_rng(0)
    d = rng.normal(size=10)
    base = participation_ratio(d)
    assert abs(participation_ratio(d * 7.5) - base) < 1e-10
    assert abs(participation_ratio(-d) - base) < 1e-10


def test_participation_ratio_zero_vector_returns_one() -> None:
    """Edge case: zero direction shouldn't blow up; treat as one-hot-like."""
    assert participation_ratio(np.zeros(5)) == 1.0


def test_near_chord_count_zero_for_far_centroids() -> None:
    """Centroids far perpendicular to a chord shouldn't be counted."""
    start = np.array([0.0, 0.0, 0.0])
    end = np.array([10.0, 0.0, 0.0])
    others = np.array([
        [5.0, 100.0, 0.0],   # interior in t, but very far perpendicular
        [5.0, 0.0, 100.0],   # ditto on other axis
        [-50.0, 0.0, 0.0],   # outside interior band
    ])
    assert near_chord_centroid_count(start, end, others, radius=1.0) == 0


def test_near_chord_count_one_for_interior_close_centroid() -> None:
    """A centroid sitting interior in t and inside the radius is counted."""
    start = np.array([0.0, 0.0])
    end = np.array([10.0, 0.0])
    others = np.array([
        [5.0, 0.5, ][:2],   # midpoint with small offset
    ])
    assert near_chord_centroid_count(start, end, others, radius=1.0) == 1


def test_near_chord_count_respects_interior_band() -> None:
    """A centroid at t=0.1 (outside default 0.2 band) shouldn't be counted."""
    start = np.array([0.0, 0.0])
    end = np.array([10.0, 0.0])
    near_endpoint = np.array([[1.0, 0.5]])     # t = 0.1
    near_midpoint = np.array([[5.0, 0.5]])     # t = 0.5
    assert near_chord_centroid_count(start, end, near_endpoint, radius=1.0) == 0
    assert near_chord_centroid_count(start, end, near_midpoint, radius=1.0) == 1


def test_pair_alignment_endpoints_are_excluded_from_near_chord() -> None:
    """The pair's own endpoints sit on the chord — they must not be counted."""
    rng = np.random.default_rng(7)
    num_emotions = 5
    hidden = 12
    # Five emotions evenly spaced 0..100 along axis 0. With chord
    # spanning the full range, interior t values are 0.25, 0.5, 0.75 —
    # cleanly inside the default (0.2, 0.8) band — and all three sit on
    # the chord line with zero perpendicular offset, so the count must
    # be 3 (endpoints e00 and e04 are excluded as the pair itself).
    vectors = rng.normal(0, 0.01, size=(num_emotions, hidden)).astype(np.float32)
    vectors[:, 0] = np.linspace(0, 100, num_emotions).astype(np.float32)
    labels = tuple(EmotionLabel(f"e{i:02d}") for i in range(num_emotions))
    ev = EmotionVectors(
        labels=labels,
        vectors=vectors,
        centroids=vectors,
        global_mean=np.zeros(hidden, dtype=np.float32),
        story_counts=np.full(num_emotions, 10, dtype=np.int64),
        hidden_size=hidden,
        skip_tokens_before=50,
    )
    manifold, _ = fit_manifold(ev, num_components=3)
    align = pair_alignment(manifold, "e00", "e04")
    assert align.near_chord_centroid_count == 3


def test_pair_alignment_top_pc_consistent_with_pc_fractions() -> None:
    rng = np.random.default_rng(11)
    num_emotions = 5
    hidden = 8
    vectors = rng.normal(size=(num_emotions, hidden)).astype(np.float32)
    labels = tuple(EmotionLabel(f"e{i:02d}") for i in range(num_emotions))
    ev = EmotionVectors(
        labels=labels,
        vectors=vectors,
        centroids=vectors,
        global_mean=np.zeros(hidden, dtype=np.float32),
        story_counts=np.full(num_emotions, 10, dtype=np.int64),
        hidden_size=hidden,
        skip_tokens_before=50,
    )
    manifold, _ = fit_manifold(ev, num_components=3)
    align = pair_alignment(manifold, "e00", "e04")
    assert align.top_pc == int(np.argmax(align.pc_fractions))
    assert abs(align.top_pc_fraction - float(align.pc_fractions[align.top_pc])) < 1e-10
    # Squared fractions must sum to 1
    assert abs(float(np.sum(align.pc_fractions)) - 1.0) < 1e-5


def test_predicted_off_my_energy_zero_for_linear_chord_through_colinear_my() -> None:
    """If every M_h centroid maps to an M_y coord that lies on a single
    line, then any geodesic that snaps to those centroids should yield
    a predicted behavior trace that's also on the line — perpendicular
    distance to the chord must be 0."""
    rng = np.random.default_rng(31)
    num_emotions = 5
    hidden = 8
    vectors = rng.normal(0, 0.01, size=(num_emotions, hidden)).astype(np.float32)
    vectors[:, 0] = np.linspace(0, 100, num_emotions).astype(np.float32)
    labels = tuple(EmotionLabel(f"e{i:02d}") for i in range(num_emotions))
    ev = EmotionVectors(
        labels=labels,
        vectors=vectors,
        centroids=vectors,
        global_mean=np.zeros(hidden, dtype=np.float32),
        story_counts=np.full(num_emotions, 10, dtype=np.int64),
        hidden_size=hidden,
        skip_tokens_before=50,
    )
    manifold, _ = fit_manifold(ev, num_components=3)

    # All M_y coords lie on the line y = x (perfectly colinear).
    my_centroids = np.stack(
        [np.arange(num_emotions, dtype=np.float32),
         np.arange(num_emotions, dtype=np.float32)],
        axis=1,
    )
    behavior = BehaviorManifold(
        labels=labels,
        centroids=my_centroids,
        stds=np.zeros_like(my_centroids),
        story_counts=np.full(num_emotions, 10, dtype=np.int64),
    )

    # Take the linear interpolation in subspace as the "geodesic" — it
    # snaps to each centroid in order, which yields a colinear M_y trace.
    waypoints = np.stack(
        [(1 - t) * manifold.centroids_subspace[0] + t * manifold.centroids_subspace[-1]
         for t in np.linspace(0, 1, 5)],
        axis=0,
    )
    pred = predicted_off_my_energy(waypoints, manifold, behavior)
    assert pred < 1e-4


def test_predicted_off_my_energy_positive_when_geodesic_visits_off_line_emotion() -> None:
    """If the geodesic visits a centroid whose M_y coord is far from the
    chord line, the predicted off-M_y energy must be positive."""
    rng = np.random.default_rng(41)
    hidden = 8
    # Three emotions: start at h=(0,...), end at h=(10,0,...), detour at h=(5,5,...)
    vectors = rng.normal(0, 0.01, size=(3, hidden)).astype(np.float32)
    vectors[0, :2] = (0.0, 0.0)
    vectors[1, :2] = (10.0, 0.0)
    vectors[2, :2] = (5.0, 5.0)
    labels = (EmotionLabel("start"), EmotionLabel("end"), EmotionLabel("detour"))
    ev = EmotionVectors(
        labels=labels,
        vectors=vectors,
        centroids=vectors,
        global_mean=np.zeros(hidden, dtype=np.float32),
        story_counts=np.full(3, 10, dtype=np.int64),
        hidden_size=hidden,
        skip_tokens_before=50,
    )
    manifold, _ = fit_manifold(ev, num_components=3)

    # M_y: start and end on V-axis, detour way off the V-axis.
    my_centroids = np.array(
        [[1.0, 1.0],   # start
         [5.0, 1.0],   # end (same arousal as start)
         [3.0, 7.0]],  # detour (high arousal, well off the chord)
        dtype=np.float32,
    )
    behavior = BehaviorManifold(
        labels=labels,
        centroids=my_centroids,
        stds=np.zeros_like(my_centroids),
        story_counts=np.full(3, 10, dtype=np.int64),
    )

    # A 3-waypoint "geodesic" that detours through the off-chord centroid.
    waypoints = np.stack(
        [manifold.centroids_subspace[0],
         manifold.centroids_subspace[2],
         manifold.centroids_subspace[1]],
        axis=0,
    )
    pred = predicted_off_my_energy(waypoints, manifold, behavior)
    assert pred > 1.0  # Detour M_y is (3, 7); chord is arousal=1 line; ~6 units off


def test_max_chord_deflection_zero_for_straight_path() -> None:
    """A path that lies exactly on its own chord has zero deflection."""
    waypoints = np.stack([
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([2.0, 0.0, 0.0]),
        np.array([3.0, 0.0, 0.0]),
    ])
    assert max_chord_deflection(waypoints) < 1e-10


def test_max_chord_deflection_recovers_explicit_offset() -> None:
    """A waypoint deliberately offset by Δ perpendicular to the chord
    must yield a max deflection of Δ."""
    # Chord from (0,0) to (10,0). Middle waypoint sits at (5, 2.5).
    waypoints = np.array([
        [0.0, 0.0],
        [3.0, 0.5],
        [5.0, 2.5],
        [7.0, 0.5],
        [10.0, 0.0],
    ])
    assert abs(max_chord_deflection(waypoints) - 2.5) < 1e-10


def test_max_chord_deflection_translation_invariant() -> None:
    base = np.array([
        [0.0, 0.0],
        [5.0, 3.0],
        [10.0, 0.0],
    ])
    offset = np.array([42.0, -7.0])
    assert abs(max_chord_deflection(base) - max_chord_deflection(base + offset)) < 1e-10


def test_max_chord_deflection_zero_for_two_waypoints() -> None:
    """No interior waypoints → no deflection to measure."""
    pts = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    assert max_chord_deflection(pts) == 0.0
