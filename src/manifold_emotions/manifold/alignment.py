"""Structural metrics characterizing each emotion pair in M_h subspace.

The Day 2 synthesis claims that whether the manifold geodesic beats the
linear chord between two emotions is a property of where the pair's
direction vector sits in the M_h PCA basis, not of where the pair sits
in the V-A plane. This module turns that claim into computable scalars
that can be evaluated for every pair in the corpus without LLM calls.

The metrics are evaluated in the *PCA subspace* of the fitted manifold.
Because the manifold's `centroids_subspace` are already PCA-projected,
each axis is a principal component by construction, so the direction
vector `d = h_end_sub - h_start_sub` has component i equal to the
projection on PC i.

Metrics:

- **Participation ratio** of the unit direction `d / ‖d‖` over the
  PCA axes: 1 if the direction lives entirely along one PC,
  ``num_components`` if it's uniformly spread. Low PR → "direction is
  a single dominant axis," which we predict will be linear-favored.
- **Top-PC fraction**: the squared component of the largest-loading
  PC. Equal to 1 / PR for one-hot directions.
- **Near-chord centroid count**: count of other emotion centroids
  whose perpendicular distance to the chord is less than a radius
  (default: the KDE bandwidth) AND whose projected position falls
  inside the interior of the chord (parameter t ∈ [0.2, 0.8]). These
  are the centroids that the G_E geodesic *can* bend toward — the raw
  "curvature material" available.
- **G_E length gap**: ``linear_length - geodesic_length`` under the
  density metric. Positive means the geodesic found a shorter path
  by exploiting curvature; near zero means the chord is already a
  geodesic and there's nothing to gain from manifold steering.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from ..behavior.manifold import BehaviorManifold
from .fit import FittedManifold


@dataclass(frozen=True, slots=True)
class PairAlignment:
    """Structural metrics for one emotion pair in M_h subspace."""

    start: str
    end: str
    h_distance: float                  # Euclidean ‖h_end_sub - h_start_sub‖
    participation_ratio: float         # 1 (one-PC) .. num_components (uniform)
    top_pc: int                        # argmax |d_unit|
    top_pc_fraction: float             # d_unit[top_pc] ** 2
    near_chord_centroid_count: int
    pc_fractions: np.ndarray           # (num_components,) — d_unit ** 2


def participation_ratio(direction: np.ndarray) -> float:
    """PR = (Σ x_i²)² / Σ x_i⁴ for x = direction/‖direction‖.

    For a unit vector aligned with one axis, PR = 1. For a uniform
    distribution over d axes, PR = d. Interpolates smoothly between.
    """
    norm_sq = float(direction @ direction)
    if norm_sq < 1e-12:
        return 1.0
    unit = direction / np.sqrt(norm_sq)
    fractions = unit * unit
    return float(1.0 / np.sum(fractions * fractions))


def near_chord_centroid_count(
    start: np.ndarray,
    end: np.ndarray,
    others: np.ndarray,
    radius: float,
    interior_band: tuple[float, float] = (0.2, 0.8),
) -> int:
    """Count `others` that sit in the chord's interior with offset < radius.

    Specifically: for each row of `others`, project onto the chord
    direction to get t ∈ ℝ (normalized so t=0 is `start` and t=1 is
    `end`), and compute the orthogonal residual to the chord line.
    Counts rows with t ∈ interior_band and residual norm < radius.

    These are the centroids whose density well the geodesic can detour
    through without leaving the chord's neighborhood — the raw count
    of available "curvature material."
    """
    d = end - start
    L_sq = float(d @ d)
    if L_sq < 1e-12:
        return 0
    L = np.sqrt(L_sq)
    d_unit = d / L
    rel = others - start[None, :]
    t_unit = rel @ d_unit
    t = t_unit / L
    proj = t_unit[:, None] * d_unit[None, :]
    residual = rel - proj
    res_norm = np.linalg.norm(residual, axis=1)
    interior_lo, interior_hi = interior_band
    interior_mask = (t > interior_lo) & (t < interior_hi)
    close_mask = res_norm < radius
    return int(np.sum(interior_mask & close_mask))


def pair_alignment(
    manifold: FittedManifold,
    start_label: str,
    end_label: str,
    *,
    radius_multiplier: float = 1.0,
    interior_band: tuple[float, float] = (0.2, 0.8),
) -> PairAlignment:
    """Compute structural metrics for a single pair.

    `radius_multiplier` scales the KDE bandwidth used as the
    perpendicular threshold for near-chord centroid counting.
    """
    i = manifold.labels.index(start_label)
    j = manifold.labels.index(end_label)
    centroids = manifold.centroids_subspace
    h_i = centroids[i].astype(np.float64)
    h_j = centroids[j].astype(np.float64)
    d = h_j - h_i
    distance = float(np.linalg.norm(d))
    pr = participation_ratio(d)

    fractions = (d / distance) ** 2 if distance > 0 else np.zeros_like(d)
    top_pc = int(np.argmax(fractions))
    top_pc_frac = float(fractions[top_pc])

    mask = np.ones(len(manifold.labels), dtype=bool)
    mask[i] = False
    mask[j] = False
    others = centroids[mask].astype(np.float64)
    radius = manifold.kde_bandwidth * radius_multiplier
    nccc = near_chord_centroid_count(
        h_i, h_j, others, radius, interior_band=interior_band
    )

    return PairAlignment(
        start=start_label,
        end=end_label,
        h_distance=distance,
        participation_ratio=pr,
        top_pc=top_pc,
        top_pc_fraction=top_pc_frac,
        near_chord_centroid_count=nccc,
        pc_fractions=fractions,
    )


def all_pair_alignments(
    manifold: FittedManifold,
    *,
    radius_multiplier: float = 1.0,
    interior_band: tuple[float, float] = (0.2, 0.8),
) -> list[PairAlignment]:
    """Compute alignment metrics for every unordered pair of emotions."""
    out: list[PairAlignment] = []
    for i, j in combinations(range(len(manifold.labels)), 2):
        out.append(
            pair_alignment(
                manifold,
                manifold.labels[i],
                manifold.labels[j],
                radius_multiplier=radius_multiplier,
                interior_band=interior_band,
            )
        )
    return out


def max_chord_deflection(
    geodesic_subspace_waypoints: np.ndarray,
) -> float:
    """Maximum perpendicular distance from a geodesic to its chord, in subspace.

    Distinguishes "geodesic that genuinely curves toward a different part
    of the manifold" from "geodesic that just shaves G_E length while
    riding almost the same path as the chord." The G_E length gap
    captures path-length shortening but treats both cases as equal;
    max deflection differentiates them.

    Returns 0 for a path that lies on the chord. For curves that bow
    away from the chord, returns the perpendicular distance of the
    farthest waypoint from the chord line. Units match the subspace
    (same units as the centroid coordinates).
    """
    if geodesic_subspace_waypoints.shape[0] < 3:
        return 0.0
    waypoints = geodesic_subspace_waypoints.astype(np.float64)
    start = waypoints[0]
    end = waypoints[-1]
    d = end - start
    L_sq = float(d @ d)
    if L_sq < 1e-12:
        return 0.0
    L = np.sqrt(L_sq)
    d_unit = d / L
    rel = waypoints - start[None, :]
    proj_scalar = rel @ d_unit
    proj = proj_scalar[:, None] * d_unit[None, :]
    perp = rel - proj
    return float(np.linalg.norm(perp, axis=1).max())


def predicted_off_my_energy(
    geodesic_subspace_waypoints: np.ndarray,
    manifold: FittedManifold,
    behavior: BehaviorManifold,
) -> float:
    """Estimate the off-M_y energy a geodesic would produce without LLM calls.

    For each waypoint along the M_h geodesic, find the nearest emotion
    centroid in subspace, look up that centroid's (V, A) in M_y, and
    treat that as the "predicted behavior" at the waypoint. Then
    compute the M_y-straight-line distance from this predicted trace
    to the target chord, averaged over waypoints.

    Rationale: when manifold steering produces real text, the behavior
    at a waypoint should resemble that of whatever emotion(s) the
    waypoint is close to in M_h. If the geodesic detours through
    centroids whose M_y coordinates already lie near the start-end
    straight line, the resulting behavior follows the line — manifold
    wins. If it detours through centroids that are M_y-line-far, the
    behavior wanders off-target — linear wins.

    Returns the mean perpendicular distance (in (V, A) units) from the
    predicted-behavior trace to the target M_y chord. Lower is better
    for the manifold framework.
    """
    if geodesic_subspace_waypoints.shape[0] < 2:
        return float("nan")

    centroids_sub = manifold.centroids_subspace.astype(np.float64)
    waypoints = geodesic_subspace_waypoints.astype(np.float64)
    # (K, num_emotions) pairwise distances in subspace.
    diffs = waypoints[:, None, :] - centroids_sub[None, :, :]
    dists = np.sqrt(np.sum(diffs * diffs, axis=-1))
    nearest_idx = np.argmin(dists, axis=1)

    # Map M_h emotion indices to M_y rows (labels may differ in order).
    my_centroid_by_label = {
        label: behavior.centroids[i] for i, label in enumerate(behavior.labels)
    }
    predicted_my: list[np.ndarray] = []
    for k in nearest_idx:
        label = manifold.labels[int(k)]
        coord = my_centroid_by_label.get(label)
        if coord is None:
            return float("nan")
        predicted_my.append(np.asarray(coord, dtype=np.float64))
    predicted = np.stack(predicted_my, axis=0)

    # Target M_y straight line: from predicted[0] to predicted[-1], by t.
    # We use the endpoints' actual M_y centroids since the geodesic by
    # construction starts and ends at the pair's emotion centroids.
    y_start = predicted[0]
    y_end = predicted[-1]
    chord = y_end - y_start
    chord_len_sq = float(chord @ chord)
    if chord_len_sq < 1e-12:
        return float(np.mean(np.linalg.norm(predicted - y_start[None, :], axis=1)))
    chord_unit = chord / np.sqrt(chord_len_sq)
    rel = predicted - y_start[None, :]
    proj_scalar = rel @ chord_unit
    proj = proj_scalar[:, None] * chord_unit[None, :]
    residual = rel - proj
    perp_dist = np.linalg.norm(residual, axis=1)
    return float(perp_dist.mean())
