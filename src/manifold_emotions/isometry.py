"""Pairwise-distance correlation between M_h (activation) and M_y (behavior).

This is the load-bearing empirical check in our project: if M_h and M_y
are isometric (the same shape up to scale), Goodfire's manifold-steering
framework applies and Phase 5 experiments are worth running. If they
aren't, the manifold view of the emotion concept space is wrong and
we need to slice the data differently before proceeding.

Concretely:
1. Compute pairwise distances on M_h (Euclidean in PCA subspace, since
   the subspace is already a flat chart of the manifold — or use
   geodesic distance along the density-geometry surface for a stricter
   comparison).
2. Compute pairwise distances on M_y (Euclidean in (valence, arousal),
   since 2-D Euclidean is a faithful metric on a flat plane).
3. Pearson and Spearman correlation between the upper-triangular
   distance vectors.
4. Also compute distances under a "linear in activation space" baseline
   (Euclidean in the original 5376-D residual stream space) so we can
   show — per Goodfire's protocol — that the manifold metric correlates
   better than the flat one.

Goodfire saw r=0.99 on simple conceptual spaces. For emotions, r>0.7
would be encouraging.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import pearsonr, spearmanr


@dataclass(frozen=True, slots=True)
class IsometryReport:
    labels: tuple[str, ...]
    # Pairwise distance matrices, all (num_emotions, num_emotions).
    m_h_subspace_distances: np.ndarray
    m_h_linear_distances: np.ndarray  # in full activation space
    m_y_distances: np.ndarray
    # Pearson / Spearman correlations across the upper-triangular entries.
    pearson_subspace_vs_behavior: float
    pearson_linear_vs_behavior: float
    spearman_subspace_vs_behavior: float
    spearman_linear_vs_behavior: float


def _pairwise_euclidean(points: np.ndarray) -> np.ndarray:
    """(N, d) → (N, N) symmetric matrix of pairwise L2 distances."""
    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt((diff * diff).sum(axis=-1))


def _upper_triangular_vector(matrix: np.ndarray) -> np.ndarray:
    """Extract the strict upper-triangle (i < j) into a 1-D vector."""
    n = matrix.shape[0]
    rows, cols = np.triu_indices(n, k=1)
    return matrix[rows, cols]


def check_isometry(
    *,
    labels: tuple[str, ...],
    m_h_subspace_centroids: np.ndarray,  # (N, d_pca)
    m_h_full_centroids: np.ndarray,  # (N, hidden_size)
    m_y_centroids: np.ndarray,  # (N, 2)  — (valence, arousal)
) -> IsometryReport:
    """Compute the cross-manifold distance correlations.

    All three centroid arrays must be aligned with the same ``labels``
    ordering — row i of every array refers to the same emotion concept.
    """
    n = len(labels)
    if (
        m_h_subspace_centroids.shape[0] != n
        or m_h_full_centroids.shape[0] != n
        or m_y_centroids.shape[0] != n
    ):
        raise ValueError(
            f"centroid row counts must equal len(labels)={n}; got "
            f"subspace={m_h_subspace_centroids.shape[0]}, "
            f"full={m_h_full_centroids.shape[0]}, "
            f"behavior={m_y_centroids.shape[0]}"
        )

    d_subspace = _pairwise_euclidean(m_h_subspace_centroids)
    d_linear = _pairwise_euclidean(m_h_full_centroids)
    d_behavior = _pairwise_euclidean(m_y_centroids)

    v_subspace = _upper_triangular_vector(d_subspace)
    v_linear = _upper_triangular_vector(d_linear)
    v_behavior = _upper_triangular_vector(d_behavior)

    pearson_sub_beh, _ = pearsonr(v_subspace, v_behavior)
    pearson_lin_beh, _ = pearsonr(v_linear, v_behavior)
    spearman_sub_beh, _ = spearmanr(v_subspace, v_behavior)
    spearman_lin_beh, _ = spearmanr(v_linear, v_behavior)

    return IsometryReport(
        labels=labels,
        m_h_subspace_distances=d_subspace,
        m_h_linear_distances=d_linear,
        m_y_distances=d_behavior,
        pearson_subspace_vs_behavior=float(pearson_sub_beh),
        pearson_linear_vs_behavior=float(pearson_lin_beh),
        spearman_subspace_vs_behavior=float(spearman_sub_beh),
        spearman_linear_vs_behavior=float(spearman_lin_beh),
    )
