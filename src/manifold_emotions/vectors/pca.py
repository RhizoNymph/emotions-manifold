"""PCA over the emotion vectors.

Anthropic reports that PC1 ≈ valence and PC2 (or PC3) ≈ arousal in
Sonnet 4.5's residual-stream emotion vectors. We need to identify these
axes in our Gemma 4 31B vectors and correlate them with LLM-judged
valence/arousal scores per the same paper's protocol.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class PCAResult:
    """Principal components of a (num_samples, hidden_size) matrix.

    ``components[i]`` is the i-th principal axis (unit vector in
    hidden_size space). ``projections[i, j]`` is sample i's projection
    onto axis j. ``explained_variance_ratio[j]`` is the fraction of
    total variance captured by axis j.
    """

    components: np.ndarray  # (n_components, hidden_size)
    projections: np.ndarray  # (num_samples, n_components)
    explained_variance: np.ndarray  # (n_components,)
    explained_variance_ratio: np.ndarray  # (n_components,)
    mean: np.ndarray  # (hidden_size,)


def fit_pca(vectors: np.ndarray, n_components: int | None = None) -> PCAResult:
    """SVD-based PCA over the row vectors.

    No external dep — uses numpy's full SVD. ``n_components`` defaults
    to ``min(num_samples, hidden_size)`` (the full rank). Sign of each
    component is arbitrary under SVD; downstream code that cares about
    direction should align via correlation with external labels (e.g.
    LLM-judged valence).
    """
    if vectors.ndim != 2:
        raise ValueError(f"vectors must be 2-D, got shape {vectors.shape}")

    num_samples, hidden_size = vectors.shape
    max_components = min(num_samples, hidden_size)
    if n_components is None:
        n_components = max_components
    if n_components > max_components:
        raise ValueError(
            f"n_components={n_components} exceeds rank {max_components} "
            f"for shape {vectors.shape}"
        )

    mean = vectors.mean(axis=0)
    centered = vectors - mean[None, :]

    # full_matrices=False yields U (N, k), s (k,), Vt (k, hidden) where k=min(N, hidden).
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)

    components = vt[:n_components]
    projections = centered @ components.T
    explained_variance = (singular_values[:n_components] ** 2) / max(1, num_samples - 1)
    total_variance = (singular_values**2).sum() / max(1, num_samples - 1)
    explained_variance_ratio = explained_variance / (total_variance or 1.0)

    return PCAResult(
        components=components,
        projections=projections,
        explained_variance=explained_variance,
        explained_variance_ratio=explained_variance_ratio,
        mean=mean,
    )


def align_axis_signs(
    pca: PCAResult,
    target_scores: np.ndarray,
    axis_indices: tuple[int, ...],
) -> PCAResult:
    """Flip the sign of selected PCA axes to maximize Pearson correlation with target_scores.

    target_scores is (num_samples,) — one score per row of the original
    vectors matrix. For each axis in axis_indices, if the correlation
    between projections[:, axis] and target_scores is negative, flip
    the sign of the component AND the projection. Other axes are
    untouched.
    """
    if target_scores.shape[0] != pca.projections.shape[0]:
        raise ValueError(
            f"target_scores length {target_scores.shape[0]} != "
            f"projections rows {pca.projections.shape[0]}"
        )

    components = pca.components.copy()
    projections = pca.projections.copy()

    for axis in axis_indices:
        corr = np.corrcoef(projections[:, axis], target_scores)[0, 1]
        if np.isfinite(corr) and corr < 0:
            components[axis] = -components[axis]
            projections[:, axis] = -projections[:, axis]

    return PCAResult(
        components=components,
        projections=projections,
        explained_variance=pca.explained_variance,
        explained_variance_ratio=pca.explained_variance_ratio,
        mean=pca.mean,
    )
