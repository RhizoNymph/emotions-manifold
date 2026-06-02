"""End-to-end manifold fitting from emotion vectors.

Takes a (num_emotions, hidden_size) emotion-vector matrix, projects it
into a lower-dimensional subspace via PCA (per Goodfire: 64 dims), fits
a Gaussian KDE through the projected centroids, and packages everything
into a ``FittedManifold`` that Phase 5 (steering) can load directly.

We save the PCA basis alongside the KDE so geodesics in the subspace
can be lifted back into the full residual-stream space when we need
to inject steering vectors into the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..vectors.diff_in_means import EmotionVectors
from ..vectors.pca import PCAResult, fit_pca
from .density import GaussianKDE, clustered_bandwidth
from .metric import DensityGeometry


@dataclass(frozen=True, slots=True)
class FittedManifold:
    """A density-geometry manifold fitted in a PCA-reduced subspace.

    ``labels`` are the emotion concept names. ``centroids_subspace[i]``
    is the projection of emotion i's vector into the PCA subspace.
    ``pca_components`` (num_components, hidden_size) and ``pca_mean``
    (hidden_size,) let callers project new vectors into the subspace
    and unproject geodesic waypoints back to the full residual stream:

        h_sub = (h_full - pca_mean) @ pca_components.T
        h_full = h_sub @ pca_components + pca_mean
    """

    labels: tuple[str, ...]
    centroids_subspace: np.ndarray  # (num_emotions, num_components)
    pca_components: np.ndarray  # (num_components, hidden_size)
    pca_mean: np.ndarray  # (hidden_size,)
    pca_explained_variance_ratio: np.ndarray  # (num_components,)
    kde_bandwidth: float
    alpha: float
    beta: float

    @property
    def num_components(self) -> int:
        return self.centroids_subspace.shape[1]

    @property
    def hidden_size(self) -> int:
        return self.pca_components.shape[1]

    def project(self, full_vectors: np.ndarray) -> np.ndarray:
        """Project (num_samples, hidden_size) vectors into the manifold subspace."""
        centered = full_vectors - self.pca_mean[None, :]
        return centered @ self.pca_components.T

    def unproject(self, subspace_vectors: np.ndarray) -> np.ndarray:
        """Lift (num_samples, num_components) subspace vectors back to full space."""
        return subspace_vectors @ self.pca_components + self.pca_mean[None, :]

    def make_density(self) -> GaussianKDE:
        return GaussianKDE.fit(
            self.centroids_subspace.astype(np.float32),
            bandwidth=self.kde_bandwidth,
        )

    def make_geometry(self) -> DensityGeometry:
        kde = self.make_density()
        return DensityGeometry(energy_fn=kde.energy, alpha=self.alpha, beta=self.beta)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            labels=np.array(self.labels, dtype=object),
            centroids_subspace=self.centroids_subspace,
            pca_components=self.pca_components,
            pca_mean=self.pca_mean,
            pca_explained_variance_ratio=self.pca_explained_variance_ratio,
            kde_bandwidth=np.array([self.kde_bandwidth], dtype=np.float64),
            alpha=np.array([self.alpha], dtype=np.float64),
            beta=np.array([self.beta], dtype=np.float64),
        )

    @classmethod
    def load(cls, path: Path) -> FittedManifold:
        with np.load(path, allow_pickle=True) as data:
            return cls(
                labels=tuple(str(x) for x in data["labels"]),
                centroids_subspace=data["centroids_subspace"],
                pca_components=data["pca_components"],
                pca_mean=data["pca_mean"],
                pca_explained_variance_ratio=data["pca_explained_variance_ratio"],
                kde_bandwidth=float(data["kde_bandwidth"][0]),
                alpha=float(data["alpha"][0]),
                beta=float(data["beta"][0]),
            )


def fit_manifold(
    emotion_vectors: EmotionVectors,
    *,
    num_components: int = 64,
    bandwidth: float | None = None,
    alpha: float = 1.0,
    beta: float = 0.01,
) -> tuple[FittedManifold, PCAResult]:
    """Fit a density-geometry manifold to the diff-in-means emotion vectors.

    Returns (FittedManifold, PCAResult) — the PCAResult is returned
    alongside so callers can inspect explained variance / projections
    without re-running PCA.

    Defaults:
    - ``num_components=64`` matches Goodfire's spline-fitting setup.
    - ``bandwidth=None`` selects Silverman's rule in the subspace.
    - ``alpha=1.0, beta=0.01`` gives ~100× dynamic range between
      on- and off-manifold; tune downward if geodesics overshoot
      centroids, upward if they hug them too tightly.
    """
    if num_components > min(emotion_vectors.vectors.shape):
        raise ValueError(
            f"num_components={num_components} exceeds rank "
            f"{min(emotion_vectors.vectors.shape)} of vectors with "
            f"shape {emotion_vectors.vectors.shape}"
        )

    pca = fit_pca(emotion_vectors.vectors.astype(np.float32), n_components=num_components)
    centroids_subspace = pca.projections.astype(np.float32)

    if bandwidth is None:
        # Clustered-data heuristic, not Silverman: emotion centroids form
        # 30 distinct clusters, not samples from a smooth unimodal density.
        # Multiplier 1.0 = bandwidth equals median nearest-neighbor distance;
        # neighbors overlap at e^{-0.5} ≈ 0.61 of peak, distant centroids
        # do not overlap — gives the geodesic optimizer real signal.
        bandwidth = clustered_bandwidth(centroids_subspace, multiplier=1.0)  # type: ignore[arg-type]

    manifold = FittedManifold(
        labels=emotion_vectors.labels,
        centroids_subspace=centroids_subspace,
        pca_components=pca.components.astype(np.float32),
        pca_mean=pca.mean.astype(np.float32),
        pca_explained_variance_ratio=pca.explained_variance_ratio.astype(np.float64),
        kde_bandwidth=float(bandwidth),
        alpha=float(alpha),
        beta=float(beta),
    )
    return manifold, pca
