"""Kernel density estimate over emotion-vector centroids.

For a small labeled point cloud (171 emotion centroids in 64-D PCA
space) a Gaussian KDE is the right energy model — smooth, differentiable,
no training, one hyperparameter (bandwidth).

    p(h) = (1 / (N * (sqrt(2 pi) sigma)^d)) * sum_i exp(-||h - c_i||^2 / (2 sigma^2))
    E(h) = -log p(h)

All math implemented in JAX so the geodesic solver can call
``jax.value_and_grad`` on path-length functionals composed from these.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class GaussianKDE:
    """Vectorized Gaussian KDE.

    ``centroids`` is (N, d), ``bandwidth`` is a positive scalar. The
    density is normalized so it integrates to 1.

    All public methods return JAX arrays so they compose with
    ``jax.value_and_grad`` in the geodesic solver. Use ``np.asarray()``
    on the output if you need a numpy result.
    """

    centroids: jax.Array  # (N, d)
    bandwidth: float

    @classmethod
    def fit(
        cls,
        centroids: np.ndarray | jax.Array,
        bandwidth: float | None = None,
    ) -> GaussianKDE:
        """Build a KDE; ``bandwidth=None`` selects Silverman's rule."""
        centroids_arr = jnp.asarray(centroids, dtype=jnp.float32)
        if bandwidth is None:
            bandwidth = silverman_bandwidth(centroids_arr)
        if bandwidth <= 0:
            raise ValueError(f"bandwidth must be positive, got {bandwidth}")
        return cls(centroids=centroids_arr, bandwidth=float(bandwidth))

    def log_density(self, points: jax.Array) -> jax.Array:
        """log p(h) for one or many query points (proper PDF, normalized).

        Accepts (d,) or (M, d) and returns scalar or (M,) accordingly.
        Uses log-sum-exp for numerical stability.
        """
        return self.log_kernel_sum(points) + self._log_norm()

    def log_kernel_sum(self, points: jax.Array) -> jax.Array:
        """log of the unnormalized kernel sum: log(sum_i exp(-||h-c_i||^2 / (2 sigma^2))).

        This is the "energy-like" quantity used in the density-geometry
        metric: G_E uses e^{-E(h)} as an UN-normalized density, since the
        metric ratio between on-manifold and off-manifold regions is
        what matters — the normalizing constant only shifts E uniformly.
        In d=8 with bandwidth ~14, the proper-PDF normalization makes
        ``density(centroid)`` ~1e-14 in absolute terms, which collapses
        ``alpha * exp(-E)`` against any sensible ``beta`` floor and
        renders the metric Euclidean. Working with the unnormalized
        kernel sum keeps values O(1) and the metric ratio meaningful.
        """
        points = jnp.asarray(points)
        single = points.ndim == 1
        if single:
            points = points[None, :]

        sigma2 = self.bandwidth**2
        diffs = points[:, None, :] - self.centroids[None, :, :]
        sq = jnp.sum(diffs * diffs, axis=-1)
        kernel_log = -sq / (2.0 * sigma2)
        out = jax.scipy.special.logsumexp(kernel_log, axis=1)
        return out[0] if single else out

    def _log_norm(self) -> jax.Array:
        n, d = self.centroids.shape
        return -0.5 * d * jnp.log(2.0 * jnp.pi * self.bandwidth**2) - jnp.log(n)

    def density(self, points: jax.Array) -> jax.Array:
        return jnp.exp(self.log_density(points))

    def energy(self, points: jax.Array) -> jax.Array:
        """E(h) for the density-geometry metric: -log of the unnormalized kernel sum.

        Uses the unnormalized form (not the proper PDF). See ``log_kernel_sum``
        for the rationale. If you need a calibrated probability density, use
        ``log_density`` instead.
        """
        return -self.log_kernel_sum(points)


def silverman_bandwidth(centroids: jax.Array) -> float:
    """Silverman's rule of thumb adapted for d-dimensional KDE.

    h = (4 / (d + 2))^(1/(d+4)) * n^(-1/(d+4)) * std

    Where ``std`` is the per-coordinate standard deviation averaged
    geometrically across dimensions. Reasonable default for smooth
    unimodal-ish data; the emotion manifold is multimodal so we may
    want to tune later.
    """
    centroids_np = np.asarray(centroids)
    n, d = centroids_np.shape
    per_dim_std = centroids_np.std(axis=0)
    # Geometric mean of per-dim stds keeps the bandwidth invariant
    # under coordinate scaling.
    mean_std = float(np.exp(np.mean(np.log(np.clip(per_dim_std, 1e-12, None)))))
    factor = (4.0 / (d + 2.0)) ** (1.0 / (d + 4.0))
    return factor * (n ** (-1.0 / (d + 4.0))) * mean_std


def clustered_bandwidth(centroids: jax.Array, multiplier: float = 1.0) -> float:
    """Bandwidth scaled to the nearest-neighbor centroid spacing.

    Silverman's rule and "fraction of median pairwise distance" both
    over-smooth for tightly clustered data: they pick a bandwidth large
    enough that the KDE looks like one big bump, so the density-geometry
    metric is nearly constant and geodesics don't bend. We want the
    bandwidth comparable to the typical NEAREST-NEIGHBOR centroid
    distance, so that:
    - kernels of adjacent centroids overlap (continuous metric, smooth
      gradients for the geodesic optimizer)
    - kernels of distant centroids do NOT overlap (density actually
      varies as the path moves between clusters)

    With ``multiplier=1.0``, density at a midpoint between two nearest
    neighbors is exp(-0.5) ≈ 0.61 of peak per kernel — good overlap.
    Density at a midpoint between distant centroids is much smaller,
    so the metric scalar varies meaningfully along inter-cluster paths.
    """
    centroids_np = np.asarray(centroids)
    n = centroids_np.shape[0]
    if n < 2:
        raise ValueError("clustered_bandwidth requires at least 2 centroids")
    diff = centroids_np[:, None, :] - centroids_np[None, :, :]
    pairwise = np.sqrt((diff * diff).sum(axis=-1))
    np.fill_diagonal(pairwise, np.inf)
    nearest_neighbor = pairwise.min(axis=1)
    return float(np.median(nearest_neighbor) * multiplier)
