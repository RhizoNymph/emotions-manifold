"""Thin-plate spline manifold: a parametric surface phi: R^2 -> R^d.

Fits a smooth 2-D surface through the (behavior-coordinate, activation-centroid)
pairs, where the behavior coordinate is the emotion's (valence, arousal) point in
M_y and the activation centroid is its projection into the d-dim PCA subspace M_h.

This is the "real Goodfire" parametric construction that the rest of the project
approximates with a non-parametric Nadaraya-Watson pullback. Goodfire parameterize
M_h by the semantic axis (day index, age, grid position) and fit a spline; the
semantic axis here is valence/arousal, so the spline is phi: (V, A) -> activation.

Because the surface is intrinsically 2-D and genuinely curved in activation space,
geodesics *on the surface* (see ``spline_geodesic``) bend away from ambient straight
lines exactly to the extent the activation manifold is curved when parameterized by
V/A. A purely affine phi reproduces the linear baseline; all deviation comes from the
nonlinear (thin-plate) warp.

Two metrics are supported for surface geodesics:
- induced (first fundamental form): geodesic = shortest path on the surface, i.e.
  the parameter path whose embedded image has minimal Euclidean length.
- induced x density: the same, reweighted by the G_E density factor
  ``(alpha * e^{-E(phi(u))} + beta)`` so the path also prefers dense regions —
  the parametric analog of the ambient density-aware metric.

The standard 2-D thin-plate spline:

    phi(u) = sum_i W_i * U(||u - c_i||) + A^T [1, u_x, u_y]
    U(r) = r^2 log r            (the 2-D TPS Green's function)

with weights (W, A) solved from the linear system

    [ K + lam*I   P ] [W]   [H]
    [ P^T         0 ] [A] = [0]

where K_ij = U(||c_i - c_j||), P_i = [1, c_i], H_i = centroid_i, and ``lam`` is a
smoothing parameter (lam=0 interpolates the centroids exactly).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .density import GaussianKDE
from .metric import DensityGeometry

# Numerical floor inside the TPS kernel's log so the r=0 term (and its gradient)
# stays finite. The kernel value at r=0 is 0 regardless; this only guards autodiff.
_KERNEL_EPS = 1e-8


def _tps_kernel_sq(r2: jax.Array) -> jax.Array:
    """2-D thin-plate kernel as a function of squared radius: U = 0.5 * r^2 * log(r^2).

    r^2 log r == 0.5 r^2 log(r^2). Adding ``_KERNEL_EPS`` inside the log keeps the
    value 0 at r=0 with a finite gradient, so JAX can differentiate paths that pass
    near control points.
    """
    return 0.5 * r2 * jnp.log(r2 + _KERNEL_EPS)


@dataclass(frozen=True, slots=True)
class SplineManifold:
    """A thin-plate-spline surface phi: (valence, arousal) -> PCA subspace.

    Self-contained: carries the PCA lift (``pca_components``/``pca_mean``) and the
    ambient KDE hyperparameters so it can both inject steering vectors into the full
    residual stream and build the density-aware surface metric, with no reference to
    the source ``FittedManifold``.

    ``control_coords`` are raw (V, A) points; the spline is solved in standardized
    coordinates (``coord_mean``/``coord_scale``) for conditioning, and ``embed``
    accepts raw (V, A) and standardizes internally.
    """

    labels: tuple[str, ...]
    control_coords: np.ndarray  # (N, 2) surface coords, one per label (see parameterization)
    weights: np.ndarray  # (N, d) TPS nonlinear weights W (standardized-coord basis)
    affine: np.ndarray  # (3, d) TPS affine part A, rows = [const, u_x, u_y]
    coord_mean: np.ndarray  # (2,)
    coord_scale: np.ndarray  # (2,)
    smoothing: float
    centroids_subspace: np.ndarray  # (N, d) PCA centroids (endpoint snapping, baselines)
    pca_components: np.ndarray  # (d, hidden_size)
    pca_mean: np.ndarray  # (hidden_size,)
    kde_bandwidth: float
    alpha: float
    beta: float
    # How ``control_coords`` was constructed. "valence_arousal": the raw M_y (V, A)
    # readout (many-to-one, so the fit is lossy). "diffusion_map_2": a bijective
    # diffusion-2 coordinate of the activation centroids (one distinct point per
    # emotion, so smoothing=0 interpolates exactly). Purely descriptive — the
    # surface machinery treats ``control_coords`` as an opaque 2-D parameter space,
    # so downstream code (chord endpoint lookup, geodesics) is parameterization-agnostic.
    parameterization: str = "valence_arousal"

    @property
    def num_components(self) -> int:
        return self.centroids_subspace.shape[1]

    @property
    def hidden_size(self) -> int:
        return self.pca_components.shape[1]

    # -- embedding -------------------------------------------------------------

    def _standardize(self, coords: jax.Array) -> jax.Array:
        mean = jnp.asarray(self.coord_mean, dtype=coords.dtype)
        scale = jnp.asarray(self.coord_scale, dtype=coords.dtype)
        return (coords - mean) / scale

    def embed(self, coords: jax.Array) -> jax.Array:
        """Map (V, A) coordinates to PCA-subspace points. JAX-traceable.

        Accepts (2,) or (M, 2) raw coordinates; returns (d,) or (M, d).
        """
        coords = jnp.asarray(coords)
        single = coords.ndim == 1
        if single:
            coords = coords[None, :]

        qs = self._standardize(coords)  # (M, 2)
        ctrl = self._standardize(jnp.asarray(self.control_coords, dtype=qs.dtype))  # (N, 2)
        diffs = qs[:, None, :] - ctrl[None, :, :]  # (M, N, 2)
        r2 = jnp.sum(diffs * diffs, axis=-1)  # (M, N)
        u_basis = _tps_kernel_sq(r2)  # (M, N)
        ones = jnp.ones((qs.shape[0], 1), dtype=qs.dtype)
        p_basis = jnp.concatenate([ones, qs], axis=1)  # (M, 3)

        out = u_basis @ jnp.asarray(self.weights, dtype=qs.dtype) + p_basis @ jnp.asarray(
            self.affine, dtype=qs.dtype
        )  # (M, d)
        return out[0] if single else out

    def embed_np(self, coords: np.ndarray) -> np.ndarray:
        """Numpy convenience wrapper around ``embed`` for non-autodiff callers."""
        return np.asarray(self.embed(jnp.asarray(coords, dtype=jnp.float32)))

    def jacobian(self, coord: np.ndarray) -> np.ndarray:
        """Embedding Jacobian J = d phi / d(V, A) at a single coord. Shape (d, 2)."""
        single = jax.jacfwd(lambda u: self.embed(u))(jnp.asarray(coord, dtype=jnp.float32))
        return np.asarray(single)

    def induced_metric(self, coord: np.ndarray) -> np.ndarray:
        """First fundamental form g = J^T J at a single coord. Shape (2, 2)."""
        j = self.jacobian(coord)
        return j.T @ j

    # -- lift + density --------------------------------------------------------

    def unproject(self, subspace_vectors: np.ndarray) -> np.ndarray:
        """Lift (M, d) subspace vectors back to the full residual stream."""
        return subspace_vectors @ self.pca_components + self.pca_mean[None, :]

    def make_density(self) -> GaussianKDE:
        return GaussianKDE.fit(
            self.centroids_subspace.astype(np.float32), bandwidth=self.kde_bandwidth
        )

    def make_geometry(self) -> DensityGeometry:
        kde = self.make_density()
        return DensityGeometry(energy_fn=kde.energy, alpha=self.alpha, beta=self.beta)

    # -- persistence -----------------------------------------------------------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            labels=np.array(self.labels, dtype=object),
            control_coords=self.control_coords,
            weights=self.weights,
            affine=self.affine,
            coord_mean=self.coord_mean,
            coord_scale=self.coord_scale,
            smoothing=np.array([self.smoothing], dtype=np.float64),
            centroids_subspace=self.centroids_subspace,
            pca_components=self.pca_components,
            pca_mean=self.pca_mean,
            kde_bandwidth=np.array([self.kde_bandwidth], dtype=np.float64),
            alpha=np.array([self.alpha], dtype=np.float64),
            beta=np.array([self.beta], dtype=np.float64),
            parameterization=np.array(self.parameterization, dtype=object),
        )

    @classmethod
    def load(cls, path: Path) -> SplineManifold:
        with np.load(path, allow_pickle=True) as data:
            # ``parameterization`` was added after the first V/A artifacts were
            # written; default to valence_arousal so those still load.
            parameterization = (
                str(data["parameterization"])
                if "parameterization" in data.files
                else "valence_arousal"
            )
            return cls(
                labels=tuple(str(x) for x in data["labels"]),
                control_coords=data["control_coords"],
                weights=data["weights"],
                affine=data["affine"],
                coord_mean=data["coord_mean"],
                coord_scale=data["coord_scale"],
                smoothing=float(data["smoothing"][0]),
                centroids_subspace=data["centroids_subspace"],
                pca_components=data["pca_components"],
                pca_mean=data["pca_mean"],
                kde_bandwidth=float(data["kde_bandwidth"][0]),
                alpha=float(data["alpha"][0]),
                beta=float(data["beta"][0]),
                parameterization=parameterization,
            )

    # -- fitting ---------------------------------------------------------------

    @classmethod
    def fit(
        cls,
        *,
        labels: tuple[str, ...],
        control_coords: np.ndarray,
        centroids_subspace: np.ndarray,
        pca_components: np.ndarray,
        pca_mean: np.ndarray,
        kde_bandwidth: float,
        alpha: float = 1.0,
        beta: float = 0.01,
        smoothing: float = 0.0,
        parameterization: str = "valence_arousal",
    ) -> SplineManifold:
        """Solve the thin-plate spline through (control_coords -> centroids_subspace).

        ``smoothing`` (lam >= 0) adds lam*I to the kernel block: lam=0 interpolates
        the centroids exactly, lam>0 trades exactness for a smoother surface (fewer
        inter-centroid wiggles, better-behaved geodesics).
        """
        coords = np.asarray(control_coords, dtype=np.float64)
        targets = np.asarray(centroids_subspace, dtype=np.float64)
        n, two = coords.shape
        if two != 2:
            raise ValueError(f"control_coords must be (N, 2), got {coords.shape}")
        if targets.shape[0] != n:
            raise ValueError(
                f"centroids_subspace has {targets.shape[0]} rows, expected {n}"
            )
        if smoothing < 0:
            raise ValueError(f"smoothing must be >= 0, got {smoothing}")

        coord_mean = coords.mean(axis=0)
        coord_scale = coords.std(axis=0)
        coord_scale[coord_scale < 1e-8] = 1.0
        std_coords = (coords - coord_mean) / coord_scale

        diffs = std_coords[:, None, :] - std_coords[None, :, :]
        r2 = np.sum(diffs * diffs, axis=-1)  # (N, N)
        k_block = 0.5 * r2 * np.log(r2 + _KERNEL_EPS)  # (N, N), diag = 0
        if smoothing > 0:
            k_block = k_block + smoothing * np.eye(n)

        p_block = np.concatenate([np.ones((n, 1)), std_coords], axis=1)  # (N, 3)

        # [[K, P], [P^T, 0]] @ [[W],[A]] = [[H],[0]]
        top = np.concatenate([k_block, p_block], axis=1)  # (N, N+3)
        bottom = np.concatenate([p_block.T, np.zeros((3, 3))], axis=1)  # (3, N+3)
        system = np.concatenate([top, bottom], axis=0)  # (N+3, N+3)
        rhs = np.concatenate([targets, np.zeros((3, targets.shape[1]))], axis=0)  # (N+3, d)

        solution = np.linalg.solve(system, rhs)  # (N+3, d)
        weights = solution[:n]  # (N, d)
        affine = solution[n:]  # (3, d)

        return cls(
            labels=labels,
            control_coords=coords.astype(np.float32),
            weights=weights.astype(np.float32),
            affine=affine.astype(np.float32),
            coord_mean=coord_mean.astype(np.float32),
            coord_scale=coord_scale.astype(np.float32),
            smoothing=float(smoothing),
            centroids_subspace=np.asarray(centroids_subspace, dtype=np.float32),
            pca_components=np.asarray(pca_components, dtype=np.float32),
            pca_mean=np.asarray(pca_mean, dtype=np.float32),
            kde_bandwidth=float(kde_bandwidth),
            alpha=float(alpha),
            beta=float(beta),
            parameterization=str(parameterization),
        )
