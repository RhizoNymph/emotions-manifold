"""Geodesics on a thin-plate-spline surface (``SplineManifold``).

We optimize in the 2-D parameter space (valence, arousal): the interior waypoints
are 2-D coordinates, the endpoints are fixed at the two emotions' V/A points, and
the objective is the length of the *embedded* path phi(gamma) measured on the
surface. Two metrics:

- ``"induced"``: Euclidean length of the embedded polyline, sum ||phi(u_{k+1}) -
  phi(u_k)||. Minimizing this gives the shortest path on the surface (the geodesic
  of the first fundamental form). A purely affine phi yields the ambient straight
  line; curvature in phi is the only thing that makes it deviate.
- ``"density"``: the same embedded polyline reweighted by the G_E density factor,
  reusing ``DensityGeometry.path_length`` on the embedded waypoints — the parametric
  analog of the ambient density-aware geodesic.

The optimization mirrors ``manifold.geodesic.fit_geodesic``: L-BFGS-B over the
flattened interior coordinates with JIT'd JAX value-and-grad. The returned waypoints
are in the PCA subspace (phi of the optimized coords), ready to be unprojected into
steering vectors, with endpoints optionally snapped to exact centroids so all
trajectories (linear, pullback, ambient geodesic, spline geodesic) share endpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from .spline import SplineManifold

SplineMetric = Literal["induced", "density"]


@dataclass(frozen=True, slots=True)
class SplineGeodesicResult:
    waypoints: np.ndarray  # (K, d) embedded subspace points
    coords: np.ndarray  # (K, 2) parameter-space (V, A) path
    metric: SplineMetric
    final_length: float
    initial_length: float
    converged: bool
    num_iterations: int
    message: str

    @property
    def num_waypoints(self) -> int:
        return self.waypoints.shape[0]


def _coord_linspace(start: np.ndarray, end: np.ndarray, num_waypoints: int) -> np.ndarray:
    if num_waypoints < 2:
        raise ValueError(f"num_waypoints must be >= 2, got {num_waypoints}")
    ts = np.linspace(0.0, 1.0, num_waypoints)
    return (1.0 - ts)[:, None] * start[None, :] + ts[:, None] * end[None, :]


def fit_spline_geodesic(
    spline: SplineManifold,
    start_coord: np.ndarray,
    end_coord: np.ndarray,
    *,
    metric: SplineMetric = "induced",
    num_waypoints: int = 30,
    max_iter: int = 300,
    gtol: float = 1e-5,
    snap_start: np.ndarray | None = None,
    snap_end: np.ndarray | None = None,
) -> SplineGeodesicResult:
    """Shortest path on the spline surface between two (V, A) endpoints.

    ``start_coord``/``end_coord`` are raw (valence, arousal) points (the two emotion
    centroids in M_y). ``snap_start``/``snap_end``, if given, replace the embedded
    endpoints with these subspace points (typically the exact PCA centroids) so the
    path shares endpoints with the linear/pullback/ambient-geodesic trajectories.
    """
    start_coord = np.asarray(start_coord, dtype=np.float64)
    end_coord = np.asarray(end_coord, dtype=np.float64)
    if start_coord.shape != (2,) or end_coord.shape != (2,):
        raise ValueError("start_coord and end_coord must be (2,) (valence, arousal)")

    start_j = jnp.asarray(start_coord, dtype=jnp.float32)
    end_j = jnp.asarray(end_coord, dtype=jnp.float32)

    init_coords = _coord_linspace(start_coord, end_coord, num_waypoints)
    interior_init = init_coords[1:-1]  # (K-2, 2)

    # Bound interior waypoints to the V/A data box (expanded by a margin) so the
    # optimizer can't push coords into the spline's extrapolation regime, where
    # phi blows up and the density energy at phi(u) overflows to nan. This is the
    # physically correct constraint: a geodesic between two in-hull points should
    # not leave the data envelope.
    ctrl = np.asarray(spline.control_coords, dtype=np.float64)
    lo = ctrl.min(axis=0)
    hi = ctrl.max(axis=0)
    margin = 0.25 * (hi - lo)
    box_lo, box_hi = lo - margin, hi + margin
    bounds = [(float(box_lo[k % 2]), float(box_hi[k % 2])) for k in range((num_waypoints - 2) * 2)]

    geometry = spline.make_geometry() if metric == "density" else None

    def assemble(interior_flat: jax.Array) -> jax.Array:
        interior = interior_flat.reshape((num_waypoints - 2, 2))
        return jnp.concatenate([start_j[None, :], interior, end_j[None, :]], axis=0)

    def embedded_length(coords_path: jax.Array) -> jax.Array:
        points = spline.embed(coords_path)  # (K, d)
        if metric == "induced":
            deltas = points[1:] - points[:-1]
            return jnp.sum(jnp.linalg.norm(deltas, axis=1))
        return geometry.path_length(points)  # type: ignore[union-attr]

    @jax.jit
    def length_and_grad(interior_flat: jax.Array) -> tuple[jax.Array, jax.Array]:
        return jax.value_and_grad(lambda f: embedded_length(assemble(f)))(interior_flat)

    initial_length = float(embedded_length(jnp.asarray(init_coords, dtype=jnp.float32)))

    def scipy_objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
        val, grad = length_and_grad(jnp.asarray(flat, dtype=jnp.float32))
        return float(val), np.asarray(grad, dtype=np.float64)

    result = minimize(
        scipy_objective,
        interior_init.flatten().astype(np.float64),
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": max_iter, "gtol": gtol},
    )

    interior_opt = result.x.reshape((num_waypoints - 2, 2))
    coords_path = np.concatenate(
        [start_coord[None, :], interior_opt, end_coord[None, :]], axis=0
    )  # (K, 2)
    waypoints = np.array(spline.embed_np(coords_path.astype(np.float32)))  # (K, d), writable

    # Guard: if the solve produced any non-finite embedded point, fall back to the
    # straight-coordinate path (always finite, in-hull) rather than emit nan steering.
    if not np.isfinite(waypoints).all():
        coords_path = init_coords
        waypoints = np.array(spline.embed_np(init_coords.astype(np.float32)))

    if snap_start is not None:
        waypoints[0] = np.asarray(snap_start, dtype=waypoints.dtype)
    if snap_end is not None:
        waypoints[-1] = np.asarray(snap_end, dtype=waypoints.dtype)

    return SplineGeodesicResult(
        waypoints=waypoints,
        coords=coords_path.astype(np.float32),
        metric=metric,
        final_length=float(result.fun),
        initial_length=initial_length,
        converged=bool(result.success),
        num_iterations=int(result.nit),
        message=str(result.message),
    )
