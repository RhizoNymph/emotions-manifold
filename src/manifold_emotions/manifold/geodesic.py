"""Geodesic path optimization under a density-geometry metric.

We discretize a path from ``start`` to ``end`` into K+1 waypoints, fix
the endpoints, and minimize path length over the K-1 interior points
using L-BFGS-B with JAX-computed gradients.

Initialization: linear interpolation between the endpoints. This is the
geodesic under the flat (Euclidean) metric — the baseline that linear
steering uses. The optimizer then bends the path to minimize length
under G_E, pulling it toward high-density regions of activation space.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from .metric import DensityGeometry


@dataclass(frozen=True, slots=True)
class GeodesicResult:
    waypoints: np.ndarray  # (K+1, d)
    final_length: float
    initial_length: float
    converged: bool
    num_iterations: int
    message: str

    @property
    def num_waypoints(self) -> int:
        return self.waypoints.shape[0]


def linear_interpolation(
    start: np.ndarray, end: np.ndarray, num_waypoints: int
) -> np.ndarray:
    """K+1 waypoints linearly interpolated from start to end."""
    if num_waypoints < 2:
        raise ValueError(f"num_waypoints must be >= 2, got {num_waypoints}")
    ts = np.linspace(0.0, 1.0, num_waypoints).astype(start.dtype)
    return (1.0 - ts)[:, None] * start[None, :] + ts[:, None] * end[None, :]


def fit_geodesic(
    geometry: DensityGeometry,
    start: np.ndarray,
    end: np.ndarray,
    num_waypoints: int = 50,
    max_iter: int = 200,
    gtol: float = 1e-5,
) -> GeodesicResult:
    """Minimize G_E path length from start to end over num_waypoints points.

    Endpoints are fixed; interior waypoints are the optimization variables.
    Returns the optimized path with both endpoints reattached.
    """
    if start.shape != end.shape:
        raise ValueError(f"start and end shape mismatch: {start.shape} vs {end.shape}")

    start_j = jnp.asarray(start, dtype=jnp.float32)
    end_j = jnp.asarray(end, dtype=jnp.float32)

    initial = linear_interpolation(np.asarray(start), np.asarray(end), num_waypoints)
    initial_j = jnp.asarray(initial, dtype=jnp.float32)
    interior_init = np.asarray(initial_j[1:-1])  # (K-1, d)
    d = start.shape[0]

    def assemble(interior_flat: jax.Array) -> jax.Array:
        interior = interior_flat.reshape((num_waypoints - 2, d))
        return jnp.concatenate(
            [start_j[None, :], interior, end_j[None, :]], axis=0
        )

    @jax.jit
    def length_and_grad(interior_flat: jax.Array) -> tuple[jax.Array, jax.Array]:
        def loss(flat: jax.Array) -> jax.Array:
            return geometry.path_length(assemble(flat))

        return jax.value_and_grad(loss)(interior_flat)

    initial_length = float(geometry.path_length(initial_j))

    def scipy_objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
        flat_j = jnp.asarray(flat, dtype=jnp.float32)
        val, grad = length_and_grad(flat_j)
        return float(val), np.asarray(grad, dtype=np.float64)

    result = minimize(
        scipy_objective,
        interior_init.flatten().astype(np.float64),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "gtol": gtol},
    )

    optimized_interior = result.x.reshape((num_waypoints - 2, d))
    waypoints = np.concatenate(
        [
            np.asarray(start)[None, :],
            optimized_interior.astype(start.dtype),
            np.asarray(end)[None, :],
        ],
        axis=0,
    )

    return GeodesicResult(
        waypoints=waypoints,
        final_length=float(result.fun),
        initial_length=initial_length,
        converged=bool(result.success),
        num_iterations=int(result.nit),
        message=str(result.message),
    )
