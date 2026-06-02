"""Density-geometry Riemannian metric G_E (Goodfire Eq. 6).

Given an energy function E(h) (typically -log p(h) from a KDE or EBM),
the density geometry is:

    G_E(h) = (alpha * exp(-E(h)) + beta)^-1 * I_n

so on-manifold regions (low E, high density) get a small metric (cheap
to traverse) and off-manifold regions (high E) get a large metric
(expensive). Geodesics under G_E thus prefer staying on the manifold.

Path length of a curve pi: [0, 1] -> R^n is

    L_G(pi) = integral_0^1 sqrt(pi_dot(t)^T G(pi(t)) pi_dot(t)) dt
            = integral_0^1 ||pi_dot(t)|| / sqrt(alpha * exp(-E(pi(t))) + beta) dt

(the second equality holds because G is a scalar times I).

We discretize as a sum over K+1 waypoints and use the midpoint of each
segment to evaluate the metric — second-order accurate, important for
the solver to follow curvature instead of greedily hugging high-density
spikes.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class DensityGeometry:
    """G_E(h) = (alpha * exp(-E(h)) + beta)^-1 * I.

    ``alpha`` controls how strongly density attracts the path; ``beta``
    keeps the metric positive in regions of zero density and bounds the
    metric's dynamic range so the solver doesn't get stuck on numerical
    cliffs. Reasonable defaults: alpha=1.0, beta=0.01 (so the metric
    varies by a factor of ~100 between on- and off-manifold).
    """

    energy_fn: object  # callable: jax.Array -> jax.Array  (must be JAX-traceable)
    alpha: float = 1.0
    beta: float = 0.01

    def metric_scalar(self, h: jax.Array) -> jax.Array:
        """Scalar f(h) such that G_E(h) = f(h) * I_n; positive everywhere."""
        return 1.0 / (self.alpha * jnp.exp(-self.energy_fn(h)) + self.beta)  # type: ignore[operator]

    def inverse_metric_scalar(self, h: jax.Array) -> jax.Array:
        """1 / f(h) = alpha * exp(-E(h)) + beta. Used in path-length integrand."""
        return self.alpha * jnp.exp(-self.energy_fn(h)) + self.beta  # type: ignore[operator]

    def path_length(self, waypoints: jax.Array) -> jax.Array:
        """Discrete path length under G_E.

        waypoints is (K+1, d). Uses the segment-midpoint to evaluate the
        metric so curvature is captured without per-vertex bias. The
        integrand at midpoint m_k is ||delta_k|| / sqrt(inverse_metric(m_k)).
        """
        deltas = waypoints[1:] - waypoints[:-1]  # (K, d)
        midpoints = 0.5 * (waypoints[1:] + waypoints[:-1])  # (K, d)
        # vmap energy over the midpoints so the user-provided energy_fn
        # can be a single-point function.
        energy_at_mid = jax.vmap(self.energy_fn)(midpoints)  # type: ignore[arg-type]
        inv_metric = self.alpha * jnp.exp(-energy_at_mid) + self.beta
        segment_lengths = jnp.linalg.norm(deltas, axis=1) / jnp.sqrt(inv_metric)
        return jnp.sum(segment_lengths)
