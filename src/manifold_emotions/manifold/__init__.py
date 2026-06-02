"""Manifold fitting and geodesic computation in activation space.

Implements the density-geometry approach from Goodfire's manifold steering
paper (Eq. 6): G_E(h) = (alpha * exp(-E(h)) + beta)^-1 * I, where E is an
energy function fit to the activation manifold (low on-manifold, high
off-manifold). Geodesics under this metric stay on the manifold without
needing an explicit parameterization s: R^k -> A — the right answer for
the emotion manifold where intrinsic coordinates aren't given.
"""
