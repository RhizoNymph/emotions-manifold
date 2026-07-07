"""Diffusion-map coordinates of the activation centroids.

A *bijective* intrinsic parameterization of the emotion cloud: unlike valence/
arousal (a lossy, many-to-one readout where ~14% of emotions collide), the
diffusion-2 embedding assigns each of the N centroids a distinct point in R^2.
A thin-plate spline fit through (diffusion coord -> centroid) at zero smoothing
therefore interpolates the centroids essentially exactly, so surface geodesics
route between the two real-emotion endpoints through near-real intermediates.

This is the faithful analog of Goodfire's parametric spline (whose day-index /
grid parameter is bijective with the concepts) for an unordered emotion cloud.
The construction here is the single source of truth for the diffusion coordinate:
both the geometry check (``scripts/analysis/bijective_spline_check.py``) and the
behavioral fitter (``scripts/fit_spline_manifold.py``) import it so the behavioral
run matches the geometric artifact exactly.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform


def diffusion_embed(centroids: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Anisotropic (alpha=1) diffusion-map embedding of ``centroids``.

    Returns the leading ``n_components`` non-trivial diffusion coordinates, each
    scaled by its eigenvalue, as an ``(N, n_components)`` float64 array. The
    kernel bandwidth is the median squared pairwise distance (a scale-free
    heuristic), and the density normalization ``Ka / (q_i q_j)`` removes the
    sampling-density bias so the coordinate reflects manifold geometry.
    """
    sq = squareform(pdist(centroids)) ** 2
    eps = float(np.median(sq[sq > 0]))
    Km = np.exp(-sq / eps)
    q = Km.sum(axis=1)
    Ka = Km / np.outer(q, q)
    d = Ka.sum(axis=1)
    ds = np.sqrt(d)
    Ps = Ka / np.outer(ds, ds)
    Ps = 0.5 * (Ps + Ps.T)
    vals, vecs = np.linalg.eigh(Ps)
    vals, vecs = vals[::-1], vecs[:, ::-1]
    psi = (vecs / ds[:, None])[:, 1 : n_components + 1]
    return (psi * vals[1 : n_components + 1][None, :]).astype(np.float64)
