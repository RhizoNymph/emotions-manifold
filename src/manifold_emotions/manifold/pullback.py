"""Pullback experiment (Goodfire §3.3): M_y geodesic → M_h path.

The forward experiment we already ran asks: do geodesics in M_h produce
trajectories in M_y that hug the behavior manifold? This module
implements the converse: given a geodesic in M_y (a straight line in
the valence-arousal plane between two emotion centroids), construct
the activation path that would *produce* that behavior trajectory, and
ask whether this path is closer to the M_h geodesic or to the M_h
straight line.

We don't have a direct inverse of the model's forward map, so we
approximate the pullback by kernel-weighted barycentric interpolation
of the emotion centroids in (V, A) space, lifted into M_h subspace
coordinates. Concretely, for a target behavior point y* in M_y:

    w_i = exp(-||y_i - y*||^2 / (2 sigma^2))
    h*  = sum_i (w_i / Z) * h_i_subspace,    Z = sum_i w_i

with bandwidth sigma equal to the median nearest-neighbor distance
between behavior centroids. This is the M_y-side analog of the KDE
bandwidth used to build M_h's density geometry.

A successful pullback should:
1. Lie closer to the M_h geodesic than to the M_h linear interpolation
   (when measured in subspace Euclidean distance per waypoint).
2. Have a path length under G_E close to the M_h geodesic's.
3. When unprojected and used as a steering trajectory, produce a
   behavior path that follows the M_y straight line.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..behavior.manifold import BehaviorManifold
from .fit import FittedManifold


def _median_nn_distance(points: np.ndarray) -> float:
    """Median Euclidean nearest-neighbor distance over a point set.

    points is (N, d). Mirrors ``density.clustered_bandwidth`` but for
    M_y centroids in 2D so the pullback kernel has the same
    "median-of-nearest-neighbor" character.
    """
    n = points.shape[0]
    diffs = points[:, None, :] - points[None, :, :]
    dists = np.sqrt((diffs * diffs).sum(axis=-1))
    np.fill_diagonal(dists, np.inf)
    nn_dists = dists.min(axis=1)
    return float(np.median(nn_dists))


def _knn_distance_from_point(
    target: np.ndarray, anchors: np.ndarray, k: int
) -> float:
    """Distance from target to its K-th nearest neighbor among anchors.

    Used for adaptive per-waypoint σ: at each target point y* on the
    M_y chord, take the distance to the K-th nearest M_y centroid as
    the local kernel bandwidth. Sparse regions get wide kernels, dense
    regions get narrow ones.

    target is (d,); anchors is (N, d). K is clipped to [1, N].
    """
    n = anchors.shape[0]
    k = max(1, min(k, n))
    dists = np.sqrt(np.sum((anchors - target[None, :]) ** 2, axis=1))
    return float(np.partition(dists, k - 1)[k - 1])


SigmaSpec = float | str | None


@dataclass(frozen=True, slots=True)
class PullbackResult:
    """Pullback path in M_h subspace, plus comparison baselines.

    All paths share the same (K, subspace_dim) shape so we can do
    per-waypoint comparisons directly. The ``_full`` arrays are the
    same paths unprojected into the full residual-stream space, ready
    to use as steering trajectories.
    """

    start_label: str
    end_label: str
    sigma: float  # M_y kernel bandwidth used to build the pullback (mean if adaptive)
    sigma_spec: str  # "<float>" for constant, "knn:<K>" for adaptive, "median_nn" for default
    sigma_per_waypoint: np.ndarray  # (K,) — actual σ used at each waypoint

    my_path: np.ndarray             # (K, 2)            — straight line in M_y
    pullback_sub: np.ndarray        # (K, subspace_dim) — kernel barycenter in M_h subspace
    geodesic_sub: np.ndarray        # (K, subspace_dim) — M_h geodesic
    linear_sub: np.ndarray          # (K, subspace_dim) — M_h linear interpolation

    pullback_full: np.ndarray       # (K, hidden_size)
    geodesic_full: np.ndarray       # (K, hidden_size)
    linear_full: np.ndarray         # (K, hidden_size)

    pullback_length: float          # under G_E
    geodesic_length: float
    linear_length: float

    # Per-waypoint subspace distance: ||pullback_k - geodesic_k||, etc.
    dist_pullback_to_geodesic: np.ndarray  # (K,)
    dist_pullback_to_linear: np.ndarray    # (K,)
    dist_geodesic_to_linear: np.ndarray    # (K,)

    @property
    def num_waypoints(self) -> int:
        return self.my_path.shape[0]

    @property
    def mean_dist_to_geodesic(self) -> float:
        return float(self.dist_pullback_to_geodesic.mean())

    @property
    def mean_dist_to_linear(self) -> float:
        return float(self.dist_pullback_to_linear.mean())

    @property
    def closer_to(self) -> str:
        return "geodesic" if self.mean_dist_to_geodesic < self.mean_dist_to_linear else "linear"


def _kernel_weights(
    target: np.ndarray, anchors: np.ndarray, sigma: float
) -> np.ndarray:
    """Softmax-normalized Gaussian weights from a target point to anchor points.

    target is (d,); anchors is (N, d). Returns (N,) summing to 1.
    Uses the log-sum-exp trick so we never under/overflow for small sigma.
    """
    sq = np.sum((anchors - target[None, :]) ** 2, axis=1)
    log_w = -sq / (2.0 * sigma * sigma)
    log_w = log_w - log_w.max()
    w = np.exp(log_w)
    return w / w.sum()


def _resolve_sigma(
    sigma: SigmaSpec, my_path: np.ndarray, behavior_centroids: np.ndarray
) -> tuple[np.ndarray, str]:
    """Turn a sigma spec into a (K,) per-waypoint σ array + a spec tag.

    Supported specs:
    - None: σ = median NN distance among behavior centroids (constant)
    - float: σ used constant across all waypoints
    - str "knn:<K>": σ at waypoint k = distance from y*_k to its K-th
      nearest behavior centroid (adaptive — sparse y* → wide σ)
    - str "knn:<K>*<scale>": scaled adaptive — σ = scale × K-NN-distance.
      Lets you preserve the adaptive shape while shifting magnitude
      down toward the dense-pair-friendly regime.
    """
    num_waypoints = my_path.shape[0]
    if sigma is None:
        s = _median_nn_distance(behavior_centroids)
        return np.full(num_waypoints, s, dtype=np.float64), "median_nn"
    if isinstance(sigma, (int, float)):
        s = float(sigma)
        return np.full(num_waypoints, s, dtype=np.float64), f"{s:.6g}"
    if isinstance(sigma, str) and sigma.startswith("knn:"):
        body = sigma.split(":", 1)[1]
        if "*" in body:
            k_str, scale_str = body.split("*", 1)
            k = int(k_str)
            scale = float(scale_str)
            tag = f"knn:{k}*{scale:g}"
        else:
            k = int(body)
            scale = 1.0
            tag = f"knn:{k}"
        per = np.array(
            [scale * _knn_distance_from_point(my_path[i], behavior_centroids, k)
             for i in range(num_waypoints)],
            dtype=np.float64,
        )
        return per, tag
    raise ValueError(f"unrecognized sigma spec: {sigma!r}")


def construct_pullback_path(
    manifold: FittedManifold,
    behavior: BehaviorManifold,
    start_label: str,
    end_label: str,
    num_waypoints: int,
    sigma: SigmaSpec = None,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, str]:
    """Build the M_y straight-line and its kernel-barycenter pullback to M_h.

    Returns (my_path, pullback_sub, mean_sigma, sigma_per_waypoint, sigma_spec).
    - my_path is (K, 2): linear interpolation between the two behavior
      centroids in (V, A).
    - pullback_sub is (K, subspace_dim): the kernel-weighted barycenter
      of M_h centroids weighted by proximity in M_y to each waypoint.
    - mean_sigma is the scalar summary of σ (mean of per-waypoint array).
    - sigma_per_waypoint is (K,): the σ used at each waypoint.
    - sigma_spec is a string tag ("median_nn", "<float>", or "knn:<K>").
    """
    if start_label not in behavior.labels or end_label not in behavior.labels:
        raise ValueError(
            f"unknown labels: {start_label!r}, {end_label!r}; "
            f"behavior has {behavior.labels}"
        )
    if start_label not in manifold.labels or end_label not in manifold.labels:
        raise ValueError(
            f"unknown labels: {start_label!r}, {end_label!r}; "
            f"manifold has {manifold.labels}"
        )

    y_start = behavior.centroids[behavior.labels.index(start_label)]
    y_end = behavior.centroids[behavior.labels.index(end_label)]
    ts = np.linspace(0.0, 1.0, num_waypoints)
    my_path = (1.0 - ts)[:, None] * y_start[None, :] + ts[:, None] * y_end[None, :]
    my_path = my_path.astype(np.float32)

    sigma_per_waypoint, sigma_spec = _resolve_sigma(
        sigma, my_path, behavior.centroids
    )

    # Align M_h centroids to behavior labels in case ordering differs.
    h_centroids = np.stack(
        [
            manifold.centroids_subspace[manifold.labels.index(lbl)]
            for lbl in behavior.labels
        ],
        axis=0,
    ).astype(np.float32)

    pullback_sub = np.zeros((num_waypoints, manifold.num_components), dtype=np.float32)
    for k in range(num_waypoints):
        weights = _kernel_weights(my_path[k], behavior.centroids, float(sigma_per_waypoint[k]))
        pullback_sub[k] = weights @ h_centroids

    mean_sigma = float(sigma_per_waypoint.mean())
    return my_path, pullback_sub, mean_sigma, sigma_per_waypoint, sigma_spec


def compute_pullback(
    manifold: FittedManifold,
    behavior: BehaviorManifold,
    start_label: str,
    end_label: str,
    *,
    num_waypoints: int,
    geodesic_max_iter: int = 300,
    sigma: SigmaSpec = None,
) -> PullbackResult:
    """Build the pullback path + the two M_h baselines and compare them.

    Geometric prediction (Goodfire §3.3): if M_h's density geometry
    matches M_y's metric structure under the model's forward map, then
    the pullback of an M_y straight line should be close to an M_h
    geodesic — not to an M_h straight line.
    """
    from .geodesic import fit_geodesic, linear_interpolation

    my_path, pullback_sub, sigma_used, sigma_per_waypoint, sigma_spec = (
        construct_pullback_path(
            manifold, behavior, start_label, end_label, num_waypoints,
            sigma=sigma,
        )
    )

    start_idx = manifold.labels.index(start_label)
    end_idx = manifold.labels.index(end_label)
    start_sub = manifold.centroids_subspace[start_idx].astype(np.float32)
    end_sub = manifold.centroids_subspace[end_idx].astype(np.float32)

    # Snap pullback endpoints to exact centroids so the three paths
    # share endpoints — otherwise their lengths are not comparable.
    pullback_sub = pullback_sub.copy()
    pullback_sub[0] = start_sub
    pullback_sub[-1] = end_sub

    geometry = manifold.make_geometry()
    geodesic_result = fit_geodesic(
        geometry, start_sub, end_sub,
        num_waypoints=num_waypoints, max_iter=geodesic_max_iter,
    )
    geodesic_sub = geodesic_result.waypoints.astype(np.float32)
    linear_sub = linear_interpolation(start_sub, end_sub, num_waypoints).astype(np.float32)

    pullback_full = manifold.unproject(pullback_sub)
    geodesic_full = manifold.unproject(geodesic_sub)
    linear_full = manifold.unproject(linear_sub)

    pullback_length = float(geometry.path_length(pullback_sub))
    geodesic_length = float(geodesic_result.final_length)
    linear_length = float(geodesic_result.initial_length)

    dist_pg = np.linalg.norm(pullback_sub - geodesic_sub, axis=1)
    dist_pl = np.linalg.norm(pullback_sub - linear_sub, axis=1)
    dist_gl = np.linalg.norm(geodesic_sub - linear_sub, axis=1)

    return PullbackResult(
        start_label=start_label,
        end_label=end_label,
        sigma=sigma_used,
        sigma_spec=sigma_spec,
        sigma_per_waypoint=sigma_per_waypoint,
        my_path=my_path,
        pullback_sub=pullback_sub,
        geodesic_sub=geodesic_sub,
        linear_sub=linear_sub,
        pullback_full=pullback_full,
        geodesic_full=geodesic_full,
        linear_full=linear_full,
        pullback_length=pullback_length,
        geodesic_length=geodesic_length,
        linear_length=linear_length,
        dist_pullback_to_geodesic=dist_pg,
        dist_pullback_to_linear=dist_pl,
        dist_geodesic_to_linear=dist_gl,
    )
