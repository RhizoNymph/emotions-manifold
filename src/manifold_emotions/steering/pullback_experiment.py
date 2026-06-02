"""Pullback steering experiment: compare pullback, geodesic, and linear paths.

Builds on ``compare_steering`` but runs *three* trajectories instead of
two:

- ``pullback``: the kernel-barycenter inverse of an M_y straight line
- ``geodesic``: the M_h geodesic between the same emotion centroids
- ``linear``: the M_h straight-line interpolation

For each trajectory we generate from every waypoint and judge the
resulting behavior. We then measure two energies per trajectory:

1. ``off_manifold_energy``: distance from the *nearest M_y centroid*
   (the same metric ``compare_steering`` uses, so we can splice the
   pullback results into the existing manifold-vs-linear plots).
2. ``my_geodesic_distance``: distance from the *target M_y straight
   line* (the pullback-natural metric — its zero is exact behavior
   isometry with the M_y geodesic).

The geometric prediction is that the pullback path should produce
behavior closer to the M_y straight line than either the geodesic or
the linear path. If true, that's evidence the M_h density geometry
encodes the behavior manifold's metric structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..behavior.judge_text import judge_texts
from ..behavior.manifold import BehaviorManifold
from ..config import Config
from ..manifold.fit import FittedManifold
from ..manifold.pullback import compute_pullback, PullbackResult, SigmaSpec
from .experiment import _aggregate_waypoint_behavior, _off_manifold_energy, _text_id
from .trajectory import SteeredContinuation, generate_along_path


@dataclass(frozen=True, slots=True)
class PullbackTrajectoryReport:
    """Behavior under one of (pullback, geodesic, linear)."""

    name: str
    waypoints_full: np.ndarray         # (K, hidden_size)
    continuations: list[SteeredContinuation]
    waypoint_behavior_mean: np.ndarray  # (K, 2)
    off_manifold_energy: float          # mean distance to nearest M_y centroid
    my_geodesic_distance: float         # mean distance to target M_y straight line


@dataclass(frozen=True, slots=True)
class PullbackExperimentReport:
    """Full results of the pullback experiment for a single pair."""

    start_label: str
    end_label: str
    my_path: np.ndarray                # (K, 2) target straight line in M_y
    geometry: PullbackResult           # path geometry without generation
    pullback: PullbackTrajectoryReport
    geodesic: PullbackTrajectoryReport
    linear: PullbackTrajectoryReport


def _distance_to_my_line(
    waypoint_behavior: np.ndarray, my_path: np.ndarray
) -> tuple[float, np.ndarray]:
    """Mean per-waypoint Euclidean distance from behavior to target M_y line.

    waypoint_behavior is (K, 2); my_path is (K, 2) and indexes by the
    *same* waypoint position k. Returns (mean, per-waypoint distances).
    Skips waypoints whose behavior is NaN (judge failed or no rating).
    """
    finite_mask = np.all(np.isfinite(waypoint_behavior), axis=1)
    per = np.full(waypoint_behavior.shape[0], np.nan, dtype=np.float64)
    if not finite_mask.any():
        return float("nan"), per
    diffs = waypoint_behavior[finite_mask] - my_path[finite_mask]
    dists = np.sqrt(np.sum(diffs * diffs, axis=1))
    per[finite_mask] = dists
    return float(dists.mean()), per


async def run_pullback_experiment(
    config: Config,
    manifold: FittedManifold,
    behavior: BehaviorManifold,
    start_label: str,
    end_label: str,
    *,
    num_waypoints: int,
    num_prompts: int,
    prompts: tuple[str, ...],
    max_tokens: int = 96,
    geodesic_max_iter: int = 300,
    concurrency: int = 16,
    steering_scale: float = 8.0,
    judge_cache_path: Path | None = None,
    sigma: SigmaSpec = None,
) -> PullbackExperimentReport:
    """End-to-end: build pullback + baselines, steer, judge, summarize."""

    geom = compute_pullback(
        manifold, behavior, start_label, end_label,
        num_waypoints=num_waypoints, geodesic_max_iter=geodesic_max_iter,
        sigma=sigma,
    )

    # Diff-in-means-style waypoints: the FittedManifold stores PCA-projected
    # diff-in-means centroids already (centroids_subspace is computed from
    # emotion_vectors.vectors via PCA), so unproject directly gives us
    # additive steering vectors. Scale matches compare_steering's default.
    pullback_steer = geom.pullback_full * steering_scale
    geodesic_steer = geom.geodesic_full * steering_scale
    linear_steer = geom.linear_full * steering_scale

    prompts_used = list(prompts[:num_prompts])

    pullback_cont = await generate_along_path(
        config, pullback_steer, prompts_used,
        max_tokens=max_tokens, concurrency=concurrency,
    )
    geodesic_cont = await generate_along_path(
        config, geodesic_steer, prompts_used,
        max_tokens=max_tokens, concurrency=concurrency,
    )
    linear_cont = await generate_along_path(
        config, linear_steer, prompts_used,
        max_tokens=max_tokens, concurrency=concurrency,
    )

    passages: list[tuple[str, str]] = []
    pair_tag = f"{start_label}_{end_label}"
    for cont in pullback_cont:
        passages.append((f"pullback_{pair_tag}_{_text_id(cont)}", cont.text))
    for cont in geodesic_cont:
        passages.append((f"geodesic_{pair_tag}_{_text_id(cont)}", cont.text))
    for cont in linear_cont:
        passages.append((f"linear_{pair_tag}_{_text_id(cont)}", cont.text))

    ratings = await judge_texts(config, passages, cache_path=judge_cache_path)

    def index_ratings(prefix: str) -> dict[str, tuple[float, float]]:
        out: dict[str, tuple[float, float]] = {}
        for text_id, rating in ratings.items():
            if text_id.startswith(prefix):
                short_id = text_id[len(prefix):]
                out[short_id] = (rating.valence, rating.arousal)
        return out

    pullback_ratings = index_ratings(f"pullback_{pair_tag}_")
    geodesic_ratings = index_ratings(f"geodesic_{pair_tag}_")
    linear_ratings = index_ratings(f"linear_{pair_tag}_")

    pullback_mean, _ = _aggregate_waypoint_behavior(pullback_cont, pullback_ratings, num_waypoints)
    geodesic_mean, _ = _aggregate_waypoint_behavior(geodesic_cont, geodesic_ratings, num_waypoints)
    linear_mean, _ = _aggregate_waypoint_behavior(linear_cont, linear_ratings, num_waypoints)

    def report(
        name: str,
        steer: np.ndarray,
        conts: list[SteeredContinuation],
        mean: np.ndarray,
    ) -> PullbackTrajectoryReport:
        ofm = _off_manifold_energy(mean, behavior.centroids)
        dist, _ = _distance_to_my_line(mean, geom.my_path)
        return PullbackTrajectoryReport(
            name=name,
            waypoints_full=steer,
            continuations=conts,
            waypoint_behavior_mean=mean,
            off_manifold_energy=ofm,
            my_geodesic_distance=dist,
        )

    return PullbackExperimentReport(
        start_label=start_label,
        end_label=end_label,
        my_path=geom.my_path,
        geometry=geom,
        pullback=report("pullback", pullback_steer, pullback_cont, pullback_mean),
        geodesic=report("geodesic", geodesic_steer, geodesic_cont, geodesic_mean),
        linear=report("linear", linear_steer, linear_cont, linear_mean),
    )
