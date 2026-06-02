"""Compare manifold steering vs. linear steering between two emotion centroids.

Per Goodfire §3.2: for each pair (e_start, e_end):
- Compute K waypoints along the M_h geodesic, lifted back to full activation space
- Compute K waypoints along the straight line in full activation space
- For each waypoint, generate N continuations from a neutral prompt
- Judge each continuation's (valence, arousal)
- Compare:
  - Smoothness: do behavioral coordinates move monotonically/smoothly?
  - Naturalness: how close is the trajectory to M_y (the behavior manifold)?
  - "Teleportation": linear paths cut through low-density regions and produce
    bursts of nonsense / off-target emotions at intermediate waypoints

We use the cumulative distance from the M_y embedding as the energy proxy
(Goodfire's E_BC is Bhattacharyya for discrete outputs; for continuous
(valence, arousal) we use Euclidean distance to the nearest M_y centroid).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..behavior.judge_text import judge_texts
from ..behavior.manifold import BehaviorManifold
from ..config import Config
from ..manifold.fit import FittedManifold
from ..manifold.geodesic import fit_geodesic, linear_interpolation
from ..vectors.diff_in_means import EmotionVectors
from .trajectory import SteeredContinuation, generate_along_path

# A short generic prompt set chosen to elicit some open-ended text the
# emotion vector can influence. Kept neutral so the prompt's own
# emotional content doesn't dominate the steering signal.
DEFAULT_NEUTRAL_PROMPTS: tuple[str, ...] = (
    "Tell me about your day in a few sentences.",
    "What's on your mind right now?",
    "Describe what you see out the window.",
)


@dataclass(frozen=True, slots=True)
class TrajectoryReport:
    """Behavioral trajectory under a particular steering path."""

    name: str  # "manifold" or "linear"
    waypoints_full: np.ndarray  # (K, hidden_size)
    continuations: list[SteeredContinuation]
    # Per-waypoint behavior centroid, averaged over the N prompts.
    waypoint_behavior_mean: np.ndarray  # (K, 2)  — (valence, arousal)
    # Cumulative off-manifold energy: sum over waypoints of distance to
    # the nearest M_y centroid, divided by K so trajectories of different
    # K are comparable.
    cumulative_off_manifold: float


@dataclass(frozen=True, slots=True)
class ComparisonReport:
    start_label: str
    end_label: str
    manifold: TrajectoryReport
    linear: TrajectoryReport


def _waypoints_along_manifold(
    manifold: FittedManifold,
    start_idx: int,
    end_idx: int,
    num_waypoints: int,
    max_iter: int,
) -> np.ndarray:
    """Geodesic in subspace, then lift back to full activation space."""
    geometry = manifold.make_geometry()
    start_sub = manifold.centroids_subspace[start_idx].astype(np.float32)
    end_sub = manifold.centroids_subspace[end_idx].astype(np.float32)
    result = fit_geodesic(
        geometry, start_sub, end_sub, num_waypoints=num_waypoints, max_iter=max_iter
    )
    return manifold.unproject(result.waypoints.astype(np.float32))


def _waypoints_along_linear(
    full_vectors: np.ndarray,  # (num_emotions, hidden_size)
    start_idx: int,
    end_idx: int,
    num_waypoints: int,
) -> np.ndarray:
    """Straight-line interpolation in full activation space."""
    return linear_interpolation(
        full_vectors[start_idx], full_vectors[end_idx], num_waypoints
    )


def _aggregate_waypoint_behavior(
    continuations: list[SteeredContinuation],
    text_id_to_rating: dict[str, tuple[float, float]],
    num_waypoints: int,
) -> tuple[np.ndarray, list[list[str]]]:
    """Average (valence, arousal) per waypoint over the prompts."""
    per_waypoint_v: list[list[float]] = [[] for _ in range(num_waypoints)]
    per_waypoint_a: list[list[float]] = [[] for _ in range(num_waypoints)]
    per_waypoint_text: list[list[str]] = [[] for _ in range(num_waypoints)]
    for cont in continuations:
        rating = text_id_to_rating.get(_text_id(cont))
        if rating is None:
            continue
        per_waypoint_v[cont.waypoint_index].append(rating[0])
        per_waypoint_a[cont.waypoint_index].append(rating[1])
        per_waypoint_text[cont.waypoint_index].append(cont.text)

    mean = np.zeros((num_waypoints, 2), dtype=np.float32)
    for k in range(num_waypoints):
        if per_waypoint_v[k]:
            mean[k, 0] = float(np.mean(per_waypoint_v[k]))
            mean[k, 1] = float(np.mean(per_waypoint_a[k]))
        else:
            mean[k] = np.nan
    return mean, per_waypoint_text


def _off_manifold_energy(
    waypoint_behavior: np.ndarray,
    my_centroids: np.ndarray,
) -> float:
    """Mean distance from each waypoint's behavior centroid to the nearest M_y point.

    Lower = trajectory hugged the behavior manifold (natural outputs);
    higher = trajectory passed through off-manifold (unnatural) regions.
    """
    if waypoint_behavior.size == 0:
        return float("nan")
    finite_mask = np.all(np.isfinite(waypoint_behavior), axis=1)
    if not finite_mask.any():
        return float("nan")
    behavior = waypoint_behavior[finite_mask]
    # Pairwise distances: (K_finite, num_emotions).
    diffs = behavior[:, None, :] - my_centroids[None, :, :]
    dists = np.sqrt((diffs * diffs).sum(axis=-1))
    nearest = dists.min(axis=1)
    return float(nearest.mean())


def _text_id(cont: SteeredContinuation) -> str:
    return f"wp{cont.waypoint_index:03d}_p{cont.prompt_index:02d}"


async def compare_steering(
    config: Config,
    emotion_vectors: EmotionVectors,
    manifold: FittedManifold,
    behavior: BehaviorManifold,
    start_label: str,
    end_label: str,
    *,
    num_waypoints: int = 20,
    num_prompts: int = 3,
    prompts: tuple[str, ...] = DEFAULT_NEUTRAL_PROMPTS,
    max_tokens: int = 96,
    geodesic_max_iter: int = 300,
    concurrency: int = 16,
    judge_cache_path: Path | None = None,
    steering_scale: float = 8.0,
) -> ComparisonReport:
    """Run the manifold-vs-linear steering comparison for one emotion pair.

    Both paths run through diff-in-means coordinates (the emotion direction
    minus the global mean), not raw centroids — for additive steering the
    direction matters; absolute residual position adds back the global mean
    of activations which has no emotion content.

    ``steering_scale`` multiplies every waypoint before sending. Diff-in-means
    norms are ~20 in this model; residual-stream norms at the steered layer
    are ~315, so an 8× scale puts steering vectors at ~50% of residual norm,
    roughly Anthropic's "strength 0.5" convention. Below ~5× the steering
    has weak behavioral effect; above ~15× fluency collapses.
    """
    if start_label not in manifold.labels or end_label not in manifold.labels:
        raise ValueError(
            f"unknown emotion labels: {start_label!r}, {end_label!r}; "
            f"manifold labels: {manifold.labels}"
        )
    start_idx_m = manifold.labels.index(start_label)
    end_idx_m = manifold.labels.index(end_label)
    start_idx_e = emotion_vectors.labels.index(start_label)
    end_idx_e = emotion_vectors.labels.index(end_label)

    manifold_waypoints = _waypoints_along_manifold(
        manifold, start_idx_m, end_idx_m, num_waypoints, geodesic_max_iter
    )
    linear_waypoints = _waypoints_along_linear(
        emotion_vectors.vectors, start_idx_e, end_idx_e, num_waypoints
    )

    manifold_waypoints = manifold_waypoints * steering_scale
    linear_waypoints = linear_waypoints * steering_scale

    # Take the first num_prompts from the prompt set.
    prompts_used = list(prompts[:num_prompts])

    manifold_cont = await generate_along_path(
        config, manifold_waypoints, prompts_used, max_tokens=max_tokens, concurrency=concurrency
    )
    linear_cont = await generate_along_path(
        config, linear_waypoints, prompts_used, max_tokens=max_tokens, concurrency=concurrency
    )

    # Judge every continuation. Use a single cache so re-runs are cheap;
    # text_ids namespace by trajectory name to avoid cross-trajectory collisions.
    passages: list[tuple[str, str]] = []
    for cont in manifold_cont:
        passages.append((f"manifold_{start_label}_{end_label}_{_text_id(cont)}", cont.text))
    for cont in linear_cont:
        passages.append((f"linear_{start_label}_{end_label}_{_text_id(cont)}", cont.text))
    ratings = await judge_texts(config, passages, cache_path=judge_cache_path)

    def index_ratings(prefix: str) -> dict[str, tuple[float, float]]:
        out: dict[str, tuple[float, float]] = {}
        for text_id, rating in ratings.items():
            if text_id.startswith(prefix):
                short_id = text_id[len(prefix):]
                out[short_id] = (rating.valence, rating.arousal)
        return out

    manifold_ratings = index_ratings(f"manifold_{start_label}_{end_label}_")
    linear_ratings = index_ratings(f"linear_{start_label}_{end_label}_")

    manifold_mean, _ = _aggregate_waypoint_behavior(
        manifold_cont, manifold_ratings, num_waypoints
    )
    linear_mean, _ = _aggregate_waypoint_behavior(
        linear_cont, linear_ratings, num_waypoints
    )

    manifold_report = TrajectoryReport(
        name="manifold",
        waypoints_full=manifold_waypoints,
        continuations=manifold_cont,
        waypoint_behavior_mean=manifold_mean,
        cumulative_off_manifold=_off_manifold_energy(manifold_mean, behavior.centroids),
    )
    linear_report = TrajectoryReport(
        name="linear",
        waypoints_full=linear_waypoints,
        continuations=linear_cont,
        waypoint_behavior_mean=linear_mean,
        cumulative_off_manifold=_off_manifold_energy(linear_mean, behavior.centroids),
    )

    return ComparisonReport(
        start_label=start_label,
        end_label=end_label,
        manifold=manifold_report,
        linear=linear_report,
    )
