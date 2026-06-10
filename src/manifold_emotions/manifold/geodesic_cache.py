"""All-pairs geodesic waypoint cache: build, save, load.

The steering experiments and the dashboard need the K-waypoint geodesic
trajectory for every emotion pair without recomputing on each use. This
module owns the (pairs × waypoints × dims) cache and its on-disk format,
which is shared by every manifold variant (per-dim, per-bandwidth).

Cache .npz format (unchanged from the original scripts):
    - waypoints:     (num_pairs, num_waypoints, num_components) float32
    - pair_indices:  (num_pairs, 2) int32 — (i, j), i < j, into ``labels``
    - num_waypoints: (1,) int
    - labels:        (num_emotions,) object — manifold labels at build time
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import structlog

from .fit import FittedManifold
from .geodesic import fit_geodesic

log = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class GeodesicCache:
    """Precomputed geodesic waypoints for all emotion pairs of a manifold."""

    labels: tuple[str, ...]
    waypoints: np.ndarray  # (num_pairs, num_waypoints, num_components)
    pair_indices: np.ndarray  # (num_pairs, 2) int32, i < j

    @property
    def num_waypoints(self) -> int:
        return self.waypoints.shape[1]

    def lookup(self, start_idx: int, end_idx: int) -> np.ndarray:
        """Waypoints for an (unordered) pair; reversed when start > end."""
        i, j = min(start_idx, end_idx), max(start_idx, end_idx)
        mask = (self.pair_indices[:, 0] == i) & (self.pair_indices[:, 1] == j)
        rows = np.nonzero(mask)[0]
        if len(rows) == 0:
            raise KeyError(f"pair ({start_idx}, {end_idx}) not in geodesic cache")
        path = self.waypoints[rows[0]]
        return path if start_idx <= end_idx else path[::-1]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            waypoints=self.waypoints,
            pair_indices=self.pair_indices,
            num_waypoints=np.array([self.num_waypoints]),
            labels=np.array(self.labels, dtype=object),
        )

    @classmethod
    def load(cls, path: Path) -> GeodesicCache:
        with np.load(path, allow_pickle=True) as data:
            return cls(
                labels=tuple(str(x) for x in data["labels"]),
                waypoints=data["waypoints"],
                pair_indices=data["pair_indices"],
            )


def build_geodesic_cache(
    manifold: FittedManifold,
    *,
    num_waypoints: int = 30,
    max_iter: int = 300,
    progress: Callable[[str], None] | None = None,
    progress_every: int = 200,
) -> GeodesicCache:
    """Fit geodesics for all label pairs of ``manifold`` under its G_E metric.

    CPU/JAX only — no vLLM or judge involvement. ``progress`` (e.g.
    ``print``) receives a rate/ETA line every ``progress_every`` pairs.
    """
    geometry = manifold.make_geometry()
    n = len(manifold.labels)
    pairs = list(combinations(range(n), 2))
    centroids = manifold.centroids_subspace.astype(np.float32)
    waypoints = np.zeros(
        (len(pairs), num_waypoints, manifold.num_components), dtype=np.float32
    )
    pair_indices = np.zeros((len(pairs), 2), dtype=np.int32)

    t0 = time.monotonic()
    for k, (i, j) in enumerate(pairs):
        if progress is not None and k % progress_every == 0:
            elapsed = time.monotonic() - t0
            rate = (k + 1) / max(elapsed, 0.01)
            eta = (len(pairs) - k - 1) / rate
            progress(
                f"  fit {k:>5d}/{len(pairs)}  "
                f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining, "
                f"{rate:.1f} pairs/s)"
            )
        result = fit_geodesic(
            geometry,
            centroids[i],
            centroids[j],
            num_waypoints=num_waypoints,
            max_iter=max_iter,
        )
        waypoints[k] = result.waypoints
        pair_indices[k] = (i, j)

    log.info(
        "manifold.geodesic_cache.built",
        num_pairs=len(pairs),
        num_waypoints=num_waypoints,
        num_components=manifold.num_components,
        wall_sec=round(time.monotonic() - t0, 1),
    )
    return GeodesicCache(
        labels=manifold.labels,
        waypoints=waypoints,
        pair_indices=pair_indices,
    )
