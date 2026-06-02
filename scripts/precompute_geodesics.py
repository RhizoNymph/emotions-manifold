"""Precompute geodesic waypoints for every pair, cache to disk.

The interactive dashboard needs path waypoints (not just lengths) for
all 435 pairs, ideally without recomputing on every UI interaction.
This script fits each geodesic once and writes the K-waypoint trajectory
to a single .npz keyed by (start_idx, end_idx) with start < end.

Cache format:
    data/geodesics_cache.npz
    - waypoints: (num_pairs, num_waypoints, num_components)
    - pair_indices: (num_pairs, 2) — (i, j) with i < j into manifold.labels
    - num_waypoints: scalar
    - manifold_path: str (which manifold this was fit against)

Run with:
    uv run python scripts/precompute_geodesics.py
    uv run python scripts/precompute_geodesics.py --num-waypoints 30
"""

from __future__ import annotations

import argparse
import time
from itertools import combinations
from pathlib import Path

import numpy as np

from manifold_emotions.config import load_config
from manifold_emotions.manifold.fit import FittedManifold
from manifold_emotions.manifold.geodesic import fit_geodesic

CACHE_PATH = Path("data/geodesics_cache.npz")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--num-waypoints", type=int, default=30)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--force", action="store_true",
                        help="recompute even if cache exists")
    args = parser.parse_args()

    config = load_config()
    manifold = FittedManifold.load(config.paths.manifold_h)

    if CACHE_PATH.exists() and not args.force:
        cached = np.load(CACHE_PATH, allow_pickle=True)
        if (cached["waypoints"].shape[1] == args.num_waypoints
                and len(cached["pair_indices"]) == len(manifold.labels)
                * (len(manifold.labels) - 1) // 2):
            print(f"cache OK at {CACHE_PATH} "
                  f"({cached['waypoints'].shape}); use --force to rebuild")
            return

    geometry = manifold.make_geometry()
    n = len(manifold.labels)
    pairs = list(combinations(range(n), 2))
    centroids = manifold.centroids_subspace.astype(np.float32)
    waypoints = np.zeros(
        (len(pairs), args.num_waypoints, manifold.num_components),
        dtype=np.float32,
    )
    pair_indices = np.zeros((len(pairs), 2), dtype=np.int32)

    start_time = time.monotonic()
    for k, (i, j) in enumerate(pairs):
        if k % 50 == 0:
            elapsed = time.monotonic() - start_time
            print(f"  geodesic {k:>3d}/{len(pairs)}  ({elapsed:.1f}s)", flush=True)
        result = fit_geodesic(
            geometry,
            centroids[i],
            centroids[j],
            num_waypoints=args.num_waypoints,
            max_iter=args.max_iter,
        )
        waypoints[k] = result.waypoints
        pair_indices[k] = (i, j)

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE_PATH,
        waypoints=waypoints,
        pair_indices=pair_indices,
        num_waypoints=np.array([args.num_waypoints]),
        labels=np.array(manifold.labels, dtype=object),
    )
    elapsed = time.monotonic() - start_time
    print(f"saved {len(pairs)} geodesics to {CACHE_PATH} ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
