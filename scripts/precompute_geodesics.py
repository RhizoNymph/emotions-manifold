"""Precompute geodesic waypoints for every pair of the production manifold.

The interactive dashboard needs path waypoints (not just lengths) for
all pairs without recomputing on every UI interaction. This script
loads the existing production manifold (no refit) and writes the cache
via ``manifold_emotions.manifold.geodesic_cache``; see that module for
the .npz format. To fit a *variant* manifold and its cache in one go,
use ``scripts/fit_manifold.py --tag ... --geodesics`` instead.

Run with:
    uv run python scripts/precompute_geodesics.py
    uv run python scripts/precompute_geodesics.py --num-waypoints 30
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from manifold_emotions.config import load_config
from manifold_emotions.manifold.fit import FittedManifold
from manifold_emotions.manifold.geodesic_cache import build_geodesic_cache

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

    cache = build_geodesic_cache(
        manifold,
        num_waypoints=args.num_waypoints,
        max_iter=args.max_iter,
        progress=print,
        progress_every=50,
    )
    cache.save(CACHE_PATH)
    print(f"saved {len(cache.pair_indices)} geodesics to {CACHE_PATH}")


if __name__ == "__main__":
    main()
