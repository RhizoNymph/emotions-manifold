"""Fit a 4-D production manifold from the full 171 emotion vectors
and precompute geodesics for all pairs. Saves to
``data/manifold_h_4d_full.npz`` and ``data/geodesics_cache_4d.npz``
so the existing 8-D production artifacts are NOT overwritten.

After this you can re-run the n=40 A-lift behavioral steering at
4-D by pointing run_pullback_experiment.py at the 4-D manifold
(needs a small CLI patch; see below) or just running:

    PYTHONPATH=src uv run python scripts/run_pullback_experiment_4d.py

(also created here as a thin wrapper).

NOTE: this script does NOT touch vLLM or Claude API. Pure JAX/CPU.
"""

from __future__ import annotations

import time
from itertools import combinations
from pathlib import Path

import numpy as np

from manifold_emotions.config import load_config
from manifold_emotions.manifold.fit import fit_manifold
from manifold_emotions.manifold.geodesic import fit_geodesic
from manifold_emotions.vectors.diff_in_means import EmotionVectors


MANIFOLD_4D_PATH = Path("data/manifold_h_4d_full.npz")
GEODESICS_4D_PATH = Path("data/geodesics_cache_4d.npz")
NUM_WAYPOINTS = 30


def main() -> None:
    cfg = load_config()
    ev = EmotionVectors.load(cfg.paths.emotion_vectors)

    print(f"Loaded {len(ev.labels)} emotion vectors")
    print(f"Fitting 4-D manifold...")
    manifold, pca = fit_manifold(ev, num_components=4)
    manifold.save(MANIFOLD_4D_PATH)
    print(f"  saved {MANIFOLD_4D_PATH}: "
          f"{manifold.num_components}-D, {len(manifold.labels)} centroids, "
          f"bandwidth={manifold.kde_bandwidth:.3f}")
    print(f"  explained variance: "
          f"per-PC {[round(r, 3) for r in pca.explained_variance_ratio.tolist()]}, "
          f"cumulative {float(pca.explained_variance_ratio.sum()):.1%}")

    geometry = manifold.make_geometry()
    n = len(manifold.labels)
    pairs = list(combinations(range(n), 2))
    centroids = manifold.centroids_subspace.astype(np.float32)
    waypoints = np.zeros((len(pairs), NUM_WAYPOINTS, manifold.num_components),
                          dtype=np.float32)
    pair_indices = np.zeros((len(pairs), 2), dtype=np.int32)

    print(f"\nPrecomputing geodesics for {len(pairs)} pairs in 4-D...")
    t0 = time.monotonic()
    for k, (i, j) in enumerate(pairs):
        if k % 200 == 0:
            elapsed = time.monotonic() - t0
            rate = (k + 1) / max(elapsed, 0.01)
            eta = (len(pairs) - k - 1) / rate
            print(f"  fit {k:>5d}/{len(pairs)}  "
                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining, "
                  f"{rate:.1f} pairs/s)")
        result = fit_geodesic(
            geometry, centroids[i], centroids[j],
            num_waypoints=NUM_WAYPOINTS, max_iter=300,
        )
        waypoints[k] = result.waypoints
        pair_indices[k] = (i, j)

    np.savez_compressed(
        GEODESICS_4D_PATH,
        waypoints=waypoints,
        pair_indices=pair_indices,
        num_waypoints=np.array([NUM_WAYPOINTS]),
        labels=np.array(manifold.labels, dtype=object),
    )
    print(f"\nsaved {GEODESICS_4D_PATH}: "
          f"{waypoints.shape} float32 ({waypoints.nbytes / 1e6:.1f} MB)")
    print(f"Total wall time: {time.monotonic() - t0:.0f}s")


if __name__ == "__main__":
    main()
