"""Fit an 8-D manifold with the SILVERMAN bandwidth heuristic and
precompute geodesics for all pairs. Saves to
``data/manifold_h_8d_silverman.npz`` and
``data/geodesics_cache_8d_silverman.npz``.

The production 8-D manifold uses the clustered-NN bandwidth heuristic
(bw=3.982). The denser dim sweep + manifold_alternatives experiments
showed Silverman bandwidth (bw=2.504) gives a small but positive edge
(+0.062 vs +0.050 production) at the geometric isometry level for the
same 800 pairs. This script sets up the infrastructure to test that
edge behaviorally on the n=40 chord pairs.
"""

from __future__ import annotations

import time
from itertools import combinations
from pathlib import Path

import numpy as np

from manifold_emotions.config import load_config
from manifold_emotions.manifold.density import silverman_bandwidth
from manifold_emotions.manifold.fit import fit_manifold
from manifold_emotions.manifold.geodesic import fit_geodesic
from manifold_emotions.vectors.diff_in_means import EmotionVectors


MANIFOLD_PATH = Path("data/manifold_h_8d_silverman.npz")
GEODESICS_PATH = Path("data/geodesics_cache_8d_silverman.npz")
NUM_WAYPOINTS = 30
NUM_COMPONENTS = 8


def main() -> None:
    cfg = load_config()
    ev = EmotionVectors.load(cfg.paths.emotion_vectors)

    print(f"Loaded {len(ev.labels)} emotion vectors")

    # Fit a temporary manifold to compute Silverman bandwidth in the
    # 8-D subspace (after PCA projection).
    print("Computing Silverman bandwidth in PCA-8 subspace...")
    tmp_manifold, _ = fit_manifold(ev, num_components=NUM_COMPONENTS)
    silverman_bw = silverman_bandwidth(tmp_manifold.centroids_subspace)
    print(f"  Silverman bandwidth: {silverman_bw:.4f}")
    print(f"  (production clustered_NN bandwidth: {tmp_manifold.kde_bandwidth:.4f})")

    print(f"\nFitting 8-D manifold with Silverman bandwidth...")
    manifold, pca = fit_manifold(
        ev, num_components=NUM_COMPONENTS, bandwidth=silverman_bw,
    )
    manifold.save(MANIFOLD_PATH)
    print(f"  saved {MANIFOLD_PATH}: "
          f"{manifold.num_components}-D, {len(manifold.labels)} centroids, "
          f"bandwidth={manifold.kde_bandwidth:.3f}")

    geometry = manifold.make_geometry()
    n = len(manifold.labels)
    pairs = list(combinations(range(n), 2))
    centroids = manifold.centroids_subspace.astype(np.float32)
    waypoints = np.zeros((len(pairs), NUM_WAYPOINTS, manifold.num_components),
                          dtype=np.float32)
    pair_indices = np.zeros((len(pairs), 2), dtype=np.int32)

    print(f"\nPrecomputing geodesics for {len(pairs)} pairs (Silverman 8-D)...")
    t0 = time.monotonic()
    for k, (i, j) in enumerate(pairs):
        if k % 200 == 0:
            elapsed = time.monotonic() - t0
            rate = (k + 1) / max(elapsed, 0.01)
            eta = (len(pairs) - k - 1) / rate
            print(f"  fit {k:>5d}/{len(pairs)}  "
                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining, "
                  f"{rate:.1f} pairs/s)", flush=True)
        result = fit_geodesic(
            geometry, centroids[i], centroids[j],
            num_waypoints=NUM_WAYPOINTS, max_iter=300,
        )
        waypoints[k] = result.waypoints
        pair_indices[k] = (i, j)

    np.savez_compressed(
        GEODESICS_PATH,
        waypoints=waypoints,
        pair_indices=pair_indices,
        num_waypoints=np.array([NUM_WAYPOINTS]),
        labels=np.array(manifold.labels, dtype=object),
    )
    print(f"\nsaved {GEODESICS_PATH}: "
          f"{waypoints.shape} float32 ({waypoints.nbytes / 1e6:.1f} MB)")
    print(f"Total wall time: {time.monotonic() - t0:.0f}s")


if __name__ == "__main__":
    main()
