"""Geodesic isometry test: do M_h geodesic distances align with V/A?

For each pair (i, j) of 171 emotion centroids:
- d_euc_sub   = ||c_i^pca - c_j^pca||   (straight line in 8-D PCA subspace; baseline)
- d_geo_arclen = Σ ||p_{k+1} - p_k||    (Euclidean arclength of G_E-minimizing path)
- d_geo_GE     = Σ ||p_{k+1} - p_k|| / √(α·e^{-E(mid)} + β)   (true G_E length)
- d_va         = ||v_i^va - v_j^va||    (Euclidean in 2-D V/A)

Test: does geodesic distance (under G_E or its arclength) correlate
MORE with V/A distance than the straight-line PCA distance does?

This is the geometric analog of the isometry edge we found at 171:
linear subspace gave r=+0.710 vs full residual r=+0.655 (edge +0.055).
Question: does the curved-metric geodesic give r > +0.710?

Output: results/riemannian_analysis/isometry_geodesic.{json,png}.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from manifold_emotions.behavior.manifold import BehaviorManifold
from manifold_emotions.config import load_config
from manifold_emotions.manifold.density import GaussianKDE
from manifold_emotions.manifold.fit import FittedManifold
from manifold_emotions.manifold.metric import DensityGeometry


CACHE_PATH = Path("data/geodesics_cache.npz")
OUT_DIR = Path("results/riemannian_analysis")


def main() -> None:
    cfg = load_config()
    mh = FittedManifold.load(cfg.paths.manifold_h)
    beh = BehaviorManifold.load(cfg.paths.manifold_y)

    if not CACHE_PATH.exists():
        raise SystemExit(
            f"Missing {CACHE_PATH}. Run scripts/precompute_geodesics.py first."
        )
    cache = np.load(CACHE_PATH, allow_pickle=True)
    waypoints = cache["waypoints"]            # (14535, 30, 8)
    pair_indices = cache["pair_indices"]      # (14535, 2)
    labels = list(cache["labels"])
    print(f"Loaded {len(pair_indices)} cached geodesics, "
          f"each with {waypoints.shape[1]} waypoints in {waypoints.shape[2]}-D")

    # Restrict to emotions that ALSO exist in M_y (some emotions are in
    # M_h but missing V/A ratings)
    beh_set = set(beh.labels)
    keep = np.array([(labels[i] in beh_set and labels[j] in beh_set)
                     for i, j in pair_indices], dtype=bool)
    pair_indices = pair_indices[keep]
    waypoints = waypoints[keep]
    print(f"Kept {len(pair_indices)} pairs after intersecting with M_y labels")

    # Build the same DensityGeometry used to fit the geodesics
    geometry = mh.make_geometry()
    # Verify by recomputing length on a sample
    sample_idx = np.random.default_rng(0).choice(len(pair_indices), size=5, replace=False)
    for k in sample_idx:
        wp = jnp.asarray(waypoints[k], dtype=jnp.float32)
        ge_len = float(geometry.path_length(wp))
        i, j = pair_indices[k]
        i, j = int(i), int(j)
        c_i = mh.centroids_subspace[i]
        c_j = mh.centroids_subspace[j]
        euc_chord = float(np.linalg.norm(c_i - c_j))
        print(f"  sanity: {labels[i]:>15s}↔{labels[j]:<15s}  euc_chord={euc_chord:7.3f}  G_E_len={ge_len:7.3f}")

    # ---- Distances ----
    pca_centroids = mh.centroids_subspace.astype(np.float32)
    va_centroids = np.zeros((len(labels), 2), dtype=np.float32)
    for k, lab in enumerate(labels):
        if lab in beh_set:
            j_va = beh.labels.index(lab)
            va_centroids[k] = beh.centroids[j_va]

    # Vectorized G_E length over all paths via JAX vmap
    print("\nComputing G_E lengths for all geodesics...")
    @jax.jit
    def ge_length_batch(paths):
        return jax.vmap(geometry.path_length)(paths)

    paths_j = jnp.asarray(waypoints, dtype=jnp.float32)
    # Chunk to avoid OOM on JAX device for 14k×30×8 paths
    chunk = 1024
    ge_lengths = np.zeros(len(paths_j), dtype=np.float32)
    for start in range(0, len(paths_j), chunk):
        end = min(start + chunk, len(paths_j))
        ge_lengths[start:end] = np.asarray(ge_length_batch(paths_j[start:end]))
        if start % 4096 == 0:
            print(f"  G_E batch {start:>5d}/{len(paths_j)}")
    print(f"  G_E lengths: min={ge_lengths.min():.3f}  median={np.median(ge_lengths):.3f}  max={ge_lengths.max():.3f}")

    # Euclidean arclength of geodesic paths
    diffs = waypoints[:, 1:, :] - waypoints[:, :-1, :]
    euc_arclen = np.sqrt((diffs * diffs).sum(axis=-1)).sum(axis=-1).astype(np.float32)
    # Straight-line PCA chord (for reference)
    euc_chord = np.linalg.norm(
        pca_centroids[pair_indices[:, 0]] - pca_centroids[pair_indices[:, 1]],
        axis=1
    ).astype(np.float32)
    # V/A target distance
    va_dist = np.linalg.norm(
        va_centroids[pair_indices[:, 0]] - va_centroids[pair_indices[:, 1]],
        axis=1
    ).astype(np.float32)

    # ---- Correlations vs V/A distance ----
    r_chord, _ = stats.pearsonr(euc_chord, va_dist)
    r_arclen, _ = stats.pearsonr(euc_arclen, va_dist)
    r_ge, _ = stats.pearsonr(ge_lengths, va_dist)

    # Spearman as a robustness check
    s_chord, _ = stats.spearmanr(euc_chord, va_dist)
    s_arclen, _ = stats.spearmanr(euc_arclen, va_dist)
    s_ge, _ = stats.spearmanr(ge_lengths, va_dist)

    print("\n" + "=" * 60)
    print(f"Isometry to V/A distance (n={len(pair_indices)} pairs)")
    print("=" * 60)
    print(f"{'distance metric':<35s} {'Pearson':>9s} {'Spearman':>9s}")
    print(f"{'8-D PCA straight chord (baseline)':<35s} {r_chord:>+9.3f} {s_chord:>+9.3f}")
    print(f"{'Euclidean arclen of geodesic path':<35s} {r_arclen:>+9.3f} {s_arclen:>+9.3f}")
    print(f"{'G_E length of geodesic (Riemannian)':<35s} {r_ge:>+9.3f} {s_ge:>+9.3f}")
    print()
    print(f"  edge (arclen − chord)  = {r_arclen - r_chord:+.4f}")
    print(f"  edge (G_E    − chord)  = {r_ge     - r_chord:+.4f}")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (xs, name, r) in zip(
        axes,
        [(euc_chord, "8-D PCA straight chord", r_chord),
         (euc_arclen, "Geodesic Euclidean arclen", r_arclen),
         (ge_lengths, "Geodesic G_E length", r_ge)],
    ):
        ax.scatter(xs, va_dist, s=2, alpha=0.15, c="steelblue")
        slope, intercept, *_ = stats.linregress(xs, va_dist)
        x_fit = np.linspace(xs.min(), xs.max(), 100)
        ax.plot(x_fit, slope * x_fit + intercept, "--", color="firebrick", lw=1.2)
        ax.set_xlabel(name)
        ax.set_ylabel("V/A distance")
        ax.set_title(f"r = {r:+.3f}")
    fig.suptitle(f"Isometry of M_h distance to V/A distance, n={len(pair_indices)} pairs", y=1.02)
    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "isometry_geodesic.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nsaved {out_path}")

    # ---- JSON summary ----
    summary = {
        "n_pairs": int(len(pair_indices)),
        "pca_chord_pearson_r": float(r_chord),
        "geodesic_arclen_pearson_r": float(r_arclen),
        "geodesic_GE_length_pearson_r": float(r_ge),
        "pca_chord_spearman_r": float(s_chord),
        "geodesic_arclen_spearman_r": float(s_arclen),
        "geodesic_GE_length_spearman_r": float(s_ge),
        "edge_arclen_vs_chord": float(r_arclen - r_chord),
        "edge_GE_vs_chord": float(r_ge - r_chord),
        "ge_lengths_summary": {
            "min": float(ge_lengths.min()),
            "median": float(np.median(ge_lengths)),
            "max": float(ge_lengths.max()),
            "mean": float(ge_lengths.mean()),
        },
        "euc_arclen_summary": {
            "min": float(euc_arclen.min()),
            "median": float(np.median(euc_arclen)),
            "max": float(euc_arclen.max()),
            "mean": float(euc_arclen.mean()),
        },
        "chord_summary": {
            "min": float(euc_chord.min()),
            "median": float(np.median(euc_chord)),
            "max": float(euc_chord.max()),
            "mean": float(euc_chord.mean()),
        },
    }
    (OUT_DIR / "isometry_geodesic.json").write_text(json.dumps(summary, indent=2))
    print(f"saved {OUT_DIR/'isometry_geodesic.json'}")


if __name__ == "__main__":
    main()
