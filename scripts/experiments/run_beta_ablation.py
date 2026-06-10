"""β ablation for the density-aware metric G_E = (α·e^{-E}+β)^{-1}·I.

Tests whether weaker curvature (large β) or stronger curvature
(small β) changes the geometric isometry edge. If the edge grows
with smaller β, that's a hint that our current setting is
under-curving; if it shrinks, the curvature regime is already
saturated and behavioral inertness isn't a too-weak-curvature
artifact.

For each β: refit geodesics on a random subset of N_PAIRS pairs
(reusing the same cached pair_indices for direct comparison),
compute G_E length under that β, correlate with V/A distance.

We don't rerun behavioral steering here — geometric-only ablation,
which is cheap (60-120 min total) and answers whether the curvature
regime is the bottleneck.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from manifold_emotions.behavior.manifold import BehaviorManifold
from manifold_emotions.config import load_config
from manifold_emotions.manifold.fit import FittedManifold
from manifold_emotions.manifold.geodesic import fit_geodesic
from manifold_emotions.manifold.metric import DensityGeometry


N_PAIRS = 1000  # random subset for tractable wall time
N_WAYPOINTS = 30
SEED = 1234

BETAS = [0.001, 0.01, 0.1, 1.0]  # 0.01 is the production default

OUT_DIR = Path("results/riemannian_analysis")


def main() -> None:
    cfg = load_config()
    mh = FittedManifold.load(cfg.paths.manifold_h)
    beh = BehaviorManifold.load(cfg.paths.manifold_y)

    cache = np.load("data/geodesics_cache.npz", allow_pickle=True)
    pair_indices_all = cache["pair_indices"]
    labels = list(cache["labels"])

    # Restrict to pairs where BOTH endpoints have V/A
    beh_set = set(beh.labels)
    keep = np.array([(labels[i] in beh_set and labels[j] in beh_set)
                     for i, j in pair_indices_all], dtype=bool)
    pair_indices_all = pair_indices_all[keep]
    print(f"Eligible pairs: {len(pair_indices_all)}")

    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(len(pair_indices_all), size=N_PAIRS, replace=False)
    pair_indices = pair_indices_all[sample_idx]
    print(f"Sampled {N_PAIRS} pairs for ablation")

    pca = mh.centroids_subspace.astype(np.float32)

    # V/A distance per pair (fixed across β)
    va_centroids = np.zeros((len(labels), 2), dtype=np.float32)
    for k, lab in enumerate(labels):
        if lab in beh_set:
            va_centroids[k] = beh.centroids[beh.labels.index(lab)]
    va_dist = np.linalg.norm(
        va_centroids[pair_indices[:, 0]] - va_centroids[pair_indices[:, 1]],
        axis=1
    )

    # Straight-chord baseline (also fixed across β)
    euc_chord = np.linalg.norm(
        pca[pair_indices[:, 0]] - pca[pair_indices[:, 1]], axis=1
    )
    r_chord, _ = stats.pearsonr(euc_chord, va_dist)
    print(f"Baseline PCA chord r vs V/A: {r_chord:+.3f}")

    results = []
    for beta in BETAS:
        print(f"\n--- β = {beta} ---")
        kde = mh.make_density()
        geom = DensityGeometry(energy_fn=kde.energy, alpha=mh.alpha, beta=beta)
        t0 = time.monotonic()
        ge_lens = np.zeros(N_PAIRS, dtype=np.float32)
        for k, (i, j) in enumerate(pair_indices):
            res = fit_geodesic(
                geom, pca[int(i)], pca[int(j)],
                num_waypoints=N_WAYPOINTS, max_iter=200,
            )
            ge_lens[k] = res.final_length
            if k % 100 == 0:
                elapsed = time.monotonic() - t0
                rate = (k + 1) / max(elapsed, 0.01)
                eta = (N_PAIRS - k - 1) / rate
                print(f"  fit {k:>4d}/{N_PAIRS}  ({elapsed:.0f}s elapsed, "
                      f"~{eta:.0f}s remaining, {rate:.1f} pairs/s)")

        r_ge, _ = stats.pearsonr(ge_lens, va_dist)
        s_ge, _ = stats.spearmanr(ge_lens, va_dist)

        # Also report median G_E length and ratio vs chord
        med_len = float(np.median(ge_lens))
        mean_ratio = float(np.mean(ge_lens / euc_chord))
        print(f"  β={beta}: r={r_ge:+.3f}  (Spearman {s_ge:+.3f}) "
              f"med_len={med_len:.2f}  mean(G_E/chord)={mean_ratio:.3f}")
        results.append({
            "beta": float(beta),
            "alpha": float(mh.alpha),
            "n_pairs": N_PAIRS,
            "pearson_r": float(r_ge),
            "spearman_r": float(s_ge),
            "edge_vs_chord": float(r_ge - r_chord),
            "median_GE_length": med_len,
            "mean_GE_over_chord_ratio": mean_ratio,
            "wall_seconds": float(time.monotonic() - t0),
        })

    summary = {
        "n_pairs": N_PAIRS,
        "alpha": float(mh.alpha),
        "baseline_PCA_chord_r": float(r_chord),
        "production_beta": float(mh.beta),
        "by_beta": results,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "beta_ablation.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {OUT_DIR/'beta_ablation.json'}")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    betas = [r["beta"] for r in results]
    rs = [r["pearson_r"] for r in results]
    edges = [r["edge_vs_chord"] for r in results]

    ax = axes[0]
    ax.semilogx(betas, rs, "o-", color="C2", markersize=8, lw=2,
                label="G_E geodesic length")
    ax.axhline(r_chord, color="C0", linestyle="--", lw=1.5,
               label=f"PCA chord baseline (r={r_chord:+.3f})")
    ax.set_xlabel("β  (smaller = more aggressive curvature)")
    ax.set_ylabel("Pearson r vs V/A distance")
    ax.set_title(f"Isometry edge across β (n={N_PAIRS})")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.semilogx(betas, edges, "o-", color="firebrick", markersize=8, lw=2)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("β")
    ax.set_ylabel("Edge: r(G_E) − r(PCA chord)")
    ax.set_title("Curvature contribution as a function of β")
    ax.grid(alpha=0.3)

    fig.suptitle("β ablation: does stronger or weaker curvature help isometry?", y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "beta_ablation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {OUT_DIR/'beta_ablation.png'}")


if __name__ == "__main__":
    main()
