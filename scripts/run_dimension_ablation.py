"""Dimension ablation: isometry edge under G_E across PCA dimensionality.

Fits 4-D, 8-D (production), 16-D, and 32-D manifolds from the same
emotion_vectors.npz, computes geodesics on a sampled pair set, and
reports the V/A isometry edge for each.

Tests the hypothesis: does the +0.049 isometry edge at 8-D grow
or shrink as we give the manifold more dimensions to bend through?

If the edge grows with dimension: production 8-D is under-fitting
the curvature. If it shrinks: 8-D already captures most of the
curvature available; higher dimensions add noise.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from manifold_emotions.behavior.manifold import BehaviorManifold
from manifold_emotions.config import load_config
from manifold_emotions.manifold.fit import fit_manifold
from manifold_emotions.manifold.geodesic import fit_geodesic
from manifold_emotions.vectors.diff_in_means import EmotionVectors


N_PAIRS = 800   # tractable per dimension setting
SEED = 5678
NUM_WAYPOINTS = 30
DIMS = [4, 8, 16, 32]
OUT_DIR = Path("results/riemannian_analysis")


def main() -> None:
    cfg = load_config()
    ev = EmotionVectors.load(cfg.paths.emotion_vectors)
    beh = BehaviorManifold.load(cfg.paths.manifold_y)
    beh_set = set(beh.labels)

    print(f"Loaded {len(ev.labels)} emotion vectors with d={ev.vectors.shape[1]}")

    # Build the pair set ONCE from the joint coverage of M_y + emotion_vectors
    common_labels = [lab for lab in ev.labels if lab in beh_set]
    print(f"Common with V/A: {len(common_labels)}")
    label_to_idx_in_ev = {lab: ev.labels.index(lab) for lab in common_labels}
    label_to_idx_in_beh = {lab: beh.labels.index(lab) for lab in common_labels}

    rng = np.random.default_rng(SEED)
    n_common = len(common_labels)
    all_pairs = [(i, j) for i in range(n_common) for j in range(i + 1, n_common)]
    sample = rng.choice(len(all_pairs), size=min(N_PAIRS, len(all_pairs)), replace=False)
    pair_labels = [(common_labels[all_pairs[k][0]], common_labels[all_pairs[k][1]])
                   for k in sample]
    print(f"Sampled {len(pair_labels)} pairs for ablation")

    # V/A distances per pair (fixed across dims)
    va_dist = np.array([
        float(np.linalg.norm(
            beh.centroids[label_to_idx_in_beh[a]] - beh.centroids[label_to_idx_in_beh[b]]
        ))
        for a, b in pair_labels
    ])

    results = []
    for d in DIMS:
        print(f"\n--- {d}-D ---")
        t0 = time.monotonic()
        manifold, _ = fit_manifold(ev, num_components=d)
        print(f"  fit {d}-D manifold in {time.monotonic()-t0:.1f}s, "
              f"bandwidth={manifold.kde_bandwidth:.3f}")
        geom = manifold.make_geometry()
        label_to_idx_in_mh = {lab: manifold.labels.index(lab) for lab in common_labels
                              if lab in manifold.labels}

        centroids = manifold.centroids_subspace.astype(np.float32)
        chord_lens = np.zeros(len(pair_labels), dtype=np.float32)
        ge_lens = np.zeros(len(pair_labels), dtype=np.float32)

        for k, (a, b) in enumerate(pair_labels):
            i = label_to_idx_in_mh[a]
            j = label_to_idx_in_mh[b]
            c_a, c_b = centroids[i], centroids[j]
            chord_lens[k] = float(np.linalg.norm(c_a - c_b))
            res = fit_geodesic(geom, c_a, c_b, num_waypoints=NUM_WAYPOINTS, max_iter=200)
            ge_lens[k] = res.final_length
            if k % 100 == 0:
                elapsed = time.monotonic() - t0
                rate = (k + 1) / max(elapsed, 0.01)
                eta = (len(pair_labels) - k - 1) / rate
                print(f"  fit {k:>4d}/{len(pair_labels)}  "
                      f"({elapsed:.0f}s, eta {eta:.0f}s, {rate:.1f}/s)")

        r_chord, _ = stats.pearsonr(chord_lens, va_dist)
        r_ge, _ = stats.pearsonr(ge_lens, va_dist)
        s_chord, _ = stats.spearmanr(chord_lens, va_dist)
        s_ge, _ = stats.spearmanr(ge_lens, va_dist)
        edge = r_ge - r_chord
        print(f"  d={d}: chord r={r_chord:+.3f}  G_E r={r_ge:+.3f}  edge={edge:+.3f}")

        results.append({
            "dim": d,
            "n_pairs": len(pair_labels),
            "chord_pearson_r": float(r_chord),
            "GE_pearson_r": float(r_ge),
            "chord_spearman_r": float(s_chord),
            "GE_spearman_r": float(s_ge),
            "edge_pearson": float(edge),
            "bandwidth": float(manifold.kde_bandwidth),
            "wall_seconds": float(time.monotonic() - t0),
        })

    summary = {
        "n_pairs": len(pair_labels),
        "by_dim": results,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "dimension_ablation.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {OUT_DIR/'dimension_ablation.json'}")

    fig, ax = plt.subplots(figsize=(8, 5))
    dims_x = [r["dim"] for r in results]
    chord_rs = [r["chord_pearson_r"] for r in results]
    ge_rs = [r["GE_pearson_r"] for r in results]
    edges = [r["edge_pearson"] for r in results]
    ax.plot(dims_x, chord_rs, "o-", color="C0", label="PCA chord (baseline)", ms=8)
    ax.plot(dims_x, ge_rs, "s-", color="C2", label="G_E geodesic length", ms=8)
    for d, e in zip(dims_x, edges):
        ax.annotate(f"+{e:.3f}" if e >= 0 else f"{e:.3f}",
                    xy=(d, max(chord_rs + ge_rs) + 0.005),
                    ha="center", fontsize=9, color="firebrick")
    ax.set_xlabel("PCA subspace dimensionality")
    ax.set_ylabel("Pearson r vs V/A distance")
    ax.set_title(f"Isometry to V/A vs PCA dimensionality (n={len(pair_labels)} pairs)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "dimension_ablation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {OUT_DIR/'dimension_ablation.png'}")


if __name__ == "__main__":
    main()
