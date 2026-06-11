"""Dimension ablation: isometry edge under G_E across PCA dimensionality.

Fits a manifold per dimensionality from the same emotion_vectors.npz,
computes geodesics on a fixed sampled pair set, and reports the V/A
isometry edge (G_E geodesic Pearson r minus straight-chord Pearson r)
for each.

Unifies run_dimension_ablation.py and run_denser_dim_sweep.py (archived
on archive/disorganized-scripts). The pair sample is seeded so runs at
different --dims values are directly comparable and mergeable:

    # original 4/8/16/32 grid
    uv run python scripts/experiments/run_dimension_ablation.py

    # the denser fill-in, merged with the original curve in the plot
    uv run python scripts/experiments/run_dimension_ablation.py \
        --dims 6,10,12,14,24 --out-name dimension_ablation_denser \
        --merge-plot-with results/riemannian_analysis/dimension_ablation.json

Keep --n-pairs/--seed at their defaults (800/5678) unless you want a
deliberately different sample — results across runs are only comparable
on the same pair set.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from manifold_emotions.behavior.manifold import BehaviorManifold
from manifold_emotions.config import load_config
from manifold_emotions.manifold.fit import fit_manifold
from manifold_emotions.manifold.geodesic import fit_geodesic
from manifold_emotions.vectors.diff_in_means import EmotionVectors

matplotlib.use("Agg")

NUM_WAYPOINTS = 30
OUT_DIR = Path("results/riemannian_analysis")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dims", default="4,8,16,32",
                        help="comma-separated PCA dimensionalities")
    parser.add_argument("--out-name", default="dimension_ablation",
                        help="basename for the output .json/.png in "
                             f"{OUT_DIR} (use a distinct name per sweep; "
                             "never overwrite an existing sweep's results)")
    parser.add_argument("--merge-plot-with", type=Path, default=None,
                        help="existing sweep JSON whose points are merged "
                             "into this run's plot (not its JSON)")
    parser.add_argument("--n-pairs", type=int, default=800)
    parser.add_argument("--seed", type=int, default=5678)
    args = parser.parse_args()

    dims = [int(d) for d in args.dims.split(",")]

    cfg = load_config()
    ev = EmotionVectors.load(cfg.paths.emotion_vectors)
    beh = BehaviorManifold.load(cfg.paths.manifold_y)
    beh_set = set(beh.labels)

    print(f"Loaded {len(ev.labels)} emotion vectors with d={ev.vectors.shape[1]}")

    # Build the pair set ONCE from the joint coverage of M_y + emotion_vectors
    common_labels = [lab for lab in ev.labels if lab in beh_set]
    print(f"Common with V/A: {len(common_labels)}")
    label_to_idx_in_beh = {lab: beh.labels.index(lab) for lab in common_labels}

    rng = np.random.default_rng(args.seed)
    n_common = len(common_labels)
    all_pairs = [(i, j) for i in range(n_common) for j in range(i + 1, n_common)]
    sample = rng.choice(len(all_pairs), size=min(args.n_pairs, len(all_pairs)),
                        replace=False)
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
    for d in dims:
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
    out_json = OUT_DIR / f"{args.out_name}.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {out_json}")

    # Plot, optionally merged with a previous sweep's points.
    combined = list(results)
    if args.merge_plot_with is not None and args.merge_plot_with.exists():
        combined.extend(json.loads(args.merge_plot_with.read_text())["by_dim"])
    combined.sort(key=lambda r: r["dim"])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    dims_x = [r["dim"] for r in combined]
    chord_rs = [r["chord_pearson_r"] for r in combined]
    ge_rs = [r["GE_pearson_r"] for r in combined]
    edges = [r["edge_pearson"] for r in combined]
    ax.plot(dims_x, chord_rs, "o-", color="C0", label="PCA chord (baseline)", ms=8)
    ax.plot(dims_x, ge_rs, "s-", color="C2", label="G_E geodesic length", ms=8)
    y_top = max(chord_rs + ge_rs)
    for d, e in zip(dims_x, edges, strict=True):
        ax.annotate(f"+{e:.3f}" if e >= 0 else f"{e:.3f}",
                    xy=(d, y_top + 0.008),
                    ha="center", fontsize=8, color="firebrick", rotation=45)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("PCA subspace dimensionality")
    ax.set_ylabel("Pearson r vs V/A distance")
    ax.set_title(f"Isometry to V/A vs PCA dimensionality (n={len(pair_labels)} pairs)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(min(chord_rs + ge_rs) - 0.02, y_top + 0.06)
    plt.tight_layout()
    out_png = OUT_DIR / f"{args.out_name}.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
