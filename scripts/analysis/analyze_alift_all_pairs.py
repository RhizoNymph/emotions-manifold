"""All-pairs A_lift analysis (purely geometric, no vLLM needed).

Computes A_lift for all 171*170/2 = 14,535 pairs, then:
  1. Distribution: histogram + quantiles
  2. Per-emotion stats: which emotions appear most in high|A_lift| pairs?
  3. V/A density of pairs at each A_lift quantile (spatial structure)
  4. Predictor: does A_lift correlate with any precomputed pair_alignment
     metric, like isometry distance or geodesic length ratio?

Outputs to results/alift_all_pairs/ for use in writeup.
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path

from manifold_emotions.behavior.manifold import BehaviorManifold
from manifold_emotions.config import load_config
from manifold_emotions.manifold.fit import FittedManifold


def main() -> None:
    cfg = load_config()
    beh = BehaviorManifold.load(cfg.paths.manifold_y)
    mh = FittedManifold.load(cfg.paths.manifold_h)
    labels = list(beh.labels)
    C = beh.centroids
    N = len(labels)
    out_dir = Path("results/alift_all_pairs")
    out_dir.mkdir(parents=True, exist_ok=True)
    SIGMA = 0.077

    def kernel_avg(target_xy):
        sq = np.sum((C - target_xy[None, :]) ** 2, axis=1)
        log_w = -sq / (2 * SIGMA * SIGMA)
        log_w -= log_w.max()
        w = np.exp(log_w)
        w /= w.sum()
        return w @ C  # (2,) — kernel-weighted V, A

    print(f"Computing A_lift for all {N*(N-1)//2} pairs (σ={SIGMA})...")
    rows = []
    for i in range(N):
        if labels[i] not in mh.labels:
            continue
        for j in range(i + 1, N):
            if labels[j] not in mh.labels:
                continue
            a, b = C[i], C[j]
            chord_len = float(np.linalg.norm(b - a))
            if chord_len < 1.0:  # skip degenerate
                continue
            ts = np.linspace(0, 1, 30)
            wps = a[None, :] * (1 - ts)[:, None] + b[None, :] * ts[:, None]
            kernel_vas = np.array([kernel_avg(w) for w in wps])
            a_lifts = kernel_vas[:, 1] - wps[:, 1]
            v_lifts = kernel_vas[:, 0] - wps[:, 0]
            rows.append({
                "i": i, "j": j,
                "label_i": labels[i], "label_j": labels[j],
                "V_i": float(C[i, 0]), "A_i": float(C[i, 1]),
                "V_j": float(C[j, 0]), "A_j": float(C[j, 1]),
                "chord_len": chord_len,
                "a_lift": float(np.mean(a_lifts)),
                "v_lift": float(np.mean(v_lifts)),
                "midpoint_V": float((C[i, 0] + C[j, 0]) / 2),
                "midpoint_A": float((C[i, 1] + C[j, 1]) / 2),
            })

    a_lifts = np.array([r["a_lift"] for r in rows])
    v_lifts = np.array([r["v_lift"] for r in rows])
    chord_lens = np.array([r["chord_len"] for r in rows])
    print(f"  N = {len(rows)} pairs (after filtering chord >= 1.0)")
    print()
    print("==== A_lift distribution ====")
    print(f"  mean   = {a_lifts.mean():+.4f}")
    print(f"  std    = {a_lifts.std():.4f}")
    print(f"  range  = [{a_lifts.min():+.3f}, {a_lifts.max():+.3f}]")
    for q in (5, 10, 25, 50, 75, 90, 95):
        print(f"  p{q:>2d}    = {np.percentile(a_lifts, q):+.4f}")

    print()
    print("==== Per-emotion stats: appearances in top/bottom A_lift decile ====")
    p10, p90 = np.percentile(a_lifts, 10), np.percentile(a_lifts, 90)
    in_top = [r for r in rows if r["a_lift"] >= p90]
    in_bot = [r for r in rows if r["a_lift"] <= p10]
    from collections import Counter
    top_counter = Counter()
    bot_counter = Counter()
    for r in in_top:
        top_counter[r["label_i"]] += 1
        top_counter[r["label_j"]] += 1
    for r in in_bot:
        bot_counter[r["label_i"]] += 1
        bot_counter[r["label_j"]] += 1
    print("  Top-10 emotions appearing in TOP decile (predict-win-favored):")
    for em, n in top_counter.most_common(10):
        print(f"    {em:>20s}  {n} pairs")
    print()
    print("  Top-10 emotions appearing in BOTTOM decile (predict-loss-favored):")
    for em, n in bot_counter.most_common(10):
        print(f"    {em:>20s}  {n} pairs")

    print()
    print("==== Spatial structure: A_lift vs midpoint V/A ====")
    # Bin pairs by midpoint V, midpoint A, compute mean A_lift per cell
    v_bins = np.linspace(1.5, 6.5, 6)
    a_bins = np.linspace(2.0, 7.0, 6)
    grid = np.full((len(a_bins) - 1, len(v_bins) - 1), np.nan)
    counts = np.zeros_like(grid)
    for r in rows:
        vi = np.searchsorted(v_bins, r["midpoint_V"]) - 1
        ai = np.searchsorted(a_bins, r["midpoint_A"]) - 1
        if 0 <= vi < grid.shape[1] and 0 <= ai < grid.shape[0]:
            if np.isnan(grid[ai, vi]):
                grid[ai, vi] = 0.0
            grid[ai, vi] += r["a_lift"]
            counts[ai, vi] += 1
    grid = grid / np.where(counts > 0, counts, 1)
    print(f"  Grid: mean A_lift per (midpoint A, midpoint V) bin")
    print(f"  V bins: {v_bins.tolist()}")
    print(f"  A bins (rows reversed for plot orientation): {a_bins.tolist()}")
    print()
    print("  A_lift heatmap (rows = A high→low, cols = V low→high):")
    print(f"  {'':10s}" + "".join(f"V<{v_bins[k+1]:.1f}".rjust(8) for k in range(len(v_bins) - 1)))
    for ai in range(len(a_bins) - 2, -1, -1):
        row = f"  A<{a_bins[ai+1]:.1f}  "
        for vi in range(len(v_bins) - 1):
            val = grid[ai, vi]
            cnt = int(counts[ai, vi])
            if cnt == 0:
                row += "    --  "
            else:
                row += f" {val:+.3f}({cnt:>3d})"[-8:].rjust(8)
        print(row)
    summary = {
        "n_pairs": len(rows),
        "sigma": SIGMA,
        "a_lift_distribution": {
            "mean": float(a_lifts.mean()),
            "std": float(a_lifts.std()),
            "min": float(a_lifts.min()),
            "max": float(a_lifts.max()),
            "percentiles": {f"p{q}": float(np.percentile(a_lifts, q))
                            for q in (5, 10, 25, 50, 75, 90, 95)},
        },
        "top_emotions_in_top_decile": top_counter.most_common(20),
        "top_emotions_in_bottom_decile": bot_counter.most_common(20),
        "spatial_grid": {
            "v_bins": v_bins.tolist(),
            "a_bins": a_bins.tolist(),
            "mean_a_lift_per_cell": grid.tolist(),
            "count_per_cell": counts.tolist(),
        },
    }
    (out_dir / "_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {out_dir/'_summary.json'}")

    # Also save raw rows for downstream plotting
    np.savez_compressed(
        out_dir / "all_pairs.npz",
        a_lift=a_lifts,
        v_lift=v_lifts,
        chord_len=chord_lens,
        midpoint_V=np.array([r["midpoint_V"] for r in rows]),
        midpoint_A=np.array([r["midpoint_A"] for r in rows]),
        i=np.array([r["i"] for r in rows]),
        j=np.array([r["j"] for r in rows]),
    )
    print(f"saved {out_dir/'all_pairs.npz'}")


if __name__ == "__main__":
    main()
