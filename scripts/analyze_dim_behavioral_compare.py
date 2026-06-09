"""3-way per-pair behavioral comparison across d ∈ {4, 6, 8}.

Reads results/pullback/{pair}.json (8-D), results/pullback_4d/{pair}.json,
and results/pullback_6d/{pair}.json — for the 40-pair common set — and
reports for each method (pullback, geodesic, linear):

- off-M_y E across the three dimensions
- M_y-line distance across the three dimensions
- Wilcoxon paired tests d=6 vs d=8 and d=6 vs d=4 for each method

Confirms or refutes the geometric finding (G_E edge peaks at d=6 in the
denser sweep, +0.085 > +0.063 at d=4 > +0.050 at d=8) by checking
whether behavioral metrics show the same dimensional ordering.

Output:
- results/riemannian_analysis_6d/dim_behavioral_compare.json
- results/figures/writeup/dim_behavioral_compare.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


DIR_4D = Path("results/pullback_4d")
DIR_6D = Path("results/pullback_6d")
DIR_8D = Path("results/pullback")
OUT_DIR = Path("results/riemannian_analysis_6d")
FIG_DIR = Path("results/figures/writeup")


def load_one(p: Path) -> dict | None:
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    t = data["trajectories"]
    return {m: {
        "off": float(t[m]["off_manifold_energy"]),
        "myl": float(t[m]["my_geodesic_distance"]),
    } for m in ("pullback", "geodesic", "linear")}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    pairs = []
    for p in sorted(DIR_8D.glob("*.json")):
        if p.name.startswith("_"):
            continue
        d8 = load_one(p)
        d4 = load_one(DIR_4D / p.name)
        d6 = load_one(DIR_6D / p.name)
        if d4 is None or d6 is None or d8 is None:
            continue
        pairs.append({"pair": p.stem, "d4": d4, "d6": d6, "d8": d8})

    print(f"Loaded {len(pairs)} pairs present at all three dimensions")
    if len(pairs) < 5:
        print("  Note: d=6 results not yet ready (need d=6 chain to complete)")
        return

    out = {"n_pairs": len(pairs), "per_method": {}}

    print(f"\n{'method':<10} {'metric':<5}  {'d=4 mean':>8} {'d=6 mean':>8} {'d=8 mean':>8}  "
          f"{'6 vs 8 Δ':>9} {'6 vs 4 Δ':>9}  {'p(6<8)':>8} {'p(6<4)':>8}")
    for m in ("pullback", "geodesic", "linear"):
        for metric in ("off", "myl"):
            v4 = np.array([p["d4"][m][metric] for p in pairs])
            v6 = np.array([p["d6"][m][metric] for p in pairs])
            v8 = np.array([p["d8"][m][metric] for p in pairs])
            d_68 = v6 - v8
            d_64 = v6 - v4
            # one-sided Wilcoxon: H1 is d=6 better (lower)
            try:
                _, p_68 = stats.wilcoxon(d_68, alternative="less")
                _, p_64 = stats.wilcoxon(d_64, alternative="less")
            except Exception:
                p_68 = float("nan"); p_64 = float("nan")
            print(f"{m:<10} {metric:<5}  {v4.mean():>8.3f} {v6.mean():>8.3f} {v8.mean():>8.3f}  "
                  f"{d_68.mean():>+9.4f} {d_64.mean():>+9.4f}  {p_68:>8.3f} {p_64:>8.3f}")
            out["per_method"].setdefault(m, {})[metric] = {
                "mean_d4": float(v4.mean()),
                "mean_d6": float(v6.mean()),
                "mean_d8": float(v8.mean()),
                "diff_6_minus_8_mean": float(d_68.mean()),
                "diff_6_minus_4_mean": float(d_64.mean()),
                "wilcoxon_p_6_lt_8": float(p_68),
                "wilcoxon_p_6_lt_4": float(p_64),
                "n": int(len(pairs)),
            }

    out["per_pair"] = pairs
    (OUT_DIR / "dim_behavioral_compare.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved {OUT_DIR/'dim_behavioral_compare.json'}")

    # Plot: 3-panel × 2-row grid (one row per metric, columns per method)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    for r_i, metric in enumerate(["off", "myl"]):
        for c_i, m in enumerate(["pullback", "geodesic", "linear"]):
            ax = axes[r_i, c_i]
            stats4 = out["per_method"][m][metric]
            xs = [4, 6, 8]
            means = [stats4["mean_d4"], stats4["mean_d6"], stats4["mean_d8"]]
            ax.plot(xs, means, "o-", color="steelblue", ms=10, lw=2)
            for x, y in zip(xs, means):
                ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                            xytext=(8, 5), fontsize=9)
            ax.set_xticks(xs)
            ax.set_xlabel("PCA dimensionality")
            ylabel = "off-M_y E" if metric == "off" else "M_y-line distance"
            ax.set_ylabel(ylabel)
            ax.set_title(f"{m} {ylabel}")
            ax.grid(alpha=0.3)
    plt.suptitle(f"Behavioral metrics across d ∈ {{4, 6, 8}} (n={len(pairs)} pairs)", y=1.0)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "dim_behavioral_compare.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {FIG_DIR/'dim_behavioral_compare.png'}")


if __name__ == "__main__":
    main()
