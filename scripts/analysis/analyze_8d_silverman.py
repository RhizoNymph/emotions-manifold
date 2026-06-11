"""Analyze 8-D Silverman bandwidth vs 8-D production (clustered_NN) bandwidth.

Reads results/pullback_8d_silverman/{pair}.json and compares against
results/pullback/{pair}.json (production 8-D, clustered_NN bandwidth).

Question: does the geometric edge of Silverman bandwidth over clustered_NN
(+0.062 vs +0.050 on 800 pairs) translate behaviorally? If yes, the
bandwidth heuristic matters; if no, our headline behavioral results are
robust to this choice.

Output:
- results/riemannian_analysis_8d_silverman/_summary.json
- results/figures/writeup/silverman_vs_clustered_nn.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


SILVERMAN_DIR = Path("results/pullback_8d_silverman")
PROD_DIR = Path("results/pullback")
OUT_DIR = Path("results/riemannian_analysis_8d_silverman")
FIG_DIR = Path("results/figures/writeup")


def load_metrics(path: Path) -> dict | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    t = data["trajectories"]
    return {m: {
        "off": float(t[m]["off_manifold_energy"]),
        "myl": float(t[m]["my_geodesic_distance"]),
    } for m in ("pullback", "geodesic", "linear")}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    pairs = []
    for p in sorted(SILVERMAN_DIR.glob("*.json")):
        if p.name.startswith("_"):
            continue
        prod = load_metrics(PROD_DIR / p.name)
        silv = load_metrics(p)
        if prod is None or silv is None:
            continue
        pairs.append({"pair": p.stem, "prod": prod, "silv": silv})

    print(f"Loaded {len(pairs)} pairs with both Silverman and clustered_NN data")
    if not pairs:
        raise SystemExit("No matching pairs")

    out = {"n_pairs": len(pairs), "per_method": {}}

    print(f"\n{'method':<10} {'metric':<5}  {'prod':>8} {'silv':>8} {'Δ':>9}  "
          f"{'CI':>20} {'Wilcoxon-p (silv<prod)':>22}")
    for m in ("pullback", "geodesic", "linear"):
        for metric in ("off", "myl"):
            prod_vals = np.array([p["prod"][m][metric] for p in pairs])
            silv_vals = np.array([p["silv"][m][metric] for p in pairs])
            diff = silv_vals - prod_vals
            mean = float(diff.mean())
            try:
                ci = stats.bootstrap((diff,), np.mean, confidence_level=0.95,
                                      random_state=0, n_resamples=2000).confidence_interval
                _, p_val = stats.wilcoxon(diff, alternative="less")
            except Exception:
                ci = type("X", (), dict(low=float("nan"), high=float("nan")))()
                p_val = float("nan")
            print(f"{m:<10} {metric:<5}  {prod_vals.mean():>8.3f} {silv_vals.mean():>8.3f} {mean:>+9.4f}  "
                  f"[{float(ci.low):+.4f}, {float(ci.high):+.4f}] {p_val:>22.3f}")
            out["per_method"].setdefault(m, {})[metric] = {
                "mean_prod": float(prod_vals.mean()),
                "mean_silv": float(silv_vals.mean()),
                "diff_mean": mean,
                "ci": [float(ci.low), float(ci.high)],
                "wilcoxon_p_silv_lt_prod": float(p_val),
                "n": int(len(pairs)),
            }

    out["per_pair"] = pairs
    (OUT_DIR / "_summary.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved {OUT_DIR/'_summary.json'}")

    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    metric_titles = {"off": "off-M_y E (lower=on-manifold)",
                     "myl": "M_y-line distance (lower=on-target)"}
    for r_i, metric in enumerate(["off", "myl"]):
        for c_i, m in enumerate(["pullback", "geodesic", "linear"]):
            ax = axes[r_i, c_i]
            prod_vals = np.array([p["prod"][m][metric] for p in pairs])
            silv_vals = np.array([p["silv"][m][metric] for p in pairs])
            ax.scatter(prod_vals, silv_vals, alpha=0.7, color="steelblue",
                       edgecolor="black", lw=0.4)
            lim = [min(prod_vals.min(), silv_vals.min()) * 0.95,
                   max(prod_vals.max(), silv_vals.max()) * 1.05]
            ax.plot(lim, lim, "k--", lw=1, alpha=0.5)
            ax.set_xlim(lim); ax.set_ylim(lim)
            ax.set_xlabel(f"clustered_NN {m}")
            ax.set_ylabel(f"Silverman {m}")
            mean_diff = (silv_vals - prod_vals).mean()
            ax.set_title(f"{m} {metric_titles[metric]}\nmean(silv − prod) = {mean_diff:+.4f}",
                         fontsize=9)
            ax.grid(alpha=0.3)
    plt.suptitle(f"Silverman vs clustered_NN bandwidth (8-D, n={len(pairs)} pairs)", y=1.0)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "silverman_vs_clustered_nn.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {FIG_DIR/'silverman_vs_clustered_nn.png'}")


if __name__ == "__main__":
    main()
