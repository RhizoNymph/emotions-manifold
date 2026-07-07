"""Figures for writeup §12.5: un-selected n=40 surrogate validation.

Left: predicted vs actual headroom per pair (calibration). Right: per-pair
real headroom sorted, with the selected top-5 highlighted. Reads
results/surrogate_optimizer/validation_results_n40.json. Outputs to
results/figures/writeup/ so existing figures are preserved.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path("results/figures/writeup")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS = Path("results/surrogate_optimizer/validation_results_n40.json")
SELECTED = {
    "hope->sensitive", "hope->unhappy", "obstinate->proud",
    "proud->sympathetic", "self-conscious->thankful",
}


def fig_n40_validation():
    d = json.loads(RESULTS.read_text())
    pp = d["per_pair"]
    pred = np.array([p["linear_actual_dist"] - p["surrogate_predicted_dist"] for p in pp])
    actual = np.array([p["real_headroom"] for p in pp])
    sel = np.array([p["pair"] in SELECTED for p in pp])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.scatter(pred[~sel], actual[~sel], s=28, color="#2b7bba", label="un-selected (35)")
    ax.scatter(pred[sel], actual[sel], s=45, color="#e07b39", marker="D",
               label="previously-selected top 5")
    lo = min(pred.min(), actual.min()) - 0.05
    hi = max(pred.max(), actual.max()) + 0.05
    ax.plot([lo, hi], [lo, hi], color="black", lw=0.8, ls=":")
    r = np.corrcoef(pred, actual)[0, 1]
    ax.set_xlabel("predicted headroom (offline surrogate)")
    ax.set_ylabel("actual headroom (judged)")
    ax.set_title(f"Calibration out-of-selection: r = {r:.3f}")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    order = np.argsort(actual)
    colors = ["#e07b39" if s else "#2b7bba" for s in sel[order]]
    ax.bar(range(len(actual)), actual[order], color=colors)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(actual.mean(), color="#444444", ls="--", lw=1.2,
               label=f"population mean {actual.mean():+.3f}")
    ax.set_xlabel("pair (sorted by actual headroom)")
    ax.set_ylabel("real headroom (linear − optimized)")
    ax.set_title(f"{int((actual > 0).sum())}/{len(actual)} pairs beat matched linear")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Un-selected n=40 behavior-first validation: optimized vs matched linear",
        fontsize=11)
    fig.tight_layout()
    out = OUT_DIR / "surrogate_n40_validation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    fig_n40_validation()
