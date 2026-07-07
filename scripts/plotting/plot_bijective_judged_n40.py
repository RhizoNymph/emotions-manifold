"""Figure for writeup §7.5: judged n=40 bijective-spline margins vs linear.

Reads results/spline_analysis_bijective_8d/_summary.json and plots the
per-pair M_y-line margins (sorted) for spline_induced and spline_density,
with the mean margin annotated. Outputs to results/figures/writeup/ so
existing figures are preserved.
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

SUMMARY = Path("results/spline_analysis_bijective_8d/_summary.json")


def fig_bijective_judged():
    d = json.loads(SUMMARY.read_text())
    pp = d["per_pair"]
    si = np.array([p["spline_induced_margin"] for p in pp])
    sd = np.array([p["spline_density_margin"] for p in pp])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, m, label, p_two in (
        (axes[0], si, "spline_induced", 0.0023),
        (axes[1], sd, "spline_density", 0.0005),
    ):
        order = np.argsort(m)
        colors = ["#2b7bba" if v > 0 else "#c44e52" for v in m[order]]
        ax.bar(range(len(m)), m[order], color=colors)
        ax.axhline(0, color="black", lw=0.8)
        ax.axhline(m.mean(), color="#e07b39", ls="--", lw=1.2,
                   label=f"mean {m.mean():+.3f} (two-sided p={p_two})")
        ax.set_title(f"{label}: {int((m > 0).sum())}/{len(m)} wins vs linear")
        ax.set_xlabel("pair (sorted by margin)")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("M_y-line margin vs linear (positive = spline better)")
    fig.suptitle(
        "Judged n=40 bijective spline: the only geometric method to beat linear "
        "at target-tracking", fontsize=11)
    fig.tight_layout()
    out = OUT_DIR / "bijective_spline_judged_n40.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    fig_bijective_judged()
