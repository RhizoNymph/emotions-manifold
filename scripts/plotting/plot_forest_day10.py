"""Day-10 forest plot of all major behavioral findings.

Supersedes forest_plot_findings_with_4d.png (kept on disk), which predates the
two day-10 results: the judged bijective-spline win and the un-selected n=40
inverse-steering validation. Two panels with separate x-scales — the magnitude
difference between "geometric methods vs linear" (percent-level) and
"behavior-first optimization vs linear" (multiples) is the paper's thesis, so
one shared axis would bury the left panel.

Aggregates that live in summary JSONs are read from them; rows whose runs
predate aggregate summaries are hardcoded with their source noted inline
(same convention as the original first-pass figure script).

Run from the repo root:
    uv run python scripts/plotting/plot_forest_day10.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from manifold_emotions.analysis.stats import bootstrap_ci

OUT_DIR = Path("results/figures/writeup")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RIE_8D = Path("results/riemannian_analysis/_summary.json")
RIE_4D = Path("results/riemannian_analysis_4d/_summary.json")
BIJ = Path("results/spline_analysis_bijective_8d/_summary.json")
N40 = Path("results/surrogate_optimizer/analysis_n40.json")


def _bij_rows() -> list[tuple]:
    d = json.loads(BIJ.read_text())
    rows = []
    for m, p2 in (("spline_induced", None), ("spline_density", None)):
        g = np.array([p[f"{m}_margin"] for p in d["per_pair"]])
        lo, hi = bootstrap_ci(g, np.mean)
        p = stats.wilcoxon(g).pvalue
        rows.append((f"bijective {m.split('_')[1]} M_y-line (n=40)",
                     float(g.mean()), lo, hi, p, "bijective"))
    for m in ("spline_induced", "spline_density"):
        g = np.array([p[f"{m}_off_gap"] for p in d["per_pair"]])
        lo, hi = bootstrap_ci(g, np.mean)
        p = stats.wilcoxon(g).pvalue
        rows.append((f"bijective {m.split('_')[1]} off-M_y E (n=40)",
                     float(g.mean()), lo, hi, p, "bijective"))
    return rows


def _rie_rows() -> list[tuple]:
    rows = []
    for path, tag in ((RIE_8D, "8-D"), (RIE_4D, "4-D")):
        s = json.loads(path.read_text())
        ge = s["geodesic_vs_linear"]
        rows.append((f"G_E geodesic off-M_y E ({tag}, n=40)",
                     ge["off_my_e_gap_mean"], *ge["off_my_e_gap_ci"],
                     ge["off_my_e_wilcoxon_p_one_sided"], "ambient"))
        rows.append((f"G_E geodesic M_y-line ({tag}, n=40)",
                     ge["my_line_margin_mean"], *ge["my_line_margin_ci"],
                     ge["my_line_wilcoxon_p_one_sided"], "ambient"))
    pb = json.loads(RIE_8D.read_text())["pullback_vs_linear_for_reference"]
    rows.append(("pullback off-M_y E (8-D, n=40)",
                 pb["off_my_e_gap_mean"], *pb["off_my_e_gap_ci"],
                 pb["off_my_e_wilcoxon_p_one_sided"], "ambient"))
    rows.append(("pullback M_y-line (8-D, n=40)",
                 pb["my_line_margin_mean"], *pb["my_line_margin_ci"],
                 pb["my_line_wilcoxon_p_one_sided"], "ambient"))
    return rows


VA_SPLINE = Path("results/spline_analysis_8d/_summary.json")


def _va_spline_rows() -> list[tuple]:
    d = json.loads(VA_SPLINE.read_text())
    rows = []
    for m in ("spline_induced", "spline_density"):
        g = np.array([p[f"{m}_margin"] for p in d["per_pair"]])
        lo, hi = bootstrap_ci(g, np.mean)
        p = stats.wilcoxon(g).pvalue
        rows.append((f"V/A spline {m.split('_')[1]} M_y-line (n=40)",
                     float(g.mean()), lo, hi, p, "vaspline"))
    return rows


# Rows whose runs predate per-pair summary files; values from the cited
# writeup/day-journal sections.
STATIC_ROWS = [
    # writeup §9 pre-registered TV primary endpoint (two-sided)
    ("TV tv8-cv8 pullback M_y-line (n=40)", 0.026, -0.084, 0.123, 0.49, "tv"),
    # writeup §5 norm-matched composition (bootstrap CI + two-sided Wilcoxon)
    ("composition off-M_y E NM (n=20)", -0.031, -0.059, -0.004, 0.058, "composition"),
]


def main() -> None:
    left = _bij_rows() + _rie_rows() + _va_spline_rows() + STATIC_ROWS

    n40 = json.loads(N40.read_text())
    rh = n40["population_real_headroom"]
    cg = n40["coherence_gap_opt_minus_lin"]
    right = [
        ("inverse steering headroom (un-selected n=40)",
         rh["mean"], rh["ci"][0], rh["ci"][1],
         rh["wilcoxon_p_greater"], "inverse"),
        ("inverse steering coherence gap (n=40)",
         cg["mean"], cg["ci"][0], cg["ci"][1],
         cg["wilcoxon_p_two-sided"], "inverse"),
    ]

    colors = {"bijective": "#2b7bba", "ambient": "#7f7f7f", "vaspline": "#c44e52",
              "tv": "#8c6bb1", "composition": "#e07b39", "inverse": "#2b7bba"}

    fig, axes = plt.subplots(
        1, 2, figsize=(15, 8), gridspec_kw={"width_ratios": [2.4, 1]})
    for ax, rows, title in (
        (axes[0], left, "geometric methods: gap vs linear (positive = method better)"),
        (axes[1], right, "behavior-first optimization vs linear"),
    ):
        for i, (name, mean, lo, hi, p, group) in enumerate(rows):
            sig = p is not None and p < 0.05
            c = colors[group]
            ax.errorbar([mean], [i], xerr=[[mean - lo], [hi - mean]],
                        fmt="o" if sig else "x", color=c, ecolor=c,
                        markerfacecolor=c if sig else "white",
                        markeredgecolor=c, markersize=9, capsize=4, linewidth=2)
            label = f"{mean:+.3f}"
            if p is not None:
                label += f" p={p:.4f}" if p >= 0.0001 else " p<1e-4"
            ax.annotate(label, (hi, i), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=8.5)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in rows], fontsize=9)
        ax.invert_yaxis()
        ax.axvline(0, color="black", lw=0.7, ls="--", alpha=0.6)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", alpha=0.3)
        lo_all = min(r[2] for r in rows)
        hi_all = max(r[3] for r in rows)
        pad = 0.35 * (hi_all - lo_all)
        ax.set_xlim(lo_all - 0.15 * (hi_all - lo_all), hi_all + pad)
    axes[0].set_xlabel("effect size (mean, 95% bootstrap CI)")
    axes[1].set_xlabel("effect size — note the axis scale")
    fig.suptitle(
        "All major behavioral findings (day 10): filled = p<0.05.\n"
        "Geometry done right buys percent-level target gains (left); "
        "learning the coupling buys multiples (right).\n"
        "p-values as reported per run: G_E/pullback rows one-sided "
        "(method-better), bijective/V-A-spline/TV/inverse rows two-sided.",
        fontsize=10)
    fig.tight_layout()
    out = OUT_DIR / "forest_plot_findings_day10.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
