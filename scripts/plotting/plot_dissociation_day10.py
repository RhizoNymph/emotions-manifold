"""Day-10 update of the geometric-vs-behavioral dissociation panel.

Ports fig_geom_vs_behavioral_dissociation from the archived
generate_first_pass_figures.py (feat/extended-analyses @ 1b8cb95) and adds the
judged bijective-spline bars to the behavioral panel. The original figure's
unqualified title ("curvature ... only routes on-manifold-ness — not
V/A-target tracking") is now false for the bijective construction (§7.5), so
this version scopes the claim to ambient constructions and shows the
exception. Saves a new file; the original is kept on disk.

Run from the repo root:
    uv run python scripts/plotting/plot_dissociation_day10.py
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


def _pullback_block(summary: dict) -> dict:
    # 8-D and 4-D summaries name the pullback-vs-linear block differently.
    return summary.get("pullback_vs_linear_for_reference",
                       summary.get("pullback_vs_linear"))


def main() -> None:
    da = json.loads(Path("results/riemannian_analysis/dimension_ablation.json").read_text())
    s8 = json.loads(Path("results/riemannian_analysis/_summary.json").read_text())
    s4 = json.loads(Path("results/riemannian_analysis_4d/_summary.json").read_text())
    bij = json.loads(Path("results/spline_analysis_bijective_8d/_summary.json").read_text())

    dims = [d["dim"] for d in da["by_dim"]]
    chord_r = [d["chord_pearson_r"] for d in da["by_dim"]]
    ge_r = [d["GE_pearson_r"] for d in da["by_dim"]]
    edge = [d["edge_pearson"] for d in da["by_dim"]]

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 9.5))

    ax_top.plot(dims, chord_r, marker="o", color="grey",
                label="Straight chord (no curvature)", linewidth=1.5)
    ax_top.plot(dims, ge_r, marker="s", color="firebrick",
                label="G_E geodesic (Riemannian)", linewidth=1.5)
    ax_top.fill_between(dims, chord_r, ge_r,
                        where=[g > c for g, c in zip(ge_r, chord_r)],
                        color="firebrick", alpha=0.18, label="G_E edge > 0")
    ax_top.fill_between(dims, chord_r, ge_r,
                        where=[g < c for g, c in zip(ge_r, chord_r)],
                        color="grey", alpha=0.18, label="G_E edge < 0")
    for d, e in zip(dims, edge):
        ax_top.annotate(f"edge={e:+.3f}",
                        (d, max(ge_r[dims.index(d)], chord_r[dims.index(d)]) + 0.015),
                        ha="center", fontsize=9,
                        color="darkgreen" if e > 0 else "darkred")
    ax_top.set_xticks(dims)
    ax_top.set_xlabel("PCA dimension of M_h")
    ax_top.set_ylabel("Pearson r vs V/A pairwise distance")
    ax_top.set_title(
        f"Geometric (ambient G_E): edge largest at 4–6-D, negative above 8-D "
        f"(n={da['n_pairs']} sampled pairs)")
    ax_top.legend(loc="lower left", fontsize=9)
    ax_top.grid(alpha=0.25)

    # Behavioral panel: ambient methods at 8-D/4-D + the judged bijective spline.
    rows = []  # (tag, method, metric, mean, lo, hi, p)
    for tag, s in [("8-D", s8), ("4-D", s4)]:
        pb = _pullback_block(s)
        ge = s["geodesic_vs_linear"]
        rows.append((tag, "geodesic", "off-M_y",
                     ge["off_my_e_gap_mean"], *ge["off_my_e_gap_ci"],
                     ge["off_my_e_wilcoxon_p_one_sided"]))
        rows.append((tag, "geodesic", "M_y-line",
                     ge["my_line_margin_mean"], *ge["my_line_margin_ci"],
                     ge["my_line_wilcoxon_p_one_sided"]))
        rows.append((tag, "pullback", "off-M_y",
                     pb["off_my_e_gap_mean"], *pb["off_my_e_gap_ci"],
                     pb["off_my_e_wilcoxon_p_one_sided"]))
        rows.append((tag, "pullback", "M_y-line",
                     pb["my_line_margin_mean"], *pb["my_line_margin_ci"],
                     pb["my_line_wilcoxon_p_one_sided"]))
    for m, metric, key in (("bijective", "off-M_y", "spline_induced_off_gap"),
                           ("bijective", "M_y-line", "spline_induced_margin")):
        g = np.array([p[key] for p in bij["per_pair"]])
        lo, hi = bootstrap_ci(g, np.mean)
        rows.append(("8-D", m, metric, float(g.mean()), lo, hi,
                     stats.wilcoxon(g).pvalue))

    groups = [("geodesic", "off-M_y"), ("geodesic", "M_y-line"),
              ("pullback", "off-M_y"), ("pullback", "M_y-line"),
              ("bijective", "off-M_y"), ("bijective", "M_y-line")]
    color_for_method = {"geodesic": "firebrick", "pullback": "steelblue",
                        "bijective": "#2b7bba"}
    hatch_for_metric = {"off-M_y": "", "M_y-line": "//"}

    x = np.arange(len(groups))
    width = 0.36
    for offset, tag, alpha in [(-width / 2, "8-D", 0.55), (+width / 2, "4-D", 1.0)]:
        xs, means, errs_lo, errs_hi, ps, cols, hats = [], [], [], [], [], [], []
        for gi, (method, metric) in enumerate(groups):
            row = next((r for r in rows
                        if r[0] == tag and r[1] == method and r[2] == metric), None)
            if row is None:
                continue
            xs.append(x[gi] + offset)
            means.append(row[3]); errs_lo.append(row[3] - row[4])
            errs_hi.append(row[5] - row[3]); ps.append(row[6])
            cols.append(color_for_method[method])
            hats.append(hatch_for_metric[metric])
        bars = ax_bot.bar(xs, means, width, color=cols, edgecolor="black",
                          linewidth=0.6, alpha=alpha, label=tag)
        for bar, h in zip(bars, hats):
            if h:
                bar.set_hatch(h)
        ax_bot.errorbar(xs, means, yerr=[errs_lo, errs_hi],
                        fmt="none", color="black", capsize=3, linewidth=0.8)
        for xi, m, p in zip(xs, means, ps):
            sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
            ax_bot.annotate(f"{m:+.3f}{sig}", (xi, m),
                            ha="center", va="bottom" if m >= 0 else "top",
                            fontsize=8)

    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels([f"{m}\n{mt}" for m, mt in groups])
    ax_bot.axhline(0, color="k", linewidth=0.5)
    ax_bot.set_ylabel("Behavioral edge vs linear\n(+ = method beats linear)")
    ax_bot.set_title(
        "Behavioral (same n=40 pairs): ambient methods route on-manifold-ness only; "
        "the bijective spline (8-D, judged) is the exception —\n"
        "it wins M_y-line too (+0.085**), the one construction tight enough and "
        "parameterized right (** = p<0.01)")
    ax_bot.legend(loc="upper left", title="PCA dim of M_h")
    ax_bot.grid(alpha=0.25, axis="y")

    fig.suptitle(
        "The geometric-vs-behavioral dissociation, day-10 version:\n"
        "ambient curvature buys cleaner isometry at low dim but routes only "
        "'on-manifold-ness'; a tight bijectively-parameterized manifold also "
        "buys percent-level target tracking", fontsize=11)
    fig.tight_layout()
    out = OUT_DIR / "geom_vs_behavioral_dissociation_day10.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
