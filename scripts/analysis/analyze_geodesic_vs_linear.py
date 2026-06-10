"""Analog of analyze_alift_expansion.py but for geodesic vs linear.

Asks: does the Riemannian density-aware geodesic in M_h (under G_E)
beat plain linear interpolation, on the same n=40 pairs we used to
test pullback vs linear?

All three trajectories (pullback / geodesic / linear) are present in
each results/pullback/{s}_{e}.json since run_pullback_experiment.py
runs them together. This script just extracts geodesic where
analyze_alift_expansion.py extracts pullback.

Outputs: results/riemannian_analysis/{summary, plots} so existing
pullback summary at results/alift_expansion/ is preserved.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from manifold_emotions.analysis.stats import bootstrap_ci


ORIGINAL_PAIRS = {
    ("happy", "sad"): +0.157,
    ("excited", "weary"): -0.013,
    ("depressed", "energized"): -0.009,
    ("terrified", "serene"): -0.098,
    ("hope", "unhappy"): +0.234,
    ("amused", "ashamed"): +0.221,
    ("grumpy", "hopeful"): +0.219,
    ("proud", "sympathetic"): -0.229,
    ("brooding", "proud"): -0.212,
    ("brooding", "pleased"): -0.209,
}


def load_pair_metrics(s, e):
    path = Path(f"results/pullback/{s}_{e}.json")
    if not path.exists():
        alt = Path(f"results/pullback/{e}_{s}.json")
        if alt.exists():
            path = alt
        else:
            return None
    d = json.loads(path.read_text())
    t = d["trajectories"]
    return {
        "pb_myl": t["pullback"]["my_geodesic_distance"],
        "ge_myl": t["geodesic"]["my_geodesic_distance"],
        "li_myl": t["linear"]["my_geodesic_distance"],
        "pb_off": t["pullback"]["off_manifold_energy"],
        "ge_off": t["geodesic"]["off_manifold_energy"],
        "li_off": t["linear"]["off_manifold_energy"],
        "g_pullback_len": d["geometry"]["pullback_length"],
        "g_geodesic_len": d["geometry"]["geodesic_length"],
        "g_linear_len": d["geometry"]["linear_length"],
    }


def main():
    plan = json.loads(Path("data/probe/alift_expansion_plan.json").read_text())
    expansion_predictions = plan["a_lift_predictions"]
    expansion_pairs = {
        (p[0], p[1]): expansion_predictions[f"{p[0]}_{p[1]}"]
        for p in plan["predict_win"] + plan["predict_loss"] + plan["predict_tie"]
    }
    all_predictions = {**ORIGINAL_PAIRS, **expansion_pairs}

    rows = []
    nan_pairs = []
    for (s, e), a_lift in all_predictions.items():
        m = load_pair_metrics(s, e)
        if m is None:
            print(f"  missing: {s}->{e}")
            continue
        vals = [m["pb_myl"], m["ge_myl"], m["li_myl"],
                m["pb_off"], m["ge_off"], m["li_off"]]
        if not all(np.isfinite(vals)):
            nan_pairs.append(f"{s}->{e}")
            continue
        rows.append({
            "pair": f"{s}->{e}",
            "a_lift": a_lift,
            "pb_myl": m["pb_myl"],
            "ge_myl": m["ge_myl"],
            "li_myl": m["li_myl"],
            "pb_off": m["pb_off"],
            "ge_off": m["ge_off"],
            "li_off": m["li_off"],
            "pb_margin": m["li_myl"] - m["pb_myl"],
            "ge_margin": m["li_myl"] - m["ge_myl"],
            "pb_vs_ge_margin": m["ge_myl"] - m["pb_myl"],
            "pb_off_gap": m["li_off"] - m["pb_off"],
            "ge_off_gap": m["li_off"] - m["ge_off"],
            "g_pullback_len": m["g_pullback_len"],
            "g_geodesic_len": m["g_geodesic_len"],
            "g_linear_len": m["g_linear_len"],
            "is_original": (s, e) in ORIGINAL_PAIRS,
        })
    if nan_pairs:
        print(f"  skipped {len(nan_pairs)} NaN pairs: {nan_pairs}")
    print(f"\nLoaded {len(rows)} pairs ({sum(1 for r in rows if r['is_original'])} "
          f"original + {sum(1 for r in rows if not r['is_original'])} expansion)")

    a_lifts = np.array([r["a_lift"] for r in rows])
    pb_margins = np.array([r["pb_margin"] for r in rows])
    ge_margins = np.array([r["ge_margin"] for r in rows])
    pb_vs_ge = np.array([r["pb_vs_ge_margin"] for r in rows])
    pb_off_gaps = np.array([r["pb_off_gap"] for r in rows])
    ge_off_gaps = np.array([r["ge_off_gap"] for r in rows])

    print("\n" + "=" * 64)
    print("GEODESIC vs LINEAR  (Riemannian metric vs straight-line in M_h)")
    print("=" * 64)

    # M_y-line margin (positive = geodesic wins)
    ge_mean = ge_margins.mean()
    ge_lo, ge_hi = bootstrap_ci(ge_margins, np.mean)
    w_stat, w_p = stats.wilcoxon(ge_margins, alternative="greater")
    print(f"\nM_y-line margin (linear − geodesic):")
    print(f"  mean = {ge_mean:+.4f}   95% CI [{ge_lo:+.4f}, {ge_hi:+.4f}]")
    print(f"  Wilcoxon one-sided p = {w_p:.4f}")
    print(f"  geodesic wins: {(ge_margins > 0).sum()}/{len(ge_margins)}")

    # Off-M_y E gap (positive = geodesic wins)
    geo_off_mean = ge_off_gaps.mean()
    geo_off_lo, geo_off_hi = bootstrap_ci(ge_off_gaps, np.mean)
    w_stat_o, w_p_o = stats.wilcoxon(ge_off_gaps, alternative="greater")
    print(f"\nOff-M_y E gap (linear − geodesic):")
    print(f"  mean = {geo_off_mean:+.4f}   95% CI [{geo_off_lo:+.4f}, {geo_off_hi:+.4f}]")
    print(f"  Wilcoxon one-sided p = {w_p_o:.4f}")
    print(f"  geodesic wins: {(ge_off_gaps > 0).sum()}/{len(ge_off_gaps)}")

    # A_lift correlation with geodesic margin
    ge_pr, ge_pp = stats.pearsonr(a_lifts, ge_margins)
    ge_pr_lo, ge_pr_hi = bootstrap_ci(
        np.arange(len(a_lifts)),
        lambda idx: stats.pearsonr(a_lifts[idx], ge_margins[idx]).statistic,
    )
    print(f"\nA_lift correlation with geodesic margin:")
    print(f"  Pearson r = {ge_pr:+.3f}   95% CI [{ge_pr_lo:+.3f}, {ge_pr_hi:+.3f}]   p = {ge_pp:.4f}")

    print("\n" + "=" * 64)
    print("PULLBACK vs GEODESIC  (does N-W shortcut match the curved-metric ideal?)")
    print("=" * 64)

    pb_vs_ge_mean = pb_vs_ge.mean()
    pvg_lo, pvg_hi = bootstrap_ci(pb_vs_ge, np.mean)
    w_p_pg = stats.wilcoxon(pb_vs_ge, alternative="two-sided").pvalue
    print(f"\nM_y-line margin (geodesic − pullback):")
    print(f"  mean = {pb_vs_ge_mean:+.4f}   95% CI [{pvg_lo:+.4f}, {pvg_hi:+.4f}]")
    print(f"  Wilcoxon two-sided p = {w_p_pg:.4f}")
    print(f"  pullback closer to M_y-line: {(pb_vs_ge > 0).sum()}/{len(pb_vs_ge)}")
    print(f"  geodesic closer to M_y-line: {(pb_vs_ge < 0).sum()}/{len(pb_vs_ge)}")

    # Correlation between pullback margin and geodesic margin: do the
    # two methods agree on which pairs are easy/hard?
    pb_ge_agreement_r, pb_ge_agreement_p = stats.pearsonr(pb_margins, ge_margins)
    print(f"\nPullback-margin vs geodesic-margin (do they agree on pair difficulty?):")
    print(f"  Pearson r = {pb_ge_agreement_r:+.3f}   p = {pb_ge_agreement_p:.4f}")

    print("\n" + "=" * 64)
    print("SUMMARY (forest plot side-by-side)")
    print("=" * 64)
    print(f"\n{'method':<25s} {'mean':>9s}  {'CI lo':>9s}  {'CI hi':>9s}  {'Wp':>8s}  wins")
    print("-" * 70)
    pb_mean = pb_margins.mean()
    pb_lo, pb_hi = bootstrap_ci(pb_margins, np.mean)
    pb_wp = stats.wilcoxon(pb_margins, alternative="greater").pvalue
    print(f"{'pullback − linear M_y':<25s} {pb_mean:+8.4f}  {pb_lo:+8.4f}  {pb_hi:+8.4f}  {pb_wp:8.4f}  {(pb_margins > 0).sum()}/{len(pb_margins)}")
    print(f"{'geodesic − linear M_y':<25s} {ge_mean:+8.4f}  {ge_lo:+8.4f}  {ge_hi:+8.4f}  {w_p:8.4f}  {(ge_margins > 0).sum()}/{len(ge_margins)}")
    pb_off_mean = pb_off_gaps.mean()
    pb_off_lo, pb_off_hi = bootstrap_ci(pb_off_gaps, np.mean)
    pb_off_wp = stats.wilcoxon(pb_off_gaps, alternative="greater").pvalue
    print(f"{'pullback − linear OFF':<25s} {pb_off_mean:+8.4f}  {pb_off_lo:+8.4f}  {pb_off_hi:+8.4f}  {pb_off_wp:8.4f}  {(pb_off_gaps > 0).sum()}/{len(pb_off_gaps)}")
    print(f"{'geodesic − linear OFF':<25s} {geo_off_mean:+8.4f}  {geo_off_lo:+8.4f}  {geo_off_hi:+8.4f}  {w_p_o:8.4f}  {(ge_off_gaps > 0).sum()}/{len(ge_off_gaps)}")

    out_dir = Path("results/riemannian_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ===== Plot 1: scatter of A_lift vs geodesic margin =====
    fig, ax = plt.subplots(figsize=(8, 6))
    orig_mask = np.array([r["is_original"] for r in rows])
    ax.scatter(a_lifts[orig_mask], ge_margins[orig_mask], s=60,
               c="steelblue", edgecolors="black", linewidths=0.5,
               alpha=0.8, label="original n=10")
    ax.scatter(a_lifts[~orig_mask], ge_margins[~orig_mask], s=60,
               c="firebrick", edgecolors="black", linewidths=0.5,
               alpha=0.7, label="expansion n=30")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.axvline(0, color="k", linewidth=0.5)
    if len(a_lifts) > 2:
        slope, intercept, *_ = stats.linregress(a_lifts, ge_margins)
        x_fit = np.linspace(a_lifts.min(), a_lifts.max(), 100)
        ax.plot(x_fit, slope * x_fit + intercept, "--", color="grey",
                label=f"Pearson r={ge_pr:+.3f} [{ge_pr_lo:+.2f}, {ge_pr_hi:+.2f}]")
    ax.set_xlabel("A_lift (predicted)")
    ax.set_ylabel("Margin = linear M_y-line − geodesic M_y-line")
    ax.set_title(f"A_lift vs Riemannian-geodesic margin, n={len(rows)}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "alift_vs_geodesic_margin.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nsaved {out_dir/'alift_vs_geodesic_margin.png'}")

    # ===== Plot 2: forest plot of all four findings =====
    fig, ax = plt.subplots(figsize=(9, 4.5))
    labels = [
        "pullback − linear  (M_y-line)",
        "geodesic − linear  (M_y-line)",
        "pullback − linear  (off-M_y E)",
        "geodesic − linear  (off-M_y E)",
    ]
    means = [pb_mean, ge_mean, pb_off_mean, geo_off_mean]
    los = [pb_lo, ge_lo, pb_off_lo, geo_off_lo]
    his = [pb_hi, ge_hi, pb_off_hi, geo_off_hi]
    wps = [pb_wp, w_p, pb_off_wp, w_p_o]
    ys = np.arange(len(labels))
    for i, (m, lo, hi, p) in enumerate(zip(means, los, his, wps)):
        is_sig = p < 0.05
        c = "C2" if is_sig else "C7"
        ax.errorbar(m, ys[i], xerr=[[m - lo], [hi - m]], fmt="o", color=c,
                    capsize=4, ms=8, markeredgecolor="black", markeredgewidth=0.5)
        ax.text(hi + 0.001, ys[i], f"p={p:.3f}", va="center", fontsize=8)
    ax.axvline(0, color="k", linewidth=0.5)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Mean gap (positive = the non-linear method wins)")
    ax.set_title(f"Behavioral gains vs linear interpolation, n={len(rows)}\n"
                 f"(filled = p<0.05 one-sided Wilcoxon)")
    plt.tight_layout()
    plt.savefig(out_dir / "forest_plot_geo_vs_linear.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {out_dir/'forest_plot_geo_vs_linear.png'}")

    # ===== Plot 3: pullback margin vs geodesic margin =====
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(pb_margins, ge_margins, s=60, c="purple",
               edgecolors="black", linewidths=0.5, alpha=0.7)
    lim_lo = min(pb_margins.min(), ge_margins.min()) - 0.05
    lim_hi = max(pb_margins.max(), ge_margins.max()) + 0.05
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "--", color="grey",
            label="y = x  (perfect agreement)")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.axvline(0, color="k", linewidth=0.5)
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_xlabel("Pullback margin (linear − pullback)")
    ax.set_ylabel("Geodesic margin (linear − geodesic)")
    ax.set_title(f"Do N-W pullback and Riemannian geodesic agree?\n"
                 f"Pearson r={pb_ge_agreement_r:+.3f} (p={pb_ge_agreement_p:.3f})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "pullback_vs_geodesic_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {out_dir/'pullback_vs_geodesic_scatter.png'}")

    # ===== Plot 4: path-length comparison (geometry, no LLM) =====
    g_pb = np.array([r["g_pullback_len"] for r in rows])
    g_ge = np.array([r["g_geodesic_len"] for r in rows])
    g_li = np.array([r["g_linear_len"] for r in rows])
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(rows))
    order = np.argsort(g_li)
    ax.plot(x, g_li[order], "o-", color="C0", label="linear", markersize=4)
    ax.plot(x, g_ge[order], "s-", color="C2", label="geodesic", markersize=4)
    ax.plot(x, g_pb[order], "^-", color="C3", label="pullback", markersize=4)
    ax.set_xlabel("Pair index (sorted by linear length)")
    ax.set_ylabel("Path length under G_E")
    ax.set_title("Per-pair geometric path lengths under density-aware metric G_E")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "ge_path_lengths.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {out_dir/'ge_path_lengths.png'}")

    summary = {
        "n_total": len(rows),
        "n_original": int(orig_mask.sum()),
        "n_expansion": int((~orig_mask).sum()),
        "geodesic_vs_linear": {
            "my_line_margin_mean": float(ge_mean),
            "my_line_margin_ci": [float(ge_lo), float(ge_hi)],
            "my_line_wilcoxon_p_one_sided": float(w_p),
            "my_line_wins": int((ge_margins > 0).sum()),
            "off_my_e_gap_mean": float(geo_off_mean),
            "off_my_e_gap_ci": [float(geo_off_lo), float(geo_off_hi)],
            "off_my_e_wilcoxon_p_one_sided": float(w_p_o),
            "off_my_e_wins": int((ge_off_gaps > 0).sum()),
            "a_lift_pearson_r": float(ge_pr),
            "a_lift_pearson_p": float(ge_pp),
            "a_lift_pearson_ci": [float(ge_pr_lo), float(ge_pr_hi)],
        },
        "pullback_vs_geodesic": {
            "my_line_margin_mean": float(pb_vs_ge_mean),
            "my_line_margin_ci": [float(pvg_lo), float(pvg_hi)],
            "wilcoxon_p_two_sided": float(w_p_pg),
            "pullback_closer_to_my_line": int((pb_vs_ge > 0).sum()),
            "geodesic_closer_to_my_line": int((pb_vs_ge < 0).sum()),
            "agreement_pearson_r": float(pb_ge_agreement_r),
            "agreement_pearson_p": float(pb_ge_agreement_p),
        },
        "pullback_vs_linear_for_reference": {
            "my_line_margin_mean": float(pb_mean),
            "my_line_margin_ci": [float(pb_lo), float(pb_hi)],
            "my_line_wilcoxon_p_one_sided": float(pb_wp),
            "off_my_e_gap_mean": float(pb_off_mean),
            "off_my_e_gap_ci": [float(pb_off_lo), float(pb_off_hi)],
            "off_my_e_wilcoxon_p_one_sided": float(pb_off_wp),
        },
        "per_pair": rows,
    }
    (out_dir / "_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"saved {out_dir/'_summary.json'}")


if __name__ == "__main__":
    main()
