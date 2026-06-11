"""Analyze the n=40 A_lift validation results.

Loads all 40 pairs (10 original + 30 expansion), computes Pearson r,
Spearman r, 95% CIs via bootstrap, and reports per-prediction-bucket
accuracy. Compares to the n=10 baseline.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


# Original 10 pairs (with the A_lift values computed in earlier work)
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
        # Try alternate ordering since some labels appear either way
        alt = Path(f"results/pullback/{e}_{s}.json")
        if alt.exists():
            path = alt
        else:
            return None
    d = json.loads(path.read_text())
    return {
        "pb_myl": d["trajectories"]["pullback"]["my_geodesic_distance"],
        "ge_myl": d["trajectories"]["geodesic"]["my_geodesic_distance"],
        "li_myl": d["trajectories"]["linear"]["my_geodesic_distance"],
        "pb_off": d["trajectories"]["pullback"]["off_manifold_energy"],
        "li_off": d["trajectories"]["linear"]["off_manifold_energy"],
    }


def main():
    plan = json.loads(Path("data/probe/alift_expansion_plan.json").read_text())
    expansion_predictions = plan["a_lift_predictions"]
    expansion_pairs = {(p[0], p[1]): expansion_predictions[f"{p[0]}_{p[1]}"]
                       for p in plan["predict_win"] + plan["predict_loss"]
                                + plan["predict_tie"]}

    all_predictions = {**ORIGINAL_PAIRS, **expansion_pairs}
    rows = []
    nan_pairs = []
    for (s, e), a_lift in all_predictions.items():
        m = load_pair_metrics(s, e)
        if m is None:
            print(f"  missing: {s}->{e}")
            continue
        # Skip pairs with NaN metrics (judge failed; credit ran out)
        if not all(np.isfinite([m["pb_myl"], m["li_myl"], m["pb_off"], m["li_off"]])):
            nan_pairs.append(f"{s}->{e}")
            continue
        margin = m["li_myl"] - m["pb_myl"]
        off_gap = m["li_off"] - m["pb_off"]
        rows.append({
            "pair": f"{s}->{e}",
            "a_lift": a_lift,
            "margin": margin,
            "pb_myl": m["pb_myl"],
            "li_myl": m["li_myl"],
            "off_gap": off_gap,
            "is_original": (s, e) in ORIGINAL_PAIRS,
        })
    if nan_pairs:
        print(f"  skipped {len(nan_pairs)} NaN pairs (judge failed): {nan_pairs}")
    print(f"\nLoaded {len(rows)} pairs total: "
          f"{sum(1 for r in rows if r['is_original'])} original, "
          f"{sum(1 for r in rows if not r['is_original'])} expansion")

    xs = np.array([r["a_lift"] for r in rows])
    ys = np.array([r["margin"] for r in rows])
    offs = np.array([r["off_gap"] for r in rows])

    # A_lift correlation
    p_r, p_p = stats.pearsonr(xs, ys)
    s_r, s_p = stats.spearmanr(xs, ys)
    rng = np.random.default_rng(42)
    n_boot = 10000
    p_bs = []
    s_bs = []
    for _ in range(n_boot):
        idx = rng.choice(len(xs), size=len(xs), replace=True)
        xb, yb = xs[idx], ys[idx]
        # need at least 3 distinct x values to compute a meaningful correlation
        if len(np.unique(xb)) < 3:
            continue
        pr = stats.pearsonr(xb, yb).statistic
        sr = stats.spearmanr(xb, yb).statistic
        if np.isfinite(pr):
            p_bs.append(pr)
        if np.isfinite(sr):
            s_bs.append(sr)
    p_bs = np.array(p_bs)
    s_bs = np.array(s_bs)
    p_lo, p_hi = np.percentile(p_bs, [2.5, 97.5])
    s_lo, s_hi = np.percentile(s_bs, [2.5, 97.5])

    print()
    print("==== A_lift correlation ====")
    print(f"  Pearson r  = {p_r:+.3f}  (p={p_p:.4f})  95% CI: [{p_lo:+.3f}, {p_hi:+.3f}]")
    print(f"  Spearman r = {s_r:+.3f}  (p={s_p:.4f})  95% CI: [{s_lo:+.3f}, {s_hi:+.3f}]")

    # Off-M_y E gap
    off_mean = offs.mean()
    bs_off = np.array([np.random.default_rng(42 + b)
                        .choice(offs, size=len(offs), replace=True).mean()
                        for b in range(n_boot)])
    off_lo, off_hi = np.percentile(bs_off, [2.5, 97.5])
    w_stat, w_p = stats.wilcoxon(offs, alternative="greater")
    print()
    print("==== off-M_y E gap (linear - pullback) ====")
    print(f"  Mean: {off_mean:+.4f}  95% CI: [{off_lo:+.4f}, {off_hi:+.4f}]")
    print(f"  Wilcoxon p (one-sided): {w_p:.4f}")
    print(f"  Pairs favoring pullback: {(offs > 0).sum()}/{len(offs)}")

    # Per-bucket accuracy of A_lift directional prediction
    def bucket(al):
        if al > 0.1: return "WIN"
        if al < -0.05: return "LOSS"
        return "TIE"
    def actual_bucket(m):
        if m > 0.05: return "WIN"
        if m < -0.05: return "LOSS"
        return "TIE"

    print()
    print("==== Directional accuracy of A_lift predictions ====")
    print(f"  {'predicted':>10s}  {'WIN':>5s}  {'TIE':>5s}  {'LOSS':>5s}  total")
    by_pred = {"WIN": [0, 0, 0], "TIE": [0, 0, 0], "LOSS": [0, 0, 0]}
    for r in rows:
        p = bucket(r["a_lift"])
        a = actual_bucket(r["margin"])
        idx = {"WIN": 0, "TIE": 1, "LOSS": 2}[a]
        by_pred[p][idx] += 1
    for pred in ("WIN", "TIE", "LOSS"):
        counts = by_pred[pred]
        total = sum(counts)
        print(f"  {pred:>10s}  {counts[0]:>5d}  {counts[1]:>5d}  {counts[2]:>5d}  {total}")

    # Save scatter plot of A_lift vs margin
    out_dir = Path("results/alift_expansion")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    orig_mask = np.array([r["is_original"] for r in rows])
    ax.scatter(xs[orig_mask], ys[orig_mask], s=60, label="original n=10",
               c="steelblue", edgecolors="black", linewidths=0.5, alpha=0.8)
    ax.scatter(xs[~orig_mask], ys[~orig_mask], s=60, label="expansion n=30",
               c="firebrick", edgecolors="black", linewidths=0.5, alpha=0.7)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.axvline(0, color="k", linewidth=0.5)
    # Fit line
    if len(xs) > 2:
        slope, intercept, _, _, _ = stats.linregress(xs, ys)
        x_fit = np.linspace(xs.min(), xs.max(), 100)
        ax.plot(x_fit, slope * x_fit + intercept, "--", color="grey",
                label=f"Pearson r={p_r:+.3f} [{p_lo:+.2f}, {p_hi:+.2f}]")
    ax.set_xlabel("A_lift (predicted)")
    ax.set_ylabel("Margin = linear M_y-line − pullback M_y-line")
    ax.set_title(f"A_lift predictor validation, n={len(rows)}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "alift_vs_margin.png", dpi=150, bbox_inches="tight")
    print(f"\nsaved {out_dir/'alift_vs_margin.png'}")

    # Save summary
    summary = {
        "n_total": len(rows),
        "n_original": int(orig_mask.sum()),
        "n_expansion": int((~orig_mask).sum()),
        "a_lift_pearson_r": float(p_r),
        "a_lift_pearson_p": float(p_p),
        "a_lift_pearson_ci": [float(p_lo), float(p_hi)],
        "a_lift_spearman_r": float(s_r),
        "a_lift_spearman_p": float(s_p),
        "a_lift_spearman_ci": [float(s_lo), float(s_hi)],
        "off_my_e_gap_mean": float(off_mean),
        "off_my_e_gap_ci": [float(off_lo), float(off_hi)],
        "off_my_e_wilcoxon_p": float(w_p),
        "off_my_e_pullback_wins": int((offs > 0).sum()),
        "per_pair": rows,
    }
    (out_dir / "_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"saved {out_dir/'_summary.json'}")


if __name__ == "__main__":
    main()
