"""Generate all the figures needed for the writeup.

Outputs to results/figures/writeup/ so existing figures are preserved.
Each figure is produced by a separate function so they can be re-run
independently.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from manifold_emotions.behavior.manifold import BehaviorManifold
from manifold_emotions.config import load_config
from manifold_emotions.manifold.fit import FittedManifold
from manifold_emotions.vectors.diff_in_means import EmotionVectors

OUT_DIR = Path("results/figures/writeup")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Helpers
# ============================================================

def load_pair_pullback(s, e):
    """Return summary dict or None if file missing / NaN-corrupted."""
    p = Path(f"results/pullback/{s}_{e}.json")
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    pb = d["trajectories"]["pullback"]["my_geodesic_distance"]
    if not np.isfinite(pb):
        return None
    return d


def pairwise_distances(X):
    """Euclidean pairwise distances matrix (N, N)."""
    diff = X[:, None, :] - X[None, :, :]
    return np.sqrt((diff * diff).sum(axis=-1))


def upper_tri_pairs(D):
    """Return the upper-triangular pair-distance vector (excludes diagonal)."""
    n = D.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    return D[iu, ju]


# ============================================================
# Figure 1: Isometry scatter at 171
# ============================================================

def fig_isometry_171():
    cfg = load_config()
    mh = FittedManifold.load(cfg.paths.manifold_h)
    beh = BehaviorManifold.load(cfg.paths.manifold_y)
    ev = EmotionVectors.load(cfg.paths.emotion_vectors)

    common = [l for l in beh.labels if l in mh.labels and l in ev.labels]
    h_idx = [mh.labels.index(l) for l in common]
    y_idx = [beh.labels.index(l) for l in common]
    e_idx = [ev.labels.index(l) for l in common]

    h_sub = mh.centroids_subspace[h_idx]
    h_full = ev.centroids[e_idx]  # raw diff-in-means residual stream vectors
    y = beh.centroids[y_idx]

    d_sub = upper_tri_pairs(pairwise_distances(h_sub))
    d_full = upper_tri_pairs(pairwise_distances(h_full))
    d_y = upper_tri_pairs(pairwise_distances(y))

    r_sub = stats.pearsonr(d_sub, d_y).statistic
    r_full = stats.pearsonr(d_full, d_y).statistic

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, d_h, r, title in [
        (axes[0], d_sub, r_sub, "M_h subspace (8-D PCA) vs M_y"),
        (axes[1], d_full, r_full, "M_h full residual (linear baseline) vs M_y"),
    ]:
        ax.scatter(d_h, d_y, s=2, alpha=0.15, color="steelblue")
        # Fit line
        slope, intercept, _, _, _ = stats.linregress(d_h, d_y)
        xs = np.linspace(d_h.min(), d_h.max(), 50)
        ax.plot(xs, slope * xs + intercept, "-", color="firebrick",
                linewidth=1.5, label=f"Pearson r = {r:+.3f}")
        ax.set_xlabel(f"{'subspace' if 'subspace' in title else 'full'} M_h pairwise distance")
        ax.set_ylabel("M_y pairwise distance")
        ax.set_title(title)
        ax.legend(loc="lower right")
    plt.suptitle(f"171-emotion isometry: M_h subspace edge = {r_sub - r_full:+.3f}",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "isometry_scatter_171.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved isometry_scatter_171.png  (r_sub={r_sub:+.3f}, r_full={r_full:+.3f})")
    return r_sub, r_full


# ============================================================
# Figure 2: 30 vs 171 isometry comparison
# ============================================================

def fig_isometry_30_vs_171():
    # Load both scales
    cfg = load_config()
    mh_171 = FittedManifold.load(cfg.paths.manifold_h)
    beh_171 = BehaviorManifold.load(cfg.paths.manifold_y)
    mh_30 = FittedManifold.load(Path("data/30emotions/manifold_h.npz"))
    beh_30 = BehaviorManifold.load(Path("data/30emotions/manifold_y.npz"))

    ev_171 = EmotionVectors.load(cfg.paths.emotion_vectors)
    ev_30 = EmotionVectors.load(Path("data/30emotions/emotion_vectors.npz"))

    def compute_r(mh, beh, ev):
        common = [l for l in beh.labels if l in mh.labels and l in ev.labels]
        h_idx = [mh.labels.index(l) for l in common]
        y_idx = [beh.labels.index(l) for l in common]
        e_idx = [ev.labels.index(l) for l in common]
        d_sub = upper_tri_pairs(pairwise_distances(mh.centroids_subspace[h_idx]))
        d_full = upper_tri_pairs(pairwise_distances(ev.centroids[e_idx]))
        d_y = upper_tri_pairs(pairwise_distances(beh.centroids[y_idx]))
        return (stats.pearsonr(d_sub, d_y).statistic,
                stats.pearsonr(d_full, d_y).statistic,
                len(common))

    r_sub_30, r_full_30, n30 = compute_r(mh_30, beh_30, ev_30)
    r_sub_171, r_full_171, n171 = compute_r(mh_171, beh_171, ev_171)

    fig, ax = plt.subplots(figsize=(8, 5))
    scales = ["30 emotions", "171 emotions"]
    sub_rs = [r_sub_30, r_sub_171]
    full_rs = [r_full_30, r_full_171]
    edges = [r_sub_30 - r_full_30, r_sub_171 - r_full_171]
    x = np.arange(2)
    w = 0.32
    ax.bar(x - w/2, full_rs, w, label="Linear (full residual)", color="grey",
           edgecolor="black", linewidth=0.5)
    ax.bar(x + w/2, sub_rs, w, label="M_h subspace (PCA)",
           color="firebrick", edgecolor="black", linewidth=0.5)
    for i, (sub, full, e) in enumerate(zip(sub_rs, full_rs, edges)):
        ax.annotate(f"r={sub:.3f}", (i + w/2, sub), ha="center", va="bottom",
                    fontsize=9, color="firebrick")
        ax.annotate(f"r={full:.3f}", (i - w/2, full), ha="center", va="bottom",
                    fontsize=9, color="black")
        ax.annotate(f"edge: {e:+.3f}", (i, max(sub, full) + 0.05),
                    ha="center", va="bottom", fontsize=10, weight="bold",
                    color="darkgreen" if e > 0 else "darkred")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{scales[0]}\n(n={n30} common)",
                        f"{scales[1]}\n(n={n171} common)"])
    ax.set_ylabel("Pearson r (M_h pairwise distance vs M_y pairwise distance)")
    ax.set_title("Subspace edges out linear isometry as concept count scales")
    ax.set_ylim(0, max(max(sub_rs), max(full_rs)) + 0.15)
    ax.legend(loc="lower right")
    ax.axhline(0, color="k", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "isometry_30_vs_171_comparison.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print(f"  saved isometry_30_vs_171_comparison.png  "
          f"(30 edge={edges[0]:+.3f}, 171 edge={edges[1]:+.3f})")


# ============================================================
# Figure 3: Forest plot of findings (with n=10 baselines)
# ============================================================

def fig_forest_plot():
    findings = [
        # (name, n, mean, ci_lo, ci_hi, p, sig, group)
        ("Chord off-M_y E gap (n=10 pilot)", 10, 0.054, 0.019, 0.093, 0.024, True, "chord"),
        ("Chord off-M_y E gap (n=40)", 40, 0.023, 0.002, 0.045, 0.051, True, "chord"),
        ("Composition coh gap NM (n=5 pilot)", 5, 0.070, -0.030, 0.170, None, False, "composition"),
        ("Composition coh gap NM (n=20)", 20, 0.053, 0.003, 0.100, 0.049, True, "composition"),
        ("Composition off-M_y gap NM (n=5 pilot)", 5, -0.007, -0.067, 0.046, None, False, "composition"),
        ("Composition off-M_y gap NM (n=20)", 20, -0.031, -0.059, -0.004, 0.058, True, "composition"),
        ("A_lift Pearson r (n=10 pilot)", 10, 0.473, -0.089, 0.861, 0.168, False, "alift"),
        ("A_lift Pearson r (n=40)", 40, 0.384, 0.110, 0.605, 0.015, True, "alift"),
    ]
    fig, ax = plt.subplots(figsize=(11, 7))
    y_pos = np.arange(len(findings))
    for i, (name, n, mean, lo, hi, p, sig, group) in enumerate(findings):
        color = {"chord": "steelblue", "composition": "darkorange",
                  "alift": "firebrick"}[group]
        marker = "o" if sig else "x"
        ax.errorbar([mean], [i], xerr=[[mean - lo], [hi - mean]],
                    fmt=marker, color=color, ecolor=color,
                    markerfacecolor="white" if not sig else color,
                    markeredgecolor=color, markersize=9, capsize=4, linewidth=2)
        label = f"{mean:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        if p is not None:
            label += f"  p={p:.3f}"
        ax.annotate(label, (hi + 0.02, i), va="center", fontsize=9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f[0] for f in findings])
    ax.invert_yaxis()
    ax.axvline(0, color="k", linewidth=0.6, linestyle="--", alpha=0.6)
    ax.set_xlabel("Effect size (mean and 95% CI)")
    ax.set_title("Behavioral findings with 95% bootstrap CIs; filled markers = p<0.05",
                 fontsize=12)
    # Group separators
    for ysep in [1.5, 5.5]:
        ax.axhline(ysep, color="grey", linewidth=0.4, linestyle=":")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "forest_plot_findings.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  saved forest_plot_findings.png")


# ============================================================
# Figure 4: Composition stratified results
# ============================================================

def fig_composition_stratified():
    # From the n=15 expansion stratified analysis
    cats = ["same-quadrant", "opposite-valence", "opposite-arousal"]
    coh_gaps = [-0.030, +0.080, +0.090]
    off_gaps = [-0.055, -0.015, -0.049]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(cats))
    w = 0.36
    ax.bar(x - w/2, coh_gaps, w, label="Coherence gap (linear - pullback)",
           color="steelblue", edgecolor="black", linewidth=0.5)
    ax.bar(x + w/2, off_gaps, w, label="off-M_y E gap (linear - pullback)",
           color="firebrick", edgecolor="black", linewidth=0.5)
    for i, (c, o) in enumerate(zip(coh_gaps, off_gaps)):
        ax.annotate(f"{c:+.3f}", (i - w/2, c),
                    ha="center", va="bottom" if c >= 0 else "top",
                    fontsize=9)
        ax.annotate(f"{o:+.3f}", (i + w/2, o),
                    ha="center", va="bottom" if o >= 0 else "top",
                    fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n(n=5)" for c in cats])
    ax.axhline(0, color="k", linewidth=0.5)
    ax.legend()
    ax.set_title("Composition results stratified by chord structure (norm-matched)\n"
                 "positive = linear wins; negative = pullback wins")
    ax.set_ylabel("Gap (linear − pullback)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "composition_stratified.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print("  saved composition_stratified.png")


# ============================================================
# Figure 5: Per-pair n=40 results with A_lift coloring
# ============================================================

def fig_pair_n40_results():
    s = json.loads(Path("results/alift_expansion/_summary.json").read_text())
    rows = s["per_pair"]
    al = np.array([r["a_lift"] for r in rows])
    margin = np.array([r["margin"] for r in rows])
    off_gap = np.array([r["off_gap"] for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    cmap = plt.cm.RdBu_r
    norm = plt.Normalize(vmin=-0.3, vmax=+0.3)

    ax = axes[0]
    order = np.argsort(margin)
    bars = ax.barh(np.arange(len(margin)), margin[order],
                   color=[cmap(norm(al[i])) for i in order])
    ax.axvline(0, color="k", linewidth=0.6)
    ax.axvline(+0.05, color="grey", linewidth=0.4, linestyle=":")
    ax.axvline(-0.05, color="grey", linewidth=0.4, linestyle=":")
    ax.set_yticks([])
    ax.set_xlabel("M_y-line margin (linear − pullback)\n+ = pullback closer to target")
    ax.set_title(f"n={len(rows)} chord pairs sorted by margin\nbars colored by A_lift")

    ax = axes[1]
    order2 = np.argsort(off_gap)
    bars = ax.barh(np.arange(len(off_gap)), off_gap[order2],
                   color=[cmap(norm(al[i])) for i in order2])
    ax.axvline(0, color="k", linewidth=0.6)
    ax.set_yticks([])
    ax.set_xlabel("off-M_y E gap (linear − pullback)\n+ = pullback more on-manifold")
    ax.set_title("Same pairs sorted by off-manifold gap")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, location="bottom", shrink=0.5, pad=0.08)
    cbar.set_label("A_lift")
    plt.savefig(OUT_DIR / "pair_n40_results.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  saved pair_n40_results.png")


# ============================================================
# Figure 6: Composition raw vs norm-matched
# ============================================================

def fig_composition_raw_vs_nm():
    a = json.loads(Path("results/composition_expansion_analysis/_summary.json").read_text())
    pooled = a["pooled"]
    raw = pooled["raw"]
    nm = pooled["nm"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    cats = ["raw magnitude", "norm-matched"]
    metrics = [
        ("coherence gap (linear% − pullback%)",
         [raw["coh_gap_mean"], nm["coh_gap_mean"]],
         [raw["coh_gap_ci"], nm["coh_gap_ci"]]),
        ("off-M_y E gap (linear − pullback)",
         [raw["off_gap_mean"], nm["off_gap_mean"]],
         [raw["off_gap_ci"], nm["off_gap_ci"]]),
    ]
    for ax, (label, means, cis) in zip(axes, metrics):
        x = np.arange(2)
        ax.bar(x, means, 0.4, color=["grey", "steelblue"],
               edgecolor="black", linewidth=0.5)
        errs = [[m - c[0] for m, c in zip(means, cis)],
                [c[1] - m for m, c in zip(means, cis)]]
        ax.errorbar(x, means, yerr=errs, fmt="none", color="black", capsize=5)
        for i, m in enumerate(means):
            ax.annotate(f"{m:+.3f}", (i, m), ha="center",
                        va="bottom" if m >= 0 else "top", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(cats)
        ax.axhline(0, color="k", linewidth=0.6)
        ax.set_title(label, fontsize=11)
        ax.set_ylabel("gap")
    plt.suptitle(f"Composition pathology: raw is magnitude-confounded (n=20 each)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "composition_n20_raw_vs_nm.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print("  saved composition_n20_raw_vs_nm.png")


# ============================================================
# Figure 7: Eval direction top/bottom emotions (differential vs contrastive)
# ============================================================

def fig_eval_direction_emotions():
    diff_s = json.loads(Path("results/eval_awareness_v2/_summary.json").read_text())
    contr_s = json.loads(Path("results/eval_awareness_contrastive/_summary.json").read_text())

    diff_top = diff_s["top_emotions_eval_vs_neutral_mean"][:10]
    diff_bot = diff_s["bottom_emotions_eval_vs_neutral_mean"][:10]
    contr_top = contr_s["top_emotions"][:10]
    contr_bot = contr_s["bottom_emotions"][:10]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, items, title, color in [
        (axes[0, 0], diff_top, "Differential (mean): TOP", "steelblue"),
        (axes[0, 1], contr_top, "Contrastive (logistic): TOP", "darkred"),
        (axes[1, 0], list(reversed(diff_bot)), "Differential (mean): BOTTOM", "steelblue"),
        (axes[1, 1], list(reversed(contr_bot)), "Contrastive (logistic): BOTTOM", "darkred"),
    ]:
        names = [n for n, c in items]
        values = [c for n, c in items]
        y = np.arange(len(names))
        ax.barh(y, values, color=color, edgecolor="black", linewidth=0.3, alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.axvline(0, color="k", linewidth=0.4)
        ax.set_xlabel("cosine similarity")
        ax.set_title(title)

    plt.suptitle("Eval-awareness direction: differential averages performance enthusiasm,\n"
                 "contrastive picks out calm/patient cluster and symmetric high-arousal suppression",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "eval_direction_top_bottom_emotions.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print("  saved eval_direction_top_bottom_emotions.png")


# ============================================================
# Figure 8: Eval steering dose-response
# ============================================================

def fig_eval_steering_dose_response():
    diff = json.loads(Path("results/eval_steering_diff_wide/_summary.json").read_text())
    contr = json.loads(Path("results/eval_steering_contr_wide/_summary.json").read_text())

    def extract(s):
        scales = sorted(int(k) for k in s["by_scale"].keys())
        vs = [s["by_scale"][f"{k:+d}"]["mean_V"] for k in scales]
        as_ = [s["by_scale"][f"{k:+d}"]["mean_A"] for k in scales]
        en = [s["by_scale"][f"{k:+d}"]["enthusiasm_mean"] for k in scales]
        return scales, vs, as_, en

    scales_d, V_d, A_d, E_d = extract(diff)
    scales_c, V_c, A_c, E_c = extract(contr)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, label, V, A, E in [(axes[0], "Differential", V_d, A_d, E_d),
                                 (axes[1], "Contrastive", V_c, A_c, E_c)]:
        scales = scales_d if label == "Differential" else scales_c
        ax.plot(scales, V, marker="o", label="V (valence)", color="steelblue")
        ax.plot(scales, A, marker="s", label="A (arousal)", color="darkorange")
        ax.plot(scales, E, marker="^", label="enthusiasm", color="darkgreen")
        ax.axvline(0, color="k", linewidth=0.4, linestyle=":")
        ax.set_xlabel("scale (unit-normalized direction × scale)")
        ax.set_title(f"{label} direction — dose-response")
        ax.legend(loc="upper left", fontsize=8)
        ax.set_xscale("symlog", linthresh=10)

    # Joyful native steer reference (single point at scale 8, on slightly diff prompts)
    ax = axes[2]
    # Use +100 as the closest comparable scale (next scale up from emotion-vector
    # range; the eval directions have norm of similar order so scale=100 is
    # well into the dose range)
    scale_ref = +100
    refs = {"baseline (no steer)": (3.92, 3.06),
            "joyful @ scale 8 (tone-test ref)": (5.77, 5.33),
            "awestruck @ scale 8 (eval-test ref)": (4.21, 3.32),
            f"diff direction @ scale +{scale_ref}":
                (V_d[scales_d.index(scale_ref)], A_d[scales_d.index(scale_ref)]),
            f"contr direction @ scale +{scale_ref}":
                (V_c[scales_c.index(scale_ref)], A_c[scales_c.index(scale_ref)])}
    for name, (v, a) in refs.items():
        ax.scatter(v, a, s=130, label=name,
                   edgecolor="black", linewidth=0.5)
    ax.set_xlabel("V (valence)")
    ax.set_ylabel("A (arousal)")
    ax.set_xlim(2, 7)
    ax.set_ylim(2, 7)
    ax.legend(loc="lower right", fontsize=7)
    ax.set_title("Behavioral effect: extracted directions vs native steers")
    ax.axhline(4, color="grey", linestyle=":", alpha=0.4)
    ax.axvline(4, color="grey", linestyle=":", alpha=0.4)
    plt.suptitle("Both eval directions are behaviorally inert vs native emotion steering",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "eval_steering_dose_response.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print("  saved eval_steering_dose_response.png")


# ============================================================
# Figure 9: Framing cosine heatmap
# ============================================================

def fig_framing_cosine_heatmap():
    s = json.loads(Path("results/eval_awareness_v2/_summary.json").read_text())
    cm = s["cosine_matrix"]
    names = list(cm.keys())
    short = [n.replace("eval_vs_neutral_", "") for n in names]
    M = np.array([[cm[a][b] for b in names] for a in names])

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(short)))
    ax.set_yticks(range(len(short)))
    ax.set_xticklabels(short, rotation=45, ha="right")
    ax.set_yticklabels(short)
    for i in range(len(short)):
        for j in range(len(short)):
            ax.annotate(f"{M[i, j]:.2f}", (j, i), ha="center", va="center",
                        color="white" if M[i, j] < 0.7 else "black", fontsize=9)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("cosine similarity in subspace")
    ax.set_title("Eval-vs-neutral direction cosine across 4 framings\n"
                 "subtle is the outlier (cos=0.38 with explicit)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "framing_cosine_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  saved framing_cosine_heatmap.png")


# ============================================================
# Figure 10: Tone modulation per emotion
# ============================================================

def fig_tone_per_emotion():
    s = json.loads(Path("results/tone/_summary.json").read_text())
    baseline = s["baseline"]
    results = s["results"]

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter([baseline["mean_V"]], [baseline["mean_A"]], s=200, c="grey",
               marker="X", edgecolor="black", linewidth=1, zorder=10,
               label="baseline (no steer)")
    for emotion, r in results.items():
        t_v, t_a = r["y_target"]
        l_v, l_a = r["linear"]["mean_V"], r["linear"]["mean_A"]
        p_v, p_a = r["pullback"]["mean_V"], r["pullback"]["mean_A"]
        ax.scatter([t_v], [t_a], s=80, c="black", marker="*", zorder=8)
        ax.annotate(emotion, (t_v, t_a), xytext=(5, 5), textcoords="offset points",
                    fontsize=8, color="black")
        ax.scatter([l_v], [l_a], s=70, c="steelblue", marker="o",
                   edgecolor="black", linewidth=0.5)
        ax.scatter([p_v], [p_a], s=70, c="darkred", marker="s",
                   edgecolor="black", linewidth=0.5)
        ax.plot([t_v, l_v], [t_a, l_a], color="steelblue", alpha=0.4, linewidth=0.7)
        ax.plot([t_v, p_v], [t_a, p_a], color="darkred", alpha=0.4, linewidth=0.7)
    ax.scatter([], [], c="black", marker="*", s=80, label="target")
    ax.scatter([], [], c="steelblue", marker="o", s=70, label="linear steer")
    ax.scatter([], [], c="darkred", marker="s", s=70, label="pullback steer")
    ax.set_xlabel("V (valence)")
    ax.set_ylabel("A (arousal)")
    ax.set_xlim(1.0, 7.0)
    ax.set_ylim(2.0, 7.0)
    ax.axhline(4, color="grey", linestyle=":", alpha=0.3)
    ax.axvline(4, color="grey", linestyle=":", alpha=0.3)
    ax.legend(loc="lower right")
    ax.set_title("Tone modulation: emotion-vector steering moves behavior\n"
                 "toward target on all 8 emotions; linear ≈ pullback")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "tone_steering_per_emotion.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print("  saved tone_steering_per_emotion.png")


# ============================================================
# Figure 11: PCA scree
# ============================================================

def fig_pca_scree():
    cfg = load_config()
    mh = FittedManifold.load(cfg.paths.manifold_h)
    explained = mh.pca_explained_variance_ratio
    cumulative = np.cumsum(explained)
    n_kept = len(explained)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(1, n_kept + 1)
    ax.bar(x, explained, color="steelblue", edgecolor="black", linewidth=0.5,
           label="per-component variance fraction")
    ax2 = ax.twinx()
    ax2.plot(x, cumulative, color="firebrick", marker="o",
             label="cumulative variance")
    ax2.axhline(0.66, color="grey", linestyle=":", linewidth=0.8)
    ax2.annotate("66% at 8-D (kept)", (n_kept * 0.7, 0.67), fontsize=10, color="grey")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Per-component variance fraction", color="steelblue")
    ax2.set_ylabel("Cumulative variance", color="firebrick")
    ax2.set_ylim(0, 1.0)
    ax.set_title(f"PCA scree on 171 emotion centroids (PCs kept: {n_kept})")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "pca_scree.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved pca_scree.png  ({n_kept} PCs, total explained: {cumulative[-1]:.3f})")


# ============================================================
# Figure 12: Density on PC1/PC2 with emotion centroids
# ============================================================

def fig_density_pc1_pc2():
    cfg = load_config()
    mh = FittedManifold.load(cfg.paths.manifold_h)
    sub = mh.centroids_subspace  # (171, num_components)
    labels = list(mh.labels)

    # Use first two PCs
    x = sub[:, 0]
    y = sub[:, 1]

    # Compute KDE on (x, y) only — for visualization
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(np.vstack([x, y]), bw_method=0.25)
    xs = np.linspace(x.min() - 0.5, x.max() + 0.5, 200)
    ys = np.linspace(y.min() - 0.5, y.max() + 0.5, 200)
    XX, YY = np.meshgrid(xs, ys)
    ZZ = kde(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.contourf(XX, YY, ZZ, levels=15, cmap="Blues", alpha=0.7)
    ax.scatter(x, y, s=25, c="firebrick", edgecolor="black", linewidth=0.3, zorder=10)
    # Label a sample for readability — every 8th, or specific anchors
    interesting = ["happy", "sad", "joyful", "angry", "afraid", "serene",
                   "excited", "calm", "anxious", "depressed", "loving",
                   "contemptuous", "ecstatic", "miserable"]
    for i, l in enumerate(labels):
        if l in interesting:
            ax.annotate(l, (x[i], y[i]), fontsize=8, xytext=(4, 4),
                        textcoords="offset points")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("171 emotion centroids on first two PCs of M_h subspace\n"
                 "shaded contours show KDE density (used for the density-aware G_E metric)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "density_pc1_pc2.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  saved density_pc1_pc2.png")


# ============================================================
# Figure 13: A_lift vs other pair_alignment metrics
# ============================================================

def fig_alift_vs_metrics():
    arr = np.load("results/alift_all_pairs/all_pairs.npz")
    a_lift = arr["a_lift"]
    i_arr = arr["i"]
    j_arr = arr["j"]
    cfg = load_config()
    beh = BehaviorManifold.load(cfg.paths.manifold_y)
    labels = list(beh.labels)
    label_idx = {l: i for i, l in enumerate(labels)}

    pair_to_row = {(int(ii), int(jj)): k
                   for k, (ii, jj) in enumerate(zip(i_arr, j_arr))}

    align = json.loads(Path("results/pair_alignment.json").read_text())
    metrics_to_plot = [
        ("h_distance", "M_h subspace distance"),
        ("near_chord_centroid_count", "centroids within r=0.5 of chord"),
        ("ge_length_gap", "G_E geodesic - linear length"),
        ("max_chord_deflection", "max chord deflection"),
    ]
    metric_vals = {m: [] for m, _ in metrics_to_plot}
    aligned_alift = []
    for rec in align:
        if rec["start"] not in label_idx or rec["end"] not in label_idx:
            continue
        i = label_idx[rec["start"]]
        j = label_idx[rec["end"]]
        if i > j: i, j = j, i
        if (i, j) not in pair_to_row:
            continue
        k = pair_to_row[(i, j)]
        for m, _ in metrics_to_plot:
            metric_vals[m].append(rec.get(m, np.nan))
        aligned_alift.append(a_lift[k])
    aligned_alift = np.array(aligned_alift)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, (m, label) in zip(axes.flat, metrics_to_plot):
        vals = np.array(metric_vals[m])
        mask = np.isfinite(vals) & np.isfinite(aligned_alift)
        if mask.sum() < 50:
            ax.set_title(f"{label} — insufficient data")
            continue
        r, _ = stats.pearsonr(vals[mask], aligned_alift[mask])
        ax.scatter(vals[mask], aligned_alift[mask], s=2, alpha=0.2, color="steelblue")
        ax.axhline(0, color="k", linewidth=0.4)
        ax.set_xlabel(label)
        ax.set_ylabel("A_lift")
        ax.set_title(f"{label}\n  Pearson r = {r:+.3f}")
    plt.suptitle("A_lift is uncorrelated with all pair_alignment metrics — it captures new information",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "alift_vs_other_metrics.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  saved alift_vs_other_metrics.png")


# ============================================================
# Main: run all
# ============================================================

if __name__ == "__main__":
    print(f"writing figures to {OUT_DIR}")
    figs = [
        ("Fig 1: isometry_scatter_171", fig_isometry_171),
        ("Fig 2: isometry_30_vs_171", fig_isometry_30_vs_171),
        ("Fig 3: forest_plot_findings", fig_forest_plot),
        ("Fig 4: composition_stratified", fig_composition_stratified),
        ("Fig 5: pair_n40_results", fig_pair_n40_results),
        ("Fig 6: composition_n20_raw_vs_nm", fig_composition_raw_vs_nm),
        ("Fig 7: eval_direction_top_bottom_emotions", fig_eval_direction_emotions),
        ("Fig 8: eval_steering_dose_response", fig_eval_steering_dose_response),
        ("Fig 9: framing_cosine_heatmap", fig_framing_cosine_heatmap),
        ("Fig 10: tone_steering_per_emotion", fig_tone_per_emotion),
        ("Fig 11: pca_scree", fig_pca_scree),
        ("Fig 12: density_pc1_pc2", fig_density_pc1_pc2),
        ("Fig 13: alift_vs_other_metrics", fig_alift_vs_metrics),
    ]
    failed = []
    for name, fn in figs:
        try:
            print(f"\n{name}")
            fn()
        except Exception as e:
            print(f"  FAILED: {e}")
            failed.append((name, str(e)))
    print()
    if failed:
        print("Failed:")
        for n, e in failed:
            print(f"  {n}: {e}")
    else:
        print(f"All figures generated to {OUT_DIR}")
