"""Analyze the n=100+ chord extension.

After the original 40 pairs + 60 extension pairs are all in
results/pullback/, this script:
- recomputes per-pair A_lift, margins, off-gaps for all 100
- reports the headline correlations with tighter CIs
- analyzes the predict-TIE bucket specifically (which previously
  had 7/12 losses instead of ties — does that hold with more data?)
- writes summary + per-quintile breakdown

Output:
- results/alift_n100_extension/_summary.json
- results/figures/writeup/alift_n100_extension.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from manifold_emotions.behavior.manifold import BehaviorManifold


PULLBACK_DIR = Path("results/pullback")
OUT_DIR = Path("results/alift_n100_extension")
FIG_DIR = Path("results/figures/writeup")


def load_pair(p: Path) -> dict | None:
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    t = data["trajectories"]
    pb = t["pullback"]
    li = t["linear"]
    return {
        "pair": p.stem,
        "pb_off": float(pb["off_manifold_energy"]),
        "li_off": float(li["off_manifold_energy"]),
        "pb_myl": float(pb["my_geodesic_distance"]),
        "li_myl": float(li["my_geodesic_distance"]),
        "off_gap": float(pb["off_manifold_energy"]) - float(li["off_manifold_energy"]),
        "margin":  float(pb["my_geodesic_distance"]) - float(li["my_geodesic_distance"]),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Load A_lift for all pairs
    beh = BehaviorManifold.load(Path("data/manifold_y.npz"))
    labels = list(beh.labels)
    arr = np.load("results/alift_all_pairs/all_pairs.npz")
    a_lift_all = arr["a_lift"]
    pair_i = arr["i"]
    pair_j = arr["j"]
    pair_alift: dict[frozenset, float] = {}
    for k in range(len(a_lift_all)):
        a = labels[pair_i[k]]
        b = labels[pair_j[k]]
        pair_alift[frozenset([a, b])] = float(a_lift_all[k])

    # Walk all pullback results and join on A_lift
    rows = []
    for p in sorted(PULLBACK_DIR.glob("*.json")):
        if p.name.startswith("_"):
            continue
        # results/pullback/{a}_{b}.json — handle multi-word labels by lookup
        stem = p.stem
        m = load_pair(p)
        if m is None:
            continue
        # Try to recover the label pair from filename
        # Many emotion labels have underscores, so we need a more robust matching
        ab = stem
        match = None
        for c_pair, av in pair_alift.items():
            la, lb = tuple(c_pair)
            f = f"{la}_{lb}"
            r = f"{lb}_{la}"
            if ab == f.replace(" ", "_") or ab == r.replace(" ", "_"):
                match = av
                break
        if match is None:
            continue
        m["a_lift"] = match
        rows.append(m)

    print(f"loaded {len(rows)} pairs with A_lift+behavioral data")
    if len(rows) < 50:
        print("  fewer than 50 — extension chain may not be done yet")

    a = np.array([r["a_lift"] for r in rows])
    margin = np.array([r["margin"] for r in rows])
    off_gap = np.array([r["off_gap"] for r in rows])

    # Headline correlations
    pr_m, pp_m = stats.pearsonr(a, margin)
    sp_m, sp_m_p = stats.spearmanr(a, margin)
    pr_o, pp_o = stats.pearsonr(a, off_gap)
    sp_o, sp_o_p = stats.spearmanr(a, off_gap)
    pr_aom, pp_aom = stats.pearsonr(np.abs(a), off_gap)
    print(f"\nn={len(rows)}")
    print(f"  A_lift → margin (Wilcoxon: pullback closer = negative margin):")
    print(f"    Pearson r={pr_m:+.3f} (p={pp_m:.4f})")
    print(f"    Spearman r={sp_m:+.3f} (p={sp_m_p:.4f})")
    print(f"  A_lift → off_gap:")
    print(f"    Pearson r={pr_o:+.3f} (p={pp_o:.4f})")
    print(f"  |A_lift| → off_gap:")
    print(f"    Pearson r={pr_aom:+.3f} (p={pp_aom:.4f})")

    # Bootstrap CIs for the headline correlations
    def boot_pearson(x, y, n_boot=2000, seed=0):
        rng = np.random.default_rng(seed)
        n = len(x)
        rs = np.empty(n_boot)
        for k in range(n_boot):
            idx = rng.choice(n, size=n, replace=True)
            rs[k] = stats.pearsonr(x[idx], y[idx])[0]
        return float(np.quantile(rs, 0.025)), float(np.quantile(rs, 0.975))

    pm_lo, pm_hi = boot_pearson(a, margin)
    po_lo, po_hi = boot_pearson(np.abs(a), off_gap)
    print(f"  A_lift → margin CI: [{pm_lo:+.3f}, {pm_hi:+.3f}]")
    print(f"  |A_lift| → off_gap CI: [{po_lo:+.3f}, {po_hi:+.3f}]")

    # Predict-TIE bucket (|A_lift| < 0.02)
    tie_mask = np.abs(a) < 0.02
    tie = np.array(rows)[tie_mask]
    if len(tie) > 0:
        margins_tie = np.array([r["margin"] for r in tie])
        wins_tie = int((margins_tie < 0).sum())
        print(f"\n  predict-TIE bucket (|A_lift|<0.02, n={len(tie)}):")
        print(f"    pullback wins: {wins_tie}/{len(tie)}  (mean margin {margins_tie.mean():+.3f})")

    out = {
        "n": len(rows),
        "a_lift_margin": {"pearson_r": float(pr_m), "p": float(pp_m),
                          "ci": [pm_lo, pm_hi],
                          "spearman_r": float(sp_m), "spearman_p": float(sp_m_p)},
        "a_lift_off_gap": {"pearson_r": float(pr_o), "p": float(pp_o)},
        "abs_alift_off_gap": {"pearson_r": float(pr_aom), "p": float(pp_aom),
                              "ci": [po_lo, po_hi]},
        "per_pair": rows,
    }
    (OUT_DIR / "_summary.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved {OUT_DIR/'_summary.json'}")

    # Scatter plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, x, y, xname, yname, color in [
        (axes[0], a, margin, "A_lift", "pullback − linear margin (M_y-line)", "steelblue"),
        (axes[1], np.abs(a), off_gap, "|A_lift|", "pullback − linear off-M_y E gap", "darkorange"),
    ]:
        ax.scatter(x, y, alpha=0.65, color=color, edgecolor="black", lw=0.4)
        r, p = stats.pearsonr(x, y)
        z = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, np.polyval(z, xs), "r--", lw=1.5, label=f"r={r:+.3f}\np={p:.4f}")
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
        ax.set_xlabel(xname); ax.set_ylabel(yname)
        ax.set_title(f"{xname} → {yname} (n={len(rows)})")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(alpha=0.3)
    plt.suptitle(f"A_lift predictor at n={len(rows)}", y=1.0)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "alift_n100_extension.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {FIG_DIR/'alift_n100_extension.png'}")


if __name__ == "__main__":
    main()
