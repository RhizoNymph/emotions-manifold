"""V_lift as a possible companion predictor to A_lift.

For each n=40 pair, compute V_lift the same way A_lift is computed
(kernel-weighted valence at the chord midpoint minus target valence),
and correlate with the pullback − linear margin reported in the n=40
chord experiment.

If V_lift correlates with margin, the framework gives us *two*
geometric predictors instead of one. If not, A_lift's asymmetry on
arousal specifically is a stronger interpretive claim (Gemma
under-produces arousal under steering and pullback compensates;
valence is already produced fine so the kernel-weighted lift on
valence is uninformative).

Outputs:
- results/alift_expansion/vlift_predictor.json
- results/alift_expansion/vlift_predictor.png
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


SUMMARY_PATH = Path("results/alift_expansion/_summary.json")
ALL_PAIRS_PATH = Path("results/alift_all_pairs/all_pairs.npz")
OUT_DIR = Path("results/alift_expansion")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = json.loads(SUMMARY_PATH.read_text())
    beh = BehaviorManifold.load(Path("data/manifold_y.npz"))
    label_to_idx = {lab: i for i, lab in enumerate(beh.labels)}

    # all_pairs.npz gives v_lift indexed by (i, j) pair index over
    # the 147-emotion common label set. Rebuild that to match.
    arr = np.load(ALL_PAIRS_PATH)
    v_lift = arr["v_lift"]
    a_lift = arr["a_lift"]
    pair_i = arr["i"]
    pair_j = arr["j"]
    # Reconstruct the label list used by the all_pairs script.
    # `analyze_alift_all_pairs.py` uses the M_y labels filtered to
    # those with valid kernel weights — for 171 emotions this is
    # typically all 147 (matching the rated set).
    # Easier path: build a label→idx using the same set as M_y.
    common_labels = list(beh.labels)
    # Index into pair_i / pair_j refers to indices in the M_y labels.

    # Now, summary['per_pair'] has "pair": "happy->sad" strings.
    pair_data = []
    matched = 0
    for entry in summary["per_pair"]:
        pair_str = entry["pair"]
        a, b = pair_str.split("->")
        a = a.strip()
        b = b.strip()
        if a not in label_to_idx or b not in label_to_idx:
            continue
        ia = label_to_idx[a]
        ib = label_to_idx[b]
        # Find the row in pair_i/pair_j (might be either order)
        mask = ((pair_i == ia) & (pair_j == ib)) | ((pair_i == ib) & (pair_j == ia))
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            continue
        idx = idxs[0]
        pair_data.append({
            "pair": pair_str,
            "a_lift": float(a_lift[idx]),
            "v_lift": float(v_lift[idx]),
            "margin": float(entry["margin"]),
            "off_gap": float(entry["off_gap"]),
        })
        matched += 1

    print(f"matched {matched}/{len(summary['per_pair'])} pairs")

    a = np.array([p["a_lift"] for p in pair_data])
    v = np.array([p["v_lift"] for p in pair_data])
    m = np.array([p["margin"] for p in pair_data])
    o = np.array([p["off_gap"] for p in pair_data])

    print("\n=== Correlations with margin (pullback − linear M_y-line) ===")
    for name, x in [("a_lift", a), ("v_lift", v), ("|a_lift|", np.abs(a)),
                    ("|v_lift|", np.abs(v))]:
        r, p = stats.pearsonr(x, m)
        s, sp = stats.spearmanr(x, m)
        print(f"  {name:>12s}: Pearson r={r:+.3f} (p={p:.3f})  Spearman r={s:+.3f} (p={sp:.3f})")

    print("\n=== Correlations with off-M_y E gap ===")
    for name, x in [("a_lift", a), ("v_lift", v), ("|a_lift|", np.abs(a)),
                    ("|v_lift|", np.abs(v))]:
        r, p = stats.pearsonr(x, o)
        s, sp = stats.spearmanr(x, o)
        print(f"  {name:>12s}: Pearson r={r:+.3f} (p={p:.3f})  Spearman r={s:+.3f} (p={sp:.3f})")

    out = {
        "n": len(pair_data),
        "per_pair": pair_data,
        "corr_with_margin": {
            "a_lift_pearson_r": float(stats.pearsonr(a, m)[0]),
            "a_lift_pearson_p": float(stats.pearsonr(a, m)[1]),
            "v_lift_pearson_r": float(stats.pearsonr(v, m)[0]),
            "v_lift_pearson_p": float(stats.pearsonr(v, m)[1]),
        },
        "corr_with_off_gap": {
            "a_lift_pearson_r": float(stats.pearsonr(a, o)[0]),
            "a_lift_pearson_p": float(stats.pearsonr(a, o)[1]),
            "v_lift_pearson_r": float(stats.pearsonr(v, o)[0]),
            "v_lift_pearson_p": float(stats.pearsonr(v, o)[1]),
        },
    }
    (OUT_DIR / "vlift_predictor.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved {OUT_DIR/'vlift_predictor.json'}")

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, x, xname in [(axes[0, 0], a, "A_lift"), (axes[0, 1], v, "V_lift")]:
        ax.scatter(x, m, alpha=0.7, color="steelblue", edgecolor="black", lw=0.4)
        if len(x) > 1:
            r, p = stats.pearsonr(x, m)
            z = np.polyfit(x, m, 1)
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, np.polyval(z, xs), "r--", lw=1.5,
                    label=f"r={r:+.3f}\np={p:.3f}")
            ax.legend(loc="upper left", fontsize=9)
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
        ax.set_xlabel(xname)
        ax.set_ylabel("pullback − linear margin (M_y-line)")
        ax.set_title(f"{xname} → pullback margin (n={len(x)})")
        ax.grid(alpha=0.3)

    for ax, x, xname in [(axes[1, 0], a, "A_lift"), (axes[1, 1], v, "V_lift")]:
        ax.scatter(x, o, alpha=0.7, color="darkorange", edgecolor="black", lw=0.4)
        if len(x) > 1:
            r, p = stats.pearsonr(x, o)
            z = np.polyfit(x, o, 1)
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, np.polyval(z, xs), "r--", lw=1.5,
                    label=f"r={r:+.3f}\np={p:.3f}")
            ax.legend(loc="upper left", fontsize=9)
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
        ax.set_xlabel(xname)
        ax.set_ylabel("pullback − linear off-M_y E gap")
        ax.set_title(f"{xname} → off-M_y E gap (n={len(x)})")
        ax.grid(alpha=0.3)

    plt.suptitle("A_lift vs V_lift as predictors of pullback advantage", y=1.0)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "vlift_predictor.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {OUT_DIR/'vlift_predictor.png'}")


if __name__ == "__main__":
    main()
