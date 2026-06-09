"""Inspect what each principal component of the 5376-D emotion-vector
space encodes, by reporting the top emotions on each PC and the
Pearson correlation of each PC with V and A.

Motivated by the denser dim sweep finding (G_E edge peaks at 6-D):
which PCs are doing the work between PC2 (already known to track
arousal) and PC6?

Outputs:
- results/manifold_alternatives/pc_loadings.json
- results/manifold_alternatives/pc_loadings.png  (heat map of PC↔V,A
  correlations and stacked bar of top |loadings| per PC)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA

from manifold_emotions.behavior.manifold import BehaviorManifold
from manifold_emotions.vectors.diff_in_means import EmotionVectors


N_COMPONENTS = 8
TOP_K = 5
OUT_DIR = Path("results/manifold_alternatives")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ev = EmotionVectors.load(Path("data/emotion_vectors.npz"))
    beh = BehaviorManifold.load(Path("data/manifold_y.npz"))

    beh_set = set(beh.labels)
    common = [lab for lab in ev.labels if lab in beh_set]
    ev_idx = [ev.labels.index(lab) for lab in common]
    beh_idx = [beh.labels.index(lab) for lab in common]
    raw = ev.vectors[ev_idx].astype(np.float32)
    va = beh.centroids[beh_idx].astype(np.float32)

    pca = PCA(n_components=N_COMPONENTS, svd_solver="full")
    proj = pca.fit_transform(raw)  # (171, n_components)
    var_ratio = pca.explained_variance_ratio_
    print(f"PCA explained variance ratio: {var_ratio.round(3)}")
    print(f"  cumulative: {var_ratio.cumsum().round(3)}")

    rows = []
    for k in range(N_COMPONENTS):
        scores = proj[:, k]
        r_v, p_v = stats.pearsonr(scores, va[:, 0])
        r_a, p_a = stats.pearsonr(scores, va[:, 1])
        top_pos = np.argsort(-scores)[:TOP_K]
        top_neg = np.argsort(scores)[:TOP_K]
        rows.append({
            "pc": k + 1,
            "explained_variance_ratio": float(var_ratio[k]),
            "pearson_v": float(r_v),
            "pearson_a": float(r_a),
            "p_v": float(p_v),
            "p_a": float(p_a),
            "top_positive": [common[i] for i in top_pos],
            "top_negative": [common[i] for i in top_neg],
        })
        print(f"\nPC{k+1}  evr={var_ratio[k]:.3f}  r_V={r_v:+.3f} (p={p_v:.2g})  "
              f"r_A={r_a:+.3f} (p={p_a:.2g})")
        print(f"  + {', '.join(common[i] for i in top_pos)}")
        print(f"  − {', '.join(common[i] for i in top_neg)}")

    (OUT_DIR / "pc_loadings.json").write_text(json.dumps({"components": rows}, indent=2))
    print(f"\nsaved {OUT_DIR/'pc_loadings.json'}")

    # Plot: heatmap of |r_V| and |r_A| per PC, alongside explained variance bar
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5),
                                    gridspec_kw={"width_ratios": [1.0, 1.6]})

    pcs = np.arange(1, N_COMPONENTS + 1)
    rv = np.array([r["pearson_v"] for r in rows])
    ra = np.array([r["pearson_a"] for r in rows])
    width = 0.4
    ax1.bar(pcs - width / 2, rv, width, label="r(PC, V)", color="C0", edgecolor="black", lw=0.5)
    ax1.bar(pcs + width / 2, ra, width, label="r(PC, A)", color="C3", edgecolor="black", lw=0.5)
    ax1.axhline(0, color="gray", lw=0.5)
    for i, (v, a) in enumerate(zip(rv, ra)):
        ax1.text(pcs[i] - width / 2, v + 0.02 * np.sign(v),
                 f"{v:+.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=7)
        ax1.text(pcs[i] + width / 2, a + 0.02 * np.sign(a),
                 f"{a:+.2f}", ha="center", va="bottom" if a >= 0 else "top", fontsize=7)
    ax1.set_xticks(pcs)
    ax1.set_xticklabels([f"PC{k}" for k in pcs])
    ax1.set_ylabel("Pearson r with V or A")
    ax1.set_title("PC alignment to valence/arousal axes")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(alpha=0.3, axis="y")
    ax1.set_ylim(-1.05, 1.05)

    ax2.axis("off")
    text = "Top-loading emotions per PC (positive / negative):\n\n"
    for r in rows:
        evr = r["explained_variance_ratio"]
        text += (
            f"PC{r['pc']:>2} (evr {evr:.3f}):\n"
            f"   +  {', '.join(r['top_positive'])}\n"
            f"   −  {', '.join(r['top_negative'])}\n\n"
        )
    ax2.text(0, 1, text, va="top", ha="left", family="monospace", fontsize=8)
    plt.suptitle("PCA loadings inspection (171 emotion centroids in 5376-D)", y=1.01)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "pc_loadings.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {OUT_DIR/'pc_loadings.png'}")


if __name__ == "__main__":
    main()
