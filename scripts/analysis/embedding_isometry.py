"""Quick embedding-isometry comparison (no geodesics).

Compares chord-distance Pearson correlation with V/A across PCA, UMAP,
and diffusion-map embeddings of the 171 emotion centroids. Standalone
and fast (< 30 s), so it can be run independently of the geodesic-heavy
manifold_alternatives.py.

Outputs:
- results/manifold_alternatives/embedding_isometry.json
- results/manifold_alternatives/embedding_isometry.png
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA

from manifold_emotions.behavior.manifold import BehaviorManifold
from manifold_emotions.vectors.diff_in_means import EmotionVectors


SEED = 5678
OUT_DIR = Path("results/manifold_alternatives")


def pca_embed(centroids: np.ndarray, n_components: int) -> np.ndarray:
    return PCA(n_components=n_components, svd_solver="full").fit_transform(centroids).astype(np.float32)


def umap_embed(centroids: np.ndarray, n_components: int, n_neighbors: int) -> np.ndarray:
    import umap
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        random_state=SEED,
        metric="euclidean",
    )
    return reducer.fit_transform(centroids).astype(np.float32)


def diffusion_map_embed(centroids: np.ndarray, n_components: int) -> np.ndarray:
    from scipy.spatial.distance import squareform, pdist
    sq = squareform(pdist(centroids)) ** 2
    epsilon = float(np.median(sq[sq > 0]))
    K = np.exp(-sq / epsilon)
    q = K.sum(axis=1)
    K_alpha = K / np.outer(q, q)
    d = K_alpha.sum(axis=1)
    d_sqrt = np.sqrt(d)
    P_sym = K_alpha / np.outer(d_sqrt, d_sqrt)
    P_sym = 0.5 * (P_sym + P_sym.T)
    vals, vecs = np.linalg.eigh(P_sym)
    vals = vals[::-1]
    vecs = vecs[:, ::-1]
    psi = vecs / d_sqrt[:, None]
    psi = psi[:, 1:n_components + 1]
    vals = vals[1:n_components + 1]
    return (psi * vals[None, :]).astype(np.float32)


def _pairwise_dists(points: np.ndarray) -> np.ndarray:
    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt((diff * diff).sum(axis=-1))


def _ut(matrix: np.ndarray) -> np.ndarray:
    n = matrix.shape[0]
    ii, jj = np.triu_indices(n, k=1)
    return matrix[ii, jj]


def isometry_corr(embedding: np.ndarray, va: np.ndarray) -> dict:
    de = _ut(_pairwise_dists(embedding))
    dv = _ut(_pairwise_dists(va))
    pr, pp = stats.pearsonr(de, dv)
    sr, sp = stats.spearmanr(de, dv)
    return {
        "pearson_r": float(pr),
        "pearson_p": float(pp),
        "spearman_r": float(sr),
        "spearman_p": float(sp),
        "n_pairs": int(len(de)),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ev = EmotionVectors.load(Path("data/emotion_vectors.npz"))
    beh = BehaviorManifold.load(Path("data/manifold_y.npz"))

    beh_set = set(beh.labels)
    common = [lab for lab in ev.labels if lab in beh_set]
    print(f"common labels (M_h ∩ M_y): {len(common)}")
    ev_idx = [ev.labels.index(lab) for lab in common]
    beh_idx = [beh.labels.index(lab) for lab in common]
    raw = ev.vectors[ev_idx].astype(np.float32)
    va = beh.centroids[beh_idx].astype(np.float32)

    results: list[dict] = []

    iso = isometry_corr(raw, va)
    print(f"  raw activation (5376-D): pearson={iso['pearson_r']:+.3f}  spearman={iso['spearman_r']:+.3f}")
    results.append({"method": "raw_activation", "n_components": raw.shape[1], **iso})

    for d in [2, 4, 8, 16, 32]:
        emb = pca_embed(raw, d)
        iso = isometry_corr(emb, va)
        print(f"  PCA-{d}: pearson={iso['pearson_r']:+.3f}")
        results.append({"method": f"PCA-{d}", "n_components": d, **iso})

    for d in [2, 4, 8]:
        for nn in [15, 30]:
            try:
                emb = umap_embed(raw, d, nn)
                iso = isometry_corr(emb, va)
                print(f"  UMAP-{d} (nn={nn}): pearson={iso['pearson_r']:+.3f}")
                results.append({"method": f"UMAP-{d}-nn{nn}", "n_components": d, "n_neighbors": nn, **iso})
            except Exception as exc:
                print(f"  UMAP-{d} (nn={nn}) FAILED: {exc}")

    for d in [2, 4, 8]:
        try:
            emb = diffusion_map_embed(raw, d)
            iso = isometry_corr(emb, va)
            print(f"  diffusion-map-{d}: pearson={iso['pearson_r']:+.3f}")
            results.append({"method": f"diffusion-{d}", "n_components": d, **iso})
        except Exception as exc:
            print(f"  diffusion-map-{d} FAILED: {exc}")

    (OUT_DIR / "embedding_isometry.json").write_text(json.dumps({"results": results}, indent=2))
    print(f"\nsaved {OUT_DIR/'embedding_isometry.json'}")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    methods = [r["method"] for r in results]
    rs = [r["pearson_r"] for r in results]
    colors = []
    for m in methods:
        if m == "raw_activation":
            colors.append("gray")
        elif m.startswith("PCA"):
            colors.append("steelblue")
        elif m.startswith("UMAP"):
            colors.append("seagreen")
        else:
            colors.append("indianred")
    bars = ax.bar(range(len(methods)), rs, color=colors, edgecolor="black", linewidth=0.5)
    for b, r in zip(bars, rs):
        ax.text(b.get_x() + b.get_width() / 2, r + 0.005,
                f"{r:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Pearson r vs V/A distance")
    ax.set_title("Geometric isometry (chord distance ↔ V/A distance) per embedding")
    ax.axhline(0, color="gray", lw=0.5)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "embedding_isometry.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {OUT_DIR/'embedding_isometry.png'}")


if __name__ == "__main__":
    main()
