"""Alternative embeddings and adaptive-bandwidth KDE for the emotion manifold.

Computes geometric isometry (chord distance correlation with V/A) for:
- PCA-{4, 8, 16, 32}-D (baselines, recomputed here for reference)
- UMAP-{2, 4, 8}-D with n_neighbors in {15, 30}
- Diffusion maps-{2, 4, 8}-D with epsilon scaling

And computes geodesic isometry (G_E geodesic length correlation with V/A) for
PCA-8 under:
- fixed Silverman bandwidth (baseline)
- fixed median-NN bandwidth (production default)
- adaptive bandwidth (per-centroid kNN with k=5)

Adaptive-bandwidth KDE here means each kernel uses bandwidth h_i = c * d_k(c_i)
where d_k is distance to k-th nearest centroid; this is the standard "balloon
estimator" variant. Implementation is JAX-traceable so we can fit geodesics
through it the same way we do for the fixed-bandwidth manifold.

Outputs:
- results/manifold_alternatives/embedding_isometry.json
- results/manifold_alternatives/embedding_isometry.png
- results/manifold_alternatives/adaptive_kde_geodesic.json
- results/manifold_alternatives/adaptive_kde_geodesic.png
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA

from manifold_emotions.behavior.manifold import BehaviorManifold
from manifold_emotions.manifold.density import (
    GaussianKDE,
    clustered_bandwidth,
    silverman_bandwidth,
)
from manifold_emotions.manifold.geodesic import fit_geodesic
from manifold_emotions.manifold.metric import DensityGeometry
from manifold_emotions.vectors.diff_in_means import EmotionVectors


SEED = 5678
N_PAIRS_GEODESIC = 800
NUM_WAYPOINTS = 30
OUT_DIR = Path("results/manifold_alternatives")


# ------------------------------------------------------------------ embeddings


def pca_embed(centroids: np.ndarray, n_components: int) -> np.ndarray:
    p = PCA(n_components=n_components, svd_solver="full")
    return p.fit_transform(centroids).astype(np.float32)


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


def diffusion_map_embed(centroids: np.ndarray, n_components: int,
                        epsilon: float | None = None) -> np.ndarray:
    """Anisotropic diffusion-map embedding (Coifman-Lafon, alpha=1.0).

    Returns the top ``n_components`` non-trivial eigenvectors of the
    normalized Markov matrix, scaled by their eigenvalues. ``epsilon``
    defaults to the median squared pairwise distance.
    """
    from scipy.spatial.distance import squareform, pdist
    sq = squareform(pdist(centroids)) ** 2
    if epsilon is None:
        # Median nonzero squared distance — standard heuristic.
        epsilon = float(np.median(sq[sq > 0]))
    K = np.exp(-sq / epsilon)
    # alpha=1 normalization (density-corrected diffusion)
    q = K.sum(axis=1)
    K_alpha = K / np.outer(q, q)
    # Row-normalize for Markov chain
    d = K_alpha.sum(axis=1)
    P = K_alpha / d[:, None]
    # Symmetrize for numerical eig: P = D^{-1/2} P_sym D^{1/2} with
    # P_sym = D^{-1/2} K_alpha D^{-1/2}
    d_sqrt = np.sqrt(d)
    P_sym = K_alpha / np.outer(d_sqrt, d_sqrt)
    P_sym = 0.5 * (P_sym + P_sym.T)
    vals, vecs = np.linalg.eigh(P_sym)
    # eigh returns ascending; reverse to descending
    vals = vals[::-1]
    vecs = vecs[:, ::-1]
    # Convert symmetric eigenvectors back to right-eigenvectors of P
    psi = vecs / d_sqrt[:, None]
    # Drop the trivial first eigenvector (all-constant, value ~1)
    psi = psi[:, 1:n_components + 1]
    vals = vals[1:n_components + 1]
    return (psi * vals[None, :]).astype(np.float32)


# ------------------------------------------------------------ adaptive KDE


@dataclass(frozen=True)
class AdaptiveKDE:
    """Variable-bandwidth Gaussian KDE: each kernel has its own bandwidth.

    log_kernel_sum(h) = log( sum_i exp( -||h - c_i||^2 / (2 * h_i^2) ) )

    Same energy semantics as the fixed-bandwidth KDE — we use the
    UNNORMALIZED kernel sum for the same reasons (see GaussianKDE docs).
    """
    centroids: jax.Array  # (N, d)
    bandwidths: jax.Array  # (N,)

    @classmethod
    def fit(cls, centroids: np.ndarray, k: int = 5, scale: float = 1.0) -> "AdaptiveKDE":
        c = np.asarray(centroids, dtype=np.float32)
        n, _ = c.shape
        diff = c[:, None, :] - c[None, :, :]
        dist = np.sqrt((diff * diff).sum(axis=-1))
        # k-th nearest neighbor (exclude self)
        sorted_d = np.sort(dist, axis=1)
        k_nn = sorted_d[:, k]  # (N,)  - self is at column 0 (distance 0)
        bw = (k_nn * scale).astype(np.float32)
        return cls(centroids=jnp.asarray(c), bandwidths=jnp.asarray(bw))

    def log_kernel_sum(self, points: jax.Array) -> jax.Array:
        points = jnp.asarray(points)
        single = points.ndim == 1
        if single:
            points = points[None, :]
        diffs = points[:, None, :] - self.centroids[None, :, :]
        sq = jnp.sum(diffs * diffs, axis=-1)  # (M, N)
        sigma2 = self.bandwidths ** 2  # (N,)
        kernel_log = -sq / (2.0 * sigma2[None, :])  # (M, N)
        out = jax.scipy.special.logsumexp(kernel_log, axis=1)
        return out[0] if single else out

    def energy(self, points: jax.Array) -> jax.Array:
        return -self.log_kernel_sum(points)


# ------------------------------------------------------------ analyses


def _pairwise_dists(points: np.ndarray) -> np.ndarray:
    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt((diff * diff).sum(axis=-1))


def _upper_triangle(matrix: np.ndarray) -> np.ndarray:
    n = matrix.shape[0]
    ii, jj = np.triu_indices(n, k=1)
    return matrix[ii, jj]


def isometry_corr(embedding: np.ndarray, va: np.ndarray) -> dict:
    de = _upper_triangle(_pairwise_dists(embedding))
    dv = _upper_triangle(_pairwise_dists(va))
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

    # Load M_y and M_h_full (raw activation centroids).
    ev = EmotionVectors.load(Path("data/emotion_vectors.npz"))
    beh = BehaviorManifold.load(Path("data/manifold_y.npz"))

    # Align to common labels.
    beh_set = set(beh.labels)
    common = [lab for lab in ev.labels if lab in beh_set]
    print(f"common labels (M_h ∩ M_y): {len(common)}")
    ev_idx = [ev.labels.index(lab) for lab in common]
    beh_idx = [beh.labels.index(lab) for lab in common]
    raw = ev.vectors[ev_idx].astype(np.float32)
    va = beh.centroids[beh_idx].astype(np.float32)

    # ---------- (A) Geometric isometry across embeddings ----------
    print("\n=== Geometric isometry across embeddings ===")
    results: list[dict] = []

    # raw activation baseline
    iso = isometry_corr(raw, va)
    print(f"  raw activation (5376-D): pearson={iso['pearson_r']:+.3f}  "
          f"spearman={iso['spearman_r']:+.3f}")
    results.append({"method": "raw_activation", "n_components": raw.shape[1], **iso})

    for d in [2, 4, 8, 16, 32]:
        emb = pca_embed(raw, d)
        iso = isometry_corr(emb, va)
        print(f"  PCA-{d}: pearson={iso['pearson_r']:+.3f}  "
              f"spearman={iso['spearman_r']:+.3f}")
        results.append({"method": f"PCA-{d}", "n_components": d, **iso})

    for d in [2, 4, 8]:
        for nn in [15, 30]:
            try:
                emb = umap_embed(raw, d, nn)
                iso = isometry_corr(emb, va)
                print(f"  UMAP-{d} (nn={nn}): pearson={iso['pearson_r']:+.3f}  "
                      f"spearman={iso['spearman_r']:+.3f}")
                results.append({
                    "method": f"UMAP-{d}-nn{nn}",
                    "n_components": d,
                    "n_neighbors": nn,
                    **iso,
                })
            except Exception as exc:
                print(f"  UMAP-{d} (nn={nn}) FAILED: {exc}")

    for d in [2, 4, 8]:
        try:
            emb = diffusion_map_embed(raw, d)
            iso = isometry_corr(emb, va)
            print(f"  diffusion-map-{d}: pearson={iso['pearson_r']:+.3f}  "
                  f"spearman={iso['spearman_r']:+.3f}")
            results.append({"method": f"diffusion-{d}", "n_components": d, **iso})
        except Exception as exc:
            print(f"  diffusion-map-{d} FAILED: {exc}")

    (OUT_DIR / "embedding_isometry.json").write_text(json.dumps({"results": results}, indent=2))
    print(f"\nsaved {OUT_DIR/'embedding_isometry.json'}")

    # Plot bar chart of pearson r per method.
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

    # ---------- (B) Adaptive bandwidth geodesic isometry ----------
    print("\n=== Adaptive-bandwidth geodesic isometry (PCA-8) ===")
    pca8 = pca_embed(raw, 8)

    # Build the same sampled pair set as run_dimension_ablation.py.
    rng = np.random.default_rng(SEED)
    n = len(common)
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    sample = rng.choice(len(all_pairs), size=min(N_PAIRS_GEODESIC, len(all_pairs)), replace=False)
    pair_idx = [all_pairs[k] for k in sample]
    va_pair_dist = np.array([
        float(np.linalg.norm(va[i] - va[j])) for i, j in pair_idx
    ])

    bw_silver = silverman_bandwidth(jnp.asarray(pca8))
    bw_clustered = clustered_bandwidth(jnp.asarray(pca8), multiplier=1.0)
    print(f"  fixed bandwidths: silverman={bw_silver:.3f}  "
          f"clustered_NN={bw_clustered:.3f}")

    settings: list[tuple[str, DensityGeometry]] = []
    # Fixed Silverman
    kde_s = GaussianKDE.fit(pca8, bandwidth=bw_silver)
    settings.append(("fixed_silverman", DensityGeometry(energy_fn=kde_s.energy, alpha=1.0, beta=0.01)))
    # Fixed clustered NN (production)
    kde_c = GaussianKDE.fit(pca8, bandwidth=bw_clustered)
    settings.append(("fixed_clustered_nn", DensityGeometry(energy_fn=kde_c.energy, alpha=1.0, beta=0.01)))
    # Adaptive k=5
    adk5 = AdaptiveKDE.fit(pca8, k=5, scale=1.0)
    settings.append(("adaptive_k5", DensityGeometry(energy_fn=adk5.energy, alpha=1.0, beta=0.01)))
    # Adaptive k=10
    adk10 = AdaptiveKDE.fit(pca8, k=10, scale=1.0)
    settings.append(("adaptive_k10", DensityGeometry(energy_fn=adk10.energy, alpha=1.0, beta=0.01)))

    chord_lens = np.array([
        float(np.linalg.norm(pca8[i] - pca8[j])) for i, j in pair_idx
    ], dtype=np.float32)
    r_chord, _ = stats.pearsonr(chord_lens, va_pair_dist)
    print(f"  baseline chord r={r_chord:+.3f}")

    geodesic_results: list[dict] = [{
        "method": "chord_baseline",
        "pearson_r": float(r_chord),
        "edge_pearson": 0.0,
        "n_pairs": len(pair_idx),
    }]
    for name, geom in settings:
        t0 = time.monotonic()
        ge_lens = np.zeros(len(pair_idx), dtype=np.float32)
        for k, (i, j) in enumerate(pair_idx):
            c_a = pca8[i].astype(np.float32)
            c_b = pca8[j].astype(np.float32)
            res = fit_geodesic(geom, c_a, c_b, num_waypoints=NUM_WAYPOINTS, max_iter=200)
            ge_lens[k] = res.final_length
            if k % 100 == 0:
                el = time.monotonic() - t0
                rate = (k + 1) / max(el, 0.01)
                eta = (len(pair_idx) - k - 1) / rate
                print(f"  {name} {k:>4d}/{len(pair_idx)}  ({el:.0f}s, eta {eta:.0f}s)")
        r_ge, _ = stats.pearsonr(ge_lens, va_pair_dist)
        s_ge, _ = stats.spearmanr(ge_lens, va_pair_dist)
        edge = r_ge - r_chord
        print(f"  {name}: r={r_ge:+.3f}  edge={edge:+.3f}  ({time.monotonic()-t0:.0f}s)")
        geodesic_results.append({
            "method": name,
            "pearson_r": float(r_ge),
            "spearman_r": float(s_ge),
            "edge_pearson": float(edge),
            "n_pairs": len(pair_idx),
            "wall_seconds": float(time.monotonic() - t0),
        })

    (OUT_DIR / "adaptive_kde_geodesic.json").write_text(json.dumps({
        "n_pairs": len(pair_idx),
        "results": geodesic_results,
    }, indent=2))
    print(f"\nsaved {OUT_DIR/'adaptive_kde_geodesic.json'}")

    # Plot adaptive KDE bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    methods = [r["method"] for r in geodesic_results]
    rs = [r["pearson_r"] for r in geodesic_results]
    bars = ax.bar(range(len(methods)), rs,
                  color=["gray" if m == "chord_baseline" else "darkorange" for m in methods],
                  edgecolor="black", linewidth=0.5)
    for b, m, r, full in zip(bars, methods, rs, geodesic_results):
        edge = full.get("edge_pearson", 0.0)
        label = f"{r:.3f}\n(edge {edge:+.3f})" if m != "chord_baseline" else f"{r:.3f}"
        ax.text(b.get_x() + b.get_width() / 2, r + 0.003,
                label, ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylabel("Pearson r vs V/A distance")
    ax.set_title(f"Geodesic isometry under fixed vs adaptive KDE bandwidth\n(PCA-8, n={len(pair_idx)} pairs)")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "adaptive_kde_geodesic.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {OUT_DIR/'adaptive_kde_geodesic.png'}")


if __name__ == "__main__":
    main()
