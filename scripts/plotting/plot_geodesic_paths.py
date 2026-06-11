"""Visualize geodesic / linear / pullback paths through PC1×PC2 + density contours.

Loads cached geodesic waypoints + on-the-fly N-W pullback waypoints +
linear interpolations for a handful of illustrative pairs, projects to
PC1×PC2, and overlays them on a KDE density heatmap.

Goal: SHOW that the curved-metric geodesic bends through dense regions
while the linear baseline cuts straight through low-density areas. This
is the geometric story behind the +0.049 isometry edge under G_E.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from manifold_emotions.behavior.manifold import BehaviorManifold
from manifold_emotions.config import load_config
from manifold_emotions.manifold.fit import FittedManifold
from manifold_emotions.manifold.pullback import compute_pullback


SHOW_PAIRS = [
    ("happy", "sad"),
    ("excited", "weary"),
    ("depressed", "energized"),
    ("terrified", "serene"),
    ("calm", "ecstatic"),
    ("amused", "ashamed"),
]


def main() -> None:
    cfg = load_config()
    mh = FittedManifold.load(cfg.paths.manifold_h)
    beh = BehaviorManifold.load(cfg.paths.manifold_y)

    cache = np.load("data/geodesics_cache.npz", allow_pickle=True)
    waypoints = cache["waypoints"]
    pair_indices = cache["pair_indices"]
    cache_labels = list(cache["labels"])
    pair_lookup = {(int(i), int(j)): k for k, (i, j) in enumerate(pair_indices)}

    kde = mh.make_density()

    # KDE heatmap over PC1×PC2 (marginalize away other PCs by sampling
    # the KDE density at (pc1, pc2, mean_other_pcs))
    centroids = mh.centroids_subspace
    pc_means = centroids.mean(axis=0)
    x_min, x_max = centroids[:, 0].min() - 5, centroids[:, 0].max() + 5
    y_min, y_max = centroids[:, 1].min() - 5, centroids[:, 1].max() + 5
    grid_n = 80
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_n),
        np.linspace(y_min, y_max, grid_n),
    )
    grid_pts = np.zeros((grid_n * grid_n, centroids.shape[1]), dtype=np.float32)
    grid_pts[:, 0] = xx.ravel()
    grid_pts[:, 1] = yy.ravel()
    for d in range(2, centroids.shape[1]):
        grid_pts[:, d] = pc_means[d]
    log_density_grid = np.asarray(kde.log_kernel_sum(jnp.asarray(grid_pts)))
    density_grid = np.exp(log_density_grid - log_density_grid.max()).reshape(grid_n, grid_n)

    n_pairs = len(SHOW_PAIRS)
    n_cols = 3
    n_rows = (n_pairs + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5.2, n_rows * 5.0))
    axes = np.atleast_2d(axes).flatten()

    for ax_idx, (start, end) in enumerate(SHOW_PAIRS):
        ax = axes[ax_idx]
        if start not in cache_labels or end not in cache_labels:
            ax.set_title(f"{start} ↔ {end}  (missing)")
            ax.axis("off")
            continue
        if start not in mh.labels or end not in mh.labels:
            ax.set_title(f"{start} ↔ {end}  (missing in M_h)")
            ax.axis("off")
            continue

        i = cache_labels.index(start)
        j = cache_labels.index(end)
        a, b = (i, j) if i < j else (j, i)
        k = pair_lookup.get((a, b))
        if k is None:
            ax.set_title(f"{start} ↔ {end}  (no cache)")
            ax.axis("off")
            continue

        geo_path = waypoints[k]  # (K, 8)
        # Orientation: cache stores start=lower-idx → end=higher-idx
        if (a, b) != (i, j):
            geo_path = geo_path[::-1]

        # Endpoints in PCA space (from mh, not cache, since cache is
        # the same data but indexed differently)
        si = mh.labels.index(start)
        ei = mh.labels.index(end)
        c_s = mh.centroids_subspace[si]
        c_e = mh.centroids_subspace[ei]

        # Linear path (30 waypoints to match)
        K = geo_path.shape[0]
        ts = np.linspace(0.0, 1.0, K)
        lin_path = (1 - ts)[:, None] * c_s[None, :] + ts[:, None] * c_e[None, :]

        # N-W pullback path
        if start in beh.labels and end in beh.labels:
            try:
                pb = compute_pullback(
                    mh, beh, start, end, num_waypoints=K, geodesic_max_iter=300,
                )
                pb_path = pb.pullback_sub
            except Exception as exc:
                print(f"  pullback failed for {start}↔{end}: {exc}")
                pb_path = None
        else:
            pb_path = None

        ax.contourf(xx, yy, density_grid, levels=20, cmap="Greys", alpha=0.6)

        # All emotion centroids as background points
        ax.scatter(centroids[:, 0], centroids[:, 1], s=8, c="lightgrey",
                   alpha=0.5, edgecolors="none", zorder=2)

        # The paths
        ax.plot(lin_path[:, 0], lin_path[:, 1], "--", color="C0",
                linewidth=2.0, label="linear", zorder=4)
        ax.plot(geo_path[:, 0], geo_path[:, 1], "-", color="C2",
                linewidth=2.0, label="geodesic (G_E)", zorder=5)
        if pb_path is not None:
            ax.plot(pb_path[:, 0], pb_path[:, 1], ":", color="C3",
                    linewidth=2.0, label="N-W pullback", zorder=6)

        # Endpoints
        ax.scatter([c_s[0], c_e[0]], [c_s[1], c_e[1]],
                   s=120, c=["C0", "C3"], edgecolors="black",
                   linewidths=1.0, zorder=7)
        ax.annotate(start, (c_s[0], c_s[1]),
                    xytext=(4, 4), textcoords="offset points", fontsize=9,
                    fontweight="bold")
        ax.annotate(end, (c_e[0], c_e[1]),
                    xytext=(4, 4), textcoords="offset points", fontsize=9,
                    fontweight="bold")

        ax.set_xlabel(f"PC1  ({mh.pca_explained_variance_ratio[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2  ({mh.pca_explained_variance_ratio[1]*100:.1f}%)")
        ax.set_title(f"{start}  →  {end}")
        if ax_idx == 0:
            ax.legend(loc="upper right", fontsize=8)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        print(f"  drew {start}↔{end}")

    for ax in axes[n_pairs:]:
        ax.axis("off")

    fig.suptitle("Geodesic / N-W pullback / linear paths in PC1×PC2\n"
                 "(grey heatmap = KDE density on PC1×PC2; "
                 "geodesics bend through dense regions)", y=1.00)
    plt.tight_layout()
    out_dir = Path("results/riemannian_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "path_visualizations.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {out}")


if __name__ == "__main__":
    main()
