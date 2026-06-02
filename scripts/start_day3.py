"""Initialize results/day3.md with the new 171-emotion manifold's structural properties.

Reads the canonical (now 171-emotion) artifacts and writes a Day 3
skeleton including: PCA spectrum, new bandwidth/NN distance, outlier
centroids by isolation, the isometry check r values, and a summary
comparing 30-emotion → 171-emotion key quantities.

Run with:
    uv run python scripts/start_day3.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from manifold_emotions.behavior.manifold import BehaviorManifold
from manifold_emotions.config import load_config
from manifold_emotions.manifold.fit import FittedManifold
from manifold_emotions.vectors.diff_in_means import EmotionVectors

DAY3 = Path("results/day3.md")


def nn_distances(centroids: np.ndarray) -> np.ndarray:
    diffs = centroids[:, None, :] - centroids[None, :, :]
    dists = np.sqrt(np.sum(diffs * diffs, axis=-1))
    np.fill_diagonal(dists, np.inf)
    return dists.min(axis=1)


def main() -> None:
    config = load_config()
    mh = FittedManifold.load(config.paths.manifold_h)
    my = BehaviorManifold.load(config.paths.manifold_y)
    ev = EmotionVectors.load(config.paths.emotion_vectors)

    n = len(mh.labels)
    mh_nn = nn_distances(mh.centroids_subspace.astype(np.float64))
    my_nn = nn_distances(my.centroids.astype(np.float64))

    # Compare to backed-up 30-emotion artifacts if they exist
    cmp_lines: list[str] = []
    backup = Path("data/30emotions")
    if backup.exists():
        try:
            mh30 = FittedManifold.load(backup / "manifold_h.npz")
            my30 = BehaviorManifold.load(backup / "manifold_y.npz")
            mh30_nn = nn_distances(mh30.centroids_subspace.astype(np.float64))
            my30_nn = nn_distances(my30.centroids.astype(np.float64))
            cmp_lines = [
                "## 30-emotion vs 171-emotion at a glance",
                "",
                "| quantity | 30 emotions | 171 emotions |",
                "|---|---:|---:|",
                f"| # centroids | {len(mh30.labels)} | {n} |",
                f"| M_h subspace dim (config) | {mh30.num_components} | {mh.num_components} |",
                f"| M_h KDE bandwidth (median NN) | {mh30.kde_bandwidth:.3f} | {mh.kde_bandwidth:.3f} |",
                f"| M_h mean NN distance | {float(mh30_nn.mean()):.3f} | {float(mh_nn.mean()):.3f} |",
                f"| M_y NN distance (median) | {float(np.median(my30_nn)):.3f} | {float(np.median(my_nn)):.3f} |",
                f"| Top-1 PCA explained variance | {float(mh30.pca_explained_variance_ratio[0]):.3f} | {float(mh.pca_explained_variance_ratio[0]):.3f} |",
                f"| Cumulative variance ({mh.num_components}-D) | {float(mh30.pca_explained_variance_ratio.sum()):.3f} | {float(mh.pca_explained_variance_ratio.sum()):.3f} |",
                "",
            ]
        except Exception as exc:
            cmp_lines = [f"(30-emotion backup load failed: {exc})\n"]

    # Outlier centroids
    order = np.argsort(-mh_nn)
    outlier_rows = "\n".join(
        f"| {mh.labels[i]} | {mh_nn[i]:.3f} | {mh_nn[i]/mh.kde_bandwidth:.2f} |"
        for i in order[:10]
    )

    pca_rows = "\n".join(
        f"| PC{i+1} | {float(mh.pca_explained_variance_ratio[i]):.3f} |"
        for i in range(mh.num_components)
    )

    # M_y bounds
    my_bounds = (
        f"valence range: [{float(my.centroids[:, 0].min()):.2f}, "
        f"{float(my.centroids[:, 0].max()):.2f}]; "
        f"arousal range: [{float(my.centroids[:, 1].min()):.2f}, "
        f"{float(my.centroids[:, 1].max()):.2f}]"
    )

    # Isometry check, if results exist (the chain runs check_isometry.py)
    iso_lines: list[str] = []
    iso_path = Path("results/isometry.json")
    if iso_path.exists():
        try:
            iso = json.loads(iso_path.read_text())
            iso_lines = [
                "## Isometry check",
                "",
                f"- Pearson r between M_h and M_y pairwise distances: "
                f"{iso.get('pearson_r', 'n/a')}",
                f"- Spearman ρ: {iso.get('spearman_rho', 'n/a')}",
                "",
            ]
        except Exception:
            pass

    body = f"""# Day 3 — 171-emotion scale-up

Day 2 closed with the corpus generation finishing at 00:32:56 and the
scale chain picking up automatically. This doc tracks what we learned
once the new manifold was fitted.

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

{chr(10).join(cmp_lines)}
## M_h structural properties (171 emotions)

- Bandwidth (median NN in subspace): **{mh.kde_bandwidth:.3f}**
- Mean NN distance in subspace: {float(mh_nn.mean()):.3f}
- M_y NN distance distribution:
  - min: {float(my_nn.min()):.3f}
  - median: {float(np.median(my_nn)):.3f}
  - mean: {float(my_nn.mean()):.3f}
  - max: {float(my_nn.max()):.3f}

### Per-PC explained variance ({mh.num_components}-D subspace)

| PC | explained variance |
|---|---:|
{pca_rows}

cumulative: **{float(mh.pca_explained_variance_ratio.sum()):.3f}**

### Most-isolated centroids (top 10 by NN distance)

| centroid | NN | × bandwidth |
|---|---:|---:|
{outlier_rows}

(Compare to 30-emotion outliers: `relaxed` 2.77×, `enthusiastic`
2.42×, `desperate` 2.25×, `energized` 2.21×, `weary` 1.96×, `calm`
1.70×, `frustrated` 1.46×, `sad` 1.44×. The denser corpus may have
shrunk these ratios since the bandwidth itself shrinks.)

## M_y at 171-scale

{my_bounds}

The 171-emotion M_y populates the affective circumplex more densely
than the 30-emotion set. Whether Anthropic's ~10 k-means cluster
structure emerges visibly is the first thing to check in the
dashboard once geodesics are precomputed.

{chr(10).join(iso_lines)}

## What's running

After the scale chain completes, `scripts/scale_chain_post.sh` kicks
off two more workloads in parallel:

1. **Pullback at 171-scale on the 4 existing pairs** (excited→weary,
   depressed→energized, happy→sad, terrified→serene). Split across
   both vLLMs, ~50 min wall time. Tests Day 2's predictions:
   - Smoother pullback paths (denser kernel neighborhoods)
   - Possibly closer to geodesic in shape
   - Narrower margin over linear (linear baseline improves)
   - Terrified→serene resolution test: sample-sparsity vs
     kernel-uninvertibility hypothesis

2. **Geodesic precompute** for all C(171, 2) = 14,535 pairs. ~4 hours
   on CPU. Needed for the dashboard at 171-scale.

Results appended below as they land.

## 171-scale pullback results

(pending — appended by scripts/append_pullback_to_day3.py)
"""

    DAY3.write_text(body)
    print(f"wrote {DAY3} ({len(body)} bytes)")


if __name__ == "__main__":
    main()
