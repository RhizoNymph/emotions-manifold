"""Fit a density-geometry manifold to the extracted emotion vectors.

Loads ``data/emotion_vectors.npz``, projects the vectors into a 64-D
PCA subspace, fits a Gaussian KDE, and writes the packaged manifold
to ``config.paths.manifold_h``.

Also exercises a few sample geodesics (happy↔sad, calm↔desperate) so
you can eyeball whether the optimization is producing sane curves
before committing to Phase 5 steering experiments.

Run with:
    uv run python scripts/fit_manifold.py
"""

from __future__ import annotations

import numpy as np

from manifold_emotions.config import load_config
from manifold_emotions.manifold.fit import fit_manifold
from manifold_emotions.manifold.geodesic import fit_geodesic
from manifold_emotions.vectors.diff_in_means import EmotionVectors

SAMPLE_PAIRS = [
    ("happy", "sad"),
    ("calm", "desperate"),
    ("excited", "weary"),
]


def main() -> None:
    config = load_config()
    ev = EmotionVectors.load(config.paths.emotion_vectors)

    num_components = min(config.extraction.pca_subspace_dim, ev.vectors.shape[0] - 1)
    manifold, pca = fit_manifold(ev, num_components=num_components)
    manifold.save(config.paths.manifold_h)

    print(
        f"saved fitted manifold to {config.paths.manifold_h}: "
        f"{manifold.num_components} PCA dims, "
        f"{len(manifold.labels)} emotion centroids, "
        f"hidden_size={manifold.hidden_size}, "
        f"bandwidth={manifold.kde_bandwidth:.4f}"
    )
    print(
        f"top-5 PCA explained variance: "
        f"{[round(r, 3) for r in pca.explained_variance_ratio[:5].tolist()]}"
    )
    cumulative = float(pca.explained_variance_ratio.sum())
    print(f"cumulative explained variance in {manifold.num_components}-D: {cumulative:.1%}")
    print()

    # Sanity check a few geodesics. Skips pairs whose labels weren't extracted
    # (e.g. the shakedown is missing emotions present in SAMPLE_PAIRS).
    label_to_idx = {label: i for i, label in enumerate(manifold.labels)}
    geometry = manifold.make_geometry()

    for start_label, end_label in SAMPLE_PAIRS:
        if start_label not in label_to_idx or end_label not in label_to_idx:
            print(f"skipping {start_label}↔{end_label} (missing from extracted set)")
            continue
        start = manifold.centroids_subspace[label_to_idx[start_label]]
        end = manifold.centroids_subspace[label_to_idx[end_label]]
        result = fit_geodesic(
            geometry,
            start.astype(np.float32),
            end.astype(np.float32),
            num_waypoints=20,
            max_iter=300,
        )
        ratio = (
            result.final_length / result.initial_length
            if result.initial_length > 0
            else float("nan")
        )
        # Find nearest centroid to each interior waypoint — gives a quick
        # human-readable trace of which emotions the geodesic passes near.
        nearest_path: list[str] = []
        for wp in result.waypoints:
            dists = np.linalg.norm(manifold.centroids_subspace - wp[None, :], axis=1)
            nearest_path.append(manifold.labels[int(np.argmin(dists))])
        # Compact: collapse consecutive duplicates.
        compact: list[str] = []
        for label in nearest_path:
            if not compact or compact[-1] != label:
                compact.append(label)

        print(
            f"{start_label} → {end_label}: "
            f"length {result.final_length:.3f} / linear {result.initial_length:.3f} "
            f"= {ratio:.2f}, iters={result.num_iterations}, "
            f"converged={result.converged}"
        )
        print(f"  nearest centroids: {' → '.join(compact)}")


if __name__ == "__main__":
    main()
