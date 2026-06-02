"""Run the M_h ↔ M_y isometry check — the gate to Phase 5 (steering).

Loads the fitted activation manifold, the behavior manifold, and the
raw emotion vectors, then computes:
- pairwise distances on M_h in the PCA subspace
- pairwise distances on M_h in full 5376-D activation space (the
  flat / "linear steering" baseline)
- pairwise distances on M_y in (valence, arousal)
- Pearson + Spearman correlation between each M_h variant and M_y

Goodfire baseline (simple 1-D conceptual spaces): r≈0.99 for the manifold
metric vs r=0.36-0.89 for linear. For our higher-d emotion manifold,
anything north of ~0.7 for the subspace metric is encouraging; the
KEY check is that the subspace correlation substantially beats the
linear baseline.

Run with:
    uv run python scripts/check_isometry.py
"""

from __future__ import annotations

import numpy as np

from manifold_emotions.behavior.manifold import BehaviorManifold
from manifold_emotions.config import load_config
from manifold_emotions.isometry import check_isometry
from manifold_emotions.manifold.fit import FittedManifold
from manifold_emotions.vectors.diff_in_means import EmotionVectors


def _common_label_order(*label_tuples: tuple[str, ...]) -> tuple[str, ...]:
    """Intersection of labels across inputs, in sorted order."""
    common = set(label_tuples[0])
    for labels in label_tuples[1:]:
        common &= set(labels)
    return tuple(sorted(common))


def main() -> None:
    config = load_config()
    ev = EmotionVectors.load(config.paths.emotion_vectors)
    mh = FittedManifold.load(config.paths.manifold_h)
    my = BehaviorManifold.load(config.paths.manifold_y)

    labels = _common_label_order(ev.labels, mh.labels, my.labels)
    print(
        f"common emotions across M_h and M_y: {len(labels)} "
        f"(M_h has {len(mh.labels)}, M_y has {len(my.labels)})"
    )

    def reorder(arr: np.ndarray, src_labels: tuple[str, ...]) -> np.ndarray:
        idx = [src_labels.index(label) for label in labels]
        return arr[idx]

    m_h_subspace = reorder(mh.centroids_subspace, mh.labels)
    m_h_full = reorder(ev.centroids, ev.labels)  # raw centroids in full activation space
    m_y = reorder(my.centroids, my.labels)

    report = check_isometry(
        labels=labels,
        m_h_subspace_centroids=m_h_subspace,
        m_h_full_centroids=m_h_full,
        m_y_centroids=m_y,
    )

    print()
    print("Pairwise-distance correlations (M_h ↔ M_y):")
    print(
        f"  Pearson  — subspace: {report.pearson_subspace_vs_behavior:+.3f}  "
        f"linear: {report.pearson_linear_vs_behavior:+.3f}"
    )
    print(
        f"  Spearman — subspace: {report.spearman_subspace_vs_behavior:+.3f}  "
        f"linear: {report.spearman_linear_vs_behavior:+.3f}"
    )

    print()
    delta_pearson = (
        report.pearson_subspace_vs_behavior - report.pearson_linear_vs_behavior
    )
    if report.pearson_subspace_vs_behavior > 0.7:
        verdict = "GOOD"
    elif report.pearson_subspace_vs_behavior > 0.5:
        verdict = "BORDERLINE"
    else:
        verdict = "WEAK"
    print(
        f"verdict: {verdict} — subspace metric beats linear by "
        f"{delta_pearson:+.3f} on Pearson distance correlation."
    )


if __name__ == "__main__":
    main()
