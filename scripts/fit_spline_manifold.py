"""Fit a thin-plate-spline manifold phi: (valence, arousal) -> PCA subspace.

Takes an existing FittedManifold (for the PCA subspace centroids + lift + KDE
hyperparameters) and the behavior manifold M_y (for the V/A control coordinates),
solves the TPS, and saves a self-contained ``SplineManifold`` artifact that the
chord experiment can load to build spline-geodesic steering trajectories.

Example:
    uv run python scripts/fit_spline_manifold.py \
        --source-manifold data/manifold_h.npz --tag 8d --smoothing 0.5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from manifold_emotions.behavior.manifold import BehaviorManifold
from manifold_emotions.manifold.fit import FittedManifold
from manifold_emotions.manifold.spline import SplineManifold


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-manifold", required=True, type=Path)
    ap.add_argument("--behavior", type=Path, default=Path("data/manifold_y.npz"))
    ap.add_argument("--tag", required=True, help="artifact tag, e.g. 8d / 4d")
    ap.add_argument("--smoothing", type=float, default=0.5)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    manifold = FittedManifold.load(args.source_manifold)
    behavior = BehaviorManifold.load(args.behavior)

    # Align V/A control coords to the manifold's label order.
    by_label = {lab: i for i, lab in enumerate(behavior.labels)}
    missing = [lab for lab in manifold.labels if lab not in by_label]
    if missing:
        raise SystemExit(f"behavior manifold missing {len(missing)} labels: {missing[:5]}")
    order = [by_label[lab] for lab in manifold.labels]
    control_coords = behavior.centroids[order].astype(np.float64)

    spline = SplineManifold.fit(
        labels=manifold.labels,
        control_coords=control_coords,
        centroids_subspace=manifold.centroids_subspace,
        pca_components=manifold.pca_components,
        pca_mean=manifold.pca_mean,
        kde_bandwidth=manifold.kde_bandwidth,
        alpha=manifold.alpha,
        beta=manifold.beta,
        smoothing=args.smoothing,
    )

    out = args.out or Path(f"data/manifold_spline_{args.tag}.npz")
    spline.save(out)

    # Report interpolation residual as a fit-quality sanity check.
    embedded = spline.embed_np(control_coords.astype(np.float32))
    resid = np.linalg.norm(embedded - manifold.centroids_subspace, axis=1)
    print(f"saved {out}  (d={spline.num_components}, N={len(spline.labels)}, smoothing={args.smoothing})")
    print(
        f"interpolation residual to centroids: mean {resid.mean():.4f}  "
        f"max {resid.max():.4f}  (subspace scale ~{np.linalg.norm(manifold.centroids_subspace, axis=1).mean():.1f})"
    )


if __name__ == "__main__":
    main()
