"""Fit a thin-plate-spline manifold phi: (2-D surface coord) -> PCA subspace.

Takes an existing FittedManifold (for the PCA subspace centroids + lift + KDE
hyperparameters), builds a 2-D surface coordinate for each emotion, solves the
TPS, and saves a self-contained ``SplineManifold`` artifact that the chord
experiment can load to build spline-geodesic steering trajectories.

Two parameterizations select where the 2-D control coordinates come from:

- ``valence_arousal`` (default): the emotion's (V, A) point in the behavior
  manifold M_y. This readout is many-to-one (~14% of emotions collide in V/A),
  so the fit is lossy (residual ~6 at 8-D). Default smoothing 0.5.
- ``diffusion``: a bijective diffusion-2 coordinate of the activation centroids
  (``manifold.diffusion.diffusion_embed`` — the SAME construction the geometry
  check uses), one distinct point per emotion, so smoothing=0.0 interpolates the
  centroids essentially exactly (residual ~0.03). This is the faithful Goodfire
  analog and the artifact behind the behavioral bijective-spline condition.

Examples:
    uv run python scripts/fit_spline_manifold.py \
        --source-manifold data/manifold_h.npz --tag 8d --smoothing 0.1

    uv run python scripts/fit_spline_manifold.py \
        --source-manifold data/manifold_h.npz --parameterization diffusion \
        --out data/manifold_spline_bijective_8d.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from manifold_emotions.behavior.manifold import BehaviorManifold
from manifold_emotions.manifold.diffusion import diffusion_embed
from manifold_emotions.manifold.fit import FittedManifold
from manifold_emotions.manifold.spline import SplineManifold

# Default smoothing per parameterization: V/A needs regularization (many-to-one,
# ill-conditioned); the bijective diffusion coord interpolates exactly at lam=0.
_DEFAULT_SMOOTHING = {"valence_arousal": 0.5, "diffusion": 0.0}


def _control_coords(
    parameterization: str,
    manifold: FittedManifold,
    behavior_path: Path,
) -> tuple[np.ndarray, str]:
    """Build (N, 2) surface control coords in ``manifold.labels`` order.

    Returns ``(control_coords, artifact_parameterization_tag)``.
    """
    match parameterization:
        case "valence_arousal":
            behavior = BehaviorManifold.load(behavior_path)
            by_label = {lab: i for i, lab in enumerate(behavior.labels)}
            missing = [lab for lab in manifold.labels if lab not in by_label]
            if missing:
                raise SystemExit(
                    f"behavior manifold missing {len(missing)} labels: {missing[:5]}"
                )
            order = [by_label[lab] for lab in manifold.labels]
            return behavior.centroids[order].astype(np.float64), "valence_arousal"
        case "diffusion":
            # Match bijective_spline_check.py exactly: diffusion-2 of the
            # activation centroids (float64), one distinct point per emotion.
            centroids = manifold.centroids_subspace.astype(np.float64)
            return diffusion_embed(centroids, 2), "diffusion_map_2"
        case _:
            raise SystemExit(f"unknown parameterization: {parameterization!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-manifold", required=True, type=Path)
    ap.add_argument("--behavior", type=Path, default=Path("data/manifold_y.npz"))
    ap.add_argument(
        "--tag", default=None, help="artifact tag, e.g. 8d / 4d (used for default --out)"
    )
    ap.add_argument(
        "--parameterization",
        choices=("valence_arousal", "diffusion"),
        default="valence_arousal",
        help="source of the 2-D surface control coordinates",
    )
    ap.add_argument(
        "--smoothing",
        type=float,
        default=None,
        help="TPS lambda; default depends on --parameterization "
        "(valence_arousal=0.5, diffusion=0.0)",
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.tag is None and args.out is None:
        raise SystemExit("provide --tag or --out for the artifact path")

    smoothing = (
        args.smoothing
        if args.smoothing is not None
        else _DEFAULT_SMOOTHING[args.parameterization]
    )

    manifold = FittedManifold.load(args.source_manifold)
    control_coords, param_tag = _control_coords(
        args.parameterization, manifold, args.behavior
    )

    spline = SplineManifold.fit(
        labels=manifold.labels,
        control_coords=control_coords,
        centroids_subspace=manifold.centroids_subspace,
        pca_components=manifold.pca_components,
        pca_mean=manifold.pca_mean,
        kde_bandwidth=manifold.kde_bandwidth,
        alpha=manifold.alpha,
        beta=manifold.beta,
        smoothing=smoothing,
        parameterization=param_tag,
    )

    out = args.out or Path(f"data/manifold_spline_{args.tag}.npz")
    spline.save(out)

    # Report interpolation residual as a fit-quality sanity check.
    embedded = spline.embed_np(control_coords.astype(np.float32))
    resid = np.linalg.norm(embedded - manifold.centroids_subspace, axis=1)
    scale = np.linalg.norm(manifold.centroids_subspace, axis=1).mean()
    print(
        f"saved {out}  (d={spline.num_components}, N={len(spline.labels)}, "
        f"parameterization={param_tag}, smoothing={smoothing})"
    )
    print(
        f"interpolation residual to centroids: mean {resid.mean():.4f}  "
        f"max {resid.max():.4f}  (subspace scale ~{scale:.1f})"
    )


if __name__ == "__main__":
    main()
