"""Fit a density-geometry manifold to the extracted emotion vectors.

Loads ``data/emotion_vectors.npz``, projects the vectors into a PCA
subspace, fits a Gaussian KDE, and writes the packaged manifold. With
``--geodesics`` it also precomputes the all-pairs geodesic waypoint
cache that the steering experiments and dashboard consume.

This is the single entry point for every manifold variant — it replaces
the per-variant setup scripts (setup_4d_pipeline.py, setup_6d_pipeline.py,
setup_8d_silverman_pipeline.py, archived on archive/disorganized-scripts):

    # production manifold (config dims + clustered_nn bandwidth, config paths)
    uv run python scripts/fit_manifold.py

    # variant manifolds: tag-suffixed artifacts, production files untouched
    uv run python scripts/fit_manifold.py --dim 4 --tag 4d --geodesics
    uv run python scripts/fit_manifold.py --dim 6 --tag 6d --geodesics
    uv run python scripts/fit_manifold.py --dim 8 --bandwidth silverman \
        --tag 8d_silverman --geodesics

Tagged runs write ``data/manifold_h_{tag}.npz`` and
``data/geodesics_cache_{tag}.npz`` (override with --out-manifold /
--out-geodesics; note the legacy 4-D/6-D artifacts were named
``manifold_h_{4d,6d}_full.npz``). Without --geodesics a few sample
geodesics are fit and printed as a sanity check.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from manifold_emotions.config import load_config
from manifold_emotions.manifold.fit import BandwidthSpec, FittedManifold, fit_manifold
from manifold_emotions.manifold.geodesic import fit_geodesic
from manifold_emotions.manifold.geodesic_cache import build_geodesic_cache
from manifold_emotions.vectors.diff_in_means import EmotionVectors

SAMPLE_PAIRS = [
    ("happy", "sad"),
    ("calm", "desperate"),
    ("excited", "weary"),
]


def _parse_bandwidth(raw: str) -> BandwidthSpec:
    if raw in ("clustered_nn", "silverman"):
        return raw
    try:
        return float(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--bandwidth must be 'clustered_nn', 'silverman', or a float; got {raw!r}"
        ) from None


def _sanity_check_geodesics(
    manifold: FittedManifold, num_waypoints: int, max_iter: int
) -> None:
    """Fit a few sample geodesics and print which centroids they pass near."""
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
            num_waypoints=num_waypoints,
            max_iter=max_iter,
        )
        ratio = (
            result.final_length / result.initial_length
            if result.initial_length > 0
            else float("nan")
        )
        # Nearest centroid per waypoint, consecutive duplicates collapsed —
        # a quick human-readable trace of the geodesic's route.
        nearest_path: list[str] = []
        for wp in result.waypoints:
            dists = np.linalg.norm(manifold.centroids_subspace - wp[None, :], axis=1)
            nearest_path.append(manifold.labels[int(np.argmin(dists))])
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dim", type=int, default=None,
        help="PCA subspace dimensionality (default: config extraction.pca_subspace_dim)",
    )
    parser.add_argument(
        "--bandwidth", type=_parse_bandwidth, default="clustered_nn",
        help="KDE bandwidth: 'clustered_nn' (default), 'silverman', or a float",
    )
    parser.add_argument(
        "--tag", default=None,
        help="suffix for output artifacts (e.g. '4d', '8d_silverman'); "
        "default: production paths from config",
    )
    parser.add_argument(
        "--out-manifold", type=Path, default=None,
        help="explicit manifold output path (overrides --tag naming)",
    )
    parser.add_argument(
        "--out-geodesics", type=Path, default=None,
        help="explicit geodesic-cache output path (overrides --tag naming)",
    )
    parser.add_argument(
        "--geodesics", action="store_true",
        help="precompute the all-pairs geodesic waypoint cache",
    )
    parser.add_argument("--num-waypoints", type=int, default=30)
    parser.add_argument("--max-iter", type=int, default=300)
    args = parser.parse_args()

    config = load_config()
    ev = EmotionVectors.load(config.paths.emotion_vectors)

    requested_dim = args.dim if args.dim is not None else config.extraction.pca_subspace_dim
    num_components = min(requested_dim, ev.vectors.shape[0] - 1)

    if args.out_manifold is not None:
        manifold_path = args.out_manifold
    elif args.tag is not None:
        manifold_path = Path(f"data/manifold_h_{args.tag}.npz")
    else:
        manifold_path = config.paths.manifold_h

    if args.out_geodesics is not None:
        geodesics_path = args.out_geodesics
    elif args.tag is not None:
        geodesics_path = Path(f"data/geodesics_cache_{args.tag}.npz")
    else:
        geodesics_path = Path("data/geodesics_cache.npz")

    print(f"Loaded {len(ev.labels)} emotion vectors")
    manifold, pca = fit_manifold(
        ev, num_components=num_components, bandwidth=args.bandwidth
    )
    manifold.save(manifold_path)

    print(
        f"saved fitted manifold to {manifold_path}: "
        f"{manifold.num_components} PCA dims, "
        f"{len(manifold.labels)} emotion centroids, "
        f"hidden_size={manifold.hidden_size}, "
        f"bandwidth={manifold.kde_bandwidth:.4f} ({args.bandwidth})"
    )
    print(
        f"top-5 PCA explained variance: "
        f"{[round(r, 3) for r in pca.explained_variance_ratio[:5].tolist()]}"
    )
    cumulative = float(pca.explained_variance_ratio.sum())
    print(f"cumulative explained variance in {manifold.num_components}-D: {cumulative:.1%}")
    print()

    if args.geodesics:
        n = len(manifold.labels)
        print(f"Precomputing geodesics for {n * (n - 1) // 2} pairs...")
        cache = build_geodesic_cache(
            manifold,
            num_waypoints=args.num_waypoints,
            max_iter=args.max_iter,
            progress=print,
        )
        cache.save(geodesics_path)
        print(
            f"saved {geodesics_path}: {cache.waypoints.shape} float32 "
            f"({cache.waypoints.nbytes / 1e6:.1f} MB)"
        )
    else:
        _sanity_check_geodesics(manifold, num_waypoints=20, max_iter=args.max_iter)


if __name__ == "__main__":
    main()
