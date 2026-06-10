"""Diagnostic: what does each path actually traverse for a given pair?

Given an emotion pair, fits both the M_h geodesic and the M_h linear
interpolation, walks each in lockstep, and reports the nearest emotion
centroid at every waypoint along with its (V, A) coordinates. The goal
is to see *why* a pair is or isn't manifold-favored without running any
LLM calls.

Specifically built to compare serene↔terrified (gap=1.57, but Δ=−0.024
— the lone unexplained linear-favored case in the Goldilocks band) to
calm↔desperate (gap=2.15, Δ=+0.112) and excited↔weary (gap=2.53,
Δ=+0.091) — pairs at similar gap regime but opposite outcomes.

The key question: do the manifold geodesic and the linear chord visit
*different* centroids, or do they both wander through the same
neighborhoods? If they visit the same neighborhoods, manifold steering
can't produce different text from linear and Δ collapses.

Run with:
    uv run python scripts/diagnose_pair.py serene terrified
    uv run python scripts/diagnose_pair.py calm desperate
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from manifold_emotions.behavior.manifold import BehaviorManifold
from manifold_emotions.config import load_config
from manifold_emotions.manifold.fit import FittedManifold
from manifold_emotions.manifold.geodesic import fit_geodesic, linear_interpolation

OUT_DIR = Path("results/figures/pair_diagnose")


def nearest_centroids(
    waypoints: np.ndarray, manifold: FittedManifold
) -> tuple[np.ndarray, np.ndarray]:
    """For each waypoint, return (index, distance) of the nearest centroid."""
    cs = manifold.centroids_subspace.astype(np.float64)
    diffs = waypoints[:, None, :].astype(np.float64) - cs[None, :, :]
    dists = np.sqrt(np.sum(diffs * diffs, axis=-1))
    idx = np.argmin(dists, axis=1)
    nearest_d = dists[np.arange(waypoints.shape[0]), idx]
    return idx, nearest_d


def compact_trace(labels: list[str]) -> list[tuple[str, int]]:
    """Collapse consecutive duplicates: ['a','a','b','b','b','a'] →
    [('a',2),('b',3),('a',1)]."""
    out: list[tuple[str, int]] = []
    for lab in labels:
        if out and out[-1][0] == lab:
            out[-1] = (lab, out[-1][1] + 1)
        else:
            out.append((lab, 1))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("start")
    parser.add_argument("end")
    parser.add_argument("--num-waypoints", type=int, default=30)
    args = parser.parse_args()

    config = load_config()
    mh = FittedManifold.load(config.paths.manifold_h)
    my = BehaviorManifold.load(config.paths.manifold_y)

    if args.start not in mh.labels or args.end not in mh.labels:
        raise SystemExit(f"missing label in M_h: {args.start} or {args.end}")

    geometry = mh.make_geometry()
    i = mh.labels.index(args.start)
    j = mh.labels.index(args.end)
    start_sub = mh.centroids_subspace[i].astype(np.float32)
    end_sub = mh.centroids_subspace[j].astype(np.float32)

    geo = fit_geodesic(geometry, start_sub, end_sub,
                       num_waypoints=args.num_waypoints, max_iter=300)
    lin = linear_interpolation(start_sub, end_sub, args.num_waypoints).astype(np.float32)

    geo_idx, geo_d = nearest_centroids(geo.waypoints, mh)
    lin_idx, lin_d = nearest_centroids(lin, mh)

    geo_labels = [mh.labels[k] for k in geo_idx]
    lin_labels = [mh.labels[k] for k in lin_idx]

    # M_y lookups by label
    my_by_label = {lab: my.centroids[k] for k, lab in enumerate(my.labels)}
    y_start = my_by_label.get(args.start)
    y_end = my_by_label.get(args.end)
    chord = y_end - y_start
    chord_len = float(np.linalg.norm(chord))
    chord_unit = chord / chord_len

    def my_offset(lab: str) -> float:
        """Perpendicular offset of `lab`'s M_y coord from the chord line."""
        y = my_by_label.get(lab)
        if y is None:
            return float("nan")
        rel = y - y_start
        proj = float(rel @ chord_unit) * chord_unit
        return float(np.linalg.norm(rel - proj))

    print(f"=== {args.start} → {args.end} ===")
    print(f"  M_h gap (linear − geodesic) under G_E: "
          f"{geo.initial_length - geo.final_length:+.3f}")
    print(f"  mean nearest-centroid distance (subspace): "
          f"geodesic={geo_d.mean():.2f}  linear={lin_d.mean():.2f}")
    print(f"  M_y chord length: {chord_len:.2f}")
    print()

    print("Per-waypoint nearest centroid (compact, with V-A offset from chord):")
    print(f"  {'k':>2}  {'method':>8}  {'nearest':>14}  "
          f"{'sub_dist':>8}  {'my_offset':>9}")
    for k in range(args.num_waypoints):
        if k % 4 != 0 and k not in (args.num_waypoints - 1,):
            continue
        gl = geo_labels[k]
        ll = lin_labels[k]
        print(
            f"  {k:>2}  {'geodesic':>8}  {gl:>14}  {geo_d[k]:>8.2f}  "
            f"{my_offset(gl):>9.3f}"
        )
        print(
            f"  {k:>2}  {'linear':>8}  {ll:>14}  {lin_d[k]:>8.2f}  "
            f"{my_offset(ll):>9.3f}"
        )

    print()
    print("Compact nearest-centroid trace (consecutive duplicates collapsed):")
    geo_compact = compact_trace(geo_labels)
    lin_compact = compact_trace(lin_labels)
    print(f"  geodesic: {' → '.join(f'{l}×{c}' for l, c in geo_compact)}")
    print(f"  linear:   {' → '.join(f'{l}×{c}' for l, c in lin_compact)}")
    print()

    geo_set = Counter(geo_labels)
    lin_set = Counter(lin_labels)
    shared = sorted(set(geo_set) & set(lin_set))
    only_geo = sorted(set(geo_set) - set(lin_set))
    only_lin = sorted(set(lin_set) - set(geo_set))
    print(f"  centroids visited by both: {shared}")
    print(f"  centroids visited only by geodesic: {only_geo}")
    print(f"  centroids visited only by linear:   {only_lin}")
    print()

    geo_offsets = [my_offset(l) for l in geo_labels]
    lin_offsets = [my_offset(l) for l in lin_labels]
    print(f"  mean perpendicular offset of nearest-centroid M_y trace "
          f"from M_y chord:")
    print(f"    geodesic = {np.nanmean(geo_offsets):.3f}")
    print(f"    linear   = {np.nanmean(lin_offsets):.3f}")

    # Plot: M_y straight line + the two nearest-centroid traces.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.scatter(my.centroids[:, 0], my.centroids[:, 1], c="lightgray", s=20, zorder=1)
    for k, lab in enumerate(my.labels):
        ax.annotate(lab, my.centroids[k], fontsize=6, color="gray", alpha=0.6)
    # Chord
    ax.plot([y_start[0], y_end[0]], [y_start[1], y_end[1]],
            "-", color="black", linewidth=1.5, alpha=0.6, label="M_y chord")
    # Geodesic nearest-centroid trace
    geo_my = np.stack([my_by_label[l] for l in geo_labels], axis=0)
    lin_my = np.stack([my_by_label[l] for l in lin_labels], axis=0)
    ax.plot(geo_my[:, 0], geo_my[:, 1], "-o", color="#0066cc",
            linewidth=1.5, markersize=4, label="geodesic nearest-centroid trace")
    ax.plot(lin_my[:, 0], lin_my[:, 1], "-s", color="#888888",
            linewidth=1.5, markersize=4, alpha=0.7,
            label="linear nearest-centroid trace")
    ax.scatter([y_start[0]], [y_start[1]], c="green", s=80, zorder=5,
               label="start")
    ax.scatter([y_end[0]], [y_end[1]], c="red", s=80, marker="X", zorder=5,
               label="end")
    ax.set_xlim(1, 7)
    ax.set_ylim(1, 7)
    ax.set_xlabel("valence")
    ax.set_ylabel("arousal")
    ax.set_title(f"{args.start} → {args.end}: nearest-centroid M_y traces")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.scatter(mh.centroids_subspace[:, 0], mh.centroids_subspace[:, 1],
               c="lightgray", s=20, zorder=1)
    for k, lab in enumerate(mh.labels):
        ax.annotate(lab, mh.centroids_subspace[k, :2], fontsize=6, color="gray", alpha=0.6)
    ax.plot(lin[:, 0], lin[:, 1], "--", color="#888888", linewidth=1.5, label="linear")
    ax.plot(geo.waypoints[:, 0], geo.waypoints[:, 1], "-", color="#0066cc",
            linewidth=2, label="geodesic")
    ax.scatter([start_sub[0]], [start_sub[1]], c="green", s=80, zorder=5)
    ax.scatter([end_sub[0]], [end_sub[1]], c="red", s=80, marker="X", zorder=5)
    ax.set_xlabel("M_h PC1")
    ax.set_ylabel("M_h PC2")
    ax.set_title(f"{args.start} → {args.end}: M_h paths (PC1, PC2)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = OUT_DIR / f"{args.start}_{args.end}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
