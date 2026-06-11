"""Visualize the pullback experiment.

For each pair with results, produces one figure with three panels:

1. M_h subspace projection (PC1, PC2): pullback, geodesic, linear, plus
   the emotion centroids that fall within the visible window. Shows
   whether the pullback bends like the geodesic.
2. M_y behavior trace: each trajectory's per-waypoint (V, A) overlaid
   on the target M_y straight line. Shows which path's behavior hugs
   the M_y geodesic most tightly.
3. Per-waypoint distance to the M_y straight line, bar chart by method.

Run with:
    uv run python scripts/plot_pullback.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from manifold_emotions.behavior.manifold import BehaviorManifold
from manifold_emotions.config import load_config
from manifold_emotions.manifold.fit import FittedManifold

RESULTS_DIR = Path("results/pullback")
DATA_DIR = Path("data/pullback")
FIG_DIR = Path("results/figures/pullback")

METHOD_COLOR = {
    "pullback": "#9933cc",
    "geodesic": "#0066cc",
    "linear":   "#999999",
}


def plot_one_pair(
    summary: dict,
    paths_npz: Path,
    manifold: FittedManifold,
    behavior: BehaviorManifold,
) -> None:
    start, end = summary["pair"]
    paths = np.load(paths_npz)
    my_path = paths["my_path"]
    pullback_sub = paths["pullback_sub"]
    geodesic_sub = paths["geodesic_sub"]
    linear_sub = paths["linear_sub"]

    fig = plt.figure(figsize=(15, 5))

    # Panel 1: M_h subspace projected onto (PC1, PC2). Manifold's first two
    # components carry valence/arousal-aligned signal (per validation),
    # so this projection is roughly the affective circumplex view of M_h.
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.scatter(
        manifold.centroids_subspace[:, 0], manifold.centroids_subspace[:, 1],
        c="lightgray", s=20, zorder=1,
    )
    for i, label in enumerate(manifold.labels):
        if label in (start, end):
            ax1.annotate(label, manifold.centroids_subspace[i, :2], fontsize=8, color="black")

    ax1.plot(pullback_sub[:, 0], pullback_sub[:, 1], "-", color=METHOD_COLOR["pullback"],
             linewidth=2, label="pullback", zorder=3)
    ax1.plot(geodesic_sub[:, 0], geodesic_sub[:, 1], "-", color=METHOD_COLOR["geodesic"],
             linewidth=2, label="geodesic", zorder=3)
    ax1.plot(linear_sub[:, 0], linear_sub[:, 1], "--", color=METHOD_COLOR["linear"],
             linewidth=1.5, label="linear", zorder=2)
    ax1.scatter(pullback_sub[[0, -1], 0], pullback_sub[[0, -1], 1], color="black", s=40, zorder=4)
    ax1.set_xlabel("M_h PC1")
    ax1.set_ylabel("M_h PC2")
    ax1.set_title(f"{start} → {end}: M_h paths (PC1, PC2)")
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel 2: M_y behavior trace
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.scatter(behavior.centroids[:, 0], behavior.centroids[:, 1], c="lightgray", s=20, zorder=1)
    ax2.plot(my_path[:, 0], my_path[:, 1], "-", color="black", linewidth=1, alpha=0.5,
             label="M_y geodesic (target)", zorder=2)
    for name in ("pullback", "geodesic", "linear"):
        traj = summary["trajectories"][name]
        v = np.array(traj["waypoint_valence"])
        a = np.array(traj["waypoint_arousal"])
        ax2.plot(v, a, "-o", color=METHOD_COLOR[name], linewidth=1.5,
                 markersize=4, label=name, zorder=3)
    ax2.scatter([summary["geometry"]["my_path_valence"][0]],
                [summary["geometry"]["my_path_arousal"][0]],
                marker="o", color="green", s=50, zorder=5)
    ax2.scatter([summary["geometry"]["my_path_valence"][-1]],
                [summary["geometry"]["my_path_arousal"][-1]],
                marker="X", color="red", s=60, zorder=5)
    ax2.set_xlim(1, 7)
    ax2.set_ylim(1, 7)
    ax2.set_xlabel("valence")
    ax2.set_ylabel("arousal")
    ax2.set_title(f"{start} → {end}: behavior traces in M_y")
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Panel 3: summary metrics bar chart
    ax3 = fig.add_subplot(1, 3, 3)
    methods = ["pullback", "geodesic", "linear"]
    off_e = [summary["trajectories"][m]["off_manifold_energy"] for m in methods]
    my_d = [summary["trajectories"][m]["my_geodesic_distance"] for m in methods]
    x = np.arange(len(methods))
    width = 0.4
    ax3.bar(x - width / 2, off_e, width, color=[METHOD_COLOR[m] for m in methods],
            alpha=0.7, label="off-M_y E (nearest centroid)")
    ax3.bar(x + width / 2, my_d, width, color=[METHOD_COLOR[m] for m in methods],
            alpha=1.0, label="M_y-line distance")
    ax3.set_xticks(x)
    ax3.set_xticklabels(methods)
    ax3.set_ylabel("distance / energy")
    ax3.set_title(f"{start} → {end}: behavior metrics")
    ax3.legend(loc="best", fontsize=9)
    ax3.grid(True, alpha=0.3, axis="y")
    for i, (e1, e2) in enumerate(zip(off_e, my_d, strict=True)):
        ax3.annotate(f"{e1:.2f}", (i - width / 2, e1), ha="center", va="bottom", fontsize=8)
        ax3.annotate(f"{e2:.2f}", (i + width / 2, e2), ha="center", va="bottom", fontsize=8)

    fig.suptitle(
        f"Pullback experiment — {start} → {end}  (sigma={summary['sigma']:.3f}, K={summary['num_waypoints']})",
        fontsize=13,
    )
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / f"{start}_{end}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    config = load_config()
    manifold = FittedManifold.load(config.paths.manifold_h)
    behavior = BehaviorManifold.load(config.paths.manifold_y)

    files = sorted(RESULTS_DIR.glob("*.json"))
    if not files:
        raise SystemExit(f"no pullback results in {RESULTS_DIR}")

    for f in files:
        summary = json.loads(f.read_text())
        start, end = summary["pair"]
        paths_npz = DATA_DIR / f"paths_{start}_{end}.npz"
        if not paths_npz.exists():
            print(f"  missing paths file {paths_npz}; skipping {f.name}")
            continue
        plot_one_pair(summary, paths_npz, manifold, behavior)


if __name__ == "__main__":
    main()
