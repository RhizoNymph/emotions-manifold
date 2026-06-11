"""Publication plots for the PCA-subspace-dim U-curve.

Reads every results/subspace_sweep/<pair>_dim<NN>.json and produces:

1. ``results/figures/subspace_sweep.png`` — two-panel summary:
   - left: Δ = (linear off-M_y E) − (manifold off-M_y E) vs subspace dim
   - right: per-method off-M_y energies vs subspace dim
2. ``results/figures/subspace_sweep_traces.png`` — per-dim trace plots
   of the manifold and linear behavior trajectories in (V, A) space,
   overlaid with the M_y emotion centroids.

Run with:
    uv run python scripts/plot_subspace_sweep.py
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from manifold_emotions.behavior.manifold import BehaviorManifold
from manifold_emotions.config import load_config

SWEEP_DIR = Path("results/subspace_sweep")
FIG_DIR = Path("results/figures")
PAIR_COLOR = {
    "excited_weary": "#0066cc",
    "terrified_serene": "#cc3300",
}
PAIR_LABEL = {
    "excited_weary": "excited → weary",
    "terrified_serene": "terrified → serene",
}

FILE_RE = re.compile(r"^(?P<pair>[a-z_]+)_dim(?P<dim>\d{2})\.json$")


def load_runs() -> dict[str, dict[int, dict]]:
    """Group sweep results by pair, indexed by subspace dim."""
    runs: dict[str, dict[int, dict]] = defaultdict(dict)
    for path in sorted(SWEEP_DIR.glob("*_dim*.json")):
        match = FILE_RE.match(path.name)
        if match is None:
            continue
        pair = match.group("pair")
        dim = int(match.group("dim"))
        runs[pair][dim] = json.loads(path.read_text())
    return runs


def plot_summary(runs: dict[str, dict[int, dict]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: Δ vs subspace dim
    ax = axes[0]
    for pair, per_dim in runs.items():
        dims_sorted = sorted(per_dim.keys())
        deltas = [per_dim[d]["delta_linear_minus_manifold"] for d in dims_sorted]
        color = PAIR_COLOR.get(pair, "black")
        ax.plot(
            dims_sorted, deltas, "-o",
            color=color, linewidth=2, markersize=8,
            label=PAIR_LABEL.get(pair, pair),
        )
        for d, val in zip(dims_sorted, deltas, strict=True):
            ax.annotate(
                f"{val:+.3f}", (d, val), fontsize=8,
                xytext=(5, 5), textcoords="offset points",
            )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("PCA subspace dim")
    ax.set_ylabel("Δ = linear E − manifold E\n(positive = manifold wins)")
    ax.set_title("Manifold-steering benefit vs subspace dim")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    # Right: off-manifold energies decomposed
    ax = axes[1]
    for pair, per_dim in runs.items():
        dims_sorted = sorted(per_dim.keys())
        man_e = [per_dim[d]["manifold_off_manifold_energy"] for d in dims_sorted]
        lin_e = [per_dim[d]["linear_off_manifold_energy"] for d in dims_sorted]
        color = PAIR_COLOR.get(pair, "black")
        ax.plot(
            dims_sorted, man_e, "-o",
            color=color, linewidth=2, markersize=7,
            label=f"{PAIR_LABEL.get(pair, pair)} manifold",
        )
        ax.plot(
            dims_sorted, lin_e, "--s",
            color=color, linewidth=1.5, markersize=6, alpha=0.6,
            label=f"{PAIR_LABEL.get(pair, pair)} linear",
        )
    ax.set_xlabel("PCA subspace dim")
    ax.set_ylabel("Cumulative off-manifold energy")
    ax.set_title("Trajectory energy by method × subspace dim")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    fig.suptitle(
        "Manifold steering benefit is non-monotonic in subspace dim",
        fontsize=13,
    )
    fig.tight_layout()
    out = FIG_DIR / "subspace_sweep.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_traces(runs: dict[str, dict[int, dict]], behavior: BehaviorManifold) -> None:
    """One panel per (pair, dim) showing the manifold vs linear traces in V-A space."""
    pairs = sorted(runs.keys())
    if not pairs:
        return
    all_dims = sorted({d for per_dim in runs.values() for d in per_dim.keys()})

    n_rows = len(pairs)
    n_cols = len(all_dims)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.6 * n_cols, 2.7 * n_rows),
        squeeze=False,
        sharex=True, sharey=True,
    )

    for r, pair in enumerate(pairs):
        per_dim = runs[pair]
        for c, dim in enumerate(all_dims):
            ax = axes[r][c]
            ax.scatter(
                behavior.centroids[:, 0], behavior.centroids[:, 1],
                s=12, color="lightgray", zorder=1,
            )
            if dim not in per_dim:
                ax.text(
                    0.5, 0.5, "—",
                    ha="center", va="center", transform=ax.transAxes,
                    color="gray",
                )
                if r == 0:
                    ax.set_title(f"dim {dim}")
                if c == 0:
                    ax.set_ylabel(PAIR_LABEL.get(pair, pair))
                continue

            run = per_dim[dim]
            mv = np.array(run["manifold_waypoint_valence"], dtype=np.float32)
            ma = np.array(run["manifold_waypoint_arousal"], dtype=np.float32)
            lv = np.array(run["linear_waypoint_valence"], dtype=np.float32)
            la = np.array(run["linear_waypoint_arousal"], dtype=np.float32)

            ax.plot(lv, la, "-", color="#999999", linewidth=1.2, alpha=0.85, label="linear")
            ax.plot(mv, ma, "-", color=PAIR_COLOR.get(pair, "black"), linewidth=1.6, label="manifold")
            ax.scatter([run["my_start_v"]], [run["my_start_a"]], marker="o", color="green", s=30, zorder=4)
            ax.scatter([run["my_end_v"]], [run["my_end_a"]], marker="X", color="red", s=40, zorder=4)
            ax.set_xlim(1, 7)
            ax.set_ylim(1, 7)
            ax.grid(True, alpha=0.2)
            delta = run["delta_linear_minus_manifold"]
            ax.text(
                0.04, 0.95,
                f"Δ={delta:+.3f}",
                ha="left", va="top",
                transform=ax.transAxes,
                fontsize=8,
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
            )
            if r == 0:
                ax.set_title(f"dim {dim}")
            if c == 0:
                ax.set_ylabel(PAIR_LABEL.get(pair, pair))

    for ax in axes[-1]:
        ax.set_xlabel("valence")

    fig.suptitle(
        "Behavior trajectories in M_y by subspace dim — green=start, red=end",
        fontsize=12,
    )
    fig.tight_layout()
    out = FIG_DIR / "subspace_sweep_traces.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    runs = load_runs()
    if not runs:
        raise SystemExit(f"no result files found in {SWEEP_DIR}")

    config = load_config()
    behavior = BehaviorManifold.load(config.paths.manifold_y)

    plot_summary(runs)
    plot_traces(runs, behavior)


if __name__ == "__main__":
    main()
