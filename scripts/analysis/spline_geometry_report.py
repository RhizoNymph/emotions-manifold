"""Geometry report for a spline manifold: path deviation + sampled isometry.

Answers the go/no-go question before any GPU spend: do geodesics on the tight
spline surface bend away from the ambient straight line *more* than the loose
ambient G_E geodesic does (which deviated only ~16% of chord length, explaining
why ambient geodesic steering ~= linear steering)?

Outputs, per source manifold (dim):
  1. Per-pair deviation table on illustrative pairs: max/mean displacement of the
     ambient G_E geodesic, spline-induced geodesic, and spline-density geodesic
     from the linear chord, normalized by chord length.
  2. Sampled isometry: Pearson r of each path-length metric vs V/A distance,
     compared to the ambient chord (+0.710) and ambient G_E (+0.758) baselines.
  3. A PC1xPC2 overlay figure of the paths on a KDE density heatmap.

Example:
    uv run python scripts/analysis/spline_geometry_report.py \
        --spline data/manifold_spline_8d.npz --tag 8d --isometry-sample 300
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import itertools

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

from manifold_emotions.behavior.manifold import BehaviorManifold
from manifold_emotions.manifold.geodesic import fit_geodesic, linear_interpolation
from manifold_emotions.manifold.spline import SplineManifold
from manifold_emotions.manifold.spline_geodesic import fit_spline_geodesic

ILLUSTRATIVE_PAIRS = [
    ("happy", "sad"),
    ("excited", "weary"),
    ("depressed", "energized"),
    ("terrified", "serene"),
    ("calm", "ecstatic"),
    ("amused", "ashamed"),
]
NUM_WAYPOINTS = 30


def _deviation(path: np.ndarray, chord: np.ndarray, chord_len: float) -> tuple[float, float]:
    """max and mean per-waypoint displacement of ``path`` from ``chord``, / chord_len."""
    d = np.linalg.norm(path - chord, axis=1)
    return float(d.max() / chord_len), float(d.mean() / chord_len)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spline", required=True, type=Path)
    ap.add_argument("--behavior", type=Path, default=Path("data/manifold_y.npz"))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--isometry-sample", type=int, default=300)
    ap.add_argument("--out-dir", type=Path, default=Path("results/spline_geometry"))
    args = ap.parse_args()

    spline = SplineManifold.load(args.spline)
    behavior = BehaviorManifold.load(args.behavior)
    geometry = spline.make_geometry()
    labels = list(spline.labels)
    idx = {lab: i for i, lab in enumerate(labels)}

    by_label = {lab: i for i, lab in enumerate(behavior.labels)}
    coords = behavior.centroids[[by_label[lab] for lab in labels]].astype(np.float64)
    centroids = spline.centroids_subspace.astype(np.float64)

    def all_paths(i: int, j: int) -> dict[str, np.ndarray]:
        c0, c1 = centroids[i], centroids[j]
        u0, u1 = coords[i], coords[j]
        linear = linear_interpolation(c0, c1, NUM_WAYPOINTS)
        ambient = fit_geodesic(geometry, c0, c1, num_waypoints=NUM_WAYPOINTS).waypoints
        ind = fit_spline_geodesic(
            spline, u0, u1, metric="induced", num_waypoints=NUM_WAYPOINTS,
            snap_start=c0, snap_end=c1,
        ).waypoints
        den = fit_spline_geodesic(
            spline, u0, u1, metric="density", num_waypoints=NUM_WAYPOINTS,
            snap_start=c0, snap_end=c1,
        ).waypoints
        return {"linear": linear, "ambient_geo": ambient, "spline_induced": ind, "spline_density": den}

    # --- 1. deviation table on illustrative pairs ---------------------------
    dev_rows = []
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    # KDE heatmap background over PC1xPC2
    pc = centroids[:, :2]
    gx = np.linspace(pc[:, 0].min() - 2, pc[:, 0].max() + 2, 120)
    gy = np.linspace(pc[:, 1].min() - 2, pc[:, 1].max() + 2, 120)
    GX, GY = np.meshgrid(gx, gy)
    bw = spline.kde_bandwidth
    dens = np.zeros_like(GX)
    for k in range(centroids.shape[0]):
        dens += np.exp(-(((GX - pc[k, 0]) ** 2 + (GY - pc[k, 1]) ** 2) / (2 * bw**2)))

    for ax, (a, b) in zip(axes.ravel(), ILLUSTRATIVE_PAIRS):
        if a not in idx or b not in idx:
            ax.set_visible(False)
            continue
        i, j = idx[a], idx[b]
        paths = all_paths(i, j)
        chord_len = float(np.linalg.norm(centroids[j] - centroids[i]))
        row = {"pair": f"{a}->{b}", "chord_len": chord_len}
        for name in ("ambient_geo", "spline_induced", "spline_density"):
            mx, mn = _deviation(paths[name], paths["linear"], chord_len)
            row[f"{name}_max_dev"] = mx
            row[f"{name}_mean_dev"] = mn
        dev_rows.append(row)

        ax.contourf(GX, GY, dens, levels=12, cmap="Greys", alpha=0.5)
        ax.scatter(pc[:, 0], pc[:, 1], s=6, c="black", alpha=0.3)
        styles = {"linear": ("0.5", "--"), "ambient_geo": ("tab:blue", "-"),
                  "spline_induced": ("tab:red", "-"), "spline_density": ("tab:orange", "-")}
        for name, (col, ls) in styles.items():
            p = paths[name]
            ax.plot(p[:, 0], p[:, 1], ls, color=col, lw=2, label=name)
        ax.set_title(f"{a}->{b}  (chord {chord_len:.1f})")
        ax.legend(fontsize=7)
    fig.suptitle(f"Spline vs ambient geodesic vs linear — PC1xPC2 ({args.tag})")
    fig.tight_layout()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = args.out_dir / f"{args.tag}_paths.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)

    # --- 2. sampled isometry ------------------------------------------------
    all_pairs = list(itertools.combinations(range(len(labels)), 2))
    rng = np.random.default_rng(0)
    sample = [all_pairs[k] for k in rng.choice(len(all_pairs), size=min(args.isometry_sample, len(all_pairs)), replace=False)]
    va_dist, chord_len, amb_len, ind_len, den_len = ([] for _ in range(5))
    for i, j in sample:
        c0, c1, u0, u1 = centroids[i], centroids[j], coords[i], coords[j]
        va_dist.append(float(np.linalg.norm(u1 - u0)))
        chord_len.append(float(np.linalg.norm(c1 - c0)))
        amb_len.append(fit_geodesic(geometry, c0, c1, num_waypoints=NUM_WAYPOINTS).final_length)
        ind_len.append(fit_spline_geodesic(spline, u0, u1, metric="induced", num_waypoints=NUM_WAYPOINTS).final_length)
        den_len.append(fit_spline_geodesic(spline, u0, u1, metric="density", num_waypoints=NUM_WAYPOINTS).final_length)
    va = np.array(va_dist)
    iso = {
        "n": len(sample),
        "chord_vs_va": float(pearsonr(chord_len, va)[0]),
        "ambient_geo_vs_va": float(pearsonr(amb_len, va)[0]),
        "spline_induced_vs_va": float(pearsonr(ind_len, va)[0]),
        "spline_density_vs_va": float(pearsonr(den_len, va)[0]),
    }

    report = {"tag": args.tag, "num_waypoints": NUM_WAYPOINTS, "deviations": dev_rows, "isometry": iso}
    out_json = args.out_dir / f"{args.tag}_report.json"
    out_json.write_text(json.dumps(report, indent=2))

    # --- stdout summary -----------------------------------------------------
    print(f"\n=== SPLINE GEOMETRY REPORT ({args.tag}) ===")
    print("\nPath deviation from linear chord (normalized by chord length):")
    print(f"{'pair':<22}{'ambient_geo':>14}{'spline_induced':>16}{'spline_density':>16}   (max dev)")
    for r in dev_rows:
        print(f"{r['pair']:<22}{r['ambient_geo_max_dev']:>14.3f}{r['spline_induced_max_dev']:>16.3f}{r['spline_density_max_dev']:>16.3f}")
    am = np.mean([r["ambient_geo_max_dev"] for r in dev_rows])
    im = np.mean([r["spline_induced_max_dev"] for r in dev_rows])
    dm = np.mean([r["spline_density_max_dev"] for r in dev_rows])
    print(f"{'MEAN':<22}{am:>14.3f}{im:>16.3f}{dm:>16.3f}")
    print(f"\n(ambient G_E geodesic deviated 0.163 in the riemannian.md n=40 study — compare spline above)")
    print(f"\nSampled isometry (n={iso['n']}) — Pearson r of path length vs V/A distance:")
    print(f"  chord (baseline):     {iso['chord_vs_va']:+.3f}")
    print(f"  ambient G_E geodesic: {iso['ambient_geo_vs_va']:+.3f}")
    print(f"  spline induced:       {iso['spline_induced_vs_va']:+.3f}")
    print(f"  spline density:       {iso['spline_density_vs_va']:+.3f}")
    print(f"\nwrote {out_json}\nwrote {fig_path}")


if __name__ == "__main__":
    main()
