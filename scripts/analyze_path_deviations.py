"""Per-pair geometric deviation: how far does the geodesic actually
bend away from the linear chord?

Reads existing results/pullback/*.json files (which contain per-waypoint
distances between every pair of paths) and the cached geodesics.

Question: if the geodesic deviates by only a few units in 8-D PCA
space, additive steering at scale 8 averages this out and behavior
won't differ from linear. Quantifying this gives us a physical
explanation for the behavioral null in Day 5's main finding.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


def main() -> None:
    plan = json.loads(Path("data/probe/alift_expansion_plan.json").read_text())
    ORIGINAL_PAIRS = {
        ("happy", "sad"), ("excited", "weary"), ("depressed", "energized"),
        ("terrified", "serene"), ("hope", "unhappy"), ("amused", "ashamed"),
        ("grumpy", "hopeful"), ("proud", "sympathetic"),
        ("brooding", "proud"), ("brooding", "pleased"),
    }
    expansion = [(p[0], p[1]) for p in
                 plan["predict_win"] + plan["predict_loss"] + plan["predict_tie"]]
    all_pairs = list(ORIGINAL_PAIRS) + expansion

    rows = []
    for s, e in all_pairs:
        p = Path(f"results/pullback/{s}_{e}.json")
        if not p.exists():
            p = Path(f"results/pullback/{e}_{s}.json")
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        g = d["geometry"]

        # mean distance pullback ↔ geodesic and pullback ↔ linear are in JSON,
        # but mean geodesic ↔ linear is not. Compute from path npz if available.
        npz_path = Path("data") / "pullback" / f"paths_{s}_{e}.npz"
        if not npz_path.exists():
            npz_path = Path("data") / "pullback" / f"paths_{e}_{s}.npz"
        if not npz_path.exists():
            continue
        npz = np.load(npz_path)
        geo_sub = npz["geodesic_sub"]
        lin_sub = npz["linear_sub"]
        pb_sub = npz["pullback_sub"]

        # Per-waypoint distances
        geo_lin_dist = np.linalg.norm(geo_sub - lin_sub, axis=1)
        pb_lin_dist = np.linalg.norm(pb_sub - lin_sub, axis=1)
        pb_geo_dist = np.linalg.norm(pb_sub - geo_sub, axis=1)

        chord_len = float(np.linalg.norm(geo_sub[0] - geo_sub[-1]))

        rows.append({
            "pair": f"{s}->{e}",
            "chord_length": chord_len,
            "geo_lin_max": float(geo_lin_dist.max()),
            "geo_lin_mean": float(geo_lin_dist.mean()),
            "geo_lin_max_frac_of_chord": float(geo_lin_dist.max() / chord_len),
            "pb_lin_mean": float(pb_lin_dist.mean()),
            "pb_geo_mean": float(pb_geo_dist.mean()),
        })

    print(f"Loaded {len(rows)} pairs with path npz")

    geo_lin_maxs = np.array([r["geo_lin_max"] for r in rows])
    geo_lin_means = np.array([r["geo_lin_mean"] for r in rows])
    chord_lens = np.array([r["chord_length"] for r in rows])
    geo_lin_max_fracs = np.array([r["geo_lin_max_frac_of_chord"] for r in rows])
    pb_lin_means = np.array([r["pb_lin_mean"] for r in rows])
    pb_geo_means = np.array([r["pb_geo_mean"] for r in rows])

    print("\nPATH DEVIATIONS (n=40 pairs)")
    print(f"{'':<40s} {'mean':>8s} {'median':>8s} {'max':>8s}")
    print(f"{'chord length (PCA Euclidean)':<40s} {chord_lens.mean():>8.2f} "
          f"{np.median(chord_lens):>8.2f} {chord_lens.max():>8.2f}")
    print(f"{'max(geodesic ↔ linear) per pair':<40s} {geo_lin_maxs.mean():>8.2f} "
          f"{np.median(geo_lin_maxs):>8.2f} {geo_lin_maxs.max():>8.2f}")
    print(f"{'mean(geodesic ↔ linear) per pair':<40s} {geo_lin_means.mean():>8.2f} "
          f"{np.median(geo_lin_means):>8.2f} {geo_lin_means.max():>8.2f}")
    print(f"{'mean(pullback ↔ linear) per pair':<40s} {pb_lin_means.mean():>8.2f} "
          f"{np.median(pb_lin_means):>8.2f} {pb_lin_means.max():>8.2f}")
    print(f"{'mean(pullback ↔ geodesic) per pair':<40s} {pb_geo_means.mean():>8.2f} "
          f"{np.median(pb_geo_means):>8.2f} {pb_geo_means.max():>8.2f}")
    print(f"{'max(geodesic ↔ linear) / chord_len':<40s} {geo_lin_max_fracs.mean():>8.3f} "
          f"{np.median(geo_lin_max_fracs):>8.3f} {geo_lin_max_fracs.max():>8.3f}")

    # ===== Plot =====
    out = Path("results/riemannian_analysis")
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.hist(geo_lin_max_fracs, bins=20, color="C2", alpha=0.7, edgecolor="black")
    ax.axvline(np.median(geo_lin_max_fracs), color="firebrick",
               linewidth=2, label=f"median = {np.median(geo_lin_max_fracs):.2f}")
    ax.set_xlabel("max(geodesic ↔ linear) / chord length")
    ax.set_ylabel("# pairs")
    ax.set_title("How far does the geodesic bend off the linear?\n"
                 "(fraction of straight-line chord length)")
    ax.legend()

    ax = axes[1]
    ax.scatter(pb_lin_means, geo_lin_means, c="purple", s=60,
               edgecolors="black", linewidths=0.5, alpha=0.7)
    lim_lo = 0
    lim_hi = max(pb_lin_means.max(), geo_lin_means.max()) * 1.1
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "--", color="grey",
            label="y = x")
    ax.set_xlabel("mean ‖pullback − linear‖ per waypoint")
    ax.set_ylabel("mean ‖geodesic − linear‖ per waypoint")
    ax.set_title("Pullback vs geodesic deviation from linear")
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.legend()

    plt.tight_layout()
    plt.savefig(out / "path_deviation_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nsaved {out/'path_deviation_distribution.png'}")

    summary = {
        "n_pairs": len(rows),
        "chord_length": {"mean": float(chord_lens.mean()), "median": float(np.median(chord_lens))},
        "geodesic_vs_linear": {
            "max_distance_mean": float(geo_lin_maxs.mean()),
            "max_distance_median": float(np.median(geo_lin_maxs)),
            "mean_distance_mean": float(geo_lin_means.mean()),
            "frac_of_chord_mean": float(geo_lin_max_fracs.mean()),
            "frac_of_chord_median": float(np.median(geo_lin_max_fracs)),
        },
        "pullback_vs_linear": {
            "mean_distance_mean": float(pb_lin_means.mean()),
            "mean_distance_median": float(np.median(pb_lin_means)),
        },
        "pullback_vs_geodesic": {
            "mean_distance_mean": float(pb_geo_means.mean()),
            "mean_distance_median": float(np.median(pb_geo_means)),
        },
        "per_pair": rows,
    }
    (out / "path_deviations.json").write_text(json.dumps(summary, indent=2))
    print(f"saved {out/'path_deviations.json'}")


if __name__ == "__main__":
    main()
