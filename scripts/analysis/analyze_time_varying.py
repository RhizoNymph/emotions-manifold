"""Analyze the time-varying steering experiment.

For each pair tested with time-varying segmented generation
(results/time_varying/{pair}.json), compare against the constant-vector
8-D baseline (results/pullback/{pair}.json).

Question: does time-varying generation produce different behavior than
holding the steering vector constant? And in particular, does it open
a behavioral advantage for the geodesic that didn't exist in the
constant-vector setting?

Output: results/time_varying/_summary.json + figure.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


TV_DIR = Path("results/time_varying")
CV_DIR = Path("results/pullback")
OUT_DIR = TV_DIR


def load_tv(p: Path) -> dict | None:
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    m = data["metrics"]
    return {
        "pair": "_".join(data["pair"]),
        "tv_pullback_off": m["pullback"]["off_my_e"],
        "tv_geodesic_off": m["geodesic"]["off_my_e"],
        "tv_linear_off":   m["linear"]["off_my_e"],
        "tv_pullback_myl": m["pullback"]["my_line"],
        "tv_geodesic_myl": m["geodesic"]["my_line"],
        "tv_linear_myl":   m["linear"]["my_line"],
    }


def load_cv(pair_name: str) -> dict | None:
    p = CV_DIR / f"{pair_name}.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    t = data["trajectories"]
    return {
        "cv_pullback_off": t["pullback"]["off_manifold_energy"],
        "cv_geodesic_off": t["geodesic"]["off_manifold_energy"],
        "cv_linear_off":   t["linear"]["off_manifold_energy"],
        "cv_pullback_myl": t["pullback"]["my_geodesic_distance"],
        "cv_geodesic_myl": t["geodesic"]["my_geodesic_distance"],
        "cv_linear_myl":   t["linear"]["my_geodesic_distance"],
    }


def main() -> None:
    pairs = []
    for p in sorted(TV_DIR.glob("*.json")):
        if p.name.startswith("_") or p.name.startswith("ratings_"):
            continue
        tv = load_tv(p)
        if tv is None:
            continue
        cv = load_cv(tv["pair"])
        if cv is None:
            print(f"  skipping {tv['pair']}: no constant-vector baseline")
            continue
        merged = {**tv, **cv}
        pairs.append(merged)

    print(f"Loaded {len(pairs)} pairs with both TV and CV data")
    if not pairs:
        raise SystemExit("No matching pairs")

    # Compute time-varying margins
    def col(name):
        return np.array([p[name] for p in pairs])

    print(f"\n{'pair':<25} {'method':<10} {'CV off':>8} {'TV off':>8} {'Δ':>7}  "
          f"{'CV myl':>8} {'TV myl':>8} {'Δ':>7}")
    for p in pairs:
        for m in ["pullback", "geodesic", "linear"]:
            d_off = p[f"tv_{m}_off"] - p[f"cv_{m}_off"]
            d_myl = p[f"tv_{m}_myl"] - p[f"cv_{m}_myl"]
            print(f"{p['pair']:<25} {m:<10} {p[f'cv_{m}_off']:>8.3f} {p[f'tv_{m}_off']:>8.3f} {d_off:>+7.3f}  "
                  f"{p[f'cv_{m}_myl']:>8.3f} {p[f'tv_{m}_myl']:>8.3f} {d_myl:>+7.3f}")

    # Summary stats: TV vs CV per method
    summary = {"n_pairs": len(pairs)}
    for m in ["pullback", "geodesic", "linear"]:
        for metric in ["off", "myl"]:
            cv_vals = col(f"cv_{m}_{metric}")
            tv_vals = col(f"tv_{m}_{metric}")
            diff = tv_vals - cv_vals
            key = f"{m}_{metric}_tv_minus_cv"
            mean = float(diff.mean())
            try:
                ci = stats.bootstrap((diff,), np.mean, confidence_level=0.95,
                                      random_state=0, n_resamples=2000).confidence_interval
                ci_lo, ci_hi = float(ci.low), float(ci.high)
            except Exception:
                ci_lo, ci_hi = float("nan"), float("nan")
            summary[key] = {"mean": mean, "ci": [ci_lo, ci_hi], "n": len(diff)}

    # Method-comparison summaries under TV
    print("\n=== Time-varying method gaps (vs linear) ===")
    for m in ["pullback", "geodesic"]:
        for metric in ["off", "myl"]:
            diff = col(f"tv_{m}_{metric}") - col(f"tv_linear_{metric}")
            try:
                stat, p_val = stats.wilcoxon(diff, alternative="less")
                ci = stats.bootstrap((diff,), np.mean, confidence_level=0.95,
                                      random_state=0, n_resamples=2000).confidence_interval
                wins = int((diff < 0).sum())
            except Exception:
                stat = float("nan"); p_val = float("nan")
                ci = type("X", (), dict(low=float("nan"), high=float("nan")))()
                wins = 0
            mean = float(diff.mean())
            print(f"  {m:>10s} vs linear ({metric}): mean={mean:+.4f}  CI [{float(ci.low):+.4f}, {float(ci.high):+.4f}]  "
                  f"wins_method={wins}/{len(diff)}  Wilcoxon-1sided p={p_val:.3f}")
            summary[f"tv_{m}_vs_linear_{metric}"] = {
                "mean_diff": mean, "ci": [float(ci.low), float(ci.high)],
                "wilcoxon_p_1sided": float(p_val),
                "wins_method_better": int(wins),
                "n": int(len(diff)),
            }

    print("\n=== Constant-vector method gaps (same pairs, for reference) ===")
    for m in ["pullback", "geodesic"]:
        for metric in ["off", "myl"]:
            diff = col(f"cv_{m}_{metric}") - col(f"cv_linear_{metric}")
            try:
                stat, p_val = stats.wilcoxon(diff, alternative="less")
                wins = int((diff < 0).sum())
            except Exception:
                p_val = float("nan"); wins = 0
            mean = float(diff.mean())
            print(f"  {m:>10s} vs linear ({metric}): mean={mean:+.4f}  "
                  f"wins_method={wins}/{len(diff)}  Wilcoxon-1sided p={p_val:.3f}")

    summary["per_pair"] = pairs
    (OUT_DIR / "_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {OUT_DIR/'_summary.json'}")

    # Plot: TV vs CV per-pair, both metrics × all 3 methods
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    metric_titles = {"off": "off-M_y E (lower=on-manifold)", "myl": "M_y-line distance (lower=on-target)"}
    for r_i, metric in enumerate(["off", "myl"]):
        for c_i, method in enumerate(["pullback", "geodesic", "linear"]):
            ax = axes[r_i, c_i]
            cv_vals = col(f"cv_{method}_{metric}")
            tv_vals = col(f"tv_{method}_{metric}")
            ax.scatter(cv_vals, tv_vals, alpha=0.7, color="steelblue", edgecolor="black", lw=0.4)
            lim = [min(cv_vals.min(), tv_vals.min()) * 0.9, max(cv_vals.max(), tv_vals.max()) * 1.1]
            ax.plot(lim, lim, "k--", lw=1, alpha=0.5)
            ax.set_xlim(lim); ax.set_ylim(lim)
            ax.set_xlabel(f"constant-vector {method}")
            ax.set_ylabel(f"time-varying {method}")
            mean_diff = (tv_vals - cv_vals).mean()
            title = f"{method} {metric_titles[metric]}\nmean(TV − CV) = {mean_diff:+.4f}"
            ax.set_title(title, fontsize=9)
            ax.grid(alpha=0.3)
    plt.suptitle(f"Time-varying vs constant-vector steering (n={len(pairs)} pairs, K=8 segments)", y=1.0)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "tv_vs_cv.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {OUT_DIR/'tv_vs_cv.png'}")


if __name__ == "__main__":
    main()
