"""Positive-control: 4-D linear steering vs 8-D linear steering.

Day 5 flagged this as the most load-bearing follow-up: the 4-D
geodesic off-M_y +0.019 (p=0.002) is currently interpreted as
"curvature helps." But it could partly be "4-D linear loses ground
that geodesic recovers" rather than "the curved metric is essential."

Compares linear-steering metrics on the SAME 40 pairs at a low dim vs
d=8 using existing per-pair chord results. Defaults reproduce the 4-D
analysis; --low-dir/--low-tag run the same control at another dim:

    uv run python scripts/analysis/analyze_4d_linear_vs_8d_linear.py \
        --low-dir results/pullback_2d --low-tag 2d \
        --out-dir results/riemannian_analysis_2d

If 4-D linear is materially worse than 8-D linear on off-M_y E or
M_y-line distance, the 4-D geodesic edge is partly compensating for
dimension loss; if they're comparable, the geodesic edge is more
purely "curved metric helps."

Outputs:
- results/riemannian_analysis_4d/linear_4d_vs_8d.json
- results/riemannian_analysis_4d/linear_4d_vs_8d.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


DIR_4D = Path("results/pullback_4d")
DIR_8D = Path("results/pullback")


def load_linear(path: Path) -> tuple[float, float] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    traj = data.get("trajectories", {}).get("linear")
    if traj is None:
        return None
    off = float(traj["off_manifold_energy"])
    myl = float(traj["my_geodesic_distance"])
    return off, myl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--low-dir", type=Path, default=DIR_4D,
                        help="chord results dir for the low-dim run")
    parser.add_argument("--low-tag", default="4d",
                        help="tag for the low dim in filenames/keys (e.g. 2d)")
    parser.add_argument("--high-dir", type=Path, default=DIR_8D)
    parser.add_argument("--high-tag", default="8d")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("results/riemannian_analysis_4d"))
    args = parser.parse_args()
    lo, hi = args.low_tag, args.high_tag
    lo_lab, hi_lab = lo.upper().replace("D", "-D"), hi.upper().replace("D", "-D")
    out_dir = args.out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = []
    for p in sorted(args.low_dir.glob("*.json")):
        if p.name.startswith("_"):
            continue
        # require matching high-dim file (same pair, same naming)
        p8 = args.high_dir / p.name
        if not p8.exists():
            continue
        m4 = load_linear(p)
        m8 = load_linear(p8)
        if m4 is None or m8 is None:
            continue
        pairs.append({
            "pair": p.stem,
            f"linear_off_{lo}": m4[0], f"linear_myl_{lo}": m4[1],
            f"linear_off_{hi}": m8[0], f"linear_myl_{hi}": m8[1],
        })

    print(f"Loaded {len(pairs)} pairs with both {lo_lab} and {hi_lab} linear data")
    if len(pairs) < 5:
        raise SystemExit("Too few pairs for analysis")

    off_4d = np.array([p[f"linear_off_{lo}"] for p in pairs])
    off_8d = np.array([p[f"linear_off_{hi}"] for p in pairs])
    myl_4d = np.array([p[f"linear_myl_{lo}"] for p in pairs])
    myl_8d = np.array([p[f"linear_myl_{hi}"] for p in pairs])

    def report(name, x4, x8, lower_is_better=True):
        diff = x4 - x8
        mean = float(diff.mean())
        ci = stats.bootstrap((diff,), np.mean, confidence_level=0.95,
                              random_state=0, n_resamples=2000).confidence_interval
        # one-sided Wilcoxon: H1 is 4-D worse than 8-D
        if lower_is_better:
            stat, p = stats.wilcoxon(diff, alternative="greater")
        else:
            stat, p = stats.wilcoxon(diff, alternative="less")
        wins_4d_better = int(np.sum(diff < 0)) if lower_is_better else int(np.sum(diff > 0))
        verdict = f"{lo_lab} WORSE" if (p < 0.05) else "no diff" if (mean == 0) else "trend"
        print(f"\n  {name}")
        print(f"    {lo_lab} mean: {x4.mean():+.4f}   {hi_lab} mean: {x8.mean():+.4f}")
        print(f"    {lo_lab} − {hi_lab}: {mean:+.4f}  CI [{ci.low:+.4f}, {ci.high:+.4f}]")
        print(f"    Wilcoxon 1-sided ({lo_lab} worse) p={p:.4f}")
        print(f"    {lo_lab}-better wins: {wins_4d_better}/{len(diff)}  → {verdict}")
        return {
            f"mean_{lo}": float(x4.mean()), f"mean_{hi}": float(x8.mean()),
            "diff_mean": mean, "ci": [float(ci.low), float(ci.high)],
            f"wilcoxon_p_{lo}_worse": float(p),
            f"wins_{lo}_better": wins_4d_better, "n": int(len(diff)),
        }

    print(f"\n=== {lo_lab} linear vs {hi_lab} linear ===")
    off_report = report("off-M_y E (lower=better; on-manifold)", off_4d, off_8d, lower_is_better=True)
    myl_report = report("M_y-line distance (lower=better)", myl_4d, myl_8d, lower_is_better=True)

    out = {
        "n_pairs": len(pairs),
        "off_my_e": off_report,
        "my_line_distance": myl_report,
        "per_pair": pairs,
    }
    out_json = out_dir / f"linear_{lo}_vs_{hi}.json"
    out_json.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {out_json}")

    # Plot per-pair 4D vs 8D for both metrics
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, x4, x8, title, ylab in [
        (axes[0], off_4d, off_8d, "Linear off-M_y E (lower = on-manifold)", "off-M_y E"),
        (axes[1], myl_4d, myl_8d, "Linear M_y-line distance (lower = on-target)", "M_y-line"),
    ]:
        ax.scatter(x8, x4, alpha=0.7, color="steelblue", edgecolor="black", lw=0.4)
        lim = [min(x8.min(), x4.min()) * 0.95, max(x8.max(), x4.max()) * 1.05]
        ax.plot(lim, lim, "k--", lw=1, alpha=0.5, label="y=x")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel(f"{hi_lab} linear {ylab}")
        ax.set_ylabel(f"{lo_lab} linear {ylab}")
        ax.set_title(title)
        ax.legend(loc="upper left")
        ax.grid(alpha=0.3)
        mean_diff = (x4 - x8).mean()
        ax.text(0.02, 0.95, f"mean({lo_lab} − {hi_lab}) = {mean_diff:+.4f}\n"
                            f"wins {lo_lab}-better: {int((x4 < x8).sum())}/{len(x4)}",
                transform=ax.transAxes, va="top", ha="left",
                fontsize=9, family="monospace",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="black", lw=0.5))

    plt.suptitle(f"{lo_lab} linear vs {hi_lab} linear positive control "
                 f"(n={len(pairs)} pairs)", y=1.0)
    plt.tight_layout()
    out_png = out_dir / f"linear_{lo}_vs_{hi}.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
