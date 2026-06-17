"""Spline-vs-linear (and reference geodesic/pullback-vs-linear) analysis.

Reads a chord chain that includes the two spline trajectories
(``spline_induced``, ``spline_density``) and computes, over the n=40 A-lift
pairs, each method's margin against the shared linear baseline on both metrics:

    margin   = li_myl - method_myl    (positive = closer to the M_y target line)
    off_gap  = li_off - method_off    (positive = more on-manifold)

with bootstrap CIs and one-sided Wilcoxon. Mirrors analyze_chord.py's pair set
and conventions so the spline numbers are directly comparable to the existing
pullback/geodesic results.

    uv run python scripts/analysis/analyze_spline_chord.py \
        --results-dir results/pullback_spline_8d --out-dir results/spline_analysis_8d
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

from manifold_emotions.analysis.stats import bootstrap_ci

ORIGINAL_PAIRS = {
    ("happy", "sad"): +0.157,
    ("excited", "weary"): -0.013,
    ("depressed", "energized"): -0.009,
    ("terrified", "serene"): -0.098,
    ("hope", "unhappy"): +0.234,
    ("amused", "ashamed"): +0.221,
    ("grumpy", "hopeful"): +0.219,
    ("proud", "sympathetic"): -0.229,
    ("brooding", "proud"): -0.212,
    ("brooding", "pleased"): -0.209,
}

# Methods compared against linear; linear itself is the baseline.
METHODS = ["pullback", "geodesic", "spline_induced", "spline_density"]


def load_pair(results_dir: Path, s: str, e: str) -> dict | None:
    path = results_dir / f"{s}_{e}.json"
    if not path.exists():
        alt = results_dir / f"{e}_{s}.json"
        if not alt.exists():
            return None
        path = alt
    d = json.loads(path.read_text())
    t = d["trajectories"]
    if "linear" not in t:
        return None
    out: dict[str, float] = {"manifold_dim": d.get("manifold_dim")}
    out["li_myl"] = t["linear"]["my_geodesic_distance"]
    out["li_off"] = t["linear"]["off_manifold_energy"]
    for m in METHODS:
        if m not in t:
            return None
        out[f"{m}_myl"] = t[m]["my_geodesic_distance"]
        out[f"{m}_off"] = t[m]["off_manifold_energy"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--plan", type=Path, default=Path("data/probe/alift_expansion_plan.json"))
    ap.add_argument("--label", default=None)
    args = ap.parse_args()
    label = args.label or args.results_dir.name

    plan = json.loads(args.plan.read_text())
    expansion = {
        (p[0], p[1]): plan["a_lift_predictions"][f"{p[0]}_{p[1]}"]
        for p in plan["predict_win"] + plan["predict_loss"] + plan["predict_tie"]
    }
    all_pairs = {**ORIGINAL_PAIRS, **expansion}

    rows = []
    nan_pairs, missing = [], []
    manifold_dim = None
    for (s, e), a_lift in all_pairs.items():
        m = load_pair(args.results_dir, s, e)
        if m is None:
            missing.append(f"{s}->{e}")
            continue
        manifold_dim = m["manifold_dim"] or manifold_dim
        vals = [m["li_myl"], m["li_off"]] + [
            m[f"{x}_{k}"] for x in METHODS for k in ("myl", "off")
        ]
        if not all(np.isfinite(vals)):
            nan_pairs.append(f"{s}->{e}")
            continue
        row = {"pair": f"{s}->{e}", "a_lift": a_lift}
        for x in METHODS:
            row[f"{x}_margin"] = m["li_myl"] - m[f"{x}_myl"]
            row[f"{x}_off_gap"] = m["li_off"] - m[f"{x}_off"]
        rows.append(row)

    if missing:
        print(f"  missing ({len(missing)}): {missing[:5]}...")
    if nan_pairs:
        print(f"  skipped {len(nan_pairs)} NaN pairs: {nan_pairs[:5]}...")
    print(f"\nLoaded {len(rows)}/{len(all_pairs)} pairs from {args.results_dir}")
    if not rows:
        print("No pairs loaded.")
        return

    a_lifts = np.array([r["a_lift"] for r in rows])
    summary: dict = {"n_pairs": len(rows), "manifold_dim": manifold_dim, "methods": {}}

    print(f"\n=== {label} (n={len(rows)}) — margin vs linear (positive favors method) ===")
    for x in METHODS:
        for metric, key in (("M_y-line", "margin"), ("off-M_y E", "off_gap")):
            arr = np.array([r[f"{x}_{key}"] for r in rows])
            lo, hi = bootstrap_ci(arr, np.mean)
            p = stats.wilcoxon(arr, alternative="greater").pvalue
            wins = int((arr > 0).sum())
            print(f"  {x:<16s} {metric:<10s} mean={arr.mean():+.4f}  "
                  f"CI [{lo:+.4f}, {hi:+.4f}]  p={p:.3f}  wins={wins}/{len(arr)}")
        # A_lift correlation against this method's M_y-line margin
        pr, pp = stats.pearsonr(a_lifts, np.array([r[f"{x}_margin"] for r in rows]))
        margin = np.array([r[f"{x}_margin"] for r in rows])
        off = np.array([r[f"{x}_off_gap"] for r in rows])
        summary["methods"][x] = {
            "my_line_margin_mean": float(margin.mean()),
            "my_line_margin_ci": list(bootstrap_ci(margin, np.mean)),
            "my_line_wilcoxon_p": float(stats.wilcoxon(margin, alternative="greater").pvalue),
            "my_line_wins": int((margin > 0).sum()),
            "off_my_e_gap_mean": float(off.mean()),
            "off_my_e_gap_ci": list(bootstrap_ci(off, np.mean)),
            "off_my_e_wilcoxon_p": float(stats.wilcoxon(off, alternative="greater").pvalue),
            "off_my_e_wins": int((off > 0).sum()),
            "a_lift_pearson_r": float(pr),
            "a_lift_pearson_p": float(pp),
        }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary["per_pair"] = rows
    (args.out_dir / "_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {args.out_dir / '_summary.json'}")


if __name__ == "__main__":
    main()
