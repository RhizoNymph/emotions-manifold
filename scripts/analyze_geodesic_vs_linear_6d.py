"""6-D variant of analyze_geodesic_vs_linear.py.

Reads results/pullback_6d/{s}_{e}.json (produced by
scripts/alift_6d_chain.sh) and computes the geodesic-vs-linear
and pullback-vs-linear comparisons, mirroring the 8-D analysis at
analyze_geodesic_vs_linear.py.

Output: results/riemannian_analysis_6d/_summary.json + figures.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

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

RESULTS_DIR = Path("results/pullback_6d")


def load_pair_metrics(s, e):
    path = RESULTS_DIR / f"{s}_{e}.json"
    if not path.exists():
        alt = RESULTS_DIR / f"{e}_{s}.json"
        if alt.exists():
            path = alt
        else:
            return None
    d = json.loads(path.read_text())
    t = d["trajectories"]
    return {
        "pb_myl": t["pullback"]["my_geodesic_distance"],
        "ge_myl": t["geodesic"]["my_geodesic_distance"],
        "li_myl": t["linear"]["my_geodesic_distance"],
        "pb_off": t["pullback"]["off_manifold_energy"],
        "ge_off": t["geodesic"]["off_manifold_energy"],
        "li_off": t["linear"]["off_manifold_energy"],
    }


def bootstrap_ci(xs, fn, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_boot):
        idx = rng.choice(len(xs), size=len(xs), replace=True)
        try:
            v = fn(xs[idx])
            if np.isfinite(v):
                samples.append(v)
        except Exception:
            pass
    samples = np.array(samples)
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def main():
    plan = json.loads(Path("data/probe/alift_expansion_plan.json").read_text())
    expansion_pairs = {
        (p[0], p[1]): plan["a_lift_predictions"][f"{p[0]}_{p[1]}"]
        for p in plan["predict_win"] + plan["predict_loss"] + plan["predict_tie"]
    }
    all_predictions = {**ORIGINAL_PAIRS, **expansion_pairs}

    rows = []
    nan_pairs = []
    missing = []
    for (s, e), a_lift in all_predictions.items():
        m = load_pair_metrics(s, e)
        if m is None:
            missing.append(f"{s}->{e}")
            continue
        vals = [m["pb_myl"], m["ge_myl"], m["li_myl"],
                m["pb_off"], m["ge_off"], m["li_off"]]
        if not all(np.isfinite(vals)):
            nan_pairs.append(f"{s}->{e}")
            continue
        rows.append({
            "pair": f"{s}->{e}",
            "a_lift": a_lift,
            "pb_margin": m["li_myl"] - m["pb_myl"],
            "ge_margin": m["li_myl"] - m["ge_myl"],
            "pb_off_gap": m["li_off"] - m["pb_off"],
            "ge_off_gap": m["li_off"] - m["ge_off"],
            "is_original": (s, e) in ORIGINAL_PAIRS,
        })
    if missing:
        print(f"  missing ({len(missing)}): {missing[:5]}...")
    if nan_pairs:
        print(f"  skipped {len(nan_pairs)} NaN pairs: {nan_pairs}")
    print(f"\nLoaded {len(rows)}/{len(all_predictions)} pairs at 6-D")

    if len(rows) == 0:
        print("\nNo pairs loaded. Did you run scripts/alift_6d_chain.sh?")
        return

    a_lifts = np.array([r["a_lift"] for r in rows])
    pb_margins = np.array([r["pb_margin"] for r in rows])
    ge_margins = np.array([r["ge_margin"] for r in rows])
    pb_off_gaps = np.array([r["pb_off_gap"] for r in rows])
    ge_off_gaps = np.array([r["ge_off_gap"] for r in rows])

    print("\n=== 6-D RESULTS (compare to 8-D in results/riemannian.md) ===")
    for name, margins in [("pullback vs linear M_y", pb_margins),
                           ("geodesic vs linear M_y", ge_margins),
                           ("pullback vs linear off-M_y", pb_off_gaps),
                           ("geodesic vs linear off-M_y", ge_off_gaps)]:
        m = margins.mean()
        lo, hi = bootstrap_ci(margins, np.mean)
        wp = stats.wilcoxon(margins, alternative="greater").pvalue
        wins = (margins > 0).sum()
        print(f"  {name:<35s}  mean={m:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  "
              f"p={wp:.3f}  wins={wins}/{len(margins)}")

    ge_pr, ge_pp = stats.pearsonr(a_lifts, ge_margins)
    pb_pr, pb_pp = stats.pearsonr(a_lifts, pb_margins)
    print(f"\n  A_lift~pullback margin r = {pb_pr:+.3f} (p={pb_pp:.3f})")
    print(f"  A_lift~geodesic margin r = {ge_pr:+.3f} (p={ge_pp:.3f})")

    out_dir = Path("results/riemannian_analysis_6d")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "n_pairs": len(rows),
        "manifold_dim": 4,
        "pullback_vs_linear": {
            "my_line_margin_mean": float(pb_margins.mean()),
            "my_line_margin_ci": list(bootstrap_ci(pb_margins, np.mean)),
            "my_line_wilcoxon_p_one_sided":
                float(stats.wilcoxon(pb_margins, alternative="greater").pvalue),
            "my_line_wins": int((pb_margins > 0).sum()),
            "off_my_e_gap_mean": float(pb_off_gaps.mean()),
            "off_my_e_gap_ci": list(bootstrap_ci(pb_off_gaps, np.mean)),
            "off_my_e_wilcoxon_p_one_sided":
                float(stats.wilcoxon(pb_off_gaps, alternative="greater").pvalue),
            "a_lift_pearson_r": float(pb_pr),
            "a_lift_pearson_p": float(pb_pp),
        },
        "geodesic_vs_linear": {
            "my_line_margin_mean": float(ge_margins.mean()),
            "my_line_margin_ci": list(bootstrap_ci(ge_margins, np.mean)),
            "my_line_wilcoxon_p_one_sided":
                float(stats.wilcoxon(ge_margins, alternative="greater").pvalue),
            "my_line_wins": int((ge_margins > 0).sum()),
            "off_my_e_gap_mean": float(ge_off_gaps.mean()),
            "off_my_e_gap_ci": list(bootstrap_ci(ge_off_gaps, np.mean)),
            "off_my_e_wilcoxon_p_one_sided":
                float(stats.wilcoxon(ge_off_gaps, alternative="greater").pvalue),
            "a_lift_pearson_r": float(ge_pr),
            "a_lift_pearson_p": float(ge_pp),
        },
        "per_pair": rows,
    }
    (out_dir / "_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {out_dir/'_summary.json'}")


if __name__ == "__main__":
    main()
