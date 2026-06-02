"""Analyze the n=20 composition expansion (5 original + 15 new).

Bootstrap CIs on coherence gap, off-M_y E gap, and dist-from-midpoint
for both norm-matched and raw-magnitude conditions. Stratifies by
structural category (same-quadrant / opposite-valence / opposite-arousal)
to see if pathology shows up in any specific structural pattern.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


def load_summary(path):
    if not Path(path).exists():
        return None
    return json.loads(Path(path).read_text())


def compute_gaps(summary):
    rows = []
    for tag, d in summary.items():
        if not isinstance(d, dict) or "linear" not in d:
            continue
        lin = d["linear"]
        pb = d["pullback"]
        rows.append({
            "tag": tag,
            "e1": d["composition"][0],
            "e2": d["composition"][1],
            "coh_gap": lin["coherence_distribution"]["coherent"] - pb["coherence_distribution"]["coherent"],
            "off_gap": lin["off_M_y_E"] - pb["off_M_y_E"],
            "dist_mid_gap": lin["dist_from_mid"] - pb["dist_from_mid"],
        })
    return rows


def bootstrap_ci(arr, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    bs = np.array([rng.choice(arr, size=len(arr), replace=True).mean()
                    for _ in range(n_boot)])
    return float(arr.mean()), tuple(np.percentile(bs, [2.5, 97.5]).tolist())


def main():
    # Load all available composition data
    sources = {
        "original_raw": "results/composition/_summary.json",
        "original_nm": "results/composition_normmatched/_summary.json",
        "expansion_raw": "results/composition_expansion_raw/_summary.json",
        "expansion_nm": "results/composition_expansion_nm/_summary.json",
    }

    print("==== Composition expansion analysis ====")
    print()
    print(f"{'source':>20s}  {'n':>3s}  {'coh gap mean':>14s}  {'95% CI':>22s}  {'off gap mean':>14s}  {'95% CI':>22s}")

    all_pooled = {"raw": [], "nm": []}
    for source_name, path in sources.items():
        summary = load_summary(path)
        if summary is None:
            print(f"  {source_name:>20s}  -- missing --")
            continue
        rows = compute_gaps(summary)
        if not rows:
            print(f"  {source_name:>20s}  -- empty --")
            continue

        coh = np.array([r["coh_gap"] for r in rows])
        off = np.array([r["off_gap"] for r in rows])
        coh_mean, coh_ci = bootstrap_ci(coh)
        off_mean, off_ci = bootstrap_ci(off)

        print(f"  {source_name:>20s}  {len(rows):>3d}  "
              f"{coh_mean:>+13.3f}  [{coh_ci[0]:>+.3f}, {coh_ci[1]:>+.3f}]  "
              f"{off_mean:>+13.3f}  [{off_ci[0]:>+.3f}, {off_ci[1]:>+.3f}]")

        if "nm" in source_name:
            all_pooled["nm"].extend(rows)
        else:
            all_pooled["raw"].extend(rows)

    print()
    print("==== Pooled (original + expansion) ====")
    for cond, rows in all_pooled.items():
        if not rows:
            continue
        coh = np.array([r["coh_gap"] for r in rows])
        off = np.array([r["off_gap"] for r in rows])
        coh_mean, coh_ci = bootstrap_ci(coh)
        off_mean, off_ci = bootstrap_ci(off)
        # Significance: is the gap detectably nonzero?
        coh_w = stats.wilcoxon(coh).pvalue if len(coh) >= 6 else float("nan")
        off_w = stats.wilcoxon(off).pvalue if len(off) >= 6 else float("nan")
        print(f"  {cond:>4s}  n={len(rows)}")
        print(f"    coh_gap: mean={coh_mean:+.3f} CI=[{coh_ci[0]:+.3f}, {coh_ci[1]:+.3f}]  "
              f"Wilcoxon p={coh_w:.4f}")
        print(f"    off_gap: mean={off_mean:+.3f} CI=[{off_ci[0]:+.3f}, {off_ci[1]:+.3f}]  "
              f"Wilcoxon p={off_w:.4f}")

    # Stratify expansion picks by structural type
    plan_path = Path("data/probe/composition_expansion_plan.json")
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        same_quad = {tuple(p) for p in plan["same_quadrant"]}
        opp_val = {tuple(p) for p in plan["opposite_valence"]}
        opp_aro = {tuple(p) for p in plan["opposite_arousal"]}

        print()
        print("==== Stratified analysis (norm-matched expansion only) ====")
        nm_summary = load_summary("results/composition_expansion_nm/_summary.json")
        if nm_summary:
            nm_rows = compute_gaps(nm_summary)
            buckets = {"same_quadrant": [], "opposite_valence": [], "opposite_arousal": []}
            for r in nm_rows:
                pair = (r["e1"], r["e2"])
                if pair in same_quad:
                    buckets["same_quadrant"].append(r)
                elif pair in opp_val:
                    buckets["opposite_valence"].append(r)
                elif pair in opp_aro:
                    buckets["opposite_arousal"].append(r)
            for bname, brows in buckets.items():
                if not brows:
                    continue
                coh = np.array([r["coh_gap"] for r in brows])
                off = np.array([r["off_gap"] for r in brows])
                print(f"  {bname:>20s}  n={len(brows)}  "
                      f"coh_gap mean={coh.mean():+.3f}  "
                      f"off_gap mean={off.mean():+.3f}")

    # Save summary
    out_dir = Path("results/composition_expansion_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_out = {
        "sources": {},
        "pooled": {},
    }
    for source_name, path in sources.items():
        summary = load_summary(path)
        if summary is None:
            continue
        rows = compute_gaps(summary)
        if not rows:
            continue
        coh = np.array([r["coh_gap"] for r in rows])
        off = np.array([r["off_gap"] for r in rows])
        coh_mean, coh_ci = bootstrap_ci(coh)
        off_mean, off_ci = bootstrap_ci(off)
        summary_out["sources"][source_name] = {
            "n": len(rows),
            "coh_gap_mean": coh_mean,
            "coh_gap_ci": list(coh_ci),
            "off_gap_mean": off_mean,
            "off_gap_ci": list(off_ci),
            "rows": rows,
        }
    for cond, rows in all_pooled.items():
        if not rows:
            continue
        coh = np.array([r["coh_gap"] for r in rows])
        off = np.array([r["off_gap"] for r in rows])
        coh_mean, coh_ci = bootstrap_ci(coh)
        off_mean, off_ci = bootstrap_ci(off)
        summary_out["pooled"][cond] = {
            "n": len(rows),
            "coh_gap_mean": coh_mean,
            "coh_gap_ci": list(coh_ci),
            "off_gap_mean": off_mean,
            "off_gap_ci": list(off_ci),
            "coh_wilcoxon_p": float(stats.wilcoxon(coh).pvalue) if len(coh) >= 6 else None,
            "off_wilcoxon_p": float(stats.wilcoxon(off).pvalue) if len(off) >= 6 else None,
        }
    (out_dir / "_summary.json").write_text(json.dumps(summary_out, indent=2))
    print(f"\nsaved {out_dir/'_summary.json'}")


if __name__ == "__main__":
    main()
