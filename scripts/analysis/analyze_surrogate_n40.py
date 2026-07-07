"""Un-selected (n=40) surrogate-validation analysis.

Consumes the judge phase output (validation_results_n40.json) and the offline
optimizer summary (_summary_n40.json) and reports the population-level questions the
selection-biased 5-pair run could not answer:

  1. Population real headroom over all 40 pairs: mean + 95% bootstrap CI + one-sided
     Wilcoxon (optimized closer than matched linear). This is the un-selected estimate
     to compare against the top-5's +1.45 best case.
  2. Out-of-selection calibration: Pearson r between the surrogate's PREDICTED headroom
     and the ACTUAL real headroom across all 40 pairs, and mean surrogate optimism
     (actual optimized distance - promised distance) with CI. Positive optimism => the
     optimizer exploited surrogate error where it wasn't checked.
  3. Coherence gap (optimized - linear coherent fraction): mean + bootstrap CI +
     two-sided Wilcoxon. Answers whether the closer-to-target optimized outputs pay for
     it in coherence.
  4. Per-pair table.

    uv run python scripts/analysis/analyze_surrogate_n40.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

from manifold_emotions.analysis.stats import bootstrap_ci, bootstrap_mean_ci, paired_gap_report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path,
                    default=Path("results/surrogate_optimizer/validation_results_n40.json"))
    ap.add_argument("--surrogate-summary", type=Path,
                    default=Path("results/surrogate_optimizer/_summary_n40.json"))
    ap.add_argument("--out", type=Path,
                    default=Path("results/surrogate_optimizer/analysis_n40.json"))
    args = ap.parse_args()

    res = json.loads(args.results.read_text())
    per_pair = res["per_pair"]
    summ = json.loads(args.surrogate_summary.read_text())
    pred_headroom = {r["pair"]: r["headroom_pred"] for r in summ["per_pair"]}

    pairs = [r["pair"] for r in per_pair]
    real_hr = np.array([r["real_headroom"] for r in per_pair], dtype=float)
    optimism = np.array([r["surrogate_optimism"] for r in per_pair], dtype=float)
    coh_gap = np.array([r["coherence_gap"] for r in per_pair], dtype=float)
    pred_hr = np.array([pred_headroom.get(p, np.nan) for p in pairs], dtype=float)

    # 1. population real headroom vs matched linear (paired, optimized should be closer)
    hr_report = paired_gap_report(real_hr, alternative="greater")

    # 2. out-of-selection calibration
    finite = np.isfinite(pred_hr) & np.isfinite(real_hr)
    pr = np.nan
    r_ci = (np.nan, np.nan)
    pearson_p = np.nan
    if finite.sum() >= 3:
        pr_arr, pa_arr = pred_hr[finite], real_hr[finite]
        pearson = scipy_stats.pearsonr(pr_arr, pa_arr)
        pr, pearson_p = float(pearson.statistic), float(pearson.pvalue)
        idx = np.arange(len(pr_arr))
        r_ci = bootstrap_ci(
            idx, lambda i: float(scipy_stats.pearsonr(pr_arr[i], pa_arr[i]).statistic))
    opt_mean, opt_ci = bootstrap_mean_ci(optimism)

    # 3. coherence gap (optimized - linear), two-sided
    coh_finite = coh_gap[np.isfinite(coh_gap)]
    coh_report = paired_gap_report(coh_finite, alternative="two-sided")

    out = {
        "n_pairs": len(per_pair),
        "population_real_headroom": {
            **hr_report.as_dict(),
            "note": "linear_dist - optimized_dist per pair; >0 => optimized closer to target",
        },
        "top5_reference_mean_real_headroom": 1.4496,  # from validation_results.json (selected)
        "calibration_out_of_selection": {
            "pearson_r_pred_vs_actual_headroom": pr,
            "pearson_r_ci": [r_ci[0], r_ci[1]],
            "pearson_p": pearson_p,
            "mean_surrogate_optimism": opt_mean,
            "mean_surrogate_optimism_ci": [opt_ci[0], opt_ci[1]],
            "note": "optimism = actual optimized dist - surrogate promised dist; >0 => exploited",
        },
        "coherence_gap_opt_minus_lin": {
            **coh_report.as_dict(),
            "mean_opt_coherent_frac": float(np.nanmean([r["opt_coherent_frac"] for r in per_pair])),
            "mean_lin_coherent_frac": float(np.nanmean([r["lin_coherent_frac"] for r in per_pair])),
        },
        "per_pair": [
            {
                "pair": r["pair"],
                "predicted_headroom": pred_headroom.get(r["pair"]),
                "real_headroom": r["real_headroom"],
                "optimized_dist": r["optimized_actual_dist"],
                "linear_dist": r["linear_actual_dist"],
                "surrogate_optimism": r["surrogate_optimism"],
                "opt_coherent_frac": r["opt_coherent_frac"],
                "lin_coherent_frac": r["lin_coherent_frac"],
                "coherence_gap": r["coherence_gap"],
            }
            for r in per_pair
        ],
    }

    print(f"=== un-selected n=40 surrogate validation ({len(per_pair)} pairs) ===")
    hr = out["population_real_headroom"]
    print(f"  population real headroom: {hr['mean']:+.3f}  "
          f"95% CI [{hr['ci'][0]:+.3f}, {hr['ci'][1]:+.3f}]  "
          f"wins {hr['wins']}/{len(per_pair)}  wilcoxon_p={hr['wilcoxon_p_greater']:.2e}")
    print("    (selected top-5 reference: +1.45)")
    cal = out["calibration_out_of_selection"]
    print(f"  calibration: Pearson r(pred,actual headroom)="
          f"{cal['pearson_r_pred_vs_actual_headroom']:.3f}  "
          f"CI [{cal['pearson_r_ci'][0]:.3f}, {cal['pearson_r_ci'][1]:.3f}]  "
          f"p={cal['pearson_p']:.2e}")
    print(f"    mean optimism {cal['mean_surrogate_optimism']:+.3f}  "
          f"CI [{cal['mean_surrogate_optimism_ci'][0]:+.3f}, "
          f"{cal['mean_surrogate_optimism_ci'][1]:+.3f}]")
    cg = out["coherence_gap_opt_minus_lin"]
    print(f"  coherence gap (opt-lin): {cg['mean']:+.3f}  "
          f"CI [{cg['ci'][0]:+.3f}, {cg['ci'][1]:+.3f}]  "
          f"(opt {cg['mean_opt_coherent_frac']:.2f} vs "
          f"lin {cg['mean_lin_coherent_frac']:.2f} coherent)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
