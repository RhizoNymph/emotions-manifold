# Positive controls

Sanity-check experiments that disentangle competing interpretations
of the headline behavioral results.

## Scope

- **4-D linear vs 8-D linear**: does 4-D linear steering perform
  worse than 8-D linear on the same 40 pairs? If yes, the 4-D
  geodesic gain reported in §6.5 partly compensates for the
  on-manifold quality that 4-D linear loses, rather than purely
  demonstrating "curved metric helps."

## Findings

### 4-D linear vs 8-D linear (n=40, July 2026)

| metric | 4-D linear | 8-D linear | 4-D − 8-D | CI | Wilcoxon (1-sided 4-D worse) |
|---|---:|---:|---:|---|---:|
| off-M_y E | +0.538 | +0.454 | **+0.085** | [+0.057, +0.115] | **p<0.0001** |
| M_y-line  | +2.183 | +2.144 | +0.039 | [−0.031, +0.111] | p=0.24 |

**4-D linear is substantially worse than 8-D linear on off-M_y E**
(9/40 wins for 4-D vs 8-D). The 4-D geodesic off-M_y +0.019 gain over
4-D linear (Finding 8) therefore recovers only ~22 % of the
on-manifold quality that 4-D linear loses relative to 8-D linear.

The 4-D curved-metric finding is best read as "the curved metric
partially compensates for dimension loss when you under-fit the
manifold" rather than "the curved metric is essential."

## Files

- `scripts/analysis/analyze_4d_linear_vs_8d_linear.py` — reads existing 4-D
  and 8-D pullback data, computes paired comparison.

## Outputs

- `results/riemannian_analysis_4d/linear_4d_vs_8d.json`
- `results/figures/writeup/linear_4d_vs_8d_positive_control.png`
