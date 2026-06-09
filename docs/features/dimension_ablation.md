# Dimension ablation

How the G_E geodesic isometry edge over a flat baseline varies as we
fit the manifold in different-dimensional PCA subspaces. Tests the
hypothesis: does the +0.049 edge at 8-D grow or shrink as we give the
manifold more dimensions to bend through?

## Scope

- Fit a fresh `FittedManifold` at each dim in {4, 8, 16, 32} (original)
  and {6, 10, 12, 14, 24} (denser sweep).
- Recompute the V/A isometry edge for chord vs G_E geodesic on the
  same 800-pair sample (SEED=5678) at each dim.

## Non-scope

- Behavioral re-runs (those live in `riemannian_analysis_4d/` for the
  4-D follow-up).
- Bandwidth or β ablations (separate scripts).

## Key findings

- **4–8-D is the sweet spot**: edge peaks at +0.063 (4-D) and is +0.050
  at 8-D.
- **Above 8-D, G_E hurts**: edge goes to −0.032 at 16-D and −0.106 at
  32-D. KDE density flattens in high-D; geodesics bend in unhelpful
  directions.
- Denser sweep (6, 10, 12, 14, 24) confirms the monotone decline above
  8 and fills the curve smoothly.

## Files

- `scripts/run_dimension_ablation.py` — original sweep, writes
  `results/riemannian_analysis/dimension_ablation.{json,png}`.
- `scripts/run_denser_dim_sweep.py` — fills 6/10/12/14/24, writes
  `results/riemannian_analysis/dimension_ablation_denser.{json,png}`.
  Plot includes both new and original points.

## Invariants

- Same SEED and pair sampling as the original sweep so the curves are
  directly comparable.
- Each dim refit uses `clustered_bandwidth` heuristic; α=1.0, β=0.01;
  num_waypoints=30; max_iter=200 for L-BFGS-B.
