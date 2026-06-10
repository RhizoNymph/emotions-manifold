# d=6 behavioral re-run

Drop-in copy of the 4-D pipeline at d=6 (PCA components). Motivated
by the denser dim sweep finding that the G_E geodesic edge over the
chord baseline peaks at **d=6 (+0.085)**, larger than d=4 (+0.063)
or d=8 (+0.050).

If the behavioral chord-experiment metrics also show a d=6 peak,
the geometric finding generalizes to behavior. If they don't, the
geometric edge doesn't translate (consistent with the broader
geometric-vs-behavioral dissociation story).

## Scope

- Fit a fresh 6-D manifold from the same emotion_vectors.npz.
- Precompute geodesics for all 14,535 emotion pairs in 6-D.
- Re-run the n=40 chord experiment (same 40 A-lift pairs) split
  20/20 across both vLLM nodes.
- Analyze with the same Wilcoxon / bootstrap pipeline as the 4-D
  and 8-D runs.

## Non-scope

- Doesn't re-derive A_lift (the 8-D pullback dynamics that A_lift
  was constructed around).
- Doesn't touch the existing 8-D or 4-D artifacts.

## Files

- `scripts/fit_manifold.py --dim 6 --geodesics` — fits + precomputes geodesics.
  Outputs `data/manifold_h_6d_full.npz` and `data/geodesics_cache_6d.npz`.
- `scripts/experiments/run_chord.py --config experiments/chord_6d.yaml` — single-pair runner;
  outputs to `results/pullback_6d/{pair}.json` and `data/pullback_6d/`.
- `scripts/orchestration/run_chain.py --config experiments/chord_6d.yaml` — runs all 40 pairs split across both vLLMs.
- `scripts/orchestration/run_chain.py` — waits for setup, then runs
  chain, then runs analysis.
- `scripts/analysis/analyze_chord.py --results-dir results/pullback_6d` — Wilcoxon / bootstrap
  / margin summary; outputs `results/riemannian_analysis_6d/_summary.json`.
- `scripts/analysis/analyze_dim_behavioral_compare.py` — 3-way d∈{4,6,8} per-pair
  comparison once all three are available.

## Invariants

- Same 40-pair set as the 4-D and 8-D runs (10 original + 30 A_lift
  expansion).
- Same scale (8.0), same K=30 waypoints, same prompts.
- Same Claude judge model + prompt; ratings cached separately under
  `data/pullback_6d/ratings_*.json` to avoid contaminating other dim caches.
