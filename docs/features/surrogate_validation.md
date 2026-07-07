# Surrogate optimizer + validation (Idea D)

## Scope

Test whether a cheap, CPU-only surrogate `f: steering vector -> judged (V, A)` can
find steering vectors that reach each chord's V/A target waypoints *closer than linear*,
and whether that promise holds when the vectors are actually generated and judged on GPU.

Three questions, escalating in rigor:

1. **Offline screen** (`surrogate_optimizer.py`): does a trust-constrained optimization
   over the surrogate predict positive headroom (`linear_pred - opt_pred`) for the n=40
   chord targets? (Surrogate bias cancels because both terms are surrogate-predicted.)
2. **Selected validation** (`validate_surrogate_vectors.py`): for the 5 top-predicted-
   headroom pairs, does the *real* judged distance beat matched linear, and does reality
   match the surrogate's promise (optimism ≈ 0)? Result: mean real headroom +1.45,
   optimism ≈ 0 — but this is a best-case, selection-biased estimate.
3. **Un-selected validation** (`validate_surrogate_n40.py` + `analyze_surrogate_n40.py`):
   the same test over ALL 40 pairs (population estimate, no selection), plus a coherence
   judgement over both optimized and linear outputs.

**Non-scope**: trajectory coherence *between* waypoints (the optimizer picks each waypoint
independently — an upper bound); closed-loop optimization with the judge in the loop;
retraining the surrogate.

## Data / control flow

### Offline (CPU, no GPU, no judge) — `surrogate_optimizer.py`

1. `harvest_training`: recompute pullback/geodesic/linear subspace geometry for every
   judged pair in `results/pullback/*.json`, join with the judged waypoint V/A →
   `(X sub-vector, Y = V/A)` training set. Held-out R² reported (~0.79 V / 0.84 A).
2. Fit a `RandomForestRegressor` (saturating, extrapolates flat → can't hallucinate far
   targets).
3. `build_candidate_pool`: perturbations + interpolations of observed vectors, filtered
   to stay within the training k-NN radius (`--trust-pct`, default 90th pct). The
   surrogate is only trusted near data.
4. Per chord target waypoint, retrieve the in-trust candidate the surrogate predicts
   lands closest; `headroom_pred = linear_pred_dist − opt_pred_dist`.
5. Dump the optimized trajectories (subspace + unprojected `x`-scale, `* 8.0`) as
   `opt_{s}_{e}.npz` (`pair`, `opt_sub`, `opt_full`, `targets`, `opt_pred`) for GPU
   validation, plus a `_summary.json`.

`--n-validate -1` dumps ALL pairs (un-selected); `--val-subdir` / `--summary-name` write
to fresh names so an existing selected dump is never overwritten. Seed is fixed (`rng =
default_rng(0)`), so the n=40 dump reproduces the top-5 vectors bit-for-bit.

Saved artifacts (untracked, under `results/surrogate_optimizer/`):
- `validation_vectors/opt_*.npz` + `_summary.json` — top-5 (selected).
- `validation_vectors_n40/opt_*.npz` + `_summary_n40.json` — all 40 (un-selected).

### GPU generation + judging — `validate_surrogate_n40.py`

Two phases so generation (vLLM) and judging (Anthropic API) are separate commands:

- **`gen`**: for each pair, load `opt_full` from the npz and recompute the matched linear
  steer (`compute_pullback(...).linear_full * scale`); generate both trajectories with
  `generate_along_path`. Pairs are sharded round-robin across `--hosts` (via
  `experiments.chain.split_pairs`), each host running on the shared event loop with its
  own `base_url` (mirrors `run_chain.py`). Completions are SAVED to
  `n40/completions/{slug}.json` (`opt` + `lin` arms). Resumable: pairs with an existing
  completions file are skipped (`--force` re-runs).
- **`judge`**: load saved completions; per pair, judge V/A (`judge_texts_batched`) and
  coherence (`judge_coherence`, the A/B/C coherent/mixed/absent judge reused verbatim
  from `run_composition_experiment.py`) over the matched opt+linear passages in the SAME
  batch — exactly the 5-pair design. Rating caches at `n40/va_ratings/` and
  `n40/coherence_ratings/`. Per-pair real headroom, surrogate optimism, and coherent
  fractions written to `validation_results_n40.json`.

`--dry-run` builds payloads/passages without any network call (CPU smoke path).

### Analysis — `analyze_surrogate_n40.py`

Consumes `validation_results_n40.json` + `_summary_n40.json`, reusing
`manifold_emotions.analysis.stats`:
- Population real headroom: `paired_gap_report` (mean, 95% bootstrap CI, one-sided
  Wilcoxon optimized-closer-than-linear, win count).
- Out-of-selection calibration: Pearson r (predicted vs actual headroom) with bootstrap
  CI + p; mean surrogate optimism with CI.
- Coherence gap (opt − lin coherent fraction): mean, bootstrap CI, two-sided Wilcoxon.
- Per-pair table. Written to `analysis_n40.json`.

## Related files

- `scripts/analysis/surrogate_optimizer.py` — offline surrogate + candidate dump.
- `scripts/experiments/validate_surrogate_vectors.py` — 5-pair (selected) GPU validation.
- `scripts/experiments/validate_surrogate_n40.py` — n=40 (un-selected) gen + judge, multi-host, saved completions, coherence.
- `scripts/analysis/analyze_surrogate_n40.py` — population stats + calibration + coherence gap.
- `src/manifold_emotions/analysis/stats.py` — `paired_gap_report`, `bootstrap_ci`, `bootstrap_mean_ci`.
- `src/manifold_emotions/steering/trajectory.py` — `generate_along_path`, `_build_payload`, `SteeredContinuation`.
- `scripts/experiments/run_composition_experiment.py` — source of `judge_coherence` (A/B/C).
- `tests/test_surrogate_n40.py` — helper + analysis-math tests.

## Invariants / constraints

- Optimizer RNG seed is 0; re-running is deterministic and never overwrites existing
  dumps (fresh `--val-subdir` / `--summary-name`).
- The n=40 opt vectors for the 5 previously-validated pairs are bit-identical to the
  selected dump, so the +1.45 selected estimate stays directly comparable.
- Completions are always persisted before judging (the n=12 TV run lost its completions).
- Matched-linear baseline is regenerated per pair and judged in the same batch as the
  optimized arm (no cross-batch judge drift).
- `_slug` replaces spaces so multi-word labels (`at ease`) yield safe filenames; the true
  pair is kept inside the npz `pair` field and each completions JSON.
