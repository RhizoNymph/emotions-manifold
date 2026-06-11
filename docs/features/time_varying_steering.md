# Time-varying steering

Stepping through geodesic waypoints **during** generation rather than
holding a single steering vector constant for the full 96 tokens.
This is Goodfire's central distinguishing claim — that the geodesic
isn't just a shape in activation space, it's a *trajectory* the model
should follow temporally.

## Scope

- Segmented generation: divide the token budget into K segments,
  applying one steering vector per segment (`/v1/completions` with a
  manually-rebuilt Gemma template between segments).
- The **schedule axis** separating time-variation from its confounds:
  - `varying` — segment k uses waypoint `w[seg_indices[k]]` (hard
    switches; finer K = smaller switches at the same token budget).
  - `constant` — every segment uses the path-midpoint waypoint
    `w[(num_waypoints-1)//2]`: the control that isolates the
    segmented call structure itself.
- Phase-split judging like the chord experiment: `judge="none"`
  generates and saves completions (resumable); re-running with
  `judge="batched"` rates everything in one Batches pass and writes
  per-pair results.
- The day-7 n=40 design: four conditions (tv8 = 8×12, tv16 = 16×6,
  cv8, cv16) on `experiments/pairs/alift_n40.json`, pairs striped
  across localhost and node1 via `VLLM_BASE_URL`.

## Non-scope

- Per-token vector switching (would require fork wire-format changes;
  the n=40 result makes it poorly motivated — see results/day7.md).
- Per-segment judging. We judge the full concatenated continuation.

## Data/control flow

1. `compute_pullback` → 30 waypoints per method (pullback / geodesic /
   linear), scaled ×8.
2. `schedule_indices(30, K, schedule)` picks the per-segment waypoint
   index ladder ([0,4,8,12,17,21,25,29] for varying K=8; [14]×K for
   constant).
3. Per (method, prompt): K sequential `/v1/completions` calls, each
   re-sending the manually-built Gemma template with the accumulated
   assistant text and that segment's steering vector. Stop-token ends
   generation early.
4. Completions saved to `{out_dir}/completions_{start}_{end}.json`
   (the n=12-era script never saved them — that data is judge-cache
   only and cannot be re-judged).
5. Judge phase collects ALL pending pairs' texts into one judge call
   (sequential or batched), then writes per-pair results in the
   original n=12 schema: `{"pair", "schedule", "metrics": {method:
   {off_my_e, my_line, ratings_va}}}`.

## Files

- `src/manifold_emotions/experiments/time_varying.py` — `TVRunConfig`,
  `run_tv_pairs`, `schedule_indices`, metrics. Key exports re-exported
  from `manifold_emotions.experiments`.
- `scripts/experiments/run_time_varying_steering.py` — CLI; defaults
  reproduce the n=12 design; `--schedule/--judge/--out-dir/--segments/
  --tokens-per-segment/--concurrency/--force`.
- `scripts/analysis/analyze_time_varying.py` — TV-vs-baseline
  comparison; `--cv-format tv` compares two TV-schema dirs with
  identical metric definitions (the chord-format myl metric measures
  a different quantity — per-waypoint distance-to-line vs single-
  completion distance-to-midpoint — so cross-format comparisons are
  descriptive only).
- `tests/test_time_varying.py` — pinned index ladders, schedule
  semantics, metric hand-values, phase/resume, failure tolerance.

## Invariants

- Same scale (8.0), same prompts, same K=30 subspace waypoints as the
  constant-vector chord baseline.
- Result files keep the n=12 schema so the analyzer reads any
  condition directory unchanged.
- One out-dir per condition; never overwrite an earlier condition.
- TV-vs-CV claims must compare same-format, same-segmentation
  conditions (tv8↔cv8, tv16↔cv16) — the n=12 analysis compared TV
  against the unsegmented chord baseline and mistook the segmentation
  artifact for a time-variation effect.

## Outputs

- n=12 era (preserved): `results/time_varying/…`
- n=40 four-condition: `results/time_varying_n40/{tv8,tv16,cv8,cv16}/`
  (completions_, per-pair results, ratings_cache.json) and
  `results/time_varying_n40/analysis_*/` (_summary.json + tv_vs_cv.png
  for tv8↔cv8, tv16↔cv16, and the four vs-chord descriptive
  comparisons). Day journal: `results/day7.md`.
