# Time-varying steering

Stepping through geodesic waypoints **during** generation rather than
holding a single steering vector constant for the full 96 tokens.
This is Goodfire's central distinguishing claim — that the geodesic
isn't just a shape in activation space, it's a *trajectory* the model
should follow temporally.

## Scope

- Implement segmented generation: divide 96-token output into K=8
  segments of 12 tokens each.
- For each segment, apply a different waypoint vector from the
  30-waypoint pullback / geodesic / linear trajectory.
- Run on a small set of pairs (default 3; chained set of 10) across
  both vLLM nodes.
- Compare time-varying (TV) behavioral metrics against the constant-
  vector (CV) baseline from the n=40 chord experiment.

## Non-scope

- Per-token vector switching. The 8-segment chunking is the simplest
  reasonable approximation; true per-token would require either
  modifying the vLLM fork (the existing wire format sends one vector
  per request) or 96× more HTTP calls per generation.
- Per-segment judging. We judge the full concatenated continuation,
  matching the constant-vector setup.

## Implementation

- Uses vLLM's **/v1/completions** endpoint (not chat/completions) with
  a manually-built Gemma chat template:

      <start_of_turn>user
      {user_prompt}<end_of_turn>
      <start_of_turn>model
      {assistant_partial...}

- After each segment, the generated text is appended to the partial
  and the prompt is rebuilt — the steering vector for the next
  segment is sent fresh in the next request.
- The chat-completions endpoint's `continue_final_message` mode does
  not work reliably with this fork because the Gemma chat template
  strips trailing tokens during re-rendering ("continue_final_message
  is set but the final message does not appear in the chat after
  applying the chat template").

## Files

- `src/manifold_emotions/manifold/pullback.py` — `compute_pullback`
  provides the per-waypoint vectors (pullback / geodesic / linear).
- `scripts/run_time_varying_steering.py` — main entry; runs a single
  pair (or default trio).
- `scripts/run_time_varying_chain.sh` — runs 10 additional pairs split
  5/5 across localhost and node1.
- `scripts/analysis/analyze_time_varying.py` — compares TV vs CV per method
  per metric; writes `results/time_varying/_summary.json` and
  `tv_vs_cv.png`.

## Invariants

- Same scale (8.0), same prompts, same K=30 waypoints in subspace as
  the constant-vector baseline so per-pair TV vs CV comparison is
  matched.
- Same Claude judge model and prompt template as the constant-vector
  experiment; ratings cached per text_id under `ratings_{pair}.json`.

## Outputs

- `results/time_varying/{pair}.json` — per-pair TV result with the
  three method completions and their off-M_y / M_y-line metrics.
- `results/time_varying/ratings_{pair}.json` — Claude judge cache.
- `results/time_varying/_summary.json` — TV vs CV aggregate.
- `results/time_varying/tv_vs_cv.png` — 2×3 grid of TV-vs-CV scatter.
