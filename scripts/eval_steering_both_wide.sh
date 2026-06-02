#!/usr/bin/env bash
# Run wide+fine scale sweeps for differential AND contrastive directions,
# serially on the same vLLM. Robust to degenerate model output at extreme
# scales (judges now return default values on empty/unparseable responses
# instead of crashing the chain).

set -euo pipefail
cd "$(dirname "$0")/.."

LOG=logs/eval_steering_both_wide.log
mkdir -p logs

SCALES="-500,-200,-100,-50,-20,-10,-5,0,+5,+10,+20,+50,+100,+200,+500"

echo "$(date '+%F %T')  ==== differential wide sweep ====" | tee -a "$LOG"
uv run python -u scripts/run_eval_awareness_steering.py \
  --direction-file results/eval_awareness_v2/full_eval_vs_neutral_mean.npy \
  --scales="$SCALES" \
  --results-dir results/eval_steering_diff_wide \
  --data-dir data/eval_steering_diff_wide \
  2>&1 | tee -a "$LOG"

echo "$(date '+%F %T')  ==== contrastive wide sweep ====" | tee -a "$LOG"
uv run python -u scripts/run_eval_awareness_steering.py \
  --direction-file results/eval_awareness_contrastive/full_contrastive_eval.npy \
  --scales="$SCALES" \
  --results-dir results/eval_steering_contr_wide \
  --data-dir data/eval_steering_contr_wide \
  2>&1 | tee -a "$LOG"

echo "$(date '+%F %T')  both sweeps complete" | tee -a "$LOG"
