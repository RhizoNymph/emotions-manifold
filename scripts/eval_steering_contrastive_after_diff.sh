#!/usr/bin/env bash
# Wait for the differential wide sweep to free the vLLM, then run
# the contrastive wide sweep with matched scale ladder.

set -euo pipefail
cd "$(dirname "$0")/.."

LOG=logs/eval_steering_contrastive_wide.log
mkdir -p logs

echo "$(date '+%F %T')  waiting for differential wide sweep to finish" | tee -a "$LOG"
deadline=$(( $(date +%s) + 3600 ))
while pgrep -f "run_eval_awareness_steering.*diff_wide" >/dev/null 2>&1; do
    if [ "$(date +%s)" -ge "$deadline" ]; then
        echo "$(date '+%F %T')  timeout" | tee -a "$LOG"; exit 1
    fi
    sleep 10
done

echo "$(date '+%F %T')  diff done, starting contrastive wide sweep" | tee -a "$LOG"

SCALES="-500,-200,-100,-50,-20,-10,-5,0,+5,+10,+20,+50,+100,+200,+500"
uv run python -u scripts/experiments/run_eval_awareness_steering.py \
  --direction-file results/eval_awareness_contrastive/full_contrastive_eval.npy \
  --scales="$SCALES" \
  --results-dir results/eval_steering_contr_wide \
  --data-dir data/eval_steering_contr_wide \
  2>&1 | tee -a "$LOG"

echo "$(date '+%F %T')  both sweeps complete" | tee -a "$LOG"
