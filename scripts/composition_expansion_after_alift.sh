#!/usr/bin/env bash
# Wait for the A-lift expansion to finish, then run composition expansion
# (15 new compositions × 2 conditions each: original-magnitude and
# norm-matched) on the freed vLLMs.

set -euo pipefail
cd "$(dirname "$0")/.."

LOG=logs/composition_expansion.log
mkdir -p logs

echo "$(date '+%F %T')  waiting for A-lift expansion to finish" | tee -a "$LOG"
deadline=$(( $(date +%s) + 21600 ))  # 6h ceiling
while pgrep -f "alift_expansion_chain|run_pullback_experiment" >/dev/null 2>&1; do
    if [ "$(date +%s)" -ge "$deadline" ]; then
        echo "$(date '+%F %T')  timeout waiting for A-lift" | tee -a "$LOG"
        exit 1
    fi
    sleep 30
done

echo "$(date '+%F %T')  A-lift done; running composition expansion" | tee -a "$LOG"

# Norm-matched (the primary condition — original was confounded by magnitude)
uv run python -u scripts/experiments/run_composition_experiment.py \
    --plan-file data/probe/composition_expansion_plan.json \
    --norm-match \
    --results-dir results/composition_expansion_nm \
    --data-dir data/composition_expansion_nm \
    2>&1 | tee -a "$LOG"

# Also run original magnitude condition so we have both
uv run python -u scripts/experiments/run_composition_experiment.py \
    --plan-file data/probe/composition_expansion_plan.json \
    --results-dir results/composition_expansion_raw \
    --data-dir data/composition_expansion_raw \
    2>&1 | tee -a "$LOG"

echo "$(date '+%F %T')  composition expansion complete" | tee -a "$LOG"
