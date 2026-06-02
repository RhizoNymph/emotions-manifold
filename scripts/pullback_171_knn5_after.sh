#!/usr/bin/env bash
# Wait for the in-flight K=3 and knn:10*0.3 chains to release both
# vLLMs, then run a 4-pair K=5 chain using the existing split layout
# (2 pairs per vLLM in parallel, pairs serial within each vLLM).

set -euo pipefail
cd "$(dirname "$0")/.."

LOG=logs/pullback_171_knn5_after.log
mkdir -p logs

echo "$(date '+%F %T')  waiting for K=3 and knn:10*0.3 chains to finish" | tee -a "$LOG"

# Poll for the existing pullback_experiment.py processes to clear.
deadline=$(( $(date +%s) + 7200 ))  # 2h ceiling
while pgrep -f run_pullback_experiment.py >/dev/null 2>&1; do
    if [ "$(date +%s)" -ge "$deadline" ]; then
        echo "$(date '+%F %T')  timed out waiting; bailing" | tee -a "$LOG"
        exit 1
    fi
    sleep 30
done

echo "$(date '+%F %T')  both vLLMs free; starting K=5 split chain" | tee -a "$LOG"

# Reuse the original split-chain pattern (2 pairs each vLLM, parallel
# across vLLMs, serial within).
K=5 bash scripts/pullback_171_adaptive_sigma.sh

echo "$(date '+%F %T')  K=5 chain complete" | tee -a "$LOG"
