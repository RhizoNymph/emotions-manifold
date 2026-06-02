#!/usr/bin/env bash
# Wait for refusal probe v2 to finish, then run eval-awareness probe.

set -euo pipefail
cd "$(dirname "$0")/.."

LOG=logs/eval_awareness_after_refusal.log
mkdir -p logs

echo "$(date '+%F %T')  waiting for refusal probe v2 to finish" | tee -a "$LOG"
deadline=$(( $(date +%s) + 3600 ))
while pgrep -f run_refusal_probe.py >/dev/null 2>&1; do
    if [ "$(date +%s)" -ge "$deadline" ]; then
        echo "$(date '+%F %T')  timed out" | tee -a "$LOG"
        exit 1
    fi
    sleep 15
done
echo "$(date '+%F %T')  refusal done, starting eval-awareness probe" | tee -a "$LOG"

uv run python -u scripts/run_eval_awareness_probe.py 2>&1 | tee -a logs/eval_awareness.log

echo "$(date '+%F %T')  eval-awareness probe complete" | tee -a "$LOG"
