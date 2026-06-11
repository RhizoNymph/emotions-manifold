#!/usr/bin/env bash
# Wait for the tone experiment to finish, then run the refusal probe.

set -euo pipefail
cd "$(dirname "$0")/.."

LOG=logs/refusal_after_tone.log
mkdir -p logs

echo "$(date '+%F %T')  waiting for tone experiment to finish" | tee -a "$LOG"
deadline=$(( $(date +%s) + 3600 ))  # 1h ceiling
while pgrep -f run_tone_experiment.py >/dev/null 2>&1; do
    if [ "$(date +%s)" -ge "$deadline" ]; then
        echo "$(date '+%F %T')  tone experiment didn't finish in 1h, bailing" | tee -a "$LOG"
        exit 1
    fi
    sleep 15
done
echo "$(date '+%F %T')  tone done, starting refusal probe" | tee -a "$LOG"

uv run python -u scripts/experiments/run_refusal_probe.py 2>&1 | tee -a logs/refusal_probe.log

echo "$(date '+%F %T')  refusal probe complete" | tee -a "$LOG"
