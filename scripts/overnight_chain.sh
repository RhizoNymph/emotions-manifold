#!/usr/bin/env bash
# Overnight chain: waits for the running subspace sweep to finish, then runs
# pullback + regenerates all plots. Designed to be backgrounded after the
# sweep has already started, so it just polls for sweep completion.
#
# Usage:
#   ./scripts/overnight_chain.sh

set -euo pipefail

cd "$(dirname "$0")/.."

LOG=logs/overnight_chain.log
mkdir -p logs

echo "$(date '+%F %T')  overnight chain started, waiting for sweep" | tee -a "$LOG"

# Poll for absence of the sweep process. Use a generous timeout (8 hours).
deadline=$(( $(date +%s) + 8 * 3600 ))
while pgrep -f run_subspace_sweep.py >/dev/null 2>&1; do
    if [ "$(date +%s)" -ge "$deadline" ]; then
        echo "$(date '+%F %T')  sweep still running after 8h, bailing" | tee -a "$LOG"
        exit 1
    fi
    sleep 60
done

echo "$(date '+%F %T')  sweep finished, regenerating subspace plots" | tee -a "$LOG"
uv run python scripts/plot_subspace_sweep.py 2>&1 | tee -a "$LOG"

echo "$(date '+%F %T')  running pullback experiment (default pairs)" | tee -a "$LOG"
uv run python scripts/run_pullback_experiment.py 2>&1 | tee -a "$LOG"

echo "$(date '+%F %T')  plotting pullback results" | tee -a "$LOG"
uv run python scripts/plot_pullback.py 2>&1 | tee -a "$LOG"

echo "$(date '+%F %T')  overnight chain complete" | tee -a "$LOG"
