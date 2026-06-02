#!/usr/bin/env bash
# Wait for the wider σ-sweep to finish, then re-run the 4-pair pullback
# comparison at the best σ found across the combined 9-point sweep
# (0.05, 0.077, 0.15, 0.30, 0.50, 1.0, 1.5, 2.0, 3.0).
#
# Provenance: results go to results/pullback/<pair>_sigmaXXXX.json,
# preserving the σ=0.077 default-σ results at results/pullback/<pair>.json.

set -euo pipefail
cd "$(dirname "$0")/.."

LOG=logs/pullback_171_best_sigma.log
mkdir -p logs

echo "$(date '+%F %T')  starting; waiting for wider σ-sweep to finish" | tee -a "$LOG"

deadline=$(( $(date +%s) + 3600 ))  # 1h ceiling on wider sweep
while pgrep -f run_pullback_sigma_sweep.py >/dev/null 2>&1; do
    if [ "$(date +%s)" -ge "$deadline" ]; then
        echo "$(date '+%F %T')  wider sweep didn't finish in 1h, bailing" | tee -a "$LOG"
        exit 1
    fi
    sleep 30
done

# Make sure the wider sweep didn't crash
if ! grep -q "σ sweep complete" logs/sigma_sweep_171_wider.log 2>/dev/null; then
    echo "$(date '+%F %T')  wider σ-sweep failed; aborting" | tee -a "$LOG"
    exit 1
fi

# Pick the best σ across the combined sweep
BEST_SIGMA=$(uv run python -c "
import json
from pathlib import Path
best_sigma = None
best_my = float('inf')
for path in sorted(Path('results/pullback_sigma').glob('excited_weary_sigma*.json')):
    row = json.load(open(path))
    if row['my_geodesic_distance'] < best_my:
        best_my = row['my_geodesic_distance']
        best_sigma = row['sigma']
print(f'{best_sigma:.3f}')
" 2>&1 | tail -1)
SIGMA_INT=$(uv run python -c "print(int(round(float('$BEST_SIGMA') * 1000)))" | tail -1)
echo "$(date '+%F %T')  best σ across combined 9-point sweep: $BEST_SIGMA (sigma${SIGMA_INT})" | tee -a "$LOG"

SUFFIX="_sigma${SIGMA_INT}"

# Snapshot existing 30-emotion pullback if not already done
mkdir -p data/30emotions/pullback
for f in results/pullback/excited_weary.json results/pullback/happy_sad.json \
         results/pullback/depressed_energized.json results/pullback/terrified_serene.json \
         results/pullback/calm_desperate.json; do
    bn=$(basename "$f")
    if [ -f "$f" ] && [ ! -f "data/30emotions/pullback/$bn" ]; then
        cp -a "$f" "data/30emotions/pullback/$bn"
    fi
done

# Run 2 pairs on each vLLM in parallel.
echo "$(date '+%F %T')  running localhost pair set: excited→weary, happy→sad at σ=$BEST_SIGMA" | tee -a "$LOG"
(
    uv run python -u scripts/run_pullback_experiment.py excited weary \
        --sigma "$BEST_SIGMA" --results-suffix "$SUFFIX" 2>&1 \
        | tee -a logs/pullback_171_best_sigma_localhost.log
    uv run python -u scripts/run_pullback_experiment.py happy sad \
        --sigma "$BEST_SIGMA" --results-suffix "$SUFFIX" 2>&1 \
        | tee -a logs/pullback_171_best_sigma_localhost.log
) &
local_pid=$!

echo "$(date '+%F %T')  running node1 pair set: depressed→energized, terrified→serene at σ=$BEST_SIGMA" | tee -a "$LOG"
(
    export VLLM_BASE_URL="http://node1:8000/v1"
    uv run python -u scripts/run_pullback_experiment.py depressed energized \
        --sigma "$BEST_SIGMA" --results-suffix "$SUFFIX" 2>&1 \
        | tee -a logs/pullback_171_best_sigma_node1.log
    uv run python -u scripts/run_pullback_experiment.py terrified serene \
        --sigma "$BEST_SIGMA" --results-suffix "$SUFFIX" 2>&1 \
        | tee -a logs/pullback_171_best_sigma_node1.log
) &
node_pid=$!

wait $local_pid
echo "$(date '+%F %T')  localhost pullbacks done" | tee -a "$LOG"
wait $node_pid
echo "$(date '+%F %T')  node1 pullbacks done" | tee -a "$LOG"

echo "$(date '+%F %T')  best-σ chain complete" | tee -a "$LOG"
