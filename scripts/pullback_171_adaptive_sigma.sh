#!/usr/bin/env bash
# Run the 4-pair pullback comparison at adaptive per-waypoint σ (K-NN
# distance from each y* to its K-th nearest M_y centroid). Tests
# whether adaptive σ rescues both:
#   - terrified→serene (sparse, needs wider σ)
#   - excited→weary, depressed→energized (dense, σ=1.0 collapsed them)
# at the same time.
#
# Outputs land at results/pullback/<pair>_knn{K}.json so they don't
# overwrite either the σ=0.077 defaults or the σ=1.0 runs.

set -euo pipefail
cd "$(dirname "$0")/.."

K=${K:-10}
LOG=logs/pullback_171_knn${K}.log
mkdir -p logs

echo "$(date '+%F %T')  starting adaptive σ chain (knn:${K})" | tee -a "$LOG"

SUFFIX="_knn${K}"

# Run 2 pairs on each vLLM in parallel; pairs sequential within each vLLM.
echo "$(date '+%F %T')  localhost: excited→weary, happy→sad" | tee -a "$LOG"
(
    uv run python -u scripts/run_pullback_experiment.py excited weary \
        --sigma "knn:${K}" --results-suffix "$SUFFIX" 2>&1 \
        | tee -a logs/pullback_171_knn${K}_localhost.log
    uv run python -u scripts/run_pullback_experiment.py happy sad \
        --sigma "knn:${K}" --results-suffix "$SUFFIX" 2>&1 \
        | tee -a logs/pullback_171_knn${K}_localhost.log
) &
local_pid=$!

echo "$(date '+%F %T')  node1: depressed→energized, terrified→serene" | tee -a "$LOG"
(
    export VLLM_BASE_URL="http://node1:8000/v1"
    uv run python -u scripts/run_pullback_experiment.py depressed energized \
        --sigma "knn:${K}" --results-suffix "$SUFFIX" 2>&1 \
        | tee -a logs/pullback_171_knn${K}_node1.log
    uv run python -u scripts/run_pullback_experiment.py terrified serene \
        --sigma "knn:${K}" --results-suffix "$SUFFIX" 2>&1 \
        | tee -a logs/pullback_171_knn${K}_node1.log
) &
node_pid=$!

wait $local_pid
echo "$(date '+%F %T')  localhost pullbacks done" | tee -a "$LOG"
wait $node_pid
echo "$(date '+%F %T')  node1 pullbacks done" | tee -a "$LOG"

echo "$(date '+%F %T')  adaptive σ chain complete (knn:${K})" | tee -a "$LOG"
