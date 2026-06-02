#!/usr/bin/env bash
# Re-run the K=3 4-pair pullback with _v2 suffix, preserving the
# original judge-error-noisy run at results/pullback/*_knn3.json.
# Split across both vLLMs (2 pairs each, sequential within vLLM).

set -euo pipefail
cd "$(dirname "$0")/.."

LOG=logs/pullback_171_knn3_v2.log
mkdir -p logs

SUFFIX="_knn3_v2"
SPEC="knn:3"

echo "$(date '+%F %T')  starting K=3 rerun (suffix $SUFFIX)" | tee -a "$LOG"

echo "$(date '+%F %T')  localhost: excited→weary, happy→sad" | tee -a "$LOG"
(
    uv run python -u scripts/run_pullback_experiment.py excited weary \
        --sigma "$SPEC" --results-suffix "$SUFFIX" 2>&1 \
        | tee -a logs/pullback_171_knn3_v2_localhost.log
    uv run python -u scripts/run_pullback_experiment.py happy sad \
        --sigma "$SPEC" --results-suffix "$SUFFIX" 2>&1 \
        | tee -a logs/pullback_171_knn3_v2_localhost.log
) &
local_pid=$!

echo "$(date '+%F %T')  node1: depressed→energized, terrified→serene" | tee -a "$LOG"
(
    export VLLM_BASE_URL="http://node1:8000/v1"
    uv run python -u scripts/run_pullback_experiment.py depressed energized \
        --sigma "$SPEC" --results-suffix "$SUFFIX" 2>&1 \
        | tee -a logs/pullback_171_knn3_v2_node1.log
    uv run python -u scripts/run_pullback_experiment.py terrified serene \
        --sigma "$SPEC" --results-suffix "$SUFFIX" 2>&1 \
        | tee -a logs/pullback_171_knn3_v2_node1.log
) &
node_pid=$!

wait $local_pid
echo "$(date '+%F %T')  localhost done" | tee -a "$LOG"
wait $node_pid
echo "$(date '+%F %T')  node1 done" | tee -a "$LOG"

echo "$(date '+%F %T')  K=3 v2 chain complete" | tee -a "$LOG"
