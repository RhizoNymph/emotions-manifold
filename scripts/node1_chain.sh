#!/usr/bin/env bash
# Run the steering-scale sweep then the pullback experiment on node1.
# Both jobs use VLLM_BASE_URL=http://node1:8000/v1 so they hit node1's
# vLLM, leaving localhost's vLLM free for the parallel corpus scale-up.
#
# Captures are not written by either of these jobs (steering returns
# generated text in the HTTP response), so no filesystem coupling.
#
# Usage:
#   bash scripts/node1_chain.sh

set -euo pipefail
cd "$(dirname "$0")/.."

export VLLM_BASE_URL="http://node1:8000/v1"
LOG=logs/node1_chain.log
mkdir -p logs

echo "$(date '+%F %T')  node1 chain started; VLLM_BASE_URL=$VLLM_BASE_URL" | tee -a "$LOG"

echo "$(date '+%F %T')  starting steering-scale sweep on depressed→energized" | tee -a "$LOG"
uv run python -u scripts/run_scale_sweep.py 2>&1 | tee -a "$LOG"

echo "$(date '+%F %T')  starting pullback experiment on depressed→energized" | tee -a "$LOG"
uv run python -u scripts/run_pullback_experiment.py depressed energized 2>&1 | tee -a "$LOG"

echo "$(date '+%F %T')  node1 chain complete" | tee -a "$LOG"
