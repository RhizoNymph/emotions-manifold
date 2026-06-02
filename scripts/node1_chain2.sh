#!/usr/bin/env bash
# Second node1 chain: pullback on calm→desperate, then σ-sweep on
# terrified→serene. Both via node1 vLLM.

set -euo pipefail
cd "$(dirname "$0")/.."

export VLLM_BASE_URL="http://node1:8000/v1"
LOG=logs/node1_chain2.log
mkdir -p logs

echo "$(date '+%F %T')  node1 chain 2 started; VLLM_BASE_URL=$VLLM_BASE_URL" | tee -a "$LOG"

echo "$(date '+%F %T')  starting pullback on calm→desperate" | tee -a "$LOG"
uv run python -u scripts/run_pullback_experiment.py calm desperate 2>&1 | tee -a "$LOG"

echo "$(date '+%F %T')  starting σ-sweep on terrified→serene" | tee -a "$LOG"
uv run python -u scripts/run_pullback_sigma_sweep.py --start terrified --end serene 2>&1 | tee -a "$LOG"

echo "$(date '+%F %T')  node1 chain 2 complete" | tee -a "$LOG"
