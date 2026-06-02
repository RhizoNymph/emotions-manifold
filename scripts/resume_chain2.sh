#!/usr/bin/env bash
# Continuation after label-mismatch fix: skip M_y judging (already
# done), re-run isometry (now 171 vs 171 instead of 168 vs 168), then
# the 4-pair pullback comparison.

set -euo pipefail
cd "$(dirname "$0")/.."

LOG=logs/resume_chain2.log
mkdir -p logs

echo "$(date '+%F %T')  resume chain 2 started" | tee -a "$LOG"

echo "$(date '+%F %T')  re-running isometry check (171 vs 171 after label fix)" | tee -a "$LOG"
uv run python -u scripts/check_isometry.py 2>&1 | tee -a "$LOG"

echo "$(date '+%F %T')  running 171-scale pullback (parallel across vLLMs)" | tee -a "$LOG"
bash scripts/pullback_171_localhost.sh 2>&1 | tee -a logs/pullback_171_localhost.log &
local_pid=$!
bash scripts/pullback_171_node1.sh 2>&1 | tee -a logs/pullback_171_node1.log &
node_pid=$!
wait $local_pid
echo "$(date '+%F %T')  localhost pullbacks done" | tee -a "$LOG"
wait $node_pid
echo "$(date '+%F %T')  node1 pullbacks done" | tee -a "$LOG"

echo "$(date '+%F %T')  refreshing pullback plots" | tee -a "$LOG"
uv run python scripts/plot_pullback.py 2>&1 | tee -a "$LOG"

echo "$(date '+%F %T')  appending pullback comparison to day3.md" | tee -a "$LOG"
uv run python -u scripts/append_pullback_to_day3.py 2>&1 | tee -a "$LOG"

echo "$(date '+%F %T')  resume chain 2 complete" | tee -a "$LOG"
