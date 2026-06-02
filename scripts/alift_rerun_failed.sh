#!/usr/bin/env bash
# Re-run the 14 A-lift expansion pairs that failed when Anthropic
# credit ran out. Split 7/7 across vLLMs; both run in parallel.

set -euo pipefail
cd "$(dirname "$0")/.."

LOG=logs/alift_rerun.log
mkdir -p logs

# Original localhost failures + 1 rebalance from node1 to keep counts even
LOCALHOST_PAIRS=(
    "energized terrified"
    "anxious droopy"
    "indifferent stressed"
    "cheerful lazy"
    "depressed vengeful"
    "calm ecstatic"
    "happy jealous"
)
NODE1_PAIRS=(
    "at_ease envious"
    "at_ease disdainful"
    "inspired irate"
    "droopy uneasy"
    "content excited"
    "bored unnerved"
    "overwhelmed sluggish"
)

echo "$(date '+%F %T')  starting A-lift rerun: 14 pairs (7 per vLLM)" | tee -a "$LOG"

(
    for pair in "${LOCALHOST_PAIRS[@]}"; do
        s=${pair% *}
        e=${pair#* }
        s=${s//_/ }
        e=${e//_/ }
        echo "$(date '+%F %T')  localhost: $s → $e" | tee -a logs/alift_rerun_localhost.log
        uv run python -u scripts/run_pullback_experiment.py "$s" "$e" 2>&1 \
            | tee -a logs/alift_rerun_localhost.log
    done
) &
local_pid=$!

(
    export VLLM_BASE_URL="http://node1:8000/v1"
    for pair in "${NODE1_PAIRS[@]}"; do
        s=${pair% *}
        e=${pair#* }
        s=${s//_/ }
        e=${e//_/ }
        echo "$(date '+%F %T')  node1: $s → $e" | tee -a logs/alift_rerun_node1.log
        uv run python -u scripts/run_pullback_experiment.py "$s" "$e" 2>&1 \
            | tee -a logs/alift_rerun_node1.log
    done
) &
node_pid=$!

wait $local_pid
echo "$(date '+%F %T')  localhost done" | tee -a "$LOG"
wait $node_pid
echo "$(date '+%F %T')  node1 done" | tee -a "$LOG"

echo "$(date '+%F %T')  A-lift rerun complete (14 pairs)" | tee -a "$LOG"
