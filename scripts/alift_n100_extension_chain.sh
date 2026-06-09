#!/usr/bin/env bash
# Run the n=100+ extension: 60 NEW pairs (stratified by A_lift quintile)
# using the production 8-D pullback experiment. Output to
# results/pullback/ alongside existing 40-pair results.
#
# Split 30/30 across both vLLMs.
#
# Prereq: scripts/sample_n100_extension_pairs.py has been run, producing
# results/alift_n100_extension/sampled_pairs.txt.

set -euo pipefail
cd "$(dirname "$0")/.."

PAIRS_FILE=results/alift_n100_extension/sampled_pairs.txt
if [[ ! -f "$PAIRS_FILE" ]]; then
    echo "ERROR: $PAIRS_FILE missing."
    echo "Run scripts/sample_n100_extension_pairs.py first."
    exit 1
fi

LOG=logs/alift_n100_extension.log
mkdir -p logs

# Read all 60 pairs, split first 30 to localhost and second 30 to node1.
mapfile -t ALL_PAIRS < "$PAIRS_FILE"
LOCALHOST_PAIRS=("${ALL_PAIRS[@]:0:30}")
NODE1_PAIRS=("${ALL_PAIRS[@]:30:30}")

echo "$(date '+%F %T')  starting n=100+ extension: 60 pairs (30 per vLLM)" | tee -a "$LOG"

(
    for pair in "${LOCALHOST_PAIRS[@]}"; do
        s=${pair% *}
        e=${pair#* }
        s=${s//_/ }
        e=${e//_/ }
        echo "$(date '+%F %T')  localhost: $s → $e" | tee -a logs/alift_n100_localhost.log
        uv run python -u scripts/run_pullback_experiment.py "$s" "$e" 2>&1 \
            | tee -a logs/alift_n100_localhost.log
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
        echo "$(date '+%F %T')  node1: $s → $e" | tee -a logs/alift_n100_node1.log
        uv run python -u scripts/run_pullback_experiment.py "$s" "$e" 2>&1 \
            | tee -a logs/alift_n100_node1.log
    done
) &
node_pid=$!

wait $local_pid
echo "$(date '+%F %T')  localhost done" | tee -a "$LOG"
wait $node_pid
echo "$(date '+%F %T')  node1 done" | tee -a "$LOG"

echo "$(date '+%F %T')  n=100+ extension complete (60 new pairs)" | tee -a "$LOG"
