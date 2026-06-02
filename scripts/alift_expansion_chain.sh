#!/usr/bin/env bash
# Run 30 more pairs for A-lift validation expansion. Split 15 across
# each vLLM (sequential within); both vLLMs in parallel.

set -euo pipefail
cd "$(dirname "$0")/.."

LOG=logs/alift_expansion.log
mkdir -p logs

# 30 pairs split into two halves of 15. Mix top/bottom/middle within
# each half so if one vLLM fails we still have a balanced sample.
LOCALHOST_PAIRS=(
    # predict win
    "contemptuous hope"
    "contemptuous playful"
    "obstinate proud"
    "disgusted pleased"
    "indignant loving"
    # predict loss
    "proud puzzled"
    "self-conscious thankful"
    "smug unhappy"
    "jubilant mad"
    "energized terrified"
    # predict tie
    "anxious droopy"
    "indifferent stressed"
    "cheerful lazy"
    "depressed vengeful"
    "calm ecstatic"
)
NODE1_PAIRS=(
    # predict win
    "afraid loving"
    "loving restless"
    "ecstatic puzzled"
    "distressed self-confident"
    "hope sensitive"
    # predict loss
    "at_ease obstinate"
    "kind vulnerable"
    "at_ease envious"
    "at_ease disdainful"
    "inspired irate"
    # predict tie
    "happy jealous"
    "droopy uneasy"
    "content excited"
    "bored unnerved"
    "overwhelmed sluggish"
)

# Note: "at ease" has a space; vLLM/run_pullback_experiment.py handles
# labels via the labels list. We need to pass them with quotes preserved.

echo "$(date '+%F %T')  starting A-lift expansion: 30 pairs (15 per vLLM)" | tee -a "$LOG"

(
    for pair in "${LOCALHOST_PAIRS[@]}"; do
        # If pair contains underscore (placeholder for space), replace it
        s=${pair% *}
        e=${pair#* }
        s=${s//_/ }
        e=${e//_/ }
        echo "$(date '+%F %T')  localhost: $s → $e" | tee -a logs/alift_expansion_localhost.log
        uv run python -u scripts/run_pullback_experiment.py "$s" "$e" 2>&1 \
            | tee -a logs/alift_expansion_localhost.log
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
        echo "$(date '+%F %T')  node1: $s → $e" | tee -a logs/alift_expansion_node1.log
        uv run python -u scripts/run_pullback_experiment.py "$s" "$e" 2>&1 \
            | tee -a logs/alift_expansion_node1.log
    done
) &
node_pid=$!

wait $local_pid
echo "$(date '+%F %T')  localhost done" | tee -a "$LOG"
wait $node_pid
echo "$(date '+%F %T')  node1 done" | tee -a "$LOG"

echo "$(date '+%F %T')  A-lift expansion complete (30 pairs)" | tee -a "$LOG"
