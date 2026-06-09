#!/usr/bin/env bash
# Run time-varying steering on a small set of pairs across both vLLM nodes.
# Split: 5 pairs on localhost, 5 pairs on node1, all using K=8 segments.
#
# These pairs are chosen as a mix of A_lift quadrants from the n=40 set
# so the time-varying vs constant-vector comparison is informative across
# the A_lift atlas.

set -euo pipefail
cd "$(dirname "$0")/.."

LOG=logs/time_varying_chain.log
mkdir -p logs

# Pairs already in default (happy_sad, excited_weary, terrified_serene)
# are handled by the default-args run; add new ones here.
LOCALHOST_PAIRS=(
    "hope unhappy"            # high +A_lift in baseline
    "amused ashamed"          # high +A_lift
    "depressed energized"     # tie / mid
    "calm ecstatic"           # tie
    "brooding proud"          # -A_lift
)
NODE1_PAIRS=(
    "grumpy hopeful"          # +A_lift
    "proud sympathetic"       # -A_lift
    "brooding pleased"        # -A_lift
    "contemptuous hope"       # predict-win
    "at ease disdainful"      # predict-loss
)

echo "$(date '+%F %T')  starting time-varying chain (10 pairs, 5 per node)" | tee -a "$LOG"

(
    for pair in "${LOCALHOST_PAIRS[@]}"; do
        s=${pair% *}
        e=${pair#* }
        echo "$(date '+%F %T')  localhost: $s → $e" | tee -a logs/time_varying_localhost.log
        uv run python -u scripts/run_time_varying_steering.py "$s" "$e" 2>&1 \
            | tee -a logs/time_varying_localhost.log
    done
) &
local_pid=$!

(
    export VLLM_BASE_URL="http://node1:8000/v1"
    for pair in "${NODE1_PAIRS[@]}"; do
        s=${pair% *}
        e=${pair#* }
        echo "$(date '+%F %T')  node1: $s → $e" | tee -a logs/time_varying_node1.log
        uv run python -u scripts/run_time_varying_steering.py "$s" "$e" 2>&1 \
            | tee -a logs/time_varying_node1.log
    done
) &
node_pid=$!

wait $local_pid
echo "$(date '+%F %T')  localhost done" | tee -a "$LOG"
wait $node_pid
echo "$(date '+%F %T')  node1 done" | tee -a "$LOG"

echo "$(date '+%F %T')  time-varying chain complete (10 pairs)" | tee -a "$LOG"
