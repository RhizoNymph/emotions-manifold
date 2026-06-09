#!/usr/bin/env bash
# Re-run the n=40 A-lift behavioral steering at 6-D, using the 6-D
# manifold from scripts/setup_6d_pipeline.py. Output to
# results/pullback_6d/. The 8-D production results are NOT touched.
#
# Split 20/20 across both vLLMs. Same pair set as alift_expansion_chain.sh
# plus the 10 original baseline pairs.
#
# Prereq: data/manifold_h_6d_full.npz must exist
# (run scripts/setup_6d_pipeline.py first).

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -f data/manifold_h_6d_full.npz ]]; then
    echo "ERROR: data/manifold_h_6d_full.npz missing."
    echo "Run scripts/setup_6d_pipeline.py first."
    exit 1
fi

LOG=logs/alift_6d.log
mkdir -p logs

# All 40 pairs (10 baseline + 30 expansion), split 20/20
LOCALHOST_PAIRS=(
    # original 5
    "happy sad"
    "excited weary"
    "depressed energized"
    "terrified serene"
    "hope unhappy"
    # expansion predict_win (5)
    "contemptuous hope"
    "contemptuous playful"
    "obstinate proud"
    "disgusted pleased"
    "indignant loving"
    # expansion predict_loss (5)
    "proud puzzled"
    "self-conscious thankful"
    "smug unhappy"
    "jubilant mad"
    "energized terrified"
    # expansion predict_tie (5)
    "anxious droopy"
    "indifferent stressed"
    "cheerful lazy"
    "depressed vengeful"
    "calm ecstatic"
)
NODE1_PAIRS=(
    # original 5
    "amused ashamed"
    "grumpy hopeful"
    "proud sympathetic"
    "brooding proud"
    "brooding pleased"
    # expansion predict_win (5)
    "afraid loving"
    "loving restless"
    "ecstatic puzzled"
    "distressed self-confident"
    "hope sensitive"
    # expansion predict_loss (5)
    "at_ease obstinate"
    "kind vulnerable"
    "at_ease envious"
    "at_ease disdainful"
    "inspired irate"
    # expansion predict_tie (5)
    "happy jealous"
    "droopy uneasy"
    "content excited"
    "bored unnerved"
    "overwhelmed sluggish"
)

echo "$(date '+%F %T')  starting 6-D A-lift: 40 pairs (20 per vLLM)" | tee -a "$LOG"

(
    for pair in "${LOCALHOST_PAIRS[@]}"; do
        s=${pair% *}
        e=${pair#* }
        s=${s//_/ }
        e=${e//_/ }
        echo "$(date '+%F %T')  localhost: $s → $e" | tee -a logs/alift_6d_localhost.log
        uv run python -u scripts/run_pullback_experiment_6d.py "$s" "$e" 2>&1 \
            | tee -a logs/alift_6d_localhost.log
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
        echo "$(date '+%F %T')  node1: $s → $e" | tee -a logs/alift_6d_node1.log
        uv run python -u scripts/run_pullback_experiment_6d.py "$s" "$e" 2>&1 \
            | tee -a logs/alift_6d_node1.log
    done
) &
node_pid=$!

wait $local_pid
echo "$(date '+%F %T')  localhost done" | tee -a "$LOG"
wait $node_pid
echo "$(date '+%F %T')  node1 done" | tee -a "$LOG"

echo "$(date '+%F %T')  6-D A-lift complete (40 pairs)" | tee -a "$LOG"
echo "$(date '+%F %T')  next: uv run python scripts/analyze_geodesic_vs_linear_4d.py" | tee -a "$LOG"
