#!/usr/bin/env bash
# Batched-judge variant of alift_8d_silverman_chain.sh (or any chain).
#
# Two phases:
#   Phase 1 — generate trajectories ONLY (no judging), split across both
#             vLLMs. Same pair set as the sequential chain it replaces.
#   Phase 2 — submit a single batch to Anthropic Batches API covering all
#             pairs' completions, wait once (~0–60 min), distribute results
#             back into per-pair summaries.
#
# Wall-time: ~max(phase1 per node, phase1 per node) + phase2
#   For n=40, ~2.5h + ~0.5h ≈ 3h, vs ~7–8 h for the sequential pipeline.
# Cost: 50 % of sequential.
#
# Usage:
#   bash scripts/alift_batched_chain.sh <chain> <manifold> <pairs_file>
#
#   <chain>       chain subdirectory under data/ and results/, e.g.
#                 pullback_8d_silverman, pullback_6d
#   <manifold>    path to the FittedManifold .npz, e.g.
#                 data/manifold_h_8d_silverman.npz
#   <pairs_file>  newline-delimited '<a> <b>' file (multi-word labels with
#                 underscores), e.g. results/alift_n100_extension/sampled_pairs.txt

set -euo pipefail
cd "$(dirname "$0")/.."

CHAIN=${1:?usage: $0 <chain> <manifold> <pairs_file>}
MANIFOLD=${2:?usage: $0 <chain> <manifold> <pairs_file>}
PAIRS_FILE=${3:?usage: $0 <chain> <manifold> <pairs_file>}

if [[ ! -f $MANIFOLD ]]; then
    echo "ERROR: manifold $MANIFOLD missing." >&2; exit 1
fi
if [[ ! -f $PAIRS_FILE ]]; then
    echo "ERROR: pairs file $PAIRS_FILE missing." >&2; exit 1
fi

LOG=logs/alift_batched_${CHAIN}.log
mkdir -p logs

mapfile -t ALL_PAIRS < "$PAIRS_FILE"
n_pairs=${#ALL_PAIRS[@]}
half=$(( (n_pairs + 1) / 2 ))
LOCALHOST_PAIRS=("${ALL_PAIRS[@]:0:$half}")
NODE1_PAIRS=("${ALL_PAIRS[@]:$half}")

echo "$(date '+%F %T')  starting batched chain $CHAIN: $n_pairs pairs (Phase 1 split $half / $((n_pairs - half)))" | tee -a "$LOG"

# ---------- Phase 1: parallel generation across nodes (NO judging) ---------
(
    for pair in "${LOCALHOST_PAIRS[@]}"; do
        s=${pair% *}; e=${pair#* }
        s=${s//_/ }; e=${e//_/ }
        echo "$(date '+%F %T')  localhost (nojudge): $s → $e" | tee -a logs/alift_batched_${CHAIN}_localhost.log
        PYTHONUNBUFFERED=1 uv run python -u scripts/run_pullback_experiment_nojudge.py \
            "$s" "$e" --manifold "$MANIFOLD" --chain "$CHAIN" 2>&1 \
            | tee -a logs/alift_batched_${CHAIN}_localhost.log
    done
) &
local_pid=$!

(
    export VLLM_BASE_URL="http://node1:8000/v1"
    for pair in "${NODE1_PAIRS[@]}"; do
        s=${pair% *}; e=${pair#* }
        s=${s//_/ }; e=${e//_/ }
        echo "$(date '+%F %T')  node1 (nojudge): $s → $e" | tee -a logs/alift_batched_${CHAIN}_node1.log
        PYTHONUNBUFFERED=1 uv run python -u scripts/run_pullback_experiment_nojudge.py \
            "$s" "$e" --manifold "$MANIFOLD" --chain "$CHAIN" 2>&1 \
            | tee -a logs/alift_batched_${CHAIN}_node1.log
    done
) &
node_pid=$!

wait $local_pid; echo "$(date '+%F %T')  localhost Phase 1 done" | tee -a "$LOG"
wait $node_pid;  echo "$(date '+%F %T')  node1 Phase 1 done"     | tee -a "$LOG"

# ---------- Phase 2: single batched judge ----------------------------------
echo "$(date '+%F %T')  Phase 2: submitting batch to Anthropic API" | tee -a "$LOG"
PYTHONUNBUFFERED=1 uv run python -u scripts/run_chain_batched_judge.py \
    --chain "$CHAIN" --pairs-file "$PAIRS_FILE" 2>&1 | tee -a "$LOG"
echo "$(date '+%F %T')  Phase 2 complete" | tee -a "$LOG"

echo "$(date '+%F %T')  batched chain $CHAIN complete" | tee -a "$LOG"
