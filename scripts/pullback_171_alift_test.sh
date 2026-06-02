#!/usr/bin/env bash
# Test the A-lift predictor: pairs with A_lift > +0.2 should show
# pullback wins (M_y-line margin positive), pairs with A_lift < -0.2
# should show pullback losses. All at the production default σ=0.077.
#
# Outputs land at results/pullback/<pair>.json (no suffix), since these
# are new pairs not yet in the dataset.

set -euo pipefail
cd "$(dirname "$0")/.."

LOG=logs/pullback_171_alift_test.log
mkdir -p logs

echo "$(date '+%F %T')  starting A-lift predictor test (6 new pairs)" | tee -a "$LOG"

# Predict WIN: high A_lift (+0.219 to +0.234)
echo "$(date '+%F %T')  localhost (predict-win): hope→unhappy, amused→ashamed, grumpy→hopeful" | tee -a "$LOG"
(
    for pair in "hope unhappy" "amused ashamed" "grumpy hopeful"; do
        set -- $pair
        uv run python -u scripts/run_pullback_experiment.py "$1" "$2" 2>&1 \
            | tee -a logs/pullback_171_alift_predict_win.log
    done
) &
win_pid=$!

# Predict LOSS: low A_lift (-0.209 to -0.229)
echo "$(date '+%F %T')  node1 (predict-loss): proud→sympathetic, brooding→proud, brooding→pleased" | tee -a "$LOG"
(
    export VLLM_BASE_URL="http://node1:8000/v1"
    for pair in "proud sympathetic" "brooding proud" "brooding pleased"; do
        set -- $pair
        uv run python -u scripts/run_pullback_experiment.py "$1" "$2" 2>&1 \
            | tee -a logs/pullback_171_alift_predict_loss.log
    done
) &
loss_pid=$!

wait $win_pid
echo "$(date '+%F %T')  predict-win batch done" | tee -a "$LOG"
wait $loss_pid
echo "$(date '+%F %T')  predict-loss batch done" | tee -a "$LOG"

echo "$(date '+%F %T')  A-lift test chain complete" | tee -a "$LOG"
