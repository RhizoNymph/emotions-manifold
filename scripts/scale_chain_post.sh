#!/usr/bin/env bash
# Post-scale-chain pipeline: runs once the 171-emotion manifold has
# been fitted and the canonical artifacts have been overwritten.
#
# Steps:
#   1. Wait for scale_chain.sh to complete
#   2. Back up the 171-emotion artifacts to data/171emotions/ so we
#      retain provenance for the new manifold
#   3. Start day3.md with the new manifold's structural properties
#   4. Re-run the 4 existing pullback pairs at 171-scale, split across
#      both vLLMs (excited→weary + happy→sad on localhost,
#      depressed→energized + terrified→serene on node1) — should
#      finish in ~50 min wall time, ~04:30
#   5. Pre-compute geodesics for the new 171-emotion manifold so the
#      dashboard works at the new scale (~4 hours on CPU)
#
# Usage:
#   bash scripts/scale_chain_post.sh

set -euo pipefail
cd "$(dirname "$0")/.."

LOG=logs/scale_chain_post.log
mkdir -p logs

echo "$(date '+%F %T')  scale_chain_post started; waiting for scale_chain" | tee -a "$LOG"

deadline=$(( $(date +%s) + 8 * 3600 ))
while pgrep -f scale_chain.sh >/dev/null 2>&1; do
    if [ "$(date +%s)" -ge "$deadline" ]; then
        echo "$(date '+%F %T')  scale_chain still running after 8h, bailing" | tee -a "$LOG"
        exit 1
    fi
    sleep 60
done

# Defensive check: make sure scale_chain succeeded
if ! grep -q "scale chain complete" logs/scale_chain.log; then
    echo "$(date '+%F %T')  scale_chain did not log completion; aborting" | tee -a "$LOG"
    exit 1
fi

echo "$(date '+%F %T')  scale_chain complete; backing up 171-emotion artifacts" | tee -a "$LOG"
mkdir -p data/171emotions
for f in emotion_vectors.npz manifold_h.npz manifold_y.npz; do
    if [ -f "data/$f" ]; then
        cp -a "data/$f" "data/171emotions/$f"
        echo "  backed up data/$f" | tee -a "$LOG"
    fi
done
for f in data/story_ratings_stories_full.json data/captures; do
    if [ -e "$f" ]; then
        # captures dir is huge; just record the path, don't copy
        if [ -d "$f" ]; then
            echo "  capture dir retained at $f" | tee -a "$LOG"
        else
            cp -a "$f" "data/171emotions/$(basename $f)"
            echo "  backed up $f" | tee -a "$LOG"
        fi
    fi
done

echo "$(date '+%F %T')  starting day3.md skeleton" | tee -a "$LOG"
uv run python -u scripts/start_day3.py 2>&1 | tee -a "$LOG"

echo "$(date '+%F %T')  kicking off 171-scale pullback experiments" | tee -a "$LOG"
# Split across both vLLMs for ~50 min wall time vs ~100 min sequential
bash scripts/pullback_171_localhost.sh 2>&1 | tee -a "$LOG" &
local_pid=$!
bash scripts/pullback_171_node1.sh 2>&1 | tee -a "$LOG" &
node_pid=$!
wait $local_pid
wait $node_pid

echo "$(date '+%F %T')  171-scale pullbacks done; appending to day3.md" | tee -a "$LOG"
uv run python -u scripts/append_pullback_to_day3.py 2>&1 | tee -a "$LOG"

echo "$(date '+%F %T')  pre-computing geodesics for the new manifold (long, ~4h)" | tee -a "$LOG"
uv run python -u scripts/precompute_geodesics.py --force 2>&1 | tee -a "$LOG"

echo "$(date '+%F %T')  scale_chain_post complete" | tee -a "$LOG"
