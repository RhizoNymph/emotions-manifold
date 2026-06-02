#!/usr/bin/env bash
# Recovery: scale_chain.sh's extract_full step false-failed (HTTP
# response missing capture_results.filesystem even though .bin files
# wrote to disk). 9,900 captures are present and 171 emotions are
# fully covered at ≥50 captures each.
#
# This script runs the remaining post-extraction pipeline:
#   1. compute_vectors → emotion_vectors.npz
#   2. fit_manifold → manifold_h.npz
#   3. build_behavior_manifold → manifold_y.npz (judges 8550 stories)
#   4. check_isometry → isometry.json
#   5. backup 171-emotion artifacts to data/171emotions/
#   6. start_day3.py
#   7. kick off 171-scale pullbacks (split across both vLLMs)
#   8. kick off 171-emotion geodesic precompute

set -euo pipefail
cd "$(dirname "$0")/.."

LOG=logs/recovery_chain.log
mkdir -p logs

echo "$(date '+%F %T')  recovery chain started" | tee -a "$LOG"

echo "$(date '+%F %T')  computing emotion vectors from existing captures" | tee -a "$LOG"
uv run python -u scripts/compute_vectors.py 2>&1 | tee -a "$LOG"

echo "$(date '+%F %T')  fitting M_h manifold" | tee -a "$LOG"
uv run python -u scripts/fit_manifold.py 2>&1 | tee -a "$LOG"

echo "$(date '+%F %T')  building behavior manifold (judges all 8550 stories)" | tee -a "$LOG"
uv run python -u scripts/build_behavior_manifold.py --corpus data/stories_full.jsonl 2>&1 | tee -a "$LOG"

echo "$(date '+%F %T')  running isometry check" | tee -a "$LOG"
uv run python -u scripts/check_isometry.py 2>&1 | tee -a "$LOG"

echo "$(date '+%F %T')  backing up 171-emotion artifacts" | tee -a "$LOG"
mkdir -p data/171emotions
for f in emotion_vectors.npz manifold_h.npz manifold_y.npz; do
    if [ -f "data/$f" ]; then
        cp -a "data/$f" "data/171emotions/$f"
        echo "  backed up data/$f" | tee -a "$LOG"
    fi
done
if [ -f data/story_ratings_stories_full.json ]; then
    cp -a data/story_ratings_stories_full.json data/171emotions/story_ratings_stories_full.json
    echo "  backed up data/story_ratings_stories_full.json" | tee -a "$LOG"
fi

echo "$(date '+%F %T')  initializing day3.md" | tee -a "$LOG"
uv run python -u scripts/start_day3.py 2>&1 | tee -a "$LOG"

echo "$(date '+%F %T')  kicking off 171-scale pullbacks (parallel across vLLMs)" | tee -a "$LOG"
bash scripts/pullback_171_localhost.sh 2>&1 | tee -a logs/pullback_171_localhost.log &
local_pid=$!
bash scripts/pullback_171_node1.sh 2>&1 | tee -a logs/pullback_171_node1.log &
node_pid=$!
wait $local_pid
echo "$(date '+%F %T')  localhost pullbacks done" | tee -a "$LOG"
wait $node_pid
echo "$(date '+%F %T')  node1 pullbacks done" | tee -a "$LOG"

echo "$(date '+%F %T')  appending pullback comparison to day3.md" | tee -a "$LOG"
uv run python -u scripts/append_pullback_to_day3.py 2>&1 | tee -a "$LOG"

echo "$(date '+%F %T')  pre-computing geodesics for the 171-emotion manifold (~4h on CPU)" | tee -a "$LOG"
uv run python -u scripts/precompute_geodesics.py --force 2>&1 | tee -a "$LOG"

echo "$(date '+%F %T')  recovery chain complete" | tee -a "$LOG"
