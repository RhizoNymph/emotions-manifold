#!/usr/bin/env bash
# Resume pipeline after API credit top-up.
# 1. Finish building the 171-emotion behavior manifold (resumes from
#    ratings cache; only judges the ~7163 stories that previously
#    errored with credit_balance_too_low)
# 2. Snapshot the now-complete M_y to data/171emotions/
# 3. Re-run isometry check (now 171 vs 171, not 28 vs 27)
# 4. Run the 4-pair pullback comparison at 171-scale, split across
#    both vLLMs
# 5. Append pullback comparison to day3.md

set -euo pipefail
cd "$(dirname "$0")/.."

LOG=logs/resume_chain.log
mkdir -p logs

echo "$(date '+%F %T')  resume chain started" | tee -a "$LOG"

echo "$(date '+%F %T')  completing behavior manifold (resumes from cache)" | tee -a "$LOG"
uv run python -u scripts/build_behavior_manifold.py --corpus data/stories_full.jsonl 2>&1 | tee -a "$LOG"

echo "$(date '+%F %T')  snapshotting completed M_y" | tee -a "$LOG"
cp -a data/manifold_y.npz data/171emotions/manifold_y.npz
cp -a data/story_ratings_stories_full.json data/171emotions/story_ratings_stories_full.json

echo "$(date '+%F %T')  re-running isometry check (171 vs 171)" | tee -a "$LOG"
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

echo "$(date '+%F %T')  resume chain complete" | tee -a "$LOG"
