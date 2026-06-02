#!/usr/bin/env bash
# Run the post-generation pipeline at 171-emotion scale:
#  1. wait for generate_full to finish
#  2. extract activations (same vLLM)
#  3. compute emotion vectors
#  4. build behavior manifold (judges every story)
#  5. fit M_h manifold at the current subspace dim (8 unless changed)
#  6. isometry check
#
# Usage:
#   bash scripts/scale_chain.sh

set -euo pipefail
cd "$(dirname "$0")/.."

LOG=logs/scale_chain.log
mkdir -p logs

echo "$(date '+%F %T')  scale chain started; waiting for generate_full" | tee -a "$LOG"

deadline=$(( $(date +%s) + 6 * 3600 ))  # 6h ceiling on the corpus generation
while pgrep -f generate_full.py >/dev/null 2>&1; do
    if [ "$(date +%s)" -ge "$deadline" ]; then
        echo "$(date '+%F %T')  generate_full still running after 6h, bailing" | tee -a "$LOG"
        exit 1
    fi
    sleep 60
done

if [ ! -s data/stories_full.jsonl ]; then
    echo "$(date '+%F %T')  generate_full exited but no corpus file; aborting" | tee -a "$LOG"
    exit 1
fi

echo "$(date '+%F %T')  generate_full finished; $(wc -l < data/stories_full.jsonl) stories" | tee -a "$LOG"

echo "$(date '+%F %T')  extracting activations" | tee -a "$LOG"
uv run python -u scripts/extract_full.py 2>&1 | tee -a "$LOG"

echo "$(date '+%F %T')  computing emotion vectors" | tee -a "$LOG"
uv run python -u scripts/compute_vectors.py 2>&1 | tee -a "$LOG"

echo "$(date '+%F %T')  building behavior manifold (judges all stories)" | tee -a "$LOG"
uv run python -u scripts/build_behavior_manifold.py --corpus data/stories_full.jsonl 2>&1 | tee -a "$LOG"

echo "$(date '+%F %T')  fitting M_h manifold" | tee -a "$LOG"
uv run python -u scripts/fit_manifold.py 2>&1 | tee -a "$LOG"

echo "$(date '+%F %T')  running isometry check" | tee -a "$LOG"
uv run python -u scripts/check_isometry.py 2>&1 | tee -a "$LOG"

echo "$(date '+%F %T')  scale chain complete" | tee -a "$LOG"
