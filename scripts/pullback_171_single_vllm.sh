#!/usr/bin/env bash
# Run all 4 pullback pairs on a single vLLM (sequential), with a given
# sigma spec. Outputs use the spec as a suffix.
#
# Env vars:
#   SIGMA_SPEC   — passed to --sigma (e.g. "knn:3", "knn:10*0.3", "1.0")
#   SUFFIX       — appended to result filenames (e.g. "_knn3", "_knn10x03")
#   BASE_URL     — VLLM_BASE_URL to export (e.g. http://localhost:8000/v1)
#   LOG_TAG      — used in log filename (default = SUFFIX without leading _)

set -euo pipefail
cd "$(dirname "$0")/.."

SPEC=${SIGMA_SPEC:?SIGMA_SPEC required}
SUF=${SUFFIX:?SUFFIX required}
URL=${BASE_URL:-http://localhost:8000/v1}
TAG=${LOG_TAG:-${SUF#_}}

LOG=logs/pullback_171_${TAG}.log
mkdir -p logs

echo "$(date '+%F %T')  starting on $URL, spec=$SPEC, suffix=$SUF" | tee -a "$LOG"

export VLLM_BASE_URL="$URL"

for pair in "excited weary" "depressed energized" "happy sad" "terrified serene"; do
    echo "$(date '+%F %T')  pair: $pair" | tee -a "$LOG"
    set -- $pair
    uv run python -u scripts/run_pullback_experiment.py "$1" "$2" \
        --sigma "$SPEC" --results-suffix "$SUF" 2>&1 \
        | tee -a "$LOG"
done

echo "$(date '+%F %T')  chain complete (spec=$SPEC)" | tee -a "$LOG"
