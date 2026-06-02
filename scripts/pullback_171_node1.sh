#!/usr/bin/env bash
# Run pullback at 171-scale on the node1 vLLM half of the 4-pair set:
# depressed→energized and terrified→serene.

set -euo pipefail
cd "$(dirname "$0")/.."

export VLLM_BASE_URL="http://node1:8000/v1"

mkdir -p data/30emotions/pullback
for f in results/pullback/depressed_energized.json results/pullback/terrified_serene.json; do
    if [ -f "$f" ] && [ ! -f "data/30emotions/pullback/$(basename $f)" ]; then
        cp -a "$f" "data/30emotions/pullback/$(basename $f)"
    fi
done

for pair in "depressed energized" "terrified serene"; do
    set -- $pair
    echo "$(date '+%F %T')  pullback 171-scale: $1 → $2 (node1)"
    uv run python -u scripts/run_pullback_experiment.py "$1" "$2"
done
