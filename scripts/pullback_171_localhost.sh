#!/usr/bin/env bash
# Run pullback at 171-scale on the localhost vLLM half of the 4-pair set:
# excited→weary and happy→sad.
#
# Results land in results/pullback/<pair>.json (same path as 30-scale
# results; we copy the 30-scale results to data/30emotions/pullback/
# beforehand for provenance).

set -euo pipefail
cd "$(dirname "$0")/.."

# Provenance: snapshot the 30-emotion pullback results before any overwrite.
mkdir -p data/30emotions/pullback
for f in results/pullback/excited_weary.json results/pullback/happy_sad.json; do
    if [ -f "$f" ] && [ ! -f "data/30emotions/pullback/$(basename $f)" ]; then
        cp -a "$f" "data/30emotions/pullback/$(basename $f)"
    fi
done

for pair in "excited weary" "happy sad"; do
    set -- $pair
    echo "$(date '+%F %T')  pullback 171-scale: $1 → $2 (localhost)"
    uv run python -u scripts/run_pullback_experiment.py "$1" "$2"
done
