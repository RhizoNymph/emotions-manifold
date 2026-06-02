#!/usr/bin/env bash
# Switch which corpus (30-emotion or 171-emotion) the canonical paths
# point at. Both backups are preserved at data/30emotions/ and
# data/171emotions/; this script copies one set into the canonical
# locations so the dashboard and other scripts using
# config.paths.{emotion_vectors,manifold_h,manifold_y} see that corpus.
#
# Usage:
#   bash scripts/switch_corpus.sh 30
#   bash scripts/switch_corpus.sh 171

set -euo pipefail
cd "$(dirname "$0")/.."

if [ $# -ne 1 ] || ! [[ "$1" =~ ^(30|171)$ ]]; then
    echo "usage: $0 {30|171}"
    exit 1
fi

SRC=data/${1}emotions
if [ ! -d "$SRC" ]; then
    echo "missing backup directory $SRC"
    exit 1
fi

echo "switching canonical artifacts to $1-emotion corpus"
for f in emotion_vectors.npz manifold_h.npz manifold_y.npz; do
    if [ -f "$SRC/$f" ]; then
        cp -a "$SRC/$f" "data/$f"
        echo "  copied $SRC/$f -> data/$f"
    else
        echo "  missing $SRC/$f"
    fi
done

# Geodesic cache (only present for 30; 171 cache lives at canonical)
if [ -f "$SRC/geodesics_cache.npz" ]; then
    cp -a "$SRC/geodesics_cache.npz" "data/geodesics_cache.npz"
    echo "  copied $SRC/geodesics_cache.npz -> data/geodesics_cache.npz"
fi

echo "done. Dashboard and downstream scripts now see $1-emotion artifacts."
