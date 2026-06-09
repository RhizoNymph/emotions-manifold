#!/usr/bin/env bash
# End-to-end batched orchestrator: re-judge broken Silverman pairs +
# run the n=100+ extension. Sequential between the two; each chain uses
# the batched pipeline (Phase 1 generation across both vLLMs, then one
# Phase 2 batch judge submission).
#
# Wall time estimate: ~6-7h end-to-end
#   Silverman gen (36 pairs / 2 nodes)         ~2h
#   Silverman batch judge                      ~30-60min
#   Silverman analyze                          ~1min
#   n=100 gen (60 pairs / 2 nodes)            ~3.5h
#   n=100 batch judge                          ~30-60min
#   n=100 analyze                              ~1min
#
# Cost estimate: ~$50 total (half of sequential pipeline)
#   Silverman re-judge: 36 × 900 × $0.0005     ~$16
#   n=100 extension:    60 × 900 × $0.0005     ~$27
#
# Prereqs:
#   - /tmp/silv_redo/pairs.txt (36 broken Silverman pairs)
#   - /tmp/n100/pairs.txt (60 sampled extension pairs)
#   - data/manifold_h_8d_silverman.npz exists
#   - data/manifold_h.npz exists (production 8-D)

set -euo pipefail
cd "$(dirname "$0")/.."

LOG=logs/batched_silv_and_n100.log
mkdir -p logs

echo "$(date '+%F %T')  ==== orchestrator started ====" | tee -a "$LOG"

# ---------- Silverman (re-judge broken 36) ----------
echo "$(date '+%F %T')  Silverman: Phase 1+2 via batched pipeline" | tee -a "$LOG"
PYTHONUNBUFFERED=1 bash scripts/alift_batched_chain.sh \
    pullback_8d_silverman \
    data/manifold_h_8d_silverman.npz \
    /tmp/silv_redo/pairs.txt 2>&1 | tee -a "$LOG"
echo "$(date '+%F %T')  Silverman complete" | tee -a "$LOG"

# Analyze Silverman now that the previously-empty pairs are filled in
echo "$(date '+%F %T')  Silverman: analyze" | tee -a "$LOG"
PYTHONUNBUFFERED=1 uv run python scripts/analyze_8d_silverman.py 2>&1 | tee -a "$LOG"
echo "$(date '+%F %T')  Silverman analyze done" | tee -a "$LOG"

# ---------- n=100+ extension ----------
echo "$(date '+%F %T')  n=100+: Phase 1+2 via batched pipeline" | tee -a "$LOG"
PYTHONUNBUFFERED=1 bash scripts/alift_batched_chain.sh \
    pullback \
    data/manifold_h.npz \
    /tmp/n100/pairs.txt 2>&1 | tee -a "$LOG"
echo "$(date '+%F %T')  n=100+ complete" | tee -a "$LOG"

# Analyze n=100+ (the analyze script globs ALL pairs in results/pullback/,
# which now contains both the original 40 and the new 60)
echo "$(date '+%F %T')  n=100+: analyze" | tee -a "$LOG"
PYTHONUNBUFFERED=1 uv run python scripts/analyze_alift_n100.py 2>&1 | tee -a "$LOG"
echo "$(date '+%F %T')  n=100+ analyze done" | tee -a "$LOG"

echo "$(date '+%F %T')  ==== orchestrator complete ====" | tee -a "$LOG"
