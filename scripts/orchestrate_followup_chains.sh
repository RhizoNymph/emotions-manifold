#!/usr/bin/env bash
# Master followup orchestrator: chain Silverman bandwidth experiment
# AFTER the d=6 chain completes.
#
# Triggered by: results/riemannian_analysis_6d/_summary.json appearing
# (which means orchestrate_6d_overnight.sh has finished Phase 3).
#
# Then runs:
#   Phase A: setup_8d_silverman_pipeline.py (~30 min)
#   Phase B: alift_8d_silverman_chain.sh (~7-8 hours)
#   Phase C: analyze (TODO: write analyze_8d_silverman.py)
#
# Logs to logs/followup_orchestration.log.

set -euo pipefail
cd "$(dirname "$0")/.."

LOG=logs/followup_orchestration.log
mkdir -p logs

echo "$(date '+%F %T')  followup orchestrator started" | tee -a "$LOG"

# Wait for d=6 to finish (the orchestrate_6d_overnight.sh writes
# _summary.json as its last step)
echo "$(date '+%F %T')  waiting for d=6 chain analysis to complete" | tee -a "$LOG"
while [[ ! -f results/riemannian_analysis_6d/_summary.json ]]; do
    sleep 300
done
echo "$(date '+%F %T')  d=6 done; starting Silverman setup" | tee -a "$LOG"

# Phase A: Silverman setup
echo "$(date '+%F %T')  Phase A: setup_8d_silverman_pipeline.py" | tee -a "$LOG"
PYTHONUNBUFFERED=1 uv run python scripts/setup_8d_silverman_pipeline.py 2>&1 | tee -a "$LOG"
echo "$(date '+%F %T')  Phase A done" | tee -a "$LOG"

# Phase B: chain
echo "$(date '+%F %T')  Phase B: alift_8d_silverman_chain.sh" | tee -a "$LOG"
bash scripts/alift_8d_silverman_chain.sh 2>&1 | tee -a "$LOG"
echo "$(date '+%F %T')  Phase B done" | tee -a "$LOG"

# Phase C: analyze
echo "$(date '+%F %T')  Phase C: analyze_8d_silverman.py" | tee -a "$LOG"
PYTHONUNBUFFERED=1 uv run python scripts/analyze_8d_silverman.py 2>&1 | tee -a "$LOG"
echo "$(date '+%F %T')  Phase C done" | tee -a "$LOG"

# Phase D: n=100+ extension (60 new pairs at production 8-D)
echo "$(date '+%F %T')  Phase D: sample_n100_extension_pairs.py" | tee -a "$LOG"
PYTHONUNBUFFERED=1 uv run python scripts/sample_n100_extension_pairs.py 2>&1 | tee -a "$LOG"
echo "$(date '+%F %T')  Phase D sampling done" | tee -a "$LOG"

echo "$(date '+%F %T')  Phase D: alift_n100_extension_chain.sh" | tee -a "$LOG"
bash scripts/alift_n100_extension_chain.sh 2>&1 | tee -a "$LOG"
echo "$(date '+%F %T')  Phase D chain done" | tee -a "$LOG"

# Phase E: analyze n=100+ extension
echo "$(date '+%F %T')  Phase E: analyze_alift_n100.py" | tee -a "$LOG"
PYTHONUNBUFFERED=1 uv run python scripts/analyze_alift_n100.py 2>&1 | tee -a "$LOG"
echo "$(date '+%F %T')  Phase E done" | tee -a "$LOG"

echo "$(date '+%F %T')  followup orchestrator complete" | tee -a "$LOG"
