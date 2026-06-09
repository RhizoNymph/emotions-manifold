#!/usr/bin/env bash
# Wait for setup_6d_pipeline to finish, then launch the n=40 6-D
# A-lift behavioral chain, then run the analysis.
#
# Spawned in background by the assistant. Logs to logs/6d_orchestration.log.

set -euo pipefail
cd "$(dirname "$0")/.."

LOG=logs/6d_orchestration.log
mkdir -p logs

echo "$(date '+%F %T')  orchestrator started" | tee -a "$LOG"

# Phase 1: wait for setup_6d_pipeline to finish (writes data/manifold_h_6d_full.npz
# and data/geodesics_cache_6d.npz at the end)
echo "$(date '+%F %T')  Phase 1: waiting for data/geodesics_cache_6d.npz" | tee -a "$LOG"
while [[ ! -f data/geodesics_cache_6d.npz ]]; do
    sleep 60
done
echo "$(date '+%F %T')  Phase 1 done: 6-D geodesics ready" | tee -a "$LOG"

# Phase 2: launch the n=40 6-D behavioral chain
echo "$(date '+%F %T')  Phase 2: launching alift_6d_chain.sh" | tee -a "$LOG"
bash scripts/alift_6d_chain.sh 2>&1 | tee -a "$LOG"
echo "$(date '+%F %T')  Phase 2 done: 40 pairs run at 6-D" | tee -a "$LOG"

# Phase 3: analyze
echo "$(date '+%F %T')  Phase 3: running analyze_geodesic_vs_linear_6d.py" | tee -a "$LOG"
uv run python scripts/analyze_geodesic_vs_linear_6d.py 2>&1 | tee -a "$LOG"
echo "$(date '+%F %T')  Phase 3 done: see results/riemannian_analysis_6d/_summary.json" | tee -a "$LOG"

echo "$(date '+%F %T')  orchestrator complete" | tee -a "$LOG"
