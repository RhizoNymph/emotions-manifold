#!/usr/bin/env bash
# Wait for the A-lift rerun to finish, then auto-run the n=40 analysis
# and save a clean summary the user can read in the morning.

set -euo pipefail
cd "$(dirname "$0")/.."

LOG=logs/alift_n40_analysis.log
mkdir -p logs

echo "$(date '+%F %T')  waiting for A-lift rerun to finish" | tee -a "$LOG"
deadline=$(( $(date +%s) + 14400 ))  # 4h ceiling
# Match the rerun chain script or its child run_pullback_experiment processes
while pgrep -f "alift_rerun_failed|run_pullback_experiment" >/dev/null 2>&1; do
    if [ "$(date +%s)" -ge "$deadline" ]; then
        echo "$(date '+%F %T')  timeout waiting for rerun" | tee -a "$LOG"
        exit 1
    fi
    sleep 30
done
echo "$(date '+%F %T')  rerun done; running n=40 analysis" | tee -a "$LOG"

uv run python -u scripts/analyze_alift_expansion.py 2>&1 | tee -a "$LOG"

echo "$(date '+%F %T')  n=40 analysis written to results/alift_expansion/_summary.json" | tee -a "$LOG"

# Compose a brief markdown summary so the user can scan a single file
uv run python << 'PYEOF' 2>&1 | tee -a logs/alift_n40_morning_summary.md
import json
from pathlib import Path

s = json.loads(Path("results/alift_expansion/_summary.json").read_text())
n = s["n_total"]
n_orig = s["n_original"]
n_exp = s["n_expansion"]
pr = s["a_lift_pearson_r"]
pp = s["a_lift_pearson_p"]
pl, ph = s["a_lift_pearson_ci"]
sr = s["a_lift_spearman_r"]
sp = s["a_lift_spearman_p"]
sl, sh = s["a_lift_spearman_ci"]
om = s["off_my_e_gap_mean"]
ol, oh = s["off_my_e_gap_ci"]
ow = s["off_my_e_wilcoxon_p"]
op = s["off_my_e_pullback_wins"]

print("# A-lift n=40 morning summary")
print()
print(f"**n = {n}** ({n_orig} original + {n_exp} expansion)")
print()
print("## A_lift correlation")
print()
print(f"- Pearson r = **{pr:+.3f}** (p={pp:.4f})  95% CI [{pl:+.3f}, {ph:+.3f}]")
print(f"- Spearman r = **{sr:+.3f}** (p={sp:.4f})  95% CI [{sl:+.3f}, {sh:+.3f}]")
print()
print(f"Compare to n=10 baseline: r=+0.473, CI [-0.089, +0.861], p=0.168")
print(f"Compare to n=26 partial: r=+0.402, CI [+0.081, +0.682], p=0.042")
print()
print("## off-M_y E gap (linear - pullback; positive = pullback more on-manifold)")
print()
print(f"- Mean: **{om:+.4f}**  95% CI [{ol:+.4f}, {oh:+.4f}]  Wilcoxon p = {ow:.4f}")
print(f"- Pullback favored in {op}/{n} pairs")
print()
print(f"Compare to n=10: +0.054, CI [+0.019, +0.093], p=0.024")
print()
print("## Next steps for you")
print()
print("1. Read this file plus `results/alift_expansion/_summary.json` for full details")
print("2. `results/alift_expansion/alift_vs_margin.png` shows the scatter plot")
print("3. day4.md has the n=26 numbers — needs updating to the n=40 values")
print("4. If results hold up, the A-lift predictor claim is now solidly defended")
PYEOF

echo "$(date '+%F %T')  morning summary at logs/alift_n40_morning_summary.md" | tee -a "$LOG"
