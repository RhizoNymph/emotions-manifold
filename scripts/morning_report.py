"""Morning status report for the overnight runs.

Prints a compact summary of what completed during the overnight chain:
- Subspace sweep: which dims × pairs have results, with Δ values
- Pullback experiment: which pairs completed, with the three energies
- Generated figures: which files exist under results/figures/
- Tail of the overnight chain log (so any errors are visible)

Run with:
    uv run python scripts/morning_report.py
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

SWEEP_DIR = Path("results/subspace_sweep")
PULLBACK_DIR = Path("results/pullback")
FIG_DIR = Path("results/figures")
CHAIN_LOG = Path("logs/overnight_chain.log")
SWEEP_LOG = Path("logs/subspace_sweep.log")

SWEEP_FILE_RE = re.compile(r"^(?P<pair>[a-z_]+)_dim(?P<dim>\d{2})\.json$")


def summarize_sweep() -> None:
    print("=== Subspace sweep ===")
    grouped: dict[str, dict[int, dict]] = defaultdict(dict)
    for path in sorted(SWEEP_DIR.glob("*_dim*.json")):
        match = SWEEP_FILE_RE.match(path.name)
        if match is None:
            continue
        grouped[match.group("pair")][int(match.group("dim"))] = json.loads(path.read_text())

    if not grouped:
        print("  (no completed runs yet)")
        return

    for pair, per_dim in grouped.items():
        print(f"  {pair}:")
        for dim in sorted(per_dim.keys()):
            row = per_dim[dim]
            print(
                f"    dim {dim:>2d}  "
                f"manifold E={row['manifold_off_manifold_energy']:.3f}  "
                f"linear E={row['linear_off_manifold_energy']:.3f}  "
                f"Δ={row['delta_linear_minus_manifold']:+.3f}"
            )
    print()


def summarize_pullback() -> None:
    print("=== Pullback experiment ===")
    files = sorted(PULLBACK_DIR.glob("*.json"))
    if not files:
        print("  (no completed runs yet)")
        return
    for f in files:
        data = json.loads(f.read_text())
        start, end = data["pair"]
        g = data["geometry"]
        t = data["trajectories"]
        print(f"  {start}→{end}:")
        print(
            f"    geometry: pullback↔geodesic={g['mean_dist_pullback_to_geodesic']:.3f}  "
            f"pullback↔linear={g['mean_dist_pullback_to_linear']:.3f}  "
            f"(closer to {g['closer_to']})"
        )
        print(
            f"             G_E length  pullback={g['pullback_length']:.3f}  "
            f"geodesic={g['geodesic_length']:.3f}  "
            f"linear={g['linear_length']:.3f}"
        )
        print(
            f"    behavior:           "
            f"pullback   geodesic    linear"
        )
        print(
            f"      off-M_y E       "
            f"{t['pullback']['off_manifold_energy']:>8.3f}  "
            f"{t['geodesic']['off_manifold_energy']:>8.3f}  "
            f"{t['linear']['off_manifold_energy']:>8.3f}"
        )
        print(
            f"      M_y-line dist   "
            f"{t['pullback']['my_geodesic_distance']:>8.3f}  "
            f"{t['geodesic']['my_geodesic_distance']:>8.3f}  "
            f"{t['linear']['my_geodesic_distance']:>8.3f}"
        )
    print()


def summarize_figures() -> None:
    print("=== Figures ===")
    if not FIG_DIR.exists():
        print("  (none)")
        return
    for path in sorted(FIG_DIR.rglob("*.png")):
        size_kb = path.stat().st_size // 1024
        print(f"  {path}  ({size_kb} KB)")
    print()


def tail_chain_log() -> None:
    print("=== Overnight chain log (last 40 lines) ===")
    if not CHAIN_LOG.exists():
        print("  (no log)")
        return
    lines = CHAIN_LOG.read_text().splitlines()
    for line in lines[-40:]:
        print(f"  {line}")
    print()


def main() -> None:
    summarize_sweep()
    summarize_pullback()
    summarize_figures()
    tail_chain_log()


if __name__ == "__main__":
    main()
