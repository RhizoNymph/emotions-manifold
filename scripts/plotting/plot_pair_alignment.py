"""Plot structural pair-alignment metrics against measured Δ.

Reads `results/pair_alignment.json` and produces a multi-panel figure
showing each candidate predictor (participation ratio, top-PC fraction,
near-chord centroid count, G_E length gap) against the measured Δ for
the pairs we've already steered. Whichever has the cleanest sign
agreement with Δ is the working predictor.

Also produces a histogram of each metric across all 435 pairs so we can
see where the measured pairs fall in the overall distribution — useful
for spotting selection biases in the pairs we've sampled.

Run with:
    uv run python scripts/plot_pair_alignment.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATA = Path("results/pair_alignment.json")
OUT = Path("results/figures/pair_alignment.png")

METRICS = (
    ("ge_length_gap", "G_E length gap (linear − geodesic)", "G_E gap"),
    ("max_chord_deflection", "Max perpendicular deflection of geodesic", "maxDef"),
    ("predicted_off_my_energy", "Predicted off-M_y E (from geodesic detour)", "pred_E"),
    ("participation_ratio", "Participation ratio", "PR"),
)


def main() -> None:
    rows = json.loads(DATA.read_text())
    measured = [r for r in rows if r["known_delta"] is not None]
    measured.sort(key=lambda r: r["known_delta"], reverse=True)

    n_panels = len(METRICS)
    fig, axes = plt.subplots(2, n_panels, figsize=(4 * n_panels, 7))

    for col, (key, title, _short) in enumerate(METRICS):
        # Top row: scatter of metric vs Δ for measured pairs
        ax = axes[0][col]
        xs, ys, labels = [], [], []
        for r in measured:
            if r[key] is None:
                continue
            xs.append(r[key])
            ys.append(r["known_delta"])
            labels.append(f"{r['start']}→{r['end']}")
        if xs:
            colors = ["#0066cc" if d > 0 else "#cc3300" if d < -0.02 else "#888888"
                      for d in ys]
            ax.scatter(xs, ys, c=colors, s=80, zorder=3)
            for x, y, lab in zip(xs, ys, labels, strict=True):
                ax.annotate(
                    lab, (x, y), fontsize=7,
                    xytext=(5, 5), textcoords="offset points",
                )
            ax.axhline(0, color="black", linewidth=0.6, alpha=0.5)
        ax.set_xlabel(title)
        if col == 0:
            ax.set_ylabel("measured Δ (linear − manifold)")
        ax.set_title(f"measured pairs vs {title}")
        ax.grid(True, alpha=0.3)

        # Bottom row: histogram of metric across all 435 pairs, with the
        # measured pairs as vertical lines colored by manifold/linear win
        ax = axes[1][col]
        all_vals = [r[key] for r in rows if r[key] is not None]
        if all_vals:
            ax.hist(all_vals, bins=30, color="lightgray", edgecolor="gray", zorder=1)
            for r in measured:
                if r[key] is None:
                    continue
                color = (
                    "#0066cc" if r["known_delta"] > 0
                    else "#cc3300" if r["known_delta"] < -0.02
                    else "#888888"
                )
                ax.axvline(r[key], color=color, linewidth=2.0, alpha=0.8, zorder=2)
                ax.annotate(
                    f"{r['start'][:3]}→{r['end'][:3]}",
                    (r[key], ax.get_ylim()[1] * 0.9),
                    fontsize=7, ha="center", color=color,
                    rotation=90, va="top",
                )
        ax.set_xlabel(title)
        if col == 0:
            ax.set_ylabel("# pairs")
        ax.set_title(f"distribution over all {len(rows)} pairs")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Pair-alignment metrics vs measured Δ — top: scatter, bottom: corpus-wide distribution\n"
        "blue = manifold wins (Δ > 0), red = linear wins (Δ < −0.02), gray = tie",
        fontsize=12,
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
