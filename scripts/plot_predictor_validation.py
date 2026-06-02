"""Plot predicted G_E gap vs measured Δ across all pairs we've steered.

Combines measured Δ values from:
- ``results/pair_validation/*.json`` (the new untested-pair runs)
- ``results/subspace_sweep/<pair>_dim08.json`` (8-D sweep)
- ``results/steering_multipair.json`` (multipair K=10/N=3)

and looks each up against ``results/pair_alignment.json`` for the
predicted G_E gap. The single panel is a scatter of measured Δ vs G_E
gap with point labels and a vertical line at the smallest gap among
the new validation pairs — visually testing whether high-gap pairs
land above the cluster of measured-pair Δ values.

Run with:
    uv run python scripts/plot_predictor_validation.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_predictions() -> dict[frozenset[str], dict]:
    rows = json.loads(Path("results/pair_alignment.json").read_text())
    out: dict[frozenset[str], dict] = {}
    for r in rows:
        key = frozenset({r["start"], r["end"]})
        out[key] = r
    return out


def load_all_measured_deltas() -> dict[frozenset[str], dict]:
    out: dict[frozenset[str], dict] = {}

    multipair = Path("results/steering_multipair.json")
    if multipair.exists():
        for row in json.loads(multipair.read_text())["pairs"]:
            key = frozenset({row["start"], row["end"]})
            out[key] = {
                "start": row["start"],
                "end": row["end"],
                "delta": row["delta_linear_minus_manifold"],
                "source": "multipair-K10N3",
            }

    # 8-D sweep overrides multipair where they overlap (more samples).
    for path in Path("results/subspace_sweep").glob("*_dim08.json"):
        row = json.loads(path.read_text())
        s, e = row["pair"]
        key = frozenset({s, e})
        out[key] = {
            "start": s, "end": e,
            "delta": row["delta_linear_minus_manifold"],
            "source": "sweep-K30N10",
        }

    # Validation pairs (new high-gap predictions).
    for path in Path("results/pair_validation").glob("*.json"):
        row = json.loads(path.read_text())
        s, e = row["pair"]
        key = frozenset({s, e})
        out[key] = {
            "start": s, "end": e,
            "delta": row["delta_linear_minus_manifold"],
            "source": "validation-K30N10",
        }

    return out


def main() -> None:
    predictions = load_predictions()
    measurements = load_all_measured_deltas()

    rows: list[dict] = []
    for key, meas in measurements.items():
        pred = predictions.get(key)
        if pred is None or pred.get("ge_length_gap") is None:
            continue
        rows.append({**meas, "ge_gap": pred["ge_length_gap"]})

    if not rows:
        raise SystemExit("no pairs with both prediction and measurement")

    rows.sort(key=lambda r: r["ge_gap"])

    fig, ax = plt.subplots(figsize=(9, 6))

    color_for_source = {
        "multipair-K10N3":      "#888888",
        "sweep-K30N10":         "#0066cc",
        "validation-K30N10":    "#9933cc",
    }
    marker_for_source = {
        "multipair-K10N3":      "o",
        "sweep-K30N10":         "s",
        "validation-K30N10":    "D",
    }

    for r in rows:
        ax.scatter(
            r["ge_gap"], r["delta"],
            c=color_for_source.get(r["source"], "black"),
            marker=marker_for_source.get(r["source"], "o"),
            s=90, zorder=3, edgecolors="black", linewidths=0.5,
        )
        ax.annotate(
            f"{r['start']}→{r['end']}",
            (r["ge_gap"], r["delta"]),
            fontsize=8,
            xytext=(6, 6), textcoords="offset points",
        )

    ax.axhline(0, color="black", linewidth=0.7, alpha=0.5)
    ax.axhspan(-0.05, 0.05, color="gray", alpha=0.12, label="±0.05 judge-noise band")

    ax.set_xlabel("Predicted G_E length gap (linear − geodesic, M_h subspace)")
    ax.set_ylabel("Measured Δ = linear off-M_y E − manifold off-M_y E")
    ax.set_title("Measured Δ vs predicted G_E gap across all steered pairs")
    ax.grid(True, alpha=0.3)

    # Custom legend by source
    handles = [
        plt.Line2D([0], [0], marker=marker_for_source[s], color="w",
                   markerfacecolor=color_for_source[s], markeredgecolor="black",
                   markersize=10, label=s)
        for s in ("multipair-K10N3", "sweep-K30N10", "validation-K30N10")
    ]
    ax.legend(handles=handles, loc="best")

    out = Path("results/figures/predictor_validation.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
