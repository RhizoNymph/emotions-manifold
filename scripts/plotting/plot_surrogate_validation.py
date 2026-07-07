"""Figure for writeup §12.5: surrogate-optimizer GPU validation (n=5 selected pairs).

Reads results/surrogate_optimizer/validation_results.json and plots, per pair,
the judged distance-to-target of the optimized trajectory vs the matched linear
baseline, with the surrogate's offline prediction overlaid so the near-zero
optimism is visible. Outputs to results/figures/writeup/ so existing figures
are preserved.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path("results/figures/writeup")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS = Path("results/surrogate_optimizer/validation_results.json")


def fig_surrogate_validation():
    data = json.loads(RESULTS.read_text())
    rows = data["per_pair"]
    labels = [r["pair"].replace("->", " → ") for r in rows]
    linear = np.array([r["linear_actual_dist"] for r in rows])
    optimized = np.array([r["optimized_actual_dist"] for r in rows])
    predicted = np.array([r["surrogate_predicted_dist"] for r in rows])

    y = np.arange(len(rows))
    h = 0.36

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(y + h / 2, linear, height=h, color="#b0b0b0", label="linear (judged)")
    ax.barh(y - h / 2, optimized, height=h, color="#2b7bba",
            label="optimized (judged)")
    ax.scatter(predicted, y - h / 2, marker="D", s=45, color="#e07b39",
               zorder=3, label="optimized (surrogate prediction)")

    for yi, (li, op) in enumerate(zip(linear, optimized)):
        ax.annotate(f"+{li - op:.2f}", xy=(li, yi + h / 2),
                    xytext=(4, -3), textcoords="offset points", fontsize=9,
                    color="#444444")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("distance to V/A target (judged, lower is better)")
    ax.set_title(
        "Behavior-first inverse steering: GPU validation on the 5 top-headroom pairs\n"
        f"mean real headroom +{data['mean_real_headroom']:.2f}, "
        f"surrogate optimism {data['mean_surrogate_optimism']:+.3f} "
        "(pairs selected by predicted headroom — best case, not a population effect)"
    , fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / "surrogate_validation_n5.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    fig_surrogate_validation()
