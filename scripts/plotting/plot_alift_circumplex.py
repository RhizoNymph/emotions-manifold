"""Plot A_lift across the V/A circumplex.

Generates two complementary visualizations:
1. Scatter: each pair plotted at its midpoint V/A, colored by A_lift sign+magnitude
2. Hexbin: density of pairs in each (mid_V, mid_A) cell with mean A_lift overlay
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl
mpl.use("Agg")

arr = np.load("results/alift_all_pairs/all_pairs.npz")
a_lift = arr["a_lift"]
mid_V = arr["midpoint_V"]
mid_A = arr["midpoint_A"]
chord = arr["chord_len"]

out_dir = Path("results/alift_all_pairs/figures")
out_dir.mkdir(parents=True, exist_ok=True)

# Plot 1: scatter, all pairs colored by A_lift
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# (a) Scatter
ax = axes[0]
order = np.argsort(np.abs(a_lift))  # plot small |A_lift| first so extremes show
sc = ax.scatter(mid_V[order], mid_A[order], c=a_lift[order],
                cmap="RdBu_r", s=4, alpha=0.5, vmin=-0.2, vmax=0.2)
ax.set_xlabel("Midpoint valence")
ax.set_ylabel("Midpoint arousal")
ax.set_title(f"A_lift across 10,584 pairs (σ=0.077)")
ax.axhline(4.5, color="grey", linestyle=":", alpha=0.4)
ax.axvline(3.5, color="grey", linestyle=":", alpha=0.4)
ax.set_xlim(1.0, 6.5)
ax.set_ylim(2.0, 7.0)
plt.colorbar(sc, ax=ax, label="A_lift  (+: pullback enthusiasm; -: pullback damping)")

# (b) Hexbin showing mean A_lift per cell
ax = axes[1]
hb = ax.hexbin(mid_V, mid_A, C=a_lift, reduce_C_function=np.mean,
               gridsize=20, cmap="RdBu_r", vmin=-0.2, vmax=0.2, mincnt=5)
ax.set_xlabel("Midpoint valence")
ax.set_ylabel("Midpoint arousal")
ax.set_title("Mean A_lift per midpoint cell")
ax.axhline(4.5, color="grey", linestyle=":", alpha=0.4)
ax.axvline(3.5, color="grey", linestyle=":", alpha=0.4)
ax.set_xlim(1.0, 6.5)
ax.set_ylim(2.0, 7.0)
plt.colorbar(hb, ax=ax, label="Mean A_lift")

plt.tight_layout()
plt.savefig(out_dir / "alift_circumplex.png", dpi=150, bbox_inches="tight")
print(f"saved {out_dir/'alift_circumplex.png'}")

# Plot 2: A_lift distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
ax = axes[0]
ax.hist(a_lift, bins=80, color="steelblue", edgecolor="white")
ax.axvline(0, color="k", linewidth=0.5)
ax.set_xlabel("A_lift")
ax.set_ylabel("count")
ax.set_title(f"A_lift distribution (n={len(a_lift)}); mean={a_lift.mean():+.4f}, std={a_lift.std():.4f}")

ax = axes[1]
ax.scatter(chord, a_lift, s=3, alpha=0.3, color="steelblue")
ax.axhline(0, color="k", linewidth=0.5)
ax.set_xlabel("Chord length")
ax.set_ylabel("A_lift")
ax.set_title("A_lift vs chord length")
plt.tight_layout()
plt.savefig(out_dir / "alift_distribution.png", dpi=150, bbox_inches="tight")
print(f"saved {out_dir/'alift_distribution.png'}")
