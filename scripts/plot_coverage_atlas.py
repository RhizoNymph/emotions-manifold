"""Plot behavioral coverage atlas — which pairs we've tested vs the 14,535-pair
geometric space."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from manifold_emotions.behavior.manifold import BehaviorManifold
from manifold_emotions.config import load_config

cfg = load_config()
beh = BehaviorManifold.load(cfg.paths.manifold_y)
labels = list(beh.labels)
C = beh.centroids

# All tested pairs (10 original + 30 expansion + 5 + 15 compositions)
all_pairs_geometric = np.load("results/alift_all_pairs/all_pairs.npz")

# Currently-tested pullback pairs
PULLBACK_TESTED = [
    ("happy", "sad"), ("excited", "weary"), ("depressed", "energized"),
    ("terrified", "serene"), ("hope", "unhappy"), ("amused", "ashamed"),
    ("grumpy", "hopeful"), ("proud", "sympathetic"),
    ("brooding", "proud"), ("brooding", "pleased"),
    # Expansion (selected pairs spanning A_lift range)
    ("contemptuous", "hope"), ("contemptuous", "playful"), ("obstinate", "proud"),
    ("disgusted", "pleased"), ("indignant", "loving"),
    ("afraid", "loving"), ("loving", "restless"), ("ecstatic", "puzzled"),
    ("distressed", "self-confident"), ("hope", "sensitive"),
    ("proud", "puzzled"), ("self-conscious", "thankful"), ("smug", "unhappy"),
    ("jubilant", "mad"), ("energized", "terrified"),
    ("at ease", "obstinate"), ("kind", "vulnerable"), ("at ease", "envious"),
    ("at ease", "disdainful"), ("inspired", "irate"),
    ("anxious", "droopy"), ("indifferent", "stressed"), ("cheerful", "lazy"),
    ("depressed", "vengeful"), ("calm", "ecstatic"),
    ("happy", "jealous"), ("droopy", "uneasy"), ("content", "excited"),
    ("bored", "unnerved"), ("overwhelmed", "sluggish"),
]
COMPOSITION_TESTED = [
    ("happy", "sad"), ("excited", "calm"), ("angry", "afraid"),
    ("happy", "excited"), ("content", "miserable"),
    # Expansion
    ("joyful", "thrilled"), ("serene", "peaceful"),
    ("terrified", "panicked"), ("melancholy", "gloomy"), ("jubilant", "ecstatic"),
    ("joyful", "gloomy"), ("thrilled", "melancholy"), ("peaceful", "enraged"),
    ("satisfied", "panicked"), ("jubilant", "depressed"),
    ("thrilled", "serene"), ("ecstatic", "satisfied"),
    ("terrified", "melancholy"), ("panicked", "lonely"), ("joyful", "fulfilled"),
]
TONE_TESTED = [
    "joyful", "excited", "content", "calm",
    "melancholy", "gloomy", "anxious", "angry",
]

def midpoint(e1, e2):
    i = labels.index(e1)
    j = labels.index(e2)
    return (C[i] + C[j]) / 2

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

# Subplot 1: all 10,584 geometric pairs in light grey + 40 pullback-tested
ax = axes[0]
ax.scatter(all_pairs_geometric["midpoint_V"], all_pairs_geometric["midpoint_A"],
           s=2, alpha=0.15, color="grey")
pb_mids = np.array([midpoint(s, e) for s, e in PULLBACK_TESTED])
ax.scatter(pb_mids[:, 0], pb_mids[:, 1], s=80, c="firebrick",
           edgecolors="black", linewidths=0.5, zorder=10,
           label=f"pullback-tested chord midpoints (n={len(PULLBACK_TESTED)})")
ax.set_xlabel("Midpoint valence")
ax.set_ylabel("Midpoint arousal")
ax.set_title("Pullback experiment coverage")
ax.axhline(4.5, color="grey", linestyle=":", alpha=0.4)
ax.axvline(3.5, color="grey", linestyle=":", alpha=0.4)
ax.legend(loc="lower left")
ax.set_xlim(1.0, 6.5)
ax.set_ylim(2.0, 7.0)

# Subplot 2: composition coverage
ax = axes[1]
ax.scatter(all_pairs_geometric["midpoint_V"], all_pairs_geometric["midpoint_A"],
           s=2, alpha=0.15, color="grey")
co_mids = np.array([midpoint(s, e) for s, e in COMPOSITION_TESTED])
ax.scatter(co_mids[:, 0], co_mids[:, 1], s=80, c="darkorange",
           edgecolors="black", linewidths=0.5, zorder=10,
           label=f"composition-tested midpoints (n={len(COMPOSITION_TESTED)})")
ax.set_xlabel("Midpoint valence")
ax.set_title("Composition experiment coverage")
ax.axhline(4.5, color="grey", linestyle=":", alpha=0.4)
ax.axvline(3.5, color="grey", linestyle=":", alpha=0.4)
ax.legend(loc="lower left")

# Subplot 3: tone modulation - single emotion targets
ax = axes[2]
all_emotion_v = C[:, 0]
all_emotion_a = C[:, 1]
ax.scatter(all_emotion_v, all_emotion_a, s=10, alpha=0.3, color="grey",
           label=f"all 171 emotion centroids")
tone_v = [C[labels.index(em), 0] for em in TONE_TESTED]
tone_a = [C[labels.index(em), 1] for em in TONE_TESTED]
ax.scatter(tone_v, tone_a, s=120, c="steelblue", edgecolors="black",
           linewidths=0.5, zorder=10,
           label=f"tone-tested emotions (n={len(TONE_TESTED)})")
for em, v, a in zip(TONE_TESTED, tone_v, tone_a):
    ax.annotate(em, (v, a), fontsize=8, xytext=(5, 5),
                textcoords="offset points")
ax.set_xlabel("Valence")
ax.set_title("Tone modulation target coverage")
ax.axhline(4.5, color="grey", linestyle=":", alpha=0.4)
ax.axvline(3.5, color="grey", linestyle=":", alpha=0.4)
ax.legend(loc="lower left")

plt.suptitle(
    f"Behavioral coverage: pullback n={len(PULLBACK_TESTED)} of {len(all_pairs_geometric['midpoint_V'])} pairs "
    f"({100*len(PULLBACK_TESTED)/len(all_pairs_geometric['midpoint_V']):.2f}%), "
    f"composition n={len(COMPOSITION_TESTED)} of {len(all_pairs_geometric['midpoint_V'])} ({100*len(COMPOSITION_TESTED)/len(all_pairs_geometric['midpoint_V']):.2f}%), "
    f"tone n={len(TONE_TESTED)} of 171 ({100*len(TONE_TESTED)/171:.1f}%)",
    fontsize=11,
)
plt.tight_layout()

out_dir = Path("results/alift_all_pairs/figures")
out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(out_dir / "coverage_atlas.png", dpi=150, bbox_inches="tight")
print(f"saved {out_dir/'coverage_atlas.png'}")
