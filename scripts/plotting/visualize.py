"""Generate the day-1 result figures.

Reads all the .npz / .json artifacts we've produced (emotion vectors,
fitted manifold, behavior manifold, judge ratings, steering experiment
outputs) and writes a small set of PNGs to ``results/figures/``.

Run with:
    uv run python scripts/visualize.py

For just the isometry scatter on alternative datasets / subspace dims:
    uv run python scripts/visualize.py --variant 171em-8d
    uv run python scripts/visualize.py --variant 30em-4d --variant 171em-4d
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from manifold_emotions.behavior.manifold import BehaviorManifold
from manifold_emotions.config import load_config
from manifold_emotions.isometry import _pairwise_euclidean, _upper_triangular_vector
from manifold_emotions.manifold.fit import FittedManifold
from manifold_emotions.vectors.diff_in_means import EmotionVectors

FIGS = Path("results/figures")


@dataclass(frozen=True)
class IsometryVariant:
    """One (emotion-set, PCA-dim) configuration for the isometry scatter."""

    name: str
    emotion_vectors_path: Path
    manifold_h_path: Path
    manifold_y_path: Path
    dataset_label: str  # human-readable, e.g. "171 emotions"
    output_name: str  # filename under FIGS/


# Preset variants: emotion count × PCA subspace dimensionality.
# - 30em-8d  : original 30-emotion data + 8-D KDE/kernel-regression subspace
#              (matches the existing isometry_scatter.png file we preserve)
# - 30em-4d  : 30-emotion data + 4-D subspace used for manifold fitting
# - 171em-8d : full 171-emotion data + 8-D KDE/kernel-regression subspace
# - 171em-4d : full 171-emotion data + 4-D subspace used for manifold fitting
ISOMETRY_VARIANTS: dict[str, IsometryVariant] = {
    "30em-8d": IsometryVariant(
        name="30em-8d",
        emotion_vectors_path=Path("data/30emotions/emotion_vectors.npz"),
        manifold_h_path=Path("data/30emotions/manifold_h.npz"),
        manifold_y_path=Path("data/30emotions/manifold_y.npz"),
        dataset_label="30 emotions",
        output_name="isometry_scatter_30em_8d.png",
    ),
    "30em-4d": IsometryVariant(
        name="30em-4d",
        emotion_vectors_path=Path("data/30emotions/emotion_vectors.npz"),
        manifold_h_path=Path("data/manifold_h_4d.npz"),
        manifold_y_path=Path("data/30emotions/manifold_y.npz"),
        dataset_label="30 emotions",
        output_name="isometry_scatter_30em_4d.png",
    ),
    "171em-8d": IsometryVariant(
        name="171em-8d",
        emotion_vectors_path=Path("data/171emotions/emotion_vectors.npz"),
        manifold_h_path=Path("data/171emotions/manifold_h.npz"),
        manifold_y_path=Path("data/171emotions/manifold_y.npz"),
        dataset_label="171 emotions",
        output_name="isometry_scatter_171em_8d.png",
    ),
    "171em-4d": IsometryVariant(
        name="171em-4d",
        emotion_vectors_path=Path("data/171emotions/emotion_vectors.npz"),
        manifold_h_path=Path("data/manifold_h_4d_full.npz"),
        manifold_y_path=Path("data/171emotions/manifold_y.npz"),
        dataset_label="171 emotions",
        output_name="isometry_scatter_171em_4d.png",
    ),
}


def _cluster_color(valence: float, arousal: float) -> str:
    """Map (valence, arousal) to a circumplex-quadrant color.

    Mirrors the four affective quadrants Anthropic discusses: positive
    high-arousal (yellow), negative high-arousal (red), positive
    low-arousal (green), negative low-arousal (blue).
    """
    high_v = valence > 4.0
    high_a = arousal > 4.0
    if high_v and high_a:
        return "#f4a300"  # warm yellow — joy/excitement
    if high_v and not high_a:
        return "#2ca02c"  # green — calm/serene
    if not high_v and high_a:
        return "#d62728"  # red — anger/fear
    return "#1f77b4"  # blue — sadness/melancholy


def _ratings_dict(path: Path) -> dict[str, tuple[float, float]]:
    rows = json.loads(path.read_text())
    return {r["emotion"]: (r["valence"], r["arousal"]) for r in rows}


def plot_affective_circumplex(
    ev: EmotionVectors,
    mh: FittedManifold,
    ratings: dict[str, tuple[float, float]],
) -> None:
    """PC1 vs PC2 scatter, colored by judge-rated valence×arousal quadrant."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: M_h subspace projection (PC1 × PC2 of activation centroids).
    ax = axes[0]
    pc1 = mh.centroids_subspace[:, 0]
    pc2 = mh.centroids_subspace[:, 1]
    for i, label in enumerate(mh.labels):
        if label in ratings:
            v, a = ratings[label]
            color = _cluster_color(v, a)
        else:
            color = "#888888"
        ax.scatter(pc1[i], pc2[i], c=color, s=80, edgecolors="black", linewidth=0.5)
        ax.annotate(
            label,
            (pc1[i], pc2[i]),
            fontsize=8,
            xytext=(4, 4),
            textcoords="offset points",
            alpha=0.7,
        )
    ax.set_xlabel(
        f"PC1 ({mh.pca_explained_variance_ratio[0]:.0%} var) — arousal axis (anticorr.)"
    )
    ax.set_ylabel(
        f"PC2 ({mh.pca_explained_variance_ratio[1]:.0%} var) — valence axis"
    )
    ax.set_title("Activation manifold $M_h$ (PCA of layer-40 residuals)")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)

    # Right: judge-rated valence × arousal (the affective circumplex).
    ax = axes[1]
    for label, (v, a) in ratings.items():
        ax.scatter(v, a, c=_cluster_color(v, a), s=80, edgecolors="black", linewidth=0.5)
        ax.annotate(
            label,
            (v, a),
            fontsize=8,
            xytext=(4, 4),
            textcoords="offset points",
            alpha=0.7,
        )
    ax.set_xlabel("Valence (Claude judge, 1–7)")
    ax.set_ylabel("Arousal (Claude judge, 1–7)")
    ax.set_xlim(0.5, 7.5)
    ax.set_ylim(0.5, 7.5)
    ax.set_title("Affective circumplex (Claude-rated)")
    ax.grid(True, alpha=0.3)
    ax.axhline(4, color="gray", linewidth=0.5)
    ax.axvline(4, color="gray", linewidth=0.5)

    legend_elements = [
        Patch(facecolor="#f4a300", label="positive high-arousal"),
        Patch(facecolor="#2ca02c", label="positive low-arousal"),
        Patch(facecolor="#d62728", label="negative high-arousal"),
        Patch(facecolor="#1f77b4", label="negative low-arousal"),
    ]
    fig.legend(
        handles=legend_elements, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02)
    )
    fig.suptitle(
        "Affective circumplex recovered in activation space AND output behavior",
        fontsize=13,
    )
    fig.tight_layout()
    out = FIGS / "affective_circumplex.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def _plot_isometry_scatter(
    ev: EmotionVectors,
    mh: FittedManifold,
    my: BehaviorManifold,
    *,
    out_path: Path,
    dataset_label: str | None = None,
) -> None:
    """Pairwise-distance scatter: M_h subspace + full-D vs M_y.

    The subspace dimensionality is inferred from ``mh.centroids_subspace``
    and the activation-space dimensionality from ``ev.centroids``, so the
    same plotter works for the 8-D KDE/kernel-regression subspace and the
    4-D manifold-fitting subspace, and for any emotion count.
    """
    common = tuple(sorted(set(mh.labels) & set(my.labels) & set(ev.labels)))
    idx_mh = [mh.labels.index(label) for label in common]
    idx_my = [my.labels.index(label) for label in common]
    idx_ev = [ev.labels.index(label) for label in common]

    d_sub = _upper_triangular_vector(_pairwise_euclidean(mh.centroids_subspace[idx_mh]))
    d_full = _upper_triangular_vector(_pairwise_euclidean(ev.centroids[idx_ev]))
    d_my = _upper_triangular_vector(_pairwise_euclidean(my.centroids[idx_my]))

    r_sub = float(np.corrcoef(d_sub, d_my)[0, 1])
    r_full = float(np.corrcoef(d_full, d_my)[0, 1])

    subspace_dim = mh.centroids_subspace.shape[1]
    full_dim = ev.centroids.shape[1]
    dataset_text = dataset_label or f"{len(common)} emotions"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    ax = axes[0]
    ax.scatter(d_sub, d_my, s=14, alpha=0.5, c="#1f77b4")
    ax.set_xlabel(f"Pairwise distance in $M_h$ ({subspace_dim}-D PCA subspace)")
    ax.set_ylabel("Pairwise distance in $M_y$ (valence, arousal)")
    ax.set_title(f"Subspace isometry — Pearson r = {r_sub:.3f}")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.scatter(d_full, d_my, s=14, alpha=0.5, c="#d62728")
    ax.set_xlabel(f"Pairwise distance in full {full_dim}-D activation space")
    ax.set_ylabel("Pairwise distance in $M_y$ (valence, arousal)")
    ax.set_title(f"Linear baseline isometry — Pearson r = {r_full:.3f}")
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Isometry between activation and behavior — {dataset_text}, "
        f"{len(d_sub)} pairs",
        fontsize=12,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}  (r_sub={r_sub:+.3f}, r_full={r_full:+.3f})")


def plot_isometry(
    ev: EmotionVectors,
    mh: FittedManifold,
    my: BehaviorManifold,
) -> None:
    """Original default: writes ``results/figures/isometry_scatter.png``.

    Preserves the legacy filename and behavior used by the no-arg run.
    """
    _plot_isometry_scatter(ev, mh, my, out_path=FIGS / "isometry_scatter.png")


def plot_isometry_variant(variant: IsometryVariant) -> None:
    """Run the isometry scatter for one preset (emotion-set, subspace-dim) combo."""
    ev = EmotionVectors.load(variant.emotion_vectors_path)
    mh = FittedManifold.load(variant.manifold_h_path)
    my = BehaviorManifold.load(variant.manifold_y_path)
    _plot_isometry_scatter(
        ev,
        mh,
        my,
        out_path=FIGS / variant.output_name,
        dataset_label=variant.dataset_label,
    )


def plot_trajectories(
    my: BehaviorManifold,
    ratings: dict[str, tuple[float, float]],
    multipair_path: Path,
    scaled_paths: list[Path],
) -> None:
    """For each pair we have data for, plot manifold + linear behavior path on M_y."""
    pair_records: list[dict] = []
    if multipair_path.exists():
        for pair in json.loads(multipair_path.read_text())["pairs"]:
            pair["source"] = "smoke (K=10, N=3)"
            pair_records.append(pair)
    for scaled_path in scaled_paths:
        if not scaled_path.exists():
            continue
        d = json.loads(scaled_path.read_text())
        d["start"] = d["pair"][0]
        d["end"] = d["pair"][1]
        d["source"] = f"scaled (K={d['num_waypoints']}, N={d['num_prompts']})"
        pair_records.append(d)

    n_pairs = len(pair_records)
    if n_pairs == 0:
        print("  no trajectory data found, skipping trajectory plot")
        return

    cols = min(3, n_pairs)
    rows = (n_pairs + cols - 1) // cols
    fig, axes_arr = plt.subplots(rows, cols, figsize=(5.5 * cols, 5 * rows))
    axes = np.atleast_2d(axes_arr).flatten()

    for i, rec in enumerate(pair_records):
        ax = axes[i]
        # Background: all M_y centroids.
        for label, (v, a) in ratings.items():
            ax.scatter(
                v,
                a,
                c=_cluster_color(v, a),
                s=24,
                alpha=0.4,
                edgecolors="none",
            )

        start_label = rec["start"]
        end_label = rec["end"]

        # Trajectories.
        m_v = rec["manifold_waypoint_valence"]
        m_a = rec["manifold_waypoint_arousal"]
        l_v = rec["linear_waypoint_valence"]
        l_a = rec["linear_waypoint_arousal"]
        ax.plot(m_v, m_a, "-o", color="#0066cc", label="manifold", markersize=4)
        ax.plot(l_v, l_a, "-o", color="#cc3300", label="linear", markersize=4)

        # Endpoint emphasis.
        if start_label in ratings:
            v, a = ratings[start_label]
            ax.scatter(v, a, c="black", s=140, marker="*", zorder=10)
            ax.annotate(
                start_label,
                (v, a),
                fontsize=10,
                fontweight="bold",
                xytext=(8, 8),
                textcoords="offset points",
            )
        if end_label in ratings:
            v, a = ratings[end_label]
            ax.scatter(v, a, c="black", s=140, marker="*", zorder=10)
            ax.annotate(
                end_label,
                (v, a),
                fontsize=10,
                fontweight="bold",
                xytext=(8, 8),
                textcoords="offset points",
            )

        delta = rec["delta_linear_minus_manifold"]
        winner = "manifold" if delta > 0.01 else "linear" if delta < -0.01 else "tie"
        ax.set_xlabel("Valence")
        ax.set_ylabel("Arousal")
        ax.set_xlim(0.5, 7.5)
        ax.set_ylim(0.5, 7.5)
        ax.set_title(
            f"{start_label} → {end_label}\nΔ={delta:+.3f} ({winner})  {rec['source']}",
            fontsize=10,
        )
        ax.grid(True, alpha=0.3)
        ax.axhline(4, color="gray", linewidth=0.5)
        ax.axvline(4, color="gray", linewidth=0.5)
        ax.legend(fontsize=9, loc="upper right")

    # Blank out unused subplots.
    for j in range(n_pairs, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Manifold vs linear steering trajectories on the behavior manifold $M_y$",
        fontsize=14,
    )
    fig.tight_layout()
    out = FIGS / "trajectories.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_delta_summary(
    multipair_path: Path,
    scaled_paths: list[Path],
) -> None:
    """Bar chart of Δ (linear − manifold) across all pair × scale combinations."""
    records: list[tuple[str, str, float]] = []
    if multipair_path.exists():
        for pair in json.loads(multipair_path.read_text())["pairs"]:
            label = f"{pair['start']}→{pair['end']}"
            records.append((label, "smoke", pair["delta_linear_minus_manifold"]))
    for scaled_path in scaled_paths:
        if not scaled_path.exists():
            continue
        d = json.loads(scaled_path.read_text())
        label = f"{d['pair'][0]}→{d['pair'][1]}"
        records.append((label, "scaled", d["delta_linear_minus_manifold"]))

    if not records:
        print("  no delta data, skipping delta summary plot")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    pair_labels = sorted({r[0] for r in records}, key=lambda lbl: lbl)
    smoke_deltas = {r[0]: r[2] for r in records if r[1] == "smoke"}
    scaled_deltas = {r[0]: r[2] for r in records if r[1] == "scaled"}

    x = np.arange(len(pair_labels))
    width = 0.38
    smoke_vals = [smoke_deltas.get(lbl, np.nan) for lbl in pair_labels]
    scaled_vals = [scaled_deltas.get(lbl, np.nan) for lbl in pair_labels]

    bars_smoke = ax.bar(x - width / 2, smoke_vals, width, label="smoke (K=10, N=3)", color="#a6cee3")
    bars_scaled = ax.bar(x + width / 2, scaled_vals, width, label="scaled (K=30, N=10)", color="#1f78b4")

    for bars in (bars_smoke, bars_scaled):
        for rect in bars:
            h = rect.get_height()
            if np.isfinite(h):
                ax.annotate(
                    f"{h:+.3f}",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3 if h >= 0 else -12),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, rotation=20, ha="right")
    ax.set_ylabel("Δ = linear E − manifold E\n(positive = manifold wins)")
    ax.set_title("Off-manifold energy gap by pair and scale")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = FIGS / "delta_summary.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_pca_variance(mh: FittedManifold) -> None:
    """Bar + cumulative line of PCA explained variance."""
    ratios = mh.pca_explained_variance_ratio
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(1, len(ratios) + 1)
    ax.bar(x, ratios, color="#7570b3", label="per-PC variance")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio")
    ax2 = ax.twinx()
    ax2.plot(x, np.cumsum(ratios), "-o", color="#d95f02", label="cumulative")
    ax2.set_ylabel("Cumulative explained variance")
    ax2.set_ylim(0, 1.02)
    ax.set_xticks(x)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title(
        f"PCA spectrum of emotion vectors\n"
        f"({len(ratios)} components, {ratios.sum():.0%} cumulative variance)"
    )
    lines = [
        Patch(facecolor="#7570b3", label="per-PC"),
        plt.Line2D([], [], color="#d95f02", marker="o", label="cumulative"),
    ]
    ax.legend(handles=lines, loc="center right")
    fig.tight_layout()
    out = FIGS / "pca_spectrum.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        action="append",
        choices=sorted(ISOMETRY_VARIANTS),
        default=None,
        help=(
            "Generate the isometry scatter for the given preset "
            "(emotion-set, subspace-dim). Repeatable. When supplied, only the "
            "isometry plot(s) for the chosen variants are produced and the "
            "default full-figure run is skipped. The legacy isometry_scatter.png "
            "is left untouched."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    FIGS.mkdir(parents=True, exist_ok=True)

    if args.variant:
        print("generating isometry scatter variants...")
        for variant_name in args.variant:
            variant = ISOMETRY_VARIANTS[variant_name]
            print(f"  variant: {variant.name}  ({variant.dataset_label})")
            plot_isometry_variant(variant)
        print(f"\nvariants saved under {FIGS}/")
        return

    config = load_config()
    ev = EmotionVectors.load(config.paths.emotion_vectors)
    mh = FittedManifold.load(config.paths.manifold_h)
    my = BehaviorManifold.load(config.paths.manifold_y)
    ratings = _ratings_dict(Path("data/emotion_ratings.json"))

    print("generating figures...")
    plot_affective_circumplex(ev, mh, ratings)
    plot_pca_variance(mh)
    plot_isometry(ev, mh, my)
    plot_trajectories(
        my,
        ratings,
        Path("results/steering_multipair.json"),
        [
            Path("results/steering_scaled.json"),
            Path("results/steering_scaled_terrified_serene.json"),
        ],
    )
    plot_delta_summary(
        Path("results/steering_multipair.json"),
        [
            Path("results/steering_scaled.json"),
            Path("results/steering_scaled_terrified_serene.json"),
        ],
    )
    print(f"\nall figures saved under {FIGS}/")


if __name__ == "__main__":
    main()
