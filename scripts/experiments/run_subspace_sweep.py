"""Phase 8 follow-up: full PCA subspace-dim sweep for the U-curve.

Refits the M_h manifold at each subspace dim in {2,4,6,8,10,12,14,16},
then runs the same scaled K=30, N=10 manifold-vs-linear comparison on
both excited↔weary (representative manifold-favored pair) and
terrified↔serene (representative linear-favored pair).

All artifacts live under ``data/subspace_sweep/`` and
``results/subspace_sweep/`` with consistent two-digit ``dimNN`` suffixes
so per-dim outputs sort lexically and don't collide with the canonical
``manifold_h.npz`` (still 8-D operating point).

Resumable: if both the manifold .npz and both result .jsons for a given
dim already exist, that dim is skipped. To force a re-run, delete the
relevant files (or pass ``--force``).

Run with:
    uv run python scripts/run_subspace_sweep.py
    uv run python scripts/run_subspace_sweep.py --dims 6 10 12
    uv run python scripts/run_subspace_sweep.py --force
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import numpy as np

from manifold_emotions.behavior.manifold import BehaviorManifold
from manifold_emotions.config import load_config
from manifold_emotions.manifold.fit import FittedManifold, fit_manifold
from manifold_emotions.steering.experiment import compare_steering
from manifold_emotions.vectors.diff_in_means import EmotionVectors

NEUTRAL_PROMPTS: tuple[str, ...] = (
    "Tell me about your day in a few sentences.",
    "What's on your mind right now?",
    "Describe what you see out the window.",
    "Tell me a short story.",
    "What did you do yesterday?",
    "Describe a simple meal.",
    "What's the weather like in your imagination?",
    "Share a memory from childhood.",
    "Talk about a hobby you enjoy.",
    "Describe a walk through a park.",
)

DEFAULT_DIMS = (2, 4, 6, 8, 10, 12, 14, 16)
PAIRS: tuple[tuple[str, str], ...] = (
    ("excited", "weary"),
    ("terrified", "serene"),
)
NUM_WAYPOINTS = 30
NUM_PROMPTS = 10

SWEEP_DATA_DIR = Path("data/subspace_sweep")
SWEEP_RESULTS_DIR = Path("results/subspace_sweep")


def manifold_path(dim: int) -> Path:
    return SWEEP_DATA_DIR / f"manifold_h_dim{dim:02d}.npz"


def ratings_path(start: str, end: str, dim: int) -> Path:
    return SWEEP_DATA_DIR / f"ratings_{start}_{end}_dim{dim:02d}.json"


def results_path(start: str, end: str, dim: int) -> Path:
    return SWEEP_RESULTS_DIR / f"{start}_{end}_dim{dim:02d}.json"


def ensure_manifold(
    emotion_vectors: EmotionVectors, dim: int, force: bool
) -> FittedManifold:
    out = manifold_path(dim)
    if out.exists() and not force:
        return FittedManifold.load(out)
    manifold, pca = fit_manifold(emotion_vectors, num_components=dim)
    manifold.save(out)
    cumulative = float(pca.explained_variance_ratio.sum())
    print(
        f"  fit M_h: {dim}-D, "
        f"explained={cumulative:.1%}, "
        f"bandwidth={manifold.kde_bandwidth:.4f}"
    )
    return manifold


async def run_pair(
    config,
    emotion_vectors: EmotionVectors,
    manifold: FittedManifold,
    behavior: BehaviorManifold,
    start: str,
    end: str,
    dim: int,
    force: bool,
) -> dict:
    result_file = results_path(start, end, dim)
    if result_file.exists() and not force:
        return json.loads(result_file.read_text())

    cache_file = ratings_path(start, end, dim)
    report = await compare_steering(
        config=config,
        emotion_vectors=emotion_vectors,
        manifold=manifold,
        behavior=behavior,
        start_label=start,
        end_label=end,
        num_waypoints=NUM_WAYPOINTS,
        num_prompts=NUM_PROMPTS,
        prompts=NEUTRAL_PROMPTS,
        max_tokens=96,
        concurrency=16,
        judge_cache_path=cache_file,
    )

    m_e = report.manifold.cumulative_off_manifold
    l_e = report.linear.cumulative_off_manifold
    delta = l_e - m_e

    my_start = behavior.centroids[behavior.labels.index(start)]
    my_end = behavior.centroids[behavior.labels.index(end)]
    chord = float(np.linalg.norm(my_end - my_start))

    summary = {
        "pair": [start, end],
        "subspace_dim": dim,
        "num_waypoints": NUM_WAYPOINTS,
        "num_prompts": NUM_PROMPTS,
        "my_chord_length": chord,
        "my_start_v": float(my_start[0]),
        "my_start_a": float(my_start[1]),
        "my_end_v": float(my_end[0]),
        "my_end_a": float(my_end[1]),
        "manifold_off_manifold_energy": m_e,
        "linear_off_manifold_energy": l_e,
        "delta_linear_minus_manifold": delta,
        "manifold_waypoint_valence": report.manifold.waypoint_behavior_mean[:, 0].tolist(),
        "manifold_waypoint_arousal": report.manifold.waypoint_behavior_mean[:, 1].tolist(),
        "linear_waypoint_valence": report.linear.waypoint_behavior_mean[:, 0].tolist(),
        "linear_waypoint_arousal": report.linear.waypoint_behavior_mean[:, 1].tolist(),
    }
    result_file.parent.mkdir(parents=True, exist_ok=True)
    result_file.write_text(json.dumps(summary, indent=2))
    return summary


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dims", nargs="*", type=int, default=list(DEFAULT_DIMS),
        help=f"subspace dims to sweep (default: {DEFAULT_DIMS})",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="re-run even if output files exist",
    )
    args = parser.parse_args()

    config = load_config()
    ev = EmotionVectors.load(config.paths.emotion_vectors)
    behavior = BehaviorManifold.load(config.paths.manifold_y)

    SWEEP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    SWEEP_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    overall_start = time.monotonic()
    summary_rows: list[dict] = []

    for dim in args.dims:
        print()
        print(f"=== subspace dim {dim} ===")
        dim_start = time.monotonic()

        manifold = ensure_manifold(ev, dim, args.force)

        for start, end in PAIRS:
            label_pair = f"{start}→{end}"
            pair_start = time.monotonic()
            print(f"  {label_pair} (dim={dim}) ...")
            summary = await run_pair(
                config=config,
                emotion_vectors=ev,
                manifold=manifold,
                behavior=behavior,
                start=start,
                end=end,
                dim=dim,
                force=args.force,
            )
            pair_elapsed = time.monotonic() - pair_start
            print(
                f"    manifold E={summary['manifold_off_manifold_energy']:.3f}  "
                f"linear E={summary['linear_off_manifold_energy']:.3f}  "
                f"Δ={summary['delta_linear_minus_manifold']:+.3f}  "
                f"({pair_elapsed:.0f}s)"
            )
            summary_rows.append(summary)

        dim_elapsed = time.monotonic() - dim_start
        print(f"  dim {dim} done in {dim_elapsed:.0f}s")

    overall_elapsed = time.monotonic() - overall_start
    print()
    print(f"=== sweep complete in {overall_elapsed/60:.1f} min ===")
    print()
    print(f"  {'dim':>3}  {'pair':>22s}  {'manifold E':>10s}  {'linear E':>10s}  {'Δ':>7s}")
    for row in summary_rows:
        print(
            f"  {row['subspace_dim']:>3d}  "
            f"{row['pair'][0]+'→'+row['pair'][1]:>22s}  "
            f"{row['manifold_off_manifold_energy']:>10.3f}  "
            f"{row['linear_off_manifold_energy']:>10.3f}  "
            f"{row['delta_linear_minus_manifold']:>+7.3f}"
        )


if __name__ == "__main__":
    asyncio.run(main())
