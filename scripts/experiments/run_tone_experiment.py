"""Creative-tone modulation test.

For each target emotion, test whether single-vector steering produces
text whose judged V/A lands near that emotion's M_y centroid. Tests
the basic premise — does emotion steering produce emotionally-coherent
behavior at all — on creative writing prompts where there's no policy
interference.

Two steering constructions per target:
  - LINEAR: just the h_emotion centroid (single-emotion steer baseline)
  - PULLBACK: kernel barycenter at that y_emotion (smooths over nearby
    M_y centroids; tests whether 'convex combination of real centroids'
    matters at the single-target level)

Both norm-matched before scaling, so we test direction/shape not
magnitude. The interesting question is whether pullback ≠ linear at
the single-target level produces any behavioral difference.

Metrics:
  - V/A judge → distance to target emotion's M_y centroid
    (the cleanest 'does tone match target' measurement)
  - Coherence judge (coherent / mixed / absent)
  - V/A scatter (mean, std) per emotion-type

Also: an unsteered baseline establishes what the model produces
without any steering, so we can separate 'is the model on-target?'
from 'is steering moving it toward target?'.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
import structlog

from manifold_emotions.behavior.judge_text import judge_texts
from manifold_emotions.behavior.manifold import BehaviorManifold
from manifold_emotions.config import load_config
from manifold_emotions.manifold.fit import FittedManifold
from manifold_emotions.manifold.pullback import _kernel_weights, _median_nn_distance
from manifold_emotions.steering.trajectory import generate_along_path

# Reuse the coherence judge from the composition script
import sys
sys.path.insert(0, str(Path(__file__).parent))
from run_composition_experiment import judge_coherence  # noqa: E402

log = structlog.get_logger(__name__)

CREATIVE_PROMPTS: tuple[str, ...] = (
    "Write a paragraph about a Tuesday morning.",
    "Describe a stranger you see at a coffee shop.",
    "Write about a walk through the city at dusk.",
    "Describe finding an old letter in a drawer.",
    "Write a paragraph about waiting at a bus stop.",
    "Tell a brief story about cooking dinner alone.",
    "Describe a phone call you didn't expect.",
    "Write about looking out the window during a long rain.",
    "Describe a small room you've spent time in.",
    "Write about an unremarkable Saturday afternoon.",
    "Describe the sound of someone walking down a hallway.",
    "Write about a book you put down without finishing.",
)

# Target emotions spanning M_y. Order: roughly clockwise from
# high-V-high-A (joyful) to high-V-low-A (calm).
TARGET_EMOTIONS: tuple[str, ...] = (
    "joyful",
    "excited",
    "content",
    "calm",
    "melancholy",
    "gloomy",
    "anxious",
    "angry",
)

STEERING_SCALE = 8.0
OUT_DATA_DIR = Path("data/tone")
OUT_RESULTS_DIR = Path("results/tone")


def _build_tone_steers(
    manifold: FittedManifold,
    behavior: BehaviorManifold,
    emotion: str,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (linear_full, pullback_full) — both at matched norm.

    linear_full = h_emotion centroid (unprojected from subspace).
    pullback_full = kernel barycenter at y_emotion, unprojected.
    Both are scaled to the linear vector's natural norm before being
    returned, so the caller can apply a single scale.
    """
    # Linear: just the emotion's h centroid
    h_sub = manifold.centroids_subspace[manifold.labels.index(emotion)]
    linear_full = manifold.unproject(h_sub[None, :])[0]

    # Pullback: kernel barycenter at y_emotion
    y = behavior.centroids[behavior.labels.index(emotion)]
    h_centroids = np.stack(
        [manifold.centroids_subspace[manifold.labels.index(l)] for l in behavior.labels],
        axis=0,
    ).astype(np.float32)
    weights = _kernel_weights(y, behavior.centroids, sigma)
    pullback_sub = (weights[:, None] * h_centroids).sum(axis=0)
    pullback_full = manifold.unproject(pullback_sub[None, :])[0]

    # Norm-match pullback to linear's natural norm
    linear_norm = float(np.linalg.norm(linear_full))
    pullback_norm = float(np.linalg.norm(pullback_full))
    pullback_full = pullback_full * (linear_norm / pullback_norm)

    return linear_full, pullback_full


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=10, help="num prompts per (emotion, type)")
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--sigma", type=float, default=None,
                        help="kernel bandwidth (default = median M_y NN distance)")
    parser.add_argument("--results-dir", default="results/tone")
    parser.add_argument("--data-dir", default="data/tone")
    args = parser.parse_args()

    config = load_config()
    manifold = FittedManifold.load(config.paths.manifold_h)
    behavior = BehaviorManifold.load(config.paths.manifold_y)

    sigma = args.sigma if args.sigma is not None else _median_nn_distance(behavior.centroids)
    prompts_used = list(CREATIVE_PROMPTS[:args.n])

    out_data_dir = Path(args.data_dir)
    out_results_dir = Path(args.results_dir)
    out_data_dir.mkdir(parents=True, exist_ok=True)
    out_results_dir.mkdir(parents=True, exist_ok=True)

    log.info("tone.start", sigma=sigma, n=args.n,
             emotions=list(TARGET_EMOTIONS))

    summary: dict = {"sigma": sigma, "scale": STEERING_SCALE,
                     "n_prompts": len(prompts_used), "results": {}}

    # ---- Unsteered baseline ----
    print()
    print("=== unsteered baseline ===")
    zero_steer = np.zeros((1, manifold.hidden_size), dtype=np.float32)
    baseline_cont = await generate_along_path(
        config, zero_steer, prompts_used,
        max_tokens=args.max_tokens, concurrency=16,
    )
    baseline_passages = [
        (f"baseline_p{c.prompt_index:02d}", c.text) for c in baseline_cont
    ]
    baseline_va = await judge_texts(config, baseline_passages,
                                    cache_path=out_data_dir / "va_baseline.json")
    baseline_co = await judge_coherence(config, baseline_passages,
                                        cache_path=out_data_dir / "coherence_baseline.json")
    base_va_pts = np.array(
        [(r.valence, r.arousal) for r in baseline_va.values()],
        dtype=np.float64,
    )
    base_co_labels = [r.label for r in baseline_co.values()]
    print(f"  baseline V mean={base_va_pts[:,0].mean():.2f}  A mean={base_va_pts[:,1].mean():.2f}")
    summary["baseline"] = {
        "mean_V": float(base_va_pts[:, 0].mean()),
        "mean_A": float(base_va_pts[:, 1].mean()),
        "std_V": float(base_va_pts[:, 0].std()),
        "std_A": float(base_va_pts[:, 1].std()),
        "coherence": {k: base_co_labels.count(k) / len(base_co_labels)
                      for k in ("coherent", "mixed", "absent")},
    }

    # ---- Per-emotion steering ----
    for emotion in TARGET_EMOTIONS:
        print()
        print(f"=== {emotion} ===")
        if emotion not in manifold.labels or emotion not in behavior.labels:
            print(f"  skipping — missing in M_h or M_y")
            continue

        y_target = behavior.centroids[behavior.labels.index(emotion)]
        linear_full, pullback_full = _build_tone_steers(
            manifold, behavior, emotion, sigma=sigma
        )

        linear_steer = (linear_full[None, :] * STEERING_SCALE).astype(np.float32)
        pullback_steer = (pullback_full[None, :] * STEERING_SCALE).astype(np.float32)
        lin_cont = await generate_along_path(
            config, linear_steer, prompts_used,
            max_tokens=args.max_tokens, concurrency=16,
        )
        pb_cont = await generate_along_path(
            config, pullback_steer, prompts_used,
            max_tokens=args.max_tokens, concurrency=16,
        )

        passages = []
        for c in lin_cont:
            passages.append((f"linear_{emotion}_p{c.prompt_index:02d}", c.text))
        for c in pb_cont:
            passages.append((f"pullback_{emotion}_p{c.prompt_index:02d}", c.text))

        va_ratings = await judge_texts(config, passages,
                                       cache_path=out_data_dir / f"va_{emotion}.json")
        co_ratings = await judge_coherence(config, passages,
                                           cache_path=out_data_dir / f"coherence_{emotion}.json")

        def aggregate(prefix: str):
            va_pts = []
            co_labels = []
            for tid, r in va_ratings.items():
                if tid.startswith(prefix):
                    va_pts.append((r.valence, r.arousal))
            for tid, r in co_ratings.items():
                if tid.startswith(prefix):
                    co_labels.append(r.label)
            return (np.array(va_pts, dtype=np.float64) if va_pts else np.empty((0, 2)),
                    co_labels)

        lin_va, lin_co = aggregate(f"linear_{emotion}_")
        pb_va, pb_co = aggregate(f"pullback_{emotion}_")

        def report(va, co, tag):
            if va.size == 0:
                return {}
            dist_target = float(np.linalg.norm(va - y_target[None, :], axis=1).mean())
            n = len(co) or 1
            return {
                "mean_V": float(va[:, 0].mean()),
                "mean_A": float(va[:, 1].mean()),
                "std_V": float(va[:, 0].std()),
                "std_A": float(va[:, 1].std()),
                "dist_to_target": dist_target,
                "coherence": {k: co.count(k) / n for k in ("coherent", "mixed", "absent")},
                "n_judged": int(va.shape[0]),
            }

        lin_rep = report(lin_va, lin_co, "lin")
        pb_rep = report(pb_va, pb_co, "pb")

        print(f"  target y=({y_target[0]:.2f}, {y_target[1]:.2f})")
        print(f"  {'metric':>16s}  {'linear':>9s}  {'pullback':>9s}")
        print(f"  {'mean V':>16s}  {lin_rep['mean_V']:>9.2f}  {pb_rep['mean_V']:>9.2f}")
        print(f"  {'mean A':>16s}  {lin_rep['mean_A']:>9.2f}  {pb_rep['mean_A']:>9.2f}")
        print(f"  {'dist to target':>16s}  {lin_rep['dist_to_target']:>9.3f}  {pb_rep['dist_to_target']:>9.3f}")
        print(f"  {'coherent%':>16s}  {lin_rep['coherence']['coherent']*100:>8.0f}%  {pb_rep['coherence']['coherent']*100:>8.0f}%")
        print(f"  {'absent%':>16s}  {lin_rep['coherence']['absent']*100:>8.0f}%  {pb_rep['coherence']['absent']*100:>8.0f}%")

        summary["results"][emotion] = {
            "y_target": [float(y_target[0]), float(y_target[1])],
            "linear": lin_rep,
            "pullback": pb_rep,
        }

        pair_out = out_results_dir / f"{emotion}.json"
        pair_out.write_text(json.dumps(summary["results"][emotion], indent=2))

    master = out_results_dir / "_summary.json"
    master.write_text(json.dumps(summary, indent=2))
    print()
    print(f"saved {master}")


if __name__ == "__main__":
    asyncio.run(main())
