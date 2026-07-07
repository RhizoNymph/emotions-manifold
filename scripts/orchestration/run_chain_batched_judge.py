"""Phase-2 batched judging for a chain generated with ``judge: none``.

Phase 1 (``run_chain.py`` with a ``judge: none`` variant) writes raw
completions per pair and NaN-skeleton summaries. This script aggregates
every pair's completions into one Batches-API submission, distributes
the ratings back into per-pair caches, and rewrites each summary with
the final off-M_y E / M_y-line metrics.

Wall-time/cost vs sequential judging: judging for ALL pairs happens in
a single batch wait (typically 10–30 min) at 50% API cost.

    uv run python scripts/orchestration/run_chain_batched_judge.py --chain pullback_8d_silverman
    uv run python scripts/orchestration/run_chain_batched_judge.py --chain pullback_4d \
        --pairs experiments/pairs/alift_n40.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import structlog

from manifold_emotions.behavior.judge_text import TextRating
from manifold_emotions.behavior.judge_text_batched import judge_texts_batched
from manifold_emotions.behavior.manifold import BehaviorManifold
from manifold_emotions.config import load_config

log = structlog.get_logger(__name__)


def _load_pair_completions(completions_path: Path) -> list[dict]:
    """Load per-pair completion records written by Phase 1."""
    if not completions_path.exists():
        return []
    return json.loads(completions_path.read_text())


def _aggregate_passages(
    chain_data_dir: Path,
    pair_names: list[str],
) -> tuple[list[tuple[str, str]], dict[str, str]]:
    """Collect (text_id, text) tuples across all pairs.

    Returns:
        passages: flat list for batch submission
        text_id_to_pair: index for re-distributing results back to per-pair caches
    """
    passages: list[tuple[str, str]] = []
    text_id_to_pair: dict[str, str] = {}
    for pair_name in pair_names:
        completions_path = chain_data_dir / f"completions_{pair_name}.json"
        records = _load_pair_completions(completions_path)
        if not records:
            log.warning("phase2.missing_completions",
                        pair=pair_name, path=str(completions_path))
            continue
        for r in records:
            tid = r["text_id"]
            passages.append((tid, r["text"]))
            text_id_to_pair[tid] = pair_name
    return passages, text_id_to_pair


def _write_per_pair_caches(
    ratings: dict,
    text_id_to_pair: dict[str, str],
    chain_data_dir: Path,
) -> dict[str, int]:
    """Split the batch results back into per-pair ratings_*.json caches.

    Same JSON shape as judge_text.judge_texts writes.
    """
    by_pair: dict[str, list[dict]] = {}
    for text_id, rating in ratings.items():
        pair_name = text_id_to_pair.get(text_id)
        if pair_name is None:
            continue
        by_pair.setdefault(pair_name, []).append({
            "text_id": rating.text_id,
            "valence": rating.valence,
            "arousal": rating.arousal,
        })

    counts: dict[str, int] = {}
    for pair_name, rows in by_pair.items():
        path = chain_data_dir / f"ratings_{pair_name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows, indent=2))
        counts[pair_name] = len(rows)
    return counts


def _rebuild_summary_for_pair(
    pair_name: str,
    chain_results_dir: Path,
    chain_data_dir: Path,
    behavior: BehaviorManifold,
) -> bool:
    """Re-derive off-M_y E and M_y-line distance for a pair from its
    now-populated ratings cache, then rewrite the summary JSON in place.

    Returns True if the summary was rewritten, False if inputs were missing.
    """
    summary_path = chain_results_dir / f"{pair_name}.json"
    ratings_path = chain_data_dir / f"ratings_{pair_name}.json"
    completions_path = chain_data_dir / f"completions_{pair_name}.json"
    paths_path = chain_data_dir / f"paths_{pair_name}.npz"

    if not summary_path.exists():
        log.warning("phase2.no_summary_to_rewrite",
                    pair=pair_name, path=str(summary_path))
        return False
    if not ratings_path.exists() or not completions_path.exists() or not paths_path.exists():
        log.warning("phase2.missing_inputs",
                    pair=pair_name,
                    ratings=ratings_path.exists(),
                    completions=completions_path.exists(),
                    paths=paths_path.exists())
        return False

    summary = json.loads(summary_path.read_text())
    ratings_rows = json.loads(ratings_path.read_text())
    ratings_by_id: dict[str, tuple[float, float]] = {
        r["text_id"]: (r["valence"], r["arousal"]) for r in ratings_rows
    }
    completions = json.loads(completions_path.read_text())
    paths_npz = np.load(paths_path)
    my_path = paths_npz["my_path"]  # (K, 2)

    # Bucket ratings by (method, waypoint), averaging across prompts —
    # matches what the sequential pipeline computes per-trajectory.
    buckets: dict[tuple[str, int], list[tuple[float, float]]] = {}
    for r in completions:
        tid = r["text_id"]
        if tid not in ratings_by_id:
            continue
        key = (r["method"], r["waypoint"])
        buckets.setdefault(key, []).append(ratings_by_id[tid])

    K = my_path.shape[0]
    # Rebuild every trajectory present in the skeleton — pullback/geodesic/linear
    # plus any spline trajectories (spline_induced/spline_density) added upstream.
    for method in list(summary["trajectories"].keys()):
        wp_va = np.full((K, 2), np.nan, dtype=np.float32)
        for wp in range(K):
            samples = buckets.get((method, wp), [])
            if samples:
                arr = np.array(samples, dtype=np.float32)
                wp_va[wp] = arr.mean(axis=0)

        finite_mask = ~np.isnan(wp_va).any(axis=1)
        wp_finite = wp_va[finite_mask]

        # off-M_y E: mean distance from each waypoint's V/A to the nearest
        # M_y centroid — matches pullback_experiment's convention.
        centroids = np.asarray(behavior.centroids, dtype=np.float32)  # (N, 2)
        if len(wp_finite) == 0:
            off_my_e = float("nan")
            my_line = float("nan")
        else:
            diffs = wp_finite[:, None, :] - centroids[None, :, :]
            dists = np.sqrt((diffs * diffs).sum(axis=-1))
            off_my_e = float(dists.min(axis=1).mean())
            # M_y-line distance: per-waypoint Euclidean to the my_path point
            # at the same index.
            my_line = float(np.sqrt(
                ((wp_finite - my_path[finite_mask]) ** 2).sum(axis=1)
            ).mean())

        summary["trajectories"][method]["off_manifold_energy"] = off_my_e
        summary["trajectories"][method]["my_geodesic_distance"] = my_line
        summary["trajectories"][method]["waypoint_valence"] = wp_va[:, 0].tolist()
        summary["trajectories"][method]["waypoint_arousal"] = wp_va[:, 1].tolist()

    summary_path.write_text(json.dumps(summary, indent=2))
    return True


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--chain", required=True,
        help="Chain name: 'pullback' (production 8-D), 'pullback_8d_silverman', "
             "'pullback_6d', 'pullback_4d', etc. Determines data/results dirs.",
    )
    parser.add_argument(
        "--pairs", type=Path, default=None,
        help="JSON pair file ([[start, end], ...], same format as run_chain.py). "
             "If omitted, processes every completions_*.json in the chain dir.",
    )
    args = parser.parse_args()

    chain = args.chain
    chain_data_dir = Path(f"data/{chain}")
    chain_results_dir = Path(f"results/{chain}")
    chain_data_dir.mkdir(parents=True, exist_ok=True)
    chain_results_dir.mkdir(parents=True, exist_ok=True)

    if args.pairs:
        pair_names = [
            f"{a}_{b}" for a, b in json.loads(args.pairs.read_text())
        ]
    else:
        pair_names = sorted(
            p.stem.removeprefix("completions_")
            for p in chain_data_dir.glob("completions_*.json")
        )

    if not pair_names:
        print(f"No completions_*.json files in {chain_data_dir}. "
              f"Run Phase 1 (a 'judge: none' chain) first.", file=sys.stderr)
        sys.exit(1)

    print(f"Phase 2 (batched judge) for chain={chain}, {len(pair_names)} pairs")

    # Aggregate ALL passages across pairs into one batch
    passages, text_id_to_pair = _aggregate_passages(chain_data_dir, pair_names)
    print(f"Aggregated {len(passages)} (text_id, text) passages from "
          f"{len(set(text_id_to_pair.values()))} pairs")

    # One shared cache file is overkill — split per-pair so reruns of
    # individual pairs still work with the sequential judge. Submit one
    # batch covering everything not already cached.
    config = load_config()
    already_cached: set[str] = set()
    for pair_name in pair_names:
        ratings_path = chain_data_dir / f"ratings_{pair_name}.json"
        if ratings_path.exists():
            for row in json.loads(ratings_path.read_text()):
                already_cached.add(row["text_id"])
    pending = [(tid, text) for tid, text in passages if tid not in already_cached]
    print(f"  cached: {len(passages) - len(pending)},  to submit: {len(pending)}")

    # Use the batched judge with NO cache_path — we split by pair ourselves
    # so we keep one cache file per pair.
    if pending:
        ratings = await judge_texts_batched(config, pending, cache_path=None)
    else:
        ratings = {}
    # Augment with previously-cached so the per-pair writer has everything
    for pair_name in pair_names:
        ratings_path = chain_data_dir / f"ratings_{pair_name}.json"
        if ratings_path.exists():
            for row in json.loads(ratings_path.read_text()):
                ratings[row["text_id"]] = TextRating(**row)

    # Distribute and write per-pair caches
    pair_rating_counts = _write_per_pair_caches(ratings, text_id_to_pair, chain_data_dir)
    print(f"Wrote per-pair caches: {len(pair_rating_counts)} pairs")

    # Re-derive metrics and rewrite the per-pair summaries
    print("Rebuilding per-pair summary JSONs (replaces nan metrics)...")
    behavior = BehaviorManifold.load(config.paths.manifold_y)
    rewritten = 0
    for pair_name in pair_names:
        if _rebuild_summary_for_pair(pair_name, chain_results_dir, chain_data_dir, behavior):
            rewritten += 1
    print(f"Rewrote {rewritten} summary files in {chain_results_dir}/")


if __name__ == "__main__":
    asyncio.run(main())
