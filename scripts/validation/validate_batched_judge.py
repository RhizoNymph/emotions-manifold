"""Validation harness for the batched-judge pipeline.

Two layers:

Layer 1 — pure-Python metric equivalence (runs offline, no API needed).
    Synthesizes a small fake pair (random V/A ratings + completion records
    + paths npz), then computes ``off_manifold_energy`` and
    ``my_geodesic_distance`` two ways:
      (a) the sequential path's helpers (``_off_manifold_energy``,
          ``_distance_to_my_line``)
      (b) ``run_chain_batched_judge._rebuild_summary_for_pair``
    Asserts they match to machine precision (well, to 1e-6). This is the
    fast test you run any time you touch the metric code.

Layer 2 — end-to-end equivalence vs the real API (requires credits).
    Documented at the bottom of this file as a runbook; not executed here.
    The plan: pick 2 small pairs, run sequential once + batched once on
    the SAME generated text, then diff the per-pair summary JSONs. Because
    the generation is non-deterministic across calls, both runs must
    judge the SAME text — we accomplish that by feeding the same
    completions JSON into both judging paths.
"""

from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from manifold_emotions.behavior.judge_text import TextRating
from manifold_emotions.behavior.manifold import BehaviorManifold
from manifold_emotions.steering.experiment import (
    _aggregate_waypoint_behavior,
    _off_manifold_energy,
)
from manifold_emotions.steering.pullback_experiment import _distance_to_my_line
from manifold_emotions.steering.trajectory import SteeredContinuation


# ------------------ Layer 1: pure-Python metric equivalence ------------------


@dataclass(frozen=True)
class SyntheticPair:
    """One synthetic pair's worth of state, enough to exercise both paths."""
    pair_name: str            # e.g. "happy_sad" — matches on-disk file stem
    start: str
    end: str
    num_waypoints: int
    num_prompts: int
    my_path: np.ndarray       # (K, 2)
    # Per-completion ground truth
    completions: list[dict]   # [{text_id, method, waypoint, prompt, text}]
    ratings: list[dict]       # [{text_id, valence, arousal}]
    behavior_centroids: np.ndarray  # (N, 2) — synthetic M_y centroids


def _make_synthetic_pair(
    *,
    pair_name: str,
    start: str,
    end: str,
    num_waypoints: int = 8,
    num_prompts: int = 4,
    n_centroids: int = 30,
    drop_rating_fraction: float = 0.0,
    seed: int = 0,
) -> SyntheticPair:
    """Build a synthetic completions+ratings+my_path triplet.

    ``drop_rating_fraction`` simulates judge failures: that fraction of
    completions get no rating (mirrors what happened with the Silverman
    credit outage — some text_ids missing from the cache).
    """
    rng = np.random.default_rng(seed)
    # M_y centroids spread across [1, 7]^2 (Russell scale)
    behavior_centroids = rng.uniform(1.5, 6.5, size=(n_centroids, 2)).astype(np.float32)
    # M_y path: random walk between two centroids
    my_path = np.linspace(
        behavior_centroids[0], behavior_centroids[-1], num_waypoints
    ).astype(np.float32)

    completions: list[dict] = []
    ratings: list[dict] = []
    for method in ("pullback", "geodesic", "linear"):
        # Each method gets a slight per-waypoint V/A offset so the three
        # show different summary metrics.
        method_bias = rng.uniform(-0.3, 0.3, size=2).astype(np.float32)
        for wp in range(num_waypoints):
            wp_center = my_path[wp] + method_bias
            for prompt in range(num_prompts):
                # text_id format MUST match the sequential pipeline exactly.
                # sequential: f"{method}_{start}_{end}_wp{N:03d}_p{N:02d}"
                # with literal spaces preserved in multi-word labels.
                tid = f"{method}_{start}_{end}_wp{wp:03d}_p{prompt:02d}"
                completions.append({
                    "text_id": tid,
                    "method": method,
                    "waypoint": wp,
                    "prompt": prompt,
                    "text": f"<synthetic text for {tid}>",
                })
                if rng.random() < drop_rating_fraction:
                    continue
                noise = rng.normal(0, 0.15, size=2)
                va = np.clip(wp_center + noise, 1.0, 7.0)
                ratings.append({
                    "text_id": tid,
                    "valence": float(va[0]),
                    "arousal": float(va[1]),
                })

    return SyntheticPair(
        pair_name=pair_name,
        start=start, end=end,
        num_waypoints=num_waypoints, num_prompts=num_prompts,
        my_path=my_path,
        completions=completions,
        ratings=ratings,
        behavior_centroids=behavior_centroids,
    )


def _sequential_metrics(s: SyntheticPair) -> dict:
    """Compute off_manifold_energy and my_geodesic_distance via the
    sequential helpers — the reference implementation."""
    text_id_to_rating: dict[str, tuple[float, float]] = {
        r["text_id"]: (r["valence"], r["arousal"]) for r in s.ratings
    }

    out: dict[str, dict[str, float]] = {}
    for method in ("pullback", "geodesic", "linear"):
        # Build a list[SteeredContinuation] for this method
        conts: list[SteeredContinuation] = []
        for c in s.completions:
            if c["method"] != method:
                continue
            conts.append(SteeredContinuation(
                waypoint_index=c["waypoint"],
                prompt_index=c["prompt"],
                text=c["text"],
                finish_reason=None,
            ))
        # _aggregate_waypoint_behavior strips the "<method>_<start>_<end>_"
        # prefix internally by looking up the cont's `_text_id`. So the
        # prefix we give to index_ratings doesn't matter here — we hand it
        # a pre-stripped {wpXXX_pYY: (V, A)} dict.
        short = {
            f"wp{c['waypoint']:03d}_p{c['prompt']:02d}":
            text_id_to_rating[c["text_id"]]
            for c in s.completions
            if c["method"] == method and c["text_id"] in text_id_to_rating
        }
        mean, _ = _aggregate_waypoint_behavior(conts, short, s.num_waypoints)

        off = _off_manifold_energy(mean, s.behavior_centroids)
        myl, _ = _distance_to_my_line(mean, s.my_path)
        out[method] = {"off_manifold_energy": off, "my_geodesic_distance": myl,
                       "wp_valence": mean[:, 0].tolist(),
                       "wp_arousal": mean[:, 1].tolist()}
    return out


def _batched_metrics(s: SyntheticPair) -> dict:
    """Compute the same metrics via the batched rebuild path.

    Writes the synthetic state to a temp dir and invokes
    ``_rebuild_summary_for_pair`` from the chain orchestrator, then reads
    the rewritten summary back.
    """
    sys.path.insert(0, str(Path(__file__).parent))
    from run_chain_batched_judge import _rebuild_summary_for_pair  # type: ignore

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        chain_data_dir = tmp_path / "data" / "synthetic_chain"
        chain_results_dir = tmp_path / "results" / "synthetic_chain"
        chain_data_dir.mkdir(parents=True)
        chain_results_dir.mkdir(parents=True)

        # ratings cache (per-pair JSON)
        ratings_path = chain_data_dir / f"ratings_{s.pair_name}.json"
        ratings_path.write_text(json.dumps(s.ratings, indent=2))

        # completions
        completions_path = chain_data_dir / f"completions_{s.pair_name}.json"
        completions_path.write_text(json.dumps(s.completions, indent=2))

        # paths npz — only my_path is read by _rebuild_summary_for_pair
        paths_path = chain_data_dir / f"paths_{s.pair_name}.npz"
        np.savez_compressed(
            paths_path,
            my_path=s.my_path,
            pullback_sub=np.zeros((s.num_waypoints, 1), dtype=np.float32),
            geodesic_sub=np.zeros((s.num_waypoints, 1), dtype=np.float32),
            linear_sub=np.zeros((s.num_waypoints, 1), dtype=np.float32),
            pullback_full=np.zeros((s.num_waypoints, 1), dtype=np.float32),
            geodesic_full=np.zeros((s.num_waypoints, 1), dtype=np.float32),
            linear_full=np.zeros((s.num_waypoints, 1), dtype=np.float32),
        )

        # Skeleton summary
        summary = {
            "pair": [s.start, s.end],
            "manifold_dim": 8,
            "num_waypoints": s.num_waypoints,
            "trajectories": {
                m: {
                    "off_manifold_energy": float("nan"),
                    "my_geodesic_distance": float("nan"),
                    "waypoint_valence": [float("nan")] * s.num_waypoints,
                    "waypoint_arousal": [float("nan")] * s.num_waypoints,
                }
                for m in ("pullback", "geodesic", "linear")
            },
        }
        summary_path = chain_results_dir / f"{s.pair_name}.json"
        summary_path.write_text(json.dumps(summary, indent=2))

        # Behavior manifold stub
        class StubBehavior:
            centroids = s.behavior_centroids
        behavior = StubBehavior()

        ok = _rebuild_summary_for_pair(
            s.pair_name, chain_results_dir, chain_data_dir, behavior,
        )
        if not ok:
            raise RuntimeError("_rebuild_summary_for_pair returned False")

        rewritten = json.loads(summary_path.read_text())
        return {
            m: {
                "off_manifold_energy": rewritten["trajectories"][m]["off_manifold_energy"],
                "my_geodesic_distance": rewritten["trajectories"][m]["my_geodesic_distance"],
                "wp_valence": rewritten["trajectories"][m]["waypoint_valence"],
                "wp_arousal": rewritten["trajectories"][m]["waypoint_arousal"],
            }
            for m in ("pullback", "geodesic", "linear")
        }


def _diff_metrics(seq: dict, bat: dict, tol: float = 1e-6) -> list[str]:
    """Return a list of human-readable diffs; empty list = identical."""
    out: list[str] = []
    for method in ("pullback", "geodesic", "linear"):
        for key in ("off_manifold_energy", "my_geodesic_distance"):
            s, b = seq[method][key], bat[method][key]
            both_nan = (s != s) and (b != b)
            if both_nan:
                continue
            if abs(s - b) > tol:
                out.append(f"{method}.{key}: seq={s:+.6f} batched={b:+.6f} Δ={b-s:+.2e}")
        # Per-waypoint vectors
        for key in ("wp_valence", "wp_arousal"):
            s_arr = np.array(seq[method][key], dtype=np.float64)
            b_arr = np.array(bat[method][key], dtype=np.float64)
            if s_arr.shape != b_arr.shape:
                out.append(f"{method}.{key}: shape mismatch {s_arr.shape} vs {b_arr.shape}")
                continue
            mask_both_nan = np.isnan(s_arr) & np.isnan(b_arr)
            mask_one_nan = np.isnan(s_arr) ^ np.isnan(b_arr)
            if mask_one_nan.any():
                idx = int(np.where(mask_one_nan)[0][0])
                out.append(f"{method}.{key}: NaN mismatch at index {idx}: "
                           f"seq={s_arr[idx]} batched={b_arr[idx]}")
                continue
            diff_mask = ~mask_both_nan & (np.abs(s_arr - b_arr) > tol)
            if diff_mask.any():
                idx = int(np.where(diff_mask)[0][0])
                out.append(f"{method}.{key}: numeric mismatch at index {idx}: "
                           f"seq={s_arr[idx]:+.6f} batched={b_arr[idx]:+.6f}")
    return out


# ------------------ Layer 2: end-to-end runbook ------------------

RUNBOOK = """\
END-TO-END EQUIVALENCE RUNBOOK (requires Anthropic API credits)

Layer 1 of this script covers the metric recomputation in isolation.
Once credits are restored, run the following to validate that the
batched judge agrees with the sequential judge on actual text:

  # 0. Pick a tiny set of pairs (2 covers the multi-word case)
  printf "happy sad\\nat_ease disdainful\\n" > /tmp/validate_pairs.txt

  # 1. Phase 1 — generate trajectories once (no judging)
  bash scripts/alift_batched_chain.sh validate_chain \\
      data/manifold_h.npz /tmp/validate_pairs.txt
  # ... wait for Phase 2 to complete, then ...

  # 2. Side-by-side sequential judge of the SAME completions, against a
  #    fresh chain dir, so we can diff the resulting summaries.
  cp -r data/validate_chain data/validate_chain_seq
  cp -r results/validate_chain results/validate_chain_seq
  rm data/validate_chain_seq/ratings_*.json     # force re-judge
  # then point a sequential re-judge at it; one liner:
  PYTHONPATH=src uv run python -c "
  import asyncio, json
  from pathlib import Path
  from manifold_emotions.behavior.judge_text import judge_texts
  from manifold_emotions.config import load_config
  cfg = load_config()
  for cf in Path('data/validate_chain_seq').glob('completions_*.json'):
      pair = cf.stem.removeprefix('completions_')
      records = json.loads(cf.read_text())
      passages = [(r['text_id'], r['text']) for r in records]
      cache = Path(f'data/validate_chain_seq/ratings_{pair}.json')
      asyncio.run(judge_texts(cfg, passages, cache_path=cache))
  "

  # 3. Diff the per-pair summary JSONs
  for f in results/validate_chain/*.json; do
    bn=$(basename "$f")
    diff "results/validate_chain/$bn" "results/validate_chain_seq/$bn" || echo "DIFFERS: $bn"
  done

EXPECTED: per-text_id V/A ratings will differ slightly (judge model has
some non-determinism even at temperature=0), so summary metrics will
differ at the ~0.005 level. Anything larger than ~0.02 is a real bug.
"""


# ------------------ Main ------------------


def run_layer1() -> int:
    """Run all Layer 1 cases; return process exit code (0 = pass)."""
    cases = [
        ("happy_sad", "happy", "sad", 0.0),
        ("at ease_disdainful", "at ease", "disdainful", 0.0),  # multi-word
        ("excited_weary", "excited", "weary", 0.2),            # 20% rating drops
        ("calm_serene", "calm", "serene", 0.5),                # 50% rating drops
    ]
    fail = 0
    for pair_name, start, end, drop_frac in cases:
        s = _make_synthetic_pair(
            pair_name=pair_name, start=start, end=end,
            drop_rating_fraction=drop_frac,
            seed=hash(pair_name) & 0xFFFFFFFF,
        )
        seq = _sequential_metrics(s)
        bat = _batched_metrics(s)
        diffs = _diff_metrics(seq, bat)
        status = "PASS" if not diffs else "FAIL"
        print(f"  [{status}] {pair_name!r}  drop_frac={drop_frac:.0%}")
        for d in diffs:
            print(f"          {d}")
        if diffs:
            fail += 1
    return 0 if fail == 0 else 1


def main() -> None:
    print("Layer 1: pure-Python metric equivalence (sequential vs batched)")
    print("-" * 70)
    code = run_layer1()
    print("-" * 70)
    if code != 0:
        print(f"\n{code} cases FAILED — fix the recomputation in run_chain_batched_judge.")
        sys.exit(code)
    print("\nAll Layer 1 cases passed.")
    print()
    print("Layer 2 (end-to-end vs the real API) is a runbook — see source for")
    print("steps to execute once API credits are restored.")
    print()
    print(RUNBOOK)


if __name__ == "__main__":
    main()
