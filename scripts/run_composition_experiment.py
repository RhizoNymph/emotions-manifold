"""Compositional pathology test.

For each (e1, e2) composition, test:
  - LINEAR: steering vector = (h_e1 + h_e2) * scale
  - PULLBACK: kernel barycenter at M_y midpoint y_mid = (y_e1 + y_e2)/2,
              unprojected to full residual stream, then * scale

Generate N continuations from each, judge for (V, A), judge for
coherence ('coherent' / 'mixed' / 'absent'), compute off-M_y E.

Tests Goodfire/Anthropic's prediction: linear composition pushes the
residual stream off-manifold and produces incoherent/degraded behavior;
the pullback-at-midpoint stays on-manifold because the kernel always
returns a convex combination of real emotion centroids.

Note on magnitude: linear h_e1 + h_e2 has roughly 2x the norm of a
single-emotion steer; pullback at midpoint is a single kernel
barycenter (unit-magnitude in the barycenter sense). We follow the
source-paper interpretation: apply both at the standard
``steering_scale=8.0`` and accept that magnitude differs. A
follow-up at norm-matched magnitudes can disambiguate magnitude vs.
shape effects.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
import structlog

from manifold_emotions.behavior.judge_text import judge_texts, TextRating
from manifold_emotions.behavior.manifold import BehaviorManifold
from manifold_emotions.config import load_config, Config
from manifold_emotions.errors import JudgeError
from manifold_emotions.manifold.fit import FittedManifold
from manifold_emotions.manifold.pullback import _kernel_weights, _median_nn_distance
from manifold_emotions.steering.experiment import _off_manifold_energy
from manifold_emotions.steering.trajectory import generate_along_path

log = structlog.get_logger(__name__)

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
    "What does Tuesday morning feel like?",
    "Write a paragraph about a stranger you saw.",
    "Talk about something small that happened recently.",
    "Describe an empty room.",
    "Tell me about your favorite season.",
    "Recount a brief conversation.",
    "Describe a journey somewhere.",
    "Share an observation about people.",
    "Write about a sound you noticed.",
    "Describe a quiet moment.",
)

COMPOSITIONS: tuple[tuple[str, str, str], ...] = (
    # (e1, e2, predicted_class)
    ("happy", "sad", "PATHOLOGY"),         # Anthropic's reported case
    ("excited", "calm", "PATHOLOGY"),      # opposite arousal
    ("angry", "afraid", "PATHOLOGY"),      # both neg-V high-A, opposite valence component
    ("happy", "excited", "COHERENT"),      # both pos-V pos-A: should be naturally coherent
    ("content", "miserable", "PATHOLOGY"), # milder happy/sad analog
)

STEERING_SCALE = 8.0  # match pullback_experiment.py default


# Coherence judge
_COHERENCE_JUDGE_PROMPT = (
    "Read this short passage and decide which one best describes its emotional content:\n\n"
    "A) COHERENT: expresses a single, clear emotional state\n"
    "B) MIXED: expresses multiple emotions, or conflicting/contradictory emotions in close succession\n"
    "C) ABSENT: no clear emotional content; the passage is flat, descriptive, or affectively neutral\n\n"
    "Respond with exactly one letter: A, B, or C. No other text.\n\n"
    "Passage:\n{text}"
)
_COHERENCE_RE = re.compile(r"^\s*([ABC])\s*$", re.IGNORECASE)
_COHERENCE_MAP = {"A": "coherent", "B": "mixed", "C": "absent"}


@dataclass(frozen=True, slots=True)
class CoherenceRating:
    text_id: str
    label: str  # "coherent" | "mixed" | "absent"


async def _judge_coherence_one(
    client: httpx.AsyncClient,
    config: Config,
    text_id: str,
    text: str,
    semaphore: asyncio.Semaphore,
) -> CoherenceRating:
    payload = {
        "model": config.judge.model,
        "max_tokens": 8,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": _COHERENCE_JUDGE_PROMPT.format(text=text)}],
    }
    headers = {
        "x-api-key": config.judge.api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    async with semaphore:
        try:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                json=payload, headers=headers, timeout=45.0,
            )
        except httpx.HTTPError as exc:
            raise JudgeError(f"coherence judge HTTP error for {text_id!r}: {exc}") from exc

    if response.status_code >= 400:
        raise JudgeError(
            f"coherence judge HTTP {response.status_code} for {text_id!r}: {response.text}"
        )

    body = response.json()
    content = body.get("content") or []
    if not content:
        # Anthropic API returned no content (often happens when input is
        # degenerate garbage that triggers a refusal). Treat as 'absent'
        # affect — closest thing to "this isn't a coherent emotional passage".
        return CoherenceRating(text_id=text_id, label="absent")
    reply = content[0].get("text", "").strip()
    m = _COHERENCE_RE.match(reply)
    if m is None:
        # Unparseable — could be a refusal, a hedge, or off-format. Treat as absent.
        return CoherenceRating(text_id=text_id, label="absent")
    return CoherenceRating(text_id=text_id, label=_COHERENCE_MAP[m.group(1).upper()])


async def judge_coherence(
    config: Config,
    passages: list[tuple[str, str]],
    cache_path: Path | None = None,
) -> dict[str, CoherenceRating]:
    """Score each passage as coherent/mixed/absent. Resumable like ``judge_texts``."""
    cache: dict[str, CoherenceRating] = {}
    if cache_path is not None and cache_path.exists():
        for row in json.loads(cache_path.read_text()):
            cache[row["text_id"]] = CoherenceRating(**row)
    missing = [(tid, t) for tid, t in passages if tid not in cache]
    if not missing:
        return {tid: cache[tid] for tid, _ in passages}

    semaphore = asyncio.Semaphore(config.judge.concurrency)
    errors: list[JudgeError] = []
    async with httpx.AsyncClient(http2=False) as client:
        async def run_one(text_id: str, text: str):
            try:
                return await _judge_coherence_one(client, config, text_id, text, semaphore)
            except JudgeError as exc:
                return exc

        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(run_one(tid, t)) for tid, t in missing]
        for task in tasks:
            r = task.result()
            if isinstance(r, JudgeError):
                errors.append(r)
            else:
                cache[r.text_id] = r

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(
            [{"text_id": r.text_id, "label": r.label} for r in cache.values()],
            indent=2,
        ))
    log.info("composition.coherence.done", rated=len(cache), errors=len(errors),
             first_errors=[str(e) for e in errors[:3]])
    return {tid: cache[tid] for tid, _ in passages if tid in cache}


def _build_steering_vectors(
    manifold: FittedManifold,
    behavior: BehaviorManifold,
    e1: str, e2: str,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (linear_full, pullback_full, y_mid) for the composition.

    linear_full = (h_e1 + h_e2) at full residual-stream resolution.
    pullback_full = kernel barycenter at M_y midpoint, unprojected.
    Both are (hidden_size,) — single steering vectors.
    """
    h1_sub = manifold.centroids_subspace[manifold.labels.index(e1)]
    h2_sub = manifold.centroids_subspace[manifold.labels.index(e2)]
    y1 = behavior.centroids[behavior.labels.index(e1)]
    y2 = behavior.centroids[behavior.labels.index(e2)]
    y_mid = ((y1 + y2) / 2.0).astype(np.float32)

    # Align M_h centroids to behavior labels
    h_centroids = np.stack(
        [manifold.centroids_subspace[manifold.labels.index(l)] for l in behavior.labels],
        axis=0,
    ).astype(np.float32)
    weights = _kernel_weights(y_mid, behavior.centroids, sigma)
    pullback_sub = (weights[:, None] * h_centroids).sum(axis=0)
    pullback_full = manifold.unproject(pullback_sub[None, :])[0]

    linear_sub = (h1_sub + h2_sub).astype(np.float32)
    linear_full = manifold.unproject(linear_sub[None, :])[0]

    return linear_full, pullback_full, y_mid


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=20, help="num prompts per (composition, type)")
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--sigma", type=float, default=None,
                        help="kernel bandwidth (default = median M_y NN distance)")
    parser.add_argument("--norm-match", action="store_true",
                        help="normalize linear and pullback to the same vector norm "
                             "before applying steering scale (isolates direction/shape "
                             "from magnitude)")
    parser.add_argument("--results-dir", default="results/composition",
                        help="where to write per-pair JSON + _summary.json")
    parser.add_argument("--data-dir", default="data/composition",
                        help="where to write V/A and coherence judge caches")
    parser.add_argument("--plan-file", default=None,
                        help="JSON file with a 'picks' key giving [[e1, e2], ...] "
                             "to use instead of the hardcoded COMPOSITIONS list. "
                             "predicted_class defaults to 'PATHOLOGY'.")
    args = parser.parse_args()

    config = load_config()
    manifold = FittedManifold.load(config.paths.manifold_h)
    behavior = BehaviorManifold.load(config.paths.manifold_y)

    sigma = args.sigma if args.sigma is not None else _median_nn_distance(behavior.centroids)

    # Use plan-file compositions if provided, otherwise the hardcoded list
    if args.plan_file is not None:
        plan = json.loads(Path(args.plan_file).read_text())
        compositions = tuple((p[0], p[1], "PATHOLOGY") for p in plan["picks"])
    else:
        compositions = COMPOSITIONS

    log.info("composition.start", sigma=sigma, n=args.n,
             norm_match=args.norm_match,
             compositions=[f"{a}+{b}" for a, b, _ in compositions])

    out_data_dir = Path(args.data_dir)
    out_results_dir = Path(args.results_dir)
    out_data_dir.mkdir(parents=True, exist_ok=True)
    out_results_dir.mkdir(parents=True, exist_ok=True)
    prompts_used = list(NEUTRAL_PROMPTS[:args.n])

    summary: dict[str, dict] = {}

    for e1, e2, predicted in compositions:
        pair_tag = f"{e1}_{e2}"
        print()
        print(f"=== composition: {e1} + {e2}  (predicted: {predicted}) ===")
        if e1 not in manifold.labels or e2 not in manifold.labels:
            print(f"  skipping — missing in M_h")
            continue
        if e1 not in behavior.labels or e2 not in behavior.labels:
            print(f"  skipping — missing in M_y")
            continue

        linear_full, pullback_full, y_mid = _build_steering_vectors(
            manifold, behavior, e1, e2, sigma=sigma
        )

        # Record raw norms for provenance
        linear_norm_raw = float(np.linalg.norm(linear_full))
        pullback_norm_raw = float(np.linalg.norm(pullback_full))

        if args.norm_match:
            # Normalize both to the pullback's natural norm. This attenuates
            # the linear sum (which is roughly sqrt(2) larger than a single
            # emotion vector) down to the kernel-barycenter magnitude, so both
            # vectors get the same total steering "energy" at scale=8.0.
            linear_full = linear_full * (pullback_norm_raw / linear_norm_raw)

        # Apply at standard scale; both as single-waypoint trajectories
        linear_steer = (linear_full[None, :] * STEERING_SCALE).astype(np.float32)
        pullback_steer = (pullback_full[None, :] * STEERING_SCALE).astype(np.float32)
        linear_norm_applied = float(np.linalg.norm(linear_steer))
        pullback_norm_applied = float(np.linalg.norm(pullback_steer))

        linear_cont = await generate_along_path(
            config, linear_steer, prompts_used,
            max_tokens=args.max_tokens, concurrency=16,
        )
        pullback_cont = await generate_along_path(
            config, pullback_steer, prompts_used,
            max_tokens=args.max_tokens, concurrency=16,
        )

        # Build passages with stable IDs
        passages: list[tuple[str, str]] = []
        for c in linear_cont:
            passages.append((f"linear_{pair_tag}_p{c.prompt_index:02d}", c.text))
        for c in pullback_cont:
            passages.append((f"pullback_{pair_tag}_p{c.prompt_index:02d}", c.text))

        # V/A judge
        va_cache = out_data_dir / f"va_{pair_tag}.json"
        va_ratings = await judge_texts(config, passages, cache_path=va_cache)
        # Coherence judge
        co_cache = out_data_dir / f"coherence_{pair_tag}.json"
        co_ratings = await judge_coherence(config, passages, cache_path=co_cache)

        def aggregate(prefix: str):
            va_pts = []
            co_labels = []
            for tid, r in va_ratings.items():
                if tid.startswith(prefix):
                    va_pts.append((r.valence, r.arousal))
            for tid, r in co_ratings.items():
                if tid.startswith(prefix):
                    co_labels.append(r.label)
            va_arr = np.array(va_pts, dtype=np.float64) if va_pts else np.empty((0, 2))
            return va_arr, co_labels

        linear_va, linear_co = aggregate(f"linear_{pair_tag}_")
        pullback_va, pullback_co = aggregate(f"pullback_{pair_tag}_")

        # off-M_y E per generation = distance from rated V/A to nearest M_y centroid
        def off_e(va_pts: np.ndarray) -> tuple[float, np.ndarray]:
            if va_pts.size == 0:
                return float("nan"), np.empty(0)
            dists = np.linalg.norm(behavior.centroids[None, :, :] - va_pts[:, None, :], axis=2)
            nn = dists.min(axis=1)
            return float(nn.mean()), nn

        def dist_from_mid(va_pts: np.ndarray) -> float:
            if va_pts.size == 0:
                return float("nan")
            return float(np.linalg.norm(va_pts - y_mid[None, :], axis=1).mean())

        def coherence_dist(labels: list[str]) -> dict[str, float]:
            if not labels:
                return {"coherent": float("nan"), "mixed": float("nan"), "absent": float("nan")}
            n = len(labels)
            return {k: labels.count(k) / n for k in ("coherent", "mixed", "absent")}

        linear_off, _ = off_e(linear_va)
        pullback_off, _ = off_e(pullback_va)
        linear_mid = dist_from_mid(linear_va)
        pullback_mid = dist_from_mid(pullback_va)
        linear_cd = coherence_dist(linear_co)
        pullback_cd = coherence_dist(pullback_co)

        # V/A scatter
        def scatter(va_pts: np.ndarray) -> dict:
            if va_pts.size == 0:
                return {"std_V": float("nan"), "std_A": float("nan"),
                        "mean_V": float("nan"), "mean_A": float("nan")}
            return {
                "std_V": float(va_pts[:, 0].std()),
                "std_A": float(va_pts[:, 1].std()),
                "mean_V": float(va_pts[:, 0].mean()),
                "mean_A": float(va_pts[:, 1].mean()),
            }
        linear_sc = scatter(linear_va)
        pullback_sc = scatter(pullback_va)

        # Print summary
        print(f"  y_mid = ({y_mid[0]:.2f}, {y_mid[1]:.2f})")
        print(f"  {'metric':>16s}  {'linear':>10s}  {'pullback':>10s}")
        print(f"  {'off-M_y E':>16s}  {linear_off:>10.3f}  {pullback_off:>10.3f}")
        print(f"  {'dist-from-mid':>16s}  {linear_mid:>10.3f}  {pullback_mid:>10.3f}")
        print(f"  {'coherent%':>16s}  {linear_cd['coherent']*100:>9.1f}%  {pullback_cd['coherent']*100:>9.1f}%")
        print(f"  {'mixed%':>16s}  {linear_cd['mixed']*100:>9.1f}%  {pullback_cd['mixed']*100:>9.1f}%")
        print(f"  {'absent%':>16s}  {linear_cd['absent']*100:>9.1f}%  {pullback_cd['absent']*100:>9.1f}%")
        print(f"  {'mean V':>16s}  {linear_sc['mean_V']:>10.2f}  {pullback_sc['mean_V']:>10.2f}")
        print(f"  {'mean A':>16s}  {linear_sc['mean_A']:>10.2f}  {pullback_sc['mean_A']:>10.2f}")
        print(f"  {'std V':>16s}  {linear_sc['std_V']:>10.2f}  {pullback_sc['std_V']:>10.2f}")
        print(f"  {'std A':>16s}  {linear_sc['std_A']:>10.2f}  {pullback_sc['std_A']:>10.2f}")

        summary[pair_tag] = {
            "composition": [e1, e2],
            "predicted_class": predicted,
            "y_mid": [float(y_mid[0]), float(y_mid[1])],
            "sigma": sigma,
            "steering_scale": STEERING_SCALE,
            "norm_match": args.norm_match,
            "linear_norm_raw": linear_norm_raw,
            "pullback_norm_raw": pullback_norm_raw,
            "linear_norm_applied": linear_norm_applied,
            "pullback_norm_applied": pullback_norm_applied,
            "n_prompts": len(prompts_used),
            "linear": {
                "off_M_y_E": linear_off,
                "dist_from_mid": linear_mid,
                "coherence_distribution": linear_cd,
                "va_scatter": linear_sc,
                "n_judged": int(len(linear_va)),
            },
            "pullback": {
                "off_M_y_E": pullback_off,
                "dist_from_mid": pullback_mid,
                "coherence_distribution": pullback_cd,
                "va_scatter": pullback_sc,
                "n_judged": int(len(pullback_va)),
            },
        }

        out = out_results_dir / f"{pair_tag}.json"
        out.write_text(json.dumps(summary[pair_tag], indent=2))
        print(f"  saved {out}")

    # Master summary
    master = out_results_dir / "_summary.json"
    master.write_text(json.dumps(summary, indent=2))
    print()
    print(f"saved master summary {master}")


if __name__ == "__main__":
    asyncio.run(main())
