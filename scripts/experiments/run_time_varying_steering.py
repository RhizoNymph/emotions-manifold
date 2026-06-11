"""Time-varying steering along the trajectory (Goodfire's temporal claim).

Tests Goodfire's central distinguishing claim — that stepping through
waypoints during generation produces different behavior than holding
a single vector constant.

Implementation: divide generation into K segments. Each segment uses
one waypoint vector. After each segment, send back the conversation
plus the assistant-partial and continue generation (manual Gemma chat
template via /v1/completions, since the chat endpoint re-renders and
strips trailing tokens).

The chord experiment baseline uses ONE waypoint per request applied to
all tokens. Here we default to K=8 segments × 12 tokens, advancing the
waypoint between segments.

For each pair, runs the segmented (time-varying) version of pullback,
geodesic, and linear, judges each completion, and reports off-M_y E and
M_y-line distance. Output: results/time_varying/{pair}.json.

Ported from the archive version with three fixes/upgrades:
- pairs come from a JSON pair file (--pairs), eliminating the bash
  word-splitting bug that dropped multi-word labels ('at ease') from
  the original n=13 chain;
- resume: pairs with existing results are skipped (--force to re-run);
- the manifold/behavior paths are configurable instead of hardcoded.

    uv run python scripts/experiments/run_time_varying_steering.py \
        --pairs experiments/pairs/time_varying_n13.json
    uv run python scripts/experiments/run_time_varying_steering.py happy sad
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import time
from pathlib import Path

import httpx
import numpy as np

from manifold_emotions.behavior.judge_text import judge_texts
from manifold_emotions.behavior.manifold import BehaviorManifold
from manifold_emotions.config import load_config
from manifold_emotions.experiments.chord import NEUTRAL_PROMPTS
from manifold_emotions.manifold.fit import FittedManifold
from manifold_emotions.manifold.pullback import compute_pullback

NUM_WAYPOINTS_TOTAL = 30
SCALE = 8.0

OUT_DIR = Path("results/time_varying")


def _segment_waypoint_indices(num_waypoints: int, num_segments: int) -> list[int]:
    """Pick representative waypoint indices for each segment.

    For num_waypoints=30, num_segments=8: indices are spread roughly
    [0, 4, 8, 13, 17, 21, 25, 29].
    """
    return [int(round(i * (num_waypoints - 1) / max(num_segments - 1, 1)))
            for i in range(num_segments)]


def _build_prompt(user_text: str, assistant_partial: str) -> str:
    """Build Gemma's chat template manually so we can append arbitrary
    assistant tokens between segment generations.

    Gemma format:
        <start_of_turn>user
        {user}<end_of_turn>
        <start_of_turn>model
        {assistant_partial...}
    """
    return (
        f"<start_of_turn>user\n{user_text}<end_of_turn>\n"
        f"<start_of_turn>model\n{assistant_partial}"
    )


def _build_payload(model_id, layer, hook, prompt: str, waypoint_full, max_tokens):
    stacked = waypoint_full.astype(np.float32).reshape(1, -1)
    return {
        "model": model_id,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 1.0,
        "top_p": 0.95,
        "stop": ["<end_of_turn>", "<start_of_turn>"],
        "steering_vectors": {
            hook: {
                "dtype": str(stacked.dtype),
                "shape": list(stacked.shape),
                "layer_indices": [layer],
                "data": base64.b64encode(stacked.tobytes()).decode("ascii"),
            },
        },
    }


async def _generate_segment(client: httpx.AsyncClient, base_url, model_id, layer, hook,
                            prompt: str, waypoint_full, max_tokens,
                            timeout=300.0):
    payload = _build_payload(model_id, layer, hook, prompt, waypoint_full, max_tokens)
    response = await client.post(
        f"{base_url}/completions",
        json=payload,
        timeout=timeout,
    )
    if response.status_code >= 400:
        raise RuntimeError(
            f"vLLM {response.status_code}: {response.text[:300]}"
        )
    body = response.json()
    choice = body["choices"][0]
    return choice["text"], choice.get("finish_reason")


async def generate_time_varying(client, base_url, model_id, layer, hook,
                                user_prompt: str, waypoints_full: np.ndarray,
                                seg_indices: list[int], tokens_per_segment: int):
    """Generate by stepping through waypoints across segments.

    Returns the concatenated assistant text.
    """
    text = ""
    for wp_idx in seg_indices:
        prompt = _build_prompt(user_prompt, text)
        seg_text, finish = await _generate_segment(
            client, base_url, model_id, layer, hook,
            prompt, waypoints_full[wp_idx], tokens_per_segment,
        )
        text = text + seg_text
        if finish == "stop":
            break
    return text


async def run_pair(start: str, end: str, manifold: FittedManifold,
                   behavior: BehaviorManifold, config,
                   num_segments: int, tokens_per_segment: int,
                   results_suffix: str) -> dict:
    print(f"\n=== time-varying steering: {start} → {end} ===")
    # Compute the trajectories the same way pullback_experiment does
    g = compute_pullback(
        manifold=manifold, behavior=behavior,
        start_label=start, end_label=end,
        num_waypoints=NUM_WAYPOINTS_TOTAL, sigma=None,
    )
    # Scale waypoints to SCALE just like the chord experiment.
    # pullback_full/geodesic_full/linear_full are (K, hidden_size).
    methods = {
        "pullback": np.asarray(g.pullback_full) * SCALE,
        "geodesic": np.asarray(g.geodesic_full) * SCALE,
        "linear": np.asarray(g.linear_full) * SCALE,
    }

    seg_indices = _segment_waypoint_indices(NUM_WAYPOINTS_TOTAL, num_segments)
    print(f"  segment waypoint indices: {seg_indices}")
    print(f"  generating with {num_segments} segments × {tokens_per_segment} tokens "
          f"= {num_segments * tokens_per_segment} total")

    base_url = config.vllm_server.base_url
    model_id = config.model.hf_id
    layer = config.model.target_layer
    hook = config.model.hook_point

    completions = {name: [] for name in methods}
    # Limit concurrency — each generation is K sequential HTTP calls so the
    # natural unit is one (method, prompt) at a time. 8 concurrent gives
    # vLLM a healthy queue without hammering it.
    semaphore = asyncio.Semaphore(8)
    t0 = time.monotonic()
    limits = httpx.Limits(max_connections=16, max_keepalive_connections=16)
    async with httpx.AsyncClient(limits=limits, timeout=httpx.Timeout(300.0)) as client:
        async def one(method_name, wp_full, prompt_idx):
            async with semaphore:
                text = await generate_time_varying(
                    client, base_url, model_id, layer, hook,
                    NEUTRAL_PROMPTS[prompt_idx], wp_full,
                    seg_indices, tokens_per_segment,
                )
            return method_name, prompt_idx, text

        tasks = [
            asyncio.create_task(one(name, wp_full, pi))
            for name, wp_full in methods.items()
            for pi in range(len(NEUTRAL_PROMPTS))
        ]
        for fut in asyncio.as_completed(tasks):
            name, pi, text = await fut
            completions[name].append((pi, text))
            preview = text.replace(chr(10), " ")[:60]
            print(f"  {name} p{pi:>2d}: {preview}…", flush=True)
    print(f"  generation done in {time.monotonic()-t0:.0f}s")

    # Judge each completion
    print("  judging…")
    cache_path = OUT_DIR / f"ratings_{start}_{end}{results_suffix}.json"
    judged = {}
    # Build passages: text_id encodes (method, pair, prompt_idx)
    all_passages = []
    id_map: dict[str, tuple[str, int]] = {}
    for name, items in completions.items():
        items.sort()
        for pi, text in items:
            tid = f"tv_{name}_{start}_{end}_p{pi:02d}"
            all_passages.append((tid, text))
            id_map[tid] = (name, pi)
    ratings = await judge_texts(config, all_passages, cache_path=cache_path)
    # Bucket by method
    by_method: dict[str, list[tuple[int, float, float]]] = {n: [] for n in completions}
    for tid, rating in ratings.items():
        name, pi = id_map[tid]
        by_method[name].append((pi, rating.valence, rating.arousal))
    for name in completions:
        by_method[name].sort()
        va = np.array([[v, a] for _, v, a in by_method[name]])
        judged[name] = va

    # Metrics
    # Approximate the target by the M_y midpoint of start and end
    y_start = behavior.centroids[behavior.labels.index(start)]
    y_end = behavior.centroids[behavior.labels.index(end)]
    target_va = 0.5 * (y_start + y_end)

    # off-M_y E for each method: mean distance from completion to nearest
    # M_y centroid
    centroids = behavior.centroids

    def off_my(va_array):
        dists = []
        for v in va_array:
            d = np.linalg.norm(centroids - v[None, :], axis=1)
            dists.append(float(d.min()))
        return float(np.mean(dists))

    # M_y-line distance: mean Euclidean to target midpoint
    def my_line(va_array):
        return float(np.mean(np.linalg.norm(va_array - target_va[None, :], axis=1)))

    metrics = {name: {
        "off_my_e": off_my(va),
        "my_line": my_line(va),
        "ratings_va": va.tolist(),
    } for name, va in judged.items()}

    print(f"\n  off-M_y E:  pullback={metrics['pullback']['off_my_e']:.3f}  "
          f"geodesic={metrics['geodesic']['off_my_e']:.3f}  "
          f"linear={metrics['linear']['off_my_e']:.3f}")
    print(f"  M_y-line:   pullback={metrics['pullback']['my_line']:.3f}  "
          f"geodesic={metrics['geodesic']['my_line']:.3f}  "
          f"linear={metrics['linear']['my_line']:.3f}")

    return {
        "pair": [start, end],
        "num_segments": num_segments,
        "tokens_per_segment": tokens_per_segment,
        "segment_waypoint_indices": seg_indices,
        "scale": SCALE,
        "target_va": target_va.tolist(),
        "metrics": metrics,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("start", nargs="?", default=None)
    parser.add_argument("end", nargs="?", default=None)
    parser.add_argument("--pairs", type=Path, default=None,
                        help="JSON pair file ([[start, end], ...]) to run "
                             "instead of a single pair")
    parser.add_argument("--manifold", type=Path, default=Path("data/manifold_h.npz"),
                        help="FittedManifold npz (default: production 8-D)")
    parser.add_argument("--segments", type=int, default=8,
                        help="number of steering segments per generation")
    parser.add_argument("--tokens-per-segment", type=int, default=12)
    parser.add_argument("--results-suffix", default="")
    parser.add_argument("--force", action="store_true",
                        help="re-run pairs whose results already exist")
    args = parser.parse_args()

    config = load_config()
    manifold = FittedManifold.load(args.manifold)
    behavior = BehaviorManifold.load(config.paths.manifold_y)

    if args.pairs is not None:
        pairs = [(a, b) for a, b in json.loads(args.pairs.read_text())]
    elif args.start and args.end:
        pairs = [(args.start, args.end)]
    else:
        parser.error("provide either start+end labels or --pairs")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []
    for start, end in pairs:
        out_path = OUT_DIR / f"{start}_{end}{args.results_suffix}.json"
        if out_path.exists() and not args.force:
            print(f"skip {start}→{end}: {out_path} exists (use --force to re-run)")
            continue
        if start not in manifold.labels or end not in manifold.labels \
                or start not in behavior.labels or end not in behavior.labels:
            print(f"skip {start}→{end}: missing centroid in M_h or M_y")
            failed.append(f"{start}->{end} (missing centroid)")
            continue
        try:
            out = await run_pair(
                start, end, manifold, behavior, config,
                num_segments=args.segments,
                tokens_per_segment=args.tokens_per_segment,
                results_suffix=args.results_suffix,
            )
        except Exception as exc:  # noqa: BLE001 — one pair must not kill the run
            print(f"  FAILED {start}→{end}: {type(exc).__name__}: {exc}")
            failed.append(f"{start}->{end} ({type(exc).__name__})")
            continue
        out_path.write_text(json.dumps(out, indent=2))
        print(f"  saved {out_path}")

    if failed:
        print(f"\n{len(failed)} pair(s) failed: {failed}")
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
