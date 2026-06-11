"""Time-varying steering: step through trajectory waypoints during generation.

Goodfire's central temporal claim is that following the trajectory *during*
generation (rather than holding one steering vector constant) is what
manifold-aware steering buys. The mechanism here divides generation into
``num_segments`` sequential ``/v1/completions`` calls of
``tokens_per_segment`` tokens each, rebuilding Gemma's chat template
manually so the assistant text accumulates across calls, and swapping the
steering vector between calls.

The n=12 result (results/time_varying, day 5) confounded three things:
time-variation itself, the *discontinuity* of hard vector switches, and
the *segmentation* of generation into separate calls. The ``schedule``
axis separates them:

- ``"varying"``: segment k uses waypoint ``w[seg_indices[k]]`` — the
  original hard-switch design. Finer segmentation (16 × 6 vs 8 × 12)
  halves each switch's jump at the same total token budget, which is the
  smoothed-TV variant.
- ``"constant"``: every segment uses the *path-midpoint waypoint*
  ``w[(num_waypoints - 1) // 2]``, delivered through the identical
  segmented call structure. This is the control that isolates the
  segmentation artifact: if constant-segmented already differs from the
  unsegmented chord baseline, segmentation (KV rebuild, stop-token
  boundaries) contaminated the TV comparison. The midpoint waypoint is
  the natural constant comparator because both behavioral metrics target
  the pair midpoint.

Judging mirrors the chord experiment's phase split: ``judge="none"``
generates and saves completions only (resumable); a later run with
``judge="batched"`` (or ``"sequential"``) loads saved completions, rates
everything in one judge call, and writes per-pair results. Result files
keep the original n=12 schema (``{"pair", "metrics": {method:
{off_my_e, my_line, ratings_va}}}``) so ``analyze_time_varying.py``
reads any condition directory unchanged.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx
import numpy as np
import structlog

from ..behavior.judge_text import judge_texts
from ..behavior.judge_text_batched import judge_texts_batched
from ..behavior.manifold import BehaviorManifold
from ..config import Config
from ..manifold.fit import FittedManifold
from ..manifold.pullback import compute_pullback
from .chord import NEUTRAL_PROMPTS

log = structlog.get_logger(__name__)

Schedule = Literal["varying", "constant"]
JudgeMode = Literal["sequential", "batched", "none"]
Pair = tuple[str, str]

METHODS = ("pullback", "geodesic", "linear")

_JUDGE_FNS = {"sequential": judge_texts, "batched": judge_texts_batched}


@dataclass(frozen=True, slots=True)
class TVRunConfig:
    """One time-varying steering condition.

    ``out_dir`` holds everything the condition produces:
    ``completions_{start}_{end}{suffix}.json``, ``{start}_{end}{suffix}.json``
    (metrics), and the shared judge cache. Use a fresh directory per
    condition — never overwrite an earlier condition's results.
    """

    out_dir: Path
    manifold_path: Path
    schedule: Schedule = "varying"
    judge: JudgeMode = "sequential"
    num_segments: int = 8
    tokens_per_segment: int = 12
    results_suffix: str = ""
    num_waypoints: int = 30
    scale: float = 8.0
    concurrency: int = 8

    def completions_path(self, start: str, end: str) -> Path:
        return self.out_dir / f"completions_{start}_{end}{self.results_suffix}.json"

    def result_path(self, start: str, end: str) -> Path:
        return self.out_dir / f"{start}_{end}{self.results_suffix}.json"

    @property
    def ratings_cache_path(self) -> Path:
        return self.out_dir / f"ratings_cache{self.results_suffix}.json"


@dataclass(frozen=True, slots=True)
class TVReport:
    """Outcome of a run over a pair list."""

    generated: tuple[Pair, ...]   # completions produced this run
    judged: tuple[Pair, ...]      # result files written this run
    skipped: tuple[Pair, ...]     # already complete for the requested phase
    failed: tuple[tuple[str, str, str], ...]  # (start, end, error)

    @property
    def ok(self) -> bool:
        return not self.failed


def segment_waypoint_indices(num_waypoints: int, num_segments: int) -> list[int]:
    """Spread segment waypoint picks evenly over the trajectory.

    For num_waypoints=30, num_segments=8: [0, 4, 8, 12, 17, 21, 25, 29].
    """
    return [int(round(i * (num_waypoints - 1) / max(num_segments - 1, 1)))
            for i in range(num_segments)]


def schedule_indices(num_waypoints: int, num_segments: int, schedule: Schedule) -> list[int]:
    """Waypoint index used for each segment under the given schedule."""
    if schedule == "varying":
        return segment_waypoint_indices(num_waypoints, num_segments)
    if schedule == "constant":
        return [(num_waypoints - 1) // 2] * num_segments
    raise ValueError(f"unknown schedule: {schedule!r}")


def off_my_energy(va: np.ndarray, centroids: np.ndarray) -> float:
    """Mean distance from each rated completion to its nearest M_y centroid."""
    dists = np.linalg.norm(centroids[None, :, :] - va[:, None, :], axis=2)
    return float(dists.min(axis=1).mean())


def my_line_distance(va: np.ndarray, target_va: np.ndarray) -> float:
    """Mean distance from each rated completion to the pair's M_y midpoint."""
    return float(np.mean(np.linalg.norm(va - target_va[None, :], axis=1)))


def _build_prompt(user_text: str, assistant_partial: str) -> str:
    """Gemma chat template, built manually so segment generations can
    continue arbitrary assistant text (the chat endpoint re-renders)."""
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
    response = await client.post(f"{base_url}/completions", json=payload, timeout=timeout)
    if response.status_code >= 400:
        raise RuntimeError(f"vLLM {response.status_code}: {response.text[:300]}")
    choice = response.json()["choices"][0]
    return choice["text"], choice.get("finish_reason")


async def generate_segmented(client, base_url, model_id, layer, hook,
                             user_prompt: str, waypoints_full: np.ndarray,
                             seg_indices: list[int], tokens_per_segment: int) -> str:
    """One generation: K sequential segment calls stepping ``seg_indices``."""
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


def _method_waypoints(manifold: FittedManifold, behavior: BehaviorManifold,
                      start: str, end: str, run: TVRunConfig) -> dict[str, np.ndarray]:
    """Scaled full-space waypoint stacks per method, as the chord experiment."""
    g = compute_pullback(
        manifold=manifold, behavior=behavior,
        start_label=start, end_label=end,
        num_waypoints=run.num_waypoints, sigma=None,
    )
    return {
        "pullback": np.asarray(g.pullback_full) * run.scale,
        "geodesic": np.asarray(g.geodesic_full) * run.scale,
        "linear": np.asarray(g.linear_full) * run.scale,
    }


async def _generate_pair(config: Config, run: TVRunConfig,
                         manifold: FittedManifold, behavior: BehaviorManifold,
                         start: str, end: str) -> dict[str, dict[str, str]]:
    """Generate all (method, prompt) completions for one pair and save them."""
    methods = _method_waypoints(manifold, behavior, start, end, run)
    seg_indices = schedule_indices(run.num_waypoints, run.num_segments, run.schedule)

    base_url = config.vllm_server.base_url
    model_id = config.model.hf_id
    layer = config.model.target_layer
    hook = config.model.hook_point

    completions: dict[str, dict[str, str]] = {name: {} for name in methods}
    semaphore = asyncio.Semaphore(run.concurrency)
    t0 = time.monotonic()
    limits = httpx.Limits(max_connections=16, max_keepalive_connections=16)
    async with httpx.AsyncClient(limits=limits, timeout=httpx.Timeout(300.0)) as client:
        async def one(method_name: str, wp_full: np.ndarray, prompt_idx: int):
            async with semaphore:
                text = await generate_segmented(
                    client, base_url, model_id, layer, hook,
                    NEUTRAL_PROMPTS[prompt_idx], wp_full,
                    seg_indices, run.tokens_per_segment,
                )
            return method_name, prompt_idx, text

        tasks = [
            asyncio.create_task(one(name, wp_full, pi))
            for name, wp_full in methods.items()
            for pi in range(len(NEUTRAL_PROMPTS))
        ]
        for fut in asyncio.as_completed(tasks):
            name, pi, text = await fut
            completions[name][str(pi)] = text

    log.info(
        "experiments.tv.pair_generated",
        pair=f"{start}->{end}", schedule=run.schedule,
        segments=run.num_segments, wall_sec=round(time.monotonic() - t0, 1),
    )
    run.out_dir.mkdir(parents=True, exist_ok=True)
    run.completions_path(start, end).write_text(json.dumps({
        "pair": [start, end],
        "schedule": run.schedule,
        "num_segments": run.num_segments,
        "tokens_per_segment": run.tokens_per_segment,
        "segment_waypoint_indices": seg_indices,
        "scale": run.scale,
        "completions": completions,
    }, indent=2))
    return completions


def _text_id(start: str, end: str, method: str, prompt_idx: int) -> str:
    return f"tv_{method}_{start}_{end}_p{prompt_idx:02d}"


def _pair_result(run: TVRunConfig, behavior: BehaviorManifold,
                 start: str, end: str, comp_meta: dict,
                 ratings_va: dict[str, dict[str, tuple[float, float]]]) -> dict:
    """Metrics in the original n=12 result schema."""
    y_start = behavior.centroids[behavior.labels.index(start)]
    y_end = behavior.centroids[behavior.labels.index(end)]
    target_va = 0.5 * (y_start + y_end)

    metrics = {}
    for method in METHODS:
        by_prompt = ratings_va[method]
        va = np.array([by_prompt[pi] for pi in sorted(by_prompt)])
        metrics[method] = {
            "off_my_e": off_my_energy(va, behavior.centroids),
            "my_line": my_line_distance(va, target_va),
            "ratings_va": va.tolist(),
        }
    return {
        "pair": [start, end],
        "schedule": comp_meta["schedule"],
        "num_segments": comp_meta["num_segments"],
        "tokens_per_segment": comp_meta["tokens_per_segment"],
        "segment_waypoint_indices": comp_meta["segment_waypoint_indices"],
        "scale": comp_meta["scale"],
        "target_va": target_va.tolist(),
        "metrics": metrics,
    }


async def run_tv_pairs(config: Config, run: TVRunConfig, pairs: list[Pair],
                       *, force: bool = False) -> TVReport:
    """Run a TV condition over ``pairs``: generate, then judge in one pass.

    - Generation is per pair, resumable: pairs with a completions file are
      not regenerated unless ``force``.
    - With ``judge="none"`` the run stops after generation.
    - Otherwise all pending texts go through *one* judge call (so the
      batched pipeline makes one batch set per condition, not per pair),
      then per-pair result files are written.
    """
    manifold = FittedManifold.load(run.manifold_path)
    behavior = BehaviorManifold.load(config.paths.manifold_y)

    generated: list[Pair] = []
    skipped: list[Pair] = []
    failed: list[tuple[str, str, str]] = []
    runnable: list[Pair] = []

    for start, end in pairs:
        if start not in manifold.labels or end not in manifold.labels \
                or start not in behavior.labels or end not in behavior.labels:
            failed.append((start, end, "missing centroid in M_h or M_y"))
            continue
        runnable.append((start, end))

    for start, end in runnable:
        if run.completions_path(start, end).exists() and not force:
            continue
        try:
            await _generate_pair(config, run, manifold, behavior, start, end)
        except Exception as exc:  # noqa: BLE001 — one pair must not kill the run
            log.error("experiments.tv.pair_failed", pair=f"{start}->{end}",
                      error=f"{type(exc).__name__}: {exc}")
            failed.append((start, end, f"{type(exc).__name__}: {exc}"))
            continue
        generated.append((start, end))

    failed_pairs = {(s, e) for s, e, _ in failed}
    if run.judge == "none":
        skipped = [p for p in runnable
                   if p not in generated and p not in failed_pairs]
        return TVReport(tuple(generated), (), tuple(skipped), tuple(failed))

    # Judge phase: collect texts for every pair missing a result file.
    pending: list[Pair] = []
    passages: list[tuple[str, str]] = []
    comp_meta: dict[Pair, dict] = {}
    for start, end in runnable:
        if (start, end) in failed_pairs:
            continue
        if run.result_path(start, end).exists() and not force:
            skipped.append((start, end))
            continue
        comp_path = run.completions_path(start, end)
        if not comp_path.exists():
            failed.append((start, end, "no completions file to judge"))
            continue
        data = json.loads(comp_path.read_text())
        comp_meta[(start, end)] = data
        pending.append((start, end))
        for method, by_prompt in data["completions"].items():
            for pi, text in by_prompt.items():
                passages.append((_text_id(start, end, method, int(pi)), text))

    judged: list[Pair] = []
    if pending:
        judge_fn = _JUDGE_FNS[run.judge]
        log.info("experiments.tv.judging", judge=run.judge,
                 pairs=len(pending), texts=len(passages))
        ratings = await judge_fn(config, passages,
                                 cache_path=run.ratings_cache_path)
        for start, end in pending:
            by_method: dict[str, dict[str, tuple[float, float]]] = {
                m: {} for m in METHODS
            }
            for method in METHODS:
                for pi in comp_meta[(start, end)]["completions"][method]:
                    tid = _text_id(start, end, method, int(pi))
                    if tid in ratings:
                        r = ratings[tid]
                        by_method[method][f"{int(pi):02d}"] = (r.valence, r.arousal)
            if not all(by_method[m] for m in METHODS):
                failed.append((start, end, "judge returned no ratings for a method"))
                continue
            result = _pair_result(run, behavior, start, end,
                                  comp_meta[(start, end)], by_method)
            run.result_path(start, end).write_text(json.dumps(result, indent=2))
            judged.append((start, end))

    report = TVReport(tuple(generated), tuple(judged), tuple(skipped), tuple(failed))
    log.info("experiments.tv.done", schedule=run.schedule, judge=run.judge,
             generated=len(report.generated), judged=len(report.judged),
             skipped=len(report.skipped), failed=len(report.failed))
    return report
