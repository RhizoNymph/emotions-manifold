"""Multi-host chord-experiment chain: split pairs, resume, tolerate failures.

Replaces the per-experiment shell chains (alift_*_chain.sh,
pullback_171_*.sh, node1_chain*.sh, overnight/recovery/resume chains —
archived on archive/disorganized-scripts). What those scripts
reimplemented in bash each time is owned here once:

- **host splitting**: pairs are striped across vLLM hosts; each worker
  gets its own ``Config`` with ``vllm_server.base_url`` replaced.
- **resume**: pairs whose outputs already exist are skipped, so an
  interrupted chain rerun picks up where it left off (this is what the
  resume_/recovery_/rerun_failed shell variants existed for).
- **failure tolerance**: one pair erroring doesn't kill the chain; the
  failure is recorded and reported at the end.

Workers share one event loop: generation (the bottleneck) overlaps
fully across hosts; the per-pair synchronous geometry step (~seconds)
briefly serializes, which is acceptable at chain scale.
"""

from __future__ import annotations

import dataclasses
import json
import math
import time
from dataclasses import dataclass

import structlog

from ..config import Config
from ..manifold.pullback import SigmaSpec
from .chord import ChordRunConfig, run_chord_pair

log = structlog.get_logger(__name__)

Pair = tuple[str, str]


@dataclass(frozen=True, slots=True)
class ChainReport:
    """Outcome of a chain run, per pair."""

    completed: tuple[Pair, ...]
    skipped: tuple[Pair, ...]  # already complete before the run (resume)
    failed: tuple[tuple[str, str, str], ...]  # (start, end, error)

    @property
    def ok(self) -> bool:
        return not self.failed


def split_pairs(pairs: list[Pair], num_workers: int) -> list[list[Pair]]:
    """Stripe pairs across workers: worker i gets pairs[i::num_workers].

    Striping (vs contiguous halves, which the shell chains used) keeps
    workers balanced when per-pair cost drifts over the list.
    """
    if num_workers < 1:
        raise ValueError(f"num_workers must be >= 1, got {num_workers}")
    return [pairs[i::num_workers] for i in range(num_workers)]


def pair_is_complete(
    run: ChordRunConfig, start: str, end: str, results_suffix: str = ""
) -> bool:
    """True when the pair's outputs for this variant already exist.

    - judge "none": the phase-1 artifacts (completions + summary skeleton)
      exist — judging happens in a separate phase-2 pass.
    - judged modes: the summary exists and every trajectory has a finite
      ``off_manifold_energy`` (a NaN skeleton from an earlier nojudge run
      does not count as complete).
    """
    summary_path = run.results_dir / f"{start}_{end}{results_suffix}.json"
    if not summary_path.exists():
        return False
    if run.judge == "none":
        completions = run.data_dir / f"completions_{start}_{end}{results_suffix}.json"
        return completions.exists()
    summary = json.loads(summary_path.read_text())
    trajectories = summary.get("trajectories", {})
    if set(trajectories) != {"pullback", "geodesic", "linear"}:
        return False
    return all(
        isinstance(t.get("off_manifold_energy"), float)
        and math.isfinite(t["off_manifold_energy"])
        for t in trajectories.values()
    )


def _worker_config(config: Config, host: str) -> Config:
    """Per-worker Config pointing generation at a specific vLLM host."""
    return dataclasses.replace(
        config,
        vllm_server=dataclasses.replace(config.vllm_server, base_url=host),
    )


async def run_chain(
    config: Config,
    run: ChordRunConfig,
    pairs: list[Pair],
    hosts: list[str] | None = None,
    *,
    num_waypoints: int | None = None,
    num_prompts: int | None = None,
    sigma: SigmaSpec | None = None,
    results_suffix: str = "",
    force: bool = False,
) -> ChainReport:
    """Run every pair of the chain across ``hosts``; return a ChainReport.

    ``hosts`` defaults to the single configured vLLM server. With
    ``force=True`` already-complete pairs are re-run instead of skipped.
    """
    import asyncio

    if hosts is None or not hosts:
        hosts = [config.vllm_server.base_url]

    if force:
        pending = list(pairs)
        skipped: list[Pair] = []
    else:
        pending = [p for p in pairs if not pair_is_complete(run, *p, results_suffix)]
        skipped = [p for p in pairs if p not in pending]

    log.info(
        "experiments.chain.start",
        chain=run.name,
        judge=run.judge,
        total=len(pairs),
        skipped=len(skipped),
        pending=len(pending),
        hosts=hosts,
    )

    completed: list[Pair] = []
    failed: list[tuple[str, str, str]] = []

    async def worker(worker_host: str, share: list[Pair]) -> None:
        worker_cfg = _worker_config(config, worker_host)
        for start, end in share:
            t0 = time.monotonic()
            try:
                summary_path = await run_chord_pair(
                    worker_cfg, run, start, end,
                    num_waypoints=num_waypoints,
                    num_prompts=num_prompts,
                    sigma=sigma,
                    results_suffix=results_suffix,
                )
            except Exception as exc:  # noqa: BLE001 — chain must outlive any pair
                log.error(
                    "experiments.chain.pair_failed",
                    chain=run.name, host=worker_host,
                    pair=f"{start}->{end}", error=str(exc),
                )
                failed.append((start, end, f"{type(exc).__name__}: {exc}"))
                continue
            if summary_path is None:
                failed.append((start, end, "skipped: missing centroid in M_h or M_y"))
                continue
            completed.append((start, end))
            log.info(
                "experiments.chain.pair_done",
                chain=run.name, host=worker_host,
                pair=f"{start}->{end}",
                wall_sec=round(time.monotonic() - t0, 1),
                done=len(completed), pending=len(pending) - len(completed) - len(failed),
            )

    shares = split_pairs(pending, len(hosts))
    await asyncio.gather(*(
        worker(host, share) for host, share in zip(hosts, shares, strict=True)
    ))

    report = ChainReport(
        completed=tuple(completed),
        skipped=tuple(skipped),
        failed=tuple(failed),
    )
    log.info(
        "experiments.chain.done",
        chain=run.name,
        completed=len(report.completed),
        skipped=len(report.skipped),
        failed=len(report.failed),
    )
    return report
