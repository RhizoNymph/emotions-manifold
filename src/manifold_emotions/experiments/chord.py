"""The chord experiment: pullback vs geodesic vs linear over an emotion pair.

One pair → three M_h trajectories (kernel-barycenter pullback, G_E
geodesic, straight line) → steered generation at every waypoint →
V/A judging → two behavioral metrics (off-M_y E, M_y-line distance).

This module unifies the per-variant scripts (run_pullback_experiment.py
and its _4d/_6d/_8d_silverman/_nojudge copies, archived on
archive/disorganized-scripts). A variant is now a ``ChordRunConfig``,
usually loaded from a YAML file under ``experiments/``:

    name: pullback_4d            # output dirs: data/{name}, results/{name}
    manifold: data/manifold_h_4d_full.npz
    judge: sequential            # sequential | batched | none
    summary_extra:               # optional, merged into the summary JSON
      bandwidth_heuristic: silverman

``judge: none`` reproduces the nojudge two-phase flow: raw completions
are written to ``data/{name}/completions_{pair}.json`` and the summary
skeleton carries NaN metrics with ``"phase": "nojudge"`` for a later
batch-judging pass to fill in.

Output filenames and JSON shapes match the original scripts so existing
results under ``results/pullback*`` stay comparable, with one addition:
the summary always includes ``manifold_dim``.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import structlog
import yaml

from ..behavior.judge_text import TextRating, judge_texts
from ..behavior.judge_text_batched import judge_texts_batched
from ..behavior.manifold import BehaviorManifold
from ..config import Config
from ..manifold.fit import FittedManifold
from ..manifold.pullback import SigmaSpec, compute_pullback
from ..manifold.spline import SplineManifold
from ..manifold.spline_geodesic import fit_spline_geodesic
from ..steering.pullback_experiment import (
    PullbackExperimentReport,
    run_pullback_experiment,
)
from ..steering.trajectory import generate_along_path

log = structlog.get_logger(__name__)

JudgeMode = Literal["sequential", "batched", "none"]

# Every trajectory the nojudge phase can emit. The three ambient trajectories are
# always available; the two spline trajectories only when a spline manifold is set.
AMBIENT_TRAJECTORIES: tuple[str, ...] = ("pullback", "geodesic", "linear")
SPLINE_TRAJECTORIES: tuple[str, ...] = ("spline_induced", "spline_density")
ALL_TRAJECTORIES: tuple[str, ...] = AMBIENT_TRAJECTORIES + SPLINE_TRAJECTORIES

JudgeFn = Callable[..., Awaitable[dict[str, TextRating]]]

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

_JUDGE_FNS: dict[str, JudgeFn] = {
    "sequential": judge_texts,
    "batched": judge_texts_batched,
}


@dataclass(frozen=True, slots=True)
class ChordRunConfig:
    """One chord-experiment variant: manifold choice, judge mode, outputs."""

    name: str
    manifold_path: Path
    judge: JudgeMode = "sequential"
    num_waypoints: int = 30
    num_prompts: int = 10
    max_tokens: int = 96
    concurrency: int = 16
    steering_scale: float = 8.0
    sigma: SigmaSpec = None
    prompts: tuple[str, ...] = NEUTRAL_PROMPTS
    summary_extra: dict = field(default_factory=dict)
    # Optional parametric-spline manifold. When set, two extra trajectories —
    # ``spline_induced`` and ``spline_density`` (surface geodesics under the
    # first-fundamental-form and density-weighted metrics) — are generated and
    # judged alongside pullback/geodesic/linear. Only supported with judge:none.
    spline_path: Path | None = None
    # Optional GPU-frugality knob (nojudge phase only): restrict which trajectories
    # are actually generated. ``None`` = all applicable (pullback/geodesic/linear,
    # plus the two spline trajectories when a spline is set). Selecting e.g.
    # ("spline_induced", "spline_density", "linear") skips pullback+geodesic
    # generation (~40% fewer waypoints steered). The pullback geometry is still
    # computed (cheap, CPU) so paths/geometry metrics stay populated.
    trajectories: tuple[str, ...] | None = None

    @property
    def data_dir(self) -> Path:
        return Path("data") / self.name

    @property
    def results_dir(self) -> Path:
        return Path("results") / self.name

    @classmethod
    def from_yaml(cls, path: Path) -> ChordRunConfig:
        raw = yaml.safe_load(Path(path).read_text())
        known = {
            "name", "manifold", "judge", "num_waypoints", "num_prompts",
            "max_tokens", "concurrency", "steering_scale", "sigma",
            "prompts", "summary_extra", "spline", "trajectories",
        }
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown keys in {path}: {sorted(unknown)}")
        if raw.get("judge", "sequential") not in ("sequential", "batched", "none"):
            raise ValueError(f"invalid judge mode in {path}: {raw['judge']!r}")
        spline = raw.get("spline")
        trajectories = raw.get("trajectories")
        if trajectories is not None:
            invalid = [t for t in trajectories if t not in ALL_TRAJECTORIES]
            if invalid:
                raise ValueError(
                    f"invalid trajectories in {path}: {invalid} "
                    f"(valid: {sorted(ALL_TRAJECTORIES)})"
                )
            if not spline and any(t in SPLINE_TRAJECTORIES for t in trajectories):
                raise ValueError(
                    f"{path}: spline trajectories selected but no 'spline' manifold set"
                )
            trajectories = tuple(trajectories)
        return cls(
            name=raw["name"],
            manifold_path=Path(raw["manifold"]),
            judge=raw.get("judge", "sequential"),
            num_waypoints=raw.get("num_waypoints", 30),
            num_prompts=raw.get("num_prompts", 10),
            max_tokens=raw.get("max_tokens", 96),
            concurrency=raw.get("concurrency", 16),
            steering_scale=raw.get("steering_scale", 8.0),
            sigma=raw.get("sigma"),
            prompts=tuple(raw.get("prompts", NEUTRAL_PROMPTS)),
            summary_extra=raw.get("summary_extra", {}),
            spline_path=Path(spline) if spline else None,
            trajectories=trajectories,
        )


@dataclass(frozen=True)
class CompletionRecord:
    """Per-(method, waypoint, prompt) raw text record for deferred judging.

    ``text_id`` matches the sequential pipeline's format
    (``{method}_{start}_{end}_wp{N:03d}_p{N:02d}``, literal spaces in
    multi-word labels preserved) so batched ratings caches can be swapped
    in for sequential ones without re-keying.
    """

    text_id: str
    method: str  # "pullback" | "geodesic" | "linear"
    waypoint: int
    prompt: int
    text: str


def _build_text_id(method: str, start: str, end: str, waypoint: int, prompt: int) -> str:
    return f"{method}_{start}_{end}_wp{waypoint:03d}_p{prompt:02d}"


def _geometry_dict(g) -> dict:
    """The pure-geometry (no LLM) section of the summary JSON."""
    return {
        "pullback_length": float(g.pullback_length),
        "geodesic_length": float(g.geodesic_length),
        "linear_length": float(g.linear_length),
        "mean_dist_pullback_to_geodesic": float(g.mean_dist_to_geodesic),
        "mean_dist_pullback_to_linear": float(g.mean_dist_to_linear),
        "closer_to": g.closer_to,
        "per_waypoint_dist_pullback_to_geodesic": g.dist_pullback_to_geodesic.tolist(),
        "per_waypoint_dist_pullback_to_linear": g.dist_pullback_to_linear.tolist(),
        "my_path_valence": g.my_path[:, 0].tolist(),
        "my_path_arousal": g.my_path[:, 1].tolist(),
    }


def _save_paths_npz(path: Path, g, extra: dict | None = None) -> None:
    """Subspace + full-residual trajectories for downstream plotting."""
    arrays = {
        "my_path": g.my_path,
        "pullback_sub": g.pullback_sub,
        "geodesic_sub": g.geodesic_sub,
        "linear_sub": g.linear_sub,
        "pullback_full": g.pullback_full,
        "geodesic_full": g.geodesic_full,
        "linear_full": g.linear_full,
    }
    if extra:
        arrays.update(extra)
    np.savez_compressed(path, **arrays)


def _spline_steer_paths(
    spline: SplineManifold,
    start: str,
    end: str,
    *,
    num_waypoints: int,
    steering_scale: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Build the two spline-geodesic steering trajectories for a pair.

    Returns ``(steer, sub)`` where ``steer[name]`` is the (K, hidden) scaled
    full-residual steering path and ``sub[name]`` the (K, d) subspace path
    (for the paths npz). Endpoints are snapped to the exact subspace centroids
    so the spline paths share endpoints with pullback/geodesic/linear.
    """
    idx = {lab: i for i, lab in enumerate(spline.labels)}
    i, j = idx[start], idx[end]
    va_s, va_e = spline.control_coords[i], spline.control_coords[j]
    c_s, c_e = spline.centroids_subspace[i], spline.centroids_subspace[j]
    steer: dict[str, np.ndarray] = {}
    sub: dict[str, np.ndarray] = {}
    for name, metric in (("spline_induced", "induced"), ("spline_density", "density")):
        res = fit_spline_geodesic(
            spline, va_s, va_e, metric=metric, num_waypoints=num_waypoints,
            snap_start=c_s, snap_end=c_e,
        )
        sub[name] = res.waypoints.astype(np.float32)
        steer[name] = (spline.unproject(res.waypoints).astype(np.float32) * steering_scale)
    return steer, sub


def _summary_dict(
    report: PullbackExperimentReport, manifold_dim: int, run: ChordRunConfig
) -> dict:
    g = report.geometry
    return {
        "pair": [report.start_label, report.end_label],
        "manifold_dim": manifold_dim,
        "num_waypoints": int(g.num_waypoints),
        "sigma": g.sigma,
        "sigma_spec": g.sigma_spec,
        "sigma_per_waypoint": g.sigma_per_waypoint.tolist(),
        "geometry": _geometry_dict(g),
        "trajectories": {
            name: {
                "off_manifold_energy": traj.off_manifold_energy,
                "my_geodesic_distance": traj.my_geodesic_distance,
                "waypoint_valence": traj.waypoint_behavior_mean[:, 0].tolist(),
                "waypoint_arousal": traj.waypoint_behavior_mean[:, 1].tolist(),
            }
            for name, traj in [
                ("pullback", report.pullback),
                ("geodesic", report.geodesic),
                ("linear", report.linear),
            ]
        },
        **run.summary_extra,
    }


async def _run_pair_nojudge(
    config: Config,
    run: ChordRunConfig,
    manifold: FittedManifold,
    behavior: BehaviorManifold,
    start: str,
    end: str,
    *,
    num_waypoints: int,
    num_prompts: int,
    sigma: SigmaSpec,
    results_suffix: str,
) -> Path:
    """Generate steered completions but defer judging (phase-1 of batched flow)."""
    g = compute_pullback(
        manifold=manifold, behavior=behavior,
        start_label=start, end_label=end,
        num_waypoints=num_waypoints, sigma=sigma,
    )

    # GPU-frugality: only build steering paths for the selected trajectories.
    # ``run.trajectories is None`` means "all applicable". The pullback geometry
    # ``g`` is always computed (cheap CPU) so geometry/paths stay populated.
    selected = run.trajectories

    def keep(name: str) -> bool:
        return selected is None or name in selected

    methods: dict[str, np.ndarray] = {}
    for name, full in (
        ("pullback", g.pullback_full),
        ("geodesic", g.geodesic_full),
        ("linear", g.linear_full),
    ):
        if keep(name):
            methods[name] = np.asarray(full, dtype=np.float32) * run.steering_scale

    spline_sub: dict[str, np.ndarray] = {}
    if run.spline_path is not None and any(keep(n) for n in SPLINE_TRAJECTORIES):
        spline = SplineManifold.load(run.spline_path)
        spline_steer, spline_sub_all = _spline_steer_paths(
            spline, start, end,
            num_waypoints=num_waypoints, steering_scale=run.steering_scale,
        )
        for name in SPLINE_TRAJECTORIES:
            if keep(name):
                methods[name] = spline_steer[name]
                spline_sub[name] = spline_sub_all[name]
    prompts_used = list(run.prompts[:num_prompts])

    records: list[CompletionRecord] = []
    for method, waypoints in methods.items():
        continuations = await generate_along_path(
            config, waypoints, prompts_used,
            max_tokens=run.max_tokens, concurrency=run.concurrency,
        )
        for c in continuations:
            records.append(CompletionRecord(
                text_id=_build_text_id(method, start, end, c.waypoint_index, c.prompt_index),
                method=method,
                waypoint=c.waypoint_index,
                prompt=c.prompt_index,
                text=c.text,
            ))

    completions_path = run.data_dir / f"completions_{start}_{end}{results_suffix}.json"
    completions_path.write_text(json.dumps([asdict(r) for r in records], indent=2))
    log.info("experiments.chord.completions_written",
             pair=f"{start}->{end}", n=len(records), path=str(completions_path))

    extra = {f"{name}_sub": arr for name, arr in spline_sub.items()}
    _save_paths_npz(run.data_dir / f"paths_{start}_{end}{results_suffix}.npz", g, extra=extra)

    # NaN skeleton — a later batch-judging pass mutates this in place. Trajectory
    # keys follow whatever methods were generated (pullback/geodesic/linear plus
    # the two spline trajectories when a spline manifold is configured).
    skeleton = {
        "pair": [start, end],
        "manifold_dim": int(manifold.num_components),
        "num_waypoints": int(num_waypoints),
        "sigma": float(g.sigma),
        "sigma_spec": g.sigma_spec,
        "sigma_per_waypoint": g.sigma_per_waypoint.tolist(),
        "geometry": _geometry_dict(g),
        "trajectories": {
            method: {
                "off_manifold_energy": float("nan"),
                "my_geodesic_distance": float("nan"),
                "waypoint_valence": [float("nan")] * num_waypoints,
                "waypoint_arousal": [float("nan")] * num_waypoints,
            }
            for method in methods
        },
        "phase": "nojudge",
        **run.summary_extra,
    }
    summary_path = run.results_dir / f"{start}_{end}{results_suffix}.json"
    summary_path.write_text(json.dumps(skeleton, indent=2))
    return summary_path


async def run_chord_pair(
    config: Config,
    run: ChordRunConfig,
    start: str,
    end: str,
    *,
    num_waypoints: int | None = None,
    num_prompts: int | None = None,
    sigma: SigmaSpec | None = None,
    results_suffix: str = "",
) -> Path | None:
    """Run one pair under ``run``'s variant; return the summary path.

    Keyword overrides (CLI-provided) take precedence over the config.
    Returns None when either label is missing from M_h or M_y.
    """
    k = num_waypoints if num_waypoints is not None else run.num_waypoints
    n = num_prompts if num_prompts is not None else run.num_prompts
    sig = sigma if sigma is not None else run.sigma

    manifold = FittedManifold.load(run.manifold_path)
    behavior = BehaviorManifold.load(config.paths.manifold_y)

    if start not in manifold.labels or end not in manifold.labels:
        log.warning("experiments.chord.skip", pair=f"{start}->{end}", missing="M_h")
        return None
    if start not in behavior.labels or end not in behavior.labels:
        log.warning("experiments.chord.skip", pair=f"{start}->{end}", missing="M_y")
        return None

    run.data_dir.mkdir(parents=True, exist_ok=True)
    run.results_dir.mkdir(parents=True, exist_ok=True)

    if run.spline_path is not None and run.judge != "none":
        raise ValueError(
            "spline trajectories are only supported with judge: none "
            "(two-phase batched flow); set judge: none in the config"
        )

    if run.judge == "none":
        return await _run_pair_nojudge(
            config, run, manifold, behavior, start, end,
            num_waypoints=k, num_prompts=n, sigma=sig,
            results_suffix=results_suffix,
        )

    judge_cache = run.data_dir / f"ratings_{start}_{end}{results_suffix}.json"
    report = await run_pullback_experiment(
        config=config,
        manifold=manifold,
        behavior=behavior,
        start_label=start,
        end_label=end,
        num_waypoints=k,
        num_prompts=n,
        prompts=run.prompts,
        max_tokens=run.max_tokens,
        concurrency=run.concurrency,
        steering_scale=run.steering_scale,
        judge_cache_path=judge_cache,
        sigma=sig,
        judge_fn=_JUDGE_FNS[run.judge],
    )

    summary = _summary_dict(report, int(manifold.num_components), run)
    summary_path = run.results_dir / f"{start}_{end}{results_suffix}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    _save_paths_npz(
        run.data_dir / f"paths_{start}_{end}{results_suffix}.npz", report.geometry
    )
    return summary_path
