"""Time-varying experiment: schedules, metrics, phase split, resume."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

import manifold_emotions.experiments.time_varying as tv_mod
from manifold_emotions.behavior.judge_text import TextRating
from manifold_emotions.behavior.manifold import BehaviorManifold
from manifold_emotions.config import load_config
from manifold_emotions.experiments.time_varying import (
    TVRunConfig,
    my_line_distance,
    off_my_energy,
    run_tv_pairs,
    schedule_indices,
    segment_waypoint_indices,
)
from manifold_emotions.manifold.fit import FittedManifold


def test_segment_waypoint_indices_pinned() -> None:
    # The original n=12 design: 30 waypoints, 8 segments.
    assert segment_waypoint_indices(30, 8) == [0, 4, 8, 12, 17, 21, 25, 29]
    # The smoothed variant: every boundary jump roughly halves.
    idx16 = segment_waypoint_indices(30, 16)
    assert idx16 == [0, 2, 4, 6, 8, 10, 12, 14, 15, 17, 19, 21, 23, 25, 27, 29]
    assert idx16[0] == 0 and idx16[-1] == 29
    assert max(b - a for a, b in zip(idx16, idx16[1:], strict=False)) <= 2
    assert segment_waypoint_indices(30, 1) == [0]


def test_schedule_indices() -> None:
    assert schedule_indices(30, 8, "varying") == segment_waypoint_indices(30, 8)
    assert schedule_indices(30, 8, "constant") == [14] * 8
    assert schedule_indices(30, 16, "constant") == [14] * 16
    with pytest.raises(ValueError, match="schedule"):
        schedule_indices(30, 8, "wavy")  # type: ignore[arg-type]


def test_metrics_hand_values() -> None:
    centroids = np.array([[0.0, 0.0], [4.0, 0.0]])
    va = np.array([[1.0, 0.0], [3.0, 0.0]])
    # nearest-centroid distances: 1.0 (to [0,0]) and 1.0 (to [4,0])
    assert off_my_energy(va, centroids) == pytest.approx(1.0)
    # distances to midpoint [2, 0]: 1.0 and 1.0
    assert my_line_distance(va, np.array([2.0, 0.0])) == pytest.approx(1.0)


def _save_fixtures(tmp_path: Path, labels: tuple[str, ...]) -> Path:
    """Write tiny M_h/M_y fixtures under tmp_path; return the M_h path."""
    n = len(labels)
    rng = np.random.default_rng(0)
    behavior = BehaviorManifold(
        labels=labels,
        centroids=rng.normal(size=(n, 2)).astype(np.float64) * 3,
        stds=np.ones((n, 2)),
        story_counts=np.full(n, 50),
    )
    behavior.save(tmp_path / "data" / "manifold_y.npz")
    manifold = FittedManifold(
        labels=labels,
        centroids_subspace=rng.normal(size=(n, 4)).astype(np.float32),
        pca_components=rng.normal(size=(4, 16)).astype(np.float32),
        pca_mean=np.zeros(16, dtype=np.float32),
        pca_explained_variance_ratio=np.full(4, 0.25),
        kde_bandwidth=1.0,
        alpha=1.0,
        beta=0.01,
    )
    manifold_path = tmp_path / "data" / "manifold_h_test.npz"
    manifold.save(manifold_path)
    return manifold_path


def _fake_waypoints(manifold, behavior, start, end, run):
    rng = np.random.default_rng(42)
    return {m: rng.normal(size=(run.num_waypoints, 16)).astype(np.float32)
            for m in ("pullback", "geodesic", "linear")}


async def test_run_tv_pairs_phases_and_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.chdir(tmp_path)
    labels = ("happy", "sad", "calm")
    manifold_path = _save_fixtures(tmp_path, labels)
    config = load_config()
    config = dataclasses.replace(config, paths=dataclasses.replace(
        config.paths, manifold_y=tmp_path / "data" / "manifold_y.npz"))

    run_nojudge = TVRunConfig(
        out_dir=Path("results/tv_test"), manifold_path=manifold_path,
        schedule="constant", judge="none", num_segments=4, tokens_per_segment=3,
    )

    monkeypatch.setattr(tv_mod, "_method_waypoints", _fake_waypoints)
    gen_calls: list[tuple[str, list[int]]] = []

    async def fake_generate(client, base_url, model_id, layer, hook,
                            user_prompt, waypoints_full, seg_indices,
                            tokens_per_segment):
        gen_calls.append((user_prompt, list(seg_indices)))
        if user_prompt is None:
            raise RuntimeError("bad prompt")
        return f"text about {user_prompt[:10]}"

    monkeypatch.setattr(tv_mod, "generate_segmented", fake_generate)

    # Phase 1: judge="none" generates and saves completions, no result files.
    report = await run_tv_pairs(config, run_nojudge,
                                [("happy", "sad"), ("happy", "missing")])
    assert report.generated == (("happy", "sad"),)
    assert report.failed == (("happy", "missing", "missing centroid in M_h or M_y"),)
    comp_path = run_nojudge.completions_path("happy", "sad")
    assert comp_path.exists()
    assert not run_nojudge.result_path("happy", "sad").exists()
    comp = json.loads(comp_path.read_text())
    assert comp["schedule"] == "constant"
    assert comp["segment_waypoint_indices"] == [14] * 4
    assert len(comp["completions"]["pullback"]) == len(tv_mod.NEUTRAL_PROMPTS)
    # every generation used the constant schedule
    assert all(idx == [14] * 4 for _, idx in gen_calls)

    # Re-run with judge="none": nothing regenerated.
    gen_calls.clear()
    report2 = await run_tv_pairs(config, run_nojudge, [("happy", "sad")])
    assert report2.skipped == (("happy", "sad"),)
    assert not gen_calls

    # Phase 2: judge="sequential" reads saved completions, one judge call.
    judge_calls: list[int] = []

    async def fake_judge(cfg, passages, cache_path=None):
        judge_calls.append(len(passages))
        return {tid: TextRating(text_id=tid, valence=4.0, arousal=3.0)
                for tid, _ in passages}

    monkeypatch.setitem(tv_mod._JUDGE_FNS, "sequential", fake_judge)
    run_judged = TVRunConfig(
        out_dir=run_nojudge.out_dir, manifold_path=manifold_path,
        schedule="constant", judge="sequential",
        num_segments=4, tokens_per_segment=3,
    )
    report3 = await run_tv_pairs(config, run_judged, [("happy", "sad")])
    assert report3.judged == (("happy", "sad"),)
    assert judge_calls == [3 * len(tv_mod.NEUTRAL_PROMPTS)]
    result = json.loads(run_judged.result_path("happy", "sad").read_text())
    assert result["schedule"] == "constant"
    assert set(result["metrics"]) == {"pullback", "geodesic", "linear"}
    for m in result["metrics"].values():
        assert np.isfinite(m["off_my_e"]) and np.isfinite(m["my_line"])
        assert len(m["ratings_va"]) == len(tv_mod.NEUTRAL_PROMPTS)

    # Re-run judged: result exists, judge not called again.
    judge_calls.clear()
    report4 = await run_tv_pairs(config, run_judged, [("happy", "sad")])
    assert report4.skipped == (("happy", "sad"),)
    assert not judge_calls


async def test_run_tv_pairs_generation_failure_tolerated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.chdir(tmp_path)
    labels = ("happy", "sad", "calm", "tense")
    manifold_path = _save_fixtures(tmp_path, labels)
    config = load_config()
    config = dataclasses.replace(config, paths=dataclasses.replace(
        config.paths, manifold_y=tmp_path / "data" / "manifold_y.npz"))
    run = TVRunConfig(
        out_dir=Path("results/tv_test"), manifold_path=manifold_path,
        judge="none", num_segments=2, tokens_per_segment=3,
    )
    monkeypatch.setattr(tv_mod, "_method_waypoints", _fake_waypoints)

    calls = {"n": 0}

    async def fake_generate(client, base_url, model_id, layer, hook,
                            user_prompt, waypoints_full, seg_indices,
                            tokens_per_segment):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("vLLM exploded")
        return "ok"

    monkeypatch.setattr(tv_mod, "generate_segmented", fake_generate)
    report = await run_tv_pairs(config, run, [("happy", "sad"), ("calm", "tense")])
    # First pair lost one generation -> whole pair recorded failed; second fine.
    assert [(s, e) for s, e, _ in report.failed] == [("happy", "sad")]
    assert report.generated == (("calm", "tense"),)
    assert not run.completions_path("happy", "sad").exists()
    assert run.completions_path("calm", "tense").exists()
