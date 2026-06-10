"""Chain runner: pair splitting, resume detection, multi-host dispatch."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import manifold_emotions.experiments.chain as chain_mod
from manifold_emotions.config import load_config
from manifold_emotions.experiments.chain import (
    pair_is_complete,
    run_chain,
    split_pairs,
)
from manifold_emotions.experiments.chord import ChordRunConfig


def test_split_pairs_stripes_and_covers_everything() -> None:
    pairs = [(f"a{i}", f"b{i}") for i in range(7)]
    shares = split_pairs(pairs, 2)
    assert shares[0] == pairs[0::2]
    assert shares[1] == pairs[1::2]
    assert sorted(shares[0] + shares[1]) == sorted(pairs)
    assert abs(len(shares[0]) - len(shares[1])) <= 1

    assert split_pairs(pairs, 1) == [pairs]
    with pytest.raises(ValueError, match="num_workers"):
        split_pairs(pairs, 0)


def _run_cfg(judge: str) -> ChordRunConfig:
    return ChordRunConfig(
        name="pullback_test", manifold_path=Path("data/m.npz"), judge=judge,
    )


def _write_summary(run: ChordRunConfig, start: str, end: str, off: float) -> None:
    run.results_dir.mkdir(parents=True, exist_ok=True)
    (run.results_dir / f"{start}_{end}.json").write_text(json.dumps({
        "trajectories": {
            m: {"off_manifold_energy": off, "my_geodesic_distance": off}
            for m in ("pullback", "geodesic", "linear")
        },
    }))


def test_pair_is_complete_judged_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    run = _run_cfg("sequential")

    assert not pair_is_complete(run, "happy", "sad")
    _write_summary(run, "happy", "sad", off=0.4)
    assert pair_is_complete(run, "happy", "sad")

    # A NaN skeleton (left by a phase-1 nojudge run) is NOT complete
    # for a judged variant.
    _write_summary(run, "calm", "ecstatic", off=float("nan"))
    assert not pair_is_complete(run, "calm", "ecstatic")


def test_pair_is_complete_nojudge_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    run = _run_cfg("none")

    _write_summary(run, "happy", "sad", off=float("nan"))
    assert not pair_is_complete(run, "happy", "sad")  # completions missing
    run.data_dir.mkdir(parents=True, exist_ok=True)
    (run.data_dir / "completions_happy_sad.json").write_text("[]")
    assert pair_is_complete(run, "happy", "sad")


async def test_run_chain_dispatches_resumes_and_tolerates_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    config = load_config()
    monkeypatch.chdir(tmp_path)
    run = _run_cfg("sequential")

    # One pair is already complete on disk: must be skipped, not re-run.
    _write_summary(run, "done", "already", off=0.3)

    calls: list[tuple[str, str, str]] = []  # (host, start, end)

    async def fake_run_chord_pair(cfg, run_cfg, start, end, **kwargs):
        calls.append((cfg.vllm_server.base_url, start, end))
        if start == "boom":
            raise RuntimeError("vLLM exploded")
        _write_summary(run_cfg, start, end, off=0.5)
        return run_cfg.results_dir / f"{start}_{end}.json"

    monkeypatch.setattr(chain_mod, "run_chord_pair", fake_run_chord_pair)

    pairs = [
        ("done", "already"),
        ("happy", "sad"),
        ("boom", "bust"),
        ("calm", "ecstatic"),
    ]
    hosts = ["http://h0:8000/v1", "http://h1:8000/v1"]
    report = await run_chain(config, run, pairs, hosts)

    assert report.skipped == (("done", "already"),)
    assert sorted(report.completed) == [("calm", "ecstatic"), ("happy", "sad")]
    assert report.failed == (("boom", "bust", "RuntimeError: vLLM exploded"),)
    assert not report.ok

    # The skipped pair never reached a worker; the rest were striped
    # across both hosts.
    assert len(calls) == 3
    assert {c[0] for c in calls} == set(hosts)
    assert ("done", "already") not in {(c[1], c[2]) for c in calls}

    # Re-running resumes: the failed pair is the only one re-attempted.
    calls.clear()
    report2 = await run_chain(config, run, pairs, hosts)
    assert [(c[1], c[2]) for c in calls] == [("boom", "bust")]
    assert len(report2.skipped) == 3
