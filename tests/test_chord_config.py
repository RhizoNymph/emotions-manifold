"""ChordRunConfig YAML loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from manifold_emotions.experiments.chord import NEUTRAL_PROMPTS, ChordRunConfig

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_from_yaml_minimal_uses_defaults(tmp_path: Path) -> None:
    cfg_path = tmp_path / "chord.yaml"
    cfg_path.write_text("name: pullback_test\nmanifold: data/m.npz\n")
    cfg = ChordRunConfig.from_yaml(cfg_path)
    assert cfg.name == "pullback_test"
    assert cfg.manifold_path == Path("data/m.npz")
    assert cfg.judge == "sequential"
    assert cfg.num_waypoints == 30
    assert cfg.num_prompts == 10
    assert cfg.steering_scale == 8.0
    assert cfg.sigma is None
    assert cfg.prompts == NEUTRAL_PROMPTS
    assert cfg.summary_extra == {}
    assert cfg.data_dir == Path("data/pullback_test")
    assert cfg.results_dir == Path("results/pullback_test")


def test_from_yaml_full_override(tmp_path: Path) -> None:
    cfg_path = tmp_path / "chord.yaml"
    cfg_path.write_text(
        "name: pullback_x\n"
        "manifold: data/m.npz\n"
        "judge: none\n"
        "num_waypoints: 12\n"
        "num_prompts: 3\n"
        "max_tokens: 64\n"
        "concurrency: 4\n"
        "steering_scale: 4.0\n"
        "sigma: 0.5\n"
        "prompts: ['a', 'b']\n"
        "summary_extra:\n"
        "  bandwidth_heuristic: silverman\n"
    )
    cfg = ChordRunConfig.from_yaml(cfg_path)
    assert cfg.judge == "none"
    assert cfg.num_waypoints == 12
    assert cfg.num_prompts == 3
    assert cfg.max_tokens == 64
    assert cfg.concurrency == 4
    assert cfg.steering_scale == 4.0
    assert cfg.sigma == 0.5
    assert cfg.prompts == ("a", "b")
    assert cfg.summary_extra == {"bandwidth_heuristic": "silverman"}


def test_from_yaml_rejects_unknown_keys(tmp_path: Path) -> None:
    cfg_path = tmp_path / "chord.yaml"
    cfg_path.write_text("name: x\nmanifold: m.npz\nmanifold_dim: 4\n")
    with pytest.raises(ValueError, match="unknown keys.*manifold_dim"):
        ChordRunConfig.from_yaml(cfg_path)


def test_from_yaml_rejects_bad_judge_mode(tmp_path: Path) -> None:
    cfg_path = tmp_path / "chord.yaml"
    cfg_path.write_text("name: x\nmanifold: m.npz\njudge: maybe\n")
    with pytest.raises(ValueError, match="invalid judge mode"):
        ChordRunConfig.from_yaml(cfg_path)


def test_all_shipped_experiment_configs_parse() -> None:
    shipped = sorted((REPO_ROOT / "experiments").glob("chord_*.yaml"))
    assert len(shipped) >= 4  # 8d, 4d, 6d, 8d_silverman
    names = set()
    for path in shipped:
        cfg = ChordRunConfig.from_yaml(path)
        names.add(cfg.name)
    assert len(names) == len(shipped)  # distinct output dirs — no clobbering


def test_alift_n40_pair_file_is_well_formed() -> None:
    import json

    pairs = json.loads(
        (REPO_ROOT / "experiments/pairs/alift_n40.json").read_text()
    )
    assert len(pairs) == 40
    assert all(len(p) == 2 for p in pairs)
    assert len({tuple(p) for p in pairs}) == 40
    # Multi-word labels carry literal spaces (underscore form is a
    # shell-chain artifact and must not leak into the pair file).
    flat = [label for pair in pairs for label in pair]
    assert "at ease" in flat
    assert not any("_" in label for label in flat)
