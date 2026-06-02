"""Config loading sanity tests. No network access required."""

from __future__ import annotations

from pathlib import Path

import pytest

from manifold_emotions.config import load_config, load_emotion_words
from manifold_emotions.errors import ConfigError


def test_load_config_succeeds_with_anthropic_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    config = load_config()
    assert config.model.target_layer == 40
    assert config.model.num_layers == 60
    assert config.model.hook_point == "post_mlp"
    assert config.vllm_server.max_model_len == 4096
    assert config.judge.api_key == "test-key"


def test_load_config_fails_without_anthropic_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # Block dotenv from re-injecting from a real .env file.
    monkeypatch.setattr("manifold_emotions.config._ENV_PATH", Path("/nonexistent"))
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        load_config()


def test_emotion_words_file_has_171_entries() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    words = load_emotion_words(repo_root / "data" / "emotion_words.txt")
    assert len(words) == 171
    # Spot-check a few well-known emotions from the Anthropic list.
    assert "happy" in words
    assert "sad" in words
    assert "desperate" in words
    assert "blissful" in words
    assert "hostile" in words


def test_story_request_id_is_stable() -> None:
    from manifold_emotions.types import EmotionLabel, Story, Topic

    s = Story(
        emotion=EmotionLabel("happy"),
        topic=Topic("a chance encounter"),
        story_idx=3,
        text="example",
    )
    assert s.request_id == "happy_a chance encounter_0003"
