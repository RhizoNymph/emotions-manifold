"""Synthetic-capture tests for the diff-in-means pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from manifold_emotions.types import EmotionLabel
from manifold_emotions.vectors.diff_in_means import (
    EmotionVectors,
    _emotion_label_from_capture_path,
    _story_centroid,
)

HIDDEN_SIZE = 8


def _bf16_bytes(arr_f32: np.ndarray) -> bytes:
    """Convert a float32 array to bf16 stored as uint16 bytes.

    bf16 is the top 16 bits of an IEEE-754 float32. We truncate (not round)
    to match what the capture consumer writes.
    """
    u32 = arr_f32.astype(np.float32).view(np.uint32)
    u16 = (u32 >> 16).astype(np.uint16)
    return u16.tobytes()


def _write_capture(root: Path, emotion: str, story_idx: int, activations: np.ndarray) -> Path:
    request_id = f"{emotion}_topic_{story_idx:04d}"
    dir_ = root / emotion / request_id
    dir_.mkdir(parents=True, exist_ok=True)
    bin_path = dir_ / "40_post_mlp.bin"
    bin_path.write_bytes(_bf16_bytes(activations))
    sidecar = {
        "request_id": request_id,
        "layer": 40,
        "hook": "post_mlp",
        "shape": list(activations.shape),
        "dtype": "bfloat16",
    }
    bin_path.with_suffix(".json").write_text(json.dumps(sidecar))
    return bin_path


def test_story_centroid_skips_leading_tokens() -> None:
    activations = np.zeros((100, HIDDEN_SIZE), dtype=np.float32)
    activations[50:, 0] = 2.0  # only the post-skip tokens have signal in dim 0
    centroid = _story_centroid(activations, skip_tokens_before=50)
    assert centroid[0] == pytest.approx(2.0)
    assert centroid[1] == 0.0


def test_story_centroid_short_story_fallback() -> None:
    activations = np.full((30, HIDDEN_SIZE), 3.0, dtype=np.float32)
    centroid = _story_centroid(activations, skip_tokens_before=50)
    # Falls back to the full-story mean rather than producing NaN.
    assert centroid[0] == pytest.approx(3.0)


def test_emotion_label_round_trips_through_capture_path(tmp_path: Path) -> None:
    bin_path = _write_capture(tmp_path, "happy", 0, np.zeros((10, HIDDEN_SIZE), dtype=np.float32))
    label = _emotion_label_from_capture_path(bin_path, tmp_path)
    assert label == EmotionLabel("happy")


def test_compute_emotion_vectors_recovers_diff_in_means(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    # Build a synthetic capture tree:
    # - "happy" stories: signal +1.0 in dim 0 past index 50
    # - "sad"   stories: signal -1.0 in dim 0 past index 50
    # - all stories: random noise in other dims (averages out)
    rng = np.random.default_rng(seed=42)

    for emotion, dim0_value in [("happy", 1.0), ("sad", -1.0)]:
        for i in range(10):
            act = rng.normal(0, 0.01, size=(80, HIDDEN_SIZE)).astype(np.float32)
            act[50:, 0] = dim0_value
            _write_capture(tmp_path, emotion, i, act)

    # Patch the config so capture root points at tmp_path.
    from manifold_emotions import config as cfg_mod

    cfg = cfg_mod.load_config()
    patched = cfg_mod.Config(
        model=cfg.model,
        vllm_server=cfg.vllm_server,
        capture=cfg_mod.CaptureConfig(
            root=tmp_path,
            writer_threads=cfg.capture.writer_threads,
            queue_size=cfg.capture.queue_size,
        ),
        judge=cfg.judge,
        corpus=cfg.corpus,
        extraction=cfg.extraction,
        manifold=cfg.manifold,
        paths=cfg.paths,
    )

    from manifold_emotions.vectors.diff_in_means import compute_emotion_vectors

    result = compute_emotion_vectors(patched)

    assert result.labels == (EmotionLabel("happy"), EmotionLabel("sad"))
    assert result.vectors.shape == (2, HIDDEN_SIZE)
    assert result.story_counts.tolist() == [10, 10]

    # Diff-in-means should put happy at +1 / sad at -1 on dim 0 (global mean is 0).
    happy_vec = result.vectors[0]
    sad_vec = result.vectors[1]
    assert happy_vec[0] == pytest.approx(1.0, abs=0.02)
    assert sad_vec[0] == pytest.approx(-1.0, abs=0.02)
    # Other dims should be near zero — random noise averages out.
    assert np.abs(happy_vec[1:]).max() < 0.05
    assert np.abs(sad_vec[1:]).max() < 0.05


def test_emotion_vectors_round_trip(tmp_path: Path) -> None:
    original = EmotionVectors(
        labels=(EmotionLabel("happy"), EmotionLabel("sad")),
        vectors=np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32),
        centroids=np.array([[1.0, 0.5], [-1.0, 0.5]], dtype=np.float32),
        global_mean=np.array([0.0, 0.5], dtype=np.float32),
        story_counts=np.array([10, 10], dtype=np.int64),
        hidden_size=2,
        skip_tokens_before=50,
    )
    path = tmp_path / "ev.npz"
    original.save(path)
    loaded = EmotionVectors.load(path)
    assert loaded.labels == original.labels
    np.testing.assert_array_equal(loaded.vectors, original.vectors)
    np.testing.assert_array_equal(loaded.centroids, original.centroids)
    np.testing.assert_array_equal(loaded.global_mean, original.global_mean)
    assert loaded.hidden_size == original.hidden_size
    assert loaded.skip_tokens_before == original.skip_tokens_before
