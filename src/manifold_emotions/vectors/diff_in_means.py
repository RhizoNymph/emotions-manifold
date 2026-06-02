"""Diff-in-means emotion vector extraction from captured residual-stream activations.

For each story we have a captured tensor of shape (num_tokens, hidden_size).
Per Anthropic, we:
1. Skip the first `skip_tokens_before` tokens of each story (emotional content
   may not be apparent in the opening).
2. Mean across remaining token positions to get one (hidden_size,) per story.
3. Mean across stories per emotion to get a per-emotion centroid.
4. Subtract the global mean centroid (mean across all per-emotion centroids).

The result is a (num_emotions, hidden_size) array of emotion vectors,
plus the parallel emotion-label index.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import structlog

from ..config import Config
from ..errors import CaptureError
from ..extraction.loader import iter_captures, load_activation
from ..types import EmotionLabel

log = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class EmotionVectors:
    """The output of diff-in-means extraction.

    `vectors[i]` is the emotion vector for `labels[i]`.
    `centroids[i]` is the raw per-emotion centroid before diff-in-means.
    `global_mean` is the mean across the per-emotion centroids.
    `story_counts[i]` is how many stories contributed to `labels[i]`.
    """

    labels: tuple[EmotionLabel, ...]
    vectors: np.ndarray  # (num_emotions, hidden_size)
    centroids: np.ndarray  # (num_emotions, hidden_size)
    global_mean: np.ndarray  # (hidden_size,)
    story_counts: np.ndarray  # (num_emotions,)
    hidden_size: int
    skip_tokens_before: int

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            labels=np.array(self.labels, dtype=object),
            vectors=self.vectors,
            centroids=self.centroids,
            global_mean=self.global_mean,
            story_counts=self.story_counts,
            hidden_size=np.array([self.hidden_size]),
            skip_tokens_before=np.array([self.skip_tokens_before]),
        )

    @classmethod
    def load(cls, path: Path) -> EmotionVectors:
        with np.load(path, allow_pickle=True) as data:
            labels = tuple(EmotionLabel(str(x)) for x in data["labels"])
            return cls(
                labels=labels,
                vectors=data["vectors"],
                centroids=data["centroids"],
                global_mean=data["global_mean"],
                story_counts=data["story_counts"],
                hidden_size=int(data["hidden_size"][0]),
                skip_tokens_before=int(data["skip_tokens_before"][0]),
            )


def _emotion_label_from_capture_path(path: Path, capture_root: Path) -> EmotionLabel:
    """The capture consumer writes to {root}/{tag_slug}/{request_id_slug}/...

    The tag we passed in is the emotion label. Slugification only replaces
    characters outside [a-zA-Z0-9._-] with _, and our emotion labels are
    all lowercase alphabetics, so the tag round-trips cleanly.
    """
    rel = path.relative_to(capture_root)
    return EmotionLabel(rel.parts[0])


def _story_centroid(activations: np.ndarray, skip_tokens_before: int) -> np.ndarray:
    """Mean across token positions past the skip threshold."""
    num_tokens = activations.shape[0]
    if num_tokens <= skip_tokens_before:
        # Short story — fall back to averaging whatever we have rather than dropping it.
        return activations.mean(axis=0)
    return activations[skip_tokens_before:].mean(axis=0)


def compute_emotion_vectors(config: Config) -> EmotionVectors:
    """Walk the capture directory, accumulate per-story centroids, diff against global mean."""
    capture_root = config.capture.root
    skip = config.extraction.skip_tokens_before

    paths = iter_captures(capture_root)
    if not paths:
        raise CaptureError(f"no .bin captures found under {capture_root}")

    log.info("vectors.compute.start", num_captures=len(paths), capture_root=str(capture_root))

    per_emotion: dict[EmotionLabel, list[np.ndarray]] = defaultdict(list)
    hidden_size: int | None = None

    for path in paths:
        emotion = _emotion_label_from_capture_path(path, capture_root)
        act = load_activation(path)
        if hidden_size is None:
            hidden_size = act.activations.shape[1]
        elif act.activations.shape[1] != hidden_size:
            raise CaptureError(
                f"hidden size mismatch: {path} has {act.activations.shape[1]}, "
                f"earlier captures had {hidden_size}"
            )
        per_emotion[emotion].append(_story_centroid(act.activations, skip))

    assert hidden_size is not None  # guaranteed by the empty-paths check above

    labels = tuple(sorted(per_emotion.keys()))
    centroids = np.zeros((len(labels), hidden_size), dtype=np.float32)
    story_counts = np.zeros(len(labels), dtype=np.int64)
    for i, label in enumerate(labels):
        stack = np.stack(per_emotion[label], axis=0)
        centroids[i] = stack.mean(axis=0)
        story_counts[i] = stack.shape[0]

    global_mean = centroids.mean(axis=0)
    vectors = centroids - global_mean[None, :]

    log.info(
        "vectors.compute.done",
        num_emotions=len(labels),
        hidden_size=hidden_size,
        total_stories=int(story_counts.sum()),
        min_stories_per_emotion=int(story_counts.min()),
        max_stories_per_emotion=int(story_counts.max()),
    )

    return EmotionVectors(
        labels=labels,
        vectors=vectors,
        centroids=centroids,
        global_mean=global_mean,
        story_counts=story_counts,
        hidden_size=hidden_size,
        skip_tokens_before=skip,
    )
