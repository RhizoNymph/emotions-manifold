"""Behavior manifold M_y: per-emotion (valence, arousal) centroid.

For each emotion label, aggregate the judge's (valence, arousal)
ratings across all stories belonging to that emotion in the corpus.
The behavior manifold is the resulting (num_emotions, 2) point cloud,
matched 1:1 with M_h's labels.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .judge_text import TextRating


@dataclass(frozen=True, slots=True)
class BehaviorManifold:
    """Per-emotion centroid in (valence, arousal) space.

    ``centroids[i]`` is the (valence, arousal) mean for ``labels[i]``.
    ``stds[i]`` is the per-dimension standard deviation across stories
    of that emotion — useful for diagnostic plots and for assessing
    whether the per-emotion behavior is tight or diffuse.
    """

    labels: tuple[str, ...]
    centroids: np.ndarray  # (num_emotions, 2)  — [valence, arousal]
    stds: np.ndarray  # (num_emotions, 2)
    story_counts: np.ndarray  # (num_emotions,)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            labels=np.array(self.labels, dtype=object),
            centroids=self.centroids,
            stds=self.stds,
            story_counts=self.story_counts,
        )

    @classmethod
    def load(cls, path: Path) -> BehaviorManifold:
        with np.load(path, allow_pickle=True) as data:
            return cls(
                labels=tuple(str(x) for x in data["labels"]),
                centroids=data["centroids"],
                stds=data["stds"],
                story_counts=data["story_counts"],
            )


def aggregate_behavior_manifold(
    text_id_to_emotion: dict[str, str],
    ratings: dict[str, TextRating],
) -> BehaviorManifold:
    """Build a BehaviorManifold from per-story ratings + an emotion mapping.

    ``text_id_to_emotion`` maps Story.request_id to the emotion label
    that conditioned that story. ``ratings`` is the output of
    ``judge_texts``. Emotions absent from either input are silently
    dropped — they won't contribute centroids.
    """
    per_emotion: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for text_id, rating in ratings.items():
        emotion = text_id_to_emotion.get(text_id)
        if emotion is None:
            continue
        per_emotion[emotion].append((rating.valence, rating.arousal))

    labels = tuple(sorted(per_emotion.keys()))
    centroids = np.zeros((len(labels), 2), dtype=np.float32)
    stds = np.zeros((len(labels), 2), dtype=np.float32)
    counts = np.zeros(len(labels), dtype=np.int64)
    for i, label in enumerate(labels):
        arr = np.array(per_emotion[label], dtype=np.float32)
        centroids[i] = arr.mean(axis=0)
        stds[i] = arr.std(axis=0)
        counts[i] = arr.shape[0]

    return BehaviorManifold(
        labels=labels,
        centroids=centroids,
        stds=stds,
        story_counts=counts,
    )
