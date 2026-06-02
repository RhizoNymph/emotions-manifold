"""Core domain types. Imported widely; keep this module dependency-light."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NewType

EmotionLabel = NewType("EmotionLabel", str)
Topic = NewType("Topic", str)


@dataclass(frozen=True, slots=True)
class Story:
    """One emotion-conditioned story used for diff-in-means extraction."""

    emotion: EmotionLabel
    topic: Topic
    story_idx: int
    text: str

    @property
    def request_id(self) -> str:
        """Stable identifier used as the capture-consumer request_id."""
        return f"{self.emotion}_{self.topic}_{self.story_idx:04d}"
