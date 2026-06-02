"""Shakedown corpus generation: 30 emotions × 1 topic × 50 stories = 1500 total.

The 30 emotions are picked to span the affective circumplex: 10 positive
high-arousal, 10 negative high-arousal, 5 positive low-arousal, 5
negative low-arousal, chosen from the full 171-word list.

Run with:
    uv run python scripts/generate_shakedown.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from manifold_emotions.config import load_config
from manifold_emotions.corpus.generate import generate_corpus
from manifold_emotions.corpus.topics import SHAKEDOWN_TOPIC
from manifold_emotions.types import EmotionLabel, Topic

# 30 emotions spanning the valence × arousal plane.
SHAKEDOWN_EMOTIONS = [
    EmotionLabel(w)
    for w in (
        # positive high-arousal
        "happy",
        "joyful",
        "ecstatic",
        "excited",
        "thrilled",
        "elated",
        "euphoric",
        "enthusiastic",
        "energized",
        "delighted",
        # negative high-arousal
        "angry",
        "furious",
        "outraged",
        "enraged",
        "panicked",
        "terrified",
        "horrified",
        "desperate",
        "hostile",
        "frustrated",
        # positive low-arousal
        "calm",
        "peaceful",
        "serene",
        "content",
        "relaxed",
        # negative low-arousal
        "sad",
        "gloomy",
        "melancholy",
        "weary",
        "depressed",
    )
]

SHAKEDOWN_OUTPUT = Path("data/stories_shakedown.jsonl")


async def main() -> None:
    config = load_config()
    if SHAKEDOWN_OUTPUT.exists():
        SHAKEDOWN_OUTPUT.unlink()

    written, rejected = await generate_corpus(
        config=config,
        emotions=SHAKEDOWN_EMOTIONS,
        topics=[Topic(SHAKEDOWN_TOPIC)],
        stories_per_pair=config.corpus.shakedown.stories_per_emotion,
        output_path=SHAKEDOWN_OUTPUT,
        concurrency=32,
    )
    print(f"wrote {written} stories to {SHAKEDOWN_OUTPUT} ({rejected} rejected)")


if __name__ == "__main__":
    asyncio.run(main())
