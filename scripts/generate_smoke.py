"""Smoke test for the corpus generation pipeline.

Generates 5 stories each for 2 emotions ("happy", "sad") with a single
neutral topic. Use this to verify the vLLM endpoint is reachable and the
prompt template produces sensible output before kicking off the shakedown
or full corpus runs.

Run with:
    uv run python scripts/generate_smoke.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from manifold_emotions.config import load_config
from manifold_emotions.corpus.generate import generate_corpus
from manifold_emotions.corpus.topics import SHAKEDOWN_TOPIC
from manifold_emotions.types import EmotionLabel, Topic

SMOKE_EMOTIONS = [EmotionLabel("happy"), EmotionLabel("sad")]
SMOKE_STORIES_PER_PAIR = 5
SMOKE_OUTPUT = Path("data/stories_smoke.jsonl")


async def main() -> None:
    config = load_config()
    if SMOKE_OUTPUT.exists():
        SMOKE_OUTPUT.unlink()

    written, rejected = await generate_corpus(
        config=config,
        emotions=SMOKE_EMOTIONS,
        topics=[Topic(SHAKEDOWN_TOPIC)],
        stories_per_pair=SMOKE_STORIES_PER_PAIR,
        output_path=SMOKE_OUTPUT,
        concurrency=4,
    )
    print(f"wrote {written} stories to {SMOKE_OUTPUT} ({rejected} rejected)")


if __name__ == "__main__":
    asyncio.run(main())
