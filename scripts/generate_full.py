"""Full corpus generation: all 171 emotions × 10 topics × 5 stories = 8,550.

Matches the Anthropic protocol's structure more closely than the
30-emotion shakedown: 10 diverse topics per emotion so the residual
emotional signal generalizes across context. With ~5 stories per
(emotion, topic) pair we get 50 stories per emotion centroid — same
sample budget as the shakedown, but spread across topics.

Wall-clock estimate: ~3 hours on a single vLLM with concurrency=32.

Run with:
    uv run python scripts/generate_full.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from manifold_emotions.config import load_config
from manifold_emotions.corpus.generate import generate_corpus
from manifold_emotions.corpus.topics import FULL_TOPICS
from manifold_emotions.types import EmotionLabel, Topic

EMOTION_WORDS_PATH = Path("data/emotion_words.txt")
FULL_OUTPUT = Path("data/stories_full.jsonl")

STORIES_PER_PAIR = 5  # 171 × 10 × 5 = 8550 stories


def load_full_emotions() -> list[EmotionLabel]:
    """Read the canonical 171 emotion words from data/emotion_words.txt.

    Each line is one emotion word; blank lines and comments (#) skipped.
    """
    if not EMOTION_WORDS_PATH.exists():
        raise SystemExit(f"emotion word list missing at {EMOTION_WORDS_PATH}")
    words: list[EmotionLabel] = []
    for raw in EMOTION_WORDS_PATH.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        words.append(EmotionLabel(line))
    return words


async def main() -> None:
    config = load_config()
    emotions = load_full_emotions()
    topics = [Topic(t) for t in FULL_TOPICS]
    total = len(emotions) * len(topics) * STORIES_PER_PAIR

    print(
        f"corpus: {len(emotions)} emotions × {len(topics)} topics × "
        f"{STORIES_PER_PAIR} stories/pair = {total} total"
    )
    print(f"writing to {FULL_OUTPUT}")
    print(f"vLLM at {config.vllm_server.base_url}")

    if FULL_OUTPUT.exists():
        # Resumable: keep existing stories if the file matches the same
        # (emotion × topic) layout. We'd need a different policy if the
        # parameters changed — for now, fail loudly so the user notices.
        existing = sum(1 for _ in FULL_OUTPUT.open())
        print(f"output exists with {existing} stories — keeping; "
              f"generate_corpus is idempotent on request_id")

    written, rejected = await generate_corpus(
        config=config,
        emotions=emotions,
        topics=topics,
        stories_per_pair=STORIES_PER_PAIR,
        output_path=FULL_OUTPUT,
        concurrency=32,
    )
    print(f"wrote {written} stories to {FULL_OUTPUT} ({rejected} rejected)")


if __name__ == "__main__":
    asyncio.run(main())
