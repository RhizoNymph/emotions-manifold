"""Capture activations for the full shakedown corpus.

Sends all 1500 shakedown stories through vLLM with the filesystem
capture consumer enabled. Each request writes a (num_prompt_tokens,
hidden_size) bf16 .bin file plus a sidecar JSON under
``data/captures/{emotion_slug}/{request_id_slug}/{layer}_{hook}.bin``.

Concurrency is bumped to 32 to match the corpus generation script's
throughput. The vLLM server must be running with
``--capture-consumers filesystem:root=<data/captures>`` (see roadmap.md).

Run with:
    uv run python scripts/extract_shakedown.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from manifold_emotions.config import load_config
from manifold_emotions.extraction.capture import capture_corpus, load_corpus
from manifold_emotions.types import Story

SHAKEDOWN_CORPUS = Path("data/stories_shakedown.jsonl")


async def main() -> None:
    config = load_config()
    if not SHAKEDOWN_CORPUS.exists():
        raise SystemExit(
            f"shakedown corpus not found at {SHAKEDOWN_CORPUS}. "
            f"Run scripts/generate_shakedown.py first."
        )

    stories: list[Story] = load_corpus(SHAKEDOWN_CORPUS)
    print(f"loaded {len(stories)} shakedown stories from {SHAKEDOWN_CORPUS}")

    succeeded, failed = await capture_corpus(config, stories, concurrency=32)
    print(f"capture: {succeeded} succeeded, {failed} failed")

    if failed:
        raise SystemExit(
            f"{failed} captures failed — see structlog output above for details. "
            f"Re-running is idempotent: existing .bin files will be overwritten."
        )


if __name__ == "__main__":
    asyncio.run(main())
