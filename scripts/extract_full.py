"""Capture activations for the full 171-emotion corpus.

Sends all stories from ``data/stories_full.jsonl`` through vLLM with the
filesystem capture consumer enabled. Each story produces one bf16 .bin
under ``data/captures/{emotion_slug}/{request_id_slug}/{layer}_{hook}.bin``.

Concurrency=32 to match generation. Idempotent: existing captures are
overwritten, so re-running after a partial failure is safe.

Run with:
    uv run python scripts/extract_full.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from manifold_emotions.config import load_config
from manifold_emotions.extraction.capture import capture_corpus, load_corpus
from manifold_emotions.types import Story

FULL_CORPUS = Path("data/stories_full.jsonl")


async def main() -> None:
    config = load_config()
    if not FULL_CORPUS.exists():
        raise SystemExit(
            f"full corpus not found at {FULL_CORPUS}. "
            f"Run scripts/generate_full.py first."
        )

    stories: list[Story] = load_corpus(FULL_CORPUS)
    print(f"loaded {len(stories)} stories from {FULL_CORPUS}")
    print(f"vLLM at {config.vllm_server.base_url}")
    print(f"captures root: {config.capture.root}")

    succeeded, failed = await capture_corpus(config, stories, concurrency=32)
    print(f"capture: {succeeded} succeeded, {failed} failed")

    if failed:
        raise SystemExit(
            f"{failed} captures failed — see structlog output above for details. "
            f"Re-running is idempotent: existing .bin files will be overwritten."
        )


if __name__ == "__main__":
    asyncio.run(main())
