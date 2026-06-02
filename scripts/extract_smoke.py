"""Run the capture pipeline against the smoke corpus.

Sends each of the 10 smoke stories through vLLM with the filesystem
capture consumer enabled and verifies that .bin + sidecar JSON files
appear on disk with the expected shapes.

Run with:
    uv run python scripts/extract_smoke.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from manifold_emotions.config import load_config
from manifold_emotions.extraction.capture import capture_corpus, load_corpus
from manifold_emotions.extraction.loader import iter_captures, load_activation

SMOKE_CORPUS = Path("data/stories_smoke.jsonl")


async def main() -> None:
    config = load_config()
    stories = load_corpus(SMOKE_CORPUS)
    print(f"loaded {len(stories)} smoke stories from {SMOKE_CORPUS}")

    succeeded, failed = await capture_corpus(config, stories, concurrency=4)
    print(f"capture: {succeeded} succeeded, {failed} failed")

    if succeeded == 0:
        print("no captures written; nothing to inspect")
        return

    captures = iter_captures(config.capture.root)
    print(f"\nfound {len(captures)} .bin files under {config.capture.root}")
    for path in captures[:3]:
        act = load_activation(path)
        print(
            f"  {path.relative_to(config.capture.root)} "
            f"layer={act.layer} hook={act.hook} "
            f"shape={act.activations.shape} dtype={act.activations.dtype} "
            f"mean={act.activations.mean():.4f} std={act.activations.std():.4f}"
        )


if __name__ == "__main__":
    asyncio.run(main())
