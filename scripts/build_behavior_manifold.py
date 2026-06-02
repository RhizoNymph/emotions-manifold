"""Build the behavior manifold M_y from the story corpus.

Loads a JSONL corpus (default: ``data/stories_shakedown.jsonl``),
judges each story's text content for (valence, arousal) via the Claude
judge with per-corpus rating cache, aggregates per emotion, and writes
the resulting BehaviorManifold to ``config.paths.manifold_y``.

Run with:
    uv run python scripts/build_behavior_manifold.py
    uv run python scripts/build_behavior_manifold.py --corpus data/stories_full.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from manifold_emotions.behavior.judge_text import judge_texts
from manifold_emotions.behavior.manifold import aggregate_behavior_manifold
from manifold_emotions.config import load_config
from manifold_emotions.extraction.capture import load_corpus

DEFAULT_CORPUS = Path("data/stories_shakedown.jsonl")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    args = parser.parse_args()

    config = load_config()
    if not args.corpus.exists():
        raise SystemExit(f"corpus not found at {args.corpus}")

    # Separate rating cache per corpus stem so shakedown and full
    # ratings don't co-mingle. The shakedown corpus keeps its legacy
    # data/story_ratings.json so we don't have to re-judge 1500 stories.
    if args.corpus == DEFAULT_CORPUS:
        RATINGS_CACHE = Path("data/story_ratings.json")
    else:
        RATINGS_CACHE = Path(f"data/story_ratings_{args.corpus.stem}.json")

    stories = load_corpus(args.corpus)
    print(f"loaded {len(stories)} stories from {args.corpus}")

    passages = [(s.request_id, s.text) for s in stories]
    text_id_to_emotion = {s.request_id: s.emotion for s in stories}

    ratings = await judge_texts(
        config=config,
        passages=passages,
        cache_path=RATINGS_CACHE,
    )

    manifold = aggregate_behavior_manifold(text_id_to_emotion, ratings)
    manifold.save(config.paths.manifold_y)

    print(
        f"saved behavior manifold to {config.paths.manifold_y}: "
        f"{len(manifold.labels)} emotion centroids"
    )
    print(
        f"per-emotion story counts: "
        f"min={int(manifold.story_counts.min())} "
        f"max={int(manifold.story_counts.max())}"
    )
    print()
    print("Per-emotion (valence, arousal) centroids:")
    for i, label in enumerate(manifold.labels):
        v, a = manifold.centroids[i]
        vs, as_ = manifold.stds[i]
        print(f"  {label:>20s}: V={v:.2f}±{vs:.2f}  A={a:.2f}±{as_:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
