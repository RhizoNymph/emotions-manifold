"""Companion to validate_batched_judge.py — runs the SEQUENTIAL judge
on the same completions/paths the batched chain consumed, then rebuilds
each pair's summary using the same metric helpers as the batched
orchestrator. Output goes to results/<chain>_seq/<pair>.json so we
can diff it against results/<chain>/<pair>.json (the batched output).

Only used for end-to-end Layer 2 validation. Not part of the production
pipeline.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from manifold_emotions.behavior.judge_text import judge_texts
from manifold_emotions.behavior.manifold import BehaviorManifold
from manifold_emotions.config import load_config

from run_chain_batched_judge import _rebuild_summary_for_pair  # type: ignore


async def main() -> None:
    if len(sys.argv) < 2:
        print("usage: validate_phase2_sequential.py <chain_name>")
        print("  e.g. validate_phase2_sequential.py validate_chain_seq")
        sys.exit(1)
    chain = sys.argv[1]

    chain_data_dir = Path(f"data/{chain}")
    chain_results_dir = Path(f"results/{chain}")

    config = load_config()
    behavior = BehaviorManifold.load(Path(config.paths.manifold_y))

    pair_names = sorted(
        p.stem.removeprefix("completions_")
        for p in chain_data_dir.glob("completions_*.json")
    )
    print(f"Sequential judging for chain={chain}, {len(pair_names)} pairs")

    for pair_name in pair_names:
        completions_path = chain_data_dir / f"completions_{pair_name}.json"
        ratings_path = chain_data_dir / f"ratings_{pair_name}.json"
        records = json.loads(completions_path.read_text())
        passages = [(r["text_id"], r["text"]) for r in records]
        print(f"  {pair_name}: {len(passages)} passages...")
        await judge_texts(config, passages, cache_path=ratings_path)

        ok = _rebuild_summary_for_pair(
            pair_name, chain_results_dir, chain_data_dir, behavior,
        )
        print(f"    summary rewritten: {ok}")


if __name__ == "__main__":
    asyncio.run(main())
