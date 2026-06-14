"""Validate that sequential and batched judging are equivalent.

Some results in the project were rated through the synchronous Messages
API (``judge_texts``) and others through the asynchronous Batches API
(``judge_texts_batched``) after a mid-project credit outage. The two
share the judge model, prompt template, response parser, and temperature
(0.0) — the batched module imports the prompt and parser from the
sequential one — so they should agree. This script proves it empirically:
it judges the *same* saved completions through both paths (no cache, so
both hit the API) and reports per-passage V/A agreement.

The number that matters is the systematic *bias* (mean signed difference);
a mixed-pipeline comparison only suffers if the two paths shift ratings
in a consistent direction. The per-passage |Δ| is temperature-0 residual
nondeterminism present between any two judge passes.

    uv run python scripts/validation/judge_pipeline_equivalence.py \\
        --completions data/pullback_2d/completions_happy_sad.json -n 120
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np

from manifold_emotions.behavior.judge_text import judge_texts
from manifold_emotions.behavior.judge_text_batched import judge_texts_batched
from manifold_emotions.config import load_config


async def run(completions_path: Path, n: int, out_path: Path | None) -> dict:
    config = load_config()
    comps = json.loads(completions_path.read_text())
    # Deterministic stride so the sample spans methods/waypoints/prompts.
    comps_sorted = sorted(comps, key=lambda c: c["text_id"])
    stride = max(1, len(comps_sorted) // n)
    sample = comps_sorted[::stride][:n]
    passages = [(c["text_id"], c["text"]) for c in sample]
    print(f"judging {len(passages)} passages both ways (temperature 0.0)")

    seq = await judge_texts(config, passages, cache_path=None)
    bat = await judge_texts_batched(config, passages, cache_path=None)

    common = [tid for tid, _ in passages if tid in seq and tid in bat]
    if not common:
        raise SystemExit("no passages rated by both paths")
    dv = np.array([seq[t].valence - bat[t].valence for t in common])
    da = np.array([seq[t].arousal - bat[t].arousal for t in common])
    exact = sum(1 for t in common
                if seq[t].valence == bat[t].valence
                and seq[t].arousal == bat[t].arousal)

    result = {
        "n_compared": len(common),
        "valence_mean_abs_diff": float(np.abs(dv).mean()),
        "valence_max_abs_diff": float(np.abs(dv).max()),
        "valence_bias_seq_minus_bat": float(dv.mean()),
        "arousal_mean_abs_diff": float(np.abs(da).mean()),
        "arousal_max_abs_diff": float(np.abs(da).max()),
        "arousal_bias_seq_minus_bat": float(da.mean()),
        "exact_match_frac": exact / len(common),
    }
    print(f"\nn compared: {result['n_compared']}")
    print(f"valence  |Δ|: mean={result['valence_mean_abs_diff']:.4f}  "
          f"max={result['valence_max_abs_diff']:.3f}  "
          f"bias(seq−bat)={result['valence_bias_seq_minus_bat']:+.4f}")
    print(f"arousal  |Δ|: mean={result['arousal_mean_abs_diff']:.4f}  "
          f"max={result['arousal_max_abs_diff']:.3f}  "
          f"bias(seq−bat)={result['arousal_bias_seq_minus_bat']:+.4f}")
    print(f"exact-match passages: {exact}/{len(common)} "
          f"({100 * result['exact_match_frac']:.0f}%)")

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        print(f"\nsaved {out_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--completions", type=Path,
                        default=Path("data/pullback_2d/completions_happy_sad.json"),
                        help="a chord completions_*.json file to sample from")
    parser.add_argument("-n", type=int, default=120,
                        help="number of passages to judge both ways")
    parser.add_argument("--out", type=Path,
                        default=Path("results/judge_pipeline_equivalence_spotcheck.json"))
    args = parser.parse_args()
    asyncio.run(run(args.completions, args.n, args.out))


if __name__ == "__main__":
    main()
