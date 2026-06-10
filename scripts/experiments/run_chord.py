"""Run the chord experiment (pullback vs geodesic vs linear) for one or more pairs.

The variant (manifold, judge mode, output dirs) comes from a YAML config
under ``experiments/``; see ``manifold_emotions.experiments.chord`` for
the schema. This single entry point replaces run_pullback_experiment.py
and its _4d/_6d/_8d_silverman/_nojudge copies (archived on
archive/disorganized-scripts).

    # production 8-D chain, one pair
    uv run python scripts/experiments/run_chord.py --config experiments/chord_8d.yaml happy sad

    # 4-D variant with CLI overrides
    uv run python scripts/experiments/run_chord.py --config experiments/chord_4d.yaml \
        excited weary --k 30 --n 10

    # all pairs from a pair file
    uv run python scripts/experiments/run_chord.py --config experiments/chord_6d.yaml \
        --pairs experiments/pairs/alift_n40.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from manifold_emotions.config import load_config
from manifold_emotions.experiments.chord import ChordRunConfig, run_chord_pair


def load_pairs(path: Path) -> list[tuple[str, str]]:
    """Pair file: JSON ``[["happy", "sad"], ...]`` (labels with literal spaces)."""
    return [(a, b) for a, b in json.loads(path.read_text())]


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("start", nargs="?", default=None)
    parser.add_argument("end", nargs="?", default=None)
    parser.add_argument(
        "--config", type=Path, default=Path("experiments/chord_8d.yaml"),
        help="experiment variant YAML (default: production 8-D)",
    )
    parser.add_argument(
        "--pairs", type=Path, default=None,
        help="JSON pair file to run instead of a single start/end pair",
    )
    parser.add_argument("--k", type=int, default=None, help="num_waypoints override")
    parser.add_argument("--n", type=int, default=None, help="num_prompts override")
    parser.add_argument(
        "--sigma", default=None,
        help="pullback kernel bandwidth override: float, 'knn:K', or omit "
        "for the config/default (median M_y NN distance)",
    )
    parser.add_argument(
        "--results-suffix", default="",
        help="appended to result+cache filenames so a non-default run "
        "doesn't overwrite the default one",
    )
    args = parser.parse_args()

    sigma: float | str | None
    if args.sigma is None:
        sigma = None
    elif args.sigma.startswith("knn:"):
        sigma = args.sigma
    else:
        sigma = float(args.sigma)

    if args.pairs is not None:
        pairs = load_pairs(args.pairs)
    elif args.start is not None and args.end is not None:
        pairs = [(args.start, args.end)]
    else:
        parser.error("provide either start+end labels or --pairs")

    run = ChordRunConfig.from_yaml(args.config)
    config = load_config()

    for start, end in pairs:
        print(f"=== chord [{run.name}, judge={run.judge}]: {start} → {end} ===")
        summary_path = await run_chord_pair(
            config, run, start, end,
            num_waypoints=args.k, num_prompts=args.n,
            sigma=sigma, results_suffix=args.results_suffix,
        )
        if summary_path is None:
            print("  skipped (missing centroid)")
            continue
        summary = json.loads(summary_path.read_text())
        if run.judge != "none":
            print(f"  {'':12s}{'pullback':>10s}  {'geodesic':>10s}  {'linear':>10s}")
            for metric in ("off_manifold_energy", "my_geodesic_distance"):
                row = [summary["trajectories"][m][metric]
                       for m in ("pullback", "geodesic", "linear")]
                label = "off-M_y E" if metric == "off_manifold_energy" else "M_y-line"
                print(f"  {label:<12s}" + "  ".join(f"{v:>10.3f}" for v in row))
        print(f"  saved {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
