"""Pullback geometry-only smoke test (no LLM, no judge).

Builds the pullback path for several emotion pairs at the 8-D operating
point and prints whether each pullback is closer to the M_h geodesic or
to the M_h linear interpolation.

Geometric prediction (Goodfire §3.3): pullbacks of M_y straight lines
should land near M_h geodesics if the two manifolds share geometry.

Run with:
    uv run python scripts/pullback_geometry_smoke.py
"""

from __future__ import annotations

import numpy as np

from manifold_emotions.behavior.manifold import BehaviorManifold
from manifold_emotions.config import load_config
from manifold_emotions.manifold.fit import FittedManifold
from manifold_emotions.manifold.pullback import compute_pullback

# Mix of pairs the forward experiment showed manifold-favored, linear-favored,
# and approximately neutral, so we can see whether the pullback prediction
# tracks the forward result.
PAIRS: tuple[tuple[str, str], ...] = (
    ("excited", "weary"),       # manifold-favored on forward
    ("calm", "desperate"),      # manifold-favored on forward
    ("ecstatic", "melancholy"),  # manifold-favored on forward
    ("happy", "sad"),           # near-tie
    ("terrified", "serene"),    # linear-favored on forward
)


def main() -> None:
    config = load_config()
    manifold = FittedManifold.load(config.paths.manifold_h)
    behavior = BehaviorManifold.load(config.paths.manifold_y)

    print(
        f"M_h subspace dim {manifold.num_components}, "
        f"{len(manifold.labels)} emotion centroids, "
        f"alpha={manifold.alpha}, beta={manifold.beta}"
    )
    print()
    print(
        f"  {'pair':>22s}  {'sigma':>5s}  {'pullL':>6s}  {'geoL':>6s}  {'linL':>6s}  "
        f"{'p↔geo':>6s}  {'p↔lin':>6s}  closer-to"
    )
    for start, end in PAIRS:
        if start not in manifold.labels or end not in manifold.labels:
            print(f"  {start+'→'+end:>22s}  (missing centroids — skipping)")
            continue
        if start not in behavior.labels or end not in behavior.labels:
            print(f"  {start+'→'+end:>22s}  (missing in M_y — skipping)")
            continue

        r = compute_pullback(manifold, behavior, start, end, num_waypoints=30)
        print(
            f"  {start+'→'+end:>22s}  {r.sigma:>5.3f}  "
            f"{r.pullback_length:>6.3f}  {r.geodesic_length:>6.3f}  {r.linear_length:>6.3f}  "
            f"{r.mean_dist_to_geodesic:>6.3f}  {r.mean_dist_to_linear:>6.3f}  "
            f"{r.closer_to}"
        )


if __name__ == "__main__":
    main()
