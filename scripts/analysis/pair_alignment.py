"""Predict manifold-vs-linear Δ from M_h subspace structure.

For every emotion pair, compute structural metrics in the 8-D M_h
subspace (no LLM calls) and cross-reference with the Δ values we have
from previous K=30/N=10 and K=10/N=3 runs. The goal is to find a
single (or small set of) scalars that distinguishes
manifold-favored from linear-favored pairs, so we can predict which
pairs to test next without burning judge calls.

Three metrics:

1. **Participation ratio** of the unit direction over PCA axes
   (instant per pair).
2. **Near-chord centroid count** — how much curvature material sits
   close enough to the chord to attract the geodesic
   (instant per pair).
3. **G_E length gap** = (linear path length) − (geodesic path length)
   under the density metric (slow: requires fitting a geodesic per
   pair, ~1-2 sec each via L-BFGS-B + JAX).

For the ~5 pairs we have measured Δ for, we plot each metric against
Δ. Whichever has the cleanest sign-consistency is our predictor.

Run with:
    uv run python scripts/pair_alignment.py
    uv run python scripts/pair_alignment.py --no-ge-gap   # skip the slow part
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from manifold_emotions.behavior.manifold import BehaviorManifold
from manifold_emotions.config import load_config
from manifold_emotions.manifold.alignment import (
    all_pair_alignments,
    max_chord_deflection,
    predicted_off_my_energy,
)
from manifold_emotions.manifold.fit import FittedManifold
from manifold_emotions.manifold.geodesic import fit_geodesic
from manifold_emotions.vectors.diff_in_means import EmotionVectors


def gather_known_deltas() -> dict[tuple[str, str], float]:
    """Pull Δ values from the multipair sweep and the K=30/N=10 8-D runs.

    Pairs appearing in both sources are overridden by the higher-K
    measurement. Returns {(start, end) → Δ}; we also store the reverse
    direction so lookups by either ordering succeed.
    """
    out: dict[tuple[str, str], float] = {}

    multipair_path = Path("results/steering_multipair.json")
    if multipair_path.exists():
        for row in json.loads(multipair_path.read_text())["pairs"]:
            d = row["delta_linear_minus_manifold"]
            out[(row["start"], row["end"])] = d
            out[(row["end"], row["start"])] = d

    sweep_8d_paths = sorted(Path("results/subspace_sweep").glob("*_dim08.json"))
    for path in sweep_8d_paths:
        row = json.loads(path.read_text())
        s, e = row["pair"]
        d = row["delta_linear_minus_manifold"]
        out[(s, e)] = d
        out[(e, s)] = d

    # Validation rounds 1+2 — same K=30/N=10 measurement quality as the
    # subspace sweep. Order after multipair so these (more recent) values
    # win on overlapping pairs.
    for path in sorted(Path("results/pair_validation").glob("*.json")):
        row = json.loads(path.read_text())
        s, e = row["pair"]
        d = row["delta_linear_minus_manifold"]
        out[(s, e)] = d
        out[(e, s)] = d

    return out


def compute_ge_gaps_and_predicted_my(
    manifold: FittedManifold,
    behavior,
    num_waypoints: int = 20,
    max_iter: int = 200,
) -> dict[tuple[str, str], dict[str, float]]:
    """For every unordered pair, fit a geodesic and record G_E lengths +
    a structural prediction of off-M_y energy.

    Slow: one L-BFGS-B per pair (~7 min for 435 pairs). The predicted
    off-M_y energy is computed from the fitted waypoints in the same
    pass, so the extra cost is negligible.
    """
    geometry = manifold.make_geometry()
    out: dict[tuple[str, str], dict[str, float]] = {}
    n = len(manifold.labels)
    total = n * (n - 1) // 2
    centroids = manifold.centroids_subspace.astype(np.float32)

    for idx, (i, j) in enumerate(combinations(range(n), 2)):
        if idx % 50 == 0:
            print(f"  geodesic {idx}/{total}", flush=True)
        result = fit_geodesic(
            geometry,
            centroids[i],
            centroids[j],
            num_waypoints=num_waypoints,
            max_iter=max_iter,
        )
        s = manifold.labels[i]
        e = manifold.labels[j]
        if s in behavior.labels and e in behavior.labels:
            pred = predicted_off_my_energy(result.waypoints, manifold, behavior)
        else:
            pred = float("nan")
        deflection = max_chord_deflection(result.waypoints)
        out[(s, e)] = {
            "geodesic_length": result.final_length,
            "linear_length": result.initial_length,
            "ge_length_gap": result.initial_length - result.final_length,
            "predicted_off_my_energy": pred,
            "max_chord_deflection": deflection,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--no-ge-gap", action="store_true",
        help="skip the per-pair geodesic fitting step",
    )
    parser.add_argument(
        "--radius-multiplier", type=float, default=1.0,
        help="multiplier on KDE bandwidth for near-chord radius",
    )
    args = parser.parse_args()

    config = load_config()
    ev = EmotionVectors.load(config.paths.emotion_vectors)
    mh = FittedManifold.load(config.paths.manifold_h)
    my = BehaviorManifold.load(config.paths.manifold_y)

    print(
        f"M_h: dim={mh.num_components}, bandwidth={mh.kde_bandwidth:.4f}, "
        f"{len(mh.labels)} emotions"
    )

    alignments = all_pair_alignments(mh, radius_multiplier=args.radius_multiplier)
    print(f"computed {len(alignments)} pair alignments")

    ge_gaps: dict[tuple[str, str], dict[str, float]] = {}
    if not args.no_ge_gap:
        print(f"fitting {len(alignments)} geodesics under G_E ...", flush=True)
        ge_gaps = compute_ge_gaps_and_predicted_my(mh, my)

    # Pair the per-emotion behavior coords for printing.
    my_lookup = {label: my.centroids[i] for i, label in enumerate(my.labels)}
    known = gather_known_deltas()

    rows: list[dict] = []
    for a in alignments:
        my_s = my_lookup.get(a.start)
        my_e = my_lookup.get(a.end)
        v_diff = float(my_e[0] - my_s[0]) if my_s is not None and my_e is not None else float("nan")
        a_diff = float(my_e[1] - my_s[1]) if my_s is not None and my_e is not None else float("nan")
        gap_row = ge_gaps.get((a.start, a.end), {})
        delta = known.get((a.start, a.end))
        rows.append({
            "start": a.start,
            "end": a.end,
            "h_distance": a.h_distance,
            "v_diff": v_diff,
            "a_diff": a_diff,
            "participation_ratio": a.participation_ratio,
            "top_pc": a.top_pc,
            "top_pc_fraction": a.top_pc_fraction,
            "near_chord_centroid_count": a.near_chord_centroid_count,
            "pc_fractions": a.pc_fractions.tolist(),
            "ge_geodesic_length": gap_row.get("geodesic_length"),
            "ge_linear_length": gap_row.get("linear_length"),
            "ge_length_gap": gap_row.get("ge_length_gap"),
            "predicted_off_my_energy": gap_row.get("predicted_off_my_energy"),
            "max_chord_deflection": gap_row.get("max_chord_deflection"),
            "known_delta": delta,
        })

    out_path = Path("results/pair_alignment.json")
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"saved {out_path}")

    # === Tabular summary ===
    measured = [r for r in rows if r["known_delta"] is not None]
    measured.sort(key=lambda r: r["known_delta"], reverse=True)

    print()
    print("Measured pairs (sorted by Δ = linear - manifold, manifold-favored first):")
    print(
        f"  {'pair':>26s}  {'Δ':>7s}  {'PR':>5s}  "
        f"{'gap_GE':>7s}  {'maxDef':>6s}  {'pred_E':>7s}  "
        f"{'topPC':>5s}  {'topfrac':>7s}"
    )
    for r in measured:
        pair_str = f"{r['start']}→{r['end']}"
        pred = r["predicted_off_my_energy"]
        defl = r.get("max_chord_deflection")
        print(
            f"  {pair_str:>26s}  {r['known_delta']:+.3f}  "
            f"{r['participation_ratio']:>5.2f}  "
            f"{('%+7.3f' % r['ge_length_gap']) if r['ge_length_gap'] is not None else '      -'}  "
            f"{('%6.2f' % defl) if defl is not None else '     -'}  "
            f"{('%7.3f' % pred) if pred is not None and not np.isnan(pred) else '      -'}  "
            f"{r['top_pc']:>5d}  "
            f"{r['top_pc_fraction']:>7.3f}"
        )

    # Top 10 by participation ratio
    print()
    print("Top 10 untested pairs by participation ratio (highest spread → predicted manifold-favored):")
    untested = [r for r in rows if r["known_delta"] is None]
    untested.sort(key=lambda r: -r["participation_ratio"])
    for r in untested[:10]:
        gap_str = f"{r['ge_length_gap']:+.3f}" if r["ge_length_gap"] is not None else "    -"
        print(
            f"  {r['start']:>14s} → {r['end']:<14s}  "
            f"PR={r['participation_ratio']:>5.2f}  nearC={r['near_chord_centroid_count']:>2d}  "
            f"gap={gap_str}"
        )

    print()
    print("Top 10 untested pairs by G_E length gap (largest curvature → predicted manifold-favored):")
    if ge_gaps:
        untested.sort(key=lambda r: -(r["ge_length_gap"] or -1))
        for r in untested[:10]:
            pred = r["predicted_off_my_energy"]
            pred_s = f"{pred:.3f}" if pred is not None and not np.isnan(pred) else "    -"
            print(
                f"  {r['start']:>14s} → {r['end']:<14s}  "
                f"PR={r['participation_ratio']:>5.2f}  nearC={r['near_chord_centroid_count']:>2d}  "
                f"gap={r['ge_length_gap']:+.3f}  pred_E={pred_s}"
            )

    print()
    print("Top 10 untested pairs by predicted off-M_y energy "
          "(LOWEST = behavior likely tracks M_y line → manifold-favored):")
    if ge_gaps:
        candidates = [
            r for r in untested
            if r["predicted_off_my_energy"] is not None
            and not np.isnan(r["predicted_off_my_energy"])
            and r["ge_length_gap"] is not None
            and r["ge_length_gap"] > 0.3  # exclude near-zero-gap pairs (no curvature)
        ]
        candidates.sort(key=lambda r: r["predicted_off_my_energy"])
        for r in candidates[:10]:
            print(
                f"  {r['start']:>14s} → {r['end']:<14s}  "
                f"PR={r['participation_ratio']:>5.2f}  gap={r['ge_length_gap']:+.3f}  "
                f"pred_E={r['predicted_off_my_energy']:.3f}"
            )


if __name__ == "__main__":
    main()
