"""CPU-only screen for the steering 'ceiling': is the V/A target reachable at all?

The central null of the project is that no steering method reaches the V/A target
line better than linear interpolation does. Before spending GPU on a closed-loop
optimizer (Idea D), this asks the cheaper question from data we already have:

  PART 1 — pure-data reachability (rock-solid, no model, no extrapolation):
    Pool every judged (valence, arousal) behavior the project has produced under
    steering, across all chord runs (~22k points). For each n=40 chord target
    waypoint y*, find the closest behavior any *other* pair's steering ever
    achieved, and compare to this pair's linear baseline distance.
      headroom = linear_dist - nearest_other_vector_dist
    Big headroom -> the target IS reachable by some additive vector; the per-pair
    methods just don't route there (a ceiling above linear exists; closed-loop
    worth trying). ~Zero headroom -> the target is genuinely hard; the null is
    fundamental, not a routing failure. This is an OPTIMISTIC upper bound: the
    reaching vector may come from an incoherent/different-emotion steer, so "no
    headroom even optimistically" is the strong, trustworthy direction.

  PART 2 — surrogate viability (informs whether the interpolation extension /
    closed-loop is even buildable): 5-fold CV R^2 of a regressor mapping the 8-D
    subspace steering point -> judged (valence, arousal). If a surrogate can't
    predict behavior from the steering vector, neither offline optimization nor a
    sample-efficient closed loop will work, and we go straight to brute GPU.

Reads existing results only; writes results/steering_ceiling/. No GPU, no judge.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ATLAS_DIRS = ["pullback", "pullback_4d", "pullback_6d", "pullback_8d_silverman"]
METHODS = ("pullback", "geodesic", "linear")


def _finite_xy(valence: list, arousal: list) -> np.ndarray:
    a = np.array([valence, arousal], dtype=np.float64).T  # (K, 2)
    return a


def harvest_atlas(results_root: Path, dirs: list[str]) -> tuple[np.ndarray, list[frozenset]]:
    """All judged (V/A) behaviors across runs, tagged by source pair (for exclusion)."""
    points: list[np.ndarray] = []
    pair_keys: list[frozenset] = []
    for d in dirs:
        rdir = results_root / d
        if not rdir.exists():
            continue
        for f in rdir.glob("*.json"):
            if f.stem.startswith("_"):
                continue
            try:
                summ = json.loads(f.read_text())
            except json.JSONDecodeError:
                continue
            pair = summ.get("pair")
            if not pair:
                continue
            key = frozenset(pair)
            for m in summ.get("trajectories", {}).values():
                xy = _finite_xy(m["waypoint_valence"], m["waypoint_arousal"])
                for row in xy:
                    if np.all(np.isfinite(row)):
                        points.append(row)
                        pair_keys.append(key)
    return np.array(points), pair_keys


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-root", type=Path, default=Path("results"))
    ap.add_argument("--target-dir", type=Path, default=Path("results/pullback"),
                    help="dir whose per-pair summaries supply the n=40 chord targets")
    ap.add_argument("--pairs", type=Path, default=Path("experiments/pairs/alift_n40.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/steering_ceiling"))
    ap.add_argument("--surrogate", action="store_true", help="also run Part 2 (8-D CV R^2)")
    ap.add_argument("--manifold", type=Path, default=Path("data/manifold_h.npz"))
    ap.add_argument("--behavior", type=Path, default=Path("data/manifold_y.npz"))
    args = ap.parse_args()

    atlas, atlas_keys = harvest_atlas(args.results_root, ATLAS_DIRS)
    atlas_keys_arr = np.array([hash(k) for k in atlas_keys])
    print(f"Atlas: {len(atlas)} judged behaviors from {len(set(atlas_keys))} distinct pairs")

    target_pairs = [tuple(p) for p in json.loads(args.pairs.read_text())]

    # PART 1 — reachability
    per_pair = []
    for s, e in target_pairs:
        path = args.target_dir / f"{s}_{e}.json"
        if not path.exists():
            path = args.target_dir / f"{e}_{s}.json"
        if not path.exists():
            continue
        summ = json.loads(path.read_text())
        g = summ["geometry"]
        targets = np.array([g["my_path_valence"], g["my_path_arousal"]], dtype=np.float64).T  # (K,2)
        lin = _finite_xy(summ["trajectories"]["linear"]["waypoint_valence"],
                         summ["trajectories"]["linear"]["waypoint_arousal"])
        key_hash = hash(frozenset((s, e)))
        other = atlas[atlas_keys_arr != key_hash]  # exclude this pair's own behaviors

        lin_d, near_d = [], []
        for k in range(targets.shape[0]):
            y = targets[k]
            if np.all(np.isfinite(lin[k])):
                lin_d.append(float(np.linalg.norm(lin[k] - y)))
            d = np.linalg.norm(other - y[None, :], axis=1)
            near_d.append(float(d.min()))
        if not lin_d:
            continue
        lin_m, near_m = float(np.mean(lin_d)), float(np.mean(near_d))
        per_pair.append({
            "pair": f"{s}->{e}",
            "linear_mean_dist": lin_m,
            "nearest_other_mean_dist": near_m,
            "headroom": lin_m - near_m,
        })

    lin_all = np.array([r["linear_mean_dist"] for r in per_pair])
    near_all = np.array([r["nearest_other_mean_dist"] for r in per_pair])
    head_all = lin_all - near_all
    print("\n=== PART 1: pure-data reachability (n={} pairs) ===".format(len(per_pair)))
    print(f"  linear mean dist to target line:        {lin_all.mean():.3f}")
    print(f"  nearest OTHER-vector mean dist:          {near_all.mean():.3f}")
    print(f"  mean headroom (linear - reachable):      {head_all.mean():.3f}")
    print(f"  pairs with headroom > 0.5:               {(head_all > 0.5).sum()}/{len(head_all)}")
    print(f"  fraction of target dist that is reachable-away: "
          f"{near_all.mean() / lin_all.mean():.2f}")

    out = {
        "part1_reachability": {
            "n_pairs": len(per_pair),
            "linear_mean_dist": float(lin_all.mean()),
            "nearest_other_mean_dist": float(near_all.mean()),
            "mean_headroom": float(head_all.mean()),
            "pairs_headroom_gt_0p5": int((head_all > 0.5).sum()),
            "per_pair": per_pair,
        }
    }

    # PART 2 — surrogate viability (8-D), optional. The per-pair path npz weren't
    # kept for the production runs, so recompute the subspace steering vectors
    # geometrically (deterministic, CPU) and join them with the existing judged
    # V/A from the summaries.
    if args.surrogate:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import cross_val_score

        from manifold_emotions.behavior.manifold import BehaviorManifold
        from manifold_emotions.manifold.fit import FittedManifold
        from manifold_emotions.manifold.pullback import compute_pullback

        manifold = FittedManifold.load(args.manifold)
        behavior = BehaviorManifold.load(args.behavior)
        sub_attr = {"pullback": "pullback_sub", "geodesic": "geodesic_sub", "linear": "linear_sub"}
        X, Y = [], []
        for s, e in target_pairs:
            rpath = args.target_dir / f"{s}_{e}.json"
            if not rpath.exists():
                continue
            if s not in manifold.labels or e not in manifold.labels:
                continue
            summ = json.loads(rpath.read_text())
            g = compute_pullback(manifold, behavior, s, e, num_waypoints=30, sigma=None)
            for m in METHODS:
                sub = getattr(g, sub_attr[m])  # (K, d)
                xy = _finite_xy(summ["trajectories"][m]["waypoint_valence"],
                                summ["trajectories"][m]["waypoint_arousal"])
                for k in range(sub.shape[0]):
                    if np.all(np.isfinite(xy[k])):
                        X.append(np.asarray(sub[k])); Y.append(xy[k])
        X, Y = np.array(X), np.array(Y)
        print(f"\n=== PART 2: surrogate viability (8-D), N={len(X)} ===")
        if X.ndim != 2 or len(X) < 50:
            print("  insufficient data for surrogate CV; skipping Part 2")
            args.out_dir.mkdir(parents=True, exist_ok=True)
            (args.out_dir / "_summary.json").write_text(json.dumps(out, indent=2))
            print(f"\nsaved {args.out_dir / '_summary.json'}")
            return
        r2 = {}
        for j, name in enumerate(("valence", "arousal")):
            rf = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=0)
            scores = cross_val_score(rf, X, Y[:, j], cv=5, scoring="r2")
            r2[name] = float(scores.mean())
            print(f"  {name}: 5-fold CV R^2 = {scores.mean():+.3f} (std {scores.std():.3f})")
        out["part2_surrogate_cv_r2"] = {"n": int(len(X)), **r2}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "_summary.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved {args.out_dir / '_summary.json'}")


if __name__ == "__main__":
    main()
