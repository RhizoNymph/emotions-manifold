"""Honest held-out R^2 for the behavior surrogate under leakage-free splits.

surrogate_optimizer.py reports its surrogate R^2 with sklearn's random
``train_test_split`` at the *waypoint* level. That split leaks:

  * adjacent waypoints on the same trajectory are near-duplicate 8-D inputs with
    highly correlated V/A targets, and
  * pullback / geodesic / linear methods share endpoint vectors per pair,

so near-copies of held-out points sit in the training set. The reported
R^2 (~0.79 valence / ~0.84 arousal) therefore overstates generalization.

This script recomputes the surrogate R^2 three ways, holding the RandomForest
hyperparameters fixed (n_estimators=300, random_state=0) so the split is the
only thing that changes:

  (a) RANDOM waypoint-level split (test_size=0.2, random_state=0) -- reproduces
      the original leaky number for the paper's before/after comparison.
  (b) GROUPED-BY-PAIR GroupKFold(k=5) -- every waypoint/method row from the same
      emotion pair stays in one fold, so no near-copy of a test pair is ever
      trained on. This measures out-of-pair generalization.
  (c) LEAVE-ONE-EMOTION-OUT (strict) -- for each emotion, all pairs touching it
      are the test set, and any train pair sharing *either* endpoint with a test
      pair is excluded, so the test emotions never appear in training. This asks
      whether the vector->behavior coupling transfers to unseen emotions at all.

Harvesting mirrors surrogate_optimizer.harvest_training exactly (same results
dir, same geometry recompute, same finite-mask), additionally tracking which
emotion pair every row came from so it can be used as the grouping key.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, train_test_split

from manifold_emotions.behavior.manifold import BehaviorManifold
from manifold_emotions.manifold.fit import FittedManifold
from manifold_emotions.manifold.pullback import compute_pullback

METHODS = ("pullback", "geodesic", "linear")
SUB_ATTR = {"pullback": "pullback_sub", "geodesic": "geodesic_sub", "linear": "linear_sub"}

# Fixed RF hyperparameters -- identical to surrogate_optimizer.py so the split is
# the only variable across the three evaluations below.
RF_KW = dict(n_estimators=300, n_jobs=-1, random_state=0)


def harvest_training(
    results_dir: Path, manifold, behavior
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[str, str]]]:
    """Recompute geometry per judged pair -> (X sub, Y V/A, pair_id, pair_list).

    Identical harvesting to surrogate_optimizer.harvest_training, but also emits
    ``pair_id[i]`` = index into ``pair_list`` for every row, so rows can be
    grouped by the emotion pair they originated from.
    """
    X: list[np.ndarray] = []
    Y: list[list[float]] = []
    pair_id: list[int] = []
    pair_list: list[tuple[str, str]] = []
    for f in sorted(results_dir.glob("*.json")):
        if f.stem.startswith("_"):
            continue
        try:
            summ = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        pair = summ.get("pair")
        if not pair or pair[0] not in manifold.labels or pair[1] not in manifold.labels:
            continue
        try:
            g = compute_pullback(manifold, behavior, pair[0], pair[1], num_waypoints=30, sigma=None)
        except Exception:
            continue
        cur_pid = len(pair_list)
        added = False
        for m in METHODS:
            if m not in summ["trajectories"]:
                continue
            sub = np.asarray(getattr(g, SUB_ATTR[m]))
            v = summ["trajectories"][m]["waypoint_valence"]
            a = summ["trajectories"][m]["waypoint_arousal"]
            for k in range(sub.shape[0]):
                if np.isfinite(v[k]) and np.isfinite(a[k]):
                    X.append(sub[k])
                    Y.append([v[k], a[k]])
                    pair_id.append(cur_pid)
                    added = True
        if added:
            pair_list.append((pair[0], pair[1]))
    return np.array(X), np.array(Y), np.array(pair_id), pair_list


def _r2_va(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    r2 = r2_score(y_true, y_pred, multioutput="raw_values")
    return float(r2[0]), float(r2[1])


def _msd(vals: list[float]) -> dict[str, float]:
    arr = np.asarray(vals, dtype=np.float64)
    return {"mean": float(arr.mean()), "sd": float(arr.std(ddof=0)), "n": int(arr.size)}


def random_split_r2(X: np.ndarray, Y: np.ndarray) -> dict:
    """Reproduce the original leaky waypoint-level split."""
    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2, random_state=0)
    rf = RandomForestRegressor(**RF_KW).fit(Xtr, Ytr)
    v, a = _r2_va(Yte, rf.predict(Xte))
    return {"valence": v, "arousal": a, "n_test": int(len(Xte))}


def grouped_by_pair_r2(X: np.ndarray, Y: np.ndarray, pair_id: np.ndarray, k: int) -> dict:
    """GroupKFold(k) with the emotion pair as the group -- no pair straddles folds."""
    gkf = GroupKFold(n_splits=k)
    folds = []
    for i, (tr, te) in enumerate(gkf.split(X, Y, groups=pair_id)):
        rf = RandomForestRegressor(**RF_KW).fit(X[tr], Y[tr])
        v, a = _r2_va(Y[te], rf.predict(X[te]))
        folds.append(
            {
                "fold": i,
                "valence": v,
                "arousal": a,
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "n_test_pairs": int(np.unique(pair_id[te]).size),
            }
        )
    return {
        "k": k,
        "folds": folds,
        "valence": _msd([f["valence"] for f in folds]),
        "arousal": _msd([f["arousal"] for f in folds]),
    }


def leave_one_emotion_out_r2(
    X: np.ndarray,
    Y: np.ndarray,
    pair_id: np.ndarray,
    pair_list: list[tuple[str, str]],
    min_train_pairs: int,
    min_test_rows: int,
) -> dict:
    """Strict leave-one-emotion-out.

    For each emotion ``e``: test = all rows whose pair contains ``e``; the test
    emotion set is every endpoint of those test pairs; train = rows whose pair
    shares *no* emotion with the test set. This guarantees the held-out
    emotions never appear (as either endpoint, in any pair, via any method) in
    training. Folds with too little train/test are skipped and counted.
    """
    pair_emos = [set(p) for p in pair_list]  # pid -> {emotion, emotion}
    row_emos = [pair_emos[p] for p in pair_id]
    all_emotions = sorted({e for p in pair_list for e in p})

    folds = []
    skipped = 0
    for e in all_emotions:
        test_mask = np.array([e in re for re in row_emos])
        if not test_mask.any():
            continue
        test_pids = np.unique(pair_id[test_mask])
        test_emotions: set[str] = set()
        for pid in test_pids:
            test_emotions |= pair_emos[pid]
        train_mask = np.array([len(re & test_emotions) == 0 for re in row_emos])

        n_train_pairs = int(np.unique(pair_id[train_mask]).size)
        n_test = int(test_mask.sum())
        if n_train_pairs < min_train_pairs or n_test < min_test_rows:
            skipped += 1
            continue
        rf = RandomForestRegressor(**RF_KW).fit(X[train_mask], Y[train_mask])
        v, a = _r2_va(Y[test_mask], rf.predict(X[test_mask]))
        folds.append(
            {
                "emotion": e,
                "valence": v,
                "arousal": a,
                "n_train": int(train_mask.sum()),
                "n_train_pairs": n_train_pairs,
                "n_test": n_test,
                "n_test_pairs": int(test_pids.size),
            }
        )
    return {
        "n_emotions_total": len(all_emotions),
        "n_folds_evaluated": len(folds),
        "n_folds_skipped": skipped,
        "min_train_pairs": min_train_pairs,
        "min_test_rows": min_test_rows,
        "folds": folds,
        "valence": _msd([f["valence"] for f in folds]) if folds else None,
        "arousal": _msd([f["arousal"] for f in folds]) if folds else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifold", type=Path, default=Path("data/manifold_h.npz"))
    ap.add_argument("--behavior", type=Path, default=Path("data/manifold_y.npz"))
    ap.add_argument("--train-dir", type=Path, default=Path("results/pullback"))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--min-train-pairs", type=int, default=10,
                    help="skip a leave-one-emotion-out fold with fewer surviving train pairs")
    ap.add_argument("--min-test-rows", type=int, default=10,
                    help="skip a leave-one-emotion-out fold with fewer test rows")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("results/surrogate_optimizer/grouped_split_r2.json"),
    )
    args = ap.parse_args()

    manifold = FittedManifold.load(args.manifold)
    behavior = BehaviorManifold.load(args.behavior)

    X, Y, pair_id, pair_list = harvest_training(args.train_dir, manifold, behavior)
    print(f"Training: {len(X)} (vector -> V/A) rows from {len(pair_list)} pairs, dim={X.shape[1]}")

    print("\n[a] random waypoint-level split (leaky baseline)")
    rnd = random_split_r2(X, Y)
    print(f"    R^2  valence {rnd['valence']:+.3f}  arousal {rnd['arousal']:+.3f}  (n_test={rnd['n_test']})")

    print(f"\n[b] grouped-by-pair GroupKFold(k={args.k})")
    grp = grouped_by_pair_r2(X, Y, pair_id, args.k)
    for f in grp["folds"]:
        print(
            f"    fold {f['fold']}: valence {f['valence']:+.3f}  arousal {f['arousal']:+.3f}"
            f"  (train {f['n_train']}, test {f['n_test']} rows / {f['n_test_pairs']} pairs)"
        )
    print(
        f"    valence {grp['valence']['mean']:+.3f} +/- {grp['valence']['sd']:.3f}"
        f"   arousal {grp['arousal']['mean']:+.3f} +/- {grp['arousal']['sd']:.3f}"
    )

    print("\n[c] strict leave-one-emotion-out")
    loeo = leave_one_emotion_out_r2(
        X, Y, pair_id, pair_list, args.min_train_pairs, args.min_test_rows
    )
    print(
        f"    evaluated {loeo['n_folds_evaluated']}/{loeo['n_emotions_total']} emotion folds"
        f" (skipped {loeo['n_folds_skipped']} for too little train/test)"
    )
    if loeo["valence"] is not None:
        print(
            f"    valence {loeo['valence']['mean']:+.3f} +/- {loeo['valence']['sd']:.3f}"
            f"   arousal {loeo['arousal']['mean']:+.3f} +/- {loeo['arousal']['sd']:.3f}"
        )

    out = {
        "n_train_rows": int(len(X)),
        "n_pairs": int(len(pair_list)),
        "dim": int(X.shape[1]),
        "rf_hyperparameters": {k: v for k, v in RF_KW.items() if k != "n_jobs"},
        "random_split": rnd,
        "grouped_by_pair": grp,
        "leave_one_emotion_out": loeo,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
