"""Offline (CPU) surrogate optimizer — the cheap half of Idea D.

The pure-data reachability screen (steering_ceiling_probe.py, Part 1) showed the V/A
targets are reachable in principle (linear wanders 2.14; some observed vector lands
within 0.57). This script asks the sharper, still-GPU-free question: can a surrogate
f: steering vector -> judged (V/A), optimized *in-distribution*, find vectors it
PREDICTS will beat linear at reaching each target — and produce concrete candidate
vectors to validate on GPU?

Pipeline (all CPU, no judge):
  1. Harvest training (8-D subspace steering vector -> judged V/A) by recomputing the
     pullback/geodesic/linear geometry for every judged pair in results/pullback and
     joining with the existing judged behaviors. Report held-out R^2.
  2. Fit a RandomForest surrogate (handles the non-linear, saturating vector->behavior
     map; extrapolates flat, so it can't hallucinate reaching far-off targets).
  3. Build a TRUST-CONSTRAINED candidate pool: perturbations + interpolations of the
     observed vectors, filtered to stay within the data's k-NN radius. The surrogate is
     only trusted near data; the optimizer is not allowed to leave it.
  4. For every n=40 chord target waypoint, retrieve the in-trust candidate the surrogate
     predicts lands closest. Headroom = linear_pred - opt_pred (both surrogate-predicted,
     so surrogate bias cancels). Positive, meaningful headroom => a closed-loop GPU run is
     worth it; ~zero => the null is robust even to optimization.
  5. Save the top-headroom optimized vectors (subspace + unprojected x scale) so a short
     GPU pass can check whether the surrogate's promise actually holds when generated.

Honesty: the surrogate can be *exploited* — minimizing predicted distance may find x where
the surrogate is wrong. The trust constraint limits this; GPU validation resolves it. And
per-waypoint-independent optimization is an upper bound (ignores trajectory coherence).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from manifold_emotions.behavior.manifold import BehaviorManifold
from manifold_emotions.manifold.fit import FittedManifold
from manifold_emotions.manifold.pullback import compute_pullback

METHODS = ("pullback", "geodesic", "linear")
SUB_ATTR = {"pullback": "pullback_sub", "geodesic": "geodesic_sub", "linear": "linear_sub"}


def harvest_training(results_dir: Path, manifold, behavior) -> tuple[np.ndarray, np.ndarray]:
    """Recompute geometry per judged pair, join with judged V/A -> (X sub, Y V/A)."""
    X, Y = [], []
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
        for m in METHODS:
            if m not in summ["trajectories"]:
                continue
            sub = np.asarray(getattr(g, SUB_ATTR[m]))
            v = summ["trajectories"][m]["waypoint_valence"]
            a = summ["trajectories"][m]["waypoint_arousal"]
            for k in range(sub.shape[0]):
                if np.isfinite(v[k]) and np.isfinite(a[k]):
                    X.append(sub[k]); Y.append([v[k], a[k]])
    return np.array(X), np.array(Y)


def build_candidate_pool(X: np.ndarray, n_target: int, rng) -> np.ndarray:
    """Perturbations + interpolations of observed vectors, staying near the data."""
    n, d = X.shape
    nn = NearestNeighbors(n_neighbors=6).fit(X)
    local_scale = nn.kneighbors(X)[0][:, 1:].mean(axis=1)  # mean dist to 5 NN per point
    cands = [X]
    per = max(1, n_target // (n * 4))
    for scale in (0.3, 0.6, 1.0):
        for _ in range(per):
            noise = rng.normal(size=(n, d)) * (scale * local_scale)[:, None]
            cands.append(X + noise)
    # interpolations between random nearby pairs
    idx_a = rng.integers(0, n, size=n_target // 4)
    nbrs = nn.kneighbors(X[idx_a])[1][:, 1:]
    idx_b = nbrs[np.arange(len(idx_a)), rng.integers(0, 5, size=len(idx_a))]
    t = rng.uniform(0.2, 0.8, size=(len(idx_a), 1))
    cands.append((1 - t) * X[idx_a] + t * X[idx_b])
    return np.concatenate(cands, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifold", type=Path, default=Path("data/manifold_h.npz"))
    ap.add_argument("--behavior", type=Path, default=Path("data/manifold_y.npz"))
    ap.add_argument("--train-dir", type=Path, default=Path("results/pullback"))
    ap.add_argument("--target-dir", type=Path, default=Path("results/pullback"))
    ap.add_argument("--pairs", type=Path, default=Path("experiments/pairs/alift_n40.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/surrogate_optimizer"))
    ap.add_argument("--pool-size", type=int, default=60000)
    ap.add_argument("--trust-pct", type=float, default=90.0,
                    help="drop candidates beyond this percentile of training k-NN radius")
    ap.add_argument("--n-validate", type=int, default=5,
                    help="top-headroom pairs to dump for GPU; -1 dumps all pairs")
    ap.add_argument("--val-subdir", default="validation_vectors",
                    help="subdir of --out-dir for opt_*.npz; use a fresh name to not overwrite")
    ap.add_argument("--summary-name", default="_summary.json",
                    help="filename under --out-dir for the run summary")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    manifold = FittedManifold.load(args.manifold)
    behavior = BehaviorManifold.load(args.behavior)

    # 1-2. harvest + surrogate
    X, Y = harvest_training(args.train_dir, manifold, behavior)
    print(f"Training: {len(X)} (vector -> V/A) points, dim={X.shape[1]}")
    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2, random_state=0)
    rf = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=0)
    rf.fit(Xtr, Ytr)
    r2 = r2_score(Yte, rf.predict(Xte), multioutput="raw_values")
    print(f"Surrogate held-out R^2: valence {r2[0]:+.3f}  arousal {r2[1]:+.3f}")
    rf_full = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=0).fit(X, Y)

    # 3. trust-constrained candidate pool
    pool = build_candidate_pool(X, args.pool_size, rng)
    nn = NearestNeighbors(n_neighbors=6).fit(X)
    train_radius = nn.kneighbors(X)[0][:, -1]
    tau = float(np.percentile(train_radius, args.trust_pct))
    pool_radius = nn.kneighbors(pool)[0][:, -1]
    pool = pool[pool_radius <= tau]
    pool_pred = rf_full.predict(pool)
    print(f"Candidate pool: {len(pool)} in-trust vectors (tau={tau:.2f}); predicting V/A")

    # 4. per-pair, per-waypoint retrieval
    target_pairs = [tuple(p) for p in json.loads(args.pairs.read_text())]
    per_pair = []
    dump = []
    for s, e in target_pairs:
        path = args.target_dir / f"{s}_{e}.json"
        if not path.exists():
            path = args.target_dir / f"{e}_{s}.json"
        if not path.exists() or s not in manifold.labels or e not in manifold.labels:
            continue
        summ = json.loads(path.read_text())
        g = summ["geometry"]
        targets = np.array([g["my_path_valence"], g["my_path_arousal"]], dtype=np.float64).T  # (K,2)
        geom = compute_pullback(manifold, behavior, s, e, num_waypoints=30, sigma=None)
        lin_sub = np.asarray(geom.linear_sub)  # (K,d)
        lin_pred = rf_full.predict(lin_sub)  # surrogate at linear's vector
        lin_v = summ["trajectories"]["linear"]["waypoint_valence"]
        lin_a = summ["trajectories"]["linear"]["waypoint_arousal"]
        lin_actual_xy = np.array([lin_v, lin_a], dtype=np.float64).T

        D = cdist(targets, pool_pred)  # (K, P) predicted distance of each candidate to each target
        best = D.argmin(axis=1)  # (K,)
        opt_pred_dist = D[np.arange(len(best)), best]  # (K,)
        lin_pred_dist = np.linalg.norm(lin_pred - targets, axis=1)
        finite = np.all(np.isfinite(lin_actual_xy), axis=1)
        lin_actual_dist = np.linalg.norm(lin_actual_xy[finite] - targets[finite], axis=1)

        per_pair.append({
            "pair": f"{s}->{e}",
            "linear_actual_mean": float(lin_actual_dist.mean()),
            "linear_pred_mean": float(lin_pred_dist.mean()),
            "opt_pred_mean": float(opt_pred_dist.mean()),
            "headroom_pred": float(lin_pred_dist.mean() - opt_pred_dist.mean()),
            "opt_trust_dist_mean": float(np.mean(nn.kneighbors(pool[best])[0][:, -1])),
        })
        dump.append({"pair": (s, e), "opt_sub": pool[best], "targets": targets,
                     "opt_pred": pool_pred[best], "headroom": per_pair[-1]["headroom_pred"]})

    head = np.array([r["headroom_pred"] for r in per_pair])
    lin_a = np.array([r["linear_actual_mean"] for r in per_pair])
    lin_p = np.array([r["linear_pred_mean"] for r in per_pair])
    opt_p = np.array([r["opt_pred_mean"] for r in per_pair])
    print(f"\n=== surrogate optimizer (n={len(per_pair)} pairs) ===")
    print(f"  linear actual mean dist:     {lin_a.mean():.3f}")
    print(f"  linear surrogate-pred dist:  {lin_p.mean():.3f}   (sanity: ~= actual if surrogate good)")
    print(f"  optimized surrogate-pred:    {opt_p.mean():.3f}")
    print(f"  predicted headroom:          {head.mean():.3f}  (linear_pred - opt_pred)")
    print(f"  pairs with headroom > 0.3:   {(head > 0.3).sum()}/{len(head)}")

    # 5. dump top-headroom optimized trajectories for GPU validation
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dump.sort(key=lambda x: -x["headroom"])
    val_dir = args.out_dir / args.val_subdir
    val_dir.mkdir(parents=True, exist_ok=True)
    scale = 8.0
    dumped = []
    n_dump = len(dump) if args.n_validate < 0 else args.n_validate
    for item in dump[:n_dump]:
        s, e = item["pair"]
        full = manifold.unproject(item["opt_sub"]).astype(np.float32) * scale  # (K, hidden)
        np.savez_compressed(val_dir / f"opt_{s}_{e}.npz",
                            pair=np.array([s, e], dtype=object),
                            opt_sub=item["opt_sub"], opt_full=full,
                            targets=item["targets"], opt_pred=item["opt_pred"])
        dumped.append(f"{s}->{e} (headroom {item['headroom']:.2f})")

    summary = {
        "surrogate_r2": {"valence": float(r2[0]), "arousal": float(r2[1])},
        "n_train": int(len(X)), "pool_size": int(len(pool)), "trust_tau": tau,
        "linear_actual_mean": float(lin_a.mean()),
        "linear_pred_mean": float(lin_p.mean()),
        "opt_pred_mean": float(opt_p.mean()),
        "mean_predicted_headroom": float(head.mean()),
        "pairs_headroom_gt_0p3": int((head > 0.3).sum()),
        "validation_dumped": dumped,
        "per_pair": per_pair,
    }
    (args.out_dir / args.summary_name).write_text(json.dumps(summary, indent=2))
    print(f"\n  dumped {len(dumped)} validation-vector sets to {val_dir}/")
    print(f"  saved {args.out_dir / args.summary_name}")


if __name__ == "__main__":
    main()
