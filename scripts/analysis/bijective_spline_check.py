"""Geometry check: a *bijective* parametric spline (the faithful Goodfire analog).

The V/A-parameterized spline failed to fit because V/A is a lossy, many-to-one readout
of the emotion (14% of emotions collide in V/A). Goodfire's spline worked because their
parameter (day index, grid) was bijective with the concepts. The faithful analog for an
unordered emotion cloud is to parameterize by a *learned intrinsic coordinate of the
activations themselves* (here: diffusion-2 of the centroids), where each emotion is one
distinct point — so the spline fits with ~0 residual and geodesics route between the two
real-emotion endpoints through the manifold (near real intermediate emotions).

This is CPU-only and decides whether a behavioral run is warranted. Three checks vs the
V/A-spline and linear:
  1. FIT residual: phi(u_i) vs centroid_i. Bijective should be ~0 (V/A-spline was ~6.4).
  2. ON-MANIFOLD routing: mean distance of interior waypoints to the nearest real centroid
     (activation subspace). Bijective should hug the data; the V/A-spline strays 30-55%.
  3. V/A-TRACKING (the real question): use the vector->V/A surrogate (R^2 ~0.8) to PREDICT
     each waypoint's behavior and measure distance to the V/A chord line. If the bijective
     spline tracks V/A no better than linear, the wall is the coupling (Wall 2), not the
     geometry — a behavioral run won't flip target-reaching.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.ensemble import RandomForestRegressor

from manifold_emotions.behavior.manifold import BehaviorManifold
from manifold_emotions.manifold.fit import FittedManifold
from manifold_emotions.manifold.geodesic import fit_geodesic, linear_interpolation
from manifold_emotions.manifold.pullback import compute_pullback
from manifold_emotions.manifold.spline import SplineManifold
from manifold_emotions.manifold.spline_geodesic import fit_spline_geodesic

K = 30


def diffusion_embed(centroids: np.ndarray, n_components: int = 2) -> np.ndarray:
    sq = squareform(pdist(centroids)) ** 2
    eps = float(np.median(sq[sq > 0]))
    Km = np.exp(-sq / eps)
    q = Km.sum(axis=1)
    Ka = Km / np.outer(q, q)
    d = Ka.sum(axis=1)
    ds = np.sqrt(d)
    Ps = Ka / np.outer(ds, ds)
    Ps = 0.5 * (Ps + Ps.T)
    vals, vecs = np.linalg.eigh(Ps)
    vals, vecs = vals[::-1], vecs[:, ::-1]
    psi = (vecs / ds[:, None])[:, 1 : n_components + 1]
    return (psi * vals[1 : n_components + 1][None, :]).astype(np.float64)


def harvest_surrogate(results_dir: Path, manifold, behavior) -> RandomForestRegressor:
    X, Y = [], []
    attr = {"pullback": "pullback_sub", "geodesic": "geodesic_sub", "linear": "linear_sub"}
    for f in sorted(results_dir.glob("*.json")):
        if f.stem.startswith("_"):
            continue
        try:
            s = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        p = s.get("pair")
        if not p or p[0] not in manifold.labels or p[1] not in manifold.labels:
            continue
        try:
            g = compute_pullback(manifold, behavior, p[0], p[1], num_waypoints=30, sigma=None)
        except Exception:
            continue
        for m, a in attr.items():
            if m not in s["trajectories"]:
                continue
            sub = np.asarray(getattr(g, a))
            v, ar = s["trajectories"][m]["waypoint_valence"], s["trajectories"][m]["waypoint_arousal"]
            for k in range(sub.shape[0]):
                if np.isfinite(v[k]) and np.isfinite(ar[k]):
                    X.append(sub[k]); Y.append([v[k], ar[k]])
    rf = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=0).fit(np.array(X), np.array(Y))
    print(f"surrogate fit on {len(X)} points")
    return rf


def nearest_centroid_dist(path: np.ndarray, centroids: np.ndarray) -> float:
    """Mean over interior waypoints of distance to nearest real centroid."""
    interior = path[1:-1]
    d = np.linalg.norm(interior[:, None, :] - centroids[None, :, :], axis=2)
    return float(d.min(axis=1).mean())


def va_track(path: np.ndarray, rf, va_chord: np.ndarray) -> float:
    pred = rf.predict(path)  # (K, 2)
    return float(np.linalg.norm(pred - va_chord, axis=1).mean())


def main() -> None:
    manifold = FittedManifold.load("data/manifold_h.npz")
    behavior = BehaviorManifold.load("data/manifold_y.npz")
    C = manifold.centroids_subspace.astype(np.float64)
    labels = list(manifold.labels)
    idx = {l: i for i, l in enumerate(labels)}
    bl = {l: i for i, l in enumerate(behavior.labels)}
    va = behavior.centroids[[bl[l] for l in labels]].astype(np.float64)  # (171,2) in manifold order

    # bijective coordinate: diffusion-2 of the activation centroids
    u = diffusion_embed(C, 2)
    bij = SplineManifold.fit(
        labels=manifold.labels, control_coords=u, centroids_subspace=C,
        pca_components=manifold.pca_components, pca_mean=manifold.pca_mean,
        kde_bandwidth=manifold.kde_bandwidth, alpha=manifold.alpha, beta=manifold.beta,
        smoothing=0.0,
    )
    resid_bij = np.linalg.norm(bij.embed_np(u.astype(np.float32)) - C, axis=1)
    va_spline = SplineManifold.load("data/manifold_spline_8d.npz")
    resid_va = np.linalg.norm(va_spline.embed_np(va_spline.control_coords.astype(np.float32))
                              - va_spline.centroids_subspace, axis=1)
    print(f"\nFIT residual to centroids (subspace scale ~{np.linalg.norm(C, axis=1).mean():.1f}):")
    print(f"  bijective (diffusion) spline: mean {resid_bij.mean():.3f}  max {resid_bij.max():.3f}")
    print(f"  V/A spline:                   mean {resid_va.mean():.3f}  max {resid_va.max():.3f}")

    rf = harvest_surrogate(Path("results/pullback"), manifold, behavior)
    geometry = manifold.make_geometry()

    pairs = [tuple(p) for p in json.loads(Path("experiments/pairs/alift_n40.json").read_text())]
    rows = []
    for s, e in pairs:
        if s not in idx or e not in idx:
            continue
        i, j = idx[s], idx[e]
        va_chord = linear_interpolation(va[i], va[j], K)  # (K,2) target line in V/A
        paths = {
            "linear": linear_interpolation(C[i], C[j], K),
            "ambient_geo": fit_geodesic(geometry, C[i], C[j], num_waypoints=K).waypoints,
            "va_spline": fit_spline_geodesic(va_spline, va[i], va[j], metric="induced",
                                             num_waypoints=K, snap_start=C[i], snap_end=C[j]).waypoints,
            "bij_spline": fit_spline_geodesic(bij, u[i], u[j], metric="induced",
                                              num_waypoints=K, snap_start=C[i], snap_end=C[j]).waypoints,
        }
        row = {"pair": f"{s}->{e}"}
        for name, p in paths.items():
            row[f"{name}_onmanifold"] = nearest_centroid_dist(p, C)
            row[f"{name}_vatrack"] = va_track(p, rf, va_chord)
        rows.append(row)

    print(f"\n=== n={len(rows)} pairs ===")
    print(f"{'method':<14}{'on-manifold (->centroid)':>26}{'V/A-track (->chord, predicted)':>32}")
    for name in ("linear", "ambient_geo", "va_spline", "bij_spline"):
        om = np.mean([r[f"{name}_onmanifold"] for r in rows])
        vt = np.mean([r[f"{name}_vatrack"] for r in rows])
        print(f"{name:<14}{om:>26.3f}{vt:>32.3f}")

    out = Path("results/bijective_spline_check")
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "fit_residual": {"bijective_mean": float(resid_bij.mean()), "bijective_max": float(resid_bij.max()),
                         "va_spline_mean": float(resid_va.mean()), "va_spline_max": float(resid_va.max())},
        "n_pairs": len(rows),
        "means": {name: {"on_manifold": float(np.mean([r[f"{name}_onmanifold"] for r in rows])),
                         "va_track_predicted": float(np.mean([r[f"{name}_vatrack"] for r in rows]))}
                  for name in ("linear", "ambient_geo", "va_spline", "bij_spline")},
        "per_pair": rows,
    }
    (out / "_summary.json").write_text(json.dumps(summary, indent=2))

    # figure: PC1xPC2 paths for a few illustrative pairs
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    illus = [("happy", "sad"), ("excited", "weary"), ("terrified", "serene")]
    for ax, (s, e) in zip(axes, illus):
        if s not in idx or e not in idx:
            continue
        i, j = idx[s], idx[e]
        ax.scatter(C[:, 0], C[:, 1], s=8, c="lightgray")
        styles = {"linear": ("0.5", "--"), "va_spline": ("tab:red", "-"), "bij_spline": ("tab:green", "-")}
        for name, (col, ls) in styles.items():
            if name == "linear":
                p = linear_interpolation(C[i], C[j], K)
            elif name == "va_spline":
                p = fit_spline_geodesic(va_spline, va[i], va[j], metric="induced", num_waypoints=K,
                                        snap_start=C[i], snap_end=C[j]).waypoints
            else:
                p = fit_spline_geodesic(bij, u[i], u[j], metric="induced", num_waypoints=K,
                                        snap_start=C[i], snap_end=C[j]).waypoints
            ax.plot(p[:, 0], p[:, 1], ls, color=col, lw=2, label=name)
        ax.set_title(f"{s}->{e}"); ax.legend(fontsize=8)
    fig.suptitle("Bijective (diffusion) vs V/A spline vs linear — PC1xPC2")
    fig.tight_layout(); fig.savefig(out / "paths.png", dpi=120); plt.close(fig)
    print(f"\nsaved {out / '_summary.json'} and paths.png")


if __name__ == "__main__":
    main()
