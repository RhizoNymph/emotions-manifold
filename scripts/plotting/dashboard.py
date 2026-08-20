"""Interactive Dash app to explore M_h and M_y manifolds.

Three projection methods for the 8-D M_h subspace down to 3-D for
display:

- ``PCA``: pick any three of the 8 principal components as the
  display axes. Since the subspace coords are already PCA-projected,
  this is exact (no further dimensionality reduction).
- ``UMAP``: nonlinear projection fitted on the 30 centroids, transforms
  geodesic and pullback waypoints through the same map.
- ``Random``: random orthogonal 3-D rotation of the 8-D space — useful
  to confirm that interesting structure isn't a PC-1/2 artifact.

For any selected (start, end) pair the dashboard overlays any of:

- The M_h geodesic (ambient G_E, from precomputed cache)
- The M_h linear chord
- The M_h kernel-barycenter pullback of the M_y geodesic
- Spline surface-geodesics, computed live from the saved SplineManifold
  artifacts (``data/manifold_spline_*.npz``) and drawn dashed:
    * Bijective spline (diffusion-2 coordinate) — induced + density metrics,
      the faithful Goodfire construction that wins target-tracking. 8-D only.
    * V/A spline (lossy coordinate) — the failure mode, in red.
  Only the methods whose artifact exists at the current dim are offered; when a
  pair was in the day-10 judged run, the info card also shows how the spline
  actually steered vs linear (off-M_y and M_y-line gaps).
- (Optional) an iso-surface of the marginal KDE density in the
  3-D projection, so you can see where the manifold lives.

Side panel shows the same pair in M_y (V × A) with the chord plus the
two endpoints, and a metrics card with G_E gap, max deflection,
predicted off-M_y energy, and the measured Δ if we have one.

Run with:
    uv run python scripts/dashboard.py
    # then open http://localhost:8050

Optional flags:
    --port 8080         # different port
    --no-precompute     # error if cache missing instead of building it
    --manifold-dim 4    # load the 4-D variant (requires data/manifold_h_4d_full.npz
                        # and data/geodesics_cache_4d.npz from setup_4d_pipeline.py)

Side-by-side comparison (8-D vs 4-D):
    uv run python scripts/dashboard.py --port 8050                   # 8-D production
    uv run python scripts/dashboard.py --port 8051 --manifold-dim 4  # 4-D
Open both URLs in separate browser windows and select the same pair to
flip between regimes.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
from sklearn.decomposition import PCA

from manifold_emotions.behavior.manifold import BehaviorManifold
from manifold_emotions.config import load_config
from manifold_emotions.manifold.fit import FittedManifold
from manifold_emotions.manifold.geodesic import linear_interpolation
from manifold_emotions.manifold.pullback import construct_pullback_path
from manifold_emotions.manifold.spline import SplineManifold
from manifold_emotions.manifold.spline_geodesic import fit_spline_geodesic

GEODESIC_CACHE = Path("data/geodesics_cache.npz")
PAIR_ALIGNMENT = Path("results/pair_alignment.json")

# Optional 4-D variant (built by scripts/setup_4d_pipeline.py).
MANIFOLD_4D_PATH = Path("data/manifold_h_4d_full.npz")
GEODESIC_CACHE_4D = Path("data/geodesics_cache_4d.npz")

# Saved bijective-spline behavioral run (the day-10 judged condition), keyed by
# pair — used to annotate the info card with the "how did it actually steer"
# numbers when the selected pair was in that experiment.
SPLINE_BEHAVIORAL_DIR = Path("results/pullback_spline_bijective_8d")

PATH_COLORS = {
    "geodesic": "#0066cc",
    "linear":   "#888888",
    "pullback": "#9933cc",
    # Spline geodesics computed live from the saved SplineManifold artifacts.
    # Bijective (diffusion-2 coord) = the faithful Goodfire method that wins;
    # V/A (lossy coord) = the failure mode, colored red so it reads as "wanders".
    "spline_bijective_induced": "#e8590c",
    "spline_bijective_density": "#f59f00",
    "spline_va_induced":        "#e03131",
}

# Registry of spline methods to offer, per manifold dim. Each entry loads one
# SplineManifold artifact and a surface-metric; only entries whose artifact file
# exists on disk are shown. The dashboard computes the surface geodesic live
# (fit_spline_geodesic) and projects it into the same 3-D view as the other paths.
# ``key`` doubles as the checklist value and the PATH_COLORS lookup.
_SPLINE_SPECS: dict[int, list[dict]] = {
    8: [
        {
            "key": "spline_bijective_induced",
            "label": " Bijective spline · induced  (faithful Goodfire, wins)",
            "artifact": Path("data/manifold_spline_bijective_8d.npz"),
            "metric": "induced",
        },
        {
            "key": "spline_bijective_density",
            "label": " Bijective spline · density",
            "artifact": Path("data/manifold_spline_bijective_8d.npz"),
            "metric": "density",
        },
        {
            "key": "spline_va_induced",
            "label": " V/A spline · induced  (lossy coord, fails)",
            "artifact": Path("data/manifold_spline_8d.npz"),
            "metric": "induced",
        },
    ],
    4: [
        {
            "key": "spline_va_induced",
            "label": " V/A spline · induced  (lossy coord)",
            "artifact": Path("data/manifold_spline_4d.npz"),
            "metric": "induced",
        },
    ],
}


def _load_spline_methods(dim: int) -> dict[str, tuple[SplineManifold, str]]:
    """Load every available spline artifact for ``dim`` into {key: (spline, metric)}.

    Silently skips specs whose artifact file is missing, so the UI only offers
    methods that can actually be drawn (bijective is 8-D-only, for instance).
    """
    out: dict[str, tuple[SplineManifold, str]] = {}
    for spec in _SPLINE_SPECS.get(dim, []):
        if spec["artifact"].exists():
            out[spec["key"]] = (SplineManifold.load(spec["artifact"]), spec["metric"])
    return out


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_geodesic_cache(
    cache_path: Path = GEODESIC_CACHE,
) -> tuple[np.ndarray, dict[tuple[int, int], int]]:
    """Load precomputed geodesic waypoints, return (waypoints, idx_map)."""
    data = np.load(cache_path, allow_pickle=True)
    waypoints = data["waypoints"]
    pair_indices = data["pair_indices"]
    idx_map: dict[tuple[int, int], int] = {}
    for k, (i, j) in enumerate(pair_indices):
        idx_map[(int(i), int(j))] = k
    return waypoints, idx_map


def _load_measured_deltas() -> dict[frozenset[str], float]:
    """Aggregate Δ values from every measurement source we have."""
    out: dict[frozenset[str], float] = {}

    multipair = Path("results/steering_multipair.json")
    if multipair.exists():
        for row in json.loads(multipair.read_text())["pairs"]:
            out[frozenset({row["start"], row["end"]})] = (
                row["delta_linear_minus_manifold"]
            )

    for path in Path("results/subspace_sweep").glob("*_dim08.json"):
        row = json.loads(path.read_text())
        s, e = row["pair"]
        out[frozenset({s, e})] = row["delta_linear_minus_manifold"]

    for path in Path("results/pair_validation").glob("*.json"):
        row = json.loads(path.read_text())
        s, e = row["pair"]
        out[frozenset({s, e})] = row["delta_linear_minus_manifold"]

    return out


def _load_alignment_predictions() -> dict[frozenset[str], dict]:
    if not PAIR_ALIGNMENT.exists():
        return {}
    out: dict[frozenset[str], dict] = {}
    for r in json.loads(PAIR_ALIGNMENT.read_text()):
        out[frozenset({r["start"], r["end"]})] = r
    return out


def _load_spline_behavioral() -> dict[frozenset[str], dict]:
    """Judged per-waypoint metrics from the saved bijective-spline run.

    Returns {pair -> {method -> {off_manifold_energy, my_geodesic_distance}}} for
    the pairs that were in the day-10 behavioral experiment, so the info card can
    show how the spline trajectory actually steered vs linear.
    """
    out: dict[frozenset[str], dict] = {}
    if not SPLINE_BEHAVIORAL_DIR.exists():
        return out
    for path in SPLINE_BEHAVIORAL_DIR.glob("*.json"):
        try:
            row = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        start, end = row["pair"]
        trajs = row.get("trajectories", {})
        out[frozenset({start, end})] = {
            name: {
                "off_manifold_energy": t.get("off_manifold_energy"),
                "my_geodesic_distance": t.get("my_geodesic_distance"),
            }
            for name, t in trajs.items()
        }
    return out


def _nn_distances(centroids: np.ndarray) -> np.ndarray:
    diffs = centroids[:, None, :] - centroids[None, :, :]
    dists = np.sqrt(np.sum(diffs * diffs, axis=-1))
    np.fill_diagonal(dists, np.inf)
    return dists.min(axis=1)


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


def project_pca(points: np.ndarray, axes: tuple[int, int, int]) -> np.ndarray:
    """Pick three PC axes from the already-PCA'd subspace."""
    return points[:, list(axes)]


def project_umap(
    points: np.ndarray, fit_points: np.ndarray, _cache: dict
) -> np.ndarray:
    """Fit UMAP on `fit_points` (centroids), transform `points`.

    Caches the fitted reducer keyed by id(fit_points) so successive
    callbacks reuse the same projection.
    """
    import umap

    key = id(fit_points)
    if key not in _cache:
        reducer = umap.UMAP(n_components=3, n_neighbors=8, min_dist=0.5,
                            random_state=42)
        reducer.fit(fit_points)
        _cache[key] = reducer
    return _cache[key].transform(points)


def project_random(
    points: np.ndarray, seed: int = 0
) -> np.ndarray:
    """Random orthogonal projection of the subspace into 3-D."""
    rng = np.random.default_rng(seed)
    d = points.shape[1]
    M = rng.normal(size=(d, 3))
    Q, _ = np.linalg.qr(M)
    return points @ Q


# ---------------------------------------------------------------------------
# Density (marginal KDE on the selected 3-D projection)
# ---------------------------------------------------------------------------


def kde_marginal_3d(
    projected_centroids: np.ndarray,
    bandwidth: float,
    grid_size: int = 24,
    pad: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Marginal Gaussian KDE on the projected centroids.

    Returns (xs, ys, zs, density_volume) where density_volume is a
    grid_size^3 array of normalized density values.
    """
    mins = projected_centroids.min(axis=0)
    maxs = projected_centroids.max(axis=0)
    span = maxs - mins
    pad_amt = pad * span
    lo, hi = mins - pad_amt, maxs + pad_amt
    xs = np.linspace(lo[0], hi[0], grid_size)
    ys = np.linspace(lo[1], hi[1], grid_size)
    zs = np.linspace(lo[2], hi[2], grid_size)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    grid = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)

    # Marginalized KDE: each centroid contributes a Gaussian on the
    # projected dims with the same bandwidth (correct for projection
    # onto an orthonormal subset of axes when the original kernel is
    # isotropic).
    diffs = grid[:, None, :] - projected_centroids[None, :, :]
    sq = np.sum(diffs * diffs, axis=-1)
    contrib = np.exp(-sq / (2.0 * bandwidth * bandwidth))
    density = contrib.sum(axis=1).reshape(grid_size, grid_size, grid_size)
    density /= density.max()
    return xs, ys, zs, density


# ---------------------------------------------------------------------------
# Figure construction
# ---------------------------------------------------------------------------


def make_mh_figure(
    centroids_proj: np.ndarray,
    labels: tuple[str, ...],
    color_values: np.ndarray | None,
    color_label: str,
    geodesic_proj: np.ndarray | None,
    linear_proj: np.ndarray | None,
    pullback_proj: np.ndarray | None,
    density: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None,
    axis_labels: tuple[str, str, str],
    extra_paths: dict[str, np.ndarray] | None = None,
) -> go.Figure:
    traces: list[go.BaseTraceType] = []

    if density is not None:
        xs, ys, zs, vol = density
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
        traces.append(
            go.Volume(
                x=gx.flatten(), y=gy.flatten(), z=gz.flatten(),
                value=vol.flatten(),
                opacity=0.05, surface_count=8,
                colorscale="Viridis",
                showscale=False,
                isomin=0.05, isomax=1.0,
            )
        )

    marker_kwargs: dict = {"size": 6, "line": {"color": "black", "width": 0.5}}
    if color_values is not None:
        marker_kwargs["color"] = color_values
        marker_kwargs["colorscale"] = "RdYlBu_r"
        marker_kwargs["showscale"] = True
        marker_kwargs["colorbar"] = {"title": color_label, "x": 1.02}
    else:
        marker_kwargs["color"] = "#444444"

    traces.append(
        go.Scatter3d(
            x=centroids_proj[:, 0],
            y=centroids_proj[:, 1],
            z=centroids_proj[:, 2],
            text=list(labels),
            hoverinfo="text",
            mode="markers+text",
            marker=marker_kwargs,
            textposition="top center",
            textfont={"size": 9, "color": "#222"},
            name="centroids",
        )
    )

    def _path_trace(pts: np.ndarray, name: str) -> go.Scatter3d:
        color = PATH_COLORS.get(name, "#333333")
        # Spline geodesics get a dashed line so they read as distinct from the
        # ambient geodesic/linear/pullback even when several overlap.
        is_spline = name.startswith("spline_")
        return go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="lines+markers",
            line={"color": color, "width": 5,
                  "dash": "dash" if is_spline else "solid"},
            marker={"size": 3, "color": color},
            name=name,
        )

    if linear_proj is not None:
        traces.append(_path_trace(linear_proj, "linear"))
    if geodesic_proj is not None:
        traces.append(_path_trace(geodesic_proj, "geodesic"))
    if pullback_proj is not None:
        traces.append(_path_trace(pullback_proj, "pullback"))
    for name, pts in (extra_paths or {}).items():
        traces.append(_path_trace(pts, name))

    fig = go.Figure(data=traces)
    fig.update_layout(
        scene={
            "xaxis_title": axis_labels[0],
            "yaxis_title": axis_labels[1],
            "zaxis_title": axis_labels[2],
            "aspectmode": "data",
        },
        margin={"l": 0, "r": 0, "b": 0, "t": 0},
        legend={"x": 0.02, "y": 0.98},
        uirevision="constant",  # preserve user camera angle across updates
    )
    return fig


def make_my_figure(
    behavior: BehaviorManifold,
    start: str | None,
    end: str | None,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=behavior.centroids[:, 0], y=behavior.centroids[:, 1],
            mode="markers+text",
            text=list(behavior.labels),
            textposition="top center",
            textfont={"size": 9, "color": "#333"},
            marker={"size": 8, "color": "#888"},
            name="emotions",
            hoverinfo="text",
        )
    )
    if start in behavior.labels and end in behavior.labels:
        y0 = behavior.centroids[behavior.labels.index(start)]
        y1 = behavior.centroids[behavior.labels.index(end)]
        fig.add_trace(
            go.Scatter(
                x=[y0[0], y1[0]], y=[y0[1], y1[1]],
                mode="lines",
                line={"color": "black", "width": 2},
                name="M_y chord",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[y0[0]], y=[y0[1]],
                mode="markers",
                marker={"size": 14, "color": "green"},
                name="start",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[y1[0]], y=[y1[1]],
                mode="markers",
                marker={"size": 14, "color": "red", "symbol": "x"},
                name="end",
            )
        )
    fig.update_layout(
        xaxis_title="valence",
        yaxis_title="arousal",
        xaxis_range=[1, 7],
        yaxis_range=[1, 7],
        margin={"l": 40, "r": 10, "b": 30, "t": 30},
        legend={"x": 0.02, "y": 0.02},
    )
    return fig


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


def build_app(
    manifold_path: Path | None = None,
    geodesic_cache_path: Path = GEODESIC_CACHE,
) -> Dash:
    config = load_config()
    manifold = FittedManifold.load(
        manifold_path if manifold_path is not None else config.paths.manifold_h
    )
    behavior = BehaviorManifold.load(config.paths.manifold_y)

    waypoints_cache, idx_map = _load_geodesic_cache(geodesic_cache_path)
    measured = _load_measured_deltas()
    predictions = _load_alignment_predictions()
    spline_behavioral = _load_spline_behavioral()

    centroids = manifold.centroids_subspace.astype(np.float64)
    labels = manifold.labels
    n = len(labels)
    nn_dists = _nn_distances(centroids)
    var_ratio = manifold.pca_explained_variance_ratio

    my_by_label = {lab: behavior.centroids[k] for k, lab in enumerate(behavior.labels)}

    umap_cache: dict[int, object] = {}

    dim = centroids.shape[1]

    # Spline methods available at this dim (bijective is 8-D-only). Computed
    # geodesics are memoized per (start, end, key) so re-selecting is instant.
    spline_methods = _load_spline_methods(dim)
    spline_specs = [s for s in _SPLINE_SPECS.get(dim, []) if s["key"] in spline_methods]
    spline_path_cache: dict[tuple[str, str, str], np.ndarray] = {}

    def _spline_waypoints(start: str, end: str, key: str) -> np.ndarray | None:
        """Live surface geodesic (K, d) for a pair, memoized. None if unavailable."""
        if key not in spline_methods:
            return None
        ck = (start, end, key)
        if ck in spline_path_cache:
            return spline_path_cache[ck]
        spline, metric = spline_methods[key]
        sidx = {lab: i for i, lab in enumerate(spline.labels)}
        if start not in sidx or end not in sidx:
            return None
        i, j = sidx[start], sidx[end]
        res = fit_spline_geodesic(
            spline, spline.control_coords[i], spline.control_coords[j],
            metric=metric, num_waypoints=30,
            snap_start=spline.centroids_subspace[i],
            snap_end=spline.centroids_subspace[j],
        )
        spline_path_cache[ck] = res.waypoints.astype(np.float64)
        return spline_path_cache[ck]
    app = Dash(__name__,
               title=f"Manifold Emotions Explorer ({dim}-D M_h)")
    # Zero the browser's default body margin so the 100vh layout fills the
    # window exactly (no stray outer scrollbar).
    app.index_string = app.index_string.replace(
        "<body>", '<body style="margin:0">'
    )

    pca_axis_options = [
        {"label": f"PC{i+1} ({var_ratio[i]*100:.1f}%)", "value": i}
        for i in range(centroids.shape[1])
    ]

    emotion_options = [{"label": lab, "value": lab} for lab in labels]

    app.layout = html.Div([
        html.Div([
            html.H2(f"Manifold Emotions Explorer  ·  {dim}-D M_h",
                    style={"margin": "0", "fontFamily": "system-ui"}),
            html.Div(
                f"{dim}-D M_h subspace, {n} emotions, "
                f"bandwidth={manifold.kde_bandwidth:.2f}, "
                f"cumulative variance={float(var_ratio.sum())*100:.1f}%",
                style={"color": "#666", "fontSize": "13px"},
            ),
        ], style={"padding": "10px 20px", "borderBottom": "1px solid #ddd",
                  "flex": "0 0 auto"}),

        html.Div([
            # Left: 3D M_h — fills the full height of the content row.
            html.Div([
                dcc.Graph(id="mh-graph", responsive=True,
                          style={"height": "100%", "width": "100%"}),
            ], style={"flex": "1 1 60%", "height": "100%", "minWidth": "0"}),

            # Right: controls + M_y + info. Scrolls independently if the
            # controls are taller than the viewport.
            html.Div([
                html.Div([
                    html.Label("Projection", style={"fontWeight": "bold"}),
                    dcc.RadioItems(
                        id="proj-method",
                        options=[
                            {"label": " PCA (pick 3 axes)", "value": "pca"},
                            {"label": " UMAP (3D)", "value": "umap"},
                            {"label": " Random orthogonal", "value": "random"},
                        ],
                        value="pca",
                        labelStyle={"display": "block", "marginBottom": "2px"},
                    ),
                ], style={"marginBottom": "10px"}),

                html.Div([
                    html.Label("PCA axes",
                               style={"fontWeight": "bold", "fontSize": "12px"}),
                    html.Div([
                        dcc.Dropdown(id="pca-x", options=pca_axis_options, value=0,
                                     clearable=False, style={"width": "120px"}),
                        dcc.Dropdown(id="pca-y", options=pca_axis_options, value=1,
                                     clearable=False, style={"width": "120px"}),
                        dcc.Dropdown(id="pca-z", options=pca_axis_options, value=2,
                                     clearable=False, style={"width": "120px"}),
                    ], style={"display": "flex", "gap": "5px"}),
                ], id="pca-axis-row", style={"marginBottom": "10px"}),

                html.Div([
                    html.Label("Color centroids by",
                               style={"fontWeight": "bold"}),
                    dcc.RadioItems(
                        id="color-by",
                        options=[
                            {"label": " None", "value": "none"},
                            {"label": " Valence", "value": "v"},
                            {"label": " Arousal", "value": "a"},
                            {"label": " NN distance (isolation)", "value": "iso"},
                        ],
                        value="v",
                        labelStyle={"display": "block", "marginBottom": "2px"},
                    ),
                ], style={"marginBottom": "10px"}),

                html.Div([
                    html.Label("Pair",
                               style={"fontWeight": "bold"}),
                    html.Div([
                        dcc.Dropdown(id="start-emotion", options=emotion_options,
                                     value="depressed",
                                     clearable=False, style={"width": "150px"}),
                        html.Span("→", style={"margin": "0 6px",
                                              "alignSelf": "center"}),
                        dcc.Dropdown(id="end-emotion", options=emotion_options,
                                     value="energized",
                                     clearable=False, style={"width": "150px"}),
                    ], style={"display": "flex"}),
                ], style={"marginBottom": "10px"}),

                html.Div([
                    html.Label("Pullback σ (kernel bandwidth in M_y)",
                               style={"fontWeight": "bold"}),
                    dcc.Slider(
                        id="pullback-sigma",
                        min=-2.0, max=0.5, step=0.1, value=0.0,
                        marks={
                            -2.0: "0.01",
                            -1.3: "0.05",
                            -1.1: "0.077",  # 171-default
                            -0.5: "0.3",
                            0.0: "1.0",     # 171 sweep optimum
                            0.3: "2.0",
                            0.5: "3.2",
                        },
                        tooltip={"placement": "bottom",
                                 "always_visible": False},
                    ),
                    html.Div(id="sigma-readout",
                             style={"fontSize": "11px", "color": "#666",
                                    "marginTop": "2px"}),
                ], style={"marginBottom": "10px"}),

                html.Div([
                    html.Label("Show paths", style={"fontWeight": "bold"}),
                    dcc.Checklist(
                        id="show-paths",
                        options=[
                            {"label": " Geodesic (ambient G_E)", "value": "geodesic"},
                            {"label": " Linear", "value": "linear"},
                            {"label": " Pullback (M_y geodesic → M_h)",
                             "value": "pullback"},
                            *[{"label": s["label"], "value": s["key"]}
                              for s in spline_specs],
                        ],
                        value=["geodesic", "linear"],
                        labelStyle={"display": "block", "marginBottom": "2px"},
                    ),
                    html.Div(
                        "Spline paths are surface geodesics computed live and "
                        "shown dashed. Bijective = diffusion-2 coordinate "
                        "(faithful Goodfire); V/A = the lossy coordinate."
                        if spline_specs else
                        "(No spline artifacts found for this dim — run "
                        "scripts/fit_spline_manifold.py to enable them.)",
                        style={"fontSize": "11px", "color": "#666",
                               "marginTop": "3px"},
                    ),
                ], style={"marginBottom": "10px"}),

                html.Div([
                    html.Label("Density isosurface", style={"fontWeight": "bold"}),
                    dcc.Checklist(
                        id="show-density",
                        options=[{"label": " Show marginal KDE in 3D",
                                  "value": "yes"}],
                        value=[],
                    ),
                ], style={"marginBottom": "10px"}),

                html.Hr(),

                html.Div(id="pair-info-card",
                         style={"fontSize": "13px", "fontFamily": "system-ui"}),

                html.Hr(),

                dcc.Graph(id="my-graph", style={"height": "320px"}),

            ], style={"flex": "0 0 39%", "height": "100%",
                      "overflowY": "auto", "padding": "10px 20px",
                      "boxSizing": "border-box"}),
        ], style={"display": "flex", "flex": "1 1 auto", "minHeight": "0"}),
    ], style={"display": "flex", "flexDirection": "column",
              "height": "100vh", "margin": "0"})

    @app.callback(
        Output("sigma-readout", "children"),
        Input("pullback-sigma", "value"),
    )
    def _sigma_label(log10_sigma: float) -> str:
        sigma = 10 ** log10_sigma
        # Note the regime: narrow / default / sweet-spot / mean-collapse
        if sigma < 0.08:
            tag = "narrow (snapping)"
        elif sigma < 0.5:
            tag = "intermediate"
        elif sigma < 1.5:
            tag = "171 sweet spot"
        else:
            tag = "wide (mean-collapse)"
        return f"σ = {sigma:.3f}  ({tag})"

    @app.callback(
        Output("pca-axis-row", "style"),
        Input("proj-method", "value"),
    )
    def _toggle_pca_axes(method: str) -> dict:
        if method == "pca":
            return {"marginBottom": "10px"}
        return {"display": "none"}

    @app.callback(
        Output("mh-graph", "figure"),
        Output("my-graph", "figure"),
        Output("pair-info-card", "children"),
        Input("proj-method", "value"),
        Input("pca-x", "value"), Input("pca-y", "value"), Input("pca-z", "value"),
        Input("color-by", "value"),
        Input("start-emotion", "value"),
        Input("end-emotion", "value"),
        Input("show-paths", "value"),
        Input("show-density", "value"),
        Input("pullback-sigma", "value"),
    )
    def _update(
        method: str,
        pcx: int, pcy: int, pcz: int,
        color_by: str,
        start: str, end: str,
        show_paths: list[str],
        show_density: list[str],
        sigma_log10: float,
    ):
        sigma = float(10 ** sigma_log10)
        # Resolve projection
        if method == "pca":
            axes = (int(pcx), int(pcy), int(pcz))
            centroids_proj = project_pca(centroids, axes)
            axis_labels = (
                f"PC{axes[0]+1}", f"PC{axes[1]+1}", f"PC{axes[2]+1}",
            )

            def project(points: np.ndarray) -> np.ndarray:
                return project_pca(points, axes)
        elif method == "umap":
            centroids_proj = project_umap(centroids, centroids, umap_cache)
            axis_labels = ("UMAP 1", "UMAP 2", "UMAP 3")

            def project(points: np.ndarray) -> np.ndarray:
                return project_umap(points, centroids, umap_cache)
        else:  # random
            centroids_proj = project_random(centroids)
            axis_labels = ("rand 1", "rand 2", "rand 3")

            def project(points: np.ndarray) -> np.ndarray:
                return project_random(points)

        # Color values
        color_values: np.ndarray | None = None
        color_label = ""
        if color_by == "v":
            color_values = np.array(
                [my_by_label[lab][0] if lab in my_by_label else np.nan
                 for lab in labels]
            )
            color_label = "valence"
        elif color_by == "a":
            color_values = np.array(
                [my_by_label[lab][1] if lab in my_by_label else np.nan
                 for lab in labels]
            )
            color_label = "arousal"
        elif color_by == "iso":
            color_values = nn_dists
            color_label = "NN dist"

        # Paths
        start_idx = labels.index(start) if start in labels else 0
        end_idx = labels.index(end) if end in labels else 1

        geo_proj: np.ndarray | None = None
        lin_proj: np.ndarray | None = None
        pull_proj: np.ndarray | None = None
        extra_proj: dict[str, np.ndarray] = {}

        if start_idx != end_idx:
            i, j = (start_idx, end_idx) if start_idx < end_idx else (end_idx, start_idx)
            geo_full = waypoints_cache[idx_map[(i, j)]]
            # Reverse if user picked endpoints in opposite order
            if start_idx > end_idx:
                geo_full = geo_full[::-1]
            lin_full = linear_interpolation(
                centroids[start_idx].astype(np.float32),
                centroids[end_idx].astype(np.float32),
                geo_full.shape[0],
            )
            if "geodesic" in show_paths:
                geo_proj = project(geo_full)
            if "linear" in show_paths:
                lin_proj = project(lin_full)
            if "pullback" in show_paths and start in behavior.labels and end in behavior.labels:
                _, pull_sub, *_ = construct_pullback_path(
                    manifold, behavior, start, end,
                    num_waypoints=geo_full.shape[0], sigma=sigma,
                )
                pull_sub[0] = centroids[start_idx]
                pull_sub[-1] = centroids[end_idx]
                pull_proj = project(pull_sub.astype(np.float32))
            for key in spline_methods:
                if key in show_paths:
                    wp = _spline_waypoints(start, end, key)
                    if wp is not None:
                        extra_proj[key] = project(wp)

        # Density
        density = None
        if "yes" in show_density:
            density = kde_marginal_3d(
                centroids_proj, manifold.kde_bandwidth,
                grid_size=20, pad=0.3,
            )

        mh_fig = make_mh_figure(
            centroids_proj, labels, color_values, color_label,
            geo_proj, lin_proj, pull_proj, density, axis_labels,
            extra_paths=extra_proj,
        )
        my_fig = make_my_figure(behavior, start, end)
        card = _build_pair_card(
            start, end, predictions, measured, spline_behavioral,
        )
        return mh_fig, my_fig, card

    return app


def _build_pair_card(
    start: str, end: str,
    predictions: dict[frozenset[str], dict],
    measured: dict[frozenset[str], float],
    spline_behavioral: dict[frozenset[str], dict] | None = None,
):
    key = frozenset({start, end})
    pred = predictions.get(key)
    delta = measured.get(key)
    rows = [html.H4(f"{start} → {end}",
                    style={"margin": "0 0 6px 0", "fontFamily": "system-ui"})]
    if pred is None:
        rows.append(html.Div("(no structural prediction)",
                             style={"color": "#888"}))
    else:
        def _fmt(v):
            return "—" if v is None else f"{v:.3f}"

        gap = pred.get("ge_length_gap")
        maxdef = pred.get("max_chord_deflection")
        pred_e = pred.get("predicted_off_my_energy")
        pr = pred.get("participation_ratio")
        rows.extend([
            html.Div([html.B("G_E gap: "), _fmt(gap)]),
            html.Div([html.B("Max chord deflection: "), _fmt(maxdef)]),
            html.Div([html.B("Predicted off-M_y E: "), _fmt(pred_e)]),
            html.Div([html.B("Participation ratio: "), _fmt(pr)]),
        ])
    if delta is None:
        rows.append(html.Div("Not yet steered (no measured Δ)",
                             style={"color": "#888", "marginTop": "6px"}))
    else:
        winner = "manifold" if delta > 0.05 else (
            "linear" if delta < -0.05 else "tie"
        )
        rows.append(html.Div(
            [html.B(f"Measured Δ: {delta:+.3f}  "), f"(winner: {winner})"],
            style={"marginTop": "6px",
                   "color": ("#0066cc" if delta > 0.05 else
                             "#cc3300" if delta < -0.05 else "#666")},
        ))

    # Bijective-spline judged run: show how the spline trajectory actually
    # steered vs linear, when this pair was in that experiment. Lower is better
    # on both metrics, so a negative gap (spline − linear) means the spline won.
    beh = (spline_behavioral or {}).get(key)
    if beh and "linear" in beh:
        lin = beh["linear"]
        rows.append(html.Hr(style={"margin": "8px 0"}))
        rows.append(html.Div(
            html.B("Bijective-spline judged run (n=40)"),
            style={"marginBottom": "3px"},
        ))
        for name in ("spline_induced", "spline_density"):
            m = beh.get(name)
            if not m:
                continue
            d_off = _card_gap(m.get("off_manifold_energy"), lin.get("off_manifold_energy"))
            d_myl = _card_gap(m.get("my_geodesic_distance"), lin.get("my_geodesic_distance"))
            rows.append(html.Div(
                f"{name.replace('spline_', '')}: "
                f"off-M_y {d_off}  ·  M_y-line {d_myl}  (vs linear)",
                style={"fontSize": "12px", "color": "#333"},
            ))
        rows.append(html.Div(
            "negative = spline closer than linear",
            style={"fontSize": "10px", "color": "#888"},
        ))
    return rows


def _card_gap(method_val, linear_val) -> str:
    """Format ``method − linear`` gap for the card, or '—' if unavailable."""
    if method_val is None or linear_val is None:
        return "—"
    return f"{method_val - linear_val:+.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument(
        "--host", default="127.0.0.1",
        help=(
            "interface to bind. Default 127.0.0.1 (localhost only). "
            "Use 0.0.0.0 to listen on every interface for LAN access "
            "(e.g. http://<your-lan-ip>:8050 from another device). "
            "There is no authentication on the dashboard — anyone with "
            "network reach can see the manifold and Δ measurements."
        ),
    )
    parser.add_argument("--no-precompute", action="store_true")
    parser.add_argument(
        "--manifold-dim", type=int, default=8, choices=(4, 8),
        help=(
            "PCA subspace dimensionality to visualize. 8 (default) loads the "
            "production manifold + geodesics_cache.npz. 4 loads "
            "manifold_h_4d_full.npz + geodesics_cache_4d.npz (built by "
            "scripts/setup_4d_pipeline.py). Run two instances on different "
            "ports (e.g. 8050 and 8051) for side-by-side comparison."
        ),
    )
    args = parser.parse_args()

    if args.manifold_dim == 4:
        manifold_path = MANIFOLD_4D_PATH
        cache_path = GEODESIC_CACHE_4D
        if not manifold_path.exists() or not cache_path.exists():
            raise SystemExit(
                f"4-D artifacts missing at {manifold_path} and/or {cache_path}. "
                f"Run scripts/setup_4d_pipeline.py first."
            )
    else:
        manifold_path = None  # default to config.paths.manifold_h
        cache_path = GEODESIC_CACHE
        if not cache_path.exists():
            if args.no_precompute:
                raise SystemExit(
                    f"geodesic cache missing at {cache_path} and "
                    "--no-precompute was set; run scripts/precompute_geodesics.py"
                )
            print(f"no cache at {cache_path}; precomputing now ...")
            import subprocess
            subprocess.check_call(["uv", "run", "python",
                                   "scripts/precompute_geodesics.py"])

    app = build_app(manifold_path=manifold_path,
                    geodesic_cache_path=cache_path)
    app.run(debug=False, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
