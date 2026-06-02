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

For any selected (start, end) pair the dashboard overlays:

- The M_h geodesic (from precomputed cache)
- The M_h linear chord
- The M_h kernel-barycenter pullback of the M_y geodesic
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

GEODESIC_CACHE = Path("data/geodesics_cache.npz")
PAIR_ALIGNMENT = Path("results/pair_alignment.json")

PATH_COLORS = {
    "geodesic": "#0066cc",
    "linear":   "#888888",
    "pullback": "#9933cc",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_geodesic_cache() -> tuple[np.ndarray, dict[tuple[int, int], int]]:
    """Load precomputed geodesic waypoints, return (waypoints, idx_map)."""
    data = np.load(GEODESIC_CACHE, allow_pickle=True)
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
        return go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="lines+markers",
            line={"color": PATH_COLORS[name], "width": 5},
            marker={"size": 3, "color": PATH_COLORS[name]},
            name=name,
        )

    if linear_proj is not None:
        traces.append(_path_trace(linear_proj, "linear"))
    if geodesic_proj is not None:
        traces.append(_path_trace(geodesic_proj, "geodesic"))
    if pullback_proj is not None:
        traces.append(_path_trace(pullback_proj, "pullback"))

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


def build_app() -> Dash:
    config = load_config()
    manifold = FittedManifold.load(config.paths.manifold_h)
    behavior = BehaviorManifold.load(config.paths.manifold_y)

    waypoints_cache, idx_map = _load_geodesic_cache()
    measured = _load_measured_deltas()
    predictions = _load_alignment_predictions()

    centroids = manifold.centroids_subspace.astype(np.float64)
    labels = manifold.labels
    n = len(labels)
    nn_dists = _nn_distances(centroids)
    var_ratio = manifold.pca_explained_variance_ratio

    my_by_label = {lab: behavior.centroids[k] for k, lab in enumerate(behavior.labels)}

    umap_cache: dict[int, object] = {}

    app = Dash(__name__, title="Manifold Emotions Explorer")

    pca_axis_options = [
        {"label": f"PC{i+1} ({var_ratio[i]*100:.1f}%)", "value": i}
        for i in range(centroids.shape[1])
    ]

    emotion_options = [{"label": lab, "value": lab} for lab in labels]

    app.layout = html.Div([
        html.Div([
            html.H2("Manifold Emotions Explorer",
                    style={"margin": "0", "fontFamily": "system-ui"}),
            html.Div(
                f"8-D M_h subspace, {n} emotions, bandwidth={manifold.kde_bandwidth:.2f}",
                style={"color": "#666", "fontSize": "13px"},
            ),
        ], style={"padding": "10px 20px", "borderBottom": "1px solid #ddd"}),

        html.Div([
            # Left: 3D M_h
            html.Div([
                dcc.Graph(id="mh-graph", style={"height": "640px"}),
            ], style={"width": "60%", "display": "inline-block",
                      "verticalAlign": "top"}),

            # Right: controls + M_y + info
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
                            {"label": " Geodesic", "value": "geodesic"},
                            {"label": " Linear", "value": "linear"},
                            {"label": " Pullback (M_y geodesic → M_h)",
                             "value": "pullback"},
                        ],
                        value=["geodesic", "linear"],
                        labelStyle={"display": "block", "marginBottom": "2px"},
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

            ], style={"width": "39%", "display": "inline-block",
                      "verticalAlign": "top", "padding": "10px 20px",
                      "boxSizing": "border-box"}),
        ]),
    ])

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
        )
        my_fig = make_my_figure(behavior, start, end)
        card = _build_pair_card(
            start, end, predictions, measured,
        )
        return mh_fig, my_fig, card

    return app


def _build_pair_card(
    start: str, end: str,
    predictions: dict[frozenset[str], dict],
    measured: dict[frozenset[str], float],
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
    return rows


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
    args = parser.parse_args()

    if not GEODESIC_CACHE.exists():
        if args.no_precompute:
            raise SystemExit(
                f"geodesic cache missing at {GEODESIC_CACHE} and "
                "--no-precompute was set; run scripts/precompute_geodesics.py"
            )
        print(f"no cache at {GEODESIC_CACHE}; precomputing now ...")
        import subprocess
        subprocess.check_call(["uv", "run", "python",
                               "scripts/precompute_geodesics.py"])

    app = build_app()
    app.run(debug=False, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
