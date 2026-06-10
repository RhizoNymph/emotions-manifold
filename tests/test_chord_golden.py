"""Golden regression: unified chord geometry must match archived results.

The refactor from per-variant scripts to ``experiments.chord`` must not
change the science. The geometry half of the pipeline (PCA manifold,
pullback weights, geodesic, path lengths) is deterministic, so we
recompute it for a pair from the archived 4-D run and compare against
the stored summary JSON.

Requires local data artifacts (gitignored); skipped when absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from manifold_emotions.behavior.manifold import BehaviorManifold
from manifold_emotions.manifold.fit import FittedManifold
from manifold_emotions.manifold.pullback import compute_pullback

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFOLD_4D = REPO_ROOT / "data/manifold_h_4d_full.npz"
MANIFOLD_Y = REPO_ROOT / "data/manifold_y.npz"
ARCHIVED_SUMMARY = REPO_ROOT / "results/pullback_4d/happy_sad.json"

pytestmark = pytest.mark.skipif(
    not (MANIFOLD_4D.exists() and MANIFOLD_Y.exists() and ARCHIVED_SUMMARY.exists()),
    reason="gitignored data/results artifacts not present in this checkout",
)


def test_geometry_matches_archived_4d_run() -> None:
    archived = json.loads(ARCHIVED_SUMMARY.read_text())
    manifold = FittedManifold.load(MANIFOLD_4D)
    behavior = BehaviorManifold.load(MANIFOLD_Y)

    g = compute_pullback(
        manifold, behavior,
        archived["pair"][0], archived["pair"][1],
        num_waypoints=archived["num_waypoints"],
        sigma=None,
    )

    assert g.sigma == pytest.approx(archived["sigma"], rel=1e-9)
    assert g.sigma_per_waypoint.tolist() == pytest.approx(
        archived["sigma_per_waypoint"], rel=1e-9
    )

    old = archived["geometry"]
    assert g.pullback_length == pytest.approx(old["pullback_length"], rel=1e-5)
    assert g.geodesic_length == pytest.approx(old["geodesic_length"], rel=1e-5)
    assert g.linear_length == pytest.approx(old["linear_length"], rel=1e-5)
    assert g.mean_dist_to_geodesic == pytest.approx(
        old["mean_dist_pullback_to_geodesic"], rel=1e-5
    )
    assert g.mean_dist_to_linear == pytest.approx(
        old["mean_dist_pullback_to_linear"], rel=1e-5
    )
    assert g.closer_to == old["closer_to"]
    assert g.my_path[:, 0].tolist() == pytest.approx(old["my_path_valence"], rel=1e-6)
    assert g.my_path[:, 1].tolist() == pytest.approx(old["my_path_arousal"], rel=1e-6)
