"""Unit tests for the un-selected n=40 surrogate-validation scripts.

The scripts under scripts/ are not an importable package, so we load the two
modules by path with importlib and test their pure (CPU-only, no-network) helpers
plus the end-to-end analysis math on synthetic per-pair data.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]


def _load(path: Path, name: str) -> ModuleType:
    # scripts/experiments must be on sys.path for the run_composition_experiment import.
    sys.path.insert(0, str(_REPO / "scripts" / "experiments"))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def n40():
    return _load(_REPO / "scripts" / "experiments" / "validate_surrogate_n40.py", "vsn40")


@pytest.fixture(scope="module")
def analyze():
    return _load(_REPO / "scripts" / "analysis" / "analyze_surrogate_n40.py", "asn40")


def test_slug_handles_spaces(n40) -> None:
    assert n40._slug("at ease", "obstinate") == "at-ease__obstinate"
    assert n40._slug("hope", "unhappy") == "hope__unhappy"


def test_serialize_roundtrip(n40) -> None:
    from manifold_emotions.steering.trajectory import SteeredContinuation

    conts = [SteeredContinuation(waypoint_index=i, prompt_index=j, text=f"t{i}{j}",
                                 finish_reason="stop")
             for i in range(3) for j in range(2)]
    back = n40._deserialize(n40._serialize(conts))
    assert [(c.waypoint_index, c.prompt_index, c.text, c.finish_reason) for c in back] == \
           [(c.waypoint_index, c.prompt_index, c.text, c.finish_reason) for c in conts]


def test_mean_dist_ignores_nonfinite(n40) -> None:
    beh = np.array([[3.0, 3.0], [np.nan, np.nan], [4.0, 4.0]])
    targets = np.array([[3.0, 3.0], [0.0, 0.0], [3.0, 3.0]])
    # only rows 0 and 2 finite: distances 0 and sqrt(2)
    assert n40._mean_dist(beh, targets) == pytest.approx((0.0 + 2 ** 0.5) / 2)


def test_coherent_frac(n40) -> None:
    assert n40._coherent_frac(["coherent", "mixed", "coherent", "absent"]) == 0.5
    assert np.isnan(n40._coherent_frac([]))


def test_index_ratings_strips_prefix(n40) -> None:
    from manifold_emotions.behavior.judge_text import TextRating

    ratings = {
        "opt_hope_unhappy_wp000_p00": TextRating("x", 5.0, 2.0),
        "lin_hope_unhappy_wp000_p00": TextRating("x", 1.0, 6.0),
    }
    idx = n40._index_ratings(ratings, "opt", "hope", "unhappy")
    assert idx == {"wp000_p00": (5.0, 2.0)}


def test_analysis_math_end_to_end(analyze, tmp_path) -> None:
    # Synthetic: optimized clearly beats linear, surrogate slightly optimistic,
    # optimized a bit less coherent than linear.
    rng = np.random.default_rng(0)
    pairs = [f"e{i}->f{i}" for i in range(40)]
    per_pair, summ_pp = [], []
    for p in pairs:
        lin_d = 2.0 + rng.normal(0, 0.2)
        opt_d = 1.0 + rng.normal(0, 0.2)
        promised = opt_d - 0.05
        per_pair.append({
            "pair": p,
            "optimized_actual_dist": opt_d,
            "linear_actual_dist": lin_d,
            "real_headroom": lin_d - opt_d,
            "surrogate_predicted_dist": promised,
            "surrogate_optimism": opt_d - promised,
            "opt_coherent_frac": 0.6,
            "lin_coherent_frac": 0.7,
            "coherence_gap": -0.1,
            "n_opt": 300, "n_lin": 300,
        })
        summ_pp.append({"pair": p, "headroom_pred": (lin_d - opt_d) + rng.normal(0, 0.1)})

    res_path = tmp_path / "validation_results_n40.json"
    summ_path = tmp_path / "_summary_n40.json"
    out_path = tmp_path / "analysis_n40.json"
    res_path.write_text(json.dumps({"per_pair": per_pair}))
    summ_path.write_text(json.dumps({"per_pair": summ_pp}))

    argv = ["analyze", "--results", str(res_path), "--surrogate-summary", str(summ_path),
            "--out", str(out_path)]
    old = sys.argv
    sys.argv = argv
    try:
        analyze.main()
    finally:
        sys.argv = old

    out = json.loads(out_path.read_text())
    assert out["n_pairs"] == 40
    hr = out["population_real_headroom"]
    assert hr["mean"] == pytest.approx(1.0, abs=0.15)
    assert hr["wins"] == 40
    assert hr["wilcoxon_p_greater"] < 0.001
    cal = out["calibration_out_of_selection"]
    assert cal["pearson_r_pred_vs_actual_headroom"] > 0.5  # noisy but correlated
    assert cal["mean_surrogate_optimism"] == pytest.approx(0.05, abs=0.02)
    cg = out["coherence_gap_opt_minus_lin"]
    assert cg["mean"] == pytest.approx(-0.1, abs=1e-6)
