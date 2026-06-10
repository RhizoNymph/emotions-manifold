"""Tests for the shared bootstrap/paired-comparison statistics.

The two bootstrap functions replaced inline copies in analysis scripts;
the *_matches_legacy tests pin them to verbatim re-implementations of
those originals so regenerated _summary.json outputs stay bit-identical.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as scipy_stats

from manifold_emotions.analysis.stats import (
    bootstrap_ci,
    bootstrap_mean_ci,
    paired_gap_report,
)


def _legacy_bootstrap_ci(xs, fn, n_boot=10000, seed=42):
    # Verbatim copy of the function formerly in analyze_geodesic_vs_linear.py.
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_boot):
        idx = rng.choice(len(xs), size=len(xs), replace=True)
        try:
            v = fn(xs[idx])
            if np.isfinite(v):
                samples.append(v)
        except Exception:
            pass
    samples = np.array(samples)
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def _legacy_bootstrap_mean_ci(arr, n_boot=10000, seed=42):
    # Verbatim copy of the function formerly in analyze_composition_expansion.py.
    rng = np.random.default_rng(seed)
    bs = np.array([rng.choice(arr, size=len(arr), replace=True).mean()
                   for _ in range(n_boot)])
    return float(arr.mean()), tuple(np.percentile(bs, [2.5, 97.5]).tolist())


def test_bootstrap_ci_matches_legacy_for_mean() -> None:
    rng = np.random.default_rng(0)
    xs = rng.normal(0.05, 0.2, size=40)
    assert bootstrap_ci(xs, np.mean, n_boot=2000) == _legacy_bootstrap_ci(
        xs, np.mean, n_boot=2000
    )


def test_bootstrap_ci_matches_legacy_for_pearson_over_indices() -> None:
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, size=40)
    b = 0.5 * a + rng.normal(0, 1, size=40)

    def stat(idx: np.ndarray) -> float:
        return scipy_stats.pearsonr(a[idx], b[idx]).statistic

    idx_arr = np.arange(len(a))
    assert bootstrap_ci(idx_arr, stat, n_boot=1000) == _legacy_bootstrap_ci(
        idx_arr, stat, n_boot=1000
    )


def test_bootstrap_ci_brackets_true_mean() -> None:
    rng = np.random.default_rng(2)
    xs = rng.normal(1.0, 0.1, size=200)
    lo, hi = bootstrap_ci(xs, np.mean, n_boot=2000)
    assert lo < 1.0 < hi
    assert hi - lo < 0.1


def test_bootstrap_ci_skips_degenerate_resamples() -> None:
    # A statistic that is undefined unless the resample contains both
    # values; most resamples of a 2-element array are degenerate.
    xs = np.array([0.0, 1.0])

    def stat(sample: np.ndarray) -> float:
        if len(np.unique(sample)) < 2:
            raise ValueError("degenerate resample")
        return float(sample.mean())

    lo, hi = bootstrap_ci(xs, stat, n_boot=500)
    assert lo == hi == 0.5  # the only non-degenerate resample value


def test_bootstrap_mean_ci_matches_legacy() -> None:
    rng = np.random.default_rng(3)
    arr = rng.normal(0.02, 0.05, size=20)
    assert bootstrap_mean_ci(arr, n_boot=2000) == _legacy_bootstrap_mean_ci(
        arr, n_boot=2000
    )


def test_paired_gap_report_positive_effect() -> None:
    rng = np.random.default_rng(4)
    gaps = rng.normal(0.05, 0.02, size=40)  # clearly positive effect
    report = paired_gap_report(gaps, n_boot=2000)
    assert report.n == 40
    assert report.ci_low > 0
    assert report.wilcoxon_p < 0.001
    assert report.wins == int((gaps > 0).sum())
    assert report.mean == float(gaps.mean())


def test_paired_gap_report_null_effect_and_serialization() -> None:
    rng = np.random.default_rng(5)
    gaps = rng.normal(0.0, 0.05, size=40)
    gaps -= gaps.mean()  # exactly-null effect regardless of the draw
    report = paired_gap_report(gaps, n_boot=2000)
    assert report.ci_low < 0 < report.ci_high
    assert report.wilcoxon_p > 0.05

    d = report.as_dict()
    assert d["n"] == 40
    assert d["ci"] == [report.ci_low, report.ci_high]
    assert d["wilcoxon_p_greater"] == report.wilcoxon_p
