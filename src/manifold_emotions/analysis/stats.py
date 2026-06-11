"""Bootstrap and paired-comparison statistics shared by analysis scripts.

``bootstrap_ci`` and ``bootstrap_mean_ci`` are verbatim ports of the two
implementations that were previously copy-pasted across
``scripts/analyze_*.py``. They intentionally use *different* resampling
streams (index-based vs value-based ``rng.choice``) so that, at the same
seed, each reproduces bit-identical output to the script it replaces —
do not merge them into one without accepting CI-bound drift in
regenerated summaries.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import stats as _scipy_stats


def bootstrap_ci(
    xs: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    n_boot: int = 10_000,
    seed: int = 42,
) -> tuple[float, float]:
    """95% percentile bootstrap CI of ``statistic`` over resamples of ``xs``.

    ``xs`` may be the data itself or an index array (resample indices and
    close over the data in ``statistic`` to bootstrap paired/correlated
    quantities, e.g. a Pearson r over two aligned arrays).

    Resamples on which ``statistic`` is undefined (degenerate draws such as
    a constant sample under Pearson r) are skipped, matching the behavior
    of the original inline copies.
    """
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_boot):
        idx = rng.choice(len(xs), size=len(xs), replace=True)
        try:
            v = statistic(xs[idx])
        except (ValueError, ZeroDivisionError, FloatingPointError):
            continue
        if np.isfinite(v):
            samples.append(v)
    arr = np.array(samples)
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def bootstrap_mean_ci(
    arr: np.ndarray,
    n_boot: int = 10_000,
    seed: int = 42,
) -> tuple[float, tuple[float, float]]:
    """Mean and its 95% bootstrap CI, value-resampled.

    Verbatim port of the ``analyze_composition_expansion.py`` variant;
    kept value-based (``rng.choice(arr, ...)``) so regenerated outputs
    match the archived ones exactly.
    """
    rng = np.random.default_rng(seed)
    bs = np.array(
        [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)]
    )
    return float(arr.mean()), tuple(np.percentile(bs, [2.5, 97.5]).tolist())


@dataclass(frozen=True, slots=True)
class PairedGapReport:
    """Mean gap, bootstrap CI, one-sided Wilcoxon p, and win count.

    The recurring per-pair comparison shape across the chord/composition
    analyses: ``gaps[i] = metric(baseline, pair i) − metric(method, pair i)``
    so positive entries mean the method beats the baseline on that pair.
    """

    n: int
    mean: float
    ci_low: float
    ci_high: float
    wilcoxon_p: float
    wins: int
    alternative: str

    def as_dict(self) -> dict:
        """JSON-serializable form for ``_summary.json`` outputs."""
        return {
            "n": self.n,
            "mean": self.mean,
            "ci": [self.ci_low, self.ci_high],
            f"wilcoxon_p_{self.alternative}": self.wilcoxon_p,
            "wins": self.wins,
        }


def paired_gap_report(
    gaps: np.ndarray,
    alternative: str = "greater",
    n_boot: int = 10_000,
    seed: int = 42,
) -> PairedGapReport:
    """Summarize a paired per-item gap array: mean, 95% CI, Wilcoxon, wins."""
    gaps = np.asarray(gaps, dtype=float)
    lo, hi = bootstrap_ci(gaps, np.mean, n_boot=n_boot, seed=seed)
    wilcoxon_p = float(_scipy_stats.wilcoxon(gaps, alternative=alternative).pvalue)
    return PairedGapReport(
        n=len(gaps),
        mean=float(gaps.mean()),
        ci_low=lo,
        ci_high=hi,
        wilcoxon_p=wilcoxon_p,
        wins=int((gaps > 0).sum()),
        alternative=alternative,
    )
