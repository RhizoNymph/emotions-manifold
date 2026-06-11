"""Shared analysis utilities (statistics, reporting) for results scripts."""

from .stats import PairedGapReport, bootstrap_ci, bootstrap_mean_ci, paired_gap_report

__all__ = [
    "PairedGapReport",
    "bootstrap_ci",
    "bootstrap_mean_ci",
    "paired_gap_report",
]
