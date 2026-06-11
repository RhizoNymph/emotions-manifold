"""Experiment definitions: config-driven runners over manifold variants."""

from .chord import (
    NEUTRAL_PROMPTS,
    ChordRunConfig,
    run_chord_pair,
)
from .time_varying import (
    TVRunConfig,
    run_tv_pairs,
)

__all__ = [
    "NEUTRAL_PROMPTS",
    "ChordRunConfig",
    "TVRunConfig",
    "run_chord_pair",
    "run_tv_pairs",
]
