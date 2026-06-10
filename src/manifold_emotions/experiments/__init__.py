"""Experiment definitions: config-driven runners over manifold variants."""

from .chord import (
    NEUTRAL_PROMPTS,
    ChordRunConfig,
    run_chord_pair,
)

__all__ = [
    "NEUTRAL_PROMPTS",
    "ChordRunConfig",
    "run_chord_pair",
]
