"""Every tracked pair file must be well-formed: the bash word-splitting
bug that dropped 'at ease' pairs from the original time-varying chain is
exactly what these files exist to prevent."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

PAIR_FILES = sorted(
    (Path(__file__).resolve().parents[1] / "experiments" / "pairs").glob("*.json")
)


@pytest.mark.parametrize("path", PAIR_FILES, ids=lambda p: p.name)
def test_pair_file_is_well_formed(path: Path) -> None:
    pairs = json.loads(path.read_text())
    assert len(pairs) > 0
    assert all(isinstance(p, list) and len(p) == 2 for p in pairs)
    assert len({tuple(p) for p in pairs}) == len(pairs)  # no duplicates
    flat = [label for pair in pairs for label in pair]
    # Labels are the literal emotion words: spaces allowed ('at ease'),
    # shell-chain underscore forms are not.
    assert not any("_" in label for label in flat)
    assert all(label == label.strip() for label in flat)


def test_pair_files_exist() -> None:
    names = {p.name for p in PAIR_FILES}
    assert {"alift_n40.json", "time_varying_n13.json"} <= names
