"""Read captured .bin / sidecar JSON files into numpy arrays."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..errors import CaptureError


@dataclass(frozen=True, slots=True)
class CapturedActivation:
    """One layer × hook activation for one story, with metadata from the sidecar."""

    request_id: str
    layer: int
    hook: str
    # Shape: (num_positions, hidden_size)
    activations: np.ndarray


def _load_sidecar(sidecar_path: Path) -> dict:
    if not sidecar_path.exists():
        raise CaptureError(f"sidecar JSON missing: {sidecar_path}")
    return json.loads(sidecar_path.read_text())


def load_activation(bin_path: Path) -> CapturedActivation:
    """Load a single .bin file (raw bf16 bytes) using its sidecar JSON for shape.

    The sidecar JSON written by the filesystem consumer carries `request_id`,
    `layer`, `hook` and shape information. .bin payloads are raw bytes in the
    model's residual dtype; bf16 is stored as raw uint16 bytes per the docs.
    """
    sidecar = _load_sidecar(bin_path.with_suffix(".json"))

    shape = tuple(sidecar.get("shape") or sidecar.get("tensor_shape") or [])
    if not shape:
        raise CaptureError(f"no shape in sidecar for {bin_path}: {sidecar}")

    raw = bin_path.read_bytes()
    expected_bytes = int(np.prod(shape)) * 2  # bf16 = 2 bytes per element
    if len(raw) != expected_bytes:
        raise CaptureError(
            f"size mismatch for {bin_path}: have {len(raw)} bytes, "
            f"expect {expected_bytes} for shape {shape}"
        )

    # bf16 → load as uint16, reinterpret as bf16 via float32 upcast.
    # numpy lacks native bf16; we upcast to float32 by left-shifting into the
    # high half of a uint32, then viewing as float32. This is exact.
    u16 = np.frombuffer(raw, dtype=np.uint16).copy()
    u32 = u16.astype(np.uint32) << 16
    f32 = u32.view(np.float32).reshape(shape)

    return CapturedActivation(
        request_id=sidecar.get("request_id", bin_path.parent.name),
        layer=int(sidecar.get("layer", -1)),
        hook=str(sidecar.get("hook", bin_path.stem.split("_", 1)[-1])),
        activations=f32,
    )


def iter_captures(capture_root: Path) -> list[Path]:
    """Return all .bin paths under the capture root, sorted for determinism."""
    return sorted(capture_root.rglob("*.bin"))
