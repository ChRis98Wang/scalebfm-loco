"""Pure helpers shared by ScaleTrack's IsaacLab motion packaging entry point."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import re
from typing import TypeVar


MotionPath = TypeVar("MotionPath")


def positive_int(value: str) -> int:
    """Parse a strictly positive CLI integer."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected a positive integer, got {value!r}"
        ) from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return parsed


def scale_track_fps(value: str) -> int:
    """Enforce the 50-Hz archive contract used by the tracking loader."""
    parsed = positive_int(value)
    if parsed != 50:
        raise argparse.ArgumentTypeError(
            f"this ScaleTrack workflow only supports 50 FPS, got {parsed}"
        )
    return parsed


def sha256_digest(value: str) -> str:
    """Parse a lowercase SHA-256 digest passed between pipeline stages."""
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise argparse.ArgumentTypeError(
            "pipeline fingerprint must be a lowercase 64-character SHA-256 digest"
        )
    return value


def select_motion_batch(
    motion_files: Sequence[MotionPath], *, batch_index: int, batch_size: int
) -> list[MotionPath]:
    """Return one bounded slice without loading motion tensors from other batches."""
    if batch_index < 0:
        raise ValueError("batch_index must be non-negative")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    start = batch_index * batch_size
    return list(motion_files[start : start + batch_size])
