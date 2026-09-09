"""Crash-safe publication helpers for retargeted motion artifacts."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Any

import joblib


def atomic_joblib_dump(value: Any, destination: Path | str) -> Path:
    """Serialize beside destination and atomically replace it after fsync."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            joblib.dump(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return destination
