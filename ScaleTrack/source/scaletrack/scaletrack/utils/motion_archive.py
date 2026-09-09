"""Safe, versioned helpers for ScaleTrack's processed motion archives."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import tempfile
from typing import Mapping

import numpy as np


FORMAT_VERSION = 3
PACKED_QUATERNION_ORDER = "wxyz"


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for one source artifact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_archive_path(input_dir: Path, motion_file: Path) -> Path:
    """Map a contained retargeted file to a collision-free relative NPZ path."""
    input_root = Path(input_dir).resolve()
    source = Path(motion_file).resolve()
    relative = source.relative_to(input_root)
    return relative.with_suffix(".npz")


def save_motion_archive(
    destination: Path,
    arrays: Mapping[str, object],
    *,
    source_path: Path,
    pipeline_fingerprint: str,
) -> None:
    """Atomically publish a processed motion with its compatibility metadata."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not re.fullmatch(r"[0-9a-f]{64}", pipeline_fingerprint):
        raise ValueError("pipeline_fingerprint must be a lowercase SHA-256 digest")
    payload = dict(arrays)
    for names_key, positions_key in (("joint_names", "joint_pos"), ("body_names", "body_pos_w")):
        if names_key not in payload:
            continue  # Existing v3 archives remain readable without names.
        names = np.asarray(payload[names_key])
        positions = np.asarray(payload.get(positions_key))
        if (names.ndim != 1 or names.dtype.kind not in "US" or positions.ndim < 2
                or len(names) != positions.shape[1] or len(set(names.tolist())) != len(names)
                or any(not str(name).strip() for name in names)):
            raise ValueError(f"{names_key} must uniquely name each {positions_key} column")
    missing_reference_fields = {
        "reference_root_pos",
        "reference_root_quat_w",
    }.difference(payload)
    if missing_reference_fields:
        raise ValueError(
            "v3 motion archive is missing reference_root fields: "
            f"{sorted(missing_reference_fields)}"
        )
    reference_root_pos = np.asarray(payload["reference_root_pos"])
    reference_root_quat = np.asarray(payload["reference_root_quat_w"])
    if reference_root_pos.ndim != 2 or reference_root_pos.shape[1:] != (3,):
        raise ValueError("reference_root_pos must have shape (N, 3)")
    if reference_root_quat.shape != (reference_root_pos.shape[0], 4):
        raise ValueError(
            "reference_root_quat_w must have shape "
            f"({reference_root_pos.shape[0]}, 4)"
        )
    if not np.isfinite(reference_root_pos).all() or not np.isfinite(
        reference_root_quat
    ).all():
        raise ValueError("reference_root fields contain NaN or Inf")
    if not np.allclose(
        np.linalg.norm(reference_root_quat, axis=-1),
        1.0,
        atol=1.0e-3,
    ):
        raise ValueError("reference_root_quat_w contains non-unit quaternions")
    payload.update(
        {
            "format_version": np.asarray(FORMAT_VERSION, dtype=np.int64),
            "quaternion_order": np.asarray(PACKED_QUATERNION_ORDER),
            "source_sha256": np.asarray(sha256_file(source_path)),
            "pipeline_fingerprint": np.asarray(pipeline_fingerprint),
        }
    )

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".npz",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            np.savez(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
