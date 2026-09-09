"""Quaternion boundary tests for old packaged data and the active IsaacLab runtime."""

from __future__ import annotations

import importlib.util

import numpy as np
import torch
from isaaclab.utils.math import quat_from_euler_xyz


def _load_compat_module():
    spec = importlib.util.find_spec("scaletrack.utils.quaternion_compat")
    assert spec is not None, "quaternion compatibility module is missing"
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_conversions_keep_disk_wxyz_and_adapt_runtime_order():
    """Catch either the simulator input or saved dataset using an implicit convention."""
    compat = _load_compat_module()
    source_xyzw = np.array([[0.1, 0.2, 0.3, 0.9]], dtype=np.float32)
    packed_wxyz = np.array([[0.9, 0.1, 0.2, 0.3]], dtype=np.float32)

    np.testing.assert_array_equal(
        compat.source_xyzw_to_runtime(source_xyzw, "xyzw"), source_xyzw
    )
    np.testing.assert_array_equal(
        compat.source_xyzw_to_runtime(source_xyzw, "wxyz"), packed_wxyz
    )
    np.testing.assert_array_equal(
        compat.packed_wxyz_to_runtime(packed_wxyz, "xyzw"), source_xyzw
    )
    np.testing.assert_array_equal(
        compat.runtime_to_packed_wxyz(source_xyzw, "xyzw"), packed_wxyz
    )


def test_runtime_detection_maps_its_identity_to_disk_wxyz():
    """Catch stale version labels selecting the convention instead of runtime behavior."""
    compat = _load_compat_module()
    zeros = torch.zeros(1)
    runtime_identity = quat_from_euler_xyz(zeros, zeros, zeros)

    assert hasattr(compat, "detect_runtime_quaternion_order"), "runtime order detector is missing"
    runtime_order = compat.detect_runtime_quaternion_order()
    packed_identity = compat.runtime_to_packed_wxyz(runtime_identity, runtime_order)

    assert runtime_order in {"wxyz", "xyzw"}
    torch.testing.assert_close(packed_identity, torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
