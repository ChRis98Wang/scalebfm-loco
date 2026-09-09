"""CPU-only tests for short-clip gait evidence metrics."""

from __future__ import annotations

from dataclasses import asdict, FrozenInstanceError
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/pretrain/rsl_rl/gait_metrics.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("gait_metrics_under_test", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _passing_clip():
    x_ref = np.linspace(0.0, 0.8, 9)
    x_actual = np.linspace(0.0, 0.6, 9)
    reference = np.column_stack((x_ref, np.zeros(9), np.full(9, 0.8)))
    actual = np.column_stack((x_actual, np.zeros(9), np.full(9, 0.8)))
    quat = np.tile([1.0, 0.0, 0.0, 0.0], (9, 1))
    phase = np.array([0.10, 0.10, -0.10, -0.10, 0.10, 0.10, -0.10, -0.10, 0.10])
    left = np.column_stack((x_actual + phase, np.full(9, 0.10), 0.03 + np.maximum(phase, 0.0)))
    right = np.column_stack((x_actual - phase, np.full(9, -0.10), 0.03 + np.maximum(-phase, 0.0)))
    return actual, reference, quat, left, right


def test_passing_clip_reports_fixed_physical_evidence_gates():
    module = _load_module()
    arrays = _passing_clip()
    result = module.summarize_gait_evidence(*arrays, sim_dt=0.02)
    assert result.sample_count == 9
    assert result.duration_s == pytest.approx(0.16)
    assert result.actual_forward_m == pytest.approx(0.6)
    assert result.reference_forward_m == pytest.approx(0.8)
    assert result.actual_path_m == pytest.approx(0.6)
    assert result.reference_path_m == pytest.approx(0.8)
    assert result.root_error_mean_m == pytest.approx(0.1)
    assert result.root_error_max_m == pytest.approx(0.2)
    assert result.min_pelvis_height_m == pytest.approx(0.8)
    assert result.min_pelvis_up_dot == pytest.approx(1.0)
    assert result.left_foot_xy_excursion_m >= 0.10
    assert result.right_foot_xy_excursion_m >= 0.10
    assert result.foot_fore_aft_swap_count == 4
    assert result.forward_gate and result.root_error_gate and result.posture_gate
    assert result.feet_excursion_gate and result.foot_swap_gate and result.passed
    assert "no contact inference" in result.claim_scope
    json.dumps(asdict(result))
    with pytest.raises(FrozenInstanceError):
        result.passed = False


def test_direction_is_defined_by_reference_not_actual_drift():
    module = _load_module()
    actual, reference, quat, left, right = _passing_clip()
    reference[:, :2] = np.column_stack((np.zeros(9), np.linspace(0.0, 0.8, 9)))
    result = module.summarize_gait_evidence(actual, reference, quat, left, right, 0.02)
    assert result.reference_forward_m == pytest.approx(0.8)
    assert result.actual_forward_m == pytest.approx(0.0)
    assert not result.forward_gate
    assert not result.passed


@pytest.mark.parametrize(
    "mutation,failed_gate",
    [
        (lambda a, r, q, l, rr: a.__setitem__((slice(None), 0), np.linspace(0, 0.49, 9)), "forward_gate"),
        (lambda a, r, q, l, rr: a.__setitem__((slice(None), 1), 0.31), "root_error_gate"),
        (lambda a, r, q, l, rr: a.__setitem__((3, 2), 0.34), "posture_gate"),
        (lambda a, r, q, l, rr: l.__setitem__((slice(None), slice(0, 2)), l[0, :2]), "feet_excursion_gate"),
        (lambda a, r, q, l, rr: (l.__setitem__((slice(None), 0), a[:, 0] + 0.1), rr.__setitem__((slice(None), 0), a[:, 0] - 0.1)), "foot_swap_gate"),
    ],
)
def test_each_fixed_gate_can_fail_independently(mutation, failed_gate):
    module = _load_module()
    arrays = list(_passing_clip())
    mutation(*arrays)
    result = module.summarize_gait_evidence(*arrays, sim_dt=0.02)
    assert not getattr(result, failed_gate)
    assert not result.passed


def test_tilt_gate_uses_wxyz_pelvis_up_direction():
    module = _load_module()
    actual, reference, quat, left, right = _passing_clip()
    quat[4] = [0.5, np.sqrt(0.75), 0.0, 0.0]
    result = module.summarize_gait_evidence(actual, reference, quat, left, right, 0.02)
    assert result.min_pelvis_up_dot == pytest.approx(-0.5)
    assert not result.posture_gate


def test_per_interval_dt_is_summed_and_inputs_are_not_mutated():
    module = _load_module()
    arrays = _passing_clip()
    originals = [array.copy() for array in arrays]
    result = module.summarize_gait_evidence(*arrays, sim_dt=np.full(8, 0.025))
    assert result.duration_s == pytest.approx(0.2)
    for array, original in zip(arrays, originals, strict=True):
        np.testing.assert_array_equal(array, original)


@pytest.mark.parametrize(
    "index,value,match",
    [
        (0, np.zeros((1, 3)), "actual_pelvis_xyz"),
        (1, np.zeros((9, 2)), "reference_pelvis_xyz"),
        (2, np.zeros((8, 4)), "same sample count"),
        (3, np.full((9, 3), np.nan), "left_ankle_xyz"),
        (4, [[0, 0, 0]] * 9, "right_ankle_xyz"),
        (4, np.zeros((9, 3), dtype=np.complex128), "right_ankle_xyz"),
    ],
)
def test_invalid_sequence_shapes_values_and_dtypes_are_rejected(index, value, match):
    module = _load_module()
    arrays = list(_passing_clip())
    arrays[index] = value
    with pytest.raises(ValueError, match=match):
        module.summarize_gait_evidence(*arrays, sim_dt=0.02)


@pytest.mark.parametrize("dt", [True, 0.0, -0.1, np.nan, np.inf, [0.02] * 7, [0.02] * 7 + [0.0]])
def test_invalid_sim_dt_is_rejected(dt):
    module = _load_module()
    with pytest.raises(ValueError, match="sim_dt"):
        module.summarize_gait_evidence(*_passing_clip(), sim_dt=dt)


def test_zero_reference_displacement_and_nonunit_quaternion_are_rejected():
    module = _load_module()
    actual, reference, quat, left, right = _passing_clip()
    reference[:] = reference[0]
    with pytest.raises(ValueError, match="nonzero horizontal"):
        module.summarize_gait_evidence(actual, reference, quat, left, right, 0.02)
    _, reference, quat, _, _ = _passing_clip()
    quat[2] *= 2.0
    with pytest.raises(ValueError, match="unit quaternions"):
        module.summarize_gait_evidence(actual, reference, quat, left, right, 0.02)
