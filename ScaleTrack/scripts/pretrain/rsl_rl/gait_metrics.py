"""Pure NumPy evidence metrics for short, physically executed gait clips.

These kinematic checks do not infer foot contact and are not a general proof of
gait stability.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class GaitEvidence:
    sample_count: int
    duration_s: float
    actual_forward_m: float
    reference_forward_m: float
    actual_path_m: float
    reference_path_m: float
    root_error_mean_m: float
    root_error_max_m: float
    min_pelvis_height_m: float
    min_pelvis_up_dot: float
    left_foot_xy_path_m: float
    right_foot_xy_path_m: float
    left_foot_xy_excursion_m: float
    right_foot_xy_excursion_m: float
    foot_fore_aft_swap_count: int
    forward_gate: bool
    root_error_gate: bool
    posture_gate: bool
    feet_excursion_gate: bool
    foot_swap_gate: bool
    passed: bool
    claim_scope: str = (
        "short-clip kinematic execution evidence only; no contact inference or general gait-stability claim"
    )


def _finite_array(name: str, value, width: int) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite (N, {width}) array") from error
    if array.ndim != 2 or array.shape[1] != width or array.shape[0] < 2:
        raise ValueError(f"{name} must have shape (N, {width}) with N >= 2")
    if array.dtype.kind != "f" or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite floating-point values")
    return array.astype(np.float64, copy=False)


def _duration(sim_dt, intervals: int) -> float:
    if isinstance(sim_dt, (bool, np.bool_)):
        raise ValueError("sim_dt must be positive and finite")
    try:
        dt = np.asarray(sim_dt)
    except (TypeError, ValueError) as error:
        raise ValueError("sim_dt must be a positive finite scalar or (N-1,) array") from error
    if dt.ndim == 0:
        if dt.dtype.kind not in "fiu" or not math.isfinite(float(dt)) or float(dt) <= 0.0:
            raise ValueError("sim_dt must be positive and finite")
        return intervals * float(dt)
    if dt.shape != (intervals,) or dt.dtype.kind not in "fiu" or not np.all(np.isfinite(dt)):
        raise ValueError("sim_dt must be a positive finite scalar or (N-1,) array")
    if np.any(dt <= 0.0):
        raise ValueError("sim_dt must be positive and finite")
    return float(np.sum(dt, dtype=np.float64))


def _path(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _xy_excursion(points: np.ndarray) -> float:
    return float(np.linalg.norm(points[:, :2] - points[0, :2], axis=1).max())


def _deadband_swaps(values: np.ndarray, deadband: float = 0.03) -> int:
    signs = np.sign(values[np.abs(values) > deadband])
    if signs.size < 2:
        return 0
    return int(np.count_nonzero(signs[1:] != signs[:-1]))


def summarize_gait_evidence(
    actual_pelvis_xyz,
    reference_pelvis_xyz,
    actual_pelvis_quat_wxyz,
    left_ankle_xyz,
    right_ankle_xyz,
    sim_dt,
) -> GaitEvidence:
    """Summarize one synchronized short walking clip and apply fixed gates."""
    actual = _finite_array("actual_pelvis_xyz", actual_pelvis_xyz, 3)
    reference = _finite_array("reference_pelvis_xyz", reference_pelvis_xyz, 3)
    quat = _finite_array("actual_pelvis_quat_wxyz", actual_pelvis_quat_wxyz, 4)
    left = _finite_array("left_ankle_xyz", left_ankle_xyz, 3)
    right = _finite_array("right_ankle_xyz", right_ankle_xyz, 3)
    count = actual.shape[0]
    arrays = (reference, quat, left, right)
    if any(array.shape[0] != count for array in arrays):
        raise ValueError("all gait sequences must have the same sample count")
    duration = _duration(sim_dt, count - 1)

    quat_norm = np.linalg.norm(quat, axis=1)
    if not np.allclose(quat_norm, 1.0, rtol=0.0, atol=1e-4):
        raise ValueError("actual_pelvis_quat_wxyz must contain unit quaternions")

    reference_delta_xy = reference[-1, :2] - reference[0, :2]
    reference_net_xy = float(np.linalg.norm(reference_delta_xy))
    if reference_net_xy <= 1e-12:
        raise ValueError("reference pelvis must have nonzero horizontal displacement")
    forward_axis = reference_delta_xy / reference_net_xy
    actual_forward = float((actual[-1, :2] - actual[0, :2]) @ forward_axis)
    reference_forward = float(reference_delta_xy @ forward_axis)

    error = np.linalg.norm(actual - reference, axis=1)
    # For wxyz quaternions, local pelvis +Z dotted with world +Z.
    up_dot = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    relative_fore_aft = (left[:, :2] - right[:, :2]) @ forward_axis

    left_excursion = _xy_excursion(left)
    right_excursion = _xy_excursion(right)
    swaps = _deadband_swaps(relative_fore_aft)
    error_mean = float(error.mean())
    error_max = float(error.max())
    min_height = float(actual[:, 2].min())
    min_up_dot = float(up_dot.min())

    forward_gate = actual_forward >= 0.5 and actual_forward >= 0.5 * reference_forward
    root_error_gate = error_mean <= 0.15 and error_max <= 0.30
    posture_gate = min_height >= 0.35 and min_up_dot >= 0.5
    feet_excursion_gate = left_excursion >= 0.10 and right_excursion >= 0.10
    foot_swap_gate = swaps >= 2
    passed = forward_gate and root_error_gate and posture_gate and feet_excursion_gate and foot_swap_gate

    return GaitEvidence(
        sample_count=count,
        duration_s=duration,
        actual_forward_m=actual_forward,
        reference_forward_m=reference_forward,
        actual_path_m=_path(actual),
        reference_path_m=_path(reference),
        root_error_mean_m=error_mean,
        root_error_max_m=error_max,
        min_pelvis_height_m=min_height,
        min_pelvis_up_dot=min_up_dot,
        left_foot_xy_path_m=_path(left[:, :2]),
        right_foot_xy_path_m=_path(right[:, :2]),
        left_foot_xy_excursion_m=left_excursion,
        right_foot_xy_excursion_m=right_excursion,
        foot_fore_aft_swap_count=swaps,
        forward_gate=bool(forward_gate),
        root_error_gate=bool(root_error_gate),
        posture_gate=bool(posture_gate),
        feet_excursion_gate=bool(feet_excursion_gate),
        foot_swap_gate=bool(foot_swap_gate),
        passed=bool(passed),
    )
