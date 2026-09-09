"""Opt-in selected-link diagnostics, measured at CommandTerm's metric boundary.

These are reference-tracking measurements, not fall detection or task success.
No training observations, rewards, termination rules or masks are changed.
"""

from contextlib import contextmanager
import math

import torch


METRIC_KEYS = (
    "error_active_body_pos_g",
    "error_active_body_pos_root_relative",
    "error_active_body_rot",
    "error_active_body_pos_g_max",
    "error_all_body_pos_g_max",
)


def _tensor(value):
    return getattr(value, "torch", value)


@torch.inference_mode()
def masked_tracking_metrics(command):
    """Compare the current reference with physical state, before reference advance.

    Positions are metres; rotation is the shortest quaternion geodesic in radians.
    Root-relative positions remove translation only, NOT heading/rotation.
    The actual articulation anchor index need not equal the reference-body index.
    """
    reference = _tensor(command.body_pos_w)
    reference_quat = _tensor(command.body_quat_w)
    actual_all = _tensor(command.robot.data.body_pos_w)
    actual = actual_all[:, command.body_indexes]
    actual_quat = _tensor(command.robot.data.body_quat_w)[:, command.body_indexes]
    mask = _tensor(command.mode)
    if reference.ndim != 3 or reference.shape[-1] != 3 or actual.shape != reference.shape:
        raise ValueError("Position tensors must share (environments, links, 3) shape")
    if mask.shape != reference.shape[:2] or not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("A binary mask matching the environment/link axes is required")
    counts = mask.sum(dim=-1)
    if torch.any(counts == 0):
        raise ValueError("Each mask must activate at least one link")
    if reference_quat.shape != (*reference.shape[:2], 4) or actual_quat.shape != reference_quat.shape:
        raise ValueError("Expected matching quaternion tensors (environments, links, 4)")
    actual_root = actual_all[:, command.robot_anchor_body_index]
    if not all(torch.isfinite(value).all() for value in (reference, actual, actual_root, reference_quat, actual_quat)):
        raise ValueError("Tracking positions and quaternions must be finite")
    reference_norm = reference_quat.norm(dim=-1, keepdim=True)
    actual_norm = actual_quat.norm(dim=-1, keepdim=True)
    if torch.any(reference_norm < 1e-8) or torch.any(actual_norm < 1e-8):
        raise ValueError("A zero quaternion is invalid")
    reference_quat = reference_quat / reference_norm
    actual_quat = actual_quat / actual_norm
    # q and -q are the same orientation. This form is stable near zero angle.
    sign = torch.where((reference_quat * actual_quat).sum(-1, keepdim=True) < 0, -1., 1.)
    actual_quat = actual_quat * sign
    angles = 4 * torch.atan2((reference_quat - actual_quat).norm(dim=-1),
                            (reference_quat + actual_quat).norm(dim=-1))
    distances = (reference - actual).norm(dim=-1)
    reference_local = reference - reference[:, [command.motion_anchor_body_index]]
    actual_local = actual - actual_root[:, None]
    local_distances = (reference_local - actual_local).norm(dim=-1)
    return {
        "error_active_body_pos_g": (distances * mask).sum(-1) / counts,
        "error_active_body_pos_root_relative": (local_distances * mask).sum(-1) / counts,
        "error_active_body_rot": (angles * mask).sum(-1) / counts,
        "error_active_body_pos_g_max": distances.masked_fill(~mask.bool(), -torch.inf).amax(-1),
        "error_all_body_pos_g_max": distances.amax(-1),
    }


@contextmanager
def masked_metrics_context(command, *, enabled=False):
    """Temporarily extend metrics at their native pre-reference-advance boundary.

    A post-env.step callback is too late: CommandTerm.compute has advanced the
    reference already. The instance-only wrapper preserves this alignment and is
    always removed, including when evaluation raises or is interrupted.
    """
    if not enabled:
        yield ()
        return
    if set(METRIC_KEYS) & command.metrics.keys():
        raise ValueError("Masked metric names already exist; refusing to overwrite")
    had_override = "_update_metrics" in vars(command)
    original = command._update_metrics

    def update():
        original()
        command.metrics.update(masked_tracking_metrics(command))

    command._update_metrics = update
    try:
        yield METRIC_KEYS
    finally:
        if had_override:
            command._update_metrics = original
        else:
            del command._update_metrics
        for key in METRIC_KEYS:
            command.metrics.pop(key, None)


def summarize_masked_tracking(rows, *, threshold=0.5):
    """A clip fails when ANY specified link exceeds threshold at ANY valid step.

    All-link means all configured reference links, not every articulation body.
    These diagnostics match the task's distance criterion, but this evaluator
    continues after a deviation instead of terminating. This is not a fall rate.
    """
    if not rows or not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("Nonempty rows and a finite positive threshold are required")
    result = {"threshold_metres": threshold, "criterion": "max over links and valid timesteps; strict > threshold"}
    for name, metric in (("active_links", "error_active_body_pos_g_max"),
                         ("all_links", "error_all_body_pos_g_max")):
        values = [row["metrics"][metric]["max"] for row in rows]
        if not all(math.isfinite(value) and value >= 0 for value in values):
            raise ValueError("Tracking maxima must be finite and nonnegative")
        failed = [row["motion"] for row, value in zip(rows, values) if value > threshold]
        result[name] = {"failed_motions": failed, "tracking_pass_rate": 1 - len(failed) / len(rows)}
    return result
