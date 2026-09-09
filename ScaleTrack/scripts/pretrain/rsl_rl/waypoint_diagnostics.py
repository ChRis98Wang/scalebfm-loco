"""Pure observation counterfactuals for diagnosing online waypoint policies."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
import math

import numpy as np
import torch


@dataclass(frozen=True)
class TaskLayout:
    """Layout derived from TrackingEnvCfg.ObservationsCfg.PolicyTaskCfg."""

    num_bodies: int = 14
    future_frames: int = 6
    blocks: tuple[tuple[str, int], ...] = (
        ("target_body_pos", 3),
        ("target_body_pos_rel", 3),
        ("target_body_rot", 6),
        ("target_body_rot_rel", 6),
    )
    mode_dim: int = 14
    with_time: bool = True

    @property
    def raw_dim(self) -> int:
        return self.num_bodies * sum(width for _, width in self.blocks) + int(self.with_time)

    @property
    def actor_dim(self) -> int:
        return self.raw_dim + self.mode_dim

    @property
    def pelvis_slices(self) -> dict[str, slice]:
        offset = 0
        result = {}
        for name, width in self.blocks:
            result[name] = slice(offset, offset + width)
            offset += self.num_bodies * width
        return result

    @property
    def time_slice(self) -> slice:
        start = self.num_bodies * sum(width for _, width in self.blocks)
        return slice(start, start + int(self.with_time))


LAYOUT = TaskLayout()


def task_layout() -> dict[str, object]:
    """Return a serialization-friendly description of the current actor task schema."""
    return {
        "num_bodies": LAYOUT.num_bodies,
        "future_frames": LAYOUT.future_frames,
        "blocks": {
            name: {"width": width, "pelvis": [slot.start, slot.stop]}
            for (name, width), slot in zip(LAYOUT.blocks, LAYOUT.pelvis_slices.values(), strict=True)
        },
        "time": [LAYOUT.time_slice.start, LAYOUT.time_slice.stop],
        "raw_task_dim": LAYOUT.raw_dim,
        "mode_dim": LAYOUT.mode_dim,
        "actor_task_dim": LAYOUT.actor_dim,
    }


def _finite_tensor(value: object, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must be finite")
    return value


def _validate_observation(obs: Mapping[str, object]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not isinstance(obs, Mapping):
        raise TypeError("obs must be a mapping of torch tensors")
    required = ("policy", "policy_task", "action", "mode", "mode_mapping")
    missing = [name for name in required if name not in obs]
    if missing:
        raise ValueError(f"obs is missing required keys: {missing}")
    for name, value in obs.items():
        _finite_tensor(value, name)
        if value.ndim < 1:
            raise ValueError(f"{name} must have a batch dimension")

    task = obs["policy_task"]
    policy = obs["policy"]
    action = obs["action"]
    mode = obs["mode"]
    mapping = obs["mode_mapping"]
    batch = task.shape[0]
    if batch <= 0:
        raise ValueError("observation batch must be nonempty")
    if any(value.shape[0] != batch for value in obs.values()):
        raise ValueError("all observation tensors must share the policy_task batch dimension")

    core = {"policy": policy, "policy_task": task, "action": action,
            "mode": mode, "mode_mapping": mapping}
    if any(not value.is_floating_point() for value in core.values()):
        raise ValueError("policy, policy_task, action, mode, and mode_mapping must be floating tensors")
    if any(value.dtype != task.dtype for value in core.values()):
        raise ValueError("core observation tensors must have the same dtype")
    if any(value.device != task.device for value in core.values()):
        raise ValueError("core observation tensors must be on the same device")

    if policy.shape != (batch, 3, 64):
        raise ValueError("policy must have shape (batch, 3, 64)")
    if task.ndim != 3 or task.shape[1:] != (LAYOUT.future_frames, LAYOUT.raw_dim):
        raise ValueError(
            f"policy_task must have shape (batch, {LAYOUT.future_frames}, {LAYOUT.raw_dim})"
        )
    if action.shape != (batch, 3, 29):
        raise ValueError("action must have shape (batch, 3, 29)")
    if mode.shape != (batch, LAYOUT.mode_dim):
        raise ValueError(f"mode must have shape (batch, {LAYOUT.mode_dim})")
    if mapping.shape != (batch, LAYOUT.raw_dim):
        raise ValueError(f"mode_mapping must have shape (batch, {LAYOUT.raw_dim})")

    expected_mode = torch.zeros_like(mode)
    expected_mode[:, 0] = 1
    if not torch.equal(mode, expected_mode):
        raise ValueError("mode must be the exact Pelvis-1 mask [1, 0, ..., 0]")
    expected_mapping = torch.zeros_like(mapping)
    for slot in LAYOUT.pelvis_slices.values():
        expected_mapping[:, slot] = 1
    expected_mapping[:, LAYOUT.time_slice] = 1
    if not torch.equal(mapping, expected_mapping):
        raise ValueError("mode_mapping does not exactly match the Pelvis-1 task layout")
    return task, mode, mapping


def stationary_pelvis_counterfactual(obs: Mapping[str, object]) -> dict[str, torch.Tensor]:
    """Clone ``obs`` and flatten only future Pelvis-1 pose features to frame zero."""
    task, _, _ = _validate_observation(obs)
    result = {name: value.clone() for name, value in obs.items()}
    counterfactual = result["policy_task"]
    for slot in LAYOUT.pelvis_slices.values():
        counterfactual[:, 1:, slot] = task[:, :1, slot]
    return result


def decode_pelvis_task(policy_task: torch.Tensor) -> dict[str, torch.Tensor]:
    """Extract independent copies of each pelvis feature block and the time offset."""
    task = _finite_tensor(policy_task, "policy_task")
    if task.ndim != 3 or task.shape[1:] != (LAYOUT.future_frames, LAYOUT.raw_dim):
        raise ValueError(
            f"policy_task must have shape (batch, {LAYOUT.future_frames}, {LAYOUT.raw_dim})"
        )
    decoded = {name: task[..., slot].clone() for name, slot in LAYOUT.pelvis_slices.items()}
    decoded["time"] = task[..., LAYOUT.time_slice].clone()
    return decoded


def summarize_actions(base, flat) -> dict[str, float]:
    """Summarize an action counterfactual difference without retaining either input."""
    if isinstance(base, torch.Tensor) and isinstance(flat, torch.Tensor):
        if base.shape != flat.shape:
            raise ValueError("action tensors must have identical shapes")
        if base.numel() == 0:
            raise ValueError("action tensors must be nonempty")
        if not bool(torch.isfinite(base).all().item()) or not bool(torch.isfinite(flat).all().item()):
            raise ValueError("action tensors must be finite")
        difference = base.to(dtype=torch.float64) - flat.to(dtype=torch.float64)
        squared = difference.square()
        return {
            "l2": float(torch.sqrt(squared.sum()).item()),
            "max_abs": float(difference.abs().max().item()),
            "rms": float(torch.sqrt(squared.mean()).item()),
        }
    if isinstance(base, np.ndarray) and isinstance(flat, np.ndarray):
        if base.shape != flat.shape:
            raise ValueError("action arrays must have identical shapes")
        if base.size == 0:
            raise ValueError("action arrays must be nonempty")
        if not np.all(np.isfinite(base)) or not np.all(np.isfinite(flat)):
            raise ValueError("action arrays must be finite")
        difference = base.astype(np.float64) - flat.astype(np.float64)
        return {
            "l2": float(np.linalg.norm(difference)),
            "max_abs": float(np.max(np.abs(difference))),
            "rms": float(math.sqrt(float(np.mean(np.square(difference))))),
        }
    raise TypeError("base and flat actions must both be torch tensors or both be numpy arrays")
