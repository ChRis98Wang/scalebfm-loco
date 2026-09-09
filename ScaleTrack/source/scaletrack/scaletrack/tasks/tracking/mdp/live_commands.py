"""Live-reference adapter that leaves the offline command implementation intact."""

from __future__ import annotations

from collections.abc import Sequence
import operator

import numpy as np
import torch

from scaletrack.tasks.tracking.mdp.commands import MotionCommand
from scaletrack.utils.online_modes import ONLINE_MODES, online_body_indices, validate_online_mode
from scaletrack.utils.quaternion_compat import (
    detect_runtime_quaternion_order,
    packed_wxyz_to_runtime,
)


class LiveMotionCommand(MotionCommand):
    """A MotionCommand which changes source only after an explicit attachment."""

    def __init__(self, cfg, env) -> None:
        self.live_reference = None
        self.live_mode_name = None
        self.runtime_quaternion_order = detect_runtime_quaternion_order()
        self._live_samples: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
        super().__init__(cfg, env)

    def _live_mode_mask(self, mode_name: str) -> torch.Tensor:
        """Build one canonical online mask without mutating command state."""
        mode_name = validate_online_mode(mode_name)
        indices = online_body_indices(mode_name)
        body_names = tuple(self.cfg.body_names)
        if len(body_names) != 14 or tuple(body_names[index] for index in indices) != ONLINE_MODES[mode_name]:
            raise ValueError(f"online mode {mode_name!r} does not match the canonical body indices")
        mask = torch.zeros((1, len(body_names)), dtype=torch.float32, device=self.device)
        mask[0, list(indices)] = 1.0
        return mask

    def attach_live_reference(self, provider, *, mode_name: str = "VR-3") -> None:
        """Switch reference getters to one validated, single-environment provider."""
        if self.live_reference is not None:
            raise RuntimeError("a live reference is already attached")
        if self.num_envs != 1:
            raise ValueError("online tracking requires exactly one environment")
        if self.cfg.ref_frame_buffer_size != 33:
            raise ValueError("online tracking requires a 33-frame reference horizon")
        if tuple(self.cfg.body_names) != tuple(provider.body_names):
            raise ValueError("provider body order must exactly match the motion command")
        step_dt = getattr(self._env, "step_dt", None)
        if step_dt is None or abs(float(step_dt) - float(provider.step_dt)) > 1e-9:
            raise ValueError("provider timestep must equal the environment control timestep")
        mode_name = validate_online_mode(mode_name)
        candidates = getattr(self.cfg, "mode_candidates", None)
        if not isinstance(candidates, dict) or tuple(candidates.get(mode_name, ())) != ONLINE_MODES[mode_name]:
            raise ValueError(f"online tracking requires the configured {mode_name} mask")
        mask = self._live_mode_mask(mode_name)
        mode_table = mask.clone()
        active_mode = mask.clone()
        # This is the provider's public horizon contract.  Check it before
        # installing the mask or retaining the provider so a failed attach is
        # side-effect free.
        provider.sample(range(33))
        self._mode_table = mode_table
        self._mode = active_mode
        self.live_reference = provider
        self.live_mode_name = mode_name
        self._live_samples.clear()
        self._future_manual_cache.clear()

    def set_live_mode(self, mode_name: str) -> None:
        """Select a canonical online mask without resetting physical or reference state."""
        if self.live_reference is None:
            raise RuntimeError("no live reference is attached")
        mode_name = validate_online_mode(mode_name)
        mask = self._live_mode_mask(mode_name)
        candidates = {mode_name: list(ONLINE_MODES[mode_name])}
        mode_table = mask.clone()
        active_mode = mask.clone()
        self.cfg.mode_candidates = candidates
        self._mode_table = mode_table
        self._mode = active_mode
        self.live_mode_name = mode_name

    def _live_dtype_device(self) -> tuple[torch.dtype, torch.device]:
        robot = getattr(self, "robot", None)
        data = getattr(robot, "data", None)
        actual = getattr(data, "joint_pos", None)
        actual = getattr(actual, "torch", actual)
        if isinstance(actual, torch.Tensor):
            return actual.dtype, actual.device
        return torch.float32, torch.device(self.device)

    def _live_offsets(self, offsets: Sequence[int]) -> tuple[int, ...]:
        resolved: list[int] = []
        random_offset = int(getattr(self._rand_timestep, "torch", self._rand_timestep)[0, 0].item())
        for offset in offsets:
            if isinstance(offset, (bool, np.bool_)) or (
                isinstance(offset, torch.Tensor) and offset.dtype == torch.bool
            ):
                raise ValueError("live reference offsets must be integers")
            try:
                value = operator.index(offset)
            except TypeError as error:
                raise ValueError("live reference offsets must be integers") from error
            if value == -1:
                value = random_offset
            if not 0 <= value <= 32:
                raise ValueError("live reference offsets must be in [0, 32] or -1")
            resolved.append(value)
        return tuple(resolved)

    def _live_sample(self, offsets: Sequence[int]) -> dict[str, torch.Tensor]:
        if self.live_reference is None:
            raise RuntimeError("no live reference is attached")
        resolved = self._live_offsets(offsets)
        key = (self.live_reference.version, self.live_reference.frame_index)
        cached = self._live_samples.get(key)
        if cached is None:
            dtype, device = self._live_dtype_device()
            raw = self.live_reference.sample(range(33))
            cached = {
                name: torch.as_tensor(value, dtype=dtype, device=device).unsqueeze(0)
                for name, value in raw.items()
            }
            cached["body_quat"] = packed_wxyz_to_runtime(
                cached["body_quat"], self.runtime_quaternion_order
            )
            self._live_samples = {key: cached}
        indices = torch.tensor(resolved, dtype=torch.long, device=cached["body_pos"].device)
        return {name: value.index_select(1, indices) for name, value in cached.items()}

    def _live_origins(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        origins = getattr(self._env.scene.env_origins, "torch", self._env.scene.env_origins)
        return torch.as_tensor(origins, dtype=dtype, device=device)

    def _live_generic_offsets(self) -> tuple[int, ...]:
        return tuple(range(33)) + self._live_offsets([-1])

    @property
    def joint_pos(self):
        if self.live_reference is None:
            return super().joint_pos
        return self._live_sample([0])["joint_pos"][:, 0]

    @property
    def joint_vel(self):
        if self.live_reference is None:
            return super().joint_vel
        return self._live_sample([0])["joint_vel"][:, 0]

    @property
    def body_pos_w(self):
        if self.live_reference is None:
            return super().body_pos_w
        value = self._live_sample([0])["body_pos"][:, 0]
        return value + self._live_origins(value.dtype, value.device)[:, None, :]

    @property
    def body_quat_w(self):
        if self.live_reference is None:
            return super().body_quat_w
        return self._live_sample([0])["body_quat"][:, 0]

    @property
    def body_lin_vel_w(self):
        if self.live_reference is None:
            return super().body_lin_vel_w
        return self._live_sample([0])["body_lin_vel"][:, 0]

    @property
    def body_ang_vel_w(self):
        if self.live_reference is None:
            return super().body_ang_vel_w
        return self._live_sample([0])["body_ang_vel"][:, 0]

    @property
    def anchor_pos_w(self):
        return self.body_pos_w[:, self.motion_anchor_body_index]

    @property
    def anchor_quat_w(self):
        return self.body_quat_w[:, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self):
        return self.body_lin_vel_w[:, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self):
        return self.body_ang_vel_w[:, self.motion_anchor_body_index]

    def _live_future(self, name: str):
        return self._live_sample(self._live_generic_offsets())[name]

    @property
    def joint_pos_future(self):
        if self.live_reference is None:
            return super().joint_pos_future
        return self._live_future("joint_pos")

    @property
    def joint_vel_future(self):
        if self.live_reference is None:
            return super().joint_vel_future
        return self._live_future("joint_vel")

    @property
    def body_pos_w_future(self):
        if self.live_reference is None:
            return super().body_pos_w_future
        value = self._live_future("body_pos")
        return value + self._live_origins(value.dtype, value.device)[:, None, None, :]

    @property
    def body_quat_w_future(self):
        if self.live_reference is None:
            return super().body_quat_w_future
        return self._live_future("body_quat")

    @property
    def body_lin_vel_w_future(self):
        if self.live_reference is None:
            return super().body_lin_vel_w_future
        return self._live_future("body_lin_vel")

    @property
    def body_ang_vel_w_future(self):
        if self.live_reference is None:
            return super().body_ang_vel_w_future
        return self._live_future("body_ang_vel")

    @property
    def anchor_pos_w_future(self):
        return self.body_pos_w_future[:, :, self.motion_anchor_body_index]

    @property
    def anchor_quat_w_future(self):
        return self.body_quat_w_future[:, :, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w_future(self):
        return self.body_lin_vel_w_future[:, :, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w_future(self):
        return self.body_ang_vel_w_future[:, :, self.motion_anchor_body_index]

    @property
    def time_offsets_future(self):
        if self.live_reference is None:
            return super().time_offsets_future
        _, device = self._live_dtype_device()
        return torch.tensor(self._live_generic_offsets(), dtype=torch.long, device=device).view(1, -1, 1)

    def _live_manual(self, future_idx: Sequence[int], name: str):
        return self._live_sample(future_idx)[name]

    def body_pos_w_future_manual(self, future_idx: Sequence[int]):
        if self.live_reference is None:
            return super().body_pos_w_future_manual(future_idx)
        value = self._live_manual(future_idx, "body_pos")
        return value + self._live_origins(value.dtype, value.device)[:, None, None, :]

    def body_quat_w_future_manual(self, future_idx: Sequence[int]):
        if self.live_reference is None:
            return super().body_quat_w_future_manual(future_idx)
        return self._live_manual(future_idx, "body_quat")

    def body_lin_vel_w_future_manual(self, future_idx: Sequence[int]):
        if self.live_reference is None:
            return super().body_lin_vel_w_future_manual(future_idx)
        return self._live_manual(future_idx, "body_lin_vel")

    def body_ang_vel_w_future_manual(self, future_idx: Sequence[int]):
        if self.live_reference is None:
            return super().body_ang_vel_w_future_manual(future_idx)
        return self._live_manual(future_idx, "body_ang_vel")

    def time_offsets_future_manual(self, future_idx: Sequence[int]):
        if self.live_reference is None:
            return super().time_offsets_future_manual(future_idx)
        _, device = self._live_dtype_device()
        return torch.tensor(self._live_offsets(future_idx), dtype=torch.long, device=device).view(1, -1, 1)

    def _resample_command(self, env_ids: Sequence[int]):
        if self.live_reference is not None:
            raise RuntimeError("online live reference rejects command resampling")
        return super()._resample_command(env_ids)

    def _update_command(self):
        if self.live_reference is None:
            return super()._update_command()
        self.live_reference.advance()
        self._live_samples.clear()
        self._future_manual_cache.clear()

    def compute(self, dt: float):
        if self.live_reference is None:
            return super().compute(dt)
        self._update_metrics()
        self._update_command()
