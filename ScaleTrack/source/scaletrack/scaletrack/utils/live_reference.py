"""Atomic, CPU-only reference trajectories for the three VR-3 body points."""

from __future__ import annotations

from dataclasses import dataclass
import numbers
from typing import Final

import numpy as np


VR3_BODY_NAMES: Final = ("pelvis", "left_wrist_yaw_link", "right_wrist_yaw_link")
_VR3_BODY_INDICES: Final = (0, 10, 13)


class ReferenceInputError(ValueError):
    """Invalid provider input, with an explicit marker for clock corruption."""

    def __init__(self, message: str, *, clock_fault: bool = False) -> None:
        super().__init__(message)
        self.clock_fault = clock_fault


@dataclass(frozen=True)
class LiveReferenceLimits:
    """Fixed rate and workspace limits for the initial single-robot mode."""

    pelvis_speed: float = 0.2
    wrist_speed: float = 0.35
    pelvis_angular_speed: float = 0.5
    wrist_angular_speed: float = 1.0
    pelvis_radius: float = 1.0
    pelvis_height_delta: float = 0.15
    pelvis_height_min: float = 0.35
    pelvis_height_max: float = 1.25
    wrist_radius: float = 1.0


@dataclass(frozen=True)
class _Packet:
    positions: np.ndarray
    quaternions: np.ndarray
    sequence: int
    stamp: float
    trajectory_positions: np.ndarray | None = None
    trajectory_quaternions: np.ndarray | None = None


def _as_float_array(value: object, shape: tuple[int, ...], label: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ReferenceInputError(f"{label} must be a numeric array") from error
    if array.shape != shape:
        raise ReferenceInputError(f"{label} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ReferenceInputError(f"{label} must be finite")
    return array.copy()


def _normalize_quaternions(value: object, shape: tuple[int, ...], label: str) -> np.ndarray:
    quaternions = _as_float_array(value, shape, label)
    norms = np.linalg.norm(quaternions, axis=-1)
    if np.any(norms == 0.0) or np.any(np.abs(norms - 1.0) > 1e-3):
        raise ReferenceInputError(f"{label} must contain nonzero, unit wxyz quaternions")
    return quaternions / norms[..., None]


def _finite_positive(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ReferenceInputError(f"{label} must be a finite positive number")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ReferenceInputError(f"{label} must be a finite positive number")
    return result


def _clock_value(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ReferenceInputError(f"{label} must be a finite nonnegative timestamp", clock_fault=True)
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ReferenceInputError(f"{label} must be a finite nonnegative timestamp", clock_fault=True)
    return result


def _quat_conjugate(quaternion: np.ndarray) -> np.ndarray:
    return np.array(
        [quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]], dtype=np.float64
    )


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )


def _step_quaternion(current: np.ndarray, goal: np.ndarray, max_angle: float) -> np.ndarray:
    """Move on the shortest unit-quaternion arc by no more than ``max_angle``."""
    destination = goal.copy()
    dot = float(np.dot(current, destination))
    if dot < 0.0:
        destination *= -1.0
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    half_angle = float(np.arccos(dot))
    angle = 2.0 * half_angle
    if angle <= max_angle or angle <= 1e-12:
        return destination
    fraction = max_angle / angle
    sine = float(np.sin(half_angle))
    if abs(sine) <= 1e-12:
        result = (1.0 - fraction) * current + fraction * destination
    else:
        result = (
            np.sin((1.0 - fraction) * half_angle) / sine * current
            + np.sin(fraction * half_angle) / sine * destination
        )
    return result / np.linalg.norm(result)


def _world_angular_velocity(current: np.ndarray, following: np.ndarray, step_dt: float) -> np.ndarray:
    """Return the shortest world-frame rotation-log difference per second."""
    difference = _quat_multiply(following, _quat_conjugate(current))
    if difference[0] < 0.0:
        difference *= -1.0
    vector = difference[1:]
    vector_norm = float(np.linalg.norm(vector))
    if vector_norm <= 1e-12:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * float(np.arctan2(vector_norm, float(np.clip(difference[0], -1.0, 1.0))))
    return vector * (angle / (vector_norm * step_dt))


class LiveReferenceProvider:
    """Own complete goal packets and create a bounded, continuous reference horizon."""

    def __init__(
        self,
        body_names: object,
        body_pos: object,
        body_quat: object,
        joint_pos: object,
        *,
        step_dt: float = 0.02,
        limits: LiveReferenceLimits | None = None,
    ) -> None:
        self._body_names = self._validate_body_names(body_names)
        self._step_dt = _finite_positive(step_dt, "step_dt")
        self._limits = limits if limits is not None else LiveReferenceLimits()
        self._validate_limits(self._limits)
        positions, quaternions, joints = self._validate_seed(
            body_pos, body_quat, joint_pos, check_workspace=True
        )
        self._workspace_pelvis = positions[0].copy()
        self._pelvis_z_min = max(
            self._limits.pelvis_height_min,
            self._workspace_pelvis[2] - self._limits.pelvis_height_delta,
        )
        self._pelvis_z_max = min(
            self._limits.pelvis_height_max,
            self._workspace_pelvis[2] + self._limits.pelvis_height_delta,
        )
        if self._pelvis_z_min > self._pelvis_z_max:
            raise ReferenceInputError("seed pelvis height has no valid workspace")
        self._seed_body_pos = positions
        self._seed_body_quat = quaternions
        self._seed_joint_pos = joints
        self._current_body_pos = positions.copy()
        self._current_body_quat = quaternions.copy()
        self._goal_positions = positions[list(_VR3_BODY_INDICES)].copy()
        self._goal_quaternions = quaternions[list(_VR3_BODY_INDICES)].copy()
        self._pending: _Packet | None = None
        self._trajectory_positions: np.ndarray | None = None
        self._trajectory_quaternions: np.ndarray | None = None
        self._trajectory_cursor = 0
        self._sequence_watermark: int | None = None
        self._stamp_watermark: float | None = None
        self._now_watermark: float | None = None
        self._last_sequence: int | None = None
        self._version = 0
        self._frame_index = 0
        self._rejected_count = 0
        self._last_error: str | None = None
        self._closed = False

    @staticmethod
    def _validate_body_names(body_names: object) -> tuple[str, ...]:
        try:
            names = tuple(body_names)  # type: ignore[arg-type]
        except TypeError as error:
            raise ReferenceInputError("body_names must be an iterable of 14 unique names") from error
        if len(names) != 14 or any(not isinstance(name, str) for name in names):
            raise ReferenceInputError("body_names must be 14 strings")
        if len(set(names)) != len(names):
            raise ReferenceInputError("body_names must be unique")
        if tuple(names[index] for index in _VR3_BODY_INDICES) != VR3_BODY_NAMES:
            raise ReferenceInputError("VR-3 bodies must resolve to indices 0, 10, and 13")
        return names

    @staticmethod
    def _validate_limits(limits: object) -> None:
        if not isinstance(limits, LiveReferenceLimits):
            raise ReferenceInputError("limits must be a LiveReferenceLimits instance")
        for name, value in vars(limits).items():
            _finite_positive(value, f"limits.{name}")
        if limits.pelvis_height_min > limits.pelvis_height_max:
            raise ReferenceInputError("pelvis height minimum must not exceed its maximum")

    def _validate_seed(
        self, body_pos: object, body_quat: object, joint_pos: object, *, check_workspace: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        positions = _as_float_array(body_pos, (14, 3), "body_pos")
        quaternions = _normalize_quaternions(body_quat, (14, 4), "body_quat")
        joints = _as_float_array(joint_pos, (29,), "joint_pos")
        if check_workspace:
            pelvis = positions[0]
            if not self._limits.pelvis_height_min <= pelvis[2] <= self._limits.pelvis_height_max:
                raise ReferenceInputError("seed pelvis height is outside the absolute workspace")
            for index in _VR3_BODY_INDICES[1:]:
                if np.linalg.norm(positions[index] - pelvis) > self._limits.wrist_radius:
                    raise ReferenceInputError("seed wrist is outside the pelvis-relative workspace")
        return positions, quaternions, joints

    def _reject(self, message: str, *, clock_fault: bool = False) -> None:
        self._rejected_count += 1
        self._last_error = message
        raise ReferenceInputError(message, clock_fault=clock_fault)

    def _ensure_open(self) -> None:
        if self._closed:
            self._reject("live reference provider is closed")

    def _validate_packet_clock(self, stamp: object, now: object) -> tuple[float, float]:
        try:
            packet_stamp = _clock_value(stamp, "stamp")
            consumer_now = _clock_value(now, "now")
        except ReferenceInputError as error:
            self._reject(str(error), clock_fault=True)
        if self._now_watermark is not None and consumer_now < self._now_watermark:
            self._reject("now cannot move backwards", clock_fault=True)
        if packet_stamp > consumer_now:
            self._reject("stamp cannot be in the future", clock_fault=True)
        if self._stamp_watermark is not None and packet_stamp < self._stamp_watermark:
            self._reject("stamp cannot move backwards", clock_fault=True)
        return packet_stamp, consumer_now

    def _validate_consumer_now(self, now: object) -> float:
        try:
            consumer_now = _clock_value(now, "now")
        except ReferenceInputError as error:
            self._reject(str(error), clock_fault=True)
        if self._now_watermark is not None and consumer_now < self._now_watermark:
            self._reject("now cannot move backwards", clock_fault=True)
        return consumer_now

    def _validate_packet(
        self,
        positions: object,
        quaternions: object,
        *,
        sequence: object,
        stamp: object,
        now: object,
        body_names: object,
        frame: object,
        quaternion_order: object,
    ) -> tuple[_Packet, float]:
        if isinstance(sequence, bool) or not isinstance(sequence, numbers.Integral):
            self._reject("sequence must be a non-bool integer")
        packet_sequence = int(sequence)
        if self._sequence_watermark is not None and packet_sequence <= self._sequence_watermark:
            self._reject("sequence must strictly increase")
        if not isinstance(frame, str) or frame != "env_local":
            self._reject("frame must be env_local")
        if not isinstance(quaternion_order, str) or quaternion_order != "wxyz":
            self._reject("quaternion_order must be wxyz")
        try:
            packet_names = tuple(body_names)  # type: ignore[arg-type]
        except TypeError:
            self._reject("packet body_names must contain each VR-3 body once")
        if (
            len(packet_names) != 3
            or any(not isinstance(name, str) for name in packet_names)
            or len(set(packet_names)) != 3
            or set(packet_names) != set(VR3_BODY_NAMES)
        ):
            self._reject("packet body_names must contain each VR-3 body once")
        try:
            packet_positions = _as_float_array(positions, (3, 3), "positions")
            packet_quaternions = _normalize_quaternions(quaternions, (3, 4), "quaternions")
        except ReferenceInputError as error:
            self._reject(str(error), clock_fault=error.clock_fault)
        packet_stamp, consumer_now = self._validate_packet_clock(stamp, now)
        ordering = [packet_names.index(name) for name in VR3_BODY_NAMES]
        packet_positions = packet_positions[ordering].copy()
        packet_quaternions = packet_quaternions[ordering].copy()
        try:
            packet_positions, packet_quaternions = self.validate_targets(packet_positions, packet_quaternions)
        except ReferenceInputError as error:
            self._reject(str(error))
        return _Packet(packet_positions, packet_quaternions, packet_sequence, packet_stamp), consumer_now

    def validate_targets(self, positions: object, quaternions: object) -> tuple[np.ndarray, np.ndarray]:
        """Check canonical targets without queuing packets or changing clocks/counters.

        A task planner can preflight its final destination before sending any
        intermediate references. Packet sequencing/liveness are still checked by submit.
        """
        if self._closed:
            raise ReferenceInputError("live reference provider is closed")
        positions = _as_float_array(positions, (3, 3), "positions")
        quaternions = _normalize_quaternions(quaternions, (3, 4), "quaternions")
        pelvis = positions[0]
        if np.linalg.norm(pelvis[:2] - self._workspace_pelvis[:2]) > self._limits.pelvis_radius:
            raise ReferenceInputError("pelvis target exceeds the horizontal workspace")
        if not self._pelvis_z_min <= pelvis[2] <= self._pelvis_z_max:
            raise ReferenceInputError("pelvis target exceeds the height workspace")
        if np.any(np.linalg.norm(positions[1:] - pelvis, axis=1) > self._limits.wrist_radius):
            raise ReferenceInputError("wrist target exceeds the pelvis-relative workspace")
        return positions.copy(), quaternions.copy()

    @property
    def body_names(self) -> tuple[str, ...]:
        return self._body_names

    @property
    def step_dt(self) -> float:
        return self._step_dt

    @property
    def limits(self) -> LiveReferenceLimits:
        return self._limits

    @property
    def version(self) -> int:
        return self._version

    @property
    def frame_index(self) -> int:
        return self._frame_index

    @property
    def last_sequence(self) -> int | None:
        return self._last_sequence

    @property
    def rejected_count(self) -> int:
        return self._rejected_count

    @property
    def last_error(self) -> str | None:
        return self._last_error

    @property
    def goal_positions(self) -> np.ndarray:
        return self._goal_positions.copy()

    @property
    def goal_quaternions(self) -> np.ndarray:
        return self._goal_quaternions.copy()

    def submit(
        self,
        positions: object,
        quaternions: object,
        *,
        sequence: object,
        stamp: object,
        now: object,
        body_names: object = VR3_BODY_NAMES,
        frame: str = "env_local",
        quaternion_order: str = "wxyz",
    ) -> None:
        """Validate and atomically replace the one-slot pending packet."""
        self._ensure_open()
        packet, consumer_now = self._validate_packet(
            positions,
            quaternions,
            sequence=sequence,
            stamp=stamp,
            now=now,
            body_names=body_names,
            frame=frame,
            quaternion_order=quaternion_order,
        )
        self._pending = packet
        self._sequence_watermark = packet.sequence
        self._stamp_watermark = packet.stamp
        self._now_watermark = consumer_now
        self._last_error = None

    def commit_pending(self, now: object) -> bool:
        """Latch the newest complete packet without moving the current reference."""
        self._ensure_open()
        if self._pending is None:
            self._now_watermark = self._validate_consumer_now(now)
            return False
        try:
            packet_stamp, consumer_now = self._validate_packet_clock(self._pending.stamp, now)
        except ReferenceInputError:
            raise
        # ``_Packet`` snapshots arrays, but copying here keeps this boundary robust to mutation.
        packet = self._pending
        trajectory_positions = trajectory_quaternions = None
        if packet.trajectory_positions is not None:
            self._validate_trajectory_anchor(packet.trajectory_positions[0], packet.trajectory_quaternions[0])
            trajectory_positions = np.broadcast_to(self._current_body_pos, (34, 14, 3)).copy()
            trajectory_quaternions = np.broadcast_to(self._current_body_quat, (34, 14, 4)).copy()
            trajectory_positions[:, list(_VR3_BODY_INDICES)] = packet.trajectory_positions
            trajectory_quaternions[:, list(_VR3_BODY_INDICES)] = packet.trajectory_quaternions
        self._goal_positions = packet.positions.copy()
        self._goal_quaternions = packet.quaternions.copy()
        self._trajectory_positions = trajectory_positions
        self._trajectory_quaternions = trajectory_quaternions
        self._trajectory_cursor = 0
        self._last_sequence = packet.sequence
        self._stamp_watermark = packet_stamp
        self._now_watermark = consumer_now
        self._pending = None
        self._last_error = None
        self._version += 1
        return True

    def _validate_trajectory_anchor(self, positions: np.ndarray, quaternions: np.ndarray) -> None:
        """A forecast may not teleport its current frame, even after queuing."""
        current_positions = self._current_body_pos[list(_VR3_BODY_INDICES)]
        current_quaternions = self._current_body_quat[list(_VR3_BODY_INDICES)]
        quaternion_difference = np.minimum(
            np.linalg.norm(quaternions - current_quaternions, axis=-1),
            np.linalg.norm(quaternions + current_quaternions, axis=-1),
        )
        if (not np.allclose(positions, current_positions, rtol=0.0, atol=1e-8)
                or np.any(quaternion_difference > 1e-8)):
            self._reject("trajectory frame 0 must match the current reference")

    def submit_trajectory(
        self, positions: object, quaternions: object, *, sequence: object, stamp: object, now: object
    ) -> None:
        """Queue 34 canonical env-local/wxyz frames, including the unchanged current frame.

        Frames 1..33 are explicit future references at ``step_dt`` intervals.
        Frame 33 supplies the velocity difference for sample offset 32. Each
        frame must satisfy the same workspace and rates as an ordinary goal.
        No frame is consumed by submit, commit, or sample; only advance moves it.
        """
        self._ensure_open()
        try:
            trajectory_positions = _as_float_array(positions, (34, 3, 3), "trajectory positions")
            trajectory_quaternions = _normalize_quaternions(quaternions, (34, 3, 4), "trajectory quaternions")
            for frame in range(34):
                self.validate_targets(trajectory_positions[frame], trajectory_quaternions[frame])
        except ReferenceInputError as error:
            self._reject(str(error), clock_fault=error.clock_fault)
        self._validate_trajectory_anchor(trajectory_positions[0], trajectory_quaternions[0])
        speeds = np.array([self._limits.pelvis_speed, self._limits.wrist_speed, self._limits.wrist_speed])
        angular_speeds = np.array([
            self._limits.pelvis_angular_speed, self._limits.wrist_angular_speed, self._limits.wrist_angular_speed
        ])
        distances = np.linalg.norm(np.diff(trajectory_positions, axis=0), axis=-1)
        dots = np.abs(np.sum(trajectory_quaternions[1:] * trajectory_quaternions[:-1], axis=-1))
        angles = 2.0 * np.arccos(np.clip(dots, 0.0, 1.0))
        if np.any(distances > speeds * self._step_dt + 1e-9):
            self._reject("trajectory exceeds reference linear speed limits")
        if np.any(angles > angular_speeds * self._step_dt + 1e-9):
            self._reject("trajectory exceeds reference angular speed limits")
        packet, consumer_now = self._validate_packet(
            trajectory_positions[-1], trajectory_quaternions[-1], sequence=sequence, stamp=stamp, now=now,
            body_names=VR3_BODY_NAMES, frame="env_local", quaternion_order="wxyz",
        )
        self._pending = _Packet(
            packet.positions, packet.quaternions, packet.sequence, packet.stamp,
            trajectory_positions, trajectory_quaternions,
        )
        self._sequence_watermark = packet.sequence
        self._stamp_watermark = packet.stamp
        self._now_watermark = consumer_now
        self._last_error = None

    def _step_toward_goal(
        self, positions: np.ndarray, quaternions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        next_positions = positions.copy()
        next_quaternions = quaternions.copy()
        speeds = (self._limits.pelvis_speed, self._limits.wrist_speed, self._limits.wrist_speed)
        angular_speeds = (
            self._limits.pelvis_angular_speed,
            self._limits.wrist_angular_speed,
            self._limits.wrist_angular_speed,
        )
        for target_index, body_index in enumerate(_VR3_BODY_INDICES):
            difference = self._goal_positions[target_index] - positions[body_index]
            distance = float(np.linalg.norm(difference))
            if distance > 0.0:
                fraction = min(1.0, speeds[target_index] * self._step_dt / distance)
                next_positions[body_index] = positions[body_index] + fraction * difference
            next_quaternions[body_index] = _step_quaternion(
                quaternions[body_index],
                self._goal_quaternions[target_index],
                angular_speeds[target_index] * self._step_dt,
            )
        return next_positions, next_quaternions

    def advance(self) -> None:
        """Advance exactly one configured reference step."""
        self._ensure_open()
        if self._trajectory_positions is None:
            self._current_body_pos, self._current_body_quat = self._step_toward_goal(
                self._current_body_pos, self._current_body_quat
            )
        else:
            self._trajectory_cursor = min(self._trajectory_cursor + 1, 33)
            self._current_body_pos = self._trajectory_positions[self._trajectory_cursor].copy()
            self._current_body_quat = self._trajectory_quaternions[self._trajectory_cursor].copy()
        self._frame_index += 1
        self._version += 1

    def sample(self, offsets: object) -> dict[str, np.ndarray]:
        """Return independent current/future buffers without advancing the reference."""
        try:
            offset_array = np.asarray(offsets)
        except (TypeError, ValueError) as error:
            raise ReferenceInputError("offsets must be a one-dimensional integer array") from error
        if offset_array.ndim != 1 or not np.issubdtype(offset_array.dtype, np.integer):
            raise ReferenceInputError("offsets must be a one-dimensional integer array")
        if np.any(offset_array < 0) or np.any(offset_array > 32):
            raise ReferenceInputError("offsets must be in [0, 32]")
        if self._trajectory_positions is None:
            frames_pos = np.empty((34, 14, 3), dtype=np.float64)
            frames_quat = np.empty((34, 14, 4), dtype=np.float64)
            frames_pos[0] = self._current_body_pos
            frames_quat[0] = self._current_body_quat
            for frame in range(1, 34):
                frames_pos[frame], frames_quat[frame] = self._step_toward_goal(
                    frames_pos[frame - 1], frames_quat[frame - 1]
                )
        else:
            indices = np.minimum(self._trajectory_cursor + np.arange(34), 33)
            frames_pos = self._trajectory_positions[indices]
            frames_quat = self._trajectory_quaternions[indices]
        body_linear_velocity = (frames_pos[1:] - frames_pos[:-1]) / self._step_dt
        body_angular_velocity = np.empty((33, 14, 3), dtype=np.float64)
        for frame in range(33):
            for body_index in range(14):
                body_angular_velocity[frame, body_index] = _world_angular_velocity(
                    frames_quat[frame, body_index], frames_quat[frame + 1, body_index], self._step_dt
                )
        selected = offset_array.astype(np.intp, copy=False)
        count = len(selected)
        return {
            "body_pos": frames_pos[selected].copy(),
            "body_quat": frames_quat[selected].copy(),
            "body_lin_vel": body_linear_velocity[selected].copy(),
            "body_ang_vel": body_angular_velocity[selected].copy(),
            "joint_pos": np.broadcast_to(self._seed_joint_pos, (count, 29)).copy(),
            "joint_vel": np.zeros((count, 29), dtype=np.float64),
        }

    def reset_reference(self, body_pos: object, body_quat: object, joint_pos: object) -> None:
        """Reseed without reopening the old packet or redefining the initial workspace."""
        self._ensure_open()
        try:
            positions, quaternions, joints = self._validate_seed(
                body_pos, body_quat, joint_pos, check_workspace=False
            )
        except ReferenceInputError as error:
            self._reject(str(error), clock_fault=error.clock_fault)
        self._seed_body_pos = positions
        self._seed_body_quat = quaternions
        self._seed_joint_pos = joints
        self._current_body_pos = positions.copy()
        self._current_body_quat = quaternions.copy()
        self._goal_positions = positions[list(_VR3_BODY_INDICES)].copy()
        self._goal_quaternions = quaternions[list(_VR3_BODY_INDICES)].copy()
        self._pending = None
        self._frame_index = 0
        self._trajectory_positions = self._trajectory_quaternions = None
        self._trajectory_cursor = 0
        self._last_error = None
        self._version += 1

    def close(self) -> None:
        """Prevent any later state-changing use of the provider."""
        self._closed = True
        self._pending = None
        self._trajectory_positions = self._trajectory_quaternions = None
        self._trajectory_cursor = 0
