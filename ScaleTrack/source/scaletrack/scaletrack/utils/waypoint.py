"""Pure-numpy bounded waypoint reference generation for online playback."""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
import numbers

import numpy as np


_TERMINAL = {"ARRIVED", "TIMED_OUT", "STALLED", "CANCELLED"}


def _positive(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite positive number")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return value


def _scalar(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def _xyz(value, name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite XYZ vector") from error
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite XYZ vector")
    return result.copy()


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class WaypointConfig:
    max_distance_m: float = 0.5
    speed_m_s: float = 0.08
    vertical_speed_m_s: float = 0.04
    max_lead_m: float = 0.10
    max_height_change_m: float = 0.15
    yaw_rate_rad_s: float = 0.25
    position_tolerance_m: float = 0.05
    height_tolerance_m: float = 0.03
    yaw_tolerance_rad: float = 5.0 * math.pi / 180.0
    speed_tolerance_m_s: float = 0.05
    angular_speed_tolerance_rad_s: float = 0.10
    settle_time_s: float = 0.5
    timeout_s: float = 15.0
    stall_time_s: float = 4.0
    progress_epsilon_m: float = 0.01
    max_heading_change_rad: float = math.pi / 4.0

    def __post_init__(self) -> None:
        for field in fields(self):
            object.__setattr__(self, field.name, _positive(getattr(self, field.name), field.name))


class WaypointFollower:
    """Rate/lead-limited reference whose terminal decisions use real state only."""

    IDLE = "IDLE"
    RUNNING = "RUNNING"
    SETTLING = "SETTLING"
    ARRIVED = "ARRIVED"
    TIMED_OUT = "TIMED_OUT"
    STALLED = "STALLED"
    CANCELLED = "CANCELLED"

    def __init__(self, cfg: WaypointConfig | None = None) -> None:
        if cfg is not None and not isinstance(cfg, WaypointConfig):
            raise TypeError("cfg must be a WaypointConfig")
        self.cfg = cfg if cfg is not None else WaypointConfig()
        self.state, self.reason = self.IDLE, None
        self._goal_xyz = self._reference_xyz = None
        self._goal_yaw = self._reference_yaw = None
        self._distance = self._yaw_error = None
        self._horizontal_error = self._vertical_error = None
        self._linear_speed = self._angular_speed = None
        self._elapsed = self._settled = 0.0
        self._best_distance = self._last_progress_elapsed = 0.0

    @property
    def goal_xyz(self):
        return None if self._goal_xyz is None else self._goal_xyz.copy()

    @property
    def goal_yaw(self):
        return self._goal_yaw

    @property
    def reference_xyz(self):
        return None if self._reference_xyz is None else self._reference_xyz.copy()

    @property
    def reference_yaw(self):
        return self._reference_yaw

    def start(
        self,
        target_xyz,
        target_yaw,
        *,
        actual_xyz,
        actual_yaw,
        reference_xyz=None,
        reference_yaw=None,
    ) -> None:
        target = _xyz(target_xyz, "target_xyz")
        actual = _xyz(actual_xyz, "actual_xyz")
        target_heading = _wrap(_scalar(target_yaw, "target_yaw"))
        actual_heading = _wrap(_scalar(actual_yaw, "actual_yaw"))
        reference = actual.copy() if reference_xyz is None else _xyz(reference_xyz, "reference_xyz")
        reference_heading = actual_heading if reference_yaw is None else _wrap(
            _scalar(reference_yaw, "reference_yaw")
        )
        distance = float(np.linalg.norm(target - actual))
        yaw_error = _wrap(target_heading - actual_heading)
        if distance > self.cfg.max_distance_m:
            raise ValueError("waypoint distance exceeds max_distance_m")
        if abs(target[2] - actual[2]) > self.cfg.max_height_change_m:
            raise ValueError("waypoint height change exceeds max_height_change_m")
        if abs(yaw_error) > self.cfg.max_heading_change_rad:
            raise ValueError("waypoint heading change exceeds max_heading_change_rad")

        height_error = abs(float(target[2] - actual[2]))
        pose_inside = (distance <= self.cfg.position_tolerance_m
                       and height_error <= self.cfg.height_tolerance_m
                       and abs(yaw_error) <= self.cfg.yaw_tolerance_rad)
        self._goal_xyz, self._goal_yaw = target, target_heading
        self._reference_xyz, self._reference_yaw = reference, reference_heading
        self.state = self.SETTLING if pose_inside else self.RUNNING
        self.reason = "within pose tolerance; waiting for stillness" if pose_inside else None
        self._distance, self._yaw_error = distance, yaw_error
        self._horizontal_error = float(np.linalg.norm((target - actual)[:2]))
        self._vertical_error = height_error
        self._linear_speed = self._angular_speed = None
        self._elapsed = self._settled = 0.0
        self._best_distance = distance
        self._last_progress_elapsed = 0.0

    def _advance_reference(self, actual: np.ndarray, dt: float) -> None:
        lead = float(np.linalg.norm(self._reference_xyz - actual))
        if lead < self.cfg.max_lead_m - 1e-12:
            delta = self._goal_xyz - self._reference_xyz
            distance = float(np.linalg.norm(delta))
            if distance > 0.0:
                move = delta * (min(distance, self.cfg.speed_m_s * dt) / distance)
                move[2] = np.clip(move[2], -self.cfg.vertical_speed_m_s * dt, self.cfg.vertical_speed_m_s * dt)
                candidate = self._reference_xyz + move
                if np.linalg.norm(candidate - actual) > self.cfg.max_lead_m:
                    offset = self._reference_xyz - actual
                    a = float(np.dot(move, move))
                    b = float(np.dot(offset, move))
                    c = float(np.dot(offset, offset) - self.cfg.max_lead_m**2)
                    fraction = (-b + math.sqrt(max(0.0, b * b - a * c))) / a
                    candidate = self._reference_xyz + np.clip(fraction, 0.0, 1.0) * move
                self._reference_xyz = candidate
        yaw_delta = _wrap(self._goal_yaw - self._reference_yaw)
        yaw_step = min(abs(yaw_delta), self.cfg.yaw_rate_rad_s * dt)
        self._reference_yaw = _wrap(self._reference_yaw + math.copysign(yaw_step, yaw_delta))

    def preview(self, actual_xyz, dt, steps=32):
        """Predict a bounded reference horizon without changing follower state."""
        if self.state not in (self.RUNNING, self.SETTLING, self.ARRIVED):
            raise RuntimeError(f"cannot preview waypoint in {self.state} state")
        actual = _xyz(actual_xyz, "actual_xyz")
        dt = _positive(dt, "dt")
        if isinstance(steps, (bool, np.bool_)) or not isinstance(steps, numbers.Integral):
            raise ValueError("steps must be a non-bool integer in [0, 32]")
        steps = int(steps)
        if not 0 <= steps <= 32:
            raise ValueError("steps must be a non-bool integer in [0, 32]")

        positions = np.empty((steps + 1, 3), dtype=np.float64)
        yaws = np.empty(steps + 1, dtype=np.float64)
        positions[0], yaws[0] = self._reference_xyz, self._reference_yaw
        if self.state == self.ARRIVED:
            positions[:] = positions[0]
            yaws[:] = yaws[0]
            return positions, yaws

        prediction = WaypointFollower(self.cfg)
        prediction._goal_xyz = self._goal_xyz.copy()
        prediction._goal_yaw = self._goal_yaw
        prediction._reference_xyz = self._reference_xyz.copy()
        prediction._reference_yaw = self._reference_yaw
        for index in range(1, steps + 1):
            prediction._advance_reference(actual, dt)
            positions[index] = prediction._reference_xyz
            yaws[index] = prediction._reference_yaw
        return positions, yaws

    def step(self, actual_xyz, actual_yaw, linear_speed, angular_speed, dt):
        if self.state == self.IDLE:
            raise RuntimeError("waypoint follower has not been started")
        if self.state in _TERMINAL:
            return self._reference_xyz.copy(), self._reference_yaw
        actual = _xyz(actual_xyz, "actual_xyz")
        actual_heading = _wrap(_scalar(actual_yaw, "actual_yaw"))
        linear = abs(_scalar(linear_speed, "linear_speed"))
        angular = abs(_scalar(angular_speed, "angular_speed"))
        dt = _positive(dt, "dt")

        self._advance_reference(actual, dt)
        self._elapsed += dt
        difference = self._goal_xyz - actual
        distance = float(np.linalg.norm(difference))
        horizontal_error = float(np.linalg.norm(difference[:2]))
        vertical_error = abs(float(difference[2]))
        yaw_error = _wrap(self._goal_yaw - actual_heading)
        if distance <= self._best_distance - self.cfg.progress_epsilon_m:
            self._best_distance = distance
            self._last_progress_elapsed = self._elapsed
        if distance <= self.cfg.position_tolerance_m:
            # Rotation/stillness waiting inside the positional gate does not
            # consume the later positional-stall budget if the robot drifts out.
            self._last_progress_elapsed = self._elapsed
        pose_inside = (distance <= self.cfg.position_tolerance_m
                       and vertical_error <= self.cfg.height_tolerance_m
                       and abs(yaw_error) <= self.cfg.yaw_tolerance_rad)
        still = linear <= self.cfg.speed_tolerance_m_s and angular <= self.cfg.angular_speed_tolerance_rad_s
        if pose_inside:
            self.state = self.SETTLING
            self.reason = "within pose tolerance; waiting for stillness"
            self._settled = self._settled + dt if still else 0.0
        else:
            self.state, self.reason, self._settled = self.RUNNING, None, 0.0

        if self._settled >= self.cfg.settle_time_s:
            self.state, self.reason = self.ARRIVED, "arrival tolerances held for settle_time_s"
        elif self._elapsed >= self.cfg.timeout_s:
            self.state, self.reason = self.TIMED_OUT, "waypoint timeout"
        elif distance > self.cfg.position_tolerance_m and self._elapsed - self._last_progress_elapsed >= self.cfg.stall_time_s:
            self.state, self.reason = self.STALLED, "no actual position progress"
        self._distance, self._yaw_error = distance, yaw_error
        self._horizontal_error, self._vertical_error = horizontal_error, vertical_error
        self._linear_speed, self._angular_speed = linear, angular
        return self._reference_xyz.copy(), self._reference_yaw

    def cancel(self, reason: str) -> None:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("cancel reason must be a nonempty string")
        if self.state == self.IDLE:
            raise RuntimeError("waypoint follower has not been started")
        self.state, self.reason = self.CANCELLED, reason

    def status(self) -> dict:
        return {
            "state": self.state,
            "reason": self.reason,
            "goal_xyz": None if self._goal_xyz is None else self._goal_xyz.copy(),
            "goal_yaw": self._goal_yaw,
            "reference_xyz": None if self._reference_xyz is None else self._reference_xyz.copy(),
            "reference_yaw": self._reference_yaw,
            "distance_m": self._distance,
            "horizontal_error_m": self._horizontal_error,
            "vertical_error_m": self._vertical_error,
            "yaw_error_rad": self._yaw_error,
            "linear_speed_m_s": self._linear_speed,
            "angular_speed_rad_s": self._angular_speed,
            "elapsed_s": self._elapsed,
            "settled_s": self._settled,
        }
