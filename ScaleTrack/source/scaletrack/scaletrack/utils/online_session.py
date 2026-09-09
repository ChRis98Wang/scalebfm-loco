"""Pure safety state for a locally supplied live reference.

This module deliberately owns no environment, simulator, or UI objects.  It
only decides whether the caller may take the next physical step.
"""

from __future__ import annotations

import numbers

import numpy as np


class OnlineSession:
    """Latch live-control pauses until a caller explicitly performs recovery."""

    READY = "READY"
    ACTIVE = "ACTIVE"
    PAUSED_USER = "PAUSED_USER"
    PAUSED_FAULT = "PAUSED_FAULT"
    CLOSED = "CLOSED"

    def __init__(self, provider, *, timeout: float = 0.5) -> None:
        if not isinstance(timeout, numbers.Real) or isinstance(timeout, bool) or timeout <= 0:
            raise ValueError("timeout must be a positive finite number")
        self.provider = provider
        self.timeout = float(timeout)
        if not np.isfinite(self.timeout):
            raise ValueError("timeout must be a positive finite number")
        self.state = self.READY
        self.reason = "ready"
        self._heartbeat_stamp: float | None = None
        self._now_watermark: float | None = None
        self._fatal_fault = False

    @staticmethod
    def _clock(value, label: str) -> float:
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise ValueError(f"{label} must be a finite nonnegative timestamp")
        result = float(value)
        if not np.isfinite(result) or result < 0.0:
            raise ValueError(f"{label} must be a finite nonnegative timestamp")
        return result

    def _fault(self, reason: str, *, fatal: bool = False) -> None:
        if self.state == self.CLOSED:
            return
        self.state = self.PAUSED_FAULT
        self.reason = str(reason)
        self._fatal_fault = self._fatal_fault or fatal

    def _validate_now(self, now) -> float:
        try:
            current = self._clock(now, "now")
        except ValueError as error:
            self._fault(str(error))
            raise RuntimeError(self.reason) from error
        if self._now_watermark is not None and current < self._now_watermark:
            self._fault("time moved backwards")
            raise RuntimeError(self.reason)
        self._now_watermark = current
        return current

    def heartbeat(self, stamp, now) -> None:
        """Record producer liveness without changing a paused state."""
        if self.state == self.CLOSED:
            raise RuntimeError("online session is closed")
        try:
            packet_stamp = self._clock(stamp, "stamp")
            current = self._validate_now(now)
            if packet_stamp > current:
                raise ValueError("heartbeat stamp is in the future")
            if self._heartbeat_stamp is not None and packet_stamp < self._heartbeat_stamp:
                raise ValueError("heartbeat stamp moved backwards")
        except (ValueError, RuntimeError) as error:
            self._fault(str(error))
            raise RuntimeError(self.reason) from error
        self._heartbeat_stamp = packet_stamp

    def _fresh(self, now: float) -> bool:
        return self._heartbeat_stamp is not None and now - self._heartbeat_stamp <= self.timeout

    @staticmethod
    def _as_array(value, shape: tuple[int, ...], label: str) -> np.ndarray:
        value = getattr(value, "torch", value)
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        try:
            array = np.asarray(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label} must have shape {shape}") from error
        if array.shape != shape or not np.issubdtype(array.dtype, np.number) or not np.all(np.isfinite(array)):
            raise ValueError(f"{label} must be finite with shape {shape}")
        return array.astype(np.float64, copy=False)

    def _healthy(self, body_pos, body_quat, joint_pos, joint_vel=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        positions = self._as_array(body_pos, (14, 3), "body_pos")
        quaternions = self._as_array(body_quat, (14, 4), "body_quat")
        joints = self._as_array(joint_pos, (29,), "joint_pos")
        if joint_vel is not None:
            self._as_array(joint_vel, (29,), "joint_vel")
        pelvis = positions[0]
        if pelvis[2] < 0.35:
            raise ValueError("pelvis fall: height is below 0.35 m")
        quaternion = quaternions[0]
        norm = np.linalg.norm(quaternion)
        if norm <= 1e-12:
            raise ValueError("pelvis orientation is nonfinite or zero")
        # The provider's canonical pose contract is wxyz.  A caller using a
        # runtime order converts at its boundary before entering this pure term.
        w, x, y, z = quaternion / norm
        local_up_dot_world_up = 1.0 - 2.0 * (x * x + y * y)
        if local_up_dot_world_up < 0.5:
            raise ValueError("pelvis fall: local up is tilted below 0.5")
        return positions, quaternions, joints

    def activate(self, now, body_pos, body_quat, joint_pos) -> None:
        if self.state != self.READY:
            raise RuntimeError(f"activate requires READY, got {self.state}")
        current = self._validate_now(now)
        if not self._fresh(current):
            self._fault("heartbeat is stale")
            raise RuntimeError(self.reason)
        try:
            positions, quaternions, joints = self._healthy(body_pos, body_quat, joint_pos)
        except ValueError as error:
            self._fault(str(error), fatal=True)
            raise RuntimeError(self.reason) from error
        self.provider.reset_reference(positions, quaternions, joints)
        self.state = self.ACTIVE
        self.reason = "active"

    def pause(self, reason: str = "user") -> None:
        if self.state == self.CLOSED:
            raise RuntimeError("online session is closed")
        if self.state == self.ACTIVE:
            self.state = self.PAUSED_USER
            self.reason = str(reason)

    def fault(self, reason: str) -> None:
        self._fault(reason, fatal=False)

    def check(self, now, body_pos, body_quat, joint_pos, joint_vel) -> bool:
        """Validate actual telemetry and liveness before a physical step."""
        if self.state != self.ACTIVE:
            return False
        try:
            current = self._validate_now(now)
        except RuntimeError:
            return False
        if not self._fresh(current):
            self._fault("heartbeat is stale")
            return False
        try:
            self._healthy(body_pos, body_quat, joint_pos, joint_vel)
        except ValueError as error:
            self._fault(str(error), fatal=True)
            return False
        return True

    def resume(self, now, body_pos, body_quat, joint_pos) -> None:
        if self.state not in (self.PAUSED_USER, self.PAUSED_FAULT):
            raise RuntimeError(f"resume requires a paused state, got {self.state}")
        if self._fatal_fault:
            raise RuntimeError(f"cannot resume fatal fault: {self.reason}")
        current = self._validate_now(now)
        if not self._fresh(current):
            raise RuntimeError("resume requires a fresh heartbeat")
        try:
            positions, quaternions, joints = self._healthy(body_pos, body_quat, joint_pos)
        except ValueError as error:
            self._fault(str(error), fatal=True)
            raise RuntimeError(self.reason) from error
        self.provider.reset_reference(positions, quaternions, joints)
        self.state = self.ACTIVE
        self.reason = "active"

    def close(self) -> None:
        if self.state != self.CLOSED:
            self.provider.close()
            self.state = self.CLOSED
            self.reason = "closed"
