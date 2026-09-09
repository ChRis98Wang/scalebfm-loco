"""Native Kit controls and a deliberately small, physics-free intent boundary."""

from __future__ import annotations

import time

import numpy as np

from scaletrack.utils.live_reference import VR3_BODY_NAMES
from scaletrack.utils.online_modes import ONLINE_MODE_NAMES, online_target_rows, validate_online_mode
from scaletrack.utils.quaternion_compat import packed_wxyz_to_runtime

_POINT_LABELS = ("Pelvis", "Left wrist", "Right wrist")
_AXES = ("X", "Y", "Z")
_SUMMARY_KEYS = ("mode", "state", "reason", "awaiting_goal", "waypoint_state",
                 "accepted_sequence", "input_age", "last_error")
_DETAIL_KEYS = ("active_body_names", "active_target_rows", "mode_rejected_count", "rejected_count",
                "active_position_errors", "active_reference_position_errors",
                "waypoint_reason", "waypoint_target_xyz", "waypoint_distance_m",
                "waypoint_horizontal_error_m", "waypoint_vertical_error_m", "waypoint_yaw_error_rad",
                "waypoint_linear_speed_m_s", "waypoint_angular_speed_rad_s", "waypoint_elapsed_s",
                "waypoint_settled_s",
                "goal_positions", "goal_quaternions_wxyz", "reference_positions",
                "reference_quaternions_wxyz", "actual_positions", "actual_quaternions_wxyz",
                "position_errors", "orientation_errors")


def _wxyz_from_xyz_degrees(degrees: np.ndarray) -> np.ndarray:
    half = np.deg2rad(degrees) * 0.5
    cx, cy, cz = np.cos(half[:, 0]), np.cos(half[:, 1]), np.cos(half[:, 2])
    sx, sy, sz = np.sin(half[:, 0]), np.sin(half[:, 1]), np.sin(half[:, 2])
    return np.column_stack((cx * cy * cz + sx * sy * sz, sx * cy * cz - cx * sy * sz,
                            cx * sy * cz + sx * cy * sz, cx * cy * sz - sx * sy * cz))


def _xyz_degrees_from_wxyz(quaternions: np.ndarray) -> np.ndarray:
    q = quaternions / np.linalg.norm(quaternions, axis=-1, keepdims=True)
    w, x, y, z = q.T
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.rad2deg(np.column_stack((roll, pitch, yaw)))


class OnlineTargetPanel:
    """Own Kit widgets/goal markers, but only ever publish immutable intents."""

    def __init__(self, *, now=time.monotonic, subscribe_updates=None, ui_module=None,
                 marker_factory=None, env_origin=None, runtime_quaternion_order="wxyz",
                 seed_positions=None, seed_quaternions_wxyz=None, mode_name="VR-3") -> None:
        self._now, self._goal_pending, self._mode_pending, self._lifecycle_pending = now, None, None, None
        self._close_sticky = self._closed = False
        self._sequence = 0
        self._heartbeat_stamp = None
        self.mode_name, self.mode_epoch = validate_online_mode(mode_name), 0
        self._editor_seed_positions = self._editor_seed_degrees = None
        self.status, self.window, self._marker = {}, None, None
        self.position_models, self.angle_models, self._labels = [], [], {}
        self.position_fields, self.angle_fields = [], []
        self.waypoint_position_models, self.waypoint_position_fields = [], []
        self.waypoint_heading_model, self.waypoint_heading_field, self._waypoint_go_button = None, None, None
        self._env_origin = np.zeros(3) if env_origin is None else np.asarray(env_origin, dtype=np.float64).reshape(3).copy()
        self._runtime_quaternion_order = runtime_quaternion_order
        self._ui = ui_module if ui_module is not None else self._load_ui()
        self._subscription = None
        try:
            self._build_window()
            self._set_editor_rows_enabled()
            self._set_waypoint_enabled()
            self._update_mode_label()
            self._marker = marker_factory() if marker_factory is not None else (self._build_markers() if self._ui is not None else None)
            if seed_positions is not None and seed_quaternions_wxyz is not None:
                self.reset_editors(seed_positions, seed_quaternions_wxyz)
                self.update_status({"state": "READY", "goal_positions": np.asarray(seed_positions, dtype=np.float64).reshape(3, 3),
                                    "goal_quaternions_wxyz": np.asarray(seed_quaternions_wxyz, dtype=np.float64).reshape(3, 4)})
            if subscribe_updates is None:
                subscribe_updates = self._kit_update_subscription
            self._subscription = subscribe_updates(self._on_update) if subscribe_updates else None
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _load_ui():
        try:
            import omni.ui
        except ImportError:
            return None
        return omni.ui

    @staticmethod
    def _kit_update_subscription(callback):
        try:
            import omni.kit.app
        except ImportError:
            return None
        return omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
            lambda _event: callback(), name="bfm_online_targets_heartbeat")

    def _build_window(self) -> None:
        if self._ui is None:
            return
        self.window = self._ui.Window("BFM Online Targets", width=600, height=940, position_x=20, position_y=60)
        self.window.set_visibility_changed_fn(self._on_visibility_changed)
        with self.window.frame:
            with self._ui.VStack(spacing=6):
                for point in _POINT_LABELS:
                    self._ui.Label(point, height=22)
                    position, angles, position_fields, angle_fields = [], [], [], []
                    with self._ui.HStack(height=24, spacing=4):
                        self._ui.Label("Position (m)", width=92)
                        for axis in _AXES:
                            self._ui.Label(axis, width=16)
                            field = self._ui.FloatField(width=self._ui.Fraction(1), height=24)
                            position_fields.append(field)
                            position.append(field.model)
                    with self._ui.HStack(height=24, spacing=4):
                        self._ui.Label("Rotation (deg)", width=92)
                        for axis in _AXES:
                            self._ui.Label(axis, width=16)
                            field = self._ui.FloatField(width=self._ui.Fraction(1), height=24)
                            angle_fields.append(field)
                            angles.append(field.model)
                    self.position_models.append(position)
                    self.angle_models.append(angles)
                    self.position_fields.append(position_fields)
                    self.angle_fields.append(angle_fields)
                with self._ui.HStack(height=30, spacing=4):
                    for mode_name in ONLINE_MODE_NAMES:
                        self._ui.Button(mode_name, clicked_fn=lambda name=mode_name: self.request_mode(name),
                                        width=self._ui.Fraction(1), height=30)
                self._ui.Label("Mode switch pauses; Resume, then submit a fresh Apply.", height=22)
                self._ui.Label("Waypoint (Pelvis-1 only): XYZ env-local m (Z=pelvis height), yaw deg", height=22)
                with self._ui.HStack(height=30, spacing=4):
                    for label in _AXES:
                        self._ui.Label(label, width=16)
                        field = self._ui.FloatField(width=self._ui.Fraction(1), height=24)
                        self.waypoint_position_fields.append(field)
                        self.waypoint_position_models.append(field.model)
                    self._ui.Label("Yaw", width=28)
                    self.waypoint_heading_field = self._ui.FloatField(width=self._ui.Fraction(1), height=24)
                    self.waypoint_heading_model = self.waypoint_heading_field.model
                    self._waypoint_go_button = self._ui.Button(
                        "Go", clicked_fn=self._apply_waypoint_widgets, width=54, height=30
                    )
                    self._ui.Button("Stop", clicked_fn=self.request_pause, width=54, height=30)
                self._ui.Label("Enable / Resume + Apply first; Go starts a short waypoint.", height=22)
                with self._ui.HStack(height=30, spacing=4):
                    for label, callback in (("Apply", self._apply_widgets), ("Enable", self.request_enable),
                                            ("Pause", self.request_pause), ("Resume", self.request_resume),
                                            ("Exit", self.request_close)):
                        self._ui.Button(label, clicked_fn=callback, width=self._ui.Fraction(1), height=30)
                for key in _SUMMARY_KEYS:
                    self._labels[key] = self._ui.Label("", height=22)
                with self._ui.ScrollingFrame(height=self._ui.Fraction(1)):
                    with self._ui.VStack(spacing=4, height=0):
                        for key in _DETAIL_KEYS:
                            self._labels[key] = self._ui.Label("", height=0, word_wrap=True)

    def _on_visibility_changed(self, visible: bool) -> None:
        if not visible:
            self.request_close()

    @staticmethod
    def _build_markers():
        from isaaclab.markers import VisualizationMarkers
        from isaaclab.markers.config import FRAME_MARKER_CFG
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.prim_path = "/Visuals/BFMOnline/goal_frames"
        frame_cfg = marker_cfg.markers["frame"].copy()
        frame_cfg.scale = (0.08, 0.08, 0.08)
        marker_cfg.markers = {"frame": frame_cfg}
        return VisualizationMarkers(marker_cfg)

    def _on_update(self) -> None:
        if self._closed:
            return
        self._heartbeat_stamp = float(self._now())
        if self.window is not None and not self.window.visible:
            self.request_close()

    @property
    def heartbeat_stamp(self):
        return self._heartbeat_stamp

    @property
    def close_requested(self) -> bool:
        return self._closed or self._close_sticky

    def _request(self, kind: str) -> None:
        if self._closed or self._close_sticky:
            return
        if kind == "close":
            self._close_sticky = True
            self._goal_pending = self._mode_pending = None
        elif self._lifecycle_pending is not None and self._lifecycle_pending["kind"] == "pause":
            return
        self._lifecycle_pending = {"kind": kind}

    def request_enable(self) -> None:
        self._request("enable")

    def request_pause(self) -> None:
        self._request("pause")

    def request_resume(self) -> None:
        self._request("resume")

    def request_close(self) -> None:
        self._request("close")

    def request_mode(self, mode_name: str) -> None:
        mode_name = validate_online_mode(mode_name)
        if self._closed or self._close_sticky:
            return
        self._mode_pending = {"kind": "mode", "mode_name": mode_name}
        self.request_pause()

    def set_mode(self, mode_name: str, mode_epoch: int) -> None:
        """Apply a controller-confirmed mode without touching simulation state."""
        mode_name = validate_online_mode(mode_name)
        if isinstance(mode_epoch, bool) or not isinstance(mode_epoch, (int, np.integer)) or int(mode_epoch) < 0:
            raise ValueError("mode_epoch must be a nonnegative integer")
        self.mode_name, self.mode_epoch = mode_name, int(mode_epoch)
        self._mode_pending = None
        self.discard_goal()
        self._set_editor_rows_enabled()
        self._set_waypoint_enabled()
        self._render_markers()
        self._update_mode_label()

    def _update_mode_label(self) -> None:
        if "mode" in self._labels:
            self._labels["mode"].text = f"mode: {self.mode_name} (epoch {self.mode_epoch})"

    def _set_editor_rows_enabled(self) -> None:
        active_rows = set(online_target_rows(self.mode_name))
        for row, fields in enumerate(self.position_fields):
            for field in (*fields, *self.angle_fields[row]):
                field.enabled = row in active_rows

    def _set_waypoint_enabled(self) -> None:
        enabled = self.mode_name == "Pelvis-1"
        for field in self.waypoint_position_fields:
            field.enabled = enabled
        if self.waypoint_heading_field is not None:
            self.waypoint_heading_field.enabled = enabled
        if self._waypoint_go_button is not None:
            self._waypoint_go_button.enabled = enabled

    def _apply_widgets(self) -> None:
        self.apply(np.array([[field.get_value_as_float() for field in fields] for fields in self.position_models]),
                   np.array([[field.get_value_as_float() for field in fields] for fields in self.angle_models]))

    def _apply_waypoint_widgets(self) -> None:
        self.request_waypoint(
            np.array([model.get_value_as_float() for model in self.waypoint_position_models]),
            self.waypoint_heading_model.get_value_as_float(),
        )

    def next_sequence(self) -> int:
        """Reserve one sequence shared by every live target producer."""
        self._sequence += 1
        return self._sequence

    def apply(self, positions, xyz_degrees) -> None:
        """Coalesce only complete goals: a lifecycle intent remains first."""
        if self._closed or self._close_sticky:
            return
        try:
            positions = np.asarray(positions, dtype=np.float64).copy()
            degrees = np.asarray(xyz_degrees, dtype=np.float64).copy()
        except (TypeError, ValueError):
            positions = degrees = np.empty(0)
        if positions.shape == (3, 3) and degrees.shape == (3, 3) and self._editor_seed_positions is not None:
            inactive_rows = set(range(3)) - set(online_target_rows(self.mode_name))
            for row in inactive_rows:
                positions[row] = self._editor_seed_positions[row]
                degrees[row] = self._editor_seed_degrees[row]
        if positions.shape != (3, 3) or degrees.shape != (3, 3) or not np.all(np.isfinite(positions)) or not np.all(np.isfinite(degrees)):
            self.status = {**self.status, "last_error": "three finite XYZ positions and degree rotations are required"}
            return
        sequence = self.next_sequence()
        self._goal_pending = {"kind": "goal", "positions": positions.copy(), "quaternions": _wxyz_from_xyz_degrees(degrees).copy(),
                              "body_names": tuple(VR3_BODY_NAMES), "frame": "env_local", "quaternion_order": "wxyz",
                              "mode_name": self.mode_name, "mode_epoch": self.mode_epoch,
                              "sequence": sequence, "stamp": float(self._now())}

    def request_waypoint(self, target_xyz, heading_degrees) -> None:
        """Publish one immutable, mode-tagged waypoint without changing lifecycle state."""
        if self._closed or self._close_sticky:
            return
        try:
            xyz = np.asarray(target_xyz, dtype=np.float64).copy()
            heading_value = np.asarray(heading_degrees, dtype=np.float64)
            if isinstance(heading_degrees, (bool, np.bool_)) or heading_value.shape != ():
                raise ValueError("heading must be one number")
            heading = float(heading_value)
        except (TypeError, ValueError, OverflowError):
            xyz, heading = np.empty(0), float("nan")
        if xyz.shape != (3,) or not np.all(np.isfinite(xyz)) or not np.isfinite(heading):
            self.status = {**self.status, "last_error": "waypoint requires finite env-local XYZ and heading"}
            return
        self._goal_pending = {"kind": "waypoint", "target_xyz": xyz.copy(), "heading_degrees": heading,
                              "mode_name": self.mode_name, "mode_epoch": self.mode_epoch,
                              "sequence": self.next_sequence(), "stamp": float(self._now())}

    def discard_goal(self) -> None:
        self._goal_pending = None

    def reset_editors(self, positions, quaternions_wxyz) -> None:
        positions = np.asarray(positions, dtype=np.float64).reshape(3, 3).copy()
        quaternions = np.asarray(quaternions_wxyz, dtype=np.float64).reshape(3, 4).copy()
        self.status = {**self.status, "editor_positions": positions, "editor_quaternions_wxyz": quaternions}
        angles = _xyz_degrees_from_wxyz(quaternions)
        self._editor_seed_positions, self._editor_seed_degrees = positions.copy(), angles.copy()
        for point, models in enumerate(self.position_models):
            for axis, model in enumerate(models):
                model.set_value(float(positions[point, axis]))
                self.angle_models[point][axis].set_value(float(angles[point, axis]))
        if len(self.waypoint_position_models) == 3:
            for axis, model in enumerate(self.waypoint_position_models):
                model.set_value(float(positions[0, axis]))
            self.waypoint_heading_model.set_value(float(angles[0, 2]))

    def consume_request(self):
        if self._lifecycle_pending is not None:
            request, self._lifecycle_pending = self._lifecycle_pending, None
            return dict(request)
        if self._mode_pending is not None:
            request, self._mode_pending = self._mode_pending, None
            return dict(request)
        if self._goal_pending is None:
            return None
        request, self._goal_pending = self._goal_pending, None
        return {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in request.items()}

    def consume_safety_request(self):
        """Consume only a pending Pause or Close, leaving all other intents queued."""
        if self._lifecycle_pending is None or self._lifecycle_pending["kind"] not in ("pause", "close"):
            return None
        request, self._lifecycle_pending = self._lifecycle_pending, None
        return dict(request)

    def update_status(self, status) -> None:
        self.status = dict(status)
        self._render_markers()
        for key, label in self._labels.items():
            value = f"{self.mode_name} (epoch {self.mode_epoch})" if key == "mode" else self.status.get(key)
            label.text = f"{key}: {self._format_status_value(value)}"

    def _render_markers(self) -> None:
        goals, quaternions = self.status.get("goal_positions"), self.status.get("goal_quaternions_wxyz")
        if self._marker is not None and goals is not None and quaternions is not None:
            rows = list(online_target_rows(self.mode_name))
            positions = np.asarray(goals, dtype=np.float64).reshape(3, 3)[rows] + self._env_origin
            runtime_quaternions = packed_wxyz_to_runtime(
                np.asarray(quaternions, dtype=np.float64).reshape(3, 4)[rows], self._runtime_quaternion_order
            )
            self._marker.visualize(positions, runtime_quaternions)

    @staticmethod
    def _format_status_value(value) -> str:
        if value is None:
            return "—"
        if isinstance(value, (bool, str, int, np.integer)):
            return str(value)
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.4f}"
        try:
            array = np.asarray(value)
        except (TypeError, ValueError):
            return str(value)
        if np.issubdtype(array.dtype, np.number):
            return np.array2string(array, precision=4, suppress_small=True, floatmode="fixed", max_line_width=88)
        return str(value)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = self._close_sticky = True
        self._goal_pending = self._mode_pending = self._lifecycle_pending = None
        subscription, self._subscription = self._subscription, None
        if subscription is not None:
            subscription.unsubscribe()
        marker, self._marker = self._marker, None
        if marker is not None:
            marker.set_visibility(False)
        if self.window is not None:
            self.window.set_visibility_changed_fn(None)
            self.window.visible = False
            destroy = getattr(self.window, "destroy", None)
            if destroy is not None:
                destroy()
