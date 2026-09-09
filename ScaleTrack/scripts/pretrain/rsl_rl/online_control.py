"""Pre-launch validation and simulator-bound online reference supervision."""

from __future__ import annotations

import time

import numpy as np


def add_online_args(parser) -> None:
    parser.add_argument("--online_targets", action="store_true", default=False,
                        help="Use local Kit targets in Pelvis-1, UMI-2 or VR-3 mode.")


def validate_online_args(args) -> None:
    """Reject unsupported combinations before ``AppLauncher`` creates an app."""
    if not getattr(args, "online_targets", False):
        return
    if not getattr(args, "motion_menu", False):
        raise ValueError("--online_targets requires --motion_menu")
    if getattr(args, "num_envs", None) != 1:
        raise ValueError("--online_targets requires --num_envs 1")
    if getattr(args, "task", None) != "G1-BFM-Transformer-Tracking":
        raise ValueError("--online_targets supports only --task G1-BFM-Transformer-Tracking")
    mode = getattr(args, "mode_index", None)
    if mode is not None and (isinstance(mode, bool) or not isinstance(mode, int) or mode not in (0, 1, 2)):
        raise ValueError("--online_targets supports --mode_index 0 (Pelvis-1), 1 (UMI-2), or 2 (VR-3)")
    if getattr(args, "local_tracking", False) or getattr(args, "video", False) or getattr(args, "headless", False):
        raise ValueError("--online_targets cannot combine with local_tracking, video, or headless")
    visualizer = getattr(args, "visualizer", None)
    if visualizer is None:
        args.visualizer = ["kit"]
    elif list(visualizer) != ["kit"]:
        raise ValueError("--online_targets requires the Kit visualizer (--viz kit)")
    args.mode_index = 2 if mode is None else mode


def configure_online_env(env_cfg) -> None:
    """Change only this playback config to live command/no hidden resets."""
    from scaletrack.tasks.tracking.mdp.live_commands import LiveMotionCommand

    env_cfg.commands.motion.class_type = LiveMotionCommand
    for name in ("time_out", "motion_time_out", "body_pos"):
        if not hasattr(env_cfg.terminations, name):
            raise ValueError(f"online config lacks termination {name}")
        setattr(env_cfg.terminations, name, None)
    for name, event in vars(env_cfg.events).items():
        if getattr(event, "mode", None) == "interval":
            setattr(env_cfg.events, name, None)
    _assert_online_config(env_cfg)


def _assert_online_config(env_cfg) -> None:
    enabled = [name for name in ("time_out", "motion_time_out", "body_pos")
               if getattr(env_cfg.terminations, name, None) is not None]
    if enabled:
        raise ValueError(f"online config retains automatic termination terms: {enabled}")
    intervals = [name for name, event in vars(env_cfg.events).items()
                 if getattr(event, "mode", None) == "interval"]
    if intervals:
        raise ValueError(f"online config retains state-writing interval events: {intervals}")


def assert_online_env(env) -> None:
    """Fail setup if the constructed environment reintroduced a reset path."""
    _assert_online_config(env.cfg)
    active_terms = getattr(getattr(env, "termination_manager", None), "active_terms", ())
    forbidden = set(active_terms)
    if forbidden:
        raise ValueError(f"online environment retains automatic termination terms: {sorted(forbidden)}")
    active_events = getattr(getattr(env, "event_manager", None), "active_terms", {})
    if active_events.get("interval", []):
        raise ValueError(f"online environment retains interval events: {active_events['interval']}")


def _as_numpy(value):
    value = getattr(value, "torch", value)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _finite_tree(value) -> bool:
    if hasattr(value, "items"):
        return all(_finite_tree(child) for _, child in value.items())
    try:
        return bool(np.all(np.isfinite(_as_numpy(value))))
    except (TypeError, ValueError):
        return False


def _yaw_wxyz(quaternion):
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,) or not np.all(np.isfinite(q)) or np.linalg.norm(q) < 1e-12:
        raise ValueError("pelvis quaternion must be finite and nonzero")
    w, x, y, z = q / np.linalg.norm(q)
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _set_yaw_wxyz(quaternion, yaw):
    """Rotate around world Z, retaining the reference pelvis roll/pitch."""
    q = np.asarray(quaternion, dtype=np.float64)
    delta = float(yaw) - _yaw_wxyz(q)
    w, x, y, z = q / np.linalg.norm(q)
    c, s = np.cos(delta / 2), np.sin(delta / 2)
    return np.array([c*w - s*z, c*x - s*y, c*y + s*x, c*z + s*w])


class OnlinePlaybackController:
    """The only online component allowed to bridge GUI intent and an env step."""

    def __init__(self, env, command, panel, *, session=None, now=time.monotonic) -> None:
        from scaletrack.utils.online_session import OnlineSession
        from scaletrack.utils import online_modes
        from scaletrack.utils.waypoint import WaypointFollower
        self.env, self.command, self.panel, self._now = env, command, panel, now
        self.provider = command.live_reference
        self.session = session if session is not None else OnlineSession(self.provider)
        self._closed = False
        self._last_error = None
        self._requires_goal = False
        self._mode_schema = online_modes
        self.mode_name = online_modes.validate_online_mode(getattr(command, "live_mode_name", "VR-3"))
        self.mode_epoch = 0
        self.mode_rejected_count = 0
        self.waypoint = WaypointFollower()
        self._waypoint_pending_reference = False

    def _cancel_waypoint(self, reason):
        self._waypoint_pending_reference = False
        if self.waypoint.state in ("RUNNING", "SETTLING"):
            self.waypoint.cancel(reason)

    def _waypoint_actual(self, actual):
        data = self.command.robot.data
        index = int(_as_numpy(self.command.body_indexes[0]).reshape(()).item())
        linear = getattr(data, "body_link_lin_vel_w", None)
        angular = getattr(data, "body_link_ang_vel_w", None)
        if linear is None:
            linear = data.body_lin_vel_w
        if angular is None:
            angular = data.body_ang_vel_w
        linear, angular = _as_numpy(linear)[0, index], _as_numpy(angular)[0, index]
        if linear.shape != (3,) or angular.shape != (3,) or not np.all(np.isfinite([linear, angular])):
            raise ValueError("waypoint actual pelvis velocities must be finite 3-vectors")
        return actual[0][0].copy(), _yaw_wxyz(actual[1][0]), float(np.linalg.norm(linear)), float(np.linalg.norm(angular))

    def _check_request_mode(self, request):
        from scaletrack.utils.live_reference import ReferenceInputError
        epoch = request.get("mode_epoch", 0)
        if (request.get("mode_name", "VR-3") != self.mode_name
                or isinstance(epoch, bool) or not isinstance(epoch, int) or epoch != self.mode_epoch):
            self.mode_rejected_count += 1
            raise ReferenceInputError("goal belongs to an old or different control mode; submit a fresh Apply")

    @staticmethod
    def _waypoint_targets(base_positions, base_quaternions, xyz, yaw):
        positions, quaternions = base_positions.copy(), base_quaternions.copy()
        positions += np.asarray(xyz) - base_positions[0]
        quaternions[0] = _set_yaw_wxyz(base_quaternions[0], yaw)
        return positions, quaternions

    def _begin_waypoint(self, request, actual, obs, now):
        from scaletrack.utils.live_reference import ReferenceInputError
        from scaletrack.utils.waypoint import WaypointFollower
        try:
            self._check_request_mode(request)
            if self.mode_name != "Pelvis-1":
                raise ValueError("waypoint Go requires Pelvis-1 mode")
            if self.session.state != self.session.ACTIVE or self._requires_goal:
                raise ValueError("waypoint Go requires ACTIVE; Enable or Resume + Apply first")
            if not self.session.check(now, *actual):
                return
            try:
                actual_xyz, actual_yaw, _, _ = self._waypoint_actual(actual)
            except ValueError as error:
                self.session.fault(str(error))
                return
            heading = request.get("heading_degrees")
            if isinstance(heading, (bool, np.bool_)) or not isinstance(heading, (int, float, np.number)) or not np.isfinite(heading):
                raise ValueError("waypoint heading must be a finite angle in degrees")
            reference = self.provider.sample([0])
            positions = reference["body_pos"][0, [0, 10, 13]].copy()
            quaternions = reference["body_quat"][0, [0, 10, 13]].copy()
            candidate = WaypointFollower(self.waypoint.cfg)
            candidate.start(request.get("target_xyz"), np.deg2rad(float(heading)),
                            actual_xyz=actual_xyz, actual_yaw=actual_yaw,
                            reference_xyz=positions[0], reference_yaw=_yaw_wxyz(quaternions[0]))
            final_targets = self._waypoint_targets(positions, quaternions, candidate.goal_xyz, candidate.goal_yaw)
            self.provider.validate_targets(*final_targets)  # Preflight before replacing any running task.
            self.provider.submit(positions, quaternions, sequence=request.get("sequence"),
                                 stamp=request.get("stamp"), now=now)
            self.provider.commit_pending(now)
        except (ReferenceInputError, ValueError) as error:
            self._last_error = str(error)
            if getattr(error, "clock_fault", False):
                self.session.fault(str(error))
            return
        self.waypoint = candidate
        self._waypoint_base_positions, self._waypoint_base_quaternions = positions, quaternions
        self._waypoint_pending_reference = False
        self._last_error = None
        self._refresh_task_groups(obs)

    def _flush_waypoint_reference(self, obs, now):
        if not self._waypoint_pending_reference:
            return
        actual_positions = self._actual()[0]
        xyz, yaw = self.waypoint.preview(actual_positions[0], self.env.step_dt, steps=32)
        current = self.provider.sample([0])
        indices = [0, 10, 13]
        positions = [current["body_pos"][0, indices]]
        quaternions = [current["body_quat"][0, indices]]
        for future_xyz, future_yaw in zip(xyz, yaw):
            next_positions, next_quaternions = self._waypoint_targets(
                self._waypoint_base_positions, self._waypoint_base_quaternions, future_xyz, future_yaw
            )
            positions.append(next_positions)
            quaternions.append(next_quaternions)
        # Allocate only after consuming UI intents. A goal arriving during env.step
        # must not become stale because an internal route update overtook it.
        # Preserve the multi-frame plan: a single next-step goal would collapse
        # the actor's remaining future window to a stationary pose.
        self.provider.submit_trajectory(
            positions, quaternions, sequence=self.panel.next_sequence(), stamp=now, now=now
        )
        self.provider.commit_pending(now)
        self._waypoint_pending_reference = False
        self._refresh_task_groups(obs)

    def _advance_waypoint(self, actual, now):
        if self.waypoint.state not in ("RUNNING", "SETTLING"):
            return
        try:
            xyz, yaw, linear_speed, angular_speed = self._waypoint_actual(actual)
            self.waypoint.step(xyz, yaw, linear_speed, angular_speed, self.env.step_dt)
        except ValueError as error:
            self.session.fault(f"waypoint telemetry: {error}")
            self._cancel_waypoint(self.session.reason)
            self._pause_sim()
            return
        if self.waypoint.state in ("TIMED_OUT", "STALLED"):
            self._waypoint_pending_reference = False
            self.session.pause(f"waypoint {self.waypoint.state}: {self.waypoint.reason}")
            self._requires_goal = False
            self._pause_sim()
        else:
            # ARRIVED holds its reference with policy/physics still running.
            # It is never declared successful merely because physics was paused.
            self._waypoint_pending_reference = True

    @property
    def close_requested(self) -> bool:
        return self._closed or self.panel.close_requested or self.session.state == self.session.CLOSED

    def _actual(self):
        data = self.command.robot.data
        raw_indexes = getattr(self.command.body_indexes, "torch", self.command.body_indexes)
        if hasattr(raw_indexes, "detach"):
            raw_indexes = raw_indexes.detach().cpu().tolist()
        indexes = [int(_as_numpy(index).reshape(()).item()) for index in raw_indexes]
        positions = _as_numpy(data.body_pos_w)[0, indexes].astype(np.float64, copy=True)
        quaternions = _as_numpy(data.body_quat_w)[0, indexes].astype(np.float64, copy=True)
        joints = _as_numpy(data.joint_pos)[0].astype(np.float64, copy=True)
        velocities = _as_numpy(data.joint_vel)[0].astype(np.float64, copy=True)
        origins = _as_numpy(self.env.scene.env_origins)[0].astype(np.float64, copy=False)
        positions -= origins
        runtime_order = getattr(self.command, "runtime_quaternion_order", "wxyz")
        if runtime_order == "xyzw":
            quaternions = quaternions[:, [3, 0, 1, 2]]
        return positions, quaternions, joints, velocities

    def _pause_sim(self) -> None:
        self._timeline_action("pause")

    def _play_sim(self) -> None:
        self._timeline_action("play")

    def _pause_native(self):
        try:
            import omni.timeline
        except ImportError:
            return
        # No app.update here: do not admit another Play callback recursively.
        omni.timeline.get_timeline_interface().pause()

    def _timeline_action(self, kind):
        """Pump GUI lifecycle without allowing Kit-owned implicit physics."""
        action = getattr(self.env.sim, kind, None)
        if action is None:
            return
        get_setting = getattr(self.env.sim, "get_setting", None)
        set_setting = getattr(self.env.sim, "set_setting", None)
        guarded = callable(get_setting) and callable(set_setting)
        key = "/app/player/playSimulations"
        previous = get_setting(key) if guarded else None
        if guarded:
            set_setting(key, False)
        try:
            action()
        finally:
            try:
                if kind == "pause":
                    self._pause_native()
            finally:
                if guarded and previous is not None:
                    set_setting(key, previous)

    def _native_playing(self):
        try:
            import omni.timeline
            return bool(omni.timeline.get_timeline_interface().is_playing())
        except (ImportError, AttributeError):
            return None

    @staticmethod
    def _orientation_errors(actual, target):
        dots = np.abs(np.sum(actual * target, axis=-1))
        return 2.0 * np.arccos(np.clip(dots, -1.0, 1.0))

    def _status(self, actual=None, *, now=None) -> None:
        if actual is None:
            actual = self._actual()
        positions, quaternions, _, _ = actual
        goals = self.provider.goal_positions
        goal_quaternions = self.provider.goal_quaternions
        reference = self.provider.sample([0])
        active_positions = positions[[0, 10, 13]].copy()
        active_quaternions = quaternions[[0, 10, 13]].copy()
        rows = list(self._mode_schema.online_target_rows(self.mode_name))
        reference_positions = reference["body_pos"][0, [0, 10, 13]]
        if now is None:
            now = float(self._now())
        waypoint = self.waypoint.status()
        self.panel.update_status({"state": self.session.state, "reason": self.session.reason,
                                  "waypoint_state": self.waypoint.state, "waypoint_reason": self.waypoint.reason,
                                  "waypoint_target_xyz": waypoint.get("goal_xyz"),
                                  **{f"waypoint_{key}": waypoint.get(key) for key in
                                     ("distance_m", "horizontal_error_m", "vertical_error_m", "yaw_error_rad",
                                      "linear_speed_m_s", "angular_speed_rad_s", "elapsed_s", "settled_s")},
                                  "mode_name": self.mode_name, "mode_epoch": self.mode_epoch,
                                  "active_body_names": self._mode_schema.ONLINE_MODES[self.mode_name],
                                  "active_target_rows": self._mode_schema.online_target_rows(self.mode_name),
                                  "mode_rejected_count": self.mode_rejected_count,
                                  "accepted_sequence": self.provider.last_sequence,
                                  "rejected_count": self.provider.rejected_count,
                                  "input_age": None if self.panel.heartbeat_stamp is None else max(0.0, now - self.panel.heartbeat_stamp),
                                  "last_error": getattr(self, "_last_error", None) or self.provider.last_error,
                                  "awaiting_goal": self._requires_goal,
                                  "goal_positions": goals, "goal_quaternions_wxyz": goal_quaternions,
                                  "reference_positions": reference["body_pos"][:, [0, 10, 13]][0],
                                  "reference_quaternions_wxyz": reference["body_quat"][:, [0, 10, 13]][0],
                                  "actual_positions": active_positions, "actual_quaternions_wxyz": active_quaternions,
                                  "position_errors": np.linalg.norm(active_positions - goals, axis=-1),
                                  "active_position_errors": np.linalg.norm(active_positions[rows] - goals[rows], axis=-1),
                                  "active_reference_position_errors": np.linalg.norm(active_positions[rows] - reference_positions[rows], axis=-1),
                                  "orientation_errors": self._orientation_errors(active_quaternions, goal_quaternions)})

    def _change_mode(self, name, actual, obs, now):
        """Switch only references/masks while stopped; recovery stays explicit."""
        try:
            self._mode_schema.validate_online_mode(name)
        except ValueError as error:
            self._last_error = str(error)
            self.mode_rejected_count += 1
            return
        if self.session.state == self.session.PAUSED_FAULT:
            self._last_error = "mode change cannot clear a fault; recover the current mode first"
            self.mode_rejected_count += 1
            return
        if name == self.mode_name:
            return
        self._cancel_waypoint("control mode changed")
        if self.session.state == self.session.ACTIVE:
            if not self.session.check(now, *actual):
                return
            self.session.pause("mode changed: Resume, then submit a fresh Apply")
        self._requires_goal = False
        self.panel.discard_goal()
        # Only the reference is reseeded. Robot/episode/history state stays intact.
        self.provider.reset_reference(*actual[:3])
        self.command.set_live_mode(name)
        self.mode_name = name
        self.mode_epoch += 1
        self.panel.set_mode(name, self.mode_epoch)
        self.panel.reset_editors(actual[0][[0, 10, 13]], actual[1][[0, 10, 13]])
        self._refresh_task_groups(obs)

    def _refresh_task_groups(self, obs):
        manager = self.env.observation_manager
        for name in ("policy_task", "critic_task", "mode", "mode_mapping"):
            value = manager.compute_group(name, update_history=False)
            obs[name] = value

    def _heartbeat(self):
        """Read the producer again after any synchronous Kit event pump."""
        now = float(self._now())
        if self.panel.heartbeat_stamp is not None:
            try:
                self.session.heartbeat(self.panel.heartbeat_stamp, now)
            except RuntimeError as error:
                if self.session.state != self.session.PAUSED_FAULT:
                    raise
                self._last_error = str(error)
        return now

    def _safety_intent(self):
        """Recheck priority intent without consuming a future Apply/Resume."""
        if self.close_requested:
            self.close()
            return
        consume = getattr(self.panel, "consume_safety_request", None)
        request = consume() if consume is not None else None
        if request is not None:
            if request["kind"] == "close":
                self.close()
            elif request["kind"] == "pause":
                self._cancel_waypoint("user pause")
                self._requires_goal = False
                self.session.pause()

    def _reseed(self, kind, now, actual, obs):
        expected = (self.session.READY,) if kind == "enable" else (self.session.PAUSED_USER, self.session.PAUSED_FAULT)
        if self.session.state not in expected:
            self._last_error = f"{kind} is unavailable in {self.session.state}"
            return False
        self._cancel_waypoint(kind)
        try:
            if kind == "enable":
                self.session.activate(now, *actual[:3])
            else:
                self.session.resume(now, *actual[:3])
        except RuntimeError as error:
            self._last_error = str(error)
            return False
        self.panel.discard_goal()
        self._requires_goal = kind == "resume"
        self.panel.reset_editors(actual[0][[0, 10, 13]], actual[1][[0, 10, 13]])
        self._refresh_task_groups(obs)
        return True

    def before_step(self, obs):
        """Consume one intent; return ``None`` whenever no physical step may run."""
        if self.close_requested:
            self.close()
            return None
        wants_play = False
        try:
            actual = self._actual()
            now = self._heartbeat()
            request = self.panel.consume_request()
            if request is not None:
                kind = request["kind"]
                if kind == "close":
                    self.close(); return None
                if kind == "pause":
                    self._cancel_waypoint("user pause")
                    self._requires_goal = False
                    self.session.pause()
                elif kind in ("enable", "resume"):
                    reseeded = self._reseed(kind, now, actual, obs)
                    wants_play = reseeded and kind == "enable"
                elif kind == "mode":
                    self._change_mode(request.get("mode_name"), actual, obs, now)
                elif kind == "waypoint":
                    self._begin_waypoint(request, actual, obs, now)
                elif kind == "goal":
                    from scaletrack.utils.live_reference import ReferenceInputError
                    try:
                        # Old packets cannot cross mode switches, even VR-3 -> UMI-2 -> VR-3.
                        self._check_request_mode(request)
                        self.provider.submit(request.get("positions"), request.get("quaternions"), sequence=request.get("sequence"), stamp=request.get("stamp"), now=now, body_names=request.get("body_names"), frame=request.get("frame"), quaternion_order=request.get("quaternion_order"))
                        committed = self.provider.commit_pending(now)
                    except ReferenceInputError as error:
                        self._last_error = str(error)
                        if error.clock_fault:
                            self.session.fault(str(error))
                    else:
                        if committed:
                            self._cancel_waypoint("manual Apply")
                            self._refresh_task_groups(obs)
                            wants_play = self._requires_goal and self.session.state == self.session.ACTIVE
                            if wants_play:
                                self._requires_goal = False

            # Every path, including a rejected Apply, passes the same gates.
            self._safety_intent()
            if self._closed:
                return None
            now = self._heartbeat()
            if self._native_playing() is False and self.session.state == self.session.ACTIVE and not (self._requires_goal or wants_play):
                self.session.pause("native timeline paused")
            allowed = self.session.check(now, *actual)
            if allowed and not self._requires_goal:
                self._flush_waypoint_reference(obs, now)
            if allowed and not _finite_tree(obs):
                self.session.fault("policy observations are nonfinite")
                allowed = False
            if self.session.state != self.session.ACTIVE:
                self._cancel_waypoint(self.session.reason)
                self._requires_goal = False
            allowed = allowed and not self._requires_goal
            if allowed and wants_play:
                self._play_sim()  # Pumps Kit: a Pause/Close can arrive here.
                self._safety_intent()
                if self._closed:
                    return None
                now = self._heartbeat()
                actual = self._actual()
                if self._native_playing() is False:
                    self.session.pause("native timeline paused")
                allowed = self.session.check(now, *actual) and not self._requires_goal
            if not allowed:
                self._pause_sim()  # Keep rendering/Exit responsive while stopped.
                self._safety_intent()
                if self._closed:
                    return None
                now = self._heartbeat()
                if self.session.state != self.session.ACTIVE:
                    self._cancel_waypoint(self.session.reason)
                    self._requires_goal = False
            self._status(actual, now=now)
            return obs if allowed else None
        except Exception:
            self.close()
            raise

    def validate_actions(self, actions) -> bool:
        self._safety_intent()
        if self._closed:
            return False
        array = _as_numpy(actions)
        if not np.all(np.isfinite(array)):
            self.session.fault("policy actions are nonfinite")
            self._cancel_waypoint(self.session.reason)
            self._pause_sim(); self._status()
            return False
        now = self._heartbeat()
        actual = self._actual()
        if self._native_playing() is False and self.session.state == self.session.ACTIVE and not self._requires_goal:
            self.session.pause("native timeline paused")
        if not self.session.check(now, *actual) or self._requires_goal:
            self._cancel_waypoint(self.session.reason)
            self._pause_sim()
            self._status(actual)
            return False
        return True

    def after_step(self, obs) -> None:
        self._safety_intent()
        if self._closed:
            return
        actual = self._actual()
        now = self._heartbeat()
        if not self.session.check(now, *actual):
            self._cancel_waypoint(self.session.reason)
            self._requires_goal = False
            self._pause_sim()
            self._safety_intent()
            if self._closed:
                return
            now = float(self._now())
        else:
            self._advance_waypoint(actual, now)
            self._safety_intent()
            if self._closed:
                return
            now = self._heartbeat()
        self._status(actual, now=now)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._cancel_waypoint("closed")
        self._pause_sim()
        try:
            self.session.close()
        finally:
            self.panel.close()
