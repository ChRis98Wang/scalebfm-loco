"""Bounded real-policy/real-physics acceptance for one combined XYZ waypoint.

Run this through the normal ``play.py`` arguments with ``--online_targets``,
``--motion_menu``, ``--mode_index 0``, ``--num_envs 1`` and ``--viz kit``.
The parent process owns the GPU/App lifetime and enforces a 180 second cgroup
deadline.  This driver does not replace policy, environment, or physics and
never writes robot state.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import sys
import time
import traceback

import numpy as np


REPORT_VALUE = os.environ.get("BFM_WAYPOINT_SMOKE_REPORT", "")
if not REPORT_VALUE:
    raise ValueError("BFM_WAYPOINT_SMOKE_REPORT must name a JSON output file")
REPORT_PATH = Path(REPORT_VALUE).expanduser().resolve()
DIAGNOSTICS = os.environ.get("BFM_WAYPOINT_DIAGNOSTICS", "0") == "1"
SCENARIO_NAME = os.environ.get("BFM_WAYPOINT_SCENARIO", "combined")
START = time.monotonic()
DEADLINE = START + 165.0
report = {
    "result": "RUNNING", "argv": list(sys.argv), "started_monotonic_s": START,
    "deadline_monotonic_s": DEADLINE, "stage": "loading_player", "stages": [],
    "physics": {"elapsed_s": 0.0, "callbacks": 0}, "real_env_steps": 0,
    "real_policy_evaluations": 0, "postattach_calls": {}, "instrumentation_errors": [],
    "observations": None, "states": {}, "trajectory": [],
    "claim_scope": "one finite combined XYZ pelvis waypoint; not a generalization claim",
    "diagnostics_enabled": DIAGNOSTICS, "policy_probes": [],
}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array(value):
    value = getattr(value, "torch", value)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value).copy()


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "detach"):
        return _array(value).tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value if isinstance(value, (str, int, float, bool)) or value is None else repr(value)


def _write_report():
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = REPORT_PATH.with_name(REPORT_PATH.name + ".tmp")
    temporary.write_text(json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(REPORT_PATH)


def _metadata(value):
    if hasattr(value, "items"):
        return {str(key): _metadata(child) for key, child in value.items()}
    raw = getattr(value, "torch", value)
    return {"shape": list(getattr(raw, "shape", ())), "dtype": str(getattr(raw, "dtype", type(raw).__name__)),
            "device": str(getattr(raw, "device", "cpu"))}


def _yaw_wxyz(quaternion):
    w, x, y, z = np.asarray(quaternion, dtype=np.float64) / np.linalg.norm(quaternion)
    return float(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _angle_error(left, right):
    return (float(left) - float(right) + math.pi) % (2.0 * math.pi) - math.pi


def _stage(name, **details):
    report["stage"] = name
    report["stages"].append({"name": name, "monotonic_s": time.monotonic(), **details})
    _write_report()


scripts = Path(__file__).resolve().parents[1] / "scripts/pretrain/rsl_rl"
source = Path(__file__).resolve().parents[1] / "source/scaletrack/scaletrack"
sys.path.insert(0, str(scripts))
from waypoint_scenarios import waypoint_scenario
scenario = waypoint_scenario(SCENARIO_NAME)
report["scenario"] = scenario.name
report["claim_scope"] = scenario.claim_scope
report["source_sha256"] = {
    name: _sha256(path) for name, path in {
        "play.py": scripts / "play.py", "online_control.py": scripts / "online_control.py",
        "online_ui.py": scripts / "online_ui.py",
        "live_commands.py": source / "tasks/tracking/mdp/live_commands.py",
        "live_reference.py": source / "utils/live_reference.py",
        "online_modes.py": source / "utils/online_modes.py", "waypoint.py": source / "utils/waypoint.py",
        "waypoint_gui_smoke.py": Path(__file__),
        "waypoint_scenarios.py": scripts / "waypoint_scenarios.py",
        "actor_critic_humanoid_transformer.py": Path(__file__).resolve().parents[1]
            / "source/my_rsl_rl/my_rsl_rl/modules/actor_critic_humanoid_transformer.py",
    }.items()
}
if DIAGNOSTICS:
    report["source_sha256"].update({name: _sha256(scripts / name) for name in
                                  ("waypoint_diagnostics.py", "waypoint_policy_probe.py")})

app = None
completed = False
controller_instance = None
runner_class = original_get_policy = original_controller = None

try:
    player = runpy.run_path(str(scripts / "play.py"))
    app = player["simulation_app"]
    args = player["args_cli"]
    if not args.online_targets or not args.motion_menu or args.mode_index != 0:
        raise AssertionError("smoke requires online targets, motion menu, and mode_index 0 (Pelvis-1)")
    if args.num_envs != 1 or list(args.visualizer or ()) != ["kit"]:
        raise AssertionError("smoke requires num_envs 1 and the Kit visualizer")

    import omni.appwindow
    import omni.kit.app
    import omni.kit.renderer.capture
    import omni.physx
    import online_control

    runner_class = player["OnPolicyRunner"]
    original_get_policy = runner_class.get_inference_policy
    policy_seen = {}

    def capture_real_policy(runner, *policy_args, **policy_kwargs):
        policy = original_get_policy(runner, *policy_args, **policy_kwargs)
        policy_seen["policy"] = policy
        if DIAGNOSTICS:
            from waypoint_policy_probe import probe_policy

            def instrumented(obs):
                controller = controller_instance
                if controller is not None and controller.should_probe():
                    before = controller._probe_boundary(obs)
                    action, details = probe_policy(policy, runner.alg.policy, obs)
                    controller.record_probe(obs, before, details)
                    return action  # Only original-reference action, never the counterfactual.
                return policy(obs)

            return instrumented
        return policy

    runner_class.get_inference_policy = capture_real_policy
    original_controller = online_control.OnlinePlaybackController

    class WaypointSmokeController(original_controller):
        def should_probe(self):
            if self._smoke_stage != "waypoint_active":
                return False
            # One probe per half-second bucket, plus a late stalled-horizon sample.
            bucket = int((float(self.waypoint.status()["elapsed_s"]) + 1e-6) / 0.5)
            return bucket in (0, 1, 2, 3, 4, 5, 6, 10) and bucket not in getattr(self, "_probed_buckets", set())

        def _probe_boundary(self, obs):
            return {"state": self._state(obs), "physics_s": self._physics_s, "callbacks": self._callbacks,
                    "frame_index": self.provider.frame_index, "version": self.provider.version,
                    "task_elapsed_s": self.waypoint.status()["elapsed_s"]}

        def record_probe(self, obs, before, details):
            from waypoint_policy_probe import expected_pelvis_features
            after = self._probe_boundary(obs)
            if any(before[key] != after[key] for key in before if key != "state"):
                raise AssertionError("policy counterfactual advanced physical/reference/task state")
            if self._state_delta(before["state"], after["state"]) != 0.0:
                raise AssertionError("policy counterfactual changed robot state/history")
            offsets = self.command._live_offsets([0, 1, 2, 3, 4, -1])
            reference = self.provider.sample(offsets)
            actual = self._actual()
            expected = expected_pelvis_features(reference["body_pos"][:, 0], reference["body_quat"][:, 0],
                                                actual[0][0], actual[1][0])
            errors = {key: float(np.max(np.abs(_array(details["moving_pelvis"][key])[0] - value)))
                      for key, value in expected.items()}
            if max(errors.values()) > 1e-5:
                raise AssertionError(f"actor pelvis features disagree with independent pose transform: {errors}")
            observed_offsets = _array(details["moving_actor_task"])[0, :, 252]
            if not np.array_equal(observed_offsets, np.asarray(offsets)):
                raise AssertionError("actor time offsets differ from reference sample offsets")
            details.update({"task_elapsed_s": before["task_elapsed_s"], "physics_s": self._physics_s,
                            "actual_xyz": actual[0][0], "actual_quaternion_wxyz": actual[1][0],
                            "reference_pelvis_xyz": reference["body_pos"][:, 0],
                            "reference_pelvis_quaternions_wxyz": reference["body_quat"][:, 0],
                            "resolved_future_offsets": offsets, "independent_transform_max_errors": errors,
                            "expected_pelvis": expected, "physics_reference_history_unchanged": True})
            self._probed_buckets = getattr(self, "_probed_buckets", set())
            self._probed_buckets.add(int((float(before["task_elapsed_s"]) + 1e-6) / 0.5))
            report["policy_probes"].append(details)
            _write_report()

        def __init__(self, env, command, panel, **kwargs):
            global controller_instance
            super().__init__(env, command, panel, **kwargs)
            controller_instance = self
            if "policy" not in policy_seen:
                raise AssertionError("real inference policy was not constructed")
            self._smoke_stage, self._stage_wall = "wait_heartbeat", time.monotonic()
            self._physics_s = self._callbacks = self._env_steps = self._policy_calls = 0
            self._last_trajectory_physics = 0.0
            self._independent_settle_s = 0.0
            self._restorers, self._capture_task, self._ready_capture_task = [], None, None
            self._closed_for_smoke = False
            self._instrument()
            self._physics_subscription = omni.physx.get_physx_interface().subscribe_physics_on_step_events(
                self._on_physics_step, pre_step=False, order=0
            )
            report["target_object"] = {"present": "target_object" in getattr(self.env.scene, "_rigid_objects", {})}
            report["waypoint_config"] = vars(self.waypoint.cfg)
            report["states"]["postattach_ready"] = self._state(None)
            _stage("wait_heartbeat", physics_s=self._physics_s, callbacks=self._callbacks)

        def _on_physics_step(self, dt):
            self._physics_s += float(dt)
            self._callbacks += 1
            report["physics"] = {"elapsed_s": self._physics_s, "callbacks": self._callbacks}

        def _wrap(self, owner, name, label):
            original = getattr(owner, name, None)
            if not callable(original):
                return
            report["postattach_calls"].setdefault(label, 0)

            def counted(*args, **kwargs):
                report["postattach_calls"][label] += 1
                return original(*args, **kwargs)

            try:
                setattr(owner, name, counted)
            except (AttributeError, TypeError) as error:
                report["instrumentation_errors"].append(f"{label}: {error}")
                return
            self._restorers.append((owner, name, original))

        def _instrument(self):
            for name in ("reset", "_reset_idx"):
                self._wrap(self.env, name, f"env.{name}")
            self._wrap(self.env.scene, "reset", "scene.reset")
            self._wrap(self.command, "_resample_command", "command._resample_command")
            for manager_name in ("event_manager", "command_manager"):
                manager = getattr(self.env, manager_name, None)
                if manager is not None:
                    self._wrap(manager, "reset", f"{manager_name}.reset")
            for kind, registry in (("articulation", getattr(self.env.scene, "_articulations", {})),
                                   ("rigid_object", getattr(self.env.scene, "_rigid_objects", {}))):
                for asset_name, asset in registry.items():
                    self._wrap(asset, "reset", f"{kind}.{asset_name}.reset")
                    for method in ("write_root_pose_to_sim", "write_root_velocity_to_sim",
                                   "write_root_state_to_sim", "write_joint_state_to_sim"):
                        owner = "robot" if asset is self.command.robot else f"{kind}.{asset_name}"
                        self._wrap(asset, method, f"{owner}.{method}")

        def _state(self, obs):
            positions, quaternions, joints, joint_velocities = self._actual()
            data = self.command.robot.data
            root = getattr(data, "root_state_w", None)
            if root is None:
                root = np.concatenate([_array(getattr(data, name))[0].reshape(-1) for name in
                                       ("root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_w")])
            else:
                root = _array(root)[0]
            history = None if obs is None else {
                name: _array(obs[name]) for name in ("policy", "action", "critic") if name in obs
            }
            return {"root_state": root, "body_positions": positions,
                    "body_quaternions_wxyz": quaternions, "joint_positions": joints,
                    "joint_velocities": joint_velocities, "observation_history": history}

        @staticmethod
        def _state_delta(left, right):
            keys = ("root_state", "body_positions", "body_quaternions_wxyz", "joint_positions", "joint_velocities")
            deltas = [float(np.max(np.abs(left[key] - right[key]))) for key in keys]
            left_history, right_history = left.get("observation_history"), right.get("observation_history")
            if left_history is not None and right_history is not None:
                if set(left_history) != set(right_history):
                    return float("inf")
                deltas.extend(float(np.max(np.abs(left_history[key] - right_history[key]))) for key in left_history)
            return max(deltas)

        def _telemetry(self):
            actual = self._actual()
            data = self.command.robot.data
            raw_index = getattr(self.command.body_indexes[0], "torch", self.command.body_indexes[0])
            index = int(_array(raw_index).reshape(()).item())
            linear_source = getattr(data, "body_link_lin_vel_w", None)
            angular_source = getattr(data, "body_link_ang_vel_w", None)
            if linear_source is None:
                linear_source = data.body_lin_vel_w
            if angular_source is None:
                angular_source = data.body_ang_vel_w
            linear = _array(linear_source)[0, index].astype(np.float64, copy=False)
            angular = _array(angular_source)[0, index].astype(np.float64, copy=False)
            xyz, quaternion = actual[0][0], actual[1][0]
            return {"xyz": xyz, "quaternion_wxyz": quaternion, "yaw_rad": _yaw_wxyz(quaternion),
                    "linear_velocity_xyz": linear, "angular_velocity_xyz": angular,
                    "linear_speed_3d_m_s": float(np.linalg.norm(linear)),
                    "angular_speed_rad_s": float(np.linalg.norm(angular))}

        def _errors(self, telemetry):
            delta = telemetry["xyz"] - self._target_xyz
            return {"position_3d_m": float(np.linalg.norm(delta)),
                    "horizontal_m": float(np.linalg.norm(delta[:2])), "height_m": abs(float(delta[2])),
                    "yaw_rad": abs(_angle_error(telemetry["yaw_rad"], self._target_yaw))}

        def _record_trajectory(self, phase, physics_dt):
            telemetry = self._telemetry()
            reference = self.provider.sample([0, 1, 2, 4, 32])
            entry = {"phase": phase, "physics_s": self._physics_s, "physics_dt_s": physics_dt,
                     "callbacks": self._callbacks,
                     "env_step": self._env_steps, "actual": telemetry,
                     "target_xyz": self._target_xyz, "target_yaw_rad": self._target_yaw,
                     "errors": self._errors(telemetry), "waypoint": self.waypoint.status(),
                     "provider_last_sequence": self.provider.last_sequence,
                     "provider_goal_positions": self.provider.goal_positions,
                     "provider_goal_quaternions_wxyz": self.provider.goal_quaternions,
                     "provider_reference_body_positions": _array(reference["body_pos"])[0],
                     "provider_reference_body_quaternions_wxyz": _array(reference["body_quat"])[0],
                     "provider_future_offsets": [0, 1, 2, 4, 32],
                     "provider_future_pelvis_xyz": _array(reference["body_pos"])[:, 0],
                     "provider_future_pelvis_linear_velocity_xyz": _array(reference["body_lin_vel"])[:, 0],
                     "panel_status": dict(self.panel.status)}
            report["trajectory"].append(entry)
            return telemetry, entry["errors"]

        def _mask_and_metadata(self, obs):
            mask = _array(self.command.mode)
            expected = np.zeros((1, 14), dtype=np.float32)
            expected[0, 0] = 1.0
            if mask.shape != expected.shape or not np.array_equal(mask, expected):
                raise AssertionError(f"Pelvis-1 command mask is not exact index [0]: {mask}")
            if self.mode_name != "Pelvis-1" or self.command.live_mode_name != "Pelvis-1" or self.panel.mode_name != "Pelvis-1":
                raise AssertionError("controller/command/panel did not start in Pelvis-1")
            if not np.array_equal(_array(obs["mode"]), expected):
                raise AssertionError("observation mode does not equal the Pelvis-1 command mask")
            report["command_mask"] = mask

        def _assert_metadata(self, obs, boundary):
            layout = _metadata(obs)
            expected = report["observations"]["after_first_online_refresh"]
            report.setdefault("observation_boundary_checks", []).append(
                {"boundary": boundary, "layout": layout, "unchanged": layout == expected}
            )
            if layout != expected:
                raise AssertionError(f"observation metadata changed at {boundary}")

        def _assert_clean(self):
            if report["instrumentation_errors"]:
                raise AssertionError(f"instrumentation was incomplete: {report['instrumentation_errors']}")
            required = {"env.reset", "env._reset_idx", "scene.reset", "command._resample_command",
                        "robot.write_root_pose_to_sim", "robot.write_root_velocity_to_sim",
                        "robot.write_root_state_to_sim", "robot.write_joint_state_to_sim"}
            missing = sorted(required - set(report["postattach_calls"]))
            if missing:
                raise AssertionError(f"required post-attach spies are missing: {missing}")
            calls = {name: count for name, count in report["postattach_calls"].items() if count}
            if calls:
                raise AssertionError(f"post-attach reset/resample/state write detected: {calls}")

        def _assert_deadline(self):
            if time.monotonic() >= DEADLINE:
                raise AssertionError(f"internal 165s deadline expired in {self._smoke_stage}")

        async def _capture_to(self, path, report_key):
            report[report_key] = {"path": str(path), "status": "requested", "exists": False}
            _write_report()
            capture = omni.kit.renderer.capture.acquire_renderer_capture_interface()
            window = omni.appwindow.get_default_app_window()
            capture.capture_next_frame_swapchain(str(path), window)
            await omni.kit.app.get_app().next_update_async()
            capture.wait_async_capture(window)
            for _ in range(20):
                if path.is_file():
                    break
                await omni.kit.app.get_app().next_update_async()
            report[report_key] = {"path": str(path), "status": "captured" if path.is_file() else "missing",
                                  "exists": path.is_file(),
                                  "size_bytes": path.stat().st_size if path.is_file() else None}
            _write_report()

        async def _capture(self):
            await self._capture_to(REPORT_PATH.with_suffix(".png"), "capture")

        async def _capture_ready(self):
            path = REPORT_PATH.with_name(f"{REPORT_PATH.stem}.ready.png")
            await self._capture_to(path, "ready_capture")

        def before_step(self, obs):
            global completed
            self._last_obs = obs
            self._assert_deadline()
            if report["observations"] is None:
                report["observations"] = {"before_online_refresh": _metadata(obs)}
            if self._smoke_stage == "wait_heartbeat" and self.panel.heartbeat_stamp is not None:
                if self._physics_s != 0.0 or self._callbacks != 0:
                    raise AssertionError("READY emitted physics before Enable")
                report["ready_zero_physics"] = True
                report["states"]["enable_request"] = self._state(obs)
                self._ready_capture_task = asyncio.ensure_future(self._capture_ready())
                self._smoke_stage = "ready_capture"
                _stage("ready_capture_requested", physics_s=self._physics_s, callbacks=self._callbacks)

            candidate = super().before_step(obs)
            now = time.monotonic()
            if self.session.state == self.session.PAUSED_FAULT:
                raise AssertionError(f"online session FAULT in {self._smoke_stage}: {self.session.reason}")
            if self.waypoint.state in ("STALLED", "TIMED_OUT"):
                status = self.waypoint.status()
                raise AssertionError(f"waypoint {self.waypoint.state}: {status}")

            if self._smoke_stage == "ready_capture":
                if self._physics_s != 0.0 or self._callbacks != 0:
                    raise AssertionError("READY GUI capture emitted physics")
                if self._ready_capture_task is not None and not self._ready_capture_task.done():
                    return candidate
                if self._ready_capture_task is None:
                    raise AssertionError("READY swapchain capture was not scheduled")
                self._ready_capture_task.result()
                if not report.get("ready_capture", {}).get("exists"):
                    raise AssertionError("READY Kit swapchain screenshot is missing")
                self.panel.request_enable()
                self._smoke_stage = "enable_requested"
                _stage("enable_requested_after_ready_capture", physics_s=self._physics_s,
                       callbacks=self._callbacks)

            elif self._smoke_stage == "enable_requested" and self.session.state == self.session.ACTIVE:
                if self._physics_s != 0.0 or self._callbacks != 0:
                    raise AssertionError("Enable pump emitted physics before the first authorized env.step")
                refreshed = _metadata(obs)
                report["observations"]["after_first_online_refresh"] = refreshed
                report["observations"]["layout_preserved"] = refreshed == report["observations"]["before_online_refresh"]
                if not report["observations"]["layout_preserved"]:
                    raise AssertionError("online refresh changed observation keys/shapes/dtypes/devices")
                self._mask_and_metadata(obs)
                self._initial_physics = self._physics_s
                self._smoke_stage = "initial_active"
                _stage("initial_active", physics_s=self._physics_s)

            elif self._smoke_stage == "waypoint_requested":
                if self.waypoint.state not in ("RUNNING", "SETTLING"):
                    raise AssertionError(f"native Go did not start waypoint: state={self.waypoint.state}, error={self._last_error}")
                if candidate is None or self.session.state != self.session.ACTIVE:
                    raise AssertionError("waypoint request stopped the active policy/physics path")
                self._waypoint_accept_physics = self._physics_s
                report["waypoint_accepted"] = {"physics_s": self._physics_s, "callbacks": self._callbacks,
                                                "provider_last_sequence": self.provider.last_sequence,
                                                "waypoint_status": self.waypoint.status()}
                self._last_trajectory_physics = self._physics_s
                self._smoke_stage = "waypoint_active"
                self._assert_metadata(obs, "waypoint_accepted")
                _stage("waypoint_active", waypoint=self.waypoint.status(), physics_s=self._physics_s)

            elif self._smoke_stage == "pause_requested" and self.session.state == self.session.PAUSED_USER:
                confirmed = self._state(obs)
                request_delta = self._state_delta(self._pause_request_state, confirmed)
                physics_delta = self._physics_s - self._pause_request_physics
                callbacks_delta = self._callbacks - self._pause_request_callbacks
                report["pause_request_to_confirmation"] = {
                    "physics_delta_s": physics_delta, "callbacks_delta": callbacks_delta,
                    "state_history_max_delta": request_delta,
                }
                if abs(physics_delta) > 1e-9 or callbacks_delta != 0 or request_delta > 1e-7:
                    raise AssertionError("Pause request-to-confirmation advanced physics/state/history")
                self._pause_confirm_state, self._pause_confirm_physics = confirmed, self._physics_s
                self._pause_confirm_callbacks, self._stage_wall = self._callbacks, now
                self._capture_task = asyncio.ensure_future(self._capture())
                self._smoke_stage = "pause_freeze"
                self._assert_metadata(obs, "pause_confirmed")
                _stage("pause_confirmed", physics_s=self._physics_s, callbacks=self._callbacks)

            elif self._smoke_stage == "pause_freeze" and now - self._stage_wall >= 0.5:
                frozen = self._state(obs)
                delta = self._state_delta(self._pause_confirm_state, frozen)
                if (abs(self._physics_s - self._pause_confirm_physics) > 1e-9
                        or self._callbacks != self._pause_confirm_callbacks or delta > 1e-7):
                    raise AssertionError("confirmed Pause did not freeze physics/state/history for 0.5s")
                if self._capture_task is not None and not self._capture_task.done():
                    return candidate
                if self._capture_task is None:
                    raise AssertionError("swapchain capture was not scheduled")
                self._capture_task.result()
                if not report.get("capture", {}).get("exists"):
                    raise AssertionError("paused Kit swapchain screenshot is missing")
                report["pause_freeze"] = {"wall_s": now - self._stage_wall,
                                           "physics_delta_s": self._physics_s - self._pause_confirm_physics,
                                           "callbacks_delta": self._callbacks - self._pause_confirm_callbacks,
                                           "state_history_max_delta": delta}
                report["states"]["paused_frozen"] = frozen
                self._assert_metadata(obs, "pause_frozen")
                self._assert_clean()
                report["result"] = "PASS"
                report["message"] = f"{scenario.name} waypoint arrived, held under live physics, then froze safely"
                report["exit_requested_monotonic_s"] = time.monotonic()
                _stage("native_x_close_requested")
                if self.panel.window is None:
                    raise AssertionError("online panel has no native window")
                completed = True
                self._smoke_stage = "exit_requested"
                self.panel.window.visible = False
            return candidate

        def validate_actions(self, actions):
            self._policy_calls += 1
            report["real_policy_evaluations"] = self._policy_calls
            return super().validate_actions(actions)

        def after_step(self, obs):
            self._last_obs = obs
            self._env_steps += 1
            report["real_env_steps"] = self._env_steps
            before_physics = getattr(self, "_last_after_physics", self._physics_s - self.env.step_dt)
            super().after_step(obs)
            self._last_after_physics = self._physics_s
            self._assert_deadline()
            if self.session.state == self.session.PAUSED_FAULT:
                raise AssertionError(f"online session FAULT in {self._smoke_stage}: {self.session.reason}")
            if self.waypoint.state in ("STALLED", "TIMED_OUT"):
                telemetry = self._telemetry()
                raise AssertionError(f"waypoint {self.waypoint.state}: {self.waypoint.reason}; actual={telemetry}; status={self.waypoint.status()}")

            if self._smoke_stage == "initial_active" and self._physics_s - self._initial_physics >= 1.0:
                initial = self._telemetry()
                self._target_yaw = initial["yaw_rad"]
                self._target_xyz = scenario.target(initial["xyz"], self._target_yaw)
                report["input"] = {"initial_actual": initial, "target_xyz": self._target_xyz,
                                   "heading_degrees": math.degrees(self._target_yaw),
                                   "horizontal_forward_offset_m": scenario.forward_m,
                                   "vertical_offset_m": scenario.height_m, "scenario": scenario.name}
                for model, value in zip(self.panel.waypoint_position_models, self._target_xyz, strict=True):
                    model.set_value(float(value))
                self.panel.waypoint_heading_model.set_value(math.degrees(self._target_yaw))
                self.panel._apply_waypoint_widgets()
                self._smoke_stage = "waypoint_requested"
                _stage("native_waypoint_go", input=report["input"], physics_s=self._physics_s)

            elif self._smoke_stage in ("waypoint_active", "hold"):
                phase = self._smoke_stage
                step_physics = self._physics_s - before_physics
                telemetry, errors = self._record_trajectory(phase, step_physics)
                if phase == "waypoint_active" and self.waypoint.state in ("RUNNING", "SETTLING"):
                    waypoint_elapsed = float(self.waypoint.status()["elapsed_s"])
                    physx_elapsed = self._physics_s - self._waypoint_accept_physics
                    if abs(waypoint_elapsed - physx_elapsed) > 1e-6:
                        raise AssertionError(
                            f"waypoint elapsed {waypoint_elapsed} differs from real PhysX elapsed {physx_elapsed}"
                        )
                within = (errors["position_3d_m"] <= 0.05 and errors["height_m"] <= 0.03
                          and errors["yaw_rad"] <= math.radians(5.0)
                          and telemetry["linear_speed_3d_m_s"] <= 0.05
                          and telemetry["angular_speed_rad_s"] <= 0.10)
                self._independent_settle_s = self._independent_settle_s + step_physics if within else 0.0

                if phase == "waypoint_active" and self.waypoint.state == "ARRIVED":
                    if not within or self._independent_settle_s + 1e-6 < 0.5:
                        raise AssertionError(f"ARRIVED lacked independent 0.5s real-telemetry proof: errors={errors}, actual={telemetry}, held={self._independent_settle_s}")
                    self._hold_start_physics, self._hold_start_steps = self._physics_s, self._env_steps
                    report["arrival"] = {"physics_s": self._physics_s, "env_step": self._env_steps,
                                         "independent_settle_s": self._independent_settle_s,
                                         "actual": telemetry, "errors": errors,
                                         "waypoint_status": self.waypoint.status()}
                    self._smoke_stage = "hold"
                    self._assert_metadata(obs, "arrived")
                    _stage("arrived_hold_live_physics", arrival=report["arrival"])
                elif phase == "hold":
                    if errors["position_3d_m"] > 0.05 or errors["height_m"] > 0.03:
                        raise AssertionError(f"live hold left position/height tolerance: {errors}")
                    if self._physics_s - self._hold_start_physics >= 2.0:
                        if telemetry["linear_speed_3d_m_s"] > 0.05 or telemetry["angular_speed_rad_s"] > 0.10:
                            raise AssertionError(f"final live-hold velocity too high: {telemetry}")
                        report["hold"] = {"physics_s": self._physics_s - self._hold_start_physics,
                                          "env_steps": self._env_steps - self._hold_start_steps,
                                          "final_actual": telemetry, "final_errors": errors,
                                          "physics_continued": self._physics_s > self._hold_start_physics}
                        if not report["hold"]["physics_continued"]:
                            raise AssertionError("ARRIVED hold used a pause instead of real physics")
                        self._assert_metadata(obs, "live_hold_complete")
                        self._assert_clean()
                        self._pause_request_state = self._state(obs)
                        self._pause_request_physics, self._pause_request_callbacks = self._physics_s, self._callbacks
                        report["states"]["pause_requested"] = self._pause_request_state
                        self.panel.request_pause()
                        self._smoke_stage = "pause_requested"
                        self._stage_wall = time.monotonic()
                        _stage("pause_requested_after_live_hold", physics_s=self._physics_s,
                               callbacks=self._callbacks)

        def close(self):
            if self._closed_for_smoke:
                return
            self._closed_for_smoke = True
            if not completed and "failure_snapshot" not in report:
                try:
                    report["failure_snapshot"] = self.failure_snapshot()
                except BaseException as snapshot_error:
                    report["failure_snapshot_error"] = f"{type(snapshot_error).__name__}: {snapshot_error}"
            capture_task, self._capture_task = getattr(self, "_capture_task", None), None
            if capture_task is not None and not capture_task.done():
                capture_task.cancel()
            ready_task, self._ready_capture_task = getattr(self, "_ready_capture_task", None), None
            if ready_task is not None and not ready_task.done():
                ready_task.cancel()
            subscription, self._physics_subscription = getattr(self, "_physics_subscription", None), None
            if subscription is not None:
                unsubscribe = getattr(subscription, "unsubscribe", None)
                if unsubscribe is not None:
                    unsubscribe()
            for owner, name, original in reversed(getattr(self, "_restorers", [])):
                setattr(owner, name, original)
            self._restorers = []
            super().close()

        def failure_snapshot(self):
            snapshot = {"stage": self._smoke_stage, "physics_s": self._physics_s,
                        "callbacks": self._callbacks, "waypoint": self.waypoint.status(),
                        "session_state": self.session.state, "session_reason": self.session.reason,
                        "last_error": self._last_error, "actual": self._telemetry()}
            obs = getattr(self, "_last_obs", None)
            if obs is not None:
                snapshot["state_and_history"] = self._state(obs)
                snapshot["observation_metadata"] = _metadata(obs)
            if hasattr(self, "_target_xyz"):
                snapshot["target_xyz"] = self._target_xyz
                snapshot["target_yaw_rad"] = self._target_yaw
                snapshot["errors"] = self._errors(snapshot["actual"])
            return snapshot

    online_control.OnlinePlaybackController = WaypointSmokeController
    _stage("player_main")
    player["main"]()
    if not completed:
        raise AssertionError("player exited before waypoint acceptance completed")
    report["result"] = "PASS"
    _write_report()
    print(f"[BFM WAYPOINT SMOKE] PASS report={REPORT_PATH}", flush=True)
except BaseException as error:
    completed = False
    report["result"] = "FAIL"
    report.setdefault("exit_requested_monotonic_s", time.monotonic())
    report["error_type"], report["error"] = type(error).__name__, str(error)
    report["traceback"] = traceback.format_exc()
    if controller_instance is not None and "failure_snapshot" not in report:
        try:
            report["failure_snapshot"] = controller_instance.failure_snapshot()
        except BaseException as snapshot_error:
            report["failure_snapshot_error"] = f"{type(snapshot_error).__name__}: {snapshot_error}"
    try:
        _write_report()
    except BaseException:
        traceback.print_exc()
    traceback.print_exc()
    print(f"[BFM WAYPOINT SMOKE] FAIL report={REPORT_PATH}", flush=True)
    raise
finally:
    if runner_class is not None and original_get_policy is not None:
        runner_class.get_inference_policy = original_get_policy
    if original_controller is not None:
        online_control.OnlinePlaybackController = original_controller
    if app is not None:
        app.close(exit_code=0 if completed and sys.exc_info()[0] is None else 1)
