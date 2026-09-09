"""Bounded real-policy/real-physics acceptance for the G1 Online Targets GUI.

Pass the normal ``play.py`` online arguments, including ``--online_targets``,
``--motion_menu``, ``--num_envs 1``, ``--mode_index 2`` and ``--viz kit``.
The parent process owns the GPU/App lifetime and must enforce a 180 second
systemd deadline.  This driver never substitutes the policy, environment, or
physics and never writes robot or target-object state.
"""

from __future__ import annotations

import asyncio
import json
import hashlib
import os
from pathlib import Path
import runpy
import sys
import time
import traceback

import numpy as np


CASE = os.environ.get("BFM_ONLINE_SMOKE_CASE", "")
REPORT_VALUE = os.environ.get("BFM_ONLINE_SMOKE_REPORT", "")
if CASE not in ("normal", "stale"):
    raise ValueError("BFM_ONLINE_SMOKE_CASE must be normal or stale")
if not REPORT_VALUE:
    raise ValueError("BFM_ONLINE_SMOKE_REPORT must name a JSON output file")

REPORT_PATH = Path(REPORT_VALUE).expanduser().resolve()
START = time.monotonic()
INTERNAL_DEADLINE_S = 165.0  # Leave 15 seconds for cleanup before the outer 180s bound.
DEADLINE = START + INTERNAL_DEADLINE_S
report = {
    "case": CASE,
    "result": "RUNNING",
    "started_monotonic_s": START,
    "deadline_monotonic_s": DEADLINE,
    "stage": "loading_player",
    "stages": [],
    "physics": {"elapsed_s": 0.0, "callbacks": 0},
    "real_policy_evaluations": 0,
    "real_env_steps": 0,
    "accepted_sequence": None,
    "accepted_sequence_count": 0,
    "accepted_sequences": [],
    "goal_commit_observation_checks": [],
    "rejected_count": None,
    "postattach_calls": {},
    "instrumentation_errors": [],
    "observations": None,
    "states": {},
    "trajectory": [],
    "argv": list(sys.argv),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value):
    value = getattr(value, "torch", value)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _write_report() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = REPORT_PATH.with_name(REPORT_PATH.name + ".tmp")
    temporary.write_text(json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(REPORT_PATH)


def _obs_metadata(value):
    if hasattr(value, "items"):
        return {str(key): _obs_metadata(child) for key, child in value.items()}
    raw = getattr(value, "torch", value)
    return {
        "shape": list(getattr(raw, "shape", ())),
        "dtype": str(getattr(raw, "dtype", type(raw).__name__)),
        "device": str(getattr(raw, "device", "cpu")),
    }


def _array(value):
    value = getattr(value, "torch", value)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value).copy()


def _max_state_delta(left, right) -> float:
    deltas = []
    for key in ("body_positions", "body_quaternions_wxyz", "root_state", "joint_positions", "joint_velocities"):
        deltas.append(float(np.max(np.abs(np.asarray(left[key]) - np.asarray(right[key])))))
    return max(deltas)


def _stage(name: str, **details) -> None:
    report["stage"] = name
    report["stages"].append({"name": name, "monotonic_s": time.monotonic(), **details})
    _write_report()


scripts = Path(__file__).resolve().parents[1] / "scripts/pretrain/rsl_rl"
sys.path.insert(0, str(scripts))
source_root = Path(__file__).resolve().parents[1] / "source/scaletrack/scaletrack"
report["source_sha256"] = {
    "online_control.py": _sha256(scripts / "online_control.py"),
    "online_ui.py": _sha256(scripts / "online_ui.py"),
    "live_commands.py": _sha256(source_root / "tasks/tracking/mdp/live_commands.py"),
    "live_reference.py": _sha256(source_root / "utils/live_reference.py"),
    "play.py": _sha256(scripts / "play.py"),
}
player = None
app = None
completed = False

try:
    player = runpy.run_path(str(scripts / "play.py"))
    app = player["simulation_app"]

    import omni.physx
    import omni.appwindow
    import omni.kit.app
    import omni.kit.renderer.capture
    import omni.timeline
    import online_control
    from online_ui import _xyz_degrees_from_wxyz

    if not player["args_cli"].online_targets:
        raise AssertionError("smoke requires --online_targets")
    if not player["args_cli"].motion_menu or player["args_cli"].mode_index != 2:
        raise AssertionError("smoke requires --motion_menu and VR-3 --mode_index 2")

    RealController = online_control.OnlinePlaybackController

    class SmokeOnlineController(RealController):
        """Test driver attached after live-reference construction and final reset."""

        def __init__(self, env, command, panel, **kwargs):
            super().__init__(env, command, panel, **kwargs)
            self._smoke_stage = "wait_heartbeat"
            self._stage_started = time.monotonic()
            self._physics_s = 0.0
            self._physics_callbacks = 0
            self._policy_evaluations = 0
            self._env_steps = 0
            self._restorers = []
            self._seed_actual = self._actual()
            self._seed_points = self._seed_actual[0][[0, 10, 13]].copy()
            self._seed_angles = _xyz_degrees_from_wxyz(self._seed_actual[1][[0, 10, 13]])
            self._submitted = 0
            self._accepted_sequences = set()
            self._pause_cycle = 0
            self._invalid_probe_cycles = set()
            self._pause_physics = None
            self._pause_state = None
            self._resume_physics = None
            self._resume_state = None
            self._timeline_probe_physics = None
            self._timeline_probe_state = None
            self._timeline_probe_policy_evaluations = None
            self._timeline_probe_env_steps = None
            self._native_play_update_subscription = None
            self._capture_task = None
            self._closed_for_smoke = False
            self._instrument_postattach_calls()
            report["target_object"] = {
                "present": "target_object" in getattr(self.env.scene, "_rigid_objects", {})
            }
            physx = omni.physx.get_physx_interface()
            self._physics_subscription = physx.subscribe_physics_on_step_events(
                self._on_physics_step, pre_step=False, order=0
            )
            report["states"]["postattach_initial"] = self._capture_state()
            _stage("wait_heartbeat")

        def _on_physics_step(self, dt):
            self._physics_s += float(dt)
            self._physics_callbacks += 1
            report["physics"] = {"elapsed_s": self._physics_s, "callbacks": self._physics_callbacks}

        def _wrap_counter(self, owner, method_name: str, label: str) -> None:
            original = getattr(owner, method_name, None)
            if original is None or not callable(original):
                return
            report["postattach_calls"].setdefault(label, 0)

            def counted(*args, **kwargs):
                report["postattach_calls"][label] += 1
                return original(*args, **kwargs)

            try:
                setattr(owner, method_name, counted)
            except (AttributeError, TypeError) as error:
                report["instrumentation_errors"].append(f"{label}: {error}")
                return
            self._restorers.append((owner, method_name, original))

        def _instrument_postattach_calls(self) -> None:
            for name in ("reset", "_reset_idx"):
                self._wrap_counter(self.env, name, f"env.{name}")
            for manager_name in ("event_manager", "command_manager"):
                manager = getattr(self.env, manager_name, None)
                if manager is not None:
                    self._wrap_counter(manager, "reset", f"{manager_name}.reset")
            scene = self.env.scene
            self._wrap_counter(scene, "reset", "scene.reset")
            registries = (("articulation", getattr(scene, "_articulations", {})),
                          ("rigid_object", getattr(scene, "_rigid_objects", {})))
            for kind, registry in registries:
                for asset_name, asset in registry.items():
                    self._wrap_counter(asset, "reset", f"{kind}.{asset_name}.reset")
                    for method_name in ("write_root_pose_to_sim", "write_root_velocity_to_sim",
                                        "write_root_state_to_sim", "write_joint_state_to_sim"):
                        self._wrap_counter(asset, method_name, f"{kind}.{asset_name}.{method_name}")

        def _capture_state(self):
            positions, quaternions, joints, velocities = self._actual()
            data = self.command.robot.data
            root = getattr(data, "root_state_w", None)
            if root is None:
                root_parts = [getattr(data, name) for name in
                              ("root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_w")]
                root = np.concatenate([_array(part)[0].reshape(-1) for part in root_parts])
            else:
                root = _array(root)[0]
            return {
                "body_names": list(self.command.cfg.body_names),
                "body_positions": positions,
                "body_quaternions_wxyz": quaternions,
                "root_state": root,
                "joint_positions": joints,
                "joint_velocities": velocities,
            }

        def _record_trajectory(self):
            actual = self._actual()
            actual_points = actual[0][[0, 10, 13]]
            actual_quaternions = actual[1][[0, 10, 13]]
            reference = self.provider.sample([0])
            reference_points = _array(reference["body_pos"])[0, [0, 10, 13]]
            reference_quaternions = _array(reference["body_quat"])[0, [0, 10, 13]]
            goals = self.provider.goal_positions
            goal_quaternions = self.provider.goal_quaternions
            report["trajectory"].append({
                "physics_s": self._physics_s,
                "provider_last_sequence": self.provider.last_sequence,
                "actual_positions": actual_points,
                "actual_quaternions_wxyz": actual_quaternions,
                "reference_positions": reference_points,
                "reference_quaternions_wxyz": reference_quaternions,
                "goal_positions": goals,
                "goal_quaternions_wxyz": goal_quaternions,
                "position_errors": np.linalg.norm(actual_points - goals, axis=-1),
                "orientation_errors": self._orientation_errors(actual_quaternions, goal_quaternions),
                "actual_reference_position_errors": np.linalg.norm(actual_points - reference_points, axis=-1),
                "actual_reference_orientation_errors": self._orientation_errors(
                    actual_quaternions, reference_quaternions
                ),
            })
            report["accepted_sequence"] = self.provider.last_sequence
            if self.provider.last_sequence is not None:
                self._accepted_sequences.add(int(self.provider.last_sequence))
            report["accepted_sequences"] = sorted(self._accepted_sequences)
            report["accepted_sequence_count"] = len(self._accepted_sequences)
            report["rejected_count"] = self.provider.rejected_count

        def _assert_deadline(self):
            if time.monotonic() >= DEADLINE:
                raise AssertionError(f"internal {INTERNAL_DEADLINE_S:g}s deadline expired in {self._smoke_stage}")

        def _assert_no_postattach_writes(self):
            unexpected = {name: count for name, count in report["postattach_calls"].items() if count}
            if unexpected:
                raise AssertionError(f"post-attach reset/state-write calls detected: {unexpected}")

        def _submit_sine_goal(self):
            self._submitted += 1
            phase = 2.0 * np.pi * self._submitted / 100.0
            positions = self._seed_points.copy()
            positions[:, 0] += 0.005 * np.sin(phase)
            self.panel.apply(positions, self._seed_angles)

        async def _capture_first_pause(self) -> None:
            capture_path = REPORT_PATH.with_suffix(".png")
            report["capture"] = {"path": str(capture_path), "status": "requested", "exists": False}
            _write_report()
            try:
                capture_path.parent.mkdir(parents=True, exist_ok=True)
                capture = omni.kit.renderer.capture.acquire_renderer_capture_interface()
                app_window = omni.appwindow.get_default_app_window()
                capture.capture_next_frame_swapchain(str(capture_path), app_window)
                await omni.kit.app.get_app().next_update_async()
                capture.wait_async_capture(app_window)
                await omni.kit.app.get_app().next_update_async()
                for _ in range(20):
                    if capture_path.is_file():
                        break
                    await omni.kit.app.get_app().next_update_async()
                exists = capture_path.is_file()
                report["capture"] = {
                    "path": str(capture_path),
                    "status": "captured" if exists else "missing",
                    "exists": exists,
                    "size_bytes": capture_path.stat().st_size if exists else None,
                }
            except BaseException as error:
                report["capture"] = {
                    "path": str(capture_path),
                    "status": "error",
                    "exists": capture_path.is_file(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            finally:
                _write_report()

        def _request_exit(self, result: str, message: str):
            global completed
            self._record_trajectory()
            if CASE == "normal" and len(self._accepted_sequences) < 100:
                raise AssertionError(
                    f"observed only {len(self._accepted_sequences)} distinct accepted sequences; need at least 100"
                )
            self._assert_no_postattach_writes()
            report["result"] = result
            report["message"] = message
            report["states"]["final"] = self._capture_state()
            report["exit_requested_monotonic_s"] = time.monotonic()
            _stage("exit_requested", message=message)
            completed = result == "PASS"
            self.panel.request_close()

        def _arm_native_play_from_kit_update(self) -> None:
            def request_native_play(_event):
                if self._smoke_stage != "native_play_armed":
                    return
                subscription, self._native_play_update_subscription = self._native_play_update_subscription, None
                if subscription is not None:
                    unsubscribe = getattr(subscription, "unsubscribe", None)
                    if unsubscribe is not None:
                        unsubscribe()
                report["native_timeline_play_requested_monotonic_s"] = time.monotonic()
                self._timeline_probe_physics = self._physics_s
                self._timeline_probe_state = self._capture_state()
                self._timeline_probe_policy_evaluations = self._policy_evaluations
                self._timeline_probe_env_steps = self._env_steps
                self._smoke_stage = "native_play_probe"
                self._stage_started = time.monotonic()
                _stage("native_timeline_play_from_pause_pump", physics_s=self._physics_s)
                omni.timeline.get_timeline_interface().play()

            stream = omni.kit.app.get_app().get_update_event_stream()
            self._native_play_update_subscription = stream.create_subscription_to_pop(
                request_native_play, name="bfm_online_smoke_native_play_once"
            )

        def before_step(self, obs):
            global completed
            self._assert_deadline()
            if report["observations"] is None:
                report["observations"] = {"before_online_refresh": _obs_metadata(obs)}
                _write_report()

            if self._smoke_stage == "wait_heartbeat" and self.panel.heartbeat_stamp is not None:
                self.panel.request_enable()
                self._smoke_stage = "enable_requested"
                self._stage_started = time.monotonic()
                _stage("enable_requested")

            sequence_before = self.provider.last_sequence
            task_before = _array(obs["policy_task"])
            candidate = super().before_step(obs)
            if not self._closed and self.provider.last_sequence != sequence_before:
                changed = not np.array_equal(task_before, _array(obs["policy_task"]))
                report["goal_commit_observation_checks"].append({
                    "sequence": self.provider.last_sequence,
                    "policy_task_changed_without_env_step": changed,
                })
                if not changed:
                    raise AssertionError("accepted changing goal did not refresh the next policy-task observation")
            now = time.monotonic()

            expected_fault = CASE == "stale" and self._smoke_stage in {
                "fault_wait", "fault_freeze", "native_play_probe", "exit_requested"
            }
            if self.session.state == self.session.PAUSED_FAULT and not expected_fault:
                raise AssertionError(f"unexpected safety fault during {self._smoke_stage}: {self.session.reason}")

            if self._smoke_stage == "enable_requested" and self.session.state == self.session.ACTIVE:
                refreshed_metadata = _obs_metadata(obs)
                report["observations"]["after_first_online_refresh"] = refreshed_metadata
                report["observations"]["layout_preserved"] = (
                    refreshed_metadata == report["observations"]["before_online_refresh"]
                )
                if not report["observations"]["layout_preserved"]:
                    raise AssertionError("first online observation refresh changed keys/shapes/dtypes/devices")
                self._smoke_stage = "active"
                self._stage_started = now
                report["states"]["enabled"] = self._capture_state()
                _stage("active")

            if self._smoke_stage == "pause_requested" and self.session.state == self.session.PAUSED_USER:
                self._pause_physics = self._physics_s
                self._pause_state = self._capture_state()
                self._smoke_stage = "pause_freeze"
                self._stage_started = now
                report["states"][f"pause_cycle_{self._pause_cycle}_paused"] = self._pause_state
                _stage(f"pause_cycle_{self._pause_cycle}_freeze", physics_s=self._pause_physics)
                if CASE == "normal" and self._pause_cycle == 1 and self._capture_task is None:
                    self._capture_task = asyncio.ensure_future(self._capture_first_pause())

            if self._smoke_stage == "pause_freeze" and now - self._stage_started >= 2.0:
                frozen = self._capture_state()
                if abs(self._physics_s - self._pause_physics) > 1e-9 or _max_state_delta(self._pause_state, frozen) > 1e-7:
                    raise AssertionError("user Pause did not freeze physical time and actual state for 2s")
                report["states"][f"pause_cycle_{self._pause_cycle}_after_2s"] = frozen
                self.panel.request_resume()
                self._smoke_stage = "resume_requested"
                self._stage_started = now
                _stage(f"pause_cycle_{self._pause_cycle}_resume_requested")

            if self._smoke_stage == "resume_requested" and self.session.state == self.session.ACTIVE:
                if not self._requires_goal or candidate is not None:
                    raise AssertionError("Resume did not latch awaiting_goal and freeze before fresh Apply")
                self._resume_physics = self._physics_s
                self._resume_state = self._capture_state()
                self._smoke_stage = "await_out_of_bounds_probe"
                self._stage_started = now
                _stage(f"pause_cycle_{self._pause_cycle}_await_out_of_bounds_probe",
                       physics_s=self._resume_physics)

            if self._smoke_stage == "await_out_of_bounds_probe" and now - self._stage_started >= 1.0:
                frozen = self._capture_state()
                if abs(self._physics_s - self._resume_physics) > 1e-9 or _max_state_delta(self._resume_state, frozen) > 1e-7:
                    raise AssertionError("Resume advanced physics before a fresh Apply")
                report["states"][f"pause_cycle_{self._pause_cycle}_resume_wait_frozen"] = frozen
                if self._pause_cycle in self._invalid_probe_cycles:
                    raise AssertionError(f"pause cycle {self._pause_cycle} repeated its out-of-bounds probe")
                self._invalid_probe_cycles.add(self._pause_cycle)
                self._invalid_goal_positions = self.provider.goal_positions
                self._invalid_goal_quaternions = self.provider.goal_quaternions
                self._invalid_last_sequence = self.provider.last_sequence
                self._invalid_rejected_count = self.provider.rejected_count
                invalid_positions = self._seed_points.copy()
                invalid_positions[0, 0] += 10.0
                self.panel.apply(invalid_positions, self._seed_angles)
                self._smoke_stage = "out_of_bounds_requested"
                self._stage_started = now
                _stage(f"pause_cycle_{self._pause_cycle}_out_of_bounds_requested")
                return candidate

            if self._smoke_stage == "out_of_bounds_requested":
                frozen = self._capture_state()
                if candidate is not None or not self._requires_goal:
                    raise AssertionError("out-of-bounds Apply bypassed the fresh-goal freeze")
                if abs(self._physics_s - self._resume_physics) > 1e-9 or _max_state_delta(self._resume_state, frozen) > 1e-7:
                    raise AssertionError("out-of-bounds Apply advanced physics or actual state")
                if self.provider.last_sequence != self._invalid_last_sequence:
                    raise AssertionError("rejected out-of-bounds Apply changed the accepted sequence")
                if (not np.array_equal(self.provider.goal_positions, self._invalid_goal_positions)
                        or not np.array_equal(self.provider.goal_quaternions, self._invalid_goal_quaternions)):
                    raise AssertionError("rejected out-of-bounds Apply changed the provider goal")
                if self.provider.rejected_count != self._invalid_rejected_count + 1:
                    raise AssertionError("out-of-bounds Apply was not rejected exactly once")
                report.setdefault("out_of_bounds_probes", []).append({
                    "pause_cycle": self._pause_cycle,
                    "candidate_was_none": candidate is None,
                    "requires_goal": self._requires_goal,
                    "physics_and_state_frozen": True,
                    "goal_unchanged": True,
                    "accepted_sequence_unchanged": True,
                    "physics_s": self._physics_s,
                    "accepted_sequence": self.provider.last_sequence,
                    "rejected_count": self.provider.rejected_count,
                })
                self._submit_sine_goal()
                self._smoke_stage = "fresh_apply_requested"
                self._stage_started = now
                _stage(f"pause_cycle_{self._pause_cycle}_out_of_bounds_rejected_then_fresh_valid_apply")

            if self._smoke_stage == "fault_wait" and self.session.state == self.session.PAUSED_FAULT:
                self._pause_physics = self._physics_s
                self._pause_state = self._capture_state()
                self._smoke_stage = "fault_freeze"
                self._stage_started = now
                report["states"]["faulted"] = self._pause_state
                _stage("fault_freeze", reason=self.session.reason, physics_s=self._pause_physics)

            if self._smoke_stage == "fault_freeze" and now - self._stage_started >= 2.0:
                frozen = self._capture_state()
                if abs(self._physics_s - self._pause_physics) > 1e-9 or _max_state_delta(self._pause_state, frozen) > 1e-7:
                    raise AssertionError("PAUSED_FAULT did not freeze physics and actual state for 2s")
                report["states"]["fault_after_2s"] = frozen
                self._smoke_stage = "native_play_armed"
                self._stage_started = now
                self._arm_native_play_from_kit_update()
                _stage("native_timeline_play_armed", physics_s=self._physics_s)

            if self._smoke_stage == "native_play_probe" and now - self._stage_started >= 2.0:
                frozen = self._capture_state()
                if abs(self._physics_s - self._timeline_probe_physics) > 1e-9 or _max_state_delta(self._timeline_probe_state, frozen) > 1e-7:
                    raise AssertionError("native timeline Play bypassed the fault latch")
                if (self._policy_evaluations != self._timeline_probe_policy_evaluations
                        or self._env_steps != self._timeline_probe_env_steps):
                    raise AssertionError("native timeline Play restarted policy inference or env.step")
                report["states"]["native_play_still_frozen"] = frozen
                report["exit_requested_monotonic_s"] = time.monotonic()
                report["result"] = "PASS"
                report["message"] = "stale heartbeat fault froze physics and native Play could not bypass it"
                self._assert_no_postattach_writes()
                _stage("native_x_close_requested")
                if self.panel.window is None:
                    raise AssertionError("online panel has no native window")
                self.panel.window.visible = False
                self._smoke_stage = "exit_requested"
                completed = True

            return candidate

        def validate_actions(self, actions) -> bool:
            self._policy_evaluations += 1
            report["real_policy_evaluations"] = self._policy_evaluations
            return super().validate_actions(actions)

        def after_step(self, obs) -> None:
            self._env_steps += 1
            report["real_env_steps"] = self._env_steps
            super().after_step(obs)
            self._assert_deadline()
            self._record_trajectory()

            if self._smoke_stage == "active":
                if CASE == "normal":
                    if self._submitted < 110:
                        self._submit_sine_goal()
                    if self._physics_s >= 0.5:
                        self._pause_cycle = 1
                        self.panel.request_pause()
                        self._smoke_stage = "pause_requested"
                        self._stage_started = time.monotonic()
                        _stage("early_pause_requested", physics_s=self._physics_s,
                               accepted_sequence=self.provider.last_sequence)
                elif self._physics_s >= 1.0:
                    subscription, self.panel._subscription = self.panel._subscription, None
                    if subscription is None:
                        raise AssertionError("panel update subscription was already absent")
                    subscription.unsubscribe()
                    report["heartbeat_subscription_cancelled_monotonic_s"] = time.monotonic()
                    self._smoke_stage = "fault_wait"
                    self._stage_started = time.monotonic()
                    _stage("heartbeat_stopped", physics_s=self._physics_s)

            elif self._smoke_stage == "streaming":
                if self._submitted < 110:
                    self._submit_sine_goal()
                accepted = len(self._accepted_sequences)
                if self._physics_s >= 10.0 and accepted >= 100:
                    self._pause_cycle = 2
                    self.panel.request_pause()
                    self._smoke_stage = "pause_requested"
                    self._stage_started = time.monotonic()
                    _stage("final_pause_requested", physics_s=self._physics_s,
                           distinct_accepted_sequences=accepted)

            elif self._smoke_stage == "fresh_apply_requested" and self._physics_s - self._resume_physics >= 0.5:
                recovered = self._capture_state()
                if _max_state_delta(self._resume_state, recovered) <= 1e-6:
                    raise AssertionError("fresh Apply resumed physical callbacks but actual robot state did not advance")
                report["states"][f"pause_cycle_{self._pause_cycle}_recovered"] = recovered
                if self._pause_cycle == 1:
                    self._smoke_stage = "streaming"
                    self._stage_started = time.monotonic()
                    _stage("streaming_live_goals", physics_s=self._physics_s)
                else:
                    self._request_exit(
                        "PASS",
                        "early and final Pause/Resume gates, 100+ live goals, 10s physics, and actual recovery passed",
                    )

        def close(self) -> None:
            if self._closed_for_smoke:
                return
            self._closed_for_smoke = True
            subscription, self._physics_subscription = getattr(self, "_physics_subscription", None), None
            if subscription is not None:
                unsubscribe = getattr(subscription, "unsubscribe", None)
                if unsubscribe is not None:
                    unsubscribe()
            update_subscription, self._native_play_update_subscription = getattr(
                self, "_native_play_update_subscription", None
            ), None
            if update_subscription is not None:
                unsubscribe = getattr(update_subscription, "unsubscribe", None)
                if unsubscribe is not None:
                    unsubscribe()
            super().close()

    online_control.OnlinePlaybackController = SmokeOnlineController
    _stage("player_main")
    try:
        player["main"]()
    finally:
        online_control.OnlinePlaybackController = RealController

    if not completed:
        raise AssertionError("player exited before the requested smoke sequence completed")
    report["result"] = "PASS"
    _write_report()
    print(f"[BFM ONLINE SMOKE] PASS case={CASE} report={REPORT_PATH}", flush=True)
except BaseException as error:
    report["result"] = "FAIL"
    report["error_type"] = type(error).__name__
    report["error"] = str(error)
    report["traceback"] = traceback.format_exc()
    try:
        _write_report()
    except BaseException:
        traceback.print_exc()
    traceback.print_exc()
    print(f"[BFM ONLINE SMOKE] FAIL case={CASE} report={REPORT_PATH}", flush=True)
    raise
finally:
    if app is not None:
        app.close(exit_code=0 if completed and sys.exc_info()[0] is None else 1)
