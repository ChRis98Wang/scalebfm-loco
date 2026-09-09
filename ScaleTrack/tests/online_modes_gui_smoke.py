"""Bounded real-Kit acceptance for live Pelvis-1/UMI-2/VR-3 switching.

Run with the normal ``play.py`` online GUI arguments and an outer 180 second
process-group deadline.  This driver keeps the real policy, environment and
PhysX stepping path; it only instruments lifecycle/state calls.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time
import traceback

import numpy as np


REPORT_VALUE = os.environ.get("BFM_ONLINE_SMOKE_REPORT", "")
if not REPORT_VALUE:
    raise ValueError("BFM_ONLINE_SMOKE_REPORT must name a JSON output file")
REPORT_PATH = Path(REPORT_VALUE).expanduser().resolve()
START = time.monotonic()
DEADLINE = START + 165.0
MODE_INDICES = {"Pelvis-1": (0,), "UMI-2": (10, 13), "VR-3": (0, 10, 13)}
TARGET_ROWS = {"Pelvis-1": (0,), "UMI-2": (1, 2), "VR-3": (0, 1, 2)}
report = {
    "result": "RUNNING", "argv": list(sys.argv), "started_monotonic_s": START,
    "deadline_monotonic_s": DEADLINE, "stage": "loading_player", "stages": [],
    "physics": {"elapsed_s": 0.0, "callbacks": 0}, "real_env_steps": 0,
    "real_policy_evaluations": 0, "postattach_calls": {}, "switches": [], "modes": [], "trajectory": [],
}


def _sha(path):
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
    return {"shape": list(raw.shape), "dtype": str(raw.dtype), "device": str(raw.device)}


def _stage(name, **details):
    report["stage"] = name
    report["stages"].append({"name": name, "monotonic_s": time.monotonic(), **details})
    _write_report()


scripts = Path(__file__).resolve().parents[1] / "scripts/pretrain/rsl_rl"
source = Path(__file__).resolve().parents[1] / "source/scaletrack/scaletrack"
sys.path.insert(0, str(scripts))
report["source_sha256"] = {
    name: _sha(path) for name, path in {
        "play.py": scripts / "play.py", "online_control.py": scripts / "online_control.py",
        "online_ui.py": scripts / "online_ui.py",
        "live_commands.py": source / "tasks/tracking/mdp/live_commands.py",
        "live_reference.py": source / "utils/live_reference.py",
        "online_modes.py": source / "utils/online_modes.py",
    }.items()
}
app = None
completed = False
runner_class = original_get_policy = controller_class = original_controller = None

try:
    player = runpy.run_path(str(scripts / "play.py"))
    app = player["simulation_app"]
    if not player["args_cli"].online_targets or not player["args_cli"].motion_menu:
        raise AssertionError("smoke requires --online_targets and --motion_menu")
    if player["args_cli"].mode_index != 2:
        raise AssertionError("acceptance run must start with --mode_index 2 (VR-3)")

    import omni.appwindow
    import omni.kit.app
    import omni.kit.renderer.capture
    import omni.physx
    import online_control
    from online_ui import _xyz_degrees_from_wxyz

    runner_class = player["OnPolicyRunner"]
    original_get_policy = runner_class.get_inference_policy
    policy_capture = {}

    def capture_policy(runner, *args, **kwargs):
        policy = original_get_policy(runner, *args, **kwargs)
        policy_capture["actor_critic"] = runner.alg.policy
        return policy

    runner_class.get_inference_policy = capture_policy
    original_controller = online_control.OnlinePlaybackController

    class ModeSmokeController(original_controller):
        def __init__(self, env, command, panel, **kwargs):
            super().__init__(env, command, panel, **kwargs)
            self._actor = policy_capture.get("actor_critic")
            if self._actor is None:
                raise AssertionError("real inference policy was not captured")
            self._smoke_stage, self._stage_wall = "wait_heartbeat", time.monotonic()
            self._physics_s = self._physics_callbacks = self._env_steps = self._policy_calls = 0
            self._mode_index = -1
            self._physics_subscription = omni.physx.get_physx_interface().subscribe_physics_on_step_events(
                self._on_physics_step, pre_step=False, order=0
            )
            self._restorers, self._capture_task = [], None
            self._closed_for_smoke = False
            self._instrument()
            _stage("wait_heartbeat")

        def _on_physics_step(self, dt):
            self._physics_s += float(dt)
            self._physics_callbacks += 1
            report["physics"] = {"elapsed_s": self._physics_s, "callbacks": self._physics_callbacks}

        def _wrap(self, owner, name, label):
            original = getattr(owner, name, None)
            if not callable(original):
                return
            report["postattach_calls"].setdefault(label, 0)

            def counted(*args, **kwargs):
                report["postattach_calls"][label] += 1
                return original(*args, **kwargs)

            setattr(owner, name, counted)
            self._restorers.append((owner, name, original))

        def _instrument(self):
            for name in ("reset", "_reset_idx"):
                self._wrap(self.env, name, f"env.{name}")
            self._wrap(self.env.scene, "reset", "scene.reset")
            self._wrap(self.command, "_resample_command", "command._resample_command")
            for asset_name, asset in getattr(self.env.scene, "_articulations", {}).items():
                self._wrap(asset, "reset", f"articulation.{asset_name}.reset")
                for name in ("write_root_pose_to_sim", "write_root_velocity_to_sim",
                             "write_root_state_to_sim", "write_joint_state_to_sim"):
                    self._wrap(asset, name, f"articulation.{asset_name}.{name}")

        def _state(self, obs):
            positions, quaternions, joints, velocities = self._actual()
            data = self.command.robot.data
            root = getattr(data, "root_state_w", None)
            if root is None:
                root = np.concatenate([_array(getattr(data, name))[0].reshape(-1) for name in
                                       ("root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_w")])
            else:
                root = _array(root)[0]
            return {
                "root": root, "body_positions": positions, "body_quaternions_wxyz": quaternions,
                "joint_positions": joints, "joint_velocities": velocities,
                "observation_history": {name: _array(obs[name]) for name in ("policy", "action", "critic")},
            }

        @staticmethod
        def _state_delta(left, right):
            values = []
            for name in ("root", "body_positions", "body_quaternions_wxyz", "joint_positions", "joint_velocities"):
                values.append(float(np.max(np.abs(left[name] - right[name]))))
            for name in ("policy", "action", "critic"):
                values.append(float(np.max(np.abs(left["observation_history"][name] - right["observation_history"][name]))))
            return max(values)

        def _assert_clean(self):
            calls = {name: count for name, count in report["postattach_calls"].items() if count}
            if calls:
                raise AssertionError(f"post-attach reset/resample/state write detected: {calls}")

        def _assert_deadline(self):
            if time.monotonic() >= DEADLINE:
                raise AssertionError(f"internal 165s deadline expired in {self._smoke_stage}")

        def _mode_mask(self):
            mask = _array(self.command.mode)
            expected = np.zeros((1, 14), dtype=np.float32)
            expected[0, list(MODE_INDICES[self.mode_name])] = 1.0
            if mask.shape != expected.shape or not np.array_equal(mask, expected):
                raise AssertionError(f"wrong {self.mode_name} command mask: {mask}")
            if (self.command.live_mode_name != self.mode_name or self.panel.mode_name != self.mode_name
                    or self.panel.mode_epoch != self.mode_epoch):
                raise AssertionError("command/controller/panel mode name or epoch diverged")
            return mask

        def _probe_observation(self, obs):
            import torch
            layout = _metadata(obs)
            if report.get("observation_layout") is None:
                report["observation_layout"] = layout
            elif layout != report["observation_layout"]:
                raise AssertionError("mode refresh changed observation keys/shapes/dtypes/devices")
            first = obs.clone()
            command_mask = self._mode_mask()
            if not np.array_equal(_array(obs["mode"]), command_mask):
                raise AssertionError("observation mode differs from the hardcoded command mask")
            actor_first = self._actor.get_actor_obs(first)
            if not isinstance(actor_first, tuple) or len(actor_first) < 2:
                raise AssertionError("expected transformer actor observation tuple")
            task_first = actor_first[1].detach().clone()
            mode = obs["mode"]
            expected_mode = mode.unsqueeze(-2).repeat(1, task_first.shape[-2], 1) if mode.dim() == 2 else mode
            if not torch.equal(task_first[..., -14:], expected_mode):
                raise AssertionError("actor observation does not contain the selected 14-bit mode")
            second = obs.clone()
            task, mapping = second["policy_task"], second["mode_mapping"]
            if mapping.dim() < task.dim():
                mapping = mapping.unsqueeze(-2).expand_as(task)
            inactive = mapping == 0
            if not bool(inactive.any().item()):
                raise AssertionError("mode mapping exposed no inactive task fields to probe")
            task[inactive] += 123.0
            task_second = self._actor.get_actor_obs(second)[1].detach()
            isolated = torch.equal(task_first, task_second)
            probe = {"mode": self.mode_name, "epoch": self.mode_epoch, "layout": layout,
                     "command_mask": command_mask, "observation_mode": _array(obs["mode"]),
                     "inactive_fields": int(inactive.sum().item()),
                     "masked_actor_task_equal": isolated,
                     "editor_rows_enabled": [all(field.enabled for field in (*position, *self.panel.angle_fields[row]))
                                             for row, position in enumerate(self.panel.position_fields)]}
            report.setdefault("mask_isolation_probes", []).append(probe)
            if not isolated:
                raise AssertionError("inactive task-field perturbation reached the actor observation")
            expected_rows = set(TARGET_ROWS[self.mode_name])
            if probe["editor_rows_enabled"] != [row in expected_rows for row in range(3)]:
                raise AssertionError("native editor row enablement does not match the active mode")

        async def _capture(self):
            path = REPORT_PATH.with_suffix(".png")
            report["capture"] = {"path": str(path), "status": "requested"}
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
            report["capture"] = {"path": str(path), "status": "captured" if path.is_file() else "missing",
                                 "exists": path.is_file(),
                                 "size_bytes": path.stat().st_size if path.is_file() else None}
            _write_report()

        def _goal(self, elapsed):
            phase = 2.0 * np.pi * elapsed / 2.0
            sine = float(np.sin(phase))
            positions, angles = self._mode_seed_positions.copy(), self._mode_seed_angles.copy()
            rows = TARGET_ROWS[self.mode_name]
            positions[list(rows), 0] += 0.02 * sine
            angles[list(rows), 2] += 2.0 * sine
            self._mode_record["commanded_x_delta_range"][0] = min(
                self._mode_record["commanded_x_delta_range"][0], 0.02 * sine)
            self._mode_record["commanded_x_delta_range"][1] = max(
                self._mode_record["commanded_x_delta_range"][1], 0.02 * sine)
            self._mode_record["commanded_yaw_delta_degrees_range"][0] = min(
                self._mode_record["commanded_yaw_delta_degrees_range"][0], 2.0 * sine)
            self._mode_record["commanded_yaw_delta_degrees_range"][1] = max(
                self._mode_record["commanded_yaw_delta_degrees_range"][1], 2.0 * sine)
            self.panel.apply(positions, angles)

        def _trajectory(self):
            positions, quaternions, _, _ = self._actual()
            actual = positions[[0, 10, 13]]
            actual_quat = quaternions[[0, 10, 13]]
            reference = self.provider.sample([0])
            ref_pos = _array(reference["body_pos"])[0, [0, 10, 13]]
            ref_quat = _array(reference["body_quat"])[0, [0, 10, 13]]
            goals, goal_quat = self.provider.goal_positions, self.provider.goal_quaternions
            rows = list(TARGET_ROWS[self.mode_name])
            inactive = sorted(set(range(3)) - set(rows))
            entry = {
                "mode": self.mode_name, "epoch": self.mode_epoch, "physics_s": self._physics_s,
                "sampled": True,
                "active_rows": rows, "actual_positions": actual[rows], "reference_positions": ref_pos[rows],
                "goal_positions": goals[rows], "actual_goal_position_errors": np.linalg.norm(actual[rows] - goals[rows], axis=-1),
                "actual_reference_position_errors": np.linalg.norm(actual[rows] - ref_pos[rows], axis=-1),
                "actual_goal_orientation_errors": self._orientation_errors(actual_quat[rows], goal_quat[rows]),
                "actual_reference_orientation_errors": self._orientation_errors(actual_quat[rows], ref_quat[rows]),
                "inactive_goal_positions_unchanged": not inactive or np.allclose(
                    goals[inactive], self._mode_seed_positions[inactive], atol=1e-9, rtol=0.0),
                "inactive_goal_orientations_unchanged": not inactive or np.allclose(
                    np.abs(np.sum(goal_quat[inactive] * self._mode_seed_quaternions[inactive], axis=-1)),
                    1.0, atol=1e-9, rtol=0.0),
            }
            if not (entry["inactive_goal_positions_unchanged"] and entry["inactive_goal_orientations_unchanged"]):
                raise AssertionError("inactive goal rows changed")
            report["trajectory"].append(entry)

        def _request_mode(self):
            self._mode_index += 1
            target = ("Pelvis-1", "UMI-2", "VR-3")[self._mode_index]
            self._switch_target = target
            self._switch_before = self._state(self._last_obs)
            self._switch_physics = self._physics_s
            self._switch_old_mode = self.mode_name
            self._switch_old_epoch = self.mode_epoch
            self._switch_before_mask = _array(self.command.mode)
            layout = _metadata(self._last_obs)
            if report.get("observation_layout") is None:
                report["observation_layout"] = layout
            elif layout != report["observation_layout"]:
                raise AssertionError("pre-switch observation layout changed")
            self._switch_record = {
                "from_mode": self._switch_old_mode, "to_mode": target,
                "epoch_before": self._switch_old_epoch, "physics_s_before": self._physics_s,
                "physics_callbacks_before": self._physics_callbacks, "command_mask_before": self._switch_before_mask,
                "state_and_history_before": self._switch_before,
            }
            report["switches"].append(self._switch_record)
            self.panel.request_mode(target)
            self._smoke_stage, self._stage_wall = "mode_requested", time.monotonic()
            _stage("mode_requested", target=target, physics_s=self._physics_s)

        def before_step(self, obs):
            global completed
            self._last_obs = obs
            self._assert_deadline()
            if self._smoke_stage == "wait_heartbeat" and self.panel.heartbeat_stamp is not None:
                self.panel.request_enable()
                self._smoke_stage = "enable_requested"
                _stage("enable_requested")
            candidate = super().before_step(obs)
            now = time.monotonic()
            if self.session.state == self.session.PAUSED_FAULT:
                raise AssertionError(f"online session faulted in {self._smoke_stage}: {self.session.reason}")
            if self._smoke_stage == "enable_requested" and self.session.state == self.session.ACTIVE:
                self._initial_physics = self._physics_s
                self._smoke_stage = "initial_active"
                _stage("initial_active", physics_s=self._physics_s)
            elif self._smoke_stage == "mode_requested" and self.session.state == self.session.PAUSED_USER:
                if self.mode_name != self._switch_old_mode:
                    raise AssertionError("mode changed before controller-confirmed PAUSED_USER")
                self._freeze_state, self._freeze_physics = self._state(obs), self._physics_s
                request_to_pause_delta = self._state_delta(self._switch_before, self._freeze_state)
                physics_delta = self._physics_s - self._switch_physics
                callbacks_delta = self._physics_callbacks - self._switch_record["physics_callbacks_before"]
                self._switch_record.update({
                    "pause_confirmed_physics_s": self._physics_s,
                    "pause_confirmed_physics_callbacks": self._physics_callbacks,
                    "request_to_pause_physics_delta_s": physics_delta,
                    "request_to_pause_callbacks_delta": callbacks_delta,
                    "request_to_pause_state_history_max_delta": request_to_pause_delta,
                })
                if abs(physics_delta) > 1e-9 or callbacks_delta != 0 or request_to_pause_delta > 1e-7:
                    raise AssertionError("request_mode allowed physics/state/history before Pause confirmation")
                self._freeze_wall = now
                self._smoke_stage = "mode_freeze"
                _stage("mode_pause_confirmed", target=self._switch_target, physics_s=self._physics_s)
            elif self._smoke_stage == "mode_freeze":
                if self.session.state != self.session.PAUSED_USER:
                    raise AssertionError("mode switch left PAUSED_USER")
                switched = self.mode_name == self._switch_target
                if switched and self.mode_epoch != self._switch_old_epoch + 1:
                    raise AssertionError("mode epoch did not advance exactly once")
                if switched and now - self._freeze_wall >= 0.5:
                    frozen = self._state(obs)
                    delta = self._state_delta(self._freeze_state, frozen)
                    if abs(self._physics_s - self._freeze_physics) > 1e-9 or delta > 1e-7:
                        raise AssertionError("mode switch did not freeze physics/state/history for 0.5s")
                    if np.array_equal(self._switch_before_mask, self._mode_mask()):
                        raise AssertionError("mode switch did not change the command mask")
                    self._switch_record.update({
                        "epoch_after": self.mode_epoch, "physics_s_after_freeze": self._physics_s,
                        "physics_callbacks_after_freeze": self._physics_callbacks,
                        "command_mask_after": self._mode_mask(), "state_and_history_after_freeze": frozen,
                        "freeze_wall_s": now - self._freeze_wall,
                        "freeze_physics_delta_s": self._physics_s - self._freeze_physics,
                        "freeze_state_history_max_delta": delta,
                    })
                    self._assert_clean()
                    self._probe_observation(obs)
                    if self._mode_index == 2 and self._capture_task is None:
                        self._capture_task = asyncio.ensure_future(self._capture())
                    if self._capture_task is not None and not self._capture_task.done():
                        return candidate
                    if self._capture_task is not None:
                        self._capture_task.result()
                        if not report.get("capture", {}).get("exists"):
                            raise AssertionError("final mode-freeze UI capture is missing")
                    actual = self._actual()
                    self._mode_seed_positions = actual[0][[0, 10, 13]].copy()
                    self._mode_seed_quaternions = actual[1][[0, 10, 13]].copy()
                    self._mode_seed_quaternions /= np.linalg.norm(
                        self._mode_seed_quaternions, axis=-1, keepdims=True
                    )
                    self._mode_seed_angles = _xyz_degrees_from_wxyz(self._mode_seed_quaternions)
                    self._mode_record = {
                        "mode": self.mode_name, "epoch": self.mode_epoch,
                        "mask": self._mode_mask(), "freeze_wall_s": now - self._freeze_wall,
                        "freeze_physics_delta_s": self._physics_s - self._freeze_physics,
                        "freeze_state_history_max_delta": delta,
                        "commanded_x_delta_range": [0.0, 0.0],
                        "commanded_yaw_delta_degrees_range": [0.0, 0.0],
                    }
                    report["modes"].append(self._mode_record)
                    self.panel.request_resume()
                    self._smoke_stage = "resume_requested"
                    _stage("resume_requested", mode=self.mode_name, epoch=self.mode_epoch)
            elif self._smoke_stage == "resume_requested" and self.session.state == self.session.ACTIVE:
                if not self._requires_goal or candidate is not None:
                    raise AssertionError("Resume did not remain frozen awaiting a fresh Apply")
                self._resume_state, self._resume_physics = self._state(obs), self._physics_s
                self._resume_wall = now
                self._smoke_stage = "await_fresh_apply"
                _stage("await_fresh_apply", mode=self.mode_name, physics_s=self._physics_s)
            elif self._smoke_stage == "await_fresh_apply" and now - self._resume_wall >= 0.5:
                frozen = self._state(obs)
                delta = self._state_delta(self._resume_state, frozen)
                if candidate is not None or not self._requires_goal:
                    raise AssertionError("fresh-Apply gate opened early")
                if abs(self._physics_s - self._resume_physics) > 1e-9 or delta > 1e-7:
                    raise AssertionError("Resume advanced physics/state/history before fresh Apply")
                self._mode_record["await_apply_wall_s"] = now - self._resume_wall
                self._mode_record["await_apply_physics_delta_s"] = self._physics_s - self._resume_physics
                self._mode_record["await_apply_state_history_max_delta"] = delta
                self._goal(0.0)
                self._smoke_stage = "apply_requested"
                _stage("fresh_apply_requested", mode=self.mode_name)
            elif self._smoke_stage == "apply_requested" and candidate is not None:
                if self.session.state != self.session.ACTIVE or self._requires_goal:
                    raise AssertionError("valid fresh Apply did not restore ACTIVE")
                self._mode_physics_start = self._physics_s
                self._mode_env_steps_start = self._env_steps
                self._mode_sequences = set()
                self._mode_rejected_start = self.provider.rejected_count
                self._last_trajectory_physics = -1.0
                self._smoke_stage = "mode_active"
                _stage("mode_active", mode=self.mode_name, epoch=self.mode_epoch, physics_s=self._physics_s)
            return candidate

        def validate_actions(self, actions):
            self._policy_calls += 1
            report["real_policy_evaluations"] = self._policy_calls
            return super().validate_actions(actions)

        def after_step(self, obs):
            global completed
            self._env_steps += 1
            report["real_env_steps"] = self._env_steps
            super().after_step(obs)
            if self.session.state == self.session.PAUSED_FAULT:
                raise AssertionError(f"online session faulted in {self._smoke_stage}: {self.session.reason}")
            self._last_obs = obs
            self._assert_deadline()
            if self._smoke_stage == "initial_active" and self._physics_s - self._initial_physics >= 0.2:
                report["initial_active_physics_s"] = self._physics_s - self._initial_physics
                self._request_mode()
            elif self._smoke_stage == "mode_active":
                elapsed = self._physics_s - self._mode_physics_start
                if self.provider.last_sequence is not None:
                    self._mode_sequences.add(int(self.provider.last_sequence))
                self._goal(elapsed)
                if elapsed - self._last_trajectory_physics >= 0.5:
                    self._trajectory()
                    self._last_trajectory_physics = elapsed
                if elapsed >= 5.0:
                    self._trajectory()
                    self._mode_record["physics_s"] = elapsed
                    self._mode_record["env_steps"] = self._env_steps - self._mode_env_steps_start
                    self._mode_record["accepted_sequence_distinct_count"] = len(self._mode_sequences)
                    self._mode_record["accepted_sequence_first"] = min(self._mode_sequences, default=None)
                    self._mode_record["accepted_sequence_last"] = max(self._mode_sequences, default=None)
                    self._mode_record["rejected_count_before"] = self._mode_rejected_start
                    self._mode_record["rejected_count_after"] = self.provider.rejected_count
                    if len(self._mode_sequences) < 2:
                        raise AssertionError(f"{self.mode_name} accepted fewer than two distinct goals")
                    if self.provider.rejected_count != self._mode_rejected_start:
                        raise AssertionError(f"{self.mode_name} rejected a supposedly valid sine goal")
                    for key, amplitude in (("commanded_x_delta_range", 0.019),
                                           ("commanded_yaw_delta_degrees_range", 1.9)):
                        low, high = self._mode_record[key]
                        if low > -amplitude or high < amplitude:
                            raise AssertionError(f"{self.mode_name} did not exercise both sine extrema: {key}={low, high}")
                    self._assert_clean()
                    if self._mode_index < 2:
                        self._request_mode()
                    else:
                        report["result"] = "PASS"
                        report["exit_requested_monotonic_s"] = time.monotonic()
                        report["message"] = "three real-policy modes each completed 5s PhysX with guarded switches"
                        _stage("exit_requested")
                        completed = True
                        self.panel.request_close()

        def close(self):
            if self._closed_for_smoke:
                return
            self._closed_for_smoke = True
            subscription, self._physics_subscription = getattr(self, "_physics_subscription", None), None
            if subscription is not None:
                unsubscribe = getattr(subscription, "unsubscribe", None)
                if unsubscribe is not None:
                    unsubscribe()
            for owner, name, original in reversed(getattr(self, "_restorers", [])):
                setattr(owner, name, original)
            self._restorers = []
            super().close()

    controller_class = ModeSmokeController
    online_control.OnlinePlaybackController = controller_class
    _stage("player_main")
    player["main"]()
    if not completed:
        raise AssertionError("player exited before all three mode stages completed")
    report["result"] = "PASS"
    _write_report()
    print(f"[BFM ONLINE MODES SMOKE] PASS report={REPORT_PATH}", flush=True)
except BaseException as error:
    report["result"] = "FAIL"
    report.setdefault("exit_requested_monotonic_s", time.monotonic())
    report["error_type"], report["error"] = type(error).__name__, str(error)
    report["traceback"] = traceback.format_exc()
    try:
        _write_report()
    except BaseException:
        traceback.print_exc()
    traceback.print_exc()
    print(f"[BFM ONLINE MODES SMOKE] FAIL report={REPORT_PATH}", flush=True)
    raise
finally:
    if runner_class is not None and original_get_policy is not None:
        runner_class.get_inference_policy = original_get_policy
    if original_controller is not None:
        online_control.OnlinePlaybackController = original_controller
    if app is not None:
        app.close(exit_code=0 if completed and sys.exc_info()[0] is None else 1)
