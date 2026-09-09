"""Behavioral tests for online player validation and supervision."""

from __future__ import annotations

import argparse

import numpy as np
import pytest
import sys
from pathlib import Path
from types import SimpleNamespace
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/pretrain/rsl_rl"))
from online_control import OnlinePlaybackController, validate_online_args
from online_ui import OnlineTargetPanel
from scaletrack.utils.live_reference import LiveReferenceProvider
from scaletrack.utils.online_session import OnlineSession


def _provider():
    names = ("pelvis", "b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8", "b9", "left_wrist_yaw_link", "b11", "b12", "right_wrist_yaw_link")
    positions = np.zeros((14, 3)); positions[0, 2] = 0.8; positions[10] = [0.2, 0.1, 0.8]; positions[13] = [0.2, -0.1, 0.8]
    quaternions = np.tile([1.0, 0.0, 0.0, 0.0], (14, 1))
    return LiveReferenceProvider(names, positions, quaternions, np.zeros(29))


def test_online_argument_rejection_happens_on_namespace_without_app_launch():
    """Break caught: an incompatible online command reaches AppLauncher."""
    args = argparse.Namespace(online_targets=True, motion_menu=True, num_envs=2,
                              task="G1-BFM-Transformer-Tracking", mode_index=None,
                              local_tracking=False, video=False, headless=False, visualizer=["kit"])
    with pytest.raises(ValueError, match="num_envs 1"):
        validate_online_args(args)


@pytest.mark.parametrize("mode", [None, 0, 1, 2])
def test_online_args_keep_three_supported_masks(mode):
    args = argparse.Namespace(online_targets=True, motion_menu=True, num_envs=1,
                              task="G1-BFM-Transformer-Tracking", mode_index=mode,
                              local_tracking=False, video=False, headless=False, visualizer=None)
    validate_online_args(args)
    assert args.mode_index == (2 if mode is None else mode)
    assert args.visualizer == ["kit"]


@pytest.mark.parametrize("mode", [3, 7, True, 1.0, "1"])
def test_online_args_reject_incompatible_mask_before_app(mode):
    args = argparse.Namespace(online_targets=True, motion_menu=True, num_envs=1,
                              task="G1-BFM-Transformer-Tracking", mode_index=mode)
    with pytest.raises(ValueError, match="mode_index"):
        validate_online_args(args)


def test_controller_commits_goal_without_reset_or_history_and_reports_actual_pose():
    """Break caught: UI commit mutates robot/reset/history or telemetry aliases its goal."""
    class Data:
        body_pos_w = np.zeros((1, 14, 3)); body_quat_w = np.tile([1., 0., 0., 0.], (1, 14, 1)); joint_pos = np.zeros((1, 29)); joint_vel = np.zeros((1, 29))
    Data.body_pos_w[0, :, :2] = [2., 3.]; Data.body_pos_w[0, 0, 2] = .8; Data.body_pos_w[0, 10] = [2.2, 3.1, .8]; Data.body_pos_w[0, 13] = [2.2, 2.9, .8]
    class Robot: data = Data()
    class Command: robot = Robot(); body_indexes = [*range(14)]; live_reference = _provider()
    class Obs:
        calls = []
        def compute_group(self, name, update_history=False):
            self.calls.append((name, update_history))
            return np.array([Command.live_reference.sample([1])["body_pos"][0, 0, 0]]) if name == "policy_task" else np.zeros(1)
    class Env:
        command_manager = type("Commands", (), {"get_term": lambda self, _: Command()})()
        observation_manager = Obs(); scene = type("Scene", (), {"env_origins": np.array([[2., 3., 0.]])})()
        sim = type("Sim", (), {"pause": lambda self: None, "play": lambda self: None})()
    class Panel:
        heartbeat_stamp = 0.0; close_requested = False
        sequence = 0
        def consume_request(self):
            if self.sequence == 100:
                return None
            self.sequence += 1
            return {"kind": "goal", "positions": np.array([[.1, 0., .8], [.2,.1,.8], [.2,-.1,.8]]), "quaternions": np.tile([1.,0.,0.,0.], (3,1)), "sequence": self.sequence, "stamp": 0.0, "body_names": ("pelvis", "left_wrist_yaw_link", "right_wrist_yaw_link"), "frame": "env_local", "quaternion_order": "wxyz"}
        def update_status(self, status): self.status = status
        def close(self): pass
    panel = Panel(); session = OnlineSession(Command.live_reference); session.heartbeat(0.0, 0.0); session.activate(0.0, Data.body_pos_w[0], Data.body_quat_w[0], Data.joint_pos[0])
    controller = OnlinePlaybackController(Env(), Command(), panel, session=session, now=lambda: 0.0)

    observation = {"policy": np.zeros(1)}
    root_before, joints_before = Data.body_pos_w.copy(), Data.joint_pos.copy()
    for _ in range(100):
        assert controller.before_step(observation) is observation

    assert Command.live_reference.last_sequence == 100
    assert session.state == session.ACTIVE
    assert observation["policy_task"][0] > 0.0
    assert np.array_equal(Data.body_pos_w, root_before) and np.array_equal(Data.joint_pos, joints_before)
    assert Env.observation_manager.calls == [("policy_task", False), ("critic_task", False), ("mode", False), ("mode_mapping", False)] * 100
    assert np.allclose(panel.status["actual_positions"][0], [0., 0., .8])
    assert not np.shares_memory(panel.status["actual_positions"], Command.live_reference.goal_positions)
    assert panel.status["position_errors"][0] > 0.0


def test_actual_unwraps_proxy_arrays_and_torch_body_indices_before_numpy_indexing():
    """Break caught: CUDA/scalar body indexes leak into NumPy indexing at the articulation boundary."""
    provider = _provider()
    class Proxy:
        def __init__(self, value): self.torch = value
    class Data:
        body_pos_w = Proxy(torch.tensor([[[0., 0., .8]] * 14])); body_quat_w = Proxy(torch.tensor([[[1., 0., 0., 0.]] * 14]))
        joint_pos = Proxy(torch.zeros((1, 29))); joint_vel = Proxy(torch.zeros((1, 29)))
    class Command: robot = type("Robot", (), {"data": Data()})(); body_indexes = [torch.tensor(index) for index in range(14)]; live_reference = provider
    panel = type("Panel", (), {"close_requested": False, "heartbeat_stamp": None, "consume_request": lambda self: None, "update_status": lambda self, status: None, "close": lambda self: None})()
    env = type("Env", (), {"scene": type("Scene", (), {"env_origins": Proxy(torch.zeros((1, 3)))})(), "sim": type("Sim", (), {"pause": lambda self: None, "play": lambda self: None})(), "observation_manager": type("Obs", (), {"compute_group": lambda *args, **kwargs: np.zeros(1)})()})()
    controller = OnlinePlaybackController(env, Command(), panel)
    positions, quaternions, joints, velocities = controller._actual()
    assert positions.shape == (14, 3) and quaternions.shape == (14, 4) and joints.shape == velocities.shape == (29,)


def test_enable_pause_resume_retains_observation_and_refreshes_after_each_reseed():
    """Break caught: paused playback loses its old observation or resumes on stale task references."""
    provider = _provider()
    class Data:
        body_pos_w = np.zeros((1, 14, 3)); body_quat_w = np.tile([1., 0., 0., 0.], (1, 14, 1)); joint_pos = np.zeros((1, 29)); joint_vel = np.zeros((1, 29))
    Data.body_pos_w[0,0,2]=.8; Data.body_pos_w[0,10]=[.2,.1,.8]; Data.body_pos_w[0,13]=[.2,-.1,.8]
    class Command: robot=type("Robot", (), {"data": Data()})(); body_indexes=list(range(14)); live_reference=provider
    clock = [0.0]
    class Panel:
        close_requested = False
        requests = iter(({"kind": "enable"}, {"kind": "pause"}, {"kind": "resume"},
                         {"kind": "goal", "positions": np.array([[.1, 0., .8], [.2,.1,.8], [.2,-.1,.8]]),
                          "quaternions": np.tile([1.,0.,0.,0.], (3,1)), "sequence": 1, "stamp": .2,
                          "body_names": ("pelvis", "left_wrist_yaw_link", "right_wrist_yaw_link"),
                          "frame": "env_local", "quaternion_order": "wxyz"}))
        @property
        def heartbeat_stamp(self): return clock[0]
        def consume_request(self): return next(self.requests, None)
        def discard_goal(self): self.discarded=True
        def reset_editors(self, positions, quaternions): self.reseeded=(positions, quaternions)
        def update_status(self, status): self.status=status
        def close(self): self.closed=True
    panel=Panel()
    class Sim:
        def pause(self): clock[0] += .1
        def play(self): clock[0] += .1
    class Obs:
        calls=[]
        def compute_group(self, name, update_history=False): self.calls.append((name, update_history)); return np.zeros(1)
    env=type("Env", (), {"scene":type("S", (), {"env_origins":np.zeros((1,3))})(), "sim":Sim(), "observation_manager":Obs()})()
    controller=OnlinePlaybackController(env, Command(), panel, now=lambda: clock[0])
    observation={"policy":np.zeros(1)}
    assert controller.before_step(observation) is observation
    assert controller.before_step(observation) is None
    assert controller.before_step(observation) is None
    assert controller.before_step(observation) is observation
    assert panel.discarded and hasattr(panel, "reseeded")
    assert Obs.calls == [(name, False) for _ in range(3) for name in ("policy_task", "critic_task", "mode", "mode_mapping")]


def test_stale_heartbeat_and_nonfinite_actions_latch_pause_and_close_cleans_panel():
    """Break caught: stale producer/fault still allows physics, or close leaks the panel."""
    provider = _provider()
    class Data:
        body_pos_w = np.zeros((1, 14, 3)); body_quat_w = np.tile([1.,0.,0.,0.], (1,14,1)); joint_pos = np.zeros((1,29)); joint_vel = np.zeros((1,29))
    Data.body_pos_w[0,0,2]=.8; Data.body_pos_w[0,10]=[.2,.1,.8]; Data.body_pos_w[0,13]=[.2,-.1,.8]
    class Command: robot=type("Robot", (), {"data": Data()})(); body_indexes=list(range(14)); live_reference=provider
    class Panel:
        heartbeat_stamp = 0.; close_requested = False; closed=False
        def consume_request(self): return None
        def update_status(self, status): pass
        def close(self): self.closed=True
    panel=Panel(); session=OnlineSession(provider); session.heartbeat(0., 0.); session.activate(0., Data.body_pos_w[0], Data.body_quat_w[0], Data.joint_pos[0])
    env=type("Env", (), {"scene":type("S", (), {"env_origins":np.zeros((1,3))})(), "sim":type("Sim", (), {"pause":lambda self:None, "play":lambda self:None})(), "observation_manager":type("O", (), {"compute_group":lambda *args, **kwargs: None})()})()
    controller=OnlinePlaybackController(env, Command(), panel, session=session, now=lambda: 1.)
    assert controller.before_step({}) is None
    assert session.state == session.PAUSED_FAULT
    assert not controller.validate_actions(np.array([np.nan]))
    controller.close()
    assert panel.closed and session.state == session.CLOSED


@pytest.fixture
def native_boundary(monkeypatch):
    """Faithful pause/play boundary: timeline state, event pumps, real intent queues."""
    clock = [0.0]
    provider = _provider()
    seed = provider.sample([0])
    data = SimpleNamespace(body_pos_w=seed["body_pos"].copy(), body_quat_w=seed["body_quat"].copy(),
                           joint_pos=np.zeros((1, 29)), joint_vel=np.zeros((1, 29)),
                           body_link_lin_vel_w=np.zeros((1, 14, 3)), body_link_ang_vel_w=np.zeros((1, 14, 3)))
    panel = OnlineTargetPanel(now=lambda: clock[0], subscribe_updates=lambda callback: None)
    panel._on_update()
    command = SimpleNamespace(robot=SimpleNamespace(data=data), body_indexes=list(range(14)), live_reference=provider,
                              live_mode_name="VR-3")
    from scaletrack.utils.online_modes import online_body_indices

    def set_live_mode(mode_name):
        command.live_mode_name = mode_name
        command._mode = np.zeros((1, 14))
        command._mode[0, list(online_body_indices(mode_name))] = 1.

    command.set_live_mode = set_live_mode
    set_live_mode("VR-3")
    calls = []

    class Sim:
        playing = False
        on_play = None
        on_pause = None
        auto_physics = True
        implicit_steps = 0

        def get_setting(self, key):
            assert key == "/app/player/playSimulations"
            return self.auto_physics

        def set_setting(self, key, value):
            assert key == "/app/player/playSimulations"
            self.auto_physics = value

        def pause(self):
            self.playing = False
            clock[0] += .01
            panel._on_update()
            if self.on_pause is not None:
                self.on_pause()
            if self.playing and self.auto_physics:
                self.implicit_steps += 1

        def play(self):
            self.playing = True
            clock[0] += .01
            panel._on_update()
            if self.on_play is not None:
                self.on_play()
            if self.playing and self.auto_physics:
                self.implicit_steps += 1

    def compute_group(name, update_history):
        calls.append((name, update_history))
        if name == "mode":
            return command._mode.copy()
        return provider.sample([1])["body_pos"].copy() if name == "policy_task" else np.zeros(1)

    sim = Sim()
    env = SimpleNamespace(sim=sim, step_dt=.02, scene=SimpleNamespace(env_origins=np.zeros((1, 3))),
                          observation_manager=SimpleNamespace(compute_group=compute_group))
    controller = OnlinePlaybackController(env, command, panel, now=lambda: clock[0])
    monkeypatch.setattr(controller, "_native_playing", lambda: sim.playing)
    monkeypatch.setattr(controller, "_pause_native", lambda: setattr(sim, "playing", False))
    observation = {"policy": np.zeros(4)}
    yield SimpleNamespace(controller=controller, panel=panel, sim=sim, clock=clock, provider=provider,
                          observation=observation, data=data, calls=calls)
    controller.close()


def _enable_pause_resume(boundary):
    b = boundary
    b.panel.request_enable()
    assert b.controller.before_step(b.observation) is b.observation
    assert b.sim.playing
    b.panel.request_pause()
    assert b.controller.before_step(b.observation) is None
    b.panel.request_resume()
    assert b.controller.before_step(b.observation) is None
    assert b.panel.status["awaiting_goal"] and not b.sim.playing


def test_resume_waits_through_invalid_apply_then_plays_on_fresh_valid_apply(native_boundary):
    b = native_boundary
    original_positions, original_joints = b.data.body_pos_w.copy(), b.data.joint_pos.copy()
    _enable_pause_resume(b)
    sequence = b.provider.last_sequence
    invalid = b.provider.goal_positions.copy()
    invalid[0, 0] += 10.
    b.panel.apply(invalid, np.zeros((3, 3)))
    assert b.controller.before_step(b.observation) is None
    assert b.provider.last_sequence == sequence
    assert b.provider.rejected_count == 1
    assert b.panel.status["awaiting_goal"] and not b.sim.playing
    assert b.controller.before_step(b.observation) is None
    valid = b.provider.goal_positions.copy()
    valid[0, 0] += .02
    b.panel.apply(valid, np.zeros((3, 3)))
    assert b.controller.before_step(b.observation) is b.observation
    assert not b.panel.status["awaiting_goal"] and b.sim.playing
    assert b.observation["policy_task"][0, 0, 0] > 0.
    np.testing.assert_array_equal(b.data.body_pos_w, original_positions)
    np.testing.assert_array_equal(b.data.joint_pos, original_joints)
    assert all(not history for _, history in b.calls)


@pytest.mark.parametrize("intent", ["pause", "close"])
def test_safety_intent_during_play_pump_prevents_next_step(native_boundary, intent):
    b = native_boundary
    b.sim.on_play = getattr(b.panel, f"request_{intent}")
    b.panel.request_enable()
    assert b.controller.before_step(b.observation) is None
    assert not b.sim.playing
    assert b.controller.session.state == ("CLOSED" if intent == "close" else "PAUSED_USER")


def test_explicit_pause_cancels_pending_resume_permission(native_boundary):
    b = native_boundary
    _enable_pause_resume(b)
    b.panel.request_pause()
    assert b.controller.before_step(b.observation) is None
    b.panel.apply(b.provider.goal_positions, np.zeros((3, 3)))
    assert b.controller.before_step(b.observation) is None
    assert not b.sim.playing and b.controller.session.state == "PAUSED_USER"


def test_native_pause_and_native_play_cannot_override_safety_latch(native_boundary):
    b = native_boundary
    b.panel.request_enable()
    assert b.controller.before_step(b.observation) is b.observation
    b.sim.playing = False
    assert b.controller.before_step(b.observation) is None
    assert b.controller.session.state == "PAUSED_USER"
    b.controller.session.fault("test stale producer")
    b.sim.playing = True
    b.panel.apply(b.provider.goal_positions, np.zeros((3, 3)))
    assert b.controller.before_step(b.observation) is None
    assert not b.sim.playing and b.controller.session.state == "PAUSED_FAULT"


def test_after_step_reads_new_heartbeat_before_liveness_check(native_boundary):
    b = native_boundary
    b.panel.request_enable()
    assert b.controller.before_step(b.observation) is b.observation
    b.clock[0] += .6
    b.panel._on_update()  # Real env.step pumps Kit and publishes a new heartbeat.
    b.controller.after_step(b.observation)
    assert b.controller.session.state == "ACTIVE"


def test_liveness_checked_again_after_slow_policy_before_any_physics(native_boundary):
    b = native_boundary
    b.panel.request_enable()
    assert b.controller.before_step(b.observation) is b.observation
    b.clock[0] += .6  # No GUI heartbeat while policy is computing.
    assert not b.controller.validate_actions(np.zeros((1, 29)))
    assert not b.sim.playing and b.controller.session.state == "PAUSED_FAULT"


def test_native_play_arriving_inside_pause_pump_cannot_step_or_remain_playing(native_boundary):
    b = native_boundary
    b.controller.session.fault("test stale producer")
    b.sim.on_pause = lambda: setattr(b.sim, "playing", True)
    assert b.controller.before_step(b.observation) is None
    assert not b.sim.playing and b.sim.implicit_steps == 0
    assert b.sim.auto_physics is True  # Restore pre-existing app setting.


def test_enable_pump_does_not_implicitly_step_physics(native_boundary):
    b = native_boundary
    b.panel.request_enable()
    assert b.controller.before_step(b.observation) is b.observation
    assert b.sim.playing and b.sim.implicit_steps == 0


def _switch_mode(boundary, name):
    b = boundary
    b.panel.request_mode(name)
    assert b.controller.before_step(b.observation) is None  # Pause has priority.
    assert b.controller.before_step(b.observation) is None  # Now confirm the mode.
    assert b.controller.mode_name == name
    assert b.controller.session.state == "PAUSED_USER" and not b.sim.playing


def test_switching_sparse_modes_preserves_robot_and_requires_explicit_recovery(native_boundary):
    b = native_boundary
    b.panel.request_enable()
    assert b.controller.before_step(b.observation) is b.observation
    actual_before = b.data.body_pos_w.copy()
    joints_before = b.data.joint_pos.copy()
    _switch_mode(b, "Pelvis-1")
    assert b.observation["mode"].sum() == 1
    assert b.controller.mode_epoch == 1
    b.panel.apply(b.provider.goal_positions, np.zeros((3, 3)))
    assert b.controller.before_step(b.observation) is None  # Apply alone cannot resume.
    b.panel.request_resume()
    assert b.controller.before_step(b.observation) is None
    assert b.panel.status["awaiting_goal"]
    b.panel.apply(b.provider.goal_positions, np.zeros((3, 3)))
    assert b.controller.before_step(b.observation) is b.observation
    _switch_mode(b, "UMI-2")
    assert b.observation["mode"].sum() == 2
    assert b.panel.status["active_body_names"] == ("left_wrist_yaw_link", "right_wrist_yaw_link")
    np.testing.assert_array_equal(b.data.body_pos_w, actual_before)
    np.testing.assert_array_equal(b.data.joint_pos, joints_before)
    assert all(not history for _, history in b.calls)


def test_mode_epoch_rejects_old_goal_even_after_switching_back(native_boundary):
    b = native_boundary
    b.panel.request_enable()
    assert b.controller.before_step(b.observation) is b.observation
    b.panel.apply(b.provider.goal_positions, np.zeros((3, 3)))
    old_goal = b.panel.consume_request()
    _switch_mode(b, "UMI-2")
    _switch_mode(b, "VR-3")
    b.panel.request_resume()
    assert b.controller.before_step(b.observation) is None
    before = b.provider.last_sequence
    b.panel._goal_pending = old_goal
    assert b.controller.before_step(b.observation) is None
    assert b.provider.last_sequence == before
    assert b.controller.mode_rejected_count == 1 and b.panel.status["awaiting_goal"]
    assert not b.sim.playing


def test_changing_modes_cannot_clear_fault_or_override_close(native_boundary):
    b = native_boundary
    b.controller.session.fault("stale producer")
    b.panel.request_mode("UMI-2")
    assert b.controller.before_step(b.observation) is None
    assert b.controller.before_step(b.observation) is None
    assert b.controller.session.state == "PAUSED_FAULT" and b.controller.mode_name == "VR-3"
    b.panel.request_mode("Pelvis-1")
    b.panel.request_close()
    assert b.controller.before_step(b.observation) is None
    assert b.controller.session.state == "CLOSED"


@pytest.mark.parametrize("mode_name,active_count", [("Pelvis-1", 1), ("UMI-2", 2)])
def test_mode_can_be_selected_before_initial_enable(native_boundary, mode_name, active_count):
    b = native_boundary
    before = b.data.body_pos_w.copy()
    b.panel.request_mode(mode_name)
    assert b.controller.before_step(b.observation) is None
    assert b.controller.before_step(b.observation) is None
    assert b.controller.session.state == "READY"
    assert b.controller.mode_name == b.panel.mode_name == mode_name
    assert b.observation["mode"].sum() == active_count
    assert not b.sim.playing and b.sim.implicit_steps == 0
    b.panel.request_enable()
    assert b.controller.before_step(b.observation) is b.observation
    np.testing.assert_array_equal(b.data.body_pos_w, before)


def test_mode_request_inside_enable_pump_stops_before_physics(native_boundary):
    b = native_boundary
    b.sim.on_play = lambda: b.panel.request_mode("UMI-2")
    b.panel.request_enable()
    assert b.controller.before_step(b.observation) is None
    assert not b.sim.playing and b.sim.implicit_steps == 0
    assert b.controller.mode_name == "VR-3"  # Not confirmed until next boundary.
    assert b.controller.before_step(b.observation) is None
    assert b.controller.mode_name == "UMI-2" and b.controller.session.state == "PAUSED_USER"


def test_mode_request_during_policy_inference_cancels_action_permission(native_boundary):
    b = native_boundary
    b.panel.request_enable()
    assert b.controller.before_step(b.observation) is b.observation
    b.panel.request_mode("Pelvis-1")
    assert not b.controller.validate_actions(np.zeros((1, 29)))
    assert not b.sim.playing and b.controller.session.state == "PAUSED_USER"
    assert b.controller.before_step(b.observation) is None
    assert b.controller.mode_name == "Pelvis-1"


def test_mid_switch_failure_closes_instead_of_running_partial_mode(native_boundary, monkeypatch):
    b = native_boundary
    b.panel.request_enable()
    assert b.controller.before_step(b.observation) is b.observation
    before = b.data.body_pos_w.copy()
    b.panel.request_mode("UMI-2")
    assert b.controller.before_step(b.observation) is None

    def failed_confirmation(*_args):
        raise RuntimeError("injected mode UI failure")

    monkeypatch.setattr(b.panel, "set_mode", failed_confirmation)
    with pytest.raises(RuntimeError, match="injected mode UI failure"):
        b.controller.before_step(b.observation)
    assert b.controller.session.state == "CLOSED" and b.panel.close_requested
    assert not b.sim.playing and b.sim.implicit_steps == 0
    np.testing.assert_array_equal(b.data.body_pos_w, before)


def _waypoint_ready(b):
    b.panel.request_mode("Pelvis-1")
    assert b.controller.before_step(b.observation) is None
    assert b.controller.before_step(b.observation) is None
    b.panel.request_enable()
    assert b.controller.before_step(b.observation) is b.observation


def test_xyz_waypoint_requires_active_pelvis_and_progresses_only_after_real_steps(native_boundary):
    b = native_boundary
    b.panel.request_waypoint([.2, 0., .74], 0.)
    assert b.controller.before_step(b.observation) is None
    assert b.controller.waypoint.state == "IDLE"
    _waypoint_ready(b)
    actual_before = b.data.body_pos_w.copy()
    b.panel.request_waypoint([.2, 0., .74], 0.)
    assert b.controller.before_step(b.observation) is b.observation
    assert b.controller.waypoint.state == "RUNNING"
    for _ in range(3):
        assert b.controller.before_step(b.observation) is b.observation
    assert b.controller.waypoint.status()["elapsed_s"] == 0.
    b.controller.after_step(b.observation)  # Faithful fake boundary: one actual-step notification.
    assert b.controller.waypoint.status()["elapsed_s"] == .02
    assert b.controller.before_step(b.observation) is b.observation
    assert b.provider.goal_positions[0, 0] > 0.
    assert b.provider.goal_positions[0, 2] < .8
    assert b.panel.status["waypoint_target_xyz"].shape == (3,)
    np.testing.assert_array_equal(b.data.body_pos_w, actual_before)
    assert all(not history for _, history in b.calls)


def test_manual_goal_queued_during_step_is_not_overtaken_by_route_sequence(native_boundary):
    b = native_boundary
    _waypoint_ready(b)
    b.panel.request_waypoint([.2, 0., .74], 0.)
    b.controller.before_step(b.observation)
    b.panel.apply(b.provider.goal_positions, np.zeros((3, 3)))
    requested_sequence = b.panel._goal_pending["sequence"]
    b.controller.after_step(b.observation)
    b.controller.before_step(b.observation)
    assert b.provider.last_sequence == requested_sequence
    assert b.provider.rejected_count == 0
    assert b.controller.waypoint.state == "CANCELLED"


def test_waypoint_forecast_exposes_continuous_xyz_future_without_advancing_current(native_boundary):
    b = native_boundary
    _waypoint_ready(b)
    b.panel.request_waypoint([.2, 0., .74], 0.)
    b.controller.before_step(b.observation)
    current = b.provider.sample([0])["body_pos"].copy()
    b.controller.after_step(b.observation)
    b.controller.before_step(b.observation)
    future = b.provider.sample([0, 1, 2, 4, 32])
    np.testing.assert_allclose(future["body_pos"][:1], current)
    assert np.all(np.diff(future["body_pos"][:, 0, 0]) > 0.)
    assert np.all(np.diff(future["body_pos"][:, 0, 2]) < 0.)
    np.testing.assert_allclose(np.linalg.norm(future["body_lin_vel"][:, 0], axis=-1), .08)
    # Only the real command-step boundary consumes frame 1, not the full preview.
    b.provider.advance()
    np.testing.assert_allclose(b.provider.sample([0])["body_pos"][0], future["body_pos"][1])
    b.controller.after_step(b.observation)
    b.controller.before_step(b.observation)
    following = b.provider.sample([0, 1, 2, 4, 32])
    np.testing.assert_allclose(following["body_pos"][0], future["body_pos"][1])
    assert np.all(np.diff(following["body_pos"][:, 0, 0]) > 0.)
    assert b.controller.waypoint.status()["elapsed_s"] == .04
    assert all(not history for _, history in b.calls)


def test_manual_apply_replaces_active_waypoint_forecast_with_hold(native_boundary):
    b = native_boundary
    _waypoint_ready(b)
    b.panel.request_waypoint([.2, 0., .74], 0.)
    b.controller.before_step(b.observation)
    b.controller.after_step(b.observation)
    b.controller.before_step(b.observation)
    assert np.linalg.norm(b.provider.sample([4])["body_lin_vel"][0, 0]) > .07
    current = b.provider.sample([0])["body_pos"][0, [0, 10, 13]]
    b.panel.apply(current, np.zeros((3, 3)))
    b.controller.before_step(b.observation)
    assert b.controller.waypoint.state == "CANCELLED"
    np.testing.assert_allclose(b.provider.sample([0, 4, 32])["body_lin_vel"], 0.)


def test_pause_cancels_route_and_resume_cannot_restart_it(native_boundary):
    b = native_boundary
    _waypoint_ready(b)
    b.panel.request_waypoint([.2, 0., .74], 0.)
    b.controller.before_step(b.observation)
    b.controller.after_step(b.observation)
    b.panel.request_pause()
    assert b.controller.before_step(b.observation) is None
    assert b.controller.waypoint.state == "CANCELLED"
    elapsed = b.controller.waypoint.status()["elapsed_s"]
    b.panel.request_resume()
    assert b.controller.before_step(b.observation) is None
    b.panel.request_waypoint([.2, 0., .74], 0.)
    assert b.controller.before_step(b.observation) is None
    assert b.panel.status["awaiting_goal"]
    assert b.controller.waypoint.status()["elapsed_s"] == elapsed


def test_invalid_new_waypoint_is_atomic_and_vertical_speed_is_real_telemetry(native_boundary):
    b = native_boundary
    _waypoint_ready(b)
    b.panel.request_waypoint([.2, 0., .74], 0.)
    b.controller.before_step(b.observation)
    running = b.controller.waypoint
    old_goal = b.provider.goal_positions.copy()
    b.panel.request_waypoint([.2, 0., 1.1], 0.)
    b.controller.before_step(b.observation)
    assert b.controller.waypoint is running
    np.testing.assert_array_equal(b.provider.goal_positions, old_goal)
    b.data.body_link_lin_vel_w[0, 0, 2] = .2
    b.controller.after_step(b.observation)
    assert b.controller.waypoint.status()["linear_speed_m_s"] == .2
    b.data.body_link_lin_vel_w[0, 0, 2] = np.nan
    b.controller.after_step(b.observation)
    assert b.controller.session.state == "PAUSED_FAULT" and not b.sim.playing
    assert b.controller.waypoint.state == "CANCELLED"


def test_waypoint_stall_pauses_and_cannot_be_cleared_by_native_play(native_boundary):
    from dataclasses import replace
    from scaletrack.utils.waypoint import WaypointFollower
    b = native_boundary
    _waypoint_ready(b)
    # Only shorten this CPU test's clock; deployed defaults are unchanged.
    b.controller.waypoint = WaypointFollower(replace(b.controller.waypoint.cfg, stall_time_s=.04))
    b.panel.request_waypoint([.2, 0., .74], 0.)
    b.controller.before_step(b.observation)
    for _ in range(2):
        b.controller.after_step(b.observation)
    assert b.controller.waypoint.state == "STALLED"
    assert b.controller.session.state == "PAUSED_USER" and not b.sim.playing
    b.sim.playing = True
    assert b.controller.before_step(b.observation) is None
    assert not b.sim.playing
    b.panel.request_resume()
    assert b.controller.before_step(b.observation) is None
    assert b.panel.status["awaiting_goal"]


def test_waypoint_arrival_holds_reference_without_pausing_physics(native_boundary):
    from dataclasses import replace
    from scaletrack.utils.waypoint import WaypointFollower
    b = native_boundary
    _waypoint_ready(b)
    b.controller.waypoint = WaypointFollower(replace(b.controller.waypoint.cfg, settle_time_s=.04))
    b.panel.request_waypoint([0., 0., .8], 0.)
    b.controller.before_step(b.observation)
    for _ in range(2):
        b.controller.after_step(b.observation)
    assert b.controller.waypoint.state == "ARRIVED"
    assert b.controller.session.state == "ACTIVE" and b.sim.playing
    assert b.controller.before_step(b.observation) is b.observation
    terminal_time = b.controller.waypoint.status()["elapsed_s"]
    reference = b.provider.goal_positions.copy()
    b.controller.after_step(b.observation)
    assert b.controller.before_step(b.observation) is b.observation
    assert b.controller.waypoint.status()["elapsed_s"] == terminal_time
    np.testing.assert_array_equal(b.provider.goal_positions, reference)


def test_old_mode_waypoint_packet_cannot_start_a_new_route(native_boundary):
    b = native_boundary
    _waypoint_ready(b)
    b.panel.request_waypoint([.2, 0., .74], 0.)
    old = b.panel.consume_request()
    _switch_mode(b, "UMI-2")
    _switch_mode(b, "Pelvis-1")
    b.panel.request_resume()
    b.controller.before_step(b.observation)
    b.panel.apply(b.provider.goal_positions, np.zeros((3, 3)))
    b.controller.before_step(b.observation)
    b.panel._goal_pending = old
    before_sequence = b.provider.last_sequence
    b.controller.before_step(b.observation)
    assert b.controller.waypoint.state == "IDLE"
    assert b.provider.last_sequence == before_sequence
    assert b.controller.mode_rejected_count == 1


@pytest.mark.parametrize("target_yaw", [-179., -30., 0., 179.])
def test_waypoint_heading_changes_preserve_reference_tilt_and_unit_quaternion(target_yaw):
    from online_control import _set_yaw_wxyz, _yaw_wxyz
    from online_ui import _wxyz_from_xyz_degrees, _xyz_degrees_from_wxyz
    before = _wxyz_from_xyz_degrees(np.array([[15., -10., 175.]]))[0]
    after = _set_yaw_wxyz(before, np.deg2rad(target_yaw))
    np.testing.assert_allclose(np.linalg.norm(after), 1.)
    np.testing.assert_allclose(_yaw_wxyz(after), np.deg2rad(target_yaw), atol=1e-12)
    np.testing.assert_allclose(_xyz_degrees_from_wxyz(after[None])[0, :2], [15., -10.], atol=1e-10)
