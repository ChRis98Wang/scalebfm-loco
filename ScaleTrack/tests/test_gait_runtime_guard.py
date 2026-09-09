"""CPU-only tests for the gait playback state-write guard."""

from __future__ import annotations

from pathlib import Path
from types import MethodType
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/pretrain/rsl_rl"))
from gait_runtime_guard import forbid_state_writes


class Calls:
    def __init__(self):
        self.names = []

    def mark(self, name):
        self.names.append(name)


def _fixture():
    calls = Calls()

    class Robot:
        def reset(self, *_args, **_kwargs): calls.mark("robot.reset")
        def write_root_pose_to_sim(self, *_args, **_kwargs): calls.mark("root_pose")
        def write_root_velocity_to_sim(self, *_args, **_kwargs): calls.mark("root_velocity")
        def write_root_state_to_sim(self, *_args, **_kwargs): calls.mark("root_state")
        def write_joint_state_to_sim(self, *_args, **_kwargs): calls.mark("joint_state")
        def set_joint_position_target(self, *_args, **_kwargs): calls.mark("joint_target")

    class Scene:
        def reset(self, *_args, **_kwargs): calls.mark("scene.reset")

    class Manager:
        def __init__(self, label): self.label = label
        def reset(self, *_args, **_kwargs): calls.mark(self.label)

    class Env:
        scene = Scene()
        command_manager = Manager("command_manager.reset")
        event_manager = Manager("event_manager.reset")
        def reset(self, *_args, **_kwargs): calls.mark("env.reset")
        def _reset_idx(self, *_args, **_kwargs): calls.mark("env._reset_idx")

    class Command:
        robot = Robot()
        def _resample_command(self, *_args, **_kwargs): calls.mark("resample")

    return Env(), Command(), calls


def test_guard_counts_and_rejects_every_required_and_optional_call_without_calling_originals():
    env, command, calls = _fixture()
    calls_to_try = (
        (env, "reset", "env.reset"),
        (env, "_reset_idx", "env._reset_idx"),
        (env.scene, "reset", "scene.reset"),
        (command, "_resample_command", "command._resample_command"),
        (command.robot, "write_root_pose_to_sim", "robot.write_root_pose_to_sim"),
        (command.robot, "write_root_velocity_to_sim", "robot.write_root_velocity_to_sim"),
        (command.robot, "write_root_state_to_sim", "robot.write_root_state_to_sim"),
        (command.robot, "write_joint_state_to_sim", "robot.write_joint_state_to_sim"),
        (env.command_manager, "reset", "command_manager.reset"),
        (env.event_manager, "reset", "event_manager.reset"),
        (command.robot, "reset", "robot.reset"),
    )
    with forbid_state_writes(env, command) as counters:
        for owner, name, label in calls_to_try:
            with pytest.raises(RuntimeError, match=label.replace(".", r"\.")):
                getattr(owner, name)()
        assert counters == {label: 1 for _, _, label in calls_to_try}
        command.robot.set_joint_position_target(object())

    assert calls.names == ["joint_target"]
    env.reset()
    command.robot.write_joint_state_to_sim(object())
    assert calls.names[-2:] == ["env.reset", "joint_state"]


def test_guard_restores_existing_instance_override_and_removes_inherited_overrides_on_exception():
    env, command, _ = _fixture()

    def special_root_state(self, *_args, **_kwargs):
        return "special"

    override = MethodType(special_root_state, command.robot)
    command.robot.write_root_state_to_sim = override
    assert "reset" not in vars(env)
    with pytest.raises(ValueError, match="body failure"):
        with forbid_state_writes(env, command):
            assert "reset" in vars(env)
            raise ValueError("body failure")

    assert "reset" not in vars(env)
    assert vars(command.robot)["write_root_state_to_sim"] is override
    assert command.robot.write_root_state_to_sim() == "special"


def test_missing_required_method_fails_before_installing_any_override():
    env, command, _ = _fixture()
    env.scene.reset = None
    before = {"env": dict(vars(env)), "scene": dict(vars(env.scene)), "command": dict(vars(command)),
              "robot": dict(vars(command.robot))}
    with pytest.raises(AttributeError, match="scene.reset"):
        with forbid_state_writes(env, command):
            pytest.fail("missing required method must prevent entry")
    assert dict(vars(env)) == before["env"]
    assert dict(vars(env.scene)) == before["scene"]
    assert dict(vars(command)) == before["command"]
    assert dict(vars(command.robot)) == before["robot"]


def test_absent_optional_methods_are_allowed_and_not_reported():
    env, command, _ = _fixture()
    env.command_manager = object()
    env.event_manager = object()
    command.robot.reset = None
    with forbid_state_writes(env, command) as counters:
        assert set(counters) == {
            "env.reset", "env._reset_idx", "scene.reset", "command._resample_command",
            "robot.write_root_pose_to_sim", "robot.write_root_velocity_to_sim",
            "robot.write_root_state_to_sim", "robot.write_joint_state_to_sim",
        }
