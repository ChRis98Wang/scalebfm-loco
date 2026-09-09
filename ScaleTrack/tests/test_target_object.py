"""Target scenes must be opt-in, physical, resettable and checkpoint-compatible."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/pretrain/rsl_rl/target_object.py"


def load_target_module():
    assert MODULE_PATH.is_file(), "target-object implementation is missing"
    spec = importlib.util.spec_from_file_location("bfm_target_object_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_object_is_opt_in_and_cli_can_load_explicit_physics(tmp_path):
    target_module = load_target_module()
    parser = argparse.ArgumentParser()
    target_module.add_target_object_args(parser)
    assert target_module.target_spec_from_args(parser.parse_args([])) is None
    config = tmp_path / "box.json"
    config.write_text(json.dumps({"mass": 2.0, "static_friction": 0.8, "dynamic_friction": 0.6}), encoding="utf-8")
    args = parser.parse_args(["--target_object", "--target_config", str(config), "--target_position", "1", "2", "0.3"])
    spec = target_module.target_spec_from_args(args)
    assert spec.mass == 2.0
    assert spec.position == (1.0, 2.0, 0.3)
    assert spec.static_friction == 0.8
    assert spec.dynamic_friction == 0.6
    assert spec.to_dict()["calibrated"] is False


def test_bundled_engineering_profile_matches_the_validated_defaults():
    module = load_target_module()
    parser = argparse.ArgumentParser()
    module.add_target_object_args(parser)
    profile = Path(__file__).resolve().parents[2] / "configs/target_objects/box_engineering.json"
    assert profile.is_file(), "portable target material profile is missing"
    spec = module.target_spec_from_args(parser.parse_args(["--target_object", "--target_config", str(profile)]))
    assert spec == module.TargetObjectSpec()


@pytest.mark.parametrize("override", [
    {"mass": 0}, {"mass": float("nan")}, {"size": (0, 0.3, 0.3)},
    {"position": (0, 0, 0)}, {"position": (0, 0, float("inf"))},
    {"dynamic_friction": 0.9, "static_friction": 0.6}, {"static_friction": -1},
    {"restitution": 1.1}, {"contact_offset": 0}, {"rest_offset": 0.01, "contact_offset": 0.005},
])
def test_invalid_physics_is_rejected_before_spawning(override):
    target_module = load_target_module()
    with pytest.raises(ValueError):
        target_module.TargetObjectSpec(**override)


def test_disabled_target_options_are_not_silently_ignored():
    target_module = load_target_module()
    parser = argparse.ArgumentParser()
    target_module.add_target_object_args(parser)
    with pytest.raises(ValueError, match="target_object"):
        target_module.target_spec_from_args(parser.parse_args(["--target_position", "1", "0", "0.2"]))


def test_attaching_object_preserves_observations_rewards_and_baseline():
    target_module = load_target_module()
    from scaletrack.tasks.tracking.config.g1_29dof.flat_env_cfg import G1BFMTrackingEnvCfg

    baseline, variant = G1BFMTrackingEnvCfg(), G1BFMTrackingEnvCfg()
    target_module.attach_target_object(baseline, None)
    spec = target_module.TargetObjectSpec(mass=2.0)
    target_module.attach_target_object(variant, spec)
    assert getattr(baseline.scene, "target_object", None) is None
    assert variant.observations.to_dict() == baseline.observations.to_dict()
    assert variant.rewards.to_dict() == baseline.rewards.to_dict()
    assert variant.scene.terrain.to_dict() == baseline.scene.terrain.to_dict()
    obj = variant.scene.target_object
    assert obj.prim_path == "{ENV_REGEX_NS}/TargetObject"
    assert obj.spawn.mass_props.mass == 2.0
    assert obj.spawn.rigid_props.kinematic_enabled is False
    assert obj.spawn.rigid_props.disable_gravity is False
    assert obj.spawn.collision_props.collision_enabled is True
    assert obj.spawn.collision_props.rest_offset == 0.0
    assert obj.spawn.physics_material.dynamic_friction == 0.5
    assert obj.spawn.physics_material.friction_combine_mode == "multiply"
    assert obj.spawn.physics_material.restitution_combine_mode == "max"
    assert variant.events.reset_target_object.mode == "reset"
    assert "disable_flag" not in variant.events.reset_target_object.params
    with pytest.raises(ValueError, match="already"):
        target_module.attach_target_object(variant, spec)


class RigidStateBoundary:
    """Real tensor state; replace only writes that otherwise require a simulator."""

    def __init__(self):
        self.device = "cpu"
        self.data = SimpleNamespace(
            default_root_pose=torch.tensor([[0.9, -0.7, 0.16, 0., 0., 0., 1.]]).repeat(3, 1),
            default_root_vel=torch.zeros((3, 6)),
        )
        self.poses = torch.full((3, 7), 99.0)
        self.velocities = torch.ones((3, 6))

    def write_root_pose_to_sim(self, root_pose, env_ids):
        self.poses[env_ids] = root_pose

    def write_root_velocity_to_sim(self, root_velocity, env_ids):
        self.velocities[env_ids] = root_velocity

    def reset(self, env_ids):
        pass


def physical_robot_command(position):
    return SimpleNamespace(
        robot_anchor_pos_w=position,
        robot_anchor_body_index=0,
        robot=SimpleNamespace(data=SimpleNamespace(body_pos_w=position[:, None, :])),
    )


def test_reset_applies_environment_origins_once_and_only_to_selected_object():
    target_module = load_target_module()
    obj = RigidStateBoundary()
    scene = {"target_object": obj}
    class Scene(dict):
        env_origins = torch.tensor([[0., 0., 0.], [3., 5., 0.], [-3., -5., 0.]])
    env = SimpleNamespace(scene=Scene(scene), num_envs=3, device="cpu")
    target_module.reset_target_object(env, torch.tensor([1]))
    torch.testing.assert_close(obj.poses[1], torch.tensor([3.9, 4.3, 0.16, 0., 0., 0., 1.]))
    assert torch.all(obj.velocities[1] == 0)
    assert torch.all(obj.poses[[0, 2]] == 99)
    assert obj.data.default_root_pose[1, 0] == pytest.approx(0.9)


def test_target_status_uses_actual_object_position_not_its_spawn_point():
    target_module = load_target_module()
    class Scene(dict):
        env_origins = torch.tensor([[10., 20., 0.]])
    scene = Scene(target_object=SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.tensor([[13., 24., 0.2]]))))
    command = physical_robot_command(torch.tensor([[10., 20., 0.2]]))
    env = SimpleNamespace(scene=scene)
    status = target_module.target_status(env, command)
    assert status["position_local"] == pytest.approx([3., 4., 0.2])
    assert status["root_distance"] == pytest.approx(5.0)


def test_scene_metrics_cover_every_environment_in_world_coordinates():
    module = load_target_module()
    assert hasattr(module, "target_metrics"), "per-environment object metrics are missing"
    env = SimpleNamespace(scene={"target_object": SimpleNamespace(data=SimpleNamespace(
        root_pos_w=torch.tensor([[3., 4., 0.2], [10., 0., 0.5]]),
        root_lin_vel_w=torch.tensor([[0., 0., 0.], [0., 3., 4.]]),
    ))})
    command = physical_robot_command(torch.tensor([[0., 0., 0.2], [7., 0., 0.5]]))
    metrics = module.target_metrics(env, command)
    torch.testing.assert_close(metrics["target_root_distance"], torch.tensor([5., 3.]))
    torch.testing.assert_close(metrics["target_height"], torch.tensor([0.2, 0.5]))
    torch.testing.assert_close(metrics["target_speed"], torch.tensor([0., 5.]))


def test_target_distance_is_physical_even_when_local_tracking_overrides_command_anchor():
    module = load_target_module()
    class Scene(dict):
        env_origins = torch.zeros((1, 3))
    env = SimpleNamespace(scene=Scene(target_object=SimpleNamespace(data=SimpleNamespace(
        root_pos_w=torch.tensor([[3., 4., 0.2]]), root_lin_vel_w=torch.zeros((1, 3)),
    ))))
    command = physical_robot_command(torch.tensor([[0., 0., 0.2]]))
    # play.py's optional local-tracking mode changes this property to the reference.
    command.robot_anchor_pos_w = torch.tensor([[3., 4., 0.2]])
    assert module.target_status(env, command)["root_distance"] == pytest.approx(5.0)
    torch.testing.assert_close(module.target_metrics(env, command)["target_root_distance"], torch.tensor([5.0]))
