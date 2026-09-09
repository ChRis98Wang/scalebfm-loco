"""CPU-only ScaleBridge lifecycle/contracts; all physics stepping is mocked."""
import importlib
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
from omegaconf import OmegaConf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ScaleBridge"))
from scalebridge import run
from scalebridge.agent.bfm_agent import BFMAgent
from scalebridge.env.motion_tracking import MotionTrackingEnv
from scalebridge.simulator.mujoco_simulator import MujocoSimulator, world_to_body_vector


def configuration(**updates):
    data = {"max_steps": 3, "warmup_steps": 2, "realtime": False, "allow_real_world": False,
            "agent": {}, "env": {}, "simulator": {
                "_target_": "simulator.mujoco_simulator.MujocoSimulator", "config": {"headless": True}}}
    data.update(updates)
    return OmegaConf.create(data)


def simulator_fixture():
    sim = MujocoSimulator.__new__(MujocoSimulator)
    sim.cfg = OmegaConf.create({"low_dt": .005, "decimation": 4, "asset": {"xml_path": "unused.xml"}})
    sim.metadata_dict = {"joint_names": ["b", "a"], "action_names": ["a", "b"],
                         "stiffness": [20., 10.], "damping": [2., 1.],
                         "default_dof_pos": [2., 1.], "torque_limit": [8., 12.]}
    sim.headless, sim.record_video, sim.marker, sim.use_joystick = True, False, False, False
    sim._closed = False
    sim._joystick_stop = threading.Event()
    sim.viewer = sim.renderer = sim.video_writer = sim.joystick_thread = sim._pygame = None
    sim.mujoco_model = SimpleNamespace(nu=2, actuator_trnid=np.array([[1, 0], [2, 0]]),
        jnt_qposadr=np.array([0, 7, 8]), jnt_dofadr=np.array([0, 6, 7]),
        actuator_gear=np.tile([1., 0., 0., 0., 0., 0.], (2, 1)),
        actuator_ctrlrange=np.array([[-10., 10.], [-8., 8.]]),
        actuator_ctrllimited=np.ones(2), jnt_actfrclimited=np.ones(3),
        jnt_actfrcrange=np.array([[0., 0.], [-20., 20.], [-8., 8.]]), opt=SimpleNamespace(timestep=0.))
    sim.mujoco_data = SimpleNamespace(qpos=np.array([0., 0., .8, 1., 0., 0., 0., 0., 0.]),
        qvel=np.zeros(8), ctrl=np.zeros(2))
    sim._get_joint_names = mock.Mock(return_value=["a", "b"])
    return sim


class ControllerTests(unittest.TestCase):
    def test_headless_requires_bound_and_real_backend_requires_explicit_opt_in(self):
        self.assertEqual(run.validate_runtime_config(configuration()), (3, 2))
        for change in ({"max_steps": 0}, {"max_steps": True}, {"max_steps": -1}, {"warmup_steps": 101}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                run.validate_runtime_config(configuration(**change))
        cfg = configuration()
        cfg.simulator._target_ = "simulator.real_world.RealWorld"
        with mock.patch.object(run, "instantiate") as instantiate, self.assertRaises(ValueError):
            run.run_controller(cfg)
        instantiate.assert_not_called()

    def test_finite_loop_and_cleanup_on_success(self):
        agent = mock.Mock()
        agent.get_meta_data.return_value = {}
        agent.get_action.return_value = (torch.zeros(1, 2), torch.zeros(1, 2))
        env = mock.Mock(dt=.02)
        env.simulator.is_running.return_value = True
        with mock.patch.object(run, "instantiate", side_effect=[agent, env]), mock.patch.object(run.time, "sleep") as sleep:
            result = run.run_controller(configuration())
        self.assertEqual(result["steps"], 3)
        self.assertEqual(env.step.call_count, 3)
        self.assertEqual(agent.get_action.call_count, 5)
        agent.close.assert_called_once()
        env.close.assert_called_once()
        sleep.assert_not_called()

    def test_failure_and_keyboard_interrupt_close_every_constructed_resource(self):
        for error in (RuntimeError("policy failed"), KeyboardInterrupt()):
            agent, env = mock.Mock(), mock.Mock(dt=.02)
            agent.get_action.side_effect = error
            with mock.patch.object(run, "instantiate", side_effect=[agent, env]), \
                    self.subTest(error=type(error).__name__), self.assertRaises(type(error)):
                run.run_controller(configuration())
            agent.close.assert_called_once()
            env.close.assert_called_once()

    def test_closed_viewer_ends_loop_cleanly_without_physics(self):
        agent, env = mock.Mock(), mock.Mock(dt=.02)
        env.simulator.is_running.return_value = False
        with mock.patch.object(run, "instantiate", side_effect=[agent, env]):
            self.assertEqual(run.run_controller(configuration(warmup_steps=0))["steps"], 0)
        env.step.assert_not_called()
        env.close.assert_called_once()


class AgentTests(unittest.TestCase):
    def make_agent(self, directory, backend="torchscript", mode=7):
        checkpoint = Path(directory) / "official_tensorrt.pt"
        checkpoint.write_bytes(b"mocked torchscript")
        checkpoint.with_name("official_tensorrt_metadata.json").write_text(json.dumps({
            "runtime_backend": backend, "joint_names": ["a", "b"], "action_names": ["a", "b"]}))
        return BFMAgent(OmegaConf.create({"checkpoint": str(checkpoint), "control_mode": mode}), "cpu")

    def test_plain_torchscript_cpu_never_imports_tensorrt(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch("torch.jit.load", return_value=mock.Mock()) as loader, \
                mock.patch.object(importlib, "import_module", side_effect=AssertionError("unexpected backend import")):
            agent = self.make_agent(directory)
        self.assertEqual(agent.runtime_backend, "torchscript")
        self.assertEqual(loader.call_args.kwargs["map_location"], "cpu")
        self.assertEqual(agent.control_mode.tolist(), [7])
        agent.close()
        agent.close()
        self.assertIsNone(agent.policy)

    def test_trt_on_cpu_unknown_backend_and_invalid_modes_rejected(self):
        for backend, mode in (("tensorrt", 7), ("unknown", 7), ("torchscript", 8), ("torchscript", True)):
            with tempfile.TemporaryDirectory() as directory, mock.patch("torch.jit.load", return_value=mock.Mock()), \
                    self.subTest(backend=backend, mode=mode), self.assertRaises(ValueError):
                self.make_agent(directory, backend, mode)

    def test_nine_input_contract_and_finite_pd_output(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch("torch.jit.load", return_value=mock.Mock()):
            agent = self.make_agent(directory)
        names = ("root_quat_buffer", "base_ang_vel_buffer", "dof_pos_buffer", "dof_vel_buffer",
                 "actions_buffer", "target_body_pos_future_to_robot_base", "target_body_rot_future_to_robot_base", "future_time_offsets")
        observations = {name: torch.zeros(1) for name in names}
        agent.policy.return_value = (torch.zeros(1, 2), torch.zeros(1, 2))
        self.assertEqual(len(agent.get_action(observations)), 2)
        self.assertEqual(len(agent.policy.call_args.args), 9)
        for result in ((torch.zeros(2), torch.zeros(1, 2)), (torch.full((1, 2), float("nan")), torch.zeros(1, 2)), torch.zeros(1, 2)):
            agent.policy.return_value = result
            with self.assertRaises(ValueError):
                agent.get_action(observations)

    def strict_agent(self, directory, **updates):
        checkpoint = Path(directory) / "verified.pt"
        checkpoint.write_bytes(b"mocked verified policy")
        metadata = {"runtime_backend": "torchscript", "export_batch_size": 1,
            "export_verification": "PASS", "policy_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "inference_input_quaternion_order": "wxyz", "history_buffer_size": 2,
            "future_idx": [0, 1, 2, 3, 4, 5], "joint_names": [f"j{i}" for i in range(29)],
            "action_names": [f"j{i}" for i in range(29)], "selected_body_names": [f"b{i}" for i in range(14)]}
        metadata.update(updates)
        checkpoint.with_name("verified_metadata.json").write_text(json.dumps(metadata))
        with mock.patch("torch.jit.load", return_value=mock.Mock()):
            agent = BFMAgent(OmegaConf.create({"checkpoint": str(checkpoint), "control_mode": 7}), "cpu")
        agent.policy.return_value = (torch.zeros(1, 29), torch.zeros(1, 29))
        return agent

    def strict_inputs(self):
        values = {"root_quat_buffer": torch.zeros(1, 2, 4), "base_ang_vel_buffer": torch.zeros(1, 2, 3),
            "dof_pos_buffer": torch.zeros(1, 2, 29), "dof_vel_buffer": torch.zeros(1, 2, 29),
            "actions_buffer": torch.zeros(1, 2, 29), "target_body_pos_future_to_robot_base": torch.zeros(1, 6, 14, 3),
            "target_body_rot_future_to_robot_base": torch.zeros(1, 6, 14, 4),
            "future_time_offsets": torch.arange(6, dtype=torch.int64)[None, :, None]}
        values["root_quat_buffer"][..., 0] = 1.
        values["target_body_rot_future_to_robot_base"][..., 0] = 1.
        return values

    def test_new_artifact_requires_verified_sha_and_batch_one_contract(self):
        for change in ({"export_verification": "FAIL"}, {"policy_sha256": "f" * 64},
                       {"export_batch_size": 2}, {"inference_input_quaternion_order": "xyzw"},
                       {"future_idx": [1, 2]}, {"history_buffer_size": 0}, {"joint_names": ["a"] * 29}):
            with tempfile.TemporaryDirectory() as directory, self.subTest(change=change), self.assertRaises(ValueError):
                self.strict_agent(directory, **change)

    def test_new_artifact_rejects_wrong_shape_dtype_finite_quaternion_or_future_values(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = self.strict_agent(directory)
        agent.get_action(self.strict_inputs())
        agent.policy.assert_called_once()
        mutations = [
            ("dof_pos_buffer", torch.zeros(1, 1, 29)), ("dof_vel_buffer", torch.zeros(1, 2, 29, dtype=torch.float64)),
            ("base_ang_vel_buffer", torch.full((1, 2, 3), float("nan"))), ("root_quat_buffer", torch.zeros(1, 2, 4)),
            ("target_body_rot_future_to_robot_base", torch.zeros(1, 6, 14, 4)),
            ("future_time_offsets", torch.arange(1, 7, dtype=torch.int64)[None, :, None]),
            ("future_time_offsets", torch.arange(6, dtype=torch.float32)[None, :, None])]
        for name, value in mutations:
            inputs = self.strict_inputs()
            inputs[name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                agent.get_action(inputs)
        agent.control_mode[0] = 8
        with self.assertRaises(ValueError):
            agent.get_action(self.strict_inputs())
        self.assertEqual(agent.policy.call_count, 1)


class MujocoLifecycleTests(unittest.TestCase):
    def test_headless_backbone_never_creates_viewer_or_renderer(self):
        sim = simulator_fixture()
        with mock.patch("mujoco.MjModel.from_xml_path", return_value=sim.mujoco_model), \
                mock.patch("mujoco.MjData", return_value=sim.mujoco_data), \
                mock.patch("mujoco.Renderer", side_effect=AssertionError("renderer launched")):
            sim._setup_backbone()
        self.assertIsNone(sim.viewer)
        sim._render()
        self.assertTrue(sim.is_running())
        sim.close()
        self.assertFalse(sim.is_running())

    def test_close_is_idempotent_closes_all_resources_and_joins_worker(self):
        sim = simulator_fixture()
        resources = [mock.Mock() for _ in range(3)]
        sim.viewer, sim.renderer, sim.video_writer = resources
        worker, pygame = mock.Mock(), mock.Mock()
        sim.joystick_thread, sim._pygame = worker, pygame
        sim.close()
        sim.close()
        for resource in resources:
            resource.close.assert_called_once()
        worker.join.assert_called_once_with(timeout=2.)
        pygame.quit.assert_called_once()
        self.assertTrue(sim._joystick_stop.is_set())

    def test_joint_pd_mapping_defaults_and_conservative_torque_limits(self):
        sim = simulator_fixture()
        sim._setup_asset()
        np.testing.assert_array_equal(sim.stiffness, [10., 20.])
        np.testing.assert_array_equal(sim.damping, [1., 2.])
        np.testing.assert_array_equal(sim.default_dof_pos, [1., 2.])
        np.testing.assert_array_equal(sim.torque_limit, [10., 8.])
        self.assertEqual(sim.torque_limit_mismatches, [{"joint": "a", "metadata": 12., "xml": 10.}])
        sim.cfg.strict_torque_limits = True
        with self.assertRaisesRegex(ValueError, "torque limits differ"):
            sim._setup_asset()

    def test_wrong_joint_transmission_order_is_rejected(self):
        sim = simulator_fixture()
        sim.mujoco_model.actuator_trnid[:, 0] = [2, 1]
        with self.assertRaisesRegex(ValueError, "unit-gear"):
            sim._setup_asset()

    def test_rsi_uses_joint_order_and_converts_world_angular_velocity(self):
        sim = simulator_fixture()
        sim._setup_asset()
        sim.has_object = False
        q = np.array([np.sqrt(.5), 0., 0., np.sqrt(.5)])
        with mock.patch("mujoco.mj_forward") as forward:
            sim.calibrate({"root_quat": q, "root_ang_vel_w": [1., 0., 0.], "dof_pos": [5., 6.], "dof_vel": [7., 8.]})
        forward.assert_called_once()
        np.testing.assert_allclose(sim.mujoco_data.qvel[3:6], [0., -1., 0.], atol=1e-12)
        np.testing.assert_array_equal(sim.mujoco_data.qpos[7:], [6., 5.])
        np.testing.assert_array_equal(sim.mujoco_data.qvel[6:], [8., 7.])

    def test_finite_target_guard_precedes_any_step_and_pd_is_explicitly_clipped(self):
        sim = simulator_fixture()
        sim._setup_asset()
        sim.decimation = 4
        with mock.patch("mujoco.mj_step") as step:
            with self.assertRaises(ValueError):
                sim.apply_action([float("nan"), 0.])
            step.assert_not_called()
            sim.apply_action([100., -100.])
            self.assertEqual(step.call_count, 4)
        np.testing.assert_array_equal(sim.mujoco_data.ctrl, [10., -8.])

    def test_world_vector_conversion_rejects_invalid_quaternion(self):
        np.testing.assert_array_equal(world_to_body_vector([1., 0., 0., 0.], [1., 2., 3.]), [1., 2., 3.])
        with self.assertRaises(ValueError):
            world_to_body_vector([0., 0., 0., 0.], [1., 2., 3.])


class ReferenceResetTests(unittest.TestCase):
    def fixture(self):
        env = MotionTrackingEnv.__new__(MotionTrackingEnv)
        env.device = "cpu"
        env.cfg = OmegaConf.create({"future_idx": [0, 1], "reference_forcing": False, "rsi": False})
        env.metadata_dict = {"selected_body_names": ["pelvis", "torso"], "body_names": ["pelvis", "torso"],
                             "joint_names": ["a", "b"], "future_idx": [0, 1]}
        positions = np.array([[[1., 2., .8], [1., 2., 1.]], [[2., 2., .9], [2., 2., 1.1]]], dtype=np.float32)
        env._setup_motion({"body_pos_w": positions, "body_quat_w": np.tile([1., 0., 0., 0.], (2, 2, 1)).astype(np.float32),
            "joint_pos": np.zeros((2, 2), np.float32), "joint_vel": np.zeros((2, 2), np.float32),
            "body_lin_vel_w": np.zeros((2, 2, 3), np.float32), "body_ang_vel_w": np.ones((2, 2, 3), np.float32)})
        env.simulator = mock.Mock()
        env.simulator.calibrate.return_value = (np.array([3., 4., .8]), np.array([1., 0., 0., 0.]))
        env._update_state_manager = mock.Mock()
        return env

    def test_non_rsi_alignment_preserves_source_height_and_repeated_reset(self):
        env = self.fixture()
        source = env._source_body_pos_w.clone()
        env._calibrate()
        aligned = env.body_pos_w.clone()
        torch.testing.assert_close(env.body_pos_w[0, 0], torch.tensor([3., 4., .8]))
        torch.testing.assert_close(env._source_body_pos_w, source, rtol=0, atol=0)
        env._calibrate()
        torch.testing.assert_close(env.body_pos_w, aligned, rtol=0, atol=0)
        torch.testing.assert_close(env._source_body_pos_w, source, rtol=0, atol=0)

    def test_rsi_labels_world_angular_velocity_and_keeps_original_reference(self):
        env = self.fixture()
        env.cfg.rsi = True
        env._calibrate()
        state = env.simulator.calibrate.call_args.args[0]
        self.assertIn("root_ang_vel_w", state)
        self.assertNotIn("root_ang_vel", state)
        torch.testing.assert_close(env.body_pos_w, env._source_body_pos_w, rtol=0, atol=0)

    def test_reset_gathers_new_future_after_base_reset(self):
        env = self.fixture()
        env._gather_reference_state = mock.Mock()
        env._update_observation_manager = mock.Mock(return_value={"new": True})
        with mock.patch("scalebridge.env.base_env.BaseEnv.reset") as reset:
            self.assertEqual(env.reset(), {"new": True})
        reset.assert_called_once()
        env._gather_reference_state.assert_called_once()


if __name__ == "__main__":
    unittest.main()
