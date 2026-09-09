"""CPU export/FK/masking contract tests; no Kit, viewer, physics or network."""
import copy
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ScaleBridge"))
from scalebridge.agent import portable_policy as portable

XML = ROOT / "ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof/g1_29dof.xml"


def fixture_metadata():
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(XML))
    # Deliberately non-XML order exercises the named permutation in FK.
    joints = [model.joint(i).name for i in range(1, 30)][::-1]
    bodies = [model.body(i).name for i in range(1, 31)]
    table = [[int(i < count) for i in range(14)] for count in (1, 2, 3, 4, 5, 6, 7, 14)]
    return {"joint_names": joints, "action_names": joints.copy(), "body_names": bodies,
            "selected_body_names": bodies[:14], "history_buffer_size": 4, "future_idx": [0, 1, 2],
            "default_dof_pos": [0.] * 29, "action_scale": [.25] * 29, "mode_table": table,
            "policy_architecture": {"prop_obs_dim": 64, "action_dim": 29, "output_dim": 29,
                "task_obs_dim": 267, "embedding_dim": 32, "num_heads": 4, "ff_dim": 32,
                "num_layers": 1, "reduced_task_dim": None, "task_embedder_hidden_dims": None}}


def fixture_policy(metadata):
    nets = portable.native_networks()
    actor = nets.HumanoidTransformer(64, 29, 29, embed_dim=32, num_heads=4, ff_dim=32, num_layers=1)
    embedder = nets.TaskEmbedder(267, 32)
    return portable.PortableBFMPolicy(actor, embedder, metadata, XML).eval()


class PortablePolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(123)
        cls.metadata = fixture_metadata()
        cls.policy = fixture_policy(cls.metadata)

    def test_native_leaf_does_not_load_isaac_or_tensorrt(self):
        before = set(sys.modules)
        portable.native_networks()
        self.assertFalse(any(name.startswith(("isaaclab", "omni.", "torch_tensorrt")) for name in set(sys.modules)-before))

    def test_mask_blocks_keep_timestamp(self):
        table = torch.tensor(self.metadata["mode_table"])
        result = portable.mode_mappings(table)
        self.assertEqual(tuple(result.shape), (8, 253))
        self.assertTrue(torch.equal(result[:, -1], torch.ones(8)))
        self.assertEqual(result[0].sum().item(), 19)
        self.assertEqual(result[7].sum().item(), 253)

    def test_bad_masks_are_rejected(self):
        for table in (torch.zeros(8, 14), torch.ones(7, 14), torch.full((8, 14), .5)):
            with self.subTest(shape=table.shape), self.assertRaises(ValueError):
                portable.mode_mappings(table)

    def test_named_fk_and_observations_agree_with_independent_mujoco_scipy(self):
        with torch.inference_mode():
            for seed in (5, 33):
                for mode in range(8):
                    inputs = list(portable.example_inputs(self.metadata, seed))
                    inputs[7] = torch.tensor([mode])
                    prop, task = self.policy.observations(*inputs[:4], *inputs[5:])
                    native_prop, native_actions, native_task = portable.independent_observations(inputs, self.metadata, XML)
                    torch.testing.assert_close(prop, native_prop, atol=2e-6, rtol=1e-6)
                    torch.testing.assert_close(task, native_task, atol=3e-6, rtol=1e-6)
                    pd, action = self.policy(*inputs)
                    direct = self.policy.actor(native_prop, native_actions, self.policy.task_embedder(native_task))
                    torch.testing.assert_close(action, direct, atol=1e-5, rtol=1e-5)
                    torch.testing.assert_close(pd, action * .25)

    def test_trace_retains_all_eight_runtime_masks(self):
        with torch.inference_mode():
            traced = torch.jit.trace(self.policy, portable.example_inputs(self.metadata), check_trace=True)
            outputs = []
            for mode in range(8):
                inputs = list(portable.example_inputs(self.metadata, seed=902))
                inputs[7] = torch.tensor([mode])
                actual = traced(*inputs)
                expected = self.policy(*inputs)
                for first, second in zip(actual, expected):
                    torch.testing.assert_close(first, second, atol=1e-5, rtol=1e-5)
                outputs.append(actual[1])
            self.assertGreater(float((outputs[0] - outputs[7]).abs().max()), 1e-5)

    def test_pd_metadata_errors_fail_closed(self):
        for key, value in (("action_scale", [0.] * 29), ("default_dof_pos", [float("nan")] * 29),
                           ("action_names", list(reversed(self.metadata["action_names"]))),
                           ("history_buffer_size", 0), ("selected_body_names", ["pelvis"] * 14)):
            candidate = copy.deepcopy(self.metadata)
            candidate[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                fixture_policy(candidate)

    def test_positive_height_is_preserved_in_task(self):
        inputs = list(portable.example_inputs(self.metadata))
        inputs[7] = torch.tensor([7])
        with torch.inference_mode():
            _, before = self.policy.observations(*inputs[:4], *inputs[5:])
            inputs[5] = inputs[5].clone()
            inputs[5][..., 0, 2] += .25
            _, after = self.policy.observations(*inputs[:4], *inputs[5:])
        torch.testing.assert_close(after[..., 2] - before[..., 2], torch.full((1, 3), .25))
        torch.testing.assert_close(after[..., 44] - before[..., 44], torch.full((1, 3), .25))


if __name__ == "__main__":
    unittest.main()
