import ast
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts import build_bfm_deployment_metadata as metadata


class DeploymentMetadataTests(unittest.TestCase):
    def expression(self, value):
        return ast.parse(value, mode="eval").body

    def test_restricted_arithmetic_and_nested_literals(self):
        result = metadata.constant(self.expression("{'x': [2*A**2, -0.312], 'y': (True, None)}"), {"A": 3})
        self.assertEqual(result, {"x": [18, -.312], "y": (True, None)})

    def test_no_python_execution(self):
        for text in ("__import__('os').system('false')", "x.y", "x[0]", "[x for x in [1]]", "2**10000"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                metadata.constant(self.expression(text))

    def test_duplicate_literal_keys_rejected(self):
        with self.assertRaises(ValueError):
            metadata.constant(self.expression("{'x': 1, 'x': 2}"))

    def test_overlap_regex_and_missing_parameter_fail(self):
        with self.assertRaises(ValueError):
            metadata.match_value({".*": 1, "left.*": 2}, "left_knee_joint")
        with self.assertRaises(ValueError):
            metadata.match_value({"right.*": 2}, "left_knee_joint")
        self.assertEqual(metadata.match_value({}, "joint", default=0.), 0.)

    def test_explicit_keyword_only(self):
        with self.assertRaises(ValueError):
            metadata.keyword(self.expression("Cfg(**other)"), "stiffness")
        self.assertEqual(metadata.constant(metadata.keyword(self.expression("Cfg(stiffness=30)"), "stiffness")), 30)

    def test_dictionary_entry_does_not_execute_unrelated_call(self):
        node = self.expression("{'asset': ImportAnything(), 'range': (-.01, .01)}")
        self.assertEqual(metadata.constant(metadata.dictionary_entry(node, "range")), (-.01, .01))
        with self.assertRaises(ValueError):
            metadata.dictionary_entry(node, "missing")

    def test_non_numeric_arithmetic_rejected(self):
        with self.assertRaises(ValueError):
            metadata.constant(self.expression("'x' * 2000"))

    def test_hash_drift_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evidence"; path.write_text("before")
            frozen = {}; metadata.checked_file(path, frozen)
            path.write_text("after")
            with self.assertRaises(ValueError):
                metadata.checked_file(path, frozen)

    def test_changed_pinned_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eval.json"; path.write_text("{}")
            with mock.patch.object(metadata, "EVIDENCE", path), self.assertRaisesRegex(ValueError, "evidence changed"):
                metadata.load_static_profile({})

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metadata.json"; path.write_text("keep")
            with self.assertRaises(SystemExit), mock.patch.object(metadata, "build_metadata") as build:
                metadata.main(["--archive", "unused", "--checkpoint", "unused", "--output", str(path)])
            build.assert_not_called()
            self.assertEqual(path.read_text(), "keep")

    def test_hash_bound_local_profile(self):
        frozen = {}; trees, joints, bodies = metadata.load_static_profile(frozen)
        profile = metadata.static_configuration(trees, list(joints), list(bodies))
        self.assertEqual(profile["policy_architecture"], {
            "embedding_dim": 256, "num_heads": 4, "ff_dim": 256, "num_layers": 4,
            "task_embedder_hidden_dims": [], "prop_obs_dim": 64, "task_obs_dim": 267,
            "action_dim": 29, "output_dim": 29, "reduced_task_dim": None})
        self.assertEqual(profile["history_buffer_size"], 3)
        self.assertEqual(profile["future_idx"], [0, 1, 2, 3, 4, 5])
        self.assertEqual(profile["training_future_idx"], [0, 1, 2, 3, 4, -1])
        self.assertEqual(profile["step_dt"], .02)
        self.assertEqual([sum(row) for row in profile["mode_table"]], [1, 2, 3, 4, 5, 6, 7, 14])
        self.assertEqual(profile["action_names"], list(joints))
        self.assertEqual(len(bodies), 30)
        self.assertIn(str(metadata.EVIDENCE), frozen)
        self.assertEqual(profile["joint_velocity_observation_scale"], .05)
        self.assertEqual(profile["training_random_future_range_frames"], [5, 33])
        self.assertEqual(profile["training_default_joint_randomization_rad"], [-.01, .01])
        for joint, offset, scale, stiffness, torque in zip(joints, profile["default_dof_pos"],
                profile["action_scale"], profile["stiffness"], profile["torque_limit"]):
            if joint == "waist_yaw_joint":
                self.assertEqual((offset, scale, stiffness, torque), (0., .25, 100., 88.))
            elif "knee" in joint:
                self.assertEqual(offset, .669)
                self.assertEqual(torque, 139.)
            elif "hip_pitch" in joint:
                self.assertEqual(offset, -.312)
            elif "wrist_yaw" in joint:
                self.assertEqual(torque, 5.)


if __name__ == "__main__":
    unittest.main()
