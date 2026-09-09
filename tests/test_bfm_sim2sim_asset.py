"""Pure CPU XML/FK checks; physics stepping is forbidden in every test."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from scripts import prepare_bfm_sim2sim_asset as asset

METADATA = asset.ROOT / "local/scalebfm_deployment_20260909a/export_v2/policy_metadata.json"


class ExistingOutputTests(unittest.TestCase):
    def test_existing_output_refused_before_any_input_or_model_read(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch("mujoco.MjModel.from_xml_path") as model, \
                self.assertRaises(FileExistsError):
            asset.prepare_asset("/missing/original", "/missing/meta", "/missing/train", directory)
        model.assert_not_called()

    def test_output_with_symlink_ancestor_refused_before_input_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "outside").mkdir()
            (root / "redirect").symlink_to(root / "outside", target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink ancestor"):
                asset.prepare_asset("/missing", "/missing", "/missing", root / "redirect/new")
            self.assertEqual(list((root / "outside").iterdir()), [])


@unittest.skipUnless(METADATA.is_file(), "Local verified export metadata is not present")
class NominalAssetTests(unittest.TestCase):
    def setUp(self):
        self.metadata = json.loads(METADATA.read_text())
        self.bridge = mujoco.MjModel.from_xml_path(str(asset.BRIDGE_XML))
        self.training = mujoco.MjModel.from_xml_path(str(asset.TRAINING_XML))
        self.no_step = mock.patch("mujoco.mj_step", side_effect=AssertionError("Physics stepping forbidden"))
        self.no_step.start()
        self.addCleanup(self.no_step.stop)

    def compare(self):
        return asset.compare_named_fk(self.bridge, self.training, self.metadata["joint_names"], self.metadata["body_names"])

    def test_current_named_fk_matches_but_extra_fixed_bodies_are_explicit(self):
        result = self.compare()
        self.assertEqual(result["result"], "PASS")
        self.assertEqual(result["random_pose_count"], 16)
        self.assertEqual(result["maximum_position_error_m"], 0.)
        self.assertEqual(result["maximum_quaternion_sign_invariant_error"], 0.)
        self.assertFalse(result["physics_stepped"])
        self.assertEqual({b["name"] for b in result["extra_fixed_bridge_bodies"]}, {"pelvis_contour_link", "imu_in_torso"})
        self.assertAlmostEqual(result["model_mass_kg"]["bridge"] - result["model_mass_kg"]["training"], .001)
        self.assertNotEqual(result["free_root_joint_names"]["bridge"], result["free_root_joint_names"]["training"])

    def test_body_local_transform_mismatch_is_blocking(self):
        index = mujoco.mj_name2id(self.bridge, mujoco.mjtObj.mjOBJ_BODY, "left_knee_link")
        self.bridge.body_pos[index, 0] += .001
        with self.assertRaisesRegex(ValueError, "FK mismatch"):
            self.compare()

    def test_joint_axis_mismatch_is_blocking(self):
        index = mujoco.mj_name2id(self.bridge, mujoco.mjtObj.mjOBJ_JOINT, "left_hip_pitch_joint")
        self.bridge.jnt_axis[index] *= -1
        with self.assertRaisesRegex(ValueError, "FK mismatch"):
            self.compare()

    def test_only_exact_two_known_hip_roll_motor_limits_can_change(self):
        changes = asset.torque_changes(self.bridge, self.metadata)
        self.assertEqual({change["joint"] for change in changes}, asset.HIPS)
        for change in changes:
            self.assertEqual(change["before"], [-88., 88.])
            self.assertEqual(change["after"], [-139., 139.])

    def test_another_motor_difference_is_not_silently_repaired(self):
        index = mujoco.mj_name2id(self.bridge, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_hip_pitch_joint")
        self.bridge.actuator_ctrlrange[index] = [-87., 87.]
        with self.assertRaisesRegex(ValueError, "Unexpected motor torque"):
            asset.torque_changes(self.bridge, self.metadata)

    def test_changed_metadata_force_limit_is_blocking(self):
        metadata = copy.deepcopy(self.metadata)
        metadata["torque_limit"][metadata["joint_names"].index("left_hip_roll_joint")] = 140.
        with self.assertRaisesRegex(ValueError, "joint-force limit mismatch"):
            asset.torque_changes(self.bridge, metadata)

    def test_variant_creation_preserves_sources_and_other_xml_properties(self):
        original_sha = asset.sha256(asset.BRIDGE_XML)
        source = asset.parse_tree(asset.BRIDGE_XML)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new_variant"
            receipt = asset.prepare_asset(asset.BRIDGE_XML, METADATA, asset.TRAINING_XML, output)
            self.assertEqual(receipt, json.loads((output / "receipt.json").read_text()))
            self.assertEqual(receipt["result"], "COMPLETE_SIMULATION_ASSET_ONLY")
            self.assertEqual(asset.sha256(asset.BRIDGE_XML), original_sha)
            self.assertFalse(receipt["original_xml_modified"])
            self.assertFalse(receipt["physically_calibrated"])
            self.assertFalse(receipt["physics_stepped"])
            self.assertTrue(receipt["protected_inputs_rechecked"])
            variant_path = Path(receipt["variant_xml"])
            self.assertEqual(asset.sha256(variant_path), receipt["variant_sha256"])
            self.assertEqual(receipt["after_fk"]["maximum_position_error_m"], 0.)
            variant = asset.parse_tree(variant_path)
            compiler = variant.getroot().find("compiler")
            self.assertTrue(Path(compiler.get("meshdir")).is_absolute())
            compiler.set("meshdir", source.getroot().find("compiler").get("meshdir"))
            for motor in variant.getroot().findall("./actuator/motor"):
                if motor.get("name") in asset.HIPS:
                    self.assertEqual(motor.get("ctrlrange"), "-139 139")
                    motor.set("ctrlrange", "-88 88")
            self.assertEqual(ET.tostring(variant.getroot()), ET.tostring(source.getroot()))
            with self.assertRaises(FileExistsError):
                asset.prepare_asset(asset.BRIDGE_XML, METADATA, asset.TRAINING_XML, output)

    def test_unverified_or_wrong_fk_metadata_fails_before_creating_output(self):
        for change in ({"export_verification": "FAIL"}, {"fk_xml_sha256": "0" * 64}):
            with tempfile.TemporaryDirectory() as directory:
                metadata = {**self.metadata, **change}
                path = Path(directory) / "policy_metadata.json"
                path.write_text(json.dumps(metadata))
                output = Path(directory) / "never_created"
                with self.subTest(change=change), self.assertRaisesRegex(ValueError, "verified, hash-bound"):
                    asset.prepare_asset(asset.BRIDGE_XML, path, asset.TRAINING_XML, output)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
