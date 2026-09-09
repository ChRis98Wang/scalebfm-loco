from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts import audit_packed_retarget_pilot as audit


class PackedRetargetAuditTests(unittest.TestCase):
    def test_reused_collision_audit_schema(self):
        import mujoco
        model = mujoco.MjModel.from_xml_path(str(
            audit.ROOT / "ScaleRetarget/assets/unitree_g1/g1_mocap_29dof.xml"))
        result = audit.refresh.audit_qpos(np.repeat(model.qpos0[None, :], 2, axis=0), model)
        self.assertIn("joint_limit_violations_gt_1e_minus6_rad", result)

    def test_unique_nonsequential_joint_order(self):
        source = np.arange(87, dtype=float).reshape(3, 29)
        permutation = np.roll(np.arange(29), 7)
        errors = audit.accumulate_order_errors(np.zeros((29, 29)), source[:, permutation], source)
        np.testing.assert_array_equal(audit.unique_joint_order(errors), permutation)

    def test_ambiguous_joint_mapping_rejected(self):
        with self.assertRaisesRegex(ValueError, "uniquely"):
            audit.unique_joint_order(np.zeros((29, 29)))

    def test_unequal_mapping_frame_counts_rejected(self):
        with self.assertRaisesRegex(ValueError, "frame counts"):
            audit.accumulate_order_errors(np.zeros((29, 29)), np.zeros((2, 29)), np.zeros((3, 29)))

    def test_unequal_paired_frame_counts_rejected(self):
        with self.assertRaisesRegex(ValueError, "frame counts"):
            audit.validate_pair(np.zeros((2, 36)), np.zeros((3, 36)), 0.02)

    def test_constant_shift_allows_float32_rounding_but_not_pose_change(self):
        before = np.zeros((3, 36))
        before[:, 3] = 1
        after = before.copy()
        after[:, 2] += np.float32(0.032)
        after[:, 3] = -1  # Quaternion sign is physically equivalent.
        proof = audit.validate_pair(before, after, 0.032)
        self.assertLess(proof["constant_z_shift_max_abs_error_m"], 2e-6)
        after[1, 7] += 0.01
        with self.assertRaisesRegex(ValueError, "changed non-Z"):
            audit.validate_pair(before, after, 0.032)

    def test_named_joint_reordering_and_root_wxyz(self):
        names = [f"joint{i}" for i in range(29)]
        packed_names = names[::-1]
        archive = {"joint_pos": np.tile(np.arange(29)[::-1], (3, 1)),
                   "reference_root_pos": np.zeros((3, 3)),
                   "reference_root_quat_w": np.tile([1, 0, 0, 0], (3, 1))}
        qpos = audit.packed_qpos(archive, packed_names, names)
        np.testing.assert_array_equal(qpos[:, 7:], np.tile(np.arange(29), (3, 1)))
        np.testing.assert_array_equal(qpos[:, 3:7], archive["reference_root_quat_w"])

    def test_static_ast_does_not_import_robot_module(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "robot.py"
            path.write_text("raise RuntimeError('MUST NOT EXECUTE')\nG1_29DOF_JOINT_NAMES = "
                            + repr([f"joint{i}" for i in range(29)]))
            self.assertEqual(len(audit.static_joint_names(path)), 29)

    def test_source_hash_misbinding_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "motion.npz"
            np.savez(path, format_version=3, fps=50, quaternion_order="wxyz", source_sha256="a" * 64,
                     pipeline_fingerprint="b" * 64, joint_pos=np.zeros((3, 29)),
                     reference_root_pos=np.zeros((3, 3)), reference_root_quat_w=np.tile([1, 0, 0, 0], (3, 1)))
            self.assertEqual(audit.load_packed(path, "a" * 64)["joint_pos"].shape, (3, 29))
            with self.assertRaisesRegex(ValueError, "PKL SHA256 mismatch"):
                audit.load_packed(path, "c" * 64)


if __name__ == "__main__":
    unittest.main()
