from __future__ import annotations

from pathlib import Path
import unittest

import mujoco
import numpy as np
from omegaconf import DictConfig

from scaleretarget.formatter.kinematic_formatter import KinematicFormatter


ROOT = Path(__file__).resolve().parents[1]
ROBOT = ROOT / "ScaleRetarget/assets/unitree_g1/g1_mocap_29dof.xml"


def configuration(**overrides):
    values = {
        "kinematic_model_device": "cpu", "height_adjust": True,
        "root_offset": False, "robot": {"robot_xml_path": str(ROBOT)},
        "height_adjust_mode": "collision_sole", "ground_offset_m": 0.002,
        "collision_sole_body_names": ["left_ankle_roll_link", "right_ankle_roll_link"],
    }
    values.update(overrides)
    return DictConfig(values)


def trajectory():
    model = mujoco.MjModel.from_xml_path(str(ROBOT))
    qpos = np.repeat(model.qpos0[None, :], 3, axis=0)
    qpos[:, :3] += [[0, 0, -0.10], [0.05, 0.02, -0.05], [0.11, 0.03, 0.0]]
    return qpos


def as_qpos(motion):
    root_rot = motion["root_rot"]
    if motion["formatter_metadata"]["quaternion_order"] == "xyzw":
        root_rot = root_rot[:, [3, 0, 1, 2]]
    return np.concatenate([motion["root_pos"], root_rot, motion["dof_pos"]], axis=1)


class KinematicFormatterTests(unittest.TestCase):
    def test_actual_g1_spheres_are_selected_and_visual_meshes_ignored(self):
        formatter = KinematicFormatter(configuration())
        self.assertEqual(len(formatter.collision_geom_ids), 8)
        self.assertTrue(all(
            formatter.collision_model.geom_type[i] == mujoco.mjtGeom.mjGEOM_SPHERE
            for i in formatter.collision_geom_ids
        ))

    def test_one_constant_shift_places_collision_minimum_at_clearance(self):
        qpos = trajectory()
        original = qpos.copy()
        formatter = KinematicFormatter(configuration())
        before = formatter._collision_lowest_z(qpos)
        result = formatter.format(qpos, {"fps": 50})
        after = as_qpos(result)
        self.assertAlmostEqual(formatter._collision_lowest_z(after), 0.002, places=12)
        np.testing.assert_array_equal(qpos, original)
        np.testing.assert_array_equal(after[:, 3:], original[:, 3:])
        np.testing.assert_array_equal(after[:, :2], original[:, :2])
        np.testing.assert_allclose(after[:, 2] - original[:, 2], 0.002 - before, atol=1e-14)
        np.testing.assert_allclose(np.diff(after[:, 2]), np.diff(original[:, 2]), atol=1e-14)
        metadata = result["formatter_metadata"]
        self.assertEqual(metadata["measured_minimum_z_m"], before)
        self.assertFalse(metadata["contact_or_dynamics_validated"])

    def test_input_and_output_fields_do_not_alias(self):
        qpos = trajectory()
        original = qpos.copy()
        result = KinematicFormatter(configuration(root_offset=True)).format(qpos, {"fps": 50})
        np.testing.assert_array_equal(result["root_pos"][0, :2], [0, 0])
        for name in ("root_pos", "root_rot", "dof_pos"):
            self.assertFalse(np.shares_memory(qpos, result[name]))
            result[name][:] = 99
        np.testing.assert_array_equal(qpos, original)

    def test_alignment_is_idempotent(self):
        formatter = KinematicFormatter(configuration())
        once = formatter.format(trajectory(), {"fps": 50})
        twice = formatter.format(as_qpos(once), {"fps": 50})
        np.testing.assert_allclose(as_qpos(twice), as_qpos(once), atol=1e-14)
        self.assertAlmostEqual(twice["formatter_metadata"]["applied_constant_z_offset_m"], 0, places=12)

    def test_geometry_support_includes_orientation_for_all_supported_types(self):
        formatter = KinematicFormatter(configuration(height_adjust=False))
        cases = (
            ("sphere", "0.05", 0.05, 0.05),
            ("capsule", "0.05 0.10", 0.15, 0.05),
            ("cylinder", "0.05 0.10", 0.10, 0.05),
            ("ellipsoid", "0.05 0.10 0.15", 0.15, 0.05),
            ("box", "0.05 0.10 0.15", 0.15, 0.05),
            ("mesh", "", 0.15, 0.05),
        )
        vertices = " ".join(
            f"{x} {y} {z}" for x in (-0.05, 0.05)
            for y in (-0.1, 0.1) for z in (-0.15, 0.15)
        )
        for kind, size, vertical, horizontal in cases:
            with self.subTest(kind=kind):
                asset = f'<asset><mesh name="sole" vertex="{vertices}"/></asset>' if kind == "mesh" else ""
                geometry = f'type="{kind}" mesh="sole"' if kind == "mesh" else f'type="{kind}" size="{size}"'
                model = mujoco.MjModel.from_xml_string(
                    f'<mujoco>{asset}<worldbody><body pos="0 0 1">'
                    f'<freejoint/><geom {geometry}/></body></worldbody></mujoco>'
                )
                formatter.collision_model = model
                formatter.collision_data = mujoco.MjData(model)
                mujoco.mj_kinematics(model, formatter.collision_data)
                self.assertAlmostEqual(formatter._geom_lowest_z(0), 1.0 - vertical, places=7)
                formatter.collision_data.qpos[3:7] = [2 ** -0.5, 0, 2 ** -0.5, 0]
                mujoco.mj_kinematics(model, formatter.collision_data)
                self.assertAlmostEqual(formatter._geom_lowest_z(0), 1.0 - horizontal, places=7)

    def test_legacy_default_preserves_body_origin_height(self):
        cfg = configuration(ground_offset_m=0.0)
        del cfg["height_adjust_mode"]
        formatter = KinematicFormatter(cfg)
        self.assertEqual(formatter.height_adjust_mode, "body_origin")
        self.assertIsNone(formatter.collision_model)
        result = formatter.format(trajectory(), {"fps": 50})
        self.assertEqual(result["formatter_metadata"]["height_adjust_mode"], "body_origin")
        collision = KinematicFormatter(configuration())
        self.assertLess(collision._collision_lowest_z(as_qpos(result)), -0.010)

    def test_legacy_fk_is_independent_of_requested_output_quaternion_order(self):
        qpos = trajectory()
        angle = 0.4
        qpos[:, 3:7] = [np.cos(angle / 2), np.sin(angle / 2), 0, 0]
        config = dict(height_adjust_mode="body_origin", ground_offset_m=0.0)
        xyzw = KinematicFormatter(configuration(**config)).format(qpos, {"fps": 50})
        wxyz = KinematicFormatter(configuration(**config, quat_order="wxyz")).format(qpos, {"fps": 50})
        np.testing.assert_array_equal(xyzw["root_pos"], wxyz["root_pos"])
        np.testing.assert_array_equal(as_qpos(xyzw), as_qpos(wxyz))

    def test_disabled_adjustment_does_not_load_robot_or_shift(self):
        formatter = KinematicFormatter(configuration(height_adjust=False, robot={"robot_xml_path": "/absent"}))
        qpos = trajectory()
        result = formatter.format(qpos, {"fps": 50})
        np.testing.assert_array_equal(as_qpos(result), qpos)
        self.assertIsNone(result["formatter_metadata"]["measured_minimum_z_m"])

    def test_invalid_configuration_fails_closed(self):
        for override in (
            {"height_adjust_mode": "unknown"}, {"ground_offset_m": -0.01},
            {"ground_offset_m": float("nan")}, {"collision_sole_body_names": []},
            {"collision_sole_body_names": ["not_a_foot"]},
            {"collision_sole_body_names": ["left_toe_link"]},
            {"collision_sole_body_names": ["left_ankle_roll_link"] * 2},
            {"collision_sole_body_names": "left_ankle_roll_link"}, {"quat_order": "invalid"},
        ):
            with self.subTest(override=override), self.assertRaises(ValueError):
                KinematicFormatter(configuration(**override))

    def test_invalid_input_fails_closed(self):
        formatter = KinematicFormatter(configuration())
        invalid = [np.zeros((0, 36)), np.zeros((2, 6)), np.zeros((2, 36))]
        nonfinite = trajectory()
        nonfinite[0, 7] = np.nan
        invalid.append(nonfinite)
        invalid.append(trajectory()[:, :-1])
        for qpos in invalid:
            with self.subTest(shape=qpos.shape), self.assertRaises(ValueError):
                formatter.format(qpos, {"fps": 50})


if __name__ == "__main__":
    unittest.main()
