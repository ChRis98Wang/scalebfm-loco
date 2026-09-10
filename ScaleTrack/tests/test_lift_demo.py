"""Pure task logic tests; no simulator imports or fake physics acceptance."""
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/pretrain/rsl_rl"))
from lift_demo import LiftSpec, LiftPlanner, box_geometry, contact_filter_paths


class LiftTests(unittest.TestCase):
    def setUp(self):
        self.spec = LiftSpec()
        self.planner = LiftPlanner([[0, 0, .79], [.1, .25, .75], [.1, -.25, .75]], [[1, 0, 0, 0]]*3)

    def measurement(self, **values):
        return dict(box_xyz=list(self.spec.goal_xyz), box_bottom_z=.56, box_tilt_rad=0.,
                    box_speed=0., box_angular_speed=0., left_force=0., right_force=0.,
                    support_force=9.81, other_robot_force=0., root_z=.79, root_up=1.,
                    inside_support_xy=True) | values

    def close(self):
        self.planner.update(2., self.measurement())
        self.planner.update(5., self.measurement())
        self.assertEqual(self.planner.phase, "CLOSE")

    def test_spec(self):
        self.spec.validate()
        self.assertAlmostEqual(self.spec.goal_xyz[2], .68)

    def test_filter_targets_rigid_pelvis_not_articulation_container(self):
        paths = contact_filter_paths(["pelvis", "left_wrist_yaw_link"])
        self.assertEqual(paths[0], "{ENV_REGEX_NS}/Robot/pelvis/pelvis")
        self.assertEqual(paths[1], "{ENV_REGEX_NS}/Robot/pelvis/left_wrist_yaw_link")
        self.assertEqual(paths[2], "{ENV_REGEX_NS}/LiftSupport")

    def test_invalid_spec(self):
        for spec in (LiftSpec(box_mass=-1), LiftSpec(lift_height=.01), LiftSpec(box_xyz=(.43, 0, .1))):
            with self.assertRaises(ValueError):
                spec.validate()

    def test_box_bottom_accounts_for_rotation(self):
        plain = box_geometry([0, 0, 1], [1, 0, 0, 0], self.spec)
        tilted = box_geometry([0, 0, 1], [np.cos(np.pi/8), np.sin(np.pi/8), 0, 0], self.spec)
        self.assertAlmostEqual(plain["bottom_z"], .88)
        self.assertLess(tilted["bottom_z"], plain["bottom_z"])

    def test_box_must_fit_full_footprint(self):
        self.assertTrue(box_geometry(self.spec.goal_xyz, [1, 0, 0, 0], self.spec)["inside_support_xy"])
        self.assertFalse(box_geometry([.60, 0, .68], [1, 0, 0, 0], self.spec)["inside_support_xy"])

    def test_no_contact_cannot_trigger_lift(self):
        self.close()
        self.planner.update(10., self.measurement())
        self.assertEqual(self.planner.reason, "grip_not_established")

    def test_one_sided_contact_is_not_grip(self):
        self.close()
        for now in (5.02, 5.22, 6):
            self.planner.update(now, self.measurement(left_force=10.))
        self.assertEqual(self.planner.phase, "CLOSE")

    def test_contact_gap_resets_duration(self):
        self.close()
        self.planner.update(5.02, self.measurement(left_force=10, right_force=10))
        self.planner.update(5.10, self.measurement())
        self.planner.update(5.22, self.measurement(left_force=10, right_force=10))
        self.assertEqual(self.planner.phase, "CLOSE")
        self.planner.update(5.42, self.measurement(left_force=10, right_force=10))
        self.assertEqual(self.planner.phase, "LIFT")

    def test_reference_lift_is_not_actual_lift(self):
        self.close()
        both = self.measurement(left_force=10, right_force=10)
        self.planner.update(5.02, both)
        self.planner.update(5.22, both)
        self.planner.update(8.22, both)
        self.assertEqual(self.planner.reason, "lift_height_not_reached")

    def test_success_requires_full_physical_sequence(self):
        self.close()
        both = self.measurement(left_force=10, right_force=10)
        for now in (5.02, 5.22):
            self.planner.update(now, both)
        lifted = self.measurement(left_force=10, right_force=10, box_bottom_z=.68,
                                  box_xyz=[.43, 0, .80], support_force=0.)
        for now in (7.22, 7.42, 7.44, 9.44):
            self.planner.update(now, lifted)
        self.assertEqual(self.planner.phase, "LOWER")
        for now in (11.44, 11.64):
            self.planner.update(now, both)
        self.assertEqual(self.planner.phase, "RELEASE")
        for now in (12., 13.):
            self.planner.update(now, self.measurement())
        self.assertEqual(self.planner.phase, "SUCCESS")

    def test_safety_timeout_and_illegal_contact(self):
        for key, value, reason in (("root_z", .2, "robot_safety_guard"),
                                   ("other_robot_force", 1., "box_contact_with_other_robot_body"),
                                   ("box_speed", np.nan, "nonfinite_telemetry")):
            self.setUp()
            self.planner.update(.02, self.measurement(**{key: value}))
            self.assertEqual(self.planner.reason, reason)
        self.setUp()
        self.planner.update(20., self.measurement())
        self.assertEqual(self.planner.reason, "task_timeout")

    def test_targets_never_mutate_input(self):
        self.close()
        measurement = self.measurement()
        before = measurement["box_xyz"].copy()
        targets, quats = self.planner.targets(measurement)
        self.assertEqual(measurement["box_xyz"], before)
        self.assertEqual(targets.shape, (3, 3))
        self.assertEqual(quats.shape, (3, 4))


if __name__ == "__main__":
    unittest.main()
