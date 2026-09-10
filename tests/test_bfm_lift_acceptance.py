"""Synthetic telemetry tests only; these are not simulator task successes."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.bfm_lift_acceptance import sequence_checks


class LiftAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.scene = dict(support_xyz=[.43, 0, .28], support_size=[.4, .46, .56],
                          box_size=[.24, .3, .24], box_mass=1., timeout=20.)
        self.rows = []
        for frame in range(1, 901):
            t = frame*.02
            phase = next(label for end, label in ((2., "SETTLE"), (5., "APPROACH"), (7., "CLOSE"),
                (9., "PRELOAD"), (11., "LIFT"), (13.02, "HOLD"), (16., "LOWER"),
                (18., "RELEASE"), (20., "SUCCESS")) if t < end)
            airborne = 10. <= t <= 13.02
            z = .8 if airborne else .68
            grip = 20. if 7. <= t <= 16. else 0.
            self.rows.append(dict(time_s=t, phase=phase, box_xyz=[.43, 0, z], box_bottom_z=z-.12,
                box_tilt_rad=0., box_speed=0., box_angular_speed=0., inside_support_xy=True,
                left_force=grip, right_force=grip, left_force_peak=grip, right_force_peak=grip,
                support_force=0. if airborne else 9.81, support_force_peak=0. if airborne else 9.81,
                root_z=.79, root_up=1., other_robot_force=0., wrist_xyz=[[.43, .18, z], [.43, -.18, z]]))

    def test_full_synthetic_sequence(self):
        self.assertTrue(all(sequence_checks(self.rows, self.scene).values()))

    def test_task_labels_cannot_replace_actual_clearance(self):
        for row in self.rows:
            row["box_bottom_z"] = .56
        self.assertFalse(sequence_checks(self.rows, self.scene)["measured_8cm_hold_2s"])

    def test_release_pulse_not_hidden_in_minimum(self):
        self.rows[-5]["left_force_peak"] = 1.
        self.assertFalse(sequence_checks(self.rows, self.scene)["released_at_xyz_goal_for_1s"])

    def test_full_footprint_required(self):
        self.rows[-1]["inside_support_xy"] = False
        self.assertFalse(sequence_checks(self.rows, self.scene)["released_at_xyz_goal_for_1s"])

    def test_wrong_xyz_goal_rejected(self):
        self.rows[-1]["box_xyz"] = [.43, .04, .68]
        self.assertFalse(sequence_checks(self.rows, self.scene)["released_at_xyz_goal_for_1s"])

    def test_airborne_support_pulse_rejected(self):
        self.rows[600]["support_force_peak"] = 1.
        self.assertFalse(sequence_checks(self.rows, self.scene)["measured_8cm_hold_2s"])

    def test_missing_lower_or_release_not_success(self):
        for row in self.rows:
            if row["phase"] == "LOWER":
                row["phase"] = "HOLD"
        self.assertFalse(sequence_checks(self.rows, self.scene)["full_phase_sequence"])

    def test_slip_still_failure(self):
        self.rows[600]["wrist_xyz"] = [[.50, .18, .8], [.50, -.18, .8]]
        self.assertFalse(sequence_checks(self.rows, self.scene)["loaded_relative_slip_le_5cm"])


if __name__ == "__main__":
    unittest.main()
