"""Logic-only tests. They are not evidence of physical lifting success."""
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/pretrain/rsl_rl"))
from lift_demo_balanced import BalanceConfig, BalancedLiftPlanner


class BalancedLiftTests(unittest.TestCase):
    def setUp(self):
        self.p = BalancedLiftPlanner([[0, 0, .79], [.1, .25, .75], [.1, -.25, .75]], [[1, 0, 0, 0]]*3)
        self.m = dict(box_xyz=[.43, 0, .68], box_quat_wxyz=[1., 0., 0., 0.],
                      wrist_xyz=[[.34, .18, .68], [.34, -.18, .70]],
                      box_bottom_z=.56, box_tilt_rad=0., box_speed=0., box_angular_speed=0.,
                      left_force=2., right_force=2., support_force=9.81, other_robot_force=0.,
                      root_z=.79, root_up=1., inside_support_xy=True)

    def preload(self):
        for now in (2., 5., 5.02, 5.22):
            self.p.update(now, self.m)
        self.assertEqual(self.p.phase, "PRELOAD")

    def test_invalid_configs(self):
        for cfg in (BalanceConfig(wrist_offset_x_m=.1), BalanceConfig(pitch_feedback_gain=float("nan")),
                    BalanceConfig(maximum_x_bias_m=.2), BalanceConfig(bias_speed_mps=-1.)):
            with self.assertRaises(ValueError):
                cfg.validate()

    def test_no_early_bias_windup(self):
        self.p.update(2., self.m)
        self.p.update(2.98, self.m)
        np.testing.assert_array_equal(self.p.tangential_bias, 0.)

    def test_bias_corrects_x_lag_and_z_asymmetry_at_bounded_rate(self):
        self.p.update(2., self.m)
        self.p.update(3., self.m)
        bias = self.p.tangential_bias
        self.assertTrue(np.all(bias[:, 0] > 0))
        self.assertLess(bias[1, 1], 0.)
        self.assertLessEqual(np.abs(bias).max(), .035*.02+1e-12)

    def test_bias_freezes_after_approach(self):
        self.p.update(2., self.m)
        self.p.update(3., self.m)
        previous = self.p.tangential_bias.copy()
        self.p.update(5., self.m)
        self.p.update(5.02, self.m)
        np.testing.assert_array_equal(self.p.tangential_bias, previous)

    def test_preload_threshold_and_duration_unchanged(self):
        self.preload()
        for now in (5.24, 5.44):
            self.p.update(now, self.m | dict(left_force=20., right_force=17.9))
        self.assertEqual(self.p.phase, "PRELOAD")
        for now in (5.46, 5.66):
            self.p.update(now, self.m | dict(left_force=20., right_force=20.))
        self.assertEqual(self.p.phase, "LIFT")

    def test_anchor_does_not_chase_object_drift(self):
        self.preload()
        before, _ = self.p.targets(self.m)
        after, _ = self.p.targets(self.m | dict(box_xyz=[.48, -.04, .68]))
        np.testing.assert_array_equal(before[:, :2], after[:, :2])

    def test_attitude_feedback_has_restoring_sign(self):
        self.preload()
        angle = .1
        _, quats = self.p.targets(self.m | dict(box_quat_wxyz=[math.cos(angle/2), 0, math.sin(angle/2), 0]))
        self.assertLess(quats[1, 2], 0.)
        points, _ = self.p.targets(self.m | dict(box_quat_wxyz=[math.cos(angle/2), math.sin(angle/2), 0, 0]))
        self.assertLess(points[1, 2], points[2, 2])

    def test_substep_force_peak_is_not_hidden_by_minimum(self):
        self.preload()
        self.p.update(5.24, self.m | dict(left_force_peak=40.01))
        self.assertEqual(self.p.reason, "instantaneous_clamp_force_exceeded")

    def test_tilt_gate_not_relaxed(self):
        self.preload()
        for now in (5.24, 5.44):
            self.p.update(now, self.m | dict(left_force=20., right_force=20.))
        self.p.update(5.46, self.m | dict(box_tilt_rad=math.radians(15.01)))
        self.assertEqual(self.p.reason, "box_tilt_exceeded")

    def test_invalid_added_telemetry_fails_closed(self):
        self.p.update(.02, self.m | dict(box_quat_wxyz=[0, 0, 0, 0]))
        self.assertEqual(self.p.reason, "invalid_balance_telemetry")

    def test_airborne_cannot_hide_intermittent_table_contact(self):
        self.preload()
        both = self.m | dict(left_force=20., right_force=20.)
        for now in (5.24, 5.44):
            self.p.update(now, both)
        lifted = both | dict(box_bottom_z=.68, support_force=0., support_force_peak=1.)
        for now in (5.46, 5.66):
            self.p.update(now, lifted)
        self.assertEqual(self.p.phase, "LIFT")

    def test_release_cannot_hide_intermittent_arm_contact(self):
        self.p.phase = "RELEASE"
        free = self.m | dict(left_force=0., right_force=0., left_force_peak=.3)
        for now in (.02, 1.02):
            self.p.update(now, free)
        self.assertEqual(self.p.phase, "RELEASE")

    def test_forward_profile_is_valid(self):
        BalanceConfig(wrist_offset_x_m=.03).validate()

    def test_clearance_feedback_changes_targets_not_success_criteria(self):
        self.p.balance = BalanceConfig(clearance_feedback_gain_per_s=1.)
        self.preload()
        both = self.m | dict(left_force=20., right_force=20.)
        for now in (5.24, 5.44, 6.44):
            self.p.update(now, both)
        self.assertGreater(self.p.clearance_bias, 0.)
        self.assertLessEqual(self.p.clearance_bias, .06*.02+1e-12)
        self.assertEqual(self.p.phase, "LIFT")
        self.assertEqual(self.p.spec.minimum_clearance, .08)
        self.p.update(8.46, both)
        self.assertEqual(self.p.reason, "lift_height_not_reached")

    def test_default_clearance_feedback_does_not_change_baseline(self):
        self.preload()
        both = self.m | dict(left_force=20., right_force=20.)
        for now in (5.24, 5.44, 6.44):
            self.p.update(now, both)
        self.assertEqual(self.p.clearance_bias, 0.)

    def test_clearance_feedback_never_runs_before_grip(self):
        self.p.balance = BalanceConfig(clearance_feedback_gain_per_s=1.)
        self.preload()
        self.p.update(5.24, self.m)
        self.assertEqual(self.p.clearance_bias, 0.)

    def test_held_clearance_loss_remains_failure(self):
        self.p.phase = "HOLD"
        self.p.update(.02, self.m)
        self.assertEqual(self.p.reason, "lost_airborne_hold")

    def test_landing_force_not_reduced_while_airborne(self):
        self.p.balance = BalanceConfig(landing_force_target_n=6.)
        self.p.phase = "LOWER"
        self.p.update(.02, self.m | dict(box_bottom_z=.65, support_force=0.))
        self.assertFalse(self.p.landing_unload)
        self.assertEqual(self.p.force_target_n, 24.)

    def test_landing_force_reduces_only_on_measured_support(self):
        self.p.balance = BalanceConfig(landing_force_target_n=6.)
        self.p.phase = "LOWER"
        self.p.update(.02, self.m)
        self.assertTrue(self.p.landing_unload)
        self.assertEqual(self.p.force_target_n, 6.)
        self.assertEqual(self.p.phase, "LOWER")
        self.assertEqual(self.p.spec.contact_threshold, .2)

    def test_landing_does_not_reduce_original_preload_gate(self):
        self.p.balance = BalanceConfig(landing_force_target_n=6.)
        self.preload()
        for now in (5.24, 5.44):
            self.p.update(now, self.m | dict(left_force=6., right_force=6.))
        self.assertEqual(self.p.phase, "PRELOAD")
        self.assertEqual(self.p.force_target_n, 24.)

    def test_lower_targets_return_to_xyz_goal_without_changing_anchor(self):
        self.p.balance = BalanceConfig(landing_xy_return_fraction=1.)
        self.p.anchor_xy = np.array([.42, .04])
        self.p.grasp_xyz = np.array(self.m["box_xyz"])
        self.p.phase = "LOWER"
        self.p.time = 2.
        points, _ = self.p.targets(self.m)
        self.assertAlmostEqual((points[1, 1]+points[2, 1])/2, 0.)
        np.testing.assert_array_equal(self.p.anchor_xy, [.42, .04])

    def test_landing_profile_still_rejects_force_peak(self):
        self.p.balance = BalanceConfig(landing_force_target_n=6.)
        self.p.phase = "LOWER"
        self.p.update(.02, self.m | dict(left_force_peak=40.01))
        self.assertEqual(self.p.reason, "instantaneous_clamp_force_exceeded")

    def test_fast_unload_keeps_gap_speed_bound(self):
        self.p.balance = BalanceConfig(landing_force_target_n=1., landing_force_gain_m_per_ns=.012)
        self.p.phase = "LOWER"
        previous = self.p.side_gaps.copy()
        self.p.update(.02, self.m | dict(left_force=20., right_force=20.))
        self.assertLessEqual(abs(self.p.side_gaps-previous).max(), .06*.02+1e-12)
        self.assertEqual(self.p.force_target_n, 1.)
        self.assertEqual(self.p.force_gain_m_per_ns, .012)

    def test_targets_are_finite_and_do_not_mutate_measurements(self):
        self.preload()
        saved = np.asarray(self.m["wrist_xyz"]).copy()
        points, quats = self.p.targets(self.m)
        self.assertTrue(np.isfinite(points).all())
        np.testing.assert_allclose(np.linalg.norm(quats, axis=-1), 1.)
        np.testing.assert_array_equal(saved, self.m["wrist_xyz"])


if __name__ == "__main__":
    unittest.main()
