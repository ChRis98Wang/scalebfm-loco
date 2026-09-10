import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/pretrain/rsl_rl"))
from lift_demo_preload_v3 import LoadBearingLiftPlanner as PreloadLiftPlanner


class LoadBearingPreloadTests(unittest.TestCase):
    def test_force_setpoint_has_margin_above_unchanged_gate(self):
        self.assertEqual(PreloadLiftPlanner.force_target_n, 24.)
        self.assertEqual(PreloadLiftPlanner.preload_min_n, 18.)

    def setUp(self):
        self.p = PreloadLiftPlanner([[0, 0, .79], [.1, .25, .75], [.1, -.25, .75]], [[1, 0, 0, 0]]*3)
        self.m = dict(box_xyz=[.43, 0, .68], box_bottom_z=.56, box_tilt_rad=0.,
                      box_speed=0., box_angular_speed=0., left_force=2., right_force=2.,
                      support_force=9.81, other_robot_force=0., root_z=.79, root_up=1., inside_support_xy=True)
        for now in (2., 5., 5.02, 5.22):
            self.p.update(now, self.m)

    def test_touch_alone_no_longer_starts_lift(self):
        self.assertEqual(self.p.phase, "PRELOAD")
        targets, _ = self.p.targets(self.m)
        np.testing.assert_allclose(targets[1:, 2], .68)

    def test_both_sides_need_load_bearing_force(self):
        for now in (5.24, 5.44):
            self.p.update(now, self.m | {"left_force": 20})
        self.assertEqual(self.p.phase, "PRELOAD")

    def test_high_force_still_needs_continuous_time(self):
        for now in (5.24, 5.42):
            self.p.update(now, self.m | {"left_force": 18., "right_force": 18.})
        self.assertEqual(self.p.phase, "PRELOAD")
        self.p.update(5.44, self.m | {"left_force": 18., "right_force": 18.})
        self.assertEqual(self.p.phase, "LIFT")

    def test_feedback_closing_is_rate_and_workspace_limited(self):
        previous = self.p.side_gaps.copy()
        self.p.update(5.24, self.m)
        delta = previous-self.p.side_gaps
        self.assertTrue(np.all(delta >= 0) and np.all(delta <= .0012 + 1e-12))
        self.assertTrue(np.all(self.p.side_gaps >= .05))

    def test_preload_timeout_is_failure(self):
        self.p.update(9.22, self.m)
        self.assertEqual(self.p.reason, "load_bearing_preload_not_established")

    def test_force_cap_is_failure(self):
        self.p.update(5.24, self.m | {"left_force": 40.1})
        self.assertEqual(self.p.reason, "excessive_clamp_force")


if __name__ == "__main__":
    unittest.main()
