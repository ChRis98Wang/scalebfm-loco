"""Actual CPU Mink/Clarabel QP on a synthetic in-memory model; never mj_step.

Run with the existing pinned UMR Python, not a newly installed solver. Missing
Mink skips this separate integration module in dependency-light test suites.
"""
import importlib.util
import unittest
import numpy as np
from scripts import umr_output_rate_limit_v3 as rate


@unittest.skipUnless(importlib.util.find_spec("mink") is not None, "Existing Mink environment required")
class MinkIntegrationTests(unittest.TestCase):
    def setUp(self):
        import mink
        import mujoco
        from mink.limits.limit import Constraint
        self.mink = mink
        children = "".join(f'<body name="body_{i}" pos="0 0 .03"><joint name="{name}" type="hinge" axis="0 0 1" limited="true" range="-3 3"/><geom type="sphere" size=".01" mass=".1"/></body>'
                           for i, name in enumerate(rate.JOINT_NAMES))
        xml = '<mujoco><compiler angle="radian"/><worldbody><body name="root"><freejoint/><geom type="sphere" size=".02" mass="1"/>' + children + '</body></worldbody></mujoco>'
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.cfg = mink.Configuration(self.model)
        self.q0 = self.cfg.q.copy()
        self.limit = rate.output_rate_limit_class(mink.Limit, Constraint)(self.model)
        self.task = mink.PostureTask(self.model, cost=1.)
        target = self.q0.copy(); target[7:] = 1.
        self.task.set_target(target)

    def solve_six(self, limits, dt=.02):
        for _ in range(6):
            v = self.mink.solve_ik(self.cfg, [self.task], dt, "clarabel", damping=1e-3, limits=limits)
            self.cfg.integrate_inplace(v, dt)

    def test_real_qp_respects_whole_output_frame_budget_and_advances(self):
        self.limit.begin_frame(0, None); self.limit.finish_frame(self.q0)
        previous = self.q0.copy()
        for frame, dt in ((1, .02), (2, .003)):
            self.limit.begin_frame(frame, previous)
            self.solve_six([self.mink.ConfigurationLimit(self.model), self.limit], dt)
            q = self.cfg.q.copy()
            self.limit.finish_frame(q)
            self.assertTrue(np.all(np.abs(q[7:]-previous[7:]) <= .24+2e-6))
            self.assertTrue(np.all(q[7:]-previous[7:] > .239))
            previous = q.copy()
        self.assertTrue(np.all(previous[7:] > .478))
        self.assertEqual(self.limit.statistics()["interval_count"], 2)
        np.testing.assert_array_equal(self.q0[7:], 0)

    def test_ordinary_velocity_limit_reissues_budget_each_inner_iteration(self):
        ordinary = self.mink.VelocityLimit(self.model, {name: 12. for name in rate.JOINT_NAMES})
        self.solve_six([ordinary])
        self.assertTrue(np.all(self.cfg.q[7:] > .9))
        self.assertGreater(np.max(np.abs(self.cfg.q[7:]-self.q0[7:])), .24)

    def test_zero_position_cost_removes_target_translation_from_qp(self):
        rotation = self.mink.SO3.from_rpy_radians(.2, -.4, .1)
        task = self.mink.FrameTask("body_28", "body", position_cost=0., orientation_cost=10., lm_damping=1.)
        task.set_target(self.mink.SE3.from_rotation_and_translation(rotation, np.zeros(3)))
        first = task.compute_qp_objective(self.cfg)
        task.set_target(self.mink.SE3.from_rotation_and_translation(rotation, np.array([5., -2., 3.])))
        second = task.compute_qp_objective(self.cfg)
        for a, b in zip(first, second):
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
