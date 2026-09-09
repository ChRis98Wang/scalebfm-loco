"""Synthetic NumPy QP/lifecycle tests; imports no physics or Mink runtime."""
from dataclasses import dataclass, replace
from types import SimpleNamespace
import unittest

import numpy as np

from scripts import umr_output_rate_limit_v3 as rate


class FakeModel:
    def __init__(self, reverse=False):
        self.nq, self.nv, self.njnt = 36, 35, 30
        self.names = ["floating_base"] + list(reversed(rate.JOINT_NAMES) if reverse else rate.JOINT_NAMES)
        self.jnt_type = np.array([0] + [3] * 29)
        self.jnt_qposadr = np.r_[0, np.arange(7, 36)]
        self.jnt_dofadr = np.r_[0, np.arange(6, 35)]

    def joint(self, name):
        index = self.names.index(name) if isinstance(name, str) else int(name)
        return SimpleNamespace(id=index, name=self.names[index])


@dataclass
class Constraint:
    G: object = None
    h: object = None

    @property
    def inactive(self):
        return self.G is None and self.h is None


class Limit:
    pass


def qpos(value=0.):
    q = np.zeros(36, dtype=np.float64)
    q[3] = 1.
    q[7:] = value
    return q


class LayoutAndBoxTests(unittest.TestCase):
    def test_exact_named_layout_and_reordered_models_are_supported(self):
        for reverse in (False, True):
            model = FakeModel(reverse)
            layout = rate.validate_hinge_layout(model)
            self.assertEqual(layout.joint_names, rate.JOINT_NAMES)
            for name, jid, qadr, dadr in zip(layout.joint_names, layout.joint_ids,
                                            layout.qpos_addresses, layout.dof_addresses):
                self.assertEqual(model.joint(name).id, jid)
                self.assertEqual(qadr, model.jnt_qposadr[jid])
                self.assertEqual(dadr, model.jnt_dofadr[jid])

    def test_missing_names_duplicates_wrong_types_and_addresses_are_rejected(self):
        changes = [lambda m: setattr(m, "nq", 35), lambda m: setattr(m, "nv", 36),
                   lambda m: setattr(m, "njnt", 31), lambda m: setattr(m, "nq", 36.),
                   lambda m: m.names.__setitem__(5, "wrong_joint"),
                   lambda m: m.names.__setitem__(5, m.names[6]),
                   lambda m: m.jnt_type.__setitem__(1, 2), lambda m: m.jnt_type.__setitem__(0, 1),
                   lambda m: m.jnt_qposadr.__setitem__(2, 7), lambda m: m.jnt_dofadr.__setitem__(2, 6),
                   lambda m: m.jnt_qposadr.__setitem__(1, 8), lambda m: m.jnt_dofadr.__setitem__(0, 1),
                   lambda m: setattr(m, "jnt_type", m.jnt_type.astype(float))]
        for change in changes:
            model = FakeModel(); change(model)
            with self.subTest(change=change), self.assertRaises(ValueError):
                rate.validate_hinge_layout(model)

    def test_pure_layout_rejects_mutable_fields_and_noninteger_dimensions(self):
        layout = rate.validate_hinge_layout(FakeModel())
        for changes in ({"nq": 36.}, {"nv": 35.}, {"nq": True},
                        {"joint_names": list(layout.joint_names)},
                        {"joint_ids": list(layout.joint_ids)},
                        {"qpos_addresses": list(layout.qpos_addresses)},
                        {"dof_addresses": list(layout.dof_addresses)}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(layout, **changes)

    def test_delta_q_formula_has_current_iterate_offset_and_unconstrained_free_base(self):
        model = FakeModel(True); layout = rate.validate_hinge_layout(model)
        anchor, current = qpos(), qpos()
        anchor[7:] = np.linspace(-.5, .5, 29)
        current[7:] = anchor[7:] + np.linspace(-.2, .2, 29)
        G, h = rate.output_box_inequalities(current, anchor, layout)
        self.assertEqual(G.shape, (58, 35)); self.assertEqual(h.shape, (58,))
        self.assertEqual(G.dtype, np.float64); self.assertEqual(h.dtype, np.float64)
        np.testing.assert_array_equal(G[:, :6], 0)
        for i, (qa, da) in enumerate(zip(layout.qpos_addresses, layout.dof_addresses)):
            self.assertEqual(G[i, da], 1.); self.assertEqual(G[29+i, da], -1.)
            self.assertAlmostEqual(h[i], anchor[qa] + .24 - current[qa])
            self.assertAlmostEqual(h[29+i], current[qa] - anchor[qa] + .24)
        # Endpoints of the anchored box are feasible for every named hinge.
        for direction in (-1., 1.):
            delta = np.zeros(35)
            delta[list(layout.dof_addresses)] = anchor[list(layout.qpos_addresses)] + direction*.24 - current[list(layout.qpos_addresses)]
            self.assertTrue(np.all(G @ delta <= h + 1e-14))

    def test_current_outside_box_requires_correction_instead_of_resetting_budget(self):
        layout = rate.validate_hinge_layout(FakeModel())
        G, h = rate.output_box_inequalities(qpos(.3), qpos(), layout)
        np.testing.assert_allclose(h[:29], -.06)
        self.assertFalse(np.all(G @ np.zeros(35) <= h))

    def test_numeric_nonfinite_shapes_and_nonunit_base_fail_closed(self):
        layout = rate.validate_hinge_layout(FakeModel())
        bad = [np.zeros(35), np.zeros((1, 36)), np.ones(36, dtype=bool),
               np.array(["0"] * 36), qpos(), qpos(), qpos()]
        bad[-3][10] = np.nan; bad[-2][10] = np.inf; bad[-1][3] = 2
        for value in bad:
            for first in (True, False):
                with self.subTest(first=first), self.assertRaises(ValueError):
                    rate.output_box_inequalities(value if first else qpos(), qpos() if first else value, layout)


class LimitLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.model = FakeModel()
        self.cls = rate.output_rate_limit_class(Limit, Constraint)
        self.limit = self.cls(self.model)
        self.config = SimpleNamespace(model=self.model, q=qpos())

    def bootstrap(self, output=None):
        self.limit.begin_frame(0, None)
        return self.limit.finish_frame(qpos() if output is None else output)

    def test_explicit_bootstrap_is_inactive_and_stores_first_output(self):
        with self.assertRaises(ValueError):
            self.limit.compute_qp_inequalities(self.config, .02)
        with self.assertRaises(ValueError):
            self.limit.finish_frame(qpos())
        with self.assertRaises(ValueError):
            self.limit.begin_frame(0, qpos())
        self.limit.begin_frame(0, None)
        self.assertTrue(self.limit.compute_qp_inequalities(self.config, .02).inactive)
        report = self.limit.finish_frame(qpos(.8))
        self.assertTrue(report["bootstrap"])
        self.assertFalse(report["rate_limit_applicable"])
        self.assertEqual(report["interval_count"], 0)
        self.limit.begin_frame(1, qpos(.8))
        self.assertFalse(self.limit.compute_qp_inequalities(self.config, .02).inactive)

    def test_six_inner_qp_steps_cannot_multiply_the_output_budget(self):
        self.bootstrap(); anchor = qpos(); self.limit.begin_frame(1, anchor)
        for _ in range(6):
            constraint = self.limit.compute_qp_inequalities(self.config, .02)
            # Exact minimizer of identity-Hessian QP towards +0.2 per hinge.
            # This projection is a TEST solver, not postprocessing in production.
            delta = np.zeros(35)
            delta[6:] = np.minimum(.2, constraint.h[:29])
            self.assertTrue(np.all(constraint.G @ delta <= constraint.h + 1e-14))
            mink_velocity = delta / .02
            self.config.q[7:] += mink_velocity[6:] * .02
        np.testing.assert_allclose(self.config.q[7:], .24)
        report = self.limit.finish_frame(self.config.q)
        self.assertTrue(report["accepted_with_fixed_tolerance"])
        self.assertEqual(self.limit.statistics()["interval_count"], 1)

    def test_solver_dt_never_changes_fixed_output_budget(self):
        self.bootstrap(); self.limit.begin_frame(1, qpos())
        reference = self.limit.compute_qp_inequalities(self.config, .02)
        for dt in (.001, .01, .2, 1.):
            c = self.limit.compute_qp_inequalities(self.config, dt)
            np.testing.assert_array_equal(c.G, reference.G)
            np.testing.assert_array_equal(c.h, reference.h)
        for dt in (0, -.02, float("nan"), float("inf"), True, "0.02"):
            with self.assertRaises(ValueError):
                self.limit.compute_qp_inequalities(self.config, dt)

    def test_missing_stale_duplicate_skipped_and_mutated_anchors_are_rejected(self):
        self.bootstrap()
        for index, previous in ((1, None), (2, qpos()), (0, None), (True, qpos()), (1, qpos(.01))):
            with self.assertRaises(ValueError):
                self.limit.begin_frame(index, previous)
        anchor = qpos(); self.limit.begin_frame(1, anchor); anchor[7] = 9
        copy = self.limit.anchor_qpos; copy[8] = 9
        np.testing.assert_array_equal(self.limit.anchor_qpos, qpos())
        with self.assertRaises(ValueError):
            self.limit.begin_frame(1, qpos())
        with self.assertRaises(ValueError):
            self.limit.statistics()
        self.limit.finish_frame(qpos(.1))
        with self.assertRaises(ValueError):
            self.limit.compute_qp_inequalities(self.config, .02)

    def test_failed_output_is_not_clipped_or_advanced_and_cannot_reset_anchor(self):
        self.bootstrap(); self.limit.begin_frame(1, qpos())
        output = qpos(.240003); before = output.copy()
        with self.assertRaises(rate.OutputRateViolation) as caught:
            self.limit.finish_frame(output)
        self.assertEqual(caught.exception.report["tolerance_exceedance_joint_count"], 29)
        self.assertEqual(self.limit.completed_frames, 1)
        np.testing.assert_array_equal(output, before)
        with self.assertRaises(ValueError):
            self.limit.begin_frame(1, qpos())

    def test_fixed_tolerance_accepts_only_2e_minus6_rad_numerical_slack(self):
        self.bootstrap(); self.limit.begin_frame(1, qpos())
        report = self.limit.finish_frame(qpos(.240001))
        self.assertTrue(report["accepted_with_fixed_tolerance"])
        self.assertEqual(report["exact_exceedance_joint_count"], 29)
        self.assertEqual(report["tolerance_exceedance_joint_count"], 0)

    def test_tolerance_boundary_is_fixed_in_radians(self):
        boundary = rate.MAX_OUTPUT_STEP_RAD + rate.STEP_TOLERANCE_RAD
        self.bootstrap(); self.limit.begin_frame(1, qpos())
        report = self.limit.finish_frame(qpos(boundary))
        self.assertTrue(report["accepted_with_fixed_tolerance"])
        self.assertEqual(report["tolerance_exceedance_joint_count"], 0)
        other = self.cls(self.model)
        other.begin_frame(0, None); other.finish_frame(qpos())
        other.begin_frame(1, qpos())
        with self.assertRaises(rate.OutputRateViolation):
            other.finish_frame(qpos(np.nextafter(boundary, np.inf)))

    def test_different_or_changed_model_is_rejected(self):
        self.bootstrap(); self.limit.begin_frame(1, qpos())
        with self.assertRaises(ValueError):
            self.limit.compute_qp_inequalities(SimpleNamespace(model=FakeModel(), q=qpos()), .02)
        self.model.jnt_dofadr[1] = 8
        with self.assertRaises(ValueError):
            self.limit.compute_qp_inequalities(self.config, .02)


class OutputIntervalTests(unittest.TestCase):
    def setUp(self):
        self.layout = rate.validate_hinge_layout(FakeModel())

    def test_first_output_interval_is_not_skipped(self):
        frames = np.stack([qpos(), qpos(.25), qpos(.25)])
        report = rate.interval_statistics(frames, self.layout)
        self.assertEqual(report["interval_count"], 2)
        self.assertTrue(report["first_interval_included"])
        self.assertEqual(report["tolerance_exceedance_frame_indices"], [1])
        self.assertEqual(report["tolerance_exceedance_joint_intervals"], 29)
        self.assertFalse(report["accepted_with_fixed_tolerance"])
        self.assertEqual(report["intervals"][0]["anchor_frame_index"], 0)

    def test_exact_and_tolerance_counts_are_separate(self):
        frames = np.stack([qpos(), qpos()])
        frames[1, 7:10] = [.24, .240001, .240003]
        report = rate.interval_statistics(frames, self.layout)
        self.assertEqual(report["exact_exceedance_joint_intervals"], 2)
        self.assertEqual(report["tolerance_exceedance_joint_intervals"], 1)
        self.assertEqual(report["per_joint"][rate.JOINT_NAMES[1]]["tolerance_exceedance_count"], 0)

    def test_single_frame_bootstrap_is_explicitly_not_applicable(self):
        report = rate.interval_statistics(qpos()[None], self.layout)
        self.assertEqual(report["interval_count"], 0)
        self.assertFalse(report["rate_limit_applicable"])
        self.assertFalse(report["first_interval_included"])
        self.assertIsNone(report["max_observed_rate_rad_s"])

    def test_no_angle_wrapping_or_frame_decimation(self):
        frames = np.stack([qpos(), qpos(2*np.pi)])
        self.assertGreater(rate.interval_statistics(frames, self.layout)["max_observed_rate_rad_s"], 300.)
        for indices in ([1, 2], [0, 2], [0., 1.], [[0, 1]]):
            with self.assertRaises(ValueError):
                rate.interval_statistics(frames, self.layout, frame_indices=indices)
        with self.assertRaises(ValueError):
            rate.interval_statistics(np.empty((0, 36)), self.layout)


if __name__ == "__main__":
    unittest.main()
