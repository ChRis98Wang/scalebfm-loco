"""Pure fixtures for the opt-in UMR v3 stage-2 controls.

No UMR, Mink, MuJoCo, physics, or solver is imported or executed. The fake
base follows the upstream warmup call followed by normal frames 0 through N-1.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
import unittest

import numpy as np

from scripts import umr_stage2_controls_v3 as controls


def rotation(axis, angle):
    """Independent Rodrigues formula, including noncommuting world rotations."""
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return (np.cos(angle) * np.eye(3)
            + (1. - np.cos(angle)) * np.outer(axis, axis)
            + np.sin(angle) * skew)


def source_rotations(frames=3):
    result = np.broadcast_to(np.eye(3), (frames, 55, 3, 3)).copy()
    for frame in range(frames):
        result[frame, 0] = rotation([1., 2., 3.], .8 + .2 * frame)
        result[frame, 20] = rotation([2., -1., 3.], .3 + .4 * frame)
        result[frame, 21] = rotation([-1., 3., 2.], -.6 - .3 * frame)
    return result


class FakeSO3:
    def __init__(self, matrix):
        self.matrix = np.array(matrix, dtype=np.float64, copy=True)

    @classmethod
    def from_matrix(cls, matrix):
        return cls(matrix)

    def as_matrix(self):
        return self.matrix.copy()


class FakeSE3:
    def __init__(self, matrix):
        self.matrix = np.array(matrix, dtype=np.float64, copy=True)

    @classmethod
    def from_rotation(cls, orientation):
        matrix = np.eye(4)
        matrix[:3, :3] = orientation.as_matrix()
        return cls(matrix)

    def as_matrix(self):
        return self.matrix.copy()

    def rotation(self):
        return FakeSO3(self.matrix[:3, :3])

    def copy(self):
        return type(self)(self.matrix)


class FakeFrameTask:
    instances = []

    def __init__(self, frame_name, frame_type, position_cost, orientation_cost,
                 gain=1., lm_damping=0.):
        self.frame_name = frame_name
        self.frame_type = frame_type
        self.position_cost = position_cost
        self.orientation_cost = orientation_cost
        self.cost = np.r_[np.full(3, position_cost), np.full(3, orientation_cost)]
        self.gain = gain
        self.lm_damping = lm_damping
        self.transform_target_to_world = None
        self.targets = []
        type(self).instances.append(self)

    def set_target(self, target):
        self.transform_target_to_world = target.copy()
        self.targets.append(target.as_matrix())


FAKE_MINK = SimpleNamespace(FrameTask=FakeFrameTask, SO3=FakeSO3, SE3=FakeSE3)


class FakeRateLimit:
    instances = []

    def __init__(self, model):
        self.model = model
        self.begins = []
        self.finishes = []
        self.active_frame = None
        type(self).instances.append(self)

    def begin_frame(self, frame, previous_output):
        # Keep the actual argument so tests catch views into mutable robot state.
        self.begins.append((frame, previous_output))
        self.active_frame = frame

    def finish_frame(self, qpos):
        self.finishes.append(qpos)
        self.active_frame = None


class FakeBase:
    """Observable upstream-shaped loop; qpos storage is mutated in place."""

    def __init__(self, human, *, iterations=6, dt=.02, fail_on_call=None,
                 failure_counts=None):
        self.human = human
        self.iterations = iterations
        self.dt = dt
        qpos = np.zeros(36)
        qpos[3] = 1.
        self.robot = SimpleNamespace(
            model=SimpleNamespace(nq=36, nv=35),
            configuration=SimpleNamespace(q=qpos, data=SimpleNamespace(qpos=qpos)),
        )
        self.original_task = SimpleNamespace(frame_name="original_task")
        self.original_limit = object()
        self.tasks = [self.original_task]
        self.limits = [self.original_limit]
        self.base_calls = []
        self.base_run_calls = []
        self.fail_on_call = fail_on_call
        self.failure_counts = failure_counts
        self.original_lock_calls = 0

    def solve_frame(self, frame, iterations=None):
        effective_iterations = self.iterations if iterations is None else iterations
        self.base_calls.append({
            "frame": int(frame), "iterations": effective_iterations,
            "tasks": tuple(self.tasks), "limits": tuple(self.limits),
            "targets": {
                task.frame_name: task.transform_target_to_world.as_matrix()
                for task in self.tasks if isinstance(task, FakeFrameTask)
            },
        })
        if len(self.base_calls) == self.fail_on_call:
            raise RuntimeError("synthetic base solve failed")
        value = 1000. if effective_iterations == 60 else float(frame + 1)
        self.robot.configuration.q[7:] = value + np.arange(29) / 100.
        failures = 0 if self.failure_counts is None else self.failure_counts[len(self.base_calls) - 1]
        return {"qpos": self.robot.configuration.q.copy(), "failures": failures}

    def run(self, frame_indices, fps, warmup_iterations=60, progress=True):
        self.base_run_calls.append((np.array(frame_indices, copy=True), fps,
                                    warmup_iterations, progress))
        self.solve_frame(0, iterations=warmup_iterations)
        qpos = np.stack([self.solve_frame(frame)["qpos"] for frame in frame_indices])
        return SimpleNamespace(qpos=qpos, frame_indices=np.array(frame_indices), fps=fps)

    def lock_ankle_roll(self, *args, **kwargs):
        self.original_lock_calls += 1
        return "base locking was called"


class Stage2ControlsTests(unittest.TestCase):
    def setUp(self):
        FakeFrameTask.instances = []
        FakeRateLimit.instances = []

    def make(self, arm="control", *, rotations=None, base_class=FakeBase, **kwargs):
        if rotations is None:
            rotations = source_rotations()
        human = SimpleNamespace(source={"joint_rotations": rotations})
        controlled_class = controls.controlled_retargeter_class(
            base_class, FAKE_MINK, FakeRateLimit)
        return controlled_class(human, arm=arm, **kwargs)

    def run_all(self, subject, **kwargs):
        frames = len(subject.human.source["joint_rotations"])
        return subject.run(np.arange(frames), 50., progress=False, **kwargs)

    def test_control_matches_unmodified_base_outputs_and_call_schedule(self):
        human = SimpleNamespace(source={"joint_rotations": source_rotations()})
        expected_base = FakeBase(human)
        expected = expected_base.run(np.arange(3), 50., progress=False)
        subject = self.make()
        result = self.run_all(subject)
        np.testing.assert_array_equal(result.qpos, expected.qpos)
        np.testing.assert_array_equal(result.frame_indices, np.arange(3))
        self.assertEqual(result.fps, 50.)
        self.assertEqual([(call["frame"], call["iterations"]) for call in subject.base_calls],
                         [(0, 60), (0, 6), (1, 6), (2, 6)])
        for call in subject.base_calls:
            self.assertEqual(call["tasks"], (subject.original_task,))
            self.assertEqual(call["limits"], (subject.original_limit,))
        self.assertEqual(FakeFrameTask.instances, [])
        self.assertEqual(FakeRateLimit.instances, [])

    def test_each_arm_adds_only_its_requested_tasks_and_limit(self):
        for arm, wrists, rate in (("control", False, False), ("rate_only", False, True),
                                  ("wrist_only", True, False), ("both", True, True)):
            with self.subTest(arm=arm):
                FakeFrameTask.instances = []
                FakeRateLimit.instances = []
                subject = self.make(arm)
                self.run_all(subject)
                self.assertEqual(len(FakeFrameTask.instances), 2 if wrists else 0)
                self.assertEqual(len(FakeRateLimit.instances), 1 if rate else 0)
                for call in subject.base_calls:
                    self.assertIs(call["tasks"][0], subject.original_task)
                    self.assertIs(call["limits"][0], subject.original_limit)
                    self.assertEqual(len(call["tasks"]), 3 if wrists else 1)
                    expected_limits = 2 if rate and call["iterations"] != 60 else 1
                    self.assertEqual(len(call["limits"]), expected_limits)

    def test_wrist_tasks_have_exact_soft_orientation_cost_and_link_names(self):
        subject = self.make("wrist_only")
        self.run_all(subject)
        tasks = {task.frame_name: task for task in FakeFrameTask.instances}
        self.assertEqual(set(tasks), {"left_wrist_yaw_link", "right_wrist_yaw_link"})
        for task in tasks.values():
            self.assertEqual(task.frame_type, "body")
            np.testing.assert_array_equal(task.cost, [0., 0., 0., 10., 10., 10.])
            self.assertEqual(task.gain, 1.)
            self.assertEqual(task.lm_damping, 1.)

    def test_world_targets_use_joint_20_21_and_right_multiply_fixed_offset(self):
        source = source_rotations()
        subject = self.make("both", rotations=source)
        self.run_all(subject)
        for call, frame in zip(subject.base_calls, (0, 0, 1, 2)):
            expected_left = source[frame, 20]
            expected_right = source[frame, 21] @ np.diag([-1., -1., 1.])
            np.testing.assert_allclose(call["targets"]["left_wrist_yaw_link"][:3, :3],
                                       expected_left, atol=1e-12)
            np.testing.assert_allclose(call["targets"]["right_wrist_yaw_link"][:3, :3],
                                       expected_right, atol=1e-12)
            for target in call["targets"].values():
                np.testing.assert_array_equal(target[:3, 3], np.zeros(3))
                np.testing.assert_array_equal(target[3], [0., 0., 0., 1.])

    def test_common_world_rotation_is_applied_once_without_extra_heading(self):
        source = source_rotations()
        common = rotation([1., -2., 3.], 1.1)
        rotated = common @ source
        subject = self.make("wrist_only", rotations=rotated)
        self.run_all(subject)
        for call, frame in zip(subject.base_calls, (0, 0, 1, 2)):
            np.testing.assert_allclose(call["targets"]["left_wrist_yaw_link"][:3, :3],
                                       common @ source[frame, 20], atol=1e-12)
            np.testing.assert_allclose(call["targets"]["right_wrist_yaw_link"][:3, :3],
                                       common @ source[frame, 21] @ np.diag([-1., -1., 1.]),
                                       atol=1e-12)

    def test_readonly_source_and_caller_frame_indices_are_preserved(self):
        source = source_rotations()
        before = source.copy()
        source.setflags(write=False)
        frame_indices = np.arange(3)
        frame_indices.setflags(write=False)
        subject = self.make("both", rotations=source)
        subject.run(frame_indices, 50., progress=False)
        np.testing.assert_array_equal(source, before)
        np.testing.assert_array_equal(frame_indices, [0, 1, 2])

    def test_rate_excludes_warmup_and_starts_normal_frame_zero_without_anchor(self):
        subject = self.make("rate_only")
        output = self.run_all(subject)
        rate, = FakeRateLimit.instances
        self.assertIs(rate.model, subject.robot.model)
        self.assertEqual([frame for frame, _ in rate.begins], [0, 1, 2])
        self.assertIsNone(rate.begins[0][1])
        self.assertEqual(len(rate.finishes), 3)
        for index in (1, 2):
            np.testing.assert_array_equal(rate.begins[index][1], output.qpos[index - 1])
        for index, finished in enumerate(rate.finishes):
            np.testing.assert_array_equal(finished, output.qpos[index])
        self.assertNotIn(rate, subject.base_calls[0]["limits"])
        self.assertTrue(all(rate in call["limits"] for call in subject.base_calls[1:]))

    def test_frame_anchors_and_finish_values_do_not_alias_mutable_qpos(self):
        subject = self.make("both")
        output = self.run_all(subject)
        rate, = FakeRateLimit.instances
        subject.robot.configuration.q[:] = -999.
        for index, (_, previous) in enumerate(rate.begins[1:], start=1):
            np.testing.assert_array_equal(previous, output.qpos[index - 1])
            self.assertFalse(np.shares_memory(previous, subject.robot.configuration.q))
        for index, finished in enumerate(rate.finishes):
            np.testing.assert_array_equal(finished, output.qpos[index])
            self.assertFalse(np.shares_memory(finished, subject.robot.configuration.q))

    def test_audit_separates_warmup_from_normal_output_frames(self):
        for arm in ("control", "rate_only", "wrist_only", "both"):
            with self.subTest(arm=arm):
                subject = self.make(arm)
                self.run_all(subject)
                audit = subject.v3_audit
                self.assertEqual(audit["schema"], "bfm.umr_stage2_controls/3")
                self.assertEqual(audit["arm"], arm)
                self.assertIsInstance(audit["warmup"], dict)
                self.assertEqual(len(audit["frames"]), 3)
                self.assertEqual(audit["warmup_failures"], 0)
                self.assertEqual(audit["output_failures"], 0)
                self.assertEqual(audit["total_observed_failures"], 0)
                self.assertEqual(audit["result"], "COMPLETE_NOT_QUALITY_ACCEPTED")
                json.dumps(audit, allow_nan=False)

    def test_unknown_arm_is_rejected(self):
        with self.assertRaises(ValueError):
            self.make("unregistered_arm")

    def test_direct_solve_before_run_is_rejected_without_calling_base(self):
        subject = self.make("both")
        with self.assertRaises((RuntimeError, ValueError)):
            subject.solve_frame(0, iterations=60)
        self.assertEqual(subject.base_calls, [])

    def test_one_instance_cannot_run_twice(self):
        subject = self.make("both")
        self.run_all(subject)
        calls = len(subject.base_calls)
        with self.assertRaises((RuntimeError, ValueError)):
            self.run_all(subject)
        self.assertEqual(len(subject.base_calls), calls)

    def test_run_requires_all_source_frames_in_order(self):
        for indices in ([1, 2], [0, 2, 1], [0, 0, 2], [0, 1], [0, 1, 2, 3]):
            with self.subTest(indices=indices):
                subject = self.make("both")
                with self.assertRaises((RuntimeError, ValueError)):
                    subject.run(np.array(indices), 50., progress=False)
                self.assertEqual(subject.base_calls, [])

    def test_wrong_fps_warmup_or_stage2_time_contract_is_rejected(self):
        for fps in (25., 49.999, np.nan, np.inf):
            with self.subTest(fps=fps):
                subject = self.make()
                with self.assertRaises(ValueError):
                    subject.run(np.arange(3), fps, progress=False)
                self.assertEqual(subject.base_calls, [])
        for iterations in (0, 6, 59, 61):
            with self.subTest(warmup_iterations=iterations):
                subject = self.make()
                with self.assertRaises(ValueError):
                    self.run_all(subject, warmup_iterations=iterations)
                self.assertEqual(subject.base_calls, [])
        for settings in ({"dt": .01}, {"iterations": 5}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                subject = self.make(**settings)
                self.run_all(subject)

    def test_ankle_roll_lock_is_always_rejected(self):
        for arm in ("control", "rate_only", "wrist_only", "both"):
            with self.subTest(arm=arm):
                subject = self.make(arm)
                with self.assertRaises((RuntimeError, ValueError)):
                    subject.lock_ankle_roll()
                self.assertEqual(subject.original_lock_calls, 0)

    def test_failure_propagates_and_does_not_finish_failed_rate_frame(self):
        subject = self.make("both", fail_on_call=3)
        with self.assertRaisesRegex(RuntimeError, "synthetic base solve failed"):
            self.run_all(subject)
        rate, = FakeRateLimit.instances
        self.assertEqual([frame for frame, _ in rate.begins], [0, 1])
        self.assertEqual(len(rate.finishes), 1)
        self.assertEqual(subject.v3_audit["result"], "ERROR")
        self.assertIn("synthetic base solve failed", json.dumps(subject.v3_audit))

    def test_failed_warmup_is_not_published_as_a_normal_frame(self):
        subject = self.make("both", fail_on_call=1)
        with self.assertRaisesRegex(RuntimeError, "synthetic base solve failed"):
            self.run_all(subject)
        self.assertEqual(subject.v3_audit["frames"], [])
        self.assertEqual(subject.v3_audit["result"], "ERROR")
        for rate in FakeRateLimit.instances:
            self.assertEqual(rate.begins, [])
            self.assertEqual(rate.finishes, [])

    def test_rate_finish_rejection_is_audited_and_never_clipped_or_continued(self):
        class RejectingRate(FakeRateLimit):
            def finish_frame(self, qpos):
                if self.active_frame == 1:
                    raise ValueError("synthetic output rate violation")
                super().finish_frame(qpos)

        klass = controls.controlled_retargeter_class(FakeBase, FAKE_MINK, RejectingRate)
        subject = klass(SimpleNamespace(source={"joint_rotations": source_rotations()}), arm="both")
        with self.assertRaisesRegex(ValueError, "synthetic output rate violation"):
            self.run_all(subject)
        self.assertEqual(subject.v3_audit["result"], "ERROR")
        self.assertEqual([r["frame"] for r in subject.v3_audit["frames"]], [0, 1])
        self.assertIn("rate_validation_error", subject.v3_audit["frames"][-1])
        # The violating raw upstream value is retained as evidence, not clipped.
        np.testing.assert_array_equal(subject.robot.configuration.q[7:], 2. + np.arange(29) / 100.)
        self.assertEqual(len(subject.base_calls), 3)

    def test_post_solve_output_modification_is_rejected(self):
        class PostprocessingBase(FakeBase):
            def run(self, *args, **kwargs):
                result = super().run(*args, **kwargs)
                result.qpos[1, 7] += .01
                return result

        subject = self.make("control", base_class=PostprocessingBase)
        with self.assertRaisesRegex(ValueError, "postprocessing forbidden"):
            self.run_all(subject)
        self.assertEqual(subject.v3_audit["result"], "ERROR")

    def test_upstream_fallback_failures_include_warmup_and_each_output(self):
        subject = self.make("both", failure_counts=[4, 2, 0, 3])
        self.run_all(subject)
        audit = subject.v3_audit
        self.assertEqual(audit["warmup"]["failures"], 4)
        self.assertEqual([row["failures"] for row in audit["frames"]], [2, 0, 3])
        self.assertEqual(audit["warmup_failures"], 4)
        self.assertEqual(audit["output_failures"], 5)
        self.assertEqual(audit["total_observed_failures"], 9)

    def test_out_of_order_calls_from_changed_base_loop_are_rejected(self):
        class SkippingBase(FakeBase):
            def run(self, frame_indices, fps, warmup_iterations=60, progress=True):
                self.solve_frame(0, iterations=warmup_iterations)
                self.solve_frame(0)
                self.solve_frame(2)
                raise AssertionError("invalid frame order reached the base loop")

        subject = self.make("both", base_class=SkippingBase)
        with self.assertRaises((RuntimeError, ValueError)):
            self.run_all(subject)
        self.assertEqual([(call["frame"], call["iterations"]) for call in subject.base_calls],
                         [(0, 60), (0, 6)])


if __name__ == "__main__":
    unittest.main()
