"""Pure NumPy/unittest checks for the opt-in wrist-target diagnostic.

Synthetic rotations only: these tests never load AMASS, SMPL-X or a simulator.
"""
from contextlib import redirect_stderr, redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from scripts import analyze_umr_wrist_targets_v2 as wrist


BODY_NAMES = ("pelvis", "left_wrist_yaw_link", "right_wrist_yaw_link")
HUMAN_PELVIS, HUMAN_LEFT_WRIST, HUMAN_RIGHT_WRIST = 0, 20, 21
PELVIS_QUATERNION = np.array([.5, -.5, -.5, -.5])
WRIST_QUATERNIONS = np.array([[1., 0., 0., 0.], [0., 0., 0., -1.]])


def axis_rotation(axis, angle):
    """Independent Rodrigues formula and wxyz quaternion for known fixtures."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    matrix = (np.cos(angle) * np.eye(3)
              + (1. - np.cos(angle)) * np.outer(axis, axis)
              + np.sin(angle) * cross)
    quaternion = np.r_[np.cos(angle / 2.), np.sin(angle / 2.) * axis]
    return matrix, quaternion


def compose(left, right):
    """Independent Hamilton product; deliberately does not use subject code."""
    left, right = np.asarray(left), np.asarray(right)
    scalar = left[..., :1] * right[..., :1] - np.sum(
        left[..., 1:] * right[..., 1:], axis=-1, keepdims=True)
    vector = (left[..., :1] * right[..., 1:]
              + right[..., :1] * left[..., 1:]
              + np.cross(left[..., 1:], right[..., 1:]))
    return np.concatenate((scalar, vector), axis=-1)


def target_fixture(frames=4):
    """Human global joints with exact offset-correct robot world orientations."""
    human = np.broadcast_to(np.eye(3), (frames, 55, 3, 3)).copy()
    robot = np.zeros((frames, 3, 4))
    for frame in range(frames):
        for slot, human_id in enumerate(
                (HUMAN_PELVIS, HUMAN_LEFT_WRIST, HUMAN_RIGHT_WRIST)):
            matrix, quaternion = axis_rotation(
                [1. + slot, 2. + frame, 3. - slot], .27 + .41 * frame + .33 * slot)
            human[frame, human_id] = matrix
            offset = PELVIS_QUATERNION if slot == 0 else WRIST_QUATERNIONS[slot - 1]
            robot[frame, slot] = compose(quaternion, offset)
    return human, robot


class RotationAngleTests(unittest.TestCase):
    def test_identity_returns_zero_with_preserved_batch_shape(self):
        identity = np.broadcast_to(np.eye(3), (2, 5, 3, 3)).copy()
        actual = wrist.rotation_angle(identity, identity)
        self.assertEqual(np.shape(actual), (2, 5))
        np.testing.assert_allclose(actual, 0., atol=5e-8)

    def test_known_angles_and_shortest_arc_are_in_radians(self):
        angles = [0., 1e-4, .3, np.pi / 2., np.pi, 1.8 * np.pi]
        rotations = np.stack([axis_rotation([1, 2, -3], value)[0] for value in angles])
        expected = [0., 1e-4, .3, np.pi / 2., np.pi, .2 * np.pi]
        actual = wrist.rotation_angle(np.repeat(np.eye(3)[None], len(angles), axis=0), rotations)
        np.testing.assert_allclose(actual, expected, atol=5e-8)

    def test_single_rotation_and_symmetry(self):
        first = axis_rotation([1, 0, 0], .6)[0]
        second = axis_rotation([0, 1, 0], -.8)[0]
        self.assertEqual(np.shape(wrist.rotation_angle(first, second)), ())
        np.testing.assert_allclose(wrist.rotation_angle(first, second),
                                   wrist.rotation_angle(second, first), atol=5e-8)

    def test_common_left_and_right_rotations_preserve_angle(self):
        first = axis_rotation([1, 2, 3], .7)[0]
        second = axis_rotation([-3, 1, 2], -.3)[0]
        common = axis_rotation([2, -1, 1], 1.1)[0]
        expected = wrist.rotation_angle(first, second)
        for a, b in ((common @ first, common @ second), (first @ common, second @ common)):
            np.testing.assert_allclose(wrist.rotation_angle(a, b), expected, atol=5e-8)

    def test_malformed_or_broadcast_only_shapes_are_rejected(self):
        cases = ((np.eye(4), np.eye(4)), (np.zeros(9), np.zeros(9)),
                 (np.eye(3), np.eye(3)[None]),
                 (np.repeat(np.eye(3)[None], 2, axis=0), np.eye(3)[None]))
        for first, second in cases:
            with self.subTest(shapes=(first.shape, second.shape)), self.assertRaises(ValueError):
                wrist.rotation_angle(first, second)

    def test_nonfinite_matrices_in_either_operand_are_rejected(self):
        for value in (np.nan, np.inf, -np.inf):
            bad = np.eye(3)
            bad[0, 1] = value
            for first, second in ((bad, np.eye(3)), (np.eye(3), bad)):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    wrist.rotation_angle(first, second)

    def test_reflections_scaling_and_shear_are_not_silently_projected_to_so3(self):
        for bad in (np.diag([-1., 1., 1.]), np.eye(3) * 1.01,
                    np.array([[1., .1, 0.], [0., 1., 0.], [0., 0., 1.]])):
            for first, second in ((bad, np.eye(3)), (np.eye(3), bad)):
                with self.subTest(matrix=bad.tolist()), self.assertRaises(ValueError):
                    wrist.rotation_angle(first, second)


class WristTargetTests(unittest.TestCase):
    def test_exported_offsets_match_fixed_gmr_frame_conventions(self):
        # wxyz [.5, -.5, -.5, -.5] maps x->z, y->x and z->y.
        np.testing.assert_allclose(wrist.PELVIS_OFFSET,
                                   [[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]], atol=1e-12)
        np.testing.assert_allclose(wrist.WRIST_OFFSETS,
                                   np.stack([np.eye(3), np.diag([-1., -1., 1.])]), atol=1e-12)

    def test_exact_corrected_targets_have_zero_world_and_relative_errors(self):
        human, robot = target_fixture()
        actual = wrist.target_errors(human, robot, BODY_NAMES)
        for key, shape in (("world_wrist", (4, 2)), ("pelvis_relative_wrist", (4, 2)),
                           ("world_pelvis", (4,))):
            self.assertEqual(np.shape(actual[key]), shape)
            np.testing.assert_allclose(actual[key], 0., atol=5e-8)

    def test_omitting_pelvis_offset_gives_120_degree_root_and_relative_errors(self):
        human = np.broadcast_to(np.eye(3), (2, 55, 3, 3)).copy()
        robot = np.broadcast_to(np.vstack(([1., 0., 0., 0.], WRIST_QUATERNIONS)), (2, 3, 4)).copy()
        actual = wrist.target_errors(human, robot, BODY_NAMES)
        np.testing.assert_allclose(actual["world_wrist"], 0., atol=5e-8)
        np.testing.assert_allclose(actual["world_pelvis"], 2. * np.pi / 3., atol=5e-8)
        np.testing.assert_allclose(actual["pelvis_relative_wrist"], 2. * np.pi / 3., atol=5e-8)

    def test_common_global_so3_change_preserves_all_nonzero_errors(self):
        human, robot = target_fixture()
        robot[:, 0] = compose(axis_rotation([2, 1, -1], .19)[1], robot[:, 0])
        robot[:, 1] = compose(axis_rotation([1, -2, 3], .38)[1], robot[:, 1])
        robot[:, 2] = compose(axis_rotation([-1, 1, 2], -.51)[1], robot[:, 2])
        expected = wrist.target_errors(human, robot, BODY_NAMES)
        for frame in range(len(human)):
            common, quaternion = axis_rotation([1, 3 + frame, -2], .63 + .27 * frame)
            human[frame] = common @ human[frame]
            robot[frame] = compose(quaternion, robot[frame])
        actual = wrist.target_errors(human, robot, BODY_NAMES)
        for key in expected:
            np.testing.assert_allclose(actual[key], expected[key], atol=5e-8)

    def test_pelvis_only_error_changes_relative_but_not_world_wrist_metrics(self):
        human, robot = target_fixture()
        robot[:, 0] = compose(axis_rotation([1, -2, 3], .31)[1], robot[:, 0])
        actual = wrist.target_errors(human, robot, BODY_NAMES)
        np.testing.assert_allclose(actual["world_wrist"], 0., atol=5e-8)
        np.testing.assert_allclose(actual["world_pelvis"], .31, atol=5e-8)
        np.testing.assert_allclose(actual["pelvis_relative_wrist"], .31, atol=5e-8)

    def test_quaternion_sign_does_not_change_any_orientation_error(self):
        human, robot = target_fixture()
        robot[:, 1] = compose(axis_rotation([1, 1, 0], .2)[1], robot[:, 1])
        expected = wrist.target_errors(human, robot, BODY_NAMES)
        robot[::2] *= -1
        robot[1::2, 0] *= -1
        actual = wrist.target_errors(human, robot, BODY_NAMES)
        for key in expected:
            np.testing.assert_allclose(actual[key], expected[key], atol=5e-8)

    def test_left_and_right_offset_are_not_interchangeable(self):
        human = np.broadcast_to(np.eye(3), (1, 55, 3, 3)).copy()
        robot = np.array([[PELVIS_QUATERNION, [1., 0., 0., 0.], [1., 0., 0., 0.]]])
        actual = wrist.target_errors(human, robot, BODY_NAMES)
        np.testing.assert_allclose(actual["world_wrist"], [[0., np.pi]], atol=5e-8)
        np.testing.assert_allclose(actual["pelvis_relative_wrist"], [[0., np.pi]], atol=5e-8)

    def test_noncanonical_body_order_and_additional_bodies_are_supported(self):
        human, robot = target_fixture()
        expected = wrist.target_errors(human, robot, BODY_NAMES)
        names = ("right_wrist_yaw_link", "untracked_extra", "pelvis", "left_wrist_yaw_link")
        extra = np.tile([1., 0., 0., 0.], (len(human), 1))
        reordered = np.stack((robot[:, 2], extra, robot[:, 0], robot[:, 1]), axis=1)
        actual = wrist.target_errors(human, reordered, names)
        for key in expected:
            np.testing.assert_allclose(actual[key], expected[key], atol=5e-8)

    def test_missing_or_duplicate_body_names_are_rejected(self):
        human, robot = target_fixture()
        cases = (BODY_NAMES[:2], ("pelvis", "left_wrist_yaw_link", "left_wrist_yaw_link"),
                 ("not_pelvis", *BODY_NAMES[1:]), (*BODY_NAMES, "pelvis"))
        for names in cases:
            with self.subTest(names=names), self.assertRaises(ValueError):
                wrist.target_errors(human, robot, names)

    def test_bad_human_and_robot_shapes_are_rejected(self):
        human, robot = target_fixture()
        cases = ((human[:, :54], robot), (human[0], robot), (human, robot[:1]),
                 (human, robot[:, :, :3]), (human, robot[:, :2]),
                 (human[:, :, :2], robot))
        for h, r in cases:
            with self.subTest(shapes=(h.shape, r.shape)), self.assertRaises(ValueError):
                wrist.target_errors(h, r, BODY_NAMES)

    def test_nonunit_and_nonfinite_robot_quaternions_are_rejected(self):
        human, robot = target_fixture()
        for quaternion in ([0., 0., 0., 0.], [2., 0., 0., 0.],
                           [.99, 0., 0., 0.], [np.nan, 0., 0., 0.], [np.inf, 0., 0., 0.]):
            bad = robot.copy()
            bad[0, 1] = quaternion
            with self.subTest(quaternion=quaternion), self.assertRaises(ValueError):
                wrist.target_errors(human, bad, BODY_NAMES)

    def test_invalid_human_rotations_are_rejected(self):
        human, robot = target_fixture()
        for matrix in (np.diag([-1., 1., 1.]), np.full((3, 3), np.nan), np.eye(3) * 1.01):
            bad = human.copy()
            bad[0, HUMAN_LEFT_WRIST] = matrix
            with self.subTest(matrix=matrix.tolist()), self.assertRaises(ValueError):
                wrist.target_errors(bad, robot, BODY_NAMES)

    def test_math_does_not_mutate_caller_inputs(self):
        human, robot = target_fixture()
        original_human, original_robot = human.copy(), robot.copy()
        wrist.target_errors(human, robot, BODY_NAMES)
        np.testing.assert_array_equal(human, original_human)
        np.testing.assert_array_equal(robot, original_robot)


class SourceClockTests(unittest.TestCase):
    def fixture(self, frames=250):
        source_count, source_fps = 721, 120.
        times = np.arange(251) / 50.
        fractional = times * source_fps
        fractional = np.where(np.abs(fractional - np.rint(fractional)) < 1e-9,
                              np.rint(fractional), fractional)
        lower = np.floor(fractional).astype(np.int64)
        upper = lower + 1
        prepared = {"times": times, "sample_lower": lower,
                    "sample_upper": upper, "sample_alpha": fractional - lower}
        metadata = {"start": 0., "target_fps": 50., "requested_duration": 5.,
                    "source_fps": source_fps, "frames": 251, "actual_duration": 5.}
        indices = np.arange(frames + 1)
        native = times[indices] * 30.
        native_lower = np.floor(native).astype(np.int64)
        proof = {"times": times[indices], "candidate_indices": indices.copy(),
                 "packed_indices": indices.copy(), "native_times": np.arange(181) / 30.,
                 "candidate_source_lower": lower[indices], "candidate_source_upper": upper[indices],
                 "candidate_source_alpha": prepared["sample_alpha"][indices],
                 "baseline_native_lower": native_lower, "baseline_native_upper": native_lower + 1,
                 "baseline_native_alpha": (native - native_lower).astype(np.float32)}
        pair = {"fps": 50., "frames": frames + 1, "expected_packed_frames": frames,
                "source_clock_proof": {"source_frames": source_count, "source_fps": source_fps,
                                       "source_endpoint_s": 6., "native_frames": 181,
                                       "native_actual_fps": 30., "native_nominal_target_fps": 30.,
                                       "native_source_frame_zero": 0},
                "common_window_proof": {"candidate_frames_before_window": 251,
                                        "pair_frames": frames + 1, "pair_start_s": 0.,
                                        "pair_last_sample_s": frames / 50.,
                                        "packed_frames_before_window": 300,
                                        "dropped_candidate_frames": 250 - frames,
                                        "dropped_packed_frames": 299 - frames}}
        return prepared, metadata, pair, proof, frames

    def test_halfopen_discards_exactly_one_inclusive_endpoint(self):
        actual = wrist.validate_source_clock(*self.fixture())
        np.testing.assert_array_equal(actual, np.arange(250))
        self.assertEqual(actual[-1] / 50., 4.98)

    def test_short_halfopen_grids_have_no_202_or_227_off_by_one(self):
        for inclusive_count in (3, 202, 227, 251):
            with self.subTest(inclusive_count=inclusive_count):
                actual = wrist.validate_source_clock(*self.fixture(inclusive_count - 1))
                np.testing.assert_array_equal(actual, np.arange(inclusive_count - 1))

    def test_source_amass_indices_cannot_be_replaced_with_native_30hz_indices(self):
        p, m, pair, proof, frames = self.fixture()
        proof["candidate_source_lower"] = proof["baseline_native_lower"].copy()
        with self.assertRaises(ValueError):
            wrist.validate_source_clock(p, m, pair, proof, frames)

    def test_logical_indices_must_be_integer_contiguous_from_zero(self):
        for name in ("candidate_indices", "packed_indices"):
            for transform in (lambda a: a + 1, lambda a: a.astype(float), lambda a: a[::-1]):
                p, m, pair, proof, frames = self.fixture()
                proof[name] = transform(proof[name])
                with self.subTest(name=name), self.assertRaises(ValueError):
                    wrist.validate_source_clock(p, m, pair, proof, frames)

    def test_source_metadata_contract_is_fixed(self):
        for name, value in (("start", .02), ("target_fps", 30.), ("requested_duration", 4.),
                            ("source_fps", 119.), ("frames", 250), ("actual_duration", 4.98)):
            p, m, pair, proof, frames = self.fixture()
            m[name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                wrist.validate_source_clock(p, m, pair, proof, frames)

    def test_prepared_sampling_arrays_are_independently_reconstructed(self):
        for name in ("times", "sample_lower", "sample_upper", "sample_alpha"):
            p, m, pair, proof, frames = self.fixture()
            p[name] = p[name].copy()
            p[name][9] += 1
            with self.subTest(name=name), self.assertRaises(ValueError):
                wrist.validate_source_clock(p, m, pair, proof, frames)

    def test_proof_sampling_arrays_must_match_prepared_source(self):
        for name in ("times", "candidate_source_lower", "candidate_source_upper", "candidate_source_alpha"):
            p, m, pair, proof, frames = self.fixture()
            proof[name] = proof[name].copy()
            proof[name][9] += 1
            with self.subTest(name=name), self.assertRaises(ValueError):
                wrist.validate_source_clock(p, m, pair, proof, frames)

    def test_native_clock_endpoint_rate_and_index_bounds_are_checked(self):
        for key, value in (("source_endpoint_s", 5.9), ("native_frames", 180),
                           ("native_actual_fps", 29.9), ("native_source_frame_zero", 1)):
            p, m, pair, proof, frames = self.fixture()
            pair["source_clock_proof"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                wrist.validate_source_clock(p, m, pair, proof, frames)
        for key, value in (("baseline_native_lower", -1), ("baseline_native_upper", 182),
                           ("baseline_native_alpha", np.nan), ("baseline_native_alpha", 1.1)):
            p, m, pair, proof, frames = self.fixture()
            proof[key][3] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                wrist.validate_source_clock(p, m, pair, proof, frames)

    def test_baseline_interpolation_must_reach_common_source_time(self):
        p, m, pair, proof, frames = self.fixture()
        proof["baseline_native_alpha"][3] = 0.
        with self.assertRaises(ValueError):
            wrist.validate_source_clock(p, m, pair, proof, frames)

    def test_common_window_counts_and_endpoint_are_bound(self):
        for key, value in (("pair_frames", 250), ("pair_start_s", .02),
                           ("pair_last_sample_s", 4.98), ("dropped_candidate_frames", 1),
                           ("dropped_packed_frames", 48), ("candidate_frames_before_window", 250)):
            p, m, pair, proof, frames = self.fixture()
            pair["common_window_proof"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                wrist.validate_source_clock(p, m, pair, proof, frames)

    def test_invalid_source_counts_rates_and_pair_counts_are_rejected(self):
        for key, value in (("source_frames", 1), ("source_frames", 721.),
                           ("source_fps", 0.), ("source_fps", np.nan)):
            p, m, pair, proof, frames = self.fixture()
            pair["source_clock_proof"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                wrist.validate_source_clock(p, m, pair, proof, frames)
        p, m, pair, proof, frames = self.fixture()
        pair["frames"] -= 1
        with self.assertRaises(ValueError):
            wrist.validate_source_clock(p, m, pair, proof, frames)

    def test_clock_validation_does_not_mutate_inputs(self):
        args = self.fixture()
        before = copy.deepcopy(args)
        wrist.validate_source_clock(*args)
        for actual, expected in zip(args[:1] + args[3:4], before[:1] + before[3:4]):
            for key in actual:
                np.testing.assert_array_equal(actual[key], expected[key])
        self.assertEqual(args[1:3], before[1:3])


class OutputContractTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dataset = self.root / "dataset.json"
        self.dataset.write_text("{}", encoding="utf-8")
        self.report = {"schema": "test-wrist-targets-v2", "origins": []}

    def call_main(self, *extra):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            result = wrist.main(["--dataset", str(self.dataset), *map(str, extra)])
        return result, stdout.getvalue()

    def assert_output_rejected_before_analysis(self, output):
        with mock.patch.object(wrist, "analyze_dataset", return_value=self.report) as analyze:
            with self.assertRaises((ValueError, OSError, SystemExit)) as raised:
                self.call_main("--outputnew", output)
            if isinstance(raised.exception, SystemExit):
                self.assertNotEqual(raised.exception.code, 0)
            analyze.assert_not_called()

    def test_default_is_readonly_json_stdout(self):
        before = set(self.root.iterdir())
        with mock.patch.object(wrist, "analyze_dataset", return_value=self.report) as analyze:
            code, stdout = self.call_main()
        self.assertEqual(code, 0)
        analyze.assert_called_once_with(self.dataset)
        self.assertEqual(json.loads(stdout), self.report)
        self.assertEqual(set(self.root.iterdir()), before)
        self.assertEqual(self.dataset.read_text(encoding="utf-8"), "{}")

    def test_explicit_new_file_contains_json_report(self):
        output = self.root / "new-report.json"
        with mock.patch.object(wrist, "analyze_dataset", return_value=self.report):
            code, stdout = self.call_main("--outputnew", output)
        self.assertEqual(code, 0)
        self.assertEqual(stdout, "")
        self.assertEqual(json.loads(output.read_text(encoding="utf-8")), self.report)

    def test_existing_output_is_never_overwritten(self):
        output = self.root / "existing.json"
        output.write_text("keep original", encoding="utf-8")
        self.assert_output_rejected_before_analysis(output)
        self.assertEqual(output.read_text(encoding="utf-8"), "keep original")

    def test_missing_output_parent_is_not_created(self):
        output = self.root / "missing" / "report.json"
        self.assert_output_rejected_before_analysis(output)
        self.assertFalse(output.parent.exists())

    def test_dangling_output_symlink_is_preserved(self):
        output, target = self.root / "report.json", self.root / "absent.json"
        output.symlink_to(target)
        self.assert_output_rejected_before_analysis(output)
        self.assertTrue(output.is_symlink())
        self.assertFalse(target.exists())

    def test_symlink_in_output_ancestors_is_rejected(self):
        actual, alias = self.root / "actual", self.root / "alias"
        (actual / "nested").mkdir(parents=True)
        alias.symlink_to(actual, target_is_directory=True)
        for output in (alias / "report.json", alias / "nested" / "report.json"):
            with self.subTest(output=output):
                self.assert_output_rejected_before_analysis(output)
        self.assertEqual(list(actual.rglob("report.json")), [])

    def test_file_created_during_analysis_is_preserved(self):
        output = self.root / "racing-report.json"

        def analyze(_dataset):
            output.write_text("another writer", encoding="utf-8")
            return self.report

        with mock.patch.object(wrist, "analyze_dataset", side_effect=analyze):
            with self.assertRaises((ValueError, OSError, SystemExit)):
                self.call_main("--outputnew", output)
        self.assertEqual(output.read_text(encoding="utf-8"), "another writer")

    def test_nonfinite_report_is_rejected_without_partial_output(self):
        output = self.root / "nonfinite-report.json"
        with mock.patch.object(wrist, "analyze_dataset", return_value={"value": np.nan}):
            with self.assertRaises(ValueError):
                self.call_main("--outputnew", output)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
