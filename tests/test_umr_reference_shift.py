"""CPU-only reference geometry/statistics and new-output contract tests."""
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from scripts import analyze_umr_reference_shift as shift

try:
    from scipy.spatial.transform import Rotation
except ImportError:
    Rotation = None


def axis_rotation(axis, angle):
    """Independent Rodrigues matrix plus its wxyz quaternion for fixtures."""
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    matrix = (np.cos(angle) * np.eye(3)
              + (1 - np.cos(angle)) * np.outer(axis, axis)
              + np.sin(angle) * cross)
    quaternion = np.r_[np.cos(angle / 2), np.sin(angle / 2) * axis]
    return matrix, quaternion


def compose(left, right):
    """Hamilton product used only to construct known world-pose fixtures."""
    left, right = np.asarray(left), np.asarray(right)
    scalar = left[..., :1] * right[..., :1] - np.sum(
        left[..., 1:] * right[..., 1:], axis=-1, keepdims=True)
    vector = (left[..., :1] * right[..., 1:]
              + right[..., :1] * left[..., 1:]
              + np.cross(left[..., 1:], right[..., 1:]))
    return np.concatenate((scalar, vector), axis=-1)


def local_fixture(frames=3):
    rng = np.random.default_rng(493)
    count = len(shift.REFERENCE_BODIES)
    positions = rng.normal(size=(frames, count, 3))
    quaternions = rng.normal(size=(frames, count, 4))
    quaternions /= np.linalg.norm(quaternions, axis=-1, keepdims=True)
    pelvis = shift.REFERENCE_BODIES.index("pelvis")
    positions[:, pelvis] = 0
    quaternions[:, pelvis] = [1., 0, 0, 0]
    return positions, quaternions


def world_fixture(positions, quaternions, side=0):
    world_positions = np.empty_like(positions)
    world_quaternions = np.empty_like(quaternions)
    for frame in range(len(positions)):
        matrix, rotation = axis_rotation([1 + side, 2 + frame, 3 - side],
                                         .4 + .6 * frame + .7 * side)
        translation = np.array([7 + side, -3 + frame, 2 - side])
        world_positions[frame] = positions[frame] @ matrix.T + translation
        world_quaternions[frame] = compose(rotation, quaternions[frame])
    return world_positions, world_quaternions


def packed_fixture(frames=2, native_sha="a" * 64, pipeline="b" * 64):
    root = np.tile([0., 0., .8], (frames, 1))
    quaternion = np.tile([1., 0., 0., 0.], (frames, 1))
    return {
        "format_version": np.array(3), "fps": np.array(50),
        "quaternion_order": np.array("wxyz"), "source_sha256": np.array(native_sha),
        "pipeline_fingerprint": np.array(pipeline),
        "joint_names": np.array(shift.dataset_contract.pairs.ARTICULATION_JOINT_NAMES),
        "body_names": np.array(shift.dataset_contract.pairs.BODY_NAMES),
        "joint_pos": np.zeros((frames, 29)), "joint_vel": np.zeros((frames, 29)),
        "body_pos_w": np.repeat(root[:, None], 30, axis=1),
        "body_quat_w": np.repeat(quaternion[:, None], 30, axis=1),
        "body_lin_vel_w": np.zeros((frames, 30, 3)),
        "body_ang_vel_w": np.zeros((frames, 30, 3)),
        "reference_root_pos": root, "reference_root_quat_w": quaternion,
    }


class ImportBehaviorTests(unittest.TestCase):
    def test_module_reload_preserves_callers_bytecode_setting(self):
        for flag in (False, True):
            with self.subTest(flag=flag), mock.patch.object(sys, "dont_write_bytecode", flag), \
                    mock.patch.object(importlib.machinery.SourceFileLoader, "set_data"):
                # Prevent reload from creating a cache even when testing False.
                importlib.reload(shift)
                self.assertIs(sys.dont_write_bytecode, flag)

    def test_code_paths_ignore_nonpath_module_file_attributes(self):
        expected = shift._code_paths()
        for value in (object(), 123, ["not-a-path"]):
            module = SimpleNamespace(__file__=value)
            with self.subTest(value=value), mock.patch.dict(sys.modules, {"synthetic_nonpath_module": module}):
                self.assertEqual(shift._code_paths(), expected)


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / "input.json"
        self.path.write_bytes(b'{"value": 1}')
        self.expected = hashlib.sha256(self.path.read_bytes()).hexdigest()

    def test_add_and_recheck_bind_real_bytes(self):
        frozen = shift.Snapshot()
        self.assertEqual(frozen.add(self.path, self.expected), self.expected)
        self.assertEqual(frozen.add(self.path), self.expected)
        self.assertEqual(frozen.hashes(), {str(self.path): self.expected})
        self.assertEqual(frozen.read_json(self.path, self.expected), {"value": 1})
        frozen.recheck()

    def test_wrong_hash_and_malformed_hash_are_rejected(self):
        for digest in ("0" * 64, self.expected.upper(), "short", 123):
            with self.subTest(digest=digest), self.assertRaises(ValueError):
                shift.Snapshot().add(self.path, digest)

    def test_changed_file_is_rejected_when_added_again(self):
        frozen = shift.Snapshot()
        frozen.add(self.path)
        self.path.write_bytes(b"changed input with a different size")
        with self.assertRaises(ValueError):
            frozen.add(self.path)

    def test_recheck_hashes_again_even_with_identical_stat_signature(self):
        frozen = shift.Snapshot()
        frozen.add(self.path)
        original_signature = frozen.files[str(self.path)][1]
        self.path.write_bytes(b'{"value": 2}')
        with mock.patch.object(frozen, "signature", return_value=original_signature):
            with self.assertRaises(ValueError):
                frozen.recheck()

    def test_alias_target_change_is_rejected_even_for_identical_bytes(self):
        alias, alternate = self.root / "alias.json", self.root / "alternate.json"
        alternate.write_bytes(self.path.read_bytes())
        alias.symlink_to(self.path)
        frozen = shift.Snapshot()
        frozen.add(self.path)
        frozen.aliases[str(alias)] = str(self.path)
        frozen.recheck()
        alias.unlink()
        alias.symlink_to(alternate)
        with self.assertRaisesRegex(ValueError, "alias changed"):
            frozen.recheck()

    def test_direct_symlink_input_is_rejected(self):
        alias = self.root / "alias.json"
        alias.symlink_to(self.path)
        with self.assertRaises(ValueError):
            shift.Snapshot().add(alias)

    def test_snapshot_supports_frozen_builder_index_reader(self):
        index = self.root / "index.yaml"
        index.write_text(json.dumps({"KIT/person/walk": str(self.path)}), encoding="utf-8")
        digest = hashlib.sha256(index.read_bytes()).hexdigest()
        frozen = shift.Snapshot()
        actual = shift.dataset_contract.read_index(index, frozen, digest)
        self.assertEqual(actual, {"KIT/person/walk": str(self.path)})
        frozen.recheck()


class PackedContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "packed.npz"

    def load(self, payload, *, native_sha="a" * 64, pipeline="b" * 64, frames=2):
        np.savez(self.path, **payload)
        digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        frozen = shift.Snapshot()
        actual = shift._load_packed(self.path, digest, native_sha, pipeline, frames, frozen)
        frozen.recheck()
        return actual

    def test_complete_named_30body_29joint_archive_loads(self):
        payload = packed_fixture()
        actual = self.load(payload)
        self.assertEqual(set(actual), set(payload))
        for key, value in payload.items():
            with self.subTest(key=key):
                np.testing.assert_array_equal(actual[key], value)

    def test_wrong_clock_format_and_quaternion_order_are_rejected(self):
        for field, value in (("fps", 49), ("fps", np.array([50])), ("fps", np.nan),
                             ("format_version", 2), ("format_version", np.array([3])),
                             ("quaternion_order", "xyzw"),
                             ("quaternion_order", np.array(["wxyz"]))):
            payload = packed_fixture()
            payload[field] = np.asarray(value)
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.load(payload)

    def test_wrong_native_source_or_pipeline_hash_is_rejected(self):
        for field in ("source_sha256", "pipeline_fingerprint"):
            for value in ("c" * 64, np.array([packed_fixture()[field].item()])):
                payload = packed_fixture()
                payload[field] = np.asarray(value)
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    self.load(payload)

    def test_body_and_joint_order_duplicates_and_shape_are_rejected(self):
        for field in ("joint_names", "body_names"):
            for kind in ("reversed", "duplicate", "missing", "rank", "numeric"):
                payload = packed_fixture()
                names = payload[field]
                if kind == "reversed":
                    payload[field] = names[::-1]
                elif kind == "duplicate":
                    names[1] = names[0]
                elif kind == "missing":
                    payload[field] = names[:-1]
                elif kind == "rank":
                    payload[field] = names[None]
                else:
                    payload[field] = np.arange(len(names))
                with self.subTest(field=field, kind=kind), self.assertRaises(ValueError):
                    self.load(payload)

    def test_frame_count_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            self.load(packed_fixture(frames=3), frames=2)
        for field, value in packed_fixture().items():
            if value.ndim > 1:
                payload = packed_fixture()
                payload[field] = value[:-1]
                with self.subTest(field=field), self.assertRaises(ValueError):
                    self.load(payload)

    def test_each_numeric_array_rejects_nonfinite_values(self):
        for field, value in packed_fixture().items():
            if value.ndim > 1:
                for bad in (np.nan, np.inf, -np.inf):
                    payload = packed_fixture()
                    payload[field].flat[0] = bad
                    with self.subTest(field=field, bad=bad), self.assertRaises(ValueError):
                        self.load(payload)

    def test_both_quaternion_arrays_require_unit_length(self):
        for field in ("body_quat_w", "reference_root_quat_w"):
            for scale in (0., 1.01):
                payload = packed_fixture()
                payload[field] *= scale
                with self.subTest(field=field, scale=scale), self.assertRaises(ValueError):
                    self.load(payload)

    def test_missing_required_arrays_and_scalars_are_rejected(self):
        for field in packed_fixture():
            payload = packed_fixture()
            del payload[field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.load(payload)


class QuaternionAngleTests(unittest.TestCase):
    def test_sign_equivalence_and_unit_roundoff(self):
        _, q = axis_rotation([1, 2, 3], 1.7)
        for other in (q, -q, q * (1 + 1e-6)):
            with self.subTest(other=other):
                np.testing.assert_allclose(shift.quaternion_angle(q, other), 0, atol=1e-7)

    def test_shortest_angle_and_pi_endpoint(self):
        for angle, expected in ((.8, .8), (1.5 * np.pi, .5 * np.pi), (np.pi, np.pi)):
            _, q = axis_rotation([0, 1, 0], angle)
            with self.subTest(angle=angle):
                self.assertAlmostEqual(float(shift.quaternion_angle([1., 0, 0, 0], q)), expected)

    def test_small_nonzero_angle_is_preserved(self):
        _, q = axis_rotation([1, 2, 3], 1e-6)
        self.assertAlmostEqual(float(shift.quaternion_angle([1., 0, 0, 0], q)), 1e-6, places=9)

    def test_bad_quaternions_are_rejected_on_either_side(self):
        valid = np.array([1., 0, 0, 0])
        for bad in ([0., 0, 0, 0], [2., 0, 0, 0], [1.0002, 0, 0, 0],
                    [np.nan, 0, 0, 1], [np.inf, 0, 0, 1], [1., 0, 0]):
            for left, right in ((valid, bad), (bad, valid)):
                with self.subTest(left=left, right=right), self.assertRaises(ValueError):
                    shift.quaternion_angle(left, right)


class RelativeLinkPoseTests(unittest.TestCase):
    def test_full_so3_recovers_known_local_link_frames(self):
        expected_positions, expected_quaternions = local_fixture()
        positions, quaternions = world_fixture(expected_positions, expected_quaternions)
        actual_positions, actual_quaternions = shift.relative_link_pose(
            positions, quaternions, shift.REFERENCE_BODIES)
        np.testing.assert_allclose(actual_positions, expected_positions, atol=2e-14)
        np.testing.assert_allclose(np.abs(np.sum(actual_quaternions * expected_quaternions, axis=-1)),
                                   1., atol=2e-14)

    def test_independent_world_transforms_preserve_nonzero_pair_errors(self):
        pa, qa = local_fixture()
        pb, qb = pa.copy(), qa.copy()
        wrist = shift.REFERENCE_BODIES.index("left_wrist_yaw_link")
        pb[:, wrist] += [.1, -.2, .3]
        _, twist = axis_rotation([1, -2, 3], .6)
        qb[:, wrist] = compose(qb[:, wrist], twist)
        wa, xa = world_fixture(pa, qa, side=0)
        wb, xb = world_fixture(pb, qb, side=1)
        ra, sa = shift.relative_link_pose(wa, xa, shift.REFERENCE_BODIES)
        rb, sb = shift.relative_link_pose(wb, xb, shift.REFERENCE_BODIES)
        np.testing.assert_allclose(np.linalg.norm(rb - ra, axis=-1),
                                   np.linalg.norm(pb - pa, axis=-1), atol=2e-14)
        np.testing.assert_allclose(shift.quaternion_angle(sa, sb),
                                   shift.quaternion_angle(qa, qb), atol=1e-7)
        np.testing.assert_allclose(shift.quaternion_angle(sa, sb)[:, wrist], .6, atol=1e-12)

    def test_arbitrary_quaternion_signs_leave_poses_unchanged(self):
        p, q = world_fixture(*local_fixture())
        signed = q.copy()
        signed[::2, ::2] *= -1
        signed[1::2, 1::2] *= -1
        pa, qa = shift.relative_link_pose(p, q, shift.REFERENCE_BODIES)
        pb, qb = shift.relative_link_pose(p, signed, shift.REFERENCE_BODIES)
        np.testing.assert_allclose(pa, pb, atol=1e-14)
        np.testing.assert_allclose(shift.quaternion_angle(qa, qb), 0, atol=1e-7)

    def test_pelvis_is_zero_position_and_identity_rotation(self):
        p, q = world_fixture(*local_fixture())
        relative_p, relative_q = shift.relative_link_pose(p, q, shift.REFERENCE_BODIES)
        pelvis = shift.REFERENCE_BODIES.index("pelvis")
        np.testing.assert_allclose(relative_p[:, pelvis], 0, atol=1e-14)
        np.testing.assert_allclose(np.abs(relative_q[:, pelvis, 0]), 1, atol=1e-14)
        np.testing.assert_allclose(relative_q[:, pelvis, 1:], 0, atol=1e-14)

    def test_names_select_and_reorder_links_with_extra_body(self):
        p, q = world_fixture(*local_fixture())
        expected_p, expected_q = shift.relative_link_pose(p, q, shift.REFERENCE_BODIES)
        names = tuple(reversed(shift.REFERENCE_BODIES)) + ("unused_body",)
        p = np.concatenate((p[:, ::-1], np.ones((len(p), 1, 3))), axis=1)
        q = np.concatenate((q[:, ::-1], np.tile([1., 0, 0, 0], (len(q), 1, 1))), axis=1)
        actual_p, actual_q = shift.relative_link_pose(p, q, names)
        np.testing.assert_allclose(actual_p, expected_p, atol=1e-14)
        np.testing.assert_allclose(shift.quaternion_angle(actual_q, expected_q), 0, atol=1e-7)

    def test_bad_names_and_shapes_are_rejected(self):
        p, q = local_fixture()
        names = shift.REFERENCE_BODIES
        duplicate = list(names)
        duplicate[1] = duplicate[0]
        missing = ["absent_pelvis" if name == "pelvis" else name for name in names]
        cases = ((p, q, names[:-1]), (p, q, duplicate), (p, q, missing),
                 (p[0], q, names), (p, q[0], names), (p[..., :2], q, names),
                 (p, q[..., :3], names), (p, q[:-1], names),
                 (p[:, :-1], q, names), (p[:0], q[:0], names))
        for case in cases:
            with self.subTest(shapes=(np.shape(case[0]), np.shape(case[1])), names=case[2]), \
                    self.assertRaises(ValueError):
                shift.relative_link_pose(*case)

    def test_nonfinite_and_nonunit_pose_data_are_rejected(self):
        p, q = local_fixture()
        for value in (np.nan, np.inf, -np.inf):
            bad_p, bad_q = p.copy(), q.copy()
            bad_p[0, 1, 2] = value
            bad_q[0, 1, 2] = value
            for positions, quaternions in ((bad_p, q), (p, bad_q)):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    shift.relative_link_pose(positions, quaternions, shift.REFERENCE_BODIES)
        for scale in (0., 1.01):
            bad_q = q.copy()
            bad_q[0, 1] *= scale
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                shift.relative_link_pose(p, bad_q, shift.REFERENCE_BODIES)

    @unittest.skipIf(Rotation is None, "optional SciPy cross-check")
    def test_random_relative_rotations_match_scipy(self):
        p, q = world_fixture(*local_fixture(frames=5))
        actual_p, actual_q = shift.relative_link_pose(p, q, shift.REFERENCE_BODIES)
        pelvis = shift.REFERENCE_BODIES.index("pelvis")
        for frame in range(len(p)):
            rotations = Rotation.from_quat(q[frame, :, [1, 2, 3, 0]].T)
            root_inverse = rotations[pelvis].inv()
            expected_p = root_inverse.apply(p[frame] - p[frame, pelvis])
            expected_q = (root_inverse * rotations).as_quat()[:, [3, 0, 1, 2]]
            np.testing.assert_allclose(actual_p[frame], expected_p, atol=2e-14)
            np.testing.assert_allclose(np.abs(np.sum(actual_q[frame] * expected_q, axis=-1)),
                                       1, atol=2e-14)


class ReferenceStatisticsTests(unittest.TestCase):
    def test_link_and_group_units_and_pelvis_exclusion(self):
        names = shift.REFERENCE_BODIES
        self.assertIsInstance(names, tuple)
        self.assertEqual(len(names), 14)
        dp = np.arange(42, dtype=float).reshape(3, 14) / 100
        da = np.deg2rad(np.arange(42, dtype=float).reshape(3, 14))
        pelvis = names.index("pelvis")
        dp[:, pelvis] = da[:, pelvis] = 0
        report = shift.summarize_errors(dp, da)
        groups = {"all14": list(range(14)),
                  "nonpelvis13": [i for i, name in enumerate(names) if name != "pelvis"],
                  "wrists": [names.index(f"{side}_wrist_yaw_link") for side in ("left", "right")],
                  "ankles": [names.index(f"{side}_ankle_roll_link") for side in ("left", "right")],
                  "torso": [names.index("torso_link")]}
        self.assertEqual(set(report["per_link"]), set(names))
        self.assertEqual(set(report["groups"]), set(groups))
        for section, selectors in (("per_link", {name: [i] for i, name in enumerate(names)}),
                                   ("groups", groups)):
            for name, indexes in selectors.items():
                for metric, data in (("position_m", dp), ("rotation_deg", np.rad2deg(da))):
                    values = data[:, indexes].ravel()
                    with self.subTest(section=section, name=name, metric=metric):
                        self.assertAlmostEqual(report[section][name][metric]["mean"], float(values.mean()))
                        self.assertAlmostEqual(report[section][name][metric]["p95"],
                                               float(np.percentile(values, 95)))
        self.assertAlmostEqual(report["groups"]["all14"]["position_m"]["mean"] * 14 / 13,
                               report["groups"]["nonpelvis13"]["position_m"]["mean"])

    def test_origins_are_equal_weighted_and_p95_is_not_pooled(self):
        records, all_dp = [], []
        for origin, values in (("KIT/a", [0., 0, 0, 0, .2]), ("KIT/b", [.02] * 101)):
            dp = np.repeat(np.asarray(values)[:, None], 14, axis=1)
            dp[:, shift.REFERENCE_BODIES.index("pelvis")] = 0
            all_dp.append(dp)
            records.append({"origin_id": origin, "frame_count": len(values),
                            "statistics": shift.summarize_errors(dp, dp * 2)})
        actual = shift.aggregate_origins(records)
        for section in ("per_link", "groups"):
            for name in actual[section]:
                for metric in ("position_m", "rotation_deg"):
                    source = [record["statistics"][section][name][metric] for record in records]
                    self.assertAlmostEqual(actual[section][name][metric]["origin_equal_mean"],
                                           np.mean([stats["mean"] for stats in source]))
                    self.assertAlmostEqual(actual[section][name][metric]["mean_of_per_origin_p95"],
                                           np.mean([stats["p95"] for stats in source]))
        nonpelvis = actual["groups"]["nonpelvis13"]["position_m"]
        self.assertAlmostEqual(nonpelvis["origin_equal_mean"], .03)
        link = shift.REFERENCE_BODIES.index("left_wrist_yaw_link")
        pooled = np.concatenate(all_dp)[:, link]
        per_link = actual["per_link"]["left_wrist_yaw_link"]["position_m"]
        self.assertAlmostEqual(per_link["mean_of_per_origin_p95"], .09)
        self.assertNotAlmostEqual(per_link["origin_equal_mean"], pooled.mean())
        self.assertNotAlmostEqual(per_link["mean_of_per_origin_p95"], np.percentile(pooled, 95))

    def test_invalid_error_arrays_are_rejected(self):
        valid = np.zeros((2, 14))
        invalid = (np.zeros((0, 14)), np.zeros((2, 13)), np.zeros((2, 14, 1)),
                   np.zeros((1, 14)), np.full((2, 14), np.nan),
                   np.full((2, 14), np.inf), np.full((2, 14), -.01))
        for bad in invalid:
            for dp, da in ((bad, valid), (valid, bad)):
                with self.subTest(shapes=(dp.shape, da.shape)), self.assertRaises(ValueError):
                    shift.summarize_errors(dp, da)


class SyntheticDatasetIntegrationTests(unittest.TestCase):
    def test_all_27_origins_bind_54_packed_inputs_and_produce_descriptive_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            batch = root / "batch"
            batch.mkdir()
            inputs, rows, fk_rows, provenance = {}, [], [], []
            indexes = {key: {} for key in ("a", "b", "baseline", "candidate")}
            origins = [f"KIT/person/motion{i:02}" for i in range(27)]
            pipeline = "b" * 64

            def bind(path):
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                inputs[str(path)] = digest
                return digest

            def json_file(path, data):
                path.write_text(json.dumps(data), encoding="utf-8")
                return bind(path)

            for index, origin in enumerate(origins):
                source = root / f"{index:02}.source.npz"
                source.write_bytes(f"synthetic source {index}".encode())
                source_sha = bind(source)
                row = {"origin_id": origin, "source": str(source), "source_sha256": source_sha,
                       "split": "train" if index < 17 else "validation", "expected_packed_frames": 2}
                provenance.append({"origin_id": origin, "canonical_source": str(source),
                                   "source_sha256": source_sha})
                receipt = root / f"{index:02}.pair.json"
                receipt_sha = json_file(receipt, {
                    "origin_id": origin, "source_sha256": source_sha,
                    "frames": 3, "expected_packed_frames": 2, "fps": 50,
                    "common_window_proof": {"pair_start_s": 0., "pair_frames": 3,
                                            "pair_last_sample_s": .04}})
                proof = root / f"{index:02}.sampling.npz"
                np.savez(proof, times=np.arange(3) / 50, packed_indices=np.arange(3),
                         candidate_indices=np.arange(3))
                row.update(pair_receipt=str(receipt), pair_receipt_sha256=receipt_sha,
                           sampling_proof=str(proof), sampling_proof_sha256=bind(proof))
                for side in ("baseline", "candidate"):
                    native = root / f"{index:02}.{side}.pkl"
                    native.write_bytes(f"synthetic native {index} {side}".encode())
                    native_sha = bind(native)
                    row.update({side + "_pkl": str(native), side + "_pkl_sha256": native_sha})
                    packed = root / f"{index:02}.{side}.npz"
                    payload = packed_fixture(native_sha=native_sha, pipeline=pipeline)
                    if side == "candidate":
                        wrist = payload["body_names"].tolist().index("left_wrist_yaw_link")
                        payload["body_pos_w"][:, wrist, 0] += .02
                    np.savez(packed, **payload)
                    digest = bind(packed)
                    fk_rows.append({"origin_id": origin, "side": side, "path": str(packed),
                                    "sha256": digest, "frames": 2, "passed": True})
                    key = ("a" if side == "baseline" else "b") if index < 17 else side
                    indexes[key][origin] = str(packed)
                rows.append(row)

            entries = {}
            for key, mapping in indexes.items():
                path = root / f"{key}.yaml"
                entries[key] = {"index": str(path), "sha256": json_file(path, mapping)}
            pilot_sha = json_file(batch / "paired_manifest.json", {"rows": rows})
            fk_sha = json_file(batch / "body_fk_audit_20260909a.json", {"results": fk_rows})
            manifest = {
                "input_sha256": inputs, "resolved_input_aliases": {},
                "evidence": {"batch": str(batch), "pins": {"manifest": pilot_sha, "fk": fk_sha},
                             "paired_packaging_fingerprint": pipeline},
                "source_provenance": provenance, "target_origins": origins[:17],
                "arms": {key: entries[key] for key in ("a", "b")},
                "development": {"origin_ids": origins[17:],
                                "indices": {key: entries[key] for key in ("baseline", "candidate")}}}
            dataset = root / "manifest.json"
            dataset.write_text(json.dumps(manifest), encoding="utf-8")
            with mock.patch.object(shift.dataset_contract, "validate_manifest", return_value=manifest) as validate:
                report = shift.analyze_dataset(dataset)
            validate.assert_called_once_with(dataset)

            self.assertEqual(report["result"], "COMPLETE_DESCRIPTIVE_DIAGNOSTIC")
            self.assertTrue(report["post_hoc"])
            self.assertTrue(report["inputs_verified_unchanged"])
            self.assertFalse(report["v1_quality_gate"])
            self.assertFalse(report["policy_errors_measured"])
            self.assertFalse(report["automatic_promotion"])
            self.assertFalse(report["physics_stepped"])
            self.assertEqual(report["training_updates"], 0)
            self.assertIsNone(report["promotion_recommendation"])
            self.assertEqual(report["dataset_manifest_sha256"],
                             hashlib.sha256(dataset.read_bytes()).hexdigest())
            for path, digest in inputs.items():
                self.assertEqual(report["input_sha256"][path], digest)
            actual_payloads = []
            for split, count in (("train17", 17), ("dev10", 10)):
                actual = report["splits"][split]
                self.assertEqual(actual["origin_count"], count)
                self.assertEqual(actual["frames_per_side"], count * 2)
                self.assertEqual(len(actual["origins"]), count)
                self.assertAlmostEqual(actual["aggregate"]["groups"]["wrists"]
                                       ["position_m"]["origin_equal_mean"], .01)
                for record in actual["origins"]:
                    self.assertEqual(record["frame_count"], 2)
                    self.assertEqual(set(record["payloads"]), {"baseline", "candidate"})
                    actual_payloads.extend(payload["path"] for payload in record["payloads"].values())
                    self.assertAlmostEqual(record["statistics"]["per_link"]["left_wrist_yaw_link"]
                                           ["position_m"]["mean"], .02)
            self.assertEqual(len(actual_payloads), 54)
            self.assertEqual(len(set(actual_payloads)), 54)


class OutputContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "dataset.json"
        self.dataset.write_text("{}", encoding="utf-8")
        self.report = {"schema": "test-reference-shift", "origins": []}

    def call_main(self, *extra):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            result = shift.main(["--dataset", str(self.dataset), *map(str, extra)])
        return result, stdout.getvalue()

    def assert_output_rejected_before_analysis(self, output):
        with mock.patch.object(shift, "analyze_dataset", return_value=self.report) as analyze:
            with self.assertRaises((ValueError, OSError, SystemExit)) as raised:
                self.call_main("--outputnew", output)
            if isinstance(raised.exception, SystemExit):
                self.assertNotEqual(raised.exception.code, 0)
            analyze.assert_not_called()

    def test_default_prints_report_without_creating_output(self):
        before = set(self.root.iterdir())
        with mock.patch.object(shift, "analyze_dataset", return_value=self.report) as analyze:
            _, stdout = self.call_main()
        analyze.assert_called_once_with(self.dataset)
        self.assertEqual(json.loads(stdout), self.report)
        self.assertEqual(set(self.root.iterdir()), before)
        self.assertEqual(self.dataset.read_text(encoding="utf-8"), "{}")

    def test_new_output_contains_json_report(self):
        output = self.root / "new-report.json"
        with mock.patch.object(shift, "analyze_dataset", return_value=self.report):
            self.call_main("--outputnew", output)
        self.assertEqual(json.loads(output.read_text(encoding="utf-8")), self.report)

    def test_existing_output_is_preserved(self):
        output = self.root / "existing.json"
        output.write_text("keep this report", encoding="utf-8")
        self.assert_output_rejected_before_analysis(output)
        self.assertEqual(output.read_text(encoding="utf-8"), "keep this report")

    def test_missing_parent_is_not_created(self):
        output = self.root / "missing" / "report.json"
        self.assert_output_rejected_before_analysis(output)
        self.assertFalse(output.parent.exists())

    def test_dangling_output_symlink_is_preserved(self):
        output = self.root / "report.json"
        target = self.root / "absent.json"
        output.symlink_to(target)
        self.assert_output_rejected_before_analysis(output)
        self.assertTrue(output.is_symlink())
        self.assertFalse(target.exists())

    def test_symlink_in_any_parent_is_rejected(self):
        actual = self.root / "actual"
        (actual / "nested").mkdir(parents=True)
        alias = self.root / "alias"
        alias.symlink_to(actual, target_is_directory=True)
        for output in (alias / "report.json", alias / "nested" / "report.json"):
            with self.subTest(output=output):
                self.assert_output_rejected_before_analysis(output)
        self.assertEqual(list(actual.rglob("report.json")), [])

    def test_exclusive_create_preserves_file_created_during_analysis(self):
        output = self.root / "racing-report.json"

        def analyze(_dataset):
            output.write_text("another writer", encoding="utf-8")
            return self.report

        with mock.patch.object(shift, "analyze_dataset", side_effect=analyze):
            with self.assertRaises((ValueError, OSError, SystemExit)):
                self.call_main("--outputnew", output)
        self.assertEqual(output.read_text(encoding="utf-8"), "another writer")


if __name__ == "__main__":
    unittest.main()
