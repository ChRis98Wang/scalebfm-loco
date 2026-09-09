import copy
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest import mock

import numpy as np

from scripts import umr_pair_dataset as pair


def payload(count=31, fps=30.):
    t = np.arange(count) / fps
    return {"fps": fps, "root_pos": np.column_stack((t, 2 * t, 1 + .1 * t)),
            "root_rot": np.tile([0., 0., 0., 1.], (count, 1)),
            "dof_pos": t[:, None] * np.linspace(.01, .29, 29)[None, :]}


def prepared_clock(count=121, fps=120., duration=5.):
    values = pair.surface.sampling_grid(count, fps, 50., 0., duration)
    return dict(zip(("times", "sample_lower", "sample_upper", "sample_alpha"), values)) | {
        "metadata": {"start": 0., "target_fps": 50., "requested_duration": duration,
                     "source_fps": fps}}


class PairClockTests(unittest.TestCase):
    def test_endpoint_preserving_actual_fps_not_relabelled_thirty(self):
        times, proof = pair.legacy_source_clock(331, 120., 83, 82 / 2.75)
        self.assertEqual(len(times), 83)
        self.assertEqual(times[-1], 2.75)
        self.assertNotEqual(proof["native_actual_fps"], 30.)
        with self.assertRaisesRegex(ValueError, "endpoint-preserving"):
            pair.legacy_source_clock(331, 120., 83, 30.)

    def test_original_thirty_hz_has_identity_sampling(self):
        times, _ = pair.legacy_source_clock(31, 30., 31, 30.)
        np.testing.assert_allclose(times, np.arange(31) / 30.)

    def test_wrong_native_count_rejected(self):
        with self.assertRaisesRegex(ValueError, "endpoint-preserving"):
            pair.legacy_source_clock(121, 120., 30, 30.)

    def test_inclusive_candidate_exclusive_baseline_intersection(self):
        pre = prepared_clock()
        indices, proof = pair.common_grid(pre, 121, 120., np.arange(50, dtype=np.float32) * .02)
        np.testing.assert_array_equal(indices, np.arange(50))
        self.assertEqual(proof["dropped_candidate_frames"], 1)
        self.assertEqual(proof["pair_last_sample_s"], .98)

    def test_different_clock_cannot_pass_by_matching_length(self):
        pre = prepared_clock()
        pre["times"] += .001
        with self.assertRaisesRegex(ValueError, "sampling proof"):
            pair.common_grid(pre, 121, 120., np.arange(50) / 50.)

    def test_source_interpolation_alpha_is_proven_independently(self):
        pre = prepared_clock()
        pre["sample_alpha"][2] += .1
        with self.assertRaisesRegex(ValueError, "sample_alpha"):
            pair.common_grid(pre, 121, 120., np.arange(50) / 50.)

    def test_nonzero_start_rejected_in_this_pilot(self):
        pre = prepared_clock()
        pre["metadata"]["start"] = .02
        with self.assertRaisesRegex(ValueError, "frame-zero"):
            pair.common_grid(pre, 121, 120., np.arange(50) / 50.)

    def test_shifted_packed_grid_rejected(self):
        with self.assertRaisesRegex(ValueError, "frame zero"):
            pair.common_grid(prepared_clock(), 121, 120., (np.arange(50) + 1) / 50.)

    def test_packer_values_prove_all_fields_with_explicit_permutation(self):
        interp = pair.package_interpolation(payload())
        names = list(pair.G1_JOINT_NAMES)[::-1]
        packed = {"joint_pos": interp["dof_pos"][:, ::-1],
                  "reference_root_pos": interp["root_pos"],
                  "reference_root_quat_w": -interp["root_quat_wxyz"]}
        qpos, proof = pair.prove_packed_values(packed, interp, names)
        self.assertEqual(qpos.shape, (50, 36))
        self.assertEqual(proof["joint_max_abs_error_rad"], 0.)
        packed["joint_pos"] = packed["joint_pos"].copy()
        packed["joint_pos"][12, 3] += .01
        with self.assertRaisesRegex(ValueError, "interpolation proof"):
            pair.prove_packed_values(packed, interp, names)

    def test_bad_packed_quaternion_does_not_hide_in_legacy_roundoff_tolerance(self):
        interp = pair.package_interpolation(payload())
        packed = {"joint_pos": interp["dof_pos"], "reference_root_pos": interp["root_pos"],
                  "reference_root_quat_w": interp["root_quat_wxyz"].copy()}
        packed["reference_root_quat_w"][1] = [np.cos(.02), 0., 0., np.sin(.02)]
        with self.assertRaisesRegex(ValueError, "quaternion"):
            pair.prove_packed_values(packed, interp, list(pair.G1_JOINT_NAMES))

    def test_constant_heading_and_initial_xy_alignment_preserves_z_and_motion(self):
        qpos, _ = pair.packed_tools.refresh.kinematic_qpos(payload(4))
        candidate = qpos.copy()
        candidate[:, :2] += [5., -2.]
        heading = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        original = qpos.copy()
        before, after, proof = pair.align_pair(qpos, candidate, heading)
        np.testing.assert_array_equal(qpos, original)
        np.testing.assert_array_equal(before[:, 2], qpos[:, 2])
        np.testing.assert_array_equal(after[:, 2], candidate[:, 2])
        np.testing.assert_allclose(before[0, :2], 0.)
        np.testing.assert_allclose(after[0, :2], 0.)
        np.testing.assert_allclose(np.diff(before[:, :3], axis=0), np.diff(qpos[:, :3], axis=0) @ heading.T)
        self.assertFalse(proof["per_frame_alignment"])
        self.assertFalse(proof["shape_or_scale_fit"])

    def test_reflection_and_non_yaw_rotation_rejected(self):
        q, _ = pair.packed_tools.refresh.kinematic_qpos(payload(4))
        for matrix in (np.diag([-1., 1., 1.]), np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])):
            with self.assertRaisesRegex(ValueError, "Z-axis-only"):
                pair.align_pair(q, q, matrix)


class PairIntegrationTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        row = {"origin_id": "KIT/actor/walk", "split": "validation", "dataset": "KIT"}
        native = payload()
        source = root / "source.npz"
        np.savez(source, trans=np.zeros((121, 3)), mocap_frame_rate=120.)
        baseline = root / "native.pkl"
        with baseline.open("wb") as stream:
            pickle.dump(native, stream, protocol=4)
        interp = pair.package_interpolation(native)
        packed = root / "packed.npz"
        np.savez(packed, format_version=3, fps=50, quaternion_order="wxyz",
                 source_sha256=pair.sha256(baseline), pipeline_fingerprint="a" * 64,
                 joint_pos=interp["dof_pos"][:, ::-1], reference_root_pos=interp["root_pos"],
                 reference_root_quat_w=interp["root_quat_wxyz"])
        for key, path in (("source", source), ("baseline_pkl", baseline), ("baseline_packed", packed)):
            row[key], row[f"{key}_sha256"] = str(path), pair.sha256(path)
        names_path = pair.ROOT / "ScaleTrack/source/scaletrack/scaletrack/robots/g1_29dof.py"
        audit = {"joint_order_proof": {"all_columns_uniquely_proven": True,
            "source_joint_names": list(pair.G1_JOINT_NAMES),
            "packed_joint_names": list(pair.G1_JOINT_NAMES)[::-1],
            "packed_to_source_indices": list(range(29))[::-1], "maximum_matching_error_rad": 0.},
            "input_sha256": {str(names_path): pair.sha256(names_path)} | {
                row[key]: row[f"{key}_sha256"] for key in ("source", "baseline_pkl", "baseline_packed")},
            "results": [{key: row[key] for key in ("origin_id", "split", "source_sha256", "baseline_packed")}]}
        audit_path = root / "audit.json"
        audit_path.write_text(json.dumps(audit))
        pre = prepared_clock()
        pre["metadata"] |= {"source_file": str(source), "source_sha256": row["source_sha256"],
                            "posed_heading_rotation": np.eye(3).tolist(), "posed_xy_offset": [0., 0.]}
        prepared_path = root / "surface.npz"
        np.savez(prepared_path, metadata_json=json.dumps(pre["metadata"]))
        candidate = root / "umr.npz"
        qpos, _ = pair.packed_tools.refresh.kinematic_qpos(payload(51, 50.))
        candidate_meta = {"schema": "bfm.umr_smplx_trial/1", "source": pre["metadata"],
                          "prepared_source_sha256": pair.sha256(prepared_path),
                          "protected_inputs_rechecked": True, "material_surface_transport": True,
                          "umr_commit": pair.surface.UMR_COMMIT}
        np.savez(candidate, qpos=qpos, fps=50., dof_names=np.array(pair.G1_JOINT_NAMES),
                 frame_indices=np.arange(51), metadata_json=json.dumps(candidate_meta))
        kwargs = dict(prepared_source=prepared_path, candidate_npz=candidate,
                      packed_audit=audit_path, output_dir=root / "pair")
        return row, pre, kwargs

    def test_complete_same_source_pair_receipt_protocol_four_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            row, pre, kwargs = self.fixture(directory)
            with mock.patch.object(pair.surface, "load_prepared_source", return_value=pre):
                receipt = pair.prepare_pair(row, **kwargs)
                self.assertEqual(receipt["frames"], 50)
                self.assertEqual(receipt["expected_packed_frames"], 49)
                self.assertEqual(receipt["split"], "validation")
                self.assertFalse(receipt["automatic_promotion"])
                for label in ("baseline", "candidate"):
                    path = Path(receipt["outputs"][label]["path"])
                    self.assertEqual(path.read_bytes()[:2], b"\x80\x04")
                    self.assertEqual(pair.sha256(path), receipt["outputs"][label]["sha256"])
                    with path.open("rb") as stream:
                        motion = pickle.load(stream)
                    self.assertEqual(motion["dof_pos"].shape, (50, 29))
                with self.assertRaisesRegex(ValueError, "existing"):
                    pair.prepare_pair(row, **kwargs)

    def test_source_hash_change_fails_before_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            row, pre, kwargs = self.fixture(directory)
            row["source_sha256"] = "f" * 64
            with self.assertRaisesRegex(ValueError, "Manifest input changed"):
                pair.prepare_pair(row, **kwargs)
            self.assertFalse(kwargs["output_dir"].exists())

    def test_candidate_prepared_source_misbinding_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            row, pre, kwargs = self.fixture(directory)
            pre = copy.deepcopy(pre)
            pre["metadata"]["source_sha256"] = "f" * 64
            with mock.patch.object(pair.surface, "load_prepared_source", return_value=pre):
                with self.assertRaisesRegex(ValueError, "exact source origin"):
                    pair.prepare_pair(row, **kwargs)
            self.assertFalse(kwargs["output_dir"].exists())

    def test_joint_order_proof_cannot_be_swapped(self):
        with tempfile.TemporaryDirectory() as directory:
            row, pre, kwargs = self.fixture(directory)
            audit = json.loads(kwargs["packed_audit"].read_text())
            audit["joint_order_proof"]["packed_to_source_indices"] = list(range(29))
            kwargs["packed_audit"].write_text(json.dumps(audit))
            with self.assertRaisesRegex(ValueError, "contradict"):
                pair.prepare_pair(row, **kwargs)


if __name__ == "__main__":
    unittest.main()
