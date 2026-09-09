from __future__ import annotations

from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import pickle
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import evaluate_umr_pairs as pairs
from scripts.umr_backend import G1_JOINT_NAMES


def native(frames=3):
    return {"fps": 50., "root_pos": np.tile([0., 0., .8], (frames, 1)),
            "root_rot": np.tile([0., 0., 0., 1.], (frames, 1)), "dof_pos": np.zeros((frames, 29))}


def archive(native_sha, frames=2):
    root = np.tile([0., 0., .8], (frames, 1))
    quat = np.tile([1., 0., 0., 0.], (frames, 1))
    return {"format_version": 3, "fps": 50, "quaternion_order": "wxyz", "source_sha256": native_sha,
            "pipeline_fingerprint": "b" * 64, "joint_pos": np.zeros((frames, 29)),
            "joint_vel": np.zeros((frames, 29)), "body_pos_w": np.repeat(root[:, None, :], 30, axis=1),
            "body_quat_w": np.repeat(quat[:, None, :], 30, axis=1),
            "body_lin_vel_w": np.zeros((frames, 30, 3)), "body_ang_vel_w": np.zeros((frames, 30, 3)),
            "reference_root_pos": root, "reference_root_quat_w": quat,
            "joint_names": np.array(pairs.ARTICULATION_JOINT_NAMES), "body_names": np.array(pairs.BODY_NAMES)}


class PairPreflightTests(unittest.TestCase):
    def fixture(self, root):
        refresh, comparison, tools = pairs.helpers()
        index_paths = {label: root / f"{label}.yaml" for label in pairs.INDEX_LABELS}
        indexes = {label: {} for label in pairs.INDEX_LABELS}
        rows = []
        for origin, label in (("ACCAD/actor/walk", "train"), ("KIT/actor/walk", "kit_validation")):
            source = root / "ScaleRetarget/dataset/amass" / f"{origin}.npz"
            source.parent.mkdir(parents=True, exist_ok=True)
            np.savez(source, trans=np.zeros((7, 3)), mocap_frame_rate=100.)
            row = {"origin_id": origin, "dataset": origin.split("/")[0], "index_label": label,
                   "split": "train" if label == "train" else "validation", "source": str(source),
                   "source_sha256": tools.sha256_file(source), "expected_packed_frames": 2}
            # Distinct source bytes across splits are essential, even in fixtures.
            if label != "train":
                np.savez(source, trans=np.ones((7, 3)), mocap_frame_rate=100.)
                row["source_sha256"] = tools.sha256_file(source)
            indexes[label][origin] = str(root / "old" / f"{origin}.npz")
            outputs = {}
            for side in pairs.SIDES:
                path = root / "input" / side / f"{origin}.pkl"
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("wb") as stream:
                    pickle.dump(native(), stream, protocol=4)
                digest = tools.sha256_file(path)
                row.update({f"{side}_pkl": str(path), f"{side}_pkl_sha256": digest})
                outputs[side] = {"path": str(path), "sha256": digest, "frames": 3}
                packed = root / "packed" / side / f"{origin}.npz"
                packed.parent.mkdir(parents=True, exist_ok=True)
                np.savez(packed, **archive(digest))
            proof = root / "pairs" / origin / "sampling_proof.npz"
            proof.parent.mkdir(parents=True)
            np.savez(proof, times=np.array([0., .02, .04]), packed_indices=np.arange(3),
                     candidate_indices=np.arange(3), native_times=np.array([0., .03, .06]),
                     baseline_native_lower=np.array([0, 0, 1]), baseline_native_upper=np.array([1, 1, 2]),
                     baseline_native_alpha=np.array([0., 2 / 3, 1 / 3]),
                     candidate_source_lower=np.array([0, 2, 4]), candidate_source_upper=np.array([1, 3, 5]),
                     candidate_source_alpha=np.zeros(3))
            outputs["sampling_proof"] = {"path": str(proof), "sha256": tools.sha256_file(proof)}
            receipt = {"schema": "bfm.umr_same_source_pair/1", "status": "PREPARED_NOT_QUALITY_ACCEPTED",
                       "automatic_promotion": False, "training_started": False, "fps": 50., "frames": 3,
                       "expected_packed_frames": 2, "origin_id": origin, "dataset": row["dataset"],
                       "split": row["split"], "source_sha256": row["source_sha256"],
                       "output_joint_names": list(G1_JOINT_NAMES), "outputs": outputs,
                       "input_sha256": {str(source): row["source_sha256"]},
                       "source_clock_proof": {"source_frames": 7, "source_fps": 100., "native_source_frame_zero": 0,
                                              "native_frames": 3, "native_actual_fps": 2 / .06},
                       "common_window_proof": {"pair_frames": 3, "pair_start_s": 0., "pair_last_sample_s": .04},
                       "alignment": {"per_frame_alignment": False, "shape_or_scale_fit": False,
                                     "baseline_constant_z_offset_m": 0., "candidate_constant_z_offset_m": 0.,
                                     "baseline_world_rotation": np.eye(3).tolist(),
                                     "candidate_world_rotation": np.eye(3).tolist()}}
            receipt_path = proof.parent / "pair_receipt.json"
            receipt_path.write_text(json.dumps(receipt))
            row.update(pair_receipt=str(receipt_path), pair_receipt_sha256=tools.sha256_file(receipt_path),
                       sampling_proof=str(proof), sampling_proof_sha256=tools.sha256_file(proof))
            rows.append(row)
        index_specs = {}
        for label, path in index_paths.items():
            path.write_text(yaml.safe_dump(indexes[label]))
            index_specs[label] = {"path": str(path), "sha256": tools.sha256_file(path)}
        selection_path = root / "selection.json"
        selected_rows = [{**row, "baseline_packed": indexes[row["index_label"]][row["origin_id"]]} for row in rows]
        selection_path.write_text(json.dumps({"schema": 1, "automatic_promotion": False, "seed": 42,
                                              "motions": selected_rows,
                                              "index_sha256": {label: value["sha256"] for label, value in index_specs.items()}}))
        self.selection_pin = tools.sha256_file(selection_path)
        manifest = {"schema": 1, "experiment_id": "unit_test", "origin_indexes": index_specs, "rows": rows,
                    "selection": str(selection_path), "selection_sha256": self.selection_pin,
                    "protected_files": {row["source"]: row["source_sha256"] for row in rows}}
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        fake_refresh = SimpleNamespace(INDEXES=index_paths, SOURCES=refresh.SOURCES,
                                       kinematic_qpos=refresh.kinematic_qpos,
                                       validate_source_disjoint=refresh.validate_source_disjoint)
        return manifest_path, root / "packed", (fake_refresh, comparison, tools)

    def validate(self, root, manifest, packed, helper):
        original_file = pairs._file

        def protected_file(path, **kwargs):
            path = Path(path)
            # Only canonical data roots are remapped into the fixture; real
            # helper source files remain immutable audit inputs.
            if str(path).startswith(str(root / "scripts")):
                path = ROOT / "scripts" / path.name
            return original_file(path, **kwargs)

        with mock.patch.object(pairs, "ROOT", root), mock.patch.object(pairs, "helpers", return_value=helper), \
                mock.patch.object(pairs, "ORIGINAL_SELECTION", root / "selection.json"), \
                mock.patch.object(pairs, "ORIGINAL_SELECTION_SHA", self.selection_pin), \
                mock.patch.object(pairs, "_file", side_effect=protected_file):
            return pairs.validate_pairs(manifest, packed, allowed_counts=(2,))

    def test_full_pair_validates_source_clock_and_named_packed_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, packed, helper = self.fixture(root)
            result = self.validate(root, manifest, packed, helper)
            self.assertEqual(len(result["rows"]), 2)
            self.assertEqual(result["packaging_fingerprint"], "b" * 64)
            self.assertTrue(result["all_origins_included"])

    def test_missing_or_extra_side_archive_rejected(self):
        for kind in ("missing", "extra"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest, packed, helper = self.fixture(root)
                if kind == "missing":
                    next((packed / "candidate").rglob("*.npz")).unlink()
                else:
                    (packed / "stray.npz").touch()
                with self.assertRaisesRegex(ValueError, "missing/extra NPZ"):
                    self.validate(root, manifest, packed, helper)

    def test_canonical_index_change_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, packed, helper = self.fixture(root)
            helper[0].INDEXES["train"].write_text("changed")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                self.validate(root, manifest, packed, helper)

    def test_moved_origin_and_changed_native_rejected(self):
        for kind in ("split", "native", "count", "name", "value"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest, packed, helper = self.fixture(root)
                data = json.loads(manifest.read_text())
                if kind == "split":
                    data["rows"][0]["split"] = "validation"
                    manifest.write_text(json.dumps(data))
                elif kind == "native":
                    Path(data["rows"][0]["candidate_pkl"]).write_bytes(b"changed")
                elif kind == "count":
                    data["rows"] = data["rows"][:1]
                    manifest.write_text(json.dumps(data))
                else:
                    path = next((packed / "candidate").rglob("*.npz"))
                    with np.load(path) as source:
                        values = dict(source)
                    if kind == "name":
                        values["joint_names"][0] = "incorrect"
                    else:
                        values["joint_pos"][0, 0] = .1
                    np.savez(path, **values)
                with self.assertRaises(ValueError):
                    self.validate(root, manifest, packed, helper)

    def test_sampling_clock_change_cannot_hide_behind_updated_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, packed, helper = self.fixture(root)
            data = json.loads(manifest.read_text())
            row = data["rows"][0]
            path = Path(row["sampling_proof"])
            with np.load(path) as original:
                values = dict(original)
            values["candidate_source_alpha"][0] = .5
            np.savez(path, **values)
            row["sampling_proof_sha256"] = helper[2].sha256_file(path)
            receipt_path = Path(row["pair_receipt"])
            receipt = json.loads(receipt_path.read_text())
            receipt["outputs"]["sampling_proof"]["sha256"] = row["sampling_proof_sha256"]
            receipt_path.write_text(json.dumps(receipt))
            row["pair_receipt_sha256"] = helper[2].sha256_file(receipt_path)
            manifest.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "different original times"):
                self.validate(root, manifest, packed, helper)

    def test_new_preselection_cannot_replace_frozen_cohort(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, packed, helper = self.fixture(root)
            selection_path = root / "selection.json"
            original = json.loads(selection_path.read_text())
            original["motions"][0]["origin_id"] = "ACCAD/different/walk"
            selection_path.write_text(json.dumps(original))
            updated = json.loads(manifest.read_text())
            updated["selection_sha256"] = helper[2].sha256_file(selection_path)
            manifest.write_text(json.dumps(updated))
            with self.assertRaisesRegex(ValueError, "original pinned preselection"):
                self.validate(root, manifest, packed, helper)

    def test_equal_body_name_sets_in_wrong_runtime_order_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "motion.npz"
            values = archive("a" * 64)
            values["body_names"] = values["body_names"][::-1]
            np.savez(path, **values)
            with self.assertRaisesRegex(ValueError, "policy articulation order"):
                pairs._archive_names(path)


class GeometryAndControllerTests(unittest.TestCase):
    def test_motion_extent_distinguishes_amplitude_without_rescaling(self):
        qpos = np.zeros((3, 36))
        qpos[:, :3] = [[0., 0., .8], [.02, 0., .9], [.02, .03, .8]]
        qpos[:, 3] = 1.
        qpos[:, 7] = [0., .1, .2]
        before = qpos.copy()
        original = pairs.motion_extent(qpos)
        angle = .8
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        qpos[:, :2] = qpos[:, :2] @ rotation.T + [12., -3.]
        transformed = pairs.motion_extent(qpos)
        for key in ("root_total_path_length_xyz_m", "root_total_path_length_xy_m", "root_horizontal_span_xy_m"):
            self.assertAlmostEqual(original[key], transformed[key], places=12)
        self.assertEqual(original["root_height_m"], transformed["root_height_m"])
        self.assertEqual(original["joint_range_rad"], transformed["joint_range_rad"])
        shrunk = before.copy()
        shrunk[:, :2] *= .8
        self.assertLess(pairs.motion_extent(shrunk)["root_horizontal_span_xy_m"], original["root_horizontal_span_xy_m"])
        self.assertAlmostEqual(original["root_height_m"]["range"], .1)
        self.assertAlmostEqual(original["joint_range_rad"]["max"], .2)

    def test_velocity_statistics_are_separate_and_quaternion_sign_invariant(self):
        qpos = np.zeros((3, 36))
        qpos[:, 0] = [0., .02, .04]
        qpos[:, 3] = [1., -1., 1.]
        qpos[:, 7] = [0., .04, .08]
        stats = pairs.velocity_statistics(qpos)
        self.assertAlmostEqual(stats["root_linear_speed_m_s"]["mean"], 1.)
        self.assertAlmostEqual(stats["root_angular_speed_rad_s"]["max"], 0.)
        self.assertAlmostEqual(stats["joint_absolute_speed_rad_s"]["max"], 2.)
        qpos[0, 3] = 0.
        with self.assertRaisesRegex(ValueError, "unit root"):
            pairs.velocity_statistics(qpos)

    def test_geometry_rejects_remain_in_report_rows(self):
        rows = [{"origin_id": "KIT/a/walk", "split": "validation", "dataset": "KIT", "packed_frames": 2,
                 "baseline_packed_sha256": "a" * 64, "candidate_packed_sha256": "b" * 64}]
        metrics = {"frames": 2, "foot_penetration_frames_gt_1mm": 0,
                   "max_all_body_ground_penetration_m": 0., "max_self_penetration_m": .006,
                   "joint_limit_violations_gt_1e_minus6_rad": 0}
        reason = pairs.geometry_reasons(metrics)
        record = {"origin_id": "KIT/a/walk", "split": "validation", "dataset": "KIT",
                  "kinematic_status": "REJECT_PACKED_KINEMATIC"}
        for side in pairs.SIDES:
            record[side] = {"metrics": metrics, "review_reasons": reason, "kinematic_pass": False,
                            "velocity_statistics": {"root": {"mean": 0., "max": 0., "p95": 0.}},
                            "packed_sha256": rows[0][f"{side}_packed_sha256"]}
        report = {"schema": 1, "experiment_id": "test", "result": "COMPLETE", "automatic_promotion": False,
                  "physics_stepped": False, "training_updates": 0, "results": [record]}
        bound = pairs.bind_geometry({"experiment_id": "test", "rows": rows}, report)
        self.assertEqual(len(bound), 1)
        self.assertEqual(bound[0]["kinematic_status"], "REJECT_PACKED_KINEMATIC")
        changed = copy.deepcopy(report)
        changed["results"][0]["kinematic_status"] = "PENDING_FROZEN_POLICY_EVALUATION"
        with self.assertRaisesRegex(ValueError, "reject must stay labelled"):
            pairs.bind_geometry({"experiment_id": "test", "rows": rows}, changed)

    def test_wrong_cgroup_or_killmode_rejected(self):
        with mock.patch.object(pairs.Path, "read_text", return_value="0::/user.slice/other.service"):
            with self.assertRaisesRegex(ValueError, "owned bfm-umr-eval"):
                pairs.owned_cgroup()
        with mock.patch.object(pairs.Path, "read_text", return_value="0::/user.slice/bfm-umr-eval-test.service"), \
                mock.patch.object(pairs.subprocess, "check_output", return_value="process"):
            with self.assertRaisesRegex(ValueError, "KillMode=control-group"):
                pairs.owned_cgroup()

    def test_default_plan_is_read_only_and_16_jobs_never_train(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new"
            fake = {"manifest": str(Path(directory) / "manifest.json"), "experiment_id": "test",
                    "rows": [{"origin_id": "KIT/a"}], "counts": {"KIT/train": 1}}
            stream = io.StringIO()
            with mock.patch.object(pairs, "validate_pairs", return_value=fake), \
                    mock.patch.object(pairs.paired, "run_job") as run, redirect_stdout(stream):
                self.assertEqual(pairs.main(["--manifest", fake["manifest"], "--packed-root", directory,
                                             "--output", str(output)]), 0)
            run.assert_not_called()
            self.assertFalse(output.exists())
            report = json.loads(stream.getvalue())
            self.assertEqual(len(report["jobs"]), 16)
            self.assertEqual({j["mode"] for j in report["jobs"]}, set(range(8)))
            self.assertFalse(report["automatic_promotion"])
            self.assertTrue(report["geometry_rejects_remain_in_evaluation"])
            for job in report["jobs"]:
                self.assertIn("--mask_metrics", job["command"])
                self.assertNotIn("train.py", " ".join(job["command"]))


if __name__ == "__main__":
    unittest.main()
