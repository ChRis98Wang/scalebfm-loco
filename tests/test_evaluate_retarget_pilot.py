from __future__ import annotations

from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import yaml

from scripts import evaluate_retarget_pilot as pilot


def archive_payload(frames=3, native_sha="a" * 64):
    root_quat = np.zeros((frames, 4), dtype=np.float32)
    root_quat[:, 0] = 1
    return {"format_version": 3, "fps": 50, "quaternion_order": "wxyz",
            "source_sha256": native_sha, "pipeline_fingerprint": "c" * 64,
            "joint_pos": np.zeros((frames, 29)), "joint_vel": np.zeros((frames, 29)),
            "body_pos_w": np.zeros((frames, 30, 3)),
            "body_quat_w": np.repeat(root_quat[:, None, :], 30, axis=1),
            "body_lin_vel_w": np.zeros((frames, 30, 3)), "body_ang_vel_w": np.zeros((frames, 30, 3)),
            "reference_root_pos": np.zeros((frames, 3)), "reference_root_quat_w": root_quat}


def pilot_rows():
    return [{"origin_id": "ACCAD/actor/walk", "packed_frames": 3, "dataset": "ACCAD",
             "baseline_packed_sha256": "c" * 64, "candidate_packed_sha256": "d" * 64,
             "split": "train", "kinematic_status": "PENDING_FROZEN_POLICY_EVALUATION"},
            {"origin_id": "KIT/actor/jump", "packed_frames": 4, "dataset": "KIT",
             "baseline_packed_sha256": "c" * 64, "candidate_packed_sha256": "d" * 64,
             "split": "validation", "kinematic_status": "REJECT_KINEMATIC"}]


def reports(side, value=.1):
    _, comparison, _ = pilot.helpers()
    result = []
    for mode, (name, bodies) in enumerate(comparison.G1_BFM_MODE_PRESETS):
        rows = [{"motion_id": index, "motion": row["origin_id"], "source_frames": row["packed_frames"],
                 "evaluated_steps": row["packed_frames"] - 1, "truncated": False,
                 "metrics": {key: {"mean": value, "max": value + .1}
                             for key in (*comparison.MASK_METRIC_KEYS, "error_body_pos_g")}}
                for index, row in enumerate(pilot_rows())]
        report = {"schema_version": 4, "task": "G1-BFM-Transformer-Tracking", "seed": 42,
                  "num_envs": 64, "max_steps": 1000, "step_dt": .02, "device": "cuda:0",
                  "torch": "test", "python": "3.12", "package_versions": {}, "gpu": "test", "cuda": "test",
                  "scene_variant": "baseline", "target_object": None, "mode_index": mode,
                  "mode": name, "active_body_names": list(bodies), "checkpoint_sha256": pilot.OFFICIAL_SHA,
                  "motion_index_sha256": ("a" if side == "baseline" else "b") * 64,
                  "input_manifest_verified_unchanged": True,
                  "input_manifest": {"checkpoint": {"sha256": pilot.OFFICIAL_SHA},
                                     "motion_index": {"sha256": ("a" if side == "baseline" else "b") * 64},
                                     "motions": {"sha256": ("c" if side == "baseline" else "d") * 64,
                                                 "files": {row["origin_id"]: {"sha256": row[f"{side}_packed_sha256"]}
                                                           for row in pilot_rows()}},
                                     "python_sources": {"sha256": "e" * 64, "files": {}}},
                  "protocol": {"training_updates": 0, "reset_disturbance": False,
                               "observation_noise": False, "interval_pushes": False,
                               "masked_tracking": {key: "test" for key in comparison.MASKED_PROTOCOL_FIELDS}},
                  "motions": rows, "summary": {"num_motions": 2, "evaluated_steps": 5, "truncated_motions": 0}}
        result.append(report)
    return result


class ArchiveTests(unittest.TestCase):
    def test_accepts_packed_v3_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "a.npz"
            np.savez(path, **archive_payload())
            self.assertEqual(pilot.validate_archive(path, native_sha="a" * 64, frames=3), "c" * 64)

    def test_rejects_wrong_provenance_frames_fps_and_quaternions(self):
        modifications = {"source_sha256": "b" * 64, "fps": 30, "format_version": 2,
                         "joint_pos": np.zeros((4, 29)), "body_quat_w": np.zeros((3, 30, 4)),
                         "reference_root_quat_w": np.full((3, 4), np.nan),
                         "pipeline_fingerprint": "untracked", "quaternion_order": "xyzw"}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "a.npz"
            for field, value in modifications.items():
                with self.subTest(field=field):
                    data = archive_payload()
                    data[field] = value
                    np.savez(path, **data)
                    with self.assertRaises(ValueError):
                        pilot.validate_archive(path, native_sha="a" * 64, frames=3)


class ReportTests(unittest.TestCase):
    def test_reference_payload_changes_are_intentional_and_rejects_remain(self):
        result = pilot.compare_reports(reports("baseline"), reports("candidate", .09), pilot_rows())
        self.assertEqual(result["aggregate_modes_passed"], 8)
        self.assertFalse(result["automatic_promotion"])
        self.assertEqual(result["training_updates"], 0)
        self.assertEqual(len(result["modes"][0]["origins"]), 2)
        self.assertEqual(result["modes"][0]["strata"]["kinematic_status/REJECT_KINEMATIC"]["num_origins"], 1)
        self.assertIn("not training acceptance", result["scope"])

    def test_all14_position_regression_fails_even_if_active_metrics_improve(self):
        candidate = reports("candidate", .09)
        for report in candidate:
            for row in report["motions"]:
                row["metrics"]["error_body_pos_g"] = {"mean": .2, "max": .3}
        result = pilot.compare_reports(reports("baseline"), candidate, pilot_rows())
        self.assertEqual(result["aggregate_modes_passed"], 0)
        self.assertFalse(any(result["per_origin_all_eight_modes_passed"].values()))

    def test_checkpoint_protocol_and_runtime_must_match(self):
        for key, value in (("checkpoint_sha256", "f" * 64), ("seed", 7),
                           ("num_envs", 128), ("step_dt", .01), ("python", "wrong")):
            with self.subTest(key=key):
                candidate = reports("candidate")
                for report in candidate:
                    report[key] = value
                with self.assertRaises(ValueError):
                    pilot.compare_reports(reports("baseline"), candidate, pilot_rows())

    def test_python_sources_cannot_change_while_payloads_change(self):
        candidate = reports("candidate")
        for report in candidate:
            report["input_manifest"]["python_sources"]["sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "Runtime Python sources"):
            pilot.compare_reports(reports("baseline"), candidate, pilot_rows())

    def test_full_windows_and_origins_are_required(self):
        for mutation in ("remove", "frames", "rename", "summary"):
            with self.subTest(mutation=mutation):
                report = reports("baseline")[0]
                if mutation == "remove":
                    report["motions"].pop()
                elif mutation == "frames":
                    report["motions"][0].update(source_frames=4, evaluated_steps=3)
                elif mutation == "rename":
                    report["motions"][0]["motion"] = "other"
                else:
                    report["summary"]["evaluated_steps"] = 4
                with self.assertRaises(ValueError):
                    pilot.validate_report(report, pilot_rows(), side="baseline", mode=0)

    def test_reports_reject_training_updates_and_nonfinite_metrics(self):
        for change in ("training", "nan", "negative"):
            report = reports("baseline")[0]
            if change == "training":
                report["protocol"]["training_updates"] = True
            else:
                report["motions"][0]["metrics"]["error_body_pos_g"]["mean"] = np.nan if change == "nan" else -1
            with self.subTest(change=change), self.assertRaises(ValueError):
                pilot.validate_report(report, pilot_rows(), side="baseline", mode=0)


class PilotPreflightTests(unittest.TestCase):
    def fixture(self, root):
        _, comparison, manifest_tools = pilot.helpers()
        rows, audits, index = [], [], {}
        candidate_root = root / "packed"
        candidate_root.mkdir()
        for i in range(2):
            name = f"ACCAD/actor/walk{i}"
            native = root / "sole_candidates" / f"{name}.pkl"
            native.parent.mkdir(parents=True, exist_ok=True)
            native.write_bytes(f"native{i}".encode())
            native_sha = manifest_tools.sha256_file(native)
            source = root / f"source{i}.npz"
            source.write_bytes(f"source{i}".encode())
            baseline = root / f"baseline{i}.npz"
            np.savez(baseline, **archive_payload(native_sha=native_sha))
            packed = candidate_root / f"{name}.npz"
            packed.parent.mkdir(parents=True, exist_ok=True)
            np.savez(packed, **archive_payload(native_sha=native_sha))
            row = {"origin_id": name, "dataset": "ACCAD", "split": "train", "index_label": "train",
                   "baseline_packed": str(baseline), "baseline_packed_sha256": manifest_tools.sha256_file(baseline),
                   "baseline_pkl": str(native), "baseline_pkl_sha256": native_sha,
                   "source": str(source), "canonical_source": str(source),
                   "source_sha256": manifest_tools.sha256_file(source), "packed_frames": 3}
            rows.append(row)
            index[name] = str(baseline)
            audits.append({"origin_id": name, "split": "train", "source_sha256": row["source_sha256"],
                           "candidate_pkl": str(native), "candidate_sha256": native_sha, "review_reasons": [],
                           "status": "REJECT_KINEMATIC" if i else "PENDING_FROZEN_POLICY_EVALUATION"})
        index_path = root / "train.yaml"
        index_path.write_text(yaml.safe_dump(index))
        manifest = {"schema": 1, "automatic_promotion": False, "seed": 42, "motions": rows,
                    "counts": {"ACCAD/train": 2}, "index_sha256": {"train": manifest_tools.sha256_file(index_path)}}
        (root / "manifest.json").write_text(json.dumps(manifest))
        (root / "sole_audit.json").write_text(json.dumps({"schema": 1, "automatic_promotion": False, "results": audits}))
        refresh = SimpleNamespace(INDEXES={"train": index_path}, resolve_inputs=lambda row: copy.deepcopy(row),
                                  validate_source_disjoint=lambda rows: None)
        return candidate_root, (refresh, comparison, manifest_tools)

    def test_valid_full_pilot_and_geometric_reject_retained(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidates, helpers = self.fixture(root)
            with mock.patch.object(pilot, "helpers", return_value=helpers):
                result = pilot.validate_pilot(root, candidates, expected_count=2)
            self.assertEqual(len(result["rows"]), 2)
            self.assertEqual(result["counts"]["REJECT_KINEMATIC"], 1)

    def test_rejects_missing_and_stray_candidate_files(self):
        for mutation in ("missing", "stray"):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                candidates, helpers = self.fixture(root)
                if mutation == "missing":
                    next(candidates.rglob("*.npz")).unlink()
                else:
                    (candidates / "stray.npz").touch()
                with mock.patch.object(pilot, "helpers", return_value=helpers):
                    with self.assertRaisesRegex(ValueError, "missing or stray"):
                        pilot.validate_pilot(root, candidates, expected_count=2)

    def test_rejects_changed_native_payload_or_canonical_index(self):
        for mutation in ("native", "index"):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                candidates, helpers = self.fixture(root)
                if mutation == "native":
                    next((root / "sole_candidates").rglob("*.pkl")).write_bytes(b"changed")
                else:
                    (root / "train.yaml").write_text("changed")
                with mock.patch.object(pilot, "helpers", return_value=helpers):
                    with self.assertRaises(ValueError):
                        pilot.validate_pilot(root, candidates, expected_count=2)


class ControllerTests(unittest.TestCase):
    def test_plan_is_read_only_and_uses_fixed_16_job_protocol(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "new"
            fake = {"rows": pilot_rows(), "counts": {"REJECT_KINEMATIC": 1}}
            stream = io.StringIO()
            with mock.patch.object(pilot, "validate_pilot", return_value=fake), redirect_stdout(stream), \
                    mock.patch.object(pilot, "run_job") as run:
                self.assertEqual(pilot.main(["--pilot", temporary, "--candidate-root", temporary, "--output", str(output)]), 0)
            run.assert_not_called()
            self.assertFalse(output.exists())
            plan = json.loads(stream.getvalue())
            self.assertEqual(len(plan["jobs"]), 16)
            self.assertEqual({job["mode"] for job in plan["jobs"]}, set(range(8)))
            for job in plan["jobs"]:
                self.assertEqual(job["timeout_seconds"], 240)
                self.assertEqual(job["command"][job["command"].index("--num_envs") + 1], "64")
                self.assertEqual(job["command"][job["command"].index("--checkpoint_path") + 1], str(pilot.OFFICIAL))

    def test_timeout_reaps_owned_process_group(self):
        process = mock.Mock(pid=12345)
        process.wait.side_effect = [subprocess.TimeoutExpired(["python"], 240), 0, 0]
        with tempfile.TemporaryDirectory() as temporary, \
                mock.patch.object(pilot.subprocess, "Popen", return_value=process) as start, \
                mock.patch.object(pilot.os, "killpg") as kill:
            with self.assertRaises(subprocess.TimeoutExpired):
                pilot.run_job({"command": ["python"], "timeout_seconds": 240}, Path(temporary) / "run.log")
            self.assertTrue(start.call_args.kwargs["start_new_session"])
            self.assertEqual(kill.call_count, 2)
            self.assertEqual(process.wait.call_count, 3)

    def test_execute_refuses_unowned_cgroup_before_data_changes(self):
        with mock.patch.object(pilot.Path, "read_text", return_value="0::/user.slice/other.service"):
            with self.assertRaisesRegex(ValueError, "owned bfm-retarget-eval"):
                pilot.owned_cgroup()


if __name__ == "__main__":
    unittest.main()
