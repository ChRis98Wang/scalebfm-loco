from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts import compare_umr_behavior_ab as comparator


def fixtures():
    names = [f"ACCAD/group/clip_{i:04d}" if i < 962 else f"KIT/group/clip_{i:04d}" for i in range(1751)]
    files = {name: {"path": f"/fixed/{name}.npz", "size_bytes": 1, "sha256": "2" * 64} for name in names}
    runtime = {"runtime.py": {"path": "/fixed/runtime.py", "size_bytes": 1, "sha256": "3" * 64}}
    checkpoints = {"official": comparator.OFFICIAL_SHA, "a": "a" * 64, "b": "b" * 64}
    families = {}
    for label, checkpoint in checkpoints.items():
        rows = []
        for index, name in enumerate(names):
            steps = 1000 if index < 98 else 308 if index < 291 else 307
            frames = 1101 if index < 98 else steps + 1
            rows.append({"motion_id": index, "motion": name, "source_frames": frames,
                "evaluated_steps": steps, "truncated": index < 98,
                "metrics": {key: {"mean": .1, "max": .2} for key in
                            (*comparator.masks.MASK_METRIC_KEYS, "error_body_pos_g")}})
        snapshot = {"checkpoint": {"path": f"/checkpoints/{label}.pt", "size_bytes": 1, "sha256": checkpoint},
                    "motion_index": {"path": "/fixed/index.yaml", "size_bytes": 1, "sha256": "4" * 64},
                    "motions": {"files": files, "sha256": comparator.group_digest(files)},
                    "python_sources": {"files": runtime, "sha256": comparator.group_digest(runtime)}}
        families[label] = []
        for mode, (name, bodies) in enumerate(comparator.masks.G1_BFM_MODE_PRESETS):
            families[label].append({"schema_version": 4, "task": comparator.masks.G1_BFM_TASK,
                "seed": 42, "num_envs": 1024, "max_steps": 1000, "step_dt": .02, "device": "cuda:0",
                "torch": "test", "python": "test", "package_versions": {}, "gpu": "test", "cuda": "test",
                "scene_variant": "baseline", "target_object": None, "input_manifest_verified_unchanged": True,
                "protocol": {"training_updates": 0, "reset_disturbance": False, "observation_noise": False,
                    "interval_pushes": False, "masked_tracking": {key: key for key in comparator.masks.MASKED_PROTOCOL_FIELDS}},
                "mode_index": mode, "mode": name, "active_body_names": list(bodies),
                "checkpoint": snapshot["checkpoint"]["path"], "checkpoint_sha256": checkpoint,
                "motion_index": snapshot["motion_index"]["path"], "motion_index_sha256": "4" * 64,
                "input_manifest": snapshot, "motions": rows,
                "summary": {"num_motions": 1751, "evaluated_steps": 605664, "truncated_motions": 98}})
    return families, checkpoints, files


class BehaviorAbComparisonTests(unittest.TestCase):
    def setUp(self):
        self.families, self.checkpoints, self.files = fixtures()

    def compare(self):
        return comparator.compare_ab(self.families["official"], self.families["a"], self.families["b"],
            expected_checkpoints=self.checkpoints, expected_motions=self.files)

    def test_equal_results_pass_this_seed_but_never_promote(self):
        result = self.compare()
        self.assertTrue(result["promotion_candidate_no_regression"])
        self.assertTrue(result["confirmation_quality_condition_met"])
        self.assertFalse(result["promotion_approved"])
        self.assertFalse(result["automatic_promotion"])
        self.assertFalse(result["independent_seed43_44_confirmation_complete"])
        self.assertEqual(set(result["comparisons"]), {"b_vs_a", "b_vs_official"})

    def test_all_body_mean_regression_cannot_hide_behind_original_four_gates(self):
        for row in self.families["b"][0]["motions"]:
            row["metrics"]["error_body_pos_g"]["mean"] = .11
        result = self.compare()["comparisons"]["b_vs_a"]
        self.assertTrue(result["original_four_gates_no_regression"])
        self.assertFalse(result["additional_three_means_no_regression"])
        self.assertFalse(result["promotion_candidate_no_regression"])

    def test_kit_mean_regression_not_hidden_by_nonkit_improvement(self):
        for row in self.families["b"][0]["motions"]:
            row["metrics"]["error_body_pos_g"]["mean"] = .101 if row["motion"].startswith("KIT/") else .09
        extra = self.compare()["comparisons"]["b_vs_a"]["additional_three_mean_comparison"]
        self.assertTrue(extra["groups"]["aggregate"]["candidate_no_regression"])
        self.assertFalse(extra["groups"]["kit"]["candidate_no_regression"])

    def test_b_beating_a_but_losing_official_is_not_promotion(self):
        for row in self.families["a"][0]["motions"]:
            row["metrics"]["error_body_pos_g"]["mean"] = .13
        for row in self.families["b"][0]["motions"]:
            row["metrics"]["error_body_pos_g"]["mean"] = .11
        result = self.compare()
        self.assertTrue(result["comparisons"]["b_vs_a"]["promotion_candidate_no_regression"])
        self.assertFalse(result["comparisons"]["b_vs_official"]["promotion_candidate_no_regression"])
        self.assertTrue(result["confirmation_quality_condition_met"])
        self.assertFalse(result["promotion_candidate_no_regression"])

    def test_active_pass_rate_regression_keeps_original_gate(self):
        self.families["b"][0]["motions"][0]["metrics"]["error_active_body_pos_g_max"]["max"] = .51
        result = self.compare()["comparisons"]["b_vs_a"]
        self.assertFalse(result["original_four_gates_no_regression"])
        self.assertTrue(result["additional_three_means_no_regression"])

    def test_missing_third_mean_rejected(self):
        for row in self.families["b"][0]["motions"]:
            del row["metrics"]["error_body_pos_g"]
        with self.assertRaisesRegex(comparator.ComparisonError, "three-mean metric missing"):
            self.compare()

    def test_negative_third_mean_rejected(self):
        self.families["b"][0]["motions"][0]["metrics"]["error_body_pos_g"]["mean"] = -.1
        with self.assertRaisesRegex(comparator.ComparisonError, "nonnegative"):
            self.compare()

    def test_distinct_checkpoint_identity_required(self):
        self.checkpoints["b"] = self.checkpoints["a"]
        with self.assertRaisesRegex(comparator.ComparisonError, "distinct"):
            self.compare()

    def test_runtime_hash_mismatch_cannot_reuse_same_group_digest(self):
        self.families["b"][0]["input_manifest"] = deepcopy(self.families["b"][0]["input_manifest"])
        self.families["b"][0]["input_manifest"]["python_sources"]["files"]["runtime.py"]["sha256"] = "7" * 64
        with self.assertRaisesRegex(comparator.ComparisonError, "group hash"):
            self.compare()

    def test_checkpoint_seed_noise_and_coverage_rejected(self):
        for field, value in (("checkpoint_sha256", "8" * 64), ("seed", 43), ("num_envs", 64)):
            before = self.families["b"][0][field]; self.families["b"][0][field] = value
            with self.subTest(field=field), self.assertRaises(comparator.ComparisonError):
                self.compare()
            self.families["b"][0][field] = before
        for report in self.families["b"]:
            report["protocol"]["observation_noise"] = True
        with self.assertRaisesRegex(comparator.ComparisonError, "perturbations"):
            self.compare()

    def test_summary_cannot_hide_incomplete_actual_windows(self):
        row = self.families["b"][0]["motions"][100]
        row.update(source_frames=308, evaluated_steps=307)
        with self.assertRaisesRegex(comparator.ComparisonError, "steps/truncations"):
            self.compare()

    def test_existing_cli_output_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "comparison.json"; path.write_text("keep")
            argv = ["--inputs", "unused", "--outputnew", str(path)]
            for label in ("official", "a", "b"):
                argv += [f"--{label}8"] + ["unused"] * 8
            with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
                comparator.main(argv)
            self.assertEqual(path.read_text(), "keep")

    def test_zero_byte_runtime_module_record_is_valid(self):
        comparator.validate_record({"path": "/runtime/__init__.py", "size_bytes": 0, "sha256": "0" * 64}, "module")

    def development(self):
        files = {name: record for name, record in list(self.files.items())[:10]}
        for label in ("a", "b"):
            rows = deepcopy(self.families[label][0]["motions"][:10])
            for row in rows:
                row.update(source_frames=20, evaluated_steps=19, truncated=False)
            for report in self.families[label]:
                report["motions"] = rows
                report["num_envs"] = 64
                report["input_manifest"] = deepcopy(report["input_manifest"])
                report["input_manifest"]["motions"] = {"files": files, "sha256": comparator.group_digest(files)}
                report["summary"] = {"num_motions": 10, "evaluated_steps": 190, "truncated_motions": 0}
        return files

    def compare_dev(self, files, passed=None):
        return comparator.compare_development(self.families["a"], self.families["b"],
            expected_checkpoints={label: self.checkpoints[label] for label in ("a", "b")},
            expected_motions=files, geometric_pass_origins=list(files)[:7] if passed is None else passed)

    def test_development_all10_pass7_reject3_are_preserved(self):
        result = self.compare_dev(self.development())
        self.assertTrue(result["diagnostic_only"])
        self.assertFalse(result["promotion_approved"])
        self.assertEqual({key: len(value) for key, value in result["cohorts"].items()},
                         {"all10": 10, "geometry_pass7": 7, "geometry_reject3": 3})
        self.assertTrue(result["three_mean_comparison"]["candidate_no_regression"])

    def test_development_reject_stratum_cannot_be_silently_dropped(self):
        files = self.development()
        self.families["b"][0]["motions"][-1]["metrics"]["error_body_pos_g"]["mean"] = .15
        result = self.compare_dev(files)["three_mean_comparison"]
        self.assertTrue(result["groups"]["geometry_pass7"]["candidate_no_regression"])
        self.assertFalse(result["groups"]["geometry_reject3"]["candidate_no_regression"])

    def test_development_mismatched_reference_rejected(self):
        files = self.development()
        changed = deepcopy(files); changed[next(iter(changed))]["sha256"] = "f" * 64
        for report in self.families["b"]:
            report["input_manifest"]["motions"] = {"files": changed, "sha256": comparator.group_digest(changed)}
        with self.assertRaisesRegex(comparator.ComparisonError, "same-reference"):
            self.compare_dev(files)

    def test_development_geometry_subset_must_be_declared_seven(self):
        files = self.development()
        with self.assertRaisesRegex(comparator.ComparisonError, "seven"):
            self.compare_dev(files, list(files)[:6])


if __name__ == "__main__":
    unittest.main()
