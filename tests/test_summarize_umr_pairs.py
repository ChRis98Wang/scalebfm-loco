"""Synthetic read-only summary tests; no simulator, Torch, downloads or jobs."""
import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts import summarize_umr_pairs as summary


def rows():
    result = []
    for dataset in summary.SOURCES:
        for index in range(8):
            result.append({"origin_id": f"{dataset}/actor/motion{index}", "dataset": dataset,
                "split": "train" if index < 6 else "validation", "source_sha256": "a" * 64,
                "packed_frames": 10, "baseline_packed_sha256": "b" * 64,
                "candidate_packed_sha256": "c" * 64,
                "kinematic_status": "PENDING_FROZEN_POLICY_EVALUATION"})
    return result


def comparison_fixture():
    source_rows = rows()
    groups = {"all": source_rows}
    for field in ("dataset", "split", "kinematic_status"):
        for value in sorted({row[field] for row in source_rows}):
            groups[f"{field}/{value}"] = [r for r in source_rows if r[field] == value]
    modes = []
    for mode in range(8):
        candidate = .06 if mode == 0 else .04
        metrics = {name: {"baseline": .05, "candidate": candidate, "delta": candidate - .05}
                   for name in summary.METRICS}
        passed = mode != 0
        modes.append({"mode_index": mode, "mode": f"mode{mode}", "origins": [
            {"origin_id": row["origin_id"], "metrics": copy.deepcopy(metrics),
             "frozen_tracking_no_regression": passed} for row in source_rows],
            "strata": {name: {"num_origins": len(selected), "mean_metrics": copy.deepcopy(metrics),
                                "frozen_tracking_no_regression": passed} for name, selected in groups.items()}})
    return {"modes": modes, "checkpoint_sha256": summary.paired.OFFICIAL_SHA,
            "training_updates": 0, "automatic_promotion": False,
            "no_regression_tolerance": summary.paired.TOLERANCE, "aggregate_modes_passed": 7,
            "per_origin_all_eight_modes_passed": {r["origin_id"]: False for r in source_rows}}


def geometry_fixture():
    records = []
    for index, row in enumerate(rows()):
        record = {k: row[k] for k in ("origin_id", "dataset", "split", "kinematic_status")}
        for side in summary.SIDES:
            ground_bad = side == "baseline"
            record[side] = {"packed_sha256": row[f"{side}_packed_sha256"],
                "metrics": {"frames": 10, "min_collision_sole_z_m": -.003 if ground_bad else .002,
                    "foot_penetration_frames_gt_1mm": 2 if ground_bad else 0,
                    "self_penetration_frames_gt_1mm": 1 if ground_bad else 0,
                    "max_self_penetration_m": .002 if ground_bad else 0.,
                    "max_all_body_ground_penetration_m": .003 if ground_bad else 0.,
                    "joint_limit_violations_gt_1e_minus6_rad": 0},
                "review_reasons": ["packed_sole_penetration_gt_1mm"] if ground_bad else [],
                "kinematic_pass": not ground_bad,
                "motion_extent": {"root_total_path_length_xyz_m": 2., "root_total_path_length_xy_m": 1.8,
                    "root_horizontal_span_xy_m": 1.5, "root_height_m": {"min": .5, "max": .7, "range": .2},
                    "joint_range_rad": {"mean": .3, "p95": .7, "max": .8}}}
        records.append(record)
    return {"result": "COMPLETE", "physics_stepped": False, "training_updates": 0,
            "automatic_promotion": False, "results": records}


class SummaryHelpersTests(unittest.TestCase):
    def test_fixed_40_source_split_contract(self):
        self.assertEqual(len(summary.validate_origins(rows())), 40)
        for bad in (rows()[:-1], [rows()[0]] * 40):
            with self.assertRaises(ValueError):
                summary.validate_origins(bad)
        bad = rows()
        bad[0]["split"] = "validation"
        with self.assertRaisesRegex(ValueError, "6 train"):
            summary.validate_origins(bad)
        bad = rows()
        bad[0]["origin_id"] = "ACCAD/../outside"
        with self.assertRaisesRegex(ValueError, "relative origin"):
            summary.validate_origins(bad)

    def test_status_requires_complete_all40_and_both_stages(self):
        state = {"result": "COMPLETE", "motions_requested": 40, "motions_completed": 40,
                 "training_started": False, "automatic_promotion": False, "inputs_verified_unchanged": True,
                 "manifest_sha256": "d" * 64, "motions": [
                     {"origin_id": row["origin_id"], "result": "COMPLETE", "stage": "complete",
                      "jobs": [{"stage": stage, "result": "COMPLETE"} for stage in ("prepare", "retarget")]} for row in rows()]}
        summary.validate_batch_status(state, rows(), "d" * 64)
        for change in ({"result": "RUNNING"}, {"motions_completed": 39}, {"training_started": True},
                       {"manifest_sha256": "e" * 64}, {"motions": state["motions"][:-1]}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                summary.validate_batch_status({**state, **change}, rows(), "d" * 64)
        bad = copy.deepcopy(state)
        bad["motions"][0]["jobs"][1]["result"] = "RUNNING"
        with self.assertRaisesRegex(ValueError, "did not complete"):
            summary.validate_batch_status(bad, rows(), "d" * 64)

    def test_tracking_units_and_all_eight_gate_counts(self):
        report = summary.tracking_summary(comparison_fixture(), rows())
        self.assertEqual(report["aggregate_modes_passed"], 7)
        self.assertEqual(report["modes"][0]["strata"]["all"]["active_position_cm"]["baseline"], 5.)
        self.assertEqual(report["modes"][0]["strata"]["all"]["active_position_cm"]["candidate"], 6.)
        self.assertEqual(report["per_stratum"]["dataset/KIT"]["origin_mode_checks_passed"], 8 * 7)
        self.assertEqual(report["per_stratum"]["split/validation"]["origins"], 10)
        self.assertEqual(report["per_stratum"]["split/validation"]["origin_mode_checks_total"], 80)
        self.assertEqual(report["per_stratum"]["dataset/KIT/split/validation"]["origins"], 2)
        self.assertEqual(report["per_stratum"]["dataset_group/legacy/split/validation"]["origins"], 8)
        self.assertEqual(report["per_stratum"]["all"]["origins_passing_all_eight_modes"], 0)

    def test_tracking_rejects_missing_mask_or_origin(self):
        for target in ("mask", "origin"):
            report = comparison_fixture()
            if target == "mask":
                report["modes"].pop()
            else:
                report["modes"][0]["origins"].pop()
            with self.assertRaises(ValueError):
                summary.tracking_summary(report, rows())

    def test_tracking_recomputes_deltas_means_and_gate_labels(self):
        for corruption in ("delta", "mean", "clip_pass", "aggregate_pass", "top_pass", "stratum"):
            report = comparison_fixture()
            mode = report["modes"][0]
            if corruption == "delta":
                mode["origins"][0]["metrics"][summary.POSITION]["delta"] = 0.
            elif corruption == "mean":
                mode["strata"]["all"]["mean_metrics"][summary.POSITION]["candidate"] = .01
            elif corruption == "clip_pass":
                mode["origins"][0]["frozen_tracking_no_regression"] = True
            elif corruption == "aggregate_pass":
                mode["strata"]["all"]["frozen_tracking_no_regression"] = True
            elif corruption == "top_pass":
                report["aggregate_modes_passed"] = 8
            else:
                mode["strata"].pop("split/validation")
            with self.subTest(corruption=corruption), self.assertRaises(ValueError):
                summary.tracking_summary(report, rows())

    def test_geometry_uses_sums_for_counts_and_maxima_for_worst_depths(self):
        report = summary.geometry_summary(geometry_fixture(), rows())
        baseline, candidate = report["sides"]["baseline"], report["sides"]["candidate"]
        self.assertEqual(baseline["frames"], 400)
        self.assertEqual(baseline["self_penetration_frames_gt_1mm"], 40)
        self.assertEqual(baseline["foot_penetration_frames_gt_1mm"], 80)
        self.assertEqual(baseline["max_self_penetration_mm"], 2.)
        self.assertEqual(baseline["max_all_body_ground_penetration_mm"], 3.)
        self.assertEqual(baseline["joint_frame_observations"], 400 * 29)
        self.assertEqual(candidate["kinematic_pass_origins"], 40)
        self.assertEqual(candidate["motion_extent_origin_macro_mean"]["root_height_range_m"], .2)

    def test_geometry_extent_is_optional_but_cannot_be_partial(self):
        report = geometry_fixture()
        report["results"][0]["candidate"].pop("motion_extent")
        with self.assertRaisesRegex(ValueError, "partial"):
            summary.geometry_summary(report, rows())
        for record in report["results"]:
            record["candidate"].pop("motion_extent", None)
        result = summary.geometry_summary(report, rows())
        self.assertIsNone(result["sides"]["candidate"]["motion_extent_origin_macro_mean"])

    def test_geometry_rejects_wrong_counts_hashes_and_false_pass(self):
        for field, value in (("frames", 11), ("self_penetration_frames_gt_1mm", 11),
                             ("joint_limit_violations_gt_1e_minus6_rad", 291),
                             ("max_self_penetration_m", float("nan"))):
            report = geometry_fixture()
            report["results"][0]["candidate"]["metrics"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                summary.geometry_summary(report, rows())
        report = geometry_fixture()
        report["results"][0]["candidate"]["packed_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "SHA"):
            summary.geometry_summary(report, rows())
        report = geometry_fixture()
        report["results"][0]["baseline"]["kinematic_pass"] = True
        with self.assertRaisesRegex(ValueError, "label"):
            summary.geometry_summary(report, rows())

    def test_umr_summary_counts_real_hits_not_number_of_files(self):
        receipts = {row["origin_id"]: {"schema": "bfm.umr_smplx_trial/1", "umr_commit": summary.UMR_COMMIT,
            "promoted_to_training": False, "protected_inputs_rechecked": True, "material_surface_transport": True,
            "source": {"source_sha256": row["source_sha256"]}, "solve_failures": index % 2,
            "setup_cache": {"enabled": True, "hit": index >= 10, "rechecked_after_retarget": True,
                            "key_sha256": f"{index % 10:064x}"}} for index, row in enumerate(rows())}
        report = summary.umr_summary(receipts, rows())
        self.assertEqual(report["cache_hits"], 30)
        self.assertEqual(report["cache_misses"], 10)
        self.assertEqual(report["unique_canonical_setups"], 10)
        self.assertEqual(report["solve_failures"], 20)
        receipts.pop(next(iter(receipts)))
        with self.assertRaisesRegex(ValueError, "40"):
            summary.umr_summary(receipts, rows())

    def test_unique_file_hash_cache_checks_conflicting_digest_and_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.npz"
            path.write_bytes(b"numeric")
            files = summary.VerifiedFiles()
            first = files.verify(path)
            with mock.patch.object(Path, "open", side_effect=AssertionError("should not reread shared model")):
                self.assertEqual(files.verify(path, first), first)
            with self.assertRaisesRegex(ValueError, "SHA256"):
                files.verify(path, "f" * 64)
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                files.recheck()

    def test_symlink_and_nonfinite_numbers_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "file"
            path.write_bytes(b"data")
            link = Path(directory) / "alias"
            link.symlink_to(path)
            with self.assertRaisesRegex(ValueError, "symlink"):
                summary.VerifiedFiles().verify(link)
        for value in (float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                summary.number(value, "test")

    def test_cli_only_prints_json_and_has_no_output_or_execute_option(self):
        result = {"result": "COMPLETE_DIAGNOSTIC_ONLY", "automatic_promotion": False}
        stream = io.StringIO()
        with mock.patch.object(summary, "summarize", return_value=result) as summarize:
            with contextlib.redirect_stdout(stream):
                self.assertEqual(summary.main(["--batch", "local/pilot", "--evaluation", "logs/eval"]), 0)
        self.assertEqual(json.loads(stream.getvalue()), result)
        summarize.assert_called_once_with(Path("local/pilot"), Path("logs/eval"))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            summary.main(["--batch", "local/pilot", "--evaluation", "logs/eval", "--execute"])


if __name__ == "__main__":
    unittest.main()
