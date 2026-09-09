#!/usr/bin/env python3
"""Read-only, evidence-bound summary of a complete 40-origin UMR pilot.

Only JSON is written to stdout. No simulator, training, data replacement, output
files or automatic promotion are started. Hash each unique input only once,
including shared body models and checkpoints; stat-check them again at the end.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import fmean
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import evaluate_retarget_pilot as paired
from scripts.umr_smplx_source import UMR_COMMIT

SOURCES = ("ACCAD", "BMLmovi", "BMLrub", "CNRS", "KIT")
SIDES = ("baseline", "candidate")
METRICS = paired.METRICS
POSITION = "error_active_body_pos_g"


def number(value, label, *, minimum=0.):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Expected finite numeric {label}")
    if minimum is not None and value < minimum:
        raise ValueError(f"Negative/out-of-range {label}")
    return value


def integer(value, label, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"Expected integer {label} >= {minimum}")
    return value


class VerifiedFiles:
    def __init__(self):
        self.files = {}

    @staticmethod
    def signature(path):
        st = path.stat()
        return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)

    def verify(self, path, expected=None):
        path = Path(path)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Expected a regular file, not a symlink: {path}")
        path = path.resolve(strict=True)
        before = self.signature(path)
        if path in self.files:
            digest, original = self.files[path]
            if before != original:
                raise ValueError(f"File changed while summarizing: {path}")
        else:
            h = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1 << 20), b""):
                    h.update(block)
            digest = h.hexdigest()
            if self.signature(path) != before:
                raise ValueError(f"File changed during hash read: {path}")
            self.files[path] = (digest, before)
        if expected is not None and (not isinstance(expected, str)
                or not re.fullmatch(r"[0-9a-f]{64}", expected) or expected != digest):
            raise ValueError(f"SHA256 binding mismatch: {path}")
        return digest

    def read_json(self, path, expected=None):
        self.verify(path, expected)
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        self.verify(path, expected)
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object: {path}")
        return value

    def recheck(self):
        for path, (_, original) in self.files.items():
            if path.is_symlink() or self.signature(path) != original:
                raise ValueError(f"Verified input changed before summary completion: {path}")


def validate_origins(rows):
    if not isinstance(rows, list) or len(rows) != 40:
        raise ValueError("Exactly all 40 preselected origins are required")
    by_origin = {}
    for row in rows:
        name = row.get("origin_id")
        if (not isinstance(name, str) or "\\" in name or any(ord(c) < 32 for c in name)
                or any(part in ("", ".", "..") for part in name.split("/"))
                or len(name.split("/")) < 2 or name.split("/")[0] != row.get("dataset")):
            raise ValueError("Invalid relative origin identity")
        if name in by_origin:
            raise ValueError("Duplicate origin identity")
        by_origin[name] = row
    expected = {f"{source}/{split}": count for source in SOURCES
                for split, count in (("train", 6), ("validation", 2))}
    if Counter(f"{r.get('dataset')}/{r.get('split')}" for r in rows) != expected:
        raise ValueError("Require exactly 6 train + 2 development origins per dataset")
    return by_origin


def validate_batch_status(status, rows, manifest_sha):
    origins = validate_origins(rows)
    if (status.get("result"), status.get("motions_requested"), status.get("motions_completed"),
            status.get("training_started"), status.get("automatic_promotion"),
            status.get("inputs_verified_unchanged"), status.get("manifest_sha256")) != (
            "COMPLETE", 40, 40, False, False, True, manifest_sha):
        raise ValueError("Batch must be COMPLETE, immutable, non-promoting and bound to its manifest")
    states = status.get("motions", [])
    if len(states) != 40 or {s.get("origin_id") for s in states} != set(origins):
        raise ValueError("Batch status omitted or duplicated an origin")
    for state in states:
        jobs = state.get("jobs", [])
        if (state.get("result"), state.get("stage")) != ("COMPLETE", "complete") or len(jobs) != 2:
            raise ValueError("Batch contains an incomplete origin")
        if {j.get("stage") for j in jobs} != {"prepare", "retarget"} or any(j.get("result") != "COMPLETE" for j in jobs):
            raise ValueError("Origin preparation or retargeting did not complete")


def tracking_summary(comparison, rows):
    origins = validate_origins(rows)
    modes = comparison.get("modes", [])
    if (len(modes) != 8 or {m.get("mode_index") for m in modes} != set(range(8))
            or comparison.get("training_updates") != 0 or comparison.get("automatic_promotion") is not False
            or comparison.get("checkpoint_sha256") != paired.OFFICIAL_SHA
            or comparison.get("no_regression_tolerance") != paired.TOLERANCE):
        raise ValueError("Require all eight frozen official-policy masks and the unchanged no-regression gate")
    groups = {"all": set(origins)}
    for field in ("dataset", "split", "kinematic_status"):
        if any(field not in row for row in rows):
            if field == "kinematic_status":
                continue
            raise ValueError(f"Missing tracking stratum {field}")
        for value in sorted({r[field] for r in rows}):
            groups[f"{field}/{value}"] = {r["origin_id"] for r in rows if r[field] == value}
    declared_groups = set(groups)
    # The evaluator preserves origin-level measurements, so these intersections
    # can be recomputed without running another evaluation or mixing splits.
    for dataset in SOURCES:
        for split in ("train", "validation"):
            groups[f"dataset/{dataset}/split/{split}"] = {
                r["origin_id"] for r in rows if r["dataset"] == dataset and r["split"] == split}
    groups["dataset_group/legacy"] = {r["origin_id"] for r in rows if r["dataset"] != "KIT"}
    groups["dataset_group/legacy/split/validation"] = {
        r["origin_id"] for r in rows if r["dataset"] != "KIT" and r["split"] == "validation"}
    pass_by_origin = {name: [] for name in origins}
    mode_rows = []
    for mode in sorted(modes, key=lambda row: row["mode_index"]):
        clips = mode.get("origins", [])
        by_name = {r["origin_id"]: r for r in clips}
        if len(clips) != 40 or set(by_name) != set(origins):
            raise ValueError("A mask omitted or duplicated an origin")
        if set(mode.get("strata", {})) != declared_groups:
            raise ValueError("Tracking strata do not cover the original datasets and splits")
        for name, clip in by_name.items():
            values = clip.get("metrics", {})
            if set(values) != set(METRICS):
                raise ValueError("Origin tracking metrics are incomplete")
            for metric, pair in values.items():
                a, b = (number(pair[side], f"{name}/{metric}/{side}") for side in SIDES)
                if not math.isclose(number(pair["delta"], "metric delta", minimum=None), b - a, abs_tol=1e-12, rel_tol=0):
                    raise ValueError("Tracking delta disagrees with original measurements")
            passed = all(pair["delta"] <= paired.TOLERANCE for pair in values.values())
            if clip.get("frozen_tracking_no_regression") is not passed:
                raise ValueError("Per-origin pass label disagrees with metric deltas")
            pass_by_origin[name].append(passed)
        strata = {}
        for label, selected in groups.items():
            reported = mode["strata"].get(label)
            means = {metric: {side: fmean(by_name[name]["metrics"][metric][side] for name in selected)
                              for side in (*SIDES, "delta")} for metric in METRICS}
            gate = all(value["delta"] <= paired.TOLERANCE for value in means.values())
            if reported is not None:
                if reported.get("num_origins") != len(selected) or reported.get("frozen_tracking_no_regression") is not gate:
                    raise ValueError("Aggregate stratum count/pass differs from original clips")
                for metric in METRICS:
                    for side in (*SIDES, "delta"):
                        if not math.isclose(reported["mean_metrics"][metric][side], means[metric][side], rel_tol=0, abs_tol=1e-12):
                            raise ValueError("Aggregate tracking values differ from original clips")
            strata[label] = {"origins": len(selected), "active_position_cm": {
                side: means[POSITION][side] * 100 for side in (*SIDES, "delta")},
                "active_rotation_rad": means["error_active_body_rot"],
                "frozen_tracking_no_regression": gate,
                "origin_checks_passed": sum(by_name[name]["frozen_tracking_no_regression"] for name in selected)}
        mode_rows.append({"mode_index": mode["mode_index"], "mode": mode["mode"], "strata": strata})
    modes_passed = sum(m["strata"]["all"]["frozen_tracking_no_regression"] for m in mode_rows)
    all_modes = {name: all(passes) for name, passes in pass_by_origin.items()}
    if comparison.get("aggregate_modes_passed") != modes_passed or comparison.get("per_origin_all_eight_modes_passed") != all_modes:
        raise ValueError("Top-level tracking pass counts disagree with the complete eight-mode results")
    return {"aggregation": "unweighted per-origin mean within each mask; no cross-mask average",
            "gate_metrics": list(METRICS), "no_regression_tolerance": paired.TOLERANCE,
            "aggregate_modes_passed": modes_passed, "modes_required": 8,
            "per_stratum": {label: {"origins": len(names), "modes_passed": sum(
                m["strata"][label]["frozen_tracking_no_regression"] for m in mode_rows),
                "origin_mode_checks_passed": sum(sum(pass_by_origin[n]) for n in names),
                "origin_mode_checks_total": len(names) * 8,
                "origins_passing_all_eight_modes": sum(all_modes[n] for n in names)} for label, names in groups.items()},
            "modes": mode_rows}


def geometry_summary(report, rows):
    origins = validate_origins(rows)
    records = report.get("results", [])
    by_origin = {r["origin_id"]: r for r in records}
    if (report.get("result") != "COMPLETE" or report.get("physics_stepped") is not False
            or report.get("training_updates") != 0 or report.get("automatic_promotion") is not False
            or len(records) != 40 or set(by_origin) != set(origins)):
        raise ValueError("Geometry audit must include all 40 origins without physics or training")
    aggregated = {}
    for side in SIDES:
        metrics, extents, passes = [], [], []
        for origin, row in origins.items():
            record = by_origin[origin]
            if record.get("dataset") != row["dataset"] or record.get("split") != row["split"]:
                raise ValueError("Geometry origin dataset/split differs")
            item = record[side]
            if item.get("packed_sha256") != row[f"{side}_packed_sha256"]:
                raise ValueError("Geometry packed payload SHA does not match this side")
            value = item["metrics"]
            frames = integer(value["frames"], "geometry frames", minimum=2)
            if frames != row["packed_frames"]:
                raise ValueError("Geometry frame count differs from evaluated origin")
            for field in ("foot_penetration_frames_gt_1mm", "self_penetration_frames_gt_1mm"):
                if integer(value[field], field) > frames:
                    raise ValueError("Penetration frame count exceeds motion frames")
            if integer(value["joint_limit_violations_gt_1e_minus6_rad"], "limit count") > frames * 29:
                raise ValueError("Joint-limit count exceeds joint-frame observations")
            for field in ("max_self_penetration_m", "max_all_body_ground_penetration_m"):
                number(value[field], field)
            number(value["min_collision_sole_z_m"], "sole minimum", minimum=None)
            reasons = []
            if value["foot_penetration_frames_gt_1mm"]:
                reasons.append("packed_sole_penetration_gt_1mm")
            if value["max_all_body_ground_penetration_m"] > .005:
                reasons.append("packed_body_ground_penetration_gt_5mm")
            if value["max_self_penetration_m"] > .005:
                reasons.append("packed_self_penetration_gt_5mm")
            if value["joint_limit_violations_gt_1e_minus6_rad"]:
                reasons.append("packed_joint_limit_violation")
            if item.get("review_reasons") != reasons or item.get("kinematic_pass") is not (not reasons):
                raise ValueError("Kinematic label disagrees with raw geometry thresholds")
            metrics.append(value)
            passes.append(not reasons)
            if "motion_extent" in item:
                extent = item["motion_extent"]
                extents.append({"root_path_xyz_m": number(extent["root_total_path_length_xyz_m"], "XYZ path"),
                    "root_path_xy_m": number(extent["root_total_path_length_xy_m"], "XY path"),
                    "root_horizontal_span_xy_m": number(extent["root_horizontal_span_xy_m"], "horizontal span"),
                    "root_height_range_m": number(extent["root_height_m"]["range"], "root height range"),
                    "mean_joint_range_rad": number(extent["joint_range_rad"]["mean"], "mean joint range"),
                    "max_joint_range_rad": number(extent["joint_range_rad"]["max"], "max joint range")})
        if extents and len(extents) != 40:
            raise ValueError("Motion-extent coverage is partial; cannot summarize a filtered subset")
        total_frames = sum(m["frames"] for m in metrics)
        aggregated[side] = {
            "origins": 40, "kinematic_pass_origins": sum(passes), "frames": total_frames,
            "self_penetration_frames_gt_1mm": sum(m["self_penetration_frames_gt_1mm"] for m in metrics),
            "foot_penetration_frames_gt_1mm": sum(m["foot_penetration_frames_gt_1mm"] for m in metrics),
            "max_self_penetration_mm": max(m["max_self_penetration_m"] for m in metrics) * 1000,
            "max_all_body_ground_penetration_mm": max(m["max_all_body_ground_penetration_m"] for m in metrics) * 1000,
            "max_collision_sole_penetration_mm": max(0., -min(m["min_collision_sole_z_m"] for m in metrics)) * 1000,
            "joint_limit_violations_gt_1e_minus6_rad": sum(m["joint_limit_violations_gt_1e_minus6_rad"] for m in metrics),
            "joint_frame_observations": total_frames * 29,
            "motion_extent_origin_macro_mean": {key: fmean(e[key] for e in extents) for key in extents[0]} if extents else None}
    return {"sides": aggregated, "limitations": "Kinematic reference metrics, not simulated slip/fall/task success. "
            "Smaller target excursion can improve followability while reducing source fidelity."}


def umr_summary(receipts, rows):
    origins = validate_origins(rows)
    if set(receipts) != set(origins):
        raise ValueError("Require exactly all 40 UMR producer receipts")
    failures, hits, keys = 0, 0, set()
    for name, receipt in receipts.items():
        source, cache = receipt.get("source", {}), receipt.get("setup_cache", {})
        if (receipt.get("schema") != "bfm.umr_smplx_trial/1" or receipt.get("umr_commit") != UMR_COMMIT
                or receipt.get("promoted_to_training") is not False or receipt.get("protected_inputs_rechecked") is not True
                or receipt.get("material_surface_transport") is not True
                or source.get("source_sha256") != origins[name]["source_sha256"]
                or cache.get("enabled") is not True or cache.get("rechecked_after_retarget") is not True
                or type(cache.get("hit")) is not bool or not re.fullmatch(r"[0-9a-f]{64}", str(cache.get("key_sha256", "")))):
            raise ValueError("UMR source/pin/cache/immutable-source receipt contract differs")
        failures += integer(receipt.get("solve_failures"), "UMR solve failures")
        hits += cache["hit"]
        keys.add(cache["key_sha256"])
    return {"motions": 40, "solve_failures": failures, "cache_hits": hits, "cache_misses": 40 - hits,
            "unique_canonical_setups": len(keys), "umr_commit": UMR_COMMIT,
            "implementation": "pinned unofficial UMR with local true SMPL-X surface adapter"}


def summarize(batch, evaluation):
    import numpy as np
    files = VerifiedFiles()
    batch, evaluation = Path(batch).resolve(strict=True), Path(evaluation).resolve(strict=True)
    status = files.read_json(batch / "status.json")
    manifest_path = batch / "paired_manifest.json"
    manifest = files.read_json(manifest_path, status.get("manifest_sha256"))
    manifest_sha = files.verify(manifest_path)
    rows = manifest.get("rows", [])
    validate_batch_status(status, rows, manifest_sha)
    if manifest.get("automatic_promotion") is not False or manifest.get("training_started") is not False:
        raise ValueError("Batch manifest cannot authorize training or promotion")
    for path, digest in manifest.get("protected_files", {}).items():
        files.verify(path, digest)
    eval_status = files.read_json(evaluation / "status.json")
    if (eval_status.get("result"), eval_status.get("jobs_requested"), eval_status.get("training_updates"),
            eval_status.get("automatic_promotion"), eval_status.get("inputs_verified_unchanged")) != (
            "COMPLETE", 16, 0, False, True):
        raise ValueError("Evaluation must complete all 16 groups without training or promotion")
    experiment = manifest.get("experiment_id")
    if eval_status.get("experiment_id") != experiment:
        raise ValueError("Batch and evaluation experiment identities differ")
    comparison_path = evaluation / "comparison.json"
    comparison = files.read_json(comparison_path, eval_status.get("comparison_sha256"))
    geometry_path = evaluation / "geometry_audit.json"
    geometry = files.read_json(geometry_path, comparison.get("geometry_audit_sha256"))
    if (comparison.get("experiment_id") != experiment or geometry.get("experiment_id") != experiment
            or geometry.get("geometry_tree_sha256") != eval_status.get("geometry_tree_sha256")
            or comparison.get("geometry_rejects_excluded") != 0):
        raise ValueError("Comparison/geometry identities differ or reject origins were removed")
    inputs = files.read_json(evaluation / "inputs.json")
    pilot = inputs["pilot"]
    eval_rows = pilot["rows"]
    validated = validate_origins(eval_rows)
    if Path(pilot["manifest"]).resolve(strict=True) != manifest_path or pilot.get("experiment_id") != experiment:
        raise ValueError("Evaluation input manifest is not this complete batch")
    for row in rows:
        if any(validated[row["origin_id"]].get(key) != value for key, value in row.items()):
            raise ValueError("Evaluation changed an original pair's identity/receipt")
    if not pilot.get("frozen_files") or not manifest.get("protected_files"):
        raise ValueError("Missing frozen source provenance")
    for path, digest in pilot["frozen_files"].items():
        files.verify(path, digest)
    reports = {side: [] for side in SIDES}
    jobs = eval_status.get("jobs", [])
    if (len(jobs) != 16 or {(j.get("side"), j.get("mode")) for j in jobs}
            != {(side, mode) for side in SIDES for mode in range(8)}):
        raise ValueError("Expected every side/mask exactly once, not an incomplete set of jobs")
    for job in jobs:
        side, mode = job["side"], job["mode"]
        if job.get("name") != f"eval_{side}_{mode}" or job.get("result") != "PASS":
            raise ValueError("Evaluation job did not complete normally")
        report = files.read_json(evaluation / f"{job['name']}.json", job["report_sha256"])
        paired.validate_report(report, eval_rows, side=side, mode=mode,
                               expected_manifest=inputs["evaluation"][side])
        reports[side].append(report)
    recomputed = paired.compare_reports(reports["baseline"], reports["candidate"], eval_rows)
    recomputed.update(experiment_id=experiment, geometry_audit_sha256=files.verify(geometry_path), geometry_rejects_excluded=0)
    if comparison != recomputed or eval_status.get("aggregate_modes_passed") != recomputed["aggregate_modes_passed"]:
        raise ValueError("Comparison does not reproduce from all 16 bound original reports")
    for side, snapshot in inputs["evaluation"].items():
        if side not in SIDES:
            raise ValueError("Unexpected evaluation side")
        for name in ("checkpoint", "motion_index"):
            files.verify(snapshot[name]["path"], snapshot[name]["sha256"])
        for group in ("motions", "python_sources"):
            for record in snapshot[group]["files"].values():
                files.verify(record["path"], record["sha256"])
    receipts = {}
    for row in rows:
        origin = row["origin_id"]
        files.verify(row["source"], row["source_sha256"])
        pair = files.read_json(row["pair_receipt"], row["pair_receipt_sha256"])
        if any(pair.get(key) != row[key] for key in ("origin_id", "dataset", "split", "source_sha256", "expected_packed_frames")):
            raise ValueError("Pair receipt changed origin identity")
        files.verify(row["sampling_proof"], row["sampling_proof_sha256"])
        for side in SIDES:
            files.verify(row[f"{side}_pkl"], row[f"{side}_pkl_sha256"])
            produced = pair["outputs"][side]
            if produced["sha256"] != row[f"{side}_pkl_sha256"]:
                raise ValueError("Pair output does not bind its published native payload")
            files.verify(produced["path"], produced["sha256"])
        for path, digest in pair["input_sha256"].items():
            files.verify(path, digest)
        directory = batch / "retargeted" / origin
        receipt = files.read_json(directory / "receipt.json")
        motion = directory / "motion.npz"
        expected = pair["input_sha256"].get(str(motion.resolve(strict=True)))
        if expected is None:
            raise ValueError("Pair does not bind its exact UMR-produced motion")
        files.verify(motion, expected)
        with np.load(motion, allow_pickle=False) as archive:
            embedded = json.loads(str(archive["metadata_json"]))
        if receipt != embedded:
            raise ValueError("UMR receipt differs from metadata in the hash-bound motion")
        prepared = batch / "surfaces" / f"{origin}.npz"
        files.verify(prepared, receipt["prepared_source_sha256"])
        if pair["input_sha256"].get(str(prepared.resolve(strict=True))) != receipt["prepared_source_sha256"]:
            raise ValueError("Pair does not bind the same prepared source as UMR")
        source = receipt["source"]
        if Path(source["source_file"]).resolve(strict=True) != Path(row["source"]).resolve(strict=True):
            raise ValueError("UMR source path differs from paired origin")
        files.verify(source["body_model"], source["body_model_sha256"])
        output = receipt["scalebfm_output"]
        if Path(output["path"]).resolve(strict=True) != directory / "scalebfm_motion.pkl":
            raise ValueError("UMR formal PKL output redirected outside its origin")
        files.verify(output["path"], output["sha256"])
        cache = receipt["setup_cache"]
        entry = Path(cache["entry"])
        if entry.resolve(strict=True) != batch / "setup_cache" / cache["key_sha256"]:
            raise ValueError("UMR canonical setup is not inside this batch")
        cache_manifest = files.read_json(entry / "manifest.json", cache["manifest_sha256"])
        if cache_manifest.get("key_sha256") != cache["key_sha256"] or cache_manifest.get("artifacts") != cache["artifacts"]:
            raise ValueError("UMR canonical cache manifest differs from its producer receipt")
        if set(cache["artifacts"]) != {"bodies.npz", "correspondence.npz"}:
            raise ValueError("Canonical cache lacks required artifacts")
        for name, record in cache["artifacts"].items():
            files.verify(entry / name, record["sha256"])
            if (entry / name).stat().st_size != record["bytes"]:
                raise ValueError("Canonical cache artifact size differs")
        receipts[origin] = receipt
    result = {"schema": "bfm.umr_paired_summary/1", "result": "COMPLETE_DIAGNOSTIC_ONLY",
              "experiment_id": experiment, "batch": str(batch), "evaluation": str(evaluation),
              "training_started": False, "training_updates": 0, "automatic_promotion": False,
              "operationally_complete": True, "data_replacement_authorized_by_report": False,
              "paired_manifest_sha256": manifest_sha, "comparison_sha256": files.verify(comparison_path),
              "geometry_audit_sha256": files.verify(geometry_path),
              "tracking": tracking_summary(comparison, eval_rows), "geometry": geometry_summary(geometry, eval_rows),
              "umr_production": umr_summary(receipts, rows),
              "limitations": ["Frozen-policy followability is not a training-improvement result.",
                              "Kinematic contact metrics are not physical slip/fall/task success.",
                              "Development validation is not an untouched final holdout.",
                              "Lower reference amplitude may improve tracking while reducing source fidelity.",
                              "No all-data replacement, training or full ScaleBFM completion is inferred."]}
    files.recheck()
    result["unique_files_verified"] = len(files.files)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(summarize(args.batch, args.evaluation), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
