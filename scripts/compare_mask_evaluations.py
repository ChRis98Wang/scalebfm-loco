#!/usr/bin/env python3
"""Compare paired schema-4 masked-tracking evaluation reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence


MASK_METRIC_KEYS = (
    "error_active_body_pos_g",
    "error_active_body_pos_root_relative",
    "error_active_body_rot",
    "error_active_body_pos_g_max",
    "error_all_body_pos_g_max",
)
OPTIONAL_METRIC_KEY = "error_body_pos_g"
THRESHOLD_METRES = 0.5
NUMERICAL_TOLERANCE = 1e-6
PAIRED_FIELDS = (
    "task",
    "seed",
    "num_envs",
    "max_steps",
    "step_dt",
    "device",
    "torch",
    "python",
    "package_versions",
    "gpu",
    "cuda",
    "scene_variant",
    "target_object",
)
MANIFEST_FIELDS = ("motion_index", "motions", "python_sources")
NULLABLE_PAIRED_FIELDS = frozenset(("gpu", "cuda", "target_object"))
MASKED_PROTOCOL_FIELDS = (
    "measurement_boundary",
    "position",
    "root_relative_position",
    "rotation",
    "tracking_failure",
    "limitations",
)
G1_BFM_TASK = "G1-BFM-Transformer-Tracking"
G1_BFM_MODE_PRESETS = (
    ("Pelvis-1", ("pelvis",)),
    ("UMI-2", ("left_wrist_yaw_link", "right_wrist_yaw_link")),
    ("VR-3", ("pelvis", "left_wrist_yaw_link", "right_wrist_yaw_link")),
    (
        "UMI-4",
        ("left_wrist_yaw_link", "right_wrist_yaw_link", "left_ankle_roll_link", "right_ankle_roll_link"),
    ),
    (
        "VR-5",
        ("pelvis", "left_wrist_yaw_link", "right_wrist_yaw_link", "left_ankle_roll_link", "right_ankle_roll_link"),
    ),
    (
        "UpperBody-6",
        (
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ),
    ),
    (
        "UpperBody-Mobile-7",
        (
            "pelvis",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ),
    ),
    (
        "WholeBody-14",
        (
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ),
    ),
)


class ComparisonError(ValueError):
    """Raised when reports cannot form a controlled paired comparison."""


def _mapping(value: Any, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ComparisonError(f"{description} must be an object")
    return value


def _sha(record: Mapping[str, Any], description: str) -> str:
    value = _mapping(record, description).get("sha256")
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise ComparisonError(f"{description}.sha256 must contain 64 hexadecimal characters")
    return value


def _finite_number(value: Any, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ComparisonError(f"{description} must be a finite number")
    return float(value)


def _validate_metric_values(metrics: Mapping[str, Any], description: str) -> None:
    for metric_name, values in metrics.items():
        values = _mapping(values, f"{description} metric {metric_name}")
        for statistic in ("mean", "max"):
            if statistic not in values:
                raise ComparisonError(f"{description} metric {metric_name} is missing {statistic}")
            _finite_number(values[statistic], f"{description} metric {metric_name}.{statistic}")
    for metric_name in MASK_METRIC_KEYS:
        values = metrics[metric_name]
        mean = float(values["mean"])
        maximum = float(values["max"])
        if mean < 0 or maximum < 0:
            raise ComparisonError(f"{description} metric {metric_name} must be nonnegative")
        if maximum + NUMERICAL_TOLERANCE < mean:
            raise ComparisonError(f"{description} metric {metric_name} has max below mean")


def _normalize_motions(
    report: Mapping[str, Any], description: str, max_steps: int
) -> dict[int, Mapping[str, Any]]:
    motions = report.get("motions")
    if not isinstance(motions, list) or not motions:
        raise ComparisonError(f"{description} must contain a nonempty motions list")
    by_id: dict[int, Mapping[str, Any]] = {}
    names: set[str] = set()
    expected_metric_names: set[str] | None = None
    for offset, raw_row in enumerate(motions):
        row = _mapping(raw_row, f"{description} motion row {offset}")
        motion_id = row.get("motion_id")
        motion_name = row.get("motion")
        if isinstance(motion_id, bool) or not isinstance(motion_id, int) or motion_id < 0:
            raise ComparisonError(f"{description} requires a nonempty clip identity with a nonnegative integer motion_id")
        if not isinstance(motion_name, str) or not motion_name:
            raise ComparisonError(f"{description} requires a nonempty clip identity")
        if motion_id in by_id:
            raise ComparisonError(f"{description} has duplicate motion_id {motion_id}")
        if motion_name in names:
            raise ComparisonError(f"{description} has duplicate motion name {motion_name!r}")
        by_id[motion_id] = row
        names.add(motion_name)

        source_frames = row.get("source_frames")
        evaluated_steps = row.get("evaluated_steps")
        if isinstance(source_frames, bool) or not isinstance(source_frames, int) or source_frames < 2:
            raise ComparisonError(f"{description} motion_id {motion_id} source_frames must be at least 2")
        if isinstance(evaluated_steps, bool) or not isinstance(evaluated_steps, int) or evaluated_steps < 1:
            raise ComparisonError(f"{description} motion_id {motion_id} evaluated_steps must be positive")
        expected_steps = min(source_frames - 1, max_steps)
        if evaluated_steps != expected_steps:
            raise ComparisonError(f"{description} motion_id {motion_id} has an invalid evaluation horizon")
        if not isinstance(row.get("truncated"), bool):
            raise ComparisonError(f"{description} motion_id {motion_id} has invalid truncated flag")
        if row["truncated"] != (expected_steps < source_frames - 1):
            raise ComparisonError(f"{description} motion_id {motion_id} has an inconsistent truncated flag")

        metrics = _mapping(row.get("metrics"), f"{description} motion_id {motion_id} metrics")
        metric_names = set(metrics)
        missing = set(MASK_METRIC_KEYS) - metric_names
        if missing:
            raise ComparisonError(
                f"{description} motion_id {motion_id} is missing required metric(s): {', '.join(sorted(missing))}"
            )
        if expected_metric_names is None:
            expected_metric_names = metric_names
        elif metric_names != expected_metric_names:
            raise ComparisonError(f"{description} has inconsistent metric sets between motions")
        _validate_metric_values(metrics, f"{description} motion_id {motion_id}")
    return by_id


def _normalize_side(reports: Iterable[Mapping[str, Any]], side: str) -> dict[int, dict[str, Any]]:
    reports = list(reports)
    by_mode: dict[int, dict[str, Any]] = {}
    for offset, raw_report in enumerate(reports):
        report = dict(_mapping(raw_report, f"{side} report {offset}"))
        mode_index = report.get("mode_index")
        if isinstance(mode_index, bool) or not isinstance(mode_index, int):
            raise ComparisonError(f"{side} report {offset} has invalid mode_index")
        if mode_index in by_mode:
            raise ComparisonError(f"{side} reports contain duplicate mode_index {mode_index}")
        by_mode[mode_index] = report
    if set(by_mode) != set(range(8)):
        raise ComparisonError(f"{side} reports must contain exactly mode_index 0..7")

    first = by_mode[0]
    if first.get("schema_version") != 4:
        raise ComparisonError(f"{side} mode_index 0 must use schema_version 4")
    manifest = _mapping(first.get("input_manifest"), f"{side} mode_index 0 input_manifest")
    checkpoint_sha = _sha(manifest.get("checkpoint"), f"{side} mode_index 0 checkpoint manifest")
    if first.get("checkpoint_sha256") != checkpoint_sha:
        raise ComparisonError(f"{side} mode_index 0 checkpoint SHA fields differ")
    manifest_hashes = {
        name: _sha(manifest.get(name), f"{side} mode_index 0 {name} manifest")
        for name in MANIFEST_FIELDS
    }
    first_motions: dict[int, Mapping[str, Any]] | None = None
    first_metric_names: set[str] | None = None

    for mode_index, report in sorted(by_mode.items()):
        description = f"{side} mode_index {mode_index}"
        if report.get("schema_version") != 4:
            raise ComparisonError(f"{description} must use schema_version 4")
        for field in PAIRED_FIELDS:
            if field not in report:
                raise ComparisonError(f"{description} is missing required field {field}")
            if report[field] is None and field not in NULLABLE_PAIRED_FIELDS:
                raise ComparisonError(f"{description} {field} must not be null")
        if report["task"] != G1_BFM_TASK:
            raise ComparisonError(f"{description} task must be {G1_BFM_TASK}")
        seed = report["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ComparisonError(f"{description} seed must be an integer")
        for field in ("num_envs", "max_steps"):
            value = report[field]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ComparisonError(f"{description} {field} must be a positive integer")
        if _finite_number(report["step_dt"], f"{description} step_dt") <= 0:
            raise ComparisonError(f"{description} step_dt must be positive")
        if report.get("input_manifest_verified_unchanged") is not True:
            raise ComparisonError(f"{description} must be verified unchanged")
        protocol = _mapping(report.get("protocol"), f"{description} protocol")
        updates = protocol.get("training_updates")
        if isinstance(updates, bool) or updates != 0:
            raise ComparisonError(f"{description} must report zero training updates")
        masked_protocol = _mapping(protocol.get("masked_tracking"), f"{description} protocol.masked_tracking")
        if any(field not in masked_protocol for field in MASKED_PROTOCOL_FIELDS):
            raise ComparisonError(f"{description} protocol.masked_tracking is incomplete")
        if protocol != first.get("protocol"):
            raise ComparisonError(f"{side} protocol differs across modes")
        for field in PAIRED_FIELDS:
            if report.get(field) != first.get(field):
                raise ComparisonError(f"{side} {field} differs across modes")

        mode_name = report.get("mode")
        active_names = report.get("active_body_names")
        expected_mode, expected_names = G1_BFM_MODE_PRESETS[mode_index]
        if mode_name != expected_mode or active_names != list(expected_names):
            raise ComparisonError(f"{description} does not match the canonical G1 BFM preset")

        current_manifest = _mapping(report.get("input_manifest"), f"{description} input_manifest")
        current_checkpoint = _sha(current_manifest.get("checkpoint"), f"{description} checkpoint manifest")
        if report.get("checkpoint_sha256") != current_checkpoint:
            raise ComparisonError(f"{description} checkpoint SHA fields differ")
        if current_checkpoint != checkpoint_sha:
            raise ComparisonError(f"{side} reports mix checkpoint content across modes")
        for name, expected_hash in manifest_hashes.items():
            if _sha(current_manifest.get(name), f"{description} {name} manifest") != expected_hash:
                raise ComparisonError(f"{side} input_manifest {name} content differs across modes")

        motions = _normalize_motions(report, description, report["max_steps"])
        metric_names = set(next(iter(motions.values()))["metrics"])
        if first_motions is None:
            first_motions = motions
            first_metric_names = metric_names
        else:
            if set(motions) != set(first_motions):
                raise ComparisonError(f"{side} clip identities differ across modes")
            for motion_id, row in motions.items():
                baseline = first_motions[motion_id]
                identity = (row["motion"], row["source_frames"], row["evaluated_steps"], row["truncated"])
                expected = (
                    baseline["motion"], baseline["source_frames"], baseline["evaluated_steps"], baseline["truncated"]
                )
                if identity != expected:
                    raise ComparisonError(f"{side} clip metadata differs across modes for motion_id {motion_id}")
            if metric_names != first_metric_names:
                raise ComparisonError(f"{side} metric sets differ across modes")
        report["_motions_by_id"] = motions
    return by_mode


def _validate_pair(reference: dict[int, dict[str, Any]], candidate: dict[int, dict[str, Any]]) -> None:
    for mode_index in range(8):
        ref = reference[mode_index]
        cand = candidate[mode_index]
        for field in PAIRED_FIELDS + ("protocol",):
            if ref.get(field) != cand.get(field):
                raise ComparisonError(f"{field} differs between reference and candidate")
        if ref["mode"] != cand["mode"]:
            raise ComparisonError(f"mode differs for mode_index {mode_index}")
        if ref["active_body_names"] != cand["active_body_names"]:
            raise ComparisonError(f"active_body_names differs for mode_index {mode_index}")

        ref_manifest = ref["input_manifest"]
        cand_manifest = cand["input_manifest"]
        for name in MANIFEST_FIELDS:
            if _sha(ref_manifest[name], f"reference {name} manifest") != _sha(
                cand_manifest[name], f"candidate {name} manifest"
            ):
                raise ComparisonError(f"input_manifest {name} content differs between reference and candidate")

        ref_motions = ref["_motions_by_id"]
        cand_motions = cand["_motions_by_id"]
        if set(ref_motions) != set(cand_motions):
            raise ComparisonError(f"clip identities differ for mode_index {mode_index}")
        for motion_id, ref_row in ref_motions.items():
            cand_row = cand_motions[motion_id]
            ref_meta = (
                ref_row["motion"], ref_row["source_frames"], ref_row["evaluated_steps"], ref_row["truncated"]
            )
            cand_meta = (
                cand_row["motion"], cand_row["source_frames"], cand_row["evaluated_steps"], cand_row["truncated"]
            )
            if ref_meta != cand_meta:
                raise ComparisonError(f"clip metadata differs for mode_index {mode_index}, motion_id {motion_id}")
            if set(ref_row["metrics"]) != set(cand_row["metrics"]):
                raise ComparisonError(f"metric sets differ for mode_index {mode_index}, motion_id {motion_id}")


def _mode_values(report: Mapping[str, Any]) -> dict[str, Any]:
    rows = list(report["_motions_by_id"].values())
    metric_names = list(MASK_METRIC_KEYS)
    if OPTIONAL_METRIC_KEY in rows[0]["metrics"]:
        metric_names.append(OPTIONAL_METRIC_KEY)
    means = {
        name: fmean(_finite_number(row["metrics"][name]["mean"], name) for row in rows)
        for name in metric_names
    }
    active_passes = sum(
        row["metrics"]["error_active_body_pos_g_max"]["max"] <= THRESHOLD_METRES for row in rows
    )
    all_passes = sum(
        row["metrics"]["error_all_body_pos_g_max"]["max"] <= THRESHOLD_METRES for row in rows
    )
    return {
        "mean_metrics": means,
        "active_max_link_tracking_pass_rate": active_passes / len(rows),
        "all_configured_14_max_link_tracking_pass_rate": all_passes / len(rows),
    }


def compare_reports(
    reference_reports: Iterable[Mapping[str, Any]], candidate_reports: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    """Validate and compare two complete eight-mode report sets."""
    reference = _normalize_side(reference_reports, "reference")
    candidate = _normalize_side(candidate_reports, "candidate")
    _validate_pair(reference, candidate)

    modes = []
    for mode_index in range(8):
        ref_values = _mode_values(reference[mode_index])
        cand_values = _mode_values(candidate[mode_index])
        mean_deltas = {
            name: cand_values["mean_metrics"][name] - value
            for name, value in ref_values["mean_metrics"].items()
        }
        active_delta = (
            cand_values["active_max_link_tracking_pass_rate"]
            - ref_values["active_max_link_tracking_pass_rate"]
        )
        all_delta = (
            cand_values["all_configured_14_max_link_tracking_pass_rate"]
            - ref_values["all_configured_14_max_link_tracking_pass_rate"]
        )
        no_regression = (
            cand_values["mean_metrics"]["error_active_body_pos_g"]
            <= ref_values["mean_metrics"]["error_active_body_pos_g"] + NUMERICAL_TOLERANCE
            and cand_values["mean_metrics"]["error_active_body_rot"]
            <= ref_values["mean_metrics"]["error_active_body_rot"] + NUMERICAL_TOLERANCE
            and active_delta >= 0
            and all_delta >= 0
        )
        modes.append(
            {
                "mode_index": mode_index,
                "mode": reference[mode_index]["mode"],
                "active_body_names": reference[mode_index]["active_body_names"],
                "num_motions": len(reference[mode_index]["_motions_by_id"]),
                "reference": ref_values,
                "candidate": cand_values,
                "delta_candidate_minus_reference": {
                    "mean_metrics": mean_deltas,
                    "active_max_link_tracking_pass_rate": active_delta,
                    "all_configured_14_max_link_tracking_pass_rate": all_delta,
                },
                "candidate_no_regression": no_regression,
            }
        )

    return {
        "schema_version": 1,
        "comparison": "candidate minus reference",
        "tracking_threshold_metres": THRESHOLD_METRES,
        "numerical_tolerance": NUMERICAL_TOLERANCE,
        "interpretation": (
            "Paired report comparison only; not a formal statistical significance test, "
            "a convergence claim, or a full-BFM validation."
        ),
        "reference_checkpoint_sha256": reference[0]["checkpoint_sha256"],
        "candidate_checkpoint_sha256": candidate[0]["checkpoint_sha256"],
        "comparator_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "overall_candidate_no_regression": all(mode["candidate_no_regression"] for mode in modes),
        "modes": modes,
    }


def _load_reports(paths: Sequence[str], side: str) -> list[Mapping[str, Any]]:
    reports = []
    for path_text in paths:
        path = Path(path_text)
        try:
            with path.open(encoding="utf-8") as stream:
                reports.append(json.load(stream))
        except (OSError, json.JSONDecodeError) as exc:
            raise ComparisonError(f"cannot read {side} report {path}: {exc}") from exc
    return reports


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", nargs="+", required=True, metavar="REPORT")
    parser.add_argument("--candidate", nargs="+", required=True, metavar="REPORT")
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise ComparisonError(f"output already exists: {args.output}")
        result = compare_reports(
            _load_reports(args.reference, "reference"),
            _load_reports(args.candidate, "candidate"),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    except (ComparisonError, OSError) as exc:
        parser.error(str(exc))
    passed = sum(mode["candidate_no_regression"] for mode in result["modes"])
    print(
        f"[BFM COMPARE] {passed}/8 modes pass no-regression; "
        f"overall={str(result['overall_candidate_no_regression']).lower()}; output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
