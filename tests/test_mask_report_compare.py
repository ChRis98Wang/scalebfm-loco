import copy
import hashlib
import json
import math
from pathlib import Path

import pytest

from scripts.compare_mask_evaluations import ComparisonError, compare_reports, main
import scripts.compare_mask_evaluations as compare_module


REQUIRED_METRICS = {
    "error_active_body_pos_g": ((0.2, 0.4), (0.1, 0.3)),
    "error_active_body_pos_root_relative": ((0.1, 0.3), (0.05, 0.25)),
    "error_active_body_rot": ((0.5, 0.7), (0.4, 0.6)),
    "error_active_body_pos_g_max": ((0.1, 0.1), (0.1, 0.1)),
    "error_all_body_pos_g_max": ((0.1, 0.1), (0.1, 0.1)),
    "error_body_pos_g": ((0.3, 0.5), (0.2, 0.4)),
}

MODE_PRESETS = (
    ("Pelvis-1", ["pelvis"]),
    ("UMI-2", ["left_wrist_yaw_link", "right_wrist_yaw_link"]),
    ("VR-3", ["pelvis", "left_wrist_yaw_link", "right_wrist_yaw_link"]),
    (
        "UMI-4",
        ["left_wrist_yaw_link", "right_wrist_yaw_link", "left_ankle_roll_link", "right_ankle_roll_link"],
    ),
    (
        "VR-5",
        ["pelvis", "left_wrist_yaw_link", "right_wrist_yaw_link", "left_ankle_roll_link", "right_ankle_roll_link"],
    ),
    (
        "UpperBody-6",
        [
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ],
    ),
    (
        "UpperBody-Mobile-7",
        [
            "pelvis",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ],
    ),
    (
        "WholeBody-14",
        [
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
        ],
    ),
)


def _motion(motion_id, side):
    index = 0 if side == "reference" else 1
    metrics = {}
    for name, (reference_means, candidate_means) in REQUIRED_METRICS.items():
        values = (reference_means, candidate_means)[index]
        metrics[name] = {"mean": values[motion_id], "max": values[motion_id]}
    if side == "reference":
        metrics["error_active_body_pos_g_max"]["max"] = (0.4, 0.6)[motion_id]
        metrics["error_all_body_pos_g_max"]["max"] = (0.5, 0.7)[motion_id]
    else:
        metrics["error_active_body_pos_g_max"]["max"] = (0.4, 0.5)[motion_id]
        metrics["error_all_body_pos_g_max"]["max"] = (0.49, 0.51)[motion_id]
    return {
        "motion_id": motion_id,
        "motion": f"dataset/clip-{motion_id}",
        "source_frames": 11 + motion_id,
        "evaluated_steps": 10 + motion_id,
        "truncated": False,
        "metrics": metrics,
    }


def _report(mode_index, side):
    checkpoint_sha = "a" * 64 if side == "reference" else "b" * 64
    return {
        "schema_version": 4,
        "checkpoint_sha256": checkpoint_sha,
        "task": "G1-BFM-Transformer-Tracking",
        "mode": MODE_PRESETS[mode_index][0],
        "mode_index": mode_index,
        "scene_variant": "baseline",
        "target_object": None,
        "seed": 42,
        "num_envs": 2,
        "max_steps": 1000,
        "step_dt": 0.02,
        "device": "cuda:0",
        "python": "3.12.3",
        "torch": "2.11.0+cu128",
        "package_versions": {"isaaclab": "6.1.14", "numpy": "2.5.2"},
        "cuda": "12.8",
        "gpu": "Synthetic GPU",
        "input_manifest_verified_unchanged": True,
        "input_manifest": {
            "checkpoint": {"sha256": checkpoint_sha},
            "motion_index": {"sha256": "c" * 64},
            "motions": {"sha256": "d" * 64},
            "python_sources": {"sha256": "e" * 64},
        },
        "protocol": {
            "training_updates": 0,
            "action_policy": "deterministic mean, selected reference-body mask",
            "masked_tracking": {
                "measurement_boundary": "post physics, pre reference advance",
                "position": "mean Euclidean distance over active links, metres",
                "root_relative_position": "translation removed only",
                "rotation": "shortest quaternion geodesic, radians",
                "tracking_failure": "any link over 0.5 m",
                "limitations": "reference deviation is not fall detection",
            },
        },
        "active_body_names": MODE_PRESETS[mode_index][1],
        # These intentionally wrong summaries prove comparison uses motion rows.
        "summary": {"mean_metrics": {"error_active_body_pos_g": 999.0}},
        "motions": [_motion(0, side), _motion(1, side)],
    }


def _eight(side):
    return [_report(mode_index, side) for mode_index in range(8)]


def _write_reports(tmp_path, prefix, reports):
    paths = []
    for report in reports:
        path = tmp_path / f"{prefix}-{report['mode_index']}.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        paths.append(path)
    return paths


def test_compare_derives_per_mode_values_and_deltas_from_motion_rows():
    result = compare_reports(_eight("reference"), list(reversed(_eight("candidate"))))

    assert result["schema_version"] == 1
    assert result["overall_candidate_no_regression"] is True
    assert result["interpretation"] == (
        "Paired report comparison only; not a formal statistical significance test, "
        "a convergence claim, or a full-BFM validation."
    )
    assert result["reference_checkpoint_sha256"] == "a" * 64
    assert result["candidate_checkpoint_sha256"] == "b" * 64
    assert result["comparator_source_sha256"] == hashlib.sha256(
        Path(compare_module.__file__).read_bytes()
    ).hexdigest()
    assert len(result["modes"]) == 8
    assert result["modes"][0] == {
        "mode_index": 0,
        "mode": "Pelvis-1",
        "active_body_names": ["pelvis"],
        "num_motions": 2,
        "reference": {
            "mean_metrics": {
                "error_active_body_pos_g": 0.30000000000000004,
                "error_active_body_pos_root_relative": 0.2,
                "error_active_body_rot": 0.6,
                "error_active_body_pos_g_max": 0.1,
                "error_all_body_pos_g_max": 0.1,
                "error_body_pos_g": 0.4,
            },
            "active_max_link_tracking_pass_rate": 0.5,
            "all_configured_14_max_link_tracking_pass_rate": 0.5,
        },
        "candidate": {
            "mean_metrics": {
                "error_active_body_pos_g": 0.2,
                "error_active_body_pos_root_relative": 0.15,
                "error_active_body_rot": 0.5,
                "error_active_body_pos_g_max": 0.1,
                "error_all_body_pos_g_max": 0.1,
                "error_body_pos_g": 0.30000000000000004,
            },
            "active_max_link_tracking_pass_rate": 1.0,
            "all_configured_14_max_link_tracking_pass_rate": 0.5,
        },
        "delta_candidate_minus_reference": {
            "mean_metrics": {
                "error_active_body_pos_g": -0.10000000000000003,
                "error_active_body_pos_root_relative": -0.05000000000000002,
                "error_active_body_rot": -0.09999999999999998,
                "error_active_body_pos_g_max": 0.0,
                "error_all_body_pos_g_max": 0.0,
                "error_body_pos_g": -0.09999999999999998,
            },
            "active_max_link_tracking_pass_rate": 0.5,
            "all_configured_14_max_link_tracking_pass_rate": 0.0,
        },
        "candidate_no_regression": True,
    }


def test_position_mean_allows_only_the_documented_numerical_tolerance():
    reference = _eight("reference")
    candidate = _eight("candidate")
    for row in candidate[0]["motions"]:
        row["metrics"]["error_active_body_pos_g"]["mean"] += 0.100002
        row["metrics"]["error_active_body_pos_g"]["max"] += 0.100002

    result = compare_reports(reference, candidate)

    assert result["modes"][0]["candidate_no_regression"] is False
    assert result["overall_candidate_no_regression"] is False

    for row in candidate[0]["motions"]:
        row["metrics"]["error_active_body_pos_g"]["mean"] -= 0.000001
    result = compare_reports(reference, candidate)
    assert result["modes"][0]["candidate_no_regression"] is True


@pytest.mark.parametrize("regression", ["rotation", "active_pass_rate", "all_pass_rate"])
def test_each_no_regression_condition_can_fail_a_mode(regression):
    reference = _eight("reference")
    candidate = _eight("candidate")
    if regression == "rotation":
        for row in candidate[0]["motions"]:
            row["metrics"]["error_active_body_rot"]["mean"] += 0.100002
            row["metrics"]["error_active_body_rot"]["max"] += 0.100002
    elif regression == "active_pass_rate":
        for row in candidate[0]["motions"]:
            row["metrics"]["error_active_body_pos_g_max"]["max"] = 0.6
    else:
        candidate[0]["motions"][0]["metrics"]["error_all_body_pos_g_max"]["max"] = 0.51

    result = compare_reports(reference, candidate)

    assert result["modes"][0]["candidate_no_regression"] is False
    assert result["overall_candidate_no_regression"] is False


def test_legacy_global_body_position_metric_is_optional_when_consistently_absent():
    reference = _eight("reference")
    candidate = _eight("candidate")
    for report in reference + candidate:
        for row in report["motions"]:
            row["metrics"].pop("error_body_pos_g")

    result = compare_reports(reference, candidate)

    assert "error_body_pos_g" not in result["modes"][0]["reference"]["mean_metrics"]


@pytest.mark.parametrize(
    ("reports", "message"),
    [
        (lambda: _eight("reference")[:-1], "exactly mode_index 0..7"),
        (
            lambda: [*_eight("reference")[:7], copy.deepcopy(_eight("reference")[0])],
            "duplicate mode_index 0",
        ),
    ],
)
def test_rejects_missing_or_duplicate_modes(reports, message):
    with pytest.raises(ComparisonError, match=message):
        compare_reports(reports(), _eight("candidate"))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda reports: reports[3]["protocol"].update({"action_policy": "different"}),
            "protocol differs",
        ),
        (
            lambda reports: reports[3]["input_manifest"]["motion_index"].update({"sha256": "f" * 64}),
            "motion_index content differs",
        ),
        (
            lambda reports: reports[3]["input_manifest"]["motions"].update({"sha256": "f" * 64}),
            "motions content differs",
        ),
        (
            lambda reports: reports[3]["input_manifest"]["python_sources"].update({"sha256": "f" * 64}),
            "python_sources content differs",
        ),
    ],
)
def test_rejects_mixed_protocol_data_or_code(mutation, message):
    candidate = _eight("candidate")
    mutation(candidate)

    with pytest.raises(ComparisonError, match=message):
        compare_reports(_eight("reference"), candidate)


def test_rejects_mixed_checkpoints_within_one_side_but_allows_different_sides():
    candidate = _eight("candidate")
    candidate[4]["checkpoint_sha256"] = "f" * 64
    candidate[4]["input_manifest"]["checkpoint"]["sha256"] = "f" * 64

    with pytest.raises(ComparisonError, match="candidate reports mix checkpoint content"):
        compare_reports(_eight("reference"), candidate)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda report: report.update({"schema_version": 3}), "schema_version 4"),
        (
            lambda report: report.update({"input_manifest_verified_unchanged": False}),
            "verified unchanged",
        ),
        (lambda report: report["protocol"].update({"training_updates": 1}), "training updates"),
        (lambda report: report.update({"task": "other"}), "task must be"),
        (lambda report: report.update({"mode": "renamed"}), "canonical G1 BFM preset"),
        (lambda report: report.update({"active_body_names": ["pelvis"]}), "canonical G1 BFM preset"),
    ],
)
def test_rejects_invalid_or_unpaired_report_metadata(mutation, message):
    candidate = _eight("candidate")
    mutation(candidate[2])

    with pytest.raises(ComparisonError, match=message):
        compare_reports(_eight("reference"), candidate)


@pytest.mark.parametrize(
    "field",
    [
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
    ],
)
def test_rejects_required_paired_field_missing_from_both_sides(field):
    reference = _eight("reference")
    candidate = _eight("candidate")
    for report in reference + candidate:
        report.pop(field)

    with pytest.raises(ComparisonError, match=f"missing required field {field}"):
        compare_reports(reference, candidate)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seed", True),
        ("num_envs", 0),
        ("max_steps", None),
        ("max_steps", 0),
        ("step_dt", 0.0),
        ("step_dt", math.inf),
    ],
)
def test_rejects_impossible_evaluation_parameters_on_both_sides(field, value):
    reference = _eight("reference")
    candidate = _eight("candidate")
    for report in reference + candidate:
        report[field] = value

    with pytest.raises(ComparisonError, match=field):
        compare_reports(reference, candidate)


@pytest.mark.parametrize("field", ["device", "torch", "python", "package_versions", "scene_variant"])
def test_rejects_null_for_nonnullable_paired_fields_on_both_sides(field):
    reference = _eight("reference")
    candidate = _eight("candidate")
    for report in reference + candidate:
        report[field] = None

    with pytest.raises(ComparisonError, match=f"{field} must not be null"):
        compare_reports(reference, candidate)


def test_rejects_missing_masked_tracking_protocol_even_when_both_sides_match():
    reference = _eight("reference")
    candidate = _eight("candidate")
    for report in reference + candidate:
        report["protocol"].pop("masked_tracking")

    with pytest.raises(ComparisonError, match="protocol.masked_tracking"):
        compare_reports(reference, candidate)


def test_rejects_incomplete_masked_tracking_protocol_even_when_both_sides_match():
    reference = _eight("reference")
    candidate = _eight("candidate")
    for report in reference + candidate:
        report["protocol"]["masked_tracking"].pop("rotation")

    with pytest.raises(ComparisonError, match="protocol.masked_tracking is incomplete"):
        compare_reports(reference, candidate)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"source_frames": 1, "evaluated_steps": 0}, "source_frames"),
        ({"evaluated_steps": 9}, "evaluation horizon"),
        ({"truncated": True}, "truncated"),
    ],
)
def test_rejects_impossible_clip_horizon_on_both_sides(updates, message):
    reference = _eight("reference")
    candidate = _eight("candidate")
    for report in reference + candidate:
        report["motions"][0].update(updates)

    with pytest.raises(ComparisonError, match=message):
        compare_reports(reference, candidate)


def test_accepts_exact_truncated_clip_horizon():
    reference = _eight("reference")
    candidate = _eight("candidate")
    for report in reference + candidate:
        report["max_steps"] = 5
        for row in report["motions"]:
            row["evaluated_steps"] = 5
            row["truncated"] = True

    assert compare_reports(reference, candidate)["overall_candidate_no_regression"] is True


@pytest.mark.parametrize("metric_name", sorted(REQUIRED_METRICS)[:5])
def test_rejects_negative_required_mask_metric_on_both_sides(metric_name):
    reference = _eight("reference")
    candidate = _eight("candidate")
    for report in reference + candidate:
        report["motions"][0]["metrics"][metric_name]["mean"] = -0.01

    with pytest.raises(ComparisonError, match="nonnegative"):
        compare_reports(reference, candidate)


@pytest.mark.parametrize("metric_name", sorted(REQUIRED_METRICS)[:5])
def test_rejects_required_mask_metric_max_below_mean(metric_name):
    reference = _eight("reference")
    candidate = _eight("candidate")
    for report in reference + candidate:
        values = report["motions"][0]["metrics"][metric_name]
        values["max"] = values["mean"] - 0.000002

    with pytest.raises(ComparisonError, match="max below mean"):
        compare_reports(reference, candidate)


def test_accepts_metric_max_within_float_tolerance_of_mean():
    reference = _eight("reference")
    candidate = _eight("candidate")
    for report in reference + candidate:
        values = report["motions"][0]["metrics"]["error_active_body_rot"]
        values["max"] = values["mean"] - 0.000001

    assert compare_reports(reference, candidate)["overall_candidate_no_regression"] is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mode", "renamed-mode"),
        ("active_body_names", ["pelvis", "right_wrist_yaw_link"]),
    ],
)
def test_rejects_noncanonical_mode_binding_even_when_both_sides_match(field, value):
    reference = _eight("reference")
    candidate = _eight("candidate")
    for report in reference + candidate:
        if report["mode_index"] == 0:
            report[field] = value

    with pytest.raises(ComparisonError, match="canonical G1 BFM preset"):
        compare_reports(reference, candidate)


def test_rejects_invalid_sha256_format_even_when_both_sides_match():
    reference = _eight("reference")
    candidate = _eight("candidate")
    for report in reference + candidate:
        report["checkpoint_sha256"] = "not-a-sha"
        report["input_manifest"]["checkpoint"]["sha256"] = "not-a-sha"

    with pytest.raises(ComparisonError, match="64 hexadecimal"):
        compare_reports(reference, candidate)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda report: report["motions"][0].update({"motion": ""}), "nonempty clip identity"),
        (
            lambda report: report["motions"][1].update({"motion_id": 0}),
            "duplicate motion_id",
        ),
        (
            lambda report: report["motions"][1].update({"motion": "dataset/clip-0"}),
            "duplicate motion name",
        ),
        (
            lambda report: report["motions"][0].update({"source_frames": 10, "evaluated_steps": 9}),
            "clip metadata differs",
        ),
    ],
)
def test_rejects_empty_duplicate_or_unpaired_clip_identity_and_metadata(mutation, message):
    candidate = _eight("candidate")
    mutation(candidate[1])

    with pytest.raises(ComparisonError, match=message):
        compare_reports(_eight("reference"), candidate)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_rejects_nonfinite_motion_metrics(value):
    candidate = _eight("candidate")
    candidate[0]["motions"][0]["metrics"]["error_active_body_rot"]["mean"] = value

    with pytest.raises(ComparisonError, match="finite"):
        compare_reports(_eight("reference"), candidate)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda report: report["motions"][0]["metrics"].pop("error_active_body_rot"),
            "required metric",
        ),
        (
            lambda report: report["motions"][0]["metrics"].pop("error_body_pos_g"),
            "inconsistent metric sets",
        ),
        (
            lambda report: [
                row["metrics"].update({"candidate_only": {"mean": 1.0, "max": 1.0}})
                for row in report["motions"]
            ],
            "metric sets differ",
        ),
    ],
)
def test_rejects_missing_or_inconsistent_metric_sets(mutation, message):
    candidate = _eight("candidate")
    mutation(candidate[0])

    with pytest.raises(ComparisonError, match=message):
        compare_reports(_eight("reference"), candidate)


def test_cli_writes_fresh_json_and_refuses_to_overwrite(tmp_path, capsys):
    reference_paths = _write_reports(tmp_path, "reference", _eight("reference"))
    candidate_paths = _write_reports(tmp_path, "candidate", _eight("candidate"))
    output = tmp_path / "comparison.json"
    args = [
        "--reference", *map(str, reference_paths),
        "--candidate", *map(str, candidate_paths),
        "--output", str(output),
    ]

    assert main(args) == 0
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["overall_candidate_no_regression"] is True
    assert "8/8 modes pass" in capsys.readouterr().out

    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2
    assert json.loads(output.read_text(encoding="utf-8")) == written
    assert "already exists" in capsys.readouterr().err
