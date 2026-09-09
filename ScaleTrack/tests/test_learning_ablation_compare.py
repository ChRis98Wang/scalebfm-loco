"""Tests for strict KIT/non-KIT learning-ablation comparison."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/pretrain/rsl_rl"))
from compare_learning_ablation import ComparisonError, compare_learning_ablation
from scripts.compare_mask_evaluations import G1_BFM_MODE_PRESETS, MASK_METRIC_KEYS


HEX = "1" * 64


def _rows(legacy_value=0.1, kit_value=0.1):
    rows = []
    for motion_id in range(962 + 789):
        kit = motion_id >= 962
        value = kit_value if kit else legacy_value
        metrics = {name: {"mean": value, "max": value} for name in MASK_METRIC_KEYS}
        metrics["error_body_pos_g"] = {"mean": value, "max": value}
        rows.append({"motion_id": motion_id,
                     "motion": f"KIT/group/clip_{motion_id}" if kit else f"Legacy/group/clip_{motion_id}",
                     "source_frames": 2, "evaluated_steps": 1, "truncated": False, "metrics": metrics})
    return rows


def _reports(checkpoint, legacy_value=0.1, kit_value=0.1):
    protocol = {"training_updates": 0, "masked_tracking": {
        "measurement_boundary": "boundary", "position": "position", "root_relative_position": "relative",
        "rotation": "rotation", "tracking_failure": "failure", "limitations": "limitations"}}
    common = {"schema_version": 4, "task": "G1-BFM-Transformer-Tracking", "seed": 42,
              "num_envs": 1024, "max_steps": 1000, "step_dt": .02, "device": "cuda:0",
              "torch": "test", "python": "test", "package_versions": {}, "gpu": "test", "cuda": "test",
              "scene_variant": "baseline", "target_object": None, "protocol": protocol,
              "input_manifest_verified_unchanged": True}
    reports = []
    for mode_index, (mode, bodies) in enumerate(G1_BFM_MODE_PRESETS):
        report = deepcopy(common)
        report.update(mode_index=mode_index, mode=mode, active_body_names=list(bodies),
                      checkpoint_sha256=checkpoint, motions=_rows(legacy_value, kit_value),
                      input_manifest={"checkpoint": {"sha256": checkpoint},
                                      "motion_index": {"sha256": "2" * 64},
                                      "motions": {"sha256": "3" * 64},
                                      "python_sources": {"sha256": "4" * 64}})
        reports.append(report)
    return reports


def test_complete_equal_union_passes_all_sixteen_stratum_mode_checks():
    result = compare_learning_ablation(_reports(HEX), _reports("5" * 64))
    assert result["promotion_candidate_no_regression"] is True
    assert result["strata"]["non_kit"]["num_motions"] == 962
    assert result["strata"]["kit"]["num_motions"] == 789
    assert sum(mode["candidate_no_regression"] for stratum in result["strata"].values()
               for mode in stratum["modes"]) == 16


def test_kit_improvement_cannot_hide_non_kit_regression_in_aggregate_mean():
    reference = _reports(HEX, legacy_value=.1, kit_value=1.0)
    candidate = _reports("5" * 64, legacy_value=.11, kit_value=0.0)
    result = compare_learning_ablation(reference, candidate)
    assert result["aggregate_comparison"]["overall_candidate_no_regression"] is True
    assert result["strata"]["kit"]["candidate_no_regression"] is True
    assert result["strata"]["non_kit"]["candidate_no_regression"] is False
    assert result["promotion_candidate_no_regression"] is False


@pytest.mark.parametrize("remove_kit", [True, False])
def test_missing_or_empty_expected_stratum_is_rejected(remove_kit):
    reference, candidate = _reports(HEX), _reports("5" * 64)
    for reports in (reference, candidate):
        for report in reports:
            if remove_kit:
                report["motions"] = [row for row in report["motions"] if not row["motion"].startswith("KIT/")]
            else:
                report["motions"] = [row for row in report["motions"] if row["motion"].startswith("KIT/")]
    with pytest.raises(ComparisonError, match="must contain exactly"):
        compare_learning_ablation(reference, candidate)


def test_mixed_python_source_manifest_is_rejected_before_stratified_comparison():
    reference, candidate = _reports(HEX), _reports("5" * 64)
    candidate[3]["input_manifest"]["python_sources"]["sha256"] = "6" * 64
    with pytest.raises(ComparisonError, match="python_sources content differs"):
        compare_learning_ablation(reference, candidate)
