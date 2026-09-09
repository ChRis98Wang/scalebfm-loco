#!/usr/bin/env python3
"""Compare complete eight-mask learning ablations on fixed KIT/non-KIT strata."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
import sys
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.compare_mask_evaluations import (  # noqa: E402
    ComparisonError,
    NUMERICAL_TOLERANCE,
    THRESHOLD_METRES,
    compare_reports,
)


EXPECTED_STRATA_COUNTS = {"non_kit": 962, "kit": 789}
POSITION_METRIC = "error_active_body_pos_g"
ROTATION_METRIC = "error_active_body_rot"
ACTIVE_MAX_METRIC = "error_active_body_pos_g_max"
ALL_MAX_METRIC = "error_all_body_pos_g_max"


def _mode_map(reports: Iterable[Mapping[str, Any]]) -> dict[int, Mapping[str, Any]]:
    return {int(report["mode_index"]): report for report in reports}


def _stratum(name: str, rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    if name == "kit":
        return [row for row in rows if row["motion"].startswith("KIT/")]
    return [row for row in rows if not row["motion"].startswith("KIT/")]


def _values(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    if not rows:
        raise ComparisonError("stratum must not be empty")
    count = len(rows)
    return {
        "mean_active_position_error_m": fmean(float(row["metrics"][POSITION_METRIC]["mean"]) for row in rows),
        "mean_active_rotation_error_rad": fmean(float(row["metrics"][ROTATION_METRIC]["mean"]) for row in rows),
        "active_max_link_tracking_pass_rate": sum(
            float(row["metrics"][ACTIVE_MAX_METRIC]["max"]) <= THRESHOLD_METRES for row in rows
        ) / count,
        "all_14_max_link_tracking_pass_rate": sum(
            float(row["metrics"][ALL_MAX_METRIC]["max"]) <= THRESHOLD_METRES for row in rows
        ) / count,
    }


def _finite(values: Mapping[str, float], description: str) -> None:
    if any(not math.isfinite(value) for value in values.values()):
        raise ComparisonError(f"{description} contains a nonfinite aggregate")


def compare_learning_ablation(
    reference_reports: Iterable[Mapping[str, Any]], candidate_reports: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    """Require aggregate and per-stratum no-regression for all eight canonical masks."""
    reference_reports = list(reference_reports)
    candidate_reports = list(candidate_reports)
    aggregate = compare_reports(reference_reports, candidate_reports)
    reference = _mode_map(reference_reports)
    candidate = _mode_map(candidate_reports)

    strata = {}
    for stratum_name, expected_count in EXPECTED_STRATA_COUNTS.items():
        modes = []
        for mode_index in range(8):
            ref_rows = _stratum(stratum_name, reference[mode_index]["motions"])
            cand_rows = _stratum(stratum_name, candidate[mode_index]["motions"])
            if len(ref_rows) != expected_count or len(cand_rows) != expected_count:
                raise ComparisonError(
                    f"{stratum_name} mode_index {mode_index} must contain exactly {expected_count} motions"
                )
            if [row["motion"] for row in ref_rows] != [row["motion"] for row in cand_rows]:
                raise ComparisonError(f"{stratum_name} clip ordering differs for mode_index {mode_index}")
            ref_values, cand_values = _values(ref_rows), _values(cand_rows)
            _finite(ref_values, f"reference {stratum_name} mode_index {mode_index}")
            _finite(cand_values, f"candidate {stratum_name} mode_index {mode_index}")
            deltas = {key: cand_values[key] - value for key, value in ref_values.items()}
            no_regression = (
                cand_values["mean_active_position_error_m"]
                <= ref_values["mean_active_position_error_m"] + NUMERICAL_TOLERANCE
                and cand_values["mean_active_rotation_error_rad"]
                <= ref_values["mean_active_rotation_error_rad"] + NUMERICAL_TOLERANCE
                and deltas["active_max_link_tracking_pass_rate"] >= 0.0
                and deltas["all_14_max_link_tracking_pass_rate"] >= 0.0
            )
            modes.append({
                "mode_index": mode_index,
                "mode": reference[mode_index]["mode"],
                "num_motions": expected_count,
                "reference": ref_values,
                "candidate": cand_values,
                "delta_candidate_minus_reference": deltas,
                "candidate_no_regression": no_regression,
            })
        strata[stratum_name] = {
            "num_motions": expected_count,
            "candidate_no_regression": all(mode["candidate_no_regression"] for mode in modes),
            "modes": modes,
        }

    promotion = aggregate["overall_candidate_no_regression"] and all(
        value["candidate_no_regression"] for value in strata.values()
    )
    return {
        "schema_version": 1,
        "comparison": "candidate minus reference",
        "interpretation": (
            "Paired 1,751-clip comparison with predeclared KIT/non-KIT strata; "
            "not a significance, convergence, or unseen-upstream-data claim."
        ),
        "reference_checkpoint_sha256": aggregate["reference_checkpoint_sha256"],
        "candidate_checkpoint_sha256": aggregate["candidate_checkpoint_sha256"],
        "tracking_threshold_metres": THRESHOLD_METRES,
        "numerical_tolerance": NUMERICAL_TOLERANCE,
        "expected_strata_counts": EXPECTED_STRATA_COUNTS,
        "aggregate_comparison": aggregate,
        "strata": strata,
        "promotion_candidate_no_regression": promotion,
        "comparator_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def _load(paths: Sequence[str], description: str) -> list[Mapping[str, Any]]:
    reports = []
    for text in paths:
        path = Path(text)
        try:
            with path.open(encoding="utf-8") as stream:
                reports.append(json.load(stream))
        except (OSError, json.JSONDecodeError) as error:
            raise ComparisonError(f"cannot read {description} report {path}: {error}") from error
    return reports


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference8", nargs=8, required=True, metavar="REPORT")
    parser.add_argument("--candidate8", nargs=8, required=True, metavar="REPORT")
    parser.add_argument("--outputnew", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.outputnew.exists():
            raise ComparisonError(f"output already exists: {args.outputnew}")
        result = compare_learning_ablation(
            _load(args.reference8, "reference"), _load(args.candidate8, "candidate")
        )
        result["reference_reports"] = list(args.reference8)
        result["candidate_reports"] = list(args.candidate8)
        args.outputnew.parent.mkdir(parents=True, exist_ok=True)
        with args.outputnew.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    except (ComparisonError, OSError) as error:
        parser.error(str(error))
    passed = sum(
        mode["candidate_no_regression"]
        for stratum in result["strata"].values()
        for mode in stratum["modes"]
    )
    print(
        f"[BFM ABLATION] {passed}/16 stratum×mode checks pass; "
        f"promotion={str(result['promotion_candidate_no_regression']).lower()}; output={args.outputnew}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
