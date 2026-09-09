#!/usr/bin/env python3
"""Compare official/A/B eight-mask results on the fixed 1,751-origin benchmark.

Read-only unless --outputnew is supplied. Exit zero means comparison completed,
not quality approval. No simulation, training, seed confirmation or promotion.
The controller supplies independently frozen final-checkpoint/reference identities.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from statistics import fmean
import sys

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "ScaleTrack/scripts/pretrain/rsl_rl"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ENTRY))
from compare_learning_ablation import compare_learning_ablation
from scripts import compare_mask_evaluations as masks

ComparisonError = masks.ComparisonError
OFFICIAL_SHA = "88d5a79946c03ed25503f48b2af71d16290844ef066ca9b6c8fa8dc3837422e3"
PROTOCOL_VERSION = "umr_behavior_ab_20260909/v1"
MEAN_METRICS = ("error_active_body_pos_g", "error_active_body_rot", "error_body_pos_g")
COUNTS = {"non_kit": 962, "kit": 789}
EXPECTED_STEPS, EXPECTED_TRUNCATED = 605664, 98
INDEXES = {
    ROOT / "ScaleRetarget/retargeted_dataset/amass_full_v1_validation.yaml": "2ec2fe800f6a3fc27e98af58b18037752058ef263a0064982671bc138aaf2529",
    ROOT / "ScaleRetarget/retargeted_dataset/amass_kit_heldout_v1.yaml": "85b1a0635d1ba5006df4a06374e2b604150cda319f290aa54cece93dcb99be71",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha(value, description):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ComparisonError(f"{description} requires a lowercase SHA256 identity")
    return value


def validate_record(record, description):
    if (not isinstance(record, dict) or not isinstance(record.get("path"), str) or not record["path"]
            or type(record.get("size_bytes")) is not int or record["size_bytes"] < 0):
        raise ComparisonError(f"{description} requires an explicit path, nonnegative size and SHA")
    require_sha(record.get("sha256"), description)


def group_digest(files):
    """Match evaluation_manifest._group's portable content fingerprint exactly."""
    if not isinstance(files, dict) or not files:
        raise ComparisonError("Expected nonempty recorded input files")
    for name, record in files.items():
        if not isinstance(name, str) or not name:
            raise ComparisonError("Recorded input names must be nonempty strings")
        validate_record(record, name)
    encoded = json.dumps({key: value["sha256"] for key, value in files.items()},
                         sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_reference_files(expected_motions):
    group_digest(expected_motions)
    if len(expected_motions) != 1751:
        raise ComparisonError("Expected the complete fixed 1,751-origin reference")
    counts = Counter("kit" if name.startswith("KIT/") else "non_kit" for name in expected_motions)
    if counts != COUNTS:
        raise ComparisonError("Fixed reference must contain 962 non-KIT and 789 KIT origins")
    for name in expected_motions:
        if (name.startswith("/") or "\\" in name or any(part in ("", ".", "..") for part in name.split("/"))):
            raise ComparisonError("Unsafe fixed reference origin")


def validate_family(reports, label, checkpoint_sha, expected_motions, seed, *,
                    num_envs=1024, expected_count=1751, expected_coverage=(EXPECTED_STEPS, EXPECTED_TRUNCATED)):
    normalized = masks._normalize_side(reports, label)
    for mode, report in normalized.items():
        expected = {"checkpoint_sha256": checkpoint_sha, "seed": seed, "num_envs": num_envs,
                    "max_steps": 1000, "step_dt": .02, "device": "cuda:0", "scene_variant": "baseline",
                    "target_object": None}
        if any(report.get(key) != value for key, value in expected.items()):
            raise ComparisonError(f"{label}/{mode} differs from the fixed checkpoint/seed/evaluation protocol")
        protocol = report["protocol"]
        if any(protocol.get(key) is not False for key in ("reset_disturbance", "observation_noise", "interval_pushes")):
            raise ComparisonError("Heldout evaluation requires all observation/reset/push perturbations disabled")
        manifest = report["input_manifest"]
        for name in ("checkpoint", "motion_index"):
            validate_record(manifest[name], f"{label}/{mode} {name}")
        for field in ("motions", "python_sources"):
            if group_digest(manifest[field].get("files")) != manifest[field].get("sha256"):
                raise ComparisonError(f"{label}/{mode} {field} group hash does not bind its file records")
        if manifest["motions"]["files"] != expected_motions:
            raise ComparisonError("Evaluation does not use the exact frozen same-reference paths/bytes")
        if report.get("checkpoint") != manifest["checkpoint"]["path"]:
            raise ComparisonError("Checkpoint path disagrees with its input record")
        if (report.get("motion_index") != manifest["motion_index"]["path"]
                or report.get("motion_index_sha256") != manifest["motion_index"]["sha256"]):
            raise ComparisonError("Reference index identity fields disagree")
        rows = list(report["_motions_by_id"].values())
        if len(rows) != expected_count or {row["motion"] for row in rows} != set(expected_motions):
            raise ComparisonError("Actual evaluated origins differ from the fixed heldout benchmark")
        coverage = (len(rows), sum(row["evaluated_steps"] for row in rows), sum(row["truncated"] for row in rows))
        if expected_coverage is not None and coverage != (expected_count, *expected_coverage):
            raise ComparisonError("Actual evaluated steps/truncations differ from the fixed windows")
        if tuple(report.get("summary", {}).get(key) for key in ("num_motions", "evaluated_steps", "truncated_motions")) != coverage:
            raise ComparisonError("Summary does not match actual heldout per-origin coverage")
        for row in rows:
            for metric in MEAN_METRICS:
                values = row["metrics"].get(metric)
                if not isinstance(values, dict):
                    raise ComparisonError(f"Required additional three-mean metric missing: {metric}")
                mean = masks._finite_number(values.get("mean"), metric + ".mean")
                maximum = masks._finite_number(values.get("max"), metric + ".max")
                if mean < 0 or maximum < 0 or maximum + masks.NUMERICAL_TOLERANCE < mean:
                    raise ComparisonError(f"Invalid nonnegative mean/max for {metric}")
    return normalized


def three_mean_comparison(reference, candidate, *, cohorts=None):
    """Per-origin equal-weight means, independently gated in each mode/stratum."""
    if cohorts is None:
        names = [row["motion"] for row in reference[0]["motions"]]
        cohorts = {"aggregate": set(names), "non_kit": {name for name in names if not name.startswith("KIT/")},
                   "kit": {name for name in names if name.startswith("KIT/")}}
    groups = {}
    for stratum, origin_names in cohorts.items():
        modes = []
        for mode in range(8):
            def rows(side):
                return [row for row in side[mode]["motions"] if row["motion"] in origin_names]
            before, after = rows(reference), rows(candidate)
            expected = len(origin_names)
            if len(before) != expected or len(after) != expected:
                raise ComparisonError("Unexpected three-mean stratum membership")
            ref = {key: fmean(row["metrics"][key]["mean"] for row in before) for key in MEAN_METRICS}
            cand = {key: fmean(row["metrics"][key]["mean"] for row in after) for key in MEAN_METRICS}
            checks = {key: cand[key] <= ref[key] + masks.NUMERICAL_TOLERANCE for key in MEAN_METRICS}
            modes.append({"mode_index": mode, "mode": reference[mode]["mode"], "num_motions": expected,
                "reference_mean_metrics": ref, "candidate_mean_metrics": cand,
                "delta_candidate_minus_reference": {key: cand[key] - ref[key] for key in MEAN_METRICS},
                "metric_no_regression": checks, "candidate_no_regression": all(checks.values())})
        groups[stratum] = {"modes": modes, "modes_passed": sum(row["candidate_no_regression"] for row in modes),
                           "candidate_no_regression": all(row["candidate_no_regression"] for row in modes)}
    return {"metrics": list(MEAN_METRICS), "aggregation": "equal weight per origin; never pooled across masks",
            "numerical_tolerance": masks.NUMERICAL_TOLERANCE, "groups": groups,
            "candidate_no_regression": all(group["candidate_no_regression"] for group in groups.values())}


def compare_ab(official_reports, a_reports, b_reports, *, expected_checkpoints, expected_motions, seed=42):
    """Pure report comparator; identities come from the caller's frozen inputs.

    This verifies evaluation evidence, not the training-final/checkpoint selection
    audit. The controller must separately establish 100-update/coverage/Adam proof.
    """
    if type(seed) is not int or seed not in (42, 43, 44):
        raise ComparisonError("This protocol only predeclares seeds 42, 43 and 44")
    if set(expected_checkpoints) != {"official", "a", "b"}:
        raise ComparisonError("Supply all official/A/B independently frozen checkpoint identities")
    for label, digest in expected_checkpoints.items():
        require_sha(digest, label)
    if expected_checkpoints["official"] != OFFICIAL_SHA or len(set(expected_checkpoints.values())) != 3:
        raise ComparisonError("Require the official checkpoint and distinct final A/B checkpoint bytes")
    validate_reference_files(expected_motions)
    supplied = {"official": list(official_reports), "a": list(a_reports), "b": list(b_reports)}
    normalized = {label: validate_family(reports, label, expected_checkpoints[label], expected_motions, seed)
                  for label, reports in supplied.items()}
    # The original pure comparator validates exact paired protocol, runtime and
    # source fingerprints before calculating any original four quality gates.
    comparisons = {}
    for reference_label in ("a", "official"):
        original = compare_learning_ablation(supplied[reference_label], supplied["b"])
        additional = three_mean_comparison(normalized[reference_label], normalized["b"])
        comparisons[f"b_vs_{reference_label}"] = {"original_four_gate_comparison": original,
            "original_four_gates_no_regression": original["promotion_candidate_no_regression"],
            "additional_three_mean_comparison": additional,
            "additional_three_means_no_regression": additional["candidate_no_regression"],
            "promotion_candidate_no_regression": original["promotion_candidate_no_regression"] and additional["candidate_no_regression"]}
    all_pass = all(item["promotion_candidate_no_regression"] for item in comparisons.values())
    return {"schema": "bfm.umr_behavior_ab_comparison/1", "protocol_version": PROTOCOL_VERSION,
        "result": "COMPLETE", "seed": seed, "checkpoint_sha256": dict(expected_checkpoints),
        "reference_motions_sha256": group_digest(expected_motions), "reference_counts": COUNTS,
        "coverage_per_model_per_mask": {"motions": 1751, "evaluated_steps": EXPECTED_STEPS, "truncated_motions": EXPECTED_TRUNCATED},
        "comparisons": comparisons, "promotion_candidate_no_regression": all_pass,
        "quality_result": "PASS_THIS_SEED_NO_REGRESSION" if all_pass else "FAIL_THIS_SEED_NO_REGRESSION",
        "confirmation_quality_condition_met": comparisons["b_vs_a"]["promotion_candidate_no_regression"] if seed == 42 else None,
        "confirmation_seeds": {str(other): "NOT_EVALUATED_IN_THIS_REPORT" for other in (42, 43, 44) if other != seed},
        "independent_seed43_44_confirmation_complete": False, "automatic_confirmation_start": False,
        "training_evidence_validated_by_comparator": False, "automatic_promotion": False, "promotion_approved": False,
        "training_updates": 0, "comparator_source_sha256": sha256(Path(__file__)),
        "limitations": ["This seed's report is not three-seed confirmation or promotion approval.",
            "Seed-42 B>A quality is only one confirmation condition; both completed engineering audits are also required.",
            "Fixed development/heldout benchmark has been used previously; not a fresh final test or unknown-upstream guarantee.",
            "98 truncated references and tracking thresholds do not establish full-duration/contact/task success.",
            "Numerical tolerance is not statistical significance; training-final checkpoint provenance is controller-owned."]}


def compare_development(a_reports, b_reports, *, expected_checkpoints, expected_motions,
                        geometric_pass_origins, seed=42):
    """Same-reference ten-origin development diagnostic, never a promotion gate.

    Invoke once for the old-reference suite and once for the UMR-reference suite.
    Their motions fingerprints differ intentionally; do not cross-pair the suites.
    """
    if type(seed) is not int or seed not in (42, 43, 44):
        raise ComparisonError("Expected a predeclared seed")
    if set(expected_checkpoints) != {"a", "b"} or len(set(expected_checkpoints.values())) != 2:
        raise ComparisonError("Development requires distinct frozen A/B final checkpoints")
    for label, digest in expected_checkpoints.items():
        require_sha(digest, label)
    group_digest(expected_motions)
    passed = list(geometric_pass_origins)
    if (len(expected_motions) != 10 or len(passed) != 7 or len(set(passed)) != 7
            or not set(passed).issubset(expected_motions)):
        raise ComparisonError("Require all ten development origins and the seven predeclared geometry-pass origins")
    normalized = {label: validate_family(reports, label, expected_checkpoints[label], expected_motions, seed,
                                         num_envs=64, expected_count=10, expected_coverage=None)
                  for label, reports in (("a", a_reports), ("b", b_reports))}
    masks._validate_pair(normalized["a"], normalized["b"])
    for side in normalized.values():
        for report in side.values():
            if any(row["source_frames"] > 250 or row["truncated"] for row in report["motions"]):
                raise ComparisonError("Development diagnostic must preserve the bounded untruncated paired windows")
    cohorts = {"all10": set(expected_motions), "geometry_pass7": set(passed),
               "geometry_reject3": set(expected_motions) - set(passed)}
    return {"schema": "bfm.umr_behavior_ab_development/1", "protocol_version": PROTOCOL_VERSION,
        "result": "COMPLETE", "seed": seed, "checkpoint_sha256": dict(expected_checkpoints),
        "reference_motions_sha256": group_digest(expected_motions), "same_reference_verified": True,
        "cohorts": {key: sorted(value) for key, value in cohorts.items()},
        "three_mean_comparison": three_mean_comparison(normalized["a"], normalized["b"], cohorts=cohorts),
        "diagnostic_only": True, "automatic_promotion": False, "promotion_approved": False, "training_updates": 0,
        "comparator_source_sha256": sha256(Path(__file__)),
        "limitations": ["All ten origins were used for development, not a fresh final test.",
            "Compare A/B within the same reference; do not compare A-old against B-UMR.",
            "Geometry-pass/reject labels describe fixed references, not learned physical contact success."]}


def verify_inputs_snapshot(snapshot, frozen):
    for field in ("motions", "python_sources"):
        if group_digest(snapshot[field]["files"]) != snapshot[field]["sha256"]:
            raise ComparisonError(f"Frozen input {field} content digest mismatch")
    records = [snapshot["checkpoint"], snapshot["motion_index"],
               *snapshot["motions"]["files"].values(), *snapshot["python_sources"]["files"].values()]
    for record in records:
        validate_record(record, "frozen input")
        path = Path(record["path"]).resolve(strict=True)
        if path.stat().st_size != record["size_bytes"]:
            raise ComparisonError(f"Frozen input size changed: {path}")
        expected = record["sha256"]
        if str(path) not in frozen:
            if sha256(path) != expected:
                raise ComparisonError(f"Frozen input content changed: {path}")
            frozen[str(path)] = expected
        elif frozen[str(path)] != expected:
            raise ComparisonError(f"Mixed frozen input identity: {path}")


def validate_canonical_reference(expected, frozen):
    import yaml
    canonical = {}
    for path, digest in INDEXES.items():
        if sha256(path) != digest:
            raise ComparisonError(f"Original heldout index changed: {path}")
        frozen[str(path)] = digest
        index = yaml.safe_load(path.read_text())
        if set(canonical) & set(index):
            raise ComparisonError("Canonical heldout indexes overlap")
        canonical.update(index)
    if set(canonical) != set(expected):
        raise ComparisonError("Input snapshot is not the fixed canonical 1,751-origin heldout")
    for origin, path in canonical.items():
        if Path(path).resolve(strict=True) != Path(expected[origin]["path"]).resolve(strict=True):
            raise ComparisonError(f"Reference payload changed from the original canonical index: {origin}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True, help="Frozen snapshots keyed official/a/b")
    for label in ("official", "a", "b"):
        parser.add_argument(f"--{label}8", nargs=8, type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), default=42)
    parser.add_argument("--outputnew", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.outputnew is not None and (args.outputnew.exists() or args.outputnew.is_symlink()):
            raise ComparisonError("Refusing existing comparison output")
        inputs = json.loads(args.inputs.read_text())
        if set(inputs) != {"official", "a", "b"}:
            raise ComparisonError("--inputs must contain exactly official/a/b input snapshots")
        frozen = {str(args.inputs.resolve()): sha256(args.inputs), str(Path(__file__)): sha256(Path(__file__))}
        for path in (ENTRY / "compare_learning_ablation.py", ROOT / "scripts/compare_mask_evaluations.py",
                     ROOT / "docs/UMR_BEHAVIOR_AB_PROTOCOL_20260909.md"):
            frozen[str(path)] = sha256(path)
        reports, checkpoints = {}, {}
        for label in ("official", "a", "b"):
            verify_inputs_snapshot(inputs[label], frozen)
            checkpoints[label] = inputs[label]["checkpoint"]["sha256"]
            reports[label] = []
            for path in getattr(args, label + "8"):
                report = json.loads(path.read_text())
                if report.get("input_manifest") != inputs[label]:
                    raise ComparisonError(f"{label} report differs from the complete independently frozen input snapshot")
                reports[label].append(report)
                frozen[str(path.resolve())] = sha256(path)
        expected = inputs["official"]["motions"]["files"]
        validate_canonical_reference(expected, frozen)
        result = compare_ab(reports["official"], reports["a"], reports["b"],
                            expected_checkpoints=checkpoints, expected_motions=expected, seed=args.seed)
        if any(sha256(path) != digest for path, digest in frozen.items()):
            raise ComparisonError("Comparison inputs changed during read-only audit")
        result.update(input_sha256=frozen, inputs_verified_unchanged=True)
        if args.outputnew is None:
            print(json.dumps(result, indent=2, allow_nan=False))
        else:
            args.outputnew.parent.mkdir(parents=True, exist_ok=True)
            with args.outputnew.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, indent=2, allow_nan=False)
                stream.write("\n")
            print(json.dumps({"output": str(args.outputnew), "quality_result": result["quality_result"], "automatic_promotion": False}))
    except (ComparisonError, OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
