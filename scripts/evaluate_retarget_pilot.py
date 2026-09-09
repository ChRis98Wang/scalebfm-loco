#!/usr/bin/env python3
"""Frozen-policy A/B diagnostics for the complete 40-origin retarget-data pilot.

References intentionally differ; checkpoint, origins, windows, masks and runtime
must match. This is NOT the policy-training regression/promotion evaluator. All
kinematic rejects stay in the paired diagnostic. No training or data promotion.

Default is read-only validation and a printed plan. To execute, use the existing
IsaacLab interpreter in an owned bfm-retarget-eval-*.service cgroup configured
with KillMode=control-group. Each evaluator process group is bounded/reaped.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import re
import signal
from statistics import fmean
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
TRACK = ROOT / "ScaleTrack"
ENTRY = TRACK / "scripts/pretrain/rsl_rl"
OFFICIAL = ROOT / "logs/rsl_rl/g1_bfm_tracking_exp/humanoid_transformer_m/model_22200.pt"
OFFICIAL_SHA = "88d5a79946c03ed25503f48b2af71d16290844ef066ca9b6c8fa8dc3837422e3"
NUM_ENVS, MAX_STEPS, SEED, JOB_TIMEOUT = 64, 1000, 42, 240
METRICS = ("error_active_body_pos_g", "error_active_body_rot", "error_body_pos_g")
TOLERANCE = 1e-6


def helpers():
    # Import only pure helpers. No IsaacLab, Torch or simulation at module load.
    for path in (ROOT, ENTRY):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from scripts import retarget_data_refresh, compare_mask_evaluations
    import evaluation_manifest
    return retarget_data_refresh, compare_mask_evaluations, evaluation_manifest


def validate_archive(path, *, native_sha, frames):
    import numpy as np
    with np.load(path, allow_pickle=False) as archive:
        for key, expected in (("format_version", 3), ("fps", 50), ("quaternion_order", "wxyz"),
                              ("source_sha256", native_sha)):
            value = archive[key]
            if value.shape != () or value.item() != expected:
                raise ValueError(f"{path}: invalid {key}, expected {expected}")
        if not re.fullmatch(r"[0-9a-f]{64}", str(archive["pipeline_fingerprint"])):
            raise ValueError(f"{path}: invalid packaging pipeline fingerprint")
        shapes = {"joint_pos": (frames, 29), "joint_vel": (frames, 29),
                  "body_pos_w": (frames, 30, 3), "body_quat_w": (frames, 30, 4),
                  "body_lin_vel_w": (frames, 30, 3), "body_ang_vel_w": (frames, 30, 3),
                  "reference_root_pos": (frames, 3), "reference_root_quat_w": (frames, 4)}
        for key, shape in shapes.items():
            value = archive[key]
            if value.shape != shape or value.dtype.kind not in "fiu" or not np.isfinite(value).all():
                raise ValueError(f"{path}: {key} must be finite and have shape {shape}")
        for key in ("body_quat_w", "reference_root_quat_w"):
            if not np.allclose(np.linalg.norm(archive[key], axis=-1), 1, rtol=0, atol=1e-3):
                raise ValueError(f"{path}: non-unit {key}")
        return str(archive["pipeline_fingerprint"])


def validate_pilot(pilot, candidate_root, *, expected_count=40):
    """Bind full pilot selection to canonical split/source/PKL/NPZ content."""
    import yaml
    refresh, _, manifest_tools = helpers()
    pilot = Path(pilot).resolve(strict=True)
    manifest_path = pilot / "manifest.json" if pilot.is_dir() else pilot
    audit_path = manifest_path.parent / "sole_audit.json"
    manifest = json.loads(manifest_path.read_text())
    audit = json.loads(audit_path.read_text())
    if (manifest.get("schema"), manifest.get("automatic_promotion"), manifest.get("seed")) != (1, False, SEED):
        raise ValueError("Unexpected pilot schema, promotion flag or seed")
    if audit.get("schema") != 1 or audit.get("automatic_promotion") is not False:
        raise ValueError("Unexpected sole audit schema or promotion flag")
    rows = manifest.get("motions", [])
    if len(rows) != expected_count:
        raise ValueError(f"Evaluate the complete preselected {expected_count}-origin pilot, not a filtered subset")
    names = [row["origin_id"] for row in rows]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate pilot origin identity")
    audit_rows = audit.get("results", [])
    if len(audit_rows) != len(rows) or {row["origin_id"] for row in audit_rows} != set(names):
        raise ValueError("Sole audit must cover exactly all pilot origins, including rejects")
    by_origin = {row["origin_id"]: row for row in audit_rows}
    if Counter(f"{r['dataset']}/{r['split']}" for r in rows) != manifest.get("counts"):
        raise ValueError("Pilot declared dataset/split counts disagree")
    indexes, frozen = {}, {str(manifest_path): manifest_tools.sha256_file(manifest_path),
                          str(audit_path): manifest_tools.sha256_file(audit_path)}
    for path in (ROOT / "scripts/retarget_data_refresh.py", ROOT / "scripts/compare_mask_evaluations.py",
                 Path(__file__)):
        frozen[str(path)] = manifest_tools.sha256_file(path)
    for label, path in refresh.INDEXES.items():
        actual = manifest_tools.sha256_file(path)
        if actual != manifest.get("index_sha256", {}).get(label):
            raise ValueError(f"Canonical {label} split index changed since pilot selection")
        frozen[str(path)] = actual
        indexes[label] = yaml.safe_load(path.read_text())
    refresh.validate_source_disjoint(rows)
    candidate_root = Path(candidate_root).resolve(strict=True)
    expected_paths = {candidate_root / f"{name}.npz" for name in names}
    actual_paths = set(candidate_root.rglob("*.npz"))
    if actual_paths != expected_paths:
        raise ValueError("Candidate NPZ set differs from the complete pilot (missing or stray files)")
    result = []
    pipelines = set()
    for row in rows:
        name = row["origin_id"]
        label = row["index_label"]
        if label not in indexes or indexes[label].get(name) != row["baseline_packed"]:
            raise ValueError(f"Origin {name} does not match its canonical split index")
        if row["split"] != ("train" if label == "train" else "validation"):
            raise ValueError(f"Origin {name} moved across train/validation")
        resolved = refresh.resolve_inputs(row)
        if resolved != row:
            raise ValueError(f"Baseline/canonical source provenance changed for {name}")
        entry = by_origin[name]
        if entry["split"] != row["split"] or entry["source_sha256"] != row["source_sha256"]:
            raise ValueError(f"Candidate origin or split changed for {name}")
        if entry.get("status") not in ("REJECT_KINEMATIC", "PENDING_FROZEN_POLICY_EVALUATION"):
            raise ValueError("Unexpected audit status; frozen evaluation cannot imply promotion")
        native = Path(entry["candidate_pkl"])
        expected_native = manifest_path.parent / "sole_candidates" / f"{name}.pkl"
        if native.resolve(strict=True) != expected_native.resolve(strict=True):
            raise ValueError(f"Candidate native path does not bind origin {name}")
        if manifest_tools.sha256_file(native) != entry["candidate_sha256"]:
            raise ValueError(f"Candidate native PKL changed for {name}")
        candidate = candidate_root / f"{name}.npz"
        if candidate.is_symlink() or not candidate.resolve(strict=True).is_relative_to(candidate_root):
            raise ValueError("Candidate archives must be owned files inside the new candidate root")
        frames = row["packed_frames"]
        if type(frames) is not int or frames < 2:
            raise ValueError("Invalid pilot frame count")
        validate_archive(row["baseline_packed"], native_sha=row["baseline_pkl_sha256"], frames=frames)
        pipelines.add(validate_archive(candidate, native_sha=entry["candidate_sha256"], frames=frames))
        for path in (row["source"], row["canonical_source"], row["baseline_pkl"], row["baseline_packed"], native, candidate):
            frozen[str(Path(path))] = manifest_tools.sha256_file(path)
        result.append({**row, "candidate_packed": str(candidate), "candidate_packed_sha256": frozen[str(candidate)],
                       "kinematic_status": entry["status"], "kinematic_review_reasons": entry["review_reasons"]})
    if len(pipelines) != 1:
        raise ValueError("Candidate packed files mix packaging pipeline fingerprints")
    return {"rows": result, "frozen_files": frozen, "candidate_packaging_fingerprint": pipelines.pop(),
            "manifest": str(manifest_path), "audit": str(audit_path),
            "counts": dict(Counter(row["kinematic_status"] for row in result))}


def build_jobs(python, output):
    output = Path(output)
    jobs = []
    for mode in range(8):
        for side in ("baseline", "candidate"):
            name = f"eval_{side}_{mode}"
            command = [str(python), "-u", str(ENTRY / "evaluate.py"),
                       "--checkpoint_path", str(OFFICIAL), "--motion_file", str(output / f"{side}.yaml"),
                       "--output", str(output / f"{name}.json"), "--task", "G1-BFM-Transformer-Tracking",
                       "--mode_index", str(mode), "--mask_metrics", "--num_envs", str(NUM_ENVS),
                       "--max_steps", str(MAX_STEPS), "--seed", str(SEED), "--device", "cuda:0"]
            jobs.append({"name": name, "side": side, "mode": mode,
                         "timeout_seconds": JOB_TIMEOUT, "command": command})
    return jobs


def validate_report(report, pilot_rows, *, side, mode, expected_manifest=None):
    _, comparison, manifest_tools = helpers()
    mode_name, body_names = comparison.G1_BFM_MODE_PRESETS[mode]
    expected = {"schema_version": 4, "task": "G1-BFM-Transformer-Tracking", "mode_index": mode,
                "mode": mode_name, "active_body_names": list(body_names), "num_envs": NUM_ENVS,
                "max_steps": MAX_STEPS, "seed": SEED, "step_dt": .02, "device": "cuda:0",
                "scene_variant": "baseline", "target_object": None,
                "checkpoint_sha256": OFFICIAL_SHA, "input_manifest_verified_unchanged": True}
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError(f"{side}/{mode}: evaluation protocol differs")
    protocol = report.get("protocol", {})
    if type(protocol.get("training_updates")) is not int or protocol["training_updates"] != 0:
        raise ValueError("Frozen evaluation must perform zero training updates")
    if any(protocol.get(key) is not False for key in ("reset_disturbance", "observation_noise", "interval_pushes")):
        raise ValueError("Frozen evaluation noise/disturbance mismatch")
    inputs = report.get("input_manifest", {})
    if inputs.get("checkpoint", {}).get("sha256") != OFFICIAL_SHA:
        raise ValueError("Frozen checkpoint input SHA differs")
    if report.get("motion_index_sha256") != inputs.get("motion_index", {}).get("sha256"):
        raise ValueError("Evaluation motion-index SHA fields disagree")
    files = inputs.get("motions", {}).get("files", {})
    if set(files) != {row["origin_id"] for row in pilot_rows} or any(
            files[row["origin_id"]].get("sha256") != row[f"{side}_packed_sha256"] for row in pilot_rows):
        raise ValueError("Evaluation reference payload SHA differs from the declared pilot side")
    if expected_manifest is not None:
        manifest_tools.verify_unchanged(expected_manifest, inputs)
    normalized = comparison._normalize_motions(report, f"{side}/{mode}", MAX_STEPS)
    actual = {row["motion"]: row for row in normalized.values()}
    if set(actual) != {row["origin_id"] for row in pilot_rows}:
        raise ValueError("Evaluated origins differ from the full pilot")
    for row in pilot_rows:
        measured = actual[row["origin_id"]]
        if measured["source_frames"] != row["packed_frames"]:
            raise ValueError("Evaluation source frame count differs from paired archives")
        for metric in METRICS:
            value = measured["metrics"].get(metric, {}).get("mean")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid required tracking metric {metric}")
    summary = report.get("summary", {})
    if (summary.get("num_motions"), summary.get("evaluated_steps"), summary.get("truncated_motions")) != (
            len(actual), sum(r["evaluated_steps"] for r in actual.values()), sum(r["truncated"] for r in actual.values())):
        raise ValueError("Reported summary disagrees with validated per-origin coverage")
    return actual


def compare_reports(baseline_reports, candidate_reports, pilot_rows):
    """Reuse strict within-side validation, intentionally not same-reference pairing."""
    _, comparison, _ = helpers()
    baseline = comparison._normalize_side(baseline_reports, "baseline")
    candidate = comparison._normalize_side(candidate_reports, "candidate")
    modes = []
    for mode in range(8):
        base, cand = baseline[mode], candidate[mode]
        left = validate_report(base, pilot_rows, side="baseline", mode=mode)
        right = validate_report(cand, pilot_rows, side="candidate", mode=mode)
        for field in (*comparison.PAIRED_FIELDS, "protocol"):
            if base.get(field) != cand.get(field):
                raise ValueError(f"Paired {field} changed; only reference payloads may differ")
        if base["input_manifest"]["python_sources"] != cand["input_manifest"]["python_sources"]:
            raise ValueError("Runtime Python sources changed between paired evaluations")
        clips = []
        for origin in left:
            a, b = left[origin], right[origin]
            for field in ("motion_id", "source_frames", "evaluated_steps", "truncated"):
                if a[field] != b[field]:
                    raise ValueError(f"Paired clip {origin} {field} changed")
            values = {metric: {"baseline": a["metrics"][metric]["mean"],
                               "candidate": b["metrics"][metric]["mean"],
                               "delta": b["metrics"][metric]["mean"] - a["metrics"][metric]["mean"]}
                      for metric in METRICS}
            clips.append({"origin_id": origin, "metrics": values,
                          "frozen_tracking_no_regression": all(v["delta"] <= TOLERANCE for v in values.values())})
        def aggregate(names):
            selected = [row for row in clips if row["origin_id"] in names]
            values = {metric: {side: fmean(row["metrics"][metric][side] for row in selected)
                               for side in ("baseline", "candidate", "delta")} for metric in METRICS}
            return {"num_origins": len(selected), "mean_metrics": values,
                    "frozen_tracking_no_regression": all(v["delta"] <= TOLERANCE for v in values.values())}
        groups = {"all": set(left)}
        for field in ("dataset", "split", "kinematic_status"):
            for value in sorted({row[field] for row in pilot_rows}):
                groups[f"{field}/{value}"] = {row["origin_id"] for row in pilot_rows if row[field] == value}
        strata = {name: aggregate(names) for name, names in groups.items()}
        modes.append({"mode_index": mode, "mode": base["mode"], "strata": strata, "origins": clips})
    return {"schema": 1, "checkpoint_sha256": OFFICIAL_SHA, "training_updates": 0,
            "automatic_promotion": False, "reference_payload_change": "intentional",
            "scope": "frozen official policy tracking only; not training acceptance, fall/contact success or full BFM",
            "sample_scope": "all preselected pilot train+development origins, including kinematic rejects; not an untouched test",
            "no_regression_tolerance": TOLERANCE, "modes": modes,
            "aggregate_modes_passed": sum(mode["strata"]["all"]["frozen_tracking_no_regression"] for mode in modes),
            "per_origin_all_eight_modes_passed": {
                row["origin_id"]: all(next(clip for clip in mode["origins"] if clip["origin_id"] == row["origin_id"])
                                      ["frozen_tracking_no_regression"] for mode in modes) for row in pilot_rows}}


def owned_cgroup():
    matches = re.findall(r"(?:^|/)(bfm-retarget-eval-[A-Za-z0-9_.-]+\.service)(?:/|$)",
                         Path("/proc/self/cgroup").read_text(), flags=re.MULTILINE)
    if len(matches) != 1:
        raise ValueError("Execute in an owned bfm-retarget-eval-*.service cgroup")
    mode = subprocess.check_output(["systemctl", "--user", "show", matches[0], "-p", "KillMode", "--value"], text=True).strip()
    if mode != "control-group":
        raise ValueError("Owned evaluation service must use KillMode=control-group")
    return matches[0]


def run_job(job, log_path):
    with Path(log_path).open("x") as log:
        child = subprocess.Popen(job["command"], cwd=TRACK, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = child.wait(timeout=job["timeout_seconds"])
            if code:
                raise subprocess.CalledProcessError(code, job["command"])
        finally:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()


def _write_json(path, payload):
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        parser.error("Output must be new; prior diagnostics are never overwritten")
    if args.execute:
        unit = owned_cgroup()
        if args.python.absolute() != Path(sys.executable).absolute():
            parser.error("Launch this controller with the same existing IsaacLab --python interpreter")
    pilot = validate_pilot(args.pilot, args.candidate_root)
    jobs = build_jobs(args.python, args.output.absolute())
    if not args.execute:
        print(json.dumps({"plan_only": True, "output": str(args.output.absolute()), "origins": len(pilot["rows"]),
                          "kinematic_counts": pilot["counts"], "jobs": jobs, "automatic_promotion": False}, indent=2))
        return 0
    import yaml
    _, _, manifest_tools = helpers()
    if manifest_tools.sha256_file(OFFICIAL) != OFFICIAL_SHA:
        raise ValueError("Frozen official checkpoint SHA changed")
    runtime = {"entrypoints": ENTRY, **manifest_tools.locate_package_sources(
        ("scaletrack", "my_rsl_rl", "isaaclab", "isaaclab_rl", "isaaclab_tasks"))}
    output = args.output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    status = {"schema": 1, "result": "RUNNING", "unit": unit, "jobs": [], "jobs_requested": 16,
              "training_updates": 0, "automatic_promotion": False, "kinematic_counts": pilot["counts"],
              "controller_sha256": manifest_tools.sha256_file(Path(__file__))}
    try:
        manifests = {}
        for side in ("baseline", "candidate"):
            index = {row["origin_id"]: row[f"{side}_packed"] for row in pilot["rows"]}
            with (output / f"{side}.yaml").open("x") as stream:
                yaml.safe_dump(index, stream, sort_keys=False)
            manifests[side] = manifest_tools.snapshot_inputs(OFFICIAL, output / f"{side}.yaml", runtime)
        _write_json(output / "inputs.json", {"pilot": pilot, "evaluation": manifests})
        _write_json(output / "status.json", status)
        reports = {"baseline": [], "candidate": []}
        for job in jobs:
            progress = {**job, "result": "RUNNING"}
            status["jobs"].append(progress)
            _write_json(output / "status.json", status)
            print(f"[RETARGET PILOT] {job['name']} START", flush=True)
            t0 = time.monotonic()
            run_job(job, output / f"{job['name']}.log")
            path = output / f"{job['name']}.json"
            report = json.loads(path.read_text())
            validate_report(report, pilot["rows"], side=job["side"], mode=job["mode"], expected_manifest=manifests[job["side"]])
            reports[job["side"]].append(report)
            progress.update(result="PASS", report_sha256=manifest_tools.sha256_file(path),
                            elapsed_seconds=time.monotonic() - t0)
            _write_json(output / "status.json", status)
            print(f"[RETARGET PILOT] {job['name']} COMPLETE", flush=True)
        result = compare_reports(reports["baseline"], reports["candidate"], pilot["rows"])
        for side, original in manifests.items():
            manifest_tools.verify_unchanged(original, manifest_tools.snapshot_inputs(OFFICIAL, output / f"{side}.yaml", runtime))
        for path, expected in pilot["frozen_files"].items():
            if manifest_tools.sha256_file(path) != expected:
                raise ValueError(f"Pilot source/provenance changed: {path}")
        if manifest_tools.sha256_file(Path(__file__)) != status["controller_sha256"]:
            raise ValueError("Evaluation controller changed during run")
        for job in status["jobs"]:
            if manifest_tools.sha256_file(output / f"{job['name']}.json") != job["report_sha256"]:
                raise ValueError("Evaluation report changed before final verification")
        _write_json(output / "comparison.json", result)
        status.update(result="COMPLETE", inputs_verified_unchanged=True,
                      aggregate_modes_passed=result["aggregate_modes_passed"],
                      comparison_sha256=manifest_tools.sha256_file(output / "comparison.json"))
    except BaseException as error:
        if status["jobs"] and status["jobs"][-1]["result"] == "RUNNING":
            status["jobs"][-1].update(result="FAIL", error=str(error))
        status.update(result="FAIL", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        status["elapsed_seconds"] = time.monotonic() - started
        _write_json(output / "status.json", status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
