#!/usr/bin/env python3
"""Owned, evaluation-only eight-mask comparison of the completed coverage run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

from run_bfm_learning_ablation import ROOT, TRACK, ENTRY, RUNS, TRAIN, LEGACY, KIT, OFFICIAL, OFFICIAL_SHA, validate_split

CANDIDATE = RUNS / "behavior_coverage_long_20260908a/model_23198.pt"
CANDIDATE_SHA = "344c74a7d7e9392077138750106922c9796d574ee585b82658a0d5b97ab73b8e"
TRAIN_OUTPUT = ROOT / "logs/behavior_learning/behavior_coverage_long_20260908a"


def build_jobs(python, output):
    output = Path(output)
    jobs = []
    for mode in range(8):
        for label, checkpoint in (("official", OFFICIAL), ("candidate", CANDIDATE)):
            command = [python, "-u", str(ENTRY / "evaluate.py"),
                       "--checkpoint_path", str(checkpoint),
                       "--motion_file", str(output / "heldout_union.yaml"),
                       "--output", str(output / f"eval_{label}_{mode}.json"),
                       "--task", "G1-BFM-Transformer-Tracking", "--mode_index", str(mode),
                       "--mask_metrics", "--num_envs", "1024", "--max_steps", "1000",
                       "--seed", "42", "--device", "cuda:0"]
            jobs.append((f"eval_{label}_{mode}", 360, command))
    return jobs


def validate_evaluation_evidence(report, label, mode, checkpoint_sha, indexed_names):
    # Reuse the strict row validator; a summary and exit 0 alone are insufficient.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.compare_mask_evaluations import G1_BFM_MODE_PRESETS, _normalize_motions

    expected_mode, expected_bodies = G1_BFM_MODE_PRESETS[mode]
    expected = {"schema_version": 4, "mode_index": mode, "mode": expected_mode,
                "active_body_names": list(expected_bodies), "task": "G1-BFM-Transformer-Tracking",
                "num_envs": 1024, "max_steps": 1000, "seed": 42, "device": "cuda:0",
                "scene_variant": "baseline", "target_object": None,
                "checkpoint_sha256": checkpoint_sha, "input_manifest_verified_unchanged": True}
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError(f"{label}/{mode} evaluation differs from the declared protocol")
    protocol = report.get("protocol", {})
    if type(protocol.get("training_updates")) is not int or protocol["training_updates"] != 0:
        raise ValueError("Evaluation must perform zero training updates")
    if any(protocol.get(key) is not False for key in ("reset_disturbance", "observation_noise", "interval_pushes")):
        raise ValueError("Evaluation disturbance/noise settings differ")
    if report.get("step_dt") != 0.02:
        raise ValueError("Expected 50 Hz evaluation")
    if report.get("input_manifest", {}).get("checkpoint", {}).get("sha256") != checkpoint_sha:
        raise ValueError("Report checkpoint fingerprint differs")
    summary = report.get("summary", {})
    if tuple(summary.get(key) for key in ("num_motions", "evaluated_steps", "truncated_motions")) != (1751, 605664, 98):
        raise ValueError("Evaluation did not finish the complete declared windows")
    rows = _normalize_motions(report, f"{label}/{mode}", 1000)
    if (len(rows) != 1751 or {row["motion"] for row in rows.values()} != set(indexed_names)
            or sum(row["evaluated_steps"] for row in rows.values()) != 605664
            or sum(row["truncated"] for row in rows.values()) != 98):
        raise ValueError("Actual per-clip evaluation coverage differs from the benchmark")
    return {key: summary[key] for key in ("num_motions", "evaluated_steps", "truncated_motions")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,70}", args.run_id):
        parser.error("Invalid run-id")
    output = ROOT / "logs/behavior_learning" / args.run_id
    jobs = build_jobs(sys.executable, output)
    if not args.execute:
        print(json.dumps({"output": str(output), "jobs": jobs, "automatic_promotion": False}, indent=2))
        return
    if f"bfm-{args.run_id}.service" not in Path("/proc/self/cgroup").read_text():
        parser.error(f"Execute inside owned bfm-{args.run_id}.service with KillMode=control-group")
    if output.exists() or output.is_symlink():
        parser.error(f"Refusing existing experiment directory: {output}")
    sys.path.insert(0, str(ENTRY))
    from evaluation_manifest import locate_package_sources, sha256_file, snapshot_inputs, verify_unchanged
    import yaml

    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    status = {"schema": 1, "result": "RUNNING", "stage": "preflight", "run_id": args.run_id,
              "jobs": [], "jobs_requested": 16, "automatic_promotion": False,
              "quality_result": "not evaluated", "training_updates": 0,
              "interpretation": "paired reference tracking only, not task/contact or full-BFM validation"}

    def save_status():
        status["elapsed_seconds"] = time.monotonic() - started
        temporary = output / "status.json.tmp"
        temporary.write_text(json.dumps(status, indent=2, allow_nan=False) + "\n")
        temporary.replace(output / "status.json")

    save_status()
    try:
        fingerprints = {"official": OFFICIAL_SHA, "candidate": CANDIDATE_SHA}
        for label, checkpoint in (("official", OFFICIAL), ("candidate", CANDIDATE)):
            if sha256_file(checkpoint) != fingerprints[label]:
                raise ValueError(f"Approved {label} checkpoint fingerprint changed")
        proof_path = TRAIN_OUTPUT / "status.json"
        proof = json.loads(proof_path.read_text())
        if (proof.get("result"), proof.get("updates_requested"), proof.get("inputs_verified_unchanged")) != (
            "COMPLETE", 1000, True
        ) or proof.get("training_returncode") != 0:
            raise ValueError("Candidate does not have a completed verified training audit")
        if (proof.get("final_checkpoint") != str(CANDIDATE)
                or proof.get("final_checkpoint_sha256") != CANDIDATE_SHA
                or proof.get("parameter_update_verification", {}).get("runner_iteration") != 23198
                or proof.get("coverage_verification", {}).get("actual_all_training_motions_sampled") is not True):
            raise ValueError("Candidate training proof differs from the approved final checkpoint")
        status["training_proof"] = {"path": str(proof_path), "sha256": sha256_file(proof_path)}
        indexes = [yaml.safe_load(path.read_text()) for path in (TRAIN, LEGACY, KIT)]
        benchmark = validate_split(*indexes)
        benchmark_path = output / "heldout_union.yaml"
        with benchmark_path.open("x") as stream:
            yaml.safe_dump(benchmark, stream, sort_keys=False)
        index_hashes = {str(path): sha256_file(path) for path in (TRAIN, LEGACY, KIT)}
        if index_hashes != proof.get("original_index_fingerprints"):
            raise ValueError("Original split indexes changed since candidate training")
        runtime_roots = {"entrypoints": ENTRY}
        runtime_roots.update(locate_package_sources(("scaletrack", "my_rsl_rl", "isaaclab", "isaaclab_rl", "isaaclab_tasks")))
        audit_roots = {**runtime_roots, "experiment_tools": ROOT / "scripts"}
        manifests = {"training_inputs": snapshot_inputs(CANDIDATE, TRAIN, audit_roots),
                     "official": snapshot_inputs(OFFICIAL, benchmark_path, runtime_roots),
                     "candidate": snapshot_inputs(CANDIDATE, benchmark_path, runtime_roots)}
        prior = json.loads((TRAIN_OUTPUT / "inputs.json").read_text())
        for key in ("motion_index", "motions"):
            if prior[key] != manifests["training_inputs"][key]:
                raise ValueError(f"Training {key} changed since candidate training")
        # Adding this evaluation controller is expected; the actual simulator,
        # policy and entrypoint Python sources must still match the training run.
        old_runtime = {key: value for key, value in prior["python_sources"]["files"].items()
                       if not key.startswith("experiment_tools/")}
        if old_runtime != manifests["official"]["python_sources"]["files"]:
            raise ValueError("Runtime Python sources changed since candidate training")
        train_hashes = {row["sha256"] for row in manifests["training_inputs"]["motions"]["files"].values()}
        eval_hashes = {row["sha256"] for row in manifests["official"]["motions"]["files"].values()}
        if train_hashes & eval_hashes:
            raise ValueError("Exact training/validation payload overlap")
        with (output / "inputs.json").open("x") as stream:
            json.dump(manifests, stream, indent=2)
        status.update(input_index_fingerprints=index_hashes, checkpoint_fingerprints=fingerprints,
                      split={"training": 7174, "legacy_validation": 962, "KIT_validation": 789,
                             "name_path_exact_payload_disjoint": True,
                             "upstream_or_near_duplicate_overlap": "not excluded"})
        for name, timeout, command in jobs:
            label, raw_mode = name.removeprefix("eval_").rsplit("_", 1)
            mode = int(raw_mode)
            status["stage"] = name
            job = {"name": name, "result": "RUNNING", "command": command, "timeout_seconds": timeout}
            status["jobs"].append(job)
            save_status()
            print(f"[BFM EVAL LONG] START {name}", flush=True)
            with (output / f"{name}.log").open("x") as log:
                child = subprocess.run(command, cwd=TRACK, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
            job["returncode"] = child.returncode
            if child.returncode:
                raise RuntimeError(f"{name} exited {child.returncode}; see its log")
            report_path = output / f"{name}.json"
            report = json.loads(report_path.read_text())
            job["evaluation_summary"] = validate_evaluation_evidence(report, label, mode, fingerprints[label], benchmark)
            verify_unchanged(manifests[label], report["input_manifest"])
            job.update(result="PASS", report_sha256=sha256_file(report_path), elapsed_seconds=report["elapsed_seconds"])
            save_status()
            print(f"[BFM EVAL LONG] COMPLETE {name} ({len(status['jobs'])}/16)", flush=True)

        status["stage"] = "comparison"
        save_status()
        comparison_path = output / "compare_official_vs_candidate.json"
        command = [sys.executable, str(ENTRY / "compare_learning_ablation.py"),
                   "--reference8", *[str(output / f"eval_official_{mode}.json") for mode in range(8)],
                   "--candidate8", *[str(output / f"eval_candidate_{mode}.json") for mode in range(8)],
                   "--outputnew", str(comparison_path)]
        comparison = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120)
        with (output / "comparison.log").open("x") as stream:
            stream.write(comparison.stdout + comparison.stderr)
        if comparison.returncode:
            raise RuntimeError(f"Comparison failed; see comparison.log (exit {comparison.returncode})")
        result = json.loads(comparison_path.read_text())
        if (result.get("reference_checkpoint_sha256"), result.get("candidate_checkpoint_sha256")) != (
            OFFICIAL_SHA, CANDIDATE_SHA
        ) or type(result.get("promotion_candidate_no_regression")) is not bool:
            raise ValueError("Comparison does not bind the selected checkpoints and quality verdict")
        for label, checkpoint, index, roots in (
            ("training_inputs", CANDIDATE, TRAIN, audit_roots),
            ("official", OFFICIAL, benchmark_path, runtime_roots),
            ("candidate", CANDIDATE, benchmark_path, runtime_roots),
        ):
            verify_unchanged(manifests[label], snapshot_inputs(checkpoint, index, roots))
        if any(sha256_file(path) != digest for path, digest in index_hashes.items()):
            raise RuntimeError("An original dataset index changed")
        if sha256_file(proof_path) != status["training_proof"]["sha256"]:
            raise RuntimeError("Historical training proof changed")
        for job in status["jobs"]:
            if sha256_file(output / f"{job['name']}.json") != job["report_sha256"]:
                raise RuntimeError("An evaluated report changed before final verification")
        quality = result["promotion_candidate_no_regression"]
        status.update(result="COMPLETE", stage="complete", inputs_verified_unchanged=True,
                      comparison_report=str(comparison_path), comparison_sha256=sha256_file(comparison_path),
                      quality_result="PASS_NO_REGRESSION" if quality else "FAIL_NO_REGRESSION",
                      aggregate_modes_passed=sum(row["candidate_no_regression"] for row in result["aggregate_comparison"]["modes"]),
                      stratum_mode_checks_passed=sum(row["candidate_no_regression"] for stratum in result["strata"].values()
                                                    for row in stratum["modes"]))
        print(f"[BFM EVAL LONG] COMPLETE; quality={status['quality_result']}; no automatic promotion", flush=True)
    except BaseException as error:
        if status["jobs"] and status["jobs"][-1]["result"] == "RUNNING":
            status["jobs"][-1].update(result="FAIL", error=str(error))
        status.update(result="FAIL", error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        save_status()


if __name__ == "__main__":
    main()
