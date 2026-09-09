#!/usr/bin/env python3
"""Bounded, zero-update evaluation of the frozen 100-update UMR A/B arms."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from scripts import run_umr_behavior_ab as train
from scripts.run_bfm_long_evaluation import validate_evaluation_evidence

SCHEMA = "bfm.umr_behavior_ab_evaluation/1"


def build_jobs(python, output, manifest, checkpoints, suite):
    if suite not in ("heldout", "development", "all"):
        raise ValueError("Unknown evaluation suite")
    jobs = []
    references = []
    if suite in ("development", "all"):
        references.extend((f"development_{side}", spec["index"], 64, ("a", "b"))
                          for side, spec in manifest["development"]["indices"].items())
    if suite in ("heldout", "all"):
        references.append(("heldout", manifest["heldout"]["index"], 1024, ("official", "a", "b")))
    for reference, index, envs, labels in references:
        for mode in range(8):
            for label in labels:
                name = f"{reference}_{label}_{mode}"
                command = [str(python), "-u", str(train.ENTRY / "evaluate.py"),
                           "--checkpoint_path", str(checkpoints[label]), "--motion_file", str(index),
                           "--output", str(Path(output) / f"{name}.json"),
                           "--task", "G1-BFM-Transformer-Tracking", "--mode_index", str(mode),
                           "--mask_metrics", "--num_envs", str(envs), "--max_steps", "1000",
                           "--seed", "42", "--device", "cuda:0"]
                jobs.append({"name": name, "reference": reference, "label": label,
                             "mode": mode, "index": index, "command": command, "timeout_seconds": 360})
    return jobs


def validate_training_proof(path, manifest, dataset):
    proof = json.loads(Path(path).read_text())
    if (proof.get("schema"), proof.get("result"), proof.get("updates_requested"),
            proof.get("inputs_verified_unchanged"), proof.get("matched_cohorts"), proof.get("seed")) != (
            train.SCHEMA, "COMPLETE", 100, True, True, 42):
        raise ValueError("Evaluation requires complete matched 100-update A/B training")
    if set(proof.get("arms", {})) != {"a", "b"}:
        raise ValueError("Both formal training arms are required")
    frozen = train.freeze_inputs(dataset, manifest, train.runtime_roots())
    if proof.get("frozen_inputs") != frozen:
        raise ValueError("Training dataset/runtime changed before evaluation")
    if json.loads((Path(path).parent / "inputs.json").read_text()) != frozen:
        raise ValueError("Training input receipt differs from its status")
    import torch
    torch.set_num_threads(2)
    initial = torch.load(train.DERIVED, map_location="cpu", weights_only=False)
    checkpoints = {"official": train.OFFICIAL}
    audits = {}
    for arm, record in proof["arms"].items():
        checkpoint = Path(record["final_checkpoint"])
        expected = train.RUNS / record["run_name"] / "model_22298.pt"
        audit_path = Path(path).parent / f"{arm}_train_audit.json"
        if (record.get("result") != "PASS" or record.get("returncode") != 0
                or checkpoint != expected or train.sha256(checkpoint) != record["final_checkpoint_sha256"]
                or train.sha256(audit_path) != record["audit_sha256"]):
            raise ValueError("Formal training artifacts changed or final checkpoint was substituted")
        audits[arm] = json.loads(audit_path.read_text())
        if (audits[arm].get("arm"), audits[arm].get("dataset_manifest_sha256")) != (
                arm, frozen["dataset_manifest_sha256"]):
            raise ValueError("Training audit belongs to a different arm/dataset")
        train.check_cohorts(audits[arm], 100, manifest["ordered_origins"])
        if train.verify_update(checkpoint, initial, 100) != record["parameter_update_verification"]:
            raise ValueError("Parameter/Adam proof differs from actual final weights")
        checkpoints[arm] = checkpoint
    if [row["motion_steps"] for row in audits["a"]["updates"]] != [
            row["motion_steps"] for row in audits["b"]["updates"]]:
        raise ValueError("Formal A/B cohorts differ")
    return proof, checkpoints


def validate_development_evidence(report, job, fingerprint, expected_frames):
    from scripts.compare_mask_evaluations import G1_BFM_MODE_PRESETS, _normalize_motions
    mode, bodies = G1_BFM_MODE_PRESETS[job["mode"]]
    expected = {"schema_version": 4, "mode_index": job["mode"], "mode": mode,
                "active_body_names": list(bodies), "task": "G1-BFM-Transformer-Tracking",
                "num_envs": 64, "max_steps": 1000, "seed": 42, "device": "cuda:0", "step_dt": .02,
                "scene_variant": "baseline", "target_object": None, "checkpoint_sha256": fingerprint,
                "input_manifest_verified_unchanged": True}
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError("Development job differs from the fixed paired evaluation protocol")
    protocol = report.get("protocol", {})
    if type(protocol.get("training_updates")) is not int or protocol["training_updates"] != 0:
        raise ValueError("Evaluation must not train")
    if any(protocol.get(key) is not False for key in ("reset_disturbance", "observation_noise", "interval_pushes")):
        raise ValueError("Development evaluation must disable noise and disturbances")
    rows = _normalize_motions(report, job["name"], 1000)
    if len(rows) != 10 or {row["motion"] for row in rows.values()} != set(expected_frames):
        raise ValueError("Incomplete ten-origin development coverage")
    for row in rows.values():
        frames = expected_frames[row["motion"]]
        if (row["source_frames"], row["evaluated_steps"], row["truncated"]) != (frames, frames - 1, False):
            raise ValueError("Development evaluation did not cover its complete paired short window")
    summary = {"num_motions": 10, "evaluated_steps": sum(expected_frames.values()) - 10, "truncated_motions": 0}
    if any(report.get("summary", {}).get(key) != value for key, value in summary.items()):
        raise ValueError("Development summary differs from its actual rows")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--training-status", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--suite", choices=("heldout", "development", "all"), default="all")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,50}", args.run_id):
        parser.error("Invalid run-id")
    dataset = args.dataset.absolute()
    output = ROOT / "logs/behavior_learning" / f"umr_behavior_{args.run_id}"
    # A command plan may inspect JSON, but never starts Kit or hashes all motions.
    manifest = json.loads(dataset.read_text())
    proof = json.loads(args.training_status.read_text())
    checkpoints = {"official": train.OFFICIAL,
                   **{arm: Path(proof["arms"][arm]["final_checkpoint"]) for arm in ("a", "b")}}
    jobs = build_jobs(sys.executable, output, manifest, checkpoints, args.suite)
    if not args.execute:
        print(json.dumps({"output": str(output), "jobs": jobs, "training_updates": 0,
                          "automatic_promotion": False}, indent=2))
        return
    service = train.bounded_unit(args.run_id)
    if Path(sys.executable).absolute() != train.ISAAC_PYTHON:
        parser.error(f"Use the existing Isaac interpreter: {train.ISAAC_PYTHON}")
    train.new_directory(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    status = {"schema": SCHEMA, "result": "RUNNING", "stage": "preflight", "service": service,
              "jobs_requested": len(jobs), "jobs": [], "training_updates": 0,
              "automatic_promotion": False, "quality_result": "NOT_EVALUATED",
              "training_proof": {"path": str(args.training_status.resolve()), "sha256": train.sha256(args.training_status)},
              "limitations": "Reference tracking only; no contact-task/hardware or independent final-test claim."}
    previous = signal.getsignal(signal.SIGTERM)
    def terminate(signum, frame):
        raise KeyboardInterrupt("Owned evaluation service stopped")
    signal.signal(signal.SIGTERM, terminate)
    def save():
        status["elapsed_seconds"] = time.monotonic() - started
        train.atomic_json(output / "status.json", status)
    save()
    try:
        from scripts.build_umr_behavior_ab import validate_manifest
        manifest = validate_manifest(dataset)
        proof, checkpoints = validate_training_proof(args.training_status, manifest, dataset)
        jobs = build_jobs(sys.executable, output, manifest, checkpoints, args.suite)
        from evaluation_manifest import snapshot_inputs, verify_unchanged
        from scripts.compare_umr_behavior_ab import compare_ab, compare_development
        roots = train.runtime_roots()
        inputs, frames = {}, {}
        fingerprints = {label: train.sha256(path) for label, path in checkpoints.items()}
        import numpy as np
        for job in jobs:
            key = f"{job['reference']}/{job['label']}"
            if key in inputs:
                continue
            inputs[key] = snapshot_inputs(checkpoints[job["label"]], job["index"], roots)
            if job["reference"].startswith("development") and job["reference"] not in frames:
                frames[job["reference"]] = {}
                for name, record in inputs[key]["motions"]["files"].items():
                    with np.load(record["path"], allow_pickle=False) as archive:
                        frames[job["reference"]][name] = int(archive["joint_pos"].shape[0])
        train.atomic_json(output / "inputs.json", inputs)
        status.update(dataset_manifest_sha256=train.sha256(dataset), checkpoint_sha256=fingerprints)
        for job in jobs:
            key = f"{job['reference']}/{job['label']}"
            status["stage"] = job["name"]
            record = {**job, "result": "RUNNING"}
            status["jobs"].append(record)
            save()
            print(f"[UMR EVAL] START {job['name']} ({len(status['jobs'])}/{len(jobs)})", flush=True)
            record["returncode"] = train.run_child(job["command"], output / f"{job['name']}.log", job["timeout_seconds"])
            if record["returncode"]:
                raise RuntimeError(f"{job['name']} exited {record['returncode']}")
            path = output / f"{job['name']}.json"
            report = json.loads(path.read_text())
            verify_unchanged(inputs[key], report["input_manifest"])
            if job["reference"] == "heldout":
                record["summary"] = validate_evaluation_evidence(
                    report, job["label"], job["mode"], fingerprints[job["label"]], inputs[key]["motions"]["files"])
            else:
                record["summary"] = validate_development_evidence(report, job, fingerprints[job["label"]], frames[job["reference"]])
            record.update(result="PASS", report_sha256=train.sha256(path))
            save()
            print(f"[UMR EVAL] COMPLETE {job['name']}", flush=True)
        def reports(reference, label):
            return [json.loads((output / f"{reference}_{label}_{mode}.json").read_text()) for mode in range(8)]
        status["stage"] = "comparison"
        save()
        comparisons = {}
        if args.suite in ("heldout", "all"):
            comparisons["heldout"] = compare_ab(
                reports("heldout", "official"), reports("heldout", "a"), reports("heldout", "b"),
                expected_checkpoints=fingerprints, expected_motions=inputs["heldout/official"]["motions"]["files"], seed=42)
        if args.suite in ("development", "all"):
            for side in ("baseline", "candidate"):
                reference = f"development_{side}"
                comparisons[reference] = compare_development(
                    reports(reference, "a"), reports(reference, "b"),
                    expected_checkpoints={label: fingerprints[label] for label in ("a", "b")},
                    expected_motions=inputs[f"{reference}/a"]["motions"]["files"],
                    geometric_pass_origins=manifest["development"]["geometric_pass_origins"], seed=42)
        status["comparisons"] = {}
        for name, comparison in comparisons.items():
            path = output / f"{name}_comparison.json"
            train.atomic_json(path, comparison)
            status["comparisons"][name] = {"path": str(path), "sha256": train.sha256(path)}
        if validate_manifest(dataset) != manifest or proof["frozen_inputs"] != train.freeze_inputs(dataset, manifest, roots):
            raise ValueError("Training data/runtime changed during evaluation")
        for key, before in inputs.items():
            label = key.rsplit("/", 1)[1]
            verify_unchanged(before, snapshot_inputs(checkpoints[label], before["motion_index"]["path"], roots))
        if train.sha256(args.training_status) != status["training_proof"]["sha256"]:
            raise ValueError("Training proof changed during evaluation")
        for record in status["jobs"]:
            if train.sha256(output / f"{record['name']}.json") != record["report_sha256"]:
                raise ValueError("Evaluated report changed before finalization")
        status.update(result="COMPLETE", stage="complete", inputs_verified_unchanged=True,
                      quality_result=comparisons["heldout"]["quality_result"] if "heldout" in comparisons else "DEVELOPMENT_DIAGNOSTICS_ONLY",
                      confirmation_quality_condition_met=comparisons.get("heldout", {}).get("confirmation_quality_condition_met"),
                      independent_seed43_44_confirmation_complete=False)
        print("[UMR EVAL] COMPLETE; inspect the comparison gates; no automatic promotion", flush=True)
    except BaseException as error:
        if status["jobs"] and status["jobs"][-1]["result"] == "RUNNING":
            status["jobs"][-1].update(result="FAIL", error=str(error))
        status.update(result="FAIL", error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        save()
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    main()
