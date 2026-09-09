#!/usr/bin/env python3
"""Owned coverage smoke / long training, without automatic model promotion."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

from run_bfm_learning_ablation import (
    ROOT, TRACK, ENTRY, RUNS, TRAIN, LEGACY, KIT, OFFICIAL, DERIVED,
    OFFICIAL_SHA, DERIVED_SHA, same_tree, validate_split, verify_update,
)


def training_command(python, run_id, output, updates):
    return [python, "-u", str(ENTRY / "training_probe.py"),
            "--report", str(output / "train_audit.json"),
            "--progress", str(output / "progress.json"),
            "--expected_schedule", "fixed", "--expected_updates", str(updates), "--expected_lr", "1e-5",
            "--expected_save_interval", "50", "--expected_sampling_strategy", "coverage",
            "--expected_resample_interval", "5", "--",
            "--task", "G1-BFM-Transformer-Tracking", "--motion_file", str(TRAIN),
            "--run_name", run_id, "--logger", "tensorboard", "--num_envs", "128",
            "--max_iterations", str(updates), "--seed", "42", "--resume", "True",
            "--load_run", "official_lr1e5_derived_20260907", "--checkpoint", "model_22200.pt",
            "--device", "cuda:0", "--headless", "agent.algorithm.entropy_coef=0.001",
            "agent.algorithm.schedule=fixed", "agent.eval_during_training=False", "agent.save_interval=50",
            "agent.motion_sampling_strategy=coverage", "agent.motion_resample_interval=5"]


def validate_coverage_audit(audit, updates, indexed_names):
    if (audit.get("result"), audit.get("completed_updates"), audit.get("completed_environment_steps")) != (
        "PASS", updates, updates * 8192
    ):
        raise ValueError("Training did not finish all requested real updates/steps")
    if audit["resolved"]["motion_sampling_strategy"] != "coverage" or audit["resolved"]["motion_resample_interval"] != 5:
        raise ValueError("Coverage schedule differs from protocol")
    if len(audit["updates"]) != updates or any(row["completed_environment_steps"] != (i + 1) * 8192
                                              for i, row in enumerate(audit["updates"])):
        raise ValueError("Expected full 8192-step rollout windows")
    first = set(audit["updates"][0]["motion_steps"])
    second = set(audit["updates"][5]["motion_steps"])
    if len(first) != 128 or len(second) != 128 or first & second:
        raise ValueError("The first two real training cohorts were not disjoint 128-clip batches")
    if any(set(row["motion_steps"]) != first for row in audit["updates"][:5]):
        raise ValueError("Motion IDs changed inside a five-update cohort")
    actual = set(audit["motion_steps"])
    if not actual <= indexed_names or sum(audit["motion_steps"].values()) != updates * 8192:
        raise ValueError("Invalid actual training-motion coverage")
    if updates >= 281 and actual != indexed_names:
        raise ValueError("The long run did not actually sample every eligible training motion")
    return {"real_cohort_boundary_verified": True, "actual_unique_motions": len(actual),
            "actual_all_training_motions_sampled": actual == indexed_names}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--updates", type=int, choices=(6, 1000), default=1000)
    parser.add_argument("--smoke-status", type=Path, help="Completed same-source six-update coverage proof")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,70}", args.run_id):
        parser.error("Invalid run-id")
    output = ROOT / "logs/behavior_learning" / args.run_id
    command = training_command(sys.executable, args.run_id, output, args.updates)
    if not args.execute:
        print(json.dumps({"output": str(output), "command": command}, indent=2))
        return
    if f"bfm-{args.run_id}.service" not in Path("/proc/self/cgroup").read_text():
        parser.error(f"Run inside owned bfm-{args.run_id}.service with KillMode=control-group")
    for path in (output, RUNS / args.run_id):
        if path.exists() or path.is_symlink():
            parser.error(f"Refusing existing output/run: {path}")
    if args.updates == 1000 and (args.smoke_status is None or not args.smoke_status.is_file()):
        parser.error("Long training requires an existing successful coverage smoke status")
    sys.path.insert(0, str(ENTRY))
    from evaluation_manifest import locate_package_sources, sha256_file, snapshot_inputs, verify_unchanged
    import torch
    import yaml

    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    status = {"schema": 1, "result": "RUNNING", "stage": "preflight", "updates_requested": args.updates,
              "command": command, "automatic_promotion": False, "validation_status": "not evaluated",
              "purpose": "behavior-learning coverage experiment, not a trained-capability claim"}

    def save_status():
        status["elapsed_seconds"] = time.monotonic() - started
        temporary = output / "status.json.tmp"
        temporary.write_text(json.dumps(status, indent=2, allow_nan=False) + "\n")
        temporary.replace(output / "status.json")

    save_status()
    try:
        if sha256_file(OFFICIAL) != OFFICIAL_SHA or sha256_file(DERIVED) != DERIVED_SHA:
            raise ValueError("Approved official/derived checkpoint fingerprint changed")
        initial = torch.load(DERIVED, map_location="cpu", weights_only=False)
        official = torch.load(OFFICIAL, map_location="cpu", weights_only=False)
        for side in ("actor", "critic"):
            key = f"{side}_optimizer_state_dict"
            if len(initial[key]["param_groups"]) != 1 or initial[key]["param_groups"][0]["lr"] != 1e-5:
                raise ValueError("Incorrect derived initial Adam LR")
            official[key]["param_groups"][0]["lr"] = 1e-5  # Compare only, never save/overwrite source.
        if not same_tree(official, initial):
            raise ValueError("Derived source differs beyond the two intended LR fields")
        indexes = [yaml.safe_load(path.read_text()) for path in (TRAIN, LEGACY, KIT)]
        validate_split(*indexes)
        index_hashes = {str(path): sha256_file(path) for path in (TRAIN, LEGACY, KIT)}
        roots = {"entrypoints": ENTRY, "experiment_tools": ROOT / "scripts"}
        roots.update(locate_package_sources(("scaletrack", "my_rsl_rl", "isaaclab", "isaaclab_rl", "isaaclab_tasks")))
        manifest = snapshot_inputs(DERIVED, TRAIN, roots)
        with (output / "inputs.json").open("x") as stream:
            json.dump(manifest, stream, indent=2)
        status["input_fingerprints"] = {key: manifest[key]["sha256"] for key in manifest}
        status["original_index_fingerprints"] = index_hashes
        if args.updates == 1000:
            proof = json.loads(args.smoke_status.read_text())
            if (proof.get("result"), proof.get("updates_requested"), proof.get("inputs_verified_unchanged")) != (
                "COMPLETE", 6, True
            ) or not proof.get("coverage_verification", {}).get("real_cohort_boundary_verified"):
                raise ValueError("Smoke proof is not a complete real coverage-boundary validation")
            if proof.get("input_fingerprints") != status["input_fingerprints"]:
                raise ValueError("Smoke input/source fingerprint differs from this long run")
            status["smoke_proof"] = str(args.smoke_status.resolve())
        status.update(stage="training", training_pool=7174, num_envs=128, steps_per_update=8192,
                      coverage_interval=5, learning_rate=1e-5, checkpoint_run=str(RUNS / args.run_id))
        save_status()
        print(f"[BFM LONG] START {args.run_id}: {args.updates} updates, coverage every 5, fixed LR=1e-5", flush=True)
        with (output / "train.log").open("x") as log:
            child = subprocess.run(command, cwd=TRACK, stdout=log, stderr=subprocess.STDOUT,
                                   timeout=300 if args.updates == 6 else 9000)
        status["training_returncode"] = child.returncode
        if child.returncode:
            raise RuntimeError(f"Training exited {child.returncode}; inspect train.log/progress.json")
        audit = json.loads((output / "train_audit.json").read_text())
        status["coverage_verification"] = validate_coverage_audit(audit, args.updates, set(indexes[0]))
        checkpoint = RUNS / args.run_id / f"model_{initial['iter'] + args.updates - 1}.pt"
        status["parameter_update_verification"] = verify_update(checkpoint, initial, args.updates)
        status["final_checkpoint"] = str(checkpoint)
        status["final_checkpoint_sha256"] = sha256_file(checkpoint)
        verify_unchanged(manifest, snapshot_inputs(DERIVED, TRAIN, roots))
        if any(sha256_file(path) != digest for path, digest in index_hashes.items()):
            raise RuntimeError("Original training or validation index changed")
        status.update(result="COMPLETE", stage="complete", inputs_verified_unchanged=True)
        print(f"[BFM LONG] COMPLETE {args.run_id}; no automatic model promotion", flush=True)
    except BaseException as error:
        status.update(result="FAIL", error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        save_status()


if __name__ == "__main__":
    main()
