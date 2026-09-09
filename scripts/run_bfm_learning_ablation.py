#!/usr/bin/env python3
"""Bounded, train-only-mask BFM LR pilot followed by paired held-out evaluation.

Run with the existing IsaacLab interpreter inside an owned systemd user cgroup.
No GUI, online reference generator, dataset mutation, or checkpoint promotion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
TRACK = ROOT / "ScaleTrack"
ENTRY = TRACK / "scripts/pretrain/rsl_rl"
DATA = ROOT / "ScaleRetarget/retargeted_dataset"
RUNS = ROOT / "logs/rsl_rl/g1_bfm_tracking_exp"
OFFICIAL = RUNS / "humanoid_transformer_m/model_22200.pt"
DERIVED = RUNS / "official_lr1e5_derived_20260907/model_22200.pt"
TRAIN = DATA / "amass_full_v2_train.yaml"
LEGACY = DATA / "amass_full_v1_validation.yaml"
KIT = DATA / "amass_kit_heldout_v1.yaml"
OFFICIAL_SHA = "88d5a79946c03ed25503f48b2af71d16290844ef066ca9b6c8fa8dc3837422e3"
DERIVED_SHA = "269f17e040ad0c27f651a25097a2650380ff1a05714dde102625aabd8d327c46"


def same_tree(left, right):
    import torch
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            same_tree(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(
            same_tree(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def validate_split(train, legacy, kit):
    """Reject name/path overlap; file-content and near-duplicate audit is separate."""
    for name, value, size in (("train", train, 7174), ("legacy", legacy, 962), ("KIT", kit, 789)):
        if not isinstance(value, dict) or len(value) != size:
            raise ValueError(f"Expected {size} motions in {name}")
        if any(not isinstance(key, str) or not isinstance(path, str) for key, path in value.items()):
            raise ValueError("Motion names/paths must be strings")
    if any(not key.startswith("KIT/") for key in kit) or any(key.startswith("KIT/") for key in legacy):
        raise ValueError("Expected disjoint legacy and KIT strata")
    groups = (train, legacy, kit)
    for i in range(3):
        for j in range(i + 1, 3):
            if set(groups[i]) & set(groups[j]):
                raise ValueError("Train/held-out names overlap")
            paths_i = {str(Path(path).resolve()) for path in groups[i].values()}
            paths_j = {str(Path(path).resolve()) for path in groups[j].values()}
            if paths_i & paths_j:
                raise ValueError("Train/held-out payload paths overlap")
    return {**legacy, **kit}


def build_jobs(python, run_id, output):
    """Predefined pilot; no result-dependent candidate selection or early promotion."""
    output = Path(output)
    jobs = []
    for schedule in ("fixed", "adaptive"):
        command = [python, "-u", str(ENTRY / "training_probe.py"),
                   "--report", str(output / f"{schedule}_train.json"),
                   "--expected_schedule", schedule, "--expected_updates", "5", "--expected_lr", "1e-5", "--",
                   "--task", "G1-BFM-Transformer-Tracking", "--motion_file", str(TRAIN),
                   "--run_name", f"{run_id}_{schedule}", "--logger", "tensorboard",
                   "--num_envs", "128", "--max_iterations", "5", "--seed", "42",
                   "--resume", "True", "--load_run", "official_lr1e5_derived_20260907",
                   "--checkpoint", "model_22200.pt", "--device", "cuda:0", "--headless",
                   "agent.algorithm.entropy_coef=0.001", f"agent.algorithm.schedule={schedule}",
                   "agent.eval_during_training=False", "agent.save_interval=1"]
        jobs.append((f"train_{schedule}", 300, command))
    checkpoints = {"official": OFFICIAL, **{schedule: RUNS / f"{run_id}_{schedule}" / "model_22203.pt"
                                           for schedule in ("fixed", "adaptive")}}
    for mode in range(8):
        for label, checkpoint in checkpoints.items():
            command = [python, "-u", str(ENTRY / "evaluate.py"), "--checkpoint_path", str(checkpoint),
                       "--motion_file", str(output / "heldout_union.yaml"),
                       "--output", str(output / f"eval_{label}_{mode}.json"),
                       "--mode_index", str(mode), "--mask_metrics", "--num_envs", "1024",
                       "--max_steps", "1000", "--seed", "42", "--device", "cuda:0"]
            jobs.append((f"eval_{label}_{mode}", 240, command))
    return jobs


def verify_update(checkpoint, initial, updates=5):
    """Verify actual model and both Adam updates, not checkpoint filename alone."""
    import torch
    result = torch.load(checkpoint, map_location="cpu", weights_only=False)
    changed = []
    for name, tensor in result["model_state_dict"].items():
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"Nonfinite checkpoint tensor: {name}")
        if not torch.equal(tensor, initial["model_state_dict"][name]):
            changed.append(name)
    if not any(name.startswith("actor.") for name in changed):
        raise ValueError("Actor backbone did not update")
    increments = {}
    for side in ("actor", "critic"):
        key = f"{side}_optimizer_state_dict"
        old, new = initial[key]["state"], result[key]["state"]
        if new.keys() != old.keys():
            raise ValueError("Optimizer state identity changed")
        values = [float(new[key]["step"]) - float(old[key]["step"]) for key in old]
        if not values or set(values) != {float(updates * 64)}:
            raise ValueError(f"Expected {updates} x 64 Adam steps for {side}")
        increments[side] = {"states": len(values), "increment": updates * 64}
    if result["iter"] != initial["iter"] + updates - 1:
        raise ValueError("Unexpected final runner iteration")
    return {"changed_model_tensors": changed, "optimizer_steps": increments,
            "runner_iteration": result["iter"]}


def validate_training_evidence(report, schedule):
    """Exit 0 alone is insufficient, especially with Kit fast shutdown."""
    if (report.get("result"), report.get("completed_updates"), report.get("completed_environment_steps")) != (
        "PASS", 5, 40960
    ):
        raise ValueError("Training observer did not publish a complete successful pilot")
    if report["resolved"]["schedule"] != schedule or report["resolved"]["initial_learning_rate"] != 1e-5:
        raise ValueError("Training schedule or initial LR differs from the paired protocol")
    if len(report["mode_steps"]) != 8 or min(report["mode_steps"].values()) <= 0:
        raise ValueError("All eight masks must actually enter the training rollout")
    for key in ("mode_steps", "motion_steps", "dataset_steps"):
        if sum(report[key].values()) != 40960:
            raise ValueError(f"Inconsistent completed rollout counts: {key}")
    if report["dataset_steps"].get("KIT", 0) <= 0 or report["changed_actor_parameters"] <= 0:
        raise ValueError("KIT sampling and actual actor parameter updates are required")
    if len(report["updates"]) != 5 or [row["completed_environment_steps"] for row in report["updates"]] != [
        8192 * i for i in range(1, 6)
    ]:
        raise ValueError("Expected five complete equal-length rollout windows")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="New local identifier, letters/digits/underscores only")
    parser.add_argument("--execute", action="store_true", help="Without this flag, print the fixed job plan only")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,70}", args.run_id):
        parser.error("Invalid run-id")
    output = ROOT / "logs/behavior_learning" / args.run_id
    jobs = build_jobs(sys.executable, args.run_id, output)
    if not args.execute:
        print(json.dumps({"output": str(output), "jobs": jobs}, indent=2))
        return
    # The outer service owns children even if this controller crashes or times out.
    if not any(f"bfm-{args.run_id}.service" in line for line in Path("/proc/self/cgroup").read_text().splitlines()):
        parser.error(f"Execute inside the owned bfm-{args.run_id}.service user cgroup")
    for path in (output, RUNS / f"{args.run_id}_fixed", RUNS / f"{args.run_id}_adaptive"):
        if path.exists() or path.is_symlink():
            parser.error(f"Refusing existing output/run: {path}")
    sys.path.insert(0, str(ENTRY))
    from evaluation_manifest import locate_package_sources, sha256_file, snapshot_inputs, verify_unchanged
    import torch
    import yaml
    output.mkdir(parents=True, exist_ok=False)
    status = {"schema": 1, "result": "RUNNING", "stage": "preflight", "jobs": [],
              "run_id": args.run_id, "updates_per_candidate": 5, "automatic_promotion": False,
              "interpretation": "learning-rate pilot, not sufficient training or general behavior mastery"}
    started = time.monotonic()

    def write_status():
        status["elapsed_seconds"] = time.monotonic() - started
        temporary = output / "status.json.tmp"
        temporary.write_text(json.dumps(status, indent=2, allow_nan=False) + "\n")
        temporary.replace(output / "status.json")

    write_status()
    try:
        if sha256_file(OFFICIAL) != OFFICIAL_SHA or sha256_file(DERIVED) != DERIVED_SHA:
            raise ValueError("Official/derived input checkpoint fingerprint changed")
        official = torch.load(OFFICIAL, map_location="cpu", weights_only=False)
        initial = torch.load(DERIVED, map_location="cpu", weights_only=False)
        for side in ("actor", "critic"):
            key = f"{side}_optimizer_state_dict"
            if len(initial[key]["param_groups"]) != 1 or initial[key]["param_groups"][0]["lr"] != 1e-5:
                raise ValueError("Derived checkpoint must begin with both Adam rates exactly 1e-5")
            official[key]["param_groups"][0]["lr"] = 1e-5  # In-memory comparison only; never save it.
        if not same_tree(official, initial):
            raise ValueError("Derived input differs from official beyond two LR fields")
        indexes = [yaml.safe_load(path.read_text()) for path in (TRAIN, LEGACY, KIT)]
        benchmark = validate_split(*indexes)
        benchmark_path = output / "heldout_union.yaml"
        with benchmark_path.open("x") as stream:
            yaml.safe_dump(benchmark, stream, sort_keys=False)
        roots = {"entrypoints": ENTRY, "experiment_tools": ROOT / "scripts"}
        roots.update(locate_package_sources(("scaletrack", "my_rsl_rl", "isaaclab", "isaaclab_rl", "isaaclab_tasks")))
        manifests = {"train": snapshot_inputs(DERIVED, TRAIN, roots),
                     "evaluation": snapshot_inputs(OFFICIAL, benchmark_path, roots)}
        for stratum, paths in (("legacy", indexes[1]), ("KIT", indexes[2])):
            train_hashes = {v["sha256"] for v in manifests["train"]["motions"]["files"].values()}
            test_hashes = {manifests["evaluation"]["motions"]["files"][name]["sha256"] for name in paths}
            if train_hashes & test_hashes:
                raise ValueError(f"Exact train/{stratum} payload duplicates")
        status["input_index_fingerprints"] = {str(path): sha256_file(path) for path in (TRAIN, LEGACY, KIT)}
        with (output / "inputs.json").open("x") as stream:
            json.dump(manifests, stream, indent=2)
        status["split"] = {"train": 7174, "legacy_validation": 962, "KIT_holdout": 789,
                           "name_path_exact_payload_disjoint": True,
                           "near_duplicate_or_upstream_overlap": "not excluded"}
        for name, timeout, command in jobs:
            status["stage"] = name
            job = {"name": name, "command": command, "timeout_seconds": timeout, "result": "RUNNING"}
            status["jobs"].append(job)
            write_status()
            print(f"[BFM LEARNING] START {name}", flush=True)
            with (output / f"{name}.log").open("x") as log:
                child = subprocess.run(command, cwd=TRACK, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
            job.update(returncode=child.returncode, result="PASS" if child.returncode == 0 else "FAIL")
            if child.returncode:
                raise RuntimeError(f"{name} failed: see {output / (name + '.log')}")
            if name.startswith("train_"):
                schedule = name.removeprefix("train_")
                evidence = json.loads((output / f"{schedule}_train.json").read_text())
                validate_training_evidence(evidence, schedule)
                checkpoint = RUNS / f"{args.run_id}_{schedule}" / "model_22203.pt"
                job["checkpoint_sha256"] = sha256_file(checkpoint)
                job["learning_verified"] = verify_update(checkpoint, initial)
            else:
                label, mode = name.removeprefix("eval_").rsplit("_", 1)
                evidence = json.loads((output / f"eval_{label}_{mode}.json").read_text())
                summary = evidence["summary"]
                if (summary["num_motions"], summary["evaluated_steps"], summary["truncated_motions"]) != (
                    1751, 605664, 98
                ) or not evidence["input_manifest_verified_unchanged"]:
                    raise ValueError("Evaluation did not complete the declared held-out windows")
                job["evaluation_summary"] = {key: summary[key] for key in
                                             ("num_motions", "evaluated_steps", "truncated_motions")}
            write_status()
            print(f"[BFM LEARNING] COMPLETE {name}", flush=True)
        status["stage"] = "comparison"
        for reference, candidate in (("official", "fixed"), ("official", "adaptive"), ("adaptive", "fixed")):
            command = [sys.executable, str(ENTRY / "compare_learning_ablation.py"),
                       "--reference8", *[str(output / f"eval_{reference}_{mode}.json") for mode in range(8)],
                       "--candidate8", *[str(output / f"eval_{candidate}_{mode}.json") for mode in range(8)],
                       "--outputnew", str(output / f"compare_{reference}_vs_{candidate}.json")]
            # Quality FAIL remains an exit-0 completed comparison; parse its JSON.
            comparison = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=60)
            print(comparison.stdout, flush=True)
            if comparison.returncode != 0:
                raise RuntimeError(f"Comparison tool failed: {comparison.stderr}")
        for name, checkpoint, index in (("train", DERIVED, TRAIN), ("evaluation", OFFICIAL, benchmark_path)):
            verify_unchanged(manifests[name], snapshot_inputs(checkpoint, index, roots))
        if any(sha256_file(path) != digest for path, digest in status["input_index_fingerprints"].items()):
            raise RuntimeError("An original dataset index changed")
        status.update(result="COMPLETE", stage="complete", inputs_verified_unchanged=True)
    except BaseException as error:
        status.update(result="FAIL", error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        write_status()


if __name__ == "__main__":
    main()
