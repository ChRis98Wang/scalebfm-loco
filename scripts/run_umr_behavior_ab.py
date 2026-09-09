#!/usr/bin/env python3
"""Finite matched-reference PPO A/B controller; never promotes a checkpoint.

The default is a read-only command plan. Execution requires an owned, bounded
user service. A successful two-arm six-update smoke with identical frozen
inputs is mandatory before either 100-update arm starts.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from scripts.run_bfm_learning_ablation import (
    TRACK, ENTRY, RUNS, OFFICIAL, DERIVED, OFFICIAL_SHA, DERIVED_SHA,
    same_tree, verify_update,
)

SCHEMA = "bfm.umr_behavior_ab_training/1"
PROBE = ROOT / "scripts/umr_behavior_training_probe.py"
PROTOCOL = ROOT / "docs/UMR_BEHAVIOR_AB_PROTOCOL_20260909.md"
ISAAC_PYTHON = Path("/home/sw/isaaclab_ws/env_isaaclab_sim6_newton/bin/python")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def bounded_unit(run_id):
    unit = f"bfm-umr-behavior-{run_id}.service"
    if not re.search(r"(?:^|/)" + re.escape(unit) + r"(?:/|$)",
                     Path("/proc/self/cgroup").read_text(), re.MULTILINE):
        raise ValueError(f"Execute inside {unit}")
    values = dict(line.split("=", 1) for line in subprocess.check_output([
        "systemctl", "--user", "show", unit, "-p", "KillMode", "-p", "Restart",
        "-p", "RuntimeMaxUSec", "-p", "MemoryMax", "-p", "TasksMax"],
        text=True, timeout=10).splitlines())
    if (values.get("KillMode") != "control-group" or values.get("Restart") != "no"
            or any(values.get(key) in (None, "", "0", "infinity")
                   for key in ("RuntimeMaxUSec", "MemoryMax", "TasksMax"))):
        raise ValueError("Finite time/memory/tasks and control-group cleanup are required")
    return {"unit": unit, **values}


def new_directory(path):
    path = Path(path).absolute()
    if path.exists() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError(f"Output must be new, without symlink ancestors: {path}")
    return path


def build_jobs(python, dataset, run_id, output, updates):
    if updates not in (6, 100):
        raise ValueError("Only the frozen 6/100-update budgets are allowed")
    return [{"arm": arm, "run_name": f"umr_behavior_{run_id}_{arm}",
             "timeout_seconds": 600 if updates == 6 else 1800,
             "command": [str(python), "-u", str(PROBE), "--dataset", str(dataset),
                         "--arm", arm, "--run-name", f"umr_behavior_{run_id}_{arm}",
                         "--updates", str(updates), "--report", str(Path(output) / f"{arm}_train_audit.json"),
                         "--progress", str(Path(output) / f"{arm}_progress.json")]} for arm in ("a", "b")]


def run_child(command, log_path, timeout):
    """Reap our own process group on timeout, signal, or any exception."""
    with Path(log_path).open("x") as log:
        child = subprocess.Popen(command, cwd=TRACK, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        try:
            return child.wait(timeout=timeout)
        finally:
            # The session was created above, so no unrelated processes share it.
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait(timeout=10)


def check_cohorts(audit, updates, origins):
    """Recompute actual use, never count only sampler assignment events."""
    if (audit.get("result"), audit.get("completed_updates"), audit.get("completed_environment_steps")) != (
            "PASS", updates, updates * 8192):
        raise ValueError("Incomplete real PPO updates / environment steps")
    rows = audit.get("updates", [])
    if len(rows) != updates or len(origins) != 256 or len(set(origins)) != 256:
        raise ValueError("Expected 256 unique origins and a complete update audit")
    cohorts, totals = [], dict.fromkeys(origins, 0)
    for i, row in enumerate(rows):
        actual = row.get("motion_steps", {})
        if (len(actual) != 128 or set(actual) - set(origins)
                or any(type(v) is not int or v != 64 for v in actual.values())
                or row.get("completed_environment_steps") != (i + 1) * 8192):
            raise ValueError("A rollout must use 128 declared origins for exactly 64 steps each")
        cohort = sorted(actual)
        if i % 5 == 0:
            cohorts.append(cohort)
        elif cohort != cohorts[-1]:
            raise ValueError("Origins changed within a five-update cohort")
        for origin, count in actual.items():
            totals[origin] += count
    for i in range(0, len(cohorts), 2):
        if i + 1 < len(cohorts) and set(cohorts[i]) & set(cohorts[i + 1]):
            raise ValueError("The two cohorts of a data epoch overlap")
    if totals != audit.get("motion_steps") or any(count <= 0 for count in totals.values()):
        raise ValueError("Actual per-origin coverage differs from the audit")
    if updates == 100 and set(totals.values()) != {3200}:
        raise ValueError("Formal training must give each origin exactly 3200 steps")
    for field in ("mode_steps", "dataset_steps"):
        counts = audit.get(field, {})
        if (not counts or any(type(n) is not int or n <= 0 for n in counts.values())
                or sum(counts.values()) != updates * 8192):
            raise ValueError(f"Invalid completed {field}")
    if len(audit["mode_steps"]) != 8 or audit["dataset_steps"].get("KIT", 0) <= 0:
        raise ValueError("All eight masks and KIT must actually be used")
    return {"cohorts": cohorts, "actual_unique_origins": 256,
            "all_origins_used": True, "per_origin_steps": totals}


def validate_smoke(proof, current, directory):
    if (proof.get("schema"), proof.get("result"), proof.get("updates_requested"),
            proof.get("inputs_verified_unchanged"), proof.get("matched_cohorts")) != (
            SCHEMA, "COMPLETE", 6, True, True):
        raise ValueError("Formal A/B requires a complete, matched two-arm six-update smoke")
    if proof.get("frozen_inputs") != current:
        raise ValueError("Smoke inputs/runtime differ from this formal experiment")
    if set(proof.get("arms", {})) != {"a", "b"}:
        raise ValueError("Both smoke arms are required")
    audits = {}
    for arm, record in proof["arms"].items():
        audit_path = Path(directory) / f"{arm}_train_audit.json"
        if (record.get("result") != "PASS" or record.get("returncode") != 0
                or sha256(audit_path) != record.get("audit_sha256")
                or sha256(record["final_checkpoint"]) != record.get("final_checkpoint_sha256")):
            raise ValueError("Smoke artifacts are missing or changed")
        audits[arm] = json.loads(audit_path.read_text())
        check_cohorts(audits[arm], 6, current["ordered_origins"])
    if [row["motion_steps"] for row in audits["a"]["updates"]] != [
            row["motion_steps"] for row in audits["b"]["updates"]]:
        raise ValueError("The two smoke arms used different real cohorts")


def runtime_roots():
    sys.path.insert(0, str(ENTRY))
    from evaluation_manifest import locate_package_sources
    return {"entrypoints": ENTRY, **locate_package_sources(
        ("scaletrack", "my_rsl_rl", "isaaclab", "isaaclab_rl", "isaaclab_tasks"))}


def freeze_inputs(dataset, manifest, roots):
    from evaluation_manifest import snapshot_inputs
    selected = [Path(__file__), PROBE, ROOT / "scripts/build_umr_behavior_ab.py",
                ROOT / "scripts/run_bfm_learning_ablation.py", PROTOCOL,
                ROOT / "scripts/evaluate_umr_behavior_ab.py", ROOT / "scripts/compare_umr_behavior_ab.py",
                ROOT / "scripts/run_bfm_long_evaluation.py", ROOT / "scripts/compare_mask_evaluations.py"]
    assets = TRACK / "source/scaletrack/scaletrack/assets/robots/g1_29dof"
    selected.extend(path for path in assets.rglob("*") if path.is_file())
    return {"dataset_manifest": str(Path(dataset).resolve()), "dataset_manifest_sha256": sha256(dataset),
            "ordered_origins": manifest["ordered_origins"],
            "arms": {arm: snapshot_inputs(DERIVED, manifest["arms"][arm]["index"], roots)
                     for arm in ("a", "b")},
            "files": {str(path.resolve()): sha256(path) for path in selected},
            "python": {"executable": sys.executable, "version": sys.version,
                       "packages": {name: importlib.metadata.version(name)
                                    for name in ("torch", "numpy", "tensordict", "hydra-core", "isaacsim")}},
            "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,name,driver_version",
                                             "--format=csv,noheader"], text=True, timeout=15).strip()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", choices=("smoke", "train"), default="smoke")
    parser.add_argument("--smoke-status", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,50}", args.run_id):
        parser.error("Invalid run-id")
    updates = 6 if args.stage == "smoke" else 100
    output = ROOT / "logs/behavior_learning" / f"umr_behavior_{args.run_id}"
    dataset = args.dataset.absolute()
    jobs = build_jobs(sys.executable, dataset, args.run_id, output, updates)
    if not args.execute:
        print(json.dumps({"output": str(output), "updates_per_arm": updates,
                          "jobs": jobs, "automatic_promotion": False}, indent=2))
        return
    service = bounded_unit(args.run_id)
    if Path(sys.executable).absolute() != ISAAC_PYTHON:
        parser.error(f"Use the existing Isaac interpreter: {ISAAC_PYTHON}")
    new_directory(output)
    for job in jobs:
        new_directory(RUNS / job["run_name"])
    if updates == 100 and (args.smoke_status is None or not args.smoke_status.is_file()):
        parser.error("Formal training requires --smoke-status from both six-update arms")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    status = {"schema": SCHEMA, "result": "RUNNING", "stage": "preflight",
              "run_id": args.run_id, "updates_requested": updates, "arms": {}, "service": service,
              "automatic_promotion": False, "quality_result": "NOT_EVALUATED", "seed": 42,
              "purpose": "matched-reference learning experiment, not task-capability evidence"}
    previous = signal.getsignal(signal.SIGTERM)
    def terminate(signum, frame):
        raise KeyboardInterrupt("Owned A/B service stopped")
    signal.signal(signal.SIGTERM, terminate)
    def save():
        status["elapsed_seconds"] = time.monotonic() - started
        atomic_json(output / "status.json", status)
    save()
    try:
        from scripts.build_umr_behavior_ab import validate_manifest
        manifest = validate_manifest(dataset)
        if sha256(OFFICIAL) != OFFICIAL_SHA or sha256(DERIVED) != DERIVED_SHA:
            raise ValueError("Common official/derived checkpoint identity changed")
        import torch
        torch.set_num_threads(2)
        initial = torch.load(DERIVED, map_location="cpu", weights_only=False)
        official = torch.load(OFFICIAL, map_location="cpu", weights_only=False)
        for side in ("actor", "critic"):
            key = f"{side}_optimizer_state_dict"
            if len(initial[key]["param_groups"]) != 1 or initial[key]["param_groups"][0]["lr"] != 1e-5:
                raise ValueError("Derived initial optimizer LR is not 1e-5")
            official[key]["param_groups"][0]["lr"] = 1e-5
        if not same_tree(official, initial) or initial["iter"] != 22199:
            raise ValueError("Common initial state differs beyond the two approved LR fields")
        del official
        roots = runtime_roots()
        frozen = freeze_inputs(dataset, manifest, roots)
        atomic_json(output / "inputs.json", frozen)
        status["frozen_inputs"] = frozen
        if updates == 100:
            proof = json.loads(args.smoke_status.read_text())
            validate_smoke(proof, frozen, args.smoke_status.parent)
            status["smoke_proof"] = {"path": str(args.smoke_status.resolve()), "sha256": sha256(args.smoke_status)}
        audits = {}
        for job in jobs:
            arm = job["arm"]
            status["stage"] = f"train_{arm}"
            record = {**job, "result": "RUNNING"}
            status["arms"][arm] = record
            save()
            print(f"[UMR A/B] START arm={arm} updates={updates}", flush=True)
            record["returncode"] = run_child(job["command"], output / f"{arm}_train.log", job["timeout_seconds"])
            if record["returncode"] != 0:
                raise RuntimeError(f"Arm {arm} exited {record['returncode']}; inspect its log")
            audit_path = output / f"{arm}_train_audit.json"
            audit = json.loads(audit_path.read_text())
            if (audit.get("schema"), audit.get("arm"), audit.get("dataset_manifest_sha256")) != (
                    "bfm.umr_behavior_training_probe/1", arm, frozen["dataset_manifest_sha256"]):
                raise ValueError("Training audit does not bind this arm and frozen dataset")
            record["coverage"] = check_cohorts(audit, updates, manifest["ordered_origins"])
            checkpoint = RUNS / job["run_name"] / f"model_{initial['iter'] + updates - 1}.pt"
            record.update(parameter_update_verification=verify_update(checkpoint, initial, updates),
                          final_checkpoint=str(checkpoint), final_checkpoint_sha256=sha256(checkpoint),
                          audit_sha256=sha256(audit_path))
            if validate_manifest(dataset) != manifest or freeze_inputs(dataset, manifest, roots) != frozen:
                raise ValueError("Frozen data/runtime changed during training")
            record["result"] = "PASS"
            audits[arm] = audit
            save()
            print(f"[UMR A/B] COMPLETE arm={arm}; optimizer and 256-origin coverage verified", flush=True)
        if [row["motion_steps"] for row in audits["a"]["updates"]] != [
                row["motion_steps"] for row in audits["b"]["updates"]]:
            raise ValueError("A/B used different actual origin cohorts")
        if updates == 100 and sha256(args.smoke_status) != status["smoke_proof"]["sha256"]:
            raise ValueError("Smoke proof changed during formal training")
        status.update(result="COMPLETE", stage="complete", matched_cohorts=True,
                      inputs_verified_unchanged=True, completed_environment_steps=updates * 8192 * 2)
        print("[UMR A/B] COMPLETE; quality not evaluated; no automatic promotion", flush=True)
    except BaseException as error:
        for record in status["arms"].values():
            if record["result"] == "RUNNING":
                record.update(result="FAIL", error=str(error))
        status.update(result="FAIL", error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        save()
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    main()
