#!/usr/bin/env python3
"""Resume a verified, paused true-UMR pilot without changing frozen producers.

Default is read-only. Completed origins are hash/clock/cache checked, unfinished
attempts are moved into a new immutable history directory, never deleted. Actual
work requires a bounded, owned service. No training, filtering or promotion.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_umr_amass_pilot as producer
from scripts import evaluate_umr_pairs as evaluation
from scripts.summarize_umr_pairs import VerifiedFiles
from scripts.umr_pair_dataset import prepare_pair
from scripts.umr_smplx_source import UMR_COMMIT, sha256, verify_umr_checkout, validate_setup_cache_entry


def completed_prefix(plan, status):
    count = status.get("motions_completed")
    if (status.get("result") not in ("PAUSED_BY_USER", "FAIL") or type(count) is not int
            or not 0 <= count < 40 or status.get("motions_requested") != 40
            or status.get("training_started") is not False or status.get("automatic_promotion") is not False):
        raise ValueError("Expected an explicitly paused/failed, non-promoting partial pilot")
    states = status.get("motions", [])
    if len(states) not in (count, count + 1):
        raise ValueError("Status must contain only the completed prefix and at most one unfinished origin")
    if [s.get("origin_id") for s in states] != [r["origin_id"] for r in plan["rows"][:len(states)]]:
        raise ValueError("Status changed the preselected origin ordering")
    for state in states[:count]:
        jobs = state.get("jobs", [])
        if ((state.get("result"), state.get("stage")) != ("COMPLETE", "complete")
                or len(jobs) != 2 or [j.get("stage") for j in jobs] != ["prepare", "retarget"]
                or any(j.get("result") != "COMPLETE" for j in jobs)):
            raise ValueError("Completed prefix contains incomplete work")
    if len(states) > count and states[count].get("result") not in ("INTERRUPTED_BY_USER", "FAIL"):
        raise ValueError("Unfinished origin must be explicitly interrupted/failed, not still running")
    return count


def service_stopped(unit):
    if not re.fullmatch(r"bfm-umr-pilot-[A-Za-z0-9_.-]+\.service", unit or ""):
        raise ValueError("Unexpected previous service identity")
    values = subprocess.check_output(["systemctl", "--user", "show", unit,
                                      "-p", "MainPID", "-p", "ActiveState"], text=True)
    fields = dict(line.split("=", 1) for line in values.splitlines())
    if fields.get("MainPID") != "0" or fields.get("ActiveState") not in ("inactive", "failed"):
        raise ValueError("Previous pilot service is still alive")


def bounded_unit():
    unit = producer.owned_unit()
    text = subprocess.check_output(["systemctl", "--user", "show", unit, "-p", "Restart",
        "-p", "RuntimeMaxUSec", "-p", "MemoryMax", "-p", "TasksMax"], text=True)
    values = dict(line.split("=", 1) for line in text.splitlines())
    if values.get("Restart") != "no" or any(values.get(key) in (None, "", "infinity", "0")
            for key in ("RuntimeMaxUSec", "MemoryMax", "TasksMax")):
        raise ValueError("Resume service needs explicit finite time/memory/task bounds and Restart=no")
    return unit


def paired_row(source, batch, files):
    origin = source["origin_id"]
    directory = batch / "pairs" / origin
    receipt_path = directory / "pair_receipt.json"
    receipt = files.read_json(receipt_path)
    row = {key: source[key] for key in ("origin_id", "dataset", "split", "index_label", "source", "source_sha256")}
    row.update(pair_receipt=str(receipt_path), pair_receipt_sha256=files.verify(receipt_path),
               sampling_proof=str(directory / "sampling_proof.npz"),
               sampling_proof_sha256=files.verify(directory / "sampling_proof.npz"),
               expected_packed_frames=receipt["expected_packed_frames"])
    for side in ("baseline", "candidate"):
        path = batch / "input" / side / f"{origin}.pkl"
        if Path(receipt["outputs"][side]["path"]) != directory / f"{side}.pkl":
            raise ValueError("Completed pair output redirected outside its origin")
        digest = receipt["outputs"][side]["sha256"]
        files.verify(path, digest)
        row.update({f"{side}_pkl": str(path), f"{side}_pkl_sha256": digest})
        count, _ = evaluation._native(path, row["expected_packed_frames"])
        if count != receipt["frames"]:
            raise ValueError("Actual completed native motion frame count changed")
    snapshots = {}
    evaluation.validate_sampling_receipt(row, snapshots)
    for path, digest in snapshots.items():
        files.verify(path, digest)
    target = batch / "retargeted" / origin
    umr = files.read_json(target / "receipt.json")
    motion = target / "motion.npz"
    files.verify(motion, receipt["input_sha256"][str(motion)])
    with np.load(motion, allow_pickle=False) as archive:
        if json.loads(str(archive["metadata_json"])) != umr:
            raise ValueError("Completed UMR receipt differs from its hash-bound motion")
    if (umr.get("umr_commit") != UMR_COMMIT
            or umr.get("protected_inputs_rechecked") is not True
            or umr["source"]["source_sha256"] != source["source_sha256"]):
        raise ValueError("Completed UMR input binding changed")
    files.verify(batch / "surfaces" / f"{origin}.npz", umr["prepared_source_sha256"])
    files.verify(umr["scalebfm_output"]["path"], umr["scalebfm_output"]["sha256"])
    cache = umr["setup_cache"]
    entry = batch / "setup_cache" / cache["key_sha256"]
    if Path(cache["entry"]) != entry or cache.get("rechecked_after_retarget") is not True:
        raise ValueError("Completed origin refers to another/unverified canonical setup")
    cache_manifest = files.read_json(entry / "manifest.json", cache["manifest_sha256"])
    evidence = validate_setup_cache_entry(entry, cache_manifest["identity"])
    if any(cache.get(key) != value for key, value in evidence.items()):
        raise ValueError("Completed canonical setup changed")
    return row


def validate_resume(batch):
    batch = Path(batch)
    local = ROOT / "local"
    if batch.is_symlink() or not batch.resolve(strict=True).is_relative_to(local) or batch.resolve() == local:
        raise ValueError("Resume only an existing dedicated real directory below local/")
    batch = batch.resolve(strict=True)
    if (batch / "paired_manifest.json").exists():
        raise ValueError("A published complete batch must not be resumed")
    files = VerifiedFiles()
    plan = files.read_json(batch / "plan.json")
    status = files.read_json(batch / "status.json")
    if (plan.get("schema"), plan.get("experiment_id"), plan.get("training_started"), plan.get("window_seconds"),
            plan.get("points"), plan.get("epochs")) != (1, batch.name, False, 5., 4096, 2500):
        raise ValueError("Frozen pilot settings changed")
    if Path(plan["selection"]) != evaluation.ORIGINAL_SELECTION or plan["selection_sha256"] != evaluation.ORIGINAL_SELECTION_SHA:
        raise ValueError("Expected the original pinned forty-origin preselection")
    if not plan.get("protected_files"):
        raise ValueError("Missing original source/model/code protection hashes")
    for path, digest in plan["protected_files"].items():
        files.verify(path, digest)
    selection = producer.validate_selection(plan["selection"])
    if any(plan.get(key) != value for key, value in selection.items()):
        raise ValueError("Source/model/index identity differs from the original plan")
    if verify_umr_checkout(producer.UMR_ROOT) != plan["umr_commit"]:
        raise ValueError("Pinned UMR changed")
    service_stopped(status["unit"])
    count = completed_prefix(plan, status)
    rows = [paired_row(source, batch, files) for source in plan["rows"][:count]]
    # No completed origin can be absent, duplicated, or partly republished.
    expected = {batch / "input" / side / f"{r['origin_id']}.pkl" for r in plan["rows"] for side in ("baseline", "candidate")}
    if not set((batch / "input").rglob("*.pkl")).issubset(expected):
        raise ValueError("Unexpected native payload outside the preselected cohort")
    files.recheck()
    return batch, plan, status, rows, files


def owned_path(path, root):
    """Reject lexical escapes and every redirected ancestor, even within root."""
    path, root = Path(path).absolute(), Path(root).absolute()
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError("Artifact path is outside owned root") from error
    if not relative.parts or ".." in relative.parts:
        raise ValueError("Artifact must be a strict descendant of owned root")
    cursor = root
    for part in ("", *relative.parts):
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError("Refuse artifact path with a symlink ancestor")
    return path


def interrupted_paths(batch, plan, count):
    paths = []
    for number, row in enumerate(plan["rows"][count:], start=count):
        command = producer.command_plan(row, batch)
        candidates = [command["surface"], command["retargeted"], batch / "pairs" / row["origin_id"],
                      *(batch / "input" / side / f"{row['origin_id']}.pkl" for side in ("baseline", "candidate")),
                      *(batch / "logs" / f"{number:02d}_{stage}.log" for stage in ("prepare", "retarget"))]
        for path in candidates:
            owned_path(path, batch)
            if path.exists():
                if path.is_dir() and any(child.is_symlink() for child in path.rglob("*")):
                    raise ValueError("Refuse moving a redirected interrupted artifact tree")
                paths.append(path)
    cache_root = batch / "setup_cache"
    owned_path(cache_root, batch)
    if cache_root.exists() and not cache_root.is_dir():
        raise ValueError("Canonical cache root is not a directory")
    for path in sorted(cache_root.iterdir()) if cache_root.exists() else []:
        if re.fullmatch(r"\.[0-9a-f]{64}\.building-[A-Za-z0-9_-]+", path.name):
            owned_path(path, batch)
            if not path.is_dir() or any(child.is_symlink() for child in path.rglob("*")):
                raise ValueError("Invalid interrupted canonical setup directory")
            paths.append(path)
    return paths


def archive_attempt(paths, batch, history):
    owned_path(history, batch)
    # Validate the entire move set before the first mutation.
    for source in paths:
        owned_path(source, batch)
        target = history / "interrupted_artifacts" / source.relative_to(batch)
        owned_path(target, history)
        if target.exists() or target.is_symlink():
            raise FileExistsError(target)
    moves = []
    for source in paths:
        target = history / "interrupted_artifacts" / source.relative_to(batch)
        if target.exists() or target.is_symlink():
            raise FileExistsError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
        moves.append({"from": str(source), "to": str(target)})
    return moves


@contextlib.contextmanager
def resume_lock(batch):
    descriptor = os.open(batch / ".resume.lock", os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(descriptor)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", args.run_id):
        parser.error("Use a safe unique run ID")
    batch, plan, status, pairs, files = validate_resume(args.batch)
    count = len(pairs)
    history = batch / "resume_history" / args.run_id
    owned_path(history, batch)
    if history.exists() or history.is_symlink():
        parser.error("Resume history must be new")
    paths = interrupted_paths(batch, plan, count)
    if not args.execute:
        print(json.dumps({"plan_only": True, "verified_completed": count, "remaining": 40 - count,
            "files_verified": len(files.files), "archive_paths": [str(p) for p in paths],
            "history": str(history), "training_started": False}, indent=2))
        return 0
    unit = bounded_unit()
    with resume_lock(batch):
        files.recheck()
        service_stopped(status["unit"])
        history.mkdir(parents=True, exist_ok=False)
        for name in ("status.json", "plan.json"):
            with (history / f"before_{name}").open("xb") as stream:
                stream.write((batch / name).read_bytes())
        moves = archive_attempt(paths, batch, history)
        inputs = dict(plan["protected_files"])
        for path in (Path(__file__), ROOT / "scripts/evaluate_umr_pairs.py", ROOT / "scripts/summarize_umr_pairs.py",
                     batch / "plan.json", history / "before_status.json", history / "before_plan.json"):
            inputs[str(path.resolve())] = sha256(path)
        resume_receipt = {"schema": 1, "run_id": args.run_id, "unit": unit, "reused_origins": count,
            "original_plan_sha256": sha256(batch / "plan.json"), "archived_artifacts": moves,
            "protected_files": inputs, "training_started": False, "automatic_promotion": False}
        producer.save_json(history / "resume_plan.json", resume_receipt)
        inputs[str(history / "resume_plan.json")] = sha256(history / "resume_plan.json")
        status = copy.deepcopy(status)
        status.update(result="RUNNING", unit=unit, motions=status["motions"][:count],
                      resume_history=str(history), reused_origins=count)
        producer.save_json(batch / "status.json", status)
        started = time.monotonic()
        previous_handler = signal.getsignal(signal.SIGTERM)

        def interrupted(signum, frame):
            raise KeyboardInterrupt("Owned service stopped; preserve explicit paused state")

        signal.signal(signal.SIGTERM, interrupted)
        try:
            for number, source in enumerate(plan["rows"][count:], start=count):
                evaluation.verify_frozen(inputs)
                command = producer.command_plan(source, batch)
                state = {"origin_id": source["origin_id"], "result": "RUNNING", "stage": "prepare", "jobs": []}
                status["motions"].append(state)
                print(f"[UMR RESUME] {number + 1}/40 {source['origin_id']}", flush=True)
                for stage in ("prepare", "retarget"):
                    state["stage"] = stage
                    job = {"stage": stage, "command": command[stage], "timeout_seconds": 300, "result": "RUNNING"}
                    state["jobs"].append(job)
                    producer.save_json(batch / "status.json", status)
                    before = time.monotonic()
                    producer.run_owned_child(command[stage], batch / "logs" / f"{number:02d}_{stage}.log", 300)
                    job.update(result="COMPLETE", elapsed_seconds=time.monotonic() - before)
                state["stage"] = "pair"
                pair_dir = batch / "pairs" / source["origin_id"]
                receipt = prepare_pair(source, prepared_source=command["surface"],
                    candidate_npz=command["retargeted"] / "motion.npz",
                    packed_audit=ROOT / "local/data_refresh_20260908b/packed_audit.json", output_dir=pair_dir)
                pairs.append(producer.publish_pair_row(source, receipt, batch, pair_dir))
                state.update(result="COMPLETE", stage="complete")
                status["motions_completed"] += 1
                producer.save_json(batch / "status.json", status)
            evaluation.verify_frozen(inputs)
            if verify_umr_checkout(producer.UMR_ROOT) != plan["umr_commit"]:
                raise ValueError("UMR pin changed during resume")
            # Check reused receipts again, not just new process exit codes.
            checks = VerifiedFiles()
            checked = [paired_row(row, batch, checks) for row in plan["rows"]]
            if checked != pairs:
                raise ValueError("Complete paired outputs changed during resume")
            checks.recheck()
            manifest = {"schema": 1, "experiment_id": batch.name, "origin_indexes": plan["origin_indexes"],
                "rows": pairs, "selection": plan["selection"], "selection_sha256": plan["selection_sha256"],
                "protected_files": inputs, "training_started": False, "automatic_promotion": False,
                "interpretation": "same-origin pipeline comparison; different human shapes, not solver-only ablation"}
            producer.save_json(batch / "paired_manifest.json", manifest)
            status.update(result="COMPLETE", inputs_verified_unchanged=True,
                          manifest_sha256=sha256(batch / "paired_manifest.json"))
        except BaseException as error:
            stopped = isinstance(error, KeyboardInterrupt)
            status.update(result="PAUSED_BY_USER" if stopped else "FAIL", error=f"{type(error).__name__}: {error}")
            if status["motions"] and status["motions"][-1]["result"] == "RUNNING":
                last = status["motions"][-1]
                last["result"] = "INTERRUPTED_BY_USER" if stopped else "FAIL"
                for job in last["jobs"]:
                    if job["result"] == "RUNNING":
                        job["result"] = last["result"]
            raise
        finally:
            signal.signal(signal.SIGTERM, previous_handler)
            status["resume_elapsed_seconds"] = time.monotonic() - started
            producer.save_json(batch / "status.json", status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
