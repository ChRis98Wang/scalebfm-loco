#!/usr/bin/env python3
"""Produce a fixed, origin-preserving true SMPL-X/UMR paired data pilot.

Default: read-only plan. Execute only inside an owned bounded user service.
Existing data, policy weights and training indexes are never changed. A failed
motion stops the batch, not a silently filtered benchmark. No training/promotion.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import retarget_data_refresh as refresh
from scripts.umr_smplx_source import load_amass, model_file_for_gender, sha256

RETARGET_PYTHON = Path("/home/sw/.cache/bfm/scaleretarget-py311/bin/python")
UMR_PYTHON = ROOT / "local/umr_trial_20260908/venv/bin/python"
UMR_ROOT = ROOT / "external/umr_trial_20260908"
ROBOT_XML = ROOT / "ScaleRetarget/assets/unitree_g1/g1_mocap_29dof.xml"
BODY_MODEL = Path("/home/sw/shuaiwang/.cache/ula_smplx/SMPLX_NEUTRAL_2020.npz")
WINDOW_SECONDS, POINTS, EPOCHS = 5.0, 4096, 2500


def save_json(path, payload):
    temporary = Path(path).with_name(f".{Path(path).name}.tmp")
    with temporary.open("x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def validate_selection(pilot, body_model=BODY_MODEL):
    """Reuse the preselected forty, never choose origins based on new outcomes."""
    pilot = Path(pilot).resolve(strict=True)
    manifest = pilot / "manifest.json" if pilot.is_dir() else pilot
    value = json.loads(manifest.read_text())
    rows = value.get("motions", [])
    if value.get("schema") != 1 or value.get("automatic_promotion") is not False or len(rows) != 40:
        raise ValueError("Expected the full preselected non-promoting 40-origin manifest")
    if len({row["origin_id"] for row in rows}) != 40:
        raise ValueError("Duplicate origin identity")
    expected_counts = {f"{source}/{split}": count for source in refresh.SOURCES
                       for split, count in (("train", 6), ("validation", 2))}
    actual_counts = Counter(f"{row['dataset']}/{row['split']}" for row in rows)
    if actual_counts != expected_counts:
        raise ValueError("Expected exactly 6 train + 2 development origins per dataset")
    indexes, index_receipts = {}, {}
    for label, path in refresh.INDEXES.items():
        digest = sha256(path)
        if digest != value["index_sha256"][label]:
            raise ValueError("Original split index changed since preselection")
        index_receipts[label] = {"path": str(path), "sha256": digest}
        indexes[label] = yaml.safe_load(path.read_text())
    checked = []
    for row in rows:
        if row["split"] != ("train" if row["index_label"] == "train" else "validation"):
            raise ValueError("Train/development origin changed split")
        if indexes[row["index_label"]].get(row["origin_id"]) != row["baseline_packed"]:
            raise ValueError("Origin no longer matches the frozen original index")
        if refresh.resolve_inputs(row) != row:
            raise ValueError("Source/PKL/archive identity differs from original manifest")
        source = load_amass(Path(row["source"]))
        model = model_file_for_gender(body_model, source["gender"])
        checked.append({**row, "raw_frames": len(source["trans"]), "raw_fps": source["fps"],
            "source_duration_seconds": (len(source["trans"]) - 1) / source["fps"],
            "body_model": str(model), "body_model_sha256": sha256(model),
            "num_betas": len(source["betas"]), "window_start_seconds": 0.,
            "window_requested_seconds": WINDOW_SECONDS})
    refresh.validate_source_disjoint(checked)
    return {"selection": str(manifest), "selection_sha256": sha256(manifest),
            "origin_indexes": index_receipts, "counts": dict(actual_counts), "rows": checked}


def command_plan(row, output, *, retarget_python=RETARGET_PYTHON, umr_python=UMR_PYTHON):
    origin = row["origin_id"]
    surface = Path(output) / "surfaces" / f"{origin}.npz"
    retargeted = Path(output) / "retargeted" / origin
    script = ROOT / "scripts/umr_smplx_source.py"
    return {"surface": surface, "retargeted": retargeted,
        "prepare": [str(retarget_python), "-u", str(script), "prepare", "--source", row["source"],
            "--body-model", row["body_model"], "--output", str(surface), "--duration", str(WINDOW_SECONDS),
            "--start", "0", "--fps", "50", "--points", str(POINTS), "--seed", "0"],
        "retarget": [str(umr_python), "-u", str(script), "retarget", "--source", str(surface),
            "--umr-root", str(UMR_ROOT), "--output", str(retargeted), "--robot-xml", str(ROBOT_XML),
            "--epochs", str(EPOCHS), "--device", "cuda", "--setup-cache", str(Path(output) / "setup_cache")]}


def run_owned_child(command, log, timeout):
    with Path(log).open("x") as stream:
        child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = child.wait(timeout=timeout)
            if code:
                raise subprocess.CalledProcessError(code, command)
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


def owned_unit():
    matches = re.findall(r"(?:^|/)(bfm-umr-pilot-[A-Za-z0-9_.-]+\.service)(?:/|$)",
                         Path("/proc/self/cgroup").read_text(), flags=re.MULTILINE)
    if len(matches) != 1:
        raise ValueError("Execute inside a bounded bfm-umr-pilot-*.service user cgroup")
    mode = subprocess.check_output(["systemctl", "--user", "show", matches[0], "-p", "KillMode", "--value"], text=True).strip()
    if mode != "control-group":
        raise ValueError("Owned batch service must use KillMode=control-group")
    return matches[0]


def publish_pair_row(source_row, receipt, output, pair_dir):
    """Hardlink only newly generated owned outputs into the packer's hierarchy."""
    row = {key: source_row[key] for key in
           ("origin_id", "split", "index_label", "dataset", "source", "source_sha256")}
    for side in ("baseline", "candidate"):
        produced = Path(receipt["outputs"][side]["path"])
        digest = receipt["outputs"][side]["sha256"]
        if produced.resolve() != (pair_dir / f"{side}.pkl").resolve() or sha256(produced) != digest:
            raise ValueError("Paired output location/content mismatch")
        target = output / "input" / side / f"{row['origin_id']}.pkl"
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(produced, target)  # Fails rather than overwriting any old output.
        row[f"{side}_pkl"] = str(target)
        row[f"{side}_pkl_sha256"] = digest
    for key, name in (("pair_receipt", "pair_receipt.json"), ("sampling_proof", "sampling_proof.npz")):
        row[key] = str(pair_dir / name)
        row[f"{key}_sha256"] = sha256(pair_dir / name)
    row["expected_packed_frames"] = receipt["expected_packed_frames"]
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, default=ROOT / "local/data_refresh_20260908b")
    parser.add_argument("--packed-audit", type=Path, default=ROOT / "local/data_refresh_20260908b/packed_audit.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        parser.error("Refusing an existing output; pilot outputs are never overwritten")
    selection = validate_selection(args.pilot)
    output = args.output.absolute()
    if not args.execute:
        print(json.dumps({"plan_only": True, "origins": len(selection["rows"]), "counts": selection["counts"],
            "max_window_seconds": WINDOW_SECONDS, "surface_points": POINTS, "correspondence_epochs": EPOCHS,
            "output": str(output), "training_started": False,
            "first_commands": {k: v for k, v in command_plan(selection["rows"][0], output).items()
                               if k in ("prepare", "retarget")}}, indent=2))
        return 0
    unit = owned_unit()
    from scripts.umr_pair_dataset import prepare_pair
    from scripts.umr_smplx_source import verify_umr_checkout
    for python in (RETARGET_PYTHON, UMR_PYTHON):
        if not python.is_file() or not os.access(python, os.X_OK):
            raise ValueError("Missing existing isolated interpreter; do not reinstall IsaacLab")
    commit = verify_umr_checkout(UMR_ROOT)
    code = ("run_umr_amass_pilot.py", "umr_smplx_source.py", "umr_backend.py", "umr_pair_dataset.py",
            "retarget_data_refresh.py", "audit_packed_retarget_pilot.py", "amass_to_scalebfm.py")
    inputs = {str(ROOT / "scripts" / name): sha256(ROOT / "scripts" / name) for name in code}
    inputs.update({str(args.packed_audit.resolve()): sha256(args.packed_audit),
                   selection["selection"]: selection["selection_sha256"], str(ROBOT_XML): sha256(ROBOT_XML)})
    for index in selection["origin_indexes"].values():
        inputs[index["path"]] = index["sha256"]
    for row in selection["rows"]:
        for key, digest_key in (("source", "source_sha256"), ("baseline_pkl", "baseline_pkl_sha256"),
                                ("baseline_packed", "baseline_packed_sha256"), ("body_model", "body_model_sha256")):
            inputs[row[key]] = row[digest_key]
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    started = time.monotonic()
    status = {"schema": 1, "result": "RUNNING", "unit": unit, "motions_requested": 40,
        "motions_completed": 0, "motions": [], "training_started": False, "automatic_promotion": False}
    save_json(output / "plan.json", {"schema": 1, "experiment_id": output.name, **selection,
        "protected_files": inputs, "umr_commit": commit, "window_seconds": WINDOW_SECONDS,
        "points": POINTS, "epochs": EPOCHS, "training_started": False})
    pair_rows = []
    try:
        for number, row in enumerate(selection["rows"]):
            for name in code:
                path = str(ROOT / "scripts" / name)
                if sha256(Path(path)) != inputs[path]:
                    raise ValueError("Pipeline code changed during batch; do not mix candidates")
            plan = command_plan(row, output)
            state = {"origin_id": row["origin_id"], "result": "RUNNING", "stage": "prepare", "jobs": []}
            status["motions"].append(state)
            save_json(output / "status.json", status)
            print(f"[UMR PILOT] {number + 1}/40 {row['origin_id']}", flush=True)
            for stage in ("prepare", "retarget"):
                state["stage"] = stage
                job = {"stage": stage, "command": plan[stage], "timeout_seconds": 300, "result": "RUNNING"}
                state["jobs"].append(job)
                save_json(output / "status.json", status)
                before = time.monotonic()
                run_owned_child(plan[stage], output / "logs" / f"{number:02d}_{stage}.log", 300)
                job.update(result="COMPLETE", elapsed_seconds=time.monotonic() - before)
            state["stage"] = "pair"
            pair_dir = output / "pairs" / row["origin_id"]
            receipt = prepare_pair(row, prepared_source=plan["surface"],
                candidate_npz=plan["retargeted"] / "motion.npz", packed_audit=args.packed_audit,
                output_dir=pair_dir)
            pair_rows.append(publish_pair_row(row, receipt, output, pair_dir))
            state.update(result="COMPLETE", stage="complete")
            status["motions_completed"] += 1
            save_json(output / "status.json", status)
        for path, expected in inputs.items():
            if sha256(Path(path)) != expected:
                raise ValueError(f"Protected source/model/index/code changed: {path}")
        if verify_umr_checkout(UMR_ROOT) != commit:
            raise ValueError("UMR checkout changed during batch")
        manifest = {"schema": 1, "experiment_id": output.name,
            "origin_indexes": selection["origin_indexes"], "rows": pair_rows,
            "selection": selection["selection"], "selection_sha256": selection["selection_sha256"],
            "protected_files": inputs, "training_started": False, "automatic_promotion": False,
            "interpretation": "same-origin pipeline comparison; different human shapes, not solver-only ablation"}
        save_json(output / "paired_manifest.json", manifest)
        status.update(result="COMPLETE", inputs_verified_unchanged=True,
                      manifest_sha256=sha256(output / "paired_manifest.json"))
    except BaseException as error:
        status.update(result="FAIL", error=f"{type(error).__name__}: {error}")
        if status["motions"] and status["motions"][-1]["result"] == "RUNNING":
            status["motions"][-1]["result"] = "FAIL"
        raise
    finally:
        status["elapsed_seconds"] = time.monotonic() - started
        save_json(output / "status.json", status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
