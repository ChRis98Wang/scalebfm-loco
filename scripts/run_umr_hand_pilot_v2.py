#!/usr/bin/env python3
"""Four-source canonical-hand mechanism pilot; no PPO, promotion or physics steps.

Default: validate frozen v1 inputs and print a read-only plan. Execution writes
only a NEW local experiment directory, inside a bounded owned systemd cgroup.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

if __name__ == "__main__":
    sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import analyze_umr_reference_shift as reference

import numpy as np

PROTOCOL = ROOT / "docs/UMR_HAND_REFERENCE_PROTOCOL_V2_20260909.md"
CORE = ROOT / "scripts/umr_smplx_source.py"
GENERATOR = ROOT / "scripts/umr_hand_reference_v2.py"
PY_PREPARE = Path("/home/sw/.cache/bfm/scaleretarget-py311/bin/python")
PY_UMR = ROOT / "local/umr_trial_20260908/venv/bin/python"
UMR = ROOT / "external/umr_trial_20260908"
XML = ROOT / "ScaleRetarget/assets/unitree_g1/g1_mocap_29dof.xml"
SELECTED = (
    "ACCAD/Female1General_c3d/A2_-_Sway_stageii",
    "BMLmovi/Subject_22_F_MoSh/Subject_22_F_9_stageii",
    "BMLrub/rub058/0012_normal_jog4_stageii",
    "KIT/205/walking_run06_stageii",
)


def select_origins(origins):
    if len(origins) != 17 or len(set(origins)) != 17:
        raise ValueError("Require all frozen seventeen train targets")
    groups = {}
    for origin in sorted(origins):
        if not isinstance(origin, str) or any(p in ("", ".", "..") for p in origin.split("/")):
            raise ValueError("Invalid origin")
        groups.setdefault(origin.split("/")[0], origin)
    chosen = tuple(groups[k] for k in sorted(groups))
    if chosen != SELECTED:
        raise ValueError("Fixed four-source first-origin selection changed")
    return chosen


def new_directory(path):
    path = Path(path).absolute()
    if (path.exists() or path.is_symlink() or any(p.is_symlink() for p in path.parents)
            or path.parent != ROOT / "local"):
        raise ValueError("Require a NEW non-symlink experiment directly under local/")
    return path


def bounded_unit(run_id):
    unit = f"bfm-umr-hand-{run_id}.service"
    matches = re.findall(r"(?:^|/)(bfm-umr-hand-[A-Za-z0-9_-]+\.service)(?:/|$)",
                         Path("/proc/self/cgroup").read_text(), flags=re.MULTILINE)
    if matches != [unit]:
        raise ValueError(f"Execute only within {unit}")
    out = subprocess.check_output(["systemctl", "--user", "show", unit, "-p", "KillMode",
        "-p", "Restart", "-p", "RuntimeMaxUSec", "-p", "MemoryMax", "-p", "TasksMax"], text=True)
    values = dict(line.split("=", 1) for line in out.splitlines())
    if (values.get("KillMode") != "control-group" or values.get("Restart") != "no"
            or any(values.get(k) in (None, "", "0", "infinity")
                   for k in ("RuntimeMaxUSec", "MemoryMax", "TasksMax"))):
        raise ValueError("Finite runtime/memory/tasks and group cleanup are required")
    return unit


def run_child(command, log_path, timeout):
    with Path(log_path).open("x", encoding="utf-8") as log:
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        try:
            result = child.wait(timeout=timeout)
            if result:
                raise RuntimeError(f"Child exit {result}; see {log_path}")
        finally:
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(child.pid, sig)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
            child.wait(timeout=10)


def write_json(path, payload, *, replace=False):
    path = Path(path)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if replace:
        # Only our already-created status file may be atomically replaced.
        if not path.is_file() or path.is_symlink():
            raise ValueError("Status replacement requires an owned regular file")
        temporary = path.with_suffix(".next.json")
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    else:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())


def locate_parent(row, manifest, frozen):
    pair = frozen.read_json(row["pair_receipt"], row["pair_receipt_sha256"])
    found = {}
    for path, digest in pair["input_sha256"].items():
        p = Path(path)
        if p.suffix != ".npz" or not p.is_relative_to(Path(manifest["evidence"]["batch"])):
            continue
        frozen.add(p, digest)
        with np.load(p, allow_pickle=False) as z:
            if "metadata_json" not in z:
                continue
            meta = json.loads(z["metadata_json"].item())
        if meta.get("schema") == "bfm.smplx_surface_source/1":
            if "prepared" in found:
                raise ValueError("Ambiguous prepared source")
            found["prepared"] = (str(p), digest, meta)
        elif meta.get("schema") == "bfm.umr_smplx_trial/1":
            if "original_motion" in found:
                raise ValueError("Ambiguous original motion")
            found["original_motion"] = (str(p), digest, meta)
    if set(found) != {"prepared", "original_motion"}:
        raise ValueError("Need exactly one original surface and one UMR motion")
    p, digest, meta = found["prepared"]
    motion, motion_digest, receipt = found["original_motion"]
    if (meta["source_sha256"] != row["source_sha256"] or meta["source_file"] != row["source"]
            or receipt["source"] != meta or receipt["prepared_source_sha256"] != digest):
        raise ValueError("Original prepared/motion source bindings disagree")
    frozen.add(meta["body_model"], meta["body_model_sha256"])
    return {"origin_id": row["origin_id"], "prepared": p, "prepared_sha256": digest,
            "original_motion": motion, "original_motion_sha256": motion_digest,
            "source_sha256": row["source_sha256"], "frames": meta["frames"]}


def plan(dataset, run_id):
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,50}", run_id):
        raise ValueError("Invalid run id")
    frozen = reference.Snapshot()
    frozen.add(dataset)
    for p in (Path(__file__), PROTOCOL, CORE, GENERATOR, Path(reference.__file__),
              ROOT / "scripts/build_umr_behavior_ab.py", ROOT / "scripts/retarget_data_refresh.py",
              ROOT / "scripts/umr_backend.py", *reference.VALIDATION_CODE):
        frozen.add(p)
    # Bind the historic offsets used by this diagnostic, not just current constants.
    for p in (ROOT / "ScaleRetarget/config/correspondence/gmr/amass_to_unitree_g1.yaml",
              ROOT / "logs/amass_preparation/accad_batch_v1_20260905.log",
              ROOT / "ScaleRetarget/scaleretarget/retargeter/gmr_retargeter.py"):
        frozen.add(p)
    manifest = reference.dataset_contract.validate_manifest(dataset)
    for p, digest in manifest["input_sha256"].items():
        frozen.add(p, digest)
    frozen.aliases.update(manifest["resolved_input_aliases"])
    pilot = frozen.read_json(Path(manifest["evidence"]["batch"]) / "paired_manifest.json",
                             manifest["evidence"]["pins"]["manifest"])
    rows = {r["origin_id"]: r for r in pilot["rows"]}
    chosen = select_origins(manifest["target_origins"])
    records = [locate_parent(rows[origin], manifest, frozen) for origin in chosen]
    for directory in (XML.parent, UMR / "umr", UMR / "configs"):
        for p in sorted(directory.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                frozen.add(p)
    for python in (PY_PREPARE, PY_UMR):
        target = python.resolve(strict=True)
        frozen.aliases[str(python)] = str(target)
        frozen.add(target)
    output = new_directory(ROOT / "local" / f"umr_hand_v2_{run_id}")
    frozen.recheck()
    return {"schema": "bfm.umr_hand_pilot/2", "result": "PLAN", "post_hoc": True,
            "automatic_promotion": False, "training_updates": 0, "policy_ppo_updates": 0, "physics_stepped": False,
            "training_updates_definition": "PPO/policy only; Stage I correspondence learning is separate",
            "stage_i_total_planned_epochs": 8 * 2500,
            "dataset": str(Path(dataset).absolute()), "output": str(output), "run_id": run_id,
            "protocol": str(PROTOCOL), "protocol_sha256": frozen.hashes()[str(PROTOCOL)],
            "selection": "lexicographically first target origin per available source",
            "origins": records, "epochs": 2500, "stage_i_seed": 0, "stage_ii_iterations": 6,
            "native_clock": "all inclusive prepared frames at 50 Hz; NOT packed half-open",
            "input_sha256": frozen.hashes(), "resolved_input_aliases": frozen.aliases}, frozen


def load_motion(path, prepared, model):
    import mujoco
    from scripts import umr_smplx_source as core
    with np.load(path, allow_pickle=False) as z:
        q = z["qpos"].astype(np.float64)
        receipt = json.loads(z["metadata_json"].item())
        names = z["dof_names"].tolist()
        if (q.shape != (len(prepared["times"]), 36) or not np.isfinite(q).all()
                or not np.allclose(np.linalg.norm(q[:, 3:7], axis=1), 1., rtol=0, atol=1e-5)
                or z["fps"].item() != 50 or not np.array_equal(z["frame_indices"], np.arange(len(q)))):
            raise ValueError("Motion/frame contract violated")
        for field in ("point_error", "normal_error", "contact_count", "floor_rows"):
            values = z[field]
            if (values.shape != (len(q),) or values.dtype.kind not in "ifu"
                    or not np.isfinite(values).all() or np.any(values < 0)):
                raise ValueError(f"Invalid native diagnostic array: {field}")
            if field in ("contact_count", "floor_rows") and values.dtype.kind not in "iu":
                raise ValueError(f"Expected integer native counts: {field}")
        if np.any(z["normal_error"] > np.pi + 1e-9):
            raise ValueError("Invalid native normal angle")
        if (receipt.get("frames") != len(q) or receipt.get("fps") != 50
                or not np.isclose(receipt.get("point_error_mean_m", np.nan), z["point_error"].mean(), rtol=0, atol=1e-12)):
            raise ValueError("Native diagnostic summary differs from its receipt")
    model_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(1, model.njnt)]
    if (names != model_names or receipt["source"] != prepared["metadata"]
            or receipt["protected_inputs_rechecked"] is not True
            or receipt["umr_commit"] != core.UMR_COMMIT
            or receipt["robot_xml_sha256"] != core.sha256(XML)
            or receipt["epochs"] != 2500 or receipt["config"]["correspondence"]["seed"] != 0
            or receipt["config"]["retarget"]["iterations"] != 6
            or receipt["config"]["retarget"]["n_selected"] != 512
            or receipt["solve_failures"] != 0):
        raise ValueError("Motion provenance/config/solver evidence violated")
    return q, receipt


def stage_evidence(log_path):
    """Read both actual resolved device and completed training device, not requested cfg."""
    log = Path(log_path).read_text(encoding="utf-8")
    start = re.findall(r"\[stage1\] 训练 (\d+) epochs，设备 = (\S+)", log)
    finish = re.findall(r"\[stage1\] 训练耗时 ([0-9.]+)s  device=(\S+)", log)
    if start != [("2500", "cuda")] or len(finish) != 1 or finish[0][1] != "cuda":
        raise ValueError("Need one fresh completed Stage I on actual CUDA; no silent CPU fallback/cache")
    return {"epochs": 2500, "resolved_device": "cuda", "completed_device": "cuda",
            "training_seconds_log_precision": float(finish[0][0]), "log_sha256": reference.sha256(log_path)}


def statistics(value):
    value = np.asarray(value, dtype=np.float64)
    if not value.size or not np.isfinite(value).all():
        raise ValueError("Nonempty finite statistics required")
    return {"mean": float(value.mean()), "p95": float(np.percentile(value, 95)), "max": float(value.max())}


def kinematic_measurements(qpos, prepared, model):
    import mujoco
    from scipy.spatial.transform import Rotation
    from scripts.retarget_data_refresh import audit_qpos
    names = ("pelvis", "left_wrist_yaw_link", "right_wrist_yaw_link")
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in names]
    if min(ids) <= 0 or len(set(ids)) != 3:
        raise ValueError("Named wrist/pelvis links missing")
    data = mujoco.MjData(model)
    q, p = np.empty((len(qpos), 3, 4)), np.empty((len(qpos), 3, 3))
    for t, pose in enumerate(qpos):
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)  # FK/contact detection only; never mj_step.
        q[t], p[t] = data.xquat[ids], data.xpos[ids]
    offsets_wxyz = np.array([[.5, -.5, -.5, -.5], [1., 0., 0., 0.], [0., 0., 0., -1.]])
    offsets = Rotation.from_quat(offsets_wxyz[:, [1, 2, 3, 0]]).as_matrix()
    target = prepared["joint_rotations"][:, [0, 20, 21]] @ offsets
    target_q = Rotation.from_matrix(target.reshape(-1, 3, 3)).as_quat().reshape(len(qpos), 3, 4)
    error = np.rad2deg(reference.quaternion_angle(q, target_q[:, :, [3, 0, 1, 2]]))
    velocity = np.abs(np.diff(qpos[:, 7:], axis=0) * 50.)
    return {"geometry": audit_qpos(qpos, model),
            "old_gmr_world_orientation_target_error_deg": {name: statistics(error[:, i]) for i, name in enumerate(names)},
            "joint_absolute_secant_speed_rad_s": statistics(velocity),
            "joint_intervals_above_unwired_12_rad_s": int(np.sum(velocity > 12.)),
            "velocity_note": "Native adjacent-frame secants; configured 12 rad/s is NOT enforced by v1 solver"}, p, q


def compare_native(first, second):
    if first.shape != second.shape or first.ndim != 2 or first.shape[1] != 36:
        raise ValueError("Native qpos pair shape mismatch")
    return {"root_translation_m": statistics(np.linalg.norm(first[:, :3] - second[:, :3], axis=1)),
            "root_orientation_deg": statistics(np.rad2deg(reference.quaternion_angle(first[:, 3:7], second[:, 3:7]))),
            "joint_absolute_difference_rad": statistics(np.abs(first[:, 7:] - second[:, 7:]))}


def execute(plan_value, frozen):
    from scripts import umr_smplx_source as core
    import mujoco

    unit = bounded_unit(plan_value["run_id"])
    core.verify_umr_checkout(UMR)
    directory = new_directory(plan_value["output"])
    frozen.recheck()
    directory.mkdir()
    status = {**plan_value, "result": "RUNNING", "unit": unit, "results": [], "jobs_complete": 0,
              "inputs_verified_unchanged": False, "runtime": {"python": sys.version, "numpy": np.__version__,
              "mujoco_fk": mujoco.__version__}, "limitations": [
                  "Four geometry-selected short train windows, not a general data or policy quality test.",
                  "Old GMR target compatibility is not a requirement of the surface objective or human-fidelity ground truth.",
                  "No policy learning/evaluation, dynamics/contact forces, manipulation task or automatic promotion."]}
    status_path = directory / "status.json"
    write_json(status_path, status)
    started = time.monotonic()
    try:
        model = mujoco.MjModel.from_xml_path(str(XML))
        if model.nq != 36 or model.njnt != 30:
            raise ValueError("Expected exactly the native free-base 29-DOF G1")
        for index, row in enumerate(plan_value["origins"]):
            case = directory / f"case_{index}"
            case.mkdir()
            original = core.load_prepared_source(Path(row["prepared"]))
            variant_dir = case / "matched_source"
            run_child([str(PY_PREPARE), "-B", str(GENERATOR), "--parent", row["prepared"],
                "--parent-sha256", row["prepared_sha256"], "--output", str(variant_dir), "--execute"],
                case / "prepare.log", 180)
            variant_path = variant_dir / "prepared_source.npz"
            variant = core.load_prepared_source(variant_path)
            if abs(float(original["actor_height"]) - float(variant["actor_height"])) > 1e-6:
                raise ValueError("Hand intervention changed actor height: not the fixed single-variable pilot")
            moved = ("sequence_points", "sequence_normals", "joint_positions", "joint_rotations", "times",
                     "foot_low", "face_indices", "barycentric", "sample_lower", "sample_upper", "sample_alpha")
            if any(not np.array_equal(original[k], variant[k]) for k in moved):
                raise ValueError("Variant changed moving targets, material points or source clock")
            record = {"origin_id": row["origin_id"], "frames": len(original["times"]), "arms": {},
                      "all_moving_arrays_unchanged": True, "variant_source": str(variant_path),
                      "variant_source_sha256": reference.sha256(variant_path)}
            all_q, all_poses = {}, {}
            for arm, source, prepared in (("control", Path(row["prepared"]), original),
                                           ("matched_hand", variant_path, variant)):
                target = case / arm
                run_child([str(PY_UMR), "-B", str(CORE), "retarget", "--source", str(source),
                    "--umr-root", str(UMR), "--output", str(target), "--robot-xml", str(XML),
                    "--epochs", "2500", "--device", "cuda", "--iterations", "6"], case / f"{arm}.log", 360)
                stage = stage_evidence(case / f"{arm}.log")
                qpos, receipt = load_motion(target / "motion.npz", prepared, model)
                if receipt != json.loads((target / "receipt.json").read_text()):
                    raise ValueError("Standalone receipt disagrees with motion metadata")
                if receipt["prepared_source_sha256"] != reference.sha256(source):
                    raise ValueError("Retarget source digest mismatch")
                measures, p, q = kinematic_measurements(qpos, prepared, model)
                record["arms"][arm] = {**measures, "motion": str(target / "motion.npz"),
                    "motion_sha256": reference.sha256(target / "motion.npz"), "solve_failures": receipt["solve_failures"],
                    "scale": receipt["scale"], "ground_offset": receipt["ground_offset"],
                    "stage_i_actual_execution": stage,
                    "setup_key": receipt["setup_cache"]["key_sha256"], "retarget_environment": receipt["environment"]}
                all_q[arm], all_poses[arm] = qpos, (p, q)
                status["jobs_complete"] += 1
                status["stage_i_total_completed_epochs"] = status["jobs_complete"] * 2500
                write_json(status_path, status, replace=True)
            old_q, _ = load_motion(Path(row["original_motion"]), original, model)
            if (record["arms"]["control"]["retarget_environment"] != record["arms"]["matched_hand"]["retarget_environment"]
                    or record["arms"]["control"]["setup_key"] == record["arms"]["matched_hand"]["setup_key"]):
                raise ValueError("Arms changed runtime or reused the same canonical identity")
            record["fresh_control_vs_frozen_original"] = compare_native(all_q["control"], old_q)
            record["matched_hand_vs_control"] = compare_native(all_q["matched_hand"], all_q["control"])
            pa, qa = all_poses["control"]
            pb, qb = all_poses["matched_hand"]
            record["wrist_world_fk_shift"] = {
                side: {"position_m": statistics(np.linalg.norm(pa[:, i] - pb[:, i], axis=1)),
                       "orientation_deg": statistics(np.rad2deg(reference.quaternion_angle(qa[:, i], qb[:, i])))}
                for i, side in ((1, "left"), (2, "right"))}
            status["results"].append(record)
            write_json(status_path, status, replace=True)
        frozen.recheck()
        core.verify_umr_checkout(UMR)
        status.update(result="COMPLETE_MECHANISM_PILOT_NOT_QUALITY_ACCEPTED", inputs_verified_unchanged=True)
    except Exception as error:
        status.update(result="ERROR", error=f"{type(error).__name__}: {error}")
    finally:
        status["elapsed_seconds"] = time.monotonic() - started
        status["output_sha256"] = {str(p): reference.sha256(p) for p in sorted(directory.rglob("*"))
                                   if p.is_file() and p != status_path}
        write_json(status_path, status, replace=True)
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    value, frozen = plan(args.dataset, args.run_id)
    report = execute(value, frozen) if args.execute else value
    if args.execute:
        print(json.dumps({k: report[k] for k in ("result", "output", "jobs_complete")}, indent=2))
    else:
        print(json.dumps(value, indent=2, sort_keys=True))
    return int(report["result"] == "ERROR")


if __name__ == "__main__":
    raise SystemExit(main())
