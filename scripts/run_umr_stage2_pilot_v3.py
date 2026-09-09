#!/usr/bin/env python3
"""Fixed four-origin/four-arm Stage II experiment, no Stage I or policy training.

Default is a read-only plan. --execute and --worker require an owned bounded
systemd unit. Frozen setup numeric arrays are read without enabling pickle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

if __name__ == "__main__":
    sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np

PINS = {
    "scripts/run_umr_hand_pilot_v2.py": "48da157de30cc2b85719cf3ec8151a0837898452ecfe50684399eca8e031ee8d",
    "scripts/umr_smplx_source.py": "d2b41fe5cffaed03637d3e37cbdf9fef1f4c30996d636b6146483bf5e9b66b94",
    "scripts/analyze_umr_reference_shift.py": "19fe46baecca3a91cc0defb68cdfd6b6cd755897e4aa4b597c250181ed19f9e3",
}
for _name, _expected in PINS.items():
    if hashlib.sha256((ROOT / _name).read_bytes()).hexdigest() != _expected:
        raise ValueError(f"Frozen v3 dependency changed: {_name}")
from scripts import run_umr_hand_pilot_v2 as v2
from scripts import umr_smplx_source as core

PARENT = ROOT / "local/umr_hand_v2_20260909a/status.json"
PARENT_SHA = "7416751c1b83c20d46f4d375123e0cc311f417b8011fbe3f0b5d6b755d0b3f0b"
PROTOCOL = ROOT / "docs/UMR_STAGE2_CONTROLS_PROTOCOL_V3_20260909.md"
MODULES = (ROOT / "scripts/umr_stage2_controls_v3.py", ROOT / "scripts/umr_output_rate_limit_v3.py")
ARMS = ("control", "rate_only", "wrist_only", "both")
SCHEMA = "bfm.umr_stage2_controls_pilot/3"


def unit_guard(run_id):
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,50}", run_id):
        raise ValueError("Invalid run id")
    name = f"bfm-umr-stage2-{run_id}.service"
    matches = re.findall(r"(?:^|/)(bfm-umr-stage2-[A-Za-z0-9_-]+\.service)(?:/|$)",
                         Path("/proc/self/cgroup").read_text(), re.MULTILINE)
    if matches != [name]:
        raise ValueError(f"Execution requires {name}")
    text = subprocess.check_output(["systemctl", "--user", "show", name, "-p", "KillMode", "-p", "Restart",
        "-p", "RuntimeMaxUSec", "-p", "MemoryMax", "-p", "TasksMax"], text=True, timeout=10)
    properties = dict(line.split("=", 1) for line in text.splitlines())
    if (properties.get("KillMode") != "control-group" or properties.get("Restart") != "no"
            or any(properties.get(k) in (None, "", "0", "infinity")
                   for k in ("RuntimeMaxUSec", "MemoryMax", "TasksMax"))):
        raise ValueError("Required finite cgroup bounds/cleanup missing")
    return name


def add_resolved(frozen, path):
    path = Path(path).absolute()
    resolved = path.resolve(strict=True)
    if path != resolved:
        frozen.aliases[str(path)] = str(resolved)
    return str(resolved), frozen.add(resolved)


def numerical_dependencies():
    """Inventory installed numeric distributions without importing the solver."""
    code = """import importlib.metadata as m, json, sys
names=('numpy','scipy','mujoco','mink','qpsolvers','clarabel','trimesh')
files=set(); versions={}
for name in names:
 d=m.distribution(name); versions[name]=d.version
 if d.files is None: raise ValueError('Missing distribution inventory '+name)
 for p in d.files:
  if '__pycache__' not in p.parts and p.suffix not in ('.pyc','.pyo'):
   f=d.locate_file(p)
   if f.is_file(): files.add(str(f.absolute()))
print(json.dumps({'files':sorted(files),'versions':versions,'python':sys.version,'executable':sys.executable}))
"""
    result = subprocess.run([str(v2.PY_UMR), "-B", "-c", code], check=True, capture_output=True,
                            text=True, timeout=30)
    return json.loads(result.stdout)


def select_cases(parent):
    if (parent.get("result") != "COMPLETE_MECHANISM_PILOT_NOT_QUALITY_ACCEPTED"
            or parent.get("jobs_complete") != 8 or parent.get("inputs_verified_unchanged") is not True
            or len(parent["origins"]) != 4 or len(parent["results"]) != 4):
        raise ValueError("Require the complete frozen v2 four-case experiment")
    cases = []
    for index, (origin, result) in enumerate(zip(parent["origins"], parent["results"])):
        if origin["origin_id"] != v2.SELECTED[index] or result["origin_id"] != v2.SELECTED[index]:
            raise ValueError("Four fixed origins/order changed")
        control = PARENT.parent / f"case_{index}" / "control"
        if result["arms"]["control"]["motion"] != str(control / "motion.npz"):
            raise ValueError("Control artifact location mismatch")
        cases.append({**origin, "index": index, "control_motion": str(control / "motion.npz"),
            "setup_bodies": str(control / "setup/bodies.npz"),
            "setup_correspondence": str(control / "setup/correspondence.npz"),
            "setup_key": result["arms"]["control"]["setup_key"], "native_frames": result["frames"]})
    return cases


def build_plan(run_id):
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,50}", run_id):
        raise ValueError("Invalid run id")
    frozen = v2.reference.Snapshot()
    parent = frozen.read_json(PARENT, PARENT_SHA)
    cases = select_cases(parent)
    for group in (parent["input_sha256"], parent["output_sha256"]):
        for path, digest in group.items():
            frozen.add(path, digest)
    frozen.aliases.update(parent["resolved_input_aliases"])
    worker_files = {str(PARENT): PARENT_SHA}
    for p in (Path(__file__).resolve(), PROTOCOL, *MODULES, *(ROOT / name for name in PINS),
              ROOT / "scripts/umr_backend.py", ROOT / "scripts/retarget_data_refresh.py"):
        worker_files[str(p)] = frozen.add(p)
    for directory in (v2.XML.parent, v2.UMR / "umr", v2.UMR / "configs"):
        for p in sorted(directory.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                worker_files[str(p)] = frozen.add(p)
    runtime = numerical_dependencies()
    for p in (*runtime.pop("files"), v2.PY_UMR):
        target, digest = add_resolved(frozen, p)
        worker_files[target] = digest
    for case in cases:
        case["input_sha256"] = {}
        for key in ("prepared", "original_motion", "control_motion", "setup_bodies", "setup_correspondence"):
            p = case[key]
            if p not in frozen.files:
                raise ValueError(f"Case input missing original binding: {p}")
            case["input_sha256"][p] = frozen.files[p][0]
    output = v2.new_directory(ROOT / "local" / f"umr_stage2_v3_{run_id}")
    frozen.recheck()
    return {"schema": SCHEMA, "result": "PLAN_ONLY", "run_id": run_id, "output": str(output),
            "protocol": str(PROTOCOL), "protocol_sha256": frozen.hashes()[str(PROTOCOL)],
            "parent": str(PARENT), "parent_sha256": PARENT_SHA, "cases": cases, "arms": list(ARMS),
            "automatic_promotion": False, "physics_stepped": False, "policy_ppo_updates": 0, "stage_i_epochs": 0,
            "fixed_parameters": {"fps": 50, "iterations": 6, "warmup_iterations": 60,
                "wrist_orientation_cost": 10., "wrist_position_cost": 0., "wrist_lm_damping": 1.,
                "hinge_max_velocity_rad_s": 12., "output_interval_tolerance_rad": 2e-6,
                "control_reproduction_absolute_tolerance": 1e-9},
            "numerical_runtime": runtime, "worker_input_sha256": worker_files,
            "input_sha256": frozen.hashes(), "resolved_input_aliases": frozen.aliases}, frozen


def exact_array(actual, expected, label):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or actual.dtype != expected.dtype or not np.array_equal(actual, expected):
        raise ValueError(f"Frozen Stage I/source mismatch: {label}")


def load_stage_inputs(case):
    """Reconstruct v1 targets and validate cached bodies against canonical source."""
    core.verify_umr_checkout(v2.UMR)
    sys.path.insert(0, str(v2.UMR))
    from umr.config import load_config
    from umr.bodies.robot import RobotBody, RobotSpec
    from umr.retarget.pipeline import UMRRetargeter
    import mink
    import mujoco
    prepared = core.load_prepared_source(Path(case["prepared"]))
    cfg = load_config("g1")
    cfg["robot"]["xml"] = str(v2.XML)
    cfg["retarget"]["tpose_offset"] = 0.
    cfg["retarget"]["iterations"] = 6
    robot = RobotBody(v2.XML, RobotSpec.from_config(cfg.robot))
    original_q, receipt = v2.load_motion(Path(case["control_motion"]), prepared, robot.model)
    if (receipt["config"] != cfg or receipt["loaded_robot_geometry"] != core.model_geometry_fingerprint(robot.model)
            or receipt["environment"] != core.environment_packages(("numpy", "scipy", "mujoco", "mink", "torch",
                "qpsolvers", "clarabel", "trimesh"))
            or prepared["metadata"]["frames"] != case["native_frames"]
            or prepared["metadata"]["points"] != 4096):
        raise ValueError("Frozen runtime/robot/source/config differs")
    human = core.SmplxSurfaceHuman(prepared, robot.height())
    if human.scale != receipt["scale"] or human.ground_offset != receipt["ground_offset"]:
        raise ValueError("Original scale/ground offset not reproduced")
    ids = prepared["binding_joint_ids"].astype(np.int64)
    rotations = human.data.xmat[ids].reshape(-1, 3, 3)
    canonical = prepared["canonical_points"] * human.scale
    local_pos = np.einsum("nji,nj->ni", rotations, canonical - human.data.xpos[ids])
    local_normal = np.einsum("nji,nj->ni", rotations, prepared["canonical_normals"])
    with np.load(case["setup_bodies"], allow_pickle=False) as bodies:
        if bodies["stamp"].item() != case["setup_key"] or bodies["robot_xml"].item() != str(v2.XML):
            raise ValueError("Canonical setup stamp/robot mismatch")
        expected = {"human_points": canonical, "human_normals": prepared["canonical_normals"],
                    "human_body_ids": ids, "human_geom_ids": np.full(len(ids), -1, dtype=np.int64),
                    "human_local_pos": local_pos, "human_local_normal": local_normal,
                    "human_segment": prepared["segment"]}
        for k, v in expected.items():
            exact_array(bodies[k], v, k)
    # Read numeric binding fields only. Upstream stores some optional descriptive
    # metadata as object arrays; this path never enables pickle for those fields.
    fields = {"bind_body_ids": (4096,), "bind_local_pos": (4096, 3), "bind_local_normal": (4096, 3),
              "inherited_segment": (4096,), "loss_history": (2500, 5)}
    with np.load(case["setup_correspondence"], allow_pickle=False) as corr:
        values = {k: corr[k] for k in fields}
        expected_stamp = hashlib.sha1(json.dumps(
            ["correspondence/2", case["setup_key"], cfg.correspondence, 2500],
            sort_keys=True, default=str, ensure_ascii=False).encode()).hexdigest()[:16]
        # Same pure stamp formula as pinned umr.stages; do not import Stage I.
        if corr["stamp"].item() != expected_stamp:
            raise ValueError("Stage I correspondence stamp differs from canonical/config")
    for k, shape in fields.items():
        value = values[k]
        if value.shape != shape or value.dtype.kind not in "iuf" or not np.isfinite(value).all():
            raise ValueError(f"Invalid numeric Stage I cache array: {k}")
    for k in ("bind_body_ids", "inherited_segment"):
        if values[k].dtype.kind not in "iu" or np.any(values[k] < 0):
            raise ValueError(f"Invalid Stage I integer labels: {k}")
    if (np.any(values["bind_body_ids"] <= 0) or np.any(values["bind_body_ids"] >= robot.model.nbody)
            or not np.allclose(np.linalg.norm(values["bind_local_normal"], axis=1), 1., rtol=0, atol=1e-6)
            or not np.array_equal(values["loss_history"][:, 0], np.arange(2500))):
        raise ValueError("Bad correspondence link/normals/epochs")
    exact_array(values["inherited_segment"], prepared["segment"], "inherited_segment")
    kwargs = {key: cfg.retarget[key] for key in (
        "n_selected", "point_selection", "tpose_offset", "iterations", "dt", "damping", "solver",
        "trust_region", "trust_region_radius", "floor_height", "floor_band", "floor_margin",
        "contact_threshold", "contact_weight", "posture_cost", "self_collision")}
    kwargs.update(human_body_ids=ids, human_local_pos=local_pos, human_local_normal=local_normal,
                  robot_body_ids=values["bind_body_ids"], robot_local_pos=values["bind_local_pos"],
                  robot_local_normal=values["bind_local_normal"], segment=values["inherited_segment"],
                  segment_names=core.SEGMENTS)
    return robot, human, prepared, cfg, kwargs, core.retargeter_class(UMRRetargeter), mink, mujoco, original_q


def validate_result(result, frames):
    shapes = {"qpos": (frames, 36), "frame_indices": (frames,), "point_error": (frames,),
              "normal_error": (frames,), "contact_count": (frames,), "floor_rows": (frames,)}
    for name, shape in shapes.items():
        value = np.asarray(getattr(result, name))
        if value.shape != shape or value.dtype.kind not in "iuf" or not np.isfinite(value).all():
            raise ValueError(f"Nonfinite or malformed result: {name}")
        if name not in ("qpos", "frame_indices") and np.any(value < 0):
            raise ValueError(f"Negative result: {name}")
        if name in ("frame_indices", "contact_count", "floor_rows") and value.dtype.kind not in "iu":
            raise ValueError(f"Noninteger result count: {name}")
    if (not np.array_equal(result.frame_indices, np.arange(frames)) or result.fps != 50
            or not np.allclose(np.linalg.norm(result.qpos[:, 3:7], axis=1), 1., rtol=0, atol=1e-5)
            or np.any(result.normal_error > np.pi + 1e-9)):
        raise ValueError("Result clock/quaternion/normal mismatch")


def validate_execution_audit(audit, arm, frames, failures):
    if (audit.get("result") != "COMPLETE_NOT_QUALITY_ACCEPTED" or audit.get("arm") != arm
            or audit["warmup"]["frame"] != 0 or audit["warmup"]["warmup"] is not True
            or audit["warmup_failures"] != audit["warmup"]["failures"]
            or [r["frame"] for r in audit["frames"]] != list(range(frames))
            or any(r["warmup"] is not False for r in audit["frames"])
            or any(type(r["failures"]) is not int or r["failures"] < 0 for r in [audit["warmup"], *audit["frames"]])
            or int(failures) != sum(r["failures"] for r in audit["frames"])
            or audit["output_failures"] != int(failures)
            or audit["total_observed_failures"] != int(failures) + audit["warmup_failures"]
            or audit["rate_enabled"] != (arm in ("rate_only", "both"))
            or len(audit["wrist_tasks"]) != (2 if arm in ("wrist_only", "both") else 0)
            or audit["postprocessing_performed"] is not False):
        raise ValueError("Actual warmup/output/task/limit accounting mismatch")


def worker(inputs, inputs_sha, case_index, arm):
    frozen = v2.reference.Snapshot()
    job = frozen.read_json(inputs, inputs_sha)
    unit_guard(job["run_id"])
    if job["schema"] != SCHEMA or job["arms"] != list(ARMS) or arm not in ARMS or not 0 <= case_index < 4:
        raise ValueError("Invalid immutable worker selection")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Stage II worker must explicitly hide CUDA; no new correspondence/policy training")
    if Path(sys.executable).resolve() != v2.PY_UMR.resolve():
        raise ValueError("Worker requires the existing pinned UMR interpreter")
    case = job["cases"][case_index]
    for group in (job["worker_input_sha256"], case["input_sha256"]):
        for p, digest in group.items():
            frozen.add(p, digest)
    frozen.aliases.update({p: t for p, t in job["resolved_input_aliases"].items()
                           if t in job["worker_input_sha256"]})
    parent = frozen.read_json(PARENT, PARENT_SHA)
    expected_case = select_cases(parent)[case_index]
    if any(case[k] != v for k, v in expected_case.items()):
        raise ValueError("Worker source/case not derived from fixed parent")
    directory = Path(job["output"]) / f"case_{case_index}" / arm
    if (directory.exists() or directory.is_symlink() or any(p.is_symlink() for p in directory.parents)
            or directory.parent.parent != ROOT / "local" / f"umr_stage2_v3_{job['run_id']}"):
        raise ValueError("Worker needs a NEW exact arm output")
    from scripts.umr_output_rate_limit_v3 import output_rate_limit_class, interval_statistics, validate_hinge_layout
    from scripts.umr_stage2_controls_v3 import controlled_retargeter_class
    robot, human, prepared, cfg, kwargs, base, mink, mujoco, old_q = load_stage_inputs(case)
    from mink.limits.limit import Constraint
    RateLimit = output_rate_limit_class(mink.Limit, Constraint)
    retargeter = controlled_retargeter_class(base, mink, RateLimit)(robot, human, arm=arm, **kwargs)
    # Retain observed per-frame evidence on a failing solve without exporting a
    # bad motion or silently selecting a different arm/clip.
    frozen.recheck()
    directory.mkdir()
    try:
        result = retargeter.run(np.arange(case["native_frames"]), 50., warmup_iterations=60, progress=False)
        validate_result(result, case["native_frames"])
    except Exception as error:
        v2.write_json(directory / "failure_receipt.json", {
            "schema": SCHEMA, "result": "ERROR", "case": case_index, "origin_id": case["origin_id"], "arm": arm,
            "error": f"{type(error).__name__}: {error}", "runtime_audit": retargeter.v3_audit,
            "exception_report": getattr(error, "report", None), "inputs_sha256": frozen.hashes(),
            "automatic_promotion": False, "physics_stepped": False, "policy_ppo_updates": 0, "stage_i_epochs": 0})
        raise
    v2.write_json(directory / "solve_audit.json", retargeter.v3_audit)
    qpos = result.qpos.copy()
    control_difference = v2.compare_native(qpos, old_q) if arm == "control" else None
    if arm == "control" and not np.allclose(qpos, old_q, rtol=0, atol=1e-9):
        raise ValueError("Control failed 1e-9 frozen qpos reproduction")
    measures, _, _ = v2.kinematic_measurements(qpos, prepared, robot.model)
    # Rename the inherited descriptive counter: rate arms DO enforce output speed.
    measures.pop("velocity_note")
    strict_count = measures.pop("joint_intervals_above_unwired_12_rad_s")
    speed = np.abs(np.diff(qpos[:, 7:], axis=0)) / .02
    tolerant_count = int(np.sum(speed > 12. + 2e-6 / .02))
    intervals = interval_statistics(qpos, validate_hinge_layout(robot.model), frame_indices=result.frame_indices)
    if (intervals["interval_count"] != len(qpos) - 1
            or intervals["exact_exceedance_joint_intervals"] != strict_count
            or intervals["tolerance_exceedance_joint_intervals"] != tolerant_count):
        raise ValueError("Independent named-joint interval accounting mismatch")
    if arm in ("rate_only", "both") and not intervals["accepted_with_fixed_tolerance"]:
        raise ValueError("Actual output frame speed exceeded the fixed numerical tolerance")
    audit = retargeter.v3_audit
    validate_execution_audit(audit, arm, len(qpos), result.solve_failures)
    online_intervals = None
    if arm in ("rate_only", "both"):
        online_intervals = retargeter._output_rate_limit.statistics()
        if online_intervals != intervals:
            raise ValueError("Online limit completion evidence differs from offline motion audit")
    report = {"schema": SCHEMA, "result": "COMPLETE_STAGE2_NOT_QUALITY_ACCEPTED", "case": case_index,
              "origin_id": case["origin_id"], "arm": arm, "fps": 50, "frames": len(qpos),
              "point_error_m": v2.statistics(result.point_error), "normal_error_rad": v2.statistics(result.normal_error),
              **measures, "output_rate_audit": intervals, "strict_joint_intervals_gt_12_rad_s": strict_count,
              "joint_intervals_gt_12_plus_tolerance_rad_s": tolerant_count,
              "rate_limit_enforced": arm in ("rate_only", "both"), "control_reproduction": control_difference,
              "warmup_failures": audit["warmup_failures"], "output_solve_failures": int(result.solve_failures),
              "runtime_audit": audit, "online_output_rate_audit": online_intervals,
              "source": prepared["metadata"], "prepared_source_sha256": case["prepared_sha256"],
              "raw_source_sha256": prepared["metadata"]["source_sha256"],
              "reused_setup_sha256": {k: case["input_sha256"][case[k]] for k in ("setup_bodies", "setup_correspondence")},
              "config": cfg, "scale": human.scale, "ground_offset": human.ground_offset,
              "environment": core.environment_packages(("numpy", "scipy", "mujoco", "mink", "torch", "qpsolvers", "clarabel", "trimesh")),
              "policy_ppo_updates": 0, "stage_i_epochs": 0, "physics_stepped": False, "automatic_promotion": False,
              "inputs_sha256": frozen.hashes(), "inputs_verified_unchanged": False}
    frozen.recheck()
    core.verify_umr_checkout(v2.UMR)
    report["inputs_verified_unchanged"] = True
    names = [mujoco.mj_id2name(robot.model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(1, robot.model.njnt)]
    core.save_npz_exclusive(directory / "motion.npz", qpos=qpos, dof_names=np.array(names), fps=50.,
        frame_indices=result.frame_indices, point_error=result.point_error, normal_error=result.normal_error,
        contact_count=result.contact_count, floor_rows=result.floor_rows, metadata_json=json.dumps(report, sort_keys=True))
    report["motion_sha256"] = core.sha256(directory / "motion.npz")
    report["scalebfm_output"] = core.export_scalebfm_motion(directory / "scalebfm_motion.pkl", qpos, 50., names)
    v2.write_json(directory / "receipt.json", report)
    print(json.dumps({k: report[k] for k in ("result", "origin_id", "arm", "frames", "output_solve_failures")}))
    return report


def execute(plan, frozen):
    unit = unit_guard(plan["run_id"])
    directory = v2.new_directory(plan["output"])
    frozen.recheck()
    directory.mkdir()
    inputs = directory / "inputs.json"
    v2.write_json(inputs, plan)
    inputs_sha = core.sha256(inputs)
    status = {"schema": SCHEMA, "result": "RUNNING", "run_id": plan["run_id"], "unit": unit,
              "inputs": str(inputs), "inputs_sha256": inputs_sha, "jobs_complete": 0, "jobs": [],
              "policy_ppo_updates": 0, "stage_i_epochs": 0, "physics_stepped": False,
              "automatic_promotion": False, "inputs_verified_unchanged": False}
    status_path = directory / "status.json"
    v2.write_json(status_path, status)
    start = time.monotonic()
    try:
        for case in plan["cases"]:
            case_dir = directory / f"case_{case['index']}"
            case_dir.mkdir()
            for arm in ARMS:
                v2.run_child([str(v2.PY_UMR), "-B", str(Path(__file__).resolve()), "--worker", "--inputs", str(inputs),
                    "--inputs-sha256", inputs_sha, "--case", str(case["index"]), "--arm", arm],
                    case_dir / f"{arm}.log", 180)
                receipt_path = case_dir / arm / "receipt.json"
                receipt = json.loads(receipt_path.read_text())
                if (receipt["result"] != "COMPLETE_STAGE2_NOT_QUALITY_ACCEPTED" or receipt["arm"] != arm
                        or receipt["origin_id"] != case["origin_id"] or receipt["frames"] != case["native_frames"]
                        or receipt["inputs_verified_unchanged"] is not True
                        or receipt["motion_sha256"] != core.sha256(case_dir / arm / "motion.npz")):
                    raise ValueError("Actual child receipt/motion mismatch")
                status["jobs"].append({"case": case["index"], "arm": arm, "receipt": str(receipt_path),
                                       "receipt_sha256": core.sha256(receipt_path)})
                status["jobs_complete"] += 1
                v2.write_json(status_path, status, replace=True)
        frozen.recheck()
        if core.sha256(inputs) != inputs_sha:
            raise ValueError("Immutable run plan changed")
        status.update(result="COMPLETE_STAGE2_MATRIX_NOT_QUALITY_ACCEPTED", inputs_verified_unchanged=True)
    except Exception as error:
        status.update(result="ERROR", error=f"{type(error).__name__}: {error}")
    finally:
        status["elapsed_seconds"] = time.monotonic() - start
        status["output_sha256"] = {str(p): core.sha256(p) for p in sorted(directory.rglob("*"))
                                   if p.is_file() and p != status_path}
        v2.write_json(status_path, status, replace=True)
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--inputs-sha256")
    parser.add_argument("--case", type=int)
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args(argv)
    if args.worker:
        if args.execute or args.run_id is not None or None in (args.inputs, args.inputs_sha256, args.case, args.arm):
            parser.error("Worker requires only --inputs --inputs-sha256 --case --arm")
        report = worker(args.inputs, args.inputs_sha256, args.case, args.arm)
    else:
        if args.run_id is None or any(v is not None for v in (args.inputs, args.inputs_sha256, args.case, args.arm)):
            parser.error("Controller requires --run-id and optional --execute")
        plan, frozen = build_plan(args.run_id)
        report = execute(plan, frozen) if args.execute else plan
        if args.execute:
            print(json.dumps(report, indent=2))
        else:
            print(json.dumps(plan, indent=2))
    return int(report["result"] == "ERROR")


if __name__ == "__main__":
    raise SystemExit(main())
