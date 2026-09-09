#!/usr/bin/env python3
"""Check all saved G1 link poses against independent named-joint MuJoCo FK.

Requires the complete forty-origin/eighty-archive pilot, never fits transforms,
and never steps physics or starts policy training. Run in a bounded owned user
service; numeric failures are retained in a new report with a nonzero exit.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import evaluate_umr_pairs as pairs
from scripts import evaluate_retarget_pilot as archives
from scripts.audit_packed_retarget_pilot import packed_qpos
from scripts.amass_to_scalebfm import semantic_fingerprint
from scripts.umr_smplx_source import sha256

POSITION_TOLERANCE_M = 1e-4
ROTATION_TOLERANCE_RAD = 1e-4


def owned_unit():
    matches = re.findall(r"(?:^|/)(bfm-umr-fk-[A-Za-z0-9_.-]+\.service)(?:/|$)",
                         Path("/proc/self/cgroup").read_text(), flags=re.MULTILINE)
    if len(matches) != 1:
        raise ValueError("Run inside a bounded bfm-umr-fk-*.service user cgroup")
    output = subprocess.check_output(["systemctl", "--user", "show", matches[0],
        "-p", "KillMode", "-p", "Restart", "-p", "RuntimeMaxUSec", "-p", "MemoryMax", "-p", "TasksMax"], text=True)
    values = dict(line.split("=", 1) for line in output.splitlines())
    if (values.get("KillMode") != "control-group" or values.get("Restart") != "no"
            or any(values.get(key) in (None, "", "0", "infinity")
                   for key in ("RuntimeMaxUSec", "MemoryMax", "TasksMax"))):
        raise ValueError("Owned FK service needs control-group cleanup and finite resource bounds")
    return matches[0]


def quaternion_angle(first, second):
    """SO(3) radians via normalized, sign-invariant float64 chord atan2.

    The unit-quaternion chord angle is half the physical rotation angle. This
    avoids acos amplifying float32 quaternion norm error near identity.
    """
    first, second = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if first.shape != second.shape or first.ndim < 1 or first.shape[-1] != 4:
        raise ValueError("Quaternion shapes must agree and end in four components")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("Nonfinite quaternion")
    norms = [np.linalg.norm(value, axis=-1, keepdims=True) for value in (first, second)]
    if any(np.any(value < 1e-12) for value in norms):
        raise ValueError("Zero quaternion")
    first, second = first / norms[0], second / norms[1]
    second = np.where(np.sum(first * second, axis=-1, keepdims=True) < 0, -second, second)
    return 4. * np.arctan2(np.linalg.norm(first - second, axis=-1),
                          np.linalg.norm(first + second, axis=-1))


def error_statistics(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Expected nonempty finite nonnegative errors")
    return {"max": float(values.max()), "p95": float(np.percentile(values, 95)),
            "rms": float(np.sqrt(np.mean(values ** 2)))}


def pose_errors(expected_position, expected_quaternion, actual_position, actual_quaternion):
    expected_position, actual_position = map(lambda x: np.asarray(x, dtype=np.float64),
                                              (expected_position, actual_position))
    if (expected_position.shape != actual_position.shape or expected_position.ndim != 3
            or expected_position.shape[1:] != (30, 3)
            or not np.isfinite(expected_position).all() or not np.isfinite(actual_position).all()):
        raise ValueError("Positions must be finite matching (frames,30,3) link arrays")
    angle = quaternion_angle(expected_quaternion, actual_quaternion)
    if angle.shape != expected_position.shape[:2]:
        raise ValueError("Position/quaternion link counts differ")
    return np.linalg.norm(expected_position - actual_position, axis=-1), angle


def worst_error(values, names):
    frame, body = np.unravel_index(np.argmax(values), values.shape)
    return {"frame_index": int(frame), "body_name": names[body], "value": float(values[frame, body])}


def remember(path, frozen, expected=None):
    path = Path(path).absolute()
    if any(parent.is_symlink() for parent in (path, *path.parents)) or not path.is_file():
        raise ValueError(f"Expected a regular nonredirected input: {path}")
    digest = sha256(path)
    if expected is not None and (not isinstance(expected, str)
            or not re.fullmatch(r"[0-9a-f]{64}", expected) or digest != expected):
        raise ValueError(f"Input SHA mismatch: {path}")
    if str(path) in frozen and frozen[str(path)] != digest:
        raise ValueError(f"Input changed while validating: {path}")
    frozen[str(path)] = digest
    return path


def validate_inputs(manifest_path, packed_root, frozen):
    manifest_path = remember(manifest_path, frozen)
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("schema") != 1 or manifest.get("training_started") is not False
            or manifest.get("automatic_promotion") is not False):
        raise ValueError("Expected a non-promoting complete pilot manifest")
    rows = manifest.get("rows", [])
    if not isinstance(rows, list) or len(rows) != 40 or len({r["origin_id"] for r in rows}) != 40:
        raise ValueError("Require exactly forty distinct preselected origins")
    selection_path = remember(manifest["selection"], frozen, manifest["selection_sha256"])
    if (selection_path != pairs.ORIGINAL_SELECTION or frozen[str(selection_path)] != pairs.ORIGINAL_SELECTION_SHA):
        raise ValueError("Expected the pinned original forty-origin selection")
    selected = json.loads(selection_path.read_text())["motions"]
    if [r["origin_id"] for r in rows] != [r["origin_id"] for r in selected]:
        raise ValueError("Original complete origin order changed")
    sources = pairs.helpers()[0].SOURCES
    counts = Counter(f"{r['dataset']}/{r['split']}" for r in rows)
    if counts != {f"{source}/{split}": n for source in sources for split, n in (("train", 6), ("validation", 2))}:
        raise ValueError("Original dataset/train/development allocation changed")
    packed_root = Path(packed_root).absolute()
    if any(p.is_symlink() for p in (packed_root, *packed_root.parents)) or not packed_root.is_dir():
        raise ValueError("Expected a real packed directory")
    expected = {packed_root / side / f"{r['origin_id']}.npz" for r in rows for side in pairs.SIDES}
    if set(packed_root.rglob("*.npz")) != expected:
        raise ValueError("Require exactly all eighty packed archives without extras")
    checked, pipelines = [], set()
    for row, original in zip(rows, selected):
        pairs._safe_origin(row["origin_id"], row["dataset"], sources)
        if any(row.get(field) != original.get(field) for field in ("source_sha256", "split", "dataset", "index_label")):
            raise ValueError("Original source identity or split changed")
        frames = row["expected_packed_frames"]
        if type(frames) is not int or not 2 <= frames <= 250:
            raise ValueError("Invalid packed frame count")
        for side in pairs.SIDES:
            native = remember(row[f"{side}_pkl"], frozen, row[f"{side}_pkl_sha256"])
            archive = remember(packed_root / side / f"{row['origin_id']}.npz", frozen)
            pipelines.add(archives.validate_archive(archive, native_sha=frozen[str(native)], frames=frames))
            names = pairs._archive_names(archive)
            checked.append({"origin_id": row["origin_id"], "dataset": row["dataset"], "split": row["split"],
                            "side": side, "path": str(archive), "sha256": frozen[str(archive)],
                            "frames": frames, **names})
    if len(pipelines) != 1:
        raise ValueError("All archives must share one packaging fingerprint")
    return manifest, checked, pipelines.pop()


def run_audit(manifest_path, packed_root, output):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    report = {"schema": "bfm.umr_packed_body_fk/1", "result": "ERROR", "physics_stepped": False,
              "training_updates": 0, "automatic_promotion": False, "inputs_verified_unchanged": False,
              "thresholds": {"position_m": POSITION_TOLERANCE_M, "rotation_rad": ROTATION_TOLERANCE_RAD},
              "interpretation": "Independent saved-link FK consistency only; not physical tracking/contact/task acceptance.",
              "input_sha256": {}, "results": []}
    frozen = report["input_sha256"]
    try:
        for name in ("audit_umr_body_fk.py", "audit_packed_retarget_pilot.py", "evaluate_umr_pairs.py",
                     "evaluate_retarget_pilot.py", "retarget_data_refresh.py", "amass_to_scalebfm.py",
                     "umr_smplx_source.py", "umr_backend.py"):
            remember(ROOT / "scripts" / name, frozen)
        remember(pairs.ROBOT_XML, frozen)
        geometry = semantic_fingerprint({"robot_geometry": pairs.ROBOT_XML.parent}, {})
        manifest, checked, pipeline = validate_inputs(manifest_path, packed_root, frozen)
        report.update(experiment_id=manifest["experiment_id"], manifest_sha256=frozen[str(Path(manifest_path).absolute())],
                      packaging_fingerprint=pipeline, geometry_tree_sha256=geometry)
        import mujoco
        model = mujoco.MjModel.from_xml_path(str(pairs.ROBOT_XML))
        if model.nq != 36 or not np.array_equal(model.jnt_qposadr[1:], np.arange(7, 36)):
            raise ValueError("Unexpected G1 model qpos layout")
        model_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(1, model.njnt)]
        data = mujoco.MjData(model)
        all_positions, all_angles = [], []
        for row in checked:
            ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in row["body_names"]]
            if min(ids) <= 0 or len(set(ids)) != 30:
                raise ValueError("Missing or duplicate named G1 link in FK model")
            with np.load(row["path"], allow_pickle=False) as archive:
                qpos = packed_qpos(archive, row["joint_names"], model_names)
                positions = np.empty((len(qpos), 30, 3)); quaternions = np.empty((len(qpos), 30, 4))
                for frame, pose in enumerate(qpos):
                    data.qpos[:] = pose
                    mujoco.mj_forward(model, data)  # FK/contact calculation only, never mj_step.
                    positions[frame], quaternions[frame] = data.xpos[ids], data.xquat[ids]
                errors, angles = pose_errors(positions, quaternions, archive["body_pos_w"], archive["body_quat_w"])
            position_stats, rotation_stats = error_statistics(errors), error_statistics(angles)
            passed = position_stats["max"] <= POSITION_TOLERANCE_M and rotation_stats["max"] <= ROTATION_TOLERANCE_RAD
            report["results"].append({**row, "position_error_m": position_stats, "rotation_error_rad": rotation_stats,
                "worst_position": worst_error(errors, row["body_names"]),
                "worst_rotation": worst_error(angles, row["body_names"]), "passed": passed})
            all_positions.append(errors.ravel()); all_angles.append(angles.ravel())
        pairs.verify_frozen(frozen)
        if semantic_fingerprint({"robot_geometry": pairs.ROBOT_XML.parent}, {}) != geometry:
            raise ValueError("Robot geometry changed during FK audit")
        passed = sum(row["passed"] for row in report["results"])
        report.update(result="PASS" if passed == 80 else "FAIL", inputs_verified_unchanged=True,
            mujoco_version=mujoco.__version__, archive_count=80, passed_archives=passed,
            total_frames=sum(row["frames"] for row in checked),
            aggregate={"position_error_m": error_statistics(np.concatenate(all_positions)),
                       "rotation_error_rad": error_statistics(np.concatenate(all_angles))})
    except Exception as error:
        report.update(result="ERROR", error=f"{type(error).__name__}: {error}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "packed-root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    owned_unit()
    report = run_audit(args.manifest, args.packed_root, args.output)
    print(json.dumps({"result": report["result"], "output": str(args.output),
                      "passed_archives": report.get("passed_archives"), "error": report.get("error")}))
    return 0 if report["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
