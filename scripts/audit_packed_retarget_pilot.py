#!/usr/bin/env python3
"""Audit actual 50 Hz candidate archives without IsaacLab imports or stepping.

Packed joint_pos is in articulation order, not the retarget source-name order.
Derive and prove its unique mapping over the entire same-origin pilot against
the packer's exact float32 source-joint interpolation before running any FK.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import retarget_data_refresh as refresh
from scripts.amass_to_scalebfm import semantic_fingerprint


def static_joint_names(path):
    """Read the literal source order without importing IsaacLab robot configs."""
    tree = ast.parse(Path(path).read_text())
    values = [ast.literal_eval(node.value) for node in tree.body
              if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id == "G1_29DOF_JOINT_NAMES"
                      for target in node.targets)]
    if len(values) != 1 or not isinstance(values[0], list):
        raise ValueError("Expected exactly one literal G1_29DOF_JOINT_NAMES list")
    names = values[0]
    if len(names) != 29 or len(set(names)) != 29 or not all(isinstance(name, str) for name in names):
        raise ValueError("Expected 29 unique named G1 source joints")
    return names


def interpolated_source_joints(payload):
    """Mirror package_motions.py float32 time grid and linear joint interpolation."""
    import torch

    qpos, fps = refresh.kinematic_qpos(payload)
    dof = torch.from_numpy(qpos[:, 7:]).float()
    frames = len(dof)
    duration = (frames - 1) / fps
    times = torch.arange(0, duration, 0.02, dtype=torch.float32)
    phase = times / duration
    lower = (phase * (frames - 1)).floor().long()
    upper = torch.minimum(lower + 1, torch.tensor(frames - 1))
    blend = phase * (frames - 1) - lower
    return (dof[lower] * (1 - blend[:, None]) + dof[upper] * blend[:, None]).numpy()


def accumulate_order_errors(errors, packed_joints, source_joints):
    if packed_joints.shape != source_joints.shape or packed_joints.ndim != 2 or packed_joints.shape[1] != 29:
        raise ValueError("Packed/source frame counts or joint dimensions differ")
    if not np.isfinite(packed_joints).all() or not np.isfinite(source_joints).all():
        raise ValueError("Nonfinite joint positions in order proof")
    return np.maximum(errors, np.max(np.abs(packed_joints[:, :, None] - source_joints[:, None, :]), axis=0))


def unique_joint_order(errors, *, atol=2e-6):
    matches = errors <= atol
    if errors.shape != (29, 29) or np.any(matches.sum(axis=0) != 1) or np.any(matches.sum(axis=1) != 1):
        raise ValueError("Packed/source joint mapping is not uniquely proven")
    return np.argmax(matches, axis=1)


def load_packed(path, expected_source_sha256):
    with np.load(path, allow_pickle=False) as archive:
        for name, expected in (("format_version", 3), ("fps", 50), ("quaternion_order", "wxyz")):
            if archive[name].shape != () or archive[name].item() != expected:
                raise ValueError(f"Unexpected packed {name}: {path}")
        if archive["source_sha256"].shape != () or archive["source_sha256"].item() != expected_source_sha256:
            raise ValueError(f"Packed source PKL SHA256 mismatch: {path}")
        fingerprint = archive["pipeline_fingerprint"].item()
        if not isinstance(fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
            raise ValueError("Invalid packed pipeline fingerprint")
        return {name: archive[name].copy() for name in
                ("joint_pos", "reference_root_pos", "reference_root_quat_w")} | {
                    "pipeline_fingerprint": fingerprint}


def packed_qpos(archive, packed_names, model_names):
    if len(set(packed_names)) != 29 or set(packed_names) != set(model_names):
        raise ValueError("Packed and MuJoCo named joint sets differ")
    joint_pos = archive["joint_pos"][:, [packed_names.index(name) for name in model_names]]
    return refresh.kinematic_qpos({"fps": 50, "dof_pos": joint_pos,
        "root_pos": archive["reference_root_pos"],
        "root_rot": archive["reference_root_quat_w"][:, [1, 2, 3, 0]]})[0]


def validate_pair(before, after, shift, *, atol=2e-6):
    if before.shape != after.shape:
        raise ValueError("Baseline/candidate frame counts or qpos dimensions differ")
    if not np.isfinite(shift):
        raise ValueError("Nonfinite native constant Z shift")
    xy_error = float(np.max(np.abs(after[:, :2] - before[:, :2])))
    joint_error = float(np.max(np.abs(after[:, 7:] - before[:, 7:])))
    quat_error = float(np.max(np.minimum(
        np.max(np.abs(after[:, 3:7] - before[:, 3:7]), axis=1),
        np.max(np.abs(after[:, 3:7] + before[:, 3:7]), axis=1))))
    z_error = float(np.max(np.abs(after[:, 2] - before[:, 2] - shift)))
    if max(xy_error, joint_error, quat_error, z_error) > atol:
        raise ValueError("Packed candidate changed non-Z fields or lost native constant shift")
    return {"root_xy_max_abs_error_m": xy_error, "joint_max_abs_error_rad": joint_error,
            "sign_invariant_root_quat_max_abs_error": quat_error,
            "constant_z_shift_max_abs_error_m": z_error, "float32_tolerance": atol}


def run_audit(pilot, packed_root, output):
    import joblib
    import mujoco

    if output.exists() or output.is_symlink():
        raise ValueError("Refusing existing packed audit output")
    manifest_path, native_path = pilot / "manifest.json", pilot / "sole_audit.json"
    manifest = json.loads(manifest_path.read_text())
    native = json.loads(native_path.read_text())
    if manifest.get("automatic_promotion") is not False or native.get("automatic_promotion") is not False:
        raise ValueError("Expected non-promoting pilot artifacts")
    native_rows = {row["origin_id"]: row for row in native["results"]}
    if len(native_rows) != len(native["results"]) or set(native_rows) != {r["origin_id"] for r in manifest["motions"]}:
        raise ValueError("Pilot manifest/native audit origin sets differ or duplicate")
    robot_xml = ROOT / "ScaleRetarget/assets/unitree_g1/g1_mocap_29dof.xml"
    names_path = ROOT / "ScaleTrack/source/scaletrack/scaletrack/robots/g1_29dof.py"
    geometry_hash = semantic_fingerprint({"robot_geometry": robot_xml.parent}, {})
    if geometry_hash != native["robot_geometry_tree_sha256"]:
        raise ValueError("Robot geometry changed since native pilot audit")
    source_names = static_joint_names(names_path)
    model = mujoco.MjModel.from_xml_path(str(robot_xml))
    model_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(1, model.njnt)]
    if len(model_names) != 29 or set(model_names) != set(source_names):
        raise ValueError("Source/model named joint sets differ")
    snapshots = {str(path): refresh.sha256(path) for path in
                 (manifest_path, native_path, names_path, Path(__file__), Path(refresh.__file__))}
    errors = np.zeros((29, 29))
    loaded = []
    for row in manifest["motions"]:
        proof = refresh.resolve_inputs(row)
        for key in ("source_sha256", "baseline_pkl_sha256", "baseline_packed_sha256"):
            if proof[key] != row[key]:
                raise ValueError(f"Pilot baseline provenance changed: {row['origin_id']}")
        native_row = native_rows[row["origin_id"]]
        if native_row["split"] != row["split"] or native_row["source_sha256"] != row["source_sha256"]:
            raise ValueError("Native candidate origin/split/source binding mismatch")
        candidate_pkl = Path(native_row["candidate_pkl"]).resolve(strict=True)
        expected_pkl = (pilot / "sole_candidates" / f"{row['origin_id']}.pkl").resolve(strict=True)
        if candidate_pkl != expected_pkl or refresh.sha256(candidate_pkl) != native_row["candidate_sha256"]:
            raise ValueError("Candidate PKL location/content differs from native audit")
        candidate_npz = (packed_root / f"{row['origin_id']}.npz").resolve(strict=True)
        candidate_npz.relative_to(packed_root.resolve(strict=True))
        baseline = load_packed(row["baseline_packed"], row["baseline_pkl_sha256"])
        candidate = load_packed(candidate_npz, native_row["candidate_sha256"])
        expected = interpolated_source_joints(joblib.load(row["baseline_pkl"]))
        errors = accumulate_order_errors(errors, baseline["joint_pos"], expected)
        for path in (Path(row["source"]), Path(row["baseline_pkl"]), Path(row["baseline_packed"]),
                     candidate_pkl, candidate_npz):
            snapshots[str(path)] = refresh.sha256(path)
        loaded.append((row, native_row, candidate_npz, baseline, candidate))
    refresh.validate_source_disjoint(manifest["motions"])
    order = unique_joint_order(errors)
    packed_names = [source_names[i] for i in order]
    results = []
    for row, native_row, candidate_npz, baseline, candidate in loaded:
        before = packed_qpos(baseline, packed_names, model_names)
        after = packed_qpos(candidate, packed_names, model_names)
        same_motion = validate_pair(before, after, native_row["constant_z_shift_m"])
        initial, final = refresh.audit_qpos(before, model), refresh.audit_qpos(after, model)
        reasons = list(native_row["review_reasons"])
        if final["foot_penetration_frames_gt_1mm"]:
            reasons.append("packed_50hz_foot_penetration_gt_1mm")
        if final["max_all_body_ground_penetration_m"] > 0.005:
            reasons.append("packed_50hz_body_ground_penetration_gt_5mm")
        if final["max_self_penetration_m"] > 0.005:
            reasons.append("packed_50hz_self_penetration_gt_5mm")
        if final["joint_limit_violations_gt_1e_minus6_rad"]:
            reasons.append("packed_50hz_joint_limit_violation")
        results.append({"origin_id": row["origin_id"], "split": row["split"],
            "source_sha256": row["source_sha256"], "baseline": initial, "candidate": final,
            "same_motion_proof": same_motion, "native_status": native_row["status"],
            "baseline_packed": row["baseline_packed"], "candidate_packed": str(candidate_npz),
            "baseline_package_fingerprint": baseline["pipeline_fingerprint"],
            "candidate_package_fingerprint": candidate["pipeline_fingerprint"],
            "review_reasons": reasons,
            "status": "REJECT_PACKED_KINEMATIC" if reasons else "PENDING_FROZEN_POLICY_EVALUATION"})
    if semantic_fingerprint({"robot_geometry": robot_xml.parent}, {}) != geometry_hash:
        raise ValueError("Robot geometry changed during packed audit")
    for path, digest in snapshots.items():
        if refresh.sha256(path) != digest:
            raise ValueError(f"Input changed during audit: {path}")
    payload = {"schema": 1, "fps": 50, "result": "COMPLETE", "automatic_promotion": False,
        "training_started": False, "physics_stepped": False,
        "interpretation": "Actual packed 50 Hz kinematic collision audit; not learned-policy or contact-dynamics acceptance",
        "robot_geometry_tree_sha256": geometry_hash, "mujoco_version": mujoco.__version__,
        "input_sha256": snapshots,
        "joint_order_proof": {"method": "unique all-clip float32 source-interpolation match",
            "source_joint_names": source_names, "packed_joint_names": packed_names,
            "packed_to_source_indices": order.tolist(), "maximum_matching_error_rad": float(errors[np.arange(29), order].max()),
            "all_columns_uniquely_proven": True, "frames": sum(len(item[3]["joint_pos"]) for item in loaded)},
        "thresholds": {"source_joint_mapping_atol": 2e-6, "same_motion_atol": 2e-6,
            "foot_penetration_m": 0.001, "self_or_body_ground_penetration_m": 0.005},
        "results": results, "counts": dict(Counter(row["status"] for row in results))}
    refresh.write_json(output, payload)
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--packed-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if "bfm-data-refresh-" not in Path("/proc/self/cgroup").read_text():
        parser.error("Run within owned bfm-data-refresh-*.service cgroup")
    result = run_audit(args.pilot.resolve(strict=True), args.packed_root.resolve(strict=True), args.output)
    print(json.dumps({"output": str(args.output), "counts": result["counts"],
                      "packed_frames": result["joint_order_proof"]["frames"], "training_started": False}))


if __name__ == "__main__":
    main()
