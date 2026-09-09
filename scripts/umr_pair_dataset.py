#!/usr/bin/env python3
"""Prepare immutable, same-source 50 Hz windows for the UMR data pilot.

This compares whole retargeting pipelines, not solvers in isolation: the legacy
pipeline uses optimized neutral shape and sparse IK; the candidate uses the
actor's SMPL-X shape and material surface UMR. No data promotion or training is
performed. Native PKLs are trusted local artifacts and are loaded with joblib.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import audit_packed_retarget_pilot as packed_tools
from scripts import umr_smplx_source as surface
from scripts.umr_backend import G1_JOINT_NAMES, convert_umr_payload, sha256

SCHEMA = "bfm.umr_same_source_pair/1"
FPS = 50.0


def _json_member(archive, name):
    return json.loads(surface.scalar_text(archive[name], name))


def legacy_source_clock(source_count, source_fps, native_count, native_fps):
    """Independently derive the historical endpoint-preserving 30 Hz clock."""
    if source_count < 3 or not np.isfinite(source_fps) or source_fps <= 0:
        raise ValueError("Invalid original source clock")
    duration = (source_count - 1) / source_fps
    if np.isclose(source_fps, 30.0):
        expected_count, expected_fps = source_count, source_fps
    else:
        expected_count = max(2, round(duration * 30.0) + 1)
        expected_fps = (expected_count - 1) / duration
    if native_count != expected_count or not np.isclose(native_fps, expected_fps, rtol=0, atol=1e-10):
        raise ValueError("Native PKL does not prove the original endpoint-preserving 30 Hz clock")
    times = np.arange(native_count, dtype=np.float64) / native_fps
    if not np.isclose(times[-1], duration, rtol=0, atol=1e-10):
        raise ValueError("Native endpoint differs from original source endpoint")
    return times, {
        "source_frames": source_count, "source_fps": source_fps,
        "source_endpoint_s": duration, "native_frames": native_count,
        "native_nominal_target_fps": 30.0, "native_actual_fps": native_fps,
        "native_clock_rule": "endpoint-preserving linspace, round(duration * 30) + 1",
        "native_source_frame_zero": 0,
    }


def package_interpolation(payload):
    """Reproduce the packer's float32 clock and position/joint interpolation.

    Quaternion interpolation is independent ideal shortest-path SLERP. The
    archived runtime's near-identical-dot early return is bounded in the proof.
    Nothing is imported from IsaacLab or silently repaired in the baseline.
    """
    import torch

    qpos, fps = packed_tools.refresh.kinematic_qpos(payload)
    frames = len(qpos)
    duration = (frames - 1) / fps
    times = torch.arange(0, duration, .02, dtype=torch.float32)
    fractional = times / duration * (frames - 1)
    lower = fractional.floor().long()
    upper = torch.minimum(lower + 1, torch.tensor(frames - 1))
    alpha = fractional - lower
    values = torch.from_numpy(np.concatenate((qpos[:, :3], qpos[:, 7:]), axis=1)).float()
    interp = (values[lower] * (1 - alpha[:, None]) + values[upper] * alpha[:, None]).numpy()
    a, b = qpos[lower.numpy(), 3:7], qpos[upper.numpy(), 3:7]
    dot = np.sum(a * b, axis=1, keepdims=True)
    b = np.where(dot < 0, -b, b)
    theta = np.arccos(np.clip(np.abs(dot), 0, 1))
    near = np.sin(theta) < 1e-7
    denominator = np.where(near, 1., np.sin(theta))
    blend = alpha.numpy()[:, None]
    quat = np.where(near, (1 - blend) * a + blend * b,
                    np.sin((1 - blend) * theta) / denominator * a
                    + np.sin(blend * theta) / denominator * b)
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    return {"times": times.numpy(), "lower": lower.numpy(), "upper": upper.numpy(),
            "alpha": alpha.numpy(), "root_pos": interp[:, :3],
            "dof_pos": interp[:, 3:], "root_quat_wxyz": quat}


def verified_packed_order(audit, row, names_path):
    proof = audit.get("joint_order_proof", {})
    source_names = packed_tools.static_joint_names(names_path)
    if source_names != list(G1_JOINT_NAMES):
        raise ValueError("Static native G1 joint order changed")
    order = np.asarray(proof.get("packed_to_source_indices", []))
    if (proof.get("all_columns_uniquely_proven") is not True
            or proof.get("source_joint_names") != source_names
            or order.dtype.kind not in "iu" or order.shape != (29,)
            or sorted(order.tolist()) != list(range(29))
            or not 0 <= float(proof.get("maximum_matching_error_rad", np.inf)) <= 2e-6):
        raise ValueError("Missing or invalid unique packed joint order proof")
    packed_names = [source_names[index] for index in order]
    if proof.get("packed_joint_names") != packed_names:
        raise ValueError("Packed names contradict the explicit order proof")
    snapshots = audit.get("input_sha256", {})
    for key in ("source", "baseline_pkl", "baseline_packed"):
        if snapshots.get(str(Path(row[key]).resolve())) != row[f"{key}_sha256"]:
            raise ValueError("Packed audit is not bound to this origin's exact input hashes")
    if snapshots.get(str(names_path.resolve())) != sha256(names_path):
        raise ValueError("Static native joint names differ from the packed audit")
    records = [r for r in audit.get("results", []) if r.get("origin_id") == row["origin_id"]]
    if (len(records) != 1 or records[0].get("split") != row["split"]
            or records[0].get("source_sha256") != row["source_sha256"]
            or records[0].get("baseline_packed") != row["baseline_packed"]):
        raise ValueError("Packed audit origin/split binding is absent or inconsistent")
    return packed_names


def prove_packed_values(packed, interp, packed_names):
    joints = packed["joint_pos"][:, [packed_names.index(name) for name in G1_JOINT_NAMES]]
    root, quat = packed["reference_root_pos"], packed["reference_root_quat_w"]
    count = len(interp["times"])
    if joints.shape != (count, 29) or root.shape != (count, 3) or quat.shape != (count, 4):
        raise ValueError("Packed arrays do not have the independently derived clock's frame count")
    if not all(np.isfinite(value).all() for value in (joints, root, quat)):
        raise ValueError("Nonfinite packed motion")
    norms = np.linalg.norm(quat, axis=1, keepdims=True)
    if not np.allclose(norms, 1., atol=1e-5, rtol=0):
        raise ValueError("Packed root quaternions are not unit length")
    angular_error = 2. * np.arccos(np.clip(np.abs(np.sum(
        quat / norms * interp["root_quat_wxyz"], axis=1)), 0., 1.))
    errors = {
        "joint_max_abs_error_rad": float(np.max(np.abs(joints - interp["dof_pos"]))),
        "root_position_max_abs_error_m": float(np.max(np.abs(root - interp["root_pos"]))),
        "quaternion_sign_invariant_max_abs_error": float(np.max(np.minimum(
            np.max(np.abs(quat - interp["root_quat_wxyz"]), axis=1),
            np.max(np.abs(quat + interp["root_quat_wxyz"]), axis=1)))),
    }
    if max(errors["joint_max_abs_error_rad"], errors["root_position_max_abs_error_m"]) > 2e-6:
        raise ValueError("Packed motion fails native float32 position/joint interpolation proof")
    # Legacy scalar helper returns q1 for |dot| within 4 float32 eps of 1.
    # Its maximum skipped rotation is 2*acos(1-4*eps) ~= 0.001953 rad.
    # Add float32 normalization/dot rounding headroom, not a fitted tolerance.
    errors["quaternion_ideal_slerp_max_angle_error_rad"] = float(angular_error.max())
    errors["quaternion_legacy_near_identical_angle_tolerance_rad"] = .0022
    if errors["quaternion_ideal_slerp_max_angle_error_rad"] > .0022:
        raise ValueError("Packed quaternion fails native shortest-path interpolation proof")
    return np.concatenate((root, quat, joints), axis=1), errors


def common_grid(prepared, raw_count, raw_fps, packed_times):
    """Join actual clocks by fixed-rate sample index, never by minimum length."""
    meta = prepared["metadata"]
    if float(meta["start"]) != 0 or float(meta["target_fps"]) != FPS:
        raise ValueError("This pilot requires a source-frame-zero 50 Hz candidate")
    expected = surface.sampling_grid(raw_count, raw_fps, FPS, 0., float(meta["requested_duration"]))
    for name, value in zip(("times", "sample_lower", "sample_upper", "sample_alpha"), expected):
        actual = prepared[name]
        if actual.shape != value.shape or not np.allclose(actual, value, atol=1e-10, rtol=0):
            raise ValueError(f"Prepared source fails original source sampling proof: {name}")
    if not np.isclose(float(meta["source_fps"]), raw_fps, atol=1e-10, rtol=0):
        raise ValueError("Prepared source rate differs from original source")
    packed_indices = np.rint(np.asarray(packed_times, dtype=np.float64) * FPS).astype(np.int64)
    if not np.array_equal(packed_indices, np.arange(len(packed_times))):
        raise ValueError("Packed clock is not contiguous from original frame zero")
    if not np.allclose(packed_times, packed_indices / FPS, atol=1e-5, rtol=0):
        raise ValueError("Packed float32 clock differs from the real 50 Hz grid")
    candidate_indices = np.rint(expected[0] * FPS).astype(np.int64)
    common = np.intersect1d(packed_indices, candidate_indices, assume_unique=True)
    if len(common) < 3 or not np.array_equal(common, np.arange(len(common))):
        raise ValueError("Insufficient contiguous same-origin paired window")
    if common[-1] / FPS > (raw_count - 1) / raw_fps + 1e-10:
        raise ValueError("Paired window exceeds the original source endpoint")
    return common, {
        "method": "intersection of independently proven original-source 50 Hz sample indices",
        "pair_start_s": 0., "pair_last_sample_s": float(common[-1] / FPS),
        "pair_frames": len(common), "packed_frames_before_window": len(packed_times),
        "candidate_frames_before_window": len(candidate_indices),
        "packed_clock_rule": "torch.arange(0, native_endpoint, 0.02, dtype=float32), endpoint exclusive",
        "candidate_clock_rule": "start + arange(floor(real_duration * 50) + 1) / 50, endpoint inclusive",
        "packed_float32_time_max_error_s": float(np.max(np.abs(packed_times - packed_indices / FPS))),
        "dropped_packed_frames": len(packed_times) - len(common),
        "dropped_candidate_frames": len(candidate_indices) - len(common),
    }


def align_pair(baseline, candidate, heading):
    """One source-derived yaw, then independent constant initial XY origins.

    This preserves both trajectories' motion, scale, relative yaw and native Z.
    It does not fit the baseline to UMR, align each frame, or remove drift.
    """
    from scipy.spatial.transform import Rotation

    heading = np.asarray(heading, dtype=np.float64)
    if (heading.shape != (3, 3) or not np.isfinite(heading).all()
            or not np.allclose(heading @ heading.T, np.eye(3), atol=1e-8, rtol=0)
            or not np.isclose(np.linalg.det(heading), 1., atol=1e-8, rtol=0)
            or not np.allclose(heading[2], [0., 0., 1.], atol=1e-8, rtol=0)):
        raise ValueError("Prepared heading must be a proper Z-axis-only rotation")
    if baseline.shape != candidate.shape or baseline.ndim != 2 or baseline.shape[1] != 36:
        raise ValueError("Paired qpos dimensions differ")
    before, after = baseline.astype(np.float64, copy=True), candidate.astype(np.float64, copy=True)
    before[:, :3] = before[:, :3] @ heading.T
    before[:, 3:7] = (Rotation.from_matrix(heading)
                      * Rotation.from_quat(before[:, [4, 5, 6, 3]])).as_quat()[:, [3, 0, 1, 2]]
    offsets = {"baseline": -before[0, :2].copy(), "candidate": -after[0, :2].copy()}
    before[:, :2] += offsets["baseline"]
    after[:, :2] += offsets["candidate"]
    return before, after, {
        "baseline_world_rotation": heading.tolist(), "candidate_world_rotation": np.eye(3).tolist(),
        "rotation_source": "prepared SMPL-X first pelvis forward (+Z local) heading",
        "xy_rule": "each robot initial root XY to zero with one constant offset",
        "baseline_constant_xy_offset_m": offsets["baseline"].tolist(),
        "candidate_constant_xy_offset_m": offsets["candidate"].tolist(),
        "baseline_constant_z_offset_m": 0., "candidate_constant_z_offset_m": 0.,
        "per_frame_alignment": False, "shape_or_scale_fit": False,
    }


def prepare_pair(row, *, prepared_source, candidate_npz, packed_audit, output_dir):
    """Verify every binding before creating this origin's new output directory."""
    import joblib
    import torch

    output_dir = Path(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError("Refusing existing pair output directory")
    if row.get("split") not in ("train", "validation"):
        raise ValueError("Expected an immutable train or development-validation origin")
    snapshots = {}
    for key in ("source", "baseline_pkl", "baseline_packed"):
        path = Path(row[key]).resolve(strict=True)
        digest = sha256(path)
        if digest != row[f"{key}_sha256"]:
            raise ValueError(f"Manifest input changed: {key}")
        snapshots[str(path)] = digest
    prepared_source, candidate_npz, packed_audit = map(Path, (prepared_source, candidate_npz, packed_audit))
    names_path = ROOT / "ScaleTrack/source/scaletrack/scaletrack/robots/g1_29dof.py"
    for path in (prepared_source, candidate_npz, packed_audit, names_path, Path(__file__),
                 ROOT / "ScaleRetarget/scaleretarget/utils/resampling.py",
                 ROOT / "ScaleTrack/scripts/pretrain/data_process/package_motions.py"):
        snapshots[str(path.resolve(strict=True))] = sha256(path)
    audit = json.loads(packed_audit.read_text())
    packed_names = verified_packed_order(audit, row, names_path)
    packed = packed_tools.load_packed(Path(row["baseline_packed"]), row["baseline_pkl_sha256"])
    payload = joblib.load(row["baseline_pkl"])
    native_qpos, native_fps = packed_tools.refresh.kinematic_qpos(payload)
    with np.load(row["source"], allow_pickle=False) as raw:
        source_count, source_fps = len(raw["trans"]), float(raw["mocap_frame_rate"])
    native_times, clock_proof = legacy_source_clock(source_count, source_fps, len(native_qpos), native_fps)
    interp = package_interpolation(payload)
    baseline, value_proof = prove_packed_values(packed, interp, packed_names)
    prepared = surface.load_prepared_source(prepared_source)
    meta = prepared["metadata"]
    if (meta.get("source_sha256") != row["source_sha256"]
            or Path(meta["source_file"]).resolve() != Path(row["source"]).resolve()):
        raise ValueError("Prepared candidate is not this exact source origin")
    with np.load(candidate_npz, allow_pickle=False) as archive:
        candidate_meta = _json_member(archive, "metadata_json")
        candidate_motion = convert_umr_payload({key: archive[key].copy() for key in ("qpos", "fps", "dof_names")})
        frame_indices = archive["frame_indices"].copy()
    if (candidate_meta.get("schema") != "bfm.umr_smplx_trial/1"
            or candidate_meta.get("source") != meta
            or candidate_meta.get("prepared_source_sha256") != snapshots[str(prepared_source.resolve())]
            or candidate_meta.get("protected_inputs_rechecked") is not True
            or candidate_meta.get("material_surface_transport") is not True
            or candidate_meta.get("umr_commit") != surface.UMR_COMMIT):
        raise ValueError("UMR artifact does not prove its exact prepared material-surface source")
    candidate, candidate_fps = packed_tools.refresh.kinematic_qpos(candidate_motion)
    if (candidate_fps != FPS or len(candidate) != len(prepared["times"])
            or not np.array_equal(frame_indices, np.arange(len(candidate)))):
        raise ValueError("UMR artifact frame clock differs from its prepared source")
    indices, common_proof = common_grid(prepared, source_count, source_fps, interp["times"])
    before, after, alignment = align_pair(baseline[indices], candidate[indices], meta["posed_heading_rotation"])
    for path, digest in snapshots.items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"Input changed while preparing pair: {path}")
    repack_times = torch.arange(0, (len(indices) - 1) / FPS, 1. / FPS, dtype=torch.float32)
    receipt = {
        "schema": SCHEMA, "status": "PREPARED_NOT_QUALITY_ACCEPTED", "automatic_promotion": False,
        "training_started": False, "origin_id": row["origin_id"], "dataset": row.get("dataset"),
        "split": row["split"], "source_sha256": row["source_sha256"], "fps": FPS,
        "frames": len(indices), "expected_packed_frames": len(repack_times),
        "input_sha256": snapshots, "source_clock_proof": clock_proof,
        "common_window_proof": common_proof, "baseline_packed_value_proof": value_proof,
        "packed_joint_names": packed_names, "output_joint_names": list(G1_JOINT_NAMES),
        "alignment": alignment, "source_posed_xy_offset_m": meta["posed_xy_offset"],
        "baseline_retarget_fingerprint": row.get("baseline_retarget_fingerprint"),
        "baseline_package_fingerprint": packed["pipeline_fingerprint"],
        "candidate_scale": candidate_meta.get("scale"), "candidate_ground_offset": candidate_meta.get("ground_offset"),
        "comparison": "whole pipeline: optimized neutral-shape sparse IK vs source-shape material-surface UMR; NOT solver ablation",
        "validation_note": "Development-validation origins stay excluded from gradient updates; not a fresh final holdout",
        "packaging_note": "Both 50 Hz PKLs use the same inclusive grid; the existing packer drops the last endpoint from both equally",
        "license_note": "Local derived licensed data; no redistribution permission inferred",
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    outputs = {}
    for label, qpos in (("baseline", before), ("candidate", after)):
        motion = convert_umr_payload({"qpos": qpos, "fps": FPS, "dof_names": G1_JOINT_NAMES})
        path = output_dir / f"{label}.pkl"
        with path.open("xb") as stream:
            pickle.dump(motion, stream, protocol=4)
        outputs[label] = {"path": str(path.resolve()), "sha256": sha256(path), "frames": len(qpos)}
    proof_path = output_dir / "sampling_proof.npz"
    with proof_path.open("xb") as stream:
        np.savez_compressed(stream, times=indices / FPS, packed_indices=indices,
                            candidate_indices=indices, native_times=native_times,
                            baseline_native_lower=interp["lower"][indices],
                            baseline_native_upper=interp["upper"][indices],
                            baseline_native_alpha=interp["alpha"][indices],
                            candidate_source_lower=prepared["sample_lower"][indices],
                            candidate_source_upper=prepared["sample_upper"][indices],
                            candidate_source_alpha=prepared["sample_alpha"][indices])
    outputs["sampling_proof"] = {"path": str(proof_path.resolve()), "sha256": sha256(proof_path)}
    receipt["outputs"] = outputs
    with (output_dir / "pair_receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--origin-id", required=True)
    parser.add_argument("--prepared-source", type=Path, required=True)
    parser.add_argument("--candidate-npz", type=Path, required=True)
    parser.add_argument("--packed-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trusted-local-pickle", action="store_true", required=True,
                        help="Acknowledge that baseline PKL is trusted and may execute code")
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    rows = [row for row in manifest["motions"] if row["origin_id"] == args.origin_id]
    if len(rows) != 1 or manifest.get("automatic_promotion") is not False:
        parser.error("Expected one matching origin in a non-promoting manifest")
    packed_tools.refresh.validate_source_disjoint(manifest["motions"])
    receipt = prepare_pair(rows[0], prepared_source=args.prepared_source, candidate_npz=args.candidate_npz,
                           packed_audit=args.packed_audit, output_dir=args.output_dir)
    print(json.dumps({"origin_id": receipt["origin_id"], "frames": receipt["frames"],
                      "status": receipt["status"], "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
