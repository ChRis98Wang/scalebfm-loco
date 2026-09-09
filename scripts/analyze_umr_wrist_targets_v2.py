#!/usr/bin/env python3
"""Read-only, post-hoc historical wrist-target compatibility diagnostic.

Uses SHA-bound prepared SMPL-X world rotations, not new SMPL-X inference or IK.
All 17 target-train and all 10 development origins are mandatory. No physics,
training, data replacement, quality-gate modification or promotion is performed.
Default output is JSON stdout; --outputnew creates exactly one new report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

if __name__ == "__main__":
    sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
PINS = {
    "scripts/analyze_umr_reference_shift.py": "19fe46baecca3a91cc0defb68cdfd6b6cd755897e4aa4b597c250181ed19f9e3",
    "scripts/build_umr_behavior_ab.py": "e9c002a90cdd8a2499b0f868a8c8e75553dce93b96379fcd77ed22f8bc225b62",
    "scripts/umr_smplx_source.py": "d2b41fe5cffaed03637d3e37cbdf9fef1f4c30996d636b6146483bf5e9b66b94",
    "ScaleRetarget/config/correspondence/gmr/amass_to_unitree_g1.yaml": "3902395bc2fb8003c72512b473348fe44ea1c6916c96a90e613f92ceb4c0c6c7",
}
# Check before importing the unchanged validator or its leaf dependencies.
for _relative, _expected in PINS.items():
    _path = ROOT / _relative
    if _path.is_symlink() or hashlib.sha256(_path.read_bytes()).hexdigest() != _expected:
        raise ValueError(f"Frozen wrist diagnostic dependency changed: {_path}")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
from scripts import analyze_umr_reference_shift as shift

SCHEMA = "bfm.umr_wrist_targets/2"
PELVIS_OFFSET = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
WRIST_OFFSETS = np.stack((np.eye(3), np.diag([-1., -1., 1.])))
WRIST_NAMES = ("left_wrist_yaw_link", "right_wrist_yaw_link")
PARENTS22 = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19])
RETARGET_FINGERPRINT = "03b685383c0c2315b2557cce93a85aa134e12ed7519acc263cd4b410cb43a651"


def _rotations(value):
    value = np.asarray(value, dtype=np.float64)
    if (value.ndim < 2 or value.shape[-2:] != (3, 3) or not value.size
            or not np.isfinite(value).all()
            or not np.allclose(value.swapaxes(-1, -2) @ value, np.eye(3), rtol=0, atol=2e-5)
            or not np.allclose(np.linalg.det(value), 1., rtol=0, atol=2e-5)):
        raise ValueError("Expected nonempty finite proper SO(3) matrices; no projection or repair")
    return value


def rotation_angle(first, second):
    """Shortest SO(3) radians using atan2, including identity and pi endpoints."""
    first, second = _rotations(first), _rotations(second)
    if first.shape != second.shape:
        raise ValueError("Rotation shapes must agree exactly, without broadcasting")
    relative = first.swapaxes(-1, -2) @ second
    skew = np.stack((relative[..., 2, 1] - relative[..., 1, 2],
                     relative[..., 0, 2] - relative[..., 2, 0],
                     relative[..., 1, 0] - relative[..., 0, 1]), axis=-1)
    sine = np.linalg.norm(skew, axis=-1) / 2
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.) / 2, -1., 1.)
    return np.arctan2(sine, cosine)


def target_errors(human_rotations, robot_quaternions, body_names):
    """World and full-SO(3) own-pelvis errors against historical local offsets.

    Human input is ALREADY heading-normalized prepared global joint rotations.
    Robot input is ALREADY aligned packed world-link wxyz, not inertial frames.
    Human joints 0/20/21 are pelvis/left wrist/right wrist. Offsets multiply on
    the RIGHT. The pelvis offset is essential in the relative target.
    """
    human = _rotations(human_rotations)
    names = list(body_names)
    if (human.ndim != 4 or human.shape[1:] != (55, 3, 3)
            or any(not isinstance(n, str) for n in names)
            or len(set(names)) != len(names)
            or not {"pelvis", *WRIST_NAMES}.issubset(names)):
        raise ValueError("Expected 55 human joints and unique named robot pelvis/wrist links")
    quaternions = shift._unit_quaternions(robot_quaternions)
    if quaternions.shape != (len(human), len(names), 4):
        raise ValueError("Human and robot frame counts / named-link shapes differ")
    robot = shift._rotation(quaternions)
    target_pelvis = human[:, 0] @ PELVIS_OFFSET
    target_wrists = human[:, [20, 21]] @ WRIST_OFFSETS
    robot_pelvis = robot[:, names.index("pelvis")]
    robot_wrists = robot[:, [names.index(n) for n in WRIST_NAMES]]
    target_relative = target_pelvis[:, None].swapaxes(-1, -2) @ target_wrists
    robot_relative = robot_pelvis[:, None].swapaxes(-1, -2) @ robot_wrists
    return {"world_wrist": rotation_angle(target_wrists, robot_wrists),
            "pelvis_relative_wrist": rotation_angle(target_relative, robot_relative),
            "world_pelvis": rotation_angle(target_pelvis, robot_pelvis)}


def _equal(actual, expected, name, *, tolerance=1e-10, integer=False):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if (actual.shape != expected.shape or actual.dtype.kind not in "fiu"
            or not np.isfinite(actual).all()
            or (integer and actual.dtype.kind not in "iu")
            or not np.allclose(actual, expected, rtol=0, atol=0 if integer else tolerance)):
        raise ValueError(f"Prepared/pair source proof mismatch: {name}")


def validate_source_clock(prepared, metadata, pair, proof, frames):
    """Pure validation; return prepared indices for the packed half-open grid."""
    clock, window = pair["source_clock_proof"], pair["common_window_proof"]
    source_count, source_fps = clock["source_frames"], clock["source_fps"]
    if (type(frames) is not int or frames < 2 or type(source_count) is not int or source_count < 3
            or not isinstance(source_fps, (int, float)) or not np.isfinite(source_fps) or source_fps <= 0
            or metadata["start"] != 0 or metadata["target_fps"] != 50
            or metadata["requested_duration"] != 5 or metadata["source_fps"] != source_fps
            or pair["fps"] != 50 or pair["frames"] != frames + 1
            or pair["expected_packed_frames"] != frames):
        raise ValueError("Invalid fixed 50 Hz source-frame-zero clock")
    endpoint = (source_count - 1) / source_fps
    _equal(clock["source_endpoint_s"], endpoint, "original endpoint")
    times = np.arange(int(np.floor(min(5., endpoint) * 50 + 1e-9)) + 1) / 50.
    fractional = np.clip(times * source_fps, 0, source_count - 1)
    nearest = np.rint(fractional)
    fractional = np.where(np.abs(fractional - nearest) < 1e-9, nearest, fractional)
    lower = np.floor(fractional).astype(np.int64)
    upper = np.minimum(lower + 1, source_count - 1)
    alpha = fractional - lower
    for name, value in (("times", times), ("sample_lower", lower),
                        ("sample_upper", upper), ("sample_alpha", alpha)):
        _equal(prepared[name], value, name, integer=name in ("sample_lower", "sample_upper"))
    if (metadata["frames"] != len(times) or window["candidate_frames_before_window"] != len(times)
            or frames + 1 > len(times) or window["pair_frames"] != frames + 1
            or window["pair_start_s"] != 0):
        raise ValueError("Prepared/common window frame count mismatch")
    _equal(metadata["actual_duration"], times[-1], "prepared actual duration")
    _equal(window["pair_last_sample_s"], frames / 50., "common endpoint")
    indices = np.arange(frames + 1, dtype=np.int64)
    for name in ("candidate_indices", "packed_indices"):
        _equal(proof[name], indices, name, integer=True)
    _equal(proof["times"], times[indices], "common source times", tolerance=1e-12)
    for key, values in (("lower", lower), ("upper", upper), ("alpha", alpha)):
        _equal(proof["candidate_source_" + key], values[indices], "candidate source " + key,
               integer=key != "alpha")
    native_count, native_fps = clock["native_frames"], clock["native_actual_fps"]
    if type(native_count) is not int or native_count < 2 or not np.isfinite(native_fps) or native_fps <= 0:
        raise ValueError("Invalid baseline native clock")
    expected_count = source_count if np.isclose(source_fps, 30.) else max(2, round(endpoint * 30.) + 1)
    expected_fps = source_fps if np.isclose(source_fps, 30.) else (expected_count - 1) / endpoint
    if (native_count != expected_count or clock["native_source_frame_zero"] != 0
            or clock["native_nominal_target_fps"] != 30.):
        raise ValueError("Native baseline does not preserve original endpoint")
    _equal(native_fps, expected_fps, "native rate")
    native_times = np.arange(native_count) / native_fps
    _equal(proof["native_times"], native_times, "native times")
    lo, hi, blend = (np.asarray(proof["baseline_native_" + k]) for k in ("lower", "upper", "alpha"))
    if (lo.shape != indices.shape or hi.shape != indices.shape or blend.shape != indices.shape
            or lo.dtype.kind not in "iu" or hi.dtype.kind not in "iu"
            or blend.dtype.kind != "f" or not np.isfinite(blend).all()
            or np.any(lo < 0) or np.any(hi >= native_count)
            or np.any(hi != np.minimum(lo + 1, native_count - 1))
            or np.any(blend < 0) or np.any(blend > 1)):
        raise ValueError("Invalid native interpolation indices/weights")
    _equal(native_times[lo] * (1 - blend.astype(float)) + native_times[hi] * blend,
           proof["times"], "native interpolation time", tolerance=1e-5)
    if (window["dropped_candidate_frames"] != len(times) - len(indices)
            or window["dropped_packed_frames"] != window["packed_frames_before_window"] - len(indices)
            or window["dropped_packed_frames"] < 0):
        raise ValueError("Common-window intersection accounting mismatch")
    # One endpoint is removed by the final paired integer-clock packer, not two.
    return indices[:-1]


def _prepared_source(row, pair, batch, frozen, manifest_hashes):
    surface_root = batch / "surfaces"
    paths = [Path(p) for p in pair["input_sha256"]
             if Path(p).is_relative_to(surface_root) and Path(p).suffix == ".npz"]
    expected = surface_root / (row["origin_id"] + ".npz")
    if len(paths) != 1 or paths[0] != expected or any(p.is_symlink() for p in expected.parents):
        raise ValueError("Prepared source is not uniquely bound to this origin")
    path = paths[0]
    digest = pair["input_sha256"][str(path)]
    if manifest_hashes.get(str(path)) != digest:
        raise ValueError("Prepared source is not bound by full A/B dataset manifest")
    frozen.add(path, digest)
    with np.load(path, allow_pickle=False) as z:
        keys = ("metadata_json", "fps", "times", "sample_lower", "sample_upper", "sample_alpha",
                "parents", "joint_rotations")
        prepared = {key: z[key] for key in keys}
    if prepared["metadata_json"].shape != () or prepared["metadata_json"].dtype.kind not in "US":
        raise ValueError("Invalid prepared scalar metadata")
    metadata = json.loads(prepared["metadata_json"].item())
    if (not isinstance(metadata, dict) or metadata["schema"] != "bfm.smplx_surface_source/1"
            or metadata["source_file"] != row["source"] or metadata["source_sha256"] != row["source_sha256"]
            or metadata["adapter_sha256"] != PINS["scripts/umr_smplx_source.py"]
            or metadata["shape_policy"] != "source_betas_all_static"
            or metadata["posed_world_assumption"] != "AMASS Z-up metres"
            or metadata["flat_hand_mean"] is not True or metadata["expression"] != "zero"
            or prepared["fps"].shape != () or prepared["fps"].item() != 50):
        raise ValueError("Prepared source identity / orientation contract mismatch")
    frozen.add(metadata["body_model"], metadata["body_model_sha256"])
    if prepared["parents"].shape != (55,):
        raise ValueError("Prepared source requires standard SMPL-X 55-joint skeleton")
    _equal(prepared["parents"][:22], PARENTS22, "SMPL-X body/wrist topology", integer=True)
    human = _rotations(prepared["joint_rotations"])
    if human.shape != (metadata["frames"], 55, 3, 3):
        raise ValueError("Invalid prepared global joint rotation array")
    alignment = pair["alignment"]
    heading = _rotations(metadata["posed_heading_rotation"])
    if heading.shape != (3, 3):
        raise ValueError("Expected fixed heading rotation, not per-frame fitting")
    _equal(alignment["baseline_world_rotation"], heading, "baseline fixed heading", tolerance=1e-12)
    _equal(alignment["candidate_world_rotation"], np.eye(3), "candidate already normalized heading", tolerance=1e-12)
    if (alignment["per_frame_alignment"] is not False or alignment["shape_or_scale_fit"] is not False
            or alignment["baseline_constant_z_offset_m"] != 0
            or alignment["candidate_constant_z_offset_m"] != 0
            or pair["baseline_retarget_fingerprint"] != RETARGET_FINGERPRINT):
        raise ValueError("Unexpected pair alignment or historical baseline version")
    output = pair["outputs"]["sampling_proof"]
    if output != {"path": row["sampling_proof"], "sha256": row["sampling_proof_sha256"]}:
        raise ValueError("Sampling proof receipt/manifest identity mismatch")
    frozen.add(row["sampling_proof"], row["sampling_proof_sha256"])
    with np.load(row["sampling_proof"], allow_pickle=False) as z:
        proof = {key: z[key] for key in z.files}
    indices = validate_source_clock(prepared, metadata, pair, proof, row["expected_packed_frames"])
    frozen.add(path, digest)
    frozen.add(row["sampling_proof"], row["sampling_proof_sha256"])
    return human[indices], {"path": str(path), "sha256": digest, "metadata": metadata,
                           "prepared_frame_count": len(human), "pair_inclusive_frames": len(indices) + 1,
                           "packed_halfopen_frames": len(indices), "prepared_index_first": int(indices[0]),
                           "prepared_index_last": int(indices[-1]), "alignment": alignment}


def _statistics(errors):
    result = {}
    for key, value in errors.items():
        degrees = np.rad2deg(value)
        groups = {"pelvis": degrees} if key == "world_pelvis" else {
            "left": degrees[:, 0], "right": degrees[:, 1], "both": degrees}
        result[key] = {label: {"mean_deg": float(np.mean(v)),
                              "p95_deg": float(np.percentile(v, 95, method="linear")),
                              "max_deg": float(np.max(v))} for label, v in groups.items()}
    return result


def _aggregate(records):
    result = {}
    for side in ("baseline", "candidate"):
        result[side] = {}
        for metric, groups in records[0]["statistics"][side].items():
            result[side][metric] = {}
            for group in groups:
                values = [r["statistics"][side][metric][group] for r in records]
                result[side][metric][group] = {
                    "origin_equal_mean_deg": float(np.mean([v["mean_deg"] for v in values])),
                    "mean_of_per_origin_p95_deg": float(np.mean([v["p95_deg"] for v in values])),
                    "maximum_deg": max(v["max_deg"] for v in values)}
    return result


def analyze_dataset(dataset):
    dataset = Path(dataset).absolute()
    frozen = shift.Snapshot()
    frozen.add(dataset)
    code = {str(ROOT / relative): frozen.add(ROOT / relative, digest) for relative, digest in PINS.items()}
    for path in shift._code_paths():
        code[str(path)] = frozen.add(path)
    code[str(Path(__file__).resolve())] = frozen.add(Path(__file__).resolve())
    manifest = shift.dataset_contract.validate_manifest(dataset)
    frozen.add(dataset)
    for path, expected in manifest["input_sha256"].items():
        frozen.add(path, expected)
    for path, target in manifest["resolved_input_aliases"].items():
        if str(Path(path).resolve(strict=True)) != target:
            raise ValueError(f"Manifest source alias changed: {path}")
        frozen.aliases[path] = target
    for path in shift._code_paths():
        code[str(path)] = frozen.add(path)
    batch = Path(manifest["evidence"]["batch"])
    pilot = frozen.read_json(batch / "paired_manifest.json", manifest["evidence"]["pins"]["manifest"])
    fk = frozen.read_json(batch / "body_fk_audit_20260909a.json", manifest["evidence"]["pins"]["fk"])
    rows = {r["origin_id"]: r for r in pilot["rows"]}
    fk_rows = {(r["origin_id"], r["side"]): r for r in fk["results"]}
    provenance = {r["origin_id"]: r for r in manifest["source_provenance"]}
    indexes = {key: shift.dataset_contract.read_index(entry["index"], frozen, entry["sha256"])
               for key, entry in {**manifest["arms"], **manifest["development"]["indices"]}.items()}
    if set(manifest["target_origins"]) & set(manifest["development"]["origin_ids"]):
        raise ValueError("Train/development origins overlap")
    splits = {}
    for split, origins, akey, bkey, count in (
            ("train17", manifest["target_origins"], "a", "b", 17),
            ("dev10", manifest["development"]["origin_ids"], "baseline", "candidate", 10)):
        if len(origins) != count or len(set(origins)) != count:
            raise ValueError("Require all 17 target train and all 10 development origins")
        records = []
        for origin in origins:
            row, source = rows[origin], provenance[origin]
            if (row["source"] != source["canonical_source"] or row["source_sha256"] != source["source_sha256"]
                    or row["split"] != ("train" if split == "train17" else "validation")):
                raise ValueError(f"Original source identity/split mismatch: {origin}")
            frozen.add(row["source"], row["source_sha256"])
            pair = frozen.read_json(row["pair_receipt"], row["pair_receipt_sha256"])
            if pair["origin_id"] != origin or pair["source_sha256"] != row["source_sha256"]:
                raise ValueError("Pair receipt does not bind original source")
            for path, digest in pair["input_sha256"].items():
                frozen.add(path, digest)
            human, prepared = _prepared_source(row, pair, batch, frozen, manifest["input_sha256"])
            frames = row["expected_packed_frames"]
            statistics, payloads = {}, {}
            for side, key in (("baseline", akey), ("candidate", bkey)):
                path = indexes[key][origin]
                digest = manifest["input_sha256"][path]
                record = fk_rows[(origin, side)]
                if (record["path"] != path or record["sha256"] != digest
                        or record["frames"] != frames or record["passed"] is not True):
                    raise ValueError(f"Packed world-link FK evidence mismatch: {origin}/{side}")
                frozen.add(row[side + "_pkl"], row[side + "_pkl_sha256"])
                z = shift._load_packed(path, digest, row[side + "_pkl_sha256"],
                                       manifest["evidence"]["paired_packaging_fingerprint"], frames, frozen)
                statistics[side] = _statistics(target_errors(human, z["body_quat_w"], z["body_names"].tolist()))
                payloads[side] = {"path": path, "sha256": digest, "frames": frames,
                                  "native_sha256": row[side + "_pkl_sha256"], "bound_body_fk_pass": True}
            records.append({"origin_id": origin, "dataset": row["dataset"], "split": row["split"],
                            "frame_count": frames, "source": row["source"], "source_sha256": row["source_sha256"],
                            "pair_receipt": row["pair_receipt"], "pair_receipt_sha256": row["pair_receipt_sha256"],
                            "sampling_proof": row["sampling_proof"], "sampling_proof_sha256": row["sampling_proof_sha256"],
                            "prepared_source": prepared, "payloads": payloads, "statistics": statistics})
        splits[split] = {"origin_count": count, "frames_per_side": sum(r["frame_count"] for r in records),
                         "aggregate": _aggregate(records), "origins": records,
                         "by_dataset": {name: _aggregate([r for r in records if r["dataset"] == name])
                                        for name in sorted({r["dataset"] for r in records})}}
    frozen.recheck()
    return {"schema": SCHEMA, "result": "COMPLETE_DESCRIPTIVE_DIAGNOSTIC", "post_hoc": True,
            "purpose": "Historical-target compatibility, not the UMR optimization objective or frozen-v1 gate",
            "v1_quality_gate": False, "policy_errors_measured": False, "promotion_recommendation": None,
            "automatic_promotion": False, "physics_stepped": False, "training_updates": 0,
            "smplx_inference_performed": False, "ik_optimization_performed": False,
            "dataset_manifest": str(dataset), "dataset_manifest_sha256": frozen.hashes()[str(dataset)],
            "human_joint_indices": {"pelvis": 0, "left_wrist": 20, "right_wrist": 21},
            "offset_wxyz": {"pelvis": [.5, -.5, -.5, -.5], "left_wrist": [1, 0, 0, 0], "right_wrist": [0, 0, 0, -1]},
            "definition": {"target_world": "T_p=R_h,0 @ C_p; T_w=R_h,20/21 @ C_w; right-multiplied historical offsets",
                           "world_wrist": "angle(T_w.T @ R_robot,w)",
                           "world_pelvis": "angle(T_p.T @ R_robot,p)",
                           "pelvis_relative_wrist": "angle((T_p.T @ T_w).T @ (R_robot,p.T @ R_robot,w))",
                           "coordinates": "Prepared human and packed robots already share heading-normalized world; do not apply H again",
                           "clock": "Use candidate_indices[:M] from SHA-bound sampling proof, M=N-1 packed frames; no nearest-time pairing",
                           "within_origin": "Degrees; numpy linear p95 over frames (both wrists pools frame x wrist)",
                           "across_origins": "Equal mean of origin means and of per-origin p95s, not pooled-frame p95"},
            "limitations": ["Descriptive compatibility with historical GMR frame targets, NOT UMR's own objective, solver-bug evidence, policy tracking error or acceptance criterion.",
                            "Targets reuse original-AMASS-to-50-Hz prepared human rotations. Historical GMR instead solved after endpoint-preserving nominal-30-Hz resampling (its actual fps may differ), then packed to 50 Hz. Resampling order differs: this is a same-source 50 Hz historical-mapping compatibility metric, NOT exact replay of the solver's actual optimization-frame residuals.",
                            "Full-SO(3) own-pelvis relative residual mixes wrist and each robot's pelvis error; world wrist and world pelvis are separately reported and cannot be additively decomposed.",
                            "Packed interpolation uses float32 times on the logical 50 Hz grid; tiny within-grid interpolation and legacy near-identical quaternion branch differences remain (recorded native tolerance 0.0022 rad).",
                            "No per-frame orientation fitting, wrist offset fitting or numerical projection of invalid rotations is performed.",
                            "All 17 geometry-selected train and all 10 development origins are included, including development geometric failures; these are not random matched populations or a fresh final test.",
                            "No joint-limit saturation is inferred without a separately bound robot-model contract; no physical calibration claim.",
                            "Licensed local source/derived motion data are not approved for redistribution by this diagnostic."],
            "splits": splits, "code_sha256": dict(sorted(code.items())), "input_sha256": frozen.hashes(),
            "resolved_input_aliases": dict(sorted(frozen.aliases.items())), "inputs_verified_unchanged": True,
            "hash_verification": "Full original dataset validation plus all report inputs/code hashed before use and rehashed after computation",
            "runtime": {"python": sys.version, "numpy": np.__version__,
                        "validator_torch": getattr(sys.modules.get("torch"), "__version__", None)}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--outputnew", type=Path, help="Exactly one NEW JSON in an existing directory; default stdout")
    args = parser.parse_args(argv)
    output = shift._new_output(args.outputnew) if args.outputnew else None
    report = analyze_dataset(args.dataset)
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if output is None:
        print(text, end="")
    else:
        output = shift._new_output(output)
        with output.open("x", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
