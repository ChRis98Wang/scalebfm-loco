#!/usr/bin/env python3
"""Complete, origin-preserving frozen-policy evaluation of true-UMR pairs.

Default is read-only preflight/plan. ``--execute`` requires an owned
bfm-umr-eval-*.service cgroup with KillMode=control-group. Actual packed baseline
and candidate references are audited with the same G1 geometry, then all eight
masks are evaluated using the same official checkpoint. Geometry rejects remain
in every paired evaluation. No training, index replacement or automatic promotion.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import evaluate_retarget_pilot as paired

ROBOT_XML = ROOT / "ScaleRetarget/assets/unitree_g1/g1_mocap_29dof.xml"
GEOMETRY_PYTHON = Path("/home/sw/.cache/bfm/scaleretarget-py311/bin/python")
ORIGINAL_SELECTION = ROOT / "local/data_refresh_20260908b/manifest.json"
ORIGINAL_SELECTION_SHA = "b440d93dfd17df42fe83f08d840fea59a0a719add9b40b32f1cfb0bef0a9f903"
SIDES = ("baseline", "candidate")
INDEX_LABELS = ("train", "legacy_validation", "kit_validation")
# The policy consumes articulation arrays by runtime index, not by stored-name
# lookup. Verify the exact audited local order, not merely an equal name set.
ARTICULATION_JOINT_NAMES = (
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint", "left_hip_roll_joint",
    "right_hip_roll_joint", "waist_roll_joint", "left_hip_yaw_joint", "right_hip_yaw_joint",
    "waist_pitch_joint", "left_knee_joint", "right_knee_joint", "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint", "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint", "left_ankle_roll_joint",
    "right_ankle_roll_joint", "left_shoulder_yaw_joint", "right_shoulder_yaw_joint", "left_elbow_joint",
    "right_elbow_joint", "left_wrist_roll_joint", "right_wrist_roll_joint", "left_wrist_pitch_joint",
    "right_wrist_pitch_joint", "left_wrist_yaw_joint", "right_wrist_yaw_joint",
)
BODY_NAMES = (
    "pelvis", "left_hip_pitch_link", "right_hip_pitch_link", "waist_yaw_link",
    "left_hip_roll_link", "right_hip_roll_link", "waist_roll_link", "left_hip_yaw_link",
    "right_hip_yaw_link", "torso_link", "left_knee_link", "right_knee_link",
    "left_shoulder_pitch_link", "right_shoulder_pitch_link", "left_ankle_pitch_link",
    "right_ankle_pitch_link", "left_shoulder_roll_link", "right_shoulder_roll_link",
    "left_ankle_roll_link", "right_ankle_roll_link", "left_shoulder_yaw_link",
    "right_shoulder_yaw_link", "left_elbow_link", "right_elbow_link", "left_wrist_roll_link",
    "right_wrist_roll_link", "left_wrist_pitch_link", "right_wrist_pitch_link",
    "left_wrist_yaw_link", "right_wrist_yaw_link",
)


def helpers():
    return paired.helpers()


def owned_cgroup():
    matches = re.findall(r"(?:^|/)(bfm-umr-eval-[A-Za-z0-9_.-]+\.service)(?:/|$)",
                         Path("/proc/self/cgroup").read_text(), flags=re.MULTILINE)
    if len(matches) != 1:
        raise ValueError("Execute in an owned bfm-umr-eval-*.service cgroup")
    mode = subprocess.check_output(["systemctl", "--user", "show", matches[0],
                                    "-p", "KillMode", "--value"], text=True).strip()
    if mode != "control-group":
        raise ValueError("Owned evaluation service must use KillMode=control-group")
    return matches[0]


def _file(path, *, expected_sha=None, frozen=None):
    _, _, tools = helpers()
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular owned file, not symlink: {path}")
    path = path.resolve(strict=True)
    digest = tools.sha256_file(path)
    if expected_sha is not None and (not isinstance(expected_sha, str)
            or not re.fullmatch("[0-9a-f]{64}", expected_sha) or digest != expected_sha):
        raise ValueError(f"Declared file SHA256 mismatch: {path}")
    if frozen is not None:
        frozen[str(path)] = digest
    return path, digest


def _safe_origin(value, dataset, sources):
    if (not isinstance(value, str) or not value or any(ord(c) < 32 for c in value)
            or value.startswith("/") or "\\" in value):
        raise ValueError("Invalid origin identity")
    parts = value.split("/")
    if len(parts) < 2 or any(part in ("", ".", "..") for part in parts):
        raise ValueError("Origin identity must be a safe relative source name")
    if parts[0] not in sources or dataset != parts[0]:
        raise ValueError("Origin dataset mismatch")


def _archive_names(path):
    import numpy as np
    from scripts.umr_backend import G1_JOINT_NAMES
    with np.load(path, allow_pickle=False) as archive:
        names = {}
        for field, expected in (("joint_names", ARTICULATION_JOINT_NAMES), ("body_names", BODY_NAMES)):
            array = archive[field]
            if array.ndim != 1 or array.dtype.kind not in "US":
                raise ValueError(f"{path}: {field} must be explicit plain strings")
            values = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in array]
            if len(values) != len(expected) or len(set(values)) != len(expected) or set(values) != set(expected):
                raise ValueError(f"{path}: wrong/duplicate G1 {field}")
            if values != list(expected):
                raise ValueError(f"{path}: {field} order differs from audited policy articulation order")
            names[field] = values
    return names


def _native(path, frames):
    """These hashes bind explicitly trusted local producer artifacts, not uploads."""
    import joblib
    import numpy as np
    refresh, _, _ = helpers()
    from scripts.umr_backend import G1_JOINT_NAMES
    payload = joblib.load(path)
    if not isinstance(payload, dict):
        raise ValueError("Native motion must be a dictionary")
    qpos, fps = refresh.kinematic_qpos(payload)
    if fps != 50. or not 3 <= len(qpos) <= 251:
        raise ValueError("Paired native windows must be bounded <= 5 s and exactly 50 Hz")
    from scripts.umr_pair_dataset import package_interpolation
    interpolation = package_interpolation(payload)
    if len(interpolation["times"]) != frames:
        raise ValueError("Native window and declared half-open packed frame counts disagree")
    for field in ("joint_names", "dof_names"):
        if field in payload and list(payload[field]) != list(G1_JOINT_NAMES):
            raise ValueError("Native source joints must be in exact G1 source order")
    return len(qpos), interpolation


def validate_sampling_receipt(row, frozen):
    """Bind both producer outputs to an independently checked common source clock."""
    import numpy as np
    from scripts.umr_backend import G1_JOINT_NAMES
    receipt_path, _ = _file(row["pair_receipt"], expected_sha=row["pair_receipt_sha256"], frozen=frozen)
    proof_path, proof_sha = _file(row["sampling_proof"], expected_sha=row["sampling_proof_sha256"], frozen=frozen)
    receipt = json.loads(receipt_path.read_text())
    for name, value in (("schema", "bfm.umr_same_source_pair/1"), ("status", "PREPARED_NOT_QUALITY_ACCEPTED"),
                        ("automatic_promotion", False), ("training_started", False), ("fps", 50.),
                        ("origin_id", row["origin_id"]), ("dataset", row["dataset"]), ("split", row["split"]),
                        ("source_sha256", row["source_sha256"]),
                        ("expected_packed_frames", row["expected_packed_frames"]),
                        ("output_joint_names", list(G1_JOINT_NAMES))):
        if receipt.get(name) != value:
            raise ValueError(f"Same-source pair receipt mismatch: {name}")
    count = receipt.get("frames")
    if type(count) is not int or not 3 <= count <= 251:
        raise ValueError("Invalid bounded source pair frame count")
    outputs = receipt.get("outputs", {})
    for side in SIDES:
        if outputs.get(side, {}).get("sha256") != row[f"{side}_pkl_sha256"] or outputs[side].get("frames") != count:
            raise ValueError("Pair receipt output hash/frame mismatch")
        _file(outputs[side]["path"], expected_sha=row[f"{side}_pkl_sha256"], frozen=frozen)
    if outputs.get("sampling_proof", {}).get("sha256") != proof_sha:
        raise ValueError("Pair receipt sampling proof hash mismatch")
    if Path(outputs["sampling_proof"]["path"]).resolve(strict=True) != proof_path:
        raise ValueError("Pair receipt sampling proof path mismatch")
    snapshots = receipt.get("input_sha256", {})
    if not isinstance(snapshots, dict) or not snapshots:
        raise ValueError("Pair producer must record all original input hashes")
    for path, digest in snapshots.items():
        _file(path, expected_sha=digest, frozen=frozen)
    alignment = receipt.get("alignment", {})
    if (alignment.get("per_frame_alignment") is not False or alignment.get("shape_or_scale_fit") is not False
            or alignment.get("baseline_constant_z_offset_m") != 0.
            or alignment.get("candidate_constant_z_offset_m") != 0.):
        raise ValueError("Pair must not fit height, scale, drift or per-frame pose to its competitor")
    for side in SIDES:
        rotation = np.asarray(alignment[f"{side}_world_rotation"], dtype=float)
        if (rotation.shape != (3, 3) or not np.isfinite(rotation).all()
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8, rtol=0)
                or not np.isclose(np.linalg.det(rotation), 1., atol=1e-8, rtol=0)
                or not np.allclose(rotation[2], [0., 0., 1.], atol=1e-8, rtol=0)):
            raise ValueError("Pair alignment must use constant proper yaw rotations")
    with np.load(row["source"], allow_pickle=False) as raw:
        raw_count, raw_fps = len(raw["trans"]), float(raw["mocap_frame_rate"])
    clock = receipt.get("source_clock_proof", {})
    if (clock.get("source_frames") != raw_count or clock.get("source_fps") != raw_fps
            or clock.get("native_source_frame_zero") != 0):
        raise ValueError("Pair source clock does not match original AMASS file")
    with np.load(proof_path, allow_pickle=False) as archive:
        proof = {key: archive[key].copy() for key in archive.files}
    times = np.arange(count, dtype=np.float64) / 50.
    for key in ("times", "packed_indices", "candidate_indices", "baseline_native_lower", "baseline_native_upper",
                "baseline_native_alpha", "candidate_source_lower", "candidate_source_upper", "candidate_source_alpha"):
        value = proof[key]
        if value.shape != (count,) or value.dtype.kind not in "fiu" or not np.isfinite(value).all():
            raise ValueError(f"Invalid same-source sampling proof {key}")
    if not np.allclose(proof["times"], times, atol=1e-10, rtol=0):
        raise ValueError("Pair timeline must start at original frame zero and use actual 50 Hz")
    for key in ("packed_indices", "candidate_indices"):
        if proof[key].dtype.kind not in "iu" or not np.array_equal(proof[key], np.arange(count)):
            raise ValueError("Pair sample indices must match, not merely have equal length")
    native_times = proof["native_times"]
    from scripts.umr_pair_dataset import legacy_source_clock
    proven_native_times, _ = legacy_source_clock(raw_count, raw_fps, clock["native_frames"], clock["native_actual_fps"])
    if (native_times.ndim != 1 or not np.isfinite(native_times).all()
            or len(native_times) != clock.get("native_frames")
            or not np.allclose(native_times, proven_native_times, atol=1e-10, rtol=0)
            or not np.isclose(native_times[-1], (raw_count - 1) / raw_fps, atol=1e-10, rtol=0)):
        raise ValueError("Native baseline timeline must span the original source endpoint")
    for prefix, max_count, grid, tolerance in (
            ("baseline_native", len(native_times), native_times, 1e-5),
            ("candidate_source", raw_count, np.arange(raw_count) / raw_fps, 1e-10)):
        lower, upper, alpha = (proof[f"{prefix}_{key}"] for key in ("lower", "upper", "alpha"))
        if (lower.dtype.kind not in "iu" or upper.dtype.kind not in "iu" or np.any(lower < 0)
                or np.any(upper < lower) or np.any(upper > lower + 1) or np.any(upper >= max_count)
                or np.any(alpha < 0) or np.any(alpha > 1)):
            raise ValueError("Invalid pair source interpolation indices/weights")
        reconstructed = (1. - alpha) * grid[lower] + alpha * grid[upper]
        if not np.allclose(reconstructed, times, atol=tolerance, rtol=0):
            raise ValueError("Baseline/candidate windows sample different original times")
    common = receipt.get("common_window_proof", {})
    if (common.get("pair_frames") != count or common.get("pair_start_s") != 0.
            or common.get("pair_last_sample_s") != times[-1]):
        raise ValueError("Pair receipt and sampling proof window differ")
    return count


def validate_pairs(manifest_path, packed_root, *, allowed_counts=(40,)):
    """Validate all preselected origin/split/source/native/packed identities."""
    import yaml
    refresh, _, tools = helpers()
    manifest_path, _ = _file(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != 1 or not re.fullmatch(r"[A-Za-z0-9_.-]+", str(manifest.get("experiment_id", ""))):
        raise ValueError("Expected versioned experiment manifest")
    if manifest.get("automatic_promotion", False) is not False:
        raise ValueError("Pair diagnostics cannot authorize promotion")
    rows = manifest.get("rows", [])
    if not isinstance(rows, list) or len(rows) not in allowed_counts:
        raise ValueError(f"Require the full preselected pilot with count in {allowed_counts}; no filtered subset")
    names = [row.get("origin_id") for row in rows]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate origin identity")
    frozen = {str(manifest_path): tools.sha256_file(manifest_path)}
    selection_path, selection_sha = _file(manifest["selection"], expected_sha=manifest["selection_sha256"], frozen=frozen)
    if selection_path != ORIGINAL_SELECTION.resolve(strict=True) or selection_sha != ORIGINAL_SELECTION_SHA:
        raise ValueError("Expected the original pinned preselection, not a newly selected cohort")
    selection = json.loads(selection_path.read_text())
    if (selection.get("schema"), selection.get("automatic_promotion"), selection.get("seed")) != (1, False, 42):
        raise ValueError("Invalid original non-promoting preselection protocol")
    selected_rows = selection.get("motions", [])
    selected = {row["origin_id"]: row for row in selected_rows}
    if len(selected) != len(selected_rows) or set(selected) != set(names):
        raise ValueError("Paired origins differ from complete original preselection")
    for row in rows:
        if any(row.get(field) != selected[row["origin_id"]].get(field)
               for field in ("source_sha256", "dataset", "split", "index_label")):
            raise ValueError("Preselected origin/source/split identity changed")
    if len(rows) == 40:
        expected_counts = {f"{dataset}/{split}": count for dataset in refresh.SOURCES
                           for split, count in (("train", 6), ("validation", 2))}
        if Counter(f"{row['dataset']}/{row['split']}" for row in rows) != expected_counts:
            raise ValueError("Expected exactly six train plus two development origins per dataset")
    protected = manifest.get("protected_files", {})
    if not isinstance(protected, dict) or not protected:
        raise ValueError("Batch source/model/helper protection hashes are required")
    for path, digest in protected.items():
        _file(path, expected_sha=digest, frozen=frozen)
    index_specs = manifest.get("origin_indexes", {})
    if set(index_specs) != set(INDEX_LABELS):
        raise ValueError("All three frozen canonical origin indexes are required")
    indexes, membership = {}, {}
    for label in INDEX_LABELS:
        declared = index_specs[label]
        path, _ = _file(declared["path"], expected_sha=declared["sha256"], frozen=frozen)
        if declared["sha256"] != selection.get("index_sha256", {}).get(label):
            raise ValueError("Canonical split SHA differs from original preselection")
        if path != Path(refresh.INDEXES[label]).resolve(strict=True):
            raise ValueError(f"Expected original canonical {label} index, not a replacement split")
        index = yaml.safe_load(path.read_text())
        if not isinstance(index, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in index.items()):
            raise ValueError("Canonical index must map origin strings to payload paths")
        for origin in index:
            if origin in membership:
                raise ValueError("Canonical origin appears in multiple split indexes")
            membership[origin] = label
        indexes[label] = index
    packed_root = Path(packed_root).resolve(strict=True)
    expected_packed = {packed_root / side / f"{origin}.npz" for origin in names for side in SIDES}
    actual_packed = set(packed_root.rglob("*.npz"))
    if actual_packed != expected_packed:
        raise ValueError("Packed file set must exactly match both complete paired sides; no missing/extra NPZ")
    native_roots = [manifest_path.parent / name for name in ("input", "inputs")]
    matching = [root for root in native_roots if all(
        Path(row[f"{side}_pkl"]).absolute() == (root / side / f"{row['origin_id']}.pkl").absolute()
        for row in rows for side in SIDES)]
    if len(matching) != 1:
        raise ValueError("All native paths must share one explicit experiment input/inputs root")
    native_root = matching[0]
    expected_native = {native_root / side / f"{origin}.pkl" for origin in names for side in SIDES}
    if set(native_root.rglob("*.pkl")) != expected_native:
        raise ValueError("Native input file set must exactly match both complete sides; no missing/extra PKL")
    pipelines, result = set(), []
    for row in rows:
        origin = row["origin_id"]
        _safe_origin(origin, row.get("dataset"), refresh.SOURCES)
        label = row.get("index_label")
        if membership.get(origin) != label or label not in indexes:
            raise ValueError(f"Origin {origin} is not in its declared canonical index")
        if indexes[label][origin] != selected[origin]["baseline_packed"]:
            raise ValueError("Original canonical baseline path no longer matches preselected origin")
        if row.get("split") != ("train" if label == "train" else "validation"):
            raise ValueError(f"Origin {origin} moved between train/development splits")
        frames = row.get("expected_packed_frames")
        if type(frames) is not int or not 2 <= frames <= 250:
            raise ValueError("Expected bounded packed frame count in [2, 250]")
        canonical = ROOT / "ScaleRetarget/dataset/amass" / f"{origin}.npz"
        _file(canonical, expected_sha=row["source_sha256"], frozen=frozen)
        source, _ = _file(row["source"], expected_sha=row["source_sha256"], frozen=frozen)
        native_count = validate_sampling_receipt(row, frozen)
        normalized = {**row, "source": str(source), "canonical_source": str(canonical),
                      "packed_frames": frames, "kinematic_status": "NOT_YET_AUDITED"}
        side_names, source_frames = {}, set()
        for side in SIDES:
            native, native_sha = _file(row[f"{side}_pkl"], expected_sha=row[f"{side}_pkl_sha256"], frozen=frozen)
            if native != (native_root / side / f"{origin}.pkl").resolve(strict=True):
                raise ValueError(f"Native {side} location does not bind origin {origin}")
            count, interpolation = _native(native, frames)
            source_frames.add(count)
            if count != native_count:
                raise ValueError("Actual native frame count differs from pair receipt")
            archive, archive_sha = _file(packed_root / side / f"{origin}.npz", frozen=frozen)
            pipelines.add(paired.validate_archive(archive, native_sha=native_sha, frames=frames))
            side_names[side] = _archive_names(archive)
            from scripts.umr_pair_dataset import prove_packed_values
            import numpy as np
            with np.load(archive, allow_pickle=False) as payload:
                prove_packed_values(payload, interpolation, side_names[side]["joint_names"])
            normalized.update({f"{side}_pkl": str(native), f"{side}_packed": str(archive),
                               f"{side}_packed_sha256": archive_sha})
        if len(source_frames) != 1 or side_names["baseline"] != side_names["candidate"]:
            raise ValueError("Paired frames or explicit articulation names differ")
        result.append(normalized)
    if len(pipelines) != 1:
        raise ValueError("Paired archives must use exactly one shared packaging fingerprint")
    refresh.validate_source_disjoint(rows)
    for name in ("evaluate_umr_pairs.py", "evaluate_retarget_pilot.py", "compare_mask_evaluations.py",
                 "retarget_data_refresh.py", "audit_packed_retarget_pilot.py", "umr_backend.py",
                 "amass_to_scalebfm.py", "umr_smplx_source.py", "umr_pair_dataset.py"):
        _file(ROOT / "scripts" / name, frozen=frozen)
    _file(ROBOT_XML, frozen=frozen)
    return {"manifest": str(manifest_path), "experiment_id": manifest["experiment_id"], "rows": result,
            "frozen_files": frozen, "packaging_fingerprint": pipelines.pop(),
            "all_origins_included": True,
            "counts": dict(Counter(f"{r['dataset']}/{r['split']}" for r in result))}


def verify_frozen(files):
    _, _, tools = helpers()
    for path, expected in files.items():
        if not Path(path).is_file() or tools.sha256_file(Path(path)) != expected:
            raise ValueError(f"Frozen source/helper/input changed: {path}")


def velocity_statistics(qpos, fps=50.):
    import numpy as np
    qpos = np.asarray(qpos, dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] != 36 or len(qpos) < 2 or not np.isfinite(qpos).all():
        raise ValueError("Velocity diagnostics require finite G1 qpos")
    if not math.isfinite(fps) or fps != 50.:
        raise ValueError("This paired velocity audit uses the fixed 50 Hz clock")
    quats = qpos[:, 3:7]
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    if not np.allclose(norms, 1., atol=1e-5, rtol=0):
        raise ValueError("Velocity diagnostics require unit root quaternions")
    quats = quats / norms
    sign = np.where(np.sum(quats[:-1] * quats[1:], axis=-1, keepdims=True) < 0, -1., 1.)
    angles = 4 * np.arctan2(np.linalg.norm(quats[:-1] - sign * quats[1:], axis=1),
                           np.linalg.norm(quats[:-1] + sign * quats[1:], axis=1))
    arrays = {"root_linear_speed_m_s": np.linalg.norm(np.diff(qpos[:, :3], axis=0), axis=1) * fps,
              "root_angular_speed_rad_s": angles * fps,
              "joint_absolute_speed_rad_s": np.abs(np.diff(qpos[:, 7:], axis=0)) * fps}
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Nonfinite derived velocities")
    return {key: {"mean": float(value.mean()), "p95": float(np.percentile(value, 95)),
                  "max": float(value.max())} for key, value in arrays.items()}


def motion_extent(qpos):
    """Target amplitude diagnostics, never a rescaling operation or quality gate."""
    import numpy as np
    qpos = np.asarray(qpos, dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] != 36 or len(qpos) < 2 or not np.isfinite(qpos).all():
        raise ValueError("Motion-extent diagnostics require finite G1 qpos")
    difference = np.diff(qpos[:, :3], axis=0)
    xy = qpos[:, :2]
    # Pairwise horizontal diameter is invariant to an arbitrary constant world
    # yaw; unlike an axis-aligned bounding box, it cannot change just by turning.
    diameter = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1).max()
    joint_range = np.ptp(qpos[:, 7:], axis=0)
    height = qpos[:, 2]
    return {"root_total_path_length_xyz_m": float(np.linalg.norm(difference, axis=1).sum()),
            "root_total_path_length_xy_m": float(np.linalg.norm(difference[:, :2], axis=1).sum()),
            "root_height_m": {"min": float(height.min()), "max": float(height.max()), "range": float(np.ptp(height))},
            "root_horizontal_span_xy_m": float(diameter),
            "horizontal_span_definition": "maximum pairwise XY distance; constant-yaw invariant",
            "joint_range_rad": {"mean": float(joint_range.mean()), "p95": float(np.percentile(joint_range, 95)),
                                "max": float(joint_range.max())}}


def geometry_reasons(metrics):
    reasons = []
    if metrics["foot_penetration_frames_gt_1mm"]:
        reasons.append("packed_sole_penetration_gt_1mm")
    if metrics["max_all_body_ground_penetration_m"] > .005:
        reasons.append("packed_body_ground_penetration_gt_5mm")
    if metrics["max_self_penetration_m"] > .005:
        reasons.append("packed_self_penetration_gt_5mm")
    if metrics["joint_limit_violations_gt_1e_minus6_rad"]:
        reasons.append("packed_joint_limit_violation")
    return reasons


def audit_geometry(validated, output):
    """Actual packed references, same named G1 model, without physics stepping."""
    import mujoco
    import numpy as np
    from scripts.audit_packed_retarget_pilot import packed_qpos
    from scripts.amass_to_scalebfm import semantic_fingerprint
    refresh, _, _ = helpers()
    if Path(output).exists() or Path(output).is_symlink():
        raise FileExistsError(output)
    verify_frozen(validated["frozen_files"])
    geometry = semantic_fingerprint({"robot_geometry": ROBOT_XML.parent}, {})
    model = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
    model_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(1, model.njnt)]
    results = []
    for row in validated["rows"]:
        result = {"origin_id": row["origin_id"], "split": row["split"], "dataset": row["dataset"]}
        for side in SIDES:
            with np.load(row[f"{side}_packed"], allow_pickle=False) as archive:
                names = [str(value) for value in archive["joint_names"]]
                qpos = packed_qpos(archive, names, model_names)
            metrics = refresh.audit_qpos(qpos, model)
            reasons = geometry_reasons(metrics)
            result[side] = {"metrics": metrics, "velocity_statistics": velocity_statistics(qpos),
                            "motion_extent": motion_extent(qpos),
                            "review_reasons": reasons, "kinematic_pass": not reasons,
                            "packed_sha256": row[f"{side}_packed_sha256"]}
        result["kinematic_status"] = ("REJECT_PACKED_KINEMATIC" if result["candidate"]["review_reasons"]
                                       else "PENDING_FROZEN_POLICY_EVALUATION")
        results.append(result)
    verify_frozen(validated["frozen_files"])
    if semantic_fingerprint({"robot_geometry": ROBOT_XML.parent}, {}) != geometry:
        raise ValueError("Robot geometry tree changed during packed audit")
    report = {"schema": 1, "experiment_id": validated["experiment_id"], "result": "COMPLETE",
              "automatic_promotion": False, "physics_stepped": False, "training_updates": 0,
              "geometry_tree_sha256": geometry, "mujoco_version": mujoco.__version__,
              "source": "actual paired packed 50 Hz reference root and named joints",
              "limitations": "Kinematic contact/velocity audit, not physical slip, fall or task-success evidence. "
                              "Different source-shape scaling can change target amplitudes; easier following is not proof "
                              "of better fidelity to the original human motion. No trajectory is rescaled here.",
              "thresholds": {"foot_penetration_m": .001, "self_or_ground_penetration_m": .005,
                             "joint_limit_tolerance_rad": 1e-6},
              "results": results,
              "counts": dict(Counter(row["kinematic_status"] for row in results))}
    refresh.write_json(Path(output), report)
    return report


def bind_geometry(validated, report):
    if (report.get("schema"), report.get("experiment_id"), report.get("result"),
            report.get("automatic_promotion"), report.get("physics_stepped"), report.get("training_updates")) != (
            1, validated["experiment_id"], "COMPLETE", False, False, 0):
        raise ValueError("Unexpected packed geometry audit protocol")
    records = report.get("results", [])
    by_origin = {row["origin_id"]: row for row in records}
    if len(by_origin) != len(records) or set(by_origin) != {row["origin_id"] for row in validated["rows"]}:
        raise ValueError("Geometry audit must include every preselected origin, including rejects")
    rows = []
    for row in validated["rows"]:
        record = by_origin[row["origin_id"]]
        if record.get("split") != row["split"] or record.get("dataset") != row["dataset"]:
            raise ValueError("Geometry audit origin/split mismatch")
        for side in SIDES:
            item = record[side]
            if item.get("packed_sha256") != row[f"{side}_packed_sha256"]:
                raise ValueError("Geometry audit packed input SHA mismatch")
            if item["metrics"]["frames"] != row["packed_frames"]:
                raise ValueError("Geometry audit frame count mismatch")
            if geometry_reasons(item["metrics"]) != item["review_reasons"]:
                raise ValueError("Geometry gate disagrees with raw audit metrics")
            if item.get("kinematic_pass") is not (not item["review_reasons"]):
                raise ValueError("Geometry pass label disagrees with raw audit reasons")
            for statistics in item["velocity_statistics"].values():
                if any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not math.isfinite(value) or value < 0 for value in statistics.values()):
                    raise ValueError("Invalid finite velocity statistics")
        expected_status = ("REJECT_PACKED_KINEMATIC" if record["candidate"]["review_reasons"]
                           else "PENDING_FROZEN_POLICY_EVALUATION")
        if record.get("kinematic_status") != expected_status:
            raise ValueError("Geometry reject must stay labelled in frozen tracking evaluation")
        rows.append({**row, "kinematic_status": record["kinematic_status"],
                     "kinematic_review_reasons": record["candidate"]["review_reasons"]})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--packed-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--geometry-python", type=Path, default=GEOMETRY_PYTHON)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--geometry-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        parser.error("Output must be new; no overwriting prior diagnostics")
    unit = None
    if args.execute or args.geometry_worker:
        unit = owned_cgroup()
    if args.execute and args.python.absolute() != Path(sys.executable).absolute():
        parser.error("Run controller with the same existing IsaacLab --python interpreter")
    validated = validate_pairs(args.manifest, args.packed_root)
    if args.geometry_worker:
        audit_geometry(validated, args.output)
        return 0
    jobs = paired.build_jobs(args.python, args.output.absolute())
    geometry_job = {"name": "geometry_audit", "timeout_seconds": 600,
                    "command": [str(args.geometry_python), "-u", str(Path(__file__).resolve()),
                                "--manifest", validated["manifest"], "--packed-root", str(args.packed_root.resolve()),
                                "--output", str(args.output.absolute() / "geometry_audit.json"), "--geometry-worker"]}
    if not args.execute:
        print(json.dumps({"plan_only": True, "experiment_id": validated["experiment_id"],
                          "origins": len(validated["rows"]), "counts": validated["counts"],
                          "geometry_job": geometry_job, "jobs": jobs, "automatic_promotion": False,
                          "geometry_rejects_remain_in_evaluation": True}, indent=2))
        return 0
    import yaml
    from scripts.amass_to_scalebfm import semantic_fingerprint
    _, _, tools = helpers()
    if tools.sha256_file(paired.OFFICIAL) != paired.OFFICIAL_SHA:
        raise ValueError("Frozen official checkpoint SHA changed")
    runtime = {"entrypoints": paired.ENTRY, **tools.locate_package_sources(
        ("scaletrack", "my_rsl_rl", "isaaclab", "isaaclab_rl", "isaaclab_tasks"))}
    geometry_tree = semantic_fingerprint({"robot_geometry": ROBOT_XML.parent}, {})
    output = args.output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    status = {"schema": 1, "experiment_id": validated["experiment_id"], "result": "RUNNING", "unit": unit,
              "jobs": [], "jobs_requested": 16, "training_updates": 0, "automatic_promotion": False,
              "geometry_tree_sha256": geometry_tree}
    try:
        paired._write_json(output / "status.json", status)
        paired.run_job(geometry_job, output / "geometry_audit.log")
        geometry_path = output / "geometry_audit.json"
        geometry_report = json.loads(geometry_path.read_text())
        if geometry_report["geometry_tree_sha256"] != geometry_tree:
            raise ValueError("Geometry worker used another robot tree")
        validated["rows"] = bind_geometry(validated, geometry_report)
        validated["frozen_files"][str(geometry_path)] = tools.sha256_file(geometry_path)
        status["geometry_counts"] = geometry_report["counts"]
        manifests = {}
        for side in SIDES:
            with (output / f"{side}.yaml").open("x") as stream:
                yaml.safe_dump({row["origin_id"]: row[f"{side}_packed"] for row in validated["rows"]},
                               stream, sort_keys=False)
            manifests[side] = tools.snapshot_inputs(paired.OFFICIAL, output / f"{side}.yaml", runtime)
        paired._write_json(output / "inputs.json", {"pilot": validated, "evaluation": manifests})
        reports = {side: [] for side in SIDES}
        for job in jobs:
            progress = {**job, "result": "RUNNING"}
            status["jobs"].append(progress)
            paired._write_json(output / "status.json", status)
            print(f"[UMR PAIRS] {job['name']} START", flush=True)
            t0 = time.monotonic()
            paired.run_job(job, output / f"{job['name']}.log")
            path = output / f"{job['name']}.json"
            report = json.loads(path.read_text())
            paired.validate_report(report, validated["rows"], side=job["side"], mode=job["mode"],
                                   expected_manifest=manifests[job["side"]])
            reports[job["side"]].append(report)
            progress.update(result="PASS", report_sha256=tools.sha256_file(path), elapsed_seconds=time.monotonic() - t0)
            paired._write_json(output / "status.json", status)
            print(f"[UMR PAIRS] {job['name']} COMPLETE", flush=True)
        comparison = paired.compare_reports(reports["baseline"], reports["candidate"], validated["rows"])
        comparison.update(experiment_id=validated["experiment_id"], geometry_audit_sha256=tools.sha256_file(geometry_path),
                          geometry_rejects_excluded=0)
        verify_frozen(validated["frozen_files"])
        if semantic_fingerprint({"robot_geometry": ROBOT_XML.parent}, {}) != geometry_tree:
            raise ValueError("Robot geometry changed during evaluation")
        for side, original in manifests.items():
            tools.verify_unchanged(original, tools.snapshot_inputs(paired.OFFICIAL, output / f"{side}.yaml", runtime))
        for job in status["jobs"]:
            if tools.sha256_file(output / f"{job['name']}.json") != job["report_sha256"]:
                raise ValueError("Evaluation report changed before final verification")
        paired._write_json(output / "comparison.json", comparison)
        status.update(result="COMPLETE", inputs_verified_unchanged=True,
                      aggregate_modes_passed=comparison["aggregate_modes_passed"],
                      comparison_sha256=tools.sha256_file(output / "comparison.json"))
    except BaseException as error:
        if status["jobs"] and status["jobs"][-1]["result"] == "RUNNING":
            status["jobs"][-1].update(result="FAIL", error=str(error))
        status.update(result="FAIL", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        status["elapsed_seconds"] = time.monotonic() - started
        paired._write_json(output / "status.json", status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
