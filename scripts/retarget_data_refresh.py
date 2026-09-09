#!/usr/bin/env python3
"""Versioned, origin-preserving AMASS retarget-data pilot (never auto-promotes).

Select a deterministic dataset/split-balanced pilot and optionally generate a
constant-Z sole-aligned candidate from trusted local kinematic PKLs. Kinematic
diagnostics alone NEVER authorize replacing the production training index.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("ACCAD", "BMLmovi", "BMLrub", "CNRS", "KIT")
DATA = ROOT / "ScaleRetarget/retargeted_dataset"
INDEXES = {"train": DATA / "amass_full_v2_train.yaml",
           "legacy_validation": DATA / "amass_full_v1_validation.yaml",
           "kit_validation": DATA / "amass_kit_heldout_v1.yaml"}


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, payload):
    with Path(path).open("x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")


def action_bucket(name):
    """Filename diversity heuristic only; not a ground-truth behavior label."""
    name = name.lower()
    for bucket, pattern in (
        ("dynamic_hint", r"run|jog|jump|hop|kick|punch|dance|martial"),
        ("locomotion_hint", r"walk|turn|step|stair|sideway"),
        ("low_or_object_hint", r"crouch|squat|lie|crawl|sit|lift|box|pick"),
        ("upper_or_stand_hint", r"stand|sway|gesture|arm|wave|reach"),
    ):
        if re.search(pattern, name):
            return bucket
    return "unlabelled"


def select_origins(indexes, *, train_per_source=6, validation_per_source=2, seed=42):
    """No moving an origin between train/dev; no scores used for selection."""
    if min(train_per_source, validation_per_source) < 1:
        raise ValueError("Each source and split must have a positive sample count")
    seen, groups = {}, defaultdict(list)
    for label, index in indexes.items():
        if label not in INDEXES or not isinstance(index, dict):
            raise ValueError("Unexpected split/index format")
        split = "train" if label == "train" else "validation"
        for origin, path in index.items():
            if not isinstance(origin, str) or not isinstance(path, str):
                raise ValueError("Index names and paths must be strings")
            parts = Path(origin).parts
            if len(parts) < 2 or Path(origin).is_absolute() or ".." in parts or parts[0] not in SOURCES:
                raise ValueError(f"Unsafe or unknown origin: {origin}")
            if origin in seen:
                raise ValueError(f"Duplicate origin across split indexes: {origin}")
            seen[origin] = label
            groups[parts[0], split].append({"origin_id": origin, "split": split,
                "index_label": label, "dataset": parts[0], "baseline_packed": path,
                "filename_hint": action_bucket(origin)})
    result = []
    for source in SOURCES:
        for split, count in (("train", train_per_source), ("validation", validation_per_source)):
            rows = groups[source, split]
            if len(rows) < count:
                raise ValueError(f"Insufficient {source}/{split} motions")
            buckets = defaultdict(list)
            for row in rows:
                buckets[row["filename_hint"]].append(row)
            for rows_in_bucket in buckets.values():
                rows_in_bucket.sort(key=lambda row: hashlib.sha256(
                    f"{seed}:{row['origin_id']}".encode()).hexdigest())
            selected = []
            while len(selected) < count:
                for bucket in sorted(buckets):
                    if buckets[bucket] and len(selected) < count:
                        selected.append(buckets[bucket].pop(0))
            result.extend(selected)
    return result


def resolve_inputs(row, repo_root=ROOT):
    """Verify packed -> native PKL -> prepared AMASS chain by file contents."""
    packed = Path(row["baseline_packed"]).resolve(strict=True)
    dataset_root = repo_root / "ScaleRetarget/retargeted_dataset"
    relative = packed.relative_to(dataset_root.resolve())
    if not relative.parts[0].endswith("_processed") or len(relative.parts) < 2:
        raise ValueError(f"Expected processed batch layout: {packed}")
    batch = relative.parts[0].removesuffix("_processed")
    tail = Path(*relative.parts[1:])
    pkl = (dataset_root / batch / tail).with_suffix(".pkl")
    source = (repo_root / "ScaleRetarget/dataset" / batch / tail).with_suffix(".npz")
    source_hash, pkl_hash = sha256(source), sha256(pkl)
    # A valid hash chain alone does not bind an arbitrary YAML label to its
    # human-motion origin. Bind to the original AMASS path/content as well.
    origin = Path(row["origin_id"])
    if origin.is_absolute() or ".." in origin.parts or origin.parts[0] not in SOURCES:
        raise ValueError("Unsafe source origin")
    if row.get("dataset", origin.parts[0]) != origin.parts[0]:
        raise ValueError("Dataset and source origin disagree")
    canonical = repo_root / "ScaleRetarget/dataset/amass" / f"{origin.as_posix()}.npz"
    if not canonical.is_file() or sha256(canonical) != source_hash:
        raise ValueError(f"Origin identity does not match prepared AMASS source: {origin}")
    source_receipt = pkl.with_suffix(".pkl.source.sha256")
    pipeline_receipt = pkl.with_suffix(".pkl.pipeline.sha256")
    if source_receipt.read_text().strip() != source_hash:
        raise ValueError(f"Prepared source provenance mismatch: {source}")
    old_pipeline = pipeline_receipt.read_text().strip()
    if not re.fullmatch("[0-9a-f]{64}", old_pipeline):
        raise ValueError("Invalid baseline retarget pipeline fingerprint")
    with np.load(packed, allow_pickle=False) as archive:
        if int(archive["format_version"]) != 3 or str(archive["quaternion_order"]) != "wxyz":
            raise ValueError("Expected packed format v3 / wxyz")
        if str(archive["source_sha256"]) != pkl_hash or int(archive["fps"]) != 50:
            raise ValueError(f"Packed PKL provenance / FPS mismatch: {packed}")
        frames = len(archive["joint_pos"])
    return {**row, "baseline_packed": str(packed), "baseline_packed_sha256": sha256(packed),
            "baseline_pkl": str(pkl), "baseline_pkl_sha256": pkl_hash,
            "source": str(source.resolve()), "source_sha256": source_hash,
            "canonical_source": str(canonical.resolve()),
            "baseline_retarget_fingerprint": old_pipeline, "packed_frames": frames}


def validate_source_disjoint(rows):
    hashes = {}
    for row in rows:
        old = hashes.setdefault(row["source_sha256"], row["split"])
        if old != row["split"]:
            raise ValueError("Same prepared AMASS source bytes appear in train and validation")


def kinematic_qpos(payload):
    fps = float(np.asarray(payload["fps"]).item())
    root = np.asarray(payload["root_pos"], dtype=np.float64)
    quat = np.asarray(payload["root_rot"], dtype=np.float64)
    dof = np.asarray(payload["dof_pos"], dtype=np.float64)
    if root.ndim != 2 or root.shape[1] != 3 or len(root) < 2:
        raise ValueError("Expected root_pos (T>=2,3)")
    if quat.shape != (len(root), 4) or dof.shape != (len(root), 29):
        raise ValueError("Expected xyzw root_rot (T,4) and native 29-DoF positions")
    qpos = np.concatenate((root, quat[:, [3, 0, 1, 2]], dof), axis=1)
    if not np.isfinite(qpos).all() or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Nonfinite motion / invalid FPS")
    if np.max(np.abs(np.linalg.norm(quat, axis=1) - 1)) > 1e-5:
        raise ValueError("Root quaternion is not unit length")
    return qpos, fps


def audit_qpos(qpos, model):
    """Native-geometry FK/contact detection, NOT simulation or BFM tracking."""
    import mujoco
    feet = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_ankle_roll_link")
            for side in ("left", "right")]
    enabled = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    geoms = np.flatnonzero(np.isin(model.geom_bodyid, feet) & enabled)
    if min(feet) < 0 or len(geoms) != 8 or np.any(model.geom_type[geoms] != mujoco.mjtGeom.mjGEOM_SPHERE):
        raise ValueError("Pilot audit requires exact G1 eight-sphere collision soles")
    if model.nq != 36 or not np.array_equal(model.jnt_qposadr[1:], np.arange(7, 36)):
        raise ValueError("Unexpected G1 qpos order")
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor < 0 or not np.allclose(model.geom_quat[floor], [1, 0, 0, 0]):
        raise ValueError("Expected horizontal floor")
    data = mujoco.MjData(model)
    foot_min = np.empty(len(qpos))
    self_depth = np.zeros(len(qpos))
    floor_depth = np.zeros(len(qpos))
    for frame, pose in enumerate(qpos):
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        foot_min[frame] = np.min(data.geom_xpos[geoms, 2] - model.geom_size[geoms, 0]) - model.geom_pos[floor, 2]
        for contact in data.contact[:data.ncon]:
            if contact.dist >= 0:
                continue
            bodies = model.geom_bodyid[[contact.geom1, contact.geom2]]
            target = floor_depth if 0 in bodies else self_depth
            target[frame] = max(target[frame], -float(contact.dist))
    violation = np.maximum(model.jnt_range[1:, 0] - qpos[:, 7:], qpos[:, 7:] - model.jnt_range[1:, 1])
    return {"frames": len(qpos), "min_collision_sole_z_m": float(foot_min.min()),
        "foot_penetration_frames_gt_1mm": int(np.sum(foot_min < -0.001)),
        "max_self_penetration_m": float(self_depth.max()),
        "self_penetration_frames_gt_1mm": int(np.sum(self_depth > 0.001)),
        "max_all_body_ground_penetration_m": float(floor_depth.max()),
        "joint_limit_violations_gt_1e_minus6_rad": int(np.sum(violation > 1e-6)),
        "interpretation": "kinematic FK only; no stepping, stance labels, slip or learned-policy evidence"}


def prepare_sole_candidates(rows, output, robot_xml, *, max_shift_m=0.05):
    import joblib
    import mujoco
    from omegaconf import OmegaConf
    sys.path.insert(0, str(ROOT / "ScaleRetarget"))
    from scaleretarget.formatter.kinematic_formatter import KinematicFormatter
    model = mujoco.MjModel.from_xml_path(str(robot_xml))
    formatter = KinematicFormatter(OmegaConf.create({"kinematic_model_device": "cpu", "height_adjust": True,
        "height_adjust_mode": "collision_sole", "ground_offset_m": 0.002, "root_offset": False,
        "collision_sole_body_names": ["left_ankle_roll_link", "right_ankle_roll_link"],
        "quat_order": "xyzw", "robot": {"robot_xml_path": str(robot_xml)}}))
    results = []
    for row in rows:
        payload = joblib.load(row["baseline_pkl"])  # Explicitly trusted local artifacts only.
        before, fps = kinematic_qpos(payload)
        initial = audit_qpos(before, model)
        converted = formatter.format(before.copy(), {"fps": fps})
        after, _ = kinematic_qpos(converted)
        shift = float(after[0, 2] - before[0, 2])
        if not np.allclose(after[:, 2] - before[:, 2], shift, atol=1e-9, rtol=0):
            raise ValueError("Candidate must have one constant vertical shift")
        if not np.array_equal(after[:, [0, 1, *range(3, 36)]], before[:, [0, 1, *range(3, 36)]]):
            raise ValueError("Sole correction changed other trajectory fields")
        final = audit_qpos(after, model)
        reasons = []
        if abs(shift) > max_shift_m:
            reasons.append("vertical_shift_exceeds_5cm_review_bound")
        if final["max_all_body_ground_penetration_m"] > 0.005:
            reasons.append("nonfoot_or_ground_penetration_gt_5mm")
        if final["max_self_penetration_m"] > 0.005:
            reasons.append("self_penetration_gt_5mm")
        if final["joint_limit_violations_gt_1e_minus6_rad"]:
            reasons.append("joint_limit_violation")
        if shift <= 0 or initial["min_collision_sole_z_m"] >= -0.001:
            reasons.append("no_material_upward_sole_fix")
        # Retain all paired candidate artifacts for honest analysis, including
        # rejects; none of these PKLs is a promoted training dataset.
        candidate = output / "sole_candidates" / f"{row['origin_id']}.pkl"
        candidate.parent.mkdir(parents=True, exist_ok=True)
        with candidate.open("xb") as stream:
            joblib.dump(converted, stream)
        results.append({"origin_id": row["origin_id"], "split": row["split"],
            "source_sha256": row["source_sha256"], "baseline": initial, "candidate": final,
            "candidate_pkl": str(candidate), "candidate_sha256": sha256(candidate),
            "constant_z_shift_m": shift, "review_reasons": reasons,
            "status": "REJECT_KINEMATIC" if reasons else "PENDING_FROZEN_POLICY_EVALUATION"})
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New pilot directory, never production index")
    parser.add_argument("--train-per-source", type=int, default=6)
    parser.add_argument("--validation-per-source", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sole-candidates", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        parser.error("Refusing existing pilot output")
    if args.sole_candidates and "bfm-data-refresh-" not in Path("/proc/self/cgroup").read_text():
        parser.error("Run candidate generation inside owned bfm-data-refresh-*.service cgroup")
    import yaml
    index_hashes = {key: sha256(path) for key, path in INDEXES.items()}
    indexes = {label: yaml.safe_load(path.read_text()) for label, path in INDEXES.items()}
    selected = select_origins(indexes, train_per_source=args.train_per_source,
                              validation_per_source=args.validation_per_source, seed=args.seed)
    rows = [resolve_inputs(row) for row in selected]
    validate_source_disjoint(rows)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": 1, "status": "PILOT_ONLY_NOT_TRAINING_ACCEPTED", "automatic_promotion": False,
        "seed": args.seed, "selection": "dataset/split balanced, hash order round-robin filename hints",
        "validation_note": "Existing development split, not untouched final test. Never use for gradient updates.",
        "counts": dict(Counter(f"{r['dataset']}/{r['split']}" for r in rows)),
        "index_sha256": index_hashes, "motions": rows, "script_sha256": sha256(Path(__file__))}
    write_json(output / "manifest.json", manifest)
    try:
        if args.sole_candidates:
            robot_xml = ROOT / "ScaleRetarget/assets/unitree_g1/g1_mocap_29dof.xml"
            from amass_to_scalebfm import semantic_fingerprint
            geometry_hash = semantic_fingerprint({"robot_geometry": robot_xml.parent}, {})
            results = prepare_sole_candidates(rows, output, robot_xml)
            if semantic_fingerprint({"robot_geometry": robot_xml.parent}, {}) != geometry_hash:
                raise ValueError("Robot XML/mesh geometry changed during audit")
            write_json(output / "sole_audit.json", {"schema": 1, "automatic_promotion": False,
                "robot_xml_sha256": sha256(robot_xml), "mujoco_version": __import__("mujoco").__version__,
                "robot_geometry_tree_sha256": geometry_hash,
                "thresholds": {"max_shift_m": 0.05, "max_penetration_m": 0.005},
                "formatter_sha256": sha256(ROOT / "ScaleRetarget/scaleretarget/formatter/kinematic_formatter.py"),
                "results": results, "counts": dict(Counter(row["status"] for row in results))})
        for label, path in INDEXES.items():
            if sha256(path) != index_hashes[label]:
                raise ValueError("Production index changed during pilot")
        for row in rows:
            for path_key, hash_key in (("source", "source_sha256"), ("baseline_pkl", "baseline_pkl_sha256"),
                                      ("baseline_packed", "baseline_packed_sha256")):
                if sha256(row[path_key]) != row[hash_key]:
                    raise ValueError(f"Input changed: {row[path_key]}")
        write_json(output / "status.json", {"result": "COMPLETE", "motions": len(rows),
            "inputs_verified_unchanged": True, "automatic_promotion": False,
            "training_started": False, "next_gate": "same-origin frozen BFM tracking and contact evaluation"})
    except BaseException as error:
        write_json(output / "failure.json", {"result": "FAIL", "error": str(error)})
        raise
    print(json.dumps({"output": str(output), "motions": len(rows), "training_started": False}))


if __name__ == "__main__":
    main()
