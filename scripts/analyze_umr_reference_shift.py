#!/usr/bin/env python3
"""Post-hoc descriptive UMR reference shift; NOT a frozen-v1 quality gate.

CPU-only: no training, physics, policy evaluation or promotion decisions.
Reference mathematics uses NumPy; the unchanged source-contract validator also
uses CPU Torch to reproduce the legacy packing clock.
The default is read-only JSON on stdout. --outputnew writes one NEW JSON file
in an existing, non-symlink directory; it never overwrites or creates indexes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

if __name__ == "__main__":
    sys.dont_write_bytecode = True
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import build_umr_behavior_ab as dataset_contract

SCHEMA = "bfm.umr_reference_shift/1"
REFERENCE_BODIES = (
    "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", "torso_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
)
GROUPS = {
    "all14": REFERENCE_BODIES,
    "nonpelvis13": REFERENCE_BODIES[1:],
    "wrists": ("left_wrist_yaw_link", "right_wrist_yaw_link"),
    "ankles": ("left_ankle_roll_link", "right_ankle_roll_link"),
    "torso": ("torso_link",),
}
UNIT_TOLERANCE = 2e-5
# Pure helpers loaded lazily by the unchanged validator. Hash their source
# before calling it, without importing physics or robot runtime packages.
VALIDATION_CODE = tuple(ROOT / "scripts" / name for name in (
    "umr_backend.py", "umr_pair_dataset.py", "audit_packed_retarget_pilot.py",
    "amass_to_scalebfm.py", "compare_mask_evaluations.py")) + (
        ROOT / "ScaleTrack/scripts/pretrain/rsl_rl/evaluation_manifest.py",)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Snapshot:
    """Hash before use and rehash after computation, including alias targets."""

    def __init__(self):
        self.files = {}
        self.aliases = {}

    @staticmethod
    def signature(path):
        st = Path(path).stat()
        return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)

    def add(self, path, expected=None):
        path = Path(path).absolute()
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Expected a regular frozen input: {path}")
        before = self.signature(path)
        if str(path) in self.files:
            digest, original = self.files[str(path)]
            if before != original:
                raise ValueError(f"Input changed during analysis: {path}")
        else:
            digest = sha256(path)
            if self.signature(path) != before:
                raise ValueError(f"Input changed while hashing: {path}")
            self.files[str(path)] = (digest, before)
        if expected is not None and (not isinstance(expected, str)
                or not re.fullmatch(r"[0-9a-f]{64}", expected) or digest != expected):
            raise ValueError(f"SHA256 binding mismatch: {path}")
        return digest

    def read_json(self, path, expected=None):
        self.add(path, expected)
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        self.add(path, expected)
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object: {path}")
        return value

    # Compatibility with the frozen index reader; it performs no writes.
    verify = add

    def recheck(self):
        for path, target in self.aliases.items():
            if str(Path(path).resolve(strict=True)) != target:
                raise ValueError(f"Frozen alias changed: {path}")
        for path, (digest, signature) in self.files.items():
            p = Path(path)
            if (p.is_symlink() or self.signature(p) != signature or sha256(p) != digest
                    or self.signature(p) != signature):
                raise ValueError(f"Frozen input changed before report completion: {path}")

    def hashes(self):
        return {path: record[0] for path, record in sorted(self.files.items())}


def _unit_quaternions(value):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim < 1 or value.shape[-1] != 4 or not np.isfinite(value).all():
        raise ValueError("Expected finite wxyz quaternions ending in four components")
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    if not value.size or np.any(np.abs(norm - 1.) > UNIT_TOLERANCE):
        raise ValueError("Quaternion must be unit length; no repair of invalid input")
    return value / norm


def quaternion_angle(first, second):
    """Shortest SO(3) radians; normalized sign-invariant chord atan2.

    Same numerical method as audit_umr_body_fk.quaternion_angle, with additional
    unit-input validation. Chord atan2 avoids acos cancellation near identity.
    """
    first, second = _unit_quaternions(first), _unit_quaternions(second)
    if first.shape != second.shape:
        raise ValueError("Quaternion shapes must agree")
    second = np.where(np.sum(first * second, axis=-1, keepdims=True) < 0, -second, second)
    return 4. * np.arctan2(np.linalg.norm(first - second, axis=-1),
                          np.linalg.norm(first + second, axis=-1))


def _multiply(q, r):
    qw, qv, rw, rv = q[..., :1], q[..., 1:], r[..., :1], r[..., 1:]
    return np.concatenate((qw * rw - np.sum(qv * rv, axis=-1, keepdims=True),
                           qw * rv + rw * qv + np.cross(qv, rv)), axis=-1)


def _rotation(q):
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                     2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                     2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), axis=-1).reshape(q.shape[:-1] + (3, 3))


def relative_link_pose(positions, quaternions, body_names):
    """Each side's own pelvis full-SO(3) link frame, never a COM/yaw-only frame."""
    names = list(body_names)
    if (any(not isinstance(n, str) for n in names) or len(set(names)) != len(names)
            or not set(REFERENCE_BODIES).issubset(names)):
        raise ValueError("Missing, duplicate or invalid named reference links")
    p = np.asarray(positions, dtype=np.float64)
    q = _unit_quaternions(quaternions)
    if (p.ndim != 3 or p.shape[0] < 1 or p.shape[1:] != (len(names), 3)
            or q.shape != p.shape[:-1] + (4,) or not np.isfinite(p).all()):
        raise ValueError("Expected matching finite (frames, named links, xyz/wxyz) arrays")
    ids = [names.index(name) for name in REFERENCE_BODIES]
    pelvis = names.index("pelvis")
    qp = q[:, pelvis:pelvis+1]
    relative_p = np.einsum("...ji,...j->...i", _rotation(qp), p[:, ids] - p[:, pelvis:pelvis+1])
    relative_q = _unit_quaternions(_multiply(qp * np.array([1., -1., -1., -1.]), q[:, ids]))
    return relative_p, relative_q


def _statistics(values):
    return {"mean": float(np.mean(values)), "p95": float(np.percentile(values, 95, method="linear"))}


def summarize_errors(position_errors_m, rotation_errors_rad):
    p, r = (np.asarray(v, dtype=np.float64) for v in (position_errors_m, rotation_errors_rad))
    if (p.ndim != 2 or p.shape[0] < 1 or p.shape[1] != 14 or r.shape != p.shape
            or not np.isfinite(p).all() or not np.isfinite(r).all()
            or np.any(p < 0) or np.any(r < 0) or np.any(r > np.pi + 1e-12)):
        raise ValueError("Expected nonempty finite nonnegative frame-by-14 errors; angle <= pi")
    r = np.rad2deg(r)
    def stats(ids):
        # Group percentiles are over frame x body, not averages of body p95s.
        return {"position_m": _statistics(p[:, ids]), "rotation_deg": _statistics(r[:, ids])}
    return {"per_link": {name: stats([i]) for i, name in enumerate(REFERENCE_BODIES)},
            "groups": {name: stats([REFERENCE_BODIES.index(n) for n in members])
                       for name, members in GROUPS.items()}}


def aggregate_origins(records):
    if not records or len({r["origin_id"] for r in records}) != len(records):
        raise ValueError("Require distinct, nonempty origin records")
    if any(type(r["frame_count"]) is not int or r["frame_count"] < 1 for r in records):
        raise ValueError("Invalid origin frame count")
    result = {}
    for section, labels in (("per_link", REFERENCE_BODIES), ("groups", GROUPS)):
        result[section] = {}
        for label in labels:
            result[section][label] = {}
            for metric in ("position_m", "rotation_deg"):
                values = np.asarray([[r["statistics"][section][label][metric][k]
                                      for k in ("mean", "p95")] for r in records], dtype=np.float64)
                if not np.isfinite(values).all() or np.any(values < 0):
                    raise ValueError("Invalid origin statistics")
                result[section][label][metric] = dict(zip(
                    ("origin_equal_mean", "mean_of_per_origin_p95"), map(float, values.mean(axis=0))))
    return result


def _load_packed(path, digest, native_digest, pipeline, frames, frozen):
    frozen.add(path, digest)
    with np.load(path, allow_pickle=False) as archive:
        z = {key: archive[key] for key in archive.files}
    for key, expected in (("format_version", 3), ("fps", 50), ("quaternion_order", "wxyz"),
                          ("source_sha256", native_digest), ("pipeline_fingerprint", pipeline)):
        if key not in z or z[key].shape != () or z[key].item() != expected:
            raise ValueError(f"Packed scalar mismatch: {path}: {key}")
    for key, expected in (("joint_names", dataset_contract.pairs.ARTICULATION_JOINT_NAMES),
                          ("body_names", dataset_contract.pairs.BODY_NAMES)):
        if (key not in z or z[key].ndim != 1 or z[key].dtype.kind not in "US"
                or z[key].tolist() != list(expected)):
            raise ValueError(f"Packed named articulation order mismatch: {path}: {key}")
    shapes = {"joint_pos": (frames, 29), "joint_vel": (frames, 29),
              "body_pos_w": (frames, 30, 3), "body_quat_w": (frames, 30, 4),
              "body_lin_vel_w": (frames, 30, 3), "body_ang_vel_w": (frames, 30, 3),
              "reference_root_pos": (frames, 3), "reference_root_quat_w": (frames, 4)}
    for key, shape in shapes.items():
        if (key not in z or z[key].shape != shape or z[key].dtype.kind not in "fiu"
                or not np.isfinite(z[key]).all()):
            raise ValueError(f"Invalid packed frame array: {path}: {key}")
    _unit_quaternions(z["body_quat_w"])
    _unit_quaternions(z["reference_root_quat_w"])
    frozen.add(path, digest)
    return z


def _code_paths():
    paths = {Path(__file__).resolve(), *VALIDATION_CODE}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if isinstance(filename, (str, Path)):
            path = Path(filename).resolve()
            if path.suffix == ".py" and path.is_relative_to(ROOT / "scripts"):
                paths.add(path)
    return sorted(paths)


def analyze_dataset(dataset):
    """Fully validate the original dataset, then compute descriptive statistics."""
    dataset = Path(dataset).absolute()
    frozen = Snapshot()
    frozen.add(dataset)
    code = {str(path): frozen.add(path) for path in _code_paths()}
    manifest = dataset_contract.validate_manifest(dataset)
    frozen.add(dataset)
    for path, expected in manifest["input_sha256"].items():
        frozen.add(path, expected)
    for path, target in manifest["resolved_input_aliases"].items():
        if str(Path(path).resolve(strict=True)) != target:
            raise ValueError(f"Manifest source alias changed: {path}")
        frozen.aliases[path] = target
    # Lazy validator helpers are pinned by the original data contract; include
    # all loaded local helper source files explicitly in this report as well.
    for path in _code_paths():
        code[str(path)] = frozen.add(path)
    batch = Path(manifest["evidence"]["batch"])
    pilot = frozen.read_json(batch / "paired_manifest.json", manifest["evidence"]["pins"]["manifest"])
    fk = frozen.read_json(batch / "body_fk_audit_20260909a.json", manifest["evidence"]["pins"]["fk"])
    rows = {r["origin_id"]: r for r in pilot["rows"]}
    fk_rows = {(r["origin_id"], r["side"]): r for r in fk["results"]}
    provenance = {r["origin_id"]: r for r in manifest["source_provenance"]}
    indexes = {}
    entries = {**manifest["arms"], **manifest["development"]["indices"]}
    for key, entry in entries.items():
        indexes[key] = dataset_contract.read_index(entry["index"], frozen, entry["sha256"])
    splits = {}
    for split, origins, akey, bkey, count in (
            ("train17", manifest["target_origins"], "a", "b", 17),
            ("dev10", manifest["development"]["origin_ids"], "baseline", "candidate", 10)):
        if len(origins) != count or len(set(origins)) != count:
            raise ValueError("Require all fixed 17 target train and 10 development origins")
        records = []
        for origin in origins:
            row, source = rows[origin], provenance[origin]
            if (row["source"] != source["canonical_source"]
                    or row["source_sha256"] != source["source_sha256"]
                    or row["split"] != ("train" if split == "train17" else "validation")):
                raise ValueError(f"Original source identity/split mismatch: {origin}")
            frozen.add(row["source"], row["source_sha256"])
            pair = frozen.read_json(row["pair_receipt"], row["pair_receipt_sha256"])
            frames = row["expected_packed_frames"]
            if (type(frames) is not int or frames < 1 or pair["frames"] != frames + 1
                    or pair["origin_id"] != origin or pair["source_sha256"] != row["source_sha256"]
                    or pair["expected_packed_frames"] != frames or pair["fps"] != 50):
                raise ValueError(f"Paired source/frame clock mismatch: {origin}")
            window = pair["common_window_proof"]
            if (window["pair_start_s"] != 0 or window["pair_frames"] != frames + 1
                    or not np.isclose(window["pair_last_sample_s"], frames / 50, rtol=0, atol=1e-12)
                    or window["pair_last_sample_s"] > 5):
                raise ValueError(f"Paired common window mismatch: {origin}")
            frozen.add(row["sampling_proof"], row["sampling_proof_sha256"])
            with np.load(row["sampling_proof"], allow_pickle=False) as proof:
                if (proof["times"].shape != (frames + 1,)
                        or not np.allclose(proof["times"], np.arange(frames + 1) / 50, rtol=0, atol=1e-12)
                        or any(not np.array_equal(proof[k], np.arange(frames + 1))
                               for k in ("packed_indices", "candidate_indices"))):
                    raise ValueError(f"Pair sample grids are not the same half-open clock: {origin}")
            local, payloads = [], {}
            for side, key in (("baseline", akey), ("candidate", bkey)):
                path = indexes[key][origin]
                digest = manifest["input_sha256"][path]
                record = fk_rows[(origin, side)]
                if (record["path"] != path or record["sha256"] != digest
                        or record["frames"] != frames or record["passed"] is not True):
                    raise ValueError(f"Link-frame FK evidence does not bind the packed data: {origin}/{side}")
                frozen.add(row[side + "_pkl"], row[side + "_pkl_sha256"])
                z = _load_packed(path, digest, row[side + "_pkl_sha256"],
                                 manifest["evidence"]["paired_packaging_fingerprint"], frames, frozen)
                local.append(relative_link_pose(z["body_pos_w"], z["body_quat_w"], z["body_names"].tolist()))
                payloads[side] = {"path": path, "sha256": digest,
                                  "native_sha256": row[side + "_pkl_sha256"]}
            (pa, qa), (pb, qb) = local
            statistics = summarize_errors(np.linalg.norm(pb - pa, axis=-1), quaternion_angle(qa, qb))
            records.append({"origin_id": origin, "frame_count": frames, "split": row["split"],
                            "source": row["source"], "source_sha256": row["source_sha256"],
                            "pair_receipt": row["pair_receipt"], "pair_receipt_sha256": row["pair_receipt_sha256"],
                            "sampling_proof": row["sampling_proof"], "sampling_proof_sha256": row["sampling_proof_sha256"],
                            "payloads": payloads, "statistics": statistics})
        splits[split] = {"origin_count": count, "frames_per_side": sum(r["frame_count"] for r in records),
                         "aggregate": aggregate_origins(records), "origins": records}
    frozen.recheck()
    return {"schema": SCHEMA, "result": "COMPLETE_DESCRIPTIVE_DIAGNOSTIC", "post_hoc": True,
            "purpose": "Developed after observing development results; reference distribution shift only",
            "v1_quality_gate": False, "policy_errors_measured": False, "promotion_recommendation": None,
            "automatic_promotion": False, "physics_stepped": False, "training_updates": 0,
            "dataset_manifest": str(dataset), "dataset_manifest_sha256": frozen.hashes()[str(dataset)],
            "reference_body_names": list(REFERENCE_BODIES), "groups": {k: list(v) for k, v in GROUPS.items()},
            "definition": {"frame": "Each side's own pelvis link frame; full SO(3), not yaw-only and not COM",
                           "position": "R_pelvis.T @ (p_body - p_pelvis); Euclidean A/B distance in metres",
                           "orientation": "conjugate(q_pelvis) * q_body; wxyz unit; shortest sign-invariant SO(3) degrees",
                           "within_origin": "mean/p95 over frames for links; frame x body for groups; numpy linear percentile",
                           "across_origins": "Equal mean of origin means and equal mean of origin p95s, NOT pooled-frame p95",
                           "clock": "Same 50 Hz half-open packed grid, native inclusive last endpoint dropped equally"},
            "limitations": ["Not policy error, solver correctness, or a frozen-v1 acceptance metric.",
                            "Removes each side's global pelvis translation and orientation; cannot measure global tracking or floor height.",
                            "Whole pipelines differ in shape/scale/constraints/retargeting; do not attribute differences solely to penetration correction.",
                            "Pelvis is zero by construction; report nonpelvis13 separately to avoid dilution.",
                            "Train is geometry-selected 17; development includes all 10 (including 3 geometric failures), not matched random populations.",
                            "Short windows are not full-motion replacement; development is not a fresh final test."],
            "target_origin_fractions": {"focused_train_pool": 17 / 256, "original_full_train": 17 / 7174},
            "splits": splits, "code_sha256": dict(sorted(code.items())), "input_sha256": frozen.hashes(),
            "resolved_input_aliases": dict(sorted(frozen.aliases.items())),
            "inputs_verified_unchanged": True, "hash_verification": "All bound files SHA256 checked before use and rehashed after computation",
            "runtime": {"python": sys.version, "numpy": np.__version__,
                        "validator_torch": getattr(sys.modules.get("torch"), "__version__", None)}}


def _new_output(path):
    path = Path(path).absolute()
    if path.exists() or path.is_symlink() or any(p.is_symlink() for p in path.parents):
        raise ValueError("--outputnew requires a NEW non-symlink file path")
    if not path.parent.is_dir():
        raise ValueError("--outputnew parent must already exist; no directories are created")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--outputnew", type=Path, help="Write exactly one NEW JSON; default stdout only")
    args = parser.parse_args(argv)
    output = _new_output(args.outputnew) if args.outputnew else None
    report = analyze_dataset(args.dataset)
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if output is None:
        print(text, end="")
    else:
        output = _new_output(output)
        with output.open("x", encoding="utf-8") as stream:
            stream.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
