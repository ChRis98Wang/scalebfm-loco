#!/usr/bin/env python3
"""Post-hoc canonical-hand v2 experiment; not a demonstrated bug fix.

Default: a read-only CPU plan. --execute creates one NEW local directory with
prepared_source.npz and receipt.json. No retargeting, physics, GPU or training.
The parent's material samples and every moving array are preserved exactly.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import re
import sys

if __name__ == "__main__":
    sys.dont_write_bytecode = True
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = ROOT / "scripts/umr_smplx_source.py"
CORE_SHA256 = "d2b41fe5cffaed03637d3e37cbdf9fef1f4c30996d636b6146483bf5e9b66b94"
EXPERIMENT_SCHEMA = "bfm.umr_canonical_hand_reference_v2/1"
CANONICAL_FIELDS = (
    "canonical_points", "canonical_normals", "canonical_joint_positions",
    "canonical_joint_rotations", "actor_height",
)
FLAT_ATOL = 1e-6


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def array_fingerprint(value):
    a = np.ascontiguousarray(value)
    if a.dtype.hasobject:
        raise ValueError("Object arrays are forbidden")
    identity = json.dumps({"dtype": a.dtype.str, "shape": list(np.asarray(value).shape)}, sort_keys=True).encode()
    return hashlib.sha256(identity + b"\0" + a.tobytes()).hexdigest()


class FrozenInputs:
    def __init__(self):
        self.files = {}
        self.aliases = {}

    @staticmethod
    def signature(path):
        s = Path(path).stat()
        return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)

    def add(self, path, expected=None, *, follow=False):
        path = Path(path).absolute()
        resolved = path.resolve(strict=True)
        if path != resolved:
            if not follow:
                raise ValueError(f"Frozen input redirects through a symlink: {path}")
            if str(path) in self.aliases and self.aliases[str(path)] != str(resolved):
                raise ValueError(f"Input alias changed: {path}")
            self.aliases[str(path)] = str(resolved)
        if not resolved.is_file():
            raise ValueError(f"Expected regular input file: {resolved}")
        before = self.signature(resolved)
        if str(resolved) in self.files:
            digest, old = self.files[str(resolved)]
            if before != old:
                raise ValueError(f"Frozen input changed: {resolved}")
        else:
            digest = sha256(resolved)
            if self.signature(resolved) != before:
                raise ValueError(f"Input changed while hashing: {resolved}")
            self.files[str(resolved)] = (digest, before)
        if expected is not None and (not isinstance(expected, str)
                or not re.fullmatch(r"[0-9a-f]{64}", expected) or digest != expected):
            raise ValueError(f"SHA256 binding mismatch: {resolved}")
        return digest

    def recheck(self):
        for path, target in self.aliases.items():
            if str(Path(path).resolve(strict=True)) != target:
                raise ValueError(f"Frozen alias changed: {path}")
        for path, (digest, signature) in self.files.items():
            p = Path(path)
            if (p.is_symlink() or self.signature(p) != signature or sha256(p) != digest
                    or self.signature(p) != signature):
                raise ValueError(f"Frozen input changed before completion: {p}")

    def hashes(self):
        return {p: row[0] for p, row in sorted(self.files.items())}


def load_frozen_core(frozen):
    # Never import an unverified replacement under the old helper's name.
    frozen.add(CORE_PATH, CORE_SHA256)
    spec = importlib.util.spec_from_file_location("umr_hand_v2_frozen_core", CORE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    frozen.add(CORE_PATH, CORE_SHA256)
    return module


def constant_source_hand(raw):
    if "pose_hand" not in raw:
        raise ValueError("Missing pose_hand: this experiment cannot discard unknown hand motion")
    h = np.asarray(raw["pose_hand"])
    count = len(raw["root_orient"])
    if h.shape != (count, 90) or count < 3 or h.dtype.kind not in "fiu" or not np.isfinite(h).all():
        raise ValueError("Expected finite full-source (frames,90) pose_hand")
    if not np.array_equal(h, np.broadcast_to(h[:1], h.shape)):
        raise ValueError("Dynamic pose_hand is outside this constant-canonical-hand experiment")
    return np.array(h[0], copy=True)


def canonical_geometry(evaluated, model, parent, core):
    """Place one inferred rest mesh as v1 did, using stored material samples."""
    v = np.asarray(evaluated["vertices"])
    j = np.asarray(evaluated["joints"])
    rotations = np.asarray(evaluated["joint_rotations"])
    faces = np.asarray(model["faces"])
    if (v.ndim != 2 or v.shape[1] != 3 or j.shape != (55, 3)
            or rotations.shape != (55, 3, 3) or faces.ndim != 2 or faces.shape[1] != 3
            or faces.dtype.kind not in "iu" or not len(faces)
            or np.any(faces < 0) or np.any(faces >= len(v))
            or any(not np.isfinite(a).all() for a in (v, j, rotations))):
        raise ValueError("Invalid canonical model output/topology")
    if (not np.allclose(rotations.swapaxes(-1, -2) @ rotations, np.eye(3), rtol=0, atol=1e-6)
            or not np.allclose(np.linalg.det(rotations), 1., rtol=0, atol=1e-6)):
        raise ValueError("Canonical joint rotations must be proper SO(3)")
    v = v @ core.CANONICAL_ROTATION.T
    j = j @ core.CANONICAL_ROTATION.T
    rotations = core.CANONICAL_ROTATION @ rotations
    offset = np.array([-j[0, 0], -j[0, 1], -v[:, 2].min()])
    v += offset
    j += offset
    height = float(np.ptp(v[:, 2]))
    if not .8 < height < 2.5:
        raise ValueError("Canonical actor height is outside the frozen loader contract")
    face_ids = np.asarray(parent["face_indices"])
    bary = np.asarray(parent["barycentric"])
    if (face_ids.ndim != 1 or face_ids.dtype.kind not in "iu" or not len(face_ids)
            or np.any(face_ids < 0) or np.any(face_ids >= len(faces))):
        raise ValueError("Parent material face indices do not fit the bound model")
    points, normals = core.barycentric_transport(v, faces, face_ids, bary)
    return {"canonical_points": points, "canonical_normals": normals,
            "canonical_joint_positions": j, "canonical_joint_rotations": rotations,
            "actor_height": np.asarray(height)}, {
                "offset": offset.tolist(), "actor_height": height,
                "vertices_sha256": array_fingerprint(v), "faces_sha256": array_fingerprint(faces)}


def _verify_material_binding(model, parent, core):
    faces = np.asarray(model["faces"])
    skinning = np.asarray(model["skinning_weights"])
    if (skinning.ndim != 2 or skinning.shape[1] != 55 or not np.isfinite(skinning).all()
            or np.any(skinning < 0) or np.any(faces >= len(skinning))):
        raise ValueError("Invalid model skinning weights")
    if not np.array_equal(model["parents"], parent["parents"]):
        raise ValueError("Parent SMPL-X hierarchy differs from the bound model")
    face_ids, bary = parent["face_indices"], parent["barycentric"]
    weights = skinning[faces[face_ids]]
    binding = np.argmax(np.einsum("nvj,nv->nj", weights, bary), axis=1)
    membership = np.zeros((55, len(core.SEGMENTS)))
    membership[np.arange(55), [core.SEGMENTS.index(n) for n in core.JOINT_SEGMENTS]] = 1.
    segment = np.argmax(weights.mean(axis=1) @ membership, axis=1)
    if not np.array_equal(binding, parent["binding_joint_ids"]) or not np.array_equal(segment, parent["segment"]):
        raise ValueError("Stored material segment/binding IDs differ from the v1 model skinning")


def build_counterfactual(parent, raw, model, core):
    """Pure transform, with an injected CPU/fixture model; never sample or write.

    model = {faces, skinning_weights, parents, evaluate(hand90)}. evaluate returns
    untransformed SMPL-X vertices, first 55 joints and global joint rotations.
    """
    metadata = parent["metadata"]
    expected = {"shape_policy": "source_betas_all_static", "flat_hand_mean": True,
                "expression": "zero", "canonical_rotation": core.CANONICAL_ROTATION.tolist()}
    if any(metadata.get(k) != v for k, v in expected.items()):
        raise ValueError("Parent canonical conventions differ from frozen v1")
    if (raw["gender"] != metadata["source_gender"] or metadata["num_betas"] != len(raw["betas"])
            or not np.array_equal(raw["betas"], parent["betas"])):
        raise ValueError("Source shape/gender differs from v1 prepared source")
    hand = constant_source_hand(raw)
    flat, flat_identity = canonical_geometry(model["evaluate"](np.zeros(90)), model, parent, core)
    _verify_material_binding(model, parent, core)
    errors = {}
    for key in CANONICAL_FIELDS:
        before, reproduced = np.asarray(parent[key]), flat[key]
        if before.shape != reproduced.shape or not np.isfinite(before).all():
            raise ValueError(f"Parent canonical array shape/nonfinite: {key}")
        error = float(np.max(np.abs(before.astype(float) - reproduced)))
        errors[key] = error
        if error > FLAT_ATOL:
            raise ValueError(f"Cannot reproduce frozen v1 flat canonical: {key} max error {error}")
    if abs(float(metadata["actor_height"]) - float(parent["actor_height"])) > FLAT_ATOL:
        raise ValueError("Parent metadata/array actor height disagree")
    candidate, candidate_identity = canonical_geometry(model["evaluate"](hand), model, parent, core)
    arrays = {key: np.array(value, copy=True) for key, value in parent.items() if key != "metadata"}
    for key in CANONICAL_FIELDS:
        arrays[key] = np.asarray(candidate[key], dtype=np.asarray(parent[key]).dtype)
    protected = sorted(set(arrays) - set(CANONICAL_FIELDS) - {"metadata_json"})
    if any(array_fingerprint(arrays[key]) != array_fingerprint(parent[key]) for key in protected):
        raise ValueError("Counterfactual unexpectedly changed a moving/protected array")
    changes = [key for key in CANONICAL_FIELDS if array_fingerprint(arrays[key]) != array_fingerprint(parent[key])]
    old_height, new_height = float(parent["actor_height"]), float(arrays["actor_height"])
    proof = {"flat_canonical_max_abs_error": errors, "flat_reproduction_atol": FLAT_ATOL,
             "flat_geometry": flat_identity, "candidate_geometry": candidate_identity,
             "constant_source_hand": hand.tolist(), "constant_hand_sha256": array_fingerprint(hand),
             "source_hand_checked_frames": len(raw["pose_hand"]), "source_hand_temporal_change": 0.,
             "changed_canonical_arrays": changes, "protected_arrays": protected,
             "protected_array_sha256": {key: array_fingerprint(arrays[key]) for key in protected},
             "canonical_array_sha256": {key: {"parent": array_fingerprint(parent[key]),
                                                "candidate": array_fingerprint(arrays[key])} for key in CANONICAL_FIELDS},
             "actor_height_parent": old_height, "actor_height_candidate": new_height,
             "actor_height_change_m": new_height - old_height,
             "downstream_scale_ratio_v2_over_v1": old_height / new_height,
             "downstream_scale_unchanged": old_height == new_height,
             "material_samples_resampled": False, "moving_arrays_unchanged": True}
    return arrays, proof


def dependency_snapshot(frozen):
    """Hash installed package files (excluding bytecode), not just version text."""
    versions, paths = {}, set()
    for name in ("numpy", "scipy", "torch", "smplx"):
        dist = importlib.metadata.distribution(name)
        versions[name] = dist.version
        if dist.files is None:
            raise ValueError(f"No installed dependency file inventory for {name}")
        for relative in dist.files:
            if "__pycache__" in relative.parts or relative.suffix in (".pyc", ".pyo"):
                continue
            path = Path(dist.locate_file(relative)).absolute()
            if path.is_file():
                paths.add(path)
    for path in sorted(paths):
        frozen.add(path, follow=True)
    return {"versions": versions, "file_count": len(paths),
            "coverage": "Installed distribution inventories, all present non-bytecode files; symlink targets bound"}


def make_cpu_model(raw, model_path, core):
    import smplx
    import torch

    body = smplx.create(str(model_path), model_type="smplx", gender=raw["gender"],
                        ext="npz", use_pca=False, flat_hand_mean=True,
                        num_betas=len(raw["betas"]), batch_size=1).to(device="cpu").eval()
    if body.num_betas != len(raw["betas"]):
        raise ValueError("Model would truncate source shape")
    parents = body.parents.detach().cpu().numpy()
    def evaluate(hand):
        tensor = lambda a: torch.as_tensor(a, dtype=torch.float32, device="cpu")
        with torch.inference_mode():
            output = body(betas=tensor(raw["betas"])[None], global_orient=tensor(np.zeros((1, 3))),
                          body_pose=tensor(np.zeros((1, 63))), transl=tensor(np.zeros((1, 3))),
                          left_hand_pose=tensor(hand[:45])[None], right_hand_pose=tensor(hand[45:])[None],
                          jaw_pose=tensor(np.zeros((1, 3))), leye_pose=tensor(np.zeros((1, 3))),
                          reye_pose=tensor(np.zeros((1, 3))),
                          expression=torch.zeros((1, body.num_expression_coeffs), device="cpu"),
                          return_verts=True, return_full_pose=True)
        return {"vertices": output.vertices[0].detach().cpu().numpy(),
                "joints": output.joints[0, :55].detach().cpu().numpy(),
                "joint_rotations": core.global_joint_rotations(output.full_pose.detach().cpu().numpy(), parents)[0]}
    return {"evaluate": evaluate, "parents": parents, "faces": np.asarray(body.faces, dtype=np.int64),
            "skinning_weights": body.lbs_weights.detach().cpu().numpy()}


def output_location(output):
    output = Path(output).absolute()
    local = (ROOT / "local").resolve()
    if (output.exists() or output.is_symlink() or any(p.is_symlink() for p in output.parents)
            or not output.parent.is_dir() or output == local or not output.is_relative_to(local)
            or ".." in output.parts):
        raise ValueError("Output must be a NEW nonredirected directory below local, with existing parent")
    return output


def prepare(parent_path, parent_sha256, output, *, execute=False):
    output = output_location(output)
    parent_path = Path(parent_path).absolute()
    frozen = FrozenInputs()
    code_hash = frozen.add(Path(__file__).resolve())
    frozen.add(parent_path, parent_sha256)
    core = load_frozen_core(frozen)
    parent = core.load_prepared_source(parent_path)
    metadata = parent["metadata"]
    if metadata.get("adapter_sha256") != CORE_SHA256 or "canonical_hand_experiment" in metadata:
        raise ValueError("Require a SHA-bound original v1 prepared source, not a prior counterfactual")
    source_path, model_path = Path(metadata["source_file"]), Path(metadata["body_model"])
    frozen.add(source_path, metadata["source_sha256"])
    frozen.add(model_path, metadata["body_model_sha256"], follow=True)
    raw = core.load_amass(source_path)
    constant_source_hand(raw)  # Fail before model imports for missing/dynamic source hands.
    model_path = core.model_file_for_gender(model_path, raw["gender"])
    frozen.add(model_path, metadata["body_model_sha256"])
    runtime = dependency_snapshot(frozen)
    model = make_cpu_model(raw, model_path, core)
    arrays, proof = build_counterfactual(parent, raw, model, core)
    frozen.recheck()
    experiment = {"schema": EXPERIMENT_SCHEMA, "post_hoc": True, "demonstrated_bug_fix": False,
                  "v1_quality_gate": False, "canonical_change": "flat zero hand -> identical constant source pose_hand",
                  "moving_hand_body_wrist_pose_changed": False, "moving_arrays_unchanged": True,
                  "material_samples_resampled": False, "parent": str(parent_path), "parent_sha256": parent_sha256,
                  "code_sha256": code_hash, "frozen_core_sha256": CORE_SHA256,
                  "source_hand_sha256": proof["constant_hand_sha256"],
                  "actor_height_change_m": proof["actor_height_change_m"],
                  "downstream_scale_unchanged": proof["downstream_scale_unchanged"],
                  "downstream_scale_ratio_v2_over_v1": proof["downstream_scale_ratio_v2_over_v1"],
                  "dependency_runtime": runtime, "input_sha256": frozen.hashes()}
    new_metadata = copy.deepcopy(metadata)
    # Keep the source schema for frozen-loader compatibility, but identify the
    # real producer of the changed canonical arrays rather than impersonate v1.
    new_metadata["actor_height"] = float(arrays["actor_height"])
    new_metadata["parent_adapter_sha256"] = CORE_SHA256
    new_metadata["adapter_sha256"] = code_hash
    new_metadata["canonical_hand_experiment"] = experiment
    arrays["metadata_json"] = np.asarray(json.dumps(new_metadata, sort_keys=True, allow_nan=False))
    report = {"schema": EXPERIMENT_SCHEMA, "result": "PLAN_ONLY", "execute": execute,
              "output_directory": str(output), "prepared_source": str(output / "prepared_source.npz"),
              "post_hoc": True, "demonstrated_bug_fix": False, "automatic_promotion": False,
              "training_updates": 0, "physics_stepped": False, "retargeting_started": False,
              "preprocessing_device": "cpu", "proof": proof, "experiment": experiment,
              "inputs_verified_unchanged": True,
              "limitations": ["Only a canonical-hand counterfactual, not a proven correction or acceptance gate.",
                              "If actor_height changes, the unmodified core also changes moving target scale; inspect the explicit scale flag.",
                              "Use a NEW retarget output and NEW setup cache; do not modify frozen v1."]}
    if execute:
        output_location(output)
        frozen.recheck()
        output.mkdir(exist_ok=False)
        target = output / "prepared_source.npz"
        with target.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        loaded = core.load_prepared_source(target)
        if any(array_fingerprint(loaded[k]) != array_fingerprint(arrays[k]) for k in arrays):
            raise ValueError("Written archive differs from planned arrays; incomplete output retained")
        frozen.recheck()
        report.update(result="COMPLETE_CANONICAL_ONLY_EXPERIMENT", prepared_source_sha256=sha256(target))
        with (output / "receipt.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--parent-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    report = prepare(args.parent, args.parent_sha256, args.output, execute=args.execute)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
