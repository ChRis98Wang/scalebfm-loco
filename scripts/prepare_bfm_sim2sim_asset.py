#!/usr/bin/env python3
"""Create a versioned simulation-only G1 torque-contract variant.

Compare named FK first; permit only the known two hip-roll motor limits to
change from 88 to the verified nominal 139 Nm. Keep the source XML and all
meshes untouched. CPU model compilation/forward kinematics only, never mj_step.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BRIDGE_XML = ROOT / "ScaleBridge/scalebridge/data/robot/g1_29dof/g1_29dof.xml"
TRAINING_XML = ROOT / "ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof/g1_29dof.xml"
HIPS = {"left_hip_roll_joint", "right_hip_roll_joint"}
SCHEMA = "bfm.g1_sim2sim_nominal_torque_variant/1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_file(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular input file: {path}")
    return path.resolve(strict=True)


def names(model, kind, count):
    import mujoco
    values = [mujoco.mj_id2name(model, kind, index) for index in range(count)]
    if any(not value for value in values) or len(set(values)) != len(values):
        raise ValueError("Models must have explicit unique body/joint/actuator names")
    return values


def difference(left, right, *, quaternion=False):
    direct = float(np.max(np.abs(np.asarray(left) - np.asarray(right))))
    return min(direct, float(np.max(np.abs(np.asarray(left) + np.asarray(right))))) if quaternion else direct


def compare_named_fk(bridge, training, joint_names, body_names):
    """Compare named local transforms, then independent MuJoCo FK on 16 poses."""
    import mujoco
    kind = mujoco.mjtObj
    models = (bridge, training)
    bodies = [names(model, kind.mjOBJ_BODY, model.nbody) for model in models]
    joints = [names(model, kind.mjOBJ_JOINT, model.njnt) for model in models]
    if len(joint_names) != 29 or len(set(joint_names)) != 29 or len(body_names) != 30 or len(set(body_names)) != 30:
        raise ValueError("Expected metadata for precisely 29 G1 joints and 30 reference bodies")
    if set(body_names) != set(bodies[1]) - {"world"}:
        raise ValueError("Training body set differs from complete policy FK metadata")
    for model, joint_list, body_list in zip(models, joints, bodies):
        hinge_names = {name for index, name in enumerate(joint_list) if model.jnt_type[index] == mujoco.mjtJoint.mjJNT_HINGE}
        if hinge_names != set(joint_names) or not set(body_names).issubset(body_list):
            raise ValueError("Model named robot joints/bodies differ from policy metadata")
        if model.nq != 36 or model.nv != 35 or model.njnt != 30 or model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("Expected exactly one free root plus 29 scalar G1 joints")
    differences = []
    for name in body_names:
        i, j = (values.index(name) for values in bodies)
        for field in ("body_pos", "body_quat"):
            error = difference(getattr(bridge, field)[i], getattr(training, field)[j], quaternion=field == "body_quat")
            if error > 1e-10:
                differences.append({"body": name, "field": field, "max_abs_error": error})
        if bodies[0][bridge.body_parentid[i]] != bodies[1][training.body_parentid[j]]:
            differences.append({"body": name, "field": "parent"})
    for name in joint_names:
        i, j = (values.index(name) for values in joints)
        for field in ("jnt_axis", "jnt_pos", "jnt_type"):
            error = difference(getattr(bridge, field)[i], getattr(training, field)[j])
            if error > 1e-10:
                differences.append({"joint": name, "field": field, "max_abs_error": error})
        if bodies[0][bridge.jnt_bodyid[i]] != bodies[1][training.jnt_bodyid[j]]:
            differences.append({"joint": name, "field": "body"})
    if differences:
        raise ValueError(f"Named robot FK mismatch; refusing asset conversion: {differences}")
    extra = []
    for name in sorted(set(bodies[0]) - set(bodies[1])):
        index = bodies[0].index(name)
        if bridge.body_jntnum[index] != 0:
            raise ValueError("Unexpected additional articulated bridge body")
        extra.append({"name": name, "parent": bodies[0][bridge.body_parentid[index]],
                      "mass_kg": float(bridge.body_mass[index]), "preserved": True})
    data = [mujoco.MjData(model) for model in models]
    rng = np.random.default_rng(42)
    maximum_position, maximum_quaternion = 0., 0.
    for _ in range(16):
        root = rng.normal(size=3)
        quaternion = rng.normal(size=4)
        quaternion /= np.linalg.norm(quaternion)
        angles = rng.uniform(-.2, .2, 29)
        for model, state, joint_list in zip(models, data, joints):
            state.qpos[:3], state.qpos[3:7] = root, quaternion
            for name, value in zip(joint_names, angles):
                state.qpos[model.jnt_qposadr[joint_list.index(name)]] = value
            mujoco.mj_forward(model, state)
        for name in body_names:
            i, j = (values.index(name) for values in bodies)
            maximum_position = max(maximum_position, difference(data[0].xpos[i], data[1].xpos[j]))
            maximum_quaternion = max(maximum_quaternion, difference(data[0].xquat[i], data[1].xquat[j], quaternion=True))
    if maximum_position > 1e-10 or maximum_quaternion > 1e-10:
        raise ValueError("Independent common-root FK does not match training robot")
    return {"result": "PASS", "named_joints": 29, "named_reference_bodies": 30,
            "random_pose_count": 16, "seed": 42, "joint_sample_range_rad": [-.2, .2],
            "maximum_position_error_m": maximum_position, "maximum_quaternion_sign_invariant_error": maximum_quaternion,
            "extra_fixed_bridge_bodies": extra,
            "free_root_joint_names": {"bridge": joints[0][0], "training": joints[1][0]},
            "model_mass_kg": {"bridge": float(bridge.body_mass.sum()), "training": float(training.body_mass.sum())},
            "physics_stepped": False, "interpretation": "Named reference FK agreement, not identical dynamics/contact assets"}


def torque_changes(model, metadata):
    import mujoco
    joint_names = metadata["joint_names"]
    declared = np.asarray(metadata["torque_limit"], dtype=float)
    if declared.shape != (29,) or not np.isfinite(declared).all() or np.any(declared <= 0):
        raise ValueError("Invalid nominal metadata torque limits")
    actuator_names = names(model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu)
    joint_list = names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)
    if set(actuator_names) != set(joint_names) or model.nu != 29:
        raise ValueError("Expected 29 named motor transmissions matching metadata")
    if not np.allclose(model.actuator_gear, np.tile([1., 0, 0, 0, 0, 0], (29, 1))):
        raise ValueError("Expected direct unit-gear joint motors")
    changes = []
    for index, name in enumerate(actuator_names):
        joint = int(model.actuator_trnid[index, 0])
        if joint_list[joint] != name or not model.actuator_ctrllimited[index] or not model.jnt_actfrclimited[joint]:
            raise ValueError("Expected bounded same-name motor/joint transmission")
        target = float(declared[joint_names.index(name)])
        motor = model.actuator_ctrlrange[index]
        force = model.jnt_actfrcrange[joint]
        if not np.allclose(force, [-target, target], atol=1e-6, rtol=0):
            raise ValueError(f"Unexpected joint-force limit mismatch for {name}; no conversion authorized")
        if not np.allclose(motor, [-target, target], atol=1e-6, rtol=0):
            if name not in HIPS or target != 139. or not np.array_equal(motor, [-88., 88.]):
                raise ValueError(f"Unexpected motor torque difference for {name}; no conversion authorized")
            changes.append({"joint": name, "element": "motor", "attribute": "ctrlrange",
                            "before": [-88., 88.], "after": [-139., 139.],
                            "reason": "Match hash-bound nominal training torque metadata; simulation-only"})
    if {change["joint"] for change in changes} != HIPS:
        raise ValueError("This version permits exactly the two known 88 -> 139 Nm hip-roll changes")
    return changes


def parse_tree(path):
    return ET.parse(path, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))


def mesh_inputs(xml):
    tree = parse_tree(xml)
    compiler = tree.getroot().find("compiler")
    meshdir = (xml.parent / compiler.get("meshdir", ".")).resolve(strict=True)
    inputs = {}
    for mesh in tree.getroot().findall("./asset/mesh"):
        if mesh.get("file"):
            path = checked_file(meshdir / mesh.get("file"))
            inputs[str(path)] = sha256(path)
    return meshdir, inputs


def prepare_asset(bridge_xml, metadata_path, training_xml, output):
    import mujoco
    output = Path(output).absolute()
    if any(path.is_symlink() for path in (output, *output.parents)):
        raise ValueError("Asset output must not redirect through a symlink ancestor")
    if output.exists() or output.is_symlink():
        raise FileExistsError("Asset output directory must be new")
    bridge_xml, metadata_path, training_xml = map(checked_file, (bridge_xml, metadata_path, training_xml))
    metadata = json.loads(metadata_path.read_text())
    if (metadata.get("runtime_backend") != "torchscript" or metadata.get("export_verification") != "PASS"
            or metadata.get("source_profile_verified_unchanged") is not True
            or metadata.get("fk_xml_sha256") != sha256(training_xml)):
        raise ValueError("Expected verified, hash-bound export metadata for this training FK XML")
    policy = checked_file(metadata_path.with_name(metadata_path.name.removesuffix("_metadata.json") + ".pt"))
    if sha256(policy) != metadata.get("policy_sha256"):
        raise ValueError("Exported policy differs from metadata SHA256")
    export_report_path = checked_file(metadata_path.parent / "export_report.json")
    export_report = json.loads(export_report_path.read_text())
    if (export_report.get("schema") != "bfm.portable_export/1" or export_report.get("result") != "PASS"
            or export_report.get("runtime_backend") != "torchscript"
            or export_report.get("metadata_sha256") != sha256(metadata_path)
            or export_report.get("artifact_sha256") != sha256(policy)
            or Path(export_report.get("artifact", "")).resolve() != policy
            or export_report.get("inputs_verified_unchanged") is not True
            or export_report.get("training_updates") != 0 or export_report.get("physics_stepped") is not False
            or export_report.get("hardware_accessed") is not False or export_report.get("automatic_promotion") is not False
            or export_report.get("masks_checked") != 8 or export_report.get("input_seeds_checked") != 2):
        raise ValueError("Export report does not bind a successful unchanged policy and metadata")
    checks = export_report.get("checks", [])
    if len(checks) != 16 or {(row.get("seed"), row.get("mode_index")) for row in checks} != {
            (seed, mode) for seed in (42, 314159) for mode in range(8)}:
        raise ValueError("Export report must cover both input seeds and all eight masks")
    for row in checks:
        for field in ("saved_vs_eager_action", "saved_vs_eager_pd_rad", "saved_vs_native_action",
                      "saved_vs_native_pd_rad", "prop_observation_max_error", "task_observation_max_error"):
            value = row.get(field)
            limit = 2e-5 if "observation" in field else 1e-4
            if (row.get("passed") is not True or isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not 0 <= value <= limit):
                raise ValueError("Export report contains an invalid/failed numerical check")
    protected = {str(path): sha256(path) for path in (bridge_xml, metadata_path, training_xml, policy,
                 export_report_path, Path(__file__).resolve())}
    export_inputs = export_report.get("input_sha256", {})
    if not export_inputs or export_inputs.get(str(training_xml)) != sha256(training_xml):
        raise ValueError("Export report does not bind this exact training FK XML")
    for path, digest in export_inputs.items():
        if sha256(checked_file(path)) != digest:
            raise ValueError(f"Export verification input changed: {path}")
        protected[str(checked_file(path))] = digest
    meshdir, bridge_meshes = mesh_inputs(bridge_xml)
    _, training_meshes = mesh_inputs(training_xml)
    protected.update(bridge_meshes)
    protected.update(training_meshes)
    bridge = mujoco.MjModel.from_xml_path(str(bridge_xml))
    training = mujoco.MjModel.from_xml_path(str(training_xml))
    before_fk = compare_named_fk(bridge, training, metadata["joint_names"], metadata["body_names"])
    changes = torque_changes(bridge, metadata)
    tree = parse_tree(bridge_xml)
    original = ET.tostring(tree.getroot())
    compiler = tree.getroot().find("compiler")
    previous_meshdir = compiler.get("meshdir")
    compiler.set("meshdir", str(meshdir))
    for change in changes:
        matches = [motor for motor in tree.getroot().findall("./actuator/motor") if motor.get("name") == change["joint"]]
        if len(matches) != 1 or matches[0].get("joint") != change["joint"]:
            raise ValueError("Unexpected XML motor layout")
        matches[0].set("ctrlrange", "-139 139")
    # Prove the XML tree has no other semantic edits, including comments/defaults.
    reverse = copy.deepcopy(tree)
    restored = reverse.getroot().find("compiler")
    if previous_meshdir is None:
        restored.attrib.pop("meshdir", None)
    else:
        restored.set("meshdir", previous_meshdir)
    for motor in reverse.getroot().findall("./actuator/motor"):
        if motor.get("name") in HIPS:
            source_motor = next(m for m in parse_tree(bridge_xml).getroot().findall("./actuator/motor") if m.get("name") == motor.get("name"))
            motor.set("ctrlrange", source_motor.get("ctrlrange"))
    if ET.tostring(reverse.getroot()) != original:
        raise ValueError("Changes exceed the explicit compiler-path and two-motor allowlist")
    output.mkdir(parents=True, exist_ok=False)
    variant_path = output / "g1_29dof_nominal_torque.xml"
    with variant_path.open("xb") as stream:
        tree.write(stream, encoding="utf-8", xml_declaration=True)
    variant = mujoco.MjModel.from_xml_path(str(variant_path))
    unchanged = ("body_mass", "body_inertia", "body_ipos", "body_iquat", "body_pos", "body_quat", "body_parentid",
                 "geom_type", "geom_size", "geom_pos", "geom_quat", "geom_friction", "geom_solref", "geom_solimp",
                 "geom_contype", "geom_conaffinity", "dof_damping", "dof_frictionloss", "dof_armature",
                 "jnt_range", "jnt_actfrcrange", "mesh_vert", "mesh_face")
    for field in unchanged:
        if not np.array_equal(getattr(bridge, field), getattr(variant, field)):
            raise ValueError(f"Unexpected non-torque model change: {field}")
    after_fk = compare_named_fk(variant, training, metadata["joint_names"], metadata["body_names"])
    variant_actuators = names(variant, mujoco.mjtObj.mjOBJ_ACTUATOR, variant.nu)
    for name, target in zip(metadata["joint_names"], metadata["torque_limit"]):
        if not np.allclose(variant.actuator_ctrlrange[variant_actuators.index(name)], [-target, target], atol=1e-6, rtol=0):
            raise ValueError("Variant does not match nominal metadata torque limits")
    if any(sha256(path) != digest for path, digest in protected.items()):
        raise ValueError("Source XML/model/metadata/policy/code changed during conversion")
    receipt = {"schema": SCHEMA, "result": "COMPLETE_SIMULATION_ASSET_ONLY", "mujoco_version": mujoco.__version__,
        "input_sha256": protected, "metadata": str(metadata_path), "original_xml": str(bridge_xml),
        "export_report": str(export_report_path), "export_report_sha256": sha256(export_report_path),
        "training_fk_xml": str(training_xml), "variant_xml": str(variant_path), "variant_sha256": sha256(variant_path),
        "changes": changes, "compiler_meshdir": {"before": previous_meshdir, "after": str(meshdir), "meshes_copied": False},
        "before_fk": before_fk, "after_fk": after_fk, "unchanged_model_arrays": list(unchanged),
        "torque_contract": "29/29 nominal metadata motor limits matched; original joint force limits unchanged",
        "protected_inputs_rechecked": True, "original_xml_modified": False, "physics_stepped": False,
        "hardware_accessed": False, "physically_calibrated": False,
        "limitations": ["Independent simulation variant; not a real-motor or contact calibration.",
            "Reference FK agreement is not physical dynamics/contact equality or policy-task success.",
            "Original Bridge collisions, mass, friction and extra fixed bodies are retained."]}
    with (output / "receipt.json").open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-xml", type=Path, default=BRIDGE_XML)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--training-xml", type=Path, default=TRAINING_XML)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(prepare_asset(args.bridge_xml, args.metadata, args.training_xml, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
