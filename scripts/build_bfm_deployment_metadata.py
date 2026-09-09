#!/usr/bin/env python3
"""Build an evidence-bound NOMINAL static profile, never an Isaac runtime dump.

Only the audited local G1-BFM-Transformer-Tracking official M checkpoint is
supported. Configuration is inspected with restricted AST arithmetic/keywords,
not executed. No Isaac, TensorDict, simulation, SMPL-X or installation required.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import operator
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRACK = ROOT / "ScaleTrack/source/scaletrack/scaletrack"
CHECKPOINT_SHA = "88d5a79946c03ed25503f48b2af71d16290844ef066ca9b6c8fa8dc3837422e3"
EVIDENCE = ROOT / "logs/behavior_learning/retarget_sole_pilot_20260908a/eval_baseline_0.json"
EVIDENCE_SHA = "7ccb964e7fb089fa552a7cd0491c1e68a6503b0eb387d34a0ad68f416981ec68"
NAMES_SOURCE = ROOT / "scripts/evaluate_umr_pairs.py"
NAMES_SOURCE_SHA = "48d04439d4c8a4a7ccec6ce75a7b43b007fc22cdc49c23561b81cd3e8036f0dc"
SOURCES = {
    "robot": TRACK / "robots/g1_29dof.py",
    "agent": TRACK / "tasks/tracking/config/g1_29dof/agents/rsl_rl_ppo_cfg.py",
    "flat": TRACK / "tasks/tracking/config/g1_29dof/flat_env_cfg.py",
    "environment": TRACK / "tasks/tracking/tracking_env_cfg.py",
    "commands": TRACK / "tasks/tracking/mdp/commands.py",
    "network": ROOT / "ScaleTrack/source/my_rsl_rl/my_rsl_rl/networks/humanoid_transformer.py",
    "official_export": ROOT / "ScaleTrack/scripts/pretrain/rsl_rl/play_export_check_humanoid_transformer.py",
}
PROFILE = "g1_bfm_official_m_audited_nominal_dense6_20260909_v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_file(path, frozen, expected=None):
    path = Path(path).resolve(strict=True)
    digest = sha256(path)
    if expected is not None and digest != expected:
        raise ValueError(f"Static profile evidence changed; review/version it before export: {path}")
    if str(path) in frozen and frozen[str(path)] != digest:
        raise ValueError(f"Evidence changed during metadata construction: {path}")
    frozen[str(path)] = digest
    return path


def constant(node, values=None):
    """No eval, calls, attributes, subscripts, imports or comprehensions."""
    values = {} if values is None else values
    if isinstance(node, ast.Constant) and type(node.value) in (int, float, str, bool, type(None)):
        return node.value
    if isinstance(node, ast.Name) and node.id in values:
        return values[node.id]
    if isinstance(node, (ast.List, ast.Tuple)):
        result = [constant(item, values) for item in node.elts]
        return tuple(result) if isinstance(node, ast.Tuple) else result
    if isinstance(node, ast.Dict) and all(key is not None for key in node.keys):
        keys = [constant(key, values) for key in node.keys]
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate literal dictionary key")
        return dict(zip(keys, [constant(value, values) for value in node.values]))
    operations = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                  ast.Div: operator.truediv, ast.Pow: operator.pow}
    if isinstance(node, ast.BinOp) and type(node.op) in operations:
        left, right = constant(node.left, values), constant(node.right, values)
        if type(left) not in (int, float) or type(right) not in (int, float):
            raise ValueError("Only numeric AST arithmetic is supported")
        if isinstance(node.op, ast.Pow) and abs(right) > 8:
            raise ValueError("Unbounded AST exponent")
        result = operations[type(node.op)](left, right)
        if not math.isfinite(result):
            raise ValueError("Nonfinite AST arithmetic")
        return result
    if isinstance(node, ast.UnaryOp) and type(node.op) in (ast.USub, ast.UAdd):
        value = constant(node.operand, values)
        if type(value) not in (int, float):
            raise ValueError("Only numeric unary arithmetic is supported")
        return -value if isinstance(node.op, ast.USub) else value
    raise ValueError(f"Unsupported static expression: {type(node).__name__}")


def dotted(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = dotted(node.value)
        return prefix + "." + node.attr if prefix else ""
    return ""


def assignments(scope):
    found = {}
    for node in scope.body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        for target in targets:
            key = dotted(target)
            if not key:
                continue
            if key in found:
                raise ValueError(f"Ambiguous repeated assignment: {key}")
            found[key] = node.value
    return found


def member(scope, name):
    found = [node for node in scope.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name]
    if len(found) != 1:
        raise ValueError(f"Expected unique static class/function: {name}")
    return found[0]


def keyword(call, name):
    if not isinstance(call, ast.Call) or any(item.arg is None for item in call.keywords):
        raise ValueError("Expected explicit configuration keyword call")
    found = [item.value for item in call.keywords if item.arg == name]
    if len(found) != 1:
        raise ValueError(f"Expected exactly one configuration keyword: {name}")
    return found[0]


def dictionary_entry(node, name):
    if not isinstance(node, ast.Dict):
        raise ValueError("Expected explicit configuration dictionary")
    found = [value for key, value in zip(node.keys, node.values) if constant(key) == name]
    if len(found) != 1:
        raise ValueError(f"Expected exactly one configuration dictionary entry: {name}")
    return found[0]


def match_value(specification, joint, *, default=None):
    matches = [value for pattern, value in specification.items() if re.fullmatch(pattern, joint)]
    if not matches and default is not None:
        return float(default)
    if len(matches) != 1:
        raise ValueError(f"Expected one unambiguous parameter for joint {joint}")
    value = float(matches[0])
    if not math.isfinite(value):
        raise ValueError("Nonfinite joint parameter")
    return value


def robot_parameters(tree, names):
    root = assignments(tree)
    values = {}
    for name, value in root.items():
        if isinstance(value, (ast.Constant, ast.BinOp, ast.UnaryOp)):
            values[name] = constant(value, values)
    config = root["G1_29DOF_CYLINDER_CFG"]
    initial = constant(keyword(keyword(config, "init_state"), "joint_pos"), values)
    actuator_nodes = keyword(config, "actuators")
    if not isinstance(actuator_nodes, ast.Dict):
        raise ValueError("Expected explicit actuator groups")
    groups = []
    for node in actuator_nodes.values:
        patterns = constant(keyword(node, "joint_names_expr"), values)
        group = {"patterns": patterns}
        for field in ("stiffness", "damping", "effort_limit_sim", "velocity_limit_sim"):
            value = constant(keyword(node, field), values)
            group[field] = value if isinstance(value, dict) else {pattern: value for pattern in patterns}
        groups.append(group)
    output = {name: [] for name in ("default_dof_pos", "action_scale", "stiffness", "damping", "torque_limit", "velocity_limit")}
    for joint in names:
        matched = [group for group in groups if any(re.fullmatch(pattern, joint) for pattern in group["patterns"])]
        if len(matched) != 1:
            raise ValueError(f"Joint must belong to exactly one actuator group: {joint}")
        group = matched[0]
        stiffness = match_value(group["stiffness"], joint)
        torque = match_value(group["effort_limit_sim"], joint)
        if stiffness <= 0 or torque <= 0:
            raise ValueError("Positive nominal stiffness/torque are required")
        output["default_dof_pos"].append(match_value(initial, joint, default=0.))
        output["stiffness"].append(stiffness)
        output["damping"].append(match_value(group["damping"], joint))
        output["torque_limit"].append(torque)
        output["velocity_limit"].append(match_value(group["velocity_limit_sim"], joint))
        # This exceptional waist rule is part of the hash-pinned robot source.
        output["action_scale"].append(.25 if joint in ("waist_roll_joint", "waist_pitch_joint", "waist_yaw_joint")
                                      else .25 * torque / stiffness)
    return output


def load_static_profile(frozen):
    evidence_path = checked_file(EVIDENCE, frozen, EVIDENCE_SHA)
    evidence = json.loads(evidence_path.read_text())
    if (evidence["checkpoint_sha256"] != CHECKPOINT_SHA or evidence["task"] != "G1-BFM-Transformer-Tracking"
            or evidence["input_manifest_verified_unchanged"] is not True or evidence["step_dt"] != .02):
        raise ValueError("Evaluation does not establish the intended official local profile")
    saved = {item["path"]: item["sha256"] for item in evidence["input_manifest"]["python_sources"]["files"].values()}
    trees = {}
    for label, path in SOURCES.items():
        if str(path) not in saved:
            raise ValueError(f"Missing frozen runtime source evidence: {label}")
        trees[label] = ast.parse(checked_file(path, frozen, saved[str(path)]).read_text())
    names_tree = ast.parse(checked_file(NAMES_SOURCE, frozen, NAMES_SOURCE_SHA).read_text())
    names = assignments(names_tree)
    return trees, constant(names["ARTICULATION_JOINT_NAMES"]), constant(names["BODY_NAMES"])


def static_configuration(trees, joint_names, body_names):
    flat = assignments(member(member(trees["flat"], "G1BFMTrackingEnvCfg"), "__post_init__"))
    selected = constant(flat["self.commands.motion.body_names"])
    modes = constant(flat["self.commands.motion.mode_candidates"])
    if len(selected) != 14 or len(set(selected)) != 14 or not set(selected).issubset(body_names) or len(modes) != 8:
        raise ValueError("Expected audited fourteen-link/eight-mode control")
    mode_table = []
    for links in modes.values():
        if len(set(links)) != len(links) or not set(links).issubset(selected):
            raise ValueError("Duplicate or unknown mode target link")
        mode_table.append([float(name in links) for name in selected])
    environment = trees["environment"]
    context = constant(assignments(environment)["BFM_CONTEXT_SIZE"])
    observation = member(environment, "BFMMaskObservationCfg")
    task = assignments(member(observation, "PolicyTaskCfg"))
    futures = [constant(keyword(call, "params"))["future_idx"] for call in task.values()]
    if not futures or any(value != futures[0] for value in futures):
        raise ValueError("Policy future indices disagree")
    export_main = member(trees["official_export"], "main")
    deployment_future_nodes = [node.value for node in export_main.body if isinstance(node, ast.Assign)
                               and any(dotted(target) == "future_idx" for target in node.targets)]
    if len(deployment_future_nodes) != 1:
        raise ValueError("Missing unique explicit official-export future index schedule")
    deployment_future = constant(deployment_future_nodes[0])
    if deployment_future != [0, 1, 2, 3, 4, 5]:
        raise ValueError("Dense-six deployment profile requires official fixed nonnegative future offsets")
    mapping = constant(keyword(assignments(member(observation, "ModeMappingCfg"))["mode_mapping"], "params"))
    policy_terms = assignments(member(observation, "PolicyCfg"))
    joint_velocity_scale = constant(keyword(policy_terms["joint_vel"], "scale"))
    for call in policy_terms.values():
        if constant(keyword(call, "history_length"), {"BFM_CONTEXT_SIZE": context}) != context:
            raise ValueError("Policy history lengths disagree")
    action = assignments(member(environment, "ActionsCfg"))["joint_pos"]
    if constant(keyword(action, "joint_names")) != [".*"] or constant(keyword(action, "use_default_offset")) is not True:
        raise ValueError("Cannot establish full articulation action order/default offset")
    timing = assignments(member(member(environment, "BFMTrackingEnvCfg"), "__post_init__"))
    step_dt = constant(timing["self.sim.dt"]) * constant(timing["self.decimation"])
    config = member(trees["agent"], "CompleteRslRlPpoActorCriticTransformerCfg")
    architecture = {key: constant(assignments(config)[key]) for key in
                    ("embedding_dim", "num_heads", "ff_dim", "num_layers", "task_embedder_hidden_dims")}
    runner = assignments(member(trees["agent"], "G1BFMTransformerPPORunnerCfg"))
    if constant(runner["empirical_normalization"]) is not False:
        raise ValueError("Empirical normalization requires a separate deployment contract")
    for item in runner["policy"].keywords:
        if item.arg in architecture:
            architecture[item.arg] = constant(item.value)
    if constant(keyword(runner["policy"], "state_dependent_std")) is not False:
        raise ValueError("Static profile requires deterministic single actor output head")
    feature_dims, with_time = mapping["feature_dims_per_link"], mapping["with_time"]
    architecture.update(prop_obs_dim=3 + 3 + 2 * len(joint_names),
                        task_obs_dim=len(selected) * sum(feature_dims) + int(with_time) + len(selected),
                        action_dim=len(joint_names), output_dim=len(joint_names), reduced_task_dim=None)
    random_future = constant(assignments(member(trees["commands"], "MotionCommandCfg"))["rand_timestep_range"])
    default_event = assignments(member(environment, "EventCfg"))["add_joint_default_pos"]
    default_parameters = keyword(default_event, "params")
    if constant(dictionary_entry(default_parameters, "operation")) != "add":
        raise ValueError("Unexpected nominal-default randomization operation")
    default_randomization = constant(dictionary_entry(default_parameters, "pos_distribution_params"))
    return {"policy_architecture": architecture, "joint_names": list(joint_names), "action_names": list(joint_names),
        "body_names": list(body_names), "selected_body_names": selected, "history_buffer_size": context,
        "future_idx": deployment_future, "future_length": len(deployment_future), "training_future_idx": futures[0],
        "deployment_future_semantics": "official export fixed dense six: current plus next five 20 ms frames",
        "mode_names": list(modes), "mode_table": mode_table,
        "mode_feature_dims": feature_dims, "mode_mapping_with_time": with_time,
        "step_dt": step_dt, "joint_velocity_observation_scale": joint_velocity_scale,
        "training_future_negative_index_semantics": "training -1 is a dynamic positive future offset, not a past frame",
        "training_random_future_range_frames": list(random_future), "training_random_future_high_exclusive": True,
        "training_default_joint_randomization_rad": list(default_randomization),
        **robot_parameters(trees["robot"], joint_names)}


def validate_checkpoint_shapes(checkpoint, architecture):
    import torch
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload["model_state_dict"]
    embed = architecture["embedding_dim"]
    expected = {"actor.prop_projection.weight": (embed, architecture["prop_obs_dim"]),
        "actor.action_projection.weight": (embed, architecture["action_dim"]),
        "actor.projection_head.weight": (architecture["output_dim"], embed),
        "actor_task_embedder.task_projection.weight": (embed, architecture["task_obs_dim"])}
    for key, shape in expected.items():
        if key not in state or tuple(state[key].shape) != shape:
            raise ValueError(f"Static profile/checkpoint shape mismatch: {key}")
    layers = {int(key.split(".")[2]) for key in state if key.startswith("actor.transformer_blocks.")}
    if layers != set(range(architecture["num_layers"])):
        raise ValueError("Actor layer count differs from explicit source configuration")
    for layer in layers:
        if tuple(state[f"actor.transformer_blocks.{layer}.feed_forward.w.weight"].shape) != (architecture["ff_dim"], embed):
            raise ValueError("Actor feed-forward dimension differs from explicit source configuration")
    return {"weights_only": True, "map_location": "cpu", "iter": payload.get("iter"),
            "num_heads_source": "explicit hash-bound agent configuration, NOT inferred from state_dict"}


def build_metadata(archive, checkpoint):
    """Pure artifact reader returning a dictionary; the caller owns publication."""
    frozen = {}
    checked_file(Path(__file__), frozen)
    checkpoint = checked_file(checkpoint, frozen, CHECKPOINT_SHA)
    archive = checked_file(archive, frozen)
    trees, expected_joints, expected_bodies = load_static_profile(frozen)
    with np.load(archive, allow_pickle=False) as motion:
        for field, expected in (("format_version", 3), ("fps", 50), ("quaternion_order", "wxyz")):
            if motion[field].shape != () or motion[field].item() != expected:
                raise ValueError(f"Expected named canonical packed archive: {field}")
        names = {}
        for field, expected in (("joint_names", expected_joints), ("body_names", expected_bodies)):
            array = motion[field]
            if array.ndim != 1 or array.dtype.kind not in "US":
                raise ValueError(f"Archive requires explicit string {field}")
            names[field] = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in array]
            if names[field] != list(expected):
                raise ValueError(f"Archive {field} differs from audited runtime order")
        native_sha = str(motion["source_sha256"].item())
        if not re.fullmatch(r"[0-9a-f]{64}", native_sha):
            raise ValueError("Named archive lacks valid native source provenance")
        if motion["joint_pos"].ndim != 2 or motion["joint_pos"].shape[1] != 29 or motion["body_pos_w"].shape[1:] != (30, 3):
            raise ValueError("Archive arrays disagree with explicit names")
    configuration = static_configuration(trees, names["joint_names"], names["body_names"])
    checkpoint_evidence = validate_checkpoint_shapes(checkpoint, configuration["policy_architecture"])
    for path, expected in tuple(frozen.items()):
        checked_file(path, frozen, expected)
    return {"schema_version": 2, "deployment_profile": PROFILE, "runtime_backend": "torchscript",
        "metadata_kind": "hash_bound_nominal_static_profile_not_runtime_dump", "task": "G1-BFM-Transformer-Tracking",
        "checkpoint": str(checkpoint), "checkpoint_sha256": CHECKPOINT_SHA,
        "source_hashes": frozen, "archive": str(archive), "archive_source_pkl_sha256": native_sha,
        "source_profile_verified_unchanged": True, "checkpoint_evidence": checkpoint_evidence,
        "default_dof_pos_semantics": "nominal zero baseline plus configured joint regex; excludes startup randomization",
        "root_quaternion_order": "wxyz", "nominal_unmatched_joint_default_rad": 0.,
        "action_output_semantics": "raw deterministic policy action; joint_position_target = action * scale + nominal_default",
        "pd_parameter_semantics": "nominal simulator settings, not physical motor/contact calibration",
        "limitations": ["Original checkpoint contains no training-config dump; profile binds audited local evaluation defaults.",
                        "No runtime actuator/default-position dump, simulator replay or hardware acceptance is implied.",
                        "Sparse modes mask 14 target links; they are not an eight-way one-hot feature.",
                        "Future -1 denotes an explicitly supplied future target timestamp, never negative physical time."],
        **configuration}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("archive", "checkpoint", "output"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        parser.error("Refusing existing metadata output")
    metadata = build_metadata(args.archive, args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "profile": PROFILE, "metadata_kind": metadata["metadata_kind"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
