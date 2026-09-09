#!/usr/bin/env python3
"""Bounded, headless native ScaleBridge rollout with real MuJoCo state metrics.

Only simulation is accessible. No reference teleportation after initialization,
no hardware, viewer, sockets, PPO, model promotion or implicit motion looping.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ScaleBridge"))
from scripts.export_bfm_torchscript import bounded_unit, sha256

ORIGINS = (
    "KIT/3/squat01_stageii",
    "KIT/359/walking_run09_stageii",
    "BMLrub/rub114/0013_knocking1_stageii",
)


def tracking_distances(actual, reference, actual_root, reference_forcing):
    import numpy as np
    actual, reference, actual_root = map(np.asarray, (actual, reference, actual_root))
    if (actual.shape != (14, 3) or reference.shape != actual.shape or actual_root.shape != (3,)
            or not all(np.isfinite(value).all() for value in (actual, reference, actual_root))):
        raise ValueError("Expected finite fourteen-link actual/reference and actual root XYZ")
    control_reference = reference - reference[0] + actual_root if reference_forcing else reference
    return np.linalg.norm(actual-control_reference, axis=-1), np.linalg.norm(actual-reference, axis=-1)


def fk_contract(source_xml, simulator_xml, metadata):
    """Same named kinematic chain across simulators, not equal contact physics."""
    import mujoco
    import numpy as np
    models = [mujoco.MjModel.from_xml_path(str(path)) for path in (source_xml, simulator_xml)]
    if any((model.nq, model.nv, model.njnt) != (36, 35, 30) for model in models):
        raise ValueError("Sim2Sim requires a named, free-root G1 29 DoF model")
    extra_bodies = []
    for model in models:
        extras = [model.body(i).name for i in range(1, model.nbody) if model.body(i).name not in metadata["body_names"]]
        if not set(extras) <= {"pelvis_contour_link", "imu_in_torso"} or any(model.body(name).jntnum[0] != 0 for name in extras):
            raise ValueError("Unexpected extra robot bodies/joints")
        extra_bodies.append(extras)
    datas = [mujoco.MjData(model) for model in models]
    generator = np.random.default_rng(42)
    max_position, max_angle = 0., 0.
    for sample in range(8):
        joints = np.array(metadata["default_dof_pos"]) + generator.uniform(-.2, .2, 29)
        poses, quats = [], []
        for model, data in zip(models, datas):
            data.qpos[:7] = [0, 0, 0, 1, 0, 0, 0]
            for name, angle in zip(metadata["joint_names"], joints):
                data.qpos[model.joint(name).qposadr[0]] = angle
            mujoco.mj_kinematics(model, data)
            ids = [model.body(name).id for name in metadata["body_names"]]
            poses.append(data.xpos[ids].copy())
            quats.append(data.xquat[ids].copy())
        second = np.where((quats[0]*quats[1]).sum(-1, keepdims=True) < 0, -quats[1], quats[1])
        max_position = max(max_position, float(np.linalg.norm(poses[0]-poses[1], axis=-1).max()))
        angle = 4*np.arctan2(np.linalg.norm(quats[0]-second, axis=-1), np.linalg.norm(quats[0]+second, axis=-1))
        max_angle = max(max_angle, float(angle.max()))
    if max_position > 1e-5 or max_angle > 1e-5:
        raise ValueError(f"Export/simulator FK differs: {max_position} m / {max_angle} rad")
    return {"passed": True, "samples": 8, "links": 30, "max_position_m": max_position, "max_rotation_rad": max_angle,
            "source_xml_sha256": sha256(source_xml), "simulator_xml_sha256": sha256(simulator_xml),
            "extra_fixed_bodies_source_simulator": extra_bodies,
            "contact_geometry_equivalence_claimed": False}


def simulator_config(xml, strict_torque_limits):
    return {"_target_": "scalebridge.simulator.mujoco_simulator.MujocoSimulator", "_recursive_": False,
            "config": {"asset": {"xml_path": str(xml)}, "low_dt": .005, "decimation": 4,
                       "headless": True, "record_video": False, "marker": False, "joystick": False,
                       "camera_follow": False, "strict_torque_limits": strict_torque_limits}}


def environment_config(motion, metadata, xml, reference_forcing, strict_torque_limits):
    from omegaconf import OmegaConf
    terms = ("root_quat_buffer", "base_ang_vel_buffer", "dof_pos_buffer", "dof_vel_buffer",
             "actions_buffer", "body_pos_w_future", "body_quat_w_future",
             "target_body_pos_future_to_robot_base", "target_body_rot_future_to_robot_base", "future_time_offsets")
    return OmegaConf.create({"motion_path": str(motion), "future_idx": metadata["future_idx"],
        "rsi": True, "reference_forcing": reference_forcing,
        "simulator": simulator_config(xml, strict_torque_limits),
        "observation": {name: {"func": {"_target_": f"scalebridge.observation.components.{name}",
                                         "_partial_": True}} for name in terms}})


def one_rollout(policy_path, motion, xml, mode, forcing, max_steps, device, strict_torque_limits):
    import mujoco
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from scalebridge.agent.bfm_agent import BFMAgent
    from scalebridge.env.motion_tracking import MotionTrackingEnv

    agent = env = None
    try:
        agent = BFMAgent(OmegaConf.create({"checkpoint": str(policy_path), "control_mode": mode}), device)
        metadata = copy.deepcopy(agent.get_meta_data())
        cfg = environment_config(motion, metadata, xml, forcing, strict_torque_limits)
        env = MotionTrackingEnv(cfg, metadata, device)
        obs = env.reset()
        sim = env.simulator
        ids = np.array([sim.mujoco_model.body(name).id for name in metadata["selected_body_names"]])
        active = np.array(metadata["mode_table"][mode], dtype=bool)
        nsteps = min(max_steps, env.motion_len - 1)
        if nsteps < 1:
            raise ValueError("Reference requires at least two frames")
        for _ in range(3):
            agent.get_action(obs)
        initial_pos = sim.mujoco_data.qpos[:3].copy()
        initial_time = sim.mujoco_data.time
        positions, world_positions, angles, root_heights, root_up, latency = [], [], [], [], [], []
        warnings_before = sim.mujoco_data.warning.number.copy()
        for step in range(nsteps):
            # Compare state AFTER one dt against reference frame step+1. Future
            # clamping near clip end is allowed; no repeated extra tail rollout.
            start = time.perf_counter()
            target, action = agent.get_action(obs)
            if torch.device(device).type == "cuda":
                torch.cuda.synchronize()
            latency.append(time.perf_counter() - start)
            obs = env.step(target, action)
            data = sim.mujoco_data
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise ValueError(f"Nonfinite physical state at step {step+1}")
            if abs((data.time-initial_time) - (step+1)*env.dt) > 1e-8:
                raise ValueError("MuJoCo clock diverged from the 50 Hz reference")
            # mj_step integrates qpos last; derived link poses may still be
            # from the preceding low-level substep. Refresh FK without stepping.
            mujoco.mj_kinematics(sim.mujoco_model, data)
            reference = env.body_pos_w[step+1].cpu().numpy()
            reference_quat = env.body_quat_w[step+1].cpu().numpy().astype(np.float64)
            actual_quat = data.xquat[ids].copy()
            if (not all(np.isfinite(value).all() for value in (reference, reference_quat, actual_quat, data.xpos[ids]))
                    or np.any(np.linalg.norm(reference_quat, axis=-1) < 1e-8)
                    or np.any(np.linalg.norm(actual_quat, axis=-1) < 1e-8)):
                raise ValueError("Nonfinite or invalid reference/derived physical pose")
            reference_quat /= np.linalg.norm(reference_quat, axis=-1, keepdims=True)
            actual_quat /= np.linalg.norm(actual_quat, axis=-1, keepdims=True)
            actual_quat *= np.where((actual_quat * reference_quat).sum(-1, keepdims=True) < 0, -1., 1.)
            angles.append(4 * np.arctan2(np.linalg.norm(reference_quat-actual_quat, axis=-1),
                                         np.linalg.norm(reference_quat+actual_quat, axis=-1)))
            control_error, world_error = tracking_distances(data.xpos[ids], reference, data.qpos[:3], forcing)
            world_positions.append(world_error)
            positions.append(control_error)
            root_heights.append(float(data.qpos[2]))
            root_up.append(float(data.xmat[ids[0]].reshape(3, 3)[2, 2]))
        positions, world_positions, angles = np.asarray(positions), np.asarray(world_positions), np.asarray(angles)
        fell = np.any(np.asarray(root_heights) < .3) or np.any(np.asarray(root_up) < .2588190451)
        warnings = sim.mujoco_data.warning.number - warnings_before
        return {"mode_index": mode, "reference_forcing": forcing,
            "control_translation": "reference-root-translation" if forcing else "actual-world-XYZ",
            "position_metric_frame": "translation-aligned local; world drift reported separately" if forcing else "world",
            "frames": env.motion_len, "steps": nsteps, "simulated_seconds": nsteps * env.dt,
            "truncated": nsteps < env.motion_len-1, "finite_physics": True,
            "initial_root_xyz": initial_pos.tolist(), "final_root_xyz": sim.mujoco_data.qpos[:3].tolist(),
            "root_displacement_m": float(np.linalg.norm(sim.mujoco_data.qpos[:3]-initial_pos)),
            "active_position_mean_m": float(positions[:, active].mean()),
            "all14_position_mean_m": float(positions.mean()),
            "world_all14_position_mean_m": float(world_positions.mean()),
            "world_all14_position_max_m": float(world_positions.max()),
            "active_rotation_mean_rad": float(angles[:, active].mean()),
            "active_position_max_m": float(positions[:, active].max()),
            "all14_position_max_m": float(positions.max()),
            "root_height_min_m": min(root_heights), "root_up_dot_min": min(root_up),
            "fall_guard_triggered": bool(fell),
            "tracking_guard_pass": bool(not fell and positions.max() <= .5 and not np.any(warnings)),
            "mujoco_warnings": warnings.tolist(),
            "torque_limit_mismatches": sim.torque_limit_mismatches,
            "policy_latency_ms": {"median": float(np.median(latency)*1000), "p95": float(np.percentile(latency,95)*1000)}}
    finally:
        if env is not None:
            env.close()
        if agent is not None:
            agent.close()


def evaluate(policy, packed_root, xml, output, max_steps, modes, forcings, device, strict_torque_limits):
    import mujoco
    import torch
    from loguru import logger
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    torch.set_num_threads(2)
    output = Path(output).absolute()
    if any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError("Output cannot redirect through symlinks")
    output.mkdir(parents=True, exist_ok=False)
    previous = signal.getsignal(signal.SIGTERM)
    def terminate(signum, frame):
        raise KeyboardInterrupt("Owned Sim2Sim service stopped")
    signal.signal(signal.SIGTERM, terminate)
    started = time.monotonic()
    report = {"schema": "bfm.sim2sim_tracking/1", "result": "ERROR", "training_updates": 0,
              "automatic_promotion": False, "hardware_accessed": False, "headless": True,
              "reference_state_initialization": True, "reset_between_jobs": True,
              "object_task_tested": False, "seed": 42, "device": device,
              "torch": torch.__version__, "mujoco": mujoco.__version__, "results": [],
              "guard_thresholds": {"all14_position_max_m": .5, "root_height_min_m": .3,
                                   "root_up_dot_min": .2588190451},
              "interpretation": "Finite-window reference tracking only, not arbitrary online targets or manipulation."}
    try:
        policy, xml, packed_root = [Path(path).resolve(strict=True) for path in (policy, xml, packed_root)]
        metadata_path = policy.with_name(policy.stem + "_metadata.json")
        export_path = policy.parent / "export_report.json"
        exported = json.loads(export_path.read_text())
        if (exported.get("result") != "PASS" or exported.get("artifact_sha256") != sha256(policy)
                or exported.get("metadata_sha256") != sha256(metadata_path)):
            raise ValueError("Require a verified, unchanged exported policy and metadata")
        metadata = json.loads(metadata_path.read_text())
        source_xmls = [Path(path) for path, digest in exported["input_sha256"].items()
                       if Path(path).suffix.lower() == ".xml" and digest == metadata["fk_xml_sha256"]]
        if len(source_xmls) != 1 or sha256(source_xmls[0]) != metadata["fk_xml_sha256"]:
            raise ValueError("Cannot establish unchanged exported FK source")
        report["kinematic_contract"] = fk_contract(source_xmls[0], xml, metadata)
        files = [policy, metadata_path, export_path, xml, Path(__file__).resolve()]
        files += source_xmls
        files += list((ROOT / "ScaleBridge/scalebridge").rglob("*.py"))
        # Generated variants may refer to an absolute mesh directory. Bind
        # compiled mesh inputs as well as the XML, without copying assets.
        import xml.etree.ElementTree as ET
        doc = ET.parse(xml).getroot()
        compiler = doc.find("compiler")
        mesh_root = xml.parent / (compiler.get("meshdir", "") if compiler is not None else "")
        files += [path for path in mesh_root.rglob("*") if path.is_file() and path.suffix.lower() in (".xml", ".stl", ".obj")]
        files += [packed_root / f"{name}.npz" for name in ORIGINS]
        frozen = {str(path): sha256(path) for path in files}
        report.update(input_sha256=frozen, source_checkpoint_sha256=metadata["checkpoint_sha256"],
                      export_report=str(export_path), planned_jobs=len(ORIGINS)*len(modes)*len(forcings),
                      strict_torque_limits=strict_torque_limits)
        torch.manual_seed(42)
        for origin in ORIGINS:
            motion = packed_root / f"{origin}.npz"
            for forcing in forcings:
                for mode in modes:
                    row = one_rollout(policy, motion, xml, mode, forcing, max_steps, device, strict_torque_limits)
                    report["results"].append({"origin_id": origin, "motion_sha256": frozen[str(motion)], **row})
                    print(json.dumps({"job": len(report["results"]), "origin_id": origin,
                                      "mode": mode, "reference_forcing": forcing,
                                      "steps": row["steps"], "tracking_guard_pass": row["tracking_guard_pass"]}), flush=True)
                    (output / "progress.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        if any(sha256(path) != digest for path, digest in frozen.items()):
            raise ValueError("Input/runtime files changed during Sim2Sim")
        rows = report["results"]
        report.update(result="COMPLETE", inputs_verified_unchanged=True, completed_jobs=len(rows),
            tracking_guard_pass_jobs=sum(row["tracking_guard_pass"] for row in rows),
            torque_contract_matches=not any(row["torque_limit_mismatches"] for row in rows),
            total_steps=sum(row["steps"] for row in rows))
        report["quality_gate_pass"] = all(row["tracking_guard_pass"] for row in rows) and report["torque_contract_matches"]
    except (Exception, KeyboardInterrupt) as error:
        report.update(result="INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "ERROR",
                      error=f"{type(error).__name__}: {error}")
    finally:
        signal.signal(signal.SIGTERM, previous)
    report["elapsed_seconds"] = time.monotonic() - started
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("policy", "packed-root", "xml", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--modes", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--translations", choices=("global", "local", "both"), default="both")
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--strict-torque-limits", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.max_steps <= 1000 or len(set(args.modes)) != len(args.modes) or any(m not in range(8) for m in args.modes):
        parser.error("Use unique masks in 0..7 and 1..1000 bounded steps")
    bounded_unit("bfm-sim2sim-")
    forcings = {"global": [False], "local": [True], "both": [False, True]}[args.translations]
    report = evaluate(args.policy, args.packed_root, args.xml, args.output, args.max_steps,
                      args.modes, forcings, args.device, args.strict_torque_limits)
    print(json.dumps({key: report[key] for key in ("result", "error", "completed_jobs", "tracking_guard_pass_jobs",
                       "quality_gate_pass", "elapsed_seconds") if key in report}))
    return 0 if report["result"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
