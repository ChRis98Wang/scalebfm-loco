#!/usr/bin/env python3
"""Native contact-only lift with centred sparse targets and full failure video."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PLAYER = ROOT / "ScaleTrack/scripts/pretrain/rsl_rl"
sys.path[:0] = [str(ROOT), str(PLAYER)]
from scripts.export_bfm_torchscript import bounded_unit, sha256

CHECKPOINT = ROOT / "ScaleTrack/logs/rsl_rl/g1_bfm_tracking_exp/humanoid_transformer_m/model_22200.pt"
CHECKPOINT_SHA = "88d5a79946c03ed25503f48b2af71d16290844ef066ca9b6c8fa8dc3837422e3"
REFERENCE = ROOT / "local/synthetic_reference_20260909b/synthetic_reference.npz"
REFERENCE_SHA = "8342a0f927013a2b7d1b4cc79c2dff5b7551e2fb72008c5eef305d0020cae04d"
METADATA = ROOT / "local/scalebfm_deployment_20260909a/export_v2/policy_metadata.json"
PROTOCOL = ROOT / "docs/ISAAC_LIFT_BALANCED_PROTOCOL_20260910.md"


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def run(args, app_factory):
    unit = bounded_unit("bfm-isaac-lift-")
    output = args.output.absolute()
    if (output.exists() or ".." in output.parts or not output.is_relative_to(ROOT / "local")
            or not output.parent.is_dir() or any(p.is_symlink() for p in (output, *output.parents))):
        raise ValueError("Use a new, non-symlink local/ output directory")
    if sha256(CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError("Only declared official checkpoint allowed in this baseline prototype")
    if sha256(REFERENCE) != REFERENCE_SHA:
        raise ValueError("Previously generated robot-only synthetic reference changed")
    busy = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                                   text=True, timeout=10).strip()
    if busy:
        raise ValueError(f"GPU compute task already exists; do not interrupt it: {busy}")
    output.mkdir()
    snapshot = output / "source_at_launch"
    snapshot.mkdir()
    for path in (Path(__file__), PROTOCOL, PLAYER / "lift_demo.py", PLAYER / "lift_demo_preload_v3.py",
                 PLAYER / "lift_demo_balanced.py", *([args.balance_config] if args.balance_config else [])):
        with (snapshot / path.name).open("xb") as stream:
            stream.write(path.read_bytes())
    files = [Path(__file__), PROTOCOL, METADATA, REFERENCE, CHECKPOINT,
             PLAYER / "lift_demo.py", PLAYER / "lift_demo_preload_v3.py", PLAYER / "lift_demo_balanced.py",
             PLAYER / "target_object.py", PLAYER / "online_control.py"]
    if args.balance_config:
        files.append(args.balance_config)
    files += list((ROOT / "ScaleTrack/source/scaletrack/scaletrack").rglob("*.py"))
    files += list((ROOT / "ScaleTrack/source/my_rsl_rl/my_rsl_rl").rglob("*.py"))
    files += [p for p in (ROOT / "ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof").rglob("*")
              if p.is_file()]
    frozen = {str(p): sha256(p) for p in files}
    save(output / "inputs.json", dict(input_sha256=frozen, unit=unit, args=vars(args) | {"output": str(output),
            "balance_config": str(args.balance_config) if args.balance_config else None},
        policy_source="official native checkpoint, not local PPO candidate", training_updates=0,
        planned_engine="IsaacLab / PhysX", reference_motion_source="robot-only synthetic FK initialization"))
    result = dict(schema="bfm.isaac_lift_demo/4", result="RUNNING", execution_complete=False,
                  task_success=False, formal_d1_d2_accepted=False, full_bfm_completed=False,
                  source_policy="official model_22200.pt", checkpoint_sha256=CHECKPOINT_SHA,
                  source_motion_data_used=False, training_updates=0, publication_performed=False,
                  planner_uses_object_feedback=True, policy_observes_object_directly=False,
                  material_calibrated=False, recording=args.record)
    app = env = writer = provider = None
    restorers, rows, image_hashes = [], [], []
    started = time.monotonic()
    previous = signal.getsignal(signal.SIGTERM)
    def stop(signum, frame):
        raise KeyboardInterrupt("Owned Isaac lift service stopped")
    signal.signal(signal.SIGTERM, stop)
    try:
        app = app_factory(args).app
        import gymnasium as gym
        import numpy as np
        import torch
        import isaaclab
        import isaaclab.sim as sim_utils
        from pxr import UsdPhysics
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab_tasks.utils import load_cfg_from_registry
        from my_rsl_rl.runners.on_policy_runner import OnPolicyRunner
        import scaletrack.tasks  # registers the existing native training task
        from scaletrack.utils.live_reference import LiveReferenceProvider
        from scaletrack.utils.quaternion_compat import runtime_to_packed_wxyz
        from online_control import configure_online_env, assert_online_env
        from lift_demo import LiftSpec, attach_lift_scene, box_geometry, rotation_wxyz
        from lift_demo_balanced import BalancedLiftPlanner as LiftPlanner, BalanceConfig
        torch.set_num_threads(1)
        np.random.seed(42)
        torch.manual_seed(42)
        spec = LiftSpec()
        balance = BalanceConfig(**json.loads(args.balance_config.read_text())) if args.balance_config else BalanceConfig()
        balance.validate()
        metadata = json.loads(METADATA.read_text())
        # The index is a generated input, not a new training dataset.
        index = output / "seed_motion.yaml"
        with index.open("x") as stream:
            stream.write("synthetic_standing_seed: " + str(REFERENCE) + "\n")
        task = "G1-BFM-Transformer-Tracking"
        cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
        agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
        cfg.scene.num_envs, cfg.scene.env_spacing, cfg.seed = 1, 4., 42
        cfg.sim.device = args.device
        agent_cfg.device = args.device
        agent_cfg.num_steps_per_env = 4
        if "Physx" not in type(cfg.sim.physics).__name__:
            raise ValueError("This protocol requires native PhysX, not another IsaacLab backend")
        configure_online_env(cfg)
        for name, term in vars(cfg.events).items():
            if getattr(term, "mode", None) == "startup":
                setattr(cfg.events, name, None)  # fixed scene, no random material/COM/hand-mass changes
        cfg.commands.motion.motion_file = str(index)
        cfg.commands.motion.enable_reset_disturbance = False
        cfg.commands.motion.debug_vis = False
        cfg.commands.motion.mode_candidates = {"VR-3": ["pelvis", "left_wrist_yaw_link", "right_wrist_yaw_link"]}
        paths = attach_lift_scene(cfg, metadata["body_names"], spec)
        cfg.scene.lift_contacts.track_contact_points = True
        cfg.viewer.eye, cfg.viewer.lookat = (2.6, 2.6, 1.7), (.25, 0., .65)
        cfg.viewer.origin_type = "world"
        cfg.viewer.resolution = (960, 720)
        cfg.video_recorder.window_width, cfg.video_recorder.window_height = 960, 720
        cfg.video_recorder.backend_source = "renderer"
        env = gym.make(task, cfg=cfg, render_mode="rgb_array" if args.record else None)
        raw = env.unwrapped
        assert_online_env(raw)
        env = RslRlVecEnvWrapper(env)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
        runner.load(str(CHECKPOINT), load_optimizer=False, map_location=args.device)
        runner._set_env_is_evaluating()
        actor = runner.alg.policy
        get_actor_obs = actor.get_actor_obs
        actor.get_actor_obs = lambda obs, inference=False: get_actor_obs(obs, inference=False)
        policy = runner.get_inference_policy(device=args.device)
        obs, _ = env.reset()  # final permitted initialization
        command = raw.command_manager.get_term("motion")
        robot = command.robot
        obj, support = raw.scene["target_object"], raw.scene["lift_support"]
        sensor = raw.scene["lift_contacts"]
        def tensor(value):
            return getattr(value, "torch", value)
        def array(value):
            return tensor(value).detach().cpu().numpy().copy()
        origin = array(raw.scene.env_origins)[0]
        body_indices = array(command.body_indexes).astype(int).tolist()
        if list(robot.body_names) != metadata["body_names"] or list(robot.joint_names) != metadata["joint_names"]:
            raise ValueError("Native robot body/joint ordering differs from bound reference metadata")
        runtime_order = command.runtime_quaternion_order
        def quat(value):
            return array(runtime_to_packed_wxyz(tensor(value), runtime_order))
        positions = array(robot.data.body_pos_w)[0, body_indices] - origin
        quaternions = quat(robot.data.body_quat_w)[0, body_indices]
        provider = LiveReferenceProvider(command.cfg.body_names, positions, quaternions,
                                        array(robot.data.joint_pos)[0], step_dt=raw.step_dt)
        command.attach_live_reference(provider, mode_name="VR-3")
        planner = LiftPlanner(positions[[0, 10, 13]], quaternions[[0, 10, 13]], spec, balance=balance)
        stage = sim_utils.get_current_stage()
        resolved_filters = [p.replace("{ENV_REGEX_NS}", "/World/envs/env_0") for p in paths]
        for path in resolved_filters:
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid() or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                raise ValueError(f"Contact filter must resolve exactly one real rigid body: {path}")
        box_prim = stage.GetPrimAtPath("/World/envs/env_0/TargetObject")
        if UsdPhysics.RigidBodyAPI(box_prim).GetKinematicEnabledAttr().Get():
            raise ValueError("Box may not be kinematic")
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.Joint):
                rels = (prim.GetRelationship("physics:body0"), prim.GetRelationship("physics:body1"))
                if any(str(target) == str(box_prim.GetPath()) for rel in rels for target in rel.GetTargets()):
                    raise ValueError("Box may not be attached to any joint")
        # Enforce no state teleportation/reset/external-force application after initialization.
        for asset in (robot, obj, support):
            for name in ("write_root_state_to_sim", "write_root_pose_to_sim", "write_root_velocity_to_sim",
                         "write_joint_state_to_sim", "set_external_force_and_torque"):
                if hasattr(asset, name):
                    original = getattr(asset, name)
                    def forbidden(*a, _name=name, **kw):
                        raise RuntimeError(f"Forbidden post-initialization state write: {_name}")
                    setattr(asset, name, forbidden)
                    restorers.append((asset, name, original))
        result.update(runtime=dict(isaaclab_path=str(Path(isaaclab.__file__).parent),
            physics_cfg=type(cfg.sim.physics).__module__ + "." + type(cfg.sim.physics).__name__,
            contact_sensor=type(sensor).__module__, torch=torch.__version__, quaternion_order=runtime_order),
            scene=asdict(spec), balance_config=asdict(balance), goal_box_center_xyz=list(spec.goal_xyz), filter_bodies=metadata["body_names"]+["LiftSupport"],
            filter_paths=resolved_filters, postinitialization_state_write_guard=True)
        save(output / "runtime.json", result)
        allowed = {side: [metadata["body_names"].index(f"{side}_{part}_link")
                         for part in ("elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]
                   for side in ("left", "right")}
        other = [i for i in range(30) if i not in allowed["left"] + allowed["right"]]
        def measure():
            xyz = array(obj.data.root_pos_w)[0] - origin
            quaternion = quat(obj.data.root_quat_w)[0]
            geom = box_geometry(xyz, quaternion, spec)
            history = array(sensor.data.force_matrix_w_history)
            if history.shape != (1, 4, 1, 31, 3) or not np.isfinite(history).all():
                raise ValueError(f"Expected one box x 31 declared partners x 4 substeps, got {history.shape}")
            forces = np.linalg.norm(history[0, :, 0], axis=-1)
            contact_points = array(sensor.data.contact_pos_w)
            tangents = array(sensor.data.friction_forces_w)
            if contact_points.shape != (1, 1, 31, 3) or tangents.shape != (1, 1, 31, 3):
                raise ValueError("Unexpected last-substep contact point/friction vector shape")
            if not np.isfinite(tangents).all():
                raise ValueError("Nonfinite measured friction force")
            contacts = [point.tolist() if np.isfinite(point).all() else None
                        for point in contact_points[0, 0]-origin]
            actual_pos = array(robot.data.body_pos_w)[0, body_indices] - origin
            actual_quats = quat(robot.data.body_quat_w)[0, body_indices]
            return dict(box_xyz=xyz.tolist(), box_quat_wxyz=quaternion.tolist(), box_bottom_z=geom["bottom_z"],
                box_tilt_rad=geom["tilt_rad"], inside_support_xy=geom["inside_support_xy"],
                box_speed=float(np.linalg.norm(array(obj.data.root_lin_vel_w)[0])),
                box_angular_speed=float(np.linalg.norm(array(obj.data.root_ang_vel_w)[0])),
                box_linear_velocity_w=array(obj.data.root_lin_vel_w)[0].tolist(),
                box_angular_velocity_w=array(obj.data.root_ang_vel_w)[0].tolist(),
                left_force_peak=float(forces[:, allowed["left"]].sum(-1).max()),
                right_force_peak=float(forces[:, allowed["right"]].sum(-1).max()),
                support_force_peak=float(forces[:, -1].max()),
                contact_points_last_w=contacts, contact_tangent_vectors_last_w=tangents[0, 0].tolist(),
                left_force=float(forces[:, allowed["left"]].sum(-1).min()),
                right_force=float(forces[:, allowed["right"]].sum(-1).min()),
                support_force=float(forces[:, -1].min()), other_robot_force=float(forces[:, other].sum(-1).max()),
                contact_normal_vectors_history_w=history[0, :, 0].tolist(),
                root_z=float(actual_pos[0, 2]), root_up=float(rotation_wxyz(actual_quats[0])[2, 2]),
                wrist_xyz=actual_pos[[10, 13]].tolist(), wrist_quat_wxyz=actual_quats[[10, 13]].tolist(),
                actual_body_xyz=actual_pos.tolist(), joint_pos=array(robot.data.joint_pos)[0].tolist(),
                joint_vel=array(robot.data.joint_vel)[0].tolist())
        measurement = measure()
        save(output / "initial_state.json", measurement)
        def render_frame(allow_unready=False):
            before = [array(asset.data.root_state_w) for asset in (robot, obj, support)]
            count = raw._sim_step_counter
            raw.sim.forward()
            play_setting = raw.sim.get_setting("/app/player/playSimulations")
            raw.sim.set_setting("/app/player/playSimulations", False)
            try:
                # With RTX enabled this installed env.render() only reads the
                # recorder; explicitly pump rendering without advancing PhysX.
                raw.sim.render()
                rgb = raw.render()
            finally:
                raw.sim.set_setting("/app/player/playSimulations", play_setting)
            if count != raw._sim_step_counter or any(
                    not np.array_equal(state, array(asset.data.root_state_w))
                    for state, asset in zip(before, (robot, obj, support))):
                raise ValueError("Recording changed physical state or clock")
            if not isinstance(rgb, np.ndarray) or rgb.shape != (720, 960, 3) or rgb.dtype != np.uint8:
                if allow_unready and (rgb is None or getattr(rgb, "size", 0) == 0):
                    return None
                raise ValueError(f"Expected native 960x720 RGB recording, got {getattr(rgb, 'shape', None)}")
            return rgb
        if args.record:
            import imageio.v2 as imageio
            writer = imageio.get_writer(str(output / "recording_raw.mp4"), fps=50, codec="libx264", quality=8,
                                       pixelformat="yuv420p", macro_block_size=16)
            result.update(initialization_frame_in_video=False, recording_starts_at_control_frame=1)
        actor_sha = hashlib.sha256(b"".join(p.detach().cpu().contiguous().numpy().tobytes() for p in actor.parameters())).hexdigest()
        initial_counter = raw._sim_step_counter
        for frame in range(1, 1001):
            goals, rotations = planner.targets(measurement)
            now = (frame-1) * raw.step_dt
            provider.submit(goals, rotations, sequence=frame, stamp=now, now=now)
            provider.commit_pending(now)
            for name in ("policy_task", "critic_task", "mode", "mode_mapping"):
                obs[name] = raw.observation_manager.compute_group(name, update_history=False)
            expected_mode = torch.zeros_like(obs["mode"])
            expected_mode[:, [0, 10, 13]] = 1.
            if not torch.equal(obs["mode"], expected_mode):
                raise ValueError("Native actor does not receive the exact VR-3 mask")
            with torch.inference_mode():
                actions = policy(obs)
                if not torch.isfinite(actions).all():
                    raise ValueError("Nonfinite policy action")
                obs, _, dones, _ = env.step(actions)
            if torch.any(dones) or raw._sim_step_counter - initial_counter != frame * 4:
                raise ValueError("Unexpected reset or physical clock divergence")
            measurement = measure()
            old_phase = planner.phase
            planner.update(frame * raw.step_dt, measurement)
            rows.append(dict(frame=frame, time_s=frame*raw.step_dt, phase=planner.phase,
                issued_targets_xyz=goals.tolist(), issued_targets_wxyz=rotations.tolist(),
                action=array(actions)[0].tolist(), controller=planner.diagnostics(), **measurement))
            if writer is not None:
                if frame == 1:
                    for warmup in range(20):
                        rgb = render_frame(allow_unready=True)
                        if warmup >= 11 and rgb is not None and rgb.std() > 2:
                            break
                    else:
                        raise ValueError("Blank native renderer after bounded warmup")
                    result["render_warmup_calls"] = warmup+1
                else:
                    rgb = render_frame()
                writer.append_data(rgb)
                image_hashes.append(hashlib.sha256(rgb.tobytes()).hexdigest())
            if old_phase != planner.phase or frame % 100 == 0:
                print(json.dumps(dict(frame=frame, phase=planner.phase, reason=planner.reason,
                    left_n=measurement["left_force"], right_n=measurement["right_force"],
                    clearance_m=measurement["box_bottom_z"]-spec.support_top,
                    tilt_deg=float(np.rad2deg(measurement["box_tilt_rad"])))), flush=True)
            if planner.phase in ("FAILED", "SUCCESS"):
                break
        current_sha = hashlib.sha256(b"".join(p.detach().cpu().contiguous().numpy().tobytes() for p in actor.parameters())).hexdigest()
        if current_sha != actor_sha or any(sha256(p) != h for p, h in frozen.items()):
            raise ValueError("Policy or source/asset inputs changed during demo")
        result.update(result="COMPLETE", execution_complete=True, task_success=planner.phase == "SUCCESS",
            terminal_phase=planner.phase, failure_reason=planner.reason, events=planner.events,
            policy_parameters_unchanged=True, inputs_verified_unchanged=True,
            policy_steps=len(rows), simulated_seconds=len(rows)*raw.step_dt,
            maximum_clearance_m=max(r["box_bottom_z"]-spec.support_top for r in rows),
            maximum_left_force_n=max(r["left_force"] for r in rows),
            maximum_right_force_n=max(r["right_force"] for r in rows),
            final_box_xyz=measurement["box_xyz"], final_root_z=measurement["root_z"],
            final_measurement=measurement, video_frames=len(image_hashes),
            maximum_left_force_peak_n=max(r["left_force_peak"] for r in rows),
            maximum_right_force_peak_n=max(r["right_force_peak"] for r in rows),
            physics_steps=raw._sim_step_counter-initial_counter)
    except BaseException as error:
        import traceback
        result.update(result="ERROR", execution_complete=False, task_success=False,
                      error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        print(result["traceback"], flush=True)
    finally:
        for asset, name, original in reversed(restorers):
            setattr(asset, name, original)
        for resource in (writer, provider, env):
            if resource is not None:
                try:
                    resource.close()
                except Exception as error:
                    result.update(result="ERROR", task_success=False, cleanup_error=str(error))
        result["elapsed_seconds"] = time.monotonic()-started
        save(output / "trajectory.json", dict(rows=rows))
        result["trajectory_sha256"] = sha256(output / "trajectory.json")
        if image_hashes:
            save(output / "video_frames.json", dict(rgb_sha256=image_hashes, fps=50,
                                                  starts_at_control_frame=1))
        if (output / "recording_raw.mp4").exists():
            result["raw_video_sha256"] = sha256(output / "recording_raw.mp4")
        save(output / "report.json", result)
        signal.signal(signal.SIGTERM, previous)
        if app is not None:
            app.close(exit_code=0 if result["result"] == "COMPLETE" and result["task_success"] else 2)
    return 0 if result["result"] == "COMPLETE" and result["task_success"] else 2


if __name__ == "__main__":
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--balance-config", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--record", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(engine="IsaacLab / PhysX", policy="official model_22200.pt", mode="VR-3",
            task="standing contact lift and replace", execute=False, max_seconds=20, record=args.record)))
    else:
        if args.record:
            args.enable_cameras = True
        raise SystemExit(run(args, AppLauncher))
