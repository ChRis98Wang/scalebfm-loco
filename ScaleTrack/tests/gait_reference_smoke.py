"""Bounded original-motion gait evidence; not an online waypoint acceptance test.

Own this process with a systemd user cgroup (180s limit, KillMode=control-group).
Only explicit between-clip initialization may reset or write physical state.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/pretrain/rsl_rl"
sys.path.insert(0, str(SCRIPTS))
from evaluation_manifest import locate_package_sources, snapshot_inputs, verify_unchanged
from gait_metrics import summarize_gait_evidence
from gait_runtime_guard import forbid_state_writes


def _array(value):
    value = getattr(value, "torch", value)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value).copy()


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(_jsonable(report), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def initialize_clip(env, command, motion_id):
    """Reset history in the same inference context that created rollout buffers."""
    import torch
    with torch.inference_mode():
        command.motion_ids.fill_(motion_id)
        command.time_steps.zero_()
        command._future_manual_cache.clear()
        return env.reset()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_path", type=Path, required=True)
    parser.add_argument("--motion_file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode_index", type=int, choices=(0, 7), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    for name in ("checkpoint_path", "motion_file"):
        path = getattr(args, name).resolve()
        if not path.is_file():
            parser.error(f"{name} must exist: {path}")
        setattr(args, name, path)
    args.output = args.output.resolve()
    if args.output.exists() or args.output.with_name(args.output.name + ".tmp").exists():
        parser.error("Refusing to overwrite an existing report or temporary report")
    started = time.monotonic()
    deadline = started + 165.
    report = {"schema_version": 1, "result": "RUNNING", "stage": "preflight", "trials": [],
              "seed": args.seed, "mode_index": args.mode_index, "argv": sys.argv,
              "started_monotonic_s": started,
              "protocol": {"claim": "original motion short-clip gait evidence, NOT online XYZ waypoint control",
                           "training_updates": 0, "num_envs": 1, "max_source_frames": 1001,
                           "sample_alignment": "post-physics actual against pre-step reference; initial sample also saved",
                           "horizon": "source_frames - 1 control steps; never cross the final-frame reset boundary",
                           "initialization": "explicit frame-zero reset before each clip; no reset/state-write inside clip",
                           "contact_claim": "none; foot alternation is a kinematic proxy",
                           "observation_noise": False, "reset_disturbance": False,
                           "interval_events": False, "automatic_terminations": False,
                           "online_waypoint_limits": "unchanged; not applied to native offline motion playback"}}
    app = None
    exit_code = 1
    try:
        import yaml
        index = yaml.safe_load(args.motion_file.read_text(encoding="utf-8"))
        if not isinstance(index, dict) or not 1 <= len(index) <= 2:
            raise ValueError("Use an isolated index with one or two explicit motion payloads")
        roots = {"entrypoints": SCRIPTS, "test_harnesses": Path(__file__).parent}
        roots.update(locate_package_sources(("scaletrack", "my_rsl_rl", "isaaclab", "isaaclab_rl", "isaaclab_tasks")))
        manifest = snapshot_inputs(args.checkpoint_path, args.motion_file, roots)
        report["input_manifest"] = manifest
        _write(args.output, report)

        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True, device=args.device).app
        import gymnasium as gym
        import omni.physx
        import torch
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
        from my_rsl_rl.runners.on_policy_runner import OnPolicyRunner
        import scaletrack.tasks  # noqa: F401
        from scaletrack.utils.quaternion_compat import runtime_to_packed_wxyz, detect_runtime_quaternion_order

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
        task = "G1-BFM-Transformer-Tracking"
        cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
        agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
        cfg.seed = agent_cfg.seed = args.seed
        cfg.scene.num_envs = 1
        cfg.sim.device = agent_cfg.device = args.device
        cfg.commands.motion.motion_file = str(args.motion_file)
        cfg.commands.motion.test_motion_file = ""
        cfg.commands.motion.debug_vis = False
        cfg.commands.motion.enable_reset_disturbance = False
        removed = []
        for name in ("time_out", "motion_time_out", "body_pos"):
            if not hasattr(cfg.terminations, name):
                raise ValueError(f"Missing expected termination config: {name}")
            setattr(cfg.terminations, name, None)
            removed.append(name)
        for name, event in vars(cfg.events).items():
            if getattr(event, "mode", None) == "interval":
                setattr(cfg.events, name, None)
        mode_name, bodies = list(cfg.commands.motion.mode_candidates.items())[args.mode_index]
        cfg.commands.motion.mode_candidates = {mode_name: bodies}
        agent_cfg.num_steps_per_env = 1
        report.update(mode=mode_name, active_body_names=bodies, removed_termination_terms=removed,
                      runtime_quaternion_order=detect_runtime_quaternion_order())

        with ExitStack() as cleanup:
            env = gym.make(task, cfg=cfg)
            cleanup.callback(lambda: env.close())
            env = RslRlVecEnvWrapper(env)
            raw = env.unwrapped
            if raw.termination_manager.active_terms or raw.event_manager.active_terms.get("interval", []):
                raise RuntimeError("Constructed gait test retains auto-reset/interval paths")
            command = raw.command_manager.get_term("motion")
            if set(command.motion_names) != set(index) or len(command.motion_names) != len(index):
                raise RuntimeError("Loaded catalog differs from requested clips")
            names = tuple(command.cfg.body_names)
            if names[0] != "pelvis":
                raise RuntimeError("Expected the canonical pelvis root")
            left, right = names.index("left_ankle_roll_link"), names.index("right_ankle_roll_link")
            origin = _array(raw.scene.env_origins)[0]
            order = report["runtime_quaternion_order"]
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
            runner.load(str(args.checkpoint_path), load_optimizer=False, map_location=args.device)
            runner._set_env_is_evaluating()
            original = runner.alg.policy.get_actor_obs
            runner.alg.policy.get_actor_obs = lambda obs, inference=False: original(obs, inference=False)
            policy = runner.get_inference_policy(device=args.device)
            expected_mask = np.array([[float(name in bodies) for name in names]])

            def pose():
                actual = _array(command.robot_body_pos_w)[0] - origin
                quat = runtime_to_packed_wxyz(command.robot_body_quat_w, order)
                value = {"actual_pelvis_xyz": actual[0], "actual_pelvis_quat_wxyz": _array(quat)[0, 0],
                         "reference_pelvis_xyz": _array(command.anchor_pos_w)[0] - origin,
                         "left_ankle_xyz": actual[left], "right_ankle_xyz": actual[right]}
                if any(not np.all(np.isfinite(item)) for item in value.values()):
                    raise RuntimeError("Nonfinite physical/reference pose")
                if abs(np.linalg.norm(value["actual_pelvis_quat_wxyz"]) - 1.) > 1e-3:
                    raise RuntimeError("Physical pelvis orientation is not a unit quaternion")
                return value

            for motion_id, motion in enumerate(command.motion_names):
                frames = int(command.time_totals[motion_id])
                if not 2 <= frames <= 1001:
                    raise ValueError("Source horizon must be 2..1001 frames; no truncation is allowed")
                obs, _ = initialize_clip(env, command, motion_id)  # Explicit between-trial initialization only.
                if int(command.time_steps[0]) != 0 or not np.array_equal(_array(command.mode), expected_mask):
                    raise RuntimeError("Initial source frame or reference mode is incorrect")
                physical = {"elapsed_s": 0., "callbacks": 0}

                def on_physics_step(dt):
                    physical["elapsed_s"] += float(dt)
                    physical["callbacks"] += 1

                trial = {"motion": motion, "source_frames": frames, "expected_control_steps": frames - 1,
                         "control_steps": 0, "physics": physical, "trajectory": [pose()], "result": "RUNNING"}
                report["trials"].append(trial)
                report["stage"] = f"running:{motion}"
                _write(args.output, report)
                subscription = omni.physx.get_physx_interface().subscribe_physics_on_step_events(
                    on_physics_step, pre_step=False, order=0)
                try:
                    with forbid_state_writes(raw, command) as counters, torch.inference_mode():
                        trial["forbidden_calls"] = counters
                        for step in range(frames - 1):
                            if time.monotonic() >= deadline:
                                raise RuntimeError("Internal 165-second deadline expired")
                            if int(command.time_steps[0]) != step:
                                raise RuntimeError("Reference frame did not advance exactly once per step")
                            reference = pose()["reference_pelvis_xyz"]
                            clock_before_policy = physical.copy()
                            actions = policy(obs)
                            if physical != clock_before_policy:
                                raise RuntimeError("Policy call advanced physics")
                            if actions.shape != (1, 29) or not bool(torch.isfinite(actions).all()):
                                raise RuntimeError("Invalid real policy actions")
                            obs, _, dones, _ = env.step(actions)
                            trial["control_steps"] += 1
                            sample = pose()
                            sample["reference_pelvis_xyz"] = reference
                            sample.update(reference_frame=step, physics_s=physical["elapsed_s"])
                            trial["trajectory"].append(sample)
                            if bool(dones.any()) or int(command.time_steps[0]) != step + 1:
                                raise RuntimeError("Unexpected reset or reference advance")
                            if not np.array_equal(_array(command.mode), expected_mask):
                                raise RuntimeError("Reference mode changed during the clip")
                            if abs(physical["elapsed_s"] - (step + 1) * raw.step_dt) > 1e-6:
                                raise RuntimeError("Physical time differs from authorized control steps")
                            q = sample["actual_pelvis_quat_wxyz"]
                            q = q / np.linalg.norm(q)
                            if sample["actual_pelvis_xyz"][2] < .35 or 1 - 2 * (q[1]**2 + q[2]**2) < .5:
                                trial["early_stop"] = "physical height/tilt safety guard"
                                break
                finally:
                    unsubscribe = getattr(subscription, "unsubscribe", None)
                    if unsubscribe is not None:
                        unsubscribe()
                    subscription = None  # Release RAII subscriptions on SDKs without unsubscribe().
                if any(counters.values()):
                    raise RuntimeError("A forbidden state-write was attempted, even if the SDK swallowed its exception")
                trial["terminal_time_step"] = int(command.time_steps[0])
                trial["terminal_reference_pelvis_xyz"] = pose()["reference_pelvis_xyz"]
                samples = trial["trajectory"]
                evidence = summarize_gait_evidence(
                    *[np.stack([sample[key] for sample in samples]) for key in
                      ("actual_pelvis_xyz", "reference_pelvis_xyz", "actual_pelvis_quat_wxyz",
                       "left_ankle_xyz", "right_ankle_xyz")], raw.step_dt)
                trial["evidence"] = asdict(evidence)
                trial["complete_horizon"] = trial["control_steps"] == frames - 1
                trial["result"] = "PASS" if evidence.passed and trial["complete_horizon"] else "FAIL"
                _write(args.output, report)
                print(f"[BFM GAIT] {mode_name} {motion}: {trial['result']} {trial['evidence']}", flush=True)

        verify_unchanged(manifest, snapshot_inputs(args.checkpoint_path, args.motion_file, roots))
        report["inputs_verified_unchanged"] = True
        report["result"] = "PASS" if all(trial["result"] == "PASS" for trial in report["trials"]) else "FAIL"
        exit_code = 0 if report["result"] == "PASS" else 1
    except BaseException as error:
        report.update(result="FAIL", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        report["exit_requested_monotonic_s"] = time.monotonic()
        report["elapsed_wall_s"] = time.monotonic() - started
        try:
            _write(args.output, report)
            print(f"[BFM GAIT] {report['result']} report={args.output}", flush=True)
        finally:
            if app is not None:
                app.close(exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
