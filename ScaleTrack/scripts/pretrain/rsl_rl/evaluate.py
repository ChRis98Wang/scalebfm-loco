"""Evaluate one trusted checkpoint without training or overwriting its run logs."""

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

from evaluation_manifest import locate_package_sources, snapshot_inputs, verify_unchanged
from target_object import add_target_object_args, attach_target_object, target_metrics, target_spec_from_args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_path", type=Path, required=True, help="Trusted local .pt checkpoint")
    parser.add_argument("--motion_file", type=Path, required=True, help="Evaluation-only YAML index")
    parser.add_argument("--output", type=Path, required=True, help="New JSON report; existing files are refused")
    parser.add_argument("--task", default="G1-BFM-Transformer-Tracking")
    parser.add_argument("--num_envs", type=int, default=128)
    parser.add_argument("--max_steps", type=int, default=1000, help="Per-clip cap; 1000 steps = 20 s at 50 Hz")
    parser.add_argument("--mode_index", type=int, default=7, help="7=WholeBody-14; 0=Pelvis-1; 2=VR-3")
    parser.add_argument("--mask_metrics", action="store_true",
                        help="Add active-link tracking diagnostics and max-link deviation rates (schema 4)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    add_target_object_args(parser)
    args = parser.parse_args()
    try:
        target_spec = target_spec_from_args(args)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    for name in ("checkpoint_path", "motion_file"):
        path = getattr(args, name).resolve()
        if not path.is_file():
            parser.error(f"{name} does not exist: {path}")
        setattr(args, name, path)
    args.output = args.output.resolve()
    if args.output.exists():
        parser.error(f"Refusing to overwrite existing report: {args.output}")
    if args.num_envs < 1 or args.max_steps < 1:
        parser.error("num_envs and max_steps must be positive")

    import yaml
    with args.motion_file.open(encoding="utf-8") as stream:
        index = yaml.safe_load(stream)
    if not isinstance(index, dict) or not index:
        parser.error("motion_file must contain a nonempty name-to-path mapping")
    started = time.monotonic()
    code_roots = {"entrypoints": Path(__file__).resolve().parent}
    try:
        code_roots.update(locate_package_sources(("scaletrack", "my_rsl_rl", "isaaclab", "isaaclab_rl", "isaaclab_tasks")))
    except ValueError as error:
        parser.error(f"{error}; use the existing IsaacLab interpreter")
    print("[BFM EVAL] Snapshotting motion payloads and runtime Python sources", flush=True)
    initial_inputs = snapshot_inputs(args.checkpoint_path, args.motion_file, code_roots)
    versions = {}
    for name in ("isaaclab", "isaaclab-rl", "isaaclab-tasks", "isaacsim", "numpy", "gymnasium"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None

    # Import/launch Kit only after CLI validation; help and missing files are cheap.
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, device=args.device).app
    exit_code = 1
    try:
        import gymnasium as gym
        import torch
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
        from my_rsl_rl.runners.on_policy_runner import OnPolicyRunner
        import scaletrack.tasks  # noqa: F401
        from evaluation import evaluate_motion_batches, summarize_motion_metrics
        from masked_metrics import masked_metrics_context, summarize_masked_tracking

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
        env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
        agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
        env_cfg.seed = agent_cfg.seed = args.seed
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.sim.device = agent_cfg.device = args.device
        env_cfg.commands.motion.motion_file = str(args.motion_file)
        env_cfg.commands.motion.test_motion_file = ""
        env_cfg.commands.motion.debug_vis = False
        env_cfg.commands.motion.enable_reset_disturbance = False
        candidates = list(env_cfg.commands.motion.mode_candidates.items())
        if not 0 <= args.mode_index < len(candidates):
            raise ValueError(f"Invalid mode_index: {args.mode_index}")
        mode_name, body_names = candidates[args.mode_index]
        env_cfg.commands.motion.mode_candidates = {mode_name: body_names}
        # No PPO rollout or optimizer updates; reduce unused rollout allocation.
        agent_cfg.num_steps_per_env = 1
        attach_target_object(env_cfg, target_spec)

        with ExitStack() as cleanup:
            env = gym.make(args.task, cfg=env_cfg)
            cleanup.callback(lambda: env.close())
            env = RslRlVecEnvWrapper(env)
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
            runner.load(str(args.checkpoint_path), load_optimizer=False, map_location=args.device)
            runner._set_env_is_evaluating()
            command = env.unwrapped.command_manager.get_term("motion")
            if len(command.motion_names) != len(index) or set(command.motion_names) != set(index):
                raise RuntimeError("Loaded motion catalog differs from the requested index; no clips may be skipped")

            original_get_actor_obs = runner.alg.policy.get_actor_obs

            def fixed_mode_observations(obs, inference=False):
                return original_get_actor_obs(obs, inference=False)

            runner.alg.policy.get_actor_obs = fixed_mode_observations
            policy = runner.get_inference_policy(device=args.device)
            metric_keys = list(agent_cfg.eval_metric_keys)
            extra_metrics = None
            if target_spec is not None:
                metric_keys += ["target_root_distance", "target_height", "target_speed"]
                extra_metrics = lambda: target_metrics(env.unwrapped, command)
            print(f"[BFM EVAL] {mode_name}: {command.num_motion} clips, checkpoint={args.checkpoint_path}", flush=True)
            with masked_metrics_context(command, enabled=args.mask_metrics) as mask_keys:
                rows = evaluate_motion_batches(
                    env, policy, command, metric_keys + list(mask_keys), max_steps=args.max_steps,
                    extra_metrics=extra_metrics,
                    progress=lambda done, total: print(f"[BFM EVAL] {done}/{total} clips", flush=True),
                )
            summary = summarize_motion_metrics(rows)
            if args.mask_metrics:
                summary["masked_tracking"] = summarize_masked_tracking(rows)
            step_dt = env.unwrapped.step_dt

        verify_unchanged(initial_inputs, snapshot_inputs(args.checkpoint_path, args.motion_file, code_roots))
        repo = Path(__file__).resolve().parents[4]
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
        ).stdout.strip()
        report = {
            "schema_version": 4 if args.mask_metrics else 3,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(args.checkpoint_path), "checkpoint_sha256": initial_inputs["checkpoint"]["sha256"],
            "motion_index": str(args.motion_file), "motion_index_sha256": initial_inputs["motion_index"]["sha256"],
            "task": args.task, "mode": mode_name, "mode_index": args.mode_index,
            "scene_variant": "target_object" if target_spec is not None else "baseline",
            "target_object": target_spec.to_dict() if target_spec is not None else None,
            "seed": args.seed, "num_envs": args.num_envs, "max_steps": args.max_steps,
            "step_dt": step_dt, "device": args.device,
            "python": platform.python_version(), "torch": torch.__version__,
            "package_versions": versions, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(args.device) if str(args.device).startswith("cuda") else None,
            "git_revision": revision,
            "input_manifest": initial_inputs,
            "input_manifest_verified_unchanged": True,
            "elapsed_seconds": time.monotonic() - started,
            "protocol": {
                "training_updates": 0,
                "startup_domain_randomization": "task defaults, same initialization seed",
                "determinism": "paired initialization seed; GPU physics is not guaranteed bitwise deterministic",
                "reset_disturbance": False, "observation_noise": False, "interval_pushes": False,
                "action_policy": "deterministic mean, selected reference-body mask",
                "ground": "local infinite static plane; configured friction unchanged",
                "object_objective": "none; optional object diagnostics are not task completion or grasp success",
                "averaging": "valid post-step command metrics, averaged within each clip then equally across clips",
                "horizon": "min(source_frames - 1, max_steps); no auto-reset samples",
                "thresholds": "maximum over time of mean-14-body global position error in metres; not task completion",
                "validation_scope": "held out from local fine-tuning; overlap with upstream pretraining is unknown",
            },
            "summary": summary, "motions": rows,
        }
        if args.mask_metrics:
            report["active_body_names"] = body_names
            report["protocol"]["masked_tracking"] = {
                "measurement_boundary": "post physics, pre reference advance; same as legacy command metrics",
                "position": "mean Euclidean distance over active links, metres",
                "root_relative_position": "subtract each physical/reference root translation only; no yaw alignment",
                "rotation": "mean shortest quaternion geodesic over active links, radians",
                "tracking_failure": "any link > 0.5 m at any valid step; active/all configured reference links reported separately",
                "limitations": "reference deviation is not fall detection; episodes continue after deviation",
            }
        encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(encoded + "\n")
        print(f"[BFM EVAL] COMPLETE: {args.output}", flush=True)
        print(json.dumps(summary["mean_metrics"], ensure_ascii=False), flush=True)
        exit_code = 0
    except BaseException:
        import traceback
        traceback.print_exc()
        raise
    finally:
        from inspect import signature
        try:
            supports_exit_code = "exit_code" in signature(app.close).parameters
        except (TypeError, ValueError):
            supports_exit_code = False
        app.close(**({"exit_code": exit_code} if supports_exit_code else {}))


if __name__ == "__main__":
    main()
