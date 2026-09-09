"""Non-invasive observer for a bounded ScaleBFM training run."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import sys
import traceback

ROOT = Path(__file__).resolve().parents[4]
SCALETRACK = ROOT / "ScaleTrack"
ENTRY = SCALETRACK / "scripts/pretrain/rsl_rl/train.py"
TRAIN_INDEX = ROOT / "ScaleRetarget/retargeted_dataset/amass_full_v2_train.yaml"
EXPERIMENT_DIR = SCALETRACK / "logs/rsl_rl/g1_bfm_tracking_exp"
SOURCE_CHECKPOINT = EXPERIMENT_DIR / "official_lr1e5_derived_20260907/model_22200.pt"
SOURCE_SHA256 = "269f17e040ad0c27f651a25097a2650380ff1a05714dde102625aabd8d327c46"
MODE_NAMES = ("Pelvis-1", "UMI-2", "VR-3", "UMI-4", "VR-5", "UpperBody-6", "UpperBody-Mobile-7", "WholeBody-14")


@dataclass(frozen=True)
class ProbeExpectations:
    schedule: str
    updates: int
    learning_rate: float
    num_envs: int = 128
    steps_per_env: int = 64
    initial_iteration: int = 22199
    entropy_coef: float = 0.001
    epochs: int = 2
    minibatches: int = 32
    save_interval: int = 1
    sampling_strategy: str = "legacy"
    resample_interval: int = 0


def _digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _option(argv, name, required=True):
    found = []
    for i, token in enumerate(argv):
        if token == name:
            if i + 1 >= len(argv) or argv[i + 1].startswith("--"):
                raise ValueError(f"{name} requires one value")
            found.append(argv[i + 1])
        elif token.startswith(name + "="):
            found.append(token.split("=", 1)[1])
    if len(found) > 1:
        raise ValueError(f"{name} must be supplied exactly once")
    if not found:
        if required:
            raise ValueError(f"missing required training argument {name}")
        return None
    return found[0]


def _validate_train_argv(argv, expected):
    checks = {
        "--task": "G1-BFM-Transformer-Tracking", "--num_envs": str(expected.num_envs),
        "--max_iterations": str(expected.updates), "--seed": "42", "--resume": "True",
        "--load_run": "official_lr1e5_derived_20260907", "--checkpoint": "model_22200.pt",
        "--logger": "tensorboard",
    }
    for option, wanted in checks.items():
        if _option(argv, option) != wanted:
            raise ValueError(f"{option} must be {wanted}")
    if Path(_option(argv, "--motion_file")).resolve() != TRAIN_INDEX.resolve():
        raise ValueError("--motion_file must be the canonical full-v2 train index")
    if _option(argv, "--test_motion_file", False) is not None:
        raise ValueError("probe forbids loading a test/heldout motion index")
    if argv.count("--headless") != 1:
        raise ValueError("probe requires exactly one --headless flag")
    required = {f"agent.algorithm.entropy_coef={expected.entropy_coef}",
                f"agent.algorithm.schedule={expected.schedule}",
                "agent.eval_during_training=False", f"agent.save_interval={expected.save_interval}"}
    if expected.sampling_strategy == "coverage":
        required.update({"agent.motion_sampling_strategy=coverage",
                         f"agent.motion_resample_interval={expected.resample_interval}"})
    if required.difference(argv):
        raise ValueError(f"missing required Hydra overrides: {sorted(required.difference(argv))}")
    run_name = _option(argv, "--run_name")
    if not run_name or run_name in {".", ".."} or Path(run_name).name != run_name:
        raise ValueError("--run_name must be one new directory basename")
    run_dir = EXPERIMENT_DIR / run_name
    if run_dir.exists() or run_dir.is_symlink():
        raise FileExistsError(f"training run directory already exists: {run_dir}")
    return run_name, run_dir


def _steps(optimizer):
    result = []
    for state in optimizer.state.values():
        if "step" not in state:
            raise RuntimeError("loaded Adam state is missing step")
        value = state["step"]
        result.append(float(value.item() if hasattr(value, "item") else value))
    if not result:
        raise RuntimeError("loaded Adam state is empty")
    return result


def _diagnostics(values):
    result = {}
    for key, value in values.items():
        value = value.item() if hasattr(value, "item") else value
        value = int(value) if isinstance(value, int) else float(value)
        if isinstance(value, float) and not math.isfinite(value):
            raise RuntimeError(f"non-finite PPO diagnostic {key}")
        result[key] = value
    return result


def _delta(now, before):
    return dict(sorted((key, now[key] - before[key]) for key in now if now[key] != before[key]))


def _atomic_json(path, payload):
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"progress temporary path already exists: {temporary}")
    try:
        with temporary.open("x") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()
        raise


def _observe_learning(runner, original_learn, expected, indexed_names, progress_callback=None):
    """Observe an already-loaded runner; kept separate for CPU fake tests."""
    import torch
    command = runner.env.unwrapped.command_manager.get_term("motion")
    if (runner.env.num_envs, runner.num_steps_per_env) != (expected.num_envs, expected.steps_per_env):
        raise RuntimeError("resolved rollout shape differs from expectation")
    if runner.current_learning_iteration != expected.initial_iteration or runner.eval_during_training:
        raise RuntimeError("unexpected starting iteration or training-time evaluation")
    if runner.save_interval != expected.save_interval:
        raise RuntimeError("resolved save interval differs from expectation")
    if (runner.motion_sampling_strategy != expected.sampling_strategy
            or runner.motion_resample_interval != expected.resample_interval):
        raise RuntimeError("resolved motion sampling configuration differs from expectation")
    def current_cohort_sampler():
        sampler = getattr(runner, "motion_cohort_sampler", None)
        return getattr(runner, "_motion_cohort_sampler", None) if sampler is None else sampler

    if expected.sampling_strategy == "legacy" and current_cohort_sampler() is not None:
        raise RuntimeError("legacy strategy unexpectedly constructed a motion cohort sampler")
    alg = runner.alg
    if (alg.schedule, alg.entropy_coef, alg.num_learning_epochs, alg.num_mini_batches) != (
        expected.schedule, expected.entropy_coef, expected.epochs, expected.minibatches):
        raise RuntimeError("resolved PPO configuration differs from expectation")
    for label, scalar, optimizer in (("actor", alg.actor_learning_rate, alg.actor_optimizer),
                                     ("critic", alg.critic_learning_rate, alg.critic_optimizer)):
        if len(optimizer.param_groups) != 1 or float(scalar) != expected.learning_rate or float(optimizer.param_groups[0]["lr"]) != expected.learning_rate:
            raise RuntimeError(f"loaded {label} Adam learning rate differs from expectation")
    if command.is_evaluating or getattr(command, "has_test_set", False):
        raise RuntimeError("heldout/evaluation motions must not be loaded")
    if set(command.motion_names_train) != indexed_names:
        raise RuntimeError("loaded motions do not exactly match the v2 index")
    mode_names = tuple(command.cfg.mode_candidates)
    if mode_names != MODE_NAMES or command._mode_table.shape[0] != 8:
        raise RuntimeError("task does not expose the canonical eight masks")

    actor_ids = {id(p) for p in alg.policy.actor_parameters}
    actor_before = {n: p.detach().cpu().clone() for n, p in alg.policy.named_parameters() if id(p) in actor_ids}
    if not actor_before:
        raise RuntimeError("actor parameter set is empty")
    actor_steps_before, critic_steps_before = _steps(alg.actor_optimizer), _steps(alg.critic_optimizer)
    motions, datasets, modes = Counter(), Counter(), Counter()
    previous = (Counter(), Counter(), Counter())
    updates, pending, completed_steps, observed_first_act = [], None, 0, False
    original_act, original_process, original_update = alg.act, alg.process_env_step, alg.update

    def observed_act(obs):
        nonlocal pending, observed_first_act
        if pending is not None or command.is_evaluating:
            raise RuntimeError("invalid action/step lifecycle or evaluation entered rollout")
        if not observed_first_act:
            sampler = current_cohort_sampler()
            if expected.sampling_strategy == "coverage" and sampler is None:
                raise RuntimeError("coverage sampler was not prepared before the first rollout action")
            if expected.sampling_strategy == "legacy" and sampler is not None:
                raise RuntimeError("legacy strategy constructed a motion cohort sampler")
            observed_first_act = True
        matches = (command._mode.detach()[:, None, :] == command._mode_table.detach()[None, :, :]).all(-1)
        if not torch.all(matches.sum(1) == 1):
            raise RuntimeError("non-canonical or ambiguous mask")
        staged = (command.motion_ids.tolist(), matches.long().argmax(1).cpu().tolist())
        actions = original_act(obs)
        pending = staged
        return actions

    def observed_process(*args, **kwargs):
        nonlocal pending, completed_steps
        if pending is None:
            raise RuntimeError("transition has no staged action metadata")
        result = original_process(*args, **kwargs)
        motion_ids, mode_ids = pending
        pending = None
        for motion_id in motion_ids:
            name = command.motion_names[motion_id]
            motions[name] += 1
            datasets[name.split("/", 1)[0]] += 1
        for mode_id in mode_ids:
            modes[mode_names[mode_id]] += 1
        completed_steps += len(motion_ids)
        return result

    def observed_update(*args, **kwargs):
        nonlocal previous
        result = original_update(*args, **kwargs)
        diag = _diagnostics(alg.update_diagnostics)
        if diag.get("minibatches") != expected.epochs * expected.minibatches:
            raise RuntimeError("PPO update did not finish all minibatches")
        scheduler_keys = [k for k in diag if k.startswith("scheduler_kl")]
        if expected.schedule == "fixed":
            lr_keys = [k for k in diag if k.startswith(("actor_lr_", "critic_lr_"))]
            if scheduler_keys or not lr_keys or any(diag[k] != expected.learning_rate for k in lr_keys):
                raise RuntimeError("fixed schedule diagnostics violate the LR contract")
        elif not scheduler_keys:
            raise RuntimeError("adaptive schedule did not publish scheduler KL")
        sampler = current_cohort_sampler()
        if expected.sampling_strategy == "coverage" and sampler is None:
            raise RuntimeError("coverage sampler disappeared during learning")
        if expected.sampling_strategy == "legacy" and sampler is not None:
            raise RuntimeError("legacy strategy constructed a motion cohort sampler")
        cohort_snapshot = sampler.snapshot() if sampler is not None else None
        updates.append({"ordinal": len(updates) + 1, "completed_environment_steps": completed_steps,
                        "motion_steps": _delta(motions, previous[0]), "dataset_steps": _delta(datasets, previous[1]),
                        "mode_steps": _delta(modes, previous[2]), "ppo_diagnostics": diag,
                        "motion_cohort_sampler": cohort_snapshot})
        previous = motions.copy(), datasets.copy(), modes.copy()
        if progress_callback is not None:
            progress_callback({
                "result": "RUNNING", "completed_updates": len(updates),
                "completed_environment_steps": completed_steps,
                "unique_motion": len(motions),
                "unique_kit": sum(name.startswith("KIT/") for name in motions),
                "motion_steps": dict(sorted(motions.items())),
                "dataset_steps": dict(sorted(datasets.items())),
                "mode_steps": {name: modes[name] for name in MODE_NAMES},
                "current_update_diagnostics": diag,
                "motion_cohort_sampler": cohort_snapshot,
            })
        return result

    alg.act, alg.process_env_step, alg.update = observed_act, observed_process, observed_update
    try:
        result = original_learn(runner, num_learning_iterations=expected.updates, init_at_random_ep_len=True)
    finally:
        alg.act, alg.process_env_step, alg.update = original_act, original_process, original_update
        if runner.writer is not None:
            runner.writer.flush()
            runner.writer.close()

    expected_env_steps = expected.updates * expected.num_envs * expected.steps_per_env
    final_iteration = expected.initial_iteration + expected.updates - 1
    if (result is not None or pending is not None or len(updates) != expected.updates
            or completed_steps != expected_env_steps or runner.current_learning_iteration != final_iteration):
        raise RuntimeError("runner did not complete the exact requested rollout/update count")
    if datasets["KIT"] <= 0 or any(modes[name] <= 0 for name in MODE_NAMES):
        raise RuntimeError("KIT or one of the eight masks did not enter the rollout")
    actor_changed = 0
    for name, parameter in alg.policy.named_parameters():
        if id(parameter) in actor_ids:
            current = parameter.detach().cpu()
            if not torch.isfinite(current).all():
                raise RuntimeError(f"non-finite actor parameter: {name}")
            actor_changed += int(not torch.equal(current, actor_before[name]))
    if actor_changed == 0:
        raise RuntimeError("no actor parameter changed")
    increment = expected.updates * expected.epochs * expected.minibatches
    actor_after, critic_after = _steps(alg.actor_optimizer), _steps(alg.critic_optimizer)
    for label, before, after in (("actor", actor_steps_before, actor_after), ("critic", critic_steps_before, critic_after)):
        if len(before) != len(after) or any(b + increment != a for b, a in zip(before, after)):
            raise RuntimeError(f"{label} Adam states did not advance by {increment}")
    if progress_callback is not None:
        final_sampler = current_cohort_sampler()
        final_progress = {
            "result": "COMPLETE", "completed_updates": len(updates),
            "completed_environment_steps": completed_steps,
            "unique_motion": len(motions),
            "unique_kit": sum(name.startswith("KIT/") for name in motions),
            "motion_steps": dict(sorted(motions.items())),
            "dataset_steps": dict(sorted(datasets.items())),
            "mode_steps": {name: modes[name] for name in MODE_NAMES},
            "current_update_diagnostics": updates[-1]["ppo_diagnostics"],
            "motion_cohort_sampler": final_sampler.snapshot() if final_sampler is not None else None,
        }
        progress_callback(final_progress)
    return {
        "resolved": {"schedule": alg.schedule, "initial_learning_rate": expected.learning_rate,
                     "entropy_coef": alg.entropy_coef, "num_envs": runner.env.num_envs,
                     "steps_per_env": runner.num_steps_per_env, "epochs": alg.num_learning_epochs,
                     "minibatches": alg.num_mini_batches, "initial_iteration": expected.initial_iteration,
                     "final_iteration": runner.current_learning_iteration,
                     "eval_during_training": runner.eval_during_training,
                     "save_interval": runner.save_interval,
                     "motion_sampling_strategy": runner.motion_sampling_strategy,
                     "motion_resample_interval": runner.motion_resample_interval,
                     "final_actor_learning_rate": alg.actor_learning_rate,
                     "final_critic_learning_rate": alg.critic_learning_rate},
        "completed_updates": len(updates), "completed_environment_steps": completed_steps,
        "changed_actor_parameters": actor_changed, "actor_adam_states": len(actor_after),
        "critic_adam_states": len(critic_after), "adam_step_increment": increment,
        "dataset_steps": dict(sorted(datasets.items())), "motion_steps": dict(sorted(motions.items())),
        "mode_steps": {name: modes[name] for name in MODE_NAMES},
        "unique_kit_motions": sum(name.startswith("KIT/") for name in motions), "updates": updates,
    }


def _run(args, report):
    import yaml
    run_name, run_dir = _validate_train_argv(args.train_args, args.expected)
    if _digest(SOURCE_CHECKPOINT) != SOURCE_SHA256:
        raise RuntimeError("approved derived checkpoint hash changed")
    index_sha = _digest(TRAIN_INDEX)
    indexed = yaml.safe_load(TRAIN_INDEX.read_text())
    if not isinstance(indexed, dict) or len(indexed) != 7174:
        raise RuntimeError("full-v2 index does not contain 7174 unique entries")
    sys.path.insert(0, str(ENTRY.parent))
    from my_rsl_rl.runners.on_policy_runner import OnPolicyRunner
    original_learn, holder, original_argv = OnPolicyRunner.learn, {}, sys.argv

    def observed_learn(runner, *positional, **keywords):
        if positional or keywords.get("num_learning_iterations") != args.expected.updates:
            raise RuntimeError("train.py invoked learn with unexpected arguments")
        if Path(runner.log_dir).resolve() != run_dir.resolve():
            raise RuntimeError("resolved log directory differs from preflight")
        callback = (lambda payload: _atomic_json(args.progress, payload)) if args.progress is not None else None
        evidence = _observe_learning(runner, original_learn, args.expected, set(indexed), callback)
        if _digest(SOURCE_CHECKPOINT) != SOURCE_SHA256 or _digest(TRAIN_INDEX) != index_sha:
            raise RuntimeError("checkpoint or training index changed during run")
        payload = {"schema": 1, "result": "PASS", "purpose": "schedule A/B execution evidence, not policy quality",
                   "argv": args.train_args, "run_name": run_name, "run_dir": str(run_dir),
                   "train_index": str(TRAIN_INDEX), "train_index_sha256": index_sha,
                   "source_checkpoint": str(SOURCE_CHECKPOINT), "source_checkpoint_sha256": SOURCE_SHA256,
                   **evidence}
        # SimulationApp.close uses fast shutdown in this environment, so persist
        # the complete learning audit before control returns to train.py.
        with report.open("x") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False); stream.write("\n")
        print("BFM_TRAINING_PROBE PASS", report, flush=True)
        holder["evidence"] = payload

    OnPolicyRunner.learn, sys.argv = observed_learn, [str(ENTRY), *args.train_args]
    try:
        runpy.run_path(str(ENTRY), run_name="__main__")
    finally:
        sys.argv, OnPolicyRunner.learn = original_argv, original_learn
    if "evidence" not in holder:
        raise RuntimeError("entry point returned without observed learning")
    return holder["evidence"]


def _parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected_schedule", choices=("fixed", "adaptive"), required=True)
    parser.add_argument("--expected_updates", type=int, required=True)
    parser.add_argument("--expected_lr", type=float, required=True)
    parser.add_argument("--expected_save_interval", type=int, default=1)
    parser.add_argument("--expected_sampling_strategy", choices=("legacy", "coverage"), default="legacy")
    parser.add_argument("--expected_resample_interval", type=int, default=0)
    parser.add_argument("--progress", type=Path)
    parser.add_argument("train_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.train_args[:1] == ["--"]:
        args.train_args = args.train_args[1:]
    if (args.expected_updates <= 0 or not math.isfinite(args.expected_lr) or args.expected_lr <= 0
            or args.expected_save_interval <= 0 or args.expected_resample_interval < 0):
        parser.error("expected updates/LR/save interval must be positive and resample interval non-negative")
    if args.expected_sampling_strategy == "coverage" and args.expected_resample_interval <= 0:
        parser.error("coverage sampling requires a positive resample interval")
    if args.expected_sampling_strategy == "legacy" and args.expected_resample_interval != 0:
        parser.error("legacy sampling requires resample interval zero")
    args.progress = args.progress.resolve() if args.progress is not None else None
    args.expected = ProbeExpectations(args.expected_schedule, args.expected_updates, args.expected_lr,
        save_interval=args.expected_save_interval, sampling_strategy=args.expected_sampling_strategy,
        resample_interval=args.expected_resample_interval)
    return args


def main(argv=None):
    args = _parse(argv)
    report = args.report.resolve()
    if report.exists() or report.is_symlink():
        raise FileExistsError(f"report already exists: {report}")
    if not report.parent.is_dir():
        raise FileNotFoundError(f"report parent does not exist: {report.parent}")
    if args.progress is not None:
        if args.progress == report:
            raise ValueError("report and progress paths must be different")
        if args.progress.exists() or args.progress.is_symlink():
            raise FileExistsError(f"progress already exists: {args.progress}")
        if not args.progress.parent.is_dir():
            raise FileNotFoundError(f"progress parent does not exist: {args.progress.parent}")
        temporary = args.progress.with_name(f".{args.progress.name}.tmp")
        if temporary.exists() or temporary.is_symlink():
            raise FileExistsError(f"progress temporary path already exists: {temporary}")
    try:
        payload = _run(args, report)
    except BaseException as error:
        payload = {"schema": 1, "result": "FAIL", "purpose": "schedule A/B execution evidence, not policy quality",
                   "argv": args.train_args, "error_type": type(error).__name__, "error": str(error),
                   "traceback": traceback.format_exc()}
        if not report.exists():
            with report.open("x") as stream:
                json.dump(payload, stream, indent=2, allow_nan=False); stream.write("\n")
        if args.progress is not None:
            prior = {}
            if args.progress.exists() and not args.progress.is_symlink():
                try:
                    prior = json.loads(args.progress.read_text())
                except (OSError, ValueError):
                    prior = {}
            prior.update({"result": "FAIL", "error_type": type(error).__name__, "error": str(error)})
            _atomic_json(args.progress, prior)
        print(payload["traceback"], file=sys.stderr, flush=True)
        raise
    # The success report was flushed before train.py called SimulationApp.close.


if __name__ == "__main__":
    main()
