#!/usr/bin/env python3
"""Versioned, bounded 256-origin UMR data A/B PPO execution observer.

This does not alter the trainer or promote a model. Both arms restart the same
approved checkpoint; a PASS proves finite learning and coverage, not quality.
Importing this module does not import Isaac/Kit or start a simulator.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import runpy
import signal
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
TRACK = ROOT / "ScaleTrack"
ENTRY = TRACK / "scripts/pretrain/rsl_rl/train.py"
RUNS = (ROOT / "logs/rsl_rl/g1_bfm_tracking_exp").resolve()
ISAAC_PYTHON = Path("/home/sw/isaaclab_ws/env_isaaclab_sim6_newton/bin/python")
OLD_PROBE = ENTRY.with_name("training_probe.py")
OLD_PROBE_SHA256 = "0404f822c28c89356e8479a85f368be28218ae7e0839258097a8c2cce80ca8d4"
DERIVED = RUNS / "official_lr1e5_derived_20260907/model_22200.pt"
DERIVED_SHA256 = "269f17e040ad0c27f651a25097a2650380ff1a05714dde102625aabd8d327c46"
SCHEMA = "bfm.umr_behavior_training_probe/1"
DATASET_SCHEMA = "bfm.umr_behavior_ab_dataset/1"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _load_old_probe():
    if digest(OLD_PROBE) != OLD_PROBE_SHA256:
        raise RuntimeError("frozen legacy training probe SHA256 changed")
    name = "_bfm_umr_legacy_probe_0404f822"
    spec = importlib.util.spec_from_file_location(name, OLD_PROBE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


legacy = _load_old_probe()


def expectations(updates):
    if type(updates) is not int or updates not in (6, 100):
        raise ValueError("only 6-update smoke or 100-update paired training is approved")
    return legacy.ProbeExpectations("fixed", updates, 1e-5, save_interval=50,
                                    sampling_strategy="coverage", resample_interval=5)


def bounded_unit():
    matches = re.findall(r"(?:^|/)(bfm-umr-behavior-[A-Za-z0-9_.-]+\.service)(?:/|$)",
                         Path("/proc/self/cgroup").read_text(), flags=re.MULTILINE)
    if len(matches) != 1:
        raise ValueError("use one owned bounded bfm-umr-behavior-*.service user cgroup")
    properties = ("KillMode", "Restart", "RuntimeMaxUSec", "MemoryMax", "TasksMax")
    command = ["systemctl", "--user", "show", matches[0]]
    for key in properties:
        command.extend(["-p", key])
    values = dict(line.split("=", 1) for line in subprocess.check_output(
        command, text=True, timeout=10).splitlines())
    unbounded = {None, "", "0", "infinity", "18446744073709551615"}
    if (values.get("KillMode") != "control-group" or values.get("Restart") != "no"
            or any(values.get(key) in unbounded for key in properties[2:])):
        raise ValueError("finite runtime/memory/tasks, Restart=no and control-group cleanup required")
    return matches[0]


def new_file(path):
    path = Path(os.path.abspath(path))
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError(f"output symlink is forbidden: {path}")
    if path.exists():
        raise FileExistsError(path)
    if not path.parent.is_dir():
        raise FileNotFoundError(path.parent)
    return path


def run_directory(name):
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,70}", name):
        raise ValueError("run-name must be a new lowercase basename, at most 71 characters")
    return new_file(RUNS / name)


def train_argv(manifest, arm, run_name, updates):
    expectations(updates)
    if arm not in ("a", "b"):
        raise ValueError("arm must be a or b")
    return ["--task", "G1-BFM-Transformer-Tracking", "--motion_file", manifest["arms"][arm]["index"],
            "--run_name", run_name, "--logger", "tensorboard", "--num_envs", "128",
            "--max_iterations", str(updates), "--seed", "42", "--resume", "True",
            "--load_run", "official_lr1e5_derived_20260907", "--checkpoint", "model_22200.pt",
            "--device", "cuda:0", "--headless", "agent.num_steps_per_env=64",
            "agent.algorithm.num_learning_epochs=2", "agent.algorithm.num_mini_batches=32",
            "agent.algorithm.actor_learning_rate=1e-5", "agent.algorithm.critic_learning_rate=1e-5",
            "agent.algorithm.entropy_coef=0.001", "agent.algorithm.schedule=fixed",
            "agent.eval_during_training=False", "agent.save_interval=50",
            "agent.motion_sampling_strategy=coverage", "agent.motion_resample_interval=5"]


def _dataset_validator(path):
    # The builder's full provenance checks remain the authority for membership.
    from scripts.build_umr_behavior_ab import validate_manifest
    return validate_manifest(path)


def prepare_dataset(path, validator=None):
    import yaml
    path = Path(path).resolve(strict=True)
    before = digest(path)
    manifest = (validator or _dataset_validator)(path)
    if manifest.get("schema") != DATASET_SCHEMA:
        raise ValueError("unsupported data A/B manifest schema")
    ordered = manifest["ordered_origins"]
    targets, replay = manifest["target_origins"], manifest["replay_origins"]
    if (len(ordered) != 256 or len(set(ordered)) != 256 or len(targets) != 17
            or len(set(targets)) != 17 or len(replay) != 239 or len(set(replay)) != 239
            or set(targets) & set(replay) or set(ordered) != set(targets) | set(replay)):
        raise ValueError("dataset must have exactly 17 targets + 239 disjoint replay origins")
    if not any(name.startswith("KIT/") for name in ordered):
        raise ValueError("KIT replay is required for the eight-mask learning audit")
    sampling = manifest["sampling"]
    wanted = dict(strategy="coverage", weights="uniform", num_envs=128,
                  resample_interval=5, weight_per_origin=1 / 256)
    if any(sampling.get(key) != value for key, value in wanted.items()):
        raise ValueError("manifest sampling differs from the fixed 256-origin protocol")
    checkpoint = manifest["checkpoint"]
    if (Path(checkpoint["path"]).resolve() != DERIVED.resolve()
            or checkpoint["sha256"] != DERIVED_SHA256 or digest(DERIVED) != DERIVED_SHA256):
        raise ValueError("both arms must restart the approved derived checkpoint")
    frozen = {str(path): before, str(DERIVED.resolve()): DERIVED_SHA256,
              str(OLD_PROBE): OLD_PROBE_SHA256, str(Path(__file__).resolve()): digest(__file__)}
    for arm in ("a", "b"):
        index = Path(manifest["arms"][arm]["index"]).resolve(strict=True)
        if str(index) != manifest["arms"][arm]["index"]:
            raise ValueError("arm indexes must be canonical absolute paths")
        expected_sha = manifest["arms"][arm]["sha256"]
        if digest(index) != expected_sha:
            raise ValueError(f"arm {arm} index SHA256 changed")
        data = yaml.safe_load(index.read_text())
        if not isinstance(data, dict) or list(data) != ordered:
            raise ValueError(f"arm {arm} index order differs from ordered_origins")
        frozen[str(index)] = expected_sha
    verify_frozen(frozen)
    return manifest, frozen


def verify_frozen(frozen):
    for path, sha256 in frozen.items():
        if digest(path) != sha256:
            raise RuntimeError(f"frozen training input SHA256 changed: {path}")


def validate_resolved_runner(runner, ordered, run_dir):
    if Path(runner.log_dir).resolve() != Path(run_dir).resolve():
        raise RuntimeError("resolved training run directory differs from preflight")
    cfg = runner.env.unwrapped.cfg
    if (runner.cfg.get("seed") != 42 or cfg.seed != 42 or str(runner.device) != "cuda:0"
            or str(cfg.sim.device) != "cuda:0" or runner.is_distributed):
        raise RuntimeError("resolved seed/device must be seed 42 and single cuda:0")
    command = runner.env.unwrapped.command_manager.get_term("motion")
    if list(command.motion_names_train) != ordered or list(command.motion_names) != ordered:
        raise RuntimeError("actual loaded origin order differs from frozen 256-origin index")
    if command.is_evaluating or command.has_test_set or runner.eval_during_training:
        raise RuntimeError("training must not load or evaluate heldout motions")
    if getattr(runner, "_pending_motion_cohort", None) is not None:
        raise RuntimeError("paired arms must not resume a previous motion cohort stream")


def validate_numeric_state(runner):
    import torch
    for name, parameter in runner.alg.policy.named_parameters():
        if not bool(torch.isfinite(parameter.detach()).all()):
            raise RuntimeError(f"non-finite policy parameter: {name}")
    for label in ("actor", "critic"):
        optimizer = getattr(runner.alg, f"{label}_optimizer")
        if not optimizer.state:
            raise RuntimeError(f"{label} optimizer state is empty")
        for state in optimizer.state.values():
            if not {"step", "exp_avg", "exp_avg_sq"} <= set(state):
                raise RuntimeError(f"{label} Adam statistics are incomplete")
            for key, value in state.items():
                if not bool(torch.isfinite(torch.as_tensor(value)).all()):
                    raise RuntimeError(f"non-finite {label} Adam statistic: {key}")
            step = float(state["step"])
            if step < 0 or not step.is_integer() or bool((state["exp_avg_sq"] < 0).any()):
                raise RuntimeError(f"invalid {label} Adam step or second moment")


def validate_coverage(evidence, ordered, updates):
    """Recompute the seed-42 stream independently of runner sampler counters."""
    import torch
    expectations(updates)
    if (evidence["completed_updates"] != updates
            or evidence["completed_environment_steps"] != updates * 8192
            or len(evidence["updates"]) != updates):
        raise RuntimeError("incorrect completed rollout/update count")
    generator = torch.Generator(device="cpu").manual_seed(42)
    cumulative = Counter()
    permutation = None
    for i, row in enumerate(evidence["updates"]):
        cohort = i // 5
        if i % 10 == 0:
            permutation = torch.randperm(256, generator=generator).tolist()
        names = [ordered[j] for j in permutation[(cohort % 2) * 128:(cohort % 2 + 1) * 128]]
        wanted = {name: 64 for name in names}
        if (row["ordinal"] != i + 1 or row["completed_environment_steps"] != (i + 1) * 8192
                or row["motion_steps"] != wanted):
            raise RuntimeError(f"update {i + 1} actual motion steps violate the seed-42 cohort stream")
        cumulative.update(wanted)
        snapshot = row["motion_cohort_sampler"]
        wanted_snapshot = {"num_motions": 256, "num_envs": 128, "seed": 42,
                           "cohorts": cohort + 1, "epochs": cohort // 2 + 1,
                           "completed_epochs": (cohort + 1) // 2,
                           "unique_ever": min((cohort + 1) * 128, 256),
                           "coverage": min((cohort + 1) / 2, 1),
                           "current_epoch_cursor": 128 if cohort % 2 == 0 else 0}
        if snapshot != wanted_snapshot:
            raise RuntimeError(f"update {i + 1} sampler counters disagree with actual cohort usage")
    if dict(cumulative) != evidence["motion_steps"] or set(cumulative) != set(ordered):
        raise RuntimeError("the paired run did not actually sample all 256 training origins")
    if updates == 100 and set(cumulative.values()) != {3200}:
        raise RuntimeError("100 updates must provide exactly 3200 steps per origin")
    return {"real_cohort_boundary_verified": True, "seeded_cohort_stream_verified": True,
            "actual_unique_motions": 256, "actual_all_training_motions_sampled": True,
            "minimum_steps_per_origin": min(cumulative.values()),
            "maximum_steps_per_origin": max(cumulative.values())}


def observe_learning(runner, original_learn, expected, ordered, run_dir, callback=None):
    validate_resolved_runner(runner, ordered, run_dir)
    validate_numeric_state(runner)
    original_update = runner.alg.update
    losses = []

    def strict_update(*args, **kwargs):
        loss = original_update(*args, **kwargs)
        if not isinstance(loss, dict) or not {"value_function", "surrogate", "entropy"} <= set(loss):
            raise RuntimeError("PPO update omitted numeric loss diagnostics")
        finite_loss = legacy._diagnostics(loss)
        diagnostics = legacy._diagnostics(runner.alg.update_diagnostics)
        required = {"minibatches", "policy_kl_mean", "policy_kl_max", "policy_kl_first", "policy_kl_last",
                    "ratio_clip_fraction_mean", "ratio_clip_fraction_max"}
        required |= {f"{part}_lr_{stat}" for part in ("actor", "critic") for stat in ("first", "last", "min", "max")}
        if not required <= set(diagnostics):
            raise RuntimeError("PPO update omitted required KL, clipping, or Adam LR diagnostics")
        if any(not 0 <= diagnostics[f"ratio_clip_fraction_{stat}"] <= 1 for stat in ("mean", "max")):
            raise RuntimeError("PPO clipping fraction is outside [0, 1]")
        losses.append(finite_loss)
        return loss

    runner.alg.update = strict_update
    try:
        evidence = legacy._observe_learning(runner, original_learn, expected, set(ordered), callback)
    finally:
        runner.alg.update = original_update
    validate_numeric_state(runner)
    evidence["coverage"] = validate_coverage(evidence, ordered, expected.updates)
    evidence["finite_loss_diagnostics"] = losses
    evidence["resolved"].update(seed=42, device="cuda:0", actual_origin_order_verified=True)
    return evidence


def durable_report(path, payload):
    # Kit fast shutdown may never return to this process's Python main/finally.
    with Path(path).open("x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(Path(path).parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _runner_class():
    from my_rsl_rl.runners.on_policy_runner import OnPolicyRunner
    return OnPolicyRunner


def _run(args):
    if Path(sys.executable).absolute() != ISAAC_PYTHON:
        raise RuntimeError(f"use the existing Isaac interpreter: {ISAAC_PYTHON}")
    unit = bounded_unit()
    run_dir = run_directory(args.run_name)
    if Path.cwd().resolve() != TRACK.resolve():
        raise ValueError(f"training must run with working directory {TRACK}")
    original_path = sys.path[:]
    sys.path[:0] = [str(ROOT), str(ENTRY.parent)]
    try:
        manifest, frozen = prepare_dataset(args.dataset)
        argv = train_argv(manifest, args.arm, args.run_name, args.updates)
        expected = expectations(args.updates)
        runner_class = _runner_class()
        return _execute(args, manifest, frozen, argv, expected, run_dir, unit, runner_class)
    finally:
        sys.path[:] = original_path


def _execute(args, manifest, frozen, argv, expected, run_dir, unit, runner_class):
    original_learn, original_load = runner_class.learn, runner_class.load
    original_argv = sys.argv
    holder, loaded = {}, []

    def observed_load(runner, path, load_optimizer=True, map_location=None):
        if (loaded or load_optimizer is not True or Path(path).resolve() != DERIVED.resolve()
                or digest(path) != DERIVED_SHA256):
            raise RuntimeError("must load exactly the approved checkpoint with both Adam states once")
        result = original_load(runner, path, load_optimizer=load_optimizer, map_location=map_location)
        loaded.append(runner)
        return result

    def observed_learn(runner, *positional, **keywords):
        if (positional or keywords != {"num_learning_iterations": expected.updates, "init_at_random_ep_len": True}
                or len(loaded) != 1 or loaded[0] is not runner or holder):
            raise RuntimeError("unexpected training learn/load invocation")

        def progress(payload):
            # COMPLETE is published only after the new strict coverage/hash checks.
            if payload["result"] == "COMPLETE":
                return
            legacy._atomic_json(args.progress, {"schema": SCHEMA, "arm": args.arm, **payload})

        evidence = observe_learning(runner, original_learn, expected,
                                    manifest["ordered_origins"], run_dir, progress)
        verify_frozen(frozen)
        payload = {"schema": SCHEMA, "result": "PASS", "purpose": "data A/B execution evidence, not policy quality",
                   "automatic_promotion": False, "arm": args.arm, "owned_unit": unit,
                   "argv": argv, "run_name": args.run_name, "run_dir": str(run_dir),
                   "dataset_manifest": str(args.dataset.resolve()),
                   "dataset_manifest_sha256": frozen[str(args.dataset.resolve())],
                   "train_index": manifest["arms"][args.arm]["index"],
                   "train_index_sha256": manifest["arms"][args.arm]["sha256"],
                   "source_checkpoint": str(DERIVED), "source_checkpoint_sha256": DERIVED_SHA256,
                   "frozen_inputs": frozen, **evidence}
        durable_report(args.report, payload)
        holder["evidence"] = payload
        legacy._atomic_json(args.progress, {"schema": SCHEMA, "result": "COMPLETE", "arm": args.arm,
            "completed_updates": evidence["completed_updates"],
            "completed_environment_steps": evidence["completed_environment_steps"],
            "unique_motion": 256, "motion_steps": evidence["motion_steps"],
            "dataset_steps": evidence["dataset_steps"], "mode_steps": evidence["mode_steps"],
            "coverage": evidence["coverage"], "report": str(args.report)})
        print("BFM_UMR_BEHAVIOR_TRAINING_PROBE PASS", args.report, flush=True)

    runner_class.learn, runner_class.load = observed_learn, observed_load
    sys.argv = [str(ENTRY), *argv]
    try:
        runpy.run_path(str(ENTRY), run_name="__main__")
    finally:
        runner_class.learn, runner_class.load = original_learn, original_load
        sys.argv = original_argv
    if "evidence" not in holder:
        raise RuntimeError("entry point returned without observed training")
    return holder["evidence"]


def _parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--arm", choices=("a", "b"), required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--updates", type=int, choices=(6, 100), required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse(argv)
    args.report, args.progress = new_file(args.report), new_file(args.progress)
    if args.report == args.progress:
        raise ValueError("report and progress must be different files")
    new_file(args.progress.with_name(f".{args.progress.name}.tmp"))
    previous_handler = signal.getsignal(signal.SIGTERM)

    def terminate(signum, frame):
        raise InterruptedError("owned training service requested termination")

    signal.signal(signal.SIGTERM, terminate)
    try:
        _run(args)
    except BaseException as error:
        payload = {"schema": SCHEMA, "result": "FAIL", "arm": args.arm,
                   "automatic_promotion": False, "error_type": type(error).__name__,
                   "error": str(error), "traceback": traceback.format_exc()}
        if not args.report.exists():
            durable_report(args.report, payload)
        prior = {}
        if args.progress.is_file() and not args.progress.is_symlink():
            try:
                prior = json.loads(args.progress.read_text())
            except (OSError, ValueError):
                pass
        prior.update({"result": "FAIL", "error_type": type(error).__name__, "error": str(error)})
        legacy._atomic_json(args.progress, prior)
        print(payload["traceback"], file=sys.stderr, flush=True)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


if __name__ == "__main__":
    main()
