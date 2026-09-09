"""CPU-only tests: no Kit, training entry point, simulator, or real checkpoint load."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("umr_behavior_probe_tested", ROOT / "scripts/umr_behavior_training_probe.py")
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)
COHORT_SPEC = importlib.util.spec_from_file_location("umr_probe_test_cohort", ROOT / "ScaleTrack/source/my_rsl_rl/my_rsl_rl/utils/motion_cohort.py")
cohorts = importlib.util.module_from_spec(COHORT_SPEC)
COHORT_SPEC.loader.exec_module(cohorts)


@pytest.fixture(autouse=True)
def cpu_threads():
    prior = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(prior)


def names():
    return [f"{'KIT' if i % 3 == 0 else 'ACCAD'}/motion_{i:03d}" for i in range(256)]


def dataset_fixture(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"test-only checkpoint, never torch loaded")
    monkeypatch.setattr(probe, "DERIVED", checkpoint)
    monkeypatch.setattr(probe, "DERIVED_SHA256", probe.digest(checkpoint))
    ordered = names()
    arms = {}
    for arm in ("a", "b"):
        index = tmp_path / f"{arm}.yaml"
        index.write_text(yaml.safe_dump({name: [f"/{arm}/{i}.npz", 1 / 256] for i, name in enumerate(ordered)}, sort_keys=False))
        arms[arm] = dict(index=str(index), sha256=probe.digest(index))
    manifest = dict(schema=probe.DATASET_SCHEMA, ordered_origins=ordered,
                    target_origins=ordered[:17], replay_origins=ordered[17:], arms=arms,
                    checkpoint=dict(path=str(checkpoint), sha256=probe.digest(checkpoint)),
                    sampling=dict(strategy="coverage", weights="uniform", num_envs=128,
                                  resample_interval=5, weight_per_origin=1 / 256))
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path, manifest


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = torch.nn.Parameter(torch.tensor([1.0]))
        self.critic = torch.nn.Parameter(torch.tensor([2.0]))
        self.actor_parameters = [self.actor]
        self.critic_parameters = [self.critic]


class Optimizer:
    def __init__(self, parameter):
        self.param_groups = [{"params": [parameter], "lr": 1e-5}]
        self.state = {parameter: dict(step=torch.tensor(10.0), exp_avg=torch.zeros(1), exp_avg_sq=torch.ones(1))}


class FakeRunner:
    def __init__(self, run_dir, fail=None):
        self.log_dir = str(run_dir)
        self.cfg, self.device, self.is_distributed = {"seed": 42}, "cuda:0", False
        self.num_steps_per_env, self.current_learning_iteration = 64, 22199
        self.eval_during_training, self.save_interval = False, 50
        self.motion_sampling_strategy, self.motion_resample_interval = "coverage", 5
        self._pending_motion_cohort = self._motion_cohort_sampler = None
        table = torch.eye(14)[:8]
        command = SimpleNamespace(is_evaluating=False, has_test_set=False,
            motion_names_train=names(), motion_names=names(), motion_ids=torch.arange(128),
            _mode_table=table, _mode=table[torch.arange(128) % 8],
            cfg=SimpleNamespace(mode_candidates=dict.fromkeys(probe.legacy.MODE_NAMES)))
        self.command = command
        self.env = SimpleNamespace(num_envs=128, unwrapped=SimpleNamespace(
            cfg=SimpleNamespace(seed=42, sim=SimpleNamespace(device="cuda:0")),
            command_manager=SimpleNamespace(get_term=lambda _: command)))
        self.writer = SimpleNamespace(flush=lambda: None, close=lambda: None)
        policy = Policy()
        self.alg = SimpleNamespace(policy=policy, schedule="fixed", entropy_coef=.001,
            num_learning_epochs=2, num_mini_batches=32, actor_learning_rate=1e-5,
            critic_learning_rate=1e-5, actor_optimizer=Optimizer(policy.actor),
            critic_optimizer=Optimizer(policy.critic), act=lambda obs: obs,
            process_env_step=lambda *a: None, update_diagnostics={})
        self.update_count = 0

        def update():
            self.update_count += 1
            if fail == "exception" and self.update_count == 2:
                raise RuntimeError("fake update failure")
            self.alg.policy.actor.data.add_(.001)
            self.alg.policy.critic.data.add_(.001)
            for optimizer in (self.alg.actor_optimizer, self.alg.critic_optimizer):
                for state in optimizer.state.values():
                    state["step"] += 64
            self.alg.update_diagnostics = {"minibatches": 64,
                "policy_kl_mean": .01, "policy_kl_max": .02, "policy_kl_first": .001, "policy_kl_last": .015,
                "ratio_clip_fraction_mean": .1, "ratio_clip_fraction_max": .2,
                **{f"{part}_lr_{stat}": 1e-5 for part in ("actor", "critic") for stat in ("first", "last", "min", "max")}}
            if fail == "nan_diagnostic" and self.update_count == 2:
                self.alg.update_diagnostics["policy_kl_max"] = float("nan")
            if fail == "missing_diagnostic":
                del self.alg.update_diagnostics["critic_lr_min"]
            return {"surrogate": float("nan") if fail == "nan_loss" else .01,
                    "value_function": .02, "entropy": .03}

        self.alg.update = update

    def load(self, path, load_optimizer=True, map_location=None):
        assert load_optimizer is True

    def learn(self, *, num_learning_iterations, init_at_random_ep_len):
        assert init_at_random_ep_len is True
        self._motion_cohort_sampler = cohorts.UniformMotionCohortSampler(256, 128, 42)
        for i in range(num_learning_iterations):
            if i % 5 == 0:
                self.command.motion_ids = self._motion_cohort_sampler.next_ids()
            for _ in range(64):
                self.alg.process_env_step(self.alg.act(None))
            self.alg.update()
            self.current_learning_iteration = 22199 + i


def arguments(tmp_path, manifest):
    return SimpleNamespace(dataset=manifest, arm="a", run_name="umr_smoke_a", updates=6,
                           report=tmp_path / "report.json", progress=tmp_path / "progress.json")


def test_leaf_import_does_not_import_kit_and_pins_old_probe(monkeypatch, tmp_path):
    import os
    import subprocess

    # Other collected tests legitimately import or stub Isaac modules. Test this
    # leaf's own import effects in a fresh, bounded CPU-only interpreter instead
    # of relying on the test suite's shared sys.modules or collection order.
    script = """
import importlib.util
import sys

def kit_modules():
    return {name for name in sys.modules if name.startswith(("isaaclab.app", "omni.kit", "isaacsim"))}

before = kit_modules()
assert not before, before
spec = importlib.util.spec_from_file_location("isolated_umr_behavior_probe", sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
after = kit_modules()
assert after == before, sorted(after - before)
print("ISOLATED_IMPORT_NO_KIT")
"""
    result = subprocess.run([sys.executable, "-c", script, str(SPEC.origin)],
                            cwd=tmp_path, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
                            capture_output=True, text=True, timeout=15, check=True)
    assert result.stdout.strip() == "ISOLATED_IMPORT_NO_KIT"
    assert probe.digest(probe.OLD_PROBE) == probe.OLD_PROBE_SHA256
    altered = tmp_path / "changed.py"
    altered.write_text("raise AssertionError('must not execute untrusted code')")
    monkeypatch.setattr(probe, "OLD_PROBE", altered)
    with pytest.raises(RuntimeError, match="SHA256"):
        probe._load_old_probe()


@pytest.mark.parametrize("updates,iteration,steps", [(6, 22204, 49152), (100, 22298, 819200)])
def test_real_observer_fake_rollouts_complete_all_origins_and_restore(updates, iteration, steps, tmp_path):
    runner = FakeRunner(tmp_path / "run")
    originals = runner.alg.act, runner.alg.process_env_step, runner.alg.update
    callbacks = []
    result = probe.observe_learning(runner, FakeRunner.learn, probe.expectations(updates), names(),
                                    tmp_path / "run", callbacks.append)
    assert result["completed_updates"] == updates and result["completed_environment_steps"] == steps
    assert result["resolved"]["final_iteration"] == iteration
    assert result["adam_step_increment"] == updates * 64
    assert result["coverage"]["actual_unique_motions"] == 256
    assert result["coverage"]["seeded_cohort_stream_verified"]
    assert result["coverage"]["minimum_steps_per_origin"] == (64 if updates == 6 else 3200)
    assert result["coverage"]["maximum_steps_per_origin"] == (320 if updates == 6 else 3200)
    assert all(value > 0 for value in result["mode_steps"].values())
    assert len(result["finite_loss_diagnostics"]) == updates
    assert (runner.alg.act, runner.alg.process_env_step, runner.alg.update) == originals
    assert callbacks[-1]["result"] == "COMPLETE"
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("failure,message,committed", [
    ("exception", "fake update", 1), ("nan_loss", "non-finite", 0),
    ("nan_diagnostic", "non-finite", 1), ("missing_diagnostic", "omitted", 0),
])
def test_failure_never_commits_bad_update_and_restores_hooks(failure, message, committed, tmp_path):
    runner = FakeRunner(tmp_path / "run", fail=failure)
    originals = runner.alg.act, runner.alg.process_env_step, runner.alg.update
    callbacks = []
    with pytest.raises(RuntimeError, match=message):
        probe.observe_learning(runner, FakeRunner.learn, probe.expectations(6), names(), tmp_path / "run", callbacks.append)
    assert len(callbacks) == committed
    assert (runner.alg.act, runner.alg.process_env_step, runner.alg.update) == originals


@pytest.mark.parametrize("change", ["seed", "env_seed", "device", "order", "heldout", "pending", "distributed", "logdir"])
def test_resolved_runtime_rejects_contract_changes(change, tmp_path):
    runner = FakeRunner(tmp_path / "run")
    if change == "seed": runner.cfg["seed"] = 7
    if change == "env_seed": runner.env.unwrapped.cfg.seed = 7
    if change == "device": runner.device = "cpu"
    if change == "order": runner.command.motion_names.reverse()
    if change == "heldout": runner.command.has_test_set = True
    if change == "pending": runner._pending_motion_cohort = {}
    if change == "distributed": runner.is_distributed = True
    if change == "logdir": runner.log_dir = str(tmp_path / "other")
    with pytest.raises(RuntimeError):
        probe.validate_resolved_runner(runner, names(), tmp_path / "run")


def test_numeric_state_rejects_nonfinite_critic_and_invalid_adam(tmp_path):
    runner = FakeRunner(tmp_path)
    runner.alg.policy.critic.data.fill_(float("inf"))
    with pytest.raises(RuntimeError, match="policy parameter"):
        probe.validate_numeric_state(runner)
    runner.alg.policy.critic.data.fill_(2)
    state = next(iter(runner.alg.actor_optimizer.state.values()))
    state["exp_avg_sq"].fill_(-1)
    with pytest.raises(RuntimeError, match="second moment"):
        probe.validate_numeric_state(runner)
    state["exp_avg_sq"].fill_(1)
    state["exp_avg"].fill_(float("nan"))
    with pytest.raises(RuntimeError, match="Adam statistic"):
        probe.validate_numeric_state(runner)


def test_coverage_rejects_wrong_seed_duplicate_origin_and_forged_snapshot(tmp_path):
    runner = FakeRunner(tmp_path)
    evidence = probe.observe_learning(runner, FakeRunner.learn, probe.expectations(6), names(), tmp_path)
    wrong_order = list(reversed(names()))
    with pytest.raises(RuntimeError, match="cohort stream"):
        probe.validate_coverage(evidence, wrong_order, 6)
    forged = copy.deepcopy(evidence)
    forged["updates"][0]["motion_cohort_sampler"]["seed"] = 0
    with pytest.raises(RuntimeError, match="sampler counters"):
        probe.validate_coverage(forged, names(), 6)
    forged = copy.deepcopy(evidence)
    del forged["motion_steps"][names()[0]]
    with pytest.raises(RuntimeError, match="all 256"):
        probe.validate_coverage(forged, names(), 6)


def test_manifest_freezes_both_arm_indexes_and_checkpoint(tmp_path, monkeypatch):
    path, manifest = dataset_fixture(tmp_path, monkeypatch)
    verified, frozen = probe.prepare_dataset(path, validator=lambda p: manifest)
    assert verified == manifest and str(path) in frozen
    assert all(item["index"] in frozen for item in manifest["arms"].values())
    Path(manifest["arms"]["b"]["index"]).write_text("changed")
    with pytest.raises(RuntimeError, match="SHA256 changed"):
        probe.verify_frozen(frozen)


@pytest.mark.parametrize("mutation", ["order", "targets", "sampling", "checkpoint", "index_sha", "manifest_race"])
def test_manifest_preflight_rejects_ineligible_or_changed_inputs(mutation, tmp_path, monkeypatch):
    path, manifest = dataset_fixture(tmp_path, monkeypatch)
    if mutation == "order": manifest["ordered_origins"].reverse()
    if mutation == "targets": manifest["target_origins"][0] = manifest["replay_origins"][0]
    if mutation == "sampling": manifest["sampling"]["resample_interval"] = 6
    if mutation == "checkpoint": manifest["checkpoint"]["sha256"] = "bad"
    if mutation == "index_sha": manifest["arms"]["b"]["sha256"] = "bad"
    def validate(given):
        if mutation == "manifest_race": given.write_text("changed during validation")
        return manifest
    with pytest.raises((ValueError, RuntimeError)):
        probe.prepare_dataset(path, validator=validate)


def test_fixed_cli_has_no_arbitrary_training_overrides(tmp_path, monkeypatch):
    path, manifest = dataset_fixture(tmp_path, monkeypatch)
    argv = probe.train_argv(manifest, "b", "trial_b", 100)
    assert argv[argv.index("--motion_file") + 1] == manifest["arms"]["b"]["index"]
    assert "--test_motion_file" not in argv and argv.count("--headless") == 1
    assert "agent.algorithm.actor_learning_rate=1e-5" in argv
    assert "agent.eval_during_training=False" in argv
    assert "agent.motion_resample_interval=5" in argv
    base = ["--dataset", str(path), "--arm", "a", "--run-name", "trial", "--updates", "6",
            "--report", str(tmp_path / "report"), "--progress", str(tmp_path / "progress")]
    assert probe._parse(base).updates == 6
    with pytest.raises(SystemExit): probe._parse(base + ["agent.seed=3"])
    with pytest.raises(ValueError): probe.expectations(5)
    with pytest.raises(ValueError): probe.expectations(True)


@pytest.mark.parametrize("invalid", ["killmode", "restart", "runtime", "memory", "tasks", "cgroup"])
def test_bounded_cgroup_required(invalid, monkeypatch):
    original_read = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda p, *a, **k: (
        "0::/user.slice/user-1000.slice/bfm-umr-behavior-test.service\n" if invalid != "cgroup" else "0::/wrong.service\n"
    ) if str(p) == "/proc/self/cgroup" else original_read(p, *a, **k))
    values = dict(KillMode="control-group", Restart="no", RuntimeMaxUSec="20min", MemoryMax="100000", TasksMax="200")
    changes = dict(killmode=("KillMode", "process"), restart=("Restart", "always"),
                   runtime=("RuntimeMaxUSec", "infinity"), memory=("MemoryMax", "0"), tasks=("TasksMax", "18446744073709551615"))
    if invalid in changes:
        key, value = changes[invalid]
        values[key] = value
    monkeypatch.setattr(probe.subprocess, "check_output", lambda *a, **k: "\n".join(f"{key}={value}" for key, value in values.items()))
    with pytest.raises(ValueError): probe.bounded_unit()


def test_outputs_are_exclusive_and_reject_symlink_ancestors(tmp_path, monkeypatch):
    existing = tmp_path / "existing"
    existing.write_text("preserve")
    with pytest.raises(FileExistsError): probe.new_file(existing)
    target = tmp_path / "target"
    target.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        probe.new_file(alias / "new.json")
    monkeypatch.setattr(probe, "RUNS", tmp_path)
    for name in ("../escape", "Bad", "a/b", "a" * 72):
        with pytest.raises(ValueError): probe.run_directory(name)


def test_execute_writes_durable_pass_before_entry_shutdown_and_restores(tmp_path, monkeypatch):
    path, manifest = dataset_fixture(tmp_path, monkeypatch)
    _, frozen = probe.prepare_dataset(path, validator=lambda p: manifest)
    args = arguments(tmp_path, path)
    run_dir = tmp_path / "run"
    runner = FakeRunner(run_dir)
    originals = FakeRunner.learn, FakeRunner.load, sys.argv
    fsync_calls = []
    real_fsync = probe.os.fsync
    monkeypatch.setattr(probe.os, "fsync", lambda fd: (fsync_calls.append(fd), real_fsync(fd))[1])
    def entry(*args_, **kwargs):
        runner.load(probe.DERIVED)
        runner.learn(num_learning_iterations=6, init_at_random_ep_len=True)
        # Represents train.py reaching env.close / SimulationApp.close.
        assert json.loads(args.report.read_text())["result"] == "PASS"
        assert json.loads(args.progress.read_text())["result"] == "COMPLETE"
        assert len(fsync_calls) >= 2
    monkeypatch.setattr(probe.runpy, "run_path", entry)
    result = probe._execute(args, manifest, frozen, [], probe.expectations(6), run_dir,
                            "bfm-umr-behavior-test.service", FakeRunner)
    assert result["arm"] == "a" and result["automatic_promotion"] is False
    assert (FakeRunner.learn, FakeRunner.load, sys.argv) == originals


@pytest.mark.parametrize("failure", ["skip_load", "wrong_load", "no_optimizer", "second_load", "update", "changed_input"])
def test_execute_failure_restores_class_hooks_and_never_writes_pass(failure, tmp_path, monkeypatch):
    path, manifest = dataset_fixture(tmp_path, monkeypatch)
    _, frozen = probe.prepare_dataset(path, validator=lambda p: manifest)
    args = arguments(tmp_path, path)
    runner = FakeRunner(tmp_path / "run", fail="exception" if failure == "update" else None)
    originals = FakeRunner.learn, FakeRunner.load, sys.argv
    def entry(*a, **k):
        if failure == "wrong_load": runner.load(tmp_path / "wrong.pt")
        if failure == "no_optimizer": runner.load(probe.DERIVED, load_optimizer=False)
        if failure != "skip_load": runner.load(probe.DERIVED)
        if failure == "second_load": runner.load(probe.DERIVED)
        if failure == "changed_input": path.write_text("changed")
        runner.learn(num_learning_iterations=6, init_at_random_ep_len=True)
    monkeypatch.setattr(probe.runpy, "run_path", entry)
    with pytest.raises(RuntimeError):
        probe._execute(args, manifest, frozen, [], probe.expectations(6), tmp_path / "run", "test", FakeRunner)
    assert not args.report.exists()
    assert (FakeRunner.learn, FakeRunner.load, sys.argv) == originals


def test_main_failure_preserves_last_successful_update_and_exclusive_report(tmp_path, monkeypatch):
    def fail(args):
        probe.legacy._atomic_json(args.progress, dict(result="RUNNING", completed_updates=1))
        raise RuntimeError("test failure")
    monkeypatch.setattr(probe, "_run", fail)
    argv = ["--dataset", str(tmp_path / "dataset"), "--arm", "b", "--run-name", "trial_b", "--updates", "6",
            "--report", str(tmp_path / "report"), "--progress", str(tmp_path / "progress")]
    with pytest.raises(RuntimeError, match="test failure"):
        probe.main(argv)
    report = json.loads((tmp_path / "report").read_text())
    progress = json.loads((tmp_path / "progress").read_text())
    assert report["result"] == "FAIL" and report["arm"] == "b"
    assert progress["result"] == "FAIL" and progress["completed_updates"] == 1
    with pytest.raises(FileExistsError): probe.main(argv)


def test_wrong_interpreter_fails_before_any_runner_import(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "ISAAC_PYTHON", Path("/not/this/python"))
    monkeypatch.setattr(probe, "_runner_class", lambda: pytest.fail("must not import runner"))
    with pytest.raises(RuntimeError, match="Isaac interpreter"):
        probe._run(arguments(tmp_path, tmp_path / "manifest"))
