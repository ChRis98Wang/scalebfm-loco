"""CPU-only tests for the bounded ScaleBFM training observer."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

PATH = Path(__file__).resolve().parents[1] / "scripts/pretrain/rsl_rl/training_probe.py"


def load_module():
    spec = importlib.util.spec_from_file_location("training_probe_tested", PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = torch.nn.Parameter(torch.tensor([1.0]))
        self.critic = torch.nn.Parameter(torch.tensor([2.0]))
        self.actor_parameters, self.critic_parameters = [self.actor], [self.critic]


class Optimizer:
    def __init__(self, parameter):
        self.param_groups = [{"params": [parameter], "lr": 1e-5}]
        self.state = {parameter: {"step": torch.tensor(10.0)}}


class Writer:
    flushed = closed = False
    def flush(self): self.flushed = True
    def close(self): self.closed = True


def fake(module, schedule="fixed", fail=False, sampling_strategy="legacy", deferred_sampler=False):
    table = torch.zeros((8, 14))
    for i in range(8): table[i, :i + 1] = 1
    names = ["KIT/a", "ACCAD/b", "KIT/c", "BMLmovi/d"]
    command = SimpleNamespace(is_evaluating=False, has_test_set=False, motion_names_train=names,
        motion_names=names, motion_ids=torch.arange(4), _mode_table=table, _mode=table[:4].clone(),
        cfg=SimpleNamespace(mode_candidates={name: [] for name in module.MODE_NAMES}))
    policy = Policy()
    ao, co = Optimizer(policy.actor), Optimizer(policy.critic)
    alg = SimpleNamespace(schedule=schedule, entropy_coef=.001, num_learning_epochs=2, num_mini_batches=2,
        actor_learning_rate=1e-5, critic_learning_rate=1e-5, actor_optimizer=ao, critic_optimizer=co,
        policy=policy, update_diagnostics={})
    alg.act = lambda obs: obs
    alg.process_env_step = lambda *a, **k: None
    calls = 0
    def update():
        nonlocal calls
        calls += 1
        if fail and calls == 2: raise RuntimeError("injected update failure")
        policy.actor.data.add_(.01); policy.critic.data.add_(.01)
        ao.state[policy.actor]["step"] += 4; co.state[policy.critic]["step"] += 4
        alg.update_diagnostics = {"minibatches": 4, "policy_kl_mean": .01,
            "actor_lr_first": 1e-5, "actor_lr_last": 1e-5,
            "critic_lr_first": 1e-5, "critic_lr_last": 1e-5}
        if schedule == "adaptive": alg.update_diagnostics["scheduler_kl_mean"] = .01
        return {"surrogate": 0.0}
    alg.update = update
    manager = SimpleNamespace(get_term=lambda _: command)
    runner = SimpleNamespace(env=SimpleNamespace(num_envs=4, unwrapped=SimpleNamespace(command_manager=manager)),
        num_steps_per_env=3, current_learning_iteration=20, eval_during_training=False, save_interval=1,
        motion_sampling_strategy=sampling_strategy,
        motion_resample_interval=1 if sampling_strategy == "coverage" else 0,
        _motion_cohort_sampler=(None if deferred_sampler else
            (SimpleNamespace(snapshot=lambda: {"cohorts": calls}) if sampling_strategy == "coverage" else None)),
        alg=alg, writer=Writer())
    def learn(given, *, num_learning_iterations, init_at_random_ep_len):
        assert given is runner and init_at_random_ep_len
        if sampling_strategy == "coverage" and deferred_sampler:
            given._motion_cohort_sampler = SimpleNamespace(snapshot=lambda: {"cohorts": calls})
        for update_i in range(num_learning_iterations):
            for step in range(3):
                command.motion_ids = torch.tensor([(step + i) % 4 for i in range(4)])
                command._mode = table[[(step * 4 + i) % 8 for i in range(4)]]
                action = alg.act(torch.tensor(0.0)); alg.process_env_step(action)
            alg.update(); runner.current_learning_iteration = 20 + update_i
    expected = module.ProbeExpectations(schedule, 2, 1e-5, num_envs=4, steps_per_env=3,
        initial_iteration=20, epochs=2, minibatches=2, sampling_strategy=sampling_strategy,
        resample_interval=1 if sampling_strategy == "coverage" else 0)
    return runner, learn, expected, set(names)


@pytest.mark.parametrize("schedule", ["fixed", "adaptive"])
def test_counts_updates_steps_kit_modes_actor_and_restores(schedule):
    module = load_module(); runner, learn, expected, names = fake(module, schedule)
    originals = runner.alg.act, runner.alg.process_env_step, runner.alg.update
    result = module._observe_learning(runner, learn, expected, names)
    assert result["completed_updates"] == 2 and result["completed_environment_steps"] == 24
    assert result["dataset_steps"]["KIT"] > 0 and result["unique_kit_motions"] == 2
    assert all(result["mode_steps"][name] > 0 for name in module.MODE_NAMES)
    assert result["changed_actor_parameters"] == 1 and result["adam_step_increment"] == 8
    assert (runner.alg.act, runner.alg.process_env_step, runner.alg.update) == originals
    assert runner.writer.flushed and runner.writer.closed
    json.dumps(result, allow_nan=False)


def test_failed_update_is_not_published_and_hooks_restore():
    module = load_module(); runner, learn, expected, names = fake(module, fail=True)
    originals = runner.alg.act, runner.alg.process_env_step, runner.alg.update
    with pytest.raises(RuntimeError, match="injected"):
        module._observe_learning(runner, learn, expected, names)
    assert (runner.alg.act, runner.alg.process_env_step, runner.alg.update) == originals
    assert runner.writer.closed


def test_coverage_progress_is_only_published_after_successful_updates_and_complete():
    module = load_module(); runner, learn, expected, names = fake(module, sampling_strategy="coverage")
    published = []
    result = module._observe_learning(runner, learn, expected, names, lambda value: published.append(value))
    assert [item["result"] for item in published] == ["RUNNING", "RUNNING", "COMPLETE"]
    assert [item["completed_updates"] for item in published] == [1, 2, 2]
    assert [item["completed_environment_steps"] for item in published] == [12, 24, 24]
    assert published[-1]["unique_motion"] == len(result["motion_steps"])
    assert published[-1]["motion_cohort_sampler"] == {"cohorts": 2}


def test_coverage_sampler_may_be_prepared_inside_original_learn_before_first_act():
    module = load_module()
    runner, learn, expected, names = fake(module, sampling_strategy="coverage", deferred_sampler=True)
    assert runner._motion_cohort_sampler is None
    result = module._observe_learning(runner, learn, expected, names)
    assert result["completed_updates"] == 2
    assert result["updates"][0]["motion_cohort_sampler"] == {"cohorts": 1}


def test_failed_update_does_not_publish_that_update_progress():
    module = load_module(); runner, learn, expected, names = fake(module, fail=True)
    published = []
    with pytest.raises(RuntimeError, match="injected"):
        module._observe_learning(runner, learn, expected, names, lambda value: published.append(value))
    assert len(published) == 1 and published[0]["completed_updates"] == 1
    assert published[0]["result"] == "RUNNING"


def test_failed_process_does_not_reach_a_completed_update():
    module = load_module(); runner, _, expected, names = fake(module)
    runner.alg.process_env_step = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("process failed"))
    def learn(given, **kwargs):
        given.alg.act(torch.tensor(0.0)); given.alg.process_env_step(None)
    with pytest.raises(RuntimeError, match="process failed"):
        module._observe_learning(runner, learn, expected, names)


def valid_argv(module, run="new-run"):
    return ["--task", "G1-BFM-Transformer-Tracking", "--motion_file", str(module.TRAIN_INDEX),
        "--run_name", run, "--logger", "tensorboard", "--num_envs", "128", "--max_iterations", "5",
        "--seed", "42", "--resume", "True", "--load_run", "official_lr1e5_derived_20260907",
        "--checkpoint", "model_22200.pt", "--device", "cuda:0", "--headless",
        "agent.algorithm.entropy_coef=0.001", "agent.algorithm.schedule=fixed",
        "agent.eval_during_training=False", "agent.save_interval=1"]


def test_cli_separator_and_bad_expectations():
    module = load_module()
    parsed = module._parse(["--report", "/tmp/x", "--expected_schedule", "fixed",
        "--expected_updates", "5", "--expected_lr", "1e-5", "--", "--task", "x"])
    assert parsed.train_args == ["--task", "x"]
    assert parsed.expected.save_interval == 1 and parsed.expected.sampling_strategy == "legacy"
    assert parsed.progress is None
    with pytest.raises(SystemExit):
        module._parse(["--report", "/tmp/x", "--expected_schedule", "fixed",
            "--expected_updates", "0", "--expected_lr", "1e-5", "--", "--task", "x"])


def test_preflight_rejects_bad_task_heldout_and_existing_run(monkeypatch, tmp_path):
    module = load_module(); expected = module.ProbeExpectations("fixed", 5, 1e-5)
    argv = valid_argv(module); argv[1] = "wrong"
    with pytest.raises(ValueError, match="--task"): module._validate_train_argv(argv, expected)
    argv = valid_argv(module) + ["--test_motion_file", "heldout.yaml"]
    with pytest.raises(ValueError, match="heldout"): module._validate_train_argv(argv, expected)
    monkeypatch.setattr(module, "EXPERIMENT_DIR", tmp_path); (tmp_path / "occupied").mkdir()
    with pytest.raises(FileExistsError): module._validate_train_argv(valid_argv(module, "occupied"), expected)


def test_coverage_preflight_requires_explicit_hydra_overrides(monkeypatch, tmp_path):
    module = load_module(); monkeypatch.setattr(module, "EXPERIMENT_DIR", tmp_path)
    expected = module.ProbeExpectations("fixed", 5, 1e-5, save_interval=50,
                                        sampling_strategy="coverage", resample_interval=5)
    argv = valid_argv(module)
    argv[argv.index("agent.save_interval=1")] = "agent.save_interval=50"
    with pytest.raises(ValueError, match="Hydra"):
        module._validate_train_argv(argv, expected)
    argv.extend(["agent.motion_sampling_strategy=coverage", "agent.motion_resample_interval=5"])
    assert module._validate_train_argv(argv, expected)[0] == "new-run"


def test_atomic_progress_replaces_complete_json(tmp_path):
    module = load_module(); path = tmp_path / "progress.json"
    module._atomic_json(path, {"result": "RUNNING", "completed_updates": 1})
    module._atomic_json(path, {"result": "COMPLETE", "completed_updates": 2})
    assert json.loads(path.read_text()) == {"result": "COMPLETE", "completed_updates": 2}
    assert not (tmp_path / ".progress.json.tmp").exists()


def test_main_writes_exclusive_failure_report(monkeypatch, tmp_path):
    module = load_module(); report = tmp_path / "failure.json"
    monkeypatch.setattr(module, "_run", lambda args, path: (_ for _ in ()).throw(RuntimeError("boom")))
    argv = ["--report", str(report), "--expected_schedule", "fixed", "--expected_updates", "5",
            "--expected_lr", "1e-5", "--", "--task", "x"]
    with pytest.raises(RuntimeError, match="boom"): module.main(argv)
    assert json.loads(report.read_text())["result"] == "FAIL"
    with pytest.raises(FileExistsError): module.main(argv)


def test_main_rejects_same_report_and_progress_path(tmp_path):
    module = load_module(); path = tmp_path / "same.json"
    argv = ["--report", str(path), "--progress", str(path), "--expected_schedule", "fixed",
            "--expected_updates", "5", "--expected_lr", "1e-5", "--", "--task", "x"]
    with pytest.raises(ValueError, match="different"):
        module.main(argv)


def test_main_marks_existing_progress_failed_on_runtime_error(monkeypatch, tmp_path):
    module = load_module(); report = tmp_path / "report.json"; progress = tmp_path / "progress.json"
    def fail(args, path):
        module._atomic_json(args.progress, {"result": "RUNNING", "completed_updates": 1})
        raise RuntimeError("late failure")
    monkeypatch.setattr(module, "_run", fail)
    argv = ["--report", str(report), "--progress", str(progress), "--expected_schedule", "fixed",
            "--expected_updates", "5", "--expected_lr", "1e-5", "--", "--task", "x"]
    with pytest.raises(RuntimeError, match="late failure"):
        module.main(argv)
    saved = json.loads(progress.read_text())
    assert saved["result"] == "FAIL" and saved["completed_updates"] == 1
