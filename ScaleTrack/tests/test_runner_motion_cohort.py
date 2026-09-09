"""CPU-only contracts for opt-in training motion-cohort rotation."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
import sys
import textwrap
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source/my_rsl_rl"))
from my_rsl_rl.runners import on_policy_runner as runner_module


OnPolicyRunner = runner_module.OnPolicyRunner


class FakeCommandManager:
    def __init__(self, command):
        self.command = command

    def get_term(self, name):
        assert name == "motion"
        return self.command


class FakePolicy:
    def __init__(self):
        self.reset_calls = []
        self.weight = torch.tensor([1.0])

    def reset(self, dones=None):
        self.reset_calls.append(None if dones is None else dones.detach().clone())

    def state_dict(self):
        return {"weight": self.weight.clone()}

    def load_state_dict(self, state):
        self.weight.copy_(state["weight"])
        return True


class FakeOptimizer:
    def __init__(self, lr=1.0e-5):
        self.param_groups = [{"lr": lr}]
        self.loaded = False

    def state_dict(self):
        return {"state": {}, "param_groups": self.param_groups}

    def load_state_dict(self, state):
        self.param_groups = state["param_groups"]
        self.loaded = True


class FakeEnv:
    def __init__(self, command, num_envs=4):
        self.num_envs = num_envs
        self.device = "cpu"
        self.unwrapped = self
        self.command_manager = FakeCommandManager(command)
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1
        obs = TensorDict(
            {"policy": torch.full((self.num_envs, 1), float(self.reset_calls))},
            batch_size=[self.num_envs],
        )
        return obs, {"reset": self.reset_calls}


def make_runner(*, strategy="coverage", interval=5, distributed=False,
                evaluating=False, probabilities=None, pending=None):
    num_motions, num_envs = 10, 4
    command = SimpleNamespace(
        num_motion_train=num_motions,
        motion_names_train=[f"motion/{index}" for index in range(num_motions)],
        motion_names=[f"motion/{index}" for index in range(num_motions)],
        motion_ids=torch.full((num_envs,), -1, dtype=torch.long),
        motion_sampling_prob=(torch.ones(num_motions) if probabilities is None else probabilities.clone()),
        is_evaluating=False,
        has_test_set=False,
        randomize_next_resampling=False,
        _future_manual_cache={"stale": object()},
    )
    env = FakeEnv(command, num_envs)
    runner = OnPolicyRunner.__new__(OnPolicyRunner)
    runner.cfg = {"seed": 17}
    runner.device = "cpu"
    runner.env = env
    runner.command_name = "motion"
    runner.alg = SimpleNamespace(
        policy=FakePolicy(),
        actor_optimizer=FakeOptimizer(),
        critic_optimizer=FakeOptimizer(),
        actor_learning_rate=1.0e-5,
        critic_learning_rate=1.0e-5,
    )
    runner.is_distributed = distributed
    runner.eval_during_training = evaluating
    runner.motion_resample_interval = interval
    runner.motion_sampling_strategy = strategy
    runner._motion_cohort_sampler = None
    runner._pending_motion_cohort = pending
    return runner, command


def test_legacy_default_does_not_construct_sampler_or_reset_environment():
    runner, command = make_runner(strategy="legacy", interval=0)
    original_ids = command.motion_ids.clone()

    assert runner._prepare_motion_cohorts() is None

    assert runner._motion_cohort_sampler is None
    assert runner.env.reset_calls == 0
    assert runner.alg.policy.reset_calls == []
    assert torch.equal(command.motion_ids, original_ids)
    assert command._future_manual_cache


def test_coverage_prepare_replaces_initial_draw_with_full_safe_reset():
    runner, command = make_runner()

    obs = runner._prepare_motion_cohorts()

    assert isinstance(obs, TensorDict)
    assert obs.device == torch.device("cpu")
    assert obs["policy"].eq(1).all()
    assert runner.env.reset_calls == 1
    assert len(torch.unique(command.motion_ids)) == runner.env.num_envs
    assert command.motion_ids.min() >= 0
    assert command.motion_ids.max() < command.num_motion_train
    assert command.randomize_next_resampling is True
    assert command._future_manual_cache == {}
    assert len(runner.alg.policy.reset_calls) == 1
    assert runner.alg.policy.reset_calls[0].dtype == torch.bool
    assert runner.alg.policy.reset_calls[0].shape == (runner.env.num_envs,)
    assert runner.alg.policy.reset_calls[0].all()


def test_advance_changes_cohort_and_returns_only_fresh_reset_observation():
    runner, command = make_runner()
    first_obs = runner._prepare_motion_cohorts()
    first_ids = command.motion_ids.clone()

    second_obs = runner._advance_motion_cohort()

    assert first_obs["policy"].eq(1).all()
    assert second_obs["policy"].eq(2).all()
    assert not torch.equal(command.motion_ids, first_ids)
    assert runner.env.reset_calls == 2
    assert len(runner.alg.policy.reset_calls) == 2


def test_checkpoint_round_trip_restores_exact_next_cohort_and_catalog_contract(tmp_path):
    original, original_command = make_runner()
    original._prepare_motion_cohorts()
    first_ids = original_command.motion_ids.clone()
    original.current_learning_iteration = 23
    checkpoint = tmp_path / "model.pt"
    original.save(checkpoint, infos={"tag": "unit"})

    resumed, resumed_command = make_runner()
    assert resumed.load(checkpoint) == {"tag": "unit"}
    assert resumed.current_learning_iteration == 23
    assert resumed._motion_cohort_sampler is None
    assert resumed._pending_motion_cohort is not None
    resumed._prepare_motion_cohorts()
    resumed_ids = resumed_command.motion_ids.clone()

    original._advance_motion_cohort()
    assert not torch.equal(first_ids, resumed_ids)
    assert torch.equal(resumed_ids, original_command.motion_ids)
    assert resumed._pending_motion_cohort is None


def test_checkpoint_catalog_order_or_interval_mismatch_fails_before_reset(tmp_path):
    original, _ = make_runner()
    original._prepare_motion_cohorts()
    original.current_learning_iteration = 4
    checkpoint = tmp_path / "model.pt"
    original.save(checkpoint)

    for mismatch in ("catalog", "interval"):
        resumed, command = make_runner(interval=5)
        resumed.load(checkpoint)
        if mismatch == "catalog":
            command.motion_names_train.reverse()
            command.motion_names = list(command.motion_names_train)
        else:
            resumed.motion_resample_interval = 6
        with pytest.raises(ValueError, match="catalog|interval"):
            resumed._prepare_motion_cohorts()
        assert resumed.env.reset_calls == 0


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"strategy": "coverage", "interval": 0}, "interval"),
        ({"strategy": "legacy", "interval": 5}, "strategy|legacy|interval"),
        ({"strategy": "unknown", "interval": 5}, "strategy"),
        ({"distributed": True}, "distributed"),
        ({"evaluating": True}, "eval"),
    ],
)
def test_invalid_or_unsafe_configuration_is_rejected_before_reset(changes, match):
    runner, _ = make_runner(
        strategy=changes.get("strategy", "coverage"),
        interval=changes.get("interval", 5),
        distributed=changes.get("distributed", False),
        evaluating=changes.get("evaluating", False),
    )

    with pytest.raises((ValueError, RuntimeError), match=match):
        runner._prepare_motion_cohorts()
    assert runner.env.reset_calls == 0


@pytest.mark.parametrize(
    "probabilities",
    [
        torch.tensor([2.0] + [1.0] * 9),
        torch.tensor([float("nan")] + [1.0] * 9),
        torch.tensor([float("inf")] + [1.0] * 9),
        torch.zeros(10),
    ],
)
def test_coverage_rejects_nonuniform_or_nonfinite_sampling_weights(probabilities):
    runner, _ = make_runner(probabilities=probabilities)

    with pytest.raises((ValueError, RuntimeError), match="uniform|finite|probab|weight"):
        runner._prepare_motion_cohorts()
    assert runner.env.reset_calls == 0


def test_cohort_advance_is_structurally_after_ppo_update_not_inside_rollout():
    """Guard against moving the full reset into the action/physics loop."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(OnPolicyRunner.learn)))
    learn = tree.body[0]
    outer = next(
        node for node in ast.walk(learn)
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "it"
    )
    update = next(
        node for node in ast.walk(outer)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "update" and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "alg"
    )
    advances = [
        node for node in ast.walk(outer)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_advance_motion_cohort"
    ]
    assert advances, "learn() must rotate an enabled cohort between updates"
    assert all(node.lineno > update.lineno for node in advances)

    rollout_loops = [
        node for node in ast.walk(outer)
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "_"
    ]
    assert rollout_loops
    for rollout in rollout_loops:
        assert not any(node in advances for node in ast.walk(rollout))
