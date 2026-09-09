"""Resume must restore the LR used by adaptive PPO, not only Adam's param groups."""

from pathlib import Path
import sys

import pytest
import torch
from tensordict import TensorDict


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source/my_rsl_rl"))
from my_rsl_rl.algorithms import PPO
from my_rsl_rl.modules import ActorCritic
from my_rsl_rl.runners.on_policy_runner import OnPolicyRunner


def make_runner():
    obs = TensorDict({name: torch.zeros(2, 1) for name in
                      ("policy", "policy_task", "critic", "critic_task", "action")}, batch_size=[2])
    policy = ActorCritic(obs, {}, 1, actor_hidden_dims=[4], critic_hidden_dims=[4])
    # Exercise the real load boundary without constructing a simulator or rollout.
    runner = OnPolicyRunner.__new__(OnPolicyRunner)
    runner.alg = PPO(policy, actor_learning_rate=2e-5, critic_learning_rate=1e-3, device="cpu")
    runner.current_learning_iteration = 0
    return runner


def saved_checkpoint(tmp_path, runner):
    actor = runner.alg.actor_optimizer.state_dict()
    critic = runner.alg.critic_optimizer.state_dict()
    actor["param_groups"][0]["lr"] = 1e-5
    critic["param_groups"][0]["lr"] = 3e-5
    path = tmp_path / "checkpoint.pt"
    torch.save({"model_state_dict": runner.alg.policy.state_dict(),
                "actor_optimizer_state_dict": actor, "critic_optimizer_state_dict": critic,
                "iter": 123, "infos": {"source": "test"}}, path)
    return path


def test_resume_restores_both_scheduler_scalars_from_saved_optimizers(tmp_path):
    runner = make_runner()
    path = saved_checkpoint(tmp_path, runner)
    assert runner.load(path, map_location="cpu") == {"source": "test"}
    assert runner.current_learning_iteration == 123
    assert runner.alg.actor_optimizer.param_groups[0]["lr"] == pytest.approx(1e-5)
    assert runner.alg.critic_optimizer.param_groups[0]["lr"] == pytest.approx(3e-5)
    assert runner.alg.actor_learning_rate == pytest.approx(1e-5)
    assert runner.alg.critic_learning_rate == pytest.approx(3e-5)


def test_inference_load_does_not_replace_optimizer_or_scheduler_rates(tmp_path):
    runner = make_runner()
    path = saved_checkpoint(tmp_path, runner)
    runner.load(path, load_optimizer=False, map_location="cpu")
    assert runner.alg.actor_learning_rate == pytest.approx(2e-5)
    assert runner.alg.critic_learning_rate == pytest.approx(1e-3)
    assert runner.alg.actor_optimizer.param_groups[0]["lr"] == pytest.approx(2e-5)
    assert runner.alg.critic_optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)
