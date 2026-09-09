"""Measure the actual PPO update without changing its gradients, RNG, or outputs."""

import copy
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source/my_rsl_rl"))
from my_rsl_rl.algorithms import PPO
from my_rsl_rl.modules import ActorCritic
from my_rsl_rl.runners.on_policy_runner import OnPolicyRunner


def make_algorithm(*, enabled=True, schedule="fixed", actor_lr=0.0, critic_lr=0.0,
                   max_grad_norm=1.0, num_actions=1):
    obs = TensorDict({key: torch.zeros(4, 1) for key in
                      ("policy", "policy_task", "critic", "critic_task", "action")},
                     batch_size=[4])
    policy = ActorCritic(obs, {}, num_actions, actor_hidden_dims=[4], critic_hidden_dims=[4],
                         init_noise_std=1.0)
    alg = PPO(policy, num_learning_epochs=2, num_mini_batches=2,
              actor_learning_rate=actor_lr, critic_learning_rate=critic_lr,
              schedule=schedule, max_grad_norm=max_grad_norm,
              log_update_diagnostics=enabled, device="cpu")
    alg.init_storage("rl", 4, 2, obs, [num_actions])
    return alg, obs


def fill_rollout(alg, obs):
    with torch.inference_mode():
        for _ in range(2):
            alg.act(obs)
            alg.process_env_step(obs, torch.tensor([0.1, 0.2, 0.3, 0.4]),
                                 torch.zeros(4), {})
        alg.compute_returns(obs)


def test_fixed_update_reports_known_gaussian_kl_and_ratio_clipping():
    alg, obs = make_algorithm()
    fill_rollout(alg, obs)
    # Unit variances with a unit old/new mean difference have KL = 1/2.
    alg.storage.mu.add_(1.0)
    # Recomputed log probability is log(2) greater, so every ratio is 2 > 1.2.
    alg.storage.actions_log_prob.sub_(math.log(2.0))
    loss = alg.update()
    assert set(loss) == {"value_function", "surrogate", "entropy"}
    stats = alg.update_diagnostics
    assert stats["minibatches"] == 4
    for suffix in ("mean", "max", "first", "last"):
        assert stats[f"policy_kl_{suffix}"] == pytest.approx(0.5, abs=1e-6)
    assert stats["ratio_clip_fraction_mean"] == 1.0
    assert stats["ratio_clip_fraction_max"] == 1.0
    for optimizer in ("actor", "critic"):
        for suffix in ("first", "last", "min", "max"):
            assert stats[f"{optimizer}_lr_{suffix}"] == 0.0
    assert "scheduler_kl_mean" not in stats


def test_29_action_kl_uses_old_to_current_direction_with_unequal_variances():
    torch.manual_seed(3)
    alg, obs = make_algorithm(num_actions=29)
    fill_rollout(alg, obs)
    alg.storage.mu.add_(0.25)
    alg.storage.sigma.mul_(2.0)
    alg.update()
    # KL(N(mu+.25, 2^2) || N(mu, 1^2)); reversing KL changes this value.
    expected = 29 * ((4.0 + 0.25**2) / 2.0 - 0.5 - math.log(2.0))
    for suffix in ("mean", "max", "first", "last"):
        assert alg.update_diagnostics[f"policy_kl_{suffix}"] == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("seed", [0, 3])
@pytest.mark.parametrize("mean_shift,first,last", [
    (0.0, 0.00015, 0.00050625),
    (1.0, 0.0000666666666667, 0.0000197530864198),
])
def test_adaptive_lr_records_rates_after_schedule_before_optimizer_step(mean_shift, first, last, seed):
    # Zero gradient clipping prevents policy changes while Adam still takes real steps.
    torch.manual_seed(seed)
    alg, obs = make_algorithm(schedule="adaptive", actor_lr=1e-4,
                              critic_lr=2e-4, max_grad_norm=0.0)
    fill_rollout(alg, obs)
    alg.storage.mu.add_(mean_shift)
    alg.update()
    stats = alg.update_diagnostics
    assert stats["actor_lr_first"] == pytest.approx(first)
    assert stats["actor_lr_last"] == pytest.approx(last)
    assert stats["actor_lr_min"] == pytest.approx(min(first, last))
    assert stats["actor_lr_max"] == pytest.approx(max(first, last))
    assert stats["critic_lr_first"] == pytest.approx(2 * first)
    assert stats["critic_lr_last"] == pytest.approx(2 * last)
    assert stats["policy_kl_first"] == pytest.approx(mean_shift**2 / 2, abs=1e-7)
    # Scheduler retains its original epsilon bias; true Gaussian KL is separate.
    # Float32 old_mu + shift and subsequent subtraction can lose an ULP;
    # seed 3 explicitly covers that case against this analytic float64 value.
    assert stats["scheduler_kl_first"] == pytest.approx(
        mean_shift**2 / 2 + math.log(1.00001), abs=1e-6)
    assert stats["actor_lr_last"] == alg.actor_optimizer.param_groups[0]["lr"]


def test_diagnostics_are_reset_between_updates_and_before_a_failed_update():
    alg, obs = make_algorithm()
    fill_rollout(alg, obs)
    alg.storage.mu.add_(1.0)
    alg.update()
    assert alg.update_diagnostics["policy_kl_mean"] == pytest.approx(0.5, abs=1e-6)
    fill_rollout(alg, obs)
    alg.update()
    assert alg.update_diagnostics["minibatches"] == 4
    assert alg.update_diagnostics["policy_kl_mean"] == pytest.approx(0.0, abs=1e-7)
    alg.storage = None
    with pytest.raises(AttributeError):
        alg.update()
    assert alg.update_diagnostics == {}


def assert_tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_tree_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for left, right in zip(a, b, strict=True):
            assert_tree_equal(left, right)
    else:
        assert a == b


def test_logging_does_not_change_policy_optimizer_losses_or_rng():
    outputs = []
    for enabled in (False, True):
        torch.manual_seed(77)
        alg, obs = make_algorithm(enabled=enabled, schedule="adaptive",
                                  actor_lr=2e-4, critic_lr=3e-4)
        before = copy.deepcopy(alg.policy.state_dict())
        fill_rollout(alg, obs)
        loss = alg.update()
        assert any(not torch.equal(before[k], v) for k, v in alg.policy.state_dict().items())
        assert bool(alg.update_diagnostics) is enabled
        outputs.append((copy.deepcopy(alg.policy.state_dict()),
                        copy.deepcopy(alg.actor_optimizer.state_dict()),
                        copy.deepcopy(alg.critic_optimizer.state_dict()),
                        loss, torch.get_rng_state().clone()))
    assert_tree_equal(outputs[0], outputs[1])


def test_diagnostics_do_not_add_per_minibatch_scalar_host_reads(monkeypatch):
    # Tensor.item() synchronizes CUDA. The new diagnostics must retain detached
    # device scalars and copy them in one batch after all optimizer updates.
    original_item = torch.Tensor.item
    counts = []
    for enabled in (False, True):
        torch.manual_seed(77)
        alg, obs = make_algorithm(enabled=enabled, schedule="adaptive",
                                  actor_lr=2e-4, critic_lr=3e-4)
        fill_rollout(alg, obs)
        reads = []

        def counted_item(tensor, *args, **kwargs):
            reads.append(tensor.shape)
            return original_item(tensor, *args, **kwargs)

        with monkeypatch.context() as context:
            context.setattr(torch.Tensor, "item", counted_item)
            alg.update()
        counts.append(len(reads))
    assert counts[0] == counts[1]


@pytest.mark.parametrize("enabled", [True, False])
def test_runner_writes_real_tensorboard_diagnostics_separately_from_losses(tmp_path, enabled):
    alg, obs = make_algorithm(enabled=enabled)
    fill_rollout(alg, obs)
    alg.storage.mu.add_(1.0)
    loss = alg.update()
    runner = OnPolicyRunner.__new__(OnPolicyRunner)
    runner.alg = alg
    runner.env = SimpleNamespace(num_envs=4)
    runner.num_steps_per_env = 2
    runner.gpu_world_size = 1
    runner.tot_timesteps = 0
    runner.tot_time = 0.0
    runner.device = "cpu"
    runner.logger_type = "tensorboard"
    runner.writer = SummaryWriter(str(tmp_path))
    try:
        runner.log({"it": 3, "tot_iter": 4, "start_iter": 3,
                    "num_learning_iterations": 1, "collection_time": 0.1,
                    "learn_time": 0.1, "ep_infos": [], "rewbuffer": [],
                    "lenbuffer": [], "loss_dict": loss})
    finally:
        runner.writer.close()
    events = EventAccumulator(str(tmp_path))
    events.Reload()
    tags = events.Tags()["scalars"]
    assert "Loss/value_function" in tags and "Loss/actor_learning_rate" in tags
    assert "Loss/policy_kl_mean" not in tags
    if enabled:
        values = events.Scalars("PPO/policy_kl_mean")
        assert len(values) == 1 and values[0].step == 3
        assert values[0].value == pytest.approx(0.5, abs=1e-6)
        assert events.Scalars("PPO/minibatches")[0].value == 4
    else:
        assert not any(tag.startswith("PPO/") for tag in tags)
