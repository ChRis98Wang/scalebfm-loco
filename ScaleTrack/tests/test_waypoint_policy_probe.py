"""Counterfactual evaluations must never replace the real action or mutate its inputs."""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/pretrain/rsl_rl"))
from waypoint_policy_probe import expected_pelvis_features, probe_policy


def observation():
    task = torch.arange(6 * 253, dtype=torch.float32).reshape(1, 6, 253) / 1000.
    task[..., 252] = torch.tensor([0, 1, 2, 3, 4, 32])
    mode = torch.zeros(1, 14)
    mode[:, 0] = 1.
    mapping = torch.cat([mode.repeat_interleave(width, dim=-1) for width in (3, 3, 6, 6)]
                        + [torch.ones(1, 1)], dim=-1)
    return {"policy_task": task, "mode": mode, "mode_mapping": mapping,
            "policy": torch.zeros(1, 3, 64), "action": torch.zeros(1, 3, 29)}


class Model(torch.nn.Module):
    is_recurrent = False

    def __init__(self, *, mutating=False, stochastic=False, wrong_mask=False):
        super().__init__()
        self.actor_task_embedder = torch.nn.Linear(267, 29)
        self.register_buffer("counter", torch.zeros(1))
        self.mutating, self.stochastic, self.wrong_mask = mutating, stochastic, wrong_mask

    def forward(self, obs):
        task = obs["policy_task"] * obs["mode_mapping"][:, None]
        mode = obs["mode"][:, None].expand(-1, 6, -1)
        if self.wrong_mask:
            mode = torch.ones_like(mode)
        result = self.actor_task_embedder(torch.cat([task, mode], dim=-1)).mean(dim=1)
        if self.mutating:
            self.counter += 1
        if self.stochastic:
            result = result + torch.rand_like(result)
        return result


def test_probe_returns_real_action_and_records_exact_actor_input_without_side_effects():
    model = Model().eval()
    obs = observation()
    saved = {key: value.clone() for key, value in obs.items()}
    rng = torch.random.get_rng_state().clone()
    action, report = probe_policy(model, model, obs)
    torch.testing.assert_close(action, model(obs), rtol=0, atol=0)
    assert report["moving_actor_task"].shape == (1, 6, 267)
    assert report["diagnostic_only_evaluations"] == 2
    assert report["repeat_action_exact"] and report["mask_exact"]
    assert not torch.equal(report["flat_actor_task"], report["moving_actor_task"])
    for key in obs:
        torch.testing.assert_close(obs[key], saved[key], rtol=0, atol=0)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert not model.actor_task_embedder._forward_pre_hooks


@pytest.mark.parametrize("option,match", [("mutating", "mutated model"), ("stochastic", "consumed RNG"),
                                         ("wrong_mask", "expected fixed")])
def test_probe_fails_closed_on_stateful_stochastic_or_wrong_mask_policy(option, match):
    model = Model(**{option: True}).eval()
    rng = torch.random.get_rng_state().clone()
    with pytest.raises(AssertionError, match=match):
        probe_policy(model, model, observation())
    assert torch.equal(rng, torch.random.get_rng_state())
    assert not model.actor_task_embedder._forward_pre_hooks


def test_training_and_recurrent_models_are_rejected():
    model = Model()
    with pytest.raises(ValueError, match="eval, non-recurrent"):
        probe_policy(model, model, observation())
    model.eval()
    model.is_recurrent = True
    with pytest.raises(ValueError, match="eval, non-recurrent"):
        probe_policy(model, model, observation())


def test_independent_pelvis_transform_contains_height_and_tangent_normal_rotation():
    yaw90 = [2**-.5, 0., 0., 2**-.5]
    features = expected_pelvis_features([[1., 0., .7], [2., 0., .6]], [yaw90, [1., 0., 0., 0.]],
                                       [0., 0., .8], yaw90)
    np.testing.assert_allclose(features["target_body_pos"], [[0., -1., -.1], [0., -2., -.2]], atol=1e-12)
    np.testing.assert_allclose(features["target_body_rot"], [[1., 0., 0., 0., 0., 1.],
                                                          [0., -1., 0., 0., 0., 1.]], atol=1e-12)
    np.testing.assert_array_equal(features["target_body_pos"], features["target_body_pos_rel"])
    np.testing.assert_array_equal(features["target_body_rot"], features["target_body_rot_rel"])


@pytest.mark.parametrize("q", [[0., 0., 0., 0.], [2., 0., 0., 0.], [np.nan, 0., 0., 0.]])
def test_invalid_pose_cannot_be_used_for_actor_verification(q):
    with pytest.raises(ValueError):
        expected_pelvis_features([[0., 0., .8]], [[1., 0., 0., 0.]], [0., 0., .8], q)
