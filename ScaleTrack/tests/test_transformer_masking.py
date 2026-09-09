"""Regression tests for side-effect-free transformer task masking."""

from __future__ import annotations

from pathlib import Path
import sys

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source/my_rsl_rl"))
from my_rsl_rl.modules.actor_critic_humanoid_transformer import ActorCriticHumanoidTransformer


def _actor_obs(obs, *, inference=False):
    # get_actor_obs is intentionally independent of initialized network layers.
    return ActorCriticHumanoidTransformer.get_actor_obs(None, obs, inference=inference)


def _observation(task, mapping):
    batch, context, _ = task.shape
    mode = torch.zeros(batch, 14, dtype=task.dtype)
    mode[:, 0] = 1.0  # Pelvis-1.
    return {
        "policy": torch.arange(batch * context * 4, dtype=task.dtype).reshape(batch, context, 4),
        "policy_task": task,
        "action": torch.zeros(batch, context, 3, dtype=task.dtype),
        "mode": mode,
        "mode_mapping": mapping,
    }


def test_training_mask_is_out_of_place_numerically_identical_and_differentiable():
    task = torch.arange(36, dtype=torch.float32).reshape(2, 3, 6).requires_grad_()
    original = task.detach().clone()
    mapping = torch.tensor([[1, 0, 1, 0, 0, 1], [0, 1, 0, 1, 0, 1]], dtype=torch.float32)
    obs = _observation(task, mapping)

    _, masked_with_mode, _ = _actor_obs(obs)
    expected_mapping = mapping.unsqueeze(1).repeat(1, task.shape[1], 1)
    torch.testing.assert_close(masked_with_mode[..., :6], original * expected_mapping)
    torch.testing.assert_close(masked_with_mode[..., 6:], obs["mode"].unsqueeze(1).repeat(1, 3, 1))
    torch.testing.assert_close(obs["policy_task"], original)

    masked_with_mode[..., :6].sum().backward()
    torch.testing.assert_close(task.grad, expected_mapping)
    torch.testing.assert_close(obs["policy_task"], original)


def test_repeated_masks_on_one_captured_observation_do_not_cross_contaminate():
    task = torch.arange(18, dtype=torch.float32).reshape(1, 3, 6)
    original = task.clone()
    first_mapping = torch.tensor([[1, 0, 0, 1, 0, 1]], dtype=torch.float32)
    second_mapping = torch.tensor([[0, 1, 1, 0, 1, 1]], dtype=torch.float32)
    obs = _observation(task, first_mapping)

    first = _actor_obs(obs)[1]
    obs["mode_mapping"] = second_mapping
    second = _actor_obs(obs)[1]

    torch.testing.assert_close(first[..., :6], original * first_mapping[:, None, :])
    torch.testing.assert_close(second[..., :6], original * second_mapping[:, None, :])
    torch.testing.assert_close(obs["policy_task"], original)


def test_inference_keeps_task_unmasked_and_appends_all_ones_without_mutation():
    task = torch.arange(18, dtype=torch.float32).reshape(1, 3, 6)
    mapping = torch.tensor([[1, 0, 0, 0, 0, 1]], dtype=torch.float32)
    obs = _observation(task.clone(), mapping)
    original_task, original_mode = obs["policy_task"].clone(), obs["mode"].clone()

    _, inference_task, _ = _actor_obs(obs, inference=True)

    torch.testing.assert_close(inference_task[..., :6], original_task)
    torch.testing.assert_close(inference_task[..., 6:], torch.ones(1, 3, 14))
    torch.testing.assert_close(obs["policy_task"], original_task)
    torch.testing.assert_close(obs["mode"], original_mode)


def test_pelvis_mapping_keeps_active_features_and_time_while_masking_other_bodies():
    # Representative mode_mapping layout: three XYZ bodies followed by time.
    task = torch.tensor([[[1., 2., 3., 10., 11., 12., 20., 21., 22., 0.25],
                          [4., 5., 6., 13., 14., 15., 23., 24., 25., 0.50]]])
    pelvis_with_time = torch.tensor([[1., 1., 1., 0., 0., 0., 0., 0., 0., 1.]])
    obs = _observation(task.clone(), pelvis_with_time)

    _, masked_with_mode, _ = _actor_obs(obs)

    expected = task * pelvis_with_time[:, None, :]
    torch.testing.assert_close(masked_with_mode[..., :10], expected)
    torch.testing.assert_close(masked_with_mode[..., -14:], obs["mode"][:, None, :].repeat(1, 2, 1))
    assert torch.equal(masked_with_mode[..., 9], task[..., 9])
    assert torch.count_nonzero(masked_with_mode[..., 3:9]) == 0
