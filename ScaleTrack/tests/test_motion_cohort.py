"""CPU tests for deterministic uniform motion cohorts."""

from __future__ import annotations

import copy

import pytest
import torch

from my_rsl_rl.utils.motion_cohort import UniformMotionCohortSampler, validate_uniform_probabilities


def test_continuous_epochs_are_complete_without_replacement():
    sampler = UniformMotionCohortSampler(10, 4, seed=7)
    cohorts = [sampler.next_ids() for _ in range(3)]
    stream = torch.cat(cohorts)
    assert torch.equal(torch.sort(stream[:10]).values, torch.arange(10))
    assert cohorts[2].shape == (4,) and cohorts[2].dtype == torch.long
    assert cohorts[2].device.type == "cpu"
    status = sampler.snapshot()
    assert status == {"num_motions": 10, "num_envs": 4, "seed": 7, "cohorts": 3,
                      "epochs": 2, "completed_epochs": 1, "unique_ever": 10,
                      "coverage": 1.0, "current_epoch_cursor": 2}


def test_7174_by_128_reaches_full_coverage_on_cohort_57():
    sampler = UniformMotionCohortSampler(7174, 128, seed=42)
    for _ in range(56): sampler.next_ids()
    assert sampler.snapshot()["unique_ever"] == 7168
    sampler.next_ids()
    assert sampler.snapshot()["unique_ever"] == 7174
    assert sampler.snapshot()["completed_epochs"] == 1


def test_seed_is_deterministic_and_isolated_from_global_rng():
    global_before = torch.random.get_rng_state().clone()
    a = UniformMotionCohortSampler(23, 7, seed=9)
    b = UniformMotionCohortSampler(23, 7, seed=9)
    c = UniformMotionCohortSampler(23, 7, seed=10)
    a_ids, b_ids, c_ids = a.next_ids(), b.next_ids(), c.next_ids()
    assert torch.equal(a_ids, b_ids) and not torch.equal(a_ids, c_ids)
    assert torch.equal(torch.random.get_rng_state(), global_before)


def test_return_and_state_tensors_do_not_alias_internal_state():
    sampler = UniformMotionCohortSampler(9, 4, seed=3)
    ids = sampler.next_ids(); state = sampler.state_dict()
    expected_next = sampler.next_ids()
    ids.fill_(99); state["permutation"].fill_(99); state["seen"].fill_(False); state["generator_state"].zero_()
    restored = UniformMotionCohortSampler(9, 4, seed=3)
    pristine = UniformMotionCohortSampler(9, 4, seed=3); pristine.next_ids()
    restored.load_state_dict(pristine.state_dict())
    assert torch.equal(restored.next_ids(), expected_next)


def test_state_round_trip_continues_exact_stream_across_epoch():
    original = UniformMotionCohortSampler(10, 4, seed=11)
    original.next_ids(); original.next_ids()
    state = original.state_dict()
    resumed = UniformMotionCohortSampler(10, 4, seed=999)
    resumed.load_state_dict(state)
    for _ in range(5):
        assert torch.equal(resumed.next_ids(), original.next_ids())
    assert resumed.snapshot() == original.snapshot()


@pytest.mark.parametrize("weights", [
    [1.0, 1.0, 1.0], torch.tensor([1.0, 1.0, 1.0]),
])
def test_uniform_positive_probabilities_are_accepted(weights):
    validate_uniform_probabilities(weights, 3)


@pytest.mark.parametrize("weights", [
    [1.0, 2.0, 1.0], [1.0, 0.0, 1.0], [1.0, -1.0, 1.0],
    [1.0, float("nan"), 1.0], [1.0, float("inf"), 1.0], [1.0, 1.0],
    torch.ones(3, dtype=torch.long), torch.ones(3, 1),
])
def test_bad_or_adaptive_probabilities_are_rejected(weights):
    with pytest.raises(ValueError): validate_uniform_probabilities(weights, 3)


@pytest.mark.parametrize("args", [(0, 4, 1), (4, 0, 1), (True, 4, 1), (4, 4, True), (4, 4, -1)])
def test_constructor_rejects_invalid_dimensions_and_seed(args):
    with pytest.raises(ValueError): UniformMotionCohortSampler(*args)


def test_invalid_restore_is_atomic():
    sampler = UniformMotionCohortSampler(10, 4, seed=5); sampler.next_ids()
    before = sampler.state_dict()
    bad_states = []
    bad = copy.deepcopy(before); bad["num_motions"] = 11; bad_states.append(bad)
    bad = copy.deepcopy(before); bad["permutation"][1] = bad["permutation"][0]; bad_states.append(bad)
    bad = copy.deepcopy(before); bad["cursor"] = 9; bad_states.append(bad)
    bad = copy.deepcopy(before); bad["seen"].fill_(False); bad_states.append(bad)
    for bad in bad_states:
        with pytest.raises(ValueError): sampler.load_state_dict(bad)
        after = sampler.state_dict()
        for key in before:
            if isinstance(before[key], torch.Tensor): assert torch.equal(after[key], before[key])
            else: assert after[key] == before[key]
