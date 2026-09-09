"""Online adapter tests using real LiveMotionCommand property implementations."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scaletrack.tasks.tracking.mdp.live_commands import LiveMotionCommand
from scaletrack.utils.live_reference import LiveReferenceProvider
from scaletrack.utils.online_modes import ONLINE_MODES


def _seed():
    names = [f"body_{index}" for index in range(14)]
    names[0], names[10], names[13] = "pelvis", "left_wrist_yaw_link", "right_wrist_yaw_link"
    body_pos = np.zeros((14, 3), dtype=np.float64)
    body_pos[0] = [0.0, 0.0, 0.8]
    body_pos[10] = [0.2, 0.2, 1.0]
    body_pos[13] = [0.2, -0.2, 1.0]
    body_quat = np.tile([1.0, 0.0, 0.0, 0.0], (14, 1))
    return tuple(names), body_pos, body_quat, np.zeros(29, dtype=np.float64)


def _command(runtime_order="wxyz", *, complete=False, mode_name="VR-3"):
    names, body_pos, body_quat, joints = _seed()
    command = LiveMotionCommand.__new__(LiveMotionCommand)
    command._env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        step_dt=0.02,
        scene=SimpleNamespace(env_origins=torch.tensor([[4.0, -3.0, 0.5]])),
    )
    command.cfg = SimpleNamespace(
        body_names=names,
        ref_frame_buffer_size=33,
        mode_candidates={mode_name: list(ONLINE_MODES[mode_name])},
    )
    command.motion_anchor_body_index = 0
    command._rand_timestep = torch.tensor([[7]], dtype=torch.long)
    command._future_manual_cache = {}
    command._live_samples = {}
    command.live_reference = None
    command.live_mode_name = None
    command.runtime_quaternion_order = runtime_order
    if complete:
        actual_pos = torch.as_tensor(body_pos, dtype=torch.float32).unsqueeze(0) + torch.tensor(
            [[[4.0, -3.0, 0.5]]]
        )
        actual_quat = torch.as_tensor(body_quat, dtype=torch.float32).unsqueeze(0)
        writes = {"joint": 0, "root": 0}

        def write_joint(*_args, **_kwargs):
            writes["joint"] += 1

        def write_root(*_args, **_kwargs):
            writes["root"] += 1

        command.robot = SimpleNamespace(
            data=SimpleNamespace(
                joint_pos=torch.zeros((1, 29)),
                joint_vel=torch.zeros((1, 29)),
                body_pos_w=actual_pos,
                body_quat_w=actual_quat,
                body_lin_vel_w=torch.zeros((1, 14, 3)),
                body_ang_vel_w=torch.zeros((1, 14, 3)),
            ),
            write_joint_state_to_sim=write_joint,
            write_root_state_to_sim=write_root,
        )
        command.body_indexes = torch.arange(14, dtype=torch.long)
        command.robot_anchor_body_index = 0
        command.metrics = {}
        command.time_left = torch.zeros(1)
        command.command_counter = torch.zeros(1, dtype=torch.long)
        command._write_counts = writes
    return command, LiveReferenceProvider(names, body_pos, body_quat, joints)


def _expected_mode(name):
    mask = torch.zeros((1, 14), dtype=torch.float32)
    mask[0, list({"Pelvis-1": (0,), "UMI-2": (10, 13), "VR-3": (0, 10, 13)}[name])] = 1.0
    return mask


@pytest.mark.parametrize("mode_name", tuple(ONLINE_MODES))
def test_attach_supports_each_canonical_online_mode_with_an_exact_14_body_mask(mode_name):
    command, provider = _command(mode_name=mode_name)
    command.attach_live_reference(provider, mode_name=mode_name)

    expected = _expected_mode(mode_name)
    assert command.live_reference is provider
    assert command.live_mode_name == mode_name
    assert torch.equal(command._mode, expected)
    assert torch.equal(command._mode_table, expected)
    assert command.cfg.mode_candidates == {mode_name: list(ONLINE_MODES[mode_name])}


@pytest.mark.parametrize(
    ("runtime_order", "expected_identity"),
    [("wxyz", [1.0, 0.0, 0.0, 0.0]), ("xyzw", [0.0, 0.0, 0.0, 1.0])],
)
def test_live_getters_share_provider_frame_apply_origin_once_and_keep_runtime_quaternion_order(
    runtime_order, expected_identity
):
    """Catch partial offline reads, double origins, or a canonical quaternion leaking to runtime."""
    command, provider = _command(runtime_order)
    command.attach_live_reference(provider)

    np.testing.assert_allclose(command.body_pos_w[0, 0].numpy(), [4.0, -3.0, 1.3])
    np.testing.assert_allclose(command.body_quat_w[0, 0].numpy(), expected_identity)
    manual = command.body_pos_w_future_manual([0, 32, -1])
    assert manual.shape == (1, 3, 14, 3)
    assert command.time_offsets_future_manual([0, 32, -1]).tolist() == [[[0], [32], [7]]]
    assert command.time_offsets_future.dtype == torch.long
    assert torch.equal(command.anchor_pos_w_future, command.body_pos_w_future[:, :, 0])
    assert command.joint_pos_future.shape == (1, 34, 29)


@pytest.mark.parametrize("invalid_offset", [1.9, True, np.bool_(True), torch.tensor(True), "1"])
def test_live_offsets_reject_lossy_or_noninteger_values_without_cache_mutation(invalid_offset):
    """Catch int/index accepting a fractional, boolean, or string offset as a valid frame."""
    command, provider = _command()
    command.attach_live_reference(provider)
    before = dict(command._live_samples)
    rand_before = command._rand_timestep.clone()
    with pytest.raises(ValueError, match="integer"):
        command.body_pos_w_future_manual([invalid_offset])
    assert command._live_samples == before
    assert torch.equal(command._rand_timestep, rand_before)


def test_live_offsets_accept_scalar_integer_tensors_without_mutating_the_random_offset():
    """Keep valid integer-like tensor offsets usable while rejecting boolean tensors."""
    command, provider = _command()
    command.attach_live_reference(provider)
    rand_before = command._rand_timestep.clone()
    assert command.body_pos_w_future_manual([torch.tensor(7)]).shape == (1, 1, 14, 3)
    assert torch.equal(command._rand_timestep, rand_before)


def test_attach_probes_the_provider_public_33_frame_sampling_contract_before_mutation():
    """Catch attachment accepting a provider that cannot serve the documented horizon."""
    command, provider = _command()
    calls = []
    original_sample = provider.sample

    def recording_sample(offsets):
        calls.append(tuple(offsets))
        return original_sample(offsets)

    provider.sample = recording_sample
    command.attach_live_reference(provider)
    assert calls == [tuple(range(33))]
    assert command.live_reference is provider
    assert torch.equal(command._mode, torch.tensor([[1.0] + [0.0] * 9 + [1.0, 0.0, 0.0, 1.0]]))


def test_attach_leaves_mask_and_reference_unset_when_public_horizon_sampling_fails():
    """Catch a failed provider probe partially attaching an online command."""
    command, provider = _command()

    def unsupported_horizon(_offsets):
        raise ValueError("unsupported provider horizon")

    provider.sample = unsupported_horizon
    with pytest.raises(ValueError, match="unsupported provider horizon"):
        command.attach_live_reference(provider)
    assert command.live_reference is None
    assert command.live_mode_name is None
    assert not hasattr(command, "_mode")
    assert not hasattr(command, "_mode_table")


@pytest.mark.parametrize(
    ("mode_name", "candidates", "error"),
    [
        ("unknown", {"VR-3": list(ONLINE_MODES["VR-3"])}, "online mode"),
        ("Pelvis-1", {"VR-3": list(ONLINE_MODES["VR-3"])}, "configured Pelvis-1"),
        ("UMI-2", {"UMI-2": ["pelvis"]}, "configured UMI-2"),
    ],
)
def test_attach_rejects_invalid_or_mismatched_mode_config_without_sampling_or_partial_state(
    mode_name, candidates, error
):
    command, provider = _command()
    command.cfg.mode_candidates = candidates
    command._live_samples["sentinel"] = object()
    command._future_manual_cache["sentinel"] = object()
    sample_calls = []
    original_sample = provider.sample

    def recording_sample(offsets):
        sample_calls.append(tuple(offsets))
        return original_sample(offsets)

    provider.sample = recording_sample
    with pytest.raises(ValueError, match=error):
        command.attach_live_reference(provider, mode_name=mode_name)

    assert sample_calls == []
    assert command.live_reference is None
    assert command.live_mode_name is None
    assert not hasattr(command, "_mode")
    assert not hasattr(command, "_mode_table")
    assert set(command._live_samples) == {"sentinel"}
    assert set(command._future_manual_cache) == {"sentinel"}
    assert command.cfg.mode_candidates is candidates


def test_attach_rejects_noncanonical_body_order_without_partial_state():
    command, provider = _command(mode_name="Pelvis-1")
    names = list(command.cfg.body_names)
    names[0], names[1] = names[1], names[0]
    command.cfg.body_names = tuple(names)

    with pytest.raises(ValueError, match="provider body order"):
        command.attach_live_reference(provider, mode_name="Pelvis-1")

    assert command.live_reference is None
    assert command.live_mode_name is None
    assert not hasattr(command, "_mode")
    assert not hasattr(command, "_mode_table")


def test_set_live_mode_switches_single_mode_cache_without_advancing_or_writing_state():
    command, provider = _command(complete=True)
    command.attach_live_reference(provider)
    command.body_pos_w_future_manual([0, 7])
    cache_before = dict(command._live_samples)
    rand_before = command._rand_timestep.clone()
    robot_state_before = {
        name: getattr(command.robot.data, name).clone()
        for name in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")
    }
    provider_state_before = (provider.version, provider.frame_index)

    def unexpected_sample(_offsets):
        raise AssertionError("mode switching must not sample or advance the provider")

    provider.sample = unexpected_sample
    for mode_name in ("Pelvis-1", "UMI-2", "VR-3"):
        command.set_live_mode(mode_name)
        expected = _expected_mode(mode_name)
        assert command.live_mode_name == mode_name
        assert torch.equal(command._mode, expected)
        assert torch.equal(command._mode_table, expected)
        assert command.cfg.mode_candidates == {mode_name: list(ONLINE_MODES[mode_name])}
        assert command.live_reference is provider
        assert (provider.version, provider.frame_index) == provider_state_before
        assert command._write_counts == {"joint": 0, "root": 0}
        assert torch.equal(command._rand_timestep, rand_before)
        assert command._live_samples.keys() == cache_before.keys()
        assert all(command._live_samples[key] is value for key, value in cache_before.items())
        for name, value in robot_state_before.items():
            assert torch.equal(getattr(command.robot.data, name), value)


def test_set_live_mode_rejects_invalid_name_without_mutation():
    command, provider = _command()
    command.attach_live_reference(provider)
    cfg_before = command.cfg.mode_candidates
    mask_before = command._mode.clone()
    table_before = command._mode_table.clone()

    with pytest.raises(ValueError, match="online mode"):
        command.set_live_mode("VR-5")

    assert command.cfg.mode_candidates is cfg_before
    assert command.live_mode_name == "VR-3"
    assert torch.equal(command._mode, mask_before)
    assert torch.equal(command._mode_table, table_before)


def test_live_getters_keep_nontrivial_frame_and_cache_consistent_across_current_manual_generic_and_anchor():
    """Catch a getter using offset zero/offline data after a moving live reference is attached."""
    command, provider = _command("xyzw")
    goal_pos = np.array([[0.6, 0.0, 0.8], [0.2, 0.6, 1.0], [0.2, -0.6, 1.0]])
    half_turn_x = np.array([0.0, 1.0, 0.0, 0.0])
    compound_rotation = np.array([0.5, 0.5, 0.5, 0.5])
    provider.submit(
        goal_pos,
        np.array([compound_rotation, half_turn_x, compound_rotation]),
        sequence=1,
        stamp=1.0,
        now=1.0,
    )
    assert provider.commit_pending(1.0)
    command.attach_live_reference(provider)
    command._update_command()
    expected = provider.sample([0, 7, 32])
    rand_before = command._rand_timestep.clone()

    current = command.body_pos_w
    manual_pos = command.body_pos_w_future_manual([0, 7, 32])
    generic_pos = command.body_pos_w_future
    manual_quat = command.body_quat_w_future_manual([0, 7, 32])
    manual_vel = command.body_ang_vel_w_future_manual([0, 7, 32])
    origin = command._env.scene.env_origins
    np.testing.assert_allclose(current[0].numpy(), expected["body_pos"][0] + origin.numpy()[0])
    np.testing.assert_allclose(manual_pos[0].numpy(), expected["body_pos"] + origin.numpy()[0])
    np.testing.assert_allclose(generic_pos[0, [0, 7, 32]].numpy(), manual_pos[0].numpy())
    np.testing.assert_allclose(command.anchor_pos_w_future[:, [0, 7, 32]].numpy(), manual_pos[:, :, 0].numpy())
    np.testing.assert_allclose(manual_quat[0, :, 0].numpy(), expected["body_quat"][:, 0, [1, 2, 3, 0]])
    np.testing.assert_allclose(manual_vel[0].numpy(), expected["body_ang_vel"])
    np.testing.assert_allclose(command.joint_pos_future[0, [0, 7, 32]].numpy(), expected["joint_pos"])
    assert manual_pos.dtype == current.dtype == torch.float32
    assert manual_pos.device == current.device == torch.device("cpu")
    assert torch.equal(command._rand_timestep, rand_before)
    assert len(command._live_samples) == 1


def test_online_compute_advances_only_provider_without_timeout_resampling_resets_or_state_writes():
    """Catch online compute inheriting the offline timer/resample/state-write path."""
    command, provider = _command(complete=True)
    command.attach_live_reference(provider)
    time_left_before = command.time_left.clone()
    counter_before = command.command_counter.clone()
    for _ in range(501):
        command.compute(0.02)
    assert provider.frame_index == 501
    assert torch.equal(command.time_left, time_left_before)
    assert torch.equal(command.command_counter, counter_before)
    assert command._write_counts == {"joint": 0, "root": 0}
    with pytest.raises(RuntimeError, match="online"):
        command._resample_command(torch.tensor([0]))
