"""Behavioral tests for the pure, atomic VR-3 live reference provider."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from dataclasses import replace
import sys

import numpy as np
import pytest


_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "source/scaletrack/scaletrack/utils/live_reference.py"
)


def _load_module():
    """Load directly so a missing provider is an intentional test failure."""
    assert _MODULE_PATH.is_file(), "live reference provider module is missing"
    spec = importlib.util.spec_from_file_location("live_reference_under_test", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert getattr(module, "LiveReferenceProvider", None) is not None, (
        "LiveReferenceProvider is missing"
    )
    return module


def _seed():
    names = tuple(f"body_{index}" for index in range(14))
    names = list(names)
    names[0] = "pelvis"
    names[10] = "left_wrist_yaw_link"
    names[13] = "right_wrist_yaw_link"
    positions = np.zeros((14, 3), dtype=np.float64)
    positions[0] = [0.0, 0.0, 0.8]
    positions[10] = [0.2, 0.2, 1.0]
    positions[13] = [0.2, -0.2, 1.0]
    quaternions = np.tile([1.0, 0.0, 0.0, 0.0], (14, 1))
    return tuple(names), positions, quaternions, np.zeros(29, dtype=np.float64)


def _test_quat_multiply(left, right):
    """Independent wxyz product used only to derive a world-frame expectation."""
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ]
    )


def _test_axis_angle(axis, angle):
    normalized_axis = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    return np.concatenate(([np.cos(angle / 2.0)], normalized_axis * np.sin(angle / 2.0)))


def _trajectory(positions, *, step_m=0.002, step_angle=0.005):
    """Build an independently predictable, valid 34-frame canonical VR-3 path."""
    base = positions[[0, 10, 13]]
    trajectory_positions = np.stack(
        [base + [0.0, step_m * frame, 0.0] for frame in range(34)]
    )
    trajectory_quaternions = np.stack(
        [np.tile(_test_axis_angle([0.0, 0.0, 1.0], step_angle * frame), (3, 1))
         for frame in range(34)]
    )
    return trajectory_positions, trajectory_quaternions


def test_commit_preserves_current_frame_and_predicts_limited_future_motion():
    """Catch a commit jump or a future trajectory that ignores the 0.2 m/s cap."""
    module = _load_module()
    names, positions, quaternions, joints = _seed()
    provider = module.LiveReferenceProvider(names, positions, quaternions, joints)

    provider.submit(
        np.array([[0.2, 0.0, 0.8], [0.2, 0.2, 1.0], [0.2, -0.2, 1.0]]),
        np.tile([1.0, 0.0, 0.0, 0.0], (3, 1)),
        sequence=1,
        stamp=1.0,
        now=1.0,
    )
    assert provider.commit_pending(1.0)
    np.testing.assert_allclose(provider.sample([0])["body_pos"][0, 0], [0.0, 0.0, 0.8])
    np.testing.assert_allclose(provider.sample([32])["body_pos"][0, 0], [0.128, 0.0, 0.8])

    provider.advance()
    np.testing.assert_allclose(provider.sample([0])["body_pos"][0, 0], [0.004, 0.0, 0.8])


def test_repeated_goal_packets_keep_motion_continuous_and_leave_seed_values_intact():
    """Catch updates that teleport a reference or modify an inactive body or joint."""
    module = _load_module()
    names, positions, quaternions, joints = _seed()
    provider = module.LiveReferenceProvider(names, positions, quaternions, joints)

    previous = provider.sample([0])["body_pos"][0, 0].copy()
    for sequence in range(1, 101):
        target = np.array(
            [
                [0.10 if sequence % 2 else -0.10, 0.0, 0.8],
                [0.10, 0.20, 1.0],
                [0.10, -0.20, 1.0],
            ]
        )
        provider.submit(
            target,
            np.tile([1.0, 0.0, 0.0, 0.0], (3, 1)),
            sequence=sequence,
            stamp=float(sequence),
            now=float(sequence),
        )
        assert provider.commit_pending(float(sequence))
        np.testing.assert_allclose(provider.sample([0])["body_pos"][0, 0], previous)
        provider.advance()
        current = provider.sample([0])
        assert np.linalg.norm(current["body_pos"][0, 0] - previous) <= 0.0040000001
        np.testing.assert_allclose(current["body_pos"][0, 1], positions[1])
        np.testing.assert_allclose(current["joint_pos"][0], joints)
        previous = current["body_pos"][0, 0].copy()


def test_velocities_are_bounded_and_terminal_frame_uses_an_extra_prediction():
    """Catch zero-padded frame-32 velocity or Euler/uncapped rotation differences."""
    module = _load_module()
    names, positions, quaternions, joints = _seed()
    provider = module.LiveReferenceProvider(names, positions, quaternions, joints)
    half_turn_z = np.array([0.0, 0.0, 0.0, 1.0])
    provider.submit(
        np.array([[0.004, 0.0, 0.8], [0.2, 0.2, 1.0], [0.2, -0.2, 1.0]]),
        np.array([half_turn_z, half_turn_z, half_turn_z]),
        sequence=1,
        stamp=1.0,
        now=1.0,
    )
    assert provider.commit_pending(1.0)
    frames = provider.sample(np.arange(33))
    linear_speeds = np.linalg.norm(frames["body_lin_vel"][:, [0, 10, 13]], axis=-1)
    angular_speeds = np.linalg.norm(frames["body_ang_vel"][:, [0, 10, 13]], axis=-1)
    assert linear_speeds[:, 0].max() <= 0.2000001
    assert linear_speeds[:, 1:].max() <= 0.3500001
    assert angular_speeds[:, 0].max() <= 0.5000001
    assert angular_speeds[:, 1:].max() <= 1.0000001
    np.testing.assert_allclose(frames["body_pos"][-1, 0], [0.004, 0.0, 0.8])
    np.testing.assert_allclose(frames["body_lin_vel"][-1, 0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(frames["body_ang_vel"][-1, 0], [0.0, 0.0, 0.5])


def test_angular_velocity_uses_shortest_world_frame_rotation_log():
    """Catch body-frame multiplication or a long arc for a negative-hemisphere goal."""
    module = _load_module()
    names, positions, quaternions, joints = _seed()
    starting_orientation = _test_quat_multiply(
        _test_axis_angle([0.0, 0.0, 1.0], 0.6),
        _test_axis_angle([0.0, 1.0, 0.0], -0.4),
    )
    world_axis = np.array([0.3, -0.4, np.sqrt(0.75)])
    world_rotation = _test_axis_angle(world_axis, 0.006)
    quaternions[0] = starting_orientation
    provider = module.LiveReferenceProvider(names, positions, quaternions, joints)
    target_quaternion = -_test_quat_multiply(world_rotation, starting_orientation)
    provider.submit(
        positions[[0, 10, 13]],
        np.array([target_quaternion, quaternions[10], quaternions[13]]),
        sequence=1,
        stamp=1.0,
        now=1.0,
    )
    assert provider.commit_pending(1.0)

    expected_world_velocity = world_axis * (0.006 / provider.step_dt)
    np.testing.assert_allclose(
        provider.sample([0])["body_ang_vel"][0, 0], expected_world_velocity, atol=1e-12
    )


def test_packet_validation_is_atomic_and_normalizes_equivalent_quaternions():
    """Catch malformed input replacing a valid pending packet or a long quaternion arc."""
    module = _load_module()
    names, positions, quaternions, joints = _seed()
    provider = module.LiveReferenceProvider(names, positions, quaternions, joints)
    valid_positions = np.array([[0.1, 0.0, 0.8], [0.1, 0.2, 1.0], [0.1, -0.2, 1.0]])
    provider.submit(
        valid_positions,
        np.tile([-1.0, 0.0, 0.0, 0.0], (3, 1)),
        sequence=1,
        stamp=1.0,
        now=1.0,
    )
    with pytest.raises(module.ReferenceInputError):
        provider.submit(
            valid_positions,
            np.array([[0.0, 0.0, 0.0, 0.0]] * 3),
            sequence=2,
            stamp=2.0,
            now=2.0,
        )
    assert provider.rejected_count == 1
    assert provider.commit_pending(1.0)
    provider.advance()
    np.testing.assert_allclose(provider.sample([0])["body_ang_vel"][0, [0, 10, 13]], 0.0)


def test_packet_body_order_and_quaternion_norm_rules_are_enforced():
    """Catch name/order mismatches or accepting a quaternion that changes rotation scale."""
    module = _load_module()
    names, positions, quaternions, joints = _seed()
    provider = module.LiveReferenceProvider(names, positions, quaternions, joints)
    targets = np.array([[0.1, 0.0, 0.8], [0.1, 0.2, 1.0], [0.1, -0.2, 1.0]])
    nearly_unit = np.tile([1.0005, 0.0, 0.0, 0.0], (3, 1))
    provider.submit(
        targets[[2, 0, 1]],
        nearly_unit[[2, 0, 1]],
        body_names=("right_wrist_yaw_link", "pelvis", "left_wrist_yaw_link"),
        sequence=1,
        stamp=1.0,
        now=1.0,
    )
    assert provider.commit_pending(1.0)
    np.testing.assert_allclose(provider.goal_positions, targets)
    with pytest.raises(module.ReferenceInputError):
        provider.submit(
            targets,
            np.tile([1.01, 0.0, 0.0, 0.0], (3, 1)),
            sequence=2,
            stamp=2.0,
            now=2.0,
        )
    with pytest.raises(module.ReferenceInputError):
        provider.submit(
            targets,
            np.tile([1.0, 0.0, 0.0, 0.0], (3, 1)),
            body_names=("pelvis", "pelvis", "right_wrist_yaw_link"),
            sequence=2,
            stamp=2.0,
            now=2.0,
        )
    with pytest.raises(module.ReferenceInputError):
        provider.submit(
            targets,
            np.tile([1.0, 0.0, 0.0, 0.0], (3, 1)),
            body_names=(["pelvis"], "left_wrist_yaw_link", "right_wrist_yaw_link"),
            sequence=2,
            stamp=2.0,
            now=2.0,
        )


def test_limits_and_public_metadata_are_immutable_and_sensible():
    """Catch mutable or nonsensical speed/bound settings entering the trajectory path."""
    module = _load_module()
    names, positions, quaternions, joints = _seed()
    provider = module.LiveReferenceProvider(names, positions, quaternions, joints)
    assert provider.body_names == names
    assert provider.step_dt == 0.02
    assert provider.frame_index == 0
    assert provider.last_sequence is None
    with pytest.raises((AttributeError, TypeError)):
        provider.limits.pelvis_speed = 5.0
    bad_limits = replace(provider.limits, wrist_speed=0.0)
    with pytest.raises(module.ReferenceInputError):
        module.LiveReferenceProvider(names, positions, quaternions, joints, limits=bad_limits)
    with pytest.raises(module.ReferenceInputError):
        module.LiveReferenceProvider(names, positions, quaternions, joints, step_dt=0.0)


def test_inputs_goals_and_samples_do_not_alias_their_callers():
    """Catch externally mutable seed, packet, goal, or sampled buffers changing state."""
    module = _load_module()
    names, positions, quaternions, joints = _seed()
    provider = module.LiveReferenceProvider(names, positions, quaternions, joints)
    positions[1] = [9.0, 9.0, 9.0]
    target = np.array([[0.1, 0.0, 0.8], [0.1, 0.2, 1.0], [0.1, -0.2, 1.0]])
    target_quaternions = np.tile([1.0, 0.0, 0.0, 0.0], (3, 1))
    provider.submit(target, target_quaternions, sequence=1, stamp=1.0, now=1.0)
    target[:] = 9.0
    target_quaternions[:] = 0.0
    assert provider.commit_pending(1.0)
    goals = provider.goal_positions
    goals[:] = 8.0
    sample = provider.sample([0])
    sample["body_pos"][:] = 7.0
    sample["joint_pos"][:] = 7.0
    fresh = provider.sample([0])
    np.testing.assert_allclose(fresh["body_pos"][0, 1], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(provider.goal_positions[0], [0.1, 0.0, 0.8])


def test_seed_workspace_offsets_and_packets_are_strictly_validated():
    """Catch invalid VR-3 topology/workspace input or unsupported sampling offsets."""
    module = _load_module()
    names, positions, quaternions, joints = _seed()
    bad_names = list(names)
    bad_names[0], bad_names[1] = bad_names[1], bad_names[0]
    with pytest.raises(module.ReferenceInputError):
        module.LiveReferenceProvider(tuple(bad_names), positions, quaternions, joints)
    bad_seed = positions.copy()
    bad_seed[10] = [1.2, 0.0, 0.8]
    with pytest.raises(module.ReferenceInputError):
        module.LiveReferenceProvider(names, bad_seed, quaternions, joints)

    provider = module.LiveReferenceProvider(names, positions, quaternions, joints)
    with pytest.raises(module.ReferenceInputError):
        provider.submit(
            np.array([[1.01, 0.0, 0.8], [1.01, 0.0, 0.8], [1.01, 0.0, 0.8]]),
            np.tile([1.0, 0.0, 0.0, 0.0], (3, 1)),
            sequence=1,
            stamp=1.0,
            now=1.0,
        )
    with pytest.raises(module.ReferenceInputError):
        provider.sample(np.array([0.0]))
    with pytest.raises(module.ReferenceInputError):
        provider.sample(np.array([33]))
    with pytest.raises(module.ReferenceInputError):
        provider.sample([[0], [1, 2]])


def test_sequence_clock_reset_and_close_keep_the_required_latches():
    """Catch stale queue acceptance, clock faults, reset replay, or post-close input."""
    module = _load_module()
    names, positions, quaternions, joints = _seed()
    provider = module.LiveReferenceProvider(names, positions, quaternions, joints)
    target = np.array([[0.1, 0.0, 0.8], [0.1, 0.2, 1.0], [0.1, -0.2, 1.0]])
    identity = np.tile([1.0, 0.0, 0.0, 0.0], (3, 1))
    provider.submit(target, identity, sequence=1, stamp=1.0, now=1.0)
    with pytest.raises(module.ReferenceInputError) as clock_error:
        provider.commit_pending(0.5)
    assert clock_error.value.clock_fault
    assert provider.commit_pending(1.0)
    with pytest.raises(module.ReferenceInputError):
        provider.submit(target, identity, sequence=True, stamp=2.0, now=2.0)
    with pytest.raises(module.ReferenceInputError):
        provider.submit(target, identity, sequence=1, stamp=2.0, now=2.0)
    with pytest.raises(module.ReferenceInputError) as future_error:
        provider.submit(target, identity, sequence=2, stamp=3.0, now=2.0)
    assert future_error.value.clock_fault

    provider.submit(target, identity, sequence=2, stamp=2.0, now=2.0)
    provider.reset_reference(positions, quaternions, joints)
    assert not provider.commit_pending(2.0)
    with pytest.raises(module.ReferenceInputError):
        provider.submit(target, identity, sequence=2, stamp=2.0, now=2.0)
    provider.close()
    with pytest.raises(module.ReferenceInputError):
        provider.submit(target, identity, sequence=3, stamp=3.0, now=3.0)


def test_preflight_validates_final_xyz_without_touching_pending_or_clocks():
    module = _load_module()
    provider = module.LiveReferenceProvider(*_seed())
    target = provider.goal_positions
    quaternions = provider.goal_quaternions
    provider.submit(target, quaternions, sequence=1, stamp=1., now=1.)
    before = (provider.version, provider.last_sequence, provider.rejected_count)
    candidate = target.copy()
    candidate[:, 2] -= .08
    with pytest.raises(module.ReferenceInputError, match="unit wxyz"):
        provider.validate_targets(candidate, quaternions * 2.)
    checked, checked_q = provider.validate_targets(candidate, quaternions * (1. + 1e-4))
    np.testing.assert_allclose(checked, candidate)
    np.testing.assert_allclose(checked_q, quaternions)
    checked[:] = 99.
    for axis, value in ((0, 2.), (2, 1.4)):
        invalid = target.copy()
        invalid[0, axis] = value
        with pytest.raises(module.ReferenceInputError, match="workspace"):
            provider.validate_targets(invalid, quaternions)
    assert (provider.version, provider.last_sequence, provider.rejected_count) == before
    assert provider.commit_pending(1.)
    np.testing.assert_array_equal(provider.goal_positions, target)
    provider.close()
    with pytest.raises(module.ReferenceInputError, match="closed"):
        provider.validate_targets(candidate, quaternions)


def test_consumer_clock_never_reverses_for_pending_or_empty_commits():
    """Catch a delayed consumer accepting a packet after a later observed GUI clock."""
    module = _load_module()
    names, positions, quaternions, joints = _seed()
    provider = module.LiveReferenceProvider(names, positions, quaternions, joints)
    target = np.array([[0.1, 0.0, 0.8], [0.1, 0.2, 1.0], [0.1, -0.2, 1.0]])
    identity = np.tile([1.0, 0.0, 0.0, 0.0], (3, 1))
    provider.submit(target, identity, sequence=1, stamp=1.0, now=10.0)
    before_goal = provider.goal_positions
    with pytest.raises(module.ReferenceInputError) as pending_clock_error:
        provider.commit_pending(9.0)
    assert pending_clock_error.value.clock_fault
    assert provider.last_sequence is None
    np.testing.assert_allclose(provider.goal_positions, before_goal)
    assert provider.commit_pending(10.0)

    assert not provider.commit_pending(11.0)
    with pytest.raises(module.ReferenceInputError) as empty_clock_error:
        provider.commit_pending(10.0)
    assert empty_clock_error.value.clock_fault
    provider.reset_reference(positions, quaternions, joints)
    with pytest.raises(module.ReferenceInputError) as reset_clock_error:
        provider.commit_pending(10.5)
    assert reset_clock_error.value.clock_fault
    provider.submit(target, identity, sequence=2, stamp=2.0, now=11.0)
    assert provider.commit_pending(11.0)


@pytest.mark.parametrize(
    ("selector", "invalid_value"),
    [
        ("frame", np.array(["env_local"])),
        ("frame", np.array(["env_local", "env_local"])),
        ("frame", ["env_local"]),
        ("quaternion_order", np.array(["wxyz"])),
        ("quaternion_order", ["wxyz"]),
        ("quaternion_order", object()),
    ],
)
def test_non_string_selectors_are_rejected_without_replacing_pending_packet(selector, invalid_value):
    """Catch array truthiness or non-string selectors bypassing atomic packet validation."""
    module = _load_module()
    names, positions, quaternions, joints = _seed()
    provider = module.LiveReferenceProvider(names, positions, quaternions, joints)
    target = np.array([[0.1, 0.0, 0.8], [0.1, 0.2, 1.0], [0.1, -0.2, 1.0]])
    identity = np.tile([1.0, 0.0, 0.0, 0.0], (3, 1))
    provider.submit(target, identity, sequence=1, stamp=1.0, now=1.0)
    kwargs = {selector: invalid_value}
    with pytest.raises(module.ReferenceInputError):
        provider.submit(target, identity, sequence=2, stamp=2.0, now=2.0, **kwargs)
    assert provider.rejected_count == 1
    assert provider.commit_pending(2.0)
    assert provider.last_sequence == 1


def test_explicit_trajectory_commit_consumption_velocity_tail_and_aliasing():
    """The 34th frame supplies velocity while public offsets remain exactly 0..32."""
    module = _load_module()
    names, positions, quaternions, joints = _seed()
    provider = module.LiveReferenceProvider(names, positions, quaternions, joints)
    path_positions, path_quaternions = _trajectory(positions)
    expected_positions = path_positions.copy()
    expected_quaternions = path_quaternions.copy()
    # Frame zero may use the opposite unit-quaternion hemisphere.
    path_quaternions[0] *= -1.0

    provider.submit_trajectory(
        path_positions, path_quaternions, sequence=1, stamp=1.0, now=1.0
    )
    path_positions[:] = 9.0
    path_quaternions[:] = 0.0
    before_commit = provider.sample([0])
    assert provider.commit_pending(1.0)
    np.testing.assert_allclose(provider.sample([0])["body_pos"], before_commit["body_pos"])

    sampled = provider.sample([0, 1, 2, 32])
    np.testing.assert_allclose(sampled["body_pos"][:, [0, 10, 13]], expected_positions[[0, 1, 2, 32]])
    for actual, expected in zip(sampled["body_quat"][:, [0, 10, 13]].reshape(-1, 4),
                                expected_quaternions[[0, 1, 2, 32]].reshape(-1, 4)):
        assert abs(float(np.dot(actual, expected))) == pytest.approx(1.0)
    np.testing.assert_allclose(sampled["body_lin_vel"][:, [0, 10, 13], 1], 0.1)
    assert np.all(np.linalg.norm(sampled["body_ang_vel"][:, [0, 10, 13]], axis=-1) <= 0.2500001)

    sampled["body_pos"][:] = 8.0
    sampled["body_quat"][:] = 8.0
    np.testing.assert_allclose(
        provider.sample([2])["body_pos"][0, [0, 10, 13]], expected_positions[2]
    )
    provider.advance()
    np.testing.assert_allclose(
        provider.sample([0])["body_pos"][0, [0, 10, 13]], expected_positions[1]
    )
    for _ in range(40):
        provider.advance()
    held = provider.sample([0, 1, 32])
    np.testing.assert_allclose(
        held["body_pos"][:, [0, 10, 13]], np.broadcast_to(expected_positions[33], (3, 3, 3))
    )
    np.testing.assert_allclose(held["body_lin_vel"][:, [0, 10, 13]], 0.0)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p, q: p.__setitem__((slice(None), slice(None), 0), np.nan),
        lambda p, q: q.__setitem__((1, 0), [2.0, 0.0, 0.0, 0.0]),
        lambda p, q: p.__setitem__((1, 0), p[0, 0] + [0.005, 0.0, 0.0]),
        lambda p, q: p.__setitem__((1, 1), p[0, 1] + [0.008, 0.0, 0.0]),
        lambda p, q: q.__setitem__((1, 0), _test_axis_angle([0.0, 0.0, 1.0], 0.011)),
        lambda p, q: q.__setitem__((1, 1), _test_axis_angle([0.0, 0.0, 1.0], 0.021)),
        lambda p, q: p.__setitem__((12, 0), [1.01, 0.0, 0.8]),
        lambda p, q: p.__setitem__((12, 1), p[12, 0] + [1.01, 0.0, 0.0]),
        lambda p, q: p.__setitem__((0, 0), p[0, 0] + [0.001, 0.0, 0.0]),
        lambda p, q: q.__setitem__((0, 0), _test_axis_angle([1.0, 0.0, 0.0], 0.001)),
    ],
)
def test_invalid_explicit_trajectory_is_atomic_and_preserves_normal_pending(mutate):
    module = _load_module()
    names, positions, quaternions, joints = _seed()
    provider = module.LiveReferenceProvider(names, positions, quaternions, joints)
    normal_target = positions[[0, 10, 13]].copy()
    normal_target[:, 0] += 0.1
    provider.submit(normal_target, quaternions[[0, 10, 13]], sequence=1, stamp=1.0, now=1.0)
    path_positions, path_quaternions = _trajectory(positions)
    mutate(path_positions, path_quaternions)

    with pytest.raises(module.ReferenceInputError):
        provider.submit_trajectory(
            path_positions, path_quaternions, sequence=2, stamp=2.0, now=2.0
        )
    assert provider.commit_pending(2.0)
    assert provider.last_sequence == 1
    np.testing.assert_allclose(provider.goal_positions, normal_target)


def test_explicit_trajectory_uses_packet_sequence_clock_and_rechecks_frame_zero_at_commit():
    module = _load_module()
    names, positions, quaternions, joints = _seed()
    provider = module.LiveReferenceProvider(names, positions, quaternions, joints)
    path_positions, path_quaternions = _trajectory(positions)
    with pytest.raises(module.ReferenceInputError):
        provider.submit_trajectory(
            path_positions, path_quaternions, sequence=True, stamp=1.0, now=1.0
        )
    with pytest.raises(module.ReferenceInputError) as future_error:
        provider.submit_trajectory(
            path_positions, path_quaternions, sequence=1, stamp=2.0, now=1.0
        )
    assert future_error.value.clock_fault

    moving_goal = positions[[0, 10, 13]].copy()
    moving_goal[:, 0] += 0.1
    provider.submit(moving_goal, quaternions[[0, 10, 13]], sequence=1, stamp=1.0, now=1.0)
    assert provider.commit_pending(1.0)
    with pytest.raises(module.ReferenceInputError, match="sequence"):
        provider.submit_trajectory(
            path_positions, path_quaternions, sequence=1, stamp=2.0, now=2.0
        )
    provider.submit_trajectory(
        path_positions, path_quaternions, sequence=2, stamp=2.0, now=2.0
    )
    provider.advance()
    current_after_advance = provider.sample([0])["body_pos"].copy()
    with pytest.raises(module.ReferenceInputError, match="frame 0"):
        provider.commit_pending(2.0)
    np.testing.assert_allclose(provider.sample([0])["body_pos"], current_after_advance)
    assert provider.last_sequence == 1


def test_stale_pending_trajectory_commit_preserves_active_trajectory_future():
    """A stale B commit must not truncate or replace the already-running trajectory A."""
    module = _load_module()
    names, positions, quaternions, joints = _seed()
    provider = module.LiveReferenceProvider(names, positions, quaternions, joints)
    path_a_positions, path_a_quaternions = _trajectory(positions)
    provider.submit_trajectory(
        path_a_positions, path_a_quaternions, sequence=1, stamp=1.0, now=1.0
    )
    assert provider.commit_pending(1.0)
    provider.advance()

    current = provider.sample([0])
    current_positions = current["body_pos"][0, [0, 10, 13]]
    current_quaternions = current["body_quat"][0, [0, 10, 13]]
    path_b_positions = np.stack(
        [current_positions + [0.001 * frame, 0.0, 0.0] for frame in range(34)]
    )
    path_b_quaternions = np.broadcast_to(current_quaternions, (34, 3, 4)).copy()
    provider.submit_trajectory(
        path_b_positions, path_b_quaternions, sequence=2, stamp=2.0, now=2.0
    )

    # A real command step advances A after B was queued, making only B's anchor stale.
    provider.advance()
    offsets = [0, 1, 5, 16, 32]
    active_a_after_advance = provider.sample(offsets)
    goal_a = provider.goal_positions
    frame_index = provider.frame_index
    version = provider.version
    with pytest.raises(module.ReferenceInputError, match="frame 0"):
        provider.commit_pending(2.0)

    preserved = provider.sample(offsets)
    for key in ("body_pos", "body_quat", "body_lin_vel", "body_ang_vel", "joint_pos", "joint_vel"):
        np.testing.assert_array_equal(preserved[key], active_a_after_advance[key])
    np.testing.assert_array_equal(provider.goal_positions, goal_a)
    assert provider.last_sequence == 1
    assert provider.frame_index == frame_index and provider.version == version

    # A newer ordinary packet can recover by atomically replacing stale pending B.
    replacement = preserved["body_pos"][0, [0, 10, 13]].copy()
    replacement[:, 0] += 0.02
    provider.submit(
        replacement, preserved["body_quat"][0, [0, 10, 13]],
        sequence=3, stamp=3.0, now=3.0,
    )
    assert provider.commit_pending(3.0)
    assert provider.last_sequence == 3


def test_normal_submit_reset_and_close_clear_an_explicit_trajectory():
    module = _load_module()
    names, positions, quaternions, joints = _seed()
    provider = module.LiveReferenceProvider(names, positions, quaternions, joints)
    path_positions, path_quaternions = _trajectory(positions)
    provider.submit_trajectory(
        path_positions, path_quaternions, sequence=1, stamp=1.0, now=1.0
    )
    assert provider.commit_pending(1.0)
    provider.advance()

    normal_target = positions[[0, 10, 13]].copy()
    normal_target[:, 0] -= 0.1
    provider.submit(normal_target, quaternions[[0, 10, 13]], sequence=2, stamp=2.0, now=2.0)
    assert provider.commit_pending(2.0)
    before = provider.sample([0])["body_pos"][0, [0, 10, 13]].copy()
    provider.advance()
    after = provider.sample([0])["body_pos"][0, [0, 10, 13]]
    assert np.all(after[:, 0] < before[:, 0])

    provider.submit_trajectory(
        np.concatenate(([after], _trajectory(positions)[0][1:])),
        path_quaternions,
        sequence=3,
        stamp=3.0,
        now=3.0,
    )
    assert provider.commit_pending(3.0)
    provider.reset_reference(positions, quaternions, joints)
    provider.advance()
    np.testing.assert_allclose(provider.sample([0])["body_pos"][0], positions)
    provider.close()
    with pytest.raises(module.ReferenceInputError):
        provider.advance()
