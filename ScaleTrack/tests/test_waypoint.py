"""CPU-only tests for bounded waypoint reference generation."""

from dataclasses import fields, replace
import math

import numpy as np
import pytest

from scaletrack.utils.waypoint import WaypointConfig, WaypointFollower


def _assert_status_equal(left, right):
    assert left.keys() == right.keys()
    for key in left:
        if isinstance(left[key], np.ndarray):
            assert np.array_equal(left[key], right[key])
        else:
            assert left[key] == right[key]


@pytest.mark.parametrize("field_name", [field.name for field in fields(WaypointConfig())])
@pytest.mark.parametrize("invalid", [0.0, -1.0, np.inf, np.nan, True])
def test_config_requires_every_value_to_be_strictly_positive_and_finite(field_name, invalid):
    with pytest.raises(ValueError, match=field_name):
        replace(WaypointConfig(), **{field_name: invalid})


def test_start_copies_inputs_wraps_heading_and_rejects_bounds_atomically():
    follower = WaypointFollower()
    target = np.array([0.2, -0.1, 0.05])
    reference = np.array([0.03, 0.04, 0.01])
    follower.start(
        target, math.radians(-179), actual_xyz=[0.0, 0.0, 0.0], actual_yaw=math.radians(179),
        reference_xyz=reference, reference_yaw=math.radians(179),
    )
    target[:] = 9.0
    reference[:] = 9.0
    assert np.array_equal(follower.goal_xyz, [0.2, -0.1, 0.05])
    assert np.array_equal(follower.reference_xyz, [0.03, 0.04, 0.01])
    before = follower.status()

    invalid_calls = [
        lambda: follower.start([0.0], 0.0, actual_xyz=[0.0, 0.0, 0.0], actual_yaw=0.0),
        lambda: follower.start([np.nan, 0.0, 0.0], 0.0, actual_xyz=[0.0, 0.0, 0.0], actual_yaw=0.0),
        lambda: follower.start([0.0, 0.0, 0.0], np.inf, actual_xyz=[0.0, 0.0, 0.0], actual_yaw=0.0),
        lambda: follower.start([0.51, 0.0, 0.0], 0.0, actual_xyz=[0.0, 0.0, 0.0], actual_yaw=0.0),
        lambda: follower.start([0.0, 0.0, 0.151], 0.0, actual_xyz=[0.0, 0.0, 0.0], actual_yaw=0.0),
        lambda: follower.start([0.0, 0.0, 0.0], 0.8, actual_xyz=[0.0, 0.0, 0.0], actual_yaw=0.0),
        lambda: follower.start([0.0, 0.0, 0.0], 0.0, actual_xyz=[0.0, 0.0, 0.0], actual_yaw=0.0,
                               reference_xyz=[0.0, np.inf, 0.0]),
    ]
    for call in invalid_calls:
        with pytest.raises(ValueError):
            call()
        _assert_status_equal(follower.status(), before)


def test_reference_obeys_translation_rate_lead_freeze_and_shortest_yaw_rate():
    follower = WaypointFollower()
    follower.start([0.5, 0.0, 0.0], 0.5, actual_xyz=[0.0, 0.0, 0.0], actual_yaw=0.0)
    first_xyz, first_yaw = follower.step([0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 1.0)
    second_xyz, second_yaw = follower.step([0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 1.0)
    held_xyz, _ = follower.step([-0.05, 0.0, 0.0], 0.0, 0.0, 0.0, 0.5)

    np.testing.assert_allclose(first_xyz, [0.08, 0.0, 0.0])
    np.testing.assert_allclose(second_xyz, [0.10, 0.0, 0.0])
    np.testing.assert_allclose(held_xyz, second_xyz)
    assert np.linalg.norm(second_xyz - first_xyz) <= follower.cfg.speed_m_s + 1e-12
    assert first_yaw == pytest.approx(0.25)
    assert second_yaw == pytest.approx(0.5)

    wrapped = WaypointFollower()
    wrapped.start([0.0, 0.0, 0.0], math.radians(-179), actual_xyz=[0.0, 0.0, 0.0],
                  actual_yaw=math.radians(179), reference_yaw=math.radians(179))
    _, yaw = wrapped.step([0.0, 0.0, 0.0], math.radians(179), 0.0, 0.0, 0.1)
    assert yaw > math.radians(179) or yaw < math.radians(-179)
    assert abs(((yaw - math.radians(179) + math.pi) % (2 * math.pi)) - math.pi) <= 0.025 + 1e-12


def test_xyz_reference_obeys_total_speed_vertical_speed_and_3d_lead_limits():
    follower = WaypointFollower(WaypointConfig(max_lead_m=0.5))
    follower.start([0.4, 0.0, 0.15], 0.0, actual_xyz=[0.0, 0.0, 0.0], actual_yaw=0.0)
    before = follower.reference_xyz
    after, _ = follower.step([0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 1.0)
    assert np.linalg.norm(after - before) <= follower.cfg.speed_m_s + 1e-12
    assert abs(after[2] - before[2]) <= follower.cfg.vertical_speed_m_s + 1e-12
    assert np.linalg.norm(after) <= follower.cfg.max_lead_m + 1e-12

    vertical = WaypointFollower(WaypointConfig(max_lead_m=0.5))
    vertical.start([0.0, 0.0, 0.15], 0.0, actual_xyz=[0.0, 0.0, 0.0], actual_yaw=0.0)
    reference, _ = vertical.step([0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 1.0)
    np.testing.assert_allclose(reference, [0.0, 0.0, 0.04])


def test_small_in_place_goal_settles_but_is_not_reported_as_arrived_immediately():
    follower = WaypointFollower()
    follower.start([0.01, 0.0, 0.01], 0.01, actual_xyz=[0.0, 0.0, 0.0], actual_yaw=0.0)
    assert follower.state == follower.SETTLING
    assert follower.status()["settled_s"] == 0.0
    follower.step([0.01, 0.0, 0.01], 0.01, 0.0, 0.0, 0.5)
    assert follower.state == follower.ARRIVED


def test_horizontal_arrival_with_height_error_neither_settles_nor_stalls_early():
    follower = WaypointFollower(WaypointConfig(stall_time_s=0.5, timeout_s=2.0))
    follower.start([0.2, 0.0, 0.04], 0.0, actual_xyz=[0.0, 0.0, 0.0], actual_yaw=0.0)
    follower.step([0.2, 0.0, 0.0], 0.0, 0.0, 0.0, 0.75)
    status = follower.status()
    assert follower.state == follower.RUNNING
    assert status["distance_m"] == pytest.approx(0.04)
    assert status["horizontal_error_m"] == pytest.approx(0.0)
    assert status["vertical_error_m"] == pytest.approx(0.04)
    assert status["settled_s"] == 0.0


def test_arrival_uses_actual_pose_and_stillness_and_resets_settling():
    follower = WaypointFollower(WaypointConfig(settle_time_s=0.5, stall_time_s=5.0))
    follower.start([0.2, 0.0, 0.0], 0.2, actual_xyz=[0.0, 0.0, 0.0], actual_yaw=0.0)
    follower.step([0.2, 0.0, 0.0], 0.2, 0.08, 0.0, 0.2)
    assert follower.state == follower.SETTLING
    assert follower.status()["settled_s"] == 0.0
    follower.step([0.2, 0.0, 0.0], 0.2, 0.0, 0.0, 0.3)
    assert follower.status()["settled_s"] == pytest.approx(0.3)
    follower.step([0.1, 0.0, 0.0], 0.2, 0.0, 0.0, 0.1)
    assert follower.state == follower.RUNNING
    assert follower.status()["settled_s"] == 0.0
    follower.step([0.2, 0.0, 0.0], 0.2, 0.0, 0.0, 0.5)
    assert follower.state == follower.ARRIVED


def test_reference_reaching_goal_cannot_arrive_while_actual_is_stationary_far_away():
    cfg = WaypointConfig(speed_m_s=0.5, max_lead_m=0.5, stall_time_s=5.0)
    follower = WaypointFollower(cfg)
    follower.start([0.2, 0.0, 0.0], 0.0, actual_xyz=[0.0, 0.0, 0.0], actual_yaw=0.0)
    follower.step([0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 1.0)
    np.testing.assert_allclose(follower.reference_xyz, follower.goal_xyz)
    assert follower.state == follower.RUNNING
    assert follower.status()["distance_m"] == pytest.approx(0.2)


def test_no_step_means_no_physical_clock_or_reference_progress():
    follower = WaypointFollower()
    follower.start([0.2, 0.0, 0.0], 0.1, actual_xyz=[0.0, 0.0, 0.0], actual_yaw=0.0)
    before = follower.status()
    after = follower.status()
    _assert_status_equal(before, after)
    assert after["elapsed_s"] == after["settled_s"] == 0.0
    assert after["linear_speed_m_s"] is None and after["angular_speed_rad_s"] is None


def test_preview_exposes_continuing_33_frame_motion_without_mutation():
    follower = WaypointFollower()
    follower.start([0.3, 0.0, -0.1], 0.5, actual_xyz=[0.0, 0.0, 0.0], actual_yaw=0.0)
    follower.step([0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.02)
    before = follower.status()
    positions, yaws = follower.preview([0.0, 0.0, 0.0], 0.02)

    np.testing.assert_array_equal(positions[0], before["reference_xyz"])
    assert yaws[0] == before["reference_yaw"]
    assert np.linalg.norm(positions[-1] - positions[1]) > 0.0
    deltas = np.diff(positions, axis=0)
    assert np.all(np.linalg.norm(deltas, axis=1) <= follower.cfg.speed_m_s * 0.02 + 1e-12)
    assert np.all(np.abs(deltas[:, 2]) <= follower.cfg.vertical_speed_m_s * 0.02 + 1e-12)
    assert np.all(np.linalg.norm(positions - [0.0, 0.0, 0.0], axis=1) <= follower.cfg.max_lead_m + 1e-12)
    assert np.all(np.abs(np.diff(yaws)) <= follower.cfg.yaw_rate_rad_s * 0.02 + 1e-12)
    _assert_status_equal(follower.status(), before)

    positions[:] = 9.0
    yaws[:] = 9.0
    _assert_status_equal(follower.status(), before)


def test_preview_arrived_holds_and_invalid_states_are_rejected():
    arrived = WaypointFollower(WaypointConfig(settle_time_s=0.1))
    arrived.start([0.01, 0.0, 0.0], 0.0, actual_xyz=[0.0, 0.0, 0.0], actual_yaw=0.0,
                  reference_xyz=[-0.02, 0.0, 0.0])
    assert arrived.state == arrived.SETTLING
    settling_positions, _ = arrived.preview([0.0, 0.0, 0.0], 0.02, steps=1)
    assert settling_positions[1, 0] > settling_positions[0, 0]
    arrived.step([0.01, 0.0, 0.0], 0.0, 0.0, 0.0, 0.1)
    positions, yaws = arrived.preview([8.0, 7.0, 6.0], 0.02, steps=32)
    np.testing.assert_array_equal(positions, np.broadcast_to(arrived.reference_xyz, (33, 3)))
    np.testing.assert_array_equal(yaws, np.full(33, arrived.reference_yaw))

    idle = WaypointFollower()
    with pytest.raises(RuntimeError, match="IDLE"):
        idle.preview([0.0, 0.0, 0.0], 0.02)
    arrived.cancel("cancel terminal task")
    with pytest.raises(RuntimeError, match="CANCELLED"):
        arrived.preview([0.0, 0.0, 0.0], 0.02)

    stalled = WaypointFollower(WaypointConfig(stall_time_s=0.1))
    stalled.start([0.2, 0.0, 0.0], 0.0, actual_xyz=[0.0, 0.0, 0.0], actual_yaw=0.0)
    stalled.step([0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.1)
    with pytest.raises(RuntimeError, match="STALLED"):
        stalled.preview([0.0, 0.0, 0.0], 0.02)

    timed_out = WaypointFollower(WaypointConfig(timeout_s=0.1, stall_time_s=1.0))
    timed_out.start([0.2, 0.0, 0.0], 0.0, actual_xyz=[0.0, 0.0, 0.0], actual_yaw=0.0)
    timed_out.step([0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.1)
    with pytest.raises(RuntimeError, match="TIMED_OUT"):
        timed_out.preview([0.0, 0.0, 0.0], 0.02)


@pytest.mark.parametrize("bad_dt", [0.0, -0.1, np.nan, np.inf, True])
@pytest.mark.parametrize("bad_steps", [-1, 33, 1.5, True])
def test_preview_rejects_invalid_inputs_without_mutation(bad_dt, bad_steps):
    follower = WaypointFollower()
    follower.start([0.2, 0.0, 0.0], 0.0, actual_xyz=[0.0, 0.0, 0.0], actual_yaw=0.0)
    before = follower.status()
    with pytest.raises(ValueError):
        follower.preview([0.0, 0.0, 0.0], bad_dt, steps=0)
    with pytest.raises(ValueError):
        follower.preview([0.0, 0.0, 0.0], 0.02, steps=bad_steps)
    with pytest.raises(ValueError, match="actual_xyz"):
        follower.preview([np.nan, 0.0, 0.0], 0.02, steps=0)
    one_position, one_yaw = follower.preview([0.0, 0.0, 0.0], 0.02, steps=np.int64(0))
    assert one_position.shape == (1, 3) and one_yaw.shape == (1,)
    _assert_status_equal(follower.status(), before)


def test_timeout_stall_and_position_tolerance_stall_exemption():
    timed = WaypointFollower(WaypointConfig(timeout_s=1.0, stall_time_s=10.0))
    timed.start([0.0, 0.0, 0.0], 0.5, actual_xyz=[0.0, 0.0, 0.0], actual_yaw=0.0)
    timed.step([0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 1.0)
    assert timed.state == timed.TIMED_OUT

    stalled = WaypointFollower(WaypointConfig(timeout_s=10.0, stall_time_s=1.0))
    stalled.start([0.3, 0.0, 0.0], 0.0, actual_xyz=[0.0, 0.0, 0.0], actual_yaw=0.0)
    stalled.step([0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.5)
    stalled.step([0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.5)
    assert stalled.state == stalled.STALLED

    rotating = WaypointFollower(WaypointConfig(timeout_s=2.0, stall_time_s=0.5))
    rotating.start([0.01, 0.0, 0.0], 0.5, actual_xyz=[0.0, 0.0, 0.0], actual_yaw=0.0)
    rotating.step([0.0, 0.0, 0.0], 0.0, 0.0, 0.2, 0.75)
    assert rotating.state == rotating.RUNNING
    rotating.step([0.07, 0.0, 0.0], 0.0, 0.0, 0.2, 0.1)
    assert rotating.state == rotating.RUNNING


@pytest.mark.parametrize("bad_dt", [0.0, -0.1, np.nan, np.inf, True])
def test_invalid_dt_does_not_advance_running_state(bad_dt):
    follower = WaypointFollower()
    follower.start([0.2, 0.0, 0.0], 0.0, actual_xyz=[0.0, 0.0, 0.0], actual_yaw=0.0)
    before = follower.status()
    with pytest.raises(ValueError, match="dt"):
        follower.step([0.0, 0.0, 0.0], 0.0, 0.0, 0.0, bad_dt)
    _assert_status_equal(follower.status(), before)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"actual_xyz": [np.nan, 0.0, 0.0]}, {"actual_yaw": np.inf},
        {"linear_speed": True}, {"angular_speed": np.nan},
    ],
)
def test_nonfinite_or_boolean_step_telemetry_is_rejected_without_mutation(kwargs):
    follower = WaypointFollower()
    follower.start([0.2, 0.0, 0.0], 0.0, actual_xyz=[0.0, 0.0, 0.0], actual_yaw=0.0)
    values = dict(actual_xyz=[0.0, 0.0, 0.0], actual_yaw=0.0, linear_speed=0.0, angular_speed=0.0, dt=0.1)
    values.update(kwargs)
    before = follower.status()
    with pytest.raises(ValueError):
        follower.step(**values)
    _assert_status_equal(follower.status(), before)


def test_status_and_step_return_copies_and_cancelled_task_can_restart():
    follower = WaypointFollower()
    follower.start([0.2, 0.0, 0.0], 0.0, actual_xyz=[0.0, 0.0, 0.0], actual_yaw=0.0)
    returned, _ = follower.step([0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.1)
    returned[:] = 8.0
    snapshot = follower.status()
    snapshot["goal_xyz"][:] = 7.0
    snapshot["reference_xyz"][:] = 7.0
    follower.goal_xyz[:] = 6.0
    follower.reference_xyz[:] = 6.0
    assert not np.any(follower.goal_xyz == 7.0)
    assert not np.any(follower.reference_xyz == 7.0)
    follower.cancel("operator pause")
    assert follower.state == follower.CANCELLED and follower.reason == "operator pause"
    held = follower.reference_xyz.copy()
    returned, _ = follower.step([9.0, 9.0, 9.0], np.nan, True, np.nan, False)
    np.testing.assert_array_equal(returned, held)
    follower.start([0.1, 0.0, 0.0], 0.0, actual_xyz=[0.0, 0.0, 0.0], actual_yaw=0.0)
    assert follower.state == follower.RUNNING and follower.reason is None


def test_cancel_before_start_is_explicit_and_leaves_idle_state_safe():
    follower = WaypointFollower()
    before = follower.status()
    with pytest.raises(RuntimeError, match="not been started"):
        follower.cancel("operator pause")
    _assert_status_equal(follower.status(), before)
    with pytest.raises(RuntimeError, match="not been started"):
        follower.step([0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.1)
