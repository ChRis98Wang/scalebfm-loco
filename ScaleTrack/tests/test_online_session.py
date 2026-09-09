"""Contract tests for the pure online session safety state machine."""

from __future__ import annotations

import numpy as np
import pytest

from scaletrack.utils.live_reference import LiveReferenceProvider
from scaletrack.utils.online_session import OnlineSession


def _seed():
    names = [f"body_{index}" for index in range(14)]
    names[0], names[10], names[13] = "pelvis", "left_wrist_yaw_link", "right_wrist_yaw_link"
    body_pos = np.zeros((14, 3), dtype=np.float64)
    body_pos[0] = [0.0, 0.0, 0.8]
    body_pos[10] = [0.2, 0.2, 1.0]
    body_pos[13] = [0.2, -0.2, 1.0]
    body_quat = np.tile([1.0, 0.0, 0.0, 0.0], (14, 1))
    joints = np.zeros(29, dtype=np.float64)
    return tuple(names), body_pos, body_quat, joints


def _session():
    names, body_pos, body_quat, joints = _seed()
    return OnlineSession(LiveReferenceProvider(names, body_pos, body_quat, joints)), body_pos, body_quat, joints


def test_stale_heartbeat_latches_fault_while_unchanged_goal_remains_valid():
    """Catch treating a UI no-edit as stale or allowing a heartbeat to clear a latch."""
    session, body_pos, body_quat, joints = _session()
    session.heartbeat(1.0, 1.0)
    session.activate(1.0, body_pos, body_quat, joints)

    assert session.check(1.4, body_pos, body_quat, joints, np.zeros(29))
    assert not session.check(1.6, body_pos, body_quat, joints, np.zeros(29))
    assert session.state == "PAUSED_FAULT"
    assert "heartbeat" in session.reason

    session.heartbeat(1.7, 1.7)
    assert session.state == "PAUSED_FAULT"


def test_resume_requires_fresh_healthy_real_state_and_never_fabricates_fall_recovery():
    """Catch an invalid resume silently entering ACTIVE after a real pelvis fall."""
    session, body_pos, body_quat, joints = _session()
    session.heartbeat(1.0, 1.0)
    session.activate(1.0, body_pos, body_quat, joints)
    fallen = body_pos.copy()
    fallen[0, 2] = 0.34

    assert not session.check(1.1, fallen, body_quat, joints, np.zeros(29))
    assert session.state == "PAUSED_FAULT"
    session.heartbeat(1.2, 1.2)

    with pytest.raises(RuntimeError, match="fall"):
        session.resume(1.2, body_pos, body_quat, joints)
    assert session.state == "PAUSED_FAULT"


def test_pause_is_latched_until_an_explicit_valid_resume_reseeds_reference():
    """Catch calls to check or heartbeat implicitly changing a user-paused session."""
    session, body_pos, body_quat, joints = _session()
    session.heartbeat(1.0, 1.0)
    session.activate(1.0, body_pos, body_quat, joints)
    session.pause()
    assert not session.check(1.1, body_pos, body_quat, joints, np.zeros(29))
    with pytest.raises(RuntimeError, match="fresh"):
        session.resume(1.6, body_pos, body_quat, joints)
    assert session.state == "PAUSED_USER"
    session.heartbeat(1.7, 1.7)
    session.resume(1.7, body_pos, body_quat, joints)
    assert session.state == "ACTIVE"
    assert session.provider.frame_index == 0
