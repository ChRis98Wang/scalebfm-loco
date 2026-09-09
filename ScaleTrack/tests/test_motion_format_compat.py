"""Regression tests for motion files produced by the official ScaleTrack pipeline."""

import numpy as np
import pytest

from scaletrack.tasks.tracking.mdp import commands


def _versioned_motion_fields(version):
    body_quat = np.zeros((2, 1, 4), dtype=np.float32)
    body_quat[..., 0] = 1.0
    fields = {
        "joint_pos": np.zeros((2, 2), dtype=np.float32),
        "joint_vel": np.zeros((2, 2), dtype=np.float32),
        "body_pos_w": np.zeros((2, 1, 3), dtype=np.float32),
        "body_quat_w": body_quat,
        "body_lin_vel_w": np.zeros((2, 1, 3), dtype=np.float32),
        "body_ang_vel_w": np.zeros((2, 1, 3), dtype=np.float32),
        "fps": np.array(50),
        "format_version": np.array(version),
        "quaternion_order": np.array("wxyz"),
    }
    if version == 3:
        fields.update(
            reference_root_pos=np.zeros((2, 3), dtype=np.float32),
            reference_root_quat_w=body_quat[:, 0].copy(),
            source_sha256=np.array("a" * 64),
            pipeline_fingerprint=np.array("b" * 64),
        )
    return fields


def test_motion_loader_converts_wxyz_for_an_xyzw_runtime(tmp_path, monkeypatch):
    """Catch passing packaged wxyz quaternions to an xyzw runtime unchanged."""
    motion_path = tmp_path / "motion.npz"
    np.savez(
        motion_path,
        joint_pos=np.zeros((1, 2), dtype=np.float32),
        joint_vel=np.zeros((1, 2), dtype=np.float32),
        body_pos_w=np.zeros((1, 1, 3), dtype=np.float32),
        body_quat_w=np.array([[[0.5, 0.1, 0.2, 0.3]]], dtype=np.float32),
        body_lin_vel_w=np.zeros((1, 1, 3), dtype=np.float32),
        body_ang_vel_w=np.zeros((1, 1, 3), dtype=np.float32),
        fps=np.array(50),
    )
    monkeypatch.setattr(commands, "RUNTIME_QUATERNION_ORDER", "xyzw")

    loaded = commands.load_motion_data_worker((0, "motion", str(motion_path), [0]))

    assert loaded is not None
    np.testing.assert_allclose(
        loaded["body_quat_w"][0, 0],
        np.array([0.1, 0.2, 0.3, 0.5], dtype=np.float32),
    )


def test_motion_loader_keeps_wxyz_for_a_wxyz_runtime(tmp_path, monkeypatch):
    """Keep bundled files usable with the upstream pinned wxyz runtime."""
    motion_path = tmp_path / "motion.npz"
    packed_wxyz = np.array([[[0.5, 0.1, 0.2, 0.3]]], dtype=np.float32)
    np.savez(
        motion_path,
        joint_pos=np.zeros((1, 2), dtype=np.float32),
        joint_vel=np.zeros((1, 2), dtype=np.float32),
        body_pos_w=np.zeros((1, 1, 3), dtype=np.float32),
        body_quat_w=packed_wxyz,
        body_lin_vel_w=np.zeros((1, 1, 3), dtype=np.float32),
        body_ang_vel_w=np.zeros((1, 1, 3), dtype=np.float32),
        fps=np.array(50),
    )
    monkeypatch.setattr(commands, "RUNTIME_QUATERNION_ORDER", "wxyz", raising=False)

    loaded = commands.load_motion_data_worker((0, "motion", str(motion_path), [0]))

    assert loaded is not None
    np.testing.assert_array_equal(loaded["body_quat_w"], packed_wxyz)


def test_motion_loader_rejects_metadata_that_declares_xyzw(tmp_path, monkeypatch):
    """Do not silently reorder an archive that explicitly declares another format."""
    motion_path = tmp_path / "motion.npz"
    packed_wxyz = np.array([[[0.5, 0.1, 0.2, 0.3]]], dtype=np.float32)
    np.savez(
        motion_path,
        joint_pos=np.zeros((1, 2), dtype=np.float32),
        joint_vel=np.zeros((1, 2), dtype=np.float32),
        body_pos_w=np.zeros((1, 1, 3), dtype=np.float32),
        body_quat_w=packed_wxyz,
        body_lin_vel_w=np.zeros((1, 1, 3), dtype=np.float32),
        body_ang_vel_w=np.zeros((1, 1, 3), dtype=np.float32),
        fps=np.array(50),
        format_version=np.array(3),
        quaternion_order=np.array("xyzw"),
    )
    monkeypatch.setattr(commands, "RUNTIME_QUATERNION_ORDER", "xyzw")

    loaded = commands.load_motion_data_worker((0, "motion", str(motion_path), [0]))

    assert loaded is None


def test_motion_loader_rejects_fractional_version_and_fps(tmp_path, monkeypatch):
    """Do not truncate malformed scalar metadata into a supported value."""
    common = {
        "joint_pos": np.zeros((1, 2), dtype=np.float32),
        "joint_vel": np.zeros((1, 2), dtype=np.float32),
        "body_pos_w": np.zeros((1, 1, 3), dtype=np.float32),
        "body_quat_w": np.array([[[1.0, 0.0, 0.0, 0.0]]], dtype=np.float32),
        "body_lin_vel_w": np.zeros((1, 1, 3), dtype=np.float32),
        "body_ang_vel_w": np.zeros((1, 1, 3), dtype=np.float32),
        "quaternion_order": np.array("wxyz"),
    }
    fractional_version = tmp_path / "fractional_version.npz"
    np.savez(
        fractional_version,
        **common,
        fps=np.array(50.0),
        format_version=np.array(3.7),
    )
    fractional_fps = tmp_path / "fractional_fps.npz"
    np.savez(
        fractional_fps,
        **common,
        fps=np.array(50.9),
        format_version=np.array(3),
    )
    monkeypatch.setattr(commands, "RUNTIME_QUATERNION_ORDER", "xyzw")

    assert commands.load_motion_data_worker(
        (0, "fractional_version", str(fractional_version), [0])
    ) is None
    assert commands.load_motion_data_worker(
        (1, "fractional_fps", str(fractional_fps), [0])
    ) is None


@pytest.mark.parametrize(
    "missing_field",
    [
        "reference_root_pos",
        "reference_root_quat_w",
        "source_sha256",
        "pipeline_fingerprint",
    ],
)
def test_motion_loader_rejects_incomplete_v3_contract(
    tmp_path, monkeypatch, missing_field
):
    """A v3 label is valid only when the full integrity contract is present."""
    motion_path = tmp_path / f"missing_{missing_field}.npz"
    fields = _versioned_motion_fields(3)
    fields.pop(missing_field)
    np.savez(motion_path, **fields)
    monkeypatch.setattr(commands, "RUNTIME_QUATERNION_ORDER", "xyzw")

    assert commands.load_motion_data_worker(
        (0, motion_path.stem, str(motion_path), [0])
    ) is None


def test_motion_loader_accepts_complete_v3_contract(tmp_path, monkeypatch):
    """Keep valid current archives loadable after tightening validation."""
    motion_path = tmp_path / "v3.npz"
    np.savez(motion_path, **_versioned_motion_fields(3))
    monkeypatch.setattr(commands, "RUNTIME_QUATERNION_ORDER", "xyzw")

    loaded = commands.load_motion_data_worker((0, "v3", str(motion_path), [0]))

    assert loaded is not None


def test_motion_loader_rejects_nan_v3_body_quaternion(tmp_path, monkeypatch):
    """NaN must not make the sign-invariant root comparison fail open."""
    motion_path = tmp_path / "nan_body_quaternion.npz"
    fields = _versioned_motion_fields(3)
    fields["body_quat_w"][0, 0, 0] = np.nan
    np.savez(motion_path, **fields)
    monkeypatch.setattr(commands, "RUNTIME_QUATERNION_ORDER", "xyzw")

    assert commands.load_motion_data_worker(
        (0, "nan_body_quaternion", str(motion_path), [0])
    ) is None


def test_motion_loader_keeps_v2_compatibility(tmp_path, monkeypatch):
    """Version 2 predates v3 root/provenance fields and remains supported."""
    motion_path = tmp_path / "v2.npz"
    np.savez(motion_path, **_versioned_motion_fields(2))
    monkeypatch.setattr(commands, "RUNTIME_QUATERNION_ORDER", "xyzw")

    loaded = commands.load_motion_data_worker((0, "v2", str(motion_path), [0]))

    assert loaded is not None
