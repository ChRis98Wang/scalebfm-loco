"""Tests for safe, versioned ScaleTrack motion archives."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scaletrack.utils import motion_archive


def test_relative_archive_path_preserves_the_input_hierarchy():
    input_dir = Path("/data/retargeted")

    assert motion_archive.relative_archive_path(
        input_dir, input_dir / "a_b/c.pkl"
    ) == Path("a_b/c.npz")
    assert motion_archive.relative_archive_path(
        input_dir, input_dir / "a/b_c.pkl"
    ) == Path("a/b_c.npz")


def test_atomic_archive_contains_the_versioned_quaternion_contract(tmp_path):
    source = tmp_path / "motion.pkl"
    source.write_bytes(b"retargeted-motion")
    output = tmp_path / "nested/motion.npz"

    motion_archive.save_motion_archive(
        output,
        {
            "fps": np.array(50),
            "joint_pos": np.zeros((3, 29)),
            "joint_names": np.array([f"joint_{i}" for i in range(29)]),
            "reference_root_pos": np.zeros((3, 3)),
            "reference_root_quat_w": np.tile([1.0, 0.0, 0.0, 0.0], (3, 1)),
        },
        source_path=source,
        pipeline_fingerprint="a" * 64,
    )

    with np.load(output, allow_pickle=False) as data:
        assert int(data["format_version"]) == motion_archive.FORMAT_VERSION
        assert str(data["quaternion_order"]) == "wxyz"
        assert str(data["source_sha256"]) == motion_archive.sha256_file(source)
        assert str(data["pipeline_fingerprint"]) == "a" * 64
        assert list(data["joint_names"]) == [f"joint_{i}" for i in range(29)]


@pytest.mark.parametrize("names", [["a", "a"], ["a"], ["a", ""], [1, 2]])
def test_archive_rejects_ambiguous_or_wrong_length_names(tmp_path, names):
    source = tmp_path / "motion.pkl"
    source.write_bytes(b"native")
    with pytest.raises(ValueError, match="joint_names"):
        motion_archive.save_motion_archive(tmp_path / "motion.npz", {
            "joint_pos": np.zeros((3, 2)), "joint_names": np.asarray(names),
            "reference_root_pos": np.zeros((3, 3)),
            "reference_root_quat_w": np.tile([1., 0., 0., 0.], (3, 1)),
        }, source_path=source, pipeline_fingerprint="b" * 64)


def test_failed_archive_write_preserves_the_previous_output(tmp_path, monkeypatch):
    source = tmp_path / "motion.pkl"
    source.write_bytes(b"retargeted-motion")
    output = tmp_path / "motion.npz"
    output.write_bytes(b"known-good-archive")
    previous = output.read_bytes()

    def fail_after_partial_write(stream, **_payload):
        stream.write(b"partial")
        raise RuntimeError("injected archive failure")

    monkeypatch.setattr(motion_archive.np, "savez", fail_after_partial_write)

    with pytest.raises(RuntimeError, match="injected archive failure"):
        motion_archive.save_motion_archive(
            output,
            {
                "fps": 50,
                "reference_root_pos": np.zeros((3, 3)),
                "reference_root_quat_w": np.tile(
                    [1.0, 0.0, 0.0, 0.0], (3, 1)
                ),
            },
            source_path=source,
            pipeline_fingerprint="a" * 64,
        )

    assert output.read_bytes() == previous
    assert list(tmp_path.glob(".motion.npz.*.npz")) == []


def test_archive_publish_replaces_a_symlink_without_touching_its_target(tmp_path):
    source = tmp_path / "motion.pkl"
    source.write_bytes(b"retargeted-motion")
    victim = tmp_path / "victim.bin"
    victim.write_bytes(b"preserve-me")
    output = tmp_path / "motion.npz"
    output.symlink_to(victim)

    motion_archive.save_motion_archive(
        output,
        {
            "fps": np.asarray(50),
            "joint_pos": np.zeros((3, 29)),
            "reference_root_pos": np.zeros((3, 3)),
            "reference_root_quat_w": np.tile([1.0, 0.0, 0.0, 0.0], (3, 1)),
        },
        source_path=source,
        pipeline_fingerprint="a" * 64,
    )

    assert not output.is_symlink()
    assert victim.read_bytes() == b"preserve-me"


def test_v3_archive_requires_independent_reference_root_fields(tmp_path):
    source = tmp_path / "motion.pkl"
    source.write_bytes(b"retargeted-motion")

    with pytest.raises(ValueError, match="reference_root"):
        motion_archive.save_motion_archive(
            tmp_path / "motion.npz",
            {"fps": 50, "joint_pos": np.zeros((3, 29))},
            source_path=source,
            pipeline_fingerprint="a" * 64,
        )
