"""Tests for collision-free motion names in generated YAML indexes."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import yaml


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/pretrain/data_process/create_yaml.py"
)


def _load_create_yaml_module():
    spec = importlib.util.spec_from_file_location("create_yaml", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_nested_motions_with_the_same_stem_get_distinct_names(tmp_path):
    data_dir = tmp_path / "motions"
    first = data_dir / "set_a/walk.npz"
    second = data_dir / "set_b/walk.npz"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    np.savez(first, fps=50)
    np.savez(second, fps=50)
    output = tmp_path / "motions.yaml"

    module = _load_create_yaml_module()
    module.create_yaml(str(data_dir), str(output))

    entries = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert entries == {
        "set_a/walk": str(first),
        "set_b/walk": str(second),
    }


def test_explicit_motion_list_excludes_stale_files(tmp_path):
    data_dir = tmp_path / "motions"
    expected = data_dir / "active/walk.npz"
    stale = data_dir / "removed/old.npz"
    expected.parent.mkdir(parents=True)
    stale.parent.mkdir(parents=True)
    np.savez(expected, fps=50)
    np.savez(stale, fps=50)
    output = tmp_path / "motions.yaml"

    module = _load_create_yaml_module()
    module.create_yaml(
        str(data_dir),
        str(output),
        motion_files=[str(expected)],
    )

    entries = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert entries == {"active/walk": str(expected)}


def test_relative_data_dir_accepts_an_absolute_selected_file(tmp_path, monkeypatch):
    data_dir = tmp_path / "motions"
    motion = data_dir / "walk.npz"
    data_dir.mkdir()
    np.savez(motion, fps=50)
    output = tmp_path / "motions.yaml"
    monkeypatch.chdir(tmp_path)

    module = _load_create_yaml_module()
    module.create_yaml("motions", str(output), motion_files=[str(motion)])

    entries = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert entries == {"walk": str(motion)}


def test_failed_yaml_write_preserves_the_previous_index(tmp_path, monkeypatch):
    data_dir = tmp_path / "motions"
    motion = data_dir / "walk.npz"
    data_dir.mkdir()
    np.savez(motion, fps=50)
    output = tmp_path / "motions.yaml"
    output.write_text("known: good\n", encoding="utf-8")

    def fail_after_partial_write(_motions, stream, **_options):
        stream.write("partial")
        raise RuntimeError("injected yaml failure")

    module = _load_create_yaml_module()
    monkeypatch.setattr(module.yaml, "safe_dump", fail_after_partial_write)

    with pytest.raises(RuntimeError, match="injected yaml failure"):
        module.create_yaml(str(data_dir), str(output))

    assert output.read_text(encoding="utf-8") == "known: good\n"
    assert list(tmp_path.glob(".motions.yaml.*.tmp")) == []


def test_yaml_publish_replaces_a_symlink_without_touching_its_target(tmp_path):
    data_dir = tmp_path / "motions"
    motion = data_dir / "walk.npz"
    data_dir.mkdir()
    np.savez(motion, fps=50)
    victim = tmp_path / "victim.yaml"
    victim.write_text("preserve: me\n", encoding="utf-8")
    output = tmp_path / "motions.yaml"
    output.symlink_to(victim)

    module = _load_create_yaml_module()
    module.create_yaml(str(data_dir), str(output))

    assert not output.is_symlink()
    assert victim.read_text(encoding="utf-8") == "preserve: me\n"
