"""The user-facing GUI command must be inspectable without starting a simulator."""

import os
from pathlib import Path
import shlex
import subprocess
import pytest


LAUNCHER = Path(__file__).resolve().parents[1] / "scripts/open_motion_browser.sh"


def launch(*args, **overrides):
    env = os.environ.copy()
    env.update(DISPLAY=":99", BFM_ISAAC_PYTHON="/missing IsaacLab/python")
    env.update(overrides)
    return subprocess.run(["bash", str(LAUNCHER), *args], env=env, capture_output=True, text=True, timeout=5)


def test_help_does_not_require_isaaclab_or_a_display():
    result = launch("--help", DISPLAY="")
    assert result.returncode == 0
    assert "--dry-run" in result.stdout
    assert "BFM_ISAAC_PYTHON" in result.stdout


def test_dry_run_keeps_paths_with_spaces_and_bounds_own_process_group():
    result = launch("--dry-run", BFM_INITIAL_MOTION="ACCAD/walk left")
    assert result.returncode == 0
    command = shlex.split(result.stdout.strip().splitlines()[-1])
    assert command[:2] == ["systemd-run", "--user"]
    assert "--unit=bfm-motion-browser" in command
    assert "--property=KillMode=control-group" in command
    assert "--property=RuntimeMaxSec=30min" in command
    assert "--property=Restart=no" in command
    assert "--wait" in command and "--pipe" in command
    assert "--setenv=DISPLAY=:99" in command
    assert "/missing IsaacLab/python" in command
    assert command[command.index("--initial_motion") + 1] == "ACCAD/walk left"
    assert command[command.index("--num_envs") + 1] == "1"
    assert command[command.index("--viz") + 1] == "kit"
    assert "--motion_menu" in command


def test_missing_display_fails_before_starting_a_unit():
    result = launch(DISPLAY="")
    assert result.returncode != 0
    assert "DISPLAY" in result.stderr


def test_missing_interpreter_fails_before_starting_a_unit():
    result = launch()
    assert result.returncode != 0
    assert "BFM_ISAAC_PYTHON" in result.stderr


def test_unknown_flags_are_not_silently_ignored():
    result = launch("--typo")
    assert result.returncode == 2
    assert "--typo" in result.stderr


def test_target_object_is_explicit_and_can_select_sparse_mode():
    result = launch("--dry-run", "--target-object", BFM_MODE_INDEX="2", BFM_TARGET_CONFIG="/tmp/box material.json")
    assert result.returncode == 0
    command = shlex.split(result.stdout.strip().splitlines()[-1])
    assert "--target_object" in command
    assert command[command.index("--target_config") + 1] == "/tmp/box material.json"
    assert command[command.index("--mode_index") + 1] == "2"
    plain = shlex.split(launch("--dry-run").stdout.strip().splitlines()[-1])
    assert "--target_object" not in plain


def test_target_config_relative_path_keeps_the_callers_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = launch("--dry-run", "--target-object", BFM_TARGET_CONFIG="box material.json")
    assert result.returncode == 0
    command = shlex.split(result.stdout.strip().splitlines()[-1])
    assert command[command.index("--target_config") + 1] == str(tmp_path / "box material.json")


def test_online_targets_dry_run_is_explicit_vr3_kit_and_never_adds_a_box():
    """Break caught: online opt-in silently changes offline defaults or enables object reset controls."""
    result = launch("--dry-run", "--online-targets")
    assert result.returncode == 0
    command = shlex.split(result.stdout.strip().splitlines()[-1])
    assert "--online_targets" in command
    assert command[command.index("--mode_index") + 1] == "2"
    assert "--target_object" not in command


@pytest.mark.parametrize("mode", ["0", "1", "2"])
def test_online_launcher_honors_supported_initial_mode(mode):
    result = launch("--dry-run", "--online-targets", BFM_MODE_INDEX=mode)
    assert result.returncode == 0
    command = shlex.split(result.stdout.strip().splitlines()[-1])
    assert command[command.index("--mode_index") + 1] == mode


@pytest.mark.parametrize("mode", ["3", "7", "-1", "typo"])
def test_online_launcher_rejects_unsupported_mode_before_creating_unit(mode):
    result = launch("--dry-run", "--online-targets", BFM_MODE_INDEX=mode)
    assert result.returncode == 2 and "BFM_MODE_INDEX" in result.stderr
