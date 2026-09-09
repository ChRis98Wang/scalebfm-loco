"""CLI preflight must not launch Kit for help or invalid input."""

from pathlib import Path
import subprocess
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/pretrain/rsl_rl/evaluate.py"


def test_evaluation_help_does_not_start_the_simulator():
    result = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0
    assert "--checkpoint_path" in result.stdout
    assert "--output" in result.stdout
    assert "--target_object" in result.stdout
    assert "--mask_metrics" in result.stdout
    assert "Simulation App" not in result.stdout


def test_evaluation_rejects_missing_checkpoint_before_simulator_start(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--checkpoint_path", str(tmp_path / "missing.pt"),
         "--motion_file", str(tmp_path / "missing.yaml"), "--output", str(tmp_path / "out.json")],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 2
    assert "checkpoint" in result.stderr
    assert "Simulation App" not in result.stdout
    assert not (tmp_path / "out.json").exists()


def test_evaluation_refuses_existing_report_without_touching_it_or_starting_kit(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"not loaded")
    index = tmp_path / "motions.yaml"
    index.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "existing.json"
    output.write_bytes(b"previous results\n")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--checkpoint_path", str(checkpoint),
         "--motion_file", str(index), "--output", str(output)],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 2
    assert "overwrite" in result.stderr
    assert "Simulation App" not in result.stdout
    assert output.read_bytes() == b"previous results\n"


def test_invalid_target_position_fails_before_loading_models_or_starting_kit(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--checkpoint_path", str(tmp_path / "absent.pt"),
         "--motion_file", str(tmp_path / "absent.yaml"), "--output", str(tmp_path / "out.json"),
         "--target_object", "--target_position", "1", "0", "-1"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 2
    assert "resting height" in result.stderr
    assert "Simulation App" not in result.stdout
