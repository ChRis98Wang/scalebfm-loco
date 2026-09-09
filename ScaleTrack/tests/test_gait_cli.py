"""The physical gait driver must reject invalid runs before AppLauncher."""

from pathlib import Path
import subprocess
import sys

SCRIPT = Path(__file__).with_name("gait_reference_smoke.py")


def run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True, timeout=15)


def test_gait_help_does_not_launch_kit():
    result = run("--help")
    assert result.returncode == 0
    assert "--mode_index {0,7}" in result.stdout
    assert "[ext:" not in result.stdout + result.stderr


def test_gait_missing_inputs_and_unsupported_modes_are_rejected_before_kit(tmp_path):
    result = run("--checkpoint_path", tmp_path / "missing.pt", "--motion_file", tmp_path / "missing.yaml",
                 "--output", tmp_path / "result.json", "--mode_index", 0)
    assert result.returncode == 2 and "must exist" in result.stderr
    assert not (tmp_path / "result.json").exists()
    assert "[ext:" not in result.stdout + result.stderr
    result = run("--checkpoint_path", "missing", "--motion_file", "missing", "--output", "missing",
                 "--mode_index", 1)
    assert result.returncode == 2 and "invalid choice" in result.stderr


def test_existing_report_is_not_overwritten(tmp_path):
    checkpoint, index, report = [tmp_path / name for name in ("trusted.pt", "motion.yaml", "report.json")]
    checkpoint.write_bytes(b"not a model; must not be loaded")
    index.write_text("clip: unused.npz\n")
    report.write_text("existing user evidence")
    result = run("--checkpoint_path", checkpoint, "--motion_file", index, "--output", report, "--mode_index", 7)
    assert result.returncode == 2 and "Refusing to overwrite" in result.stderr
    assert report.read_text() == "existing user evidence"
    assert "[ext:" not in result.stdout + result.stderr


def test_between_clip_reset_stays_inside_inference_mode_for_existing_history():
    import importlib.util
    from types import SimpleNamespace
    import torch
    spec = importlib.util.spec_from_file_location("gait_driver_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with torch.inference_mode():
        history = torch.ones(1, 3, 64)
        ids = torch.zeros(1, dtype=torch.long)
        steps = torch.ones(1, dtype=torch.long)
    command = SimpleNamespace(motion_ids=ids, time_steps=steps, _future_manual_cache={"stale": 1})

    def reset():
        assert torch.is_inference_mode_enabled()
        history.zero_()  # This fails outside inference_mode for a buffer created inside it.
        return {"policy": history}, {}

    env = SimpleNamespace(reset=reset)
    for motion in (0, 1):
        obs, _ = module.initialize_clip(env, command, motion)
        assert int(ids[0]) == motion and int(steps[0]) == 0
        assert not command._future_manual_cache
        assert torch.count_nonzero(obs["policy"]) == 0
