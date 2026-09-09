import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("long_training_plan", ROOT / "scripts/run_bfm_long_training.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_long_command_keeps_holdout_out_and_preserves_original_checkpoint(tmp_path):
    command = module.training_command("/existing/python", "new_long", tmp_path, 1000)
    assert command[command.index("--max_iterations") + 1] == "1000"
    assert command[command.index("--expected_updates") + 1] == "1000"
    assert "--test_motion_file" not in command
    for option in ("agent.algorithm.schedule=fixed", "agent.motion_sampling_strategy=coverage",
                   "agent.motion_resample_interval=5", "agent.save_interval=50", "agent.eval_during_training=False"):
        assert option in command


def evidence():
    groups = [{f"KIT/{i}": 64 for i in range(128)}, {f"KIT/{i}": 64 for i in range(128, 256)}]
    return {"result": "PASS", "completed_updates": 6, "completed_environment_steps": 49152,
            "resolved": {"motion_sampling_strategy": "coverage", "motion_resample_interval": 5},
            "updates": [{"completed_environment_steps": (i + 1) * 8192, "motion_steps": groups[i // 5]}
                        for i in range(6)],
            "motion_steps": {**{key: value * 5 for key, value in groups[0].items()}, **groups[1]}}


def test_real_rotation_requires_distinct_second_cohort_and_completed_steps():
    audit = evidence()
    result = module.validate_coverage_audit(audit, 6, set(audit["motion_steps"]))
    assert result["actual_unique_motions"] == 256
    audit["updates"][5]["motion_steps"] = audit["updates"][0]["motion_steps"]
    with pytest.raises(ValueError, match="disjoint"):
        module.validate_coverage_audit(audit, 6, set(audit["motion_steps"]))


def test_dry_run_never_starts_simulator_or_training():
    result = subprocess.run([sys.executable, str(ROOT / "scripts/run_bfm_long_training.py"),
                             "--run-id", "dry_only"], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0 and "--max_iterations" in result.stdout
    assert "[ext:" not in result.stdout + result.stderr
