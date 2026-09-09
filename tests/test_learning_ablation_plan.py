"""No simulator required for the fixed learning experiment protocol."""
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_bfm_learning_ablation.py"
spec = importlib.util.spec_from_file_location("learning_ablation_plan", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def value(command, flag):
    return command[command.index(flag) + 1]


def test_plan_is_two_training_jobs_and_three_complete_mask_sets(tmp_path):
    jobs = module.build_jobs("/existing/python", "unit_test", tmp_path)
    assert len(jobs) == 26
    assert [job[0] for job in jobs[:2]] == ["train_fixed", "train_adaptive"]
    for name, deadline, command in jobs[:2]:
        assert deadline == 300
        assert value(command, "--max_iterations") == "5"
        assert value(command, "--num_envs") == "128"
        assert "--test_motion_file" not in command
        assert "agent.eval_during_training=False" in command
    for label in ("official", "fixed", "adaptive"):
        selected = [job for job in jobs if job[0].startswith(f"eval_{label}_")]
        assert {value(job[2], "--mode_index") for job in selected} == set(map(str, range(8)))
        for _, deadline, command in selected:
            assert deadline == 240
            assert value(command, "--num_envs") == "1024"
            assert value(command, "--max_steps") == "1000"
            assert "--mask_metrics" in command
            assert "--target_object" not in command


def indexes():
    return ({f"TRAIN/{i}": f"/training/{i}.npz" for i in range(7174)},
            {f"OLD/{i}": f"/legacy/{i}.npz" for i in range(962)},
            {f"KIT/{i}": f"/kit/{i}.npz" for i in range(789)})


def test_union_preserves_inputs_and_requires_all_heldout_rows():
    train, legacy, kit = indexes()
    union = module.validate_split(train, legacy, kit)
    assert len(union) == 1751
    assert list(union) == list(legacy) + list(kit)
    assert len(train) == 7174 and len(legacy) == 962 and len(kit) == 789


@pytest.mark.parametrize("kind", ["name", "path", "missing", "bad_stratum"])
def test_invalid_or_overlapping_split_is_rejected(kind):
    train, legacy, kit = indexes()
    if kind == "name":
        train["KIT/0"] = train.pop("TRAIN/0")
    elif kind == "path":
        train["TRAIN/0"] = kit["KIT/0"]
    elif kind == "missing":
        kit.pop("KIT/0")
    else:
        kit["WRONG/0"] = kit.pop("KIT/0")
    with pytest.raises(ValueError):
        module.validate_split(train, legacy, kit)


def test_plan_only_and_bad_identifier_never_import_kit():
    output = subprocess.run([sys.executable, str(SCRIPT), "--run-id", "test_only"],
                            capture_output=True, text=True, timeout=10)
    assert output.returncode == 0 and '"train_fixed"' in output.stdout
    assert "[ext:" not in output.stdout + output.stderr
    output = subprocess.run([sys.executable, str(SCRIPT), "--run-id", "../../unsafe"],
                            capture_output=True, text=True, timeout=10)
    assert output.returncode == 2


def training_evidence():
    return {"result": "PASS", "completed_updates": 5, "completed_environment_steps": 40960,
            "resolved": {"schedule": "fixed", "initial_learning_rate": 1e-5},
            "mode_steps": {str(i): 5120 for i in range(8)}, "motion_steps": {"KIT/a": 40960},
            "dataset_steps": {"KIT": 40960}, "changed_actor_parameters": 1,
            "updates": [{"completed_environment_steps": i * 8192} for i in range(1, 6)]}


def test_complete_training_evidence_is_required_even_after_exit_zero():
    module.validate_training_evidence(training_evidence(), "fixed")
    for key, value in (("result", "FAIL"), ("completed_updates", 4), ("changed_actor_parameters", 0),
                       ("dataset_steps", {"KIT": 1}), ("mode_steps", {"one": 40960}), ("updates", [])):
        data = training_evidence()
        data[key] = value
        with pytest.raises(ValueError):
            module.validate_training_evidence(data, "fixed")
