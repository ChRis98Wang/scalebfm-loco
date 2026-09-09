"""The public CPU workflow must not need simulator installs or licensed inputs."""

from pathlib import Path
import re
import shlex

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_cpu_workflow_is_read_only_bounded_and_uses_pinned_actions():
    workflow = yaml.safe_load((ROOT / ".github/workflows/cpu-tests.yml").read_text())
    assert workflow["permissions"] == {"contents": "read"}
    job = workflow["jobs"]["cpu-tests"]
    assert job["runs-on"] == "ubuntu-latest"
    assert job["timeout-minutes"] <= 10
    assert job["strategy"]["matrix"]["python-version"] == ["3.11", "3.12"]
    actions = [step for step in job["steps"] if "uses" in step]
    assert len(actions) == 2
    assert all(re.fullmatch(r"actions/[a-z-]+@[0-9a-f]{40}", step["uses"]) for step in actions)
    assert actions[0]["with"]["persist-credentials"] is False


def test_cpu_workflow_limits_dependencies_and_test_scope():
    requirements = (ROOT / "requirements-ci.txt").read_text().splitlines()
    assert requirements == ["pytest==8.4.2", "PyYAML==6.0.3"]
    workflow = yaml.safe_load((ROOT / ".github/workflows/cpu-tests.yml").read_text())
    commands = "\n".join(step.get("run", "") for step in workflow["jobs"]["cpu-tests"]["steps"])
    assert "python -m pip install -r requirements-ci.txt" in commands
    assert "python -m pytest tests/test_ci_workflow.py tests/test_gui_launcher.py ScaleTrack/tests/test_evaluation_cli.py ScaleTrack/tests/test_evaluation_manifest.py -q" in commands
    assert "bash -n scripts/open_motion_browser.sh" in commands
    assert "git diff --check" in commands
    assert "isaacsim" not in commands and "retargeted_dataset" not in commands


def test_cpu_workflow_runs_mask_report_validation_without_a_simulator():
    workflow = yaml.safe_load((ROOT / ".github/workflows/cpu-tests.yml").read_text())
    selected_tests = set()
    for step in workflow["jobs"]["cpu-tests"]["steps"]:
        for command in step.get("run", "").splitlines():
            args = shlex.split(command)
            if args[:3] == ["python", "-m", "pytest"]:
                selected_tests.update(arg for arg in args[3:] if not arg.startswith("-"))
    assert "tests/test_mask_report_compare.py" in selected_tests


def test_cpu_workflow_runs_umr_ab_pure_contracts_without_licensed_inputs():
    workflow = yaml.safe_load((ROOT / ".github/workflows/cpu-tests.yml").read_text())
    commands = "\n".join(step.get("run", "") for step in workflow["jobs"]["cpu-tests"]["steps"])
    assert "python -m pytest tests/test_umr_behavior_ab_training.py tests/test_umr_behavior_ab_evaluation.py tests/test_umr_behavior_ab_comparison.py -q" in commands
    # The provenance builder and real-run probe have additional local-only deps.
    assert "tests/test_umr_behavior_ab_dataset.py" not in commands
    assert "tests/test_umr_behavior_training_probe.py" not in commands
