"""CPU-only contract tests for the long ScaleBFM evaluation controller."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
DRIVER = ROOT / "scripts/run_bfm_long_evaluation.py"


def load_driver():
    scripts = str(DRIVER.parent)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location("long_evaluation_under_test", DRIVER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def option(command, name):
    assert command.count(name) == 1
    return command[command.index(name) + 1]


def test_jobs_are_interleaved_paired_evaluation_only(tmp_path):
    module = load_driver()
    jobs = module.build_jobs("/python", tmp_path)
    assert [name for name, _, _ in jobs] == [
        f"eval_{label}_{mode}" for mode in range(8) for label in ("official", "candidate")
    ]
    assert all(timeout == 360 for _, timeout, _ in jobs)
    for offset, (name, _, command) in enumerate(jobs):
        mode, label = offset // 2, ("official" if offset % 2 == 0 else "candidate")
        assert command[:2] == ["/python", "-u"]
        assert Path(command[2]).name == "evaluate.py"
        assert option(command, "--task") == "G1-BFM-Transformer-Tracking"
        assert option(command, "--mode_index") == str(mode)
        assert option(command, "--num_envs") == "1024"
        assert option(command, "--max_steps") == "1000"
        assert option(command, "--seed") == "42"
        assert option(command, "--device") == "cuda:0"
        assert option(command, "--motion_file") == str(tmp_path / "heldout_union.yaml")
        assert option(command, "--output") == str(tmp_path / f"eval_{label}_{mode}.json")
        assert option(command, "--checkpoint_path") == str(
            module.OFFICIAL if label == "official" else module.CANDIDATE
        )
        assert command.count("--mask_metrics") == 1
        assert "train.py" not in " ".join(command)
        assert "--resume" not in command and "--execute" not in command


def test_default_cli_is_dry_run_and_creates_nothing(monkeypatch, tmp_path, capsys):
    module = load_driver()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", [str(DRIVER), "--run-id", "fresh_eval"])
    module.main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["output"] == str(tmp_path / "logs/behavior_learning/fresh_eval")
    assert len(payload["jobs"]) == 16 and payload["automatic_promotion"] is False
    assert not (tmp_path / "logs").exists()


@pytest.fixture(scope="module")
def valid_evidence():
    module = load_driver()
    from compare_mask_evaluations import G1_BFM_MODE_PRESETS, MASK_METRIC_KEYS

    sha = "a" * 64
    names = [f"OLD/clip_{i}" for i in range(962)] + [f"KIT/clip_{i}" for i in range(789)]
    rows = []
    for motion_id, name in enumerate(names):
        if motion_id < 98:
            steps, source_frames, truncated = 1000, 1002, True
        elif motion_id < 98 + 193:
            steps, source_frames, truncated = 308, 309, False
        else:
            steps, source_frames, truncated = 307, 308, False
        rows.append({
            "motion_id": motion_id,
            "motion": name,
            "source_frames": source_frames,
            "evaluated_steps": steps,
            "truncated": truncated,
            "metrics": {key: {"mean": 0.1, "max": 0.2} for key in MASK_METRIC_KEYS},
        })
    mode, bodies = G1_BFM_MODE_PRESETS[3]
    report = {
        "schema_version": 4,
        "mode_index": 3,
        "mode": mode,
        "active_body_names": list(bodies),
        "task": "G1-BFM-Transformer-Tracking",
        "num_envs": 1024,
        "max_steps": 1000,
        "seed": 42,
        "device": "cuda:0",
        "scene_variant": "baseline",
        "target_object": None,
        "step_dt": 0.02,
        "checkpoint_sha256": sha,
        "input_manifest_verified_unchanged": True,
        "input_manifest": {"checkpoint": {"sha256": sha}},
        "protocol": {"training_updates": 0, "reset_disturbance": False,
                     "observation_noise": False, "interval_pushes": False},
        "summary": {"num_motions": 1751, "evaluated_steps": 605664, "truncated_motions": 98},
        "motions": rows,
    }
    assert sum(row["evaluated_steps"] for row in rows) == 605664
    return module, report, sha, names


def test_full_schema4_evidence_is_accepted(valid_evidence):
    module, report, sha, names = valid_evidence
    result = module.validate_evaluation_evidence(report, "candidate", 3, sha, names)
    assert result == {"num_motions": 1751, "evaluated_steps": 605664, "truncated_motions": 98}


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda report: report.update(schema_version=3), "protocol"),
        (lambda report: report["protocol"].update(training_updates=1), "zero training"),
        (lambda report: report["protocol"].update(observation_noise=True), "noise settings"),
        (lambda report: report.update(step_dt=0.01), "50 Hz"),
        (lambda report: report["input_manifest"]["checkpoint"].update(sha256="b" * 64), "fingerprint"),
        (lambda report: report["summary"].update(evaluated_steps=605663), "complete declared"),
    ],
)
def test_protocol_summary_and_checkpoint_tampering_is_rejected(valid_evidence, mutation, match):
    module, original, sha, names = valid_evidence
    report = copy.deepcopy(original)
    mutation(report)
    with pytest.raises(ValueError, match=match):
        module.validate_evaluation_evidence(report, "candidate", 3, sha, names)


@pytest.mark.parametrize("kind", ["duplicate_id", "bad_horizon", "wrong_name", "bad_metric"])
def test_per_clip_evidence_is_recomputed_not_trusted_from_summary(valid_evidence, kind):
    module, original, sha, names = valid_evidence
    report = copy.deepcopy(original)
    if kind == "duplicate_id":
        report["motions"][1]["motion_id"] = report["motions"][0]["motion_id"]
    elif kind == "bad_horizon":
        report["motions"][100]["evaluated_steps"] -= 1
    elif kind == "wrong_name":
        report["motions"][100]["motion"] = "KIT/not_indexed"
    else:
        report["motions"][100]["metrics"]["error_active_body_pos_g"]["mean"] = float("nan")
    with pytest.raises(ValueError):
        module.validate_evaluation_evidence(report, "candidate", 3, sha, names)
