"""Evaluation must count real motion steps, never training or auto-reset samples."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/pretrain/rsl_rl/evaluation.py"


def evaluation_module():
    assert MODULE_PATH.is_file(), "independent evaluation helpers are missing"
    spec = importlib.util.spec_from_file_location("bfm_evaluation", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RecordedMetricEnv:
    """A finite trace at the simulator boundary, with no policy/trainer mock."""

    def __init__(self, *, nonfinite=False, unexpected_reset=False):
        self.num_envs = 2
        self.device = "cpu"
        self.command = SimpleNamespace(
            num_motion=3,
            motion_names=["A/one", "B/three", "A/two"],
            time_totals=torch.tensor([2, 4, 3]),
            motion_ids=torch.zeros(2, dtype=torch.long),
            time_steps=torch.zeros(2, dtype=torch.long),
            _future_manual_cache={"stale": True},
            metrics={},
        )
        self.traces = [[0.1], [0.2, 0.4, 0.6], [1.0, 2.0]]
        self.nonfinite = nonfinite
        self.unexpected_reset = unexpected_reset
        self.resets = 0

    def reset(self):
        self.command.time_steps.zero_()
        self.resets += 1
        return torch.zeros((2, 1)), {}

    def step(self, actions):
        assert actions.shape == (2, 1)
        assert not torch.is_grad_enabled()
        values, dones = [], []
        for motion, step in zip(self.command.motion_ids.tolist(), self.command.time_steps.tolist()):
            trace = self.traces[motion]
            # A huge reset sample must never enter this clip's statistics.
            values.append(trace[step] if step < len(trace) else 99.0)
            dones.append(step >= len(trace))
        if self.nonfinite:
            values[0] = float("nan")
        if self.unexpected_reset:
            dones[0] = True
        self.command.metrics = {"error_body_pos_g": torch.tensor(values)}
        self.command.time_steps += 1
        return torch.zeros((2, 1)), None, torch.tensor(dones), {}


def test_evaluation_counts_each_valid_step_and_preserves_motion_identity():
    module = evaluation_module()
    env = RecordedMetricEnv()
    rows = module.evaluate_motion_batches(
        env, lambda obs: torch.zeros_like(obs), env.command, ["error_body_pos_g"], max_steps=1000
    )
    by_name = {row["motion"]: row for row in rows}
    assert set(by_name) == {"A/one", "B/three", "A/two"}
    assert [by_name[name]["evaluated_steps"] for name in env.command.motion_names] == [1, 3, 2]
    assert [by_name[name]["metrics"]["error_body_pos_g"]["mean"] for name in env.command.motion_names] == pytest.approx([0.1, 0.4, 1.5])
    assert by_name["B/three"]["metrics"]["error_body_pos_g"]["max"] == pytest.approx(0.6)
    assert env.resets == 2
    assert not env.command._future_manual_cache


def test_evaluation_horizon_is_explicit_and_marks_truncated_clips():
    module = evaluation_module()
    env = RecordedMetricEnv()
    rows = module.evaluate_motion_batches(
        env, lambda obs: torch.zeros_like(obs), env.command, ["error_body_pos_g"], max_steps=1
    )
    by_name = {row["motion"]: row for row in rows}
    assert all(row["evaluated_steps"] == 1 for row in rows)
    assert not by_name["A/one"]["truncated"]
    assert by_name["B/three"]["truncated"]
    assert by_name["A/two"]["truncated"]
    assert by_name["B/three"]["metrics"]["error_body_pos_g"]["mean"] == pytest.approx(0.2)


@pytest.mark.parametrize("fault,match", [("nonfinite", "finite"), ("unexpected_reset", "reset")])
def test_corrupt_or_reset_samples_cannot_produce_a_successful_report(fault, match):
    module = evaluation_module()
    env = RecordedMetricEnv(**{fault: True})
    with pytest.raises(RuntimeError, match=match):
        module.evaluate_motion_batches(
            env, lambda obs: torch.zeros_like(obs), env.command, ["error_body_pos_g"], max_steps=1000
        )


def test_summary_is_clip_weighted_and_separates_datasets_and_strict_thresholds():
    module = evaluation_module()
    env = RecordedMetricEnv()
    rows = module.evaluate_motion_batches(
        env, lambda obs: torch.zeros_like(obs), env.command, ["error_body_pos_g"], max_steps=1000
    )
    summary = module.summarize_motion_metrics(rows, thresholds=(0.2, 0.5))
    assert summary["num_motions"] == 3
    assert summary["mean_metrics"]["error_body_pos_g"] == pytest.approx(2.0 / 3.0)
    assert summary["thresholds"]["0.5"]["failed_motions"] == ["B/three", "A/two"]
    assert summary["thresholds"]["0.5"]["success_rate"] == pytest.approx(1.0 / 3.0)
    assert summary["by_dataset"]["A"]["mean_metrics"]["error_body_pos_g"] == pytest.approx(0.8)
    assert summary["by_dataset"]["B"]["num_motions"] == 1


def test_optional_scene_metrics_follow_the_same_valid_step_mask():
    module = evaluation_module()
    env = RecordedMetricEnv()
    rows = module.evaluate_motion_batches(
        env, lambda obs: torch.zeros_like(obs), env.command,
        ["error_body_pos_g", "target_height"], max_steps=1000,
        extra_metrics=lambda: {"target_height": env.command.time_steps.float()},
    )
    assert [row["metrics"]["target_height"]["mean"] for row in rows] == pytest.approx([1., 2., 1.5])
    assert "target_height" not in env.command.metrics
