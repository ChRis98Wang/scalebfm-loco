"""Selected-link metrics must not reward invisible goals or change frame alignment."""

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


PATH = Path(__file__).resolve().parents[1] / "scripts/pretrain/rsl_rl/masked_metrics.py"


def module():
    assert PATH.is_file(), "masked tracking diagnostics are not implemented"
    spec = importlib.util.spec_from_file_location("masked_metrics_test", PATH)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def command_fixture():
    # Reference ordering [pelvis, wrist, foot] differs from articulation ordering.
    identity = torch.tensor([1., 0., 0., 0.])
    return SimpleNamespace(
        body_pos_w=torch.zeros(2, 3, 3),
        body_quat_w=identity.repeat(2, 3, 1),
        mode=torch.tensor([[1., 0., 0.], [0., 1., 1.]]),
        body_indexes=[2, 0, 1],
        motion_anchor_body_index=0,
        robot_anchor_body_index=2,
        robot=SimpleNamespace(data=SimpleNamespace(
            body_pos_w=torch.tensor([[[9., 0., 0.], [8., 0., 0.], [.2, 0., 0.]],
                                     [[.3, 0., 0.], [.5, 0., 0.], [.1, 0., 0.]]]),
            body_quat_w=identity.repeat(2, 3, 1),
        )),
        # A player can override this property: it must NOT influence diagnostics.
        robot_anchor_pos_w=torch.full((2, 3), 999.),
        metrics={"legacy": torch.tensor([7., 8.])},
    )


def test_metrics_use_only_active_links_and_actual_articulation_root():
    api = module()
    command = command_fixture()
    result = api.masked_tracking_metrics(command)
    assert result["error_active_body_pos_g"].tolist() == pytest.approx([.2, .4])
    assert result["error_active_body_pos_root_relative"].tolist() == pytest.approx([0., .3])
    assert result["error_active_body_pos_g_max"].tolist() == pytest.approx([.2, .5])
    assert result["error_all_body_pos_g_max"].tolist() == pytest.approx([9., .5])
    assert result["error_active_body_rot"].tolist() == pytest.approx([0., 0.])
    assert set(command.metrics) == {"legacy"}


def test_rotation_uses_shortest_quaternion_angle_and_ignores_sign_and_inactive_links():
    api = module()
    command = command_fixture()
    command.robot.data.body_quat_w[0, 2] = torch.tensor([-1., 0., 0., 0.])
    command.robot.data.body_quat_w[1, 0] = torch.tensor([math.sqrt(.5), 0., 0., math.sqrt(.5)])
    command.robot.data.body_quat_w[1, 1] = torch.tensor([0., 0., 0., 1.])
    assert api.masked_tracking_metrics(command)["error_active_body_rot"].tolist() == pytest.approx([0., 3 * math.pi / 4])


@pytest.mark.parametrize("mask", [torch.zeros(2, 3), torch.ones(2, 2), torch.full((2, 3), .5)])
def test_invalid_or_empty_masks_are_rejected(mask):
    api = module()
    command = command_fixture()
    command.mode = mask
    with pytest.raises(ValueError, match="mask"):
        api.masked_tracking_metrics(command)


def test_invalid_quaternion_cannot_look_like_zero_orientation_error():
    api = module()
    command = command_fixture()
    command.robot.data.body_quat_w[0, 2].zero_()
    with pytest.raises(ValueError, match="quaternion"):
        api.masked_tracking_metrics(command)


def test_proxyarray_states_are_unwrapped_before_indexing():
    api = module()
    command = command_fixture()
    command.robot.data.body_pos_w = SimpleNamespace(torch=command.robot.data.body_pos_w)
    command.robot.data.body_quat_w = SimpleNamespace(torch=command.robot.data.body_quat_w)
    assert api.masked_tracking_metrics(command)["error_active_body_pos_g"].tolist() == pytest.approx([.2, .4])


def test_hook_measures_pre_reference_advance_and_restores_on_exception():
    api = module()
    command = command_fixture()
    calls = []
    command._update_metrics = lambda: calls.append("legacy")
    original = command._update_metrics
    with pytest.raises(RuntimeError, match="stop"):
        with api.masked_metrics_context(command, enabled=True) as keys:
            assert "error_active_body_pos_g" in keys
            command._update_metrics()
            # Simulate the following command._update_command(): next reference changes.
            command.body_pos_w += 100
            assert command.metrics["error_active_body_pos_g"].tolist() == pytest.approx([.2, .4])
            raise RuntimeError("stop")
    assert calls == ["legacy"]
    assert command._update_metrics is original
    assert set(command.metrics) == {"legacy"}


def test_disabled_context_does_not_require_or_mutate_a_command():
    api = module()
    command = SimpleNamespace()
    with api.masked_metrics_context(command, enabled=False) as keys:
        assert keys == ()
    assert vars(command) == {}


def test_tracking_threshold_is_max_link_error_not_mean_and_keeps_boundary_inclusive():
    api = module()
    rows = [
        {"motion": "A/one", "metrics": {"error_active_body_pos_g_max": {"max": .5},
                                        "error_all_body_pos_g_max": {"max": .6}}},
        {"motion": "B/two", "metrics": {"error_active_body_pos_g_max": {"max": .51},
                                        "error_all_body_pos_g_max": {"max": .7}}},
    ]
    result = api.summarize_masked_tracking(rows)
    assert result["active_links"]["failed_motions"] == ["B/two"]
    assert result["active_links"]["tracking_pass_rate"] == .5
    assert result["all_links"]["tracking_pass_rate"] == 0.


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_positions_cannot_produce_valid_diagnostics(value):
    api = module()
    command = command_fixture()
    command.robot.data.body_pos_w[0, 2, 0] = value
    with pytest.raises(ValueError, match="finite"):
        api.masked_tracking_metrics(command)
