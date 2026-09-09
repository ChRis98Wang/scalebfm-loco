"""CPU tests for waypoint actor-observation counterfactual diagnostics."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/pretrain/rsl_rl/waypoint_diagnostics.py"
_ENV_CFG_PATH = (
    Path(__file__).resolve().parents[1]
    / "source/scaletrack/scaletrack/tasks/tracking/tracking_env_cfg.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("waypoint_diagnostics_under_test", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _observation(module, *, batch=2, dtype=torch.float32):
    layout = module.LAYOUT
    task = torch.arange(
        batch * layout.future_frames * layout.raw_dim, dtype=dtype
    ).reshape(batch, layout.future_frames, layout.raw_dim)
    mode = torch.zeros(batch, layout.mode_dim, dtype=dtype)
    mode[:, 0] = 1
    mapping = torch.zeros(batch, layout.raw_dim, dtype=dtype)
    for slot in layout.pelvis_slices.values():
        mapping[:, slot] = 1
    mapping[:, layout.time_slice] = 1
    return {
        "policy": torch.ones(batch, 3, 64, dtype=dtype),
        "policy_task": task,
        "action": torch.ones(batch, 3, 29, dtype=dtype),
        "critic": torch.ones(batch, 3, 266, dtype=dtype),
        "mode": mode,
        "mode_mapping": mapping,
    }


def test_layout_matches_observation_and_actor_task_dimensions_without_magic_offsets():
    module = _load_module()
    description = module.task_layout()
    assert description["raw_task_dim"] == 14 * (3 + 3 + 6 + 6) + 1 == 253
    assert description["actor_task_dim"] == 253 + 14 == 267
    assert description["blocks"] == {
        "target_body_pos": {"width": 3, "pelvis": [0, 3]},
        "target_body_pos_rel": {"width": 3, "pelvis": [42, 45]},
        "target_body_rot": {"width": 6, "pelvis": [84, 90]},
        "target_body_rot_rel": {"width": 6, "pelvis": [168, 174]},
    }
    assert description["time"] == [252, 253]


def test_layout_statically_matches_policy_task_and_mode_mapping_config():
    """Fail without importing Isaac when the production observation schema changes."""
    module = _load_module()
    tree = ast.parse(_ENV_CFG_PATH.read_text())
    classes = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    policy_task = classes["PolicyTaskCfg"]
    terms = [node for node in policy_task.body if isinstance(node, ast.Assign)]
    expected_names = [name for name, _ in module.LAYOUT.blocks] + ["timestamp"]
    assert [node.targets[0].id for node in terms] == expected_names
    for term in terms:
        params = next(keyword.value for keyword in term.value.keywords if keyword.arg == "params")
        assert ast.literal_eval(params)["future_idx"] == [0, 1, 2, 3, 4, -1]

    mapping_class = classes["ModeMappingCfg"]
    mapping_term = next(node for node in mapping_class.body if isinstance(node, ast.Assign))
    params = next(keyword.value for keyword in mapping_term.value.keywords if keyword.arg == "params")
    mapping_params = ast.literal_eval(params)
    assert mapping_params["feature_dims_per_link"] == [width for _, width in module.LAYOUT.blocks]
    assert mapping_params["with_time"] is module.LAYOUT.with_time


def test_counterfactual_clones_every_tensor_and_changes_only_future_pelvis_pose():
    module = _load_module()
    obs = _observation(module, dtype=torch.float64)
    originals = {name: value.clone() for name, value in obs.items()}
    flat = module.stationary_pelvis_counterfactual(obs)

    for name in obs:
        assert flat[name] is not obs[name]
        assert flat[name].dtype == obs[name].dtype
        assert flat[name].device == obs[name].device
    for name, value in obs.items():
        torch.testing.assert_close(value, originals[name])

    expected_task = originals["policy_task"].clone()
    for slot in module.LAYOUT.pelvis_slices.values():
        expected_task[:, 1:, slot] = expected_task[:, :1, slot]
    torch.testing.assert_close(flat["policy_task"], expected_task)
    for name in obs.keys() - {"policy_task"}:
        torch.testing.assert_close(flat[name], originals[name])
    torch.testing.assert_close(
        flat["policy_task"][..., module.LAYOUT.time_slice],
        originals["policy_task"][..., module.LAYOUT.time_slice],
    )

    flat["policy_task"].zero_()
    flat["policy"].zero_()
    for name, value in obs.items():
        torch.testing.assert_close(value, originals[name])


def test_decode_returns_exact_independent_pelvis_blocks_and_time():
    module = _load_module()
    task = _observation(module)["policy_task"]
    decoded = module.decode_pelvis_task(task)
    expected_widths = {
        "target_body_pos": 3,
        "target_body_pos_rel": 3,
        "target_body_rot": 6,
        "target_body_rot_rel": 6,
        "time": 1,
    }
    for name, width in expected_widths.items():
        assert decoded[name].shape == (2, 6, width)
    for name, slot in module.LAYOUT.pelvis_slices.items():
        torch.testing.assert_close(decoded[name], task[..., slot])
    torch.testing.assert_close(decoded["time"], task[..., module.LAYOUT.time_slice])
    decoded["target_body_pos"].zero_()
    assert torch.count_nonzero(task[..., module.LAYOUT.pelvis_slices["target_body_pos"]]) > 0


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda module, obs: obs["mode"].fill_(1), "Pelvis-1 mask"),
        (lambda module, obs: obs["mode_mapping"].zero_(), "mode_mapping"),
        (lambda module, obs: obs["policy_task"].fill_(float("nan")), "finite"),
        (lambda module, obs: obs.__setitem__("policy_task", obs["policy_task"][:, :, :-1]), "policy_task"),
        (lambda module, obs: obs.__setitem__("mode", obs["mode"][:, :-1]), "mode"),
        (lambda module, obs: obs.__setitem__("action", obs["action"][:1]), "batch"),
    ],
)
def test_counterfactual_strictly_rejects_bad_values_shapes_and_masks(mutate, match):
    module = _load_module()
    obs = _observation(module)
    mutate(module, obs)
    with pytest.raises(ValueError, match=match):
        module.stationary_pelvis_counterfactual(obs)


def test_counterfactual_rejects_missing_or_non_tensor_observation_values():
    module = _load_module()
    obs = _observation(module)
    del obs["action"]
    with pytest.raises(ValueError, match="missing"):
        module.stationary_pelvis_counterfactual(obs)
    obs = _observation(module)
    obs["extra"] = np.zeros(1)
    with pytest.raises(TypeError, match="extra"):
        module.stationary_pelvis_counterfactual(obs)


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda obs: obs.__setitem__("policy", obs["policy"][:, :, :-1]), "policy must have shape"),
        (lambda obs: obs.__setitem__("action", obs["action"][:, :-1]), "action must have shape"),
        (lambda obs: obs.__setitem__("extra", torch.tensor(1.0)), "batch dimension"),
        (lambda obs: obs.__setitem__("policy", obs["policy"].to(torch.float64)), "same dtype"),
        (lambda obs: obs.__setitem__("mode", obs["mode"].to(torch.int64)), "floating tensors"),
        (lambda obs: {key: obs.__setitem__(key, value[:0]) for key, value in tuple(obs.items())}, "nonempty"),
    ],
)
def test_counterfactual_rejects_bad_core_shapes_dtypes_and_empty_or_scalar_batches(mutate, match):
    module = _load_module()
    obs = _observation(module)
    mutate(obs)
    with pytest.raises(ValueError, match=match):
        module.stationary_pelvis_counterfactual(obs)


@pytest.mark.parametrize("backend", ["torch", "numpy"])
def test_summary_actions_reports_l2_max_and_rms(backend):
    module = _load_module()
    base = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    flat = np.array([[0.0, 2.0], [1.0, 0.0]], dtype=np.float32)
    if backend == "torch":
        base, flat = torch.from_numpy(base), torch.from_numpy(flat)
    summary = module.summarize_actions(base, flat)
    assert summary == pytest.approx({"l2": np.sqrt(21.0), "max_abs": 4.0, "rms": np.sqrt(21.0 / 4.0)})


@pytest.mark.parametrize(
    "base,flat,error",
    [
        (torch.zeros(1), np.zeros(1), TypeError),
        (np.zeros(1), np.zeros(2), ValueError),
        (torch.empty(0), torch.empty(0), ValueError),
        (np.array([np.nan]), np.zeros(1), ValueError),
    ],
)
def test_summary_actions_rejects_mixed_empty_mismatched_or_nonfinite(base, flat, error):
    module = _load_module()
    with pytest.raises(error):
        module.summarize_actions(base, flat)
