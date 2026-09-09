"""CPU-only tests for fixed waypoint acceptance and diagnostic scenarios."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import importlib.util
import math
from pathlib import Path
import sys

import numpy as np
import pytest


_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/pretrain/rsl_rl/waypoint_scenarios.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("waypoint_scenarios_under_test", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "name,forward,height,scope_fragment",
    [
        ("combined", 0.20, -0.06, "only scenario supporting the combined acceptance claim"),
        ("horizontal_only", 0.20, 0.0, "diagnostic isolation"),
        ("height_only", 0.0, -0.06, "diagnostic isolation"),
    ],
)
def test_scenarios_are_fixed_frozen_records_with_explicit_claim_scope(
    name, forward, height, scope_fragment
):
    module = _load_module()
    scenario = module.waypoint_scenario(name)
    assert scenario.name == name
    assert scenario.forward_m == forward
    assert scenario.height_m == height
    assert scope_fragment in scenario.claim_scope
    if name != "combined":
        assert "not combined waypoint acceptance" in scenario.claim_scope
    with pytest.raises(FrozenInstanceError):
        scenario.forward_m = 9.0


@pytest.mark.parametrize(
    "name,yaw,expected",
    [
        ("combined", 0.0, [1.2, 2.0, 2.94]),
        ("horizontal_only", math.pi / 2.0, [1.0, 2.2, 3.0]),
        ("height_only", -2.3, [1.0, 2.0, 2.94]),
    ],
)
def test_target_applies_fixed_heading_relative_xy_and_height(name, yaw, expected):
    module = _load_module()
    initial = np.array([1.0, 2.0, 3.0])
    result = module.waypoint_scenario(name).target(initial, yaw)
    np.testing.assert_allclose(result, expected, atol=1e-14)
    initial[:] = 8.0
    np.testing.assert_allclose(result, expected, atol=1e-14)


def test_target_returns_an_independent_copy_even_for_zero_horizontal_offset():
    module = _load_module()
    initial = np.array([0.1, -0.2, 0.8])
    result = module.waypoint_scenario("height_only").target(initial, 0.0)
    assert not np.shares_memory(result, initial)
    result[:] = 7.0
    np.testing.assert_array_equal(initial, [0.1, -0.2, 0.8])


@pytest.mark.parametrize("name", [None, 0, True, "", "Combined", "forward", "combined "])
def test_only_canonical_scenario_names_are_supported(name):
    module = _load_module()
    with pytest.raises(ValueError, match="waypoint scenario"):
        module.waypoint_scenario(name)


@pytest.mark.parametrize(
    "initial",
    [[0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, np.nan, 0.0], [0.0, np.inf, 0.0], "xyz", None],
)
def test_target_rejects_invalid_initial_xyz(initial):
    module = _load_module()
    with pytest.raises(ValueError, match="initial_xyz"):
        module.waypoint_scenario("combined").target(initial, 0.0)


@pytest.mark.parametrize("yaw", [True, np.bool_(False), None, "0", np.nan, np.inf, -np.inf])
def test_target_rejects_nonfinite_or_boolean_yaw(yaw):
    module = _load_module()
    with pytest.raises(ValueError, match="yaw"):
        module.waypoint_scenario("combined").target([0.0, 0.0, 0.8], yaw)


def test_public_lookup_has_no_arbitrary_distance_or_height_parameters():
    module = _load_module()
    with pytest.raises(TypeError):
        module.waypoint_scenario("combined", forward_m=0.3)
