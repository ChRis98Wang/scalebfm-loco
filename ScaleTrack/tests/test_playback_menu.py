"""Playback selection and index isolation without Isaac Sim."""

from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/pretrain/rsl_rl/playback_menu.py"


@lru_cache
def menu_module():
    spec = importlib.util.spec_from_file_location("playback_menu", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def selection():
    return menu_module().MotionSelection(
        ["ACCAD/walk_left", "BMLrub/jumping", "ACCAD/walk_right"], [2.0, 3.0, 4.0]
    )


def test_live_mode_change_updates_real_mask_then_restarts_the_selected_motion():
    module = menu_module()
    assert hasattr(module, "switch_mode"), "live reference-mode switching is missing"
    command = SimpleNamespace(
        cfg=SimpleNamespace(body_names=["pelvis", "left_wrist", "right_wrist", "torso"], mode_candidates={}),
        _mode_table=torch.ones((1, 4)), _mode=torch.ones((1, 4)),
        num_motion=2, motion_ids=torch.zeros(1, dtype=torch.long), time_steps=torch.ones(1, dtype=torch.long),
        _future_manual_cache={"stale": True}, device="cpu",
    )
    class Env:
        def reset(self):
            command._mode[:] = command._mode_table[0]
            return "fresh observation", {}
    obs = module.switch_mode(Env(), command, "VR-3", ["pelvis", "left_wrist", "right_wrist"], 1)
    assert obs == "fresh observation"
    torch.testing.assert_close(command._mode, torch.tensor([[1., 1., 1., 0.]]))
    assert command.cfg.mode_candidates == {"VR-3": ["pelvis", "left_wrist", "right_wrist"]}
    assert command.motion_ids.item() == 1 and command.time_steps.item() == 0
    assert not command._future_manual_cache


def test_unknown_control_bodies_fail_before_mutating_the_command():
    module = menu_module()
    assert hasattr(module, "switch_mode"), "live reference-mode switching is missing"
    command = SimpleNamespace(cfg=SimpleNamespace(body_names=["pelvis"], mode_candidates={"old": ["pelvis"]}),
                              num_motion=1, _mode_table=torch.ones((1, 1)), device="cpu")
    with pytest.raises(ValueError, match="body"):
        module.switch_mode(None, command, "unknown", ["missing_hand"], 0)
    assert command.cfg.mode_candidates == {"old": ["pelvis"]}


@pytest.mark.parametrize("has_gui,expected", [(True, True), (False, False), (lambda: True, True), (lambda: False, False)])
def test_gui_detection_supports_isaaclab_property_and_legacy_method(has_gui, expected):
    assert menu_module().gui_is_available(SimpleNamespace(has_gui=has_gui)) is expected


@pytest.mark.parametrize("headless,expected", [(False, True), (True, False)])
def test_gui_detection_prefers_actual_kit_launch_config_to_unset_sim_cache(headless, expected):
    assert menu_module().gui_is_available(SimpleNamespace(has_gui=False), {"headless": headless}) is expected


def test_search_is_case_insensitive_and_does_not_change_the_playing_motion():
    browser = selection()
    assert browser.filter("  ACCAD RIGHT ") == (2,)
    assert browser.current_id == 0
    assert browser.consume_request() is None


def test_filtered_selection_uses_original_motion_id_and_is_consumed_once():
    browser = selection()
    browser.filter("jump")
    browser.request_filtered(0)
    assert browser.current_id == 0
    assert browser.consume_request() == 1
    assert browser.consume_request() is None
    browser.mark_playing(1)
    assert browser.current_id == 1


def test_previous_next_wrap_and_rapid_clicks_preserve_navigation_order():
    browser = selection()
    browser.filter("walk")
    browser.request_next(-1)
    assert browser.consume_request() == 2
    browser.mark_playing(2)
    browser.request_next(1)
    browser.request_next(1)
    assert browser.consume_request() == 2


def test_empty_search_cannot_queue_an_invalid_motion():
    browser = selection()
    assert browser.filter("does-not-exist") == ()
    browser.request_next(1)
    assert browser.consume_request() is None
    with pytest.raises(IndexError):
        browser.request_filtered(0)


@pytest.mark.parametrize("names,durations", [([], []), (["a"], []), (["a"], [0]), (["a"], [float('nan')])])
def test_bad_catalogs_are_rejected(names, durations):
    with pytest.raises(ValueError):
        menu_module().MotionSelection(names, durations)


def test_combined_index_resolves_relative_paths_and_removes_only_its_temporary_file(tmp_path):
    first = tmp_path / "train.yaml"
    second = tmp_path / "validation.yaml"
    (tmp_path / "walk.npz").touch()
    (tmp_path / "jump.npz").touch()
    first.write_text("train/walk: walk.npz\n")
    second.write_text("validation/jump: jump.npz\n")
    original = first.read_bytes()
    with menu_module().combined_motion_index([first, second]) as merged:
        merged = Path(merged)
        assert yaml.safe_load(merged.read_text()) == {
            "train/walk": str(tmp_path / "walk.npz"),
            "validation/jump": str(tmp_path / "jump.npz"),
        }
    assert not merged.exists()
    assert first.read_bytes() == original
    assert (tmp_path / "walk.npz").exists()


def test_combined_index_is_cleaned_up_when_simulator_creation_fails(tmp_path):
    motion = tmp_path / "walk.npz"
    motion.touch()
    index = tmp_path / "motions.yaml"
    index.write_text("walk: walk.npz\n")
    with pytest.raises(RuntimeError, match="simulator failed"):
        with menu_module().combined_motion_index([index]) as merged:
            merged = Path(merged)
            raise RuntimeError("simulator failed")
    assert not merged.exists()
    assert index.exists() and motion.exists()


def test_conflicting_names_are_not_silently_overwritten(tmp_path):
    first, second = tmp_path / "a.yaml", tmp_path / "b.yaml"
    (tmp_path / "a.npz").touch()
    (tmp_path / "b.npz").touch()
    first.write_text("same_name: a.npz\n")
    second.write_text("same_name: b.npz\n")
    with pytest.raises(ValueError, match="same_name"):
        with menu_module().combined_motion_index([first, second]):
            pass


def test_missing_motion_is_reported_before_starting_the_environment(tmp_path):
    module = menu_module()
    index = tmp_path / "motions.yaml"
    index.write_text("missing: missing.npz\n")
    with pytest.raises(FileNotFoundError):
        with module.combined_motion_index([index]):
            pass


def test_switch_resets_frame_cache_and_all_envs_before_getting_new_observations():
    command = SimpleNamespace(
        num_motion=3, motion_ids=torch.tensor([0, 0]), time_steps=torch.tensor([8, 9]),
        _future_manual_cache={"old": 1},
    )

    class ResetBoundary:
        def reset(self):
            assert command.time_steps.tolist() == [0, 0]
            assert command._future_manual_cache == {}
            return command.motion_ids.clone(), {}

    obs = menu_module().switch_motion(ResetBoundary(), command, 2)
    assert obs.tolist() == [2, 2]
    assert command.motion_ids.tolist() == [2, 2]


def test_invalid_switch_preserves_current_state():
    command = SimpleNamespace(num_motion=2, motion_ids=torch.tensor([1]), time_steps=torch.tensor([8]))
    with pytest.raises(IndexError):
        menu_module().switch_motion(None, command, 2)
    assert command.motion_ids.tolist() == [1]
    assert command.time_steps.tolist() == [8]
