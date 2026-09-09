"""Tests for package-stage CLI constraints and bounded batch selection."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/pretrain/data_process/motion_processing.py"
)


def _load_motion_processing_module():
    spec = importlib.util.spec_from_file_location("motion_processing", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_package_output_fps_matches_the_training_loader_contract():
    module = _load_motion_processing_module()
    assert module.scale_track_fps("50") == 50
    with pytest.raises(argparse.ArgumentTypeError, match="only supports 50 FPS"):
        module.scale_track_fps("60")


def test_only_one_requested_motion_batch_is_selected_at_a_time():
    module = _load_motion_processing_module()
    motions = [f"motion-{index}.pkl" for index in range(10)]

    assert module.select_motion_batch(motions, batch_index=0, batch_size=3) == motions[:3]
    assert module.select_motion_batch(motions, batch_index=2, batch_size=3) == motions[6:9]
    assert module.select_motion_batch(motions, batch_index=3, batch_size=3) == motions[9:]
    assert module.select_motion_batch(motions, batch_index=4, batch_size=3) == []
