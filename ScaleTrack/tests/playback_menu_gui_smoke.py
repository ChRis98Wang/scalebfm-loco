"""Run with the existing IsaacLab Python and --viz kit; no robot or training job."""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
panel = None
try:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/pretrain/rsl_rl"))
    from playback_ui import MotionPlaybackPanel

    panel = MotionPlaybackPanel(
        ["ACCAD/walk", "BMLrub/jump", "ACCAD/run"], [2.0, 3.0, 4.0],
        current_id=0,
    )
    for _ in range(5):
        app.update()
    assert hasattr(panel, "configure_interaction_controls"), "mode and target controls are missing"
    panel.configure_interaction_controls(["Pelvis-1", "VR-3", "WholeBody-14"], 2, target_enabled=True)
    panel.mode_combo.model.get_item_value_model().set_value(1)
    panel.request_selected_mode()
    assert panel.consume_mode_request() == 1
    assert panel.consume_mode_request() is None
    panel.request_target_reset()
    assert panel.consume_target_reset() is True
    assert panel.consume_target_reset() is False
    panel.update_target({"position_local": [0.9, -0.7, 0.16], "root_distance": 1.234})
    assert "1.23" in panel._target_label.text
    assert "0.90" in panel._target_label.text
    panel.search_model.set_value("JUMP")
    panel.apply_filter()
    assert panel.selection.filtered_ids == (1,)
    assert panel.selection.current_id == 0
    panel.play_selected()
    assert panel.selection.consume_request() == 1
    panel.mark_playing(1)
    panel.search_model.set_value("")
    panel.apply_filter()
    panel.next_motion()
    assert panel.selection.consume_request() == 2
    panel.previous_motion()
    assert panel.selection.consume_request() == 0
    panel.selection.request_restart()
    assert panel.selection.consume_request() == 1
    panel.update_progress(0.4)
    assert panel._progress_label.text == "Clip time: 0.40 / 3.00 s"
    panel.search_model.set_value("no such motion")
    panel.apply_filter()
    panel.play_selected()
    panel.next_motion()
    assert panel.selection.consume_request() is None
    panel.request_close()
    assert panel.close_requested
    panel.close()
    assert panel.window is None
    panel = None
    original_rebuild = MotionPlaybackPanel._rebuild_choices
    failed_panels = []

    def fail_after_window_creation(self):
        assert self.window is not None
        failed_panels.append(self)
        raise RuntimeError("simulated panel setup failure")

    MotionPlaybackPanel._rebuild_choices = fail_after_window_creation
    try:
        try:
            MotionPlaybackPanel(["walk"], [2.0])
        except RuntimeError as error:
            assert str(error) == "simulated panel setup failure"
        else:
            raise AssertionError("Expected a panel setup failure")
        assert failed_panels[0].window is None
        assert failed_panels[0].close_requested
    finally:
        MotionPlaybackPanel._rebuild_choices = original_rebuild
    print("[BFM MENU TEST] PASS: real Kit UI, search, selection, progress, exit and setup-failure cleanup", flush=True)
except BaseException:
    import traceback
    traceback.print_exc()
    print("[BFM MENU TEST] FAIL", flush=True)
    raise
finally:
    if panel is not None:
        panel.close()
    app.close(exit_code=1 if sys.exc_info()[0] else 0)
