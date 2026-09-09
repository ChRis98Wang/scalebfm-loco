"""Native Kit controls for playback only; import after AppLauncher has started."""

from __future__ import annotations

import omni.ui as ui
from playback_menu import MotionSelection


class MotionPlaybackPanel:
    """Queue UI requests; the playback loop applies them outside simulation callbacks."""

    def __init__(self, names, durations, current_id=0):
        self.selection = MotionSelection(names, durations, current_id)
        self.close_requested = False
        self.combo = None
        self.window = None
        self.search_model = ui.SimpleStringModel("")
        self.mode_names = ()
        self.mode_combo = None
        self._pending_mode = None
        self._pending_target_reset = False
        self._target_label = None
        self._target_enabled = False
        try:
            self._build_ui()
            self._rebuild_choices()
            self.mark_playing(current_id)
        except BaseException:
            self.close()
            raise

    def _build_ui(self):
        self.window = ui.Window("BFM Motion Browser", width=860, height=360, position_x=80, position_y=90)
        self.window.set_visibility_changed_fn(lambda visible: self.request_close() if not visible else None)
        with self.window.frame:
            with ui.VStack(spacing=7):
                ui.Label("G1 reference tracking | choose a clip, then Play selected", height=24)
                with ui.HStack(height=28, spacing=6):
                    ui.Label("Search", width=55)
                    ui.StringField(model=self.search_model, tooltip="Examples: walk, run, jump, crouch, ACCAD")
                    ui.Button("Search", width=80, clicked_fn=self.apply_filter)
                    ui.Button("Clear", width=65, clicked_fn=self.clear_filter)
                self._count_label = ui.Label("", height=20)
                self._choices_frame = ui.Frame(height=30)
                with ui.HStack(height=30, spacing=6):
                    ui.Button("Previous", clicked_fn=self.previous_motion)
                    ui.Button("Play selected", clicked_fn=self.play_selected)
                    ui.Button("Next", clicked_fn=self.next_motion)
                    ui.Button("Restart", clicked_fn=self.selection.request_restart)
                self._current_label = ui.Label("", height=44, word_wrap=True)
                self._progress_label = ui.Label("", height=20)
                self._interaction_frame = ui.Frame(height=0)
                with ui.HStack(height=24, spacing=6):
                    ui.Button("Exit player", width=110, clicked_fn=self.request_close)
                ui.Label("Select a clip to restart it from frame zero. Closing this panel exits playback.",
                         height=28, word_wrap=True)

    def _rebuild_choices(self):
        ids = self.selection.filtered_ids
        self._count_label.text = f"{len(ids)} matching / {len(self.selection.names)} loaded clips"
        labels = [f"{index + 1}: {self.selection.names[index]} ({self.selection.durations[index]:.2f} s)" for index in ids]
        selected = ids.index(self.selection.current_id) if self.selection.current_id in ids else 0
        self._choices_frame.clear()
        with self._choices_frame:
            self.combo = ui.ComboBox(selected, *(labels or ["No matching motions"]), enabled=bool(ids))
        # Browsing a dropdown does not reset the robot until Play selected is pressed.
        # Do not register a reset callback: Kit may emit selection events while building widgets.

    def apply_filter(self):
        self.selection.filter(self.search_model.as_string)
        self._rebuild_choices()

    def clear_filter(self):
        self.search_model.set_value("")
        self.apply_filter()

    def play_selected(self):
        if self.selection.filtered_ids:
            index = self.combo.model.get_item_value_model().as_int
            self.selection.request_filtered(index)

    def next_motion(self):
        self.selection.request_next(1)

    def previous_motion(self):
        self.selection.request_next(-1)

    def mark_playing(self, motion_id):
        self.selection.mark_playing(motion_id)
        name = self.selection.names[motion_id]
        self._current_label.text = f"Playing #{motion_id + 1}: {name}"
        self._current_label.tooltip = name
        ids = self.selection.filtered_ids
        if motion_id in ids:
            self.combo.model.get_item_value_model().set_value(ids.index(motion_id))

    def update_progress(self, elapsed_seconds):
        duration = self.selection.durations[self.selection.current_id]
        self._progress_label.text = f"Clip time: {elapsed_seconds:.2f} / {duration:.2f} s"

    def configure_interaction_controls(self, mode_names, current_mode, *, target_enabled=False):
        self.mode_names = tuple(mode_names)
        if not 0 <= current_mode < len(self.mode_names):
            raise ValueError("Invalid current reference mode")
        self._target_enabled = target_enabled
        self._interaction_frame.clear()
        self._interaction_frame.height = ui.Pixel(120 if target_enabled else 36)
        self.window.height = 490 if target_enabled else 405
        with self._interaction_frame:
            with ui.VStack(spacing=5):
                with ui.HStack(height=28, spacing=6):
                    ui.Label("Reference mode", width=125)
                    self.mode_combo = ui.ComboBox(current_mode, *self.mode_names)
                    ui.Button("Apply mode + restart", width=185, clicked_fn=self.request_selected_mode)
                if target_enabled:
                    self._target_label = ui.Label("Target state: waiting for simulation", height=26)
                    ui.Button("Reset target only", height=26, clicked_fn=self.request_target_reset)
                    ui.Label("Physical box enabled | engineering material values, not calibrated | no grasp objective", height=25)

    def request_selected_mode(self):
        self._pending_mode = self.mode_combo.model.get_item_value_model().as_int

    def consume_mode_request(self):
        result, self._pending_mode = self._pending_mode, None
        return result

    def request_target_reset(self):
        if self._target_enabled:
            self._pending_target_reset = True

    def consume_target_reset(self):
        result, self._pending_target_reset = self._pending_target_reset, False
        return result

    def update_target(self, status):
        if self._target_label is not None:
            x, y, z = status["position_local"]
            self._target_label.text = f"Target local XYZ: ({x:.2f}, {y:.2f}, {z:.2f}) m | root distance: {status['root_distance']:.2f} m"

    def request_close(self):
        self.close_requested = True

    def close(self):
        self.close_requested = True
        if self.window is not None:
            self.window.set_visibility_changed_fn(lambda visible: None)
            self.window.destroy()
            self.window = None
        self.combo = None
        self.mode_combo = None
