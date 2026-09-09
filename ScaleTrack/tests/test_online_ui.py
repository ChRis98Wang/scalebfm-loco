"""Behavioral tests for the local target-panel boundary."""

from __future__ import annotations

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/pretrain/rsl_rl"))
from online_ui import OnlineTargetPanel
from scaletrack.utils.live_reference import VR3_BODY_NAMES


def test_apply_is_a_complete_copied_wxyz_snapshot_and_edits_coalesce():
    """Break caught: partial or caller-owned target packets reach the controller."""
    clock = iter((1.0, 1.1, 1.2))
    panel = OnlineTargetPanel(now=lambda: next(clock), subscribe_updates=lambda callback: None)
    positions = np.array([[0.0, 0.0, 0.8], [0.2, 0.1, 0.8], [0.2, -0.1, 0.8]])
    degrees = np.array([[0.0, 0.0, 90.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    panel.apply(positions, degrees)
    positions[:] = 99.0
    panel.apply(np.array([[0.01, 0.0, 0.8], [0.2, 0.1, 0.8], [0.2, -0.1, 0.8]]), degrees)
    request = panel.consume_request()

    assert request["kind"] == "goal"
    assert request["mode_name"] == "VR-3" and request["mode_epoch"] == 0
    assert request["body_names"] == VR3_BODY_NAMES
    assert request["positions"][0, 0] == 0.01
    assert np.allclose(request["quaternions"][0], [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
    assert panel.consume_request() is None


def test_heartbeat_is_produced_by_subscription_and_close_unsubscribes():
    """Break caught: consumer calls fabricate producer liveness or leak subscriptions."""
    callbacks = []

    class Handle:
        closed = False

        def unsubscribe(self):
            self.closed = True

    handle = Handle()
    panel = OnlineTargetPanel(now=lambda: 4.0, subscribe_updates=lambda callback: (callbacks.append(callback), handle)[1])
    assert panel.heartbeat_stamp is None
    callbacks[0]()
    assert panel.heartbeat_stamp == 4.0
    panel.close()
    assert handle.closed
    assert panel.close_requested


def test_native_widget_apply_and_priority_lifecycle_keep_close_sticky():
    """Break caught: production buttons do not submit complete goals or a goal erases Pause/Close."""
    class Model:
        def __init__(self, value=0.0): self.value = value
        def get_value_as_float(self): return self.value
        def set_value(self, value): self.value = value
    class Context:
        def __init__(self, *_args, **kwargs): self.kwargs = kwargs
        def __enter__(self): return self
        def __exit__(self, *args): return False
    class Window(Context):
        def __init__(self, *_args, **kwargs): super().__init__(**kwargs); self.visible = True; self.frame = Context(); self.destroyed = False; self.visibility_callback = None
        def destroy(self): self.destroyed = True
        def set_visibility_changed_fn(self, callback): self.visibility_callback = callback
    class Ui:
        def __init__(self): self.buttons = []; self.button_kwargs = {}
        def Fraction(self, value): return value
        def FloatField(self, *_args, **_kwargs): return type("Field", (), {"model": Model()})()
        def Label(self, *_args, **_kwargs): return type("Label", (), {"text": ""})()
        def Button(self, label, clicked_fn, **kwargs):
            button = type("Button", (), {"enabled": True})()
            self.buttons.append((label, clicked_fn)); self.button_kwargs[label] = kwargs
            return button
    ui = Ui()
    ui.Window = Window; ui.VStack = Context; ui.HStack = Context; ui.ScrollingFrame = Context
    panel = OnlineTargetPanel(now=lambda: 1.0, subscribe_updates=lambda callback: None, ui_module=ui,
                              marker_factory=lambda: type("Marker", (), {"set_visibility": lambda self, visible: None, "visualize": lambda self, positions, quaternions: None})())
    panel.position_models[0][2].set_value(0.8)
    panel.position_models[1][0].set_value(0.2); panel.position_models[1][1].set_value(0.1); panel.position_models[1][2].set_value(0.8)
    panel.position_models[2][0].set_value(0.2); panel.position_models[2][1].set_value(-0.1); panel.position_models[2][2].set_value(0.8)
    next(callback for label, callback in ui.buttons if label == "Apply")()
    panel.request_pause(); panel.apply(np.array([[.01, 0, .8], [.2,.1,.8], [.2,-.1,.8]]), np.zeros((3,3)))
    assert panel.consume_request()["kind"] == "pause"
    assert panel.consume_request()["kind"] == "goal"
    panel.request_close(); panel.request_resume(); panel.apply(np.zeros((3, 3)), np.zeros((3, 3)))
    assert panel.close_requested and panel.consume_request()["kind"] == "close"
    panel.update_status({"awaiting_goal": True, "input_age": 0.123456})
    assert panel._labels["awaiting_goal"].text == "awaiting_goal: True"
    assert panel._labels["input_age"].text == "input_age: 0.1235"
    assert panel.window.kwargs == {"width": 600, "height": 940, "position_x": 20, "position_y": 60}
    assert [label for label, _ in ui.buttons] == ["Pelvis-1", "UMI-2", "VR-3", "Go", "Stop",
                                                    "Apply", "Enable", "Pause", "Resume", "Exit"]
    assert all(kwargs["height"] == 30 for kwargs in ui.button_kwargs.values())
    assert panel._labels["mode"].text == "mode: VR-3 (epoch 0)"
    panel.close()
    assert ui.buttons and panel.window.destroyed and panel.window.visibility_callback is None


def test_safety_consumer_preserves_non_safety_requests_and_pause_priority():
    """Break caught: a same-pump Resume/Enable erases Pause or the safety poll consumes other work."""
    panel = OnlineTargetPanel(now=lambda: 1.0, subscribe_updates=lambda callback: None)
    positions = np.array([[0.0, 0.0, 0.8], [0.2, 0.1, 0.8], [0.2, -0.1, 0.8]])

    panel.request_enable()
    assert panel.consume_safety_request() is None
    assert panel.consume_request()["kind"] == "enable"
    panel.apply(positions, np.zeros((3, 3)))
    assert panel.consume_safety_request() is None
    assert panel.consume_request()["kind"] == "goal"

    panel.request_enable()
    panel.request_pause()
    panel.request_resume()
    panel.request_enable()
    assert panel.consume_safety_request() == {"kind": "pause"}
    assert panel.consume_request() is None

    panel.request_resume()
    panel.request_close()
    panel.request_pause()
    assert panel.consume_safety_request() == {"kind": "close"}
    assert panel.close_requested and panel.consume_request() is None


def test_mode_request_pauses_first_coalesces_and_confirmed_mode_discards_stale_goal():
    """Break caught: UI confirms a mode itself or an old-mode goal survives controller confirmation."""
    panel = OnlineTargetPanel(now=lambda: 1.0, subscribe_updates=lambda callback: None)
    positions = np.array([[0.0, 0.0, 0.8], [0.2, 0.1, 0.8], [0.2, -0.1, 0.8]])
    panel.apply(positions, np.zeros((3, 3)))
    panel.request_mode("Pelvis-1")
    panel.request_mode("UMI-2")

    assert panel.mode_name == "VR-3" and panel.mode_epoch == 0
    assert panel.consume_safety_request() == {"kind": "pause"}
    assert panel.consume_request() == {"kind": "mode", "mode_name": "UMI-2"}
    panel.set_mode("UMI-2", 4)
    assert panel.consume_request() is None
    panel.apply(positions, np.zeros((3, 3)))
    request = panel.consume_request()
    assert request["mode_name"] == "UMI-2" and request["mode_epoch"] == 4
    panel.apply(positions, np.zeros((3, 3)))
    panel.set_mode("VR-3", 5)
    assert panel.consume_request() is None


def test_waypoint_is_copied_tagged_and_coalesces_with_goals_on_one_sequence():
    """Break caught: waypoint aliases caller data or collides with manual/provider sequences."""
    clock = iter((1.0, 1.1, 1.2, 1.3))
    panel = OnlineTargetPanel(now=lambda: next(clock), subscribe_updates=lambda callback: None,
                              mode_name="Pelvis-1")
    xyz = np.array([0.4, -0.2, 0.82])
    panel.request_waypoint(xyz, 30.0)
    xyz[:] = 99.0
    request = panel.consume_request()

    assert request["kind"] == "waypoint"
    assert np.array_equal(request["target_xyz"], [0.4, -0.2, 0.82])
    assert request["heading_degrees"] == 30.0
    assert request["mode_name"] == "Pelvis-1" and request["mode_epoch"] == 0
    assert request["sequence"] == 1 and request["stamp"] == 1.0

    reserved = panel.next_sequence()
    panel.apply(np.array([[0., 0., .8], [.2, .1, .8], [.2, -.1, .8]]), np.zeros((3, 3)))
    assert reserved == 2 and panel.consume_request()["sequence"] == 3
    panel.request_waypoint([0.1, 0.2, 0.81], -10.0)
    panel.apply(np.array([[.01, 0., .8], [.2, .1, .8], [.2, -.1, .8]]), np.zeros((3, 3)))
    latest = panel.consume_request()
    assert latest["kind"] == "goal" and latest["sequence"] == 5


def test_waypoint_validation_priority_discard_and_close():
    """Break caught: malformed waypoints enqueue or bypass lifecycle/cleanup priority."""
    panel = OnlineTargetPanel(now=lambda: 2.0, subscribe_updates=lambda callback: None,
                              mode_name="Pelvis-1")
    panel.request_waypoint([0.1, 0.2], 0.0)
    assert panel.consume_request() is None
    assert "finite" in panel.status["last_error"]
    panel.request_waypoint([0.1, 0.2, np.nan], 0.0)
    assert panel.consume_request() is None

    panel.request_waypoint([0.1, 0.2, 0.8], 5.0)
    panel.request_pause()
    assert panel.consume_request() == {"kind": "pause"}
    assert panel.consume_request()["kind"] == "waypoint"
    panel.request_waypoint([0.2, 0.3, 0.82], 6.0)
    panel.discard_goal()
    assert panel.consume_request() is None
    assert panel.next_sequence() == 3  # discard does not reuse the waypoint's sequence
    panel.request_waypoint([0.3, 0.4, 0.84], 7.0)
    panel.request_close()
    assert panel.consume_request() == {"kind": "close"}
    assert panel.consume_request() is None


def test_waypoint_widgets_follow_confirmed_mode_and_reset_from_pelvis_pose():
    """Break caught: Go is usable outside Pelvis-1 or reset overwrites it with a non-pelvis pose."""
    class Model:
        def __init__(self): self.value = 0.0
        def get_value_as_float(self): return self.value
        def set_value(self, value): self.value = value
    class Field:
        def __init__(self): self.model = Model(); self.enabled = True
    class Button:
        def __init__(self, callback): self.callback = callback; self.enabled = True
    class Context:
        def __init__(self, *_args, **_kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *_args): return False
    class Window:
        def __init__(self, *_args, **_kwargs): self.frame = Context(); self.visible = True
        def set_visibility_changed_fn(self, callback): self.callback = callback
    class Ui:
        def __init__(self): self.buttons = {}
        def Fraction(self, value): return value
        def FloatField(self, **_kwargs): return Field()
        def Label(self, *_args, **_kwargs): return type("Label", (), {"text": ""})()
        def Button(self, label, clicked_fn, **_kwargs):
            button = Button(clicked_fn); self.buttons[label] = button; return button
    ui = Ui(); ui.Window = Window; ui.VStack = Context; ui.HStack = Context; ui.ScrollingFrame = Context
    positions = np.array([[1.25, -0.5, .8], [.2, .1, .8], [.2, -.1, .8]])
    pelvis_yaw = 35.0
    quaternions = np.tile([1., 0., 0., 0.], (3, 1))
    quaternions[0] = [np.cos(np.deg2rad(pelvis_yaw) / 2), 0., 0., np.sin(np.deg2rad(pelvis_yaw) / 2)]
    panel = OnlineTargetPanel(now=lambda: 3.0, subscribe_updates=lambda callback: None, ui_module=ui,
                              marker_factory=lambda: None, seed_positions=positions,
                              seed_quaternions_wxyz=quaternions, mode_name="VR-3")

    assert not ui.buttons["Go"].enabled
    assert not any(field.enabled for field in panel.waypoint_position_fields)
    assert not panel.waypoint_heading_field.enabled
    panel.set_mode("Pelvis-1", 7)
    assert ui.buttons["Go"].enabled and all(field.enabled for field in panel.waypoint_position_fields)
    assert panel.waypoint_heading_field.enabled
    assert np.allclose([model.value for model in panel.waypoint_position_models], [1.25, -0.5, 0.8])
    assert np.isclose(panel.waypoint_heading_model.value, pelvis_yaw)
    panel.waypoint_position_models[0].set_value(8.0)
    panel.update_status({"waypoint_state": "IDLE", "waypoint_reason": "ready"})
    assert panel.waypoint_position_models[0].value == 8.0  # telemetry must not overwrite user editing
    ui.buttons["Go"].callback()
    request = panel.consume_request()
    assert request["kind"] == "waypoint" and np.allclose(request["target_xyz"], [8.0, -0.5, 0.8])
    assert request["mode_epoch"] == 7
    ui.buttons["Stop"].callback()
    assert panel.consume_request() == {"kind": "pause"}


def test_sparse_mode_disables_and_masks_inactive_widget_rows():
    """Break caught: disabled NaN fields poison a complete sparse-mode packet."""
    class Model:
        def __init__(self): self.value = 0.0
        def get_value_as_float(self): return self.value
        def set_value(self, value): self.value = value
    class Field:
        def __init__(self): self.model = Model(); self.enabled = True
    class Context:
        def __init__(self, *_args, **_kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *_args): return False
    class Window:
        def __init__(self, *_args, **_kwargs): self.frame = Context(); self.visible = True
        def set_visibility_changed_fn(self, callback): self.callback = callback
    class Ui:
        def __init__(self): self.buttons = {}
        def Fraction(self, value): return value
        def FloatField(self, **_kwargs): return Field()
        def Label(self, *_args, **_kwargs): return type("Label", (), {"text": ""})()
        def Button(self, label, clicked_fn, **_kwargs): self.buttons[label] = clicked_fn
    ui = Ui(); ui.Window = Window; ui.VStack = Context; ui.HStack = Context; ui.ScrollingFrame = Context
    seed_positions = np.array([[0., 0., .8], [.2, .1, .8], [.2, -.1, .8]])
    seed_quaternions = np.tile([1., 0., 0., 0.], (3, 1))
    panel = OnlineTargetPanel(now=lambda: 2.0, subscribe_updates=lambda callback: None, ui_module=ui,
                              marker_factory=lambda: None, seed_positions=seed_positions,
                              seed_quaternions_wxyz=seed_quaternions)
    panel.set_mode("Pelvis-1", 1)

    assert all(field.enabled for field in (*panel.position_fields[0], *panel.angle_fields[0]))
    assert not any(field.enabled for row in (1, 2) for field in (*panel.position_fields[row], *panel.angle_fields[row]))
    for row in (1, 2):
        for model in (*panel.position_models[row], *panel.angle_models[row]):
            model.set_value(float("nan"))
    panel.position_models[0][0].set_value(0.01)
    ui.buttons["Apply"]()
    request = panel.consume_request()

    assert request["mode_name"] == "Pelvis-1" and request["mode_epoch"] == 1
    assert request["positions"][0, 0] == 0.01
    assert np.array_equal(request["positions"][1:], seed_positions[1:])
    assert np.array_equal(request["quaternions"][1:], seed_quaternions[1:])


def test_marker_receives_world_goal_with_runtime_xyzw_and_status_fields():
    """Break caught: goal markers use local/wxyz data or omit actual/reference/error telemetry."""
    class Marker:
        visible = True
        def set_visibility(self, visible): self.visible = visible
        def visualize(self, translations, orientations): self.translations, self.orientations = translations, orientations
    marker = Marker()
    seed_positions = np.array([[0., 0., .8], [.2,.1,.8], [.2,-.1,.8]])
    seed_quaternions = np.tile([1.,0.,0.,0.], (3,1))
    panel = OnlineTargetPanel(now=lambda: 1.0, subscribe_updates=lambda callback: None,
                              marker_factory=lambda: marker, env_origin=np.array([2., 3., 0.]), runtime_quaternion_order="xyzw",
                              seed_positions=seed_positions, seed_quaternions_wxyz=seed_quaternions)
    assert panel.status["state"] == "READY"
    assert np.allclose(marker.translations[0], [2., 3., .8])
    assert np.allclose(marker.orientations[0], [0., 0., 0., 1.])
    panel.update_status({"goal_positions": np.array([[0., 0., .8], [.2,.1,.8], [.2,-.1,.8]]),
                         "goal_quaternions_wxyz": np.tile([1.,0.,0.,0.], (3,1)),
                         "reference_positions": np.zeros((3,3)), "reference_quaternions_wxyz": np.tile([1.,0.,0.,0.], (3,1)),
                         "actual_positions": np.ones((3,3)), "actual_quaternions_wxyz": np.tile([1.,0.,0.,0.], (3,1)),
                         "position_errors": np.ones(3), "orientation_errors": np.zeros(3), "last_error": "bad packet"})
    assert np.allclose(marker.translations[0], [2., 3., .8])
    assert np.allclose(marker.orientations[0], [0., 0., 0., 1.])
    assert panel.status["last_error"] == "bad packet"
    panel.set_mode("Pelvis-1", 1)
    assert marker.translations.shape == (1, 3)
    assert np.allclose(marker.translations[0], [2., 3., .8])
    panel.set_mode("UMI-2", 2)
    assert marker.translations.shape == (2, 3)
    assert np.allclose(marker.translations, [[2.2, 3.1, .8], [2.2, 2.9, .8]])


def test_marker_cfg_keeps_only_a_small_frame_prototype(monkeypatch):
    """Break caught: the default connecting line and 0.5 m frame obscure the robot."""
    import isaaclab.markers as markers
    from isaaclab.markers.config import FRAME_MARKER_CFG

    class CapturedMarkers:
        def __init__(self, cfg): self.cfg = cfg

    monkeypatch.setattr(markers, "VisualizationMarkers", CapturedMarkers)
    captured = OnlineTargetPanel._build_markers()

    assert captured.cfg.prim_path == "/Visuals/BFMOnline/goal_frames"
    assert tuple(captured.cfg.markers) == ("frame",)
    assert captured.cfg.markers["frame"].scale == (0.08, 0.08, 0.08)
    assert set(FRAME_MARKER_CFG.markers) == {"frame", "connecting_line"}
    assert FRAME_MARKER_CFG.markers["frame"].scale == (0.5, 0.5, 0.5)


def test_window_close_callback_latches_exit_without_a_heartbeat():
    """Break caught: an X-close is ignored when the Kit update subscription stops."""
    class Context:
        def __init__(self, *_args, **kwargs): self.kwargs=kwargs
        def __enter__(self): return self
        def __exit__(self, *args): return False
    class Window(Context):
        def __init__(self, *_args, **kwargs): super().__init__(**kwargs); self.visible=True; self.frame=Context(); self.callback=None; self.destroyed=False
        def set_visibility_changed_fn(self, callback): self.callback=callback
        def destroy(self): self.destroyed=True
    class Ui:
        def Fraction(self, value): return value
        def FloatField(self, *_args, **_kwargs): return type("F", (), {"model": type("M", (), {"set_value": lambda self, value: None, "get_value_as_float": lambda self: 0.})()})()
        def Label(self, *_args, **_kwargs): return type("L", (), {"text": ""})()
        def Button(self, *_args, **_kwargs): pass
    ui = Ui()
    ui.Window = Window; ui.VStack = Context; ui.HStack = Context; ui.ScrollingFrame = Context
    callbacks = []
    panel = OnlineTargetPanel(subscribe_updates=lambda callback: callbacks.append(callback), ui_module=ui,
                              marker_factory=lambda: type("Marker", (), {"set_visibility": lambda self, visible: None, "visualize": lambda self, positions, quaternions: None})())
    assert callbacks and panel.heartbeat_stamp is None
    visibility_callback = panel.window.callback
    panel.window.visible = False
    visibility_callback(False)
    assert panel.close_requested and panel.consume_request()["kind"] == "close"
    panel.close()
    assert panel.window.callback is None and panel.window.destroyed
