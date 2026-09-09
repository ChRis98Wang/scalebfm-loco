"""Exercise the real player and Kit callbacks with a trusted local checkpoint.

Pass the same arguments as play.py, including --motion_menu --target_object
--num_envs 1 --mode_index 7 --viz kit. Own this test with a bounded process group.
No synthetic policy or fake physics is substituted; no desktop input is sent.
"""

from pathlib import Path
import runpy
import sys

scripts = Path(__file__).resolve().parents[1] / "scripts/pretrain/rsl_rl"
sys.path.insert(0, str(scripts))
player = runpy.run_path(str(scripts / "play.py"))
app = player["simulation_app"]
completed = False
try:
    import gymnasium as gym
    import torch
    from playback_ui import MotionPlaybackPanel
    from target_object import _tensor

    assert player["args_cli"].motion_menu and player["target_spec"] is not None
    assert player["args_cli"].mode_index == 7
    original_make = gym.make
    original_progress = MotionPlaybackPanel.update_progress
    state = {"steps": 0}

    def capture_real_env(*args, **kwargs):
        env = original_make(*args, **kwargs)
        state["env"] = env.unwrapped
        return env

    def exercise_controls(panel, elapsed):
        global completed
        original_progress(panel, elapsed)
        state["steps"] += 1
        step = state["steps"]
        env = state["env"]
        command = env.command_manager.get_term("motion")
        if step == 25:
            assert int(command._mode[0].sum()) == 14
            state["initial_motion"] = panel.selection.current_id
            panel.next_motion()
        elif step == 26:
            assert panel.selection.current_id != state["initial_motion"]
            assert int(command.motion_ids[0]) == panel.selection.current_id
            assert int(command.time_steps[0]) == 1
        elif step in (50, 75, 100):
            index = {50: 2, 75: 0, 100: 7}[step]
            panel.mode_combo.model.get_item_value_model().set_value(index)
            panel.request_selected_mode()
        elif step in (51, 76, 101):
            expected = {51: ("VR-3", 3), 76: ("Pelvis-1", 1), 101: ("WholeBody-14", 14)}[step]
            assert list(command.cfg.mode_candidates) == [expected[0]]
            assert int(command._mode[0].sum()) == expected[1]
            assert int(command.time_steps[0]) == 1
            print(f"[BFM ROBOT UI TEST] Applied {expected[0]} with {expected[1]} active bodies", flush=True)
        elif step == 110:
            obj = env.scene["target_object"]
            pose = _tensor(obj.data.root_pose_w).clone()
            pose[:, 0] += 0.5
            pose[:, 2] += 0.3
            obj.write_root_pose_to_sim(pose)
            state["robot_motion_frame"] = int(command.time_steps[0])
            panel.request_target_reset()
        elif step == 111:
            obj = env.scene["target_object"]
            local = _tensor(obj.data.root_pos_w) - env.scene.env_origins
            expected = torch.tensor(player["target_spec"].position, device=env.device)
            torch.testing.assert_close(local[0], expected, atol=0.02, rtol=0)
            assert int(command.time_steps[0]) == state["robot_motion_frame"] + 1
            print("[BFM ROBOT UI TEST] Target reset without restarting the robot clip", flush=True)
        elif step == 130:
            assert "Target local XYZ" in panel._target_label.text
            completed = True
            panel.request_close()

    gym.make = capture_real_env
    MotionPlaybackPanel.update_progress = exercise_controls
    try:
        player["main"]()
        assert completed, "Player exited before the complete interaction sequence"
    finally:
        gym.make = original_make
        MotionPlaybackPanel.update_progress = original_progress
    print("[BFM ROBOT UI TEST] PASS: real policy, clip switching, three reference modes, target reset, HUD and Exit player", flush=True)
except BaseException:
    import traceback
    traceback.print_exc()
    print("[BFM ROBOT UI TEST] FAIL", flush=True)
    raise
finally:
    app.close(exit_code=1 if sys.exc_info()[0] else 0)
