"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from contextlib import ExitStack, nullcontext

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip
from online_control import add_online_args, validate_online_args
from target_object import (add_target_object_args, attach_target_object, reset_target_object,
                           target_spec_from_args, target_status)

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
add_online_args(parser)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to the motion file.")
parser.add_argument("--motion_menu", action="store_true", help="Show the native Kit motion browser.")
parser.add_argument("--additional_motion_file", action="append", default=[], help="Additional YAML index for playback; repeatable.")
parser.add_argument("--initial_motion", type=str, default=None, help="Exact initial motion name in the playback catalog.")
parser.add_argument("--mode_index", type=int, default=None, help="Fixed motion mode index to use during playback.")
parser.add_argument(
    "--local_tracking",
    action="store_true",
    default=False,
    help="Use the reference root position for observations; requires a root-containing --mode_index.",
)
# append RSL-RL cli arguments
add_target_object_args(parser)
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
try:
    validate_online_args(args_cli)
except ValueError as error:
    parser.error(str(error))
try:
    target_spec = target_spec_from_args(args_cli)
except (ValueError, OSError) as error:
    parser.error(str(error))
if args_cli.motion_menu and (args_cli.video or args_cli.headless):
    parser.error("--motion_menu requires GUI playback; do not combine it with --headless or --video.")
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import pathlib
import torch

from my_rsl_rl.runners.on_policy_runner import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import scaletrack.tasks  # noqa: F401
from playback_menu import combined_motion_index, gui_is_available, switch_mode, switch_motion


class _HiddenMarker:

    def __init__(self, marker):
        self._marker = marker
        self._marker.set_visibility(False)

    def set_visibility(self, _visible):
        self._marker.set_visibility(False)

    def visualize(self, *_args, **_kwargs):
        pass


class _LocalReferenceMarker:

    def __init__(self, marker, motion_command, robot_root_position):
        self._marker = marker
        self._motion_command = motion_command
        self._robot_root_position = robot_root_position

    def set_visibility(self, visible):
        self._marker.set_visibility(visible)

    def visualize(self, positions, orientations, *args, **kwargs):
        root_offset = self._robot_root_position(self._motion_command) - self._motion_command.anchor_pos_w
        self._marker.visualize(positions + root_offset, orientations, *args, **kwargs)


def _enable_local_tracking(motion_command):

    command_type = type(motion_command)
    robot_root_position = command_type.robot_anchor_pos_w.fget

    class _LocalTrackingMotionCommand(command_type):
        @property
        def robot_anchor_pos_w(self):
            return self.anchor_pos_w

    motion_command.__class__ = _LocalTrackingMotionCommand
    return robot_root_position


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    with ExitStack() as cleanup:
        _play(env_cfg, agent_cfg, cleanup)


def _play(env_cfg, agent_cfg, cleanup):
    """Own playback resources from creation, including failures during setup."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    if args_cli.online_targets:
        from online_control import configure_online_env
        configure_online_env(env_cfg)
    if args_cli.motion_menu:
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 1
        if env_cfg.scene.num_envs != 1:
            raise ValueError("The motion menu currently supports --num_envs 1 only.")
        env_cfg.commands.motion.debug_vis = False  # robot-only playback, without reference overlays

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    
    print(f"[INFO]: Using motion file from CLI: {args_cli.motion_file}")
    env_cfg.commands.motion.motion_file = args_cli.motion_file
    env_cfg.commands.motion.enable_reset_disturbance = False
    available_modes = list(env_cfg.commands.motion.mode_candidates.items())
    if args_cli.motion_menu and args_cli.mode_index is None:
        args_cli.mode_index = next(index for index, (name, _) in enumerate(available_modes) if name == "WholeBody-14")
    if args_cli.mode_index is not None:
        mode_candidates = list(env_cfg.commands.motion.mode_candidates.items())
        if not 0 <= args_cli.mode_index < len(mode_candidates):
            raise ValueError(
                f"mode_index must be between 0 and {len(mode_candidates) - 1}; received {args_cli.mode_index}."
            )
        mode_name, link_names = mode_candidates[args_cli.mode_index]
        env_cfg.commands.motion.mode_candidates = {mode_name: link_names}
        print(f"[INFO]: Using fixed mode index {args_cli.mode_index}: {mode_name} ({link_names})")
    if args_cli.local_tracking:
        if args_cli.mode_index is None:
            raise ValueError("--local_tracking requires --mode_index with a mode that includes the root link.")
        root_link_name = env_cfg.commands.motion.anchor_body_name
        if root_link_name not in link_names:
            raise ValueError(
                f"Local tracking requires the selected mode to include the root link '{root_link_name}'; "
                f"mode '{mode_name}' contains {link_names}."
            )
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # create isaac environment
    attach_target_object(env_cfg, target_spec)
    indexes = [args_cli.motion_file, *args_cli.additional_motion_file]
    index_context = combined_motion_index(indexes) if args_cli.additional_motion_file else nullcontext(args_cli.motion_file)
    with index_context as motion_file:
        env_cfg.commands.motion.motion_file = motion_file
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
        # Capture the current outer wrapper at cleanup time, including setup failures.
        cleanup.callback(lambda: env.close())
    if args_cli.online_targets:
        from online_control import assert_online_env
        assert_online_env(env.unwrapped)

    if args_cli.mode_index is not None or args_cli.local_tracking or args_cli.motion_menu or args_cli.initial_motion or args_cli.online_targets:
        motion_command = env.unwrapped.command_manager.get_term("motion")

    if args_cli.local_tracking:
        robot_root_position = _enable_local_tracking(motion_command)
        if hasattr(motion_command, "goal_body_visualizers"):
            for index, marker in enumerate(motion_command.goal_body_visualizers):
                motion_command.goal_body_visualizers[index] = _LocalReferenceMarker(
                    marker, motion_command, robot_root_position
                )
        print("[INFO]: Local tracking enabled: observations use the reference root position.")

    if args_cli.mode_index is not None:
        active_link_names = set(link_names)
        if hasattr(motion_command, "current_body_visualizers"):
            for index, body_name in enumerate(motion_command.cfg.body_names):
                if body_name not in active_link_names:
                    motion_command.current_body_visualizers[index] = _HiddenMarker(
                        motion_command.current_body_visualizers[index]
                    )
                    motion_command.goal_body_visualizers[index] = _HiddenMarker(
                        motion_command.goal_body_visualizers[index]
                    )

    log_dir = os.path.dirname(resume_path)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    
    ppo_runner._set_env_is_evaluating()

    if args_cli.mode_index is not None:
        # During normal inference the policy intentionally uses the complete task observation and an all-ones mode.
        # For fixed-mode playback, reuse the training observation path so the selected mode masks the task observation
        # and is passed to the actor explicitly. This instance-only override does not affect training.
        actor_critic = ppo_runner.alg.policy
        get_actor_obs = actor_critic.get_actor_obs

        def get_fixed_mode_actor_obs(obs, inference=False):
            return get_actor_obs(obs, inference=False)

        actor_critic.get_actor_obs = get_fixed_mode_actor_obs

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    menu = None
    online_controller = None
    if args_cli.online_targets:
        # The wrapper constructor and evaluation setup perform the last allowed
        # initialization reset.  Attach only after the explicit final reset.
        from online_ui import OnlineTargetPanel
        from online_control import OnlinePlaybackController
        from scaletrack.utils.live_reference import LiveReferenceProvider
        from scaletrack.utils.quaternion_compat import runtime_to_packed_wxyz

        if args_cli.initial_motion:
            obs = switch_motion(env, motion_command, motion_command.motion_names.index(args_cli.initial_motion))
        else:
            obs, _ = env.reset()
        robot_data = motion_command.robot.data
        body_indexes = getattr(motion_command.body_indexes, "torch", motion_command.body_indexes)
        if hasattr(body_indexes, "detach"):
            body_indexes = body_indexes.detach().cpu().tolist()
        body_indexes = [int(index) for index in body_indexes]
        body_pos = getattr(robot_data.body_pos_w, "torch", robot_data.body_pos_w)[0, body_indexes].detach().cpu().numpy()
        origins = getattr(env.unwrapped.scene.env_origins, "torch", env.unwrapped.scene.env_origins)[0].detach().cpu().numpy()
        body_pos -= origins
        body_quat = getattr(robot_data.body_quat_w, "torch", robot_data.body_quat_w)[0, body_indexes]
        body_quat = runtime_to_packed_wxyz(body_quat, motion_command.runtime_quaternion_order).detach().cpu().numpy()
        joints = getattr(robot_data.joint_pos, "torch", robot_data.joint_pos)[0].detach().cpu().numpy()
        provider = LiveReferenceProvider(motion_command.cfg.body_names, body_pos, body_quat, joints, step_dt=env.unwrapped.step_dt)
        motion_command.attach_live_reference(provider, mode_name=mode_name)
        panel = OnlineTargetPanel(env_origin=origins, runtime_quaternion_order=motion_command.runtime_quaternion_order,
                                  seed_positions=body_pos[[0, 10, 13]], seed_quaternions_wxyz=body_quat[[0, 10, 13]],
                                  mode_name=mode_name)
        cleanup.callback(panel.close)
        if panel.window is None:
            raise RuntimeError("Online targets require a native Kit window; the UI extension is unavailable.")
        online_controller = OnlinePlaybackController(env.unwrapped, motion_command, panel)
        cleanup.callback(online_controller.close)
        online_controller._pause_sim()
        print("[BFM ONLINE] Ready: Enable, then edit pelvis/wrist targets and Apply. Resume requires a fresh Apply.", flush=True)
    elif args_cli.motion_menu:
        if not gui_is_available(env.unwrapped.sim, getattr(simulation_app, "config", None)):
            raise ValueError("The motion menu requires the Kit visualizer: pass --viz kit.")
        from playback_ui import MotionPlaybackPanel

        current_id = 0
        if args_cli.initial_motion:
            current_id = motion_command.motion_names.index(args_cli.initial_motion)
        menu = MotionPlaybackPanel(
            motion_command.motion_names,
            [frames * env.unwrapped.step_dt for frames in motion_command.time_totals.tolist()],
            current_id=current_id,
        )
        cleanup.callback(menu.close)
        # Local-tracking modifies the root frame; avoid offering incompatible modes there.
        menu_modes = [(mode_name, link_names)] if args_cli.local_tracking else available_modes
        menu.configure_interaction_controls([name for name, _ in menu_modes],
                                            0 if args_cli.local_tracking else args_cli.mode_index,
                                            target_enabled=target_spec is not None)
        obs = switch_motion(env, motion_command, current_id)
        print(f"[BFM MENU] Ready: {len(motion_command.motion_names)} clips; playing {motion_command.motion_names[current_id]}", flush=True)
    elif args_cli.initial_motion:
        obs = switch_motion(env, motion_command, motion_command.motion_names.index(args_cli.initial_motion))
    else:
        obs, _ = env.reset()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        if online_controller is not None:
            with torch.inference_mode():
                candidate_obs = online_controller.before_step(obs)
                if online_controller.close_requested:
                    break
                if candidate_obs is None:
                    continue
                obs = candidate_obs
                actions = policy(obs)
                if not online_controller.validate_actions(actions):
                    continue
                obs, _, _, _ = env.step(actions)
                online_controller.after_step(obs)
            continue
        if menu is not None and menu.close_requested:
            break
        with torch.inference_mode():
            if menu is not None:
                requested = menu.selection.consume_request()
                requested_mode = menu.consume_mode_request()
                if requested_mode is not None:
                    mode_name, body_names = menu_modes[requested_mode]
                    current_motion = menu.selection.current_id if requested is None else requested
                    obs = switch_mode(env, motion_command, mode_name, body_names, current_motion)
                    menu.mark_playing(current_motion)
                    print(f"[BFM MENU] Mode: {mode_name}; motion={current_motion}", flush=True)
                elif requested is not None:
                    obs = switch_motion(env, motion_command, requested)
                    menu.mark_playing(requested)
                    print(f"[BFM MENU] Switched: id={requested} name={motion_command.motion_names[requested]} frame={int(motion_command.time_steps[0])}", flush=True)
                if menu.consume_target_reset():
                    reset_target_object(env.unwrapped)
                    print("[BFM TARGET] Reset target only", flush=True)
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            if menu is not None:
                menu.update_progress(float(motion_command.time_steps[0]) * env.unwrapped.step_dt)
                if target_spec is not None and timestep % 5 == 0:
                    menu.update_target(target_status(env.unwrapped, motion_command))
        timestep += 1
        if args_cli.video:
            if timestep == args_cli.video_length:
                break


if __name__ == "__main__":
    # run the main function
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        raise
    finally:
        # New Kit versions otherwise mask exceptions with fast-shutdown exit status 0.
        from inspect import signature
        close_kwargs = {}
        try:
            supports_exit_code = "exit_code" in signature(simulation_app.close).parameters
        except (ValueError, TypeError):
            supports_exit_code = False
        if supports_exit_code:
            close_kwargs["exit_code"] = 1 if sys.exc_info()[0] else 0
        simulation_app.close(**close_kwargs)
