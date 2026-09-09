import os
import torch
import numpy as np
from loguru import logger
from scalebridge.env.base_env import BaseEnv
from scalebridge.utils.torch_utils import calc_heading_quat, calc_heading_quat_inv, quat_mul, quat_apply, quat_inv

class MotionTrackingEnv(BaseEnv):

    def _setup_metadata(self):
        # weishuai: We explicitly pass motion dict as arguments to support heritage on motion data processing
        assert os.path.exists(self.cfg.motion_path), f"You have to ensure the correct path to motion file. Current motion path is: {self.cfg.motion_path}."
        logger.info(f"[Env] Loading offline motion trajectory from {self.cfg.motion_path}.")
        with np.load(self.cfg.motion_path, allow_pickle=False) as motion_data:
            assert motion_data['fps'] == 50, f"You should first process the data to ensure the same format and FPS with IsaacLab compatible format."
            self._setup_motion(motion_data)

        self.reference_forcing = self.cfg.get('reference_forcing', True)
        logger.info(f"[Env] Using reference root position to apply forcing: {self.reference_forcing}")
        self.metadata_dict["enable_root_localization"] = not self.reference_forcing

    def _global_time_indices(self, frame_offsets):
        return torch.clamp(
            self.episode_length_buf.unsqueeze(-1) + frame_offsets,
            min=0, max=self.motion_len - 1
        )
    
    def _setup_motion(self, motion_data):
        selected_links = self.metadata_dict["selected_body_names"]
        complete_links = self.metadata_dict["body_names"]
        if not selected_links or selected_links[0] != "pelvis" or complete_links[0] != "pelvis":
            raise ValueError("Motion tracking requires pelvis as the first selected/reference body")
        for key, expected in (("body_names", complete_links), ("joint_names", self.metadata_dict["joint_names"])):
            if key in motion_data and list(motion_data[key]) != list(expected):
                raise ValueError(f"Packed {key} order differs from policy metadata")
        if "quaternion_order" in motion_data and str(motion_data["quaternion_order"]) != "wxyz":
            raise ValueError("ScaleBridge packed root/body quaternions must be wxyz")
        body_indexes = np.array([complete_links.index(link) for link in selected_links])

        self.body_pos_w = torch.from_numpy(motion_data["body_pos_w"][:, body_indexes]).to(self.device)
        self.body_quat_w = torch.from_numpy(motion_data["body_quat_w"][:, body_indexes]).to(self.device) # wxyz
        self.motion_len = len(self.body_pos_w)
        self._source_body_pos_w = self.body_pos_w.clone()
        self._source_body_quat_w = self.body_quat_w.clone()
        future_idx = list(self.cfg.future_idx)
        if future_idx != list(self.metadata_dict.get("future_idx", future_idx)):
            raise ValueError("Future frame offsets differ from policy export metadata")
        if not future_idx or future_idx[0] != 0 or any(type(i) is not int or i < 0 for i in future_idx):
            raise ValueError("Future offsets must be nonnegative integer frame indices starting at zero")
        self.future_frame_offset = torch.as_tensor(future_idx, dtype=torch.long, device=self.device)
        self.joint_pos = motion_data["joint_pos"].copy()
        self.joint_vel = motion_data["joint_vel"].copy()
        self.body_lin_vel_w = motion_data["body_lin_vel_w"].copy()
        self.body_ang_vel_w = motion_data["body_ang_vel_w"].copy()
        # self.future_frames = self.future_frame_offset.shape[-1]

    def _gather_reference_state(self):
        temporal_index = self._global_time_indices(self.future_frame_offset).reshape(-1)
        self.state_buffer.update({
            "body_pos_w_future": self.body_pos_w.index_select(0, temporal_index).unsqueeze(0),
            "body_quat_w_wxyz_future": self.body_quat_w.index_select(0, temporal_index).unsqueeze(0),
            "future_frame_offset": self.future_frame_offset[None, :, None]
        })
        
        if self.reference_forcing:
            self.state_buffer["root_pos_buffer"][:, -1] = self.state_buffer["body_pos_w_future"][:, 0, 0]

    def _setup_state_manager(self):
        super()._setup_state_manager()
        self._gather_reference_state()
        
    def _update_state_manager(self):
        super()._update_state_manager()
        self._gather_reference_state()
        
    def _calibrate(self):
        # Each reset starts from the immutable source, not an already-aligned
        # previous episode. Never zero a height through a tensor view.
        self.body_pos_w = self._source_body_pos_w.clone()
        self.body_quat_w = self._source_body_quat_w.clone()
        use_rsi = self.cfg.get('rsi', False)
        
        if use_rsi:
            logger.warning(f"Reference State Initialization activated! This should only be used in simulator!")
            init_state_dict = {
                "root_pos": self.body_pos_w[0,0].cpu().numpy(),
                "root_quat": self.body_quat_w[0,0].cpu().numpy(),
                "root_lin_vel": self.body_lin_vel_w[0,0],
                "root_ang_vel_w": self.body_ang_vel_w[0,0],
                "dof_pos": self.joint_pos[0],
                "dof_vel": self.joint_vel[0],
            }
        else:
            init_state_dict = {}

        root_pos, root_quat = self.simulator.calibrate(init_state_dict) # weishuai: We default to not applying RSI to pure motion tracking
        
        if not use_rsi:
            self._update_state_manager() # weishuai: This would not affect the initial model context; Later it would get overwritten in reset
            
            # weishuai: We do not use RSI but adjust motion based on the current state;
            logger.info(f"[Env] Adjusting the xy-offset of offline trajectories ...")
            pos_offset = self.body_pos_w[0,0].clone()
            pos_offset[..., -1] = 0

            logger.info(f"[Env] Adjusting the heading direction of offline trajectories ...")
            root_quat_wxyz = torch.from_numpy(root_quat).float().to(self.device)
            target_heading = calc_heading_quat(root_quat_wxyz)
            source_q0 = self.body_quat_w[0,0]
            source_heading_inv = calc_heading_quat_inv(source_q0)
            
            q_align = quat_mul(target_heading, source_heading_inv) # (4)
            q_align_expand = q_align[None,None,:].expand(self.body_quat_w.shape[0], self.body_quat_w.shape[1], -1)
            
            self.body_quat_w = quat_mul(q_align_expand, self.body_quat_w)            
            self.body_pos_w = quat_apply(q_align_expand, self.body_pos_w - pos_offset)
            robot_xy = torch.as_tensor(root_pos, device=self.device, dtype=self.body_pos_w.dtype).clone()
            robot_xy[-1] = 0
            self.body_pos_w += robot_xy

    def reset(self):
        super().reset()
        self._gather_reference_state()
        return self._update_observation_manager()

    def step(self, tgt_dof_pos, action):
        obs_dict = super().step(tgt_dof_pos, action)
        self.simulator.update_marker_pos(obs_dict["body_pos_w_future"][0,0])
        return obs_dict
