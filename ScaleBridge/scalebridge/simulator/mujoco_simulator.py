import os
import time
import torch
import threading
import mujoco
import numpy as np
from loguru import logger
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from scalebridge.simulator.base_simulator import BaseSimulator
from scalebridge.utils.merge_robot_object_xml import merge_robot_object_xml

def draw_marker(pos,v):
    geom = v.user_scn.geoms[v.user_scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.03,0.03,0.03],
        pos=pos,
        mat=np.eye(3).flatten(),
        rgba=[1,0,0,1]
    )
    v.user_scn.ngeom += 1

class MujocoSimulator(BaseSimulator):
    def __init__(self, config, metadata_dict):
        self.headless = config.get('headless', False)
        self.viewer = self.renderer = self.video_writer = None
        self.joystick = self.joystick_thread = self._pygame = None
        self._joystick_stop = threading.Event()
        self._closed = False
        self.record_video = config.get('record_video', False)
        self.marker = config.get('marker', False)
        self.use_joystick = config.get('joystick', False)
        self.camera_follow = config.get('camera_follow', False) or self.record_video

        try:
            super().__init__(config, metadata_dict)
            self._setup_joystick()
        except BaseException:
            self.close()
            raise

    def _setup_backbone(self):
        super()._setup_backbone()

        xml_path = self.cfg.asset.xml_path

        # pase object from metadata if exists
        self.has_object = "object_names" in self.metadata_dict
        if self.has_object:
            xml_path = merge_robot_object_xml(xml_path, self.metadata_dict)

        self.mujoco_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mujoco_data = mujoco.MjData(self.mujoco_model)
        self.mujoco_model.opt.timestep=self.low_dt
        
        if not self.headless:
            import mujoco.viewer as mjv
            self.viewer = mjv.launch_passive(
                model=self.mujoco_model, data=self.mujoco_data,
                show_left_ui=False, show_right_ui=False,
            )
        self.marker_pos = None
        if self.record_video:
            import imageio
            save_dir = HydraConfig.get().runtime.output_dir
            video_name = os.path.join(save_dir, 'recording.mp4')
            self.video_writer = imageio.get_writer(video_name, fps=50)
            self.renderer = mujoco.Renderer(self.mujoco_model, height=480, width=640)
            self.render_scene = mujoco.MjvScene(self.mujoco_model, maxgeom=1000)
            self.camera = mujoco.MjvCamera()


    def _setup_asset(self):
        super()._setup_asset()

        self.default_qpos = self.mujoco_data.qpos.copy()
        self.default_qvel = self.mujoco_data.qvel.copy()

        self.default_dof_pos = self.mujoco_data.qpos[7:7+self.num_joints].copy()
        if "default_dof_pos" in self.metadata_dict:
            pose = np.asarray(self.metadata_dict["default_dof_pos"], dtype=float)
            if pose.shape != (self.num_joints,) or not np.isfinite(pose).all():
                raise ValueError("Invalid joint-ordered default pose metadata")
            self.default_dof_pos[self.sim_to_env_joint_idx] = pose
            self.default_qpos[7:7+self.num_joints] = self.default_dof_pos
        model = self.mujoco_model
        joint_ids = model.actuator_trnid[:, 0]
        if (not np.array_equal(joint_ids, np.arange(1, self.num_joints + 1))
                or not np.array_equal(model.jnt_qposadr[joint_ids], np.arange(7, 7+self.num_joints))
                or not np.array_equal(model.jnt_dofadr[joint_ids], np.arange(6, 6+self.num_joints))
                or not np.allclose(model.actuator_gear, np.tile([1., 0, 0, 0, 0, 0], (self.num_joints, 1)))):
            raise ValueError("Expected ordered unit-gear joint motors after the free root")
        self._setup_torque_limits(joint_ids)

    def _setup_torque_limits(self, joint_ids):
        model = self.mujoco_model
        ranges = np.asarray(model.actuator_ctrlrange)
        joint_ranges = np.asarray(model.jnt_actfrcrange)[joint_ids]
        if (not np.all(model.actuator_ctrllimited) or not np.all(model.jnt_actfrclimited[joint_ids])
                or not np.isfinite(ranges).all() or not np.isfinite(joint_ranges).all()
                or not np.allclose(ranges[:, 0], -ranges[:, 1])
                or not np.allclose(joint_ranges[:, 0], -joint_ranges[:, 1])):
            raise ValueError("Require explicit finite symmetric XML motor and joint force limits")
        xml_limits = np.minimum(ranges[:, 1], joint_ranges[:, 1])
        if np.any(xml_limits <= 0):
            raise ValueError("XML torque limits must be positive")
        self.torque_limit = xml_limits.copy()
        self.torque_limit_mismatches = []
        if "torque_limit" in self.metadata_dict:
            declared = np.asarray(self.metadata_dict["torque_limit"], dtype=float)
            if declared.shape != (self.num_joints,) or not np.isfinite(declared).all() or np.any(declared <= 0):
                raise ValueError("Invalid joint-ordered torque limit metadata")
            in_sim = np.zeros(self.num_joints)
            in_sim[self.sim_to_env_joint_idx] = declared
            names = self._get_joint_names()
            self.torque_limit_mismatches = [{"joint": names[i], "metadata": float(in_sim[i]),
                "xml": float(xml_limits[i])} for i in range(self.num_joints)
                if not np.isclose(in_sim[i], xml_limits[i], atol=1e-6, rtol=0)]
            self.torque_limit = np.minimum(xml_limits, in_sim)
            if self.torque_limit_mismatches:
                if self.cfg.get("strict_torque_limits", False):
                    raise ValueError(f"XML/metadata torque limits differ: {self.torque_limit_mismatches}")
                logger.warning(f"[Simulator] XML/metadata torque mismatch; conservatively clipping: {self.torque_limit_mismatches}")

    def _setup_joystick(self):
        if self.use_joystick:
            import pygame
            self._pygame = pygame
            logger.info(f'[Simulator] Using Joystick as command sender; Make sure you have connected the joystick to the PC!')
            pygame.init()
            pygame.joystick.init()
            if pygame.joystick.get_count() > 0:
                self.joystick = pygame.joystick.Joystick(0)
                self.joystick.init()
                logger.info(f"[Simulator] Joystick detected: {self.joystick.get_name()}")
            else:
                pygame.quit()
                raise RuntimeError("Requested joystick was not detected")
            self.lin_vel_x_tmp = 0
            self.lin_vel_y_tmp = 0
            self.ang_vel_z_tmp = 0
            self.joystick_thread = threading.Thread(target=self._handle_joystick, args=(pygame,),daemon=True)
            self.joystick_thread.start()
    
    def _handle_joystick(self, pygame):
        try:
            while not self._joystick_stop.is_set():
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self._joystick_stop.set()
                        return
                self.lin_vel_x_tmp = self.joystick.get_axis(1) * -1
                self.lin_vel_y_tmp = self.joystick.get_axis(0) * -1
                self.ang_vel_z_tmp = self.joystick.get_axis(3) * -1
                self._joystick_stop.wait(0.001)
        except Exception:
            self._joystick_stop.set()

    def _render(self):
        if not self.is_running():
            raise RuntimeError("MuJoCo simulator/viewer has been closed")
        if self.headless and not self.record_video:
            return

        if self.marker_pos is not None and self.marker:
            if self.viewer is not None:
                self.viewer.user_scn.ngeom = 0
                for i in range(self.marker_pos.shape[0]):
                    draw_marker(self.marker_pos[i], self.viewer)
            if self.record_video:
                self.render_scene.ngeom = 0
                for i in range(self.marker_pos.shape[0]):
                    geom = self.render_scene.geoms[self.render_scene.ngeom]
                    mujoco.mjv_initGeom(
                        geom,
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.03, 0.03, 0.03],
                        pos=self.marker_pos[i],
                        mat=np.eye(3).flatten(),
                        rgba=[1,0,0,1]
                    )
                    self.render_scene.ngeom += 1
        camera = self.viewer.cam if self.viewer is not None else self.camera
        if self.camera_follow:
            camera.lookat = self.mujoco_data.qpos[:3].copy()
            camera.elevation = 0
            camera.azimuth = 180
            camera.distance = 3.0

        if self.viewer is not None:
            self.viewer.sync()

        if self.record_video:
            self.renderer.update_scene(self.mujoco_data, camera=camera)
        
            # Then manually add markers to the renderer's scene
            if self.marker_pos is not None and self.marker:
                # The renderer's scene might need to be updated after update_scene
                # So we add markers after the update
                for i in range(self.marker_pos.shape[0]):
                    geom = self.renderer.scene.geoms[self.renderer.scene.ngeom]
                    mujoco.mjv_initGeom(
                        geom,
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.03, 0.03, 0.03],
                        pos=self.marker_pos[i],
                        mat=np.eye(3).flatten(),
                        rgba=[1,0,0,1]
                    )
                    self.renderer.scene.ngeom += 1

            img = self.renderer.render()
            self.video_writer.append_data(img)

    def update_marker_pos(self, marker_pos):
        self.marker_pos = marker_pos.cpu().numpy()

    def refresh_sim(self):

        state_dict = { # weishuai: FIXME
            "root_pos": self.mujoco_data.qpos[:3],
            "root_quat_wxyz": self.mujoco_data.qpos[3:7],
            "base_ang_vel": self.mujoco_data.qvel[3:6],
            "dof_pos": self.mujoco_data.qpos[7:7+self.num_joints][self.sim_to_env_joint_idx],
            "dof_vel": self.mujoco_data.qvel[6:6+self.num_joints][self.sim_to_env_joint_idx],
        }

        if self.has_object:
            object_root_pose = self.mujoco_data.qpos[7+self.num_joints:].reshape(-1, 7)
            state_dict.update({
                "object_root_pos": object_root_pose[:, :3],
                "object_root_quat_wxyz": object_root_pose[:, 3:7],
            })

        if self.use_joystick:
            self.commands = np.array([self.lin_vel_x_tmp, self.lin_vel_y_tmp, self.ang_vel_z_tmp], dtype=np.float32)
            self.commands = np.where(np.abs(self.commands) < 0.05, 0, self.commands)
            state_dict.update({'commands': self.commands})
        
        return {k:torch.from_numpy(v).float() for k,v in state_dict.items()}

    def calibrate(self, init_state_dict=None):
        init_state_dict = init_state_dict or {}
        init_qpos = self.default_qpos.copy()
        init_qvel = self.default_qvel.copy()

        if "root_pos" in init_state_dict:
            init_qpos[:3] = init_state_dict["root_pos"]
        if "root_quat" in init_state_dict:
            init_qpos[3:7] = init_state_dict["root_quat"]
        if "dof_pos" in init_state_dict:
            init_qpos[7:7+self.num_joints][self.sim_to_env_joint_idx] = init_state_dict["dof_pos"]
        if "root_lin_vel" in init_state_dict:
            init_qvel[:3] = init_state_dict["root_lin_vel"]
        if "root_ang_vel_w" in init_state_dict or "root_ang_vel" in init_state_dict:
            # Packed reference angular velocity is world-frame, but a MuJoCo
            # free joint's rotational qvel is body-local.
            angular = init_state_dict.get("root_ang_vel_w", init_state_dict.get("root_ang_vel"))
            init_qvel[3:6] = world_to_body_vector(init_qpos[3:7], angular)
        if "dof_vel" in init_state_dict:
            init_qvel[6:6+self.num_joints][self.sim_to_env_joint_idx] = init_state_dict["dof_vel"]
        
        if self.has_object:
            
            init_obj_pose = init_qpos[7+self.num_joints:].copy().reshape(-1, 7)
            init_obj_vel = init_qvel[6+self.num_joints:].copy().reshape(-1, 6)
            
            if "object_root_pos" in init_state_dict:
                init_obj_pose[:, :3] = init_state_dict["object_root_pos"]
            if "object_root_quat" in init_state_dict:
                init_obj_pose[:, 3:7] = init_state_dict["object_root_quat"]
            if "object_root_lin_vel" in init_state_dict:
                init_obj_vel[:, :3] = init_state_dict["object_root_lin_vel"]
            if "object_root_ang_vel" in init_state_dict:
                init_obj_vel[:, 3:] = init_state_dict["object_root_ang_vel"]

            init_qpos[7+self.num_joints:] = init_obj_pose.flatten()
            init_qvel[6+self.num_joints:] = init_obj_vel.flatten()

        if (not np.isfinite(init_qpos).all() or not np.isfinite(init_qvel).all()
                or not np.isclose(np.linalg.norm(init_qpos[3:7]), 1., atol=1e-5, rtol=0)):
            raise ValueError("Invalid/nonfinite initial state; refusing MuJoCo forward")
        self.mujoco_data.qpos[:] = init_qpos
        self.mujoco_data.qvel[:] = init_qvel
        self.mujoco_data.ctrl[:] = 0

        mujoco.mj_forward(self.mujoco_model, self.mujoco_data)

        return init_qpos[:3], init_qpos[3:7]

    def apply_action(self, tgt_dof_pos):
        if not self.is_running():
            raise RuntimeError("MuJoCo simulator has been closed")
        tgt_dof_pos = np.asarray(tgt_dof_pos).reshape(-1)
        if tgt_dof_pos.shape != (self.num_joints,) or not np.isfinite(tgt_dof_pos).all():
            raise ValueError("Expected one finite target per controlled joint")

        target_dof_pos_in_sim = self.default_dof_pos.copy()
        target_dof_pos_in_sim[self.env_action_to_sim_idx] = tgt_dof_pos

        for _ in range(self.decimation):
            torque = (target_dof_pos_in_sim - self.mujoco_data.qpos[7:7+self.num_joints]) * self.stiffness - self.mujoco_data.qvel[6:6+self.num_joints] * self.damping # no clip here
            if not np.isfinite(torque).all():
                raise ValueError("Nonfinite PD torque; refusing to step physics")
            self.mujoco_data.ctrl[:] = np.clip(torque, -self.torque_limit, self.torque_limit)
            mujoco.mj_step(self.mujoco_model, self.mujoco_data)

        self._render()

    def _get_joint_names(self):
        joint_names = []
        for i in range(self.mujoco_model.nu):
            dof_name = mujoco.mj_id2name(self.mujoco_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            joint_names.append(dof_name)
        return joint_names

    def is_running(self):
        return not self._closed and (self.headless or (self.viewer is not None and self.viewer.is_running()))

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._joystick_stop.set()
        thread = self.joystick_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.)
        for name in ("video_writer", "renderer", "viewer"):
            resource = getattr(self, name, None)
            if resource is not None:
                try:
                    resource.close()
                except Exception as error:
                    logger.warning(f"[Simulator] Closing {name} failed: {error}")
                setattr(self, name, None)
        if self._pygame is not None:
            self._pygame.quit()
            self._pygame = None


def world_to_body_vector(quaternion_wxyz, vector):
    quaternion = np.asarray(quaternion_wxyz, dtype=float)
    vector = np.asarray(vector, dtype=float)
    if (quaternion.shape != (4,) or vector.shape != (3,) or not np.isfinite(quaternion).all()
            or not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(quaternion), 1., atol=1e-5, rtol=0)):
        raise ValueError("Expected finite world vector and unit wxyz root quaternion")
    quaternion = quaternion / np.linalg.norm(quaternion)
    xyz = -quaternion[1:]
    twice_cross = 2. * np.cross(xyz, vector)
    return vector + quaternion[0] * twice_cross + np.cross(xyz, twice_cross)
