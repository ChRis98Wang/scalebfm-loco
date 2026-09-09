"""Non-mutating motion formatting with explicit, versionable height alignment.

The opt-in collision_sole mode is a flat-ground, constant-height kinematic
correction. It is not contact estimation or a dynamics feasibility check.
"""

import numpy as np
import torch
from loguru import logger

from scaleretarget.formatter.base_formatter import BaseFormatter
from scaleretarget.utils.kinematic_model.kinematics_model import KinematicsModel


class KinematicFormatter(BaseFormatter):
    def __init__(self, config):
        super().__init__(config)
        if self.quat_order not in ("xyzw", "wxyz"):
            raise ValueError(f"Unsupported quaternion order: {self.quat_order}")
        self.kinematic_model_device = self.config.kinematic_model_device
        self.height_adjust = self.config.height_adjust
        self.height_adjust_mode = self.config.get("height_adjust_mode", "body_origin")
        if self.height_adjust_mode not in ("body_origin", "collision_sole"):
            raise ValueError(f"Unknown height_adjust_mode: {self.height_adjust_mode}")
        self.ground_offset_m = float(self.config.get("ground_offset_m", 0.0))
        if not np.isfinite(self.ground_offset_m) or self.ground_offset_m < 0:
            raise ValueError("ground_offset_m must be finite and non-negative")
        self.root_offset = self.config.root_offset
        self.kinematic_model = None
        self.collision_model = None
        self.collision_data = None
        self.collision_geom_ids = ()
        self.collision_body_names = ()
        if self.height_adjust:
            if self.height_adjust_mode == "body_origin":
                self.kinematic_model = KinematicsModel(
                    file_path=self.config.robot.robot_xml_path,
                    device=self.kinematic_model_device,
                )
            else:
                self._initialize_collision_sole()
        logger.info(
            "[Formatter] height_adjust: {}, mode: {}, ground_offset_m: {}, root offset: {}",
            self.height_adjust, self.height_adjust_mode, self.ground_offset_m, self.root_offset,
        )

    def _initialize_collision_sole(self):
        # Only the explicitly selected geometry-based rule requires MuJoCo.
        import mujoco

        names = self.config.get("collision_sole_body_names", ())
        if isinstance(names, str) or not names or len(set(names)) != len(names):
            raise ValueError("collision_sole_body_names must contain unique foot body names")
        self.collision_body_names = tuple(names)
        model = mujoco.MjModel.from_xml_path(str(self.config.robot.robot_xml_path))
        if (
            model.njnt == 0
            or model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE
            or model.jnt_qposadr[0] != 0
            or np.count_nonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE) != 1
        ):
            raise ValueError("collision_sole requires exactly one leading free root joint")
        # Explicit pairs can enable collisions even with both contact masks zero.
        explicitly_paired = set(map(int, model.pair_geom1)) | set(map(int, model.pair_geom2))
        selected = set()
        supported = {
            int(mujoco.mjtGeom.mjGEOM_SPHERE), int(mujoco.mjtGeom.mjGEOM_CAPSULE),
            int(mujoco.mjtGeom.mjGEOM_ELLIPSOID), int(mujoco.mjtGeom.mjGEOM_CYLINDER),
            int(mujoco.mjtGeom.mjGEOM_BOX), int(mujoco.mjtGeom.mjGEOM_MESH),
        }
        for name in names:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id <= 0:
                raise ValueError(f"Unknown or world collision sole body: {name}")
            descendants = {body_id}
            for candidate in range(body_id + 1, model.nbody):
                if int(model.body_parentid[candidate]) in descendants:
                    descendants.add(candidate)
            body_geoms = {
                i for i in range(model.ngeom)
                if int(model.geom_bodyid[i]) in descendants
                and (model.geom_contype[i] or model.geom_conaffinity[i] or i in explicitly_paired)
            }
            if not body_geoms:
                raise ValueError(f"No collision-enabled geometry on sole body: {name}")
            for geom_id in body_geoms:
                if int(model.geom_type[geom_id]) not in supported:
                    raise ValueError(f"Unsupported sole geometry type on {name}: {model.geom_type[geom_id]}")
            selected.update(body_geoms)
        self.collision_model = model
        self.collision_data = mujoco.MjData(model)
        self.collision_geom_ids = tuple(sorted(selected))

    def _geom_lowest_z(self, geom_id):
        """Exact vertical support of a primitive or compiled convex mesh."""
        import mujoco

        model, data = self.collision_model, self.collision_data
        kind = model.geom_type[geom_id]
        size = model.geom_size[geom_id]
        z = float(data.geom_xpos[geom_id, 2])
        row = data.geom_xmat[geom_id].reshape(3, 3)[2]
        if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            extent = size[0]
        elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            extent = size[0] + size[1] * abs(row[2])
        elif kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            extent = np.linalg.norm(row * size)
        elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            extent = size[0] * np.linalg.norm(row[:2]) + size[1] * abs(row[2])
        elif kind == mujoco.mjtGeom.mjGEOM_BOX:
            extent = np.dot(np.abs(row), size)
        elif kind == mujoco.mjtGeom.mjGEOM_MESH:
            mesh_id = model.geom_dataid[geom_id]
            start = model.mesh_vertadr[mesh_id]
            count = model.mesh_vertnum[mesh_id]
            return z + float(np.min(model.mesh_vert[start:start + count] @ row))
        else:
            raise ValueError(f"Unsupported sole geometry type: {kind}")
        return z - float(extent)

    def _collision_lowest_z(self, qpos):
        import mujoco

        if qpos.shape[1] != self.collision_model.nq:
            raise ValueError(
                f"qpos has {qpos.shape[1]} columns, robot model requires {self.collision_model.nq}"
            )
        lowest = np.inf
        for frame in qpos:
            self.collision_data.qpos[:] = frame
            mujoco.mj_kinematics(self.collision_model, self.collision_data)
            lowest = min(lowest, *(self._geom_lowest_z(i) for i in self.collision_geom_ids))
        return float(lowest)

    def format(self, qpos_list, extras):
        qpos = np.array(qpos_list, dtype=float, copy=True)
        if qpos.ndim != 2 or qpos.shape[0] == 0 or qpos.shape[1] < 7:
            raise ValueError("qpos must be a non-empty (frames, 7 + joints) array")
        if not np.all(np.isfinite(qpos)):
            raise ValueError("qpos must contain only finite values")
        if not np.allclose(np.linalg.norm(qpos[:, 3:7], axis=1), 1.0, atol=1e-5, rtol=0):
            raise ValueError("qpos root quaternions must be unit length in wxyz order")
        root_pos = qpos[:, :3].copy()
        root_rot_xyzw = qpos[:, [4, 5, 6, 3]].copy()
        root_rot = root_rot_xyzw.copy() if self.quat_order == "xyzw" else qpos[:, 3:7].copy()
        dof_pos = qpos[:, 7:].copy()
        lowest_height = None
        height_offset = 0.0
        if self.height_adjust:
            if self.height_adjust_mode == "collision_sole":
                lowest_height = self._collision_lowest_z(qpos)
            else:
                with torch.inference_mode():
                    body_pos, _ = self.kinematic_model.forward_kinematics(
                        torch.from_numpy(root_pos).float().to(self.kinematic_model_device),
                        # FK expects xyzw regardless of requested output order.
                        torch.from_numpy(root_rot_xyzw).float().to(self.kinematic_model_device),
                        torch.from_numpy(dof_pos).float().to(self.kinematic_model_device),
                    )
                lowest_height = torch.min(body_pos[..., 2]).item()
            height_offset = self.ground_offset_m - lowest_height
            root_pos[:, 2] += height_offset
        xy_offset = root_pos[0, :2].copy() if self.root_offset else np.zeros(2)
        root_pos[:, :2] -= xy_offset
        return {
            "fps": extras["fps"],
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "formatter_metadata": {
                "schema_version": 1,
                "height_adjust": bool(self.height_adjust),
                "height_adjust_mode": self.height_adjust_mode,
                "ground_offset_m": self.ground_offset_m,
                "measured_minimum_z_m": lowest_height,
                "applied_constant_z_offset_m": height_offset,
                "removed_initial_xy_m": xy_offset.tolist(),
                "quaternion_order": self.quat_order,
                "collision_sole_body_names": list(self.collision_body_names),
                "collision_sole_geom_ids": list(self.collision_geom_ids),
                "contact_or_dynamics_validated": False,
            },
        }
