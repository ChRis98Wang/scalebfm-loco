"""ScaleBFM's native Transformer with the deployment observation/PD contract.

No Kit, TensorRT, training runner, network receiver or simulator is started on
import. The exported artifact has a fixed batch/history/future shape; control
mode remains a tensor input. This is inference, not a new learned policy.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
NETWORK_SOURCE = ROOT / "ScaleTrack/source/my_rsl_rl/my_rsl_rl/networks/humanoid_transformer.py"


def native_networks():
    # Loading this dependency-free leaf preserves the original actor, without
    # importing the training package __init__ and its unrelated dependencies.
    name = "bfm_native_deployment_networks"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, NETWORK_SOURCE)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def load_native_actor(checkpoint, metadata):
    architecture = metadata["policy_architecture"]
    if (architecture["prop_obs_dim"], architecture["action_dim"], architecture["output_dim"],
            architecture["task_obs_dim"]) != (64, 29, 29, 267):
        raise ValueError("This deployment contract requires the deterministic 29 DoF BFM actor")
    nets = native_networks()
    actor = nets.HumanoidTransformer(
        prop_obs_dim=64, action_dim=29, output_dim=29,
        embed_dim=architecture["embedding_dim"], num_heads=architecture["num_heads"],
        ff_dim=architecture["ff_dim"], num_layers=architecture["num_layers"])
    embedder = nets.TaskEmbedder(
        task_obs_dim=267, embedding_dim=architecture["embedding_dim"],
        reduced_task_dim=architecture.get("reduced_task_dim"),
        hidden_dims=architecture.get("task_embedder_hidden_dims"))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload["model_state_dict"]
    for prefix, module in (("actor.", actor), ("actor_task_embedder.", embedder)):
        selected = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
        if not selected or not all(torch.isfinite(value).all() for value in selected.values()):
            raise ValueError(f"Invalid or missing actor tensors: {prefix}")
        module.load_state_dict(selected, strict=True)
        module.eval().requires_grad_(False)
    return actor, embedder


def quat_mul(left, right):
    w1, v1 = left[..., :1], left[..., 1:]
    w2, v2 = right[..., :1], right[..., 1:]
    return torch.cat((w1 * w2 - (v1 * v2).sum(-1, keepdim=True),
                      w1 * v2 + w2 * v1 + torch.cross(v1, v2, dim=-1)), dim=-1)


def quat_conjugate(quat):
    return torch.cat((quat[..., :1], -quat[..., 1:]), dim=-1)


def quat_apply(quat, vector):
    tangent = 2 * torch.cross(quat[..., 1:], vector, dim=-1)
    return vector + quat[..., :1] * tangent + torch.cross(quat[..., 1:], tangent, dim=-1)


def mode_mappings(table):
    if table.shape != (8, 14) or not torch.all((table == 0) | (table == 1)) or torch.any(table.sum(-1) == 0):
        raise ValueError("Expected eight nonempty binary fourteen-link masks")
    return torch.cat([table.repeat_interleave(width, -1) for width in (3, 3, 6, 6)]
                     + [torch.ones((8, 1), dtype=table.dtype, device=table.device)], -1)


class PortableBFMPolicy(nn.Module):
    def __init__(self, actor, task_embedder, metadata, xml_path):
        super().__init__()
        import mujoco

        self.actor = actor
        self.task_embedder = task_embedder
        joints = metadata["joint_names"]
        if len(joints) != 29 or len(set(joints)) != 29 or metadata["action_names"] != joints:
            raise ValueError("Require identical, explicit unique 29-joint observation/action order")
        selected = metadata["selected_body_names"]
        if len(selected) != 14 or len(set(selected)) != 14 or selected[0] != "pelvis":
            raise ValueError("Expected fourteen unique tracked links starting at pelvis")
        self.context_len = int(metadata["history_buffer_size"])
        self.future_len = len(metadata["future_idx"])
        if not 1 <= self.context_len <= 64 or not 1 <= self.future_len <= 16:
            raise ValueError("Unbounded history/future contract")
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        if (model.nq, model.nv, model.nbody, model.njnt) != (36, 35, 31, 30):
            raise ValueError("Expected the free-root G1 29 DoF, without fingers or objects")
        body_names = [model.body(i).name for i in range(1, model.nbody)]
        xml_joints = [model.joint(i).name for i in range(1, model.njnt)]
        if (body_names[0] != "pelvis" or set(joints) != set(xml_joints)
                or any(name not in body_names for name in selected)
                or not np.array_equal(model.jnt_bodyid, np.arange(1, 31))
                or not np.all(model.jnt_type[1:] == mujoco.mjtJoint.mjJNT_HINGE)
                or not np.allclose(model.jnt_pos[1:], 0, atol=1e-12)):
            raise ValueError("Unsupported FK hierarchy, joint placement or named-order mismatch")
        # Python topology constants are intentionally fixed in the traced model.
        self.parents = tuple(int(value - 1) for value in model.body_parentid[2:])
        if any(parent < 0 or parent >= index + 1 for index, parent in enumerate(self.parents)):
            raise ValueError("FK parents must precede children")
        self.register_buffer("xml_joint_indices", torch.tensor([joints.index(name) for name in xml_joints]))
        self.register_buffer("selected_indices", torch.tensor([body_names.index(name) for name in selected]))
        self.register_buffer("local_pos", torch.tensor(model.body_pos[2:], dtype=torch.float32))
        self.register_buffer("local_quat", torch.tensor(model.body_quat[2:], dtype=torch.float32))
        self.register_buffer("joint_axis", torch.tensor(model.jnt_axis[1:], dtype=torch.float32))
        for name in ("default_dof_pos", "action_scale"):
            value = torch.tensor(metadata[name], dtype=torch.float32)
            if value.shape != (29,) or not torch.isfinite(value).all():
                raise ValueError(f"Invalid {name}")
            if name == "action_scale" and not torch.all(value > 0):
                raise ValueError("Action scales must be positive")
            self.register_buffer(name, value)
        table = torch.tensor(metadata["mode_table"], dtype=torch.float32)
        self.register_buffer("mode_table", table)
        self.register_buffer("mode_mapping", mode_mappings(table))

    def body_fk(self, joint_pos):
        angles = joint_pos[:, self.xml_joint_indices] / 2
        rotations = torch.cat((torch.cos(angles)[..., None],
                               self.joint_axis[None] * torch.sin(angles)[..., None]), -1)
        zero = torch.zeros_like(joint_pos[:, :3])
        identity = torch.cat((torch.ones_like(joint_pos[:, :1]), zero), -1)
        positions, quaternions = [zero], [identity]
        for index, parent in enumerate(self.parents):
            translation = self.local_pos[index][None].expand_as(zero)
            local_quat = self.local_quat[index][None].expand_as(identity)
            positions.append(positions[parent] + quat_apply(quaternions[parent], translation))
            quaternions.append(quat_mul(quaternions[parent], quat_mul(local_quat, rotations[:, index])))
        return (torch.stack(positions, 1)[:, self.selected_indices],
                torch.stack(quaternions, 1)[:, self.selected_indices])

    def observations(self, root_quat_buffer, base_ang_vel_buffer, dof_pos_buffer, dof_vel_buffer,
                     target_body_pos_future_to_robot_base, target_body_rot_future_to_robot_base,
                     mode_index, time_offsets):
        gravity = torch.zeros_like(base_ang_vel_buffer)
        gravity[..., 2] = -1
        prop = torch.cat((quat_apply(quat_conjugate(root_quat_buffer), gravity), base_ang_vel_buffer,
                          dof_pos_buffer - self.default_dof_pos, dof_vel_buffer * .05), -1)
        position, quaternion = self.body_fk(dof_pos_buffer[:, -1])
        relative_position = target_body_pos_future_to_robot_base - position[:, None]
        target = target_body_rot_future_to_robot_base
        current = quaternion[:, None].expand_as(target)
        relative_rotation = quat_mul(target, quat_conjugate(current))
        tangent = torch.zeros_like(target_body_pos_future_to_robot_base)
        normal = torch.zeros_like(tangent)
        tangent[..., 0] = 1
        normal[..., 2] = 1
        absolute_tan_norm = torch.cat((quat_apply(target, tangent), quat_apply(target, normal)), -1)
        relative_tan_norm = torch.cat((quat_apply(relative_rotation, tangent), quat_apply(relative_rotation, normal)), -1)
        task = torch.cat((target_body_pos_future_to_robot_base.flatten(2), relative_position.flatten(2),
                          absolute_tan_norm.flatten(2), relative_tan_norm.flatten(2), time_offsets), -1)
        masked = task * self.mode_mapping[mode_index][:, None]
        return prop, torch.cat((masked, self.mode_table[mode_index][:, None].expand(-1, self.future_len, -1)), -1)

    def forward(self, root_quat_buffer, base_ang_vel_buffer, dof_pos_buffer, dof_vel_buffer,
                last_action_buffer, target_body_pos_future_to_robot_base,
                target_body_rot_future_to_robot_base, mode_index, time_offsets):
        prop, task = self.observations(root_quat_buffer, base_ang_vel_buffer, dof_pos_buffer,
                                      dof_vel_buffer, target_body_pos_future_to_robot_base,
                                      target_body_rot_future_to_robot_base, mode_index, time_offsets)
        action = self.actor(prop, last_action_buffer, self.task_embedder(task))
        return action * self.action_scale + self.default_dof_pos, action


def example_inputs(metadata, seed=42):
    """Diverse fixed-seed nonzero inputs, not measurements of task competence."""
    generator = torch.Generator().manual_seed(seed)
    history, future = metadata["history_buffer_size"], len(metadata["future_idx"])
    def rand(*shape):
        return torch.randn(shape, generator=generator)
    roots = rand(1, history, 4)
    roots /= roots.norm(dim=-1, keepdim=True)
    targets = rand(1, future, 14, 4)
    targets /= targets.norm(dim=-1, keepdim=True)
    return (roots, rand(1, history, 3), torch.tensor(metadata["default_dof_pos"]) + .15 * rand(1, history, 29),
            rand(1, history, 29), .2 * rand(1, history, 29), rand(1, future, 14, 3), targets,
            torch.tensor([0], dtype=torch.long), torch.tensor(metadata["future_idx"], dtype=torch.long)[None, :, None])


def independent_observations(inputs, metadata, xml_path):
    """NumPy/SciPy rotations + MuJoCo FK; does not reuse the wrapper math.

    Provides the original actor's three inputs for export-contract verification.
    This is not an IsaacLab-runtime observation comparison.
    """
    import mujoco
    from scipy.spatial.transform import Rotation

    root, angular, joints, velocity, actions, target_pos, target_quat, mode, offsets = (
        item.detach().cpu().numpy() for item in inputs)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    data.qpos[:7] = [0, 0, 0, 1, 0, 0, 0]
    for name, angle in zip(metadata["joint_names"], joints[0, -1]):
        data.qpos[model.joint(name).qposadr[0]] = angle
    mujoco.mj_forward(model, data)
    ids = [model.body(name).id for name in metadata["selected_body_names"]]
    def rotation(wxyz):
        return Rotation.from_quat(wxyz.reshape(-1, 4)[:, [1, 2, 3, 0]])
    gravity = rotation(root).inv().apply(np.tile([0., 0., -1.], (root.shape[1], 1))).reshape(1, -1, 3)
    prop = np.concatenate((gravity, angular, joints - np.array(metadata["default_dof_pos"]), velocity * .05), -1)
    target_rotation = rotation(target_quat)
    current_quat = np.broadcast_to(data.xquat[ids][None, None], target_quat.shape)
    relative_rotation = target_rotation * rotation(current_quat).inv()
    def tan_norm(value):
        count = target_quat.size // 4
        return np.concatenate((value.apply(np.tile([1., 0., 0.], (count, 1))),
                               value.apply(np.tile([0., 0., 1.], (count, 1)))), -1).reshape(1, -1, 84)
    task = np.concatenate((target_pos.reshape(1, -1, 42),
                           (target_pos - data.xpos[ids][None, None]).reshape(1, -1, 42),
                           tan_norm(target_rotation), tan_norm(relative_rotation), offsets), -1)
    mask = np.asarray(metadata["mode_table"])[int(mode[0])]
    mapping = np.concatenate([np.repeat(mask, width) for width in (3, 3, 6, 6)] + [[1]])
    task = np.concatenate((task * mapping, np.broadcast_to(mask, (1, task.shape[1], 14))), -1)
    return tuple(torch.tensor(value, dtype=torch.float32) for value in (prop, actions, task))
