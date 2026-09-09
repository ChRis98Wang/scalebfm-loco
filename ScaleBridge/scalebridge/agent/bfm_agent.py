import os
import json
import importlib
import hashlib
import re
import torch
from loguru import logger
from scalebridge.agent.base_agent import BaseAgent

CONTROL_MODE_DICT = {
    0: "Pelvis",
    1: "Double Hands",
    2: "Pelvis and Double Hands",
    3: "Double Hands and Feet",
    4: "Pelvis and Double Hands and Feet",
    5: "Left and Right Shoulders, Elbows and Hands",
    6: "Pelvis and Left and Right Shoulders, Elbows and Hands",
    7: "Pelvis, Torso, Left and Right Shoulders, Elbows and Hands, Left and Right Hips, Knees and Feet"
}

class BFMAgent(BaseAgent):
    def __init__(self, config, device):

        super().__init__(config, device)

        if type(self.config.control_mode) is not int or self.config.control_mode not in CONTROL_MODE_DICT:
            raise ValueError("Control mode must be an integer in 0..7")
        self.control_mode = torch.tensor([self.config.control_mode], dtype=torch.long, device=self.device)
        logger.info(f"[Agent] Activating control mode {self.config.control_mode}: {CONTROL_MODE_DICT[self.config.control_mode]}")

    def _load_policy(self):

        path_base, _ = os.path.splitext(self.checkpoint)
        meta_path = path_base + "_metadata.json"
        logger.info(f"[Agent] loading order dict from: {meta_path}")

        with open(meta_path, "r") as f:
            self.meta_data_dict = json.load(f)

        # Old official artifacts used the _tensorrt filename convention before
        # an explicit backend field existed. Plain TorchScript needs no TRT.
        self.runtime_backend = self.meta_data_dict.get("runtime_backend",
            "tensorrt" if "_tensorrt" in os.path.basename(path_base) else "torchscript")
        if self.runtime_backend not in ("torchscript", "tensorrt"):
            raise ValueError(f"Unsupported policy backend: {self.runtime_backend}")
        self._strict_contract = "export_batch_size" in self.meta_data_dict
        if self.runtime_backend == "torchscript":
            if "export_verification" in self.meta_data_dict and self.meta_data_dict["export_verification"] != "PASS":
                raise ValueError("TorchScript export verification did not pass")
            declared = self.meta_data_dict.get("policy_sha256")
            if declared is not None:
                digest = hashlib.sha256()
                with open(self.checkpoint, "rb") as stream:
                    for block in iter(lambda: stream.read(1 << 20), b""):
                        digest.update(block)
                if not isinstance(declared, str) or not re.fullmatch("[0-9a-f]{64}", declared) or digest.hexdigest() != declared:
                    raise ValueError("TorchScript policy SHA256 differs from export metadata")
        if self._strict_contract:
            self._validate_export_contract()
        if self.runtime_backend == "tensorrt":
            if torch.device(self.device).type != "cuda":
                raise ValueError("TensorRT policies require a CUDA device")
            importlib.import_module("torch_tensorrt")
        logger.info(f"[Agent] Loading {self.runtime_backend} checkpoint from {self.checkpoint}")
        self.policy = torch.jit.load(self.checkpoint, map_location=self.device)
        self.policy.eval()

        for key, item in self.meta_data_dict.items():
            logger.debug(f"Metadata {key}: {item}")
        
    def get_meta_data(self):
        return self.meta_data_dict

    def _validate_export_contract(self):
        meta = self.meta_data_dict
        if (type(meta["export_batch_size"]) is not int or meta["export_batch_size"] != 1
                or meta.get("export_verification") != "PASS" or "policy_sha256" not in meta
                or meta.get("inference_input_quaternion_order") != "wxyz"):
            raise ValueError("New exports require a verified batch-one wxyz input contract and policy hash")
        if type(meta.get("history_buffer_size")) is not int or meta["history_buffer_size"] < 1:
            raise ValueError("Invalid exported history length")
        for field, count in (("joint_names", 29), ("action_names", 29), ("selected_body_names", 14)):
            names = meta.get(field)
            if not isinstance(names, list) or len(names) != count or any(not isinstance(n, str) or not n for n in names) or len(set(names)) != count:
                raise ValueError(f"Invalid exported {field}")
        if set(meta["joint_names"]) != set(meta["action_names"]):
            raise ValueError("Exported action and observation joint sets differ")
        offsets = meta.get("future_idx")
        if (not isinstance(offsets, list) or not offsets or offsets[0] != 0
                or any(type(i) is not int or i < 0 for i in offsets)
                or offsets != sorted(set(offsets))):
            raise ValueError("Invalid exported future frame indices")

    def _validate_inputs(self, obs):
        meta = self.meta_data_dict
        batch, history, future = meta["export_batch_size"], meta["history_buffer_size"], len(meta["future_idx"])
        shapes = {"root_quat_buffer": (batch, history, 4), "base_ang_vel_buffer": (batch, history, 3),
                  "dof_pos_buffer": (batch, history, 29), "dof_vel_buffer": (batch, history, 29),
                  "actions_buffer": (batch, history, 29),
                  "target_body_pos_future_to_robot_base": (batch, future, 14, 3),
                  "target_body_rot_future_to_robot_base": (batch, future, 14, 4),
                  "future_time_offsets": (batch, future, 1)}
        device = torch.device(self.device)
        for name, shape in shapes.items():
            value = obs.get(name)
            dtype = torch.int64 if name == "future_time_offsets" else torch.float32
            if (not isinstance(value, torch.Tensor) or value.shape != shape or value.dtype != dtype
                    or value.device.type != device.type or (device.index is not None and value.device.index != device.index)
                    or not torch.isfinite(value).all()):
                raise ValueError(f"Export input {name} has wrong shape/dtype/device or nonfinite values")
        for name in ("root_quat_buffer", "target_body_rot_future_to_robot_base"):
            norm = torch.linalg.vector_norm(obs[name], dim=-1)
            if not torch.allclose(norm, torch.ones_like(norm), atol=1e-4, rtol=0):
                raise ValueError(f"Export input {name} requires unit wxyz quaternions")
        expected = torch.tensor(meta["future_idx"], dtype=torch.int64, device=self.device)[None, :, None]
        if not torch.equal(obs["future_time_offsets"], expected):
            raise ValueError("Future frame values differ from exported policy contract")
        if (self.control_mode.shape != (batch,) or self.control_mode.dtype != torch.int64
                or self.control_mode.device != expected.device or not torch.all((self.control_mode >= 0) & (self.control_mode <= 7))):
            raise ValueError("Export control mode requires one int64 value in 0..7")
    
    @torch.inference_mode()
    def get_action(self, obs_dict):
        if self._strict_contract:
            self._validate_inputs(obs_dict)
        result = self.policy(
            *[
                obs_dict["root_quat_buffer"],
                obs_dict["base_ang_vel_buffer"],
                obs_dict["dof_pos_buffer"],
                obs_dict["dof_vel_buffer"],
                obs_dict["actions_buffer"],
                obs_dict["target_body_pos_future_to_robot_base"],
                obs_dict["target_body_rot_future_to_robot_base"],
                self.control_mode,
                obs_dict["future_time_offsets"]
            ]
        )
        target_count = len(self.meta_data_dict.get("target_joint_names", self.meta_data_dict["action_names"]))
        action_count = len(self.meta_data_dict["action_names"])
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            raise ValueError("Policy must return (PD joint targets, raw policy action)")
        if any(value.shape != (1, count) or not torch.isfinite(value).all()
               for value, count in zip(result, (target_count, action_count))):
            raise ValueError("Policy returned invalid/nonfinite joint targets or actions")
        return result

    def close(self):
        self.policy = None
