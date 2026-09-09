"""Opt-in rigid target for playback/evaluation; CLI validation does not start Kit."""

from dataclasses import asdict, dataclass
import json
import math
from numbers import Real
from pathlib import Path


@dataclass(frozen=True)
class TargetObjectSpec:
    """Engineering starting values for a box, NOT measured material calibration."""

    size: tuple = (0.30, 0.30, 0.30)
    position: tuple = (0.9, -0.7, 0.16)
    mass: float = 1.0
    static_friction: float = 0.6
    dynamic_friction: float = 0.5
    restitution: float = 0.05
    contact_offset: float = 0.005
    rest_offset: float = 0.0
    linear_damping: float = 0.0
    angular_damping: float = 0.02
    color: tuple = (0.62, 0.32, 0.10)
    roughness: float = 0.75

    def __post_init__(self):
        for name in ("size", "position", "color"):
            value = getattr(self, name)
            if not isinstance(value, (list, tuple)) or len(value) != 3:
                raise ValueError(f"{name} must contain three numbers")
            object.__setattr__(self, name, tuple(value))
        for name, value in asdict(self).items():
            values = value if isinstance(value, tuple) else (value,)
            if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) for v in values):
                raise ValueError(f"{name} must be finite numeric values")
        if self.mass <= 0 or min(self.size) <= 0:
            raise ValueError("mass and all dimensions must be positive")
        if self.position[2] < self.size[2] / 2 + self.rest_offset:
            raise ValueError("position puts the target below its ground resting height")
        if not 0 <= self.dynamic_friction <= self.static_friction:
            raise ValueError("Require 0 <= dynamic_friction <= static_friction")
        if not 0 <= self.restitution <= 1 or not 0 <= self.roughness <= 1 or not all(0 <= c <= 1 for c in self.color):
            raise ValueError("restitution, roughness and color must be in [0, 1]")
        if not 0 <= self.rest_offset < self.contact_offset:
            raise ValueError("Require 0 <= rest_offset < contact_offset")
        if min(self.linear_damping, self.angular_damping) < 0:
            raise ValueError("damping must be nonnegative")

    def to_dict(self):
        return {**asdict(self), "shape": "cuboid", "calibrated": False,
                "friction_combine_mode": "multiply", "restitution_combine_mode": "max"}


def add_target_object_args(parser):
    parser.add_argument("--target_object", action="store_true", help="Add a physical target box; not a grasp-trained policy")
    parser.add_argument("--target_config", type=Path, help="Local JSON containing TargetObjectSpec parameters")
    parser.add_argument("--target_position", type=float, nargs=3, metavar=("X", "Y", "Z"),
                        help="Override initial box centre in environment-local metres")


def target_spec_from_args(args):
    if not args.target_object:
        if args.target_config is not None or args.target_position is not None:
            raise ValueError("target_config/target_position require --target_object")
        return None
    values = {}
    if args.target_config is not None:
        with Path(args.target_config).open(encoding="utf-8") as stream:
            values = json.load(stream)
        if not isinstance(values, dict):
            raise ValueError("target_config must be a JSON object")
    if args.target_position is not None:
        values["position"] = tuple(args.target_position)
    try:
        return TargetObjectSpec(**values)
    except TypeError as error:
        raise ValueError(f"Invalid target_config fields: {error}") from error


def attach_target_object(env_cfg, spec):
    if spec is None:
        return
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg
    from isaaclab.managers import EventTermCfg

    if getattr(env_cfg.scene, "target_object", None) is not None or getattr(env_cfg.events, "reset_target_object", None) is not None:
        raise ValueError("A target object or target reset event already exists")
    obj = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TargetObject",
        spawn=sim_utils.CuboidCfg(
            size=spec.size,
            mass_props=sim_utils.MassPropertiesCfg(mass=spec.mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False, disable_gravity=False,
                linear_damping=spec.linear_damping, angular_damping=spec.angular_damping,
                solver_position_iteration_count=16, solver_velocity_iteration_count=4,
                max_depenetration_velocity=2.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True, contact_offset=spec.contact_offset, rest_offset=spec.rest_offset,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=spec.static_friction, dynamic_friction=spec.dynamic_friction,
                restitution=spec.restitution, friction_combine_mode="multiply", restitution_combine_mode="max",
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=spec.color, roughness=spec.roughness),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=spec.position),
    )
    env_cfg.scene.target_object = obj
    # Deliberately no disable_flag: the object must reset during evaluation too.
    env_cfg.events.reset_target_object = EventTermCfg(func=reset_target_object, mode="reset")


def _tensor(value):
    """Accept the installed SDK's ProxyArray as well as Torch-backed test/older data."""
    return getattr(value, "torch", value)


def _physical_anchor_position(command):
    # The optional local-tracking player overrides command.robot_anchor_pos_w.
    # Object telemetry must continue to report the actual simulated robot.
    return _tensor(command.robot.data.body_pos_w)[:, command.robot_anchor_body_index]


def reset_target_object(env, env_ids=None):
    import torch

    obj = env.scene["target_object"]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    pose = _tensor(obj.data.default_root_pose)[env_ids].clone()
    velocity = _tensor(obj.data.default_root_vel)[env_ids].clone()
    pose[:, :3] += env.scene.env_origins[env_ids]
    obj.reset(env_ids)
    obj.write_root_pose_to_sim(pose, env_ids=env_ids)
    obj.write_root_velocity_to_sim(velocity, env_ids=env_ids)


def target_status(env, command):
    import torch

    position = _tensor(env.scene["target_object"].data.root_pos_w)
    local = position - env.scene.env_origins
    distance = torch.linalg.vector_norm(position - _physical_anchor_position(command), dim=-1)
    if not torch.isfinite(local).all() or not torch.isfinite(distance).all():
        raise RuntimeError("Nonfinite target object state")
    return {"position_local": local[0].detach().cpu().tolist(), "root_distance": float(distance[0])}


def target_metrics(env, command):
    """Diagnostics only: proximity and object motion do not establish grasp success."""
    import torch

    data = env.scene["target_object"].data
    position = _tensor(data.root_pos_w)
    return {
        "target_root_distance": torch.linalg.vector_norm(position - _physical_anchor_position(command), dim=-1),
        "target_height": position[:, 2],
        "target_speed": torch.linalg.vector_norm(_tensor(data.root_lin_vel_w), dim=-1),
    }
