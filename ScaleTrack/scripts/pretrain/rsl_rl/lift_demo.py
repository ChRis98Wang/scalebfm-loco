"""IsaacLab-only contact lift development task; pure logic is import-safe.

The planner changes sparse reference targets, never robot/object physical state.
Default scene values are declared simulation assumptions, not measured materials.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import numpy as np


@dataclass(frozen=True)
class LiftSpec:
    box_size: tuple = (.24, .30, .24)
    box_xyz: tuple = (.43, 0., .685)
    box_mass: float = 1.
    support_size: tuple = (.40, .46, .56)
    support_xyz: tuple = (.43, 0., .28)
    lift_height: float = .12
    minimum_clearance: float = .08
    contact_threshold: float = .2
    grip_time: float = .20
    hold_time: float = 2.
    release_time: float = 1.
    timeout: float = 20.

    @property
    def support_top(self):
        return self.support_xyz[2] + self.support_size[2] / 2

    @property
    def goal_xyz(self):
        return (*self.support_xyz[:2], self.support_top + self.box_size[2] / 2)

    def validate(self):
        for key, value in asdict(self).items():
            values = value if isinstance(value, tuple) else (value,)
            if any(isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) for x in values):
                raise ValueError(f"Invalid {key}")
        if min(*self.box_size, *self.support_size, self.box_mass, self.contact_threshold,
               self.grip_time, self.hold_time, self.release_time, self.timeout) <= 0:
            raise ValueError("Sizes/mass/thresholds/times must be positive")
        if not 0 < self.minimum_clearance < self.lift_height <= .20:
            raise ValueError("Invalid lift height or clearance")
        if self.box_xyz[2] - self.box_size[2] / 2 < self.support_top:
            raise ValueError("Box initially intersects support")
        if any(self.box_size[k] >= self.support_size[k] for k in (0, 1)):
            raise ValueError("Box footprint must fit support")


def rotation_wxyz(quaternion):
    q = np.asarray(quaternion, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q) - 1) > 1e-3:
        raise ValueError("Require unit wxyz quaternion")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def box_geometry(xyz, quaternion, spec):
    xyz = np.asarray(xyz, dtype=float)
    if xyz.shape != (3,) or not np.isfinite(xyz).all():
        raise ValueError("Invalid box centre")
    rotation = rotation_wxyz(quaternion)
    half = np.asarray(spec.box_size) / 2
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * half
    corners = corners @ rotation.T + xyz
    tilt = math.acos(float(np.clip(rotation[2, 2], -1, 1)))
    inside = bool(np.all(np.abs(corners[:, :2] - np.asarray(spec.support_xyz[:2]))
                         <= np.asarray(spec.support_size[:2]) / 2))
    return dict(bottom_z=float(corners[:, 2].min()), tilt_rad=tilt,
                inside_support_xy=inside, corners_world=corners.tolist())


class LiftPlanner:
    """Feedback-gated D1 prototype. A terminal state alone is not acceptance."""

    def __init__(self, initial_points, initial_quats, spec=None):
        self.spec = spec or LiftSpec()
        self.spec.validate()
        self.initial = np.asarray(initial_points, dtype=float).copy()
        self.quats = np.asarray(initial_quats, dtype=float).copy()
        if self.initial.shape != (3, 3) or self.quats.shape != (3, 4) or not np.isfinite(self.initial).all():
            raise ValueError("Require pelvis/left wrist/right wrist targets")
        for q in self.quats:
            rotation_wxyz(q)
        self.phase = "SETTLE"
        self.entered = self.time = 0.
        self.good_since = None
        self.lost_since = None
        self.events = []
        self.reason = None
        self.grasp_xyz = None
        self.grasp_relative = None
        self.half_gap = .24

    def _enter(self, phase, now, reason):
        self.events.append(dict(from_phase=self.phase, to_phase=phase, time_s=now, reason=reason))
        self.phase, self.entered, self.good_since = phase, now, None

    def _continuous(self, valid, now, duration):
        if not valid:
            self.good_since = None
            return False
        if self.good_since is None:
            self.good_since = now
        return now - self.good_since >= duration - 1e-9

    def fail(self, now, reason):
        self.reason = reason
        self._enter("FAILED", now, reason)

    def update(self, now, measurement):
        if not math.isfinite(now) or now < self.time:
            raise ValueError("Non-monotonic simulation clock")
        self.time = now
        if self.phase in ("SUCCESS", "FAILED"):
            return
        required = ("box_xyz", "box_bottom_z", "box_tilt_rad", "box_speed", "box_angular_speed",
                    "left_force", "right_force", "support_force", "other_robot_force", "root_z", "root_up")
        if any(not np.isfinite(measurement[key]).all() for key in required):
            self.fail(now, "nonfinite_telemetry")
            return
        m, s = measurement, self.spec
        if m["root_z"] < .30 or m["root_up"] < math.cos(math.radians(75)):
            self.fail(now, "robot_safety_guard")
            return
        if m["other_robot_force"] > s.contact_threshold:
            self.fail(now, "box_contact_with_other_robot_body")
            return
        if now >= s.timeout:
            self.fail(now, "task_timeout")
            return
        elapsed = now - self.entered
        bilateral = m["left_force"] >= s.contact_threshold and m["right_force"] >= s.contact_threshold
        if self.phase in ("LIFT", "HOLD", "LOWER"):
            if not bilateral:
                self.lost_since = now if self.lost_since is None else self.lost_since
            else:
                self.lost_since = None
            if self.lost_since is not None and now - self.lost_since > .10 + 1e-9:
                self.fail(now, "bilateral_contact_lost")
                return
            if m["box_tilt_rad"] > math.radians(15):
                self.fail(now, "box_tilt_exceeded")
                return
            if self.grasp_relative is not None and "wrist_xyz" in m:
                relative = np.asarray(m["box_xyz"]) - np.mean(m["wrist_xyz"], axis=0)
                if np.linalg.norm(relative - self.grasp_relative) > .05:
                    self.fail(now, "box_slip_exceeded")
                    return
        if self.phase == "SETTLE" and elapsed >= 2:
            self._enter("APPROACH", now, "initial_settle_elapsed")
        elif self.phase == "APPROACH" and elapsed >= 3:
            self._enter("CLOSE", now, "side_approach_completed")
        elif self.phase == "CLOSE":
            self.half_gap = max(.13, .24 - elapsed * .025)
            if self._continuous(bilateral, now, s.grip_time):
                self.grasp_xyz = np.asarray(m["box_xyz"]).copy()
                if "wrist_xyz" in m:
                    self.grasp_relative = self.grasp_xyz - np.mean(m["wrist_xyz"], axis=0)
                self._enter("LIFT", now, "measured_bilateral_contact_for_020s")
            elif elapsed >= 5:
                self.fail(now, "grip_not_established")
        elif self.phase == "LIFT":
            lifted = m["box_bottom_z"] - s.support_top >= s.minimum_clearance
            if self._continuous(bilateral and lifted and m["support_force"] < s.contact_threshold, now, .20):
                self._enter("HOLD", now, "measured_airborne_clearance")
            elif elapsed >= 3:
                self.fail(now, "lift_height_not_reached")
        elif self.phase == "HOLD":
            lifted = m["box_bottom_z"] - s.support_top >= s.minimum_clearance
            if not lifted or m["support_force"] >= s.contact_threshold:
                self.fail(now, "lost_airborne_hold")
            elif self._continuous(bilateral, now, s.hold_time):
                self._enter("LOWER", now, "measured_airborne_hold_for_2s")
        elif self.phase == "LOWER":
            supported = (m["support_force"] >= .8 * s.box_mass * 9.81
                         and abs(m["box_bottom_z"] - s.support_top) <= .01
                         and m["box_speed"] <= .05 and m["box_angular_speed"] <= .10)
            if self._continuous(supported, now, .20):
                self._enter("RELEASE", now, "support_carries_box_weight")
            elif elapsed >= 4:
                self.fail(now, "support_not_recovered")
        elif self.phase == "RELEASE":
            stable = (m["left_force"] < s.contact_threshold and m["right_force"] < s.contact_threshold
                      and m["support_force"] >= .8 * s.box_mass * 9.81 and m["box_speed"] <= .05
                      and m["box_angular_speed"] <= .10 and m.get("inside_support_xy", False)
                      and np.linalg.norm(np.asarray(m["box_xyz"]) - s.goal_xyz) <= .03)
            if self._continuous(stable, now, s.release_time):
                self._enter("SUCCESS", now, "supported_released_at_xyz_goal_for_1s")

    def targets(self, measurement):
        points, quats = self.initial.copy(), self.quats.copy()
        if self.phase in ("SETTLE", "SUCCESS", "FAILED"):
            return points, quats
        box = np.asarray(measurement["box_xyz"])
        z = self.spec.goal_xyz[2]
        gap = self.half_gap
        if self.phase == "APPROACH":
            gap = .24
        elif self.phase in ("LIFT", "HOLD", "LOWER"):
            if self.phase == "LIFT":
                z = self.grasp_xyz[2] + self.spec.lift_height * min(1., (self.time-self.entered)/2.)
            elif self.phase == "HOLD":
                z = self.grasp_xyz[2] + self.spec.lift_height
            else:
                z = self.spec.goal_xyz[2] + self.spec.lift_height * max(0., 1.-(self.time-self.entered)/2.)
        elif self.phase == "RELEASE":
            gap = .27
        for row, sign in ((1, 1), (2, -1)):
            points[row] = [box[0] - .09, box[1] + sign * gap, z]
            quats[row] = [1., 0., 0., 0.]
        return points, quats


def contact_filter_paths(body_names):
    # The native USD has an articulation container named pelvis and a separate
    # rigid pelvis body below it. Never filter on the container itself.
    return [f"{{ENV_REGEX_NS}}/Robot/pelvis/{name}" for name in body_names] + ["{ENV_REGEX_NS}/LiftSupport"]


def attach_lift_scene(env_cfg, body_names, spec):
    """Add objects and one-to-many filtered contact reporting, before gym.make."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
    from isaaclab.sensors import ContactSensorCfg
    from target_object import TargetObjectSpec, attach_target_object
    spec.validate()
    attach_target_object(env_cfg, TargetObjectSpec(size=spec.box_size, position=spec.box_xyz,
        mass=spec.box_mass, static_friction=.8, dynamic_friction=.6, restitution=0., angular_damping=0.))
    env_cfg.scene.target_object.spawn.activate_contact_sensors = True
    env_cfg.scene.lift_support = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/LiftSupport",
        spawn=sim_utils.CuboidCfg(size=spec.support_size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.005, rest_offset=0.),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=.8, dynamic_friction=.6,
                restitution=0., friction_combine_mode="multiply", restitution_combine_mode="max"),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.22, .28, .34), roughness=.8)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=spec.support_xyz))
    env_cfg.scene.lift_goal = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/LiftGoal",
        spawn=sim_utils.CuboidCfg(size=(.38, .44, .002),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.08, .65, .20))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(*spec.support_xyz[:2], spec.support_top + .001)))
    paths = contact_filter_paths(body_names)
    env_cfg.scene.lift_contacts = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/TargetObject",
        filter_prim_paths_expr=paths, history_length=4, update_period=0.,
        track_friction_forces=True, max_contact_data_count_per_prim=256)
    env_cfg.scene.lazy_sensor_update = False
    return paths
