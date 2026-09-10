"""V2: measured force preloading before lift; unchanged robot and scene physics."""
import numpy as np

from lift_demo import LiftPlanner


class PreloadLiftPlanner(LiftPlanner):
    force_target_n = 18.
    force_limit_n = 40.
    minimum_half_gap_m = .05
    closing_speed_mps = .06
    force_gain_m_per_ns = .004

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.side_gaps = np.full(2, self.half_gap)

    def _enter(self, phase, now, reason):
        if phase == "LIFT" and self.phase == "CLOSE":
            self.side_gaps[:] = self.half_gap
            return super()._enter("PRELOAD", now, "bilateral_touch_confirmed_preload_required")
        return super()._enter(phase, now, reason)

    def update(self, now, measurement):
        dt = min(.02, max(0., now-self.time))
        super().update(now, measurement)
        if self.phase not in ("PRELOAD", "LIFT", "HOLD", "LOWER"):
            return
        force = np.array([measurement["left_force"], measurement["right_force"]])
        if np.max(force) > self.force_limit_n:
            self.fail(now, "excessive_clamp_force")
            return
        # Only reference positions are adjusted. No torque, simulator state,
        # friction, mass, collision proxy, or direct object actuation changes.
        velocity = np.clip(self.force_gain_m_per_ns * (self.force_target_n-force),
                           -self.closing_speed_mps, self.closing_speed_mps)
        self.side_gaps = np.clip(self.side_gaps-velocity*dt, self.minimum_half_gap_m, .24)
        if self.phase == "PRELOAD":
            if self._continuous(bool(np.all(force >= self.force_target_n)), now, self.spec.grip_time):
                self.grasp_xyz = np.asarray(measurement["box_xyz"]).copy()
                if "wrist_xyz" in measurement:
                    self.grasp_relative = self.grasp_xyz - np.mean(measurement["wrist_xyz"], axis=0)
                self._enter("LIFT", now, "measured_both_sides_ge_18n_for_020s")
            elif now-self.entered >= 4.:
                self.fail(now, "load_bearing_preload_not_established")

    def targets(self, measurement):
        points, quats = super().targets(measurement)
        if self.phase in ("PRELOAD", "LIFT", "HOLD", "LOWER"):
            points[1, 1] = measurement["box_xyz"][1] + self.side_gaps[0]
            points[2, 1] = measurement["box_xyz"][1] - self.side_gaps[1]
        return points, quats
