"""Bounded sparse-target compensation for the native, contact-only lift task.

No simulator API is used here: outputs are pelvis/wrist reference poses, not
physical robot/object states. Success is still judged by the original planner.
"""
from dataclasses import asdict, dataclass
import math

import numpy as np

from lift_demo import rotation_wxyz
from lift_demo_preload_v3 import LoadBearingLiftPlanner


@dataclass(frozen=True)
class BalanceConfig:
    wrist_offset_x_m: float = -.03
    approach_bias_gain_per_s: float = .6
    bias_speed_mps: float = .035
    maximum_x_bias_m: float = .08
    maximum_z_bias_m: float = .04
    pitch_feedback_gain: float = 1.5
    maximum_pitch_correction_rad: float = .25
    roll_height_gain_m_per_rad: float = .08
    maximum_roll_height_correction_m: float = .025
    clearance_feedback_gain_per_s: float = 0.
    maximum_clearance_bias_m: float = .08
    clearance_bias_speed_mps: float = .06
    landing_force_target_n: float = 24.
    landing_xy_return_fraction: float = 0.
    landing_force_gain_m_per_ns: float = .004

    def validate(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(f"Invalid balance parameter: {name}")
            if name != "wrist_offset_x_m" and value < 0:
                raise ValueError(f"Negative balance parameter: {name}")
        if not -.09 <= self.wrist_offset_x_m <= .03:
            raise ValueError("Wrist X offset outside declared development workspace")
        if not 1. <= self.landing_force_target_n <= 24. or not 0. <= self.landing_xy_return_fraction <= 1.:
            raise ValueError("Landing force/XY reference outside declared bounds")
        if not 0. < self.landing_force_gain_m_per_ns <= .02:
            raise ValueError("Landing force feedback gain outside declared bounds")
        if (self.maximum_x_bias_m > .10 or self.maximum_z_bias_m > .05
                or self.bias_speed_mps > .06 or self.maximum_pitch_correction_rad > .35
                or self.maximum_roll_height_correction_m > .04
                or self.maximum_clearance_bias_m > .10 or self.clearance_bias_speed_mps > .06
                or self.clearance_feedback_gain_per_s > 2.):
            raise ValueError("Balance compensation exceeds bounded workspace/rate")


class BalancedLiftPlanner(LoadBearingLiftPlanner):
    """Centre the grip, freeze tangential calibration and resist object tipping."""

    def __init__(self, *args, balance=None, **kwargs):
        self.balance = balance or BalanceConfig()
        self.balance.validate()
        super().__init__(*args, **kwargs)
        self.tangential_bias = np.zeros((2, 2))  # x/z, learned only before contact
        self.anchor_xy = None
        self.last_issued_targets = None
        self.clearance_bias = 0.
        self.landing_unload = False

    def update(self, now, measurement):
        old_phase, old_time = self.phase, self.time
        dt = min(.02, max(0., now-old_time))
        # Validate added telemetry before it can influence a target.
        if old_phase not in ("SUCCESS", "FAILED"):
            try:
                rotation_wxyz(measurement["box_quat_wxyz"])
                wrists = np.asarray(measurement["wrist_xyz"], dtype=float)
                if wrists.shape != (2, 3) or not np.isfinite(wrists).all():
                    raise ValueError("Invalid wrists")
            except (KeyError, TypeError, ValueError):
                self.fail(now, "invalid_balance_telemetry")
                return
        if old_phase == "LOWER" and self.balance.landing_force_target_n < 24.:
            touching = (measurement["support_force"] >= .5*self.spec.box_mass*9.81
                        and abs(measurement["box_bottom_z"]-self.spec.support_top) <= .005
                        and measurement["box_tilt_rad"] <= math.radians(10)
                        and measurement["box_speed"] <= .10)
            self.landing_unload = self.landing_unload or touching
        # Change the reference force setpoint only after measured landing contact;
        # never lower preload/airborne grip or any physical acceptance threshold.
        self.force_target_n = self.balance.landing_force_target_n if self.landing_unload else 24.
        self.force_gain_m_per_ns = self.balance.landing_force_gain_m_per_ns if self.landing_unload else .004
        gate_measurement = measurement
        if old_phase in ("LIFT", "HOLD"):
            gate_measurement = measurement | {
                "support_force": measurement.get("support_force_peak", measurement["support_force"])}
        elif old_phase == "RELEASE":
            gate_measurement = measurement | {
                "left_force": measurement.get("left_force_peak", measurement["left_force"]),
                "right_force": measurement.get("right_force_peak", measurement["right_force"])}
        super().update(now, gate_measurement)
        if self.phase in ("FAILED", "SUCCESS"):
            return
        # Never integrate against contact constraints or wind up during approach
        # slewing. Freeze this bias for closing, preload and the full loaded path.
        if old_phase == self.phase == "APPROACH" and now-self.entered >= 1.:
            desired = np.array([measurement["box_xyz"][0]+self.balance.wrist_offset_x_m,
                                self.spec.goal_xyz[2]])
            error = desired - wrists[:, [0, 2]]
            velocity = np.clip(self.balance.approach_bias_gain_per_s*error,
                               -self.balance.bias_speed_mps, self.balance.bias_speed_mps)
            limit = np.array([self.balance.maximum_x_bias_m, self.balance.maximum_z_bias_m])
            self.tangential_bias = np.clip(self.tangential_bias+dt*velocity, -limit, limit)
        if self.phase == "PRELOAD" and old_phase != "PRELOAD":
            self.anchor_xy = np.asarray(measurement["box_xyz"][:2], dtype=float).copy()
        # The original preload force gate uses each side's minimum over four
        # physics substeps. The new instantaneous cap is deliberately stricter.
        if self.phase in ("PRELOAD", "LIFT", "HOLD", "LOWER"):
            peaks = np.asarray([measurement.get("left_force_peak", measurement["left_force"]),
                                measurement.get("right_force_peak", measurement["right_force"])])
            if not np.isfinite(peaks).all() or np.max(peaks) > self.force_limit_n:
                self.fail(now, "instantaneous_clamp_force_exceeded")
                return
        if self.phase in ("LIFT", "HOLD"):
            progress = min(1., (now-self.entered)/2.) if self.phase == "LIFT" else 1.
            requested_clearance = self.spec.lift_height*progress
            measured_clearance = measurement["box_bottom_z"]-self.spec.support_top
            velocity = float(np.clip(self.balance.clearance_feedback_gain_per_s*(
                requested_clearance-measured_clearance), -self.balance.clearance_bias_speed_mps,
                self.balance.clearance_bias_speed_mps))
            self.clearance_bias = float(np.clip(self.clearance_bias+dt*velocity,
                                               0., self.balance.maximum_clearance_bias_m))

    def targets(self, measurement):
        points, quats = super().targets(measurement)
        if self.phase not in ("SETTLE", "FAILED", "SUCCESS"):
            c = self.balance
            anchor = np.asarray(measurement["box_xyz"][:2]) if self.anchor_xy is None else self.anchor_xy
            if self.anchor_xy is not None and self.phase in ("LOWER", "RELEASE"):
                progress = min(1., (self.time-self.entered)/2.) if self.phase == "LOWER" else 1.
                blend = progress*c.landing_xy_return_fraction
                anchor = (1.-blend)*anchor + blend*np.asarray(self.spec.goal_xyz[:2])
            points[1:, 0] = anchor[0] + c.wrist_offset_x_m + self.tangential_bias[:, 0]
            points[1:, 2] += self.tangential_bias[:, 1]
            if self.phase in ("PRELOAD", "LIFT", "HOLD", "LOWER"):
                points[1, 1] = anchor[1] + self.side_gaps[0]
                points[2, 1] = anchor[1] - self.side_gaps[1]
                rotation = rotation_wxyz(measurement["box_quat_wxyz"])
                roll = math.atan2(rotation[2, 1], rotation[2, 2])
                pitch = math.asin(float(np.clip(-rotation[2, 0], -1., 1.)))
                correction = float(np.clip(-c.pitch_feedback_gain*pitch,
                                           -c.maximum_pitch_correction_rad, c.maximum_pitch_correction_rad))
                quats[1:] = [math.cos(correction/2), 0., math.sin(correction/2), 0.]
                dz = float(np.clip(-c.roll_height_gain_m_per_rad*roll,
                                   -c.maximum_roll_height_correction_m, c.maximum_roll_height_correction_m))
                points[1, 2] += dz
                points[2, 2] -= dz
                if self.phase in ("LIFT", "HOLD"):
                    points[1:, 2] += self.clearance_bias
                elif self.phase == "LOWER":
                    points[1:, 2] += self.clearance_bias*max(0., 1.-(self.time-self.entered)/2.)
            elif self.phase == "RELEASE" and self.anchor_xy is not None:
                points[1, 1], points[2, 1] = anchor[1]+.27, anchor[1]-.27
        self.last_issued_targets = points.copy()
        return points, quats

    def diagnostics(self):
        return dict(tangential_bias_xz_m=self.tangential_bias.tolist(),
                    anchor_xy=None if self.anchor_xy is None else self.anchor_xy.tolist(),
                    side_gaps_m=self.side_gaps.tolist(), clearance_bias_m=self.clearance_bias,
                    landing_unload=self.landing_unload, force_control_setpoint_n=self.force_target_n)
