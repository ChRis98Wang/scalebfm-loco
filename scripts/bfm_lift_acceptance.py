"""Independent physical sequence checks, separate from the online planner.

Inputs are audited 50 Hz measured rows. Passing these checks is a single-state
development success, never the formal multi-seed D1/D2 success-rate requirement.
"""
import math

import numpy as np


def sequence_checks(rows, scene):
    top = scene["support_xyz"][2]+scene["support_size"][2]/2
    goal = np.array([*scene["support_xyz"][:2], top+scene["box_size"][2]/2])
    weight = scene["box_mass"]*9.81
    phases = []
    for row in rows:
        if not phases or phases[-1] != row["phase"]:
            phases.append(row["phase"])
    expected = ["SETTLE", "APPROACH", "CLOSE", "PRELOAD", "LIFT", "HOLD", "LOWER", "RELEASE", "SUCCESS"]
    checks = dict(full_phase_sequence=phases == expected,
                  within_20s=bool(rows) and rows[-1]["time_s"] < scene["timeout"],
                  no_robot_guard=bool(rows) and all(r["root_z"] >= .3 and r["root_up"] >= math.cos(math.radians(75))
                                                   and r["other_robot_force"] <= .2 for r in rows))
    first = {phase: next((r["time_s"] for r in rows if r["phase"] == phase), None) for phase in expected}
    def interval(start, end):
        if start is None or end is None:
            return []
        return [r for r in rows if start-1e-9 <= r["time_s"] <= end+1e-9]
    def covers(samples, seconds):
        return bool(samples) and samples[-1]["time_s"]-samples[0]["time_s"] >= seconds-1e-9
    def both(row):
        return min(row["left_force"], row["right_force"]) >= .2
    preload = interval(None if first["LIFT"] is None else first["LIFT"]-.2, first["LIFT"])
    checks["measured_18n_preload_020s"] = covers(preload, .2) and all(
        min(r["left_force"], r["right_force"]) >= 18. for r in preload)
    hold = interval(first["HOLD"], first["LOWER"])
    checks["measured_8cm_hold_2s"] = covers(hold, 2.) and all(
        r["box_bottom_z"]-top >= .08 and r["support_force_peak"] < .2
        and r["box_tilt_rad"] <= math.radians(15) and both(r) for r in hold)
    loaded = interval(first["LIFT"], first["RELEASE"])
    checks["loaded_tilt_and_peak_force"] = bool(loaded) and all(
        r["box_tilt_rad"] <= math.radians(15)
        and max(r["left_force_peak"], r["right_force_peak"]) <= 40. for r in loaded)
    relative = [np.asarray(r["box_xyz"])-np.asarray(r["wrist_xyz"]).mean(axis=0) for r in loaded]
    checks["loaded_relative_slip_le_5cm"] = bool(relative) and max(
        np.linalg.norm(value-relative[0]) for value in relative) <= .05
    lost_since = None
    maximum_lost = 0.
    for row in loaded:
        if both(row):
            lost_since = None
        else:
            lost_since = row["time_s"] if lost_since is None else lost_since
            maximum_lost = max(maximum_lost, row["time_s"]-lost_since)
    checks["loaded_bilateral_coverage"] = bool(loaded) and sum(both(r) for r in loaded)/len(loaded) >= .9 and maximum_lost <= .1+1e-9
    supported = interval(None if first["RELEASE"] is None else first["RELEASE"]-.2, first["RELEASE"])
    checks["support_before_release"] = covers(supported, .2) and all(
        r["support_force"] >= .8*weight and abs(r["box_bottom_z"]-top) <= .01
        and r["box_speed"] <= .05 and r["box_angular_speed"] <= .10 for r in supported)
    released = interval(None if first["SUCCESS"] is None else first["SUCCESS"]-1., first["SUCCESS"])
    checks["released_at_xyz_goal_for_1s"] = covers(released, 1.) and all(
        max(r["left_force_peak"], r["right_force_peak"]) < .2
        and r["support_force"] >= .8*weight and r["box_speed"] <= .05
        and r["box_angular_speed"] <= .10 and r["inside_support_xy"]
        and np.linalg.norm(np.asarray(r["box_xyz"])-goal) <= .03 for r in released)
    return {key: bool(value) for key, value in checks.items()}
