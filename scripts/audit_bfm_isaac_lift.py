#!/usr/bin/env python3
"""Independently recompute native lift geometry/contacts and audit provenance.

An evidence-consistent failed episode is still a failed task. Contact centroids
below are normal-weighted *pair-average* locations, not exact centres of pressure.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.export_bfm_torchscript import bounded_unit, sha256
from scripts.bfm_lift_acceptance import sequence_checks


def audit(folder):
    report = json.loads((folder / "report.json").read_text())
    inputs = json.loads((folder / "inputs.json").read_text())["input_sha256"]
    provenance = {}
    for filename, digest in inputs.items():
        original = Path(filename)
        candidates = (original, folder / "source_at_launch" / original.name)
        matching = next((p for p in candidates if p.is_file() and sha256(p) == digest), None)
        if matching is None:
            raise ValueError(f"No unchanged original or launch snapshot: {filename}")
        provenance[filename] = str(matching)
    trajectory = folder / "trajectory.json"
    if sha256(trajectory) != report["trajectory_sha256"]:
        raise ValueError("Trajectory digest mismatch")
    rows = json.loads(trajectory.read_text())["rows"]
    if not report["execution_complete"] or report["result"] != "COMPLETE":
        raise ValueError("Runtime error, not a complete physical episode")
    if report["policy_steps"] != len(rows) or report["physics_steps"] != len(rows)*4:
        raise ValueError("Step count mismatch")
    if report.get("recording"):
        frames = json.loads((folder / "video_frames.json").read_text())["rgb_sha256"]
        if report["video_frames"] != len(rows) or len(frames) != len(rows):
            raise ValueError("Not every physical step has an encoded frame")
        if sha256(folder / "recording_raw.mp4") != report["raw_video_sha256"]:
            raise ValueError("Video digest mismatch")
    half = np.asarray(report["scene"]["box_size"])/2
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])*half
    top = report["scene"]["support_xyz"][2]+report["scene"]["support_size"][2]/2
    names = report["filter_bodies"]
    sides = {side: [names.index(f"{side}_{part}_link")
                    for part in ("elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]
             for side in ("left", "right")}
    longest = 0.
    since = None
    last_contact = {}
    for frame, row in enumerate(rows, 1):
        if row["frame"] != frame or abs(row["time_s"]-.02*frame) > 1e-9:
            raise ValueError("Noncontiguous 50 Hz physical trajectory")
        q = np.asarray(row["box_quat_wxyz"], dtype=float)
        if abs(np.linalg.norm(q)-1) > 1e-3:
            raise ValueError("Invalid measured box quaternion")
        q /= np.linalg.norm(q)
        # Quaternion vector rotation, separate implementation from box_geometry.
        def rotate(v):
            return v+2*np.cross(q[1:], np.cross(q[1:], v)+q[0]*v)
        xyz = np.asarray(row["box_xyz"])
        world_corners = rotate(corners)+xyz
        bottom = float(world_corners[:, 2].min())
        inside = bool(np.all(np.abs(world_corners[:, :2]-np.asarray(report["scene"]["support_xyz"][:2]))
                             <= np.asarray(report["scene"]["support_size"][:2])/2))
        if inside != row["inside_support_xy"]:
            raise ValueError("Full box footprint measurement differs")
        tilt = float(np.arccos(np.clip(rotate(np.array([0., 0., 1.]))[2], -1, 1)))
        for key, expected in (("box_bottom_z", bottom), ("box_tilt_rad", tilt)):
            if abs(row[key]-expected) > 2e-6:
                raise ValueError(f"Geometry mismatch at {frame}/{key}")
        normals = np.asarray(row["contact_normal_vectors_history_w"])
        if normals.shape != (4, 31, 3) or not np.isfinite(normals).all():
            raise ValueError("Invalid four-substep force history")
        force = np.linalg.norm(normals, axis=-1)
        for side in ("left", "right"):
            totals = force[:, sides[side]].sum(-1)
            for suffix, value in (("force", totals.min()), ("force_peak", totals.max())):
                if abs(row[f"{side}_{suffix}"]-value) > 2e-5:
                    raise ValueError(f"Contact mismatch at frame {frame}")
        if abs(row["support_force"]-force[:, -1].min()) > 2e-5 or abs(
                row["support_force_peak"]-force[:, -1].max()) > 2e-5:
            raise ValueError("Support minimum/maximum mismatch")
        airborne = (bottom-top >= .08 and row["support_force_peak"] < .2
                    and min(row["left_force"], row["right_force"]) >= .2
                    and tilt <= np.deg2rad(15))
        since = row["time_s"] if airborne and since is None else since if airborne else None
        if since is not None:
            longest = max(longest, row["time_s"]-since)
        contacts = row["contact_points_last_w"]
        for side, indices in sides.items():
            valid = [i for i in indices if contacts[i] is not None and force[0, i] > .2]
            if valid:
                point = np.average(np.asarray([contacts[i] for i in valid]), axis=0, weights=force[0, valid])
                last_contact[side] = dict(time_s=row["time_s"], pair_average_centroid_w=point.tolist(),
                                          centroid_offset_from_box_w=(point-xyz).tolist(),
                                          active_partners=[names[i] for i in valid])
    terminal = rows[-1]
    if bool(report["task_success"]) != (terminal["phase"] == "SUCCESS"):
        raise ValueError("Task result disagrees with terminal phase")
    if report["task_success"] and longest < 2.-1e-9:
        raise ValueError("Claimed success without continuous real airborne hold")
    checks = sequence_checks(rows, report["scene"])
    if report["task_success"] != all(checks.values()):
        raise ValueError(f"Physical success disagrees with independently measured sequence: {checks}")
    return dict(run=str(folder), evidence_consistent=True, task_success=report["task_success"],
                physical_sequence_checks=checks,
                failure_reason=report["failure_reason"], policy_steps=len(rows),
                physical_seconds=len(rows)*.02, longest_measured_8cm_airborne_s=longest,
                maximum_clearance_m=max(r["box_bottom_z"]-top for r in rows),
                final_tilt_deg=float(np.rad2deg(terminal["box_tilt_rad"])),
                final_box_xyz=terminal["box_xyz"], final_wrist_xyz=terminal["wrist_xyz"],
                final_contact_centroids=last_contact, events=report["events"],
                provenance_verified=provenance, formal_d1_d2_accepted=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bounded_unit("bfm-isaac-lift-audit-")
    dest = args.output.absolute()
    if (not dest.is_relative_to(ROOT / "local") or ".." in dest.parts
            or any(p.is_symlink() for p in (dest, *dest.parents))):
        raise ValueError("Audit output must be a new non-symlink local file")
    result = audit(args.run)
    with dest.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "provenance_verified"}))
