"""Versioned 50 Hz paired-window loader for the unchanged IsaacLab packer.

The historical entry point computes duration as (N-1)*(1/fps), which can
round upward and produce an extra endpoint. This opt-in entry point uses exactly
N-1 samples for an inclusive N-sample 50 Hz source. It recomputes velocities and
all simulator link states; it never trims an already-produced archive. The
frozen historical entry point remains byte-for-byte unchanged.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def paired_clock(frames, fps, output_dt):
    import torch

    if type(frames) is not int or not 3 <= frames <= 251:
        raise ValueError("Paired pilot sources require 3..251 inclusive samples")
    if fps != 50. or output_dt != 1. / 50.:
        raise ValueError("Paired native and output clocks must both be exactly 50 Hz")
    duration = (frames - 1) / fps
    # Match the frozen proof's float32 sample values, but constrain the logical
    # half-open interval by integer sample index, not binary64 ceil(end/step).
    times = torch.arange(0, duration, output_dt, dtype=torch.float32)[:frames - 1]
    if len(times) != frames - 1:
        raise ValueError("Runtime could not construct every required paired sample")
    return duration, times


def load_paired_motion(input_dir, motion_file, output_dt, *, runtime):
    import joblib
    import numpy as np
    import torch

    motion = joblib.load(motion_file)  # Explicitly trusted, hash-bound local pilot inputs.
    root = np.asarray(motion["root_pos"])
    quat = np.asarray(motion["root_rot"])
    joints = np.asarray(motion["dof_pos"])
    count = len(root)
    if root.shape != (count, 3) or quat.shape != (count, 4) or joints.shape != (count, 29):
        raise ValueError("Expected paired G1 29DoF native arrays")
    if not all(np.isfinite(value).all() for value in (root, quat, joints)):
        raise ValueError("Nonfinite paired source")
    if not np.allclose(np.linalg.norm(quat.astype(np.float64), axis=1), 1., atol=1e-5, rtol=0):
        raise ValueError("Paired root quaternion is not unit length")
    duration, times = paired_clock(count, float(motion["fps"]), output_dt)
    base_pos = torch.from_numpy(root).float()
    base_rot = runtime.source_xyzw_to_runtime(torch.from_numpy(quat).float(), runtime.RUNTIME_QUATERNION_ORDER)
    dof_pos = torch.from_numpy(joints).float()
    phase = times / duration
    lower = (phase * (count - 1)).floor().long()
    upper = torch.minimum(lower + 1, torch.tensor(count - 1))
    blend = phase * (count - 1) - lower
    position = base_pos[lower] * (1 - blend[:, None]) + base_pos[upper] * blend[:, None]
    dofs = dof_pos[lower] * (1 - blend[:, None]) + dof_pos[upper] * blend[:, None]
    rotations = torch.stack([runtime.quat_slerp(base_rot[a], base_rot[b], weight)
                             for a, b, weight in zip(lower, upper, blend)])
    linear = torch.gradient(position, spacing=output_dt, dim=0)[0]
    velocity = torch.gradient(dofs, spacing=output_dt, dim=0)[0]
    wxyz = runtime.runtime_to_packed_wxyz(rotations, runtime.RUNTIME_QUATERNION_ORDER)
    if len(wxyz) < 3:
        relative = runtime.quat_mul(wxyz[1:], runtime.quat_conjugate(wxyz[:-1]))
        angular = (runtime.axis_angle_from_quat(relative) / output_dt).repeat(2, 1)
    else:
        relative = runtime.quat_mul(wxyz[2:], runtime.quat_conjugate(wxyz[:-2]))
        omega = runtime.axis_angle_from_quat(relative) / (2. * output_dt)
        angular = torch.cat([omega[:1], omega, omega[-1:]], dim=0)
    if any(not torch.isfinite(value).all() for value in (position, dofs, rotations, linear, velocity, angular)):
        raise ValueError("Nonfinite paired interpolated position/velocity")
    print(f"[paired clock] {Path(motion_file).stem}: {count} inclusive -> {len(times)} half-open samples")
    return {"base_pos": position, "base_rot": rotations, "base_lin_vel": linear,
            "base_ang_vel": angular, "dof_pos": dofs, "dof_vel": velocity, "output_frames": len(times),
            "file_name": runtime.relative_archive_path(input_dir, motion_file).with_suffix("").as_posix(),
            "source_path": str(Path(motion_file).resolve())}


def main():
    # Import intentionally delegates AppLauncher/CLI/robot/FK to the historical
    # entry point without executing its __main__. Only the loader is replaced.
    import package_motions as runtime

    try:
        output = Path(runtime.args_cli.output_dir)
        if output.exists() or output.is_symlink():
            raise FileExistsError("Paired packaging requires a new output directory")
        wrapper_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        parent_fingerprint = runtime.PIPELINE_FINGERPRINT
        identity = {"schema": "bfm.paired_integer_half_open_clock/1", "parent_pipeline": parent_fingerprint,
                    "entrypoint_sha256": wrapper_hash, "rule": "50 Hz inclusive N native samples -> N-1 packed samples",
                    "velocity_rule": "recompute all derivatives before simulator link-state generation"}
        fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        runtime.PIPELINE_FINGERPRINT = fingerprint
        runtime.load_single_motion = lambda *args: load_paired_motion(*args, runtime=runtime)
        runtime.main()
        if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != wrapper_hash:
            raise ValueError("Paired packing entry point changed during execution")
        with (output / "paired_clock_receipt.json").open("x") as stream:
            json.dump({**identity, "pipeline_fingerprint": fingerprint, "result": "COMPLETE",
                       "original_packer_modified": False, "posthoc_archive_cropping": False}, stream, indent=2)
    finally:
        runtime.simulation_app.close()


if __name__ == "__main__":
    main()
