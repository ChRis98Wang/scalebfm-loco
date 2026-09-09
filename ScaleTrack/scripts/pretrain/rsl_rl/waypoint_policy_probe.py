"""Non-actuating counterfactual inspection of the exact fixed-mode inference path."""

from __future__ import annotations

import numpy as np
import torch

from waypoint_diagnostics import decode_pelvis_task, stationary_pelvis_counterfactual, summarize_actions


def expected_pelvis_features(reference_xyz, reference_wxyz, actual_xyz, actual_wxyz):
    """Independent NumPy world-to-actual-pelvis transform, including tangent/normal rotation."""
    positions = np.asarray(reference_xyz, dtype=np.float64)
    quaternions = np.asarray(reference_wxyz, dtype=np.float64)
    actual = np.asarray(actual_xyz, dtype=np.float64)
    orientation = np.asarray(actual_wxyz, dtype=np.float64)
    if (positions.ndim != 2 or positions.shape[1:] != (3,)
            or quaternions.shape != (len(positions), 4) or actual.shape != (3,) or orientation.shape != (4,)):
        raise ValueError("expected finite reference [frames,3/4] and actual [3/4] arrays")
    if any(not np.all(np.isfinite(value)) for value in (positions, quaternions, actual, orientation)):
        raise ValueError("pose arrays must be finite")

    def rotation(q):
        norm = np.linalg.norm(q)
        if abs(norm - 1.) > 1e-3:
            raise ValueError("expected unit wxyz quaternion")
        w, x, y, z = q / norm
        return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                         [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                         [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])

    inverse = rotation(orientation).T
    relative_positions = (positions - actual) @ inverse.T
    relative_rotations = np.stack([inverse @ rotation(q) for q in quaternions])
    rotations = np.concatenate([relative_rotations[:, :, 0], relative_rotations[:, :, 2]], axis=-1)
    return {"target_body_pos": relative_positions, "target_body_pos_rel": relative_positions.copy(),
            "target_body_rot": rotations, "target_body_rot_rel": rotations.copy()}


def probe_policy(policy, model, obs):
    """Return the original moving-reference action plus a side-effect-checked diagnostic.

    All policy inputs are copies. Only the first, unmodified-reference action is
    returned to the normal actuation path; counterfactual actions are never applied.
    Fail closed for stochastic, training-mode, recurrent, or state-mutating models.
    """
    if any(module.training for module in model.modules()) or getattr(model, "is_recurrent", False):
        raise ValueError("diagnostics require an eval, non-recurrent policy")
    flat = stationary_pelvis_counterfactual(obs)
    moving = {key: value.clone() for key, value in obs.items()}
    repeat = {key: value.clone() for key, value in obs.items()}
    original = {key: value.clone() for key, value in obs.items()}
    tensors = {**dict(model.named_parameters()), **dict(model.named_buffers())}
    state = {key: value.clone() for key, value in tensors.items()}
    devices = sorted({value.device.index for value in obs.values() if value.is_cuda})
    captured = []

    def capture(_module, inputs):
        captured.append(inputs[0].clone())

    handle = model.actor_task_embedder.register_forward_pre_hook(capture)
    try:
        with torch.inference_mode(), torch.random.fork_rng(devices=devices):
            cpu_rng = torch.random.get_rng_state()
            cuda_rng = {device: torch.cuda.get_rng_state(device) for device in devices}
            moving_action = policy(moving)
            flat_action = policy(flat)
            repeat_action = policy(repeat)
            if not torch.equal(cpu_rng, torch.random.get_rng_state()) or any(
                not torch.equal(value, torch.cuda.get_rng_state(device)) for device, value in cuda_rng.items()
            ):
                raise AssertionError("diagnostic inference consumed RNG")
    finally:
        handle.remove()
    if any(not torch.equal(original[key], value) for key, value in obs.items()):
        raise AssertionError("diagnostics mutated the original observations")
    if any(not torch.equal(state[key], value) for key, value in tensors.items()):
        raise AssertionError("diagnostic inference mutated model parameters/buffers")
    if not torch.equal(moving_action, repeat_action):
        raise AssertionError("identical moving observations did not produce identical actions")
    if len(captured) != 3:
        raise AssertionError("expected exactly one task embedder call per inference")
    mapping = obs["mode_mapping"][:, None, :]
    mode = obs["mode"][:, None, :].expand(-1, obs["policy_task"].shape[1], -1)
    expected = torch.cat([obs["policy_task"] * mapping, mode], dim=-1)
    if not torch.equal(captured[0], expected) or not torch.equal(captured[0], captured[2]):
        raise AssertionError("actual actor task differs from the expected fixed Pelvis-1 mask")
    return moving_action, {
        "moving_action": moving_action.clone(), "flat_action": flat_action.clone(),
        "action_difference": summarize_actions(moving_action, flat_action),
        "moving_actor_task": captured[0], "flat_actor_task": captured[1],
        "moving_pelvis": decode_pelvis_task(captured[0][..., :253]),
        "flat_pelvis": decode_pelvis_task(captured[1][..., :253]),
        "policy_history": original["policy"], "action_history": original["action"],
        "mask_exact": True, "repeat_action_exact": True, "model_state_unchanged": True,
        "input_unchanged": True, "rng_unchanged": True,
        "real_action_evaluations": 1, "diagnostic_only_evaluations": 2,
    }
