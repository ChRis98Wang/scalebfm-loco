"""Explicit quaternion conversions at ScaleTrack's file/runtime boundaries."""

from __future__ import annotations

from typing import Literal, TypeVar


QuaternionArray = TypeVar("QuaternionArray")
QuaternionOrder = Literal["wxyz", "xyzw"]


def detect_runtime_quaternion_order() -> QuaternionOrder:
    """Detect IsaacLab's quaternion convention from an identity rotation.

    IsaacLab releases used by ScaleBFM exist with both ``wxyz`` and ``xyzw``
    math APIs.  Detecting the behavior avoids coupling the data path to a
    directory name or version string.
    """
    import torch
    from isaaclab.utils.math import quat_from_euler_xyz

    zeros = torch.zeros(1)
    identity = quat_from_euler_xyz(zeros, zeros, zeros).detach().cpu()
    wxyz_identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=identity.dtype)
    xyzw_identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=identity.dtype)

    if torch.allclose(identity, wxyz_identity):
        return "wxyz"
    if torch.allclose(identity, xyzw_identity):
        return "xyzw"
    raise RuntimeError(
        "Unable to detect IsaacLab quaternion order from identity rotation: "
        f"{identity.tolist()}"
    )


def _convert(
    quaternions: QuaternionArray,
    source_order: QuaternionOrder,
    target_order: QuaternionOrder,
) -> QuaternionArray:
    if source_order == target_order:
        return quaternions
    if source_order == "wxyz" and target_order == "xyzw":
        return quaternions[..., [1, 2, 3, 0]]
    if source_order == "xyzw" and target_order == "wxyz":
        return quaternions[..., [3, 0, 1, 2]]
    raise ValueError(f"Unsupported quaternion conversion: {source_order} -> {target_order}")


def source_xyzw_to_runtime(
    quaternions: QuaternionArray, runtime_order: QuaternionOrder
) -> QuaternionArray:
    """Convert ScaleRetarget's xyzw root rotations to the active runtime."""
    return _convert(quaternions, "xyzw", runtime_order)


def packed_wxyz_to_runtime(
    quaternions: QuaternionArray, runtime_order: QuaternionOrder
) -> QuaternionArray:
    """Convert ScaleTrack's stable on-disk wxyz representation to the runtime."""
    return _convert(quaternions, "wxyz", runtime_order)


def runtime_to_packed_wxyz(
    quaternions: QuaternionArray, runtime_order: QuaternionOrder
) -> QuaternionArray:
    """Convert runtime body rotations to ScaleTrack's stable on-disk wxyz form."""
    return _convert(quaternions, runtime_order, "wxyz")
