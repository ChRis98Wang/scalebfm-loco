"""Canonical sparse online modes, matching the existing G1 training masks."""

from types import MappingProxyType

from scaletrack.utils.live_reference import VR3_BODY_NAMES

ONLINE_MODES = MappingProxyType({
    "Pelvis-1": (VR3_BODY_NAMES[0],),
    "UMI-2": VR3_BODY_NAMES[1:],
    "VR-3": VR3_BODY_NAMES,
})
ONLINE_MODE_NAMES = tuple(ONLINE_MODES)


def validate_online_mode(name: str) -> str:
    if not isinstance(name, str) or name not in ONLINE_MODES:
        raise ValueError(f"online mode must be one of {ONLINE_MODE_NAMES}, got {name!r}")
    return name


def online_target_rows(name: str) -> tuple[int, ...]:
    return tuple(VR3_BODY_NAMES.index(body) for body in ONLINE_MODES[validate_online_mode(name)])


def online_body_indices(name: str) -> tuple[int, ...]:
    return tuple((0, 10, 13)[row] for row in online_target_rows(name))
