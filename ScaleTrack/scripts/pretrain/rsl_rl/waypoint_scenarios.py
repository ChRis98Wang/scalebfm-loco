"""Fixed waypoint acceptance and diagnostic scenarios without runtime dependencies."""

from __future__ import annotations

from dataclasses import dataclass
import math
import numbers
from types import MappingProxyType

import numpy as np


@dataclass(frozen=True)
class _WaypointScenario:
    name: str
    forward_m: float
    height_m: float
    claim_scope: str

    def target(self, initial_xyz, yaw) -> np.ndarray:
        """Return this fixed scenario relative to one finite initial pelvis pose."""
        try:
            initial = np.asarray(initial_xyz, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("initial_xyz must be a finite XYZ vector") from error
        if initial.shape != (3,) or not np.all(np.isfinite(initial)):
            raise ValueError("initial_xyz must be a finite XYZ vector")
        if isinstance(yaw, (bool, np.bool_)) or not isinstance(yaw, numbers.Real):
            raise ValueError("yaw must be a finite non-bool angle")
        heading = float(yaw)
        if not math.isfinite(heading):
            raise ValueError("yaw must be a finite non-bool angle")
        result = initial.copy()
        result[:2] += self.forward_m * np.array([math.cos(heading), math.sin(heading)])
        result[2] += self.height_m
        return result


_SCENARIOS = MappingProxyType(
    {
        "combined": _WaypointScenario(
            "combined",
            0.20,
            -0.06,
            "combined XYZ waypoint acceptance; the only scenario supporting the combined acceptance claim",
        ),
        "horizontal_only": _WaypointScenario(
            "horizontal_only",
            0.20,
            0.0,
            "diagnostic isolation of horizontal response only; not combined waypoint acceptance",
        ),
        "height_only": _WaypointScenario(
            "height_only",
            0.0,
            -0.06,
            "diagnostic isolation of height response only; not combined waypoint acceptance",
        ),
    }
)


def waypoint_scenario(name: str) -> _WaypointScenario:
    """Return one of the three fixed scenarios; arbitrary target tuning is unsupported."""
    if not isinstance(name, str) or name not in _SCENARIOS:
        raise ValueError(f"waypoint scenario must be one of {tuple(_SCENARIOS)}, got {name!r}")
    return _SCENARIOS[name]
