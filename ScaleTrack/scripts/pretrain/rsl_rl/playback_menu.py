"""Simulator-independent motion selection and playback reset helpers."""

from __future__ import annotations

import math
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml


def gui_is_available(sim, app_config=None):
    """Use Kit's launch mode; some Lab builds leave the simulation GUI cache unset."""
    if app_config is not None and "headless" in app_config:
        return not app_config["headless"]
    # Legacy fallback: older releases used a method instead of a property.
    value = sim.has_gui
    return bool(value() if callable(value) else value)


class MotionSelection:
    """Keep browsing separate from the motion applied at a simulation boundary."""

    def __init__(self, names, durations, current_id=0):
        self.names = tuple(names)
        self.durations = tuple(float(value) for value in durations)
        if not self.names or len(self.names) != len(self.durations):
            raise ValueError("Motion names and durations must be nonempty and have equal lengths.")
        if any(not math.isfinite(value) or value <= 0 for value in self.durations):
            raise ValueError("Motion durations must be finite and positive.")
        self._check_id(current_id)
        self.current_id = current_id
        self.filtered_ids = tuple(range(len(self.names)))
        self._pending_id = None

    def _check_id(self, motion_id):
        if not 0 <= motion_id < len(self.names):
            raise IndexError(f"Motion id out of range: {motion_id}")

    def filter(self, query):
        tokens = query.casefold().split()
        self.filtered_ids = tuple(
            index for index, name in enumerate(self.names)
            if all(token in name.casefold() for token in tokens)
        )
        return self.filtered_ids

    def request_filtered(self, index):
        if not 0 <= index < len(self.filtered_ids):
            raise IndexError(f"Filtered selection out of range: {index}")
        self._pending_id = self.filtered_ids[index]

    def request_next(self, direction):
        if not self.filtered_ids:
            return
        current = self.current_id if self._pending_id is None else self._pending_id
        if current in self.filtered_ids:
            position = (self.filtered_ids.index(current) + direction) % len(self.filtered_ids)
        else:
            position = 0 if direction >= 0 else len(self.filtered_ids) - 1
        self.request_filtered(position)

    def request_restart(self):
        self._pending_id = self.current_id

    def consume_request(self):
        motion_id, self._pending_id = self._pending_id, None
        return motion_id

    def mark_playing(self, motion_id):
        self._check_id(motion_id)
        self.current_id = motion_id


@contextmanager
def combined_motion_index(index_paths):
    """Load explicit indexes into a temporary, absolute-path index, never editing sources."""
    motions = {}
    for index_path in index_paths:
        index_path = Path(index_path).resolve()
        with index_path.open(encoding="utf-8") as stream:
            entries = yaml.safe_load(stream)
        if not isinstance(entries, dict) or not entries:
            raise ValueError(f"Expected a nonempty motion mapping: {index_path}")
        for name, value in entries.items():
            if not isinstance(name, str) or not name or not isinstance(value, str):
                raise ValueError(f"Motion names and paths must be strings: {index_path}")
            path = Path(value)
            path = (path if path.is_absolute() else index_path.parent / path).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Motion {name}: {path}")
            if name in motions and motions[name] != str(path):
                raise ValueError(f"Conflicting motion name: {name}")
            motions[name] = str(path)
    if not motions:
        raise ValueError("At least one nonempty motion index is required.")
    with TemporaryDirectory(prefix="bfm-play-motions-") as directory:
        combined = Path(directory) / "motions.yaml"
        with combined.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(motions, stream, sort_keys=False, allow_unicode=True)
        yield str(combined)


def switch_motion(env, command, motion_id):
    """Reset all playback environments onto a validated clip before recomputing observations."""
    if not 0 <= motion_id < command.num_motion:
        raise IndexError(f"Motion id out of range: {motion_id}")
    command.motion_ids.fill_(motion_id)
    command.time_steps.zero_()
    command._future_manual_cache.clear()
    obs, _ = env.reset()
    return obs


def switch_mode(env, command, mode_name, body_names, motion_id):
    """Apply a sparse reference mask, then reset motion/history at a safe step boundary."""
    import torch

    if not body_names or len(set(body_names)) != len(body_names) or not set(body_names).issubset(command.cfg.body_names):
        raise ValueError("Every mode body must be known, nonempty and unique")
    if not 0 <= motion_id < command.num_motion:
        raise IndexError(f"Motion id out of range: {motion_id}")
    mask = torch.tensor([[float(name in body_names) for name in command.cfg.body_names]],
                        device=command.device, dtype=command._mode_table.dtype)
    command.cfg.mode_candidates = {mode_name: list(body_names)}
    command._mode_table = mask
    return switch_motion(env, command, motion_id)
