"""Deterministic uniform motion cohorts using an isolated CPU RNG."""

from __future__ import annotations

import math
import numbers
from typing import Mapping

import torch


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive non-bool integer")
    return int(value)


def validate_uniform_probabilities(probabilities, num_motions: int) -> None:
    """Reject non-positive, non-finite, or non-uniform sampling weights."""
    count = _positive_int("num_motions", num_motions)
    try:
        weights = torch.as_tensor(probabilities)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError("probabilities must be a finite positive uniform CPU vector") from error
    if weights.device.type != "cpu" or weights.ndim != 1 or weights.shape[0] != count:
        raise ValueError(f"probabilities must be a CPU vector of shape ({count},)")
    if not weights.dtype.is_floating_point:
        raise ValueError("probabilities must use a floating-point dtype")
    if not torch.isfinite(weights).all() or not torch.all(weights > 0):
        raise ValueError("probabilities must be finite and strictly positive")
    if not math.isfinite(float(weights.sum(dtype=torch.float64))):
        raise ValueError("probability sum must be finite")
    if not torch.equal(weights, weights[0].expand_as(weights)):
        raise ValueError("only exactly uniform probabilities are supported")


class UniformMotionCohortSampler:
    """Emit a continuous stream of shuffled, no-replacement motion epochs."""

    _STATE_KEYS = {
        "version", "num_motions", "num_envs", "seed", "generator_state",
        "permutation", "cursor", "seen", "cohorts", "epochs", "completed_epochs",
    }

    def __init__(self, num_motions: int, num_envs: int, seed: int):
        self.num_motions = _positive_int("num_motions", num_motions)
        self.num_envs = _positive_int("num_envs", num_envs)
        if isinstance(seed, bool) or not isinstance(seed, numbers.Integral) or not 0 <= int(seed) <= 2**63 - 1:
            raise ValueError("seed must be a non-bool integer in [0, 2**63 - 1]")
        self.seed = int(seed)
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(self.seed)
        self._permutation: torch.Tensor | None = None
        self._cursor = 0
        self._seen = torch.zeros(self.num_motions, dtype=torch.bool, device="cpu")
        self._cohorts = 0
        self._epochs = 0
        self._completed_epochs = 0

    def _start_epoch(self) -> None:
        self._permutation = torch.randperm(self.num_motions, generator=self._generator, device="cpu")
        self._cursor = 0
        self._epochs += 1

    def next_ids(self) -> torch.Tensor:
        """Return the next independent CPU-long cohort of ``num_envs`` IDs."""
        pieces = []
        needed = self.num_envs
        while needed:
            if self._permutation is None:
                self._start_epoch()
            available = self.num_motions - self._cursor
            take = min(needed, available)
            piece = self._permutation[self._cursor : self._cursor + take]
            pieces.append(piece)
            self._seen[piece] = True
            self._cursor += take
            needed -= take
            if self._cursor == self.num_motions:
                self._completed_epochs += 1
                self._permutation = None
                self._cursor = 0
        self._cohorts += 1
        return torch.cat(pieces).clone() if len(pieces) > 1 else pieces[0].clone()

    def snapshot(self) -> dict[str, int | float]:
        """Return JSON-compatible progress counters (not observed rollout usage)."""
        unique = int(self._seen.sum().item())
        return {
            "num_motions": self.num_motions,
            "num_envs": self.num_envs,
            "seed": self.seed,
            "cohorts": self._cohorts,
            "epochs": self._epochs,
            "completed_epochs": self._completed_epochs,
            "unique_ever": unique,
            "coverage": unique / self.num_motions,
            "current_epoch_cursor": self._cursor,
        }

    def state_dict(self) -> dict[str, object]:
        """Return an alias-free CPU state suitable for ``torch.save``."""
        permutation = self._permutation
        return {
            "version": 1,
            "num_motions": self.num_motions,
            "num_envs": self.num_envs,
            "seed": self.seed,
            "generator_state": self._generator.get_state().clone(),
            "permutation": (torch.empty(0, dtype=torch.long) if permutation is None else permutation.clone()),
            "cursor": self._cursor,
            "seen": self._seen.clone(),
            "cohorts": self._cohorts,
            "epochs": self._epochs,
            "completed_epochs": self._completed_epochs,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Atomically validate and restore a state returned by :meth:`state_dict`."""
        if not isinstance(state, Mapping) or set(state) != self._STATE_KEYS:
            raise ValueError("motion cohort state has missing or unexpected keys")

        def integer(name, minimum=0):
            value = state[name]
            if isinstance(value, bool) or not isinstance(value, numbers.Integral) or int(value) < minimum:
                raise ValueError(f"state {name} must be an integer >= {minimum}")
            return int(value)

        if integer("version", 1) != 1:
            raise ValueError("unsupported motion cohort state version")
        if integer("num_motions", 1) != self.num_motions or integer("num_envs", 1) != self.num_envs:
            raise ValueError("motion cohort state dimensions do not match this sampler")
        seed = integer("seed")
        if seed > 2**63 - 1:
            raise ValueError("state seed is out of range")
        cursor, cohorts = integer("cursor"), integer("cohorts")
        epochs, completed = integer("epochs"), integer("completed_epochs")
        generator_state, permutation, seen = state["generator_state"], state["permutation"], state["seen"]
        for name, tensor in (("generator_state", generator_state), ("permutation", permutation), ("seen", seen)):
            if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu":
                raise ValueError(f"state {name} must be a CPU tensor")
        if generator_state.dtype != torch.uint8 or generator_state.ndim != 1:
            raise ValueError("state generator_state must be a one-dimensional uint8 tensor")
        test_generator = torch.Generator(device="cpu")
        try:
            test_generator.set_state(generator_state.clone())
        except RuntimeError as error:
            raise ValueError("invalid CPU generator state") from error
        if permutation.dtype != torch.long or permutation.ndim != 1 or permutation.numel() not in (0, self.num_motions):
            raise ValueError("state permutation must be empty or one complete long permutation")
        active = permutation.numel() != 0
        if active:
            if not torch.equal(torch.sort(permutation).values, torch.arange(self.num_motions)):
                raise ValueError("state permutation is not a unique in-range permutation")
            if not 0 <= cursor < self.num_motions or epochs != completed + 1:
                raise ValueError("active epoch cursor/counters are inconsistent")
        elif cursor != 0 or epochs != completed:
            raise ValueError("inactive epoch cursor/counters are inconsistent")
        if seen.dtype != torch.bool or seen.shape != (self.num_motions,):
            raise ValueError("state seen must be a bool vector matching num_motions")
        emitted = cohorts * self.num_envs
        if emitted != completed * self.num_motions + (cursor if active else 0):
            raise ValueError("state cohort and epoch counters are inconsistent")
        if completed:
            expected_seen = torch.ones_like(seen)
        elif active:
            expected_seen = torch.zeros_like(seen)
            expected_seen[permutation[:cursor]] = True
        else:
            expected_seen = torch.zeros_like(seen)
        if not torch.equal(seen, expected_seen):
            raise ValueError("state seen mask is inconsistent with emitted IDs")

        # Commit only after every field and invariant has passed validation.
        self.seed = seed
        self._generator.set_state(generator_state.clone())
        self._permutation = permutation.clone() if active else None
        self._cursor = cursor
        self._seen = seen.clone()
        self._cohorts = cohorts
        self._epochs = epochs
        self._completed_epochs = completed
