#!/usr/bin/env python3
"""Quick regression check for randomize_rigid_body_com without launching Isaac Sim."""

from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg
from scaletrack.tasks.tracking.mdp import events


class FakePhysxView:
    """Minimal stand-in PhysX view for set/get com calls."""

    def __init__(self, coms, fallback_once: bool = False):
        self._coms = coms
        self._fallback_once = fallback_once
        self.calls = []

    def get_coms(self):
        return self._coms

    def set_coms(self, coms, env_ids):
        self.calls.append((type(coms), getattr(coms, "shape", None), type(env_ids).__name__))
        if self._fallback_once:
            self._fallback_once = False
            raise TypeError("mock fallback")
        self._coms = coms


class FakeArticulation:
    def __init__(self, coms, fallback_once: bool = False):
        self.root_physx_view = FakePhysxView(coms, fallback_once=fallback_once)
        self.num_bodies = len(coms[0])


class FakeScene:
    def __init__(self, artifact):
        self._artifact = artifact
        self.num_envs = len(artifact.root_physx_view.get_coms())

    def __getitem__(self, key):
        if key != "robot":
            raise KeyError(key)
        return self._artifact


class FakeEnv:
    def __init__(self, coms, fallback_once: bool = False):
        self.scene = FakeScene(FakeArticulation(coms, fallback_once=fallback_once))


def run_regression() -> None:
    com_range = {"x": (-0.1, 0.1), "y": (0.0, 0.0), "z": (0.0, 0.0)}
    cfg = SceneEntityCfg(name="robot", body_ids=[1, 3])

    # Case 1: full env_ids + torch input
    env = FakeEnv(torch.zeros(4, 5, 3, dtype=torch.float64))
    events.randomize_rigid_body_com(env, None, com_range, cfg)
    out = env.scene["robot"].root_physx_view._coms
    assert isinstance(out, torch.Tensor) and out.dtype == torch.float32, out.dtype

    # Case 2: partial env_ids + list input
    env = FakeEnv([[[0.0, 0.0, 0.0] for _ in range(5)] for _ in range(4)], fallback_once=False)
    events.randomize_rigid_body_com(env, torch.tensor([0, 2], dtype=torch.int32), com_range, cfg)
    out = env.scene["robot"].root_physx_view._coms
    assert isinstance(out, torch.Tensor) and out.shape == (4, 5, 3), out.shape

    # Case 3: setter fallback path (TypeError -> warp path)
    env = FakeEnv(torch.zeros(4, 5, 3), fallback_once=True)
    events.randomize_rigid_body_com(env, torch.tensor([0, 2], dtype=torch.int32), com_range, cfg)
    calls = env.scene["robot"].root_physx_view.calls
    assert len(calls) == 2, calls
    assert calls[0][0].__name__ == "Tensor", calls[0][0]

    print("randomize_rigid_body_com regression checks passed.")


if __name__ == "__main__":
    run_regression()
