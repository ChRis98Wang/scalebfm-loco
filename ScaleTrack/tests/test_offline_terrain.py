"""Catch network-dependent ground creation and changes to trained ground physics.

Run with the existing IsaacLab Python; USD authoring needs no Kit window or GPU.
Only the simulator singleton (for its device) and external asset I/O are replaced.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

import isaaclab.sim as sim_utils
from isaaclab.sim.utils import prims
from scaletrack.tasks.tracking.tracking_env_cfg import MySceneCfg


@pytest.fixture
def ground(monkeypatch):
    def local_files_only(path):
        assert "://" not in str(path), f"Online asset access forbidden: {path}"
        return int(Path(path).is_file())

    monkeypatch.setattr(prims, "check_file_path", local_files_only)
    monkeypatch.setattr(
        sim_utils.SimulationContext, "instance", staticmethod(lambda: SimpleNamespace(device="cpu"))
    )
    stage = Usd.Stage.CreateInMemory()
    cfg = MySceneCfg(num_envs=4, env_spacing=2.5).terrain.replace(num_envs=4, env_spacing=2.5)
    with sim_utils.use_stage(stage):
        yield stage, cfg


def test_configured_ground_spawns_without_external_assets(ground):
    stage, cfg = ground
    terrain = cfg.class_type(cfg)

    assert stage.GetPrimAtPath("/World/ground/terrain").IsValid()
    assert terrain.terrain_prim_paths == ["/World/ground/terrain"]
    assert stage.GetRootLayer().GetExternalReferences() == ()
    for prim in stage.Traverse():
        assert not prim.HasAuthoredReferences()
        assert not prim.HasAuthoredPayloads()
        for attr in prim.GetAttributes():
            assert attr.GetTypeName() not in (Sdf.ValueTypeNames.Asset, Sdf.ValueTypeNames.AssetArray)


def test_ground_keeps_static_plane_physics_and_grid_origins(ground):
    stage, cfg = ground
    terrain = cfg.class_type(cfg)
    colliders = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.CollisionAPI)]

    assert len(colliders) == 1
    collider = colliders[0]
    assert collider.GetTypeName() == "Plane"
    assert UsdGeom.Plane(collider).GetAxisAttr().Get() == "Z"
    assert not collider.HasAPI(UsdPhysics.RigidBodyAPI)
    assert UsdGeom.Xformable(collider).ComputeLocalToWorldTransform(0).ExtractTranslation()[2] == 0.0
    material, _ = UsdShade.MaterialBindingAPI(collider).ComputeBoundMaterial(materialPurpose="physics")
    physics = UsdPhysics.MaterialAPI(material.GetPrim())
    assert physics.GetStaticFrictionAttr().Get() == 1.0
    assert physics.GetDynamicFrictionAttr().Get() == 1.0
    assert physics.GetRestitutionAttr().Get() == 0.0
    assert material.GetPrim().GetAttribute("physxMaterial:frictionCombineMode").Get() == "multiply"
    assert material.GetPrim().GetAttribute("physxMaterial:restitutionCombineMode").Get() == "multiply"
    torch.testing.assert_close(
        terrain.env_origins,
        torch.tensor([[1.25, -1.25, 0.0], [1.25, 1.25, 0.0], [-1.25, -1.25, 0.0], [-1.25, 1.25, 0.0]]),
    )


def test_ground_material_overrides_are_not_hardcoded(ground):
    stage, cfg = ground
    cfg.physics_material.static_friction = 0.75
    cfg.physics_material.dynamic_friction = 0.5
    cfg.class_type(cfg)
    collider = next(prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.CollisionAPI))
    material, _ = UsdShade.MaterialBindingAPI(collider).ComputeBoundMaterial(materialPurpose="physics")

    physics = UsdPhysics.MaterialAPI(material.GetPrim())
    assert physics.GetStaticFrictionAttr().Get() == 0.75
    assert physics.GetDynamicFrictionAttr().Get() == 0.5


def test_visual_ground_vertices_retain_millimetre_precision(ground):
    """Million-metre float32 vertices produced severe RTX shadow banding."""
    stage, cfg = ground
    cfg.class_type(cfg)
    meshes = [UsdGeom.Mesh(prim) for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]

    assert len(meshes) == 1
    points = np.asarray(meshes[0].GetPointsAttr().Get(), dtype=np.float32)
    assert np.max(np.spacing(np.abs(points))) < 0.001
    material, _ = UsdShade.MaterialBindingAPI(meshes[0].GetPrim()).ComputeBoundMaterial()
    surface = material.ComputeSurfaceSource()[0]
    assert surface.GetIdAttr().Get() == "UsdPreviewSurface"
    np.testing.assert_allclose(surface.GetInput("diffuseColor").Get(), [0.18, 0.20, 0.24])


def test_duplicate_ground_does_not_overwrite_existing_stage(ground):
    stage, cfg = ground
    terrain = cfg.class_type(cfg)
    before = stage.GetRootLayer().ExportToString()

    with pytest.raises(ValueError, match="already exists"):
        terrain.import_ground_plane("terrain")

    assert stage.GetRootLayer().ExportToString() == before


@pytest.mark.parametrize("size", [(0.0, 2.0), (-1.0, 2.0), (float("nan"), 2.0)])
def test_invalid_visual_size_does_not_mutate_the_stage(ground, size):
    stage, cfg = ground
    terrain = cfg.class_type(cfg)
    before = stage.GetRootLayer().ExportToString()

    with pytest.raises(ValueError, match="size"):
        terrain.import_ground_plane("bad", size=size)

    assert stage.GetRootLayer().ExportToString() == before
    assert terrain.terrain_prim_paths == ["/World/ground/terrain"]
