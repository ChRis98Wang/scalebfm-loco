"""A network-independent flat terrain for the existing tracking task.

Keep IsaacLab's terrain origins and material configuration, but author the
static collision plane and its visual surface directly into the USD stage.
No downloaded USD, MDL, texture, or global IsaacLab patch is needed.
"""

import math

from pxr import Sdf, UsdGeom, UsdPhysics, UsdShade

import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporter


class OfflinePlaneTerrainImporter(TerrainImporter):
    """Use a local infinite collision plane for ``terrain_type='plane'``.

    Other terrain types and origin layout remain implemented by IsaacLab.
    As with its default plane importer, only the configured visual material's
    diffuse color is used. The physics material is applied without overrides.
    """

    def import_ground_plane(self, name: str, size: tuple[float, float] = (2000.0, 2000.0)):
        # Only the visible surface is finite. Million-metre mesh vertices lose
        # float32 precision in RTX; the separate collision Plane stays infinite.
        if len(size) != 2 or any(not math.isfinite(value) or value <= 0 for value in size):
            raise ValueError(f"Ground size must contain two finite positive dimensions: {size}")
        stage = sim_utils.get_current_stage()
        prim_path = f"{self.cfg.prim_path}/{name}"
        if prim_path in self.terrain_prim_paths or stage.GetPrimAtPath(prim_path).IsValid():
            raise ValueError(f"A terrain prim already exists at '{prim_path}'.")

        UsdGeom.Xform.Define(stage, prim_path)
        try:
            # A static Plane collider (not a thin box or a triangle mesh) keeps
            # the original flat-ground contact geometry at z=0.
            plane = UsdGeom.Plane.Define(stage, f"{prim_path}/CollisionPlane")
            plane.CreateAxisAttr("Z")
            plane.MakeInvisible()
            UsdPhysics.CollisionAPI.Apply(plane.GetPrim()).CreateCollisionEnabledAttr(True)
            if self.cfg.physics_material is not None:
                material_path = f"{prim_path}/physicsMaterial"
                self.cfg.physics_material.func(material_path, self.cfg.physics_material)
                sim_utils.bind_physics_material(str(plane.GetPath()), material_path)

            half_x, half_y = size[0] / 2.0, size[1] / 2.0
            mesh = UsdGeom.Mesh.Define(stage, f"{prim_path}/Surface")
            mesh.CreatePointsAttr([
                (-half_x, -half_y, 0.0), (half_x, -half_y, 0.0),
                (half_x, half_y, 0.0), (-half_x, half_y, 0.0),
            ])
            mesh.CreateFaceVertexCountsAttr([4])
            mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
            mesh.CreateSubdivisionSchemeAttr("none")
            mesh.CreateExtentAttr([(-half_x, -half_y, 0.0), (half_x, half_y, 0.0)])

            color = getattr(self.cfg.visual_material, "diffuse_color", (0.18, 0.20, 0.24))
            material = UsdShade.Material.Define(stage, f"{prim_path}/visualMaterial")
            shader = UsdShade.Shader.Define(stage, f"{material.GetPath()}/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
            shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
        except Exception:
            stage.RemovePrim(prim_path)
            raise

        self.terrain_prim_paths.append(prim_path)
