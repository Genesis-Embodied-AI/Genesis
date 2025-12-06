import pxr
import genesis as gs
import numpy as np
import trimesh
from typing import List, Dict, Literal
from enum import Enum
from pxr import Usd, UsdGeom
from genesis.utils.usd.usd_parser_context import UsdParserContext
from genesis.utils.usd.usd_parser_utils import usd_mesh_to_gs_trimesh, compute_gs_related_transform
from genesis.utils import geom as gu
from genesis.utils import mesh as mu
from genesis.utils.usd.usd_parser_utils import bfs_iterator


class UsdGeometryAdapter:
    """
    A adapter to convert USD geometry to Genesis geometry info.
    Receive: UsdGeom.Mesh, UsdGeom.Plane, UsdGeom.Sphere, UsdGeom.Capsule, UsdGeom.Cube
    Return: Genesis geometry info
    """

    SupportUsdGeoms = [UsdGeom.Mesh, UsdGeom.Plane, UsdGeom.Sphere, UsdGeom.Capsule, UsdGeom.Cube]

    def __init__(self, ctx: UsdParserContext, prim: Usd.Prim, ref_prim: Usd.Prim, mesh_type: Literal["mesh", "vmesh"]):
        self._prim: Usd.Prim = prim
        self._ref_prim: Usd.Prim = ref_prim
        self._ctx: UsdParserContext = ctx
        self._mesh_type: Literal["mesh", "vmesh"] = mesh_type
        pass

    def create_gs_geo_info(self) -> Dict:
        g_info = dict()
        geom_is_col = self._mesh_type == "mesh"
        g_info["contype"] = 1 if geom_is_col else 0
        g_info["conaffinity"] = 1 if geom_is_col else 0
        g_info["friction"] = gu.default_friction()
        g_info["sol_params"] = gu.default_solver_params()

        if self._prim.IsA(UsdGeom.Mesh):
            r = self._create_gs_mesh_geo_info()
            g_info.update(r)
        elif self._prim.IsA(UsdGeom.Plane):
            r = self._create_gs_plane_geo_info()
            g_info.update(r)
        elif self._prim.IsA(UsdGeom.Sphere):
            r = self._create_gs_sphere_geo_info()
            g_info.update(r)
        elif self._prim.IsA(UsdGeom.Capsule):
            r = self._create_gs_capsule_geo_info()
            g_info.update(r)
        elif self._prim.IsA(UsdGeom.Cube):
            r = self._create_gs_cube_geo_info()
            g_info.update(r)
        else:
            return None

        return g_info

    def _create_gs_mesh_geo_info(self) -> Dict:
        """Create geometry info for USD Mesh."""
        mesh_prim = UsdGeom.Mesh(self._prim)
        Q_rel, tmesh = usd_mesh_to_gs_trimesh(mesh_prim, self._ref_prim)

        mesh = self._create_mesh_from_trimesh(tmesh)

        return {
            self._mesh_type: mesh,
            "type": gs.GEOM_TYPE.MESH,
            "data": None,
            "pos": Q_rel[:3, 3],
            "quat": gu.R_to_quat(Q_rel[:3, :3]),
        }

    def _get_surface(self) -> gs.surfaces.Surface:
        """Get the surface material for the geometry."""
        if self._mesh_type == "mesh":
            default_surface = gs.surfaces.Collision()
            default_surface.color = (1.0, 0.0, 1.0)
            return default_surface
        else:
            return self._ctx.find_material(self._prim)

    def _get_usd_file_path(self) -> str:
        """Get the USD file path from the stage."""
        if self._ctx.stage.GetRootLayer():
            return self._ctx.stage.GetRootLayer().realPath
        return ""

    def _create_mesh_from_trimesh(self, tmesh: trimesh.Trimesh) -> gs.Mesh:
        """Create a Genesis Mesh from a trimesh with common parameters."""
        return gs.Mesh.from_trimesh(
            tmesh,
            scale=1.0,
            surface=self._get_surface(),
            metadata={"mesh_path": f"{self._get_usd_file_path()}::{self._prim.GetPath()}"},
        )

    def _create_gs_plane_geo_info(self) -> Dict:
        """Create geometry info for USD Plane."""
        plane_prim = UsdGeom.Plane(self._prim)

        # Get plane properties
        width_attr = plane_prim.GetWidthAttr()
        length_attr = plane_prim.GetLengthAttr()
        axis_attr = plane_prim.GetAxisAttr()

        # Get plane dimensions (default to large size if not specified)
        width = width_attr.Get() if width_attr and width_attr.HasValue() else 1e3
        length = length_attr.Get() if length_attr and length_attr.HasValue() else 1e3

        # Get plane axis (default to "Z" for Z-up)
        axis_str = axis_attr.Get() if axis_attr and axis_attr.HasValue() else "Z"

        # Convert axis string to normal vector
        if axis_str == "X":
            plane_normal_local = np.array([1.0, 0.0, 0.0])
        elif axis_str == "Y":
            plane_normal_local = np.array([0.0, 1.0, 0.0])
        elif axis_str == "Z":
            plane_normal_local = np.array([0.0, 0.0, 1.0])
        else:
            gs.logger.warning(f"Unsupported plane axis {axis_str}, defaulting to Z.")
            plane_normal_local = np.array([0.0, 0.0, 1.0])

        # Get plane transform relative to reference prim (includes scale S)
        Q_rel, S = compute_gs_related_transform(self._prim, self._ref_prim)
        S_diag = np.diag(S)

        # Apply scale to plane dimensions
        # For plane, scale width and length based on the plane's orientation
        # If axis is X, scale by Y and Z components; if Y, scale by X and Z; if Z, scale by X and Y
        if axis_str == "X":
            width *= S_diag[1]  # Y scale
            length *= S_diag[2]  # Z scale
        elif axis_str == "Y":
            width *= S_diag[0]  # X scale
            length *= S_diag[2]  # Z scale
        else:  # Z
            width *= S_diag[0]  # X scale
            length *= S_diag[1]  # Y scale

        # Transform normal to reference prim's local space
        plane_normal = Q_rel[:3, :3] @ plane_normal_local
        plane_normal = (
            plane_normal / np.linalg.norm(plane_normal) if np.linalg.norm(plane_normal) > 1e-10 else plane_normal
        )

        # Create plane geometry using mesh utility
        plane_size = (width, length)
        vmesh, cmesh = mu.create_plane(normal=plane_normal, plane_size=plane_size)
        plane_mesh = vmesh if self._mesh_type == "vmesh" else cmesh
        mesh_gs = self._create_mesh_from_trimesh(plane_mesh)

        return {
            self._mesh_type: mesh_gs,
            "type": gs.GEOM_TYPE.PLANE,
            "data": plane_normal_local.copy(),
            "pos": Q_rel[:3, 3],
            "quat": gu.R_to_quat(Q_rel[:3, :3]),
        }

    def _create_gs_sphere_geo_info(self) -> Dict:
        """Create geometry info for USD Sphere."""
        sphere_prim = UsdGeom.Sphere(self._prim)

        # Get sphere radius
        radius_attr = sphere_prim.GetRadiusAttr()
        radius = radius_attr.Get() if radius_attr and radius_attr.HasValue() else 0.5

        # Get transform relative to reference prim (includes scale S)
        Q_rel, S = compute_gs_related_transform(self._prim, self._ref_prim)
        S_diag = np.diag(S)

        if not np.allclose(S_diag, S_diag[0]):
            gs.logger.warning(
                f"Sphere: {self._prim.GetPath()} scale is not uniform: {S}, take the mean of the three components"
            )
            radius *= np.mean(S_diag)
        else:
            radius *= S_diag[0]

        # Create sphere mesh (use fewer subdivisions for collision, more for visual)
        subdivisions = 2 if self._mesh_type == "mesh" else 3
        tmesh = mu.create_sphere(radius=radius, subdivisions=subdivisions)

        mesh = self._create_mesh_from_trimesh(tmesh)

        # Sphere data is just the radius
        geom_data = np.array([radius])

        return {
            self._mesh_type: mesh,
            "type": gs.GEOM_TYPE.SPHERE,
            "data": geom_data,
            "pos": Q_rel[:3, 3],
            "quat": gu.R_to_quat(Q_rel[:3, :3]),
        }

    def _create_gs_capsule_geo_info(self) -> Dict:
        """Create geometry info for USD Capsule."""
        capsule_prim = UsdGeom.Capsule(self._prim)

        # Get capsule properties
        radius_attr = capsule_prim.GetRadiusAttr()
        height_attr = capsule_prim.GetHeightAttr()
        axis_attr = capsule_prim.GetAxisAttr()

        # Get capsule dimensions (defaults)
        radius = radius_attr.Get() if radius_attr and radius_attr.HasValue() else 0.5
        height = height_attr.Get() if height_attr and height_attr.HasValue() else 1.0

        # Get axis (default to "Z")
        axis_str = axis_attr.Get() if axis_attr and axis_attr.HasValue() else "Z"

        # Get transform relative to reference prim (includes scale S)
        Q_rel, S = compute_gs_related_transform(self._prim, self._ref_prim)
        S_diag = np.diag(S)

        # Apply scale to capsule dimensions
        # Height scales along the axis direction, radius scales perpendicular to axis
        if axis_str == "X":
            height *= S_diag[0]  # X scale
            radius *= np.mean([S_diag[1], S_diag[2]])
        elif axis_str == "Y":
            height *= S_diag[1]  # Y scale
            # Radius scales by average of X and Z
            radius *= np.mean([S_diag[0], S_diag[2]])
        elif axis_str == "Z":
            height *= S_diag[2]  # Z scale
            # Radius scales by average of X and Y
            radius *= np.mean([S_diag[0], S_diag[1]])

        # Create capsule mesh (use fewer subdivisions for collision, more for visual)
        # Note: trimesh capsule uses count parameter (radial, height)
        count = (8, 12) if self._mesh_type == "mesh" else (16, 24)
        tmesh = trimesh.creation.capsule(radius=radius, height=height, count=count)

        mesh = self._create_mesh_from_trimesh(tmesh)

        # Capsule data: radius and height
        geom_data = np.array([radius, height])

        return {
            self._mesh_type: mesh,
            "type": gs.GEOM_TYPE.CAPSULE,
            "data": geom_data,
            "pos": Q_rel[:3, 3],
            "quat": gu.R_to_quat(Q_rel[:3, :3]),
        }

    def _create_gs_cube_geo_info(self) -> Dict:
        """Create geometry info for USD Cube (Box)."""
        cube_prim = UsdGeom.Cube(self._prim)

        # Get cube size/extents
        size_attr = cube_prim.GetSizeAttr()
        if size_attr and size_attr.HasValue():
            # If size is a single value, create uniform cube
            size_val = size_attr.Get()
            if isinstance(size_val, (int, float)):
                extents = np.array([size_val, size_val, size_val])
            else:
                extents = np.array(size_val)
        else:
            # Try to get extent (bounding box)
            extent_attr = cube_prim.GetExtentAttr()
            if extent_attr and extent_attr.HasValue():
                extent = extent_attr.Get()
                # Extent is typically [min, max] for each axis
                if len(extent) == 6:
                    extents = np.array([extent[1] - extent[0], extent[3] - extent[2], extent[5] - extent[4]])
                else:
                    extents = np.array([1.0, 1.0, 1.0])
            else:
                # Default size
                extents = np.array([1.0, 1.0, 1.0])

        # Get transform relative to reference prim (includes scale S)
        Q_rel, S = compute_gs_related_transform(self._prim, self._ref_prim)
        S_diag = np.diag(S)
        # Apply scale to extents (element-wise multiplication)
        extents = S_diag * extents

        # Create box mesh
        tmesh = mu.create_box(extents=extents)

        mesh = self._create_mesh_from_trimesh(tmesh)

        # Box data is the extents
        geom_data = extents.copy()

        return {
            self._mesh_type: mesh,
            "type": gs.GEOM_TYPE.BOX,
            "data": geom_data,
            "pos": Q_rel[:3, 3],
            "quat": gu.R_to_quat(Q_rel[:3, :3]),
        }


class BatchedUsdGeometryAdapater:
    """
    A adapter to convert USD geometry to Genesis geometry info.
    Receive: List[UsdGeom.Mesh], List[UsdGeom.Plane], List[UsdGeom.Sphere], List[UsdGeom.Capsule], List[UsdGeom.Cube]
    Return: List[Dict]
    """

    def __init__(
        self, ctx: UsdParserContext, start_prim: Usd.Prim, ref_prim: Usd.Prim, mesh_type: Literal["mesh", "vmesh"]
    ):
        self._ctx: UsdParserContext = ctx
        self._start_prim: Usd.Prim = start_prim
        self._ref_prim: Usd.Prim = ref_prim
        self._mesh_type: Literal["mesh", "vmesh"] = mesh_type
        self._geometries: List[Usd.Prim] = self._find_all_geometries()

    def _find_all_geometries(self) -> List[Usd.Prim]:
        """Find all geometries under the start prim."""
        geometries: List[Usd.Prim] = []

        # consider the start prim itself
        for geom_type in UsdGeometryAdapter.SupportUsdGeoms:
            if self._start_prim.IsA(geom_type):
                geometries.append(self._start_prim)
                break

        # consider the children of the start prim
        for prim in bfs_iterator(self._start_prim):
            for geom_type in UsdGeometryAdapter.SupportUsdGeoms:
                if prim.IsA(geom_type):
                    geometries.append(prim)
                    break
        return geometries

    def create_gs_geo_infos(self) -> List[Dict]:
        """Create geometry info for all geometries."""
        g_infos: List[Dict] = []
        for geometry in self._geometries:
            g_info = UsdGeometryAdapter(self._ctx, geometry, self._ref_prim, self._mesh_type).create_gs_geo_info()
            assert g_info is not None, f"Geometry: {geometry.GetPath()} create gs geo info failed"
            g_infos.append(g_info)
        return g_infos
