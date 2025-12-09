import pxr
import genesis as gs
import numpy as np
import trimesh
from typing import List, Dict, Literal
from enum import Enum
from pxr import Usd, UsdGeom, UsdShade
from genesis.utils.usd.usd_parser_context import UsdParserContext
from genesis.utils.usd.usd_parser_utils import compute_gs_related_transform, compute_gs_global_transform
from genesis.utils import geom as gu
from genesis.utils import mesh as mu
from genesis.utils.usd.usd_parser_utils import bfs_iterator


class UsdGeometryAdapter:
    """
    A adapter to convert USD geometry to Genesis geometry info.
    Receive: UsdGeom.Mesh, UsdGeom.Plane, UsdGeom.Sphere, UsdGeom.Capsule, UsdGeom.Cube
    Return: Genesis geometry info
    """

    SupportUsdGeoms = [UsdGeom.Mesh, UsdGeom.Plane, UsdGeom.Sphere, UsdGeom.Capsule, UsdGeom.Cube, UsdGeom.Cylinder]

    def __init__(self, ctx: UsdParserContext, prim: Usd.Prim, ref_prim: Usd.Prim, mesh_type: Literal["mesh", "vmesh"]):
        self._prim: Usd.Prim = prim
        self._ref_prim: Usd.Prim = ref_prim
        self._ctx: UsdParserContext = ctx
        self._mesh_type: Literal["mesh", "vmesh"] = mesh_type
    
    def is_primitive(self) -> bool:
        return self._is_primitive

    def create_gs_geo_info(self) -> Dict:
        g_info = dict()
        geom_is_col = self._mesh_type == "mesh"
        g_info["contype"] = 1 if geom_is_col else 0
        g_info["conaffinity"] = 1 if geom_is_col else 0
        g_info["friction"] = gu.default_friction()
        g_info["sol_params"] = gu.default_solver_params()

        if self._prim.IsA(UsdGeom.Mesh):
            if self._mesh_type == "vmesh":
                r = self._create_gs_visual_mesh_geo_info()
            else:
                r = self._create_gs_collision_mesh_geo_info()
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
        elif self._prim.IsA(UsdGeom.Cylinder):
            r = self._create_gs_cylinder_geo_info()
            g_info.update(r)
        else:
            # gs.logger.warning(f"Unsupported Geometry Type: {self._prim.GetTypeName()} of {self._prim}")
            return None

        return g_info

    def _extract_mesh_geometry(self, mesh_prim: UsdGeom.Mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract basic mesh geometry (points, face_vertex_indices, face_vertex_counts, triangles).
        Parameters:
            mesh_prim: UsdGeom.Mesh
                The USD mesh to extract geometry from.
        Returns:
            tuple: (Q_rel, points, triangles)
                - Q_rel: np.ndarray, shape (4, 4) - The Genesis transformation matrix (rotation and translation)
                  relative to ref_prim. This is the Q transform without scaling.
                - points: np.ndarray, shape (n, 3) - The points of the mesh.
                - normals: np.ndarray, shape (n, 3) - The normals of the mesh.
                - uvs: np.ndarray, shape (n, 2) - The UVs of the mesh.
                - triangles: np.ndarray, shape (m, 3) - The triangles of the mesh.
        """
        # Compute Genesis transform relative to ref_prim (Q^i_j)
        Q_rel, S = compute_gs_related_transform(mesh_prim.GetPrim(), self._ref_prim)

        # Get USD mesh attributes
        points_attr = mesh_prim.GetPointsAttr()
        face_vertex_counts_attr = mesh_prim.GetFaceVertexCountsAttr()
        face_vertex_indices_attr = mesh_prim.GetFaceVertexIndicesAttr()

        if not points_attr.HasValue():
            gs.raise_exception(f"Mesh {mesh_prim.GetPath()} has no points.")

        # Get points and apply scaling
        points = np.array(points_attr.Get(), dtype=np.float32)
        points = points @ S  # Apply scaling

        # Get face data
        face_vertex_indices = (
            np.array(face_vertex_indices_attr.Get(), dtype=np.int32)
            if face_vertex_indices_attr.HasValue()
            else np.array([], dtype=np.int32)
        )
        face_vertex_counts = (
            np.array(face_vertex_counts_attr.Get())
            if face_vertex_counts_attr.HasValue()
            else np.array([], dtype=np.int32)
        )

        points_faces_varying = False
        # Parse normals
        normals = None
        normal_attr = mesh_prim.GetNormalsAttr()
        if normal_attr.HasValue():
            normals = np.array(normal_attr.Get(), dtype=np.float32)
            if normals.shape[0] != points.shape[0]:
                if normals.shape[0] == face_vertex_indices.shape[0]:  # face varying meshes
                    points_faces_varying = True
                else:
                    gs.logger.warning(
                        f"Size of normals mismatch for mesh {mesh_prim.GetPath()} in usd file {self._get_usd_file_path()}."
                    )
                    normals = None

        uv_name = self._get_uv_name()

        # Parse UVs
        uvs = None
        if uv_name is not None:
            uv_var = UsdGeom.PrimvarsAPI(self._prim).GetPrimvar(uv_name)
            if uv_var.IsDefined() and uv_var.HasValue():
                uvs = np.array(uv_var.ComputeFlattened(), dtype=np.float32)
                uvs[:, 1] = 1.0 - uvs[:, 1]  # Flip V coordinate
                if uvs.shape[0] != points.shape[0]:
                    if uvs.shape[0] == face_vertex_indices.shape[0]:
                        points_faces_varying = True
                    elif uvs.shape[0] == 1:
                        uvs = None
                    else:
                        gs.logger.warning(
                            f"Size of uvs mismatch for mesh {mesh_prim.GetPath()} in usd file {self._get_usd_file_path()}."
                        )
                        uvs = None

        # Triangulate faces
        if len(face_vertex_counts) == 0:
            triangles = np.array([], dtype=np.int32).reshape(0, 3)
        else:
            # rearrange points and faces
            if points_faces_varying:
                points = points[face_vertex_indices]
                face_vertex_indices = np.arange(face_vertex_indices.shape[0])

            # triangulate faces
            if np.max(face_vertex_counts) > 3:
                triangles = []
                bi = 0
                for face_vertex_count in face_vertex_counts:
                    if face_vertex_count == 3:
                        triangles.append(
                            [face_vertex_indices[bi + 0], face_vertex_indices[bi + 1], face_vertex_indices[bi + 2]]
                        )
                    elif face_vertex_count > 3:
                        for i in range(1, face_vertex_count - 1):
                            triangles.append(
                                [
                                    face_vertex_indices[bi + 0],
                                    face_vertex_indices[bi + i],
                                    face_vertex_indices[bi + i + 1],
                                ]
                            )
                    bi += face_vertex_count
                triangles = np.array(triangles, dtype=np.int32)
            else:
                triangles = face_vertex_indices.reshape(-1, 3)
                # Get UV name from material (needed for UV extraction)

        return Q_rel, points, normals, uvs, triangles

    def _create_gs_visual_mesh_geo_info(self) -> Dict:
        """Create geometry info for USD visual Mesh with rendering information."""
        mesh_prim = UsdGeom.Mesh(self._prim)

        # Extract basic geometry
        Q_rel, points, normals, uvs, triangles = self._extract_mesh_geometry(mesh_prim)

        # Create trimesh with normals and UVs
        tmesh = trimesh.Trimesh(
            vertices=points,
            faces=triangles,
            vertex_normals=normals,
            visual=trimesh.visual.TextureVisuals(uv=uvs) if uvs is not None else None,
            process=True,
        )

        # Update normals and UVs from processed mesh
        if tmesh.vertex_normals is not None:
            normals = tmesh.vertex_normals
        if uvs is not None and tmesh.visual is not None:
            uvs = tmesh.visual.uv

        # Create Genesis mesh from trimesh
        mesh = self._create_mesh_from_trimesh(tmesh)

        return {
            self._mesh_type: mesh,
            "type": gs.GEOM_TYPE.MESH,
            "data": None,
            "pos": Q_rel[:3, 3],
            "quat": gu.R_to_quat(Q_rel[:3, :3]),
        }
    
    def _create_gs_collision_mesh_geo_info(self) -> Dict:
        """Create geometry info for USD collision Mesh without rendering information."""
        mesh_prim = UsdGeom.Mesh(self._prim)

        # Extract basic geometry (no rendering info needed)
        Q_rel, points, normals, uvs, triangles = self._extract_mesh_geometry(mesh_prim)

        # Create trimesh without normals or UVs (collision meshes don't need rendering info)
        tmesh = trimesh.Trimesh(
            vertices=points,
            faces=triangles,
            vertex_normals=normals,
            process=True,
        )

        # Create Genesis mesh from trimesh (uses Collision surface from _get_surface)
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

    def _get_uv_name(self) -> str:
        """Get the UV name from the material for the geometry."""
        if self._mesh_type == "mesh":
            return "st"  # Default UV name for collision meshes
        else:
            # Get UV name from material in context
            if self._prim.HasRelationship("material:binding"):
                if not self._prim.HasAPI(UsdShade.MaterialBindingAPI):
                    UsdShade.MaterialBindingAPI.Apply(self._prim)
                prim_bindings = UsdShade.MaterialBindingAPI(self._prim)
                material_usd = prim_bindings.ComputeBoundMaterial()[0]
                if material_usd.GetPrim().IsValid():
                    material_spec = material_usd.GetPrim().GetPrimStack()[-1]
                    material_id = material_spec.layer.identifier + material_spec.path.pathString
                    material_result = self._ctx.materials.get(material_id)
                    if material_result is not None:
                        _, uv_name = material_result
                        return uv_name
            return "st"  # Default UV name

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
            "type": gs.GEOM_TYPE.MESH,
            "pos": Q_rel[:3, 3],
            "quat": gu.R_to_quat(Q_rel[:3, :3]),
        }

    def _create_gs_sphere_geo_info(self) -> Dict:
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

        return {
            self._mesh_type: mesh,
            "type": gs.GEOM_TYPE.MESH,
            "pos": Q_rel[:3, 3],
            "quat": gu.R_to_quat(Q_rel[:3, :3]),
        }

    def _create_gs_capsule_geo_info(self) -> Dict:
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

        return {
            self._mesh_type: mesh,
            "type": gs.GEOM_TYPE.MESH,
            "pos": Q_rel[:3, 3],
            "quat": gu.R_to_quat(Q_rel[:3, :3]),
        }

    def _create_gs_cube_geo_info(self) -> Dict:
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

        return {
            self._mesh_type: mesh,
            "type": gs.GEOM_TYPE.MESH,
            "pos": Q_rel[:3, 3],
            "quat": gu.R_to_quat(Q_rel[:3, :3]),
        }

    def _create_gs_cylinder_geo_info(self) -> Dict:
        """Create geometry info for USD Cylinder as a mesh (transform baked into vertices)."""
        cylinder_prim = UsdGeom.Cylinder(self._prim)

        # Get cylinder properties
        radius_attr = cylinder_prim.GetRadiusAttr()
        height_attr = cylinder_prim.GetHeightAttr()
        axis_attr = cylinder_prim.GetAxisAttr()

        # Get cylinder dimensions (defaults)
        radius = radius_attr.Get() if radius_attr and radius_attr.HasValue() else 0.5
        height = height_attr.Get() if height_attr and height_attr.HasValue() else 1.0

        # Get axis (default to "Z")
        axis_str = axis_attr.Get() if axis_attr and axis_attr.HasValue() else "Z"

        # Get transform relative to reference prim (includes scale S)
        Q_rel, S = compute_gs_related_transform(self._prim, self._ref_prim)
        S_diag = np.diag(S)

        # Apply scale to cylinder dimensions
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

        # Create cylinder mesh (use fewer sections for collision, more for visual)
        sections = 8 if self._mesh_type == "mesh" else 16
        tmesh = mu.create_cylinder(radius=radius, height=height, sections=sections)

        # Apply transform to mesh vertices (bake pos and quat into vertices)
        vertices = tmesh.vertices
        vertices_transformed = (Q_rel[:3, :3] @ vertices.T).T + Q_rel[:3, 3]
        tmesh.vertices = vertices_transformed

        mesh = self._create_mesh_from_trimesh(tmesh)

        return {
            self._mesh_type: mesh,
            "type": gs.GEOM_TYPE.MESH,
            "data": None,
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
