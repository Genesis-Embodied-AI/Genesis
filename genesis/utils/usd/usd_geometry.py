import re
from typing import List, Dict, Literal
from enum import Enum

import numpy as np
import trimesh
from pxr import Usd, UsdGeom, UsdShade

import genesis as gs
from .usd_context import UsdContext
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

    def __init__(self, ctx: UsdContext, prim: Usd.Prim, ref_prim: Usd.Prim, mesh_type: Literal["mesh", "vmesh"]):
        self._prim: Usd.Prim = prim
        self._ref_prim: Usd.Prim = ref_prim
        self._ctx: UsdContext = ctx
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

        # Create plane geometry using mesh utility (for visualization)
        plane_size = (width, length)
        vmesh, cmesh = mu.create_plane(normal=plane_normal, plane_size=plane_size)
        plane_mesh = vmesh if self._mesh_type == "vmesh" else cmesh
        mesh_gs = self._create_mesh_from_trimesh(plane_mesh)

        return {
            self._mesh_type: mesh_gs,
            "type": gs.GEOM_TYPE.PLANE,
            "data": plane_normal,
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
            "type": gs.GEOM_TYPE.SPHERE,
            "data": np.array([radius]),
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
            "type": gs.GEOM_TYPE.CAPSULE,
            "data": np.array([radius, height]),
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

        # Create box mesh (for visualization)
        tmesh = mu.create_box(extents=extents)

        mesh = self._create_mesh_from_trimesh(tmesh)

        return {
            self._mesh_type: mesh,
            "type": gs.GEOM_TYPE.BOX,
            "data": extents,
            "pos": Q_rel[:3, 3],
            "quat": gu.R_to_quat(Q_rel[:3, :3]),
        }

    def _create_gs_cylinder_geo_info(self) -> Dict:
        """Create geometry info for USD Cylinder as a primitive."""
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

        mesh = self._create_mesh_from_trimesh(tmesh)

        return {
            self._mesh_type: mesh,
            "type": gs.GEOM_TYPE.CYLINDER,
            "data": np.array([radius, height]),
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


visual_pattern = re.compile(r"^(visual|Visual).*")
collision_pattern = re.compile(r"^(collision|Collision).*")


def geom_warning(geom_type, geom_id, stage_file, reason_msg, action_msg):
    gs.logger.warning(f"{reason_msg} for {geom_type} {geom_id} in usd file {stage_file}. {action_msg}.")


def parse_mesh_geometry(prim: Usd.Prim, scale, group_by_material, surface: gs.surfaces.Surface):
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
    # Get USD mesh attributes
    mesh_usd = UsdGeom.Mesh(prim)
    mesh_id = context.get_prim_id(prim)

    # parse vertices
    points_attr = mesh_usd.GetPointsAttr()
    if not points_attr.HasValue():
        geom_warning("Mesh", mesh_id, stage_file, "No vertices", "Skip this mesh")
        return None
    points = np.array(mesh_usd.GetPointsAttr().Get(), dtype=np.float32)

    # parse faces
    faces_attr = mesh_usd.GetFaceVertexCountsAttr()
    faces_vertex_counts_attr = mesh_usd.GetFaceVertexIndicesAttr()
    faces = np.array(faces_attr.Get(), dtype=np.int32) if faces_attr.HasValue() else np.array([], dtype=np.int32)
    faces_vertex_counts = (
        np.array(faces_vertex_counts_attr.Get())
        if faces_vertex_counts_attr.HasValue()
        else np.array([], dtype=np.int32)
    )
    points_faces_varying = False

    # parse normals
    normals = None
    normals_attr = mesh_usd.GetNormalsAttr()
    if normals_attr.HasValue():
        normals = np.array(normals_attr.Get(), dtype=np.float32)
        if normals.shape[0] != points.shape[0]:
            if normals.shape[0] == faces.shape[0]:  # face varying meshes, adjacent faces do not share vertices
                points_faces_varying = True
            else:
                geom_warning("Mesh", mesh_id, stage_file, "Size of normals mismatch", "Discard the normals")
                normals = None

    # parse materials
    if prim.HasRelationship("material:binding"):
        if not prim.HasAPI(UsdShade.MaterialBindingAPI):
            UsdShade.MaterialBindingAPI.Apply(prim)
    prim_bindings = UsdShade.MaterialBindingAPI(prim)
    material_usd = prim_bindings.ComputeBoundMaterial()[0]
    if material_usd.GetPrim().IsValid():
        material_id = context.get_prim_id(material_usd.GetPrim())
        material, uv_name = context.apply_surface(material_id, surface)
    else:
        material, uv_name, material_id = surface.copy(), "st", None

    # parse UVs
    uvs = None
    if uv_name is not None:
        uv_var = UsdGeom.PrimvarsAPI(prim).GetPrimvar(uv_name)
        if uv_var.IsDefined() and uv_var.HasValue():
            uvs = np.array(uv_var.ComputeFlattened(), dtype=np.float32)
            uvs[:, 1] = 1.0 - uvs[:, 1]  # Flip V coordinate
            if uvs.shape[0] != points.shape[0]:
                if uvs.shape[0] == faces.shape[0]:
                    points_faces_varying = True
                elif uvs.shape[0] == 1:
                    uvs = None
                else:
                    geom_warning("Mesh", mesh_id, stage_file, "Size of uvs mismatch", "Discard the uvs")
                    uvs = None

    # process faces
    if faces_vertex_counts.size == 0:
        triangles = np.array([], dtype=np.int32).reshape(0, 3)
    else:
        # rearrange points and faces
        if points_faces_varying:
            points = points[faces]
            faces = np.arange(faces.shape[0])

        # triangulate faces
        if np.max(faces_vertex_counts) > 3:
            triangles = []
            bi = 0
            for face_vertex_count in faces_vertex_counts:
                if face_vertex_count == 3:
                    triangles.append([faces[bi + 0], faces[bi + 1], faces[bi + 2]])
                elif face_vertex_count > 3:
                    for i in range(1, face_vertex_count - 1):
                        triangles.append([faces[bi + 0], faces[bi + i], faces[bi + i + 1]])
                bi += face_vertex_count
            triangles = np.array(triangles, dtype=np.int32)
        else:
            triangles = faces.reshape(-1, 3)

    # process mesh
    processed_mesh = trimesh.Trimesh(
        vertices=points,
        faces=triangles,
        vertex_normals=normals,
        visual=trimesh.visual.TextureVisuals(uv=uvs) if uvs is not None else None,
        process=True,
    )
    points = processed_mesh.vertices
    triangles = processed_mesh.faces
    normals = processed_mesh.vertex_normals
    if uvs is not None:
        uvs = processed_mesh.visual.uv


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


def parse_all_geometries(self, prim: Usd.Prim, surface: gs.surfaces.Surface, match_visual=False, match_collision=False):
    g_infos = []

    if visual_pattern.match(prim.GetName()):
        match_visual = True
    if collision_pattern.match(prim.GetName()):
        match_collision = True

    if prim.IsA(UsdGeom.Mesh):
        mesh = parse_mesh_geometry(prim)
        gs_type = gs.GEOM_TYPE.MESH
    elif prim.IsA(UsdGeom.Plane):
        mesh = parse_plane_geometry(prim)
        gs_type = gs.GEOM_TYPE.PLANE
    elif prim.IsA(UsdGeom.Sphere):
        mesh = parse_sphere_geometry(prim)
        gs_type = gs.GEOM_TYPE.SPHERE
    elif prim.IsA(UsdGeom.Capsule):
        mesh = parse_capsule_geometry(prim)
        gs_type = gs.GEOM_TYPE.CAPSULE
    elif prim.IsA(UsdGeom.Cube):
        mesh = parse_cube_geometry(prim)
        gs_type = gs.GEOM_TYPE.BOX
    elif prim.IsA(UsdGeom.Cylinder):
        mesh = parse_cylinder_geometry(prim)
        gs_type = gs.GEOM_TYPE.CYLINDER
    else:
        mesh = None
        gs_type = None

    if mesh is not None:
        geom_transform = compute_gs_related_transform(prim, ref_prim)
        geom_pos = geom_transform[:3, 3]
        geom_quat = gu.R_to_quat(geom_transform[:3, :3])
        is_visual = match_visual or not (match_collision or match_visual)
        is_collision = match_collision or not (match_collision or match_visual)
        if is_visual:
            g_infos.append(
                dict(
                    vmesh=mesh,
                    pos=geom_pos,
                    quat=geom_quat,
                    contype=0,
                    conaffinity=0,
                    type=gs_type,
                    data=geom_data,
                )
            )
        if is_collision:
            # TODO: use "physics:material:binding" to extract frictions
            g_infos.append(
                dict(
                    mesh=mesh,
                    pos=geom_pos,
                    quat=geom_quat,
                    contype=1,
                    conaffinity=1,
                    type=gs_type,
                    data=geom_data,
                    friction=gu.default_friction(),
                    sol_params=gu.default_solver_params(),
                )
            )

    for child in prim.GetChildren():
        g_infos.extend(parse_all_geometries(child, surface, match_visual, match_collision))

    return g_infos
