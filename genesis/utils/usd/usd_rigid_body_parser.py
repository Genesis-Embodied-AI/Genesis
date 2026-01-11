"""
USD Rigid Body Parser

Parser for extracting rigid body information from USD stages.
The parser is agnostic to genesis structures, focusing only on USD rigid body structure.

Also includes Genesis-specific parsing functions that translate USD structures into Genesis info structures.
"""

from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf
from typing import List, Dict, Tuple, Literal
import genesis as gs
import numpy as np
import re
from scipy.spatial.transform import Rotation as R
from .usd_parser_context import UsdParserContext
from .usd_parser_utils import bfs_iterator, compute_gs_global_transform, compute_gs_related_transform
from .usd_articulation_parser import UsdArticulationParser
from .usd_geo_adapter import BatchedUsdGeometryAdapater, UsdGeometryAdapter
from .. import geom as gu
from .. import mesh as mu


class UsdRigidBodyParser:
    """
    A parser to extract rigid body information from a USD stage.
    The Parser is agnostic to genesis structures, it only focuses on USD rigid body structure.
    """

    def __init__(self, stage: Usd.Stage, rigid_body_prim: Usd.Prim):
        """
        Initialize the rigid body parser.

        Parameters
        ----------
        stage : Usd.Stage
            The USD stage.
        rigid_body_prim : Usd.Prim
            The rigid body prim (must have RigidBodyAPI or CollisionAPI).
        """
        self._stage: Usd.Stage = stage
        self._root: Usd.Prim = rigid_body_prim
        self._g_infos: List[Dict] = []

        is_rigid_body = UsdRigidBodyParser.regard_as_rigid_body(rigid_body_prim)
        if not is_rigid_body:
            gs.raise_exception(
                f"Provided prim {rigid_body_prim.GetPath()} is not a rigid body, APIs found: {rigid_body_prim.GetAppliedSchemas()}"
            )

        collision_api_only = rigid_body_prim.HasAPI(UsdPhysics.CollisionAPI) and not rigid_body_prim.HasAPI(
            UsdPhysics.RigidBodyAPI
        )
        kinematic_enabled = False
        if rigid_body_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_body_api = UsdPhysics.RigidBodyAPI(rigid_body_prim)
            kinematic_enabled = (
                rigid_body_api.GetKinematicEnabledAttr().Get()
                if rigid_body_api.GetKinematicEnabledAttr().Get()
                else False
            )
        self.is_fixed = collision_api_only or kinematic_enabled

    # ==================== Properties ====================

    @property
    def stage(self) -> Usd.Stage:
        """Get the USD stage."""
        return self._stage

    @property
    def rigid_body_prim(self) -> Usd.Prim:
        """Get the rigid body prim."""
        return self._root

    # ==================== Geometry Collection Methods ====================

    all_pattern = re.compile(r"^.*")

    @staticmethod
    def is_geo_prim(prim: Usd.Prim) -> bool:
        return any(prim.IsA(geo_type) for geo_type in UsdGeometryAdapter.SupportUsdGeoms)

    @staticmethod
    def create_gs_geo_infos(
        context: UsdParserContext, rigid_body_prim: Usd.Prim, pattern, mesh_type: Literal["mesh", "vmesh"]
    ) -> List[Dict]:
        # if the rigid body itself is a geometry
        geo_infos: List[Dict] = []
        rigid_body_geo_adapter = UsdGeometryAdapter(context, rigid_body_prim, rigid_body_prim, mesh_type)
        rigid_body_geo_info = rigid_body_geo_adapter.create_gs_geo_info()
        if rigid_body_geo_info is not None:
            geo_infos.append(rigid_body_geo_info)

        # - RigidBody
        #     - Visuals
        #     - Collisions
        search_roots: list[Usd.Prim] = []
        for child in rigid_body_prim.GetChildren():
            if pattern.match(child.GetName()):
                search_roots.append(child)

        for search_root in search_roots:
            adapter = BatchedUsdGeometryAdapater(context, search_root, rigid_body_prim, mesh_type)
            geo_infos.extend(adapter.create_gs_geo_infos())

        return geo_infos

    def get_visual_geometries(self, context: UsdParserContext) -> List[Dict]:
        vis_geo_infos = UsdRigidBodyParser.create_gs_geo_infos(
            context, self._root, UsdRigidBodyParser.visual_pattern, "vmesh"
        )
        if len(vis_geo_infos) == 0:
            # if no visual geometries found, use any pattern to find visual geometries
            gs.logger.info(
                f"No visual geometries found, using any pattern to find visual geometries in {self._root.GetPath()}"
            )
            vis_geo_infos = UsdRigidBodyParser.create_gs_geo_infos(
                context, self._root, UsdRigidBodyParser.all_pattern, "vmesh"
            )
        return vis_geo_infos

    def get_collision_geometries(self, context: UsdParserContext) -> List[Dict]:
        col_geo_infos = UsdRigidBodyParser.create_gs_geo_infos(
            context, self._root, UsdRigidBodyParser.collision_pattern, "mesh"
        )
        return col_geo_infos

    def get_visual_and_collision_geometries(self, context: UsdParserContext) -> tuple[List[Dict], List[Dict]]:
        """
        Get visual and collision geometries.

        Parameters
        ----------
        context : UsdParserContext
            The parser context.

        Returns
        -------
        visual_g_infos : List[Dict]
            List of visual geometry info dictionaries.
        collision_g_infos : List[Dict]
            List of collision geometry info dictionaries.
        """
        visual_g_infos = self.get_visual_geometries(context)
        collision_g_infos = self.get_collision_geometries(context)

        return visual_g_infos, collision_g_infos


# ==================== Helper Functions for Genesis Parsing ====================


def _finalize_geometry_info(g_info: Dict, morph: gs.morphs.USDRigidBody, is_visual: bool) -> Dict:
    """
    Finalize geometry info dictionary by adding parser-specific fields.

    Parameters
    ----------
    g_info : dict
        Geometry info dictionary from adapter.
    morph : gs.morphs.USDRigidBody
        The rigid body morph (for visualization/collision flags).
    is_visual : bool
        Whether this is a visual geometry.

    Returns
    -------
    dict or None
        Finalized geometry info dictionary, or None if should be skipped.
    """
    visualization = getattr(morph, "visualization", True)
    collision = getattr(morph, "collision", True)

    # Return None if geometry should not be added
    if is_visual and not visualization:
        return None
    if not is_visual and not collision:
        return None

    # Add solver params if not present (for collision geometries)
    if not is_visual and "sol_params" not in g_info:
        g_info["sol_params"] = gu.default_solver_params()

    # Override contype/conaffinity for collision if specified in morph
    if not is_visual:
        g_info["contype"] = getattr(morph, "contype", g_info.get("contype", 1))
        g_info["conaffinity"] = getattr(morph, "conaffinity", g_info.get("conaffinity", 1))

    return g_info


def _create_rigid_body_link_info(rigid_body_prim: Usd.Prim, is_fixed: bool) -> Tuple[Dict, Dict]:
    """
    Create link and joint info dictionaries for a rigid body.

    Parameters
    ----------
    rigid_body_prim : Usd.Prim
        The rigid body prim.
    is_fixed : bool
        Whether the rigid body is fixed.

    Returns
    -------
    l_info : dict
        Link info dictionary.
    j_info : dict
        Joint info dictionary.
    """
    # Get the global position and quaternion of the rigid body
    Q_w, S = compute_gs_global_transform(rigid_body_prim)
    body_pos = Q_w[:3, 3]
    body_quat = gu.R_to_quat(Q_w[:3, :3])

    # Determine joint type and init_qpos
    if is_fixed:
        joint_type = gs.JOINT_TYPE.FIXED
        n_qs = 0
        n_dofs = 0
        init_qpos = np.zeros(0)
    else:
        joint_type = gs.JOINT_TYPE.FREE
        n_qs = 7
        n_dofs = 6
        init_qpos = np.concatenate([body_pos, body_quat])

    # Generate link name from prim path
    link_name = str(rigid_body_prim.GetPath())

    # Create link info
    l_info = dict(
        is_robot=False,
        name=f"{link_name}",
        pos=body_pos,
        quat=body_quat,
        inertial_pos=None,  # we will compute the COM later based on the geometry
        inertial_quat=gu.identity_quat(),
        parent_idx=-1,
    )

    # Create joint info
    j_info = dict(
        name=f"{link_name}_joint",  # we only have one joint for the rigid body
        n_qs=n_qs,
        n_dofs=n_dofs,
        type=joint_type,
        init_qpos=init_qpos,
    )

    return l_info, j_info


# ==================== Main Parsing Function ====================
def is_rigid_body(prim: Usd.Prim) -> bool:
    """
    Check if a prim should be regarded as a rigid body.

    Note: We regard CollisionAPI also as rigid body (they are fixed rigid body).

    Parameters
    ----------
    prim : Usd.Prim
        The prim to check.

    Returns
    -------
    bool
        True if the prim should be regarded as a rigid body, False otherwise.
    """
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        return False

        return True

    return False


def parse_all_geometries(context: UsdParserContext, rigid_body_prim: Usd.Prim) -> List[Dict]:
    pass


def parse_usd_rigid_body(morph: gs.morphs.USDRigidBody, surface: gs.surfaces.Surface):
    """
    Parse USD rigid body from the given USD file and prim path, returning info structures
    suitable for mesh-style loading (similar to gs.morph.Mesh).

    Returns
    -------
    l_info : dict
        Link info dictionary.
    j_infos : list
        List of joint info dictionaries (single joint for rigid body).
    g_infos : list
        List of geometry info dictionaries.
    """
    # Validate inputs and setup
    scale = getattr(morph, "scale", 1.0)

    context = morph.parser_ctx
    stage = context.stage
    rigid_body_prim = stage.GetPrimAtPath(morph.prim_path)
    if not rigid_body_prim.IsValid():
        gs.raise_exception(f"Invalid prim path {morph.prim_path} in USD file {morph.file}.")

    # Create parser for USD-agnostic extraction
    rigid_body = UsdRigidBodyParser(stage, rigid_body_prim)

    # Get visual and collision geometries
    visual_g_infos, collision_g_infos = rigid_body.parse_visual_and_collision_geometries(context)

    # Finalize and add visual geometries
    for g_info in visual_g_infos:
        finalized = _finalize_geometry_info(g_info, morph, is_visual=True)
        if finalized:
            g_infos.append(finalized)

    # Finalize and add collision geometries
    for g_info in collision_g_infos:
        finalized = _finalize_geometry_info(g_info, morph, is_visual=False)
        if finalized:
            g_infos.append(finalized)

    # Create link and joint info
    l_info, j_info = _create_rigid_body_link_info(rigid_body_prim, rigid_body.is_fixed)
    j_infos = [j_info]

    return l_info, j_infos, g_infos


# entrance
def parse_mesh_usd(path: str, group_by_material: bool, scale, surface: gs.surfaces.Surface):

    stage = Usd.Stage.Open(path)
    scale *= UsdGeom.GetStageMetersPerUnit(stage)
    yup = UsdGeom.GetStageUpAxis(stage) == "Y"
    xform_cache = UsdGeom.XformCache()

    mesh_infos = mu.MeshInfoGroup()
    materials = {}
    baked_materials = {}

    # parse geometries
    for prim in stage.Traverse():
        if prim.HasRelationship("material:binding"):
            if not prim.HasAPI(UsdShade.MaterialBindingAPI):
                UsdShade.MaterialBindingAPI.Apply(prim)
    for i, prim in enumerate(stage.Traverse()):
        if prim.IsA(UsdGeom.Mesh):
            matrix = np.asarray(xform_cache.GetLocalToWorldTransform(prim), dtype=np.float32)
            if yup:
                matrix @= mu.Y_UP_TRANSFORM
            mesh_usd = UsdGeom.Mesh(prim)
            mesh_spec = prim.GetPrimStack()[-1]
            mesh_id = mesh_spec.layer.identifier + mesh_spec.path.pathString

            if not mesh_usd.GetPointsAttr().HasValue():
                continue
            points = np.array(mesh_usd.GetPointsAttr().Get(), dtype=np.float32)
            faces = np.array(mesh_usd.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
            faces_vertex_counts = np.array(mesh_usd.GetFaceVertexCountsAttr().Get())
            points_faces_varying = False

            # parse normals
            normals = None
            normal_attr = mesh_usd.GetNormalsAttr()
            if normal_attr.HasValue():
                normals = np.array(normal_attr.Get(), dtype=np.float32)
                if normals.shape[0] != points.shape[0]:
                    if normals.shape[0] == faces.shape[0]:  # face varying meshes, adjacent faces do not share vertices
                        points_faces_varying = True
                    else:
                        gs.raise_exception(f"Size of normals mismatch for mesh {mesh_id} in usd file {path}.")

            # parse materials
            prim_bindings = UsdShade.MaterialBindingAPI(prim)
            material_usd = prim_bindings.ComputeBoundMaterial()[0]
            if material_usd.GetPrim().IsValid():
                material_spec = material_usd.GetPrim().GetPrimStack()[-1]
                material_file = material_spec.layer.identifier
                material_file = path if material_file == baked_path else material_file
                material_id = material_file + material_spec.path.pathString
                material, uv_name = materials.get(material_id, (None, "st"))
            else:
                material, uv_name, material_id = surface.copy(), "st", None

            # parse uvs
            uvs = None
            if uv_name is not None:
                uv_var = UsdGeom.PrimvarsAPI(prim).GetPrimvar(uv_name)
                if uv_var.IsDefined() and uv_var.HasValue():
                    uvs = np.array(uv_var.ComputeFlattened(), dtype=np.float32)
                    uvs[:, 1] = 1.0 - uvs[:, 1]
                    if uvs.shape[0] != points.shape[0]:
                        if uvs.shape[0] == faces.shape[0]:
                            points_faces_varying = True
                        elif uvs.shape[0] == 1:
                            uvs = None
                        else:
                            gs.raise_exception(f"Size of uvs mismatch for mesh {mesh_id} in usd file {path}.")

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
                gs.logger.warning(f"Mesh {mesh_usd} has non-triangle faces.")
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

            # apply tranform
            points, normals = mu.apply_transform(matrix, points, normals)

            group_idx = material_id if group_by_material else i
            mesh_info, first_created = mesh_infos.get(group_idx)
            if first_created:
                mesh_info.set_property(
                    surface=material,
                    metadata={
                        "path": path,  # unbaked file or cache
                        "name": material_id if group_by_material else mesh_id,
                        "require_bake": material_id in baked_materials,
                        "bake_success": material_id in baked_materials and material is not None,
                    },
                )
            mesh_info.append(points, triangles, normals, uvs)

    return mesh_infos.export_meshes(scale=scale)
