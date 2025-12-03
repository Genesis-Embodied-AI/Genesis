"""
USD Rigid Body Parser

Parser for extracting rigid body information from USD stages.
The parser is agnostic to genesis structures, focusing only on USD rigid body structure.

Also includes Genesis-specific parsing functions that translate USD structures into Genesis info structures.
"""

from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf
from typing import List, Dict, Tuple
import genesis as gs
import numpy as np
import re
from scipy.spatial.transform import Rotation as R
from genesis.utils.usd.usd_parser_context import UsdParserContext
from genesis.utils.usd.usd_parser_utils import (bfs_iterator, 
                                                compute_gs_global_transform,
                                                compute_gs_related_transform,
                                                usd_mesh_to_gs_trimesh)
from genesis.utils.usd.usd_articulation_parser import UsdArticulationParser
from genesis.utils import geom as gu
from genesis.utils import mesh as mu

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
        
        is_rigid_body = UsdRigidBodyParser.regard_as_rigid_body(rigid_body_prim)
        if not is_rigid_body:
            gs.raise_exception(
                f"Provided prim {rigid_body_prim.GetPath()} is not a rigid body, APIs found: {rigid_body_prim.GetAppliedSchemas()}"
            )
            
        collision_api_only = rigid_body_prim.HasAPI(UsdPhysics.CollisionAPI) and not rigid_body_prim.HasAPI(UsdPhysics.RigidBodyAPI)
        kinematic_enabled = False
        if rigid_body_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_body_api = UsdPhysics.RigidBodyAPI(rigid_body_prim)
            kinematic_enabled = rigid_body_api.GetKinematicEnabledAttr().Get() if rigid_body_api.GetKinematicEnabledAttr().Get() else False
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
    
    # ==================== Mesh Collection Methods ====================
    
    @staticmethod
    def _get_meshes(prim: Usd.Prim) -> List[UsdGeom.Mesh]:
        """
        Get all meshes under the given prim.
        
        Parameters
        ----------
        prim : Usd.Prim
            The prim to search under.
            
        Returns
        -------
        List[UsdGeom.Mesh]
            List of meshes found.
        """
        meshes: List[UsdGeom.Mesh] = []
        
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            meshes.append(mesh)
        
        for child_prim in bfs_iterator(prim):
            if child_prim != prim and child_prim.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(child_prim)
                meshes.append(mesh)
        
        return meshes
    
    visual_pattern = re.compile(r'^(visual|Visual).*')
    
    def get_visual_meshes(self) -> List[UsdGeom.Mesh]:
        """
        Get all visual meshes under the rigid body prim.
        
        Returns
        -------
        List[UsdGeom.Mesh]
            List of visual meshes found.
        """
        # Find any child name starting with "visual" or "Visual"

        visual_prims: list[Usd.Prim] = []
        for child in self._root.GetChildren():
            if UsdRigidBodyParser.visual_pattern.match(child.GetName()):
                visual_prims.append(child)
        
        meshes: List[UsdGeom.Mesh] = []
        for visual_prim in visual_prims:
            meshes.extend(self._get_meshes(visual_prim))
        return meshes
    
    collision_pattern = re.compile(r'^(collision|Collision).*')
    
    def get_collision_meshes(self) -> List[UsdGeom.Mesh]:
        """
        Get all collision meshes under the rigid body prim.
        
        Returns
        -------
        List[UsdGeom.Mesh]
            List of collision meshes found.
        """
        # Find any child name starting with "collision" or "Collision"

        collision_prims: list[Usd.Prim] = []
        for child in self._root.GetChildren():
            if UsdRigidBodyParser.collision_pattern.match(child.GetName()):
                collision_prims.append(child)
        
        meshes: List[UsdGeom.Mesh] = []
        for collision_prim in collision_prims:
            meshes.extend(self._get_meshes(collision_prim))
        return meshes
    
    def get_all_meshes(self) -> List[UsdGeom.Mesh]:
        """
        Get all meshes under the rigid body prim.
        
        Returns
        -------
        List[UsdGeom.Mesh]
            List of all meshes found.
        """
        return self._get_meshes(self._root)
    
    def get_visual_and_collision_meshes(self) -> tuple[List[UsdGeom.Mesh], List[UsdGeom.Mesh]]:
        """
        Get visual and collision meshes. If none found, returns all meshes as both visual and collision.
        
        Returns
        -------
        visual_meshes : List[UsdGeom.Mesh]
            List of visual meshes.
        collision_meshes : List[UsdGeom.Mesh]
            List of collision meshes.
        """
        visual_meshes = self.get_visual_meshes()
        collision_meshes = self.get_collision_meshes()
        
        # If no visual/collision children found, look for meshes directly under the prim
        if not visual_meshes and not collision_meshes:
            all_meshes = self.get_all_meshes()
            # Use all meshes for both visual and collision if no explicit structure
            visual_meshes = all_meshes
            collision_meshes = all_meshes
        
        return visual_meshes, collision_meshes
    
    # ==================== Plane Collection Methods ====================
    
    @staticmethod
    def _get_planes(prim: Usd.Prim) -> List[UsdGeom.Plane]:
        """
        Get all plane prims under the given prim.
        
        Parameters
        ----------
        prim : Usd.Prim
            The prim to search under.
            
        Returns
        -------
        List[UsdGeom.Plane]
            List of planes found.
        """
        planes: List[UsdGeom.Plane] = []
        
        if prim.IsA(UsdGeom.Plane):
            plane = UsdGeom.Plane(prim)
            planes.append(plane)
        
        for child_prim in bfs_iterator(prim):
            if child_prim != prim and child_prim.IsA(UsdGeom.Plane):
                plane = UsdGeom.Plane(child_prim)
                planes.append(plane)
        
        return planes
    
    def get_planes(self) -> List[UsdGeom.Plane]:
        """
        Get all plane prims under the rigid body prim.
        
        Returns
        -------
        List[UsdGeom.Plane]
            List of planes found.
        """
        return self._get_planes(self._root)

    # ==================== Static Methods: Finding Rigid Bodies ====================
    
    @staticmethod
    def regard_as_rigid_body(prim: Usd.Prim) -> bool:
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
        
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return True
        
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            return True
        
        return False
    
    @staticmethod
    def find_all_rigid_bodies(stage: Usd.Stage, context: UsdParserContext = None) -> List[Usd.Prim]:
        """
        Find all top-most rigid body prims that are not part of an articulation.
        Only finds the head of each rigid body subtree (prims with RigidBodyAPI or CollisionAPI),
        and skips descendants to avoid recursive finding.
        
        Parameters
        ----------
        stage : Usd.Stage
            The USD stage.
        context : UsdParserContext, optional
            If provided, rigid body top prims will be added to the context.
            
        Returns
        -------
        List[Usd.Prim]
            List of top-most rigid body prims.
        """
        # First, collect all articulation roots and their descendants
        articulation_roots = UsdArticulationParser.find_all_articulation_roots(stage, context)
        articulation_descendants = set()
        for root in articulation_roots:
            for prim in bfs_iterator(root):
                articulation_descendants.add(prim.GetPath())
        
        # Track rigid body subtrees to skip descendants
        rigid_body_descendants = set()
        
        # Find all rigid bodies that are not part of any articulation
        rigid_bodies = []
        for prim in bfs_iterator(stage.GetPseudoRoot()):
            # Skip if already part of a rigid body subtree
            if prim.GetPath() in rigid_body_descendants:
                continue
            
            # Check for RigidBodyAPI or CollisionAPI
            if UsdRigidBodyParser.regard_as_rigid_body(prim):
                # Check if this prim is a descendant of any articulation
                is_descendant = False
                current_prim = prim
                while current_prim and current_prim != stage.GetPseudoRoot():
                    if current_prim.GetPath() in articulation_descendants:
                        is_descendant = True
                        break
                    current_prim = current_prim.GetParent()
                
                if not is_descendant:
                    rigid_bodies.append(prim)
                    if context:
                        context._rigid_body_top_prims[prim.GetPath()] = prim
                    # Mark all descendants of this rigid body as part of its subtree
                    # (they will be merged, not treated as separate rigid bodies)
                    for descendant in bfs_iterator(prim):
                        if descendant != prim:  # Don't mark the head itself
                            rigid_body_descendants.add(descendant.GetPath())
        
        return rigid_bodies

# ==================== Helper Functions for Genesis Parsing ====================

def _create_mesh_geometry_info(
    usd_mesh: UsdGeom.Mesh,
    rigid_body_prim: Usd.Prim,
    morph: gs.morphs.USDRigidBody,
    context: UsdParserContext,
    geom_is_col: bool
) -> Dict:
    """
    Create geometry info dictionary from a USD mesh.
    
    Parameters
    ----------
    usd_mesh : UsdGeom.Mesh
        The USD mesh to convert.
    rigid_body_prim : Usd.Prim
        The rigid body prim this geometry belongs to.
    morph : gs.morphs.USDRigidBody
        The rigid body morph (for scale and file path).
    context : UsdParserContext
        The parser context (for material lookup).
    geom_is_col : bool
        Whether this is a collision geometry.
        
    Returns
    -------
    dict
        Geometry info dictionary.
    """
    # Convert USD mesh to trimesh (genesis-agnostic conversion)
    Q_rel, tmesh = usd_mesh_to_gs_trimesh(usd_mesh, ref_prim=rigid_body_prim)
    
    # Get material for this mesh
    mesh_prim = usd_mesh.GetPrim()
        
    # Create mesh (genesis-specific)
    material = gs.surfaces.Collision() if geom_is_col else context.find_material(mesh_prim)
    
    mesh = gs.Mesh.from_trimesh(
        tmesh,
        scale=morph.scale,
        surface=material,
        metadata={"mesh_path": f"{morph.file}::{usd_mesh.GetPath()}"}
    )
    
    # Get transform relative to the rigid body
    geom_pos = Q_rel[:3, 3]
    geom_quat = gu.R_to_quat(Q_rel[:3, :3])
    
    visualization = getattr(morph, 'visualization', True)
    collision = getattr(morph, 'collision', True)
    
    # Return None if geometry should not be added
    if visualization and not geom_is_col:
        return dict(
            contype=0,
            conaffinity=0,
            vmesh=mesh,
            pos=geom_pos,
            quat=geom_quat,
        )
    elif collision and geom_is_col:
        return dict(
            contype=getattr(morph, 'contype', 1),
            conaffinity=getattr(morph, 'conaffinity', 1),
            mesh=mesh,
            type=gs.GEOM_TYPE.MESH,
            sol_params=gu.default_solver_params(),
            pos=geom_pos,
            quat=geom_quat,
        )
    else:
        return None


def _create_plane_geometry_info(
    usd_plane: UsdGeom.Plane,
    morph: gs.morphs.USDRigidBody,
    context: UsdParserContext
) -> Tuple[Dict, Dict]:
    """
    Create visual and collision geometry info dictionaries from a USD plane.
    
    Parameters
    ----------
    usd_plane : UsdGeom.Plane
        The USD plane to convert.
    morph : gs.morphs.USDRigidBody
        The rigid body morph (for scale and file path).
    context : UsdParserContext
        The parser context (for material lookup).
        
    Returns
    -------
    visual_g_info : dict or None
        Visual geometry info dictionary, or None if visualization is disabled.
    collision_g_info : dict or None
        Collision geometry info dictionary, or None if collision is disabled.
    """
    plane_prim = usd_plane.GetPrim()
    
    # Get plane properties
    width_attr = usd_plane.GetWidthAttr()
    length_attr = usd_plane.GetLengthAttr()
    axis_attr = usd_plane.GetAxisAttr()
    
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
    
    # Get plane transform relative to rigid body
    Q_rel, _ = compute_gs_related_transform(plane_prim, plane_prim)
    
    # Transform normal to world space (then to rigid body local space)
    plane_normal = Q_rel[:3, :3] @ plane_normal_local
    plane_normal = plane_normal / np.linalg.norm(plane_normal) if np.linalg.norm(plane_normal) > 1e-10 else plane_normal
    
    # Create plane geometry using mesh utility
    plane_size = (width, length)
    vmesh, cmesh = mu.create_plane(normal=plane_normal, plane_size=plane_size)
    
    # Get material for the plane
    material = context.find_material(plane_prim)
    
    # Create visual mesh
    vmesh_gs = gs.Mesh.from_trimesh(
        vmesh,
        scale=morph.scale,
        surface=material,
        metadata={"mesh_path": f"{morph.file}::{plane_prim.GetPath()}"}
    )
    
    # Create collision mesh
    cmesh_gs = gs.Mesh.from_trimesh(
        cmesh,
        scale=morph.scale,
        surface=gs.surfaces.Collision(),
        metadata={"mesh_path": f"{morph.file}::{plane_prim.GetPath()}"}
    )
    
    # Get plane position and orientation relative to rigid body
    geom_pos = Q_rel[:3, 3]
    geom_quat = gu.R_to_quat(Q_rel[:3, :3])
    
    # Plane normal for geom_data (in plane's local space, which is the axis direction)
    geom_data = plane_normal_local.copy()
    
    visualization = getattr(morph, 'visualization', True)
    collision = getattr(morph, 'collision', True)
    
    visual_g_info = None
    collision_g_info = None
    
    if visualization:
        visual_g_info = dict(
            contype=0,
            conaffinity=0,
            vmesh=vmesh_gs,
            pos=geom_pos,
            quat=geom_quat,
        )
    
    if collision:
        collision_g_info = dict(
            contype=getattr(morph, 'contype', 1),
            conaffinity=getattr(morph, 'conaffinity', 1),
            mesh=cmesh_gs,
            type=gs.GEOM_TYPE.PLANE,
            data=geom_data,
            sol_params=gu.default_solver_params(),
            pos=geom_pos,
            quat=geom_quat,
        )
    
    return visual_g_info, collision_g_info


def _create_rigid_body_link_info(
    rigid_body_prim: Usd.Prim,
    is_fixed: bool
) -> Tuple[Dict, Dict]:
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
    scale = getattr(morph, 'scale', 1.0)
    if scale != 1.0:
        gs.logger.warning("USD rigid body parsing currently only supports scale=1.0. Scale will be ignored.")
        scale = 1.0
    
    assert morph.parser_ctx is not None, "USDRigidBody must have a parser context."
    assert morph.prim_path is not None, "USDRigidBody must have a prim path."
    
    context: UsdParserContext = morph.parser_ctx
    stage: Usd.Stage = context.stage
    rigid_body_prim = stage.GetPrimAtPath(Sdf.Path(morph.prim_path))
    assert rigid_body_prim.IsValid(), f"Invalid prim path {morph.prim_path} in USD file {morph.file}."
    
    # Create parser for USD-agnostic extraction
    rigid_body = UsdRigidBodyParser(stage, rigid_body_prim)
    
    # Build geometry infos
    g_infos = []
    
    if rigid_body_prim.IsA(UsdGeom.Plane):
        usd_plane = UsdGeom.Plane(rigid_body_prim)
        visual_g_info, collision_g_info = _create_plane_geometry_info(
            usd_plane, morph, context)
        if visual_g_info:
            g_infos.append(visual_g_info)
        if collision_g_info:
            g_infos.append(collision_g_info)
    else:
        visual_meshes, collision_meshes = rigid_body.get_visual_and_collision_meshes()
        n_visuals = len(visual_meshes)
        
        for i, usd_mesh in enumerate((*visual_meshes, *collision_meshes)):
            geom_is_col = i >= n_visuals
            g_info = _create_mesh_geometry_info(usd_mesh, rigid_body_prim, morph, context, geom_is_col)
            if g_info:  # Only append if geometry info was created (respects visualization/collision flags)
                g_infos.append(g_info)
    
    # Create link and joint info
    l_info, j_info = _create_rigid_body_link_info(rigid_body_prim, rigid_body.is_fixed)
    j_infos = [j_info]
    
    return l_info, j_infos, g_infos
