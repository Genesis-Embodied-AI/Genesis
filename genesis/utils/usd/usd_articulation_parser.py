"""
USD Articulation Parser

Parser for extracting articulation information from USD stages.
The parser is agnostic to genesis structures, focusing only on USD articulation structure.

Also includes Genesis-specific parsing functions that translate USD structures into Genesis info structures.
"""

from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf
from typing import List
import genesis as gs
import numpy as np
import re
from scipy.spatial.transform import Rotation as R
from genesis.utils.usd.usd_parser_context import UsdParserContext
from genesis.utils.usd.usd_parser_utils import (bfs_iterator, 
                                               compute_usd_global_transform, 
                                               compute_usd_related_transform, 
                                               compute_gs_global_transform,
                                               compute_gs_related_transform,
                                               extract_rotation_and_scale,
                                               extract_quat_from_transform,
                                               apply_transform_to_pos,
                                               usd_mesh_to_gs_trimesh, 
                                               usd_quat_to_numpy,
                                               convert_usd_joint_axis_to_gs_link_space,
                                               convert_usd_joint_pos_to_gs_link_space,
                                               compute_joint_axis_scaling_factor)
from genesis.utils import geom as gu

class UsdArticulationParser:
    """
    A parser to extract articulation information from a USD stage.
    The Parser is agnostic to genesis structures, it only focuses on USD articulation structure.
    """
    
    def __init__(self, stage: Usd.Stage, articulation_root_prim: Usd.Prim):
        """
        Initialize the articulation parser.
        
        Parameters
        ----------
        stage : Usd.Stage
            The USD stage.
        articulation_root_prim : Usd.Prim
            The root prim of the articulation (must have ArticulationRootAPI).
        """
        self._stage: Usd.Stage = stage
        self._root: Usd.Prim = articulation_root_prim
        if not articulation_root_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            gs.raise_exception(
                f"Provided prim {articulation_root_prim.GetPath()} is not an Articulation Root. Now we only support articulation parsing from ArticulationRootAPI."
            )
        
        gs.logger.info(f"Parsing USD articulation from {articulation_root_prim.GetPath()}.")
        
        self.joints: List[UsdPhysics.Joint] = []
        self.fixed_joints: List[UsdPhysics.FixedJoint] = []
        self.revolute_joints: List[UsdPhysics.RevoluteJoint] = []
        self.prismatic_joints: List[UsdPhysics.PrismaticJoint] = []
        self.spherical_joints: List[UsdPhysics.SphericalJoint] = []
        self._collect_joints()
        
        self.links: List[Usd.Prim] = []
        self._collect_links()
    
    @staticmethod
    def find_first_articulation_root(stage: Usd.Stage) -> Usd.Prim:
        """
        Find the first articulation root in the stage.
        
        Parameters
        ----------
        stage : Usd.Stage
            The USD stage.
            
        Returns
        -------
        Usd.Prim or None
            The first articulation root prim, or None if not found.
        """
        default_prim = stage.GetDefaultPrim()
        if default_prim:
            for prim in bfs_iterator(default_prim):
                if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                    return prim
        return None
    
    @staticmethod
    def find_all_articulation_roots(stage: Usd.Stage, context: UsdParserContext = None) -> List[Usd.Prim]:
        """
        Find all prims with ArticulationRootAPI in the stage.
        
        Parameters
        ----------
        stage : Usd.Stage
            The USD stage.
        context : UsdParserContext, optional
            If provided, articulation roots will be added to the context.
            
        Returns
        -------
        List[Usd.Prim]
            List of articulation root prims.
        """
        articulation_roots = []
        for prim in bfs_iterator(stage.GetPseudoRoot()):
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                articulation_roots.append(prim)
                if context:
                    context.add_articulation_root(prim)
        return articulation_roots
    
    def _collect_joints(self):
        """Collect all joints in the articulation."""
        for child in bfs_iterator(self._root):
            if child.IsA(UsdPhysics.Joint):
                joint_api = UsdPhysics.Joint(child)
                self.joints.append(joint_api)
                if child.IsA(UsdPhysics.RevoluteJoint):
                    revolute_joint_api = UsdPhysics.RevoluteJoint(child)
                    self.revolute_joints.append(revolute_joint_api)
                elif child.IsA(UsdPhysics.FixedJoint):
                    fixed_joint_api = UsdPhysics.FixedJoint(child)
                    self.fixed_joints.append(fixed_joint_api)
                elif child.IsA(UsdPhysics.PrismaticJoint):
                    prismatic_joint_api = UsdPhysics.PrismaticJoint(child)
                    self.prismatic_joints.append(prismatic_joint_api)
                elif child.IsA(UsdPhysics.SphericalJoint):
                    spherical_joint_api = UsdPhysics.SphericalJoint(child)
                    self.spherical_joints.append(spherical_joint_api)
    
    def _collect_links(self):
        """Collect all links connected by joints in the articulation."""
        # Now we have joints collected, we can find links connected by these joints
        paths = set()
        for joint in self.joints:
            body0_targets = joint.GetBody0Rel().GetTargets()
            body1_targets = joint.GetBody1Rel().GetTargets()
            for target_path in body0_targets + body1_targets:
                # Check target is valid
                if self._stage.GetPrimAtPath(target_path):
                    paths.add(target_path)
                else:
                    gs.raise_exception(
                        f"Joint {joint.GetPath()} has invalid target body reference {target_path}."
                    )
        for path in paths:
            prim = self._stage.GetPrimAtPath(path)
            self.links.append(prim)
    
    @staticmethod
    def get_meshes(prim: Usd.Prim) -> List[UsdGeom.Mesh]:
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
    
    @staticmethod
    def get_visual_meshes(link: Usd.Prim) -> List[UsdGeom.Mesh]:
        """
        Get all visual meshes under the given link prim.
        
        Parameters
        ----------
        link : Usd.Prim
            The link prim to search under.
            
        Returns
        -------
        List[UsdGeom.Mesh]
            List of visual meshes found.
        """
        meshes: List[UsdGeom.Mesh] = []
        
        # Check if this prim is a mesh
        if link.IsA(UsdGeom.Mesh):
            meshes.append(UsdGeom.Mesh(link))
        
        # Find any child name starting with "visual" or "Visual"
        visual_pattern = re.compile(r'^(visual|Visual).*')
        visual_prims: list[Usd.Prim] = []
        for child in link.GetChildren():
            if visual_pattern.match(child.GetName()):
                visual_prims.append(child)
        
        for visual_prim in visual_prims:
            meshes.extend(UsdArticulationParser.get_meshes(visual_prim))
        return meshes
    
    @staticmethod
    def get_collision_meshes(link: Usd.Prim) -> List[UsdGeom.Mesh]:
        """
        Get all collision meshes under the given link prim.
        
        Parameters
        ----------
        link : Usd.Prim
            The link prim to search under.
            
        Returns
        -------
        List[UsdGeom.Mesh]
            List of collision meshes found.
        """
        meshes: List[UsdGeom.Mesh] = []
        
        # Check if this prim is a mesh
        if link.IsA(UsdGeom.Mesh):
            meshes.append(UsdGeom.Mesh(link))
        
        # Find any child name starting with "collision" or "Collision"
        collision_pattern = re.compile(r'^(collision|Collision).*')
        collision_prims: list[Usd.Prim] = []
        for child in link.GetChildren():
            if collision_pattern.match(child.GetName()):
                collision_prims.append(child)
        
        for collision_prim in collision_prims:
            meshes.extend(UsdArticulationParser.get_meshes(collision_prim))
        return meshes


def parse_usd_articulation(morph: gs.morphs.USDArticulation, surface: gs.surfaces.Surface):
    """
    Parse USD articulation from the given USD file and prim path, translating it into genesis structures.
    
    Returns
    -------
    l_infos : list
        List of link info dictionaries.
    links_j_infos : list
        List of lists of joint info dictionaries.
    links_g_infos : list
        List of lists of geometry info dictionaries.
    eqs_info : list
        List of equality constraint info dictionaries.
    """
    
    if morph.scale:
        morph.scale = 1.0
    
    assert morph.scale == 1.0, "Currently we only support scale=1.0 for USD articulation parsing."
    
    assert morph.parser_ctx is not None, "USDArticulation must have a parser context."
    
    context:UsdParserContext = morph.parser_ctx
    stage: Usd.Stage = context.stage
    
    assert morph.prim_path is not None, "USDArticulation must have a prim path."
    root_prim: Usd.Prim = stage.GetPrimAtPath(Sdf.Path(morph.prim_path))
    assert root_prim.IsValid(), f"Invalid prim path {morph.prim_path} in USD file {morph.file}."
    
    robot = UsdArticulationParser(stage, root_prim)
    
    
    link_name_to_idx = dict()
    for idx, link in enumerate(robot.links):
        link_name = link.GetPath()
        link_name_to_idx[link_name] = idx
    
    n_links = len(robot.links)
    
    l_infos = [dict() for _ in range(n_links)]
    links_j_infos = [[] for _ in range(n_links)]
    links_g_infos = [[] for _ in range(n_links)]
    
    for link, l_info, link_g_infos in zip(robot.links, l_infos, links_g_infos):
        l_info["name"] = link.GetPath()
        
        # No parent by default. It will be overwritten latter on if appropriate.
        l_info["parent_idx"] = -1
        
        # placeholder for pos and quat, will be updated when parsing joints
        l_info["pos"] = gu.zero_pos()
        l_info["quat"] = gu.identity_quat()
        
        # we compute urdf's invweight later
        l_info["invweight"] = np.full((2,), fill_value=-1.0)
        
        # leave the inertial info empty for now
        l_info["inertial_pos"] = gu.zero_pos()
        l_info["inertial_quat"] = gu.identity_quat()
        l_info["inertial_i"] = None
        l_info["inertial_mass"] = None
        
        # collect geometry info
        visual_meshes = UsdArticulationParser.get_visual_meshes(link)
        collision_meshes = UsdArticulationParser.get_collision_meshes(link)
        
        # if not (len(visual_meshes) > 0 or len(collision_meshes) > 0):
           # gs.logger.warning(f"Link {link.GetPath()} has no visual or collision meshes.")
        
        assert morph.scale == 1.0, "Currently we only support scale=1.0 for USD articulation parsing."
        
        if(len(visual_meshes) == 0):
            visual_meshes = collision_meshes.copy()

        n_visuals = len(visual_meshes)
        # collect visual and collision meshes
        for i, usd_mesh in enumerate((*visual_meshes, *collision_meshes)):
            geom_is_col = i >= n_visuals
            geom_type = gs.GEOM_TYPE.MESH
            geom_data = None

            # Get Genesis transform and trimesh relative to link
            Q_rel, tmesh = usd_mesh_to_gs_trimesh(usd_mesh, link)
            
            # Get material for this mesh
            mesh_prim = usd_mesh.GetPrim()
            # Try to get material binding if not collision
            
            default_surface = gs.surfaces.Collision()
            default_surface.color = (1.0, 0.0, 1.0)
            
            mesh_material:gs.surfaces.Surface = None
            if geom_is_col:
                mesh_material = default_surface
            else:
                mesh_material = context.find_material(mesh_prim)
            
            mesh = gs.Mesh.from_trimesh(tmesh, 
                                        scale=morph.scale,
                                        surface=mesh_material,
                                        metadata={"mesh_path":
                                            f"{morph.file}::{usd_mesh.GetPath()}"
                                            }
                                        )
            
            g_info = {"mesh" if geom_is_col else "vmesh": mesh}
            
            g_info["type"] = geom_type
            g_info["data"] = geom_data
            g_info["pos"] = Q_rel[:3, 3]
            g_info["quat"] = gu.R_to_quat(Q_rel[:3, :3])
            g_info["contype"] = 1 if geom_is_col else 0
            g_info["conaffinity"] = 1 if geom_is_col else 0
            g_info["friction"] = gu.default_friction()
            g_info["sol_params"] = gu.default_solver_params()
            
            link_g_infos.append(g_info)

    #########################  non-base joints and links #########################
    for joint in robot.joints:
        # Get the child link for this joint
        body0_targets = joint.GetBody0Rel().GetTargets()
        parent_link = None
        
        if body0_targets and len(body0_targets) > 0:
            parent_link_path = body0_targets[0]
            parent_link = stage.GetPrimAtPath(parent_link_path)
        
        body1_targets = joint.GetBody1Rel().GetTargets()
        if not body1_targets:
            gs.raise_exception(
                f"Joint {joint.GetPath()} has no body1 target."
            )
        child_link_path = body1_targets[0]
        child_link = stage.GetPrimAtPath(child_link_path)
        
        # Find the child link index
        idx = link_name_to_idx.get(child_link_path)
        if idx is None:
            gs.raise_exception(
                f"Joint {joint.GetPath()} references unknown child link {child_link_path}."
            )
        
        l_info = l_infos[idx]
        if parent_link:
            # Compute Genesis transform relative to parent link (Q^i_j)
            # This uses Genesis tree structure, not USD tree structure
            trans_mat, _ = compute_gs_related_transform(child_link, parent_link)
        else:
            # For base links, compute Genesis global transform (Q^w)
            trans_mat, _ = compute_gs_global_transform(child_link)
            
        l_info["pos"] = trans_mat[:3, 3]
        l_info["quat"] = extract_quat_from_transform(trans_mat)
        
        j_info = dict()
        links_j_infos[idx].append(j_info)
        
        # Get joint name and basic properties
        joint_prim = joint.GetPrim()
        j_info["name"] = joint_prim.GetPath()
        
        # Convert joint position from USD world space to Genesis link local space
        # Use body0 (parent) as reference for joint position
        if parent_link:
            j_info["pos"] = convert_usd_joint_pos_to_gs_link_space(joint_prim, parent_link, child_link)
        else:
            # For base joints, convert from world space
            j_info["pos"] = convert_usd_joint_pos_to_gs_link_space(joint_prim, child_link, child_link)
        j_info["quat"] = gu.identity_quat()
        
        # Get parent link
        body0_targets = joint.GetBody0Rel().GetTargets()
        if body0_targets:
            parent_link_path = body0_targets[0]
            parent_idx = link_name_to_idx.get(parent_link_path)
            if parent_idx is not None:
                l_info["parent_idx"] = parent_idx
        
        
        if joint_prim.IsA(UsdPhysics.RevoluteJoint):
            revolute_joint = UsdPhysics.RevoluteJoint(joint_prim)
            
            # Get joint axis string
            axis_attr = revolute_joint.GetAxisAttr()
            axis_str = axis_attr.Get() if axis_attr else "X"
            
            # Convert joint axis from USD world space to Genesis link local space
            # Use body0 (parent) as reference for joint axis definition
            if parent_link:
                axis = convert_usd_joint_axis_to_gs_link_space(joint_prim, parent_link, axis_str, child_link)
            else:
                # For base joints, use child link as reference
                axis = convert_usd_joint_axis_to_gs_link_space(joint_prim, child_link, axis_str, child_link)
            
            # Normalize the axis (should already be normalized, but ensure it)
            axis = axis / np.linalg.norm(axis) if np.linalg.norm(axis) > 1e-10 else axis
            
            j_info["dofs_motion_ang"] = np.array([axis])
            j_info["dofs_motion_vel"] = np.zeros((1, 3))
            
            # Get joint limits (angle limits are preserved under proportional scaling)
            lower_limit_attr = revolute_joint.GetLowerLimitAttr()
            upper_limit_attr = revolute_joint.GetUpperLimitAttr()
            lower_limit = lower_limit_attr.Get() if lower_limit_attr else -np.inf
            upper_limit = upper_limit_attr.Get() if upper_limit_attr else np.inf
            lower_limit = np.deg2rad(lower_limit)
            upper_limit = np.deg2rad(upper_limit)
            
            j_info["dofs_limit"] = np.array([[lower_limit, upper_limit]])
            j_info["dofs_stiffness"] = np.array([0.0])
            
            j_info["type"] = gs.JOINT_TYPE.REVOLUTE
            j_info["n_qs"] = 1
            j_info["n_dofs"] = 1
            j_info["init_qpos"] = np.zeros(1)
        elif joint_prim.IsA(UsdPhysics.PrismaticJoint):
            prismatic_joint = UsdPhysics.PrismaticJoint(joint_prim)
            
            # Get joint axis string
            axis_attr = prismatic_joint.GetAxisAttr()
            axis_str = axis_attr.Get() if axis_attr else "X"
            
            # Convert joint axis from USD world space to Genesis link local space
            # Use body0 (parent) as reference for joint axis definition
            if parent_link:
                axis = convert_usd_joint_axis_to_gs_link_space(joint_prim, parent_link, axis_str, child_link)
            else:
                # For base joints, use child link as reference
                axis = convert_usd_joint_axis_to_gs_link_space(joint_prim, child_link, axis_str, child_link)
            
            # Normalize the axis (should already be normalized, but ensure it)
            axis = axis / np.linalg.norm(axis) if np.linalg.norm(axis) > 1e-10 else axis
            
            # Prismatic joints use dofs_motion_vel (linear motion) instead of dofs_motion_ang
            j_info["dofs_motion_ang"] = np.zeros((1, 3))
            j_info["dofs_motion_vel"] = np.array([axis])
            
            # Get joint limits (in linear units, not degrees)
            lower_limit_attr = prismatic_joint.GetLowerLimitAttr()
            upper_limit_attr = prismatic_joint.GetUpperLimitAttr()
            lower_limit = lower_limit_attr.Get() if lower_limit_attr else -np.inf
            upper_limit = upper_limit_attr.Get() if upper_limit_attr else np.inf
            
            # Apply distance limit scaling: β = ||axis^w||
            # The distance limit should be scaled by β
            if parent_link:
                beta = compute_joint_axis_scaling_factor(joint_prim, parent_link, axis_str)
            else:
                beta = compute_joint_axis_scaling_factor(joint_prim, child_link, axis_str)
            
            # Scale limits by β
            lower_limit = lower_limit * beta if lower_limit != -np.inf else -np.inf
            upper_limit = upper_limit * beta if upper_limit != np.inf else np.inf
            
            j_info["dofs_limit"] = np.array([[lower_limit, upper_limit]])
            j_info["dofs_stiffness"] = np.array([0.0])
            
            j_info["type"] = gs.JOINT_TYPE.PRISMATIC
            j_info["n_qs"] = 1
            j_info["n_dofs"] = 1
            j_info["init_qpos"] = np.zeros(1)
        elif joint_prim.IsA(UsdPhysics.SphericalJoint):
            spherical_joint = UsdPhysics.SphericalJoint(joint_prim)
            
            # Spherical joints have 3 DOF (rotation around all 3 axes)
            # No axis needed - can rotate around all axes
            j_info["dofs_motion_ang"] = np.eye(3)  # Identity matrix for 3 rotational axes
            j_info["dofs_motion_vel"] = np.zeros((3, 3))
            
            # Spherical joints typically don't have simple limits
            # If limits exist, they would be complex (cone limits), which we don't support yet
            # For now, set unlimited range
            j_info["dofs_limit"] = np.tile([-np.inf, np.inf], (3, 1))
            j_info["dofs_stiffness"] = np.zeros(3)
            
            j_info["type"] = gs.JOINT_TYPE.SPHERICAL
            j_info["n_qs"] = 4  # Quaternion representation
            j_info["n_dofs"] = 3  # 3 rotational DOF
            j_info["init_qpos"] = gu.identity_quat()  # Initial quaternion
        else:
            # Parse joint type and properties
            if not joint_prim.IsA(UsdPhysics.FixedJoint):
                gs.logger.warning(f"Unsupported USD joint type: <{joint_prim.GetTypeName()}> in joint {joint_prim.GetPath()}. Treating as fixed joint.")
            
            if not body0_targets:
                gs.logger.info(f"Root Fixed Joint detected {joint_prim.GetPath()}")
            else:
                gs.logger.info(f"Fixed Joint detected {joint_prim.GetPath()}")
            j_info["dofs_motion_ang"] = np.zeros((0, 3))
            j_info["dofs_motion_vel"] = np.zeros((0, 3))
            j_info["dofs_limit"] = np.zeros((0, 2))
            j_info["dofs_stiffness"] = np.zeros((0))
            
            j_info["type"] = gs.JOINT_TYPE.FIXED
            j_info["n_qs"] = 0
            j_info["n_dofs"] = 0
            j_info["init_qpos"] = np.zeros(0)
                    
        # Common joint properties
        j_info["sol_params"] = gu.default_solver_params()
        j_info["dofs_invweight"] = np.full((j_info["n_dofs"],), fill_value=-1.0)
        
        # Joint friction and damping
        joint_friction = 0.0
        joint_damping = 0.0
        
        j_info["dofs_frictionloss"] = np.full(j_info["n_dofs"], joint_friction)
        j_info["dofs_damping"] = np.full(j_info["n_dofs"], joint_damping)
        j_info["dofs_armature"] = np.zeros(j_info["n_dofs"])
        
        # Default control gains
        j_info["dofs_kp"] = gu.default_dofs_kp(j_info["n_dofs"])
        j_info["dofs_kv"] = gu.default_dofs_kv(j_info["n_dofs"])
        
        # Force limits
        j_info["dofs_force_range"] = np.tile([-np.inf, np.inf], (j_info["n_dofs"], 1))
    
    # Add FREE joint to base links that have no incoming joints (unless morph is fixed)
    # This prevents base links from being incorrectly marked as fixed to world
    morph_fixed = getattr(morph, 'fixed', False)
    if not morph_fixed:
        for idx, (l_info, link_j_infos) in enumerate(zip(l_infos, links_j_infos)):
            # Base link (no parent) with no joints should get a FREE joint
            if l_info["parent_idx"] == -1 and len(link_j_infos) == 0:
                j_info = dict()
                j_info["name"] = f"{l_info['name']}_root_joint"
                j_info["type"] = gs.JOINT_TYPE.FREE
                j_info["n_qs"] = 7
                j_info["n_dofs"] = 6
                j_info["init_qpos"] = np.concatenate([l_info["pos"], l_info["quat"]])
                j_info["pos"] = gu.zero_pos()
                j_info["quat"] = gu.identity_quat()
                j_info["dofs_motion_ang"] = np.eye(6, 3, -3)
                j_info["dofs_motion_vel"] = np.eye(6, 3)
                j_info["dofs_limit"] = np.tile([-np.inf, np.inf], (6, 1))
                j_info["dofs_stiffness"] = np.zeros(6)
                j_info["dofs_invweight"] = np.zeros(6)
                j_info["dofs_frictionloss"] = np.zeros(6)
                j_info["dofs_damping"] = np.zeros(6)
                j_info["dofs_armature"] = np.zeros(6)
                j_info["dofs_kp"] = np.zeros((6,), dtype=gs.np_float)
                j_info["dofs_kv"] = np.zeros((6,), dtype=gs.np_float)
                j_info["dofs_force_range"] = np.tile([-np.inf, np.inf], (6, 1))
                j_info["sol_params"] = gu.default_solver_params()
                link_j_infos.append(j_info)
    
    # Apply scaling factor
    for l_info, link_j_infos, link_g_infos in zip(l_infos, links_j_infos, links_g_infos):
        l_info["pos"] *= morph.scale
        l_info["inertial_pos"] *= morph.scale
        
        if l_info["inertial_mass"] is not None:
            l_info["inertial_mass"] *= morph.scale**3
        if l_info["inertial_i"] is not None:
            l_info["inertial_i"] *= morph.scale**5
        
        for j_info in link_j_infos:
            j_info["pos"] *= morph.scale
        
        for g_info in link_g_infos:
            g_info["pos"] *= morph.scale
    
    # Post Process
    # Re-order kinematic tree info
    from .. import urdf as urdf_utils
    l_infos, links_j_infos, links_g_infos, _ = urdf_utils._order_links(l_infos, links_j_infos, links_g_infos)
    
    # For now, no equalities
    eqs_info = []
    
    return l_infos, links_j_infos, links_g_infos, eqs_info