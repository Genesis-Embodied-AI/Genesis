from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf
from typing import List
                            
import genesis as gs
from . import geom as gu
from . import usda
import numpy as np
import trimesh
import re
from . import usd_parser_utils as usd_utils

# Design Summary

# - Genesis Agnostic: The parser is agnostic to genesis structures, it only focuses on USD structure.
#   - UsdArticulationParser: parse articulation information from a USD stage.
#   - UsdRigidBodyParser: parse rigid body information from a USD stage.

# - Genesis Specific: The parser returns info structures to genesis
#   - parse_usd_articulation: parse USD articulation from the given USD file (or Context) and prim path.
#   - parse_usd_rigid_body: parse USD rigid body from the given USD file (or Context) and prim path.

class UsdArticulationParser:
    """
    A parser to extract articulation information from a USD stage.
    The Parser is agnostic to genesis structures, it only focuses on USD articulation structure.
    """
    def __init__(self, stage:Usd.Stage, articulation_root_prim:Usd.Prim):
        self._stage:Usd.Stage = stage
        self._root:Usd.Prim = articulation_root_prim
        if not articulation_root_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            gs.raise_exception(
                f"Provided prim {articulation_root_prim.GetPath()} is not an Articulation Root. Now we only support articulation parsing from ArticulationRootAPI."
            )
        
        self.joints:List[UsdPhysics.Joint] = []
        self.fixed_joints:List[UsdPhysics.FixedJoint] = []
        self.revolute_joints:List[UsdPhysics.RevoluteJoint] = []
        self.prismatic_joints:List[UsdPhysics.PrismaticJoint] = []
        self.spherical_joints:List[UsdPhysics.SphericalJoint] = []
        self._collect_joints()
        
        self.links:List[Usd.Prim] = []
        self._collect_links()
    
    @staticmethod
    def find_first_articulation_root(stage:Usd.Stage) -> Usd.Prim:
        for prim in usd_utils.bfs_iterator(stage.GetDefaultPrim()):
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                return prim
        return None
    
    @staticmethod
    def find_all_articulation_roots(stage: Usd.Stage) -> List[Usd.Prim]:
        """
        Find all prims with ArticulationRootAPI in the stage.
        """
        articulation_roots = []
        for prim in usd_utils.bfs_iterator(stage.GetPseudoRoot()):
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                articulation_roots.append(prim)
        return articulation_roots
    
    def _collect_joints(self):
        for child in usd_utils.bfs_iterator(self._root):
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
        # now we have joints collected, we can find links connected by these joints
        paths = set()
        for joint in self.joints:
            body0_targets = joint.GetBody0Rel().GetTargets()
            body1_targets = joint.GetBody1Rel().GetTargets()
            for target_path in body0_targets + body1_targets:
                # check target is valid
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
    def get_meshes(prim:Usd.Prim) -> List[UsdGeom.Mesh]:
        """
        Get all meshes under the articulation root prim.
        """
        meshes:List[UsdGeom.Mesh] = []
        
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            meshes.append(mesh)
        
        for prim in usd_utils.bfs_iterator(prim):
            if prim.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(prim)
                meshes.append(mesh)
        
        return meshes
    
    @staticmethod
    def get_visual_meshes(link:Usd.Prim) -> List[UsdGeom.Mesh]:
        """
        Get all meshes under the given link prim.
        """
        # find any child name starting with "visual" or "Visual"
        visual_pattern = re.compile(r'^(visual|Visual).*')
        visual_prims:list[Usd.Prim] = []
        for child in link.GetChildren():
            child:Usd.Prim
            if visual_pattern.match(child.GetName()):
                visual_prims.append(child)
        
        meshes:List[UsdGeom.Mesh] = []
        for visual_prim in visual_prims:
            meshes.extend(UsdArticulationParser.get_meshes(visual_prim))
        return meshes
    
    @staticmethod
    def get_collision_meshes(link:Usd.Prim) -> List[UsdGeom.Mesh]:
        """
        Get all collision meshes under the given link prim.
        """
        # find any child name starting with "collision" or "Collision"
        collision_pattern = re.compile(r'^(collision|Collision).*')
        collision_prims:list[Usd.Prim] = []
        for child in link.GetChildren():
            child:Usd.Prim
            if collision_pattern.match(child.GetName()):
                collision_prims.append(child)
        
        meshes:List[UsdGeom.Mesh] = []
        for collision_prim in collision_prims:
            meshes.extend(UsdArticulationParser.get_meshes(collision_prim))
        return meshes
    

class UsdRigidBodyParser:
    """
    A parser to extract rigid body information from a USD stage.
    The Parser is agnostic to genesis structures, it only focuses on USD rigid body structure.
    """
    def __init__(self, stage:Usd.Stage, rigid_body_prim:Usd.Prim):
        self._stage:Usd.Stage = stage
        self._root:Usd.Prim = rigid_body_prim
        
        is_rigid_body = UsdRigidBodyParser.regard_as_rigid_body(rigid_body_prim)
        if not is_rigid_body:
            gs.raise_exception(
                f"Provided prim {rigid_body_prim.GetPath()} is not a rigid body, APIs found: {rigid_body_prim.GetAppliedSchemas()}"
            )
        self.is_fixed = rigid_body_prim.HasAPI(UsdPhysics.CollisionAPI) and not rigid_body_prim.HasAPI(UsdPhysics.RigidBodyAPI)
        
    
    @property
    def stage(self) -> Usd.Stage:
        """Get the USD stage."""
        return self._stage
    
    @property
    def rigid_body_prim(self) -> Usd.Prim:
        """Get the rigid body prim."""
        return self._root
    
    @staticmethod
    def _get_meshes(prim: Usd.Prim) -> List[UsdGeom.Mesh]:
        """
        Get all meshes under the given prim.
        """
        meshes: List[UsdGeom.Mesh] = []
        
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            meshes.append(mesh)
        
        for child_prim in usd_utils.bfs_iterator(prim):
            if child_prim.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(child_prim)
                meshes.append(mesh)
        
        return meshes
    
    def get_visual_meshes(self) -> List[UsdGeom.Mesh]:
        """
        Get all visual meshes under the rigid body prim.
        """
        # find any child name starting with "visual" or "Visual"
        visual_pattern = re.compile(r'^(visual|Visual).*')
        visual_prims: list[Usd.Prim] = []
        for child in self._root.GetChildren():
            if visual_pattern.match(child.GetName()):
                visual_prims.append(child)
        
        meshes: List[UsdGeom.Mesh] = []
        for visual_prim in visual_prims:
            meshes.extend(self._get_meshes(visual_prim))
        return meshes
    
    def get_collision_meshes(self) -> List[UsdGeom.Mesh]:
        """
        Get all collision meshes under the rigid body prim.
        """
        # find any child name starting with "collision" or "Collision"
        collision_pattern = re.compile(r'^(collision|Collision).*')
        collision_prims: list[Usd.Prim] = []
        for child in self._root.GetChildren():
            if collision_pattern.match(child.GetName()):
                collision_prims.append(child)
        
        meshes: List[UsdGeom.Mesh] = []
        for collision_prim in collision_prims:
            meshes.extend(self._get_meshes(collision_prim))
        return meshes
    
    def get_all_meshes(self) -> List[UsdGeom.Mesh]:
        """
        Get all meshes under the rigid body prim.
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
    
    def get_global_transform(self) -> np.ndarray:
        """
        Get the global transform matrix of the rigid body.
        
        Returns
        -------
        transform : np.ndarray, shape (4, 4)
            The global transform matrix.
        """
        return usd_utils.compute_global_transform(self._root)
    
    def get_mesh_transform(self, mesh: UsdGeom.Mesh) -> np.ndarray:
        """
        Get the transform matrix of a mesh relative to the rigid body.
        
        Parameters
        ----------
        mesh : UsdGeom.Mesh
            The mesh to get the transform for.
        
        Returns
        -------
        transform : np.ndarray, shape (4, 4)
            The transform matrix relative to the rigid body.
        """
        return usd_utils.compute_related_transform(mesh, self._root)
    
    def get_material_binding(self, mesh_prim: Usd.Prim) -> UsdShade.Material:
        """
        Get the material binding for a mesh prim.
        
        Parameters
        ----------
        mesh_prim : Usd.Prim
            The mesh prim to get the material binding for.
        
        Returns
        -------
        material : UsdShade.Material or None
            The bound material, or None if no binding exists.
        """
        if not mesh_prim.HasRelationship("material:binding"):
            return None
        
        if not mesh_prim.HasAPI(UsdShade.MaterialBindingAPI):
            UsdShade.MaterialBindingAPI.Apply(mesh_prim)
        
        prim_bindings = UsdShade.MaterialBindingAPI(mesh_prim)
        material_usd = prim_bindings.ComputeBoundMaterial()[0]
        
        if material_usd.GetPrim().IsValid():
            return material_usd
        return None
    
    def get_material_id(self, material: UsdShade.Material) -> str:
        """
        Get a unique identifier for a material.
        
        Parameters
        ----------
        material : UsdShade.Material
            The material to get the ID for.
        
        Returns
        -------
        material_id : str
            A unique identifier for the material.
        """
        material_spec = material.GetPrim().GetPrimStack()[-1]
        return material_spec.layer.identifier + material_spec.path.pathString
    
    @staticmethod
    def regard_as_rigid_body(prim: Usd.Prim) -> bool:
        """
        Check if a prim should be regarded as a rigid body.
        """
        
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            return False
        
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return True
        
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            return True
        
        return False
    
    @staticmethod
    def find_all_rigid_bodies(stage: Usd.Stage) -> List[Usd.Prim]:
        """
        Find all top-level prims with RigidBodyAPI that are not part of an articulation.
        A rigid body is considered top-level if it's not a descendant of any articulation root.
        """
        # First, collect all articulation roots and their descendants
        articulation_roots = UsdArticulationParser.find_all_articulation_roots(stage)
        articulation_descendants = set()
        for root in articulation_roots:
            for prim in usd_utils.bfs_iterator(root):
                articulation_descendants.add(prim.GetPath())
        
        # Find all rigid bodies that are not part of any articulation
        rigid_bodies = []
        for prim in usd_utils.bfs_iterator(stage.GetPseudoRoot()):
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
        
        return rigid_bodies
    
    @staticmethod
    def find_first_rigid_body(stage: Usd.Stage) -> Usd.Prim:
        """
        Find the first prim with RigidBodyAPI that is not part of an articulation.
        """
        rigid_bodies = UsdRigidBodyParser.find_all_rigid_bodies(stage)
        return rigid_bodies[0] if rigid_bodies else None


def parse_usd_articulation(morph:gs.morphs.USDArticulation, surface:gs.surfaces.Surface) -> None:
    """
    Parse USD articulation from the given USD file and prim path, translating it into genesis structures.
    """
    
    if morph.scale:
        morph.scale = 1.0
    
    assert morph.scale == 1.0, "Currently we only support scale=1.0 for USD articulation parsing."
    
    stage:Usd.Stage = morph.parser_ctx.stage if morph.parser_ctx else Usd.Stage.Open(morph.file)
    
    if morph.prim_path:
        root_prim = stage.GetPrimAtPath(Sdf.Path(morph.prim_path))
    else:
        root_prim = UsdArticulationParser.find_first_articulation_root(stage)
        
    if not root_prim:
        gs.raise_exception(
            f"Cannot find articulation root prim in USD file {morph.file}."
        )
    
    robot = UsdArticulationParser(stage, root_prim)
    
    # Get materials from parser context if available
    if morph.parser_ctx:
        materials = morph.parser_ctx.materials
    else:
        # Fallback: empty materials dict if no context provided
        materials = {}
    
    link_name_to_idx = dict()
    for idx, link in enumerate(robot.links):
        link_name = link.GetPath()
        link_name_to_idx[link_name] = idx
    
    n_links = len(robot.links)
    n_joints = len(robot.joints)
    
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
        n_visuals = len(visual_meshes)
        collision_meshes = UsdArticulationParser.get_collision_meshes(link)
        
        assert morph.scale == 1.0, "Currently we only support scale=1.0 for USD articulation parsing."

        # collect visual and collision meshes
        for i, usd_mesh in enumerate((*visual_meshes, *collision_meshes)):
            geom_is_col = i >= n_visuals
            geom_type = gs.GEOM_TYPE.MESH
            geom_data = None
            tmesh = usd_utils.usd_mesh_to_trimesh(usd_mesh)
            
            # Get material for this mesh
            mesh_prim = usd_mesh.GetPrim()
            mesh_material = surface.copy()
            
            # Try to get material binding if not collision
            if not geom_is_col:
                # Check for material binding
                if mesh_prim.HasRelationship("material:binding"):
                    if not mesh_prim.HasAPI(UsdShade.MaterialBindingAPI):
                        UsdShade.MaterialBindingAPI.Apply(mesh_prim)
                    prim_bindings = UsdShade.MaterialBindingAPI(mesh_prim)
                    material_usd = prim_bindings.ComputeBoundMaterial()[0]
                    if material_usd.GetPrim().IsValid():
                        material_spec = material_usd.GetPrim().GetPrimStack()[-1]
                        material_id = material_spec.layer.identifier + material_spec.path.pathString
                        material_result = materials.get(material_id)
                        if material_result is not None:
                            mesh_material, _ = material_result
            
            mesh = gs.Mesh.from_trimesh(tmesh, 
                                        scale=morph.scale,
                                        surface=gs.surfaces.Collision() if geom_is_col else mesh_material,
                                        metadata={"mesh_path":
                                            f"{morph.file}::{usd_mesh.GetPath()}"
                                            }
                                        )
            
            g_info = {"mesh" if geom_is_col else "vmesh": mesh}
            
            trans_mat = usd_utils.compute_related_transform(usd_mesh, link)
            g_info["type"] = geom_type
            g_info["data"] = geom_data
            g_info["pos"] = trans_mat[:3, 3]
            g_info["quat"] = gu.R_to_quat(trans_mat[:3, :3])
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
            trans_mat = usd_utils.compute_related_transform(child_link, parent_link)
        else:
            trans_mat = usd_utils.compute_global_transform(child_link)
        l_info["pos"] = trans_mat[:3, 3]
        l_info["quat"] = gu.R_to_quat(trans_mat[:3, :3])
        
        j_info = dict()
        links_j_infos[idx].append(j_info)
        
        # Get joint name and basic properties
        joint_prim = joint.GetPrim()
        j_info["name"] = joint_prim.GetPath()
        j_info["pos"] = np.array(joint.GetLocalPos1Attr().Get(), dtype=np.float64) if joint.GetLocalPos1Attr() else gu.zero_pos()
        j_info["quat"] = gu.identity_quat()
        # j_info["quat"] = gu.identity_quat()
        
        # Get parent link
        body0_targets = joint.GetBody0Rel().GetTargets()
        if body0_targets:
            parent_link_path = body0_targets[0]
            parent_idx = link_name_to_idx.get(parent_link_path)
            if parent_idx is not None:
                l_info["parent_idx"] = parent_idx
        
        # Parse joint type and properties
        if joint_prim.IsA(UsdPhysics.FixedJoint):
            if not body0_targets:
                gs.logger.info(f"Root Fixed Joint detected {joint_prim.GetPath()}")
            j_info["dofs_motion_ang"] = np.zeros((0, 3))
            j_info["dofs_motion_vel"] = np.zeros((0, 3))
            j_info["dofs_limit"] = np.zeros((0, 2))
            j_info["dofs_stiffness"] = np.zeros((0))
            
            j_info["type"] = gs.JOINT_TYPE.FIXED
            j_info["n_qs"] = 0
            j_info["n_dofs"] = 0
            j_info["init_qpos"] = np.zeros(0)
            
        elif joint_prim.IsA(UsdPhysics.RevoluteJoint):
            revolute_joint = UsdPhysics.RevoluteJoint(joint_prim)
            
            # Get joint axis
            axis_attr = revolute_joint.GetAxisAttr()
            axis_str = axis_attr.Get() if axis_attr else "X"
            if axis_str == "X":
                axis = np.array([1.0, 0.0, 0.0])
            elif axis_str == "Y":
                axis = np.array([0.0, 1.0, 0.0])
            elif axis_str == "Z":
                axis = np.array([0.0, 0.0, 1.0])
            else:
                gs.raise_exception(
                    f"Unsupported joint axis {axis_str} in USD revolute joint {joint_prim.GetPath()}."
                )
            
            quat=usd_utils.usd_quat_to_np(joint.GetLocalRot0Attr().Get()) if joint.GetLocalRot0Attr() else gu.identity_quat()
            R_joint = gu.quat_to_R(quat)
            axis = R_joint @ axis
            
            j_info["dofs_motion_ang"] = np.array([axis])
            j_info["dofs_motion_vel"] = np.zeros((1, 3))
            
            # Get joint limits
            lower_limit_attr = revolute_joint.GetLowerLimitAttr()
            upper_limit_attr = revolute_joint.GetUpperLimitAttr()
            lower_limit = lower_limit_attr.Get() if lower_limit_attr else -np.inf
            upper_limit = upper_limit_attr.Get() if upper_limit_attr else np.inf
            lower_limit = np.deg2rad(lower_limit)
            upper_limit = np.deg2rad(upper_limit)
            # gs.logger.info(f"Revolute joint {joint_prim.GetPath()} limits: [{lower_limit}, {upper_limit}]")
            
            j_info["dofs_limit"] = np.array([[lower_limit, upper_limit]])
            j_info["dofs_stiffness"] = np.array([0.0])
            
            j_info["type"] = gs.JOINT_TYPE.REVOLUTE
            j_info["n_qs"] = 1
            j_info["n_dofs"] = 1
            j_info["init_qpos"] = np.zeros(1)
        elif joint_prim.IsA(UsdPhysics.PrismaticJoint):
            gs.raise_exception(f"Unsupported USD joint type: {joint_prim.GetTypeName()}")
        elif joint_prim.IsA(UsdPhysics.SphericalJoint):
            gs.raise_exception(f"Unsupported USD joint type: {joint_prim.GetTypeName()}")
        else:
            gs.raise_exception(f"Unsupported USD joint type: {joint_prim.GetTypeName()}")
        
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
    
    # Re-order kinematic tree info
    from . import urdf as urdf_utils
    l_infos, links_j_infos, links_g_infos, _ = urdf_utils._order_links(l_infos, links_j_infos, links_g_infos)
    
    # For now, no equalities
    eqs_info = []
    
    return l_infos, links_j_infos, links_g_infos, eqs_info

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
    from scipy.spatial.transform import Rotation as R
    
    # Get scale (default to 1.0 for now, as USD rigid body parsing currently only supports scale=1.0)
    scale = getattr(morph, 'scale', 1.0)
    if scale != 1.0:
        gs.logger.warning("USD rigid body parsing currently only supports scale=1.0. Scale will be ignored.")
        scale = 1.0
    
    # Open stage and get rigid body prim
    stage: Usd.Stage = morph.parser_ctx.stage if morph.parser_ctx else Usd.Stage.Open(morph.file)
    
    if morph.prim_path:
        rigid_body_prim = stage.GetPrimAtPath(Sdf.Path(morph.prim_path))
        if not rigid_body_prim.IsValid():
            gs.raise_exception(
                f"Invalid prim path {morph.prim_path} in USD file {morph.file}."
            )
    else:
        rigid_body_prim = UsdRigidBodyParser.find_first_rigid_body(stage)
        if not rigid_body_prim:
            gs.raise_exception(
                f"Cannot find rigid body prim in USD file {morph.file}."
            )
    
    # Create parser for USD-agnostic extraction
    rigid_body = UsdRigidBodyParser(stage, rigid_body_prim)
    
    # Get visual and collision meshes using parser
    visual_meshes, collision_meshes = rigid_body.get_visual_and_collision_meshes()
    n_visuals = len(visual_meshes)
    
    # Get materials from parser context if available, otherwise parse them
    if morph.parser_ctx:
        materials = morph.parser_ctx.materials
    else:
        # Fallback: parse materials if no context provided
        materials = {}
        for prim in stage.Traverse():
            if prim.IsA(UsdShade.Material):
                material_usd = UsdShade.Material(prim)
                material_id = rigid_body.get_material_id(material_usd)
                if material_id not in materials:
                    material, uv_name, require_bake = usda.parse_usd_material(material_usd, surface)
                    materials[material_id] = (material, uv_name)
    
    # Get the global transform of the rigid body using parser
    global_transform = rigid_body.get_global_transform()
    body_pos = global_transform[:3, 3]
    body_quat = gu.R_to_quat(global_transform[:3, :3])
    
    # Apply scale and morph.pos/quat as offsets
    final_pos = body_pos * scale
    if hasattr(morph, 'pos') and morph.pos is not None:
        final_pos = final_pos + np.array(morph.pos)
    final_quat = body_quat.copy()
    if hasattr(morph, 'quat') and morph.quat is not None:
        # Multiply quaternions: result = morph.quat * final_quat
        r_morph = R.from_quat(np.array(morph.quat)[[1, 2, 3, 0]])  # wxyz to xyzw
        r_init = R.from_quat(final_quat[[1, 2, 3, 0]])  # wxyz to xyzw
        r_result = r_morph * r_init
        final_quat = np.array([r_result.as_quat()[3], *r_result.as_quat()[:3]])  # xyzw to wxyz
    
    # Determine joint type and init_qpos
    if rigid_body.is_fixed:
        joint_type = gs.JOINT_TYPE.FIXED
        n_qs = 0
        n_dofs = 0
        init_qpos = np.zeros(0)
    else:
        joint_type = gs.JOINT_TYPE.FREE
        n_qs = 7
        n_dofs = 6
        init_qpos = np.concatenate([final_pos, final_quat])
    
    # Build geometry infos
    g_infos = []
    
    # Process visual and collision meshes
    for i, usd_mesh in enumerate((*visual_meshes, *collision_meshes)):
        geom_is_col = i >= n_visuals
        
        # Convert USD mesh to trimesh (genesis-agnostic conversion)
        tmesh = usd_utils.usd_mesh_to_trimesh(usd_mesh)
        
        # Get material for this mesh using parser
        mesh_prim = usd_mesh.GetPrim()
        material = surface.copy()
        
        # Try to get material binding using parser
        material_usd = rigid_body.get_material_binding(mesh_prim)
        if material_usd is not None:
            material_id = rigid_body.get_material_id(material_usd)
            material_result = materials.get(material_id)
            if material_result is not None:
                material, _ = material_result
        
        # Create mesh (genesis-specific)
        mesh = gs.Mesh.from_trimesh(
            tmesh,
            scale=scale,
            surface=gs.surfaces.Collision() if geom_is_col else material,
            metadata={"mesh_path": f"{morph.file}::{usd_mesh.GetPath()}"}
        )
        
        # Get transform relative to the rigid body using parser
        trans_mat = rigid_body.get_mesh_transform(usd_mesh)
        geom_pos = trans_mat[:3, 3] * scale
        geom_quat = gu.R_to_quat(trans_mat[:3, :3])
        
        visualization = getattr(morph, 'visualization', True)
        collision = getattr(morph, 'collision', True)
        
        if visualization and not geom_is_col:
            g_infos.append(
                dict(
                    contype=0,
                    conaffinity=0,
                    vmesh=mesh,
                    pos=geom_pos,
                    quat=geom_quat,
                )
            )
        if collision and geom_is_col:
            g_infos.append(
                dict(
                    contype=getattr(morph, 'contype', 1),
                    conaffinity=getattr(morph, 'conaffinity', 1),
                    mesh=mesh,
                    type=gs.GEOM_TYPE.MESH,
                    sol_params=gu.default_solver_params(),
                    pos=geom_pos,
                    quat=geom_quat,
                )
            )
    
    # Generate link name from prim path
    link_name = str(rigid_body_prim.GetPath())
    # Create link info
    l_info = dict(
        is_robot=False,
        name=f"{link_name}",
        pos=final_pos,
        quat=final_quat,
        inertial_pos=None,  # we will compute the COM later based on the geometry
        inertial_quat=gu.identity_quat(),
        parent_idx=-1,
    )
    
    # Create joint info
    j_infos = [
        dict(
            name=f"{link_name}_joint", # we only have one joint for the rigid body
            n_qs=n_qs,
            n_dofs=n_dofs,
            type=joint_type,
            init_qpos=init_qpos,
        )
    ]
    
    return l_info, j_infos, g_infos