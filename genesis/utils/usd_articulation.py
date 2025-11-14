from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf
from typing import List
                            
import genesis as gs
from . import geom as gu
import numpy as np
import trimesh

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
    def bfs_iterator(root:Usd.Prim):
        from collections import deque
        queue = deque([root])
        while queue:
            prim = queue.popleft()
            yield prim
            for child in prim.GetChildren():
                queue.append(child)
    
    @staticmethod
    def find_first_articulation_root(stage:Usd.Stage) -> Usd.Prim:
        for prim in UsdArticulationParser.bfs_iterator(stage.GetDefaultPrim()):
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                return prim
        return None
    
    def _collect_joints(self):
        for child in UsdArticulationParser.bfs_iterator(self._root):
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
        for prim in UsdArticulationParser.bfs_iterator(prim):
            if prim.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(prim)
                meshes.append(mesh)
        return meshes
    
    @staticmethod
    def get_visual_meshes(link:Usd.Prim) -> List[UsdGeom.Mesh]:
        """
        Get all meshes under the given link prim.
        """
        # TO DISCUSS: how to identify visual meshes under a link prim?
        # Now just hard code, assuming there is a Prim named "visuals" under each link
        visuals_prim = link.GetChild("visuals")
        return UsdArticulationParser.get_meshes(visuals_prim) if visuals_prim else []
    
    @staticmethod
    def get_collision_meshes(link:Usd.Prim) -> List[UsdGeom.Mesh]:
        """
        Get all collision meshes under the given link prim.
        """
        # TO DISCUSS: how to identify collision meshes under a link prim?
        # Now just hard code, assuming there is a Prim named "collisions" under each link
        collisions_prim = link.GetChild("collisions")
        return UsdArticulationParser.get_meshes(collisions_prim) if collisions_prim else []
    
    
    @staticmethod
    def usd_mesh_to_trimesh(usd_mesh:UsdGeom.Mesh) -> trimesh.Trimesh:
        """
        Convert a USD mesh to a genesis trimesh mesh.
        """
        points_attr = usd_mesh.GetPointsAttr()
        face_vertex_counts_attr = usd_mesh.GetFaceVertexCountsAttr()
        face_vertex_indices_attr = usd_mesh.GetFaceVertexIndicesAttr()
        
        points = np.array(points_attr.Get())
        face_vertex_counts = np.array(face_vertex_counts_attr.Get())
        face_vertex_indices = np.array(face_vertex_indices_attr.Get())
        faces = []
        
        offset = 0
        for i, count in enumerate(face_vertex_counts):
            face_vertex_counts[i] = count
            if count == 3:
                faces.append(face_vertex_indices[offset:offset+count])
            elif count == 4:
                quad = face_vertex_indices[offset:offset+count]
                faces.append([quad[0], quad[1], quad[2]])
                faces.append([quad[0], quad[2], quad[3]])
            else:
                gs.raise_exception(
                    f"Unsupported face vertex count {count} in USD mesh {usd_mesh.GetPath()}. Only triangles and quads are supported."
                )
            offset += count
        faces = np.array(faces)
        return trimesh.Trimesh(vertices=points, faces=faces)
    
    @staticmethod
    def usd_global_transform_to_np(prim:Usd.Prim) -> np.ndarray:
        """
        Convert a USD transform to a 4x4 numpy transformation matrix.
        """
        imageable = UsdGeom.Imageable(prim)
        if not imageable:
            return np.eye(4)
        # USD's transform is left-multiplied, while we use right-multiplied convention in genesis.
        t = imageable.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetTranspose()
        return np.array(t)
    
    @staticmethod
    def usd_compute_related_transform(prim:Usd.Prim, ref_prim:Usd.Prim) -> np.ndarray:
        """
        Compute the transformation matrix from the related_prim to the prim.
        """
        prim_world_transform = UsdArticulationParser.usd_global_transform_to_np(prim)
        ref_prim_to_world = UsdArticulationParser.usd_global_transform_to_np(ref_prim)
        world_to_ref_prim = np.linalg.inv(ref_prim_to_world)
        prim_to_ref_prim_transform = world_to_ref_prim @ prim_world_transform
        return prim_to_ref_prim_transform

    @staticmethod
    def usd_quat_to_np(usd_quat:Gf.Quatf) -> np.ndarray:
        """
        Convert a USD Gf.Quatf to a numpy array.
        """
        return np.array([usd_quat.GetReal(), *usd_quat.GetImaginary()])

def parse_usd(morph:gs.morphs.USDArticulation, surface:gs.surfaces.Surface) -> None:
    """
    Parse USD articulation from the given USD file and prim path, translating it into genesis structures.
    """
    
    if morph.scale:
        morph.scale = 1.0
    
    assert morph.scale == 1.0, "Currently we only support scale=1.0 for USD articulation parsing."
    
    stage = Usd.Stage.Open(morph.file)
    if morph.prim_path:
        root_prim = stage.GetPrimAtPath(Sdf.Path(morph.prim_path))
    else:
        root_prim = UsdArticulationParser.find_first_articulation_root(stage)
        
    if not root_prim:
        gs.raise_exception(
            f"Cannot find articulation root prim in USD file {morph.file}."
        )
    
    robot = UsdArticulationParser(stage, root_prim)
    
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
            tmesh = UsdArticulationParser.usd_mesh_to_trimesh(usd_mesh)
            mesh = gs.Mesh.from_trimesh(tmesh, 
                                        scale=morph.scale,
                                        surface=gs.surfaces.Collision() if geom_is_col else surface,
                                        metadata={"mesh_path":
                                            f"{morph.file}::{usd_mesh.GetPath()}"
                                            }
                                        )
            
            # TO FIX:
            # now just hard code a color for visualization
            mesh.set_color([1.0, 0.7, 0.3, 1.0])  # set a default color for visualization
            g_info = {"mesh" if geom_is_col else "vmesh": mesh}
            
            trans_mat = UsdArticulationParser.usd_compute_related_transform(usd_mesh, link)
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
            trans_mat = UsdArticulationParser.usd_compute_related_transform(child_link, parent_link)
        else:
            trans_mat = UsdArticulationParser.usd_global_transform_to_np(child_link)
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
            
            quat=UsdArticulationParser.usd_quat_to_np(joint.GetLocalRot0Attr().Get()) if joint.GetLocalRot0Attr() else gu.identity_quat()
            R_joint = gu.quat_to_R(quat)
            axis = R_joint @ axis
            
            j_info["dofs_motion_ang"] = np.array([axis])
            j_info["dofs_motion_vel"] = np.zeros((1, 3))
            
            # Get joint limits
            lower_limit_attr = revolute_joint.GetLowerLimitAttr()
            upper_limit_attr = revolute_joint.GetUpperLimitAttr()
            lower_limit = lower_limit_attr.Get() if lower_limit_attr else -np.inf
            upper_limit = upper_limit_attr.Get() if upper_limit_attr else np.inf
            
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
        # Try to get these from USD joint drive attributes if available
        joint_friction = 0.0
        joint_damping = 0.0
        
        # Check for drive API which may contain damping
        if joint_prim.HasAPI(UsdPhysics.DriveAPI):
            # Get drive for the specific axis
            if joint_prim.IsA(UsdPhysics.RevoluteJoint):
                drive = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
            elif joint_prim.IsA(UsdPhysics.PrismaticJoint):
                drive = UsdPhysics.DriveAPI.Apply(joint_prim, "linear")
            else:
                drive = None
            
            if drive:
                damping_attr = drive.GetDampingAttr()
                stiffness_attr = drive.GetStiffnessAttr()
                if damping_attr:
                    joint_damping = damping_attr.Get() or 0.0
                if stiffness_attr and j_info["n_dofs"] > 0:
                    stiffness_val = stiffness_attr.Get() or 0.0
                    j_info["dofs_stiffness"] = np.full(j_info["n_dofs"], stiffness_val)
        
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


if __name__ == "__main__":
    stage = Usd.Stage.Open("D:/MyStorage/Project/GenesisProject/Genesis/playground/assets/input_mesh_simple.usda")
    root_prim = stage.GetPrimAtPath(Sdf.Path("/World/orcahand_right"))
    parser = UsdArticulationParser(stage, root_prim)
    print(f"Found {len(parser.joints)} joints.")
    print(f"Found {len(parser.revolute_joints)} revolute joints.")
    print(f"Found {len(parser.fixed_joints)} fixed joints.")
    for link in parser.links:
        print(f"Link prim: {link.GetPath()}")
    