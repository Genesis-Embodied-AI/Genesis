from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf
from typing import List
import genesis as gs
import numpy as np
import trimesh
import re
import scipy

class UsdParserContext:
    """
    A context class for USD Parsing, can be pass as arguments to various usd entity parser
    """
    def __init__(self, stage:Usd.Stage):
        self._stage = stage
        self._materials: dict[str, tuple[gs.surfaces.Surface, str]] = {}  # material_id -> (material_surface, uv_name)
    
    @property
    def stage(self) -> Usd.Stage:
        return self._stage
    
    @property
    def materials(self) -> dict:
        """
        Get the parsed materials dictionary.
        Key: material_id (str)
        Value: tuple of (material_surface, uv_name)
        """
        return self._materials
    
    def get_material(self, material_id: str):
        """
        Get a parsed material by its ID.
        
        Parameters
        ----------
        material_id : str
            The material ID.
        
        Returns
        -------
        tuple or None
            Tuple of (material_surface, uv_name) if found, None otherwise.
        """
        return self._materials.get(material_id)

def bfs_iterator(root:Usd.Prim):
    from collections import deque
    queue = deque([root])
    while queue:
        prim = queue.popleft()
        yield prim
        for child in prim.GetChildren():
            queue.append(child)

def compute_global_transform(prim:Usd.Prim) -> np.ndarray:
    """
    Convert a USD transform to a 4x4 numpy transformation matrix.
    """
    imageable = UsdGeom.Imageable(prim)
    if not imageable:
        return np.eye(4)
    # USD's transform is left-multiplied, while we use right-multiplied convention in genesis.
    t = imageable.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetTranspose()
    return np.array(t)

def compute_related_transform(prim:Usd.Prim, ref_prim:Usd.Prim) -> np.ndarray:
    """
    Compute the transformation matrix from the related_prim to the prim.
    """
    prim_world_transform = compute_global_transform(prim)
    ref_prim_to_world = compute_global_transform(ref_prim)
    world_to_ref_prim = np.linalg.inv(ref_prim_to_world)
    prim_to_ref_prim_transform = world_to_ref_prim @ prim_world_transform
    return prim_to_ref_prim_transform

def usd_quat_to_np(usd_quat:Gf.Quatf) -> np.ndarray:
    """
    Convert a USD Gf.Quatf to a numpy array.
    """
    return np.array([usd_quat.GetReal(), *usd_quat.GetImaginary()])

def extract_rotation_and_scale(global_transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    U, S, Vh = np.linalg.svd(global_transform[:3, :3])
    # check if reflection is present
    if np.linalg.det(U @ Vh) < 0:
        Vh[-1, :] *= -1
    R = U @ Vh
    scale = np.diag(S)
    return R, scale
    

def usd_mesh_to_trimesh(usd_mesh: UsdGeom.Mesh) -> trimesh.Trimesh:
    """
    Convert a USD mesh to a trimesh mesh.
    
    Parameters
    ----------
    usd_mesh : UsdGeom.Mesh
        The USD mesh to convert.
    
    Returns
    -------
    trimesh.Trimesh
        The converted trimesh object.
    """
    
    global_transform = compute_global_transform(usd_mesh.GetPrim())
    R, S = extract_rotation_and_scale(global_transform)
    
    points_attr = usd_mesh.GetPointsAttr()
    
    face_vertex_counts_attr = usd_mesh.GetFaceVertexCountsAttr()
    face_vertex_indices_attr = usd_mesh.GetFaceVertexIndicesAttr()
    
    points = np.array(points_attr.Get())
    # apply scaling to points
    points = points @ S
    face_vertex_counts = np.array(face_vertex_counts_attr.Get())
    face_vertex_indices = np.array(face_vertex_indices_attr.Get())
    faces = []
    
    offset = 0
    has_non_tri_quads = False
    for i, count in enumerate(face_vertex_counts):
        face_vertex_counts[i] = count
        if count == 3:
            # Triangle - use directly
            faces.append(face_vertex_indices[offset:offset+count])
        elif count == 4:
            # Quad - split into two triangles
            quad = face_vertex_indices[offset:offset+count]
            faces.append([quad[0], quad[1], quad[2]])
            faces.append([quad[0], quad[2], quad[3]])
        elif count > 4:
            # Polygon with more than 4 vertices - triangulate using triangle fan
            # Use the first vertex as the fan center and connect to each pair of consecutive vertices
            polygon = face_vertex_indices[offset:offset+count]
            for j in range(1, count - 1):
                faces.append([polygon[0], polygon[j], polygon[j + 1]])
            has_non_tri_quads = True
        else:
            # Invalid face (count < 3)
            gs.logger.warning(
                f"Invalid face vertex count {count} in USD mesh {usd_mesh.GetPath()}. Skipping face."
            )
        offset += count
    
    if has_non_tri_quads:
        gs.logger.info(
            f"USD mesh {usd_mesh.GetPath()} contains polygons with more than 4 vertices. Triangulated using triangle fan method."
        )
    faces = np.array(faces)
    return trimesh.Trimesh(vertices=points, faces=faces)