"""
USD Parser Utilities

Utility functions for USD parsing, including transform conversions,
mesh conversions, and other helper functions.
"""

from pxr import Usd, UsdGeom, Gf
from typing import List
import genesis as gs
import numpy as np
import trimesh
from collections import deque
from .. import geom as gu

def bfs_iterator(root: Usd.Prim):
    """
    Breadth-first iterator over USD prims.
    
    Parameters
    ----------
    root : Usd.Prim
        Root prim to start iteration from.
        
    Yields
    ------
    Usd.Prim
        Prims in breadth-first order.
    """
    queue = deque([root])
    while queue:
        prim = queue.popleft()
        yield prim
        for child in prim.GetChildren():
            queue.append(child)

def compute_global_transform(prim: Usd.Prim) -> np.ndarray:
    """
    Convert a USD transform to a 4x4 numpy transformation matrix.
    
    Parameters
    ----------
    prim : Usd.Prim
        The prim to get the global transform for.
        
    Returns
    -------
    np.ndarray, shape (4, 4)
        The global transformation matrix.
    """
    imageable = UsdGeom.Imageable(prim)
    if not imageable:
        return np.eye(4)
    # USD's transform is left-multiplied, while we use right-multiplied convention in genesis.
    t = imageable.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetTranspose()
    return np.array(t)

def compute_related_transform(prim: Usd.Prim, ref_prim: Usd.Prim = None) -> np.ndarray:
    """
    Compute the transformation matrix from the reference prim to the prim.
    
    Parameters
    ----------
    prim : Usd.Prim
        The prim to get the transform for.
    ref_prim : Usd.Prim
        The reference prim (transform will be relative to this).
        
    Returns
    -------
    np.ndarray, shape (4, 4)
        The transformation matrix relative to ref_prim.
    """
    prim_world_transform = compute_global_transform(prim)
    if ref_prim is None:
        return prim_world_transform
    ref_prim_to_world = compute_global_transform(ref_prim)
    world_to_ref_prim = np.linalg.inv(ref_prim_to_world)
    prim_to_ref_prim_transform = world_to_ref_prim @ prim_world_transform
    return prim_to_ref_prim_transform

def gf_quat_to_numpy(usd_quat: Gf.Quatf) -> np.ndarray:
    """
    Convert a USD Gf.Quatf to a numpy array (w, x, y, z format).
    
    Parameters
    ----------
    usd_quat : Gf.Quatf
        The USD quaternion.
        
    Returns
    -------
    np.ndarray, shape (4,)
        Quaternion as numpy array [w, x, y, z].
    """
    return np.array([usd_quat.GetReal(), *usd_quat.GetImaginary()])

def extract_rotation_and_scale(trans_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract rotation R and scale S from a 3x3 matrix using SVD.
    
    Parameters
    ----------
    matrix_3x3 : np.ndarray, shape (3, 3)
        The 3x3 transformation matrix.
        
    Returns
    -------
    R : np.ndarray, shape (3, 3)
        Rotation matrix.
    S : np.ndarray, shape (3,)
        Scale factors as diagonal.
    """
    # SVD to get rotation and scale
    U, S, Vh = np.linalg.svd(trans_matrix[:3, :3])
    
    # Check if reflection is present
    if np.linalg.det(U @ Vh) < 0:
        Vh[-1, :] *= -1
    R = U @ Vh
    scale = np.diag(S)
    return R, scale

def usd_mesh_to_trimesh(usd_mesh: UsdGeom.Mesh, ref_prim: Usd.Prim = None) -> trimesh.Trimesh:
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
    
    # genesis mesh transform only has pos/quat
    # the scaling from usd mesh should be applied to the points
    global_transform = compute_global_transform(usd_mesh.GetPrim())
    _R, S = extract_rotation_and_scale(global_transform[:3, :3])
    
    points_attr = usd_mesh.GetPointsAttr()
    face_vertex_counts_attr = usd_mesh.GetFaceVertexCountsAttr()
    face_vertex_indices_attr = usd_mesh.GetFaceVertexIndicesAttr()
    
    points = np.array(points_attr.Get())
    # Apply only scaling to every point
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

def extract_quat_from_transform(trans_matrix: np.ndarray) -> np.ndarray:
    """
    Extract quaternion from a 4x4 transformation matrix.
    
    Parameters
    ----------
    trans_matrix : np.ndarray, shape (4, 4) or (3, 3)
        The transformation matrix.
        
    Returns
    -------
    np.ndarray, shape (4,)
    """
    R,S = extract_rotation_and_scale(trans_matrix)
    quat = gu.R_to_quat(R)
    return quat

def apply_transform_to_pos(trans_matrix: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """
    Apply a transformation matrix to a position.
    
    Parameters
    ----------
    trans_matrix : np.ndarray, shape (4, 4) or (3, 3)
    pos : np.ndarray, shape (3,)
        The position to apply the transformation to.
        
    Returns
    -------
    np.ndarray, shape (3,)
        The transformed position.
    """
    return trans_matrix[:3, :3] @ pos + trans_matrix[:3, 3]