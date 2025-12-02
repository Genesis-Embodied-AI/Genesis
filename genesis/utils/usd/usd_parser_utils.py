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

def compute_usd_global_transform(prim: Usd.Prim) -> np.ndarray:
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

def compute_usd_related_transform(prim: Usd.Prim, ref_prim: Usd.Prim = None) -> np.ndarray:
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
    prim_world_transform = compute_usd_global_transform(prim)
    if ref_prim is None:
        return prim_world_transform
    ref_prim_to_world = compute_usd_global_transform(ref_prim)
    world_to_ref_prim = np.linalg.inv(ref_prim_to_world)
    prim_to_ref_prim_transform = world_to_ref_prim @ prim_world_transform
    return prim_to_ref_prim_transform

def usd_quat_to_numpy(usd_quat: Gf.Quatf) -> np.ndarray:
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

def usd_mesh_to_gs_trimesh(usd_mesh: UsdGeom.Mesh, ref_prim: Usd.Prim = None) -> tuple[np.ndarray, trimesh.Trimesh]:
    """
    Convert a USD mesh to a trimesh mesh and compute its Genesis transform relative to ref_prim.
    
    Parameters
    ----------
    usd_mesh : UsdGeom.Mesh
        The USD mesh to convert.
    ref_prim : Usd.Prim, optional
        The reference prim to compute the transform relative to. If None, regard as the world frame.
        
    Returns
    -------
    tuple[np.ndarray, trimesh.Trimesh]
        A tuple of (Q, trimesh) where:
        - Q: np.ndarray, shape (4, 4) - The Genesis transformation matrix (rotation and translation)
          relative to ref_prim. This is the Q transform without scaling.
        - S: np.ndarray, shape (3,) - The scaling factors extracted from the prim's USD global transform.
        - trimesh: trimesh.Trimesh - The converted trimesh object with scaling applied to vertices.
    """
    
    # Compute Genesis transform relative to ref_prim (Q^i_j)
    Q_rel, S = compute_gs_related_transform(usd_mesh.GetPrim(), ref_prim)
    
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
    tmesh = trimesh.Trimesh(vertices=points, faces=faces)
    
    return Q_rel, tmesh

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
        Quaternion as numpy array [w, x, y, z].
    """
    R, _ = extract_rotation_and_scale(trans_matrix)
    quat = gu.R_to_quat(R)
    return quat

def compute_gs_global_transform(prim: Usd.Prim) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Genesis global transform (Q^w) from USD prim.
    This extracts the rigid transform (rotation + translation) without scaling.
    
    In Genesis, transforms are Q (rotation R + translation t), while USD uses T (R + t + scaling S).
    The relationship is: T^w = Q^w · S in world space.
    
    Parameters
    ----------
    prim : Usd.Prim
        The prim to get the Genesis global transform for.
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple of (Q, S) where:
        - Q: np.ndarray, shape (4, 4) - The Genesis global transformation matrix Q^w (without scaling).
        - S: np.ndarray, shape (3,) - The scaling factors extracted from the prim's USD transform.
    """
    # Get USD global transform T^w (with scaling)
    T_w = compute_usd_global_transform(prim)
    
    # Extract rotation R and scale S from T^w
    R, S = extract_rotation_and_scale(T_w[:3, :3])
    
    # Build Genesis transform Q^w = [R | t; 0 | 1] (no scaling)
    Q_w = np.eye(4)
    Q_w[:3, :3] = R
    Q_w[:3, 3] = T_w[:3, 3]  # Translation is preserved
    
    return Q_w, S

def compute_gs_related_transform(prim: Usd.Prim, ref_prim: Usd.Prim = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Genesis transform (Q^i_j) relative to a reference prim.
    This computes the transform in Genesis tree structure (without scaling).
    
    Parameters
    ----------
    prim : Usd.Prim
        The prim to get the transform for.
    ref_prim : Usd.Prim, optional
        The reference prim (parent link). If None, returns global transform.
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple of (Q, S) where:
        - Q: np.ndarray, shape (4, 4) - The Genesis transformation matrix Q^i_j relative to ref_prim.
        - S: np.ndarray, shape (3,) - The scaling factors extracted from the prim's USD transform.
    """

    # Get Genesis global transforms
    Q_w_prim, S_prim = compute_gs_global_transform(prim)
    
    if ref_prim is None:
        return Q_w_prim, S_prim
    
    Q_w_ref, _ = compute_gs_global_transform(ref_prim)
    
    # Compute relative transform: Q^i_j = (Q^w_i)^(-1) · Q^w_j
    Q_w_ref_inv = np.linalg.inv(Q_w_ref)
    Q_i_j = Q_w_ref_inv @ Q_w_prim
    
    return Q_i_j, S_prim

def convert_usd_joint_axis_to_gs_link_space(
    joint_prim: Usd.Prim,
    body_prim: Usd.Prim,
    axis_str: str,
    gs_link_prim: Usd.Prim
) -> np.ndarray:
    """
    Convert USD joint axis from USD world space to Genesis link local space.
    
    The joint axis is defined in USD body local space. We convert it to world space,
    then to Genesis link local space.
    
    Parameters
    ----------
    joint_prim : Usd.Prim
        The joint prim.
    body_prim : Usd.Prim
        The body (link) prim that defines the joint axis local space.
    axis_str : str
        The axis string ('X', 'Y', or 'Z').
    gs_link_prim : Usd.Prim
        The Genesis link prim to convert the axis to.
        
    Returns
    -------
    np.ndarray, shape (3,)
        The axis vector in Genesis link local space.
    """
    # Get axis vector in body local space (axis_str defines axis in body0's local space)
    if axis_str == "X":
        axis_local = np.array([1.0, 0.0, 0.0])
    elif axis_str == "Y":
        axis_local = np.array([0.0, 1.0, 0.0])
    elif axis_str == "Z":
        axis_local = np.array([0.0, 0.0, 1.0])
    else:
        gs.raise_exception(f"Unsupported joint axis {axis_str}.")
    
    # Apply joint rotation if available (LocalRot0Attr rotates the axis in body0's local space)
    from pxr import UsdPhysics
    joint_api = UsdPhysics.Joint(joint_prim)
    if joint_api.GetLocalRot0Attr():
        quat = usd_quat_to_numpy(joint_api.GetLocalRot0Attr().Get())
        R_joint = gu.quat_to_R(quat)
        axis_local = R_joint @ axis_local
    
    # Transform axis to world space: axis^w = T^w_body · axis
    # The axis is already in body0's local space (after LocalRot0Attr), so transform directly by body0's global transform
    T_w_body = compute_usd_global_transform(body_prim)
    axis_world = T_w_body[:3, :3] @ axis_local
    
    # Convert to Genesis link local space: axis^0' = (Q^w_0')^(-1) · axis^w
    Q_w_gs_link, _ = compute_gs_global_transform(gs_link_prim)
    Q_w_gs_link_inv = np.linalg.inv(Q_w_gs_link)
    axis_gs_link = Q_w_gs_link_inv[:3, :3] @ axis_world
    
    return axis_gs_link

def convert_usd_joint_pos_to_gs_link_space(
    joint_prim: Usd.Prim,
    body_prim: Usd.Prim,
    gs_link_prim: Usd.Prim
) -> np.ndarray:
    """
    Convert USD joint position from USD world space to Genesis link local space.
    
    Parameters
    ----------
    joint_prim : Usd.Prim
        The joint prim.
    body_prim : Usd.Prim
        The body (link) prim that defines the joint position local space.
    gs_link_prim : Usd.Prim
        The Genesis link prim to convert the position to.
        
    Returns
    -------
    np.ndarray, shape (3,)
        The position in Genesis link local space.
    """
    from pxr import UsdPhysics
    joint_api = UsdPhysics.Joint(joint_prim)
    
    # Get joint position in body local space (LocalPos0Attr is relative to body0)
    # LocalPos0Attr defines the joint frame origin in body0's local space
    if joint_api.GetLocalPos0Attr():
        pos_local = np.array(joint_api.GetLocalPos0Attr().Get(), dtype=np.float64)
    else:
        pos_local = gu.zero_pos()
    
    # Transform position to world space: P^w = T^w_body · P
    # LocalPos0Attr is already in body0's local space, so we just need to transform by body0's global transform
    T_w_body = compute_usd_global_transform(body_prim)
    pos_world = T_w_body[:3, :3] @ pos_local + T_w_body[:3, 3]
    
    # Convert to Genesis link local space: P^0' = (Q^w_0')^(-1) · (P^w - t^w_0')
    Q_w_gs_link, _ = compute_gs_global_transform(gs_link_prim)
    Q_w_gs_link_inv = np.linalg.inv(Q_w_gs_link)
    pos_gs_link = Q_w_gs_link_inv[:3, :3] @ (pos_world - Q_w_gs_link[:3, 3])
    
    return pos_gs_link

def compute_joint_axis_scaling_factor(
    joint_prim: Usd.Prim,
    body_prim: Usd.Prim,
    axis_str: str
) -> float:
    """
    Compute the scaling factor for joint axis (for distance limit scaling).
    
    The scaling factor β = ||axis^w|| = α||axis||, where ||axis|| = 1 by definition.
    Under proportional scaling, β = α = ||axis^w||.
    
    Parameters
    ----------
    joint_prim : Usd.Prim
        The joint prim.
    body_prim : Usd.Prim
        The body (link) prim.
    axis_str : str
        The axis string ('X', 'Y', or 'Z').
        
    Returns
    -------
    float
        The scaling factor β.
    """
    # Get axis vector in body local space (axis_str defines axis in body0's local space)
    if axis_str == "X":
        axis_local = np.array([1.0, 0.0, 0.0])
    elif axis_str == "Y":
        axis_local = np.array([0.0, 1.0, 0.0])
    elif axis_str == "Z":
        axis_local = np.array([0.0, 0.0, 1.0])
    else:
        gs.raise_exception(f"Unsupported joint axis {axis_str}.")
    
    # Apply joint rotation if available (LocalRot0Attr rotates the axis in body0's local space)
    from pxr import UsdPhysics
    joint_api = UsdPhysics.Joint(joint_prim)
    if joint_api.GetLocalRot0Attr():
        quat = usd_quat_to_numpy(joint_api.GetLocalRot0Attr().Get())
        R_joint = gu.quat_to_R(quat)
        axis_local = R_joint @ axis_local
    
    # Transform axis to world space: axis^w = T^w_body · axis
    # The axis is already in body0's local space (after LocalRot0Attr), so transform directly by body0's global transform
    T_w_body = compute_usd_global_transform(body_prim)
    axis_world = T_w_body[:3, :3] @ axis_local
    
    # Compute scaling factor: β = ||axis^w||
    beta = np.linalg.norm(axis_world)
    
    return beta