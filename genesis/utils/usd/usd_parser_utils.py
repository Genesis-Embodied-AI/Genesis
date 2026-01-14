"""
USD Parser Utilities

Utility functions for USD parsing, including transform conversions, mesh conversions, and other helper functions.

Reference: ./UsdParserSpec.md
"""

from collections import deque
from typing import Callable, List, Tuple, Literal

import numpy as np
import trimesh
from pxr import Gf, Usd, UsdGeom

import genesis as gs

from .. import geom as gu


def usd_quat_to_numpy(usd_quat: Gf.Quatf) -> np.ndarray:
    """
    Convert a USD Gf.Quatf to a numpy array (w, x, y, z) format.

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


def extract_rotation_and_scale(trans_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    R, S = gu.polar(trans_matrix[:3, :3], pure_rotation=True, side="right")
    assert np.linalg.det(R) > 0, "Rotation matrix must contain only pure rotations."
    return R, S


def usd_mesh_to_gs_trimesh(usd_mesh: UsdGeom.Mesh, ref_prim: Usd.Prim | None) -> Tuple[np.ndarray, trimesh.Trimesh]:
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
        - trimesh: trimesh.Trimesh - The converted trimesh object with scaling applied to vertices.
    """

    # Compute Genesis transform relative to ref_prim (Q^i_j)
    Q_rel, S = compute_gs_relative_transform(usd_mesh.GetPrim(), ref_prim)

    points_attr = usd_mesh.GetPointsAttr()
    face_vertex_counts_attr = usd_mesh.GetFaceVertexCountsAttr()
    face_vertex_indices_attr = usd_mesh.GetFaceVertexIndicesAttr()

    points = np.asarray(points_attr.Get())
    # Apply only scaling to every point
    points = points @ S
    face_vertex_counts = np.asarray(face_vertex_counts_attr.Get())
    face_vertex_indices = np.asarray(face_vertex_indices_attr.Get())
    faces = []

    offset = 0
    has_non_tri_quads = False
    for i, count in enumerate(face_vertex_counts):
        face_vertex_counts[i] = count
        if count == 3:
            # Triangle - use directly
            faces.append(face_vertex_indices[offset : offset + count])
        elif count == 4:
            # Quad - split into two triangles
            quad = face_vertex_indices[offset : offset + count]
            faces.append([quad[0], quad[1], quad[2]])
            faces.append([quad[0], quad[2], quad[3]])
        elif count > 4:
            # Polygon with more than 4 vertices - triangulate using triangle fan
            # Use the first vertex as the fan center and connect to each pair of consecutive vertices
            polygon = face_vertex_indices[offset : offset + count]
            for j in range(1, count - 1):
                faces.append([polygon[0], polygon[j], polygon[j + 1]])
            has_non_tri_quads = True
        else:
            # Invalid face (count < 3)
            gs.logger.warning(f"Invalid face vertex count {count} in USD mesh {usd_mesh.GetPath()}. Skipping face.")
        offset += count

    if has_non_tri_quads:
        gs.logger.info(
            f"USD mesh {usd_mesh.GetPath()} contains polygons with more than 4 vertices. Triangulated using triangle fan method."
        )
    faces = np.asarray(faces)
    tmesh = trimesh.Trimesh(vertices=points, faces=faces)
    return Q_rel, tmesh


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
    return np.asarray(imageable.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetTranspose())


def compute_usd_relative_transform(prim: Usd.Prim, ref_prim: Usd.Prim | None) -> np.ndarray:
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
    return world_to_ref_prim @ prim_world_transform


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


def compute_gs_relative_transform(prim: Usd.Prim, ref_prim: Usd.Prim | None) -> tuple[np.ndarray, np.ndarray]:
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


def compute_gs_joint_pos_from_usd_prim(usd_local_joint_pos: np.ndarray, usd_link_prim: Usd.Prim | None) -> np.ndarray:
    """
    Compute Genesis joint position from USD joint position in USD link local space.
    """
    T_w = compute_usd_global_transform(usd_link_prim)
    pos_w = T_w[:3, :3] @ usd_local_joint_pos + T_w[:3, 3]
    Q_w, _ = compute_gs_global_transform(usd_link_prim)
    Q_w_inv = np.linalg.inv(Q_w)
    return Q_w_inv[:3, :3] @ (pos_w - Q_w[:3, 3])


def compute_gs_joint_axis_and_pos_from_usd_prim(
    usd_local_joint_axis: np.ndarray, usd_local_joint_pos: np.ndarray, usd_link_prim: Usd.Prim | None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Genesis joint axis and position from USD joint axis and position in USD link local space.
    """
    T_w = compute_usd_global_transform(usd_link_prim)
    axis_w = T_w[:3, :3] @ usd_local_joint_axis
    pos_w = T_w[:3, :3] @ usd_local_joint_pos + T_w[:3, 3]
    Q_w, _ = compute_gs_global_transform(usd_link_prim)
    Q_w_inv = np.linalg.inv(Q_w)
    return Q_w_inv[:3, :3] @ axis_w, Q_w_inv[:3, :3] @ (pos_w - Q_w[:3, 3])
