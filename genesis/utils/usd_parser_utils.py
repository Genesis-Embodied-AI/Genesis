from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf
from typing import List
import genesis as gs
import numpy as np
import trimesh
import re

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

class UsdParserContext:
    """
    A context class for USD Parsing, can be pass as arguments to various usd entity parser
    """
    def __init__(self, stage:Usd.Stage):
        self._stage = stage
    
    @property
    def stage(self) -> Usd.Stage:
        return self._stage
    pass