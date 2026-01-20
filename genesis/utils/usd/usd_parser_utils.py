"""
USD Parser Utilities

Utility functions for USD parsing, including transform conversions, mesh conversions, and other helper functions.

Reference: ./UsdParserSpec.md
"""
from typing import Callable, List, Tuple, Literal

import numpy as np
from pxr import Gf, Usd, UsdGeom

import genesis as gs

from .. import geom as gu


AXES_VECTOR = {
    "X": np.array([1, 0, 0], dtype=gs.np_float),
    "Y": np.array([0, 1, 0], dtype=gs.np_float),
    "Z": np.array([0, 0, 1], dtype=gs.np_float),
}


AXES_T = {
    "X": np.array(
        [[0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=gs.np_float
    ),
    "Y": np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=gs.np_float
    ),
    "Z": np.eye(4, dtype=gs.np_float),
}


def usd_pos_to_numpy(usd_pos: Gf.Vec3f) -> np.ndarray:
    if usd_pos is None:
        return gu.zero_pos()
    return np.asarray(usd_pos, dtype=gs.np_float)


def usd_quat_to_numpy(usd_quat: Gf.Quatf) -> np.ndarray:
    if usd_quat is None:
        return gu.identity_quat()
    return np.asarray([usd_quat.GetReal(), *usd_quat.GetImaginary()], dtype=gs.np_float)


def extract_scale(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    R, S = gu.polar(T[:3, :3], pure_rotation=True, side="right")
    assert np.linalg.det(R) > 0, "Rotation matrix must contain only pure rotations."
    Q = np.eye(4, dtype=T.dtype)
    Q[:3, :3] = R
    Q[:3, 3] = T[:3, 3]
    return Q, S


def get_attr_value_by_candidates(
    prim: Usd.Prim, candidates: List[str], attr_name: str, default_value: float
):
    for candidate in candidates:
        attr_value = prim.GetAttribute(candidate).Get()
        if attr_value is not None:
            return attr_value
    
    gs.logger.debug(
        f"No matching attribute `{attr_name}` found in {prim.GetPath()} "
        f"given candidates: {candidates}. Using default value: {default_value}."
    )
    return default_value
