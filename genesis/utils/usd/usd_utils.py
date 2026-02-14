import math
from typing import List, Tuple

import numpy as np
from pxr import Gf, Usd, UsdGeom

import genesis as gs
from genesis.utils import geom as gu


AXES_VECTOR = {
    "X": np.array([1, 0, 0], dtype=np.float32),
    "Y": np.array([0, 1, 0], dtype=np.float32),
    "Z": np.array([0, 0, 1], dtype=np.float32),
}


AXES_T = {
    "X": np.array(
        [[0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float32
    ),
    "Y": np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float32
    ),
    "Z": np.eye(4, dtype=np.float32),
}


def usd_pos_to_numpy(usd_pos: Gf.Vec3f | None) -> np.ndarray:
    """
    Convert USD position to numpy array, handling None values.

    Parameters
    ----------
    usd_pos : Gf.Vec3f | None
        USD position attribute value. If None, returns zero vector.

    Returns
    -------
    np.ndarray
        Position as numpy array, or zero vector if input is None.
    """
    if usd_pos is None:
        return gu.zero_pos()
    return np.asarray(usd_pos, dtype=np.float32)


def usd_quat_to_numpy(usd_quat: Gf.Quatf | None) -> np.ndarray:
    """
    Convert USD quaternion to numpy array, handling None values.

    Parameters
    ----------
    usd_quat : Gf.Quatf | None
        USD quaternion attribute value. If None, returns identity quaternion.

    Returns
    -------
    np.ndarray
        Quaternion as numpy array, or identity quaternion if input is None.
    """
    if usd_quat is None:
        return gu.identity_quat()
    return np.asarray([usd_quat.GetReal(), *usd_quat.GetImaginary()], dtype=np.float32)


def usd_center_of_mass_to_numpy(usd_pos: Gf.Vec3f) -> np.ndarray | None:
    """
    Convert USD center of mass position to numpy array, handling invalid default values.

    The USD Physics MassAPI defines centerOfMass with default value (-inf, -inf, -inf),
    which is invalid and indicates that the center of mass should be computed from geometry.
    This function returns None for the default invalid value, allowing the system to
    recompute it from geometry.

    Parameters
    ----------
    usd_pos : Gf.Vec3f
        USD center of mass position attribute value.

    Returns
    -------
    np.ndarray | None
        Valid center of mass position as numpy array, or None if invalid/default.
    """
    pos = usd_pos_to_numpy(usd_pos)
    # Default invalid value is (-inf, -inf, -inf) - all negative infinity
    if np.all(np.isinf(pos) & (pos < 0)):
        return None
    return pos


def usd_principal_axes_to_numpy(usd_quat: Gf.Quatf) -> np.ndarray | None:
    """
    Convert USD principal axes quaternion to numpy array, handling invalid default values.

    The USD Physics MassAPI defines principalAxes with default value (0, 0, 0, 0),
    which is invalid (identity quaternion should be (1, 0, 0, 0)) and indicates that
    the principal axes should be computed from geometry. This function returns None
    for the default invalid value.

    Parameters
    ----------
    usd_quat : Gf.Quatf
        USD principal axes quaternion attribute value.

    Returns
    -------
    np.ndarray | None
        Valid principal axes quaternion as numpy array, or None if invalid/default.
    """
    quat = usd_quat_to_numpy(usd_quat)
    # Default invalid value is (0, 0, 0, 0) - identity quaternion should be (1, 0, 0, 0)
    if np.allclose(quat, [0, 0, 0, 0]):
        return None
    return quat


def usd_inertia_to_numpy(inertia: Gf.Vec3f) -> np.ndarray | None:
    """
    Convert USD diagonal inertia to numpy diagonal matrix, handling default and invalid values.

    The USD Physics MassAPI defines diagonalInertia with default value (0, 0, 0),
    which is valid but means the inertia should be ignored/computed from geometry.
    This function returns None for the default ignored value or invalid values (non-finite).

    Parameters
    ----------
    inertia : Gf.Vec3f
        USD diagonal inertia attribute value.

    Returns
    -------
    np.ndarray | None
        Valid diagonal inertia matrix (3x3), or None if default ignored or invalid.
    """
    diagonal_inertia = usd_diagonal_inertia_to_numpy(inertia)
    if diagonal_inertia is None:
        return None
    return np.diag(diagonal_inertia)


def usd_diagonal_inertia_to_numpy(usd_pos: Gf.Vec3f) -> np.ndarray | None:
    """
    Convert USD diagonal inertia to numpy array, handling default and invalid values.

    The USD Physics MassAPI defines diagonalInertia with default value (0, 0, 0),
    which is valid but means the inertia should be ignored/computed from geometry.
    This function returns None for the default ignored value or invalid values (negative or inf/nan).

    Parameters
    ----------
    usd_pos : Gf.Vec3f
        USD diagonal inertia attribute value.

    Returns
    -------
    np.ndarray | None
        Valid diagonal inertia as numpy array, or None if default ignored or invalid.
    """
    inertia = usd_pos_to_numpy(usd_pos)
    # Default is (0, 0, 0) which means ignored - only return if non-zero and valid
    if np.allclose(inertia, 0):
        return None
    # Check for invalid values (non-finite)
    if not all(math.isfinite(e) for e in inertia):
        return None
    return inertia


def usd_mass_to_float(usd_mass: float) -> float | None:
    """
    Convert USD mass to float, handling default and invalid values.

    The USD Physics MassAPI defines mass with default value 0, which is valid but
    means the mass should be ignored/computed from geometry. This function returns
    None for the default ignored value or invalid values (non-positive, inf, or nan).

    Parameters
    ----------
    usd_mass : float
        USD mass attribute value.

    Returns
    -------
    float | None
        Valid mass value, or None if default ignored or invalid.
    """
    # Default is 0 which means ignored
    if usd_mass == 0:
        return None
    # Check for invalid values (non-finite)
    if not math.isfinite(usd_mass):
        return None
    return float(usd_mass)


def usd_attr_array_to_numpy(attr: Usd.Attribute, dtype: np.dtype, return_none: bool = False) -> np.ndarray | None:
    if attr.HasValue():
        return np.array(attr.Get(), dtype=dtype)
    return None if return_none else np.empty(0, dtype=dtype)


def usd_primvar_array_to_numpy(
    primvar: UsdGeom.Primvar, dtype: np.dtype, return_none: bool = False
) -> np.ndarray | None:
    if primvar.IsDefined() and primvar.HasValue():
        return np.array(primvar.ComputeFlattened(), dtype=dtype)
    return None if return_none else np.empty(0, dtype=dtype)


def extract_scale(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    R, S = gu.polar(T[:3, :3], pure_rotation=True, side="right")
    if np.linalg.det(R) <= 0:
        gs.raise_exception(f"Negative determinant of rotation matrix detected. Got {np.linalg.det(R)}.")
    Q = np.eye(4, dtype=T.dtype)
    Q[:3, :3] = R
    Q[:3, 3] = T[:3, 3]
    return Q, S


def get_attr_value_by_candidates(prim: Usd.Prim, candidates: List[str], attr_name: str, default_value: float):
    for candidate in candidates:
        attr_value = prim.GetAttribute(candidate).Get()
        if attr_value is not None:
            return attr_value

    gs.logger.debug(
        f"No matching attribute `{attr_name}` found in {prim.GetPath()} "
        f"given candidates: {candidates}. Using default value: {default_value}."
    )
    return default_value
