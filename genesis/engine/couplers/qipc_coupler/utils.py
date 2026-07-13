from __future__ import annotations

import numpy as np

import genesis as gs
import genesis.utils.geom as gu


def rotate_inertia_to_link_frame(inertia_i, inertial_quat):
    """Rotate inertia tensor from principal frame to link frame.

    Args:
        inertia_i: (3, 3) inertia tensor in principal frame.
        inertial_quat: (4,) quaternion (w, x, y, z) of principal frame relative to link frame.

    Returns:
        (3, 3) inertia tensor in link frame.
    """
    if inertia_i is None:
        gs.raise_exception("QIPCCoupler: link has no inertia data. All links must have valid inertial properties.")
    R = gu.quat_to_R(np.array(inertial_quat, dtype=np.float64))
    return R @ np.array(inertia_i, dtype=np.float64) @ R.T
