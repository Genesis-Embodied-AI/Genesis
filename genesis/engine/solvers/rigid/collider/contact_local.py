"""
Thread-local versions of contact helper functions.

This module provides thread-local versions of contact-related helper functions
that accept pos/quat as direct parameters instead of reading from geoms_state.
"""

import gstaichi as ti

import genesis as gs
from genesis.utils import geom_utils as gu


@ti.func
def func_rotate_frame_local(
    pos: ti.types.vector(3, dtype=gs.ti_float),
    quat: ti.types.vector(4, dtype=gs.ti_float),
    contact_pos: ti.types.vector(3, dtype=gs.ti_float),
    qrot: ti.types.vector(4, dtype=gs.ti_float),
) -> ti.types.struct(
    pos=ti.types.vector(3, dtype=gs.ti_float),
    quat=ti.types.vector(4, dtype=gs.ti_float),
):
    """
    Thread-local version of func_rotate_frame.

    Instead of modifying geoms_state in place, this function takes thread-local
    pos/quat and returns the updated values.

    Args:
        pos: Thread-local geometry position
        quat: Thread-local geometry quaternion
        contact_pos: Contact position to rotate around
        qrot: Rotation quaternion for perturbation

    Returns:
        Struct containing updated pos and quat
    """
    # Update quaternion
    new_quat = gu.ti_transform_quat_by_quat(quat, qrot)

    # Update position
    rel = contact_pos - pos
    vec = gu.ti_transform_by_quat(rel, qrot)
    vec = vec - rel
    new_pos = pos - vec

    # Return struct with both updated values
    return ti.Struct(pos=new_pos, quat=new_quat)
