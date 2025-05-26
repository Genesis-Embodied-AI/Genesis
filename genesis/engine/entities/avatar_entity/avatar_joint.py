import taichi as ti

from ..rigid_entity import RigidJoint


@ti.data_oriented
class AvatarJoint(RigidJoint):
    """AvatarJoint resembles RigidJoint in rigid_solver, but is only used for collision checking."""

    pass
