import taichi as ti

from ..rigid_entity import RigidGeom, RigidVisGeom


@ti.data_oriented
class AvatarGeom(RigidGeom):
    """AvatarGeom resembles RigidGeom in rigid_solver, but is only used for collision checking."""

    pass


@ti.data_oriented
class AvatarVisGeom(RigidVisGeom):
    pass
