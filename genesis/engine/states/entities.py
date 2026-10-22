import genesis as gs
from genesis.repr_base import RBC


class ToolEntityState:
    """
    Dynamic state queried from a genesis ToolEntity.
    """

    def __init__(self, entity, s_global):
        self.entity = entity
        self.s_global = s_global

        self.pos = gs.zeros((self.entity.sim._B, 3), dtype=gs.tc_float)
        self.quat = gs.zeros((self.entity.sim._B, 4), dtype=gs.tc_float)
        self.vel = gs.zeros((self.entity.sim._B, 3), dtype=gs.tc_float)
        self.ang = gs.zeros((self.entity.sim._B, 3), dtype=gs.tc_float)
        self.entity.sim.track_state(self.pos, self.quat, self.vel, self.ang)

    # def __repr__(self):
    #     return f'{self.__repr_name__()}\n' \
    #            f'entity : {_repr(self.entity)}\n' \
    #            f'pos    : {_repr(self.pos)}\n' \
    #            f'quat   : {_repr(self.quat)}\n' \
    #            f'vel    : {_repr(self.vel)}\n' \
    #            f'ang    : {_repr(self.ang)}'


class MPMEntityState(RBC):
    """
    Dynamic state queried from a genesis MPMEntity.
    """

    def __init__(self, entity, s_global):
        self._entity = entity
        self._s_global = s_global
        base_shape = (self.entity.sim._B, self._entity.n_particles)
        self._pos = gs.zeros(base_shape + (3,), dtype=gs.tc_float)
        self._vel = gs.zeros(base_shape + (3,), dtype=gs.tc_float)
        self._C = gs.zeros(base_shape + (3, 3), dtype=gs.tc_float)
        self._F = gs.zeros(base_shape + (3, 3), dtype=gs.tc_float)
        self._Jp = gs.zeros(base_shape, dtype=gs.tc_float)
        self._active = gs.zeros(base_shape, dtype=gs.tc_int)
        self.entity.sim.track_state(self._pos, self._vel, self._C, self._F, self._Jp)

    @property
    def entity(self):
        return self._entity

    @property
    def s_global(self):
        return self._s_global

    @property
    def pos(self):
        return self._pos

    @property
    def vel(self):
        return self._vel

    @property
    def C(self):
        return self._C

    @property
    def F(self):
        return self._F

    @property
    def Jp(self):
        return self._Jp

    @property
    def active(self):
        return self._active


class SPHEntityState(RBC):
    """
    Dynamic state queried from a genesis SPHEntity.
    """

    def __init__(self, entity, s_global):
        self._entity = entity
        self._s_global = s_global
        base_shape = (self.entity.sim._B, self._entity.n_particles)
        self._pos = gs.zeros(base_shape + (3,), dtype=gs.tc_float)
        self._vel = gs.zeros(base_shape + (3,), dtype=gs.tc_float)

    @property
    def entity(self):
        return self._entity

    @property
    def s_global(self):
        return self._s_global

    @property
    def pos(self):
        return self._pos

    @property
    def vel(self):
        return self._vel


class FEMEntityState:
    """
    Dynamic state queried from a genesis FEMEntity.
    """

    def __init__(self, entity, s_global):
        self._entity = entity
        self._s_global = s_global
        base_shape = (self.entity.sim._B, self._entity.n_vertices, 3)

        self._pos = gs.zeros(base_shape, dtype=gs.tc_float)
        self._vel = gs.zeros(base_shape, dtype=gs.tc_float)
        self._active = gs.zeros((self.entity.sim._B, self.entity.n_elements), dtype=gs.tc_int)

    @property
    def entity(self):
        return self._entity

    @property
    def s_global(self):
        return self._s_global

    @property
    def pos(self):
        return self._pos

    @property
    def vel(self):
        return self._vel

    @property
    def active(self):
        return self._active


class RigidEntityState(RBC):
    """
    Dynamic state queried from a genesis RigidEntity.
    """

    def __init__(self, entity, s_global, pos, quat):
        self._entity = entity
        self._s_global = s_global
        self._pos = pos
        self._quat = quat

    @property
    def entity(self):
        return self._entity

    @property
    def s_global(self):
        return self._s_global

    @property
    def pos(self):
        return self._pos

    @property
    def quat(self):
        return self._quat
