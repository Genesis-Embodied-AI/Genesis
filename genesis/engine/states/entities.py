import genesis as gs
from genesis.repr_base import RBC


class ToolEntityState:
    """
    Dynamic state queried from a genesis ToolEntity.
    """

    def __init__(self, entity, s_global):
        self.entity = entity
        self.s_global = s_global

        self.pos = gs.zeros(3, dtype=float, requires_grad=self.entity.scene.requires_grad, scene=self.entity.scene)
        self.quat = gs.zeros(4, dtype=float, requires_grad=self.entity.scene.requires_grad, scene=self.entity.scene)
        self.vel = gs.zeros(3, dtype=float, requires_grad=self.entity.scene.requires_grad, scene=self.entity.scene)
        self.ang = gs.zeros(3, dtype=float, requires_grad=self.entity.scene.requires_grad, scene=self.entity.scene)

    def serializable(self):
        self.entity = None

        self.pos = self.pos.detach()
        self.quat = self.quat.detach()
        self.vel = self.vel.detach()
        self.ang = self.ang.detach()

    # def __repr__(self):
    #     return f'{self._repr_type()}\n' \
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

        self._pos = gs.zeros(
            (self._entity.n_particles, 3),
            dtype=float,
            requires_grad=self._entity.scene.requires_grad,
            scene=self._entity.scene,
        )
        self._vel = gs.zeros(
            (self._entity.n_particles, 3),
            dtype=float,
            requires_grad=self._entity.scene.requires_grad,
            scene=self._entity.scene,
        )
        self._C = gs.zeros(
            (self._entity.n_particles, 3, 3),
            dtype=float,
            requires_grad=self._entity.scene.requires_grad,
            scene=self._entity.scene,
        )
        self._F = gs.zeros(
            (self._entity.n_particles, 3, 3),
            dtype=float,
            requires_grad=self._entity.scene.requires_grad,
            scene=self._entity.scene,
        )
        self._Jp = gs.zeros(
            (self._entity.n_particles,),
            dtype=float,
            requires_grad=self._entity.scene.requires_grad,
            scene=self._entity.scene,
        )
        self._active = gs.zeros((self._entity.n_particles,), dtype=int, requires_grad=False, scene=self._entity.scene)

    def serializable(self):
        self._entity = None

        self._pos = self._pos.detach()
        self._vel = self._vel.detach()
        self._C = self._C.detach()
        self._F = self._F.detach()
        self._Jp = self._Jp.detach()
        self._active = self._active.detach()

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

        self._pos = gs.zeros((self._entity.n_particles, 3), dtype=float, requires_grad=False, scene=self._entity.scene)
        self._vel = gs.zeros((self._entity.n_particles, 3), dtype=float, requires_grad=False, scene=self._entity.scene)

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

        self._pos = gs.zeros((self.entity.n_vertices, 3), dtype=float, requires_grad=False, scene=self.entity.scene)
        self._vel = gs.zeros((self.entity.n_vertices, 3), dtype=float, requires_grad=False, scene=self.entity.scene)
        self._active = gs.zeros((self.entity.n_elements,), dtype=int, requires_grad=False, scene=self.entity.scene)

    def serializable(self):
        self._entity = None

        self._pos = self._pos.detach()
        self._vel = self._vel.detach()
        self._active = self._active.detach()

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
