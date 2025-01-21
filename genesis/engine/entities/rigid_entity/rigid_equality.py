import taichi as ti
import torch

import genesis as gs
import genesis.utils.geom as gu
from genesis.repr_base import RBC


@ti.data_oriented
class RigidEquality(RBC):
    """
    Equality class for rigid body entities.
    """

    def __init__(
        self,
        entity,
        name,
        idx,
        type,
        link1_idx,
        link2_idx,
        anchor1_pos,
        anchor2_pos,
        rel_pose,
        torque_scale,
        sol_params,
    ):
        self._name = name
        self._entity = entity
        self._solver = entity.solver

        self._uid = gs.UID()
        self._idx = idx
        self._type = type

        self._link1_idx = link1_idx
        self._link2_idx = link2_idx
        self._anchor1_pos = anchor1_pos
        self._anchor2_pos = anchor2_pos
        self._rel_pose = rel_pose
        self._torque_scale = torque_scale
        self._sol_params = sol_params

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def uid(self):
        """
        Returns the unique id of the equality.
        """
        return self._uid

    @property
    def name(self):
        """
        Returns the name of the equality.
        """
        return self._name

    @property
    def entity(self):
        """
        Returns the entity that the equality belongs to.
        """
        return self._entity

    @property
    def solver(self):
        """
        The RigidSolver object that the equality belongs to.
        """
        return self._solver

    @property
    def idx(self):
        """
        Returns the global index of the equality in the rigid solver.
        """
        return self._idx

    @property
    def idx_local(self):
        """
        Returns the local index of the equality in the entity.
        """
        return self._idx - self._entity._equality_start

    @property
    def type(self):
        """
        Returns the type of the equality.
        """
        return self._type

    @property
    def link1_idx(self):
        """
        Returns the index of the first link.
        """
        return self._link1_idx

    @property
    def link2_idx(self):
        """
        Returns the index of the second link.
        """
        return self._link2_idx

    @property
    def anchor1_pos(self):
        """
        Returns the position of the first anchor.
        """
        return self._anchor1_pos

    @property
    def anchor2_pos(self):
        """
        Returns the position of the second anchor.
        """
        return self._anchor2_pos

    @property
    def rel_pose(self):
        """
        Returns the relative pose between the two links.
        """
        return self._rel_pose

    @property
    def torque_scale(self):
        """
        Returns the torque scale of the equality.
        """
        return self._torque_scale

    @property
    def sol_params(self):
        """
        Returns the solver parameters of the equality.
        """
        return self._sol_params
