import gstaichi as ti
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
        eq_obj1id,
        eq_obj2id,
        eq_data,
        sol_params,
    ):
        self._name = name
        self._entity = entity
        self._solver = entity.solver

        self._uid = gs.UID()
        self._idx = idx
        self._type = type

        self._eq_obj1id = eq_obj1id
        self._eq_obj2id = eq_obj2id
        self._eq_data = eq_data
        self._sol_params = sol_params

    # ------------------------------------------------------------------------------------
    # -------------------------------- real-time state -----------------------------------
    # ------------------------------------------------------------------------------------

    def set_sol_params(self, sol_params):
        """
        Set the solver parameters of this equality constraint.
        """
        if self.is_built:
            self._solver.set_sol_params(sol_params[..., None, :], eqs_idx=self._idx, envs_idx=None, unsafe=False)
        else:
            self._sol_params = sol_params

    @property
    def sol_params(self):
        """
        Returns the solver parameters of this equality constraint.
        """
        if self.is_built:
            return self._solver.get_sol_params(eqs_idx=self._idx, envs_idx=None, unsafe=True)[..., 0, :]
        return self._sol_params

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
    def eq_obj1id(self):
        """
        Returns the index of the first link.
        """
        return self._eq_obj1id

    @property
    def eq_obj2id(self):
        """
        Returns the index of the second link.
        """
        return self._eq_obj2id

    @property
    def eq_data(self):
        """
        Returns the eq_data of this equality constraint.
        """
        return self._eq_data

    @property
    def is_built(self):
        """
        Whether the rigid entity this equality constraint belongs to is built.
        """
        return self.entity.is_built
