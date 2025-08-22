import gstaichi as ti
import torch

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.misc import DeprecationError
from genesis.repr_base import RBC


@ti.data_oriented
class RigidJoint(RBC):
    """
    Joint class for rigid body entities. Each RigidLink is connected to its parent link via a RigidJoint.
    """

    def __init__(
        self,
        entity,
        name,
        idx,
        link_idx,
        q_start,
        dof_start,
        n_qs,
        n_dofs,
        type,
        pos,
        quat,
        init_qpos,
        sol_params,
        dofs_motion_ang,
        dofs_motion_vel,
        dofs_limit,
        dofs_invweight,
        dofs_frictionloss,
        dofs_stiffness,
        dofs_damping,
        dofs_armature,
        dofs_kp,
        dofs_kv,
        dofs_force_range,
    ):
        self._name = name
        self._entity = entity
        self._solver = entity.solver

        self._uid = gs.UID()
        self._idx = idx
        self._link_idx = link_idx
        self._q_start = q_start
        self._dof_start = dof_start
        self._n_qs = n_qs
        self._n_dofs = n_dofs
        self._type = type
        self._pos = pos
        self._quat = quat
        self._init_qpos = init_qpos
        self._sol_params = sol_params

        self._dofs_motion_ang = dofs_motion_ang
        self._dofs_motion_vel = dofs_motion_vel
        self._dofs_limit = dofs_limit
        self._dofs_invweight = dofs_invweight
        self._dofs_frictionloss = dofs_frictionloss
        self._dofs_stiffness = dofs_stiffness
        self._dofs_damping = dofs_damping
        self._dofs_armature = dofs_armature
        self._dofs_kp = dofs_kp
        self._dofs_kv = dofs_kv
        self._dofs_force_range = dofs_force_range

    # ------------------------------------------------------------------------------------
    # -------------------------------- real-time state -----------------------------------
    # ------------------------------------------------------------------------------------

    def get_pos(self):
        """
        Get the position of the joint in the world frame.
        """
        raise DeprecationError(
            "This method has been removed. Please consider operating at link-level to get the cartesian position in "
            "word frame. Alternatively, 'get_anchor_pos' returns the anchor position of the joint in the world frame."
        )

    def get_quat(self):
        """
        Get the quaternion of the joint in the world frame.
        """
        raise DeprecationError(
            "This method has been removed. Please consider operating at link-level to get the cartesian orientation in "
            "word frame. Alternatively, 'get_anchor_axis' returns the anchor axis of the joint in the world frame."
        )

    @gs.assert_built
    def get_anchor_pos(self):
        """
        Get the anchor position of the joint in the world frame.

        Mathematically, the anchor point corresponds to the point that is fixed wrt parent link and is coincident with
        the joint for the neutral configuration qpos0. This means that this point moves under the effect of the
        generalized coordinates corresponding to this joint (and all its ancestors in the kinematic tree). Physically,
        the anchor point is the "output" of the joint transmission, on which the child body is welded.
        """
        tensor = torch.empty(self._solver._batch_shape(3, True), dtype=gs.tc_float, device=gs.device)
        self._kernel_get_anchor_pos(tensor)
        if self._solver.n_envs == 0:
            tensor = tensor.squeeze(0)
        return tensor

    @ti.kernel
    def _kernel_get_anchor_pos(self, tensor: ti.types.ndarray()):
        for i_b in range(self._solver._B):
            xpos = self._solver.joints_state.xanchor[self._idx, i_b]
            for i in ti.static(range(3)):
                tensor[i_b, i] = xpos[i]

    @gs.assert_built
    def get_anchor_axis(self):
        """
        Get the anchor axis of the joint in the world frame.

        See `RigidJoint.get_anchor_pos` documentation for details about the notion on anchor point.
        """
        tensor = torch.empty(self._solver._batch_shape(3, True), dtype=gs.tc_float, device=gs.device)
        self._kernel_get_anchor_axis(tensor)
        if self._solver.n_envs == 0:
            tensor = tensor.squeeze(0)
        return tensor

    @ti.kernel
    def _kernel_get_anchor_axis(self, tensor: ti.types.ndarray()):
        for i_b in range(self._solver._B):
            xaxis = self._solver.joints_state.xaxis[self._idx, i_b]
            for i in ti.static(range(3)):
                tensor[i_b, i] = xaxis[i]

    def set_sol_params(self, sol_params):
        """
        Set the solver parameters of this joint.
        """
        if self.is_built:
            self._solver.set_sol_params(sol_params[..., None, :], joints_idx=self._idx, envs_idx=None, unsafe=False)
        else:
            self._sol_params = sol_params

    @property
    def sol_params(self):
        """
        Retruns the solver parameters of the joint.
        """
        if self.is_built:
            return self._solver.get_sol_params(joints_idx=self._idx, envs_idx=None, unsafe=True)[..., 0, :]
        return self._sol_params

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def uid(self):
        """
        Returns the unique id of the joint.
        """
        return self._uid

    @property
    def name(self):
        """
        Returns the name of the joint.
        """
        return self._name

    @property
    def entity(self):
        """
        Returns the entity that the joint belongs to.
        """
        return self._entity

    @property
    def solver(self):
        """
        The RigidSolver object that the joint belongs to.
        """
        return self._solver

    @property
    def link(self):
        """
        Returns the child link that of the joint.
        """
        return self._solver.links[self._link_idx]

    @property
    def idx(self):
        """
        Returns the global index of the joint in the rigid solver.
        """
        return self._idx

    @property
    def idx_local(self):
        """
        Returns the local index of the joint in the entity.
        """
        return self._idx - self._entity.joint_start

    @property
    def init_qpos(self):
        """
        Returns the initial joint position.
        """
        return self._init_qpos

    @property
    def n_qs(self):
        """
        Returns the number of `q` (generalized coordinate) variables that the joint has.
        """
        return self._n_qs

    @property
    def n_dofs(self):
        """
        Returns the number of dofs that the joint has.
        """
        return self._n_dofs

    @property
    def type(self):
        """
        Returns the type of the joint.
        """
        return self._type

    @property
    def pos(self):
        """
        Returns the initial position of the joint in the world frame.
        """
        return self._pos

    @property
    def quat(self):
        """
        Returns the initial quaternion of the joint in the world frame.
        """
        return self._quat

    @property
    def q_start(self):
        """
        Returns the starting index of the `q` variables of the joint in the rigid solver.
        """
        return self._q_start

    @property
    def dof_start(self):
        """
        Returns the starting index of the dofs of the joint in the rigid solver.
        """
        return self._dof_start

    @property
    def q_end(self):
        """
        Returns the ending index of the `q` variables of the joint in the rigid solver.
        """
        return self._n_qs + self.q_start

    @property
    def dof_end(self):
        """
        Returns the ending index of the dofs of the joint in the rigid solver.
        """
        return self._n_dofs + self.dof_start

    @property
    def dof_idx(self):
        """
        Returns all the Degrees' of Freedom (DoF) indices of the joint in the rigid solver.

        This property either returns a list, an integer, or None depending on whether the joint has multiple DoFs, a
        single one, or none, respectively.
        """
        gs.logger.warning(
            "This property is deprecated and will be removed in future release. Please use 'dofs_idx' instead."
        )
        if self.n_dofs == 1:
            return self.dof_start
        if self.n_dofs == 0:
            return None
        return self.dofs_idx

    @property
    def dofs_idx(self):
        """
        Returns all the Degrees' of Freedom (DoF) indices of the joint in the rigid solver as a sequence.
        """
        return list(range(self.dof_start, self.dof_end))

    @property
    def dof_idx_local(self):
        """
        Returns the local dof index of the joint in the entity.

        This property either returns a list, an integer, or None depending on whether the joint has multiple DoFs, a
        single one, or none, respectively.
        """
        gs.logger.warning(
            "This property is deprecated and will be removed in future release. Please use 'dofs_idx_local' instead."
        )
        if self.n_dofs == 1:
            return self.dof_start - self._entity.dof_start
        if self.n_dofs == 0:
            return None
        return self.dofs_idx_local

    @property
    def dofs_idx_local(self):
        """
        Returns the local Degrees of Freedom indices of the joint in the entity.
        """
        return list(range(self.dof_start - self._entity.dof_start, self.dof_end - self._entity.dof_start))

    @property
    def q_idx(self):
        """
        Returns all the position indices of the joint in the rigid solver.

        This property either returns a list, an integer, or None depending on whether the joint has multiple position
        indices, a single one, or none, respectively.
        """
        gs.logger.warning(
            "This property is deprecated and will be removed in future release. Please use 'qs_idx' instead."
        )
        if self.n_qs == 1:
            return self.q_start
        elif self.n_qs == 0:
            return None
        else:
            return self.qs_idx

    @property
    def qs_idx(self):
        """
        Returns all the position indices of the joint in the rigid solver.
        """
        return list(range(self.q_start, self.q_end))

    @property
    def q_idx_local(self):
        """
        Returns all the local `q` indices of the joint in the entity.
        """
        gs.logger.warning(
            "This property is deprecated and will be removed in future release. Please use 'qs_idx_local' instead."
        )
        if self.n_qs == 1:
            return self.q_start - self._entity.q_start
        elif self.n_qs == 0:
            return None
        else:
            return self.qs_idx_local

    @property
    def qs_idx_local(self):
        """
        Returns all the local `q` indices of the joint in the entity.
        """
        return list(range(self.q_start - self._entity.q_start, self.q_end - self._entity.q_start))

    @property
    def dofs_motion_ang(self):
        return self._dofs_motion_ang

    @property
    def dofs_motion_vel(self):
        return self._dofs_motion_vel

    @property
    def dofs_limit(self):
        """
        Returns the range limit of the dofs of the joint.
        """
        return self._dofs_limit

    @property
    def dofs_invweight(self):
        """
        Returns the invweight of the dofs of the joint.
        """
        return self._dofs_invweight

    @property
    def dofs_frictionloss(self):
        """
        Returns the frictionloss of the dofs of the joint.
        """
        return self._dofs_frictionloss

    @property
    def dofs_stiffness(self):
        """
        Returns the stiffness of the dofs of the joint.
        """
        return self._dofs_stiffness

    @property
    def dofs_damping(self):
        """
        Returns the damping of the dofs of the joint.
        """
        return self._dofs_damping

    @property
    def dofs_armature(self):
        """
        Returns the armature of the dofs of the joint.
        """
        return self._dofs_armature

    @property
    def dofs_kp(self):
        """
        Returns the kp (positional gain) of the dofs of the joint.
        """
        return self._dofs_kp

    @property
    def dofs_kv(self):
        """
        Returns the kv (velocity gain) of the dofs of the joint.
        """
        return self._dofs_kv

    @property
    def dofs_force_range(self):
        """
        Returns the force range of the dofs of the joint.
        """
        return self._dofs_force_range

    @property
    def is_built(self):
        """
        Returns whether the entity the joint belongs to is built.
        """
        return self.entity.is_built

    # ------------------------------------------------------------------------------------
    # -------------------------------------- repr ----------------------------------------
    # ------------------------------------------------------------------------------------

    def _repr_brief(self):
        return f"{(self._repr_type())}: {self._uid}, name: '{self._name}', idx: {self._idx}, type: {self._type}"
