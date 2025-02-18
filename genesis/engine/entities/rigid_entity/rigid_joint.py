import taichi as ti
import torch

import genesis as gs
import genesis.utils.geom as gu
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
        q_start,
        dof_start,
        n_qs,
        n_dofs,
        type,
        pos,
        quat,
        dofs_motion_ang,
        dofs_motion_vel,
        dofs_limit,
        dofs_invweight,
        dofs_stiffness,
        dofs_sol_params,
        dofs_damping,
        dofs_armature,
        dofs_kp,
        dofs_kv,
        dofs_force_range,
        init_qpos,
    ):
        self._name = name
        self._entity = entity
        self._solver = entity.solver

        self._uid = gs.UID()
        self._idx = idx
        self._q_start = q_start
        self._dof_start = dof_start
        self._n_qs = n_qs
        self._n_dofs = n_dofs
        self._type = type
        self._pos = pos
        self._quat = quat
        self._init_qpos = init_qpos

        self._dofs_motion_ang = dofs_motion_ang
        self._dofs_motion_vel = dofs_motion_vel
        self._dofs_limit = dofs_limit
        self._dofs_invweight = dofs_invweight
        self._dofs_stiffness = dofs_stiffness
        self._dofs_sol_params = dofs_sol_params
        self._dofs_damping = dofs_damping
        self._dofs_armature = dofs_armature
        self._dofs_kp = dofs_kp
        self._dofs_kv = dofs_kv
        self._dofs_force_range = dofs_force_range

        # NOTE: temp hack to use 0 damping/armature for drone
        if isinstance(self._entity, gs.engine.entities.DroneEntity) and self._type == gs.JOINT_TYPE.FREE:
            import numpy as np

            self._dofs_damping = np.zeros_like(self._dofs_damping)
            self._dofs_armature = np.zeros_like(self._dofs_armature)

    # ------------------------------------------------------------------------------------
    # -------------------------------- real-time state -----------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    def get_pos(self):
        """
        Get the position of the joint in the world frame.
        """
        tensor = torch.empty(self._solver._batch_shape(3, True), dtype=gs.tc_float, device=gs.device)
        self._kernel_get_pos(tensor)
        if self._solver.n_envs == 0:
            tensor = tensor.squeeze(0)
        return tensor

    @ti.kernel
    def _kernel_get_pos(self, tensor: ti.types.ndarray()):

        for i_b in range(self._solver._B):
            I_l = [self._idx, i_b] if ti.static(self._solver._options.batch_links_info) else self._idx
            l_info = self._solver.links_info[I_l]
            i_p = l_info.parent_idx

            p_pos = ti.Vector.zero(gs.ti_float, 3)
            p_quat = gu.ti_identity_quat()

            if i_p != -1:
                p_pos = self._solver.links_state[i_p, i_b].pos
                p_quat = self._solver.links_state[i_p, i_b].quat

            tmp_pos, tmp_quat = gu.ti_transform_pos_quat_by_trans_quat(l_info.pos, l_info.quat, p_pos, p_quat)

            joint_pos, joint_quat = gu.ti_transform_pos_quat_by_trans_quat(
                l_info.joint_pos, l_info.joint_quat, tmp_pos, tmp_quat
            )

            for i in ti.static(range(3)):
                tensor[i_b, i] = joint_pos[i]

    @gs.assert_built
    def get_quat(self):
        """
        Get the quaternion of the joint in the world frame.
        """
        tensor = torch.empty(self._solver._batch_shape(4, True), dtype=gs.tc_float, device=gs.device)
        self._kernel_get_quat(tensor)
        if self._solver.n_envs == 0:
            tensor = tensor.squeeze(0)
        return tensor

    @ti.kernel
    def _kernel_get_quat(self, tensor: ti.types.ndarray()):

        for i_b in range(self._solver._B):
            I_l = [self._idx, i_b] if ti.static(self._solver._options.batch_links_info) else self._idx
            l_info = self._solver.links_info[I_l]
            i_p = l_info.parent_idx

            p_pos = ti.Vector.zero(gs.ti_float, 3)
            p_quat = gu.ti_identity_quat()

            if i_p != -1:
                p_pos = self._solver.links_state[i_p, i_b].pos
                p_quat = self._solver.links_state[i_p, i_b].quat

            tmp_pos, tmp_quat = gu.ti_transform_pos_quat_by_trans_quat(l_info.pos, l_info.quat, p_pos, p_quat)

            joint_pos, joint_quat = gu.ti_transform_pos_quat_by_trans_quat(
                l_info.joint_pos, l_info.joint_quat, tmp_pos, tmp_quat
            )

            for i in ti.static(range(4)):
                tensor[i_b, i] = joint_quat[i]

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
        return self._solver.links[self._idx]

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
        return self._idx - self._entity._joint_start

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
        Returns all the dof indices of the joint in the rigid solver.
        """
        if self.n_dofs == 1:
            return self.dof_start
        elif self.n_dofs == 0:
            return None
        else:
            return list(range(self.dof_start, self.dof_end))

    @property
    def dof_idx_local(self):
        """
        Returns the local dof index of the joint in the entity.
        """
        if self.n_dofs == 1:
            return self.dof_idx - self._entity._dof_start
        elif self.n_dofs == 0:
            return None
        else:
            return [dof_idx - self._entity._dof_start for dof_idx in self.dof_idx]

    @property
    def q_idx(self):
        """
        Returns all the `q` indices of the joint in the rigid solver.
        """
        if self.n_qs == 1:
            return self.q_start
        elif self.n_qs == 0:
            return None
        else:
            return list(range(self.q_start, self.q_end))

    @property
    def q_idx_local(self):
        """
        Returns all the local `q` indices of the joint in the entity.
        """
        if self.n_qs == 1:
            return self.q_start - self._entity._q_start
        elif self.n_qs == 0:
            return None
        else:
            return [q_idx - self._entity._q_start for q_idx in self.q_idx]

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
    def dofs_stiffness(self):
        """
        Returns the stiffness of the dofs of the joint.
        """
        return self._dofs_stiffness

    @property
    def dofs_sol_params(self):
        """
        Retruns the solver parameters of the dofs of the joint.
        """
        return self._dofs_sol_params

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
