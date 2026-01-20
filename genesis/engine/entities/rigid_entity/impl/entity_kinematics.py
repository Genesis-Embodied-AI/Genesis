"""
Kinematics mixin for RigidEntity.

Contains methods for Jacobian computation, inverse kinematics, forward kinematics, and path planning.
"""

import gstaichi as ti
import numpy as np
import torch

import genesis as gs
from genesis.utils import array_class
from genesis.utils import geom as gu
from genesis.utils import linalg as lu
from genesis.utils.misc import broadcast_tensor, ti_to_torch


class RigidEntityKinematicsMixin:
    """Mixin class providing kinematics functionality for RigidEntity."""

    def get_jacobian(self, link, local_point=None):
        """
        Get the spatial Jacobian for a point on a target link.

        Parameters
        ----------
        link : RigidLink
            The target link.
        local_point : torch.Tensor or None, shape (3,)
            Coordinates of the point in the link's *local* frame.
            If None, the link origin is used (back-compat).

        Returns
        -------
        jacobian : torch.Tensor
            The Jacobian matrix of shape (n_envs, 6, entity.n_dofs) or (6, entity.n_dofs) if n_envs == 0.
        """
        if not self._requires_jac_and_IK:
            gs.raise_exception(
                "Inverse kinematics and jacobian are disabled for this entity. Set `morph.requires_jac_and_IK` to True if you need them."
            )

        if self.n_dofs == 0:
            gs.raise_exception("Entity has zero dofs.")

        if local_point is None:
            sol = self._solver
            self._kernel_get_jacobian_zero(
                tgt_link_idx=link.idx,
                dofs_info=sol.dofs_info,
                joints_info=sol.joints_info,
                links_info=sol.links_info,
                links_state=sol.links_state,
            )
        else:
            p_local = torch.as_tensor(local_point, dtype=gs.tc_float, device=gs.device)
            if p_local.shape != (3,):
                gs.raise_exception("Must be a vector of length 3")
            sol = self._solver
            self._kernel_get_jacobian(
                tgt_link_idx=link.idx,
                p_local=p_local,
                dofs_info=sol.dofs_info,
                joints_info=sol.joints_info,
                links_info=sol.links_info,
                links_state=sol.links_state,
            )

        jacobian = ti_to_torch(self._jacobian, transpose=True, copy=True)
        if self._solver.n_envs == 0:
            jacobian = jacobian[0]

        return jacobian

    @ti.func
    def _impl_get_jacobian(
        self,
        tgt_link_idx,
        i_b,
        p_vec,
        dofs_info: array_class.DofsInfo,
        joints_info: array_class.JointsInfo,
        links_info: array_class.LinksInfo,
        links_state: array_class.LinksState,
    ):
        self._func_get_jacobian(
            tgt_link_idx=tgt_link_idx,
            i_b=i_b,
            p_local=p_vec,
            pos_mask=ti.Vector.one(gs.ti_int, 3),
            rot_mask=ti.Vector.one(gs.ti_int, 3),
            dofs_info=dofs_info,
            joints_info=joints_info,
            links_info=links_info,
            links_state=links_state,
        )

    @ti.kernel
    def _kernel_get_jacobian(
        self,
        tgt_link_idx: ti.i32,
        p_local: ti.types.ndarray(),
        dofs_info: array_class.DofsInfo,
        joints_info: array_class.JointsInfo,
        links_info: array_class.LinksInfo,
        links_state: array_class.LinksState,
    ):
        p_vec = ti.Vector([p_local[0], p_local[1], p_local[2]], dt=gs.ti_float)
        for i_b in range(self._solver._B):
            self._impl_get_jacobian(
                tgt_link_idx=tgt_link_idx,
                i_b=i_b,
                p_vec=p_vec,
                dofs_info=dofs_info,
                joints_info=joints_info,
                links_info=links_info,
                links_state=links_state,
            )

    @ti.kernel
    def _kernel_get_jacobian_zero(
        self,
        tgt_link_idx: ti.i32,
        dofs_info: array_class.DofsInfo,
        joints_info: array_class.JointsInfo,
        links_info: array_class.LinksInfo,
        links_state: array_class.LinksState,
    ):
        for i_b in range(self._solver._B):
            self._impl_get_jacobian(
                tgt_link_idx=tgt_link_idx,
                i_b=i_b,
                p_vec=ti.Vector.zero(gs.ti_float, 3),
                dofs_info=dofs_info,
                joints_info=joints_info,
                links_info=links_info,
                links_state=links_state,
            )

    @ti.func
    def _func_get_jacobian(
        self,
        tgt_link_idx,
        i_b,
        p_local,
        pos_mask,
        rot_mask,
        dofs_info: array_class.DofsInfo,
        joints_info: array_class.JointsInfo,
        links_info: array_class.LinksInfo,
        links_state: array_class.LinksState,
    ):
        for i_row, i_d in ti.ndrange(6, self.n_dofs):
            self._jacobian[i_row, i_d, i_b] = 0.0

        tgt_link_pos = links_state.pos[tgt_link_idx, i_b] + gu.ti_transform_by_quat(
            p_local, links_state.quat[tgt_link_idx, i_b]
        )
        i_l = tgt_link_idx
        while i_l > -1:
            I_l = [i_l, i_b] if ti.static(self.solver._options.batch_links_info) else i_l

            dof_offset = 0
            for i_j in range(links_info.joint_start[I_l], links_info.joint_end[I_l]):
                I_j = [i_j, i_b] if ti.static(self.solver._options.batch_joints_info) else i_j

                if joints_info.type[I_j] == gs.JOINT_TYPE.FIXED:
                    pass

                elif joints_info.type[I_j] == gs.JOINT_TYPE.REVOLUTE:
                    i_d = joints_info.dof_start[I_j]
                    I_d = [i_d, i_b] if ti.static(self.solver._options.batch_dofs_info) else i_d
                    i_d_jac = i_d + dof_offset - self._dof_start
                    rotation = gu.ti_transform_by_quat(dofs_info.motion_ang[I_d], links_state.quat[i_l, i_b])
                    translation = rotation.cross(tgt_link_pos - links_state.pos[i_l, i_b])

                    self._jacobian[0, i_d_jac, i_b] = translation[0] * pos_mask[0]
                    self._jacobian[1, i_d_jac, i_b] = translation[1] * pos_mask[1]
                    self._jacobian[2, i_d_jac, i_b] = translation[2] * pos_mask[2]
                    self._jacobian[3, i_d_jac, i_b] = rotation[0] * rot_mask[0]
                    self._jacobian[4, i_d_jac, i_b] = rotation[1] * rot_mask[1]
                    self._jacobian[5, i_d_jac, i_b] = rotation[2] * rot_mask[2]

                elif joints_info.type[I_j] == gs.JOINT_TYPE.PRISMATIC:
                    i_d = joints_info.dof_start[I_j]
                    I_d = [i_d, i_b] if ti.static(self.solver._options.batch_dofs_info) else i_d
                    i_d_jac = i_d + dof_offset - self._dof_start
                    translation = gu.ti_transform_by_quat(dofs_info.motion_vel[I_d], links_state.quat[i_l, i_b])

                    self._jacobian[0, i_d_jac, i_b] = translation[0] * pos_mask[0]
                    self._jacobian[1, i_d_jac, i_b] = translation[1] * pos_mask[1]
                    self._jacobian[2, i_d_jac, i_b] = translation[2] * pos_mask[2]

                elif joints_info.type[I_j] == gs.JOINT_TYPE.FREE:
                    # translation
                    for i_d_ in ti.static(range(3)):
                        i_d = joints_info.dof_start[I_j] + i_d_
                        i_d_jac = i_d + dof_offset - self._dof_start

                        self._jacobian[i_d_, i_d_jac, i_b] = 1.0 * pos_mask[i_d_]

                    # rotation
                    for i_d_ in ti.static(range(3)):
                        i_d = joints_info.dof_start[I_j] + i_d_ + 3
                        i_d_jac = i_d + dof_offset - self._dof_start
                        I_d = [i_d, i_b] if ti.static(self.solver._options.batch_dofs_info) else i_d
                        rotation = dofs_info.motion_ang[I_d]
                        translation = rotation.cross(tgt_link_pos - links_state.pos[i_l, i_b])

                        self._jacobian[0, i_d_jac, i_b] = translation[0] * pos_mask[0]
                        self._jacobian[1, i_d_jac, i_b] = translation[1] * pos_mask[1]
                        self._jacobian[2, i_d_jac, i_b] = translation[2] * pos_mask[2]
                        self._jacobian[3, i_d_jac, i_b] = rotation[0] * rot_mask[0]
                        self._jacobian[4, i_d_jac, i_b] = rotation[1] * rot_mask[1]
                        self._jacobian[5, i_d_jac, i_b] = rotation[2] * rot_mask[2]

                dof_offset = dof_offset + joints_info.n_dofs[I_j]

            i_l = links_info.parent_idx[I_l]

    @gs.assert_built
    def inverse_kinematics(
        self,
        link,
        pos=None,
        quat=None,
        init_qpos=None,
        respect_joint_limit=True,
        max_samples=50,
        max_solver_iters=20,
        damping=0.01,
        pos_tol=5e-4,  # 0.5 mm
        rot_tol=5e-3,  # 0.28 degree
        pos_mask=[True, True, True],
        rot_mask=[True, True, True],
        max_step_size=0.5,
        dofs_idx_local=None,
        return_error=False,
        envs_idx=None,
    ):
        """
        Compute inverse kinematics for a single target link.

        Parameters
        ----------
        link : RigidLink
            The link to be used as the end-effector.
        pos : None | array_like, shape (3,), optional
            The target position. If None, position error will not be considered. Defaults to None.
        quat : None | array_like, shape (4,), optional
            The target orientation. If None, orientation error will not be considered. Defaults to None.
        init_qpos : None | array_like, shape (n_dofs,), optional
            Initial qpos used for solving IK. If None, the current qpos will be used. Defaults to None.
        respect_joint_limit : bool, optional
            Whether to respect joint limits. Defaults to True.
        max_samples : int, optional
            Number of resample attempts. Defaults to 50.
        max_solver_iters : int, optional
            Maximum number of solver iterations per sample. Defaults to 20.
        damping : float, optional
            Damping for damped least squares. Defaults to 0.01.
        pos_tol : float, optional
            Position tolerance for normalized position error (in meter). Defaults to 1e-4.
        rot_tol : float, optional
            Rotation tolerance for normalized rotation vector error (in radian). Defaults to 1e-4.
        pos_mask : list, shape (3,), optional
            Mask for position error. Defaults to [True, True, True]. E.g.: If you only care about position along x and y, you can set it to [True, True, False].
        rot_mask : list, shape (3,), optional
            Mask for rotation axis alignment. Defaults to [True, True, True]. E.g.: If you only want the link's Z-axis to be aligned with the Z-axis in the given quat, you can set it to [False, False, True].
        max_step_size : float, optional
            Maximum step size in q space for each IK solver step. Defaults to 0.5.
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to set. If None, all dofs will be set. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None. This is used to specify which dofs the IK is applied to.
        return_error : bool, optional
            Whether to return the final errorqpos. Defaults to False.
        envs_idx: None | array_like, optional
            The indices of the environments to set. If None, all environments will be set. Defaults to None.

        Returns
        -------
        qpos : array_like, shape (n_dofs,) or (n_envs, n_dofs) or (len(envs_idx), n_dofs)
            Solver qpos (joint positions).
        (optional) error_pose : array_like, shape (6,) or (n_envs, 6) or (len(envs_idx), 6)
            Pose error for each target. The 6-vector is [err_pos_x, err_pos_y, err_pos_z, err_rot_x, err_rot_y, err_rot_z]. Only returned if `return_error` is True.
        """
        if self._solver.n_envs > 0:
            envs_idx = self._scene._sanitize_envs_idx(envs_idx)

            if pos is not None:
                if pos.shape[0] != len(envs_idx):
                    gs.raise_exception("First dimension of `pos` must be equal to `scene.n_envs`.")
            if quat is not None:
                if quat.shape[0] != len(envs_idx):
                    gs.raise_exception("First dimension of `quat` must be equal to `scene.n_envs`.")

        ret = self.inverse_kinematics_multilink(
            links=[link],
            poss=[pos] if pos is not None else [],
            quats=[quat] if quat is not None else [],
            init_qpos=init_qpos,
            respect_joint_limit=respect_joint_limit,
            max_samples=max_samples,
            max_solver_iters=max_solver_iters,
            damping=damping,
            pos_tol=pos_tol,
            rot_tol=rot_tol,
            pos_mask=pos_mask,
            rot_mask=rot_mask,
            max_step_size=max_step_size,
            dofs_idx_local=dofs_idx_local,
            return_error=return_error,
            envs_idx=envs_idx,
        )

        if return_error:
            qpos, error_pose = ret
            return qpos, error_pose[..., 0, :]
        return ret

    @gs.assert_built
    def inverse_kinematics_multilink(
        self,
        links,
        poss=None,
        quats=None,
        init_qpos=None,
        respect_joint_limit=True,
        max_samples=50,
        max_solver_iters=20,
        damping=0.01,
        pos_tol=5e-4,  # 0.5 mm
        rot_tol=5e-3,  # 0.28 degree
        pos_mask=[True, True, True],
        rot_mask=[True, True, True],
        max_step_size=0.5,
        dofs_idx_local=None,
        return_error=False,
        envs_idx=None,
    ):
        """
        Compute inverse kinematics for  multiple target links.

        Parameters
        ----------
        links : list of RigidLink
            List of links to be used as the end-effectors.
        poss : list, optional
            List of target positions. If empty, position error will not be considered. Defaults to None.
        quats : list, optional
            List of target orientations. If empty, orientation error will not be considered. Defaults to None.
        init_qpos : array_like, shape (n_dofs,), optional
            Initial qpos used for solving IK. If None, the current qpos will be used. Defaults to None.
        respect_joint_limit : bool, optional
            Whether to respect joint limits. Defaults to True.
        max_samples : int, optional
            Number of resample attempts. Defaults to 50.
        max_solver_iters : int, optional
            Maximum number of solver iterations per sample. Defaults to 20.
        damping : float, optional
            Damping for damped least squares. Defaults to 0.01.
        pos_tol : float, optional
            Position tolerance for normalized position error (in meter). Defaults to 1e-4.
        rot_tol : float, optional
            Rotation tolerance for normalized rotation vector error (in radian). Defaults to 1e-4.
        pos_mask : list, shape (3,), optional
            Mask for position error. Defaults to [True, True, True]. E.g.: If you only care about position along x and y, you can set it to [True, True, False].
        rot_mask : list, shape (3,), optional
            Mask for rotation axis alignment. Defaults to [True, True, True]. E.g.: If you only want the link's Z-axis to be aligned with the Z-axis in the given quat, you can set it to [False, False, True].
        max_step_size : float, optional
            Maximum step size in q space for each IK solver step. Defaults to 0.5.
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to set. If None, all dofs will be set. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None. This is used to specify which dofs the IK is applied to.
        return_error : bool, optional
            Whether to return the final errorqpos. Defaults to False.
        envs_idx : None | array_like, optional
            The indices of the environments to set. If None, all environments will be set. Defaults to None.

        Returns
        -------
        qpos : array_like, shape (n_dofs,) or (n_envs, n_dofs) or (len(envs_idx), n_dofs)
            Solver qpos (joint positions).
        (optional) error_pose : array_like, shape (6,) or (n_envs, 6) or (len(envs_idx), 6)
            Pose error for each target. The 6-vector is [err_pos_x, err_pos_y, err_pos_z, err_rot_x, err_rot_y, err_rot_z]. Only returned if `return_error` is True.
        """
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)

        if not self._requires_jac_and_IK:
            gs.raise_exception(
                "Inverse kinematics and jacobian are disabled for this entity. Set `morph.requires_jac_and_IK` to True if you need them."
            )

        if self.n_dofs == 0:
            gs.raise_exception("Entity has zero dofs.")

        n_links = len(links)
        if n_links == 0:
            gs.raise_exception("Target link not provided.")

        poss = list(poss) if poss is not None else []
        if not poss:
            poss = [None for _ in range(n_links)]
            pos_mask = [False, False, False]
        elif len(poss) != n_links:
            gs.raise_exception("Accepting only `poss` with length equal to `links` or empty list.")

        quats = list(quats) if quats is not None else []
        if not quats:
            quats = [None for _ in range(n_links)]
            rot_mask = [False, False, False]
        elif len(quats) != n_links:
            gs.raise_exception("Accepting only `quats` with length equal to `links` or empty list.")

        link_pos_mask, link_rot_mask = [], []
        for i, (pos, quat) in enumerate(zip(poss, quats)):
            if pos is None and quat is None:
                gs.raise_exception("At least one of `poss` or `quats` must be provided.")
            link_pos_mask.append(pos is not None)
            poss[i] = broadcast_tensor(pos, gs.tc_float, (len(envs_idx), 3), ("envs_idx", "")).contiguous()
            link_rot_mask.append(quat is not None)
            if quat is None:
                quat = gu.identity_quat()
            quats[i] = broadcast_tensor(quat, gs.tc_float, (len(envs_idx), 4), ("envs_idx", "")).contiguous()
        link_pos_mask = torch.tensor(link_pos_mask, dtype=gs.tc_int, device=gs.device)
        link_rot_mask = torch.tensor(link_rot_mask, dtype=gs.tc_int, device=gs.device)
        poss = torch.stack(poss, dim=0)
        quats = torch.stack(quats, dim=0)

        custom_init_qpos = init_qpos is not None
        init_qpos = broadcast_tensor(
            init_qpos, gs.tc_float, (len(envs_idx), self.n_qs), ("envs_idx", "qs_idx")
        ).contiguous()

        # pos and rot mask
        pos_mask = broadcast_tensor(pos_mask, gs.tc_bool, (3,)).contiguous()
        rot_mask = broadcast_tensor(rot_mask, gs.tc_bool, (3,)).contiguous()
        if (num_axis := rot_mask.sum()) == 1:
            rot_mask = ~rot_mask if gs.tc_bool == torch.bool else 1 - rot_mask
        elif num_axis == 2:
            gs.raise_exception("You can only align 0, 1 axis or all 3 axes.")

        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs)
        n_dofs = len(dofs_idx)
        if n_dofs == 0:
            gs.raise_exception("Target dofs not provided.")

        links_idx = torch.tensor([link.idx for link in links], dtype=gs.tc_int, device=gs.device)
        links_idx_by_dofs = []
        for link in self.links:
            for joint in link.joints:
                if any(i in dofs_idx for i in joint.dofs_idx_local):
                    links_idx_by_dofs.append(link.idx_local)
                    break
        links_idx_by_dofs = self._get_global_idx(links_idx_by_dofs, self.n_links, self._link_start)
        n_links_by_dofs = len(links_idx_by_dofs)

        kernel_rigid_entity_inverse_kinematics(
            self,
            links_idx,
            poss,
            quats,
            n_links,
            dofs_idx,
            n_dofs,
            links_idx_by_dofs,
            n_links_by_dofs,
            custom_init_qpos,
            init_qpos,
            max_samples,
            max_solver_iters,
            damping,
            pos_tol,
            rot_tol,
            pos_mask,
            rot_mask,
            link_pos_mask,
            link_rot_mask,
            max_step_size,
            respect_joint_limit,
            envs_idx,
            self._solver.links_state,
            self._solver.links_info,
            self._solver.joints_state,
            self._solver.joints_info,
            self._solver.dofs_state,
            self._solver.dofs_info,
            self._solver.entities_info,
            self._solver._rigid_global_info,
            self._solver._static_rigid_sim_config,
        )

        qpos = ti_to_torch(self._IK_qpos_best, transpose=True, copy=True)
        qpos = qpos[0] if self._solver.n_envs == 0 else qpos[envs_idx]

        if return_error:
            error_pose = ti_to_torch(self._IK_err_pose_best, transpose=True, copy=True).reshape(
                (-1, self._IK_n_tgts, 6)
            )[:, :n_links]
            error_pose = error_pose[0] if self._solver.n_envs == 0 else error_pose[envs_idx]
            return qpos, error_pose
        return qpos

    @gs.assert_built
    def forward_kinematics(self, qpos, qs_idx_local=None, links_idx_local=None, envs_idx=None):
        """
        Compute forward kinematics for a single target link.

        Parameters
        ----------
        qpos : array_like, shape (n_qs,) or (n_envs, n_qs) or (len(envs_idx), n_qs)
            The joint positions.
        qs_idx_local : None | array_like, optional
            The indices of the qpos to set. If None, all qpos will be set. Defaults to None.
        links_idx_local : None | array_like, optional
            The indices of the links to get. If None, all links will be returned. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments to set. If None, all environments will be set. Defaults to None.

        Returns
        -------
        links_pos : array_like, shape (n_links, 3) or (n_envs, n_links, 3) or (len(envs_idx), n_links, 3)
            The positions of the links (link frame origins).
        links_quat : array_like, shape (n_links, 4) or (n_envs, n_links, 4) or (len(envs_idx), n_links, 4)
            The orientations of the links.
        """

        if self._solver.n_envs == 0:
            qpos = qpos[None]
            envs_idx = torch.zeros(1, dtype=gs.tc_int)
        else:
            envs_idx = self._scene._sanitize_envs_idx(envs_idx)

        links_idx = self._get_global_idx(links_idx_local, self.n_links, self._link_start)
        links_pos = torch.empty((len(envs_idx), len(links_idx), 3), dtype=gs.tc_float, device=gs.device)
        links_quat = torch.empty((len(envs_idx), len(links_idx), 4), dtype=gs.tc_float, device=gs.device)

        self._kernel_forward_kinematics(
            links_pos,
            links_quat,
            qpos,
            self._get_global_idx(qs_idx_local, self.n_qs, self._q_start),
            links_idx,
            envs_idx,
            self._solver.links_state,
            self._solver.links_info,
            self._solver.joints_state,
            self._solver.joints_info,
            self._solver.dofs_state,
            self._solver.dofs_info,
            self._solver.entities_info,
            self._solver._rigid_global_info,
            self._solver._static_rigid_sim_config,
        )

        if self._solver.n_envs == 0:
            links_pos = links_pos[0]
            links_quat = links_quat[0]
        return links_pos, links_quat

    @ti.kernel
    def _kernel_forward_kinematics(
        self,
        links_pos: ti.types.ndarray(),
        links_quat: ti.types.ndarray(),
        qpos: ti.types.ndarray(),
        qs_idx: ti.types.ndarray(),
        links_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
        links_state: array_class.LinksState,
        links_info: array_class.LinksInfo,
        joints_state: array_class.JointsState,
        joints_info: array_class.JointsInfo,
        dofs_state: array_class.DofsState,
        dofs_info: array_class.DofsInfo,
        entities_info: array_class.EntitiesInfo,
        rigid_global_info: array_class.RigidGlobalInfo,
        static_rigid_sim_config: ti.template(),
    ):
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_q_, i_b_ in ti.ndrange(qs_idx.shape[0], envs_idx.shape[0]):
            # save original qpos
            # NOTE: reusing the IK_qpos_orig as cache (should not be a problem)
            self._IK_qpos_orig[qs_idx[i_q_], envs_idx[i_b_]] = rigid_global_info.qpos[qs_idx[i_q_], envs_idx[i_b_]]
            # set new qpos
            rigid_global_info.qpos[qs_idx[i_q_], envs_idx[i_b_]] = qpos[i_b_, i_q_]

        # run FK
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_b_ in range(envs_idx.shape[0]):
            gs.engine.solvers.rigid.rigid_solver.func_forward_kinematics_entity(
                self._idx_in_solver,
                envs_idx[i_b_],
                links_state,
                links_info,
                joints_state,
                joints_info,
                dofs_state,
                dofs_info,
                entities_info,
                rigid_global_info,
                static_rigid_sim_config,
                is_backward=False,
            )

        ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
        for i_l_, i_b_ in ti.ndrange(links_idx.shape[0], envs_idx.shape[0]):
            for i in ti.static(range(3)):
                links_pos[i_b_, i_l_, i] = links_state.pos[links_idx[i_l_], envs_idx[i_b_]][i]
            for i in ti.static(range(4)):
                links_quat[i_b_, i_l_, i] = links_state.quat[links_idx[i_l_], envs_idx[i_b_]][i]

        # restore original qpos
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_q_, i_b_ in ti.ndrange(qs_idx.shape[0], envs_idx.shape[0]):
            rigid_global_info.qpos[qs_idx[i_q_], envs_idx[i_b_]] = self._IK_qpos_orig[qs_idx[i_q_], envs_idx[i_b_]]

        # run FK
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_b_ in range(envs_idx.shape[0]):
            gs.engine.solvers.rigid.rigid_solver.func_forward_kinematics_entity(
                self._idx_in_solver,
                envs_idx[i_b_],
                links_state,
                links_info,
                joints_state,
                joints_info,
                dofs_state,
                dofs_info,
                entities_info,
                rigid_global_info,
                static_rigid_sim_config,
                is_backward=False,
            )

    # ------------------------------------------------------------------------------------
    # --------------------------------- motion planing -----------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    def plan_path(
        self,
        qpos_goal,
        qpos_start=None,
        max_nodes=2000,
        resolution=0.05,
        timeout=None,
        max_retry=1,
        smooth_path=True,
        num_waypoints=300,
        ignore_collision=False,
        planner="RRTConnect",
        envs_idx=None,
        return_valid_mask=False,
        *,
        ee_link_name=None,
        with_entity=None,
        **kwargs,
    ):
        """
        Plan a path from `qpos_start` to `qpos_goal`.

        Parameters
        ----------
        qpos_goal : array_like
            The goal state. [B, Nq] or [1, Nq]
        qpos_start : None | array_like, optional
            The start state. If None, the current state of the rigid entity will be used.
            Defaults to None. [B, Nq] or [1, Nq]
        resolution : float, optiona
            Joint-space resolution. It corresponds to the maximum distance between states to be checked
            for validity along a path segment.
        timeout : float, optional
            The max time to spend for each planning in seconds. Note that the timeout is not exact.
        max_retry : float, optional
            Maximum number of retry in case of timeout or convergence failure. Default to 1.
        smooth_path : bool, optional
            Whether to smooth the path after finding a solution. Defaults to True.
        num_waypoints : int, optional
            The number of waypoints to interpolate the path. If None, no interpolation will be performed.
            Defaults to 100.
        ignore_collision : bool, optional
            Whether to ignore collision checking during motion planning. Defaults to False.
        ignore_joint_limit : bool, optional
            This option has been deprecated and is not longer doing anything.
        planner : str, optional
            The name of the motion planning algorithm to use.
            Supported planners: 'RRT', 'RRTConnect'. Defaults to 'RRTConnect'.
        envs_idx : None | array_like, optional
            The indices of the environments to set. If None, all environments will be set. Defaults to None.
        return_valid_mask: bool
            Obtain valid mask of the succesful planed path over batch.
        ee_link_name: str
            The name of the link, which we "attach" the object during the planning
        with_entity: RigidEntity
            The (non-articulated) object to "attach" during the planning

        Returns
        -------
        path : torch.Tensor
            A tensor of waypoints representing the planned path.
            Each waypoint is an array storing the entity's qpos of a single time step.
        is_invalid: torch.Tensor
            A tensor of boolean mask indicating the batch indices with failed plan.
        """
        if self._solver.n_envs > 0:
            n_envs = len(self._scene._sanitize_envs_idx(envs_idx))
        else:
            n_envs = 1

        if "ignore_joint_limit" in kwargs:
            gs.logger.warning("`ignore_joint_limit` is deprecated")

        ee_link_idx = None
        if ee_link_name is not None:
            assert with_entity is not None, "`with_entity` must be specified."
            ee_link_idx = self.get_link(ee_link_name).idx
        if with_entity is not None:
            assert ee_link_name is not None, "reference link of the robot must be specified."
            assert len(with_entity.links) == 1, "only non-articulated object is supported for now."

        # import here to avoid circular import
        from genesis.utils.path_planning import RRT, RRTConnect

        match planner:
            case "RRT":
                planner_obj = RRT(self)
            case "RRTConnect":
                planner_obj = RRTConnect(self)
            case _:
                gs.raise_exception(f"invalid planner {planner} specified.")

        path = torch.empty((num_waypoints, n_envs, self.n_qs), dtype=gs.tc_float, device=gs.device)
        is_invalid = torch.ones((n_envs,), dtype=torch.bool, device=gs.device)
        for i in range(1 + max_retry):
            retry_path, retry_is_invalid = planner_obj.plan(
                qpos_goal,
                qpos_start=qpos_start,
                resolution=resolution,
                timeout=timeout,
                max_nodes=max_nodes,
                smooth_path=smooth_path,
                num_waypoints=num_waypoints,
                ignore_collision=ignore_collision,
                envs_idx=envs_idx,
                ee_link_idx=ee_link_idx,
                obj_entity=with_entity,
            )
            # NOTE: update the previously failed path with the new results
            path[:, is_invalid] = retry_path[:, is_invalid]

            is_invalid &= retry_is_invalid
            if not is_invalid.any():
                break
            gs.logger.info(f"Planning failed. Retrying for {is_invalid.sum()} environments...")

        if self._solver.n_envs == 0:
            if return_valid_mask:
                return path.squeeze(1), ~is_invalid[0]
            return path.squeeze(1)

        if return_valid_mask:
            return path, ~is_invalid
        return path


@ti.kernel(fastcache=False)
def kernel_rigid_entity_inverse_kinematics(
    rigid_entity: ti.template(),
    links_idx: ti.types.ndarray(),
    poss: ti.types.ndarray(),
    quats: ti.types.ndarray(),
    n_links: ti.i32,
    dofs_idx: ti.types.ndarray(),
    n_dofs: ti.i32,
    links_idx_by_dofs: ti.types.ndarray(),
    n_links_by_dofs: ti.i32,
    custom_init_qpos: ti.i32,
    init_qpos: ti.types.ndarray(),
    max_samples: ti.i32,
    max_solver_iters: ti.i32,
    damping: ti.f32,
    pos_tol: ti.f32,
    rot_tol: ti.f32,
    pos_mask_: ti.types.ndarray(),
    rot_mask_: ti.types.ndarray(),
    link_pos_mask: ti.types.ndarray(),
    link_rot_mask: ti.types.ndarray(),
    max_step_size: ti.f32,
    respect_joint_limit: ti.i32,
    envs_idx: ti.types.ndarray(),
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    joints_state: array_class.JointsState,
    joints_info: array_class.JointsInfo,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    EPS = rigid_global_info.EPS[None]

    # convert to ti Vector
    pos_mask = ti.Vector([pos_mask_[0], pos_mask_[1], pos_mask_[2]], dt=gs.ti_float)
    rot_mask = ti.Vector([rot_mask_[0], rot_mask_[1], rot_mask_[2]], dt=gs.ti_float)
    n_error_dims = 6 * n_links

    for i_b_ in range(envs_idx.shape[0]):
        i_b = envs_idx[i_b_]

        # save original qpos
        for i_q in range(rigid_entity.n_qs):
            rigid_entity._IK_qpos_orig[i_q, i_b] = rigid_global_info.qpos[i_q + rigid_entity._q_start, i_b]

        if custom_init_qpos:
            for i_q in range(rigid_entity.n_qs):
                rigid_global_info.qpos[i_q + rigid_entity._q_start, i_b] = init_qpos[i_b_, i_q]

        for i_error in range(n_error_dims):
            rigid_entity._IK_err_pose_best[i_error, i_b] = 1e4

        solved = False
        for i_sample in range(max_samples):
            for _ in range(max_solver_iters):
                # run FK to update link states using current q
                gs.engine.solvers.rigid.rigid_solver.func_forward_kinematics_entity(
                    rigid_entity._idx_in_solver,
                    i_b,
                    links_state,
                    links_info,
                    joints_state,
                    joints_info,
                    dofs_state,
                    dofs_info,
                    entities_info,
                    rigid_global_info,
                    static_rigid_sim_config,
                    is_backward=False,
                )
                # compute error
                solved = True
                for i_ee in range(n_links):
                    i_l_ee = links_idx[i_ee]

                    tgt_pos_i = ti.Vector([poss[i_ee, i_b_, 0], poss[i_ee, i_b_, 1], poss[i_ee, i_b_, 2]])
                    err_pos_i = tgt_pos_i - links_state.pos[i_l_ee, i_b]
                    for k in range(3):
                        err_pos_i[k] *= pos_mask[k] * link_pos_mask[i_ee]
                    if err_pos_i.norm() > pos_tol:
                        solved = False

                    tgt_quat_i = ti.Vector(
                        [quats[i_ee, i_b_, 0], quats[i_ee, i_b_, 1], quats[i_ee, i_b_, 2], quats[i_ee, i_b_, 3]]
                    )
                    err_rot_i = gu.ti_quat_to_rotvec(
                        gu.ti_transform_quat_by_quat(gu.ti_inv_quat(links_state.quat[i_l_ee, i_b]), tgt_quat_i), EPS
                    )
                    for k in range(3):
                        err_rot_i[k] *= rot_mask[k] * link_rot_mask[i_ee]
                    if err_rot_i.norm() > rot_tol:
                        solved = False

                    # put into multi-link error array
                    for k in range(3):
                        rigid_entity._IK_err_pose[i_ee * 6 + k, i_b] = err_pos_i[k]
                        rigid_entity._IK_err_pose[i_ee * 6 + k + 3, i_b] = err_rot_i[k]

                if solved:
                    break

                # compute multi-link jacobian
                for i_ee in range(n_links):
                    # update jacobian for ee link
                    i_l_ee = links_idx[i_ee]
                    rigid_entity._func_get_jacobian(
                        tgt_link_idx=i_l_ee,
                        i_b=i_b,
                        p_local=ti.Vector.zero(gs.ti_float, 3),
                        pos_mask=pos_mask,
                        rot_mask=rot_mask,
                        dofs_info=dofs_info,
                        joints_info=joints_info,
                        links_info=links_info,
                        links_state=links_state,
                    )  # NOTE: we still compute jacobian for all dofs as we haven't found a clean way to implement this

                    # copy to multi-link jacobian (only for the effective n_dofs instead of self.n_dofs)
                    for i_dof in range(n_dofs):
                        for i_error in ti.static(range(6)):
                            i_row = i_ee * 6 + i_error
                            i_dof_ = dofs_idx[i_dof]
                            rigid_entity._IK_jacobian[i_row, i_dof, i_b] = rigid_entity._jacobian[i_error, i_dof_, i_b]

                # compute dq = jac.T @ inverse(jac @ jac.T + diag) @ error (only for the effective n_dofs instead of self.n_dofs)
                lu.mat_transpose(rigid_entity._IK_jacobian, rigid_entity._IK_jacobian_T, n_error_dims, n_dofs, i_b)
                lu.mat_mul(
                    rigid_entity._IK_jacobian,
                    rigid_entity._IK_jacobian_T,
                    rigid_entity._IK_mat,
                    n_error_dims,
                    n_dofs,
                    n_error_dims,
                    i_b,
                )
                lu.mat_add_eye(rigid_entity._IK_mat, damping**2, n_error_dims, i_b)
                lu.mat_inverse(
                    rigid_entity._IK_mat,
                    rigid_entity._IK_L,
                    rigid_entity._IK_U,
                    rigid_entity._IK_y,
                    rigid_entity._IK_inv,
                    n_error_dims,
                    i_b,
                )
                lu.mat_mul_vec(
                    rigid_entity._IK_inv,
                    rigid_entity._IK_err_pose,
                    rigid_entity._IK_vec,
                    n_error_dims,
                    n_error_dims,
                    i_b,
                )

                for i_d in range(rigid_entity.n_dofs):  # IK_delta_qpos = IK_jacobian_T @ IK_vec
                    rigid_entity._IK_delta_qpos[i_d, i_b] = 0
                for i_d in range(n_dofs):
                    for j in range(n_error_dims):
                        # NOTE: IK_delta_qpos uses the original indexing instead of the effective n_dofs
                        i_d_ = dofs_idx[i_d]
                        rigid_entity._IK_delta_qpos[i_d_, i_b] += (
                            rigid_entity._IK_jacobian_T[i_d, j, i_b] * rigid_entity._IK_vec[j, i_b]
                        )

                for i_d in range(rigid_entity.n_dofs):
                    rigid_entity._IK_delta_qpos[i_d, i_b] = ti.math.clamp(
                        rigid_entity._IK_delta_qpos[i_d, i_b], -max_step_size, max_step_size
                    )

                # update q
                gs.engine.solvers.rigid.rigid_solver.func_integrate_dq_entity(
                    rigid_entity._IK_delta_qpos,
                    rigid_entity._idx_in_solver,
                    i_b,
                    respect_joint_limit,
                    links_info,
                    joints_info,
                    dofs_info,
                    entities_info,
                    rigid_global_info,
                    static_rigid_sim_config,
                )

            if not solved:
                # re-compute final error if exited not due to solved
                gs.engine.solvers.rigid.rigid_solver.func_forward_kinematics_entity(
                    rigid_entity._idx_in_solver,
                    i_b,
                    links_state,
                    links_info,
                    joints_state,
                    joints_info,
                    dofs_state,
                    dofs_info,
                    entities_info,
                    rigid_global_info,
                    static_rigid_sim_config,
                    is_backward=False,
                )
                solved = True
                for i_ee in range(n_links):
                    i_l_ee = links_idx[i_ee]

                    tgt_pos_i = ti.Vector([poss[i_ee, i_b_, 0], poss[i_ee, i_b_, 1], poss[i_ee, i_b_, 2]])
                    err_pos_i = tgt_pos_i - links_state.pos[i_l_ee, i_b]
                    for k in range(3):
                        err_pos_i[k] *= pos_mask[k] * link_pos_mask[i_ee]
                    if err_pos_i.norm() > pos_tol:
                        solved = False

                    tgt_quat_i = ti.Vector(
                        [quats[i_ee, i_b_, 0], quats[i_ee, i_b_, 1], quats[i_ee, i_b_, 2], quats[i_ee, i_b_, 3]]
                    )
                    err_rot_i = gu.ti_quat_to_rotvec(
                        gu.ti_transform_quat_by_quat(gu.ti_inv_quat(links_state.quat[i_l_ee, i_b]), tgt_quat_i), EPS
                    )
                    for k in range(3):
                        err_rot_i[k] *= rot_mask[k] * link_rot_mask[i_ee]
                    if err_rot_i.norm() > rot_tol:
                        solved = False

                    # put into multi-link error array
                    for k in range(3):
                        rigid_entity._IK_err_pose[i_ee * 6 + k, i_b] = err_pos_i[k]
                        rigid_entity._IK_err_pose[i_ee * 6 + k + 3, i_b] = err_rot_i[k]

            if solved:
                for i_q in range(rigid_entity.n_qs):
                    rigid_entity._IK_qpos_best[i_q, i_b] = rigid_global_info.qpos[i_q + rigid_entity._q_start, i_b]
                for i_error in range(n_error_dims):
                    rigid_entity._IK_err_pose_best[i_error, i_b] = rigid_entity._IK_err_pose[i_error, i_b]
                break

            else:
                # copy to _IK_qpos if this sample is better
                improved = True
                for i_ee in range(n_links):
                    error_pos_i = ti.Vector(
                        [rigid_entity._IK_err_pose[i_ee * 6 + i_error, i_b] for i_error in range(3)]
                    )
                    error_rot_i = ti.Vector(
                        [rigid_entity._IK_err_pose[i_ee * 6 + i_error, i_b] for i_error in range(3, 6)]
                    )
                    error_pos_best = ti.Vector(
                        [rigid_entity._IK_err_pose_best[i_ee * 6 + i_error, i_b] for i_error in range(3)]
                    )
                    error_rot_best = ti.Vector(
                        [rigid_entity._IK_err_pose_best[i_ee * 6 + i_error, i_b] for i_error in range(3, 6)]
                    )
                    if error_pos_i.norm() > error_pos_best.norm() or error_rot_i.norm() > error_rot_best.norm():
                        improved = False
                        break

                if improved:
                    for i_q in range(rigid_entity.n_qs):
                        rigid_entity._IK_qpos_best[i_q, i_b] = rigid_global_info.qpos[i_q + rigid_entity._q_start, i_b]
                    for i_error in range(n_error_dims):
                        rigid_entity._IK_err_pose_best[i_error, i_b] = rigid_entity._IK_err_pose[i_error, i_b]

                # Resample init q
                if respect_joint_limit and i_sample < max_samples - 1:
                    for i_l_ in range(n_links_by_dofs):
                        i_l = links_idx_by_dofs[i_l_]
                        I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l

                        for i_j in range(links_info.joint_start[I_l], links_info.joint_end[I_l]):
                            I_j = [i_j, i_b] if ti.static(static_rigid_sim_config.batch_joints_info) else i_j
                            i_d = joints_info.dof_start[I_j]
                            I_d = [i_d, i_b] if ti.static(static_rigid_sim_config.batch_dofs_info) else i_d

                            dof_limit = dofs_info.limit[I_d]
                            if (
                                joints_info.type[I_j] == gs.JOINT_TYPE.REVOLUTE
                                or joints_info.type[I_j] == gs.JOINT_TYPE.PRISMATIC
                            ) and not (ti.math.isinf(dof_limit[0]) or ti.math.isinf(dof_limit[1])):
                                q_start = joints_info.q_start[I_j]
                                rigid_global_info.qpos[q_start, i_b] = dof_limit[0] + ti.random() * (
                                    dof_limit[1] - dof_limit[0]
                                )
                else:
                    pass  # When respect_joint_limit=False, we can simply continue from the last solution

        # restore original qpos and link state
        for i_q in range(rigid_entity.n_qs):
            rigid_global_info.qpos[i_q + rigid_entity._q_start, i_b] = rigid_entity._IK_qpos_orig[i_q, i_b]
        gs.engine.solvers.rigid.rigid_solver.func_forward_kinematics_entity(
            rigid_entity._idx_in_solver,
            i_b,
            links_state,
            links_info,
            joints_state,
            joints_info,
            dofs_state,
            dofs_info,
            entities_info,
            rigid_global_info,
            static_rigid_sim_config,
            is_backward=False,
        )
