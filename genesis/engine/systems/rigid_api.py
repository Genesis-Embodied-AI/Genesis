"""User-facing queries and setters of the rigid bodies, outside every simulation pipeline."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
import torch

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
from genesis.constants import link_ref_frame
from genesis.engine.core import Find, Require, System, register_system
from genesis.engine.entities import RigidEntity
from genesis.engine.solvers.base_solver import kernel_set_gravity
from genesis.engine.solvers.rigid.kinematic_tools import offset_world_shift as _offset_world_shift
from genesis.engine.solvers.rigid.kinematic_tools import select_links_offset as _select_links_offset
from genesis.engine.solvers.rigid.abd.accessor import (
    ConstraintType,
    kernel_control_dofs_force,
    kernel_control_dofs_position,
    kernel_control_dofs_position_velocity,
    kernel_control_dofs_velocity,
    kernel_get_dofs_control_force,
    kernel_get_links_acc,
    kernel_get_links_vel,
    kernel_get_terrain_height,
    kernel_set_dofs_act_bias,
    kernel_set_dofs_act_gain,
    kernel_set_dofs_armature,
    kernel_set_dofs_damping,
    kernel_set_dofs_force_range,
    kernel_set_dofs_frictionloss,
    kernel_set_dofs_kp,
    kernel_set_dofs_kv,
    kernel_set_dofs_limit,
    kernel_set_dofs_position,
    kernel_set_dofs_stiffness,
    kernel_set_dofs_velocity,
    kernel_set_dofs_zero_velocity,
    kernel_set_drone_rpm,
    kernel_set_geom_friction,
    kernel_set_geom_friction_rolling,
    kernel_set_geom_friction_torsional,
    kernel_set_geoms_friction,
    kernel_set_geoms_friction_ratio,
    kernel_set_geoms_friction_rolling,
    kernel_set_geoms_friction_torsional,
    kernel_set_global_sol_params,
    kernel_set_links_COM,
    kernel_set_links_inertia,
    kernel_set_links_mass,
    kernel_set_links_pos,
    kernel_set_links_quat,
    kernel_set_qpos,
    kernel_set_sol_params,
    kernel_set_vverts,
    kernel_update_drone_propeller_vgeoms,
)
from genesis.engine.solvers.rigid.abd.forward_kinematics import (
    kernel_update_verts_for_geoms,
    kernel_update_vgeoms,
    kernel_update_vverts_for_vgeoms,
)
from genesis.engine.solvers.rigid.abd.inverse_kinematics import (
    kernel_forward_kinematics_query,
    kernel_get_jacobian,
    kernel_get_jacobian_zero,
    kernel_inverse_kinematics_entity,
    kernel_set_ik_targets,
)
from genesis.engine.solvers.rigid.abd.misc import (
    kernel_apply_links_external_wrench,
    kernel_apply_links_external_wrench_at_pos,
    kernel_set_zero,
)
from genesis.engine.solvers.rigid.rigid_tools import (
    sanitize_io_variables,
    sanitize_ref_frame,
    sanitize_sol_params,
)
from genesis.utils.misc import (
    DeprecationError,
    assign_indexed_tensor,
    broadcast_tensor,
    indices_to_mask,
    qd_to_torch,
    tensor_to_array,
)

from ..simulator_engine import SimulatorEngine
from .rigid_constraint import RigidConstraint
from .rigid_dynamics import RigidDynamics
from .rigid_frictionloss_constraint import RigidFrictionLossConstraint
from .rigid_island import RigidIsland

if TYPE_CHECKING:
    from genesis.engine.scene import Scene

TensorLike: TypeAlias = ArrayLike | torch.Tensor
IndexLike: TypeAlias = ArrayLike | torch.Tensor | None
HostTensor: TypeAlias = NDArray[np.generic] | torch.Tensor

IS_OLD_TORCH = tuple(map(int, torch.__version__.split(".")[:2])) < (2, 8)


class RigidAPI(System):
    """The user-facing queries and setters of the rigid bodies, which the public facade delegates to.

    RigidAPI takes part in no simulation pipeline and declares no Action. It writes the primary data of RigidDynamics
    with Layer-1 functions, and requests the derived state from its owners after a write: RigidDynamics maintains the
    cartesian state, its forward-state flags and the inverse weights, and RigidIsland the hibernation state.
    """

    _TERRAIN_HEIGHT_QUERY_TILT_TOLERANCE = 1e-3

    # -------------------------------- Dependencies --------------------------------

    rigid_dynamics = Require(RigidDynamics)
    rigid_island = Require(RigidIsland)
    rigid_constraint = Find(RigidConstraint)
    rigid_frictionloss_constraint = Find(RigidFrictionLossConstraint)

    def __init__(self, scene: "Scene") -> None:
        super().__init__()
        self._scene = scene
        self._kinematics_scratch_cache: array_class.KinematicsScratch | None = None
        self._ik_scratch_cache: array_class.IKScratch | None = None

    # -------------------------------- Public get/set API --------------------------------

    def get_terrain_height(self, positions: TensorLike, link_idx: int, envs_idx: IndexLike = None) -> torch.Tensor:
        dynamics = self.rigid_dynamics
        terrain = dynamics.links[link_idx].entity

        positions = torch.as_tensor(positions, dtype=gs.tc_float, device=gs.device)
        if positions.ndim == 0 or positions.ndim > 3 or positions.shape[-1] != 2:
            gs.raise_exception("`positions` must have shape (2,), (n_points, 2), or (n_envs, n_points, 2).")

        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        n_envs = len(envs_idx)
        is_single_position = positions.ndim == 1
        n_points = 1 if is_single_position else positions.shape[-2]
        if n_points == 0:
            gs.raise_exception("`positions` must contain at least one position.")

        is_per_env = positions.ndim == 3 and positions.shape[0] != 1
        positions = broadcast_tensor(
            positions, gs.tc_float, (n_envs if is_per_env else 1, n_points, 2), ("envs_idx", "positions", "")
        )

        heights = torch.empty((n_envs, n_points), dtype=gs.tc_float, device=gs.device)
        kernel_get_terrain_height(
            envs_idx,
            link_idx,
            terrain._morph.horizontal_scale,
            positions.contiguous(),
            terrain._terrain_height_field,
            heights,
            dynamics.data.dyn_state,
            dynamics.rigid_config,
            tilt_tolerance=self._TERRAIN_HEIGHT_QUERY_TILT_TOLERANCE,
            is_per_env=is_per_env,
        )

        if dynamics.n_envs == 0:
            heights = heights[0]
        if is_single_position:
            heights = heights[..., 0]
        return heights

    def get_links_quat(
        self, links_idx: IndexLike = None, envs_idx: IndexLike = None, *, relative: bool = False
    ) -> torch.Tensor:
        dynamics = self.rigid_dynamics
        tensor = qd_to_torch(dynamics.data.dyn_state.links.quat, envs_idx, links_idx, transpose=True, copy=True)
        idx = links_idx if isinstance(links_idx, int) else slice(None)
        if relative and not dynamics.links_offset_quat_is_identity[idx].all():
            offset_quat = _select_links_offset(dynamics.links_offset_quat, links_idx, envs_idx)
            tensor = gu.transform_quat_by_quat(gu.inv_quat(offset_quat), tensor)
        return tensor[0] if dynamics.n_envs == 0 else tensor

    def get_joints_anchor_pos(self, joints_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        dynamics = self.rigid_dynamics
        tensor = qd_to_torch(dynamics.data.dyn_state.joints.xanchor, envs_idx, joints_idx, transpose=True, copy=True)
        return tensor[0] if dynamics.n_envs == 0 else tensor

    def get_joints_anchor_axis(self, joints_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        dynamics = self.rigid_dynamics
        tensor = qd_to_torch(dynamics.data.dyn_state.joints.xaxis, envs_idx, joints_idx, transpose=True, copy=True)
        return tensor[0] if dynamics.n_envs == 0 else tensor

    def get_vgeoms_pos(
        self, vgeoms_idx: IndexLike = None, envs_idx: IndexLike = None, *, relative: bool = False
    ) -> torch.Tensor:
        dynamics = self.rigid_dynamics
        tensor = qd_to_torch(dynamics.data.dyn_state.vgeoms.pos, envs_idx, vgeoms_idx, transpose=True, copy=True)
        if relative and dynamics.vgeoms_offset_pos is not None:
            quat = qd_to_torch(dynamics.data.dyn_state.vgeoms.quat, envs_idx, vgeoms_idx, transpose=True, copy=True)
            offset_pos = dynamics.vgeoms_offset_pos if vgeoms_idx is None else dynamics.vgeoms_offset_pos[vgeoms_idx]
            offset_quat = dynamics.vgeoms_offset_quat if vgeoms_idx is None else dynamics.vgeoms_offset_quat[vgeoms_idx]
            tensor -= _offset_world_shift(offset_pos, offset_quat, quat)
        return tensor[0] if dynamics.n_envs == 0 else tensor

    def get_vgeoms_quat(
        self, vgeoms_idx: IndexLike = None, envs_idx: IndexLike = None, *, relative: bool = False
    ) -> torch.Tensor:
        dynamics = self.rigid_dynamics
        tensor = qd_to_torch(dynamics.data.dyn_state.vgeoms.quat, envs_idx, vgeoms_idx, transpose=True, copy=True)
        if relative and dynamics.vgeoms_offset_quat is not None:
            offset_quat = dynamics.vgeoms_offset_quat if vgeoms_idx is None else dynamics.vgeoms_offset_quat[vgeoms_idx]
            tensor = gu.transform_quat_by_quat(gu.inv_quat(offset_quat), tensor)
        return tensor[0] if dynamics.n_envs == 0 else tensor

    def get_links_ang(self, links_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        dynamics = self.rigid_dynamics
        tensor = qd_to_torch(dynamics.data.dyn_state.links.cd_ang, envs_idx, links_idx, transpose=True, copy=True)
        return tensor[0] if dynamics.n_envs == 0 else tensor

    def get_qpos(self, qs_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        dynamics = self.rigid_dynamics
        tensor = qd_to_torch(dynamics.data.rigid_info.qpos, envs_idx, qs_idx, transpose=True, copy=True)
        return tensor[0] if dynamics.n_envs == 0 else tensor

    def get_dofs_velocity(self, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        dynamics = self.rigid_dynamics
        tensor = qd_to_torch(dynamics.data.dyn_state.dofs.vel, envs_idx, dofs_idx, transpose=True, copy=True)
        return tensor[0] if dynamics.n_envs == 0 else tensor

    def get_dofs_position(self, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        """Read current DOF positions."""
        dynamics = self.rigid_dynamics
        tensor = qd_to_torch(dynamics.data.dyn_state.dofs.pos, envs_idx, dofs_idx, transpose=True, copy=True)
        return tensor[0] if dynamics.n_envs == 0 else tensor

    def get_dofs_limit(
        self, dofs_idx: IndexLike = None, envs_idx: IndexLike = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dynamics = self.rigid_dynamics
        if not dynamics.options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        tensor = qd_to_torch(dynamics.data.dyn_info.dofs.limit, envs_idx, dofs_idx, transpose=True, copy=True)
        if dynamics.n_envs == 0 and dynamics.options.batch_dofs_info:
            tensor = tensor[0]
        return tensor[..., 0], tensor[..., 1]

    def update_vgeoms(self) -> None:
        dynamics = self.rigid_dynamics
        kernel_update_vgeoms(dynamics.data.dyn_state, dynamics.data.dyn_info, dynamics.rigid_config)

    def update_vverts_for_vgeoms(self, vgeoms_idx: IndexLike) -> None:
        """Refresh the vverts_state.pos slice for the requested vgeoms by re-running FK.

        Used by set_vverts(None, ...) and at scene-build time to initialize the custom buffer with a meaningful pose.
        The kernel is a no-op for vgeoms whose vverts have no state slot.
        """
        dynamics = self.rigid_dynamics
        if dynamics.n_custom_vverts == 0:
            return
        kernel_update_vverts_for_vgeoms(
            vgeoms_idx, dynamics.data.dyn_state, dynamics.data.dyn_info, dynamics.rigid_config
        )

    def set_vverts(
        self,
        custom_vvert_start: int,
        custom_vvert_end: int,
        vgeoms_idx: IndexLike,
        vverts: TensorLike | None,
        envs_idx: IndexLike = None,
    ) -> None:
        """Write the slice [custom_vvert_start:custom_vvert_end] of vverts_state.pos.

        vverts=None re-populates the slice from FK by running update_vverts_for_vgeoms over the vgeoms owning the slice.
        Otherwise vverts is broadcast to the slice shape and written directly. vverts_state.pos is always batched;
        envs_idx selects which envs to write to.
        """
        dynamics = self.rigid_dynamics
        if vverts is None:
            self.update_vverts_for_vgeoms(vgeoms_idx)
            return

        if gs.use_zerocopy:
            data = qd_to_torch(dynamics.data.dyn_state.vverts.pos, transpose=True, copy=False)
            if isinstance(envs_idx, torch.Tensor) and envs_idx.dtype == torch.bool:
                pos_slice = data[:, custom_vvert_start:custom_vvert_end]
                if vverts.ndim == 3 and len(vverts) != len(pos_slice):
                    pos_slice.masked_scatter_(envs_idx[:, None, None], vverts.view_as(vverts))
                else:
                    vverts_b = broadcast_tensor(vverts, gs.tc_float, pos_slice.shape)
                    torch.where(envs_idx[:, None, None], vverts_b, pos_slice, out=pos_slice)
            else:
                vverts_mask = indices_to_mask(slice(custom_vvert_start, custom_vvert_end))
                pos_mask = (0, *vverts_mask) if dynamics.n_envs == 0 else indices_to_mask(envs_idx, *vverts_mask)
                assign_indexed_tensor(data, pos_mask, vverts)
            return

        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        target_shape = (envs_idx.shape[0], custom_vvert_end - custom_vvert_start, 3)
        vverts = broadcast_tensor(vverts, gs.tc_float, target_shape, ("envs", "vverts", "xyz")).contiguous()
        kernel_set_vverts(custom_vvert_start, envs_idx, vverts, dynamics.data.dyn_state, dynamics.rigid_config)

    def get_vverts(self, custom_vvert_start: int, custom_vvert_end: int, envs_idx: IndexLike = None) -> torch.Tensor:
        """Return a copy of the vverts_state.pos slice for the given custom-vvert range.

        Shape: (len(envs_idx), custom_vvert_end - custom_vvert_start, 3). envs_idx=None returns every env.
        """
        dynamics = self.rigid_dynamics
        tensor = qd_to_torch(
            dynamics.data.dyn_state.vverts.pos,
            envs_idx,
            slice(custom_vvert_start, custom_vvert_end),
            transpose=True,
            copy=True,
        )
        return tensor[0] if dynamics.n_envs == 0 else tensor

    def get_links_jacobian(
        self,
        link_idx: int,
        dof_start: int,
        n_dofs: int,
        local_point: TensorLike | None = None,
    ) -> torch.Tensor:
        """Spatial Jacobian of a link-local point, as (n_envs, 6, n_dofs), for the entity owning that link.

        dof_start offsets the entity degrees of freedom into the solver and n_dofs narrows the shared buffer down to
        that entity. local_point defaults to the link origin.
        """
        dynamics = self.rigid_dynamics
        jacobian = self._kinematics_scratch.jacobian
        if local_point is None:
            kernel_get_jacobian_zero(
                link_idx,
                dof_start,
                jacobian,
                dynamics.data.dyn_state,
                dynamics.data.dyn_info,
                dynamics.rigid_config,
                dynamics.batch_size,
            )
        else:
            kernel_get_jacobian(
                link_idx,
                dof_start,
                local_point,
                jacobian,
                dynamics.data.dyn_state,
                dynamics.data.dyn_info,
                dynamics.rigid_config,
                dynamics.batch_size,
            )
        return qd_to_torch(jacobian, transpose=True, copy=True)[..., :n_dofs]

    def forward_kinematics_query(
        self, entity: RigidEntity, qpos: TensorLike, envs_idx: IndexLike = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Link poses one entity would have at the given configuration, leaving the live configuration as it was.

        Distinct from the forward kinematics the step propagates: this evaluates a configuration the solver does not
        hold, and restores the one it does. Returns positions (n_envs, n_links, 3) and orientations
        (n_envs, n_links, 4), without the batch dimension when the scene is not batched.
        """
        dynamics = self.rigid_dynamics
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        qpos = broadcast_tensor(qpos, gs.tc_float, (len(envs_idx), entity.n_qs), ("envs_idx", "qs_idx")).contiguous()

        qs_idx = torch.arange(entity._q_start, entity._q_start + entity.n_qs, dtype=gs.tc_int, device=gs.device)
        links_idx = torch.arange(
            entity._link_start, entity._link_start + entity.n_links, dtype=gs.tc_int, device=gs.device
        )
        links_pos = torch.empty((len(envs_idx), entity.n_links, 3), dtype=gs.tc_float, device=gs.device)
        links_quat = torch.empty((len(envs_idx), entity.n_links, 4), dtype=gs.tc_float, device=gs.device)
        kernel_forward_kinematics_query(
            entity._idx_in_solver,
            qs_idx,
            links_idx,
            envs_idx,
            links_pos,
            links_quat,
            qpos,
            self._kinematics_scratch.qpos_cache,
            dynamics.data.dyn_state,
            dynamics.data.dyn_info,
            dynamics.data.rigid_info,
            dynamics.rigid_config,
        )

        if dynamics.n_envs == 0:
            links_pos = links_pos[0]
            links_quat = links_quat[0]
        return links_pos, links_quat

    def inverse_kinematics(
        self,
        entity_idx: int,
        q_start: int,
        link_start: int,
        joint_start: int,
        links_idx: torch.Tensor,
        dofs_idx: torch.Tensor,
        envs_idx: torch.Tensor,
        poss: torch.Tensor,
        quats: torch.Tensor,
        local_points: torch.Tensor,
        init_qpos: torch.Tensor,
        pos_mask: torch.Tensor,
        rot_mask: torch.Tensor,
        link_pos_mask: torch.Tensor,
        link_rot_mask: torch.Tensor,
        n_qs: int,
        n_dofs: int,
        custom_init_qpos: bool,
        max_samples: int,
        max_solver_iters: int,
        damping: float,
        pos_tol: float,
        rot_tol: float,
        max_step_size: float,
        seed: int,
        respect_joint_limit: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Damped-least-squares inverse kinematics of one entity for the given link targets.

        Returns the best configuration per environment, spanning the entity, and the stacked pose residual of every
        target it carries. Callers narrow both to the environments and targets they asked for.
        """
        dynamics = self.rigid_dynamics
        n_links = len(links_idx)
        n_tgts = dynamics.options.IK_max_targets
        if n_links > n_tgts:
            gs.raise_exception(f"Cannot solve for {n_links} targets. Raise 'IK_max_targets' above {n_tgts}.")

        ik_state, ik_fk, targets = self._ik_scratch

        # Only the leading rows each buffer holds for this solve are written, and the kernel reads no further.
        if gs.use_zerocopy:
            for member, data in (
                (targets.links_idx, links_idx),
                (targets.local_point, local_points),
                (targets.link_pos_mask, link_pos_mask),
                (targets.link_rot_mask, link_rot_mask),
            ):
                member_t = qd_to_torch(member, copy=False)
                member_t[:n_links] = data
            dofs_idx_t = qd_to_torch(targets.dofs_idx, copy=False)
            dofs_idx_t[: len(dofs_idx)] = dofs_idx
            envs_idx_t = qd_to_torch(targets.envs_idx, copy=False)
            envs_idx_t[: len(envs_idx)] = envs_idx
            pos_mask_t = qd_to_torch(targets.pos_mask, copy=False)
            pos_mask_t[:] = pos_mask
            rot_mask_t = qd_to_torch(targets.rot_mask, copy=False)
            rot_mask_t[:] = rot_mask
            for member, data in ((targets.pos, poss), (targets.quat, quats)):
                member_t = qd_to_torch(member, copy=False)
                member_t[:n_links, envs_idx] = data
            init_qpos_t = qd_to_torch(targets.init_qpos, copy=False)
            init_qpos_t[envs_idx, :n_qs] = init_qpos
            if gs.backend == gs.metal:
                torch.mps.synchronize()
        else:
            kernel_set_ik_targets(
                links_idx,
                dofs_idx,
                envs_idx,
                poss,
                quats,
                local_points,
                init_qpos,
                pos_mask,
                rot_mask,
                link_pos_mask,
                link_rot_mask,
                targets,
            )

        kernel_inverse_kinematics_entity(
            entity_idx,
            q_start,
            link_start,
            joint_start,
            dynamics.data.dyn_state,
            ik_state,
            ik_fk,
            targets,
            dynamics.data.dyn_info,
            dynamics.data.rigid_info,
            dynamics.rigid_config,
            n_qs,
            n_dofs,
            n_links,
            len(dofs_idx),
            len(envs_idx),
            custom_init_qpos,
            max_samples,
            max_solver_iters,
            damping,
            pos_tol,
            rot_tol,
            max_step_size,
            seed,
            respect_joint_limit,
        )

        qpos = qd_to_torch(ik_state.qpos_best, transpose=True, copy=True)[..., :n_qs]
        err_pose = qd_to_torch(ik_state.err_pose_best, transpose=True, copy=True).reshape((-1, n_tgts, 6))
        return qpos, err_pose[:, :n_links]

    def set_gravity(self, gravity: TensorLike, envs_idx: IndexLike = None) -> None:
        """
        Set the gravity acting on the bodies of this solver, in m/s^2.

        Parameters
        ----------
        gravity : array_like, shape (3,) or (n_envs, 3)
            The gravity vector, one per environment when several are named.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        dynamics = self.rigid_dynamics
        if dynamics.data.rigid_info.gravity is None:
            gs.raise_exception(f"{type(self).__name__} holds no body for a gravity to act on.")
        # Indices are not checked on purpose: deciding to raise means reading a comparison back from the device, which
        # syncs, and this may run every step.
        if gs.use_zerocopy:
            envs_mask = indices_to_mask(envs_idx)
            gravity_t = qd_to_torch(dynamics.data.rigid_info.gravity, transpose=True, copy=False)
            assign_indexed_tensor(
                gravity_t, envs_mask, broadcast_tensor(gravity, gs.tc_float, gravity_t[envs_mask].shape)
            )
            if gs.backend == gs.metal:
                torch.mps.synchronize()
            return

        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        gravity = broadcast_tensor(gravity, gs.tc_float, (len(envs_idx), 3), ("envs_idx", ""))
        kernel_set_gravity(gravity.contiguous(), envs_idx, dynamics.data.rigid_info.gravity)

    def get_gravity(self, envs_idx: IndexLike = None) -> torch.Tensor:
        """
        Get the gravity acting on the bodies of this solver, in m/s^2.

        Parameters
        ----------
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments are returned. Defaults to None.

        Returns
        -------
        gravity : torch.Tensor, shape (3,) or (n_envs, 3)
            The gravity vector of each environment.
        """
        dynamics = self.rigid_dynamics
        if dynamics.data.rigid_info.gravity is None:
            gs.raise_exception(f"{type(self).__name__} holds no body for a gravity to act on.")
        tensor = qd_to_torch(dynamics.data.rigid_info.gravity, envs_idx, transpose=True, copy=True)
        return tensor[0] if dynamics.n_envs == 0 else tensor

    def set_links_pos(self, pos: TensorLike, links_idx: IndexLike = None, envs_idx: IndexLike = None) -> None:
        raise DeprecationError("This method has been removed. Please use 'set_base_links_pos' instead.")

    def set_base_links_pos(
        self,
        pos: TensorLike,
        links_idx: IndexLike = None,
        envs_idx: IndexLike = None,
        *,
        relative: bool = False,
        skip_forward: bool = False,
    ) -> None:
        dynamics = self.rigid_dynamics
        if links_idx is None:
            links_idx = dynamics.base_links_idx

        # Without any offset position on the targeted links, the authored and internal link origins coincide, so a
        # relative set is just an absolute one.
        idx = links_idx if isinstance(links_idx, int) else slice(None)
        if relative and dynamics.links_offset_pos_is_identity[idx].all():
            relative = False

        # Map a single base link's authored position to world here (keeping the current orientation) so the zero-copy
        # in-place write below still applies, both for an environment-uniform and a per-environment offset. Multi-link
        # relative sets are composed in the kernel branch instead.
        if relative and isinstance(links_idx, int):
            cur_quat = self.get_links_quat(links_idx, envs_idx, relative=False)[..., 0, :]
            offset_pos = _select_links_offset(dynamics.links_offset_pos, links_idx, envs_idx)[..., 0, :]
            offset_quat = _select_links_offset(dynamics.links_offset_quat, links_idx, envs_idx)[..., 0, :]
            pos = torch.as_tensor(pos, dtype=gs.tc_float, device=gs.device) + _offset_world_shift(
                offset_pos, offset_quat, cur_quat
            )
            relative = False

        # Zero-copy fast path: single base link, non-relative. Write the position buffer in place instead of
        # launching a kernel. The kernel path below handles relative or multi-link updates, as well as waking up
        # hibernated entities, which is required whenever hibernation is enabled.
        if (
            gs.use_zerocopy
            and not relative
            and isinstance(links_idx, int)
            and not dynamics.rigid_config.use_hibernation
        ):
            link = dynamics.links[links_idx]
            if link.is_fixed:
                data = qd_to_torch(dynamics.data.dyn_state.links.pos, transpose=True, copy=False)
                target = data[:, links_idx]
            else:
                data = qd_to_torch(dynamics.data.rigid_info.qpos, transpose=True, copy=False)
                target = data[:, link.q_start : link.q_start + 3]
            if isinstance(envs_idx, torch.Tensor) and envs_idx.dtype == torch.bool:
                if pos.ndim == 2 and len(pos) not in (1, len(target)):
                    # A fresh source view is needed because masked_scatter_ may reshape it in-place. Metal
                    # mis-scatters a stride-0 broadcast mask, so it must be materialized to a dense mask there.
                    envs_mask = envs_idx[:, None]
                    if gs.backend == gs.metal:
                        envs_mask = envs_mask.expand_as(target).contiguous()
                    target.masked_scatter_(envs_mask, pos.view_as(pos))
                else:
                    pos = broadcast_tensor(pos, gs.tc_float, target.shape)
                    torch.where(envs_idx[:, None], pos, target, out=target)
            else:
                # Fixed links with at least one geom and non-batched vertices cannot take env-specific positions
                if link.is_fixed and link.geoms and not link.entity._batch_fixed_verts:
                    pos = torch.as_tensor(pos, dtype=gs.tc_float, device=gs.device)
                    same_pos = pos.ndim < 2 or len(pos) == 1 or (torch.diff(pos, dim=0).abs() < gs.EPS).all()
                    set_all_envs = envs_idx is None or torch.equal(
                        torch.sort(self._scene._sanitize_envs_idx(envs_idx)).values, self._scene._envs_idx
                    )
                    if not (set_all_envs and same_pos):
                        gs.raise_exception(
                            "Specifying env-specific pos for fixed links with at least one geometry requires "
                            "setting morph option 'batch_fixed_verts=True'."
                        )
                mask = (0,) if dynamics.n_envs == 0 else indices_to_mask(envs_idx)
                assign_indexed_tensor(target, mask, pos)
            if gs.backend == gs.metal:
                torch.mps.synchronize()
        else:
            pos, links_idx, envs_idx = sanitize_io_variables(
                self._scene, pos, links_idx, dynamics.n_links, "links_idx", envs_idx, (3,), skip_allocation=True
            )
            if dynamics.n_envs == 0:
                pos = pos[None]

            if relative:
                # Compose the body-frame offset onto the authored position, keeping the current orientation, then set
                # the resulting world position absolutely.
                cur_quat = qd_to_torch(
                    dynamics.data.dyn_state.links.quat, envs_idx, links_idx, transpose=True, copy=True
                )
                offset_pos = _select_links_offset(dynamics.links_offset_pos, links_idx, envs_idx)
                offset_quat = _select_links_offset(dynamics.links_offset_quat, links_idx, envs_idx)
                pos = pos + _offset_world_shift(offset_pos, offset_quat, cur_quat)
                relative = False

            # Raise exception for fixed links with at least one geom and non-batched fixed vertices, except if setting
            # same location for all envs at once
            set_all_envs = torch.equal(torch.sort(envs_idx).values, self._scene._envs_idx)
            has_fixed_verts = any(
                link.is_fixed and link.geoms and not link.entity._batch_fixed_verts
                for link in (dynamics.links[i_l] for i_l in links_idx)
            )
            if has_fixed_verts and not (set_all_envs and (torch.diff(pos, dim=0).abs() < gs.EPS).all()):
                gs.raise_exception(
                    "Specifying env-specific pos for fixed links with at least one geometry requires setting morph "
                    "option 'batch_fixed_verts=True'."
                )

            # Wake up hibernated entities before setting position (fixed links don't need wake-up)
            if dynamics.rigid_config.use_hibernation and not all(dynamics.links[i_l].is_fixed for i_l in links_idx):
                self.rigid_island.wake_entities_by_links(links_idx, envs_idx)

            kernel_set_links_pos(
                links_idx,
                envs_idx,
                pos,
                dynamics.data.dyn_state,
                dynamics.data.dyn_info,
                dynamics.data.rigid_info,
                dynamics.rigid_config,
            )

        if not skip_forward:
            if not isinstance(envs_idx, torch.Tensor):
                envs_idx = self._scene._sanitize_envs_idx(envs_idx)
            dynamics.update_forward_kinematics(envs_idx)
        else:
            dynamics.invalidate_forward_state()

    def set_links_quat(self, quat: TensorLike, links_idx: IndexLike = None, envs_idx: IndexLike = None) -> None:
        raise DeprecationError("This method has been removed. Please use 'set_base_links_quat' instead.")

    def set_base_links_quat(
        self,
        quat: TensorLike,
        links_idx: IndexLike = None,
        envs_idx: IndexLike = None,
        *,
        relative: bool = False,
        skip_forward: bool = False,
    ) -> None:
        dynamics = self.rigid_dynamics
        if links_idx is None:
            links_idx = dynamics.base_links_idx

        # Without any pose offset on the targeted links, the authored and internal link origins coincide, so a relative
        # set is just an absolute one.
        idx = links_idx if isinstance(links_idx, int) else slice(None)
        if relative and (
            dynamics.links_offset_quat_is_identity[idx].all() and dynamics.links_offset_pos_is_identity[idx].all()
        ):
            relative = False

        # Reused by the int fast path and the kernel branch below.
        relative_pos_passthrough = relative and dynamics.links_offset_pos_is_identity[idx].all()

        # Compose a single base link's authored orientation to world here so the zero-copy in-place write below still
        # applies, both for an environment-uniform and a per-environment offset. This only preserves the authored-frame
        # position when the offset position is identity; a non-zero offset position rotates with the orientation and
        # is handled (together with multi-link relative sets) in the kernel branch instead.
        if isinstance(links_idx, int) and relative_pos_passthrough:
            offset_quat = _select_links_offset(dynamics.links_offset_quat, links_idx, envs_idx)[..., 0, :]
            quat = gu.transform_quat_by_quat(offset_quat, torch.as_tensor(quat, dtype=gs.tc_float, device=gs.device))
            relative = False

        # Zero-copy fast path: single base link, non-relative. Write the quaternion buffer in place instead of
        # launching a kernel. The kernel path below handles relative or multi-link updates, as well as waking up
        # hibernated entities, which is required whenever hibernation is enabled.
        if (
            gs.use_zerocopy
            and not relative
            and isinstance(links_idx, int)
            and not dynamics.rigid_config.use_hibernation
        ):
            link = dynamics.links[links_idx]
            if link.is_fixed:
                data = qd_to_torch(dynamics.data.dyn_state.links.quat, transpose=True, copy=False)
                target = data[:, links_idx]
            else:
                data = qd_to_torch(dynamics.data.rigid_info.qpos, transpose=True, copy=False)
                target = data[:, link.q_start + 3 : link.q_start + 7]
            if isinstance(envs_idx, torch.Tensor) and envs_idx.dtype == torch.bool:
                if quat.ndim == 2 and len(quat) not in (1, len(target)):
                    # A fresh source view is needed because masked_scatter_ may reshape it in-place. Metal
                    # mis-scatters a stride-0 broadcast mask, so it must be materialized to a dense mask there.
                    envs_mask = envs_idx[:, None]
                    if gs.backend == gs.metal:
                        envs_mask = envs_mask.expand_as(target).contiguous()
                    target.masked_scatter_(envs_mask, quat.view_as(quat))
                else:
                    quat = broadcast_tensor(quat, gs.tc_float, target.shape)
                    torch.where(envs_idx[:, None], quat, target, out=target)
            else:
                # Fixed links with at least one geom and non-batched vertices cannot take env-specific orientations
                if link.is_fixed and link.geoms and not link.entity._batch_fixed_verts:
                    quat = torch.as_tensor(quat, dtype=gs.tc_float, device=gs.device)
                    same_quat = quat.ndim < 2 or len(quat) == 1 or (torch.diff(quat, dim=0).abs() < gs.EPS).all()
                    set_all_envs = envs_idx is None or torch.equal(
                        torch.sort(self._scene._sanitize_envs_idx(envs_idx)).values, self._scene._envs_idx
                    )
                    if not (set_all_envs and same_quat):
                        gs.raise_exception(
                            "Impossible to set env-specific quat for fixed links with at least one geometry."
                        )
                mask = (0,) if dynamics.n_envs == 0 else indices_to_mask(envs_idx)
                assign_indexed_tensor(target, mask, quat)
            if gs.backend == gs.metal:
                torch.mps.synchronize()
        else:
            quat, links_idx, envs_idx = sanitize_io_variables(
                self._scene, quat, links_idx, dynamics.n_links, "links_idx", envs_idx, (4,), skip_allocation=True
            )
            if dynamics.n_envs == 0:
                quat = quat[None]

            if relative:
                offset_quat = _select_links_offset(dynamics.links_offset_quat, links_idx, envs_idx)
                if not relative_pos_passthrough:
                    # The offset position rotates with the orientation, so keep the authored-frame position fixed by
                    # rewriting the world position from the current authored position and the new authored orientation.
                    cur_pos = qd_to_torch(
                        dynamics.data.dyn_state.links.pos, envs_idx, links_idx, transpose=True, copy=True
                    )
                    cur_quat = qd_to_torch(
                        dynamics.data.dyn_state.links.quat, envs_idx, links_idx, transpose=True, copy=True
                    )
                    offset_pos = _select_links_offset(dynamics.links_offset_pos, links_idx, envs_idx)
                    authored_pos = cur_pos - _offset_world_shift(offset_pos, offset_quat, cur_quat)
                    world_pos = authored_pos + gu.transform_by_quat(offset_pos, quat)
                    kernel_set_links_pos(
                        links_idx,
                        envs_idx,
                        world_pos,
                        dynamics.data.dyn_state,
                        dynamics.data.dyn_info,
                        dynamics.data.rigid_info,
                        dynamics.rigid_config,
                    )
                # Compose the offset onto the authored orientation, then set the resulting world orientation absolutely.
                quat = gu.transform_quat_by_quat(offset_quat, quat)
                relative = False

            set_all_envs = torch.equal(torch.sort(envs_idx).values, self._scene._envs_idx)
            has_fixed_verts = any(
                link.is_fixed and link.geoms and not link.entity._batch_fixed_verts
                for link in (dynamics.links[i_l] for i_l in links_idx)
            )
            if has_fixed_verts and not (set_all_envs and (torch.diff(quat, dim=0).abs() < gs.EPS).all()):
                gs.raise_exception("Impossible to set env-specific quat for fixed links with at least one geometry.")

            # Wake up hibernated entities before setting quaternion (fixed links don't need wake-up)
            if dynamics.rigid_config.use_hibernation and not all(dynamics.links[i_l].is_fixed for i_l in links_idx):
                self.rigid_island.wake_entities_by_links(links_idx, envs_idx)

            kernel_set_links_quat(
                links_idx,
                envs_idx,
                quat,
                dynamics.data.dyn_state,
                dynamics.data.dyn_info,
                dynamics.data.rigid_info,
                dynamics.rigid_config,
            )

        if not skip_forward:
            if not isinstance(envs_idx, torch.Tensor):
                envs_idx = self._scene._sanitize_envs_idx(envs_idx)
            dynamics.update_forward_kinematics(envs_idx)
        else:
            dynamics.invalidate_forward_state()

    def set_links_mass(
        self,
        mass: TensorLike,
        links_idx: IndexLike = None,
        envs_idx: IndexLike = None,
        scale_inertia: bool = False,
    ) -> None:
        """Set the mass of the given links, in kg.

        Set 'scale_inertia' to scale their inertia by the same factor, which gives what a body of the same shape made
        heavier behaves like, its inertia being proportional to its mass. Without it, the mass of a link changes and
        the inertia it was given stays as it is.

        Every mass must be finite and strictly positive, which is not checked.

        The value written survives `scene.reset()`: a reset restores the configuration the scene was built at and
        leaves inertial properties alone.
        """
        dynamics = self.rigid_dynamics
        # Neither the masses nor the links they are written to are read back for checking: a condition drawn from
        # values on the device forces a synchronization at every call. The mass is spread over the links given, so
        # they have to carry some between them. A selection already on the device is left as it is, for that reason.
        if not isinstance(links_idx, torch.Tensor):
            if links_idx is None:
                links = dynamics.links
            elif isinstance(links_idx, slice):
                links = dynamics.links[links_idx]
            elif isinstance(links_idx, (int, np.integer)):
                links = (dynamics.links[links_idx],)
            else:
                links = [dynamics.links[i_l] for i_l in links_idx]
            if sum(0.0 if link.is_fixed else link.desc.mass for link in links) <= gs.EPS:
                gs.raise_exception(
                    "None of the links being set holds a mass to spread out, so there is no mass to set."
                )

        self._set_links_info(mass, links_idx, "mass", envs_idx, scale_inertia=scale_inertia)

    def set_links_inertia(self, inertia: TensorLike, links_idx: IndexLike = None, envs_idx: IndexLike = None) -> None:
        """Set the inertia tensor of the given links, expressed in their inertial frame.

        The caller is responsible for making sure that the inertia specified is physically sound, that is symmetric
        positive definite. There is no runtime check of this precondition, for the sake of efficiency.
        """
        self._set_links_info(inertia, links_idx, "inertia", envs_idx)

    def set_links_COM(self, com: TensorLike, links_idx: IndexLike = None, envs_idx: IndexLike = None) -> None:
        """Set the center of mass (COM) of the given links, as an offset from the origin of their local frame."""
        self._set_links_info(com, links_idx, "COM", envs_idx)

    def set_geoms_friction_ratio(
        self, friction_ratio: TensorLike, geoms_idx: IndexLike = None, envs_idx: IndexLike = None
    ) -> None:
        dynamics = self.rigid_dynamics
        friction_ratio, geoms_idx, envs_idx = sanitize_io_variables(
            self._scene, friction_ratio, geoms_idx, dynamics.n_geoms, "geoms_idx", envs_idx, skip_allocation=True
        )
        if dynamics.n_envs == 0:
            friction_ratio = friction_ratio[None]
        kernel_set_geoms_friction_ratio(
            geoms_idx, envs_idx, friction_ratio, dynamics.data.dyn_state, dynamics.rigid_config
        )

    def set_qpos(
        self,
        qpos: TensorLike,
        qs_idx: IndexLike = None,
        envs_idx: IndexLike = None,
        *,
        skip_forward: bool = False,
    ) -> None:
        dynamics = self.rigid_dynamics
        if gs.use_zerocopy:
            data = qd_to_torch(dynamics.data.rigid_info.qpos, transpose=True, copy=False)
            errno = qd_to_torch(dynamics.data.errno, copy=False)
            qs_mask = indices_to_mask(qs_idx)
            if (
                (not qs_mask or isinstance(qs_mask[0], slice))
                and isinstance(envs_idx, torch.Tensor)
                and envs_idx.dtype == torch.bool
            ):
                qs_data = data[(slice(None), *qs_mask)]
                if qpos.ndim == 2 and len(qpos) not in (1, len(qs_data)):
                    # A fresh source view is needed because masked_scatter_ may reshape it in-place. Metal mis-scatters
                    # a stride-0 broadcast mask, so it must be materialized to a dense full-shape mask there.
                    envs_mask = envs_idx[:, None]
                    if gs.backend == gs.metal:
                        envs_mask = envs_mask.expand_as(qs_data).contiguous()
                    qs_data.masked_scatter_(envs_mask, qpos.view_as(qpos))
                else:
                    qpos = broadcast_tensor(qpos, gs.tc_float, qs_data.shape)
                    torch.where(envs_idx[:, None], qpos, qs_data, out=qs_data)
                errno.masked_fill_(envs_idx, 0.0)
            else:
                mask = (
                    (0, *qs_mask) if dynamics.n_envs == 0 else indices_to_mask(envs_idx, *qs_mask, boolean_mask=False)
                )
                assign_indexed_tensor(data, mask, qpos)
                errno[envs_idx] = 0
                if mask and isinstance(mask[0], torch.Tensor):
                    envs_idx = mask[0].reshape((-1,))
            if gs.backend == gs.metal:
                torch.mps.synchronize()
        else:
            qpos, qs_idx, envs_idx = sanitize_io_variables(
                self._scene, qpos, qs_idx, dynamics.n_qs, "qs_idx", envs_idx, skip_allocation=True
            )
            if dynamics.n_envs == 0:
                qpos = qpos[None]

            # Teleporting a sleeping body must revive it, otherwise the new pose is silently dropped. Nothing can be
            # hibernated while the scene is still being built.
            if dynamics.rigid_config.use_hibernation and self._scene.is_built:
                self.rigid_island.wake_entities_by_qs(qs_idx, envs_idx)

            kernel_set_qpos(qs_idx, envs_idx, qpos, dynamics.data.rigid_info, dynamics.rigid_config)
            kernel_set_zero(envs_idx, dynamics.data.errno)

        if not skip_forward:
            if not isinstance(envs_idx, torch.Tensor):
                envs_idx = self._scene._sanitize_envs_idx(envs_idx)
            dynamics.update_forward_kinematics(envs_idx)
        else:
            dynamics.invalidate_forward_state()

    def set_global_sol_params(self, sol_params: TensorLike) -> None:
        """
        Set constraint solver parameters.

        Reference: https://mujoco.readthedocs.io/en/latest/modeling.html#solver-parameters

        Parameters
        ----------
        sol_params: Tuple[float] | List[float] | np.ndarray | torch.tensor
            array of length 7 in which each element corresponds to
            (timeconst, dampratio, dmin, dmax, width, mid, power)
        """
        dynamics = self.rigid_dynamics
        sol_params_ = broadcast_tensor(sol_params, gs.tc_float, (7,), ("",))
        sol_params_ = sanitize_sol_params(sol_params_.clone(), dynamics.sol_min_timeconst)
        kernel_set_global_sol_params(sol_params_, dynamics.data.dyn_info, dynamics.rigid_config)

    def set_sol_params(
        self,
        sol_params: TensorLike,
        geoms_idx: IndexLike = None,
        envs_idx: IndexLike = None,
        *,
        joints_idx: IndexLike = None,
        eqs_idx: IndexLike = None,
    ) -> None:
        """
        Set constraint solver parameters.

        See :func:`genesis.utils.geom.default_solver_params` for the parameter semantics, in particular the
        relationship between ``dampratio``, spring stiffness, and velocity damping.

        Reference: https://mujoco.readthedocs.io/en/latest/modeling.html#solver-parameters

        Parameters
        ----------
        sol_params: Tuple[float] | List[float] | np.ndarray | torch.tensor
            array of length 7 in which each element corresponds to
            (timeconst, dampratio, dmin, dmax, width, mid, power)
        """
        dynamics = self.rigid_dynamics
        # Make sure that a single constraint type has been selected at once
        if sum(inputs_idx is not None for inputs_idx in (geoms_idx, joints_idx, eqs_idx)) > 1:
            gs.raise_exception("Cannot set more than one constraint type at once.")

        # Select the right input type
        if eqs_idx is not None:
            constraint_type = ConstraintType.EQUALITY
            idx_name = "eqs_idx"
            inputs_idx = eqs_idx
            inputs_length = dynamics.n_equalities
            batched = True
        elif joints_idx is not None:
            constraint_type = ConstraintType.JOINT
            idx_name = "joints_idx"
            inputs_idx = joints_idx
            inputs_length = dynamics.n_joints
            batched = dynamics.options.batch_joints_info
        else:
            constraint_type = ConstraintType.GEOM
            idx_name = "geoms_idx"
            inputs_idx = geoms_idx
            inputs_length = dynamics.n_geoms
            batched = False

        # Sanitize input arguments
        sol_params_, inputs_idx, envs_idx = sanitize_io_variables(
            self._scene,
            sol_params,
            inputs_idx,
            inputs_length,
            idx_name,
            envs_idx,
            (7,),
            batched=batched,
            skip_allocation=True,
        )
        # Geom values resolve an unspecified time constant to the default, then defer the floor to contact
        # assembly, which floors the mixed pair value (see func_set_contact_data), mirroring the build-time
        # sanitization; joint and equality values are consumed unmixed, so they are floored here.
        sol_params_ = sanitize_sol_params(
            sol_params_.clone(),
            dynamics.sol_min_timeconst,
            dynamics.sol_default_timeconst if constraint_type == ConstraintType.GEOM else None,
            floor_timeconst=constraint_type != ConstraintType.GEOM,
        )
        if dynamics.n_envs == 0 and batched:
            sol_params_ = sol_params_[None]

        kernel_set_sol_params(
            inputs_idx, envs_idx, sol_params_, dynamics.data.dyn_info, dynamics.rigid_config, int(constraint_type)
        )

    def set_dofs_kp(self, kp: TensorLike, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> None:
        self._set_dofs_info([kp], dofs_idx, "kp", envs_idx)

    def set_dofs_kv(self, kv: TensorLike, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> None:
        self._set_dofs_info([kv], dofs_idx, "kv", envs_idx)

    def set_dofs_act_gain(self, act_gain: TensorLike, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> None:
        self._set_dofs_info([act_gain], dofs_idx, "act_gain", envs_idx)

    def set_dofs_act_bias(
        self,
        bias0: TensorLike,
        bias1: TensorLike,
        bias2: TensorLike,
        dofs_idx: IndexLike = None,
        envs_idx: IndexLike = None,
    ) -> None:
        self._set_dofs_info([bias0, bias1, bias2], dofs_idx, "act_bias", envs_idx)

    def set_dofs_force_range(
        self,
        lower: TensorLike,
        upper: TensorLike,
        dofs_idx: IndexLike = None,
        envs_idx: IndexLike = None,
    ) -> None:
        self._set_dofs_info([lower, upper], dofs_idx, "force_range", envs_idx)

    def set_dofs_stiffness(self, stiffness: TensorLike, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> None:
        self._set_dofs_info([stiffness], dofs_idx, "stiffness", envs_idx)

    def set_dofs_armature(self, armature: TensorLike, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> None:
        self._set_dofs_info([armature], dofs_idx, "armature", envs_idx)

    def set_dofs_damping(self, damping: TensorLike, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> None:
        self._set_dofs_info([damping], dofs_idx, "damping", envs_idx)

    def set_dofs_frictionloss(
        self, frictionloss: TensorLike, dofs_idx: IndexLike = None, envs_idx: IndexLike = None
    ) -> None:
        self._set_dofs_info([frictionloss], dofs_idx, "frictionloss", envs_idx)

    def set_dofs_limit(
        self,
        lower: TensorLike,
        upper: TensorLike,
        dofs_idx: IndexLike = None,
        envs_idx: IndexLike = None,
    ) -> None:
        self._set_dofs_info([lower, upper], dofs_idx, "limit", envs_idx)

    def set_dofs_position(self, position: TensorLike, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> None:
        dynamics = self.rigid_dynamics
        position, dofs_idx, envs_idx = sanitize_io_variables(
            self._scene, position, dofs_idx, dynamics.n_dofs, "dofs_idx", envs_idx, skip_allocation=True
        )
        if dynamics.n_envs == 0:
            position = position[None]

        self.rigid_island.wake_entities_by_dofs(dofs_idx, envs_idx)

        kernel_set_dofs_position(
            dofs_idx,
            envs_idx,
            position,
            dynamics.data.dyn_state,
            dynamics.data.dyn_info,
            dynamics.data.rigid_info,
            dynamics.rigid_config,
        )

        if gs.use_zerocopy:
            errno = qd_to_torch(dynamics.data.errno, copy=False)
            errno[envs_idx] = 0
            if gs.backend == gs.metal:
                torch.mps.synchronize()
        else:
            kernel_set_zero(envs_idx, dynamics.data.errno)

        dynamics.update_forward_kinematics(envs_idx)

    def set_dofs_velocity(
        self,
        velocity: TensorLike,
        dofs_idx: IndexLike = None,
        envs_idx: IndexLike = None,
        *,
        skip_forward: bool = False,
    ) -> None:
        # Wake the owning entities before delegating to the base setter, which re-sanitizes and applies the write.
        dynamics = self.rigid_dynamics
        if dynamics.rigid_config.use_hibernation:
            _, wake_dofs_idx, wake_envs_idx = sanitize_io_variables(
                self._scene, velocity, dofs_idx, dynamics.n_dofs, "dofs_idx", envs_idx, skip_allocation=True
            )
            self.rigid_island.wake_entities_by_dofs(wake_dofs_idx, wake_envs_idx)
        self._set_kinematic_dofs_velocity(velocity, dofs_idx, envs_idx, skip_forward=skip_forward)

    def control_dofs_force(self, force: TensorLike, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> None:
        # A control target is consumed by the actuation pass of the next step, which wakes a sleeping link it actuates
        # (see func_torque_and_passive_force), so the target is written without waking anything here.
        dynamics = self.rigid_dynamics
        if gs.use_zerocopy:
            mask = (0, *indices_to_mask(dofs_idx)) if dynamics.n_envs == 0 else indices_to_mask(envs_idx, dofs_idx)
            ctrl_mode = qd_to_torch(dynamics.data.dyn_state.dofs.ctrl_mode, transpose=True, copy=False)
            ctrl_mode[mask] = gs.CTRL_MODE.FORCE
            ctrl_force = qd_to_torch(dynamics.data.dyn_state.dofs.ctrl_force, transpose=True, copy=False)
            assign_indexed_tensor(ctrl_force, mask, force)
            if gs.backend == gs.metal:
                torch.mps.synchronize()
            return

        force, dofs_idx, envs_idx = sanitize_io_variables(
            self._scene, force, dofs_idx, dynamics.n_dofs, "dofs_idx", envs_idx, skip_allocation=True
        )
        if dynamics.n_envs == 0:
            force = force[None]

        kernel_control_dofs_force(dofs_idx, envs_idx, force, dynamics.data.dyn_state, dynamics.rigid_config)

    def control_dofs_velocity(
        self, velocity: TensorLike, dofs_idx: IndexLike = None, envs_idx: IndexLike = None
    ) -> None:
        dynamics = self.rigid_dynamics
        if gs.use_zerocopy:
            mask = (0, *indices_to_mask(dofs_idx)) if dynamics.n_envs == 0 else indices_to_mask(envs_idx, dofs_idx)
            ctrl_mode = qd_to_torch(dynamics.data.dyn_state.dofs.ctrl_mode, transpose=True, copy=False)
            ctrl_mode[mask] = gs.CTRL_MODE.VELOCITY
            ctrl_pos = qd_to_torch(dynamics.data.dyn_state.dofs.ctrl_pos, transpose=True, copy=False)
            ctrl_pos[mask] = 0.0
            ctrl_vel = qd_to_torch(dynamics.data.dyn_state.dofs.ctrl_vel, transpose=True, copy=False)
            assign_indexed_tensor(ctrl_vel, mask, velocity)
            if gs.backend == gs.metal:
                torch.mps.synchronize()
            return

        velocity, dofs_idx, envs_idx = sanitize_io_variables(
            self._scene, velocity, dofs_idx, dynamics.n_dofs, "dofs_idx", envs_idx, skip_allocation=True
        )
        if dynamics.n_envs == 0:
            velocity = velocity[None]

        kernel_control_dofs_velocity(dofs_idx, envs_idx, velocity, dynamics.data.dyn_state, dynamics.rigid_config)

    def control_dofs_position(
        self, position: TensorLike, dofs_idx: IndexLike = None, envs_idx: IndexLike = None
    ) -> None:
        dynamics = self.rigid_dynamics
        if gs.use_zerocopy:
            mask = (0, *indices_to_mask(dofs_idx)) if dynamics.n_envs == 0 else indices_to_mask(envs_idx, dofs_idx)
            ctrl_mode = qd_to_torch(dynamics.data.dyn_state.dofs.ctrl_mode, transpose=True, copy=False)
            ctrl_mode[mask] = gs.CTRL_MODE.POSITION
            ctrl_pos = qd_to_torch(dynamics.data.dyn_state.dofs.ctrl_pos, transpose=True, copy=False)
            assign_indexed_tensor(ctrl_pos, mask, position)
            ctrl_vel = qd_to_torch(dynamics.data.dyn_state.dofs.ctrl_vel, transpose=True, copy=False)
            ctrl_vel[mask] = 0.0
            if gs.backend == gs.metal:
                torch.mps.synchronize()
            return

        position, dofs_idx, envs_idx = sanitize_io_variables(
            self._scene, position, dofs_idx, dynamics.n_dofs, "dofs_idx", envs_idx, skip_allocation=True
        )
        if dynamics.n_envs == 0:
            position = position[None]

        kernel_control_dofs_position(dofs_idx, envs_idx, position, dynamics.data.dyn_state, dynamics.rigid_config)

    def control_dofs_position_velocity(
        self,
        position: TensorLike,
        velocity: TensorLike,
        dofs_idx: IndexLike = None,
        envs_idx: IndexLike = None,
    ) -> None:
        dynamics = self.rigid_dynamics
        if gs.use_zerocopy:
            mask = (0, *indices_to_mask(dofs_idx)) if dynamics.n_envs == 0 else indices_to_mask(envs_idx, dofs_idx)
            ctrl_mode = qd_to_torch(dynamics.data.dyn_state.dofs.ctrl_mode, transpose=True, copy=False)
            ctrl_mode[mask] = gs.CTRL_MODE.POSITION
            ctrl_pos = qd_to_torch(dynamics.data.dyn_state.dofs.ctrl_pos, transpose=True, copy=False)
            assign_indexed_tensor(ctrl_pos, mask, position)
            ctrl_vel = qd_to_torch(dynamics.data.dyn_state.dofs.ctrl_vel, transpose=True, copy=False)
            assign_indexed_tensor(ctrl_vel, mask, velocity)
            if gs.backend == gs.metal:
                torch.mps.synchronize()
            return

        position, dofs_idx, _ = sanitize_io_variables(
            self._scene, position, dofs_idx, dynamics.n_dofs, "dofs_idx", envs_idx, skip_allocation=True
        )
        velocity, dofs_idx, envs_idx = sanitize_io_variables(
            self._scene, velocity, dofs_idx, dynamics.n_dofs, "dofs_idx", envs_idx, skip_allocation=True
        )
        if dynamics.n_envs == 0:
            position = position[None]
            velocity = velocity[None]

        kernel_control_dofs_position_velocity(
            dofs_idx, envs_idx, position, velocity, dynamics.data.dyn_state, dynamics.rigid_config
        )

    def get_sol_params(
        self,
        geoms_idx: IndexLike = None,
        envs_idx: IndexLike = None,
        *,
        joints_idx: IndexLike = None,
        eqs_idx: IndexLike = None,
    ) -> torch.Tensor:
        """
        Get constraint solver parameters.
        """
        dynamics = self.rigid_dynamics
        if eqs_idx is not None:
            # Always batched
            tensor = qd_to_torch(
                dynamics.data.dyn_info.equalities.sol_params, envs_idx, eqs_idx, transpose=True, copy=True
            )
            if dynamics.n_envs == 0:
                tensor = tensor[0]
        elif joints_idx is not None:
            # Conditionally batched
            assert envs_idx is None
            # batch_shape = (envs_idx, joints_idx) if dynamics.options.batch_joints_info else (joints_idx,) tensor =
            # qd_to_torch(dynamics.data.dyn_info.joints.sol_params, *batch_shape, transpose=True)
            tensor = qd_to_torch(
                dynamics.data.dyn_info.joints.sol_params, envs_idx, joints_idx, transpose=True, copy=True
            )
            if dynamics.n_envs == 0 and dynamics.options.batch_joints_info:
                tensor = tensor[0]
        else:  # geoms_idx is not None
            # Never batched
            assert envs_idx is None
            tensor = qd_to_torch(dynamics.data.dyn_info.geoms.sol_params, geoms_idx, transpose=True, copy=True)
        return tensor

    def get_links_pos(
        self,
        links_idx: IndexLike = None,
        envs_idx: IndexLike = None,
        *,
        ref: link_ref_frame = link_ref_frame.link_origin,
        relative: bool = False,
    ) -> torch.Tensor:
        dynamics = self.rigid_dynamics
        idx = links_idx if isinstance(links_idx, int) else slice(None)
        if not gs.use_zerocopy:
            _, links_idx, envs_idx = sanitize_io_variables(
                self._scene, None, links_idx, dynamics.n_links, "links_idx", envs_idx, (3,), skip_allocation=True
            )

        ref_frame = sanitize_ref_frame(ref)
        if ref_frame == link_ref_frame.root_COM:
            tensor = qd_to_torch(dynamics.data.dyn_state.links.root_COM, envs_idx, links_idx, transpose=True, copy=True)
        elif ref_frame == link_ref_frame.link_COM:
            i_pos = qd_to_torch(dynamics.data.dyn_state.links.i_pos, envs_idx, links_idx, transpose=True)
            root_COM = qd_to_torch(dynamics.data.dyn_state.links.root_COM, envs_idx, links_idx, transpose=True)
            tensor = i_pos + root_COM
        else:
            tensor = qd_to_torch(dynamics.data.dyn_state.links.pos, envs_idx, links_idx, transpose=True, copy=True)

        # The pose offset is defined on the link origin, so it is only stripped for the 'link_origin' reference.
        if (
            relative
            and ref_frame == link_ref_frame.link_origin
            and not dynamics.links_offset_pos_is_identity[idx].all()
        ):
            quat = qd_to_torch(dynamics.data.dyn_state.links.quat, envs_idx, links_idx, transpose=True)
            offset_pos = _select_links_offset(dynamics.links_offset_pos, links_idx, envs_idx)
            offset_quat = _select_links_offset(dynamics.links_offset_quat, links_idx, envs_idx)
            tensor -= _offset_world_shift(offset_pos, offset_quat, quat)

        return tensor[0] if dynamics.n_envs == 0 else tensor

    def get_links_vel(
        self,
        links_idx: IndexLike = None,
        envs_idx: IndexLike = None,
        *,
        ref: link_ref_frame = link_ref_frame.link_origin,
        relative: bool = False,
    ) -> torch.Tensor:
        dynamics = self.rigid_dynamics
        ref_frame = sanitize_ref_frame(ref, has_root_com=False)
        idx = links_idx if isinstance(links_idx, int) else slice(None)
        # Only the 'link_origin' reference carries the offset, as in 'get_links_pos'.
        is_relative = (
            relative
            and ref_frame == link_ref_frame.link_origin
            and not dynamics.links_offset_pos_is_identity[idx].all()
        )
        if gs.use_zerocopy:
            mask = indices_to_mask(envs_idx, links_idx)
            cd_vel = qd_to_torch(dynamics.data.dyn_state.links.cd_vel, transpose=True)
            cd_ang = qd_to_torch(dynamics.data.dyn_state.links.cd_ang, transpose=True)
            if ref_frame == link_ref_frame.link_COM:
                i_pos = qd_to_torch(dynamics.data.dyn_state.links.i_pos, transpose=True)
                delta = i_pos[mask]
            else:
                pos = qd_to_torch(dynamics.data.dyn_state.links.pos, transpose=True)
                root_COM = qd_to_torch(dynamics.data.dyn_state.links.root_COM, transpose=True)
                delta = pos[mask] - root_COM[mask]
                if is_relative:
                    quat = qd_to_torch(dynamics.data.dyn_state.links.quat, envs_idx, links_idx, transpose=True)
                    offset_pos = _select_links_offset(dynamics.links_offset_pos, links_idx, envs_idx)
                    offset_quat = _select_links_offset(dynamics.links_offset_quat, links_idx, envs_idx)
                    delta = delta - _offset_world_shift(offset_pos, offset_quat, quat)
            tensor = cd_vel[mask] + cd_ang[mask].cross(delta, dim=-1)
            return tensor[0] if dynamics.n_envs == 0 else tensor

        _tensor, links_idx, envs_idx = sanitize_io_variables(
            self._scene, None, links_idx, dynamics.n_links, "links_idx", envs_idx, (3,)
        )
        tensor = _tensor[None] if dynamics.n_envs == 0 else _tensor
        kernel_get_links_vel(
            links_idx,
            envs_idx,
            tensor,
            dynamics.links_offset_pos,
            dynamics.links_offset_quat,
            dynamics.data.dyn_state,
            dynamics.rigid_config,
            ref_frame,
            is_relative=is_relative,
        )
        return _tensor

    def get_links_acc(
        self, links_idx: IndexLike = None, envs_idx: IndexLike = None, *, relative: bool = False
    ) -> torch.Tensor:
        dynamics = self.rigid_dynamics
        idx = links_idx if isinstance(links_idx, int) else slice(None)
        _tensor, links_idx, envs_idx = sanitize_io_variables(
            self._scene, None, links_idx, dynamics.n_links, "links_idx", envs_idx, (3,)
        )
        tensor = _tensor[None] if dynamics.n_envs == 0 else _tensor
        kernel_get_links_acc(
            links_idx,
            envs_idx,
            tensor,
            dynamics.links_offset_pos,
            dynamics.links_offset_quat,
            dynamics.data.dyn_state,
            dynamics.rigid_config,
            is_relative=relative and not dynamics.links_offset_pos_is_identity[idx].all(),
        )
        return _tensor

    def get_links_acc_ang(self, links_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        dynamics = self.rigid_dynamics
        tensor = qd_to_torch(dynamics.data.dyn_state.links.cacc_ang, envs_idx, links_idx, transpose=True, copy=True)
        return tensor[0] if dynamics.n_envs == 0 else tensor

    def get_links_root_COM(self, links_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        """
        Returns the center of mass (COM) of the entire kinematic tree to which the specified links belong.

        This corresponds to the global COM of each entity, assuming a single-rooted structure - that is, as long as no
        two successive links are connected by a free-floating joint (ie a joint that allows all 6 degrees of freedom).
        """
        dynamics = self.rigid_dynamics
        tensor = qd_to_torch(dynamics.data.dyn_state.links.root_COM, envs_idx, links_idx, transpose=True, copy=True)
        return tensor[0] if dynamics.n_envs == 0 else tensor

    def get_links_COM(self, links_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        """The center of mass (COM) of each link, as an offset from the origin of its local frame."""
        dynamics = self.rigid_dynamics
        if not dynamics.options.batch_links_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched links info.")
        tensor = qd_to_torch(dynamics.data.dyn_info.links.inertial_pos, envs_idx, links_idx, transpose=True, copy=True)
        return tensor[0] if dynamics.n_envs == 0 and dynamics.options.batch_links_info else tensor

    def is_links_info_batched(self) -> bool:
        """Whether the inertial properties of a link are stored per environment rather than shared by the batch."""
        return self.rigid_dynamics.options.batch_links_info

    def get_links_mass(self, links_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        """The mass of each link, as the solver currently uses it."""
        dynamics = self.rigid_dynamics
        if not dynamics.options.batch_links_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched links info.")
        tensor = qd_to_torch(dynamics.data.dyn_info.links.inertial_mass, envs_idx, links_idx, transpose=True, copy=True)
        return tensor[0] if dynamics.n_envs == 0 and dynamics.options.batch_links_info else tensor

    def get_links_inertia(self, links_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        """The inertia matrix of each link, in its inertial frame, as the solver currently uses it."""
        dynamics = self.rigid_dynamics
        if not dynamics.options.batch_links_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched links info.")
        tensor = qd_to_torch(dynamics.data.dyn_info.links.inertial_i, envs_idx, links_idx, transpose=True, copy=True)
        return tensor[0] if dynamics.n_envs == 0 and dynamics.options.batch_links_info else tensor

    def get_links_invweight(self, links_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        dynamics = self.rigid_dynamics
        if not dynamics.options.batch_links_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched links info.")
        tensor = qd_to_torch(dynamics.data.dyn_info.links.invweight, envs_idx, links_idx, transpose=True, copy=True)
        return tensor[0] if dynamics.n_envs == 0 and dynamics.options.batch_links_info else tensor

    def get_geoms_friction_ratio(self, geoms_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        dynamics = self.rigid_dynamics
        tensor = qd_to_torch(
            dynamics.data.dyn_state.geoms.friction_ratio, envs_idx, geoms_idx, transpose=True, copy=True
        )
        return tensor[0] if dynamics.n_envs == 0 else tensor

    def get_geoms_pos(
        self, geoms_idx: IndexLike = None, envs_idx: IndexLike = None, *, relative: bool = False
    ) -> torch.Tensor:
        dynamics = self.rigid_dynamics
        tensor = qd_to_torch(dynamics.data.dyn_state.geoms.pos, envs_idx, geoms_idx, transpose=True, copy=True)
        if relative and dynamics.geoms_offset_pos is not None:
            quat = qd_to_torch(dynamics.data.dyn_state.geoms.quat, envs_idx, geoms_idx, transpose=True, copy=True)
            offset_pos = dynamics.geoms_offset_pos if geoms_idx is None else dynamics.geoms_offset_pos[geoms_idx]
            offset_quat = dynamics.geoms_offset_quat if geoms_idx is None else dynamics.geoms_offset_quat[geoms_idx]
            tensor -= _offset_world_shift(offset_pos, offset_quat, quat)
        return tensor[0] if dynamics.n_envs == 0 else tensor

    def get_geoms_quat(
        self, geoms_idx: IndexLike = None, envs_idx: IndexLike = None, *, relative: bool = False
    ) -> torch.Tensor:
        dynamics = self.rigid_dynamics
        tensor = qd_to_torch(dynamics.data.dyn_state.geoms.quat, envs_idx, geoms_idx, transpose=True, copy=True)
        if relative and dynamics.geoms_offset_quat is not None:
            offset_quat = dynamics.geoms_offset_quat if geoms_idx is None else dynamics.geoms_offset_quat[geoms_idx]
            tensor = gu.transform_quat_by_quat(gu.inv_quat(offset_quat), tensor)
        return tensor[0] if dynamics.n_envs == 0 else tensor

    def get_dofs_control_force(self, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        dynamics = self.rigid_dynamics
        _tensor, dofs_idx, envs_idx = sanitize_io_variables(
            self._scene, None, dofs_idx, dynamics.n_dofs, "dofs_idx", envs_idx
        )
        tensor = _tensor[None] if dynamics.n_envs == 0 else _tensor
        kernel_get_dofs_control_force(
            dofs_idx, envs_idx, tensor, dynamics.data.dyn_state, dynamics.data.dyn_info, dynamics.rigid_config
        )
        return _tensor

    def get_dofs_force(self, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        dynamics = self.rigid_dynamics
        tensor = qd_to_torch(dynamics.data.dyn_state.dofs.force, envs_idx, dofs_idx, transpose=True, copy=True)
        return tensor[0] if dynamics.n_envs == 0 else tensor

    def get_dofs_acc(self, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        dynamics = self.rigid_dynamics
        tensor = qd_to_torch(dynamics.data.dyn_state.dofs.acc, envs_idx, dofs_idx, transpose=True, copy=True)
        return tensor[0] if dynamics.n_envs == 0 else tensor

    def get_dofs_kp(self, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        dynamics = self.rigid_dynamics
        if not dynamics.options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        gain = qd_to_torch(dynamics.data.dyn_info.dofs.act_gain, envs_idx, dofs_idx, transpose=True, copy=True)
        bias = qd_to_torch(dynamics.data.dyn_info.dofs.act_bias, envs_idx, dofs_idx, transpose=True, copy=True)
        if dynamics.n_envs == 0 and dynamics.options.batch_dofs_info:
            gain, bias = gain[0], bias[0]
        if not (torch.abs(gain + bias[..., 1]) < gs.EPS * torch.clamp(torch.abs(gain), min=1.0)).all():
            gs.raise_exception(
                "Some DOFs use a non-PD-reducible actuator (act_gain != -act_bias[1]). "
                "Use get_dofs_act_gain() and get_dofs_act_bias() instead."
            )
        if not (torch.abs(bias[..., 0]) < gs.EPS).all():
            gs.raise_exception(
                "Some DOFs use a non-PD-reducible actuator (act_bias[0] != 0). "
                "Use get_dofs_act_gain() and get_dofs_act_bias() instead."
            )
        return gain

    def get_dofs_kv(self, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        dynamics = self.rigid_dynamics
        if not dynamics.options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        gain = qd_to_torch(dynamics.data.dyn_info.dofs.act_gain, envs_idx, dofs_idx, transpose=True, copy=True)
        bias = qd_to_torch(dynamics.data.dyn_info.dofs.act_bias, envs_idx, dofs_idx, transpose=True, copy=True)
        if dynamics.n_envs == 0 and dynamics.options.batch_dofs_info:
            gain, bias = gain[0], bias[0]
        if not (torch.abs(gain + bias[..., 1]) < gs.EPS * torch.clamp(torch.abs(gain), min=1.0)).all():
            gs.raise_exception(
                "Some DOFs use a non-PD-reducible actuator (act_gain != -act_bias[1]). "
                "Use get_dofs_act_gain() and get_dofs_act_bias() instead."
            )
        if not (torch.abs(bias[..., 0]) < gs.EPS).all():
            gs.raise_exception(
                "Some DOFs use a non-PD-reducible actuator (act_bias[0] != 0). "
                "Use get_dofs_act_gain() and get_dofs_act_bias() instead."
            )
        return -bias[..., 2]

    def get_dofs_act_gain(self, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        dynamics = self.rigid_dynamics
        if not dynamics.options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        tensor = qd_to_torch(dynamics.data.dyn_info.dofs.act_gain, envs_idx, dofs_idx, transpose=True, copy=True)
        return tensor[0] if dynamics.n_envs == 0 and dynamics.options.batch_dofs_info else tensor

    def get_dofs_act_bias(
        self, dofs_idx: IndexLike = None, envs_idx: IndexLike = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dynamics = self.rigid_dynamics
        if not dynamics.options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        tensor = qd_to_torch(dynamics.data.dyn_info.dofs.act_bias, envs_idx, dofs_idx, transpose=True, copy=True)
        if dynamics.n_envs == 0 and dynamics.options.batch_dofs_info:
            tensor = tensor[0]
        return tensor[..., 0], tensor[..., 1], tensor[..., 2]

    def get_dofs_force_range(
        self, dofs_idx: IndexLike = None, envs_idx: IndexLike = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dynamics = self.rigid_dynamics
        if not dynamics.options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        tensor = qd_to_torch(dynamics.data.dyn_info.dofs.force_range, envs_idx, dofs_idx, transpose=True, copy=True)
        if dynamics.n_envs == 0 and dynamics.options.batch_dofs_info:
            tensor = tensor[0]
        return tensor[..., 0], tensor[..., 1]

    def get_dofs_stiffness(self, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        dynamics = self.rigid_dynamics
        if not dynamics.options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        tensor = qd_to_torch(dynamics.data.dyn_info.dofs.stiffness, envs_idx, dofs_idx, transpose=True, copy=True)
        return tensor[0] if dynamics.n_envs == 0 and dynamics.options.batch_dofs_info else tensor

    def get_dofs_invweight(self, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        dynamics = self.rigid_dynamics
        if not dynamics.options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        tensor = qd_to_torch(dynamics.data.dyn_info.dofs.invweight, envs_idx, dofs_idx, transpose=True, copy=True)
        return tensor[0] if dynamics.n_envs == 0 and dynamics.options.batch_dofs_info else tensor

    def get_dofs_armature(self, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        dynamics = self.rigid_dynamics
        if not dynamics.options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        tensor = qd_to_torch(dynamics.data.dyn_info.dofs.armature, envs_idx, dofs_idx, transpose=True, copy=True)
        return tensor[0] if dynamics.n_envs == 0 and dynamics.options.batch_dofs_info else tensor

    def get_dofs_damping(self, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        dynamics = self.rigid_dynamics
        if not dynamics.options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        tensor = qd_to_torch(dynamics.data.dyn_info.dofs.damping, envs_idx, dofs_idx, transpose=True, copy=True)
        return tensor[0] if dynamics.n_envs == 0 and dynamics.options.batch_dofs_info else tensor

    def get_dofs_frictionloss(self, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        dynamics = self.rigid_dynamics
        if not dynamics.options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        tensor = qd_to_torch(dynamics.data.dyn_info.dofs.frictionloss, envs_idx, dofs_idx, transpose=True, copy=True)
        return tensor[0] if dynamics.n_envs == 0 and dynamics.options.batch_dofs_info else tensor

    def get_mass_mat(
        self,
        dofs_idx: IndexLike = None,
        envs_idx: IndexLike = None,
        decompose: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        dynamics = self.rigid_dynamics
        tensor = qd_to_torch(
            dynamics.data.rigid_info.mass_mat_L if decompose else dynamics.data.rigid_info.mass_mat,
            envs_idx,
            transpose=True,
            copy=True,
        )
        if dofs_idx is not None:
            tensor = tensor[indices_to_mask(None, dofs_idx, dofs_idx)]
        if dynamics.n_envs == 0:
            tensor = tensor[0]

        if decompose:
            mass_mat_D_inv = qd_to_torch(
                dynamics.data.rigid_info.mass_mat_D_inv, envs_idx, dofs_idx, transpose=True, copy=True
            )
            if dynamics.n_envs == 0:
                mass_mat_D_inv = mass_mat_D_inv[0]
            return tensor, mass_mat_D_inv

        return tensor

    def get_kinetic_energy(
        self,
        links_idx: IndexLike = None,
        dofs_idx: IndexLike = None,
        envs_idx: IndexLike = None,
    ) -> torch.Tensor:
        """Get the kinetic energy of the specified links and DOFs in Joules [J] (translational + rotational).

        Summed over the links, each contributing ``0.5 * V^T * I * V`` for its spatial velocity ``V`` and spatial
        inertia ``I`` about the center of mass (COM) of its kinematic tree, plus the motor armature contribution
        ``0.5 * sum_d(armature_d * dq_d^2)`` of the DOFs. This equals the joint-space form ``0.5 * dq^T * M(q) * dq``
        while reading the link velocities and inertias that forward kinematics already maintains, so it needs no mass
        matrix and stays consistent with the current configuration whatever the integrator.

        A link and the DOFs driving it contribute independently, so selecting one without the other is meaningful only
        to isolate that one term.

        Parameters
        ----------
        links_idx : None | array_like, optional
            The indices of the links. If None, all links will be considered. Defaults to None.
        dofs_idx : None | array_like, optional
            The indices of the degrees of freedom. If None, all of them will be considered. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        kinetic_energy : torch.Tensor, shape () or (n_envs,)
        """
        dynamics = self.rigid_dynamics
        # Spatial velocity and inertia are both referenced to the tree COM, so the quadratic form expands as
        # 0.5 * m * |v|^2 + v . (w x (m * c)) + 0.5 * w . (I * w), with `cinr_pos` holding the first moment m * c.
        cd_ang = qd_to_torch(dynamics.data.dyn_state.links.cd_ang, envs_idx, links_idx, transpose=True)
        cd_vel = qd_to_torch(dynamics.data.dyn_state.links.cd_vel, envs_idx, links_idx, transpose=True)
        cinr_inertial = qd_to_torch(dynamics.data.dyn_state.links.cinr_inertial, envs_idx, links_idx, transpose=True)
        cinr_pos = qd_to_torch(dynamics.data.dyn_state.links.cinr_pos, envs_idx, links_idx, transpose=True)
        cinr_mass = qd_to_torch(dynamics.data.dyn_state.links.cinr_mass, envs_idx, links_idx, transpose=True)
        translational = cinr_mass * torch.sum(cd_vel * cd_vel, dim=-1)
        coupling = torch.sum(cd_vel * torch.cross(cd_ang, cinr_pos, dim=-1), dim=-1)
        rotational = torch.sum(cd_ang * torch.matmul(cinr_inertial, cd_ang.unsqueeze(-1)).squeeze(-1), dim=-1)
        kinetic_energy = torch.sum(0.5 * (translational + rotational) + coupling, dim=-1)

        dofs_vel = qd_to_torch(dynamics.data.dyn_state.dofs.vel, envs_idx, dofs_idx, transpose=True)
        armature = self.get_dofs_armature(dofs_idx, envs_idx if dynamics.options.batch_dofs_info else None)
        kinetic_energy += 0.5 * torch.sum(armature * dofs_vel * dofs_vel, dim=-1)

        return kinetic_energy[0] if dynamics.n_envs == 0 else kinetic_energy

    def get_potential_energy(
        self,
        links_idx: IndexLike = None,
        dofs_idx: IndexLike = None,
        envs_idx: IndexLike = None,
    ) -> torch.Tensor:
        """Get the potential energy of the specified links and DOFs in Joules [J] (gravitational + joint springs).

        Gravity contributes ``-sum_i(m_i * g^T * p_i)`` over the links free to move, where ``p_i`` is the center of
        mass (COM) position of link *i*. A link fixed to the world is left out, as it is of the mass of an entity: its
        potential is a constant of the scene. Joint springs contribute ``0.5 * sum_d(stiffness_d * (q_d - q0_d)^2)``,
        the elastic energy stored by holding each DOF away from its neutral position. Both are state functions, so
        their sum with the kinetic energy is conserved by a passive, frictionless, contact-free self.

        Contacts contribute nothing: they are resolved by the constraint solver, which stabilizes a penetration
        rather than storing it as an elastic potential.

        Parameters
        ----------
        links_idx : None | array_like, optional
            The indices of the links. If None, all links will be considered. Defaults to None.
        dofs_idx : None | array_like, optional
            The indices of the degrees of freedom. If None, all of them will be considered. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        potential_energy : torch.Tensor, shape () or (n_envs,)
        """
        dynamics = self.rigid_dynamics
        gravity = self.get_gravity(envs_idx=envs_idx)  # (3,) or (n_envs, 3)
        links_pos = self.get_links_pos(links_idx, envs_idx, ref=link_ref_frame.link_COM)  # (..., n_links, 3)
        links_envs_idx = envs_idx if dynamics.options.batch_links_info else None
        links_mass = qd_to_torch(dynamics.data.dyn_info.links.inertial_mass, links_envs_idx, links_idx, transpose=True)
        links_is_fixed = qd_to_torch(dynamics.data.dyn_info.links.is_fixed, links_envs_idx, links_idx, transpose=True)
        links_mass = links_mass.masked_fill(links_is_fixed.to(torch.bool), 0.0)
        if dynamics.n_envs == 0 and dynamics.options.batch_links_info:
            links_mass = links_mass[0]

        # PE_i = m_i * g^T * p_i => PE = sum_i(m_i * (g . p_i))
        # g is (..., 3), links_pos is (..., n_links, 3) -> broadcast g to (..., 1, 3)
        g_dot_p = torch.sum(gravity.unsqueeze(-2) * links_pos, dim=-1)  # (..., n_links)
        potential_energy = -torch.sum(links_mass * g_dot_p, dim=-1)

        # `dofs.pos` holds the deflection `qpos - qpos0` that the spring force is proportional to, so the elastic
        # energy integrates directly against it whatever the neutral configuration.
        dofs_pos = qd_to_torch(dynamics.data.dyn_state.dofs.pos, envs_idx, dofs_idx, transpose=True)
        stiffness = self.get_dofs_stiffness(dofs_idx, envs_idx if dynamics.options.batch_dofs_info else None)
        spring_energy = 0.5 * torch.sum(stiffness * dofs_pos * dofs_pos, dim=-1)
        if dynamics.n_envs == 0:
            spring_energy = spring_energy[0]

        return potential_energy + spring_energy

    def get_total_energy(self, envs_idx: IndexLike = None) -> torch.Tensor:
        """Get the total mechanical energy of all entities in Joules [J] (kinetic + potential).

        Parameters
        ----------
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        total_energy : torch.Tensor, shape () or (n_envs,)
        """
        return self.get_kinetic_energy(envs_idx=envs_idx) + self.get_potential_energy(envs_idx=envs_idx)

    def get_geoms_friction(self, geoms_idx: IndexLike = None) -> torch.Tensor:
        return qd_to_torch(self.rigid_dynamics.data.dyn_info.geoms.friction, geoms_idx, copy=True)

    def get_geoms_friction_torsional(self, geoms_idx: IndexLike = None) -> torch.Tensor:
        return qd_to_torch(self.rigid_dynamics.data.dyn_info.geoms.friction_torsional, geoms_idx, copy=True)

    def get_geoms_friction_rolling(self, geoms_idx: IndexLike = None) -> torch.Tensor:
        return qd_to_torch(self.rigid_dynamics.data.dyn_info.geoms.friction_rolling, geoms_idx, copy=True)

    def get_AABB(self, entities_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        dynamics = self.rigid_dynamics
        aabb_min = qd_to_torch(dynamics.data.dyn_state.geoms.aabb_min, envs_idx, transpose=True)
        aabb_max = qd_to_torch(dynamics.data.dyn_state.geoms.aabb_max, envs_idx, transpose=True)

        aabb = torch.stack([aabb_min, aabb_max], dim=-2)

        if entities_idx is not None:
            entity_geom_starts = []
            entity_geom_ends = []
            for entity_idx in entities_idx:
                entity = dynamics.entities[entity_idx]
                entity_geom_starts.append(entity._geom_start)
                entity_geom_ends.append(entity._geom_start + entity.n_geoms)

            entity_aabbs = []
            for start, end in zip(entity_geom_starts, entity_geom_ends):
                if start < end:
                    entity_geoms_aabb = aabb[..., start:end, :, :]
                    entity_min = entity_geoms_aabb[..., :, 0, :].min(dim=-2)[0]
                    entity_max = entity_geoms_aabb[..., :, 1, :].max(dim=-2)[0]
                    entity_aabb = torch.stack([entity_min, entity_max], dim=-2)
                else:
                    entity_aabb = torch.zeros_like(aabb[..., 0:1, :, :])
                entity_aabbs.append(entity_aabb)

            aabb = torch.stack(entity_aabbs, dim=-2)

        return aabb[0] if dynamics.n_envs == 0 else aabb

    def set_geom_friction(self, friction: TensorLike, geoms_idx: IndexLike) -> None:
        kernel_set_geom_friction(geoms_idx, friction, self.rigid_dynamics.data.dyn_info)

    def set_geom_friction_torsional(self, friction_torsional: TensorLike, geoms_idx: IndexLike) -> None:
        kernel_set_geom_friction_torsional(geoms_idx, friction_torsional, self.rigid_dynamics.data.dyn_info)

    def set_geom_friction_rolling(self, friction_rolling: TensorLike, geoms_idx: IndexLike) -> None:
        kernel_set_geom_friction_rolling(geoms_idx, friction_rolling, self.rigid_dynamics.data.dyn_info)

    def set_geoms_friction(self, friction: TensorLike, geoms_idx: IndexLike = None) -> None:
        dynamics = self.rigid_dynamics
        friction, geoms_idx, _ = sanitize_io_variables(
            self._scene,
            friction,
            geoms_idx,
            dynamics.n_geoms,
            "geoms_idx",
            envs_idx=None,
            batched=False,
            skip_allocation=True,
        )
        kernel_set_geoms_friction(geoms_idx, friction, dynamics.data.dyn_info, dynamics.rigid_config)

    def set_geoms_friction_torsional(self, friction_torsional: TensorLike, geoms_idx: IndexLike = None) -> None:
        dynamics = self.rigid_dynamics
        friction_torsional, geoms_idx, _ = sanitize_io_variables(
            self._scene,
            friction_torsional,
            geoms_idx,
            dynamics.n_geoms,
            "geoms_idx",
            envs_idx=None,
            batched=False,
            skip_allocation=True,
        )
        kernel_set_geoms_friction_torsional(
            geoms_idx, friction_torsional, dynamics.data.dyn_info, dynamics.rigid_config
        )

    def set_geoms_friction_rolling(self, friction_rolling: TensorLike, geoms_idx: IndexLike = None) -> None:
        dynamics = self.rigid_dynamics
        friction_rolling, geoms_idx, _ = sanitize_io_variables(
            self._scene,
            friction_rolling,
            geoms_idx,
            dynamics.n_geoms,
            "geoms_idx",
            envs_idx=None,
            batched=False,
            skip_allocation=True,
        )
        kernel_set_geoms_friction_rolling(geoms_idx, friction_rolling, dynamics.data.dyn_info, dynamics.rigid_config)

    def update_drone_propeller_vgeoms(
        self,
        propellers_vgeom_idxs: TensorLike,
        propellers_revs: TensorLike,
        propellers_spin: TensorLike,
    ) -> None:
        dynamics = self.rigid_dynamics
        kernel_update_drone_propeller_vgeoms(
            propellers_vgeom_idxs,
            propellers_revs,
            propellers_spin,
            dynamics.data.dyn_state,
            dynamics.data.rigid_info,
            dynamics.rigid_config,
        )

    def set_drone_rpm(
        self,
        propellers_link_idx: TensorLike,
        kf: TensorLike,
        km: TensorLike,
        propellers_rpm: TensorLike,
        propellers_spin: TensorLike,
        invert: bool,
    ) -> None:
        dynamics = self.rigid_dynamics
        kernel_set_drone_rpm(
            propellers_link_idx,
            kf,
            km,
            propellers_rpm,
            propellers_spin,
            dynamics.data.dyn_state,
            dynamics.rigid_config,
            invert,
        )

    def update_verts_for_geoms(self, geoms_idx: IndexLike) -> None:
        dynamics = self.rigid_dynamics
        _, geoms_idx, _ = sanitize_io_variables(
            self._scene, None, geoms_idx, dynamics.n_geoms, "geoms_idx", envs_idx=None, skip_allocation=True
        )
        kernel_update_verts_for_geoms(geoms_idx, dynamics.data.dyn_state, dynamics.data.dyn_info, dynamics.rigid_config)

    def get_fixed_collision_verts(self, verts_idx: IndexLike) -> torch.Tensor:
        """Read fixed collision vertices from the Rigid-owned state."""
        return qd_to_torch(self.rigid_dynamics.data.dyn_state.fixed_verts.pos, verts_idx, copy=True)

    def get_free_collision_verts(self, verts_idx: IndexLike) -> torch.Tensor:
        """Read per-environment collision vertices from the Rigid-owned state."""
        dynamics = self.rigid_dynamics
        tensor = qd_to_torch(dynamics.data.dyn_state.free_verts.pos, None, verts_idx, transpose=True, copy=True)
        return tensor[0] if dynamics.n_envs == 0 else tensor

    def get_vgeom_transforms_batched(self, envs_idx: IndexLike = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Read visual-geometry transforms while preserving the physical batch dimension."""
        dynamics = self.rigid_dynamics
        self.update_vgeoms()
        pos = qd_to_torch(dynamics.data.dyn_state.vgeoms.pos, envs_idx, transpose=True, copy=True)
        quat = qd_to_torch(dynamics.data.dyn_state.vgeoms.quat, envs_idx, transpose=True, copy=True)
        return pos, quat

    def get_error_envs_mask(self) -> torch.Tensor:
        return qd_to_torch(self.rigid_dynamics.data.errno) > 0

    def apply_links_external_wrench(
        self,
        force: TensorLike | None = None,
        torque: TensorLike | None = None,
        links_idx: IndexLike = None,
        envs_idx: IndexLike = None,
        *,
        pos: TensorLike | None = None,
        ref: link_ref_frame = link_ref_frame.link_origin,
        local: bool = False,
    ) -> None:
        """
        Apply an external wrench over one simulation step on a set of links.

        Parameters
        ----------
        force : None | array_like, optional
            The linear force to apply. None for a pure torque. Defaults to None.
        torque : None | array_like, optional
            The torque to apply, on top of the moment induced by the linear force. Defaults to None.
        links_idx : None | array_like, optional
            The indices of the links on which to apply the wrench. None to specify all links. Default to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        pos : None | array_like, optional
            Where the linear force is applied, which sets the moment arm of the induced torque. With `local=True`, an
            offset from the origin of the `ref` frame, so the point follows each link as it moves; otherwise a world
            position. None applies the force at the origin of the `ref` frame. Defaults to None.
        ref: gs.link_ref_frame, optional
            The reference frame: the origin of each link ('link_origin'), or its center of mass ('link_COM'). It fixes
            where the linear force acts when `pos` is None, and the axes of the input coordinates when `local=True`.
            Defaults to 'link_origin'.
        local: bool, optional
            Whether `force`, `torque` and `pos` are expressed in the coordinates of the `ref` frame rather than the
            world frame. Defaults to False.
        """
        dynamics = self.rigid_dynamics
        if force is None and torque is None:
            gs.raise_exception("Either 'force' or 'torque' must be specified.")
        if force is None and pos is not None:
            gs.raise_exception("'pos' requires 'force', as a torque acts the same wherever it is applied.")
        ref_frame = sanitize_ref_frame(ref, has_root_com=False)

        if pos is not None:
            pos, _, _ = sanitize_io_variables(
                self._scene, pos, links_idx, dynamics.n_links, "links_idx", envs_idx, (3,), skip_allocation=True
            )
        # An absent wrench component contributes nothing, which the kernel spells out as zeros.
        force = 0.0 if force is None else force
        torque = 0.0 if torque is None else torque
        force, _, _ = sanitize_io_variables(
            self._scene, force, links_idx, dynamics.n_links, "links_idx", envs_idx, (3,), skip_allocation=True
        )
        torque, links_idx, envs_idx = sanitize_io_variables(
            self._scene, torque, links_idx, dynamics.n_links, "links_idx", envs_idx, (3,), skip_allocation=True
        )
        if dynamics.n_envs == 0:
            force, torque = force[None], torque[None]
            if pos is not None:
                pos = pos[None]

        # A wrench on a sleeping body must revive it, otherwise the input is silently dropped.
        if dynamics.rigid_config.use_hibernation:
            self.rigid_island.wake_entities_by_links(links_idx, envs_idx)

        if pos is None:
            kernel_apply_links_external_wrench(
                links_idx, envs_idx, force, torque, dynamics.data.dyn_state, dynamics.rigid_config, ref_frame, local
            )
        else:
            kernel_apply_links_external_wrench_at_pos(
                links_idx,
                envs_idx,
                pos,
                force,
                torque,
                dynamics.data.dyn_state,
                dynamics.rigid_config,
                ref_frame,
                local,
            )

    def get_links_net_contact_force(self, links_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        dynamics = self.rigid_dynamics
        tensor = qd_to_torch(
            dynamics.data.dyn_state.links.contact_force,
            envs_idx,
            links_idx,
            transpose=True,
            copy=True,
        )
        return tensor[0] if dynamics.n_envs == 0 else tensor

    def get_dofs_actuator_force(self, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        dynamics = self.rigid_dynamics
        dofs_state, dofs_info = dynamics.data.dyn_state.dofs, dynamics.data.dyn_info.dofs
        qf_applied = qd_to_torch(dofs_state.qf_applied, envs_idx, transpose=True)
        if self.rigid_constraint is None:
            qacc = qd_to_torch(dofs_state.acc, envs_idx, transpose=True)
        else:
            qacc = self.rigid_constraint.get_constrained_acc(envs_idx)
        qf_passive = qd_to_torch(dofs_state.qf_passive, envs_idx, transpose=True)
        if dynamics.options.batch_dofs_info:
            armature = qd_to_torch(dofs_info.armature, envs_idx, transpose=True)
        else:
            armature = qd_to_torch(dofs_info.armature, transpose=True)
        if self.rigid_frictionloss_constraint is None:
            qf_frictionloss = torch.zeros_like(qf_applied)
        else:
            qf_frictionloss = self.rigid_frictionloss_constraint.get_dofs_frictionloss_force(envs_idx)

        actuator_force = qf_applied - armature * qacc + qf_frictionloss + qf_passive
        if dofs_idx is not None:
            actuator_force = actuator_force[indices_to_mask(None, dofs_idx)]
        return actuator_force[0] if dynamics.n_envs == 0 else actuator_force

    # -------------------------------- Implementation --------------------------------

    def _set_links_info(
        self,
        values: TensorLike,
        links_idx: IndexLike,
        name: str,
        envs_idx: IndexLike = None,
        *,
        scale_inertia: bool = False,
    ) -> None:
        """Write one inertial property of the given links, then recompute the inverse weights of their trees once.

        The property is the mass, the center of mass or the inertia, selected by `name`. The inverse weights are per
        environment even when the property itself is shared by the whole batch, so writing shared link info recomputes
        them for every environment.
        """
        dynamics = self.rigid_dynamics
        if not dynamics.options.batch_links_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched links info.")
        shape = {"mass": (), "COM": (3,), "inertia": (3, 3)}[name]
        # A value of fewer dimensions than the property holds would be spread over it, which for an inertia means a
        # tensor of one number repeated: no inertia of any body.
        if shape and np.shape(values)[-len(shape) :] != shape:
            gs.raise_exception(f"{name} is {shape} per link, and {np.shape(values)} cannot be read as that.")
        # The anchor check below reads the written indices as given, in any form the sanitizer accepts. A tensor on a
        # GPU device is read back in debug mode only, since that stalls the GPU.
        links_idx_ = None
        if links_idx is None:
            links_idx_ = range(dynamics.n_links)
        elif not isinstance(links_idx, torch.Tensor) or gs.debug or links_idx.device.type == "cpu":
            (links_idx_,) = indices_to_mask(links_idx, keepdim=False, to_torch=False, boolean_mask=False)
            if isinstance(links_idx_, slice):
                links_idx_ = range(*links_idx_.indices(dynamics.n_links))
            elif isinstance(links_idx_, torch.Tensor):
                links_idx_ = tensor_to_array(links_idx_)
            if np.ndim(links_idx_) == 0:
                links_idx_ = (links_idx_,)
        values, links_idx, envs_idx = sanitize_io_variables(
            self._scene,
            values,
            links_idx,
            dynamics.n_links,
            "links_idx",
            envs_idx,
            element_shape=shape,
            batched=dynamics.options.batch_links_info,
            skip_allocation=True,
        )
        if links_idx_ is not None:
            # TODO: the anchor of an aligned root keeps its mass block diagonal (see _init_tree_fields). A center of
            # mass or an inertia written on any link of the body moves the anchor. A mass write keeps it only as one
            # rescale of every link of the body, inertia included. A frame move shifts the local pose of every geom
            # ('GeomsInfo.pos' / 'quat', one entry per geom for all environments). It also shifts 'qpos', quoted in the
            # anchored frame, and the origin a fixed child reports. A per-environment value has no frame to go to, so
            # the guard stays until 'GeomsInfo' gets a batch dimension.
            links_anchor = []
            for link in dynamics.links:
                anchor = link
                while anchor.parent_idx != -1 and all(joint.type == gs.JOINT_TYPE.FIXED for joint in anchor.joints):
                    anchor = dynamics.links[anchor.parent_idx]
                links_anchor.append(anchor if anchor.aligned else None)
            values_idx_by_link = {i_l: i_col for i_col, i_l in enumerate(links_idx_)}
            for i_l in values_idx_by_link:
                anchor = links_anchor[i_l]
                if anchor is None:
                    continue
                links_idx_body = [link.idx for link in dynamics.links if links_anchor[link.idx] is anchor]
                if name == "mass" and len(links_idx_body) == 1:
                    continue
                if name == "mass" and scale_inertia and all(i_b in values_idx_by_link for i_b in links_idx_body):
                    # The rescale check reads the values back and stalls the GPU, so it runs in debug mode only.
                    if not gs.debug:
                        continue
                    ratios = values[..., [values_idx_by_link[i_b] for i_b in links_idx_body]]
                    ratios = ratios / self.get_links_mass(links_idx_body, envs_idx)
                    if torch.allclose(ratios, ratios[..., :1], rtol=gs.EPS, atol=gs.EPS):
                        continue
                link = dynamics.links[i_l]
                remedy = (
                    "Load the entity with 'align=False' to write inertial properties at runtime."
                    if isinstance(link.entity.main_morph, gs.options.morphs.FileMorph)
                    else "Writing the inertial properties of a primitive morph is not supported yet."
                )
                what = {"mass": "mass", "COM": "center of mass", "inertia": "inertia"}[name]
                gs.raise_exception(
                    f"Cannot set the {what} of link '{link.name}': the frame of its body is anchored on the center of "
                    f"mass and principal axes of all its links. Set the mass of the whole entity instead. {remedy}"
                )

        envs_idx = envs_idx if dynamics.options.batch_links_info else self._scene._envs_idx
        if not dynamics.options.batch_links_info or dynamics.n_envs == 0:
            values = values[None]
        # The weighing reads each tree at its neutral configuration, which a sleeping body is not carried to, so the
        # islands wake before it.
        self.rigid_island.wake_link_islands(links_idx, envs_idx)
        if name == "mass":
            kernel_set_links_mass(
                links_idx,
                envs_idx,
                values,
                dynamics.data.dyn_info,
                dynamics.rigid_config,
                scale_inertia,
            )
        elif name == "COM":
            kernel_set_links_COM(links_idx, envs_idx, values, dynamics.data.dyn_info, dynamics.rigid_config)
        else:
            kernel_set_links_inertia(links_idx, envs_idx, values, dynamics.data.dyn_info, dynamics.rigid_config)
        # The weights are solved again rather than scaled by the mass ratio: a weight follows the mass alone for a body
        # whose degrees of freedom carry no armature, which a model loaded from a file has. An inertia moves no center
        # of mass, so every velocity stands where it stood.
        dynamics.refresh_links_inertial_weights(links_idx, envs_idx, moves_com=name != "inertia")

    def _set_dofs_info(
        self,
        tensor_list: Sequence[TensorLike],
        dofs_idx: IndexLike,
        name: str,
        envs_idx: IndexLike = None,
    ) -> None:
        dynamics = self.rigid_dynamics
        if gs.use_zerocopy and name in {
            "kp",
            "kv",
            "act_gain",
            "act_bias",
            "force_range",
            "stiffness",
            "damping",
            "frictionloss",
            "limit",
        }:
            mask = indices_to_mask(*((envs_idx, dofs_idx) if dynamics.options.batch_dofs_info else (dofs_idx,)))
            if name == "kp":
                # kp sets act_gain, act_bias[0] = 0, act_bias[1] = -kp (full PD reset)
                kp = torch.as_tensor(tensor_list[0], dtype=gs.tc_float, device=gs.device)
                gain = qd_to_torch(dynamics.data.dyn_info.dofs.act_gain, transpose=True, copy=False)
                assign_indexed_tensor(gain, mask, kp)
                bias = qd_to_torch(dynamics.data.dyn_info.dofs.act_bias, transpose=True, copy=False)
                bias[(*mask, ..., 0)] = 0.0
                assign_indexed_tensor(bias, (*mask, ..., 1), -kp)
            elif name == "kv":
                # kv sets act_bias[..., 2] = -kv
                kv = torch.as_tensor(tensor_list[0], dtype=gs.tc_float, device=gs.device)
                bias = qd_to_torch(dynamics.data.dyn_info.dofs.act_bias, transpose=True, copy=False)
                assign_indexed_tensor(bias, (*mask, ..., 2), -kv)
            else:
                data = qd_to_torch(getattr(dynamics.data.dyn_info.dofs, name), transpose=True, copy=False)
                num_values = len(tensor_list)
                for j, mask_j in enumerate(((*mask, ..., j) for j in range(num_values)) if num_values > 1 else (mask,)):
                    assign_indexed_tensor(data, mask_j, tensor_list[j])
            if gs.backend == gs.metal:
                torch.mps.synchronize()
            return

        tensor_list = list(tensor_list)
        for j, tensor in enumerate(tensor_list):
            tensor, dofs_idx, envs_idx_ = sanitize_io_variables(
                self._scene,
                tensor,
                dofs_idx,
                dynamics.n_dofs,
                "dofs_idx",
                envs_idx,
                batched=dynamics.options.batch_dofs_info,
                skip_allocation=True,
            )
            if dynamics.n_envs == 0 and dynamics.options.batch_dofs_info:
                tensor = tensor[None]
            tensor_list[j] = tensor
        if name == "kp":
            kernel_set_dofs_kp(dofs_idx, envs_idx_, *tensor_list, dynamics.data.dyn_info, dynamics.rigid_config)
        elif name == "kv":
            kernel_set_dofs_kv(dofs_idx, envs_idx_, *tensor_list, dynamics.data.dyn_info, dynamics.rigid_config)
        elif name == "force_range":
            kernel_set_dofs_force_range(
                dofs_idx, envs_idx_, *tensor_list, dynamics.data.dyn_info, dynamics.rigid_config
            )
        elif name == "stiffness":
            kernel_set_dofs_stiffness(dofs_idx, envs_idx_, *tensor_list, dynamics.data.dyn_info, dynamics.rigid_config)
        elif name == "armature":
            kernel_set_dofs_armature(dofs_idx, envs_idx_, *tensor_list, dynamics.data.dyn_info, dynamics.rigid_config)
            dynamics.refresh_invweight_and_meaninertia(envs_idx=envs_idx, force_update=True)
        elif name == "damping":
            kernel_set_dofs_damping(dofs_idx, envs_idx_, *tensor_list, dynamics.data.dyn_info, dynamics.rigid_config)
        elif name == "frictionloss":
            kernel_set_dofs_frictionloss(
                dofs_idx, envs_idx_, *tensor_list, dynamics.data.dyn_info, dynamics.rigid_config
            )
        elif name == "limit":
            kernel_set_dofs_limit(dofs_idx, envs_idx_, *tensor_list, dynamics.data.dyn_info, dynamics.rigid_config)
        elif name == "act_gain":
            kernel_set_dofs_act_gain(dofs_idx, envs_idx_, *tensor_list, dynamics.data.dyn_info, dynamics.rigid_config)
        elif name == "act_bias":
            kernel_set_dofs_act_bias(dofs_idx, envs_idx_, *tensor_list, dynamics.data.dyn_info, dynamics.rigid_config)
        else:
            gs.raise_exception(f"Invalid `name` {name}.")

    def _set_kinematic_dofs_velocity(
        self,
        velocity: TensorLike | None,
        dofs_idx: IndexLike = None,
        envs_idx: IndexLike = None,
        *,
        skip_forward: bool = False,
    ) -> None:
        dynamics = self.rigid_dynamics
        if gs.use_zerocopy:
            vel = qd_to_torch(dynamics.data.dyn_state.dofs.vel, transpose=True, copy=False)
            dofs_mask = indices_to_mask(dofs_idx)
            if (
                (not dofs_mask or isinstance(dofs_mask[0], slice))
                and isinstance(envs_idx, torch.Tensor)
                and (
                    (velocity is None and (not IS_OLD_TORCH or envs_idx.dtype == torch.bool))
                    or (velocity is not None and velocity.ndim == 2 and envs_idx.dtype == torch.bool)
                )
            ):
                dofs_vel = vel[(slice(None), *dofs_mask)]
                if velocity is None:
                    if envs_idx.dtype == torch.bool:
                        dofs_vel.masked_fill_(envs_idx[:, None], 0.0)
                    else:
                        dofs_vel.scatter_(0, envs_idx[:, None].expand((-1, dofs_vel.shape[1])), 0.0)
                else:
                    if velocity.ndim == 2 and len(dofs_vel) != len(velocity):
                        # Note that it is necessary to create a new temporary view because it will be reshaped in-place
                        dofs_vel.masked_scatter_(envs_idx[:, None], velocity.view_as(velocity))
                    else:
                        velocity = broadcast_tensor(velocity, gs.tc_float, dofs_vel.shape)
                        torch.where(envs_idx[:, None], velocity, dofs_vel, out=dofs_vel)
            else:
                mask = (
                    (0, *dofs_mask)
                    if dynamics.n_envs == 0
                    else indices_to_mask(envs_idx, *dofs_mask, boolean_mask=False)
                )
                if velocity is None:
                    vel[mask] = 0.0
                else:
                    assign_indexed_tensor(vel, mask, velocity)
                if mask and isinstance(mask[0], torch.Tensor):
                    envs_idx = mask[0].reshape((-1,))
            if gs.backend == gs.metal:
                torch.mps.synchronize()
            if not skip_forward and not isinstance(envs_idx, torch.Tensor):
                envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        else:
            velocity, dofs_idx, envs_idx = sanitize_io_variables(
                self._scene, velocity, dofs_idx, dynamics.n_dofs, "dofs_idx", envs_idx, skip_allocation=True
            )
            if velocity is None:
                kernel_set_dofs_zero_velocity(dofs_idx, envs_idx, dynamics.data.dyn_state, dynamics.rigid_config)
            else:
                if dynamics.n_envs == 0:
                    velocity = velocity[None]
                kernel_set_dofs_velocity(dofs_idx, envs_idx, velocity, dynamics.data.dyn_state, dynamics.rigid_config)

        if not skip_forward:
            dynamics.update_forward_velocity(envs_idx)
        else:
            dynamics.invalidate_forward_velocity()

    @property
    def _kinematics_scratch(self) -> array_class.KinematicsScratch:
        """The scratch of the Jacobian and forward-kinematics queries, allocated on first use."""
        if self._kinematics_scratch_cache is None:
            dynamics = self.rigid_dynamics
            self._kinematics_scratch_cache = array_class.get_kinematics_scratch(
                entities=dynamics.entities, batch_size=dynamics.batch_size, n_qs_actual=dynamics.n_qs
            )
        return self._kinematics_scratch_cache

    @property
    def _ik_scratch(self) -> array_class.IKScratch:
        """The scratch of inverse kinematics, allocated on the first call as upstream does."""
        if self._ik_scratch_cache is None:
            dynamics = self.rigid_dynamics
            self._ik_scratch_cache = array_class.get_ik_scratch(
                options=dynamics.options, entities=dynamics.entities, batch_size=dynamics.batch_size
            )
        return self._ik_scratch_cache


@register_system(SimulatorEngine)
def create_rigid_api(engine: SimulatorEngine) -> RigidAPI | None:
    """Admit the Rigid API under the condition that admits RigidDynamics."""
    return RigidAPI(engine.scene) if any(isinstance(entity, RigidEntity) for entity in engine.scene.entities) else None
