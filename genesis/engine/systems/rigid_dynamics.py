"""Manager System of the rigid solver, owning its data and exposing stage Actions composed around a host solve.

The initialization of the rigid data extends its kinematic fields through the implementation sections at the end of
this class.
"""

from collections.abc import Callable, Iterator
from dataclasses import asdict
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
import quadrants as qd
import torch

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
from genesis.engine.core import Data, System, register_system
from genesis.engine.entities import DroneEntity, RigidEntity
from genesis.engine.solvers.rigid.kinematic_tools import balanced_variant_mapping as _balanced_variant_mapping
from genesis.engine.solvers.rigid.kinematic_tools import fill_base_link_geom_offsets as _fill_base_link_geom_offsets
from genesis.engine.solvers.rigid.abd.accessor import (
    kernel_get_state,
    kernel_get_state_grad,
    kernel_refresh_links_invweight_and_meaninertia,
    kernel_set_dofs_force_grad,
    kernel_set_dofs_velocity_grad,
    kernel_set_links_pos_grad,
    kernel_set_links_quat_grad,
    kernel_set_state,
)
from genesis.engine.solvers.rigid.abd.diff import (
    kernel_begin_backward_substep,
    kernel_copy_acc,
    kernel_copy_next_to_curr_no_check,
    kernel_prepare_backward_substep,
    kernel_save_adjoint_cache,
)
from genesis.engine.solvers.rigid.abd.forward_dynamics import (
    func_forward_dynamics,
    kernel_forward_dynamics_without_qacc,
    kernel_refresh_invweight_and_meaninertia,
    kernel_update_acc,
    update_qacc_from_qvel_delta,
    update_qvel,
)
from genesis.engine.solvers.rigid.abd.forward_kinematics import (
    kernel_COM_links_replay,
    kernel_forward_kinematics_links_geoms,
    kernel_forward_kinematics_replay,
    kernel_forward_velocity,
    kernel_masked_forward_kinematics_links_geoms,
    kernel_masked_forward_velocity,
    kernel_update_cartesian_space,
    kernel_update_geoms_replay,
    kernel_update_vgeoms,
    kernel_update_vverts_for_vgeoms,
)
from genesis.engine.solvers.rigid.abd.manual_bw import (
    kernel_manual_compute_qacc_bw,
    kernel_manual_forward_kinematics_bw,
    kernel_manual_forward_velocity_bw,
)
from genesis.engine.solvers.rigid.abd.misc import (
    kernel_clear_external_force,
    kernel_init_dof_fields,
    kernel_init_entity_fields,
    kernel_init_equality_fields,
    kernel_init_geom_fields,
    kernel_init_joint_fields,
    kernel_init_link_dynamics,
    kernel_init_link_fields,
    kernel_init_vert_fields,
    kernel_init_vgeom_fields,
    kernel_init_vvert_fields,
    kernel_reset_hibernation,
    kernel_set_zero,
    kernel_update_heterogeneous_link_info,
    kernel_update_heterogeneous_links_vgeom,
)
from genesis.engine.solvers.rigid.rigid_tools import (
    jacobi_mass_spread_bound,
    resolve_broadphase_traversal,
    sanitize_io_variables,
    sanitize_sol_params,
)
from genesis.engine.states import RigidSolverState
from genesis.engine.states.cache import QueriedStates
from genesis.options.solvers import RigidOptions
from genesis.utils.misc import (
    get_gpu_core_count,
    get_gpu_shared_tile_sizes,
    indices_to_mask,
    qd_to_numpy,
    qd_to_torch,
    qd_zero_grad,
)

from genesis.engine.solvers.rigid.substep import func_step_1, func_step_2, func_substep_wakeup_coupled_links
from ..simulator_engine import SimulatorEngine

if TYPE_CHECKING:
    from genesis.engine.scene import Scene

TensorLike: TypeAlias = ArrayLike | torch.Tensor
IndexLike: TypeAlias = ArrayLike | torch.Tensor | None
HostTensor: TypeAlias = NDArray[np.generic] | torch.Tensor


IS_OLD_TORCH = tuple(map(int, torch.__version__.split(".")[:2])) < (2, 8)


class RigidDynamics(System):
    """Own the Rigid model, dynamics Data, numerical Actions and ordinary backend APIs.

    SimulatorEngine initialization has it allocate and initialize rigid dynamics from the collected Scene model.
    It does not collect or order Contact/Constraint Actions; SimulatorEngine composes those independent owners.
    """

    # -------------------------------- Data --------------------------------

    @qd.data_oriented
    class Data(Data):
        """The array_class structs owned by rigid dynamics.

        ``@qd.data_oriented`` is a QD discovery workaround: ``GlobalDataContext`` can expose this root only in a form
        QD traverses. Remove it when QD derives ndarray parameters through ordinary non-data-oriented objects.
        """

        geoms_init_AABB: array_class.GeomsInitAABB
        dyn_state: array_class.DynState
        dyn_info: array_class.DynInfo
        rigid_info: array_class.RigidInfo
        weight_scratch: array_class.WeightScratch
        rigid_adjoint_cache: array_class.RigidAdjointCache
        dyn_state_adjoint_cache: array_class.DynState | None
        errno: qd.Tensor

    data: Data

    def __init__(self, scene: "Scene") -> None:
        super().__init__()
        self._scene = scene
        self.data = self.Data()
        self.scene = scene
        self.sim = self._sim = scene.sim
        self._options = scene.options.rigid
        self._B = scene._B
        self.n_envs = scene.n_envs
        self._para_level = scene._para_level
        self._requires_grad = self.sim.options.requires_grad
        self._enable_collision = self._options.enable_collision
        self._enable_multi_contact = self._options.enable_multi_contact
        self._enable_mujoco_compatibility = self._options.enable_mujoco_compatibility
        self._enable_joint_limit = self._options.enable_joint_limit
        self._enable_self_collision = self._options.enable_self_collision
        self._enable_neutral_collision = self._options.enable_neutral_collision
        self._enable_adjacent_collision = self._options.enable_adjacent_collision
        self._disable_constraint = self._options.disable_constraint
        self._queried_states = QueriedStates()
        self._ckpt = {}

    def build(self) -> None:
        from .rigid_island import RigidIsland

        self.rigid_island = self.require(RigidIsland)

    # -------------------------------- Actions --------------------------------

    @System.action(kind="host")
    def collect(self):
        return self.collect_model

    @System.action(kind="host")
    def initialize(self):
        return self.initialize_data

    @System.action(kind="host")
    def initialize_inverse_weights(self):
        return self._initialize_inverse_weights

    @System.action(kind="stage")
    def wake_coupled_links_stage(self):
        return func_substep_wakeup_coupled_links, self.data, self.rigid_island.data, self.rigid_config

    @System.action(kind="host")
    def wake_coupled_links_kernel(self):
        return kernel_substep_wakeup_coupled_links, self.data, self.rigid_island.data, self.rigid_config

    @System.action(kind="stage")
    def forward_dynamics_stage(self):
        return func_step_1, self.data, self.rigid_island.data, self.rigid_config

    @System.action(kind="host")
    def forward_dynamics_kernel(self):
        return kernel_step_1, self.data, self.rigid_island.data, self.rigid_config

    @System.action(kind="stage")
    def integrate_stage(self):
        return func_step_2, self.data, self.rigid_config

    @System.action(kind="host")
    def integrate_kernel(self):
        return kernel_step_2, self.data, self.rigid_config

    @System.action(kind="host")
    def mark_forward_updated(self):
        return self._mark_forward_updated

    @System.action(kind="host")
    def save_adjoint_cache(self):
        return self._save_adjoint_cache

    @System.action(kind="host")
    def restore_substep_inputs(self):
        return self._restore_substep_inputs

    @System.action(kind="host")
    def reverse_integration(self):
        return self._reverse_integration

    @System.action(kind="host")
    def reverse_smooth_acceleration(self):
        return self._reverse_smooth_acceleration

    @System.action(kind="host")
    def reverse_forward_dynamics(self):
        return self._reverse_forward_dynamics

    # -------------------------------- Public get/set API --------------------------------

    @property
    def options(self) -> RigidOptions:
        return self._options

    @property
    def batch_size(self) -> int:
        return self._B

    def get_n_candidate_equalities(self) -> int:
        return self.n_candidate_equalities_

    @property
    def weight_scratch(self) -> array_class.WeightScratch:
        return self.data.weight_scratch

    def refresh_invweight_and_meaninertia(
        self,
        envs_idx: IndexLike = None,
        *,
        force_update: bool = True,
        in_place: bool = False,
    ) -> None:
        """Recompute the inverse weights and the mean inertia from the inertial properties of the links."""
        # Every kinematic tree of the solver is recomputed. A change limited to some links is handled by the setter
        # writing them, which recomputes their own tree only.
        # Early return if no DoFs. This is essential to avoid segfault on CUDA.
        if self._n_dofs == 0:
            return

        # Handling default arguments
        batched = self._options.batch_dofs_info or self._options.batch_links_info
        if not batched and envs_idx is not None:
            gs.raise_exception(
                "Links and dofs must be batched to selectively update invweight and meaninertia for some environment."
            )
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)

        jac_row = self.weight_scratch.jac_row
        solve_out = self.weight_scratch.solve_out
        kernel_refresh_invweight_and_meaninertia(
            envs_idx,
            jac_row,
            solve_out,
            self.dyn_state,
            self.rigid_island.data.island,
            self.dyn_info,
            self.rigid_info,
            self.rigid_config,
            force_update,
            self._is_forward_pos_updated,
            self._is_forward_vel_updated,
        )

    def update_forward_pos(self) -> None:
        if self._is_forward_pos_updated:
            return
        kernel_forward_kinematics_links_geoms(
            self._scene._envs_idx,
            self.dyn_state,
            self.dyn_info,
            self.rigid_info,
            self.rigid_config,
        )
        self._is_forward_pos_updated = True

    def update_forward_kinematics(self, envs_idx: torch.Tensor) -> None:
        """Bring the cartesian state of these environments up to date with the configuration written to them."""
        if envs_idx.dtype == torch.bool:
            fn = kernel_masked_forward_kinematics_links_geoms
        else:
            fn = kernel_forward_kinematics_links_geoms
        fn(envs_idx, self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config)
        self._is_forward_pos_updated = True
        self._is_forward_vel_updated = True

    def update_forward_velocity(self, envs_idx: torch.Tensor) -> None:
        """Bring the cartesian velocities of these environments up to date with the dof velocities written to them."""
        if envs_idx.dtype == torch.bool:
            fn = kernel_masked_forward_velocity
        else:
            fn = kernel_forward_velocity
        fn(envs_idx, self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config, is_backward=False)
        self._is_forward_vel_updated = True

    def invalidate_forward_velocity(self) -> None:
        self._is_forward_vel_updated = False

    def refresh_links_inertial_weights(
        self, links_idx: torch.Tensor, envs_idx: torch.Tensor, *, moves_com: bool
    ) -> None:
        """Weigh again the trees of these links once their inertial properties are written."""
        kernel_refresh_links_invweight_and_meaninertia(
            links_idx,
            envs_idx,
            self.data.weight_scratch.jac_row,
            self.data.weight_scratch.solve_out,
            self.dyn_state,
            self.dyn_info,
            self.rigid_info,
            self.rigid_config,
            self._is_forward_pos_updated,
            self._is_forward_vel_updated and moves_com,
        )

    def iter_data(self) -> Iterator[array_class.DataItem | array_class.ConfigItem]:
        """Yield public Rigid data through the owner composition boundary."""
        yield from array_class.iter_data(self, "rigid_config")
        yield from array_class.iter_data(self.data, "errno", array_class.DataKind.STATE)
        structs = ["rigid_info", "dyn_info", "dyn_state", "rigid_adjoint_cache"]
        if self.rigid_config.requires_grad:
            structs.append("dyn_state_adjoint_cache")
        for name in structs:
            yield from array_class.iter_data(self.data, name)
        yield from array_class.iter_data(self, "_is_forward_pos_updated", array_class.DataKind.DERIVED)
        yield from array_class.iter_data(self, "_is_forward_vel_updated", array_class.DataKind.DERIVED)

    def read_errno(self) -> int:
        if gs.use_zerocopy:
            return int(np.bitwise_or.reduce(qd_to_numpy(self._errno)))
        from genesis.engine.solvers.rigid.abd.misc import kernel_bit_reduction

        return int(kernel_bit_reduction(self._errno))

    def add_grad_from_state(self, state: RigidSolverState) -> None:
        qpos_grad = gs.zeros_like(state.qpos)
        dofs_vel_grad = gs.zeros_like(state.dofs_vel)
        links_pos_grad = gs.zeros_like(state.links_pos)
        links_quat_grad = gs.zeros_like(state.links_quat)

        if state.qpos.grad is not None:
            qpos_grad = state.qpos.grad
        if state.dofs_vel.grad is not None:
            dofs_vel_grad = state.dofs_vel.grad
        if state.links_pos.grad is not None:
            links_pos_grad = state.links_pos.grad
        if state.links_quat.grad is not None:
            links_quat_grad = state.links_quat.grad

        kernel_get_state_grad(
            qpos_grad,
            dofs_vel_grad,
            links_pos_grad,
            links_quat_grad,
            self.dyn_state,
            self.rigid_info,
            self.rigid_config,
        )

    def collect_output_grads(self) -> None:
        """
        Collect gradients from downstream queried states.
        """
        if self._sim.cur_step_global in self._queried_states:
            # one step could have multiple states
            assert len(self._queried_states[self._sim.cur_step_global]) == 1
            state = self._queried_states[self._sim.cur_step_global][0]
            self.add_grad_from_state(state)

    def set_base_links_pos_grad(
        self,
        links_idx: IndexLike,
        envs_idx: IndexLike,
        relative: bool,
        pos_grad: torch.Tensor,
    ) -> None:
        if links_idx is None:
            links_idx = self.base_links_idx
        # A relative 'set_pos' adds 'R(authored_quat) @ offset_pos', which reads the current orientation; that
        # dependency is not propagated, so with a non-zero offset position the backward pass is only supported on links
        # without one. 'relative=False' sets the world position directly and is a plain passthrough.
        idx = links_idx if isinstance(links_idx, int) else slice(None)
        if relative and not self.links_offset_pos_is_identity[idx].all():
            gs.raise_exception(
                "Backward pass for 'set_pos' with 'relative=True' is only supported on links without an offset "
                "position (no inertial alignment shifting the link origin). Use 'relative=False' to set the world "
                "position."
            )
        pos_grad_, links_idx, envs_idx = sanitize_io_variables(
            self._scene,
            pos_grad.unsqueeze(-2),
            links_idx,
            self.n_links,
            "links_idx",
            envs_idx,
            (3,),
            skip_allocation=True,
        )
        if self.n_envs == 0:
            pos_grad_ = pos_grad_.unsqueeze(0)
        kernel_set_links_pos_grad(
            links_idx, envs_idx, pos_grad_, self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config
        )

    def set_base_links_quat_grad(
        self,
        links_idx: IndexLike,
        envs_idx: IndexLike,
        relative: bool,
        quat_grad: torch.Tensor,
    ) -> None:
        if links_idx is None:
            links_idx = self.base_links_idx
        # A relative 'set_quat' composes the orientation offset (a constant right-multiplication) and, when the offset
        # position is non-zero, also rewrites the world position from the input orientation. Neither jacobian is
        # propagated, so the backward pass is only supported when the targeted links carry no pose offset. With
        # 'relative=False' the world orientation is set directly, which is a plain passthrough and always differentiable.
        idx = links_idx if isinstance(links_idx, int) else slice(None)
        if relative and not (
            self.links_offset_quat_is_identity[idx].all() and self.links_offset_pos_is_identity[idx].all()
        ):
            gs.raise_exception(
                "Backward pass for 'set_quat' with 'relative=True' is only supported on links without a pose offset "
                "(no up-axis conversion or inertial alignment). Use 'relative=False' to set the world orientation."
            )
        quat_grad_, links_idx, envs_idx = sanitize_io_variables(
            self._scene,
            quat_grad.unsqueeze(-2),
            links_idx,
            self.n_links,
            "links_idx",
            envs_idx,
            (4,),
            skip_allocation=True,
        )
        if self.n_envs == 0:
            quat_grad_ = quat_grad_.unsqueeze(0)
        kernel_set_links_quat_grad(
            links_idx, envs_idx, quat_grad_, self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config
        )

    def set_dofs_velocity_grad(self, dofs_idx: IndexLike, envs_idx: IndexLike, velocity_grad: torch.Tensor) -> None:
        velocity_grad_, dofs_idx, envs_idx = sanitize_io_variables(
            self._scene, velocity_grad, dofs_idx, self.n_dofs, "dofs_idx", envs_idx, skip_allocation=True
        )
        if self.n_envs == 0:
            velocity_grad_ = velocity_grad_.unsqueeze(0)
        kernel_set_dofs_velocity_grad(dofs_idx, envs_idx, velocity_grad_, self.dyn_state, self.rigid_config)

    def set_dofs_force_grad(self, dofs_idx: IndexLike, envs_idx: IndexLike, force_grad: torch.Tensor) -> None:
        force_grad_, dofs_idx, envs_idx = sanitize_io_variables(
            self._scene, force_grad, dofs_idx, self.n_dofs, "dofs_idx", envs_idx, skip_allocation=True
        )
        if self.n_envs == 0:
            force_grad_ = force_grad_.unsqueeze(0)
        kernel_set_dofs_force_grad(dofs_idx, envs_idx, force_grad_, self.dyn_state, self.rigid_config)

    def clear_external_force(self) -> None:
        if gs.use_zerocopy:
            for tensor in (self.dyn_state.links.cfrc_applied_ang, self.dyn_state.links.cfrc_applied_vel):
                out = qd_to_torch(tensor, copy=False)
                out.zero_()
            if gs.backend == gs.metal:
                torch.mps.synchronize()
            return

        kernel_clear_external_force(self.dyn_state, self.rigid_info, self.rigid_config)

    def reset_grad(self) -> None:
        # Rigid additionally owns `geoms_state`, `entities_state`, and the `*_adjoint_cache` structs written by the
        # backward substep chain. All carry `needs_grad=True` fields that accumulate via `atomic_add` during backward,
        # so they must start at zero between consecutive `loss.backward()`s.
        self._reset_kinematic_grad()
        if self._requires_grad:
            qd_zero_grad(self.dyn_state.geoms)
            qd_zero_grad(self.dyn_state.entities)
            qd_zero_grad(self.dyn_state_adjoint_cache.dofs)
            qd_zero_grad(self.dyn_state_adjoint_cache.links)
            qd_zero_grad(self.dyn_state_adjoint_cache.joints)
            qd_zero_grad(self.dyn_state_adjoint_cache.geoms)
            qd_zero_grad(self._rigid_adjoint_cache)
            # One flush for the zeroing batch; see qd_zero_grad in misc.py.
            if gs.use_zerocopy and gs.backend == gs.metal:
                torch.mps.synchronize()

    def get_state(self, f: int | None = None) -> RigidSolverState:
        s_global = self.sim.cur_step_global
        if s_global in self._queried_states:
            return self._queried_states[s_global][0]

        state = RigidSolverState(self._scene, s_global)

        # A captured state must be self-consistent: links_pos / links_quat have to be the Cartesian pose implied by
        # the qpos captured alongside them, otherwise restoring it does not reproduce the configuration it was taken
        # from. The step loop leaves that pose one integration behind qpos whenever it defers the post-integrate
        # refresh, so catch up here - a no-op when the pose is already current, which is the common case.
        self.update_forward_pos()

        kernel_get_state(
            state.qpos,
            state.dofs_vel,
            state.dofs_acc,
            state.links_pos,
            state.links_quat,
            state.friction_ratio,
            self.dyn_state,
            self.rigid_info,
            self.rigid_config,
        )
        self._queried_states.append(state)
        return state

    def discard_queried_state(self, state: RigidSolverState) -> None:
        """Forget a state 'get_state' returned, once the scene state holding it is dropped."""
        self._queried_states.discard(state)

    def set_state(
        self,
        f: int,
        state: RigidSolverState,
        envs_idx: IndexLike = None,
        *,
        partial: bool = False,
    ) -> None:
        if (
            not self._requires_grad
            and gs.use_zerocopy
            and (not isinstance(envs_idx, torch.Tensor) or (not IS_OLD_TORCH or envs_idx.dtype == torch.bool))
        ):
            errno = qd_to_torch(self._errno, copy=False)
            qpos_dst = qd_to_torch(self.rigid_info.qpos, transpose=True, copy=False)
            vel_dst = qd_to_torch(self.dyn_state.dofs.vel, transpose=True, copy=False)
            acc_dst = qd_to_torch(self.dyn_state.dofs.acc, transpose=True, copy=False)
            ctrl_force_dst = qd_to_torch(self.dyn_state.dofs.ctrl_force, transpose=True, copy=False)
            ctrl_mode_dst = qd_to_torch(self.dyn_state.dofs.ctrl_mode, transpose=True, copy=False)
            pos_dst = qd_to_torch(self.dyn_state.links.pos, transpose=True, copy=False)
            quat_dst = qd_to_torch(self.dyn_state.links.quat, transpose=True, copy=False)
            cfrc_vel_dst = qd_to_torch(self.dyn_state.links.cfrc_applied_vel, transpose=True, copy=False)
            cfrc_ang_dst = qd_to_torch(self.dyn_state.links.cfrc_applied_ang, transpose=True, copy=False)
            fric_dst = qd_to_torch(self.dyn_state.geoms.friction_ratio, transpose=True, copy=False)
            # Setting the state is a discontinuity: wake every body in the affected envs (a body left hibernated would
            # stay frozen), restoring the flags and the awake-dof count alongside the other state buffers.
            if self._use_hibernation:
                links_hibernated_dst = qd_to_torch(self.dyn_state.links.is_hibernated, transpose=True, copy=False)
                awake_steps_dst = qd_to_torch(self.dyn_state.links.awake_steps, transpose=True, copy=False)
                dofs_hibernated_dst = qd_to_torch(self.dyn_state.dofs.is_hibernated, transpose=True, copy=False)
                geoms_hibernated_dst = qd_to_torch(self.dyn_state.geoms.is_hibernated, transpose=True, copy=False)
                entities_hibernated_dst = qd_to_torch(self.dyn_state.entities.is_hibernated, transpose=True, copy=False)
                islands_hibernated_dst = qd_to_torch(
                    self.rigid_island.data.island.is_hibernated, transpose=True, copy=False
                )
                islands_next_link_dst = qd_to_torch(
                    self.rigid_island.data.island.hibernated_next_link, transpose=True, copy=False
                )
                n_awake_dofs_dst = qd_to_torch(self.rigid_info.n_awake_dofs, copy=False)

            if envs_idx is not None and not isinstance(envs_idx, torch.Tensor):
                (envs_idx,) = indices_to_mask(envs_idx)
            if isinstance(envs_idx, torch.Tensor):
                if envs_idx.dtype == torch.bool:
                    envs_mask = envs_idx
                else:
                    envs_mask = torch.zeros(self._B, dtype=torch.bool, device=gs.device)
                    envs_mask[envs_idx] = True

                errno.masked_fill_(envs_mask, 0)
                if self.n_qs:
                    torch.where(envs_mask[:, None], state.qpos, qpos_dst, out=qpos_dst)
                    torch.where(envs_mask[:, None], state.dofs_vel, vel_dst, out=vel_dst)
                    torch.where(envs_mask[:, None], state.dofs_acc, acc_dst, out=acc_dst)
                    ctrl_force_dst.masked_fill_(envs_mask[:, None], 0.0)
                    ctrl_mode_dst.masked_fill_(envs_mask[:, None], gs.CTRL_MODE.FORCE)
                torch.where(envs_mask[:, None, None], state.links_pos, pos_dst, out=pos_dst)
                torch.where(envs_mask[:, None, None], state.links_quat, quat_dst, out=quat_dst)
                cfrc_vel_dst.masked_fill_(envs_mask[:, None, None], 0.0)
                cfrc_ang_dst.masked_fill_(envs_mask[:, None, None], 0.0)
                if self.n_geoms:
                    torch.where(envs_mask[:, None], state.friction_ratio, fric_dst, out=fric_dst)
                if self._use_hibernation:
                    links_hibernated_dst.masked_fill_(envs_mask[:, None], 0)
                    awake_steps_dst.masked_fill_(envs_mask[:, None], 0)
                    dofs_hibernated_dst.masked_fill_(envs_mask[:, None], 0)
                    geoms_hibernated_dst.masked_fill_(envs_mask[:, None], 0)
                    entities_hibernated_dst.masked_fill_(envs_mask[:, None], 0)
                    islands_hibernated_dst.masked_fill_(envs_mask[:, None], 0)
                    islands_next_link_dst.masked_fill_(envs_mask[:, None], -1)
                    n_awake_dofs_dst.masked_fill_(envs_mask, self.n_dofs)
            else:
                if self.n_qs:
                    errno[envs_idx] = 0
                    qpos_dst[envs_idx] = state.qpos[envs_idx]
                    vel_dst[envs_idx] = state.dofs_vel[envs_idx]
                    acc_dst[envs_idx] = state.dofs_acc[envs_idx]
                    ctrl_force_dst[envs_idx] = 0.0
                    ctrl_mode_dst[envs_idx] = gs.CTRL_MODE.FORCE
                pos_dst[envs_idx] = state.links_pos[envs_idx]
                quat_dst[envs_idx] = state.links_quat[envs_idx]
                cfrc_vel_dst[envs_idx] = 0.0
                cfrc_ang_dst[envs_idx] = 0.0
                if self.n_geoms:
                    fric_dst[envs_idx] = state.friction_ratio[envs_idx]
                if self._use_hibernation:
                    links_hibernated_dst[envs_idx] = 0
                    awake_steps_dst[envs_idx] = 0
                    dofs_hibernated_dst[envs_idx] = 0
                    geoms_hibernated_dst[envs_idx] = 0
                    entities_hibernated_dst[envs_idx] = 0
                    islands_hibernated_dst[envs_idx] = 0
                    islands_next_link_dst[envs_idx] = -1
                    n_awake_dofs_dst[envs_idx] = self.n_dofs
            if gs.backend == gs.metal:
                torch.mps.synchronize()
        else:
            envs_idx = self._scene._sanitize_envs_idx(envs_idx)
            kernel_set_zero(envs_idx, self._errno)
            kernel_set_state(
                envs_idx,
                state.qpos,
                state.dofs_vel,
                state.dofs_acc,
                state.links_pos,
                state.links_quat,
                state.friction_ratio,
                self.dyn_state,
                self.rigid_info,
                self.rigid_config,
            )
            if self._use_hibernation:
                kernel_reset_hibernation(
                    envs_idx,
                    self.dyn_state,
                    self.rigid_island.data.island,
                    self.dyn_info,
                    self.rigid_info,
                    self.rigid_config,
                )

        if not partial:
            if not isinstance(envs_idx, torch.Tensor):
                envs_idx = self._scene._sanitize_envs_idx(envs_idx)
            if envs_idx.dtype == torch.bool:
                fn = kernel_masked_forward_kinematics_links_geoms
            else:
                fn = kernel_forward_kinematics_links_geoms
            fn(envs_idx, self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config)
            self._is_forward_pos_updated = True
            self._is_forward_vel_updated = True
        else:
            self._is_forward_pos_updated = False
            self._is_forward_vel_updated = False

    def restart_owner(self) -> None:
        """Re-arm owner-local host state after a state discontinuity."""
        for entity in self.entities:
            if isinstance(entity, DroneEntity):
                entity._prev_prop_t = -1

    def process_input(self, in_backward: bool = False) -> None:
        for entity in self.entities:
            entity.process_input(in_backward=in_backward)

    def process_input_grad(self) -> None:
        for entity in self.entities:
            entity.process_input_grad()

    def save_ckpt(self, ckpt_name: str) -> None:
        # Save ckpt only if we need gradients, because this operation is costly
        if self._requires_grad:
            if ckpt_name not in self._ckpt:
                self._ckpt[ckpt_name] = dict()

            # copy=True required: with the zerocopy backend qd_to_numpy returns a view, so later substeps would
            # overwrite this ckpt's buffer in place.
            self._ckpt[ckpt_name]["qpos"] = qd_to_numpy(self._rigid_adjoint_cache.qpos, copy=True)
            self._ckpt[ckpt_name]["dofs_vel"] = qd_to_numpy(self._rigid_adjoint_cache.dofs_vel, copy=True)
            self._ckpt[ckpt_name]["dofs_acc"] = qd_to_numpy(self._rigid_adjoint_cache.dofs_acc, copy=True)

            for entity in self.entities:
                entity.save_ckpt(ckpt_name)

    def load_ckpt(self, ckpt_name: str) -> None:
        # Set first frame
        self.rigid_info.qpos.from_numpy(self._ckpt[ckpt_name]["qpos"][0])
        self.dyn_state.dofs.vel.from_numpy(self._ckpt[ckpt_name]["dofs_vel"][0])
        self.dyn_state.dofs.acc.from_numpy(self._ckpt[ckpt_name]["dofs_acc"][0])

        if not self._enable_mujoco_compatibility:
            # Mirror the post-integrate refresh of kernel_step_2: the replayed substeps skip their own cartesian /
            # velocity updates (is_forward_pos_updated / is_forward_vel_updated), so both must be recomputed here
            # from the restored qpos / vel. A stale link velocity would corrupt the velocity-product terms of every
            # backward primal in the window.
            kernel_update_cartesian_space(
                self.dyn_state,
                self.dyn_info,
                self.rigid_info,
                self.rigid_config,
                force_update_all_geoms=False,
                is_backward=False,
            )
            kernel_forward_velocity(
                self._scene._sanitize_envs_idx(None),
                self.dyn_state,
                self.dyn_info,
                self.rigid_info,
                self.rigid_config,
                is_backward=False,
            )

        for entity in self.entities:
            entity.load_ckpt(ckpt_name)

    def _restore_substep_inputs(self, f: int) -> None:
        """Restore the substep inputs and switch owner kernels to backward replay mode."""
        self._is_backward = True
        kernel_prepare_backward_substep(
            f,
            self.dyn_state,
            self.dyn_state_adjoint_cache,
            self._rigid_adjoint_cache,
            self.dyn_info,
            self.rigid_info,
            self.rigid_config,
        )

    def _reverse_integration(self, f: int) -> None:
        """Reverse integration after Engine has replayed the composed forward substep."""
        envs_idx = self._scene._sanitize_envs_idx(None)
        if not self._enable_mujoco_compatibility:
            # The FK backward below builds its Jacobian at the post-integrate qpos / vel, so copy the integrator's
            # _next outputs into the current slots first.
            kernel_copy_next_to_curr_no_check(self.dyn_state, self.rigid_info, self.rigid_config)
            self._update_cartesian_grad(envs_idx)

        is_grad_valid = kernel_begin_backward_substep(
            f,
            self.dyn_state,
            self.dyn_state_adjoint_cache,
            self._rigid_adjoint_cache,
            self.dyn_info,
            self.rigid_info,
            self.rigid_config,
        )
        if not is_grad_valid:
            gs.raise_exception(f"Nan grad in qpos or dofs_vel found at step {self._sim.cur_step_global}")

        kernel_step_2.grad(
            self.data,
            self.rigid_config,
            is_backward=True,
        )

    def _reverse_smooth_acceleration(self) -> None:
        """Reverse the unconstrained acceleration solve when no Constraint System exists."""
        kernel_manual_compute_qacc_bw(self.dyn_state, self.rigid_info, self.rigid_config)

    def _reverse_forward_dynamics(self, f: int) -> None:
        """Reverse forward dynamics and restore forward execution mode."""
        envs_idx = self._scene._sanitize_envs_idx(None)
        kernel_copy_acc(f, self.dyn_state, self._rigid_adjoint_cache, self.rigid_config)

        kernel_forward_dynamics_without_qacc.grad(
            self.dyn_state,
            self.rigid_island.data.island,
            self.dyn_info,
            self.rigid_info,
            self.rigid_config,
            is_backward=True,
        )

        # If it was the very first substep, we need to backpropagate through the initial update of the cartesian space
        if self._enable_mujoco_compatibility or self._sim.cur_substep_global == 0:
            self._update_cartesian_grad(envs_idx)

        # Change back to forward mode
        self._is_backward = False

    # -------------------------------- Implementation --------------------------------

    def compute_forward_dynamics(self) -> None:
        """Compute the unconstrained accelerations on the host, outside the substep pipeline."""
        kernel_forward_dynamics(
            self.dyn_state,
            self.rigid_island.data.island,
            self.dyn_info,
            self.rigid_info,
            self.rigid_config,
            False,
        )

    def update_acceleration(self) -> None:
        """Update the cartesian accelerations from the dof accelerations on the host, outside the substep pipeline."""
        kernel_update_acc(self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config)

    def _update_cartesian_grad(self, envs_idx: torch.Tensor) -> None:
        """Forward-replay the post-integrate cartesian-space update (FK -> COM -> geom poses -> velocity) under
        is_backward=True, then reverse it stage by stage.

        Velocity and forward kinematics are reversed manually (kernel_manual_*_bw in manual_bw.py), while COM and
        the link->geom transform are reversed by Quadrants autodiff (.grad). Shared by the post-integrate reverse
        and the first-substep initial-state reverse in substep_backward.
        """
        # Forward replay in dependency order (FK -> COM -> geoms -> velocity).
        kernel_forward_kinematics_replay(
            envs_idx, self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config, is_backward=True
        )
        kernel_COM_links_replay(self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config)
        kernel_update_geoms_replay(self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config, is_backward=True)
        kernel_forward_velocity(
            envs_idx, self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config, is_backward=True
        )

        # Reverse the stages: velocity first, forward kinematics last. COM and geoms both consume only FK
        # outputs, so their mutual order is free.
        kernel_manual_forward_velocity_bw(self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config)
        kernel_COM_links_replay.grad(self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config)
        kernel_update_geoms_replay.grad(
            self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config, is_backward=True
        )
        kernel_manual_forward_kinematics_bw(self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config)

    def collect_model(self) -> None:
        """Collect the authored Rigid entities from Scene and compile their host self."""
        self._entities = gs.List(entity for entity in self._scene.entities if isinstance(entity, RigidEntity))
        self.entities = self._entities
        self.links = gs.List(link for entity in self._entities for link in entity.links)
        self.joints = gs.List(joint for entity in self._entities for joint in entity.joints)
        self.vgeoms = gs.List(vgeom for entity in self._entities for vgeom in entity.vgeoms)
        self.geoms = gs.List(geom for entity in self._entities for geom in entity.geoms)
        self.equalities = gs.List(equality for entity in self._entities for equality in entity.equalities)
        self.n_entities = len(self._entities)
        self.n_links = len(self.links)
        self.n_joints = len(self.joints)
        self.n_vgeoms = len(self.vgeoms)
        self.n_vverts = sum(entity.n_vverts for entity in self._entities)
        self.n_vfaces = sum(entity.n_vfaces for entity in self._entities)
        self.n_custom_vverts = sum(entity.n_vverts for entity in self._entities if entity._morph.enable_custom_vverts)
        self.n_custom_vfaces = sum(entity.n_vfaces for entity in self._entities if entity._morph.enable_custom_vverts)
        self.n_qs = sum(entity.n_qs for entity in self._entities)
        self.n_dofs = sum(entity.n_dofs for entity in self._entities)
        self._enable_heterogeneous = any(entity.desc.variants for entity in self._entities)
        self.init_qpos = (
            np.concatenate([entity.init_qpos for entity in self._entities], dtype=gs.np_float)
            if self._entities
            else np.array([], dtype=gs.np_float)
        )
        self._build_kinematic_model()
        self.geoms = gs.List(geom for entity in self._entities for geom in entity.geoms)
        self.equalities = gs.List(equality for entity in self._entities for equality in entity.equalities)
        self.n_geoms = len(self.geoms)
        self.n_verts = sum(entity.n_verts for entity in self._entities)
        self.n_free_verts = sum(
            link.n_verts
            for entity in self._entities
            for link in entity.links
            if not link.is_fixed or entity._batch_fixed_verts
        )
        self.n_fixed_verts = sum(
            link.n_verts
            for entity in self._entities
            for link in entity.links
            if link.is_fixed and not entity._batch_fixed_verts
        )
        self.n_faces = sum(entity.n_faces for entity in self._entities)
        self.n_edges = sum(entity.n_edges for entity in self._entities)
        self.n_equalities = len(self.equalities)
        self._n_geoms = self.n_geoms
        self._n_verts = self.n_verts
        self._n_free_verts = self.n_free_verts
        self._n_fixed_verts = self.n_fixed_verts
        self._n_faces = self.n_faces
        self._n_edges = self.n_edges
        self._n_equalities = self.n_equalities
        self.n_geoms_ = max(1, self.n_geoms)
        self.n_verts_ = max(1, self.n_verts)
        self.n_faces_ = max(1, self.n_faces)
        self.n_edges_ = max(1, self.n_edges)
        self.n_free_verts_ = max(1, self.n_free_verts)
        self.n_fixed_verts_ = max(1, self.n_fixed_verts)
        self.n_candidate_equalities_ = max(1, self.n_equalities + self._options.max_dynamic_constraints)
        geoms_offset_pos = np.zeros((self.n_geoms, 3), dtype=gs.np_float)
        geoms_offset_quat = np.tile(gu.identity_quat(), (self.n_geoms, 1))
        for entity in self._entities:
            ranges = entity.base_link._variant_geom_ranges if entity._desc.variants else None
            _fill_base_link_geom_offsets(geoms_offset_pos, geoms_offset_quat, entity, entity.geoms, ranges)
        self.geoms_offset_pos = self.geoms_offset_quat = None
        if not (
            np.allclose(geoms_offset_pos, 0.0, atol=gs.EPS)
            and np.allclose(gu.quat_to_xyz(geoms_offset_quat), 0.0, atol=gs.EPS)
        ):
            self.geoms_offset_pos = torch.from_numpy(geoms_offset_pos).to(device=gs.device, dtype=gs.tc_float)
            self.geoms_offset_quat = torch.from_numpy(geoms_offset_quat).to(device=gs.device, dtype=gs.tc_float)
        self._enable_collision = self._options.enable_collision
        self._enable_multi_contact = self._options.enable_multi_contact
        self._enable_mujoco_compatibility = self._options.enable_mujoco_compatibility
        self._enable_joint_limit = self._options.enable_joint_limit
        self._enable_self_collision = self._options.enable_self_collision
        self._enable_neutral_collision = self._options.enable_neutral_collision
        self._enable_adjacent_collision = self._options.enable_adjacent_collision
        self._disable_constraint = self._options.disable_constraint
        self._max_collision_pairs = self._options.max_collision_pairs
        self.max_collision_pairs = self._max_collision_pairs
        self._integrator = self._options.integrator
        self._box_box_detection = self._options.box_box_detection
        self._use_hibernation = self._options.use_hibernation and not self._requires_grad
        if self._options.hibernation_thresh_vel is None:
            self._hibernation_thresh_vel = 1e-4 if self._enable_mujoco_compatibility else 2e-3
        else:
            self._hibernation_thresh_vel = self._options.hibernation_thresh_vel
        self._substep_dt = self.sim._substep_dt
        self.sol_min_timeconst = 2.0 * self._substep_dt
        self.sol_default_timeconst = (
            None
            if self._options.constraint_timeconst is None
            else max(self._options.constraint_timeconst, self.sol_min_timeconst)
        )
        if self._options.friction_cone == gs.friction_cone.elliptic and self._requires_grad:
            gs.raise_exception("The elliptic friction cone is not supported yet when 'requires_grad' is True.")
        blocker = ""
        if self._enable_mujoco_compatibility:
            blocker = "'enable_mujoco_compatibility' is True"
        elif self._options.friction_cone != gs.friction_cone.elliptic:
            blocker = "it requires 'friction_cone' to be 'gs.friction_cone.elliptic'"
        elif self._options.constraint_solver != gs.constraint_solver.Newton:
            blocker = "it requires 'constraint_solver' to be 'gs.constraint_solver.Newton'"
        if self._options.contact_resolution is None:
            self._options.contact_resolution = (
                gs.contact_resolution.convex if blocker else gs.contact_resolution.signorini
            )
        elif self._options.contact_resolution == gs.contact_resolution.signorini and blocker:
            gs.raise_exception(f"'contact_resolution' cannot be 'gs.contact_resolution.signorini' when {blocker}.")
        self._contact_resolution = self._options.contact_resolution
        if self._options.impratio is None:
            self._options.impratio = (
                100.0
                if self._options.friction_cone == gs.friction_cone.elliptic and not self._enable_mujoco_compatibility
                else 1.0
            )
        if self._options.tolerance is None:
            self._options.tolerance = 1e-5 if gs.qd_float == qd.f32 else 1e-8
            if not self._enable_mujoco_compatibility:
                self._options.tolerance *= 0.1
        self._is_forward_pos_updated = False
        self._is_forward_vel_updated = False
        self._is_backward = False
        self._build_static_config()

    def initialize_data(self) -> None:
        """Allocate and initialize the rigid data, which the solver forwards to from here on."""
        self._create_data()
        self._init_dof_fields(self.dyn_state, self.dyn_info, self.rigid_info)
        self._init_vvert_fields(self.dyn_info)
        self._init_vgeom_fields(self.dyn_info)
        self._init_link_fields(self.dyn_state, self.dyn_info, self.rigid_info)
        self.qpos = self.rigid_info.qpos
        self.qpos0 = self.rigid_info.qpos0
        self.links_T = self.rigid_info.links_T
        self._init_tree_fields(self.rigid_info)
        self.mass_mat = self.rigid_info.mass_mat
        self.mass_mat_L = self.rigid_info.mass_mat_L
        self.mass_mat_D_inv = self.rigid_info.mass_mat_D_inv
        self.meaninertia = self.rigid_info.meaninertia
        self._init_entity_fields(self.dyn_state, self.dyn_info, self.rigid_info)
        self._vfaces_raycast_mask = self._init_vfaces_raycast_mask(self.dyn_info)
        self._init_vverts_state(self.dyn_state, self.dyn_info)
        self._init_dynamics()

    def get_forward_update_flags(self) -> tuple[bool, bool]:
        return self._is_forward_pos_updated, self._is_forward_vel_updated

    def invalidate_forward_state(self) -> None:
        self._is_forward_pos_updated = False
        self._is_forward_vel_updated = False

    def _save_adjoint_cache(self, f: int) -> None:
        kernel_save_adjoint_cache(f, self.dyn_state, self._rigid_adjoint_cache, self.rigid_info, self.rigid_config)

    def get_forward_kernel_args(self) -> tuple[bool, bool, bool]:
        return self._is_forward_pos_updated, self._is_forward_vel_updated, self._is_backward

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad

    @property
    def is_backward(self) -> bool:
        return self._is_backward

    def update_velocity_for_coupling(self) -> None:
        update_qvel(self.dyn_state, self.rigid_info, self.rigid_config)

    def update_acceleration_after_coupling(self) -> None:
        update_qacc_from_qvel_delta(self.dyn_state, self.rigid_info, self.rigid_config)

    def _mark_forward_updated(self) -> None:
        """Record that the integration has updated the cartesian space and the velocities the next substep starts
        from, which the MuJoCo-compatible mode recomputes at the start of every substep instead."""
        self._is_forward_pos_updated = not self._enable_mujoco_compatibility
        self._is_forward_vel_updated = not self._enable_mujoco_compatibility

    # ------------------------------- Initialization ---------------------------------

    def _build_static_config(self) -> None:
        # The scene has multi-island block structure when it holds several independent DOF-carrying bodies or free
        # joints (the Hessian then splits into per-island blocks instead of one dense tree), where the sparse Jacobian
        # of the CPU solve pays off; a single dense-coupled tree (e.g. one big robot) gains nothing from it.
        n_dof_entities = sum(entity.n_dofs > 0 for entity in self.entities)
        n_free_joints = sum(joint.type == gs.JOINT_TYPE.FREE for joint in self.joints)
        has_multi_island_structure = n_dof_entities >= 2 or n_free_joints >= 2

        # sparse_solve=None resolves automatically: the sparse Jacobian makes the per-iteration Jacobian-vector
        # products, the constraint-to-island lookup and the per-island Hessian assembly cost O(nonzeros) instead of
        # O(n_constraints * n_dofs), which pays off on CPU when the scene has block structure, whereas a single
        # dense-coupled tree gains nothing. An explicit value overrides this on CPU. On GPU the dense tiled path is
        # faster, so the sparse representation is dropped there whatever the option says.
        if self._options.sparse_solve is None:
            sparse_solve = gs.backend == gs.cpu and not self._enable_mujoco_compatibility and has_multi_island_structure
        else:
            sparse_solve = self._options.sparse_solve
            if sparse_solve and gs.backend != gs.cpu:
                gs.logger.warning(
                    "'sparse_solve' is ignored on the GPU backend, where the dense tiled factorization is faster."
                )
        sparse_solve = sparse_solve and gs.backend == gs.cpu

        # Under the elliptic cone any middle-zone contact invalidates the Cholesky factor every Newton iteration, so
        # a contact-dense solve rebuilds it nearly every iteration; persisting the cone-free assembled Hessian turns
        # each rebuild into an envelope copy (see the nt_H declaration in array_class.py for the packed storage).
        # Confined to the CPU per-island skyline path that owns per-iteration rebuilds, the GPU arms rebuilding through
        # the tiled assembly.
        enable_cone_free_hessian_reuse = (
            self._options.friction_cone == gs.friction_cone.elliptic
            and gs.backend == gs.cpu
            and sparse_solve
            and self._options.constraint_solver == gs.constraint_solver.Newton
        )

        # The tiled per-island seed (see enable_tiled_island_seed in array_class.py) replaces a per-env thread walking
        # O(n^3) dependent loads, slower than the whole Newton iteration above the core count. The cooperative body
        # kernels run one warp per env, which pays off once an env carries enough dofs to keep its lanes busy, and stops
        # paying once the envs alone saturate the GPU. Both are dense-only. Outside performance mode the ndarray kernels
        # serve every scene shape, so the static config drops the scene bounds whose generic value costs little.
        is_generic = gs.use_ndarray
        enable_tiled_island_seed = (
            gs.backend != gs.cpu
            and not self.sim.options.requires_grad
            and not sparse_solve
            and (self.n_dofs >= 16 or (is_generic and self.n_dofs > 0))
        )
        enable_cooperative_constraint_kernels = enable_tiled_island_seed and (
            is_generic or self._sim._B <= get_gpu_core_count()
        )
        # The noslip sweep of an island is the last one-thread process of the cooperative regime, so its block-per-island
        # variant takes the same bound (see func_noslip in noslip.py).
        enable_cooperative_noslip = enable_cooperative_constraint_kernels and self._options.noslip_iterations > 0
        # Dofs per kinematic tree, counted from the links here since _init_tree_fields runs once the fields this config
        # sizes are allocated. A scene holding one tree forms at most one island per env, and the per-island passes
        # read the env's plain ranges.
        tree_links = np.flatnonzero(self._links_tree_root_idx >= 0)
        trees_n_dofs = np.bincount(
            self._links_tree_root_idx[tree_links], np.array([link.n_dofs for link in self.links])[tree_links]
        )
        is_single_island = self._n_trees == 1 and not is_generic
        # Above the cooperative bound one thread per env saturates the GPU, where the scalar dense Cholesky of an env's
        # one block beats the tiled factor, so the seed kernel assembles the block and the monolith factors it.
        has_scalar_seed_factor = (
            enable_tiled_island_seed and is_single_island and not enable_cooperative_constraint_kernels
        )
        constraint_layout_batch_first = (
            enable_cooperative_constraint_kernels or self.sim._para_level < gs.PARA_LEVEL.ALL
        )

        # The level sweep (see func_sweep_links_by_level) syncs the lanes of a block, which neither the CPU nor the
        # differentiable runs offer. The serial walks run one thread per tree or root and env through all the links of
        # its group, so they take as long as the longest group. The lanes of an env step through a level in chunks of
        # their count, so the sweep stays within that time while its lane steps over all the levels and envs fit in the
        # cores times the longest group. Past that bound it only adds instructions to a busy GPU, whether the links are
        # grouped by tree or by root.
        enable_level_sweep = gs.backend != gs.cpu and not self.sim.options.requires_grad
        if enable_level_sweep and not is_generic:
            n_lanes_per_env = array_class.RigidSimStaticConfig.level_sweep_n_lanes_per_env
            for links_group_idx, links_level in (
                (self._links_tree_root_idx, self._links_tree_level),
                (self._links_root_idx, self._links_root_level),
            ):
                group_links = np.flatnonzero(links_level >= 0)
                if group_links.size:
                    _, groups_n_links = np.unique(links_group_idx[group_links], return_counts=True)
                    _, levels_n_links = np.unique(
                        links_group_idx[group_links] * self.n_links + links_level[group_links], return_counts=True
                    )
                    n_lane_steps = n_lanes_per_env * (-(-levels_n_links // n_lanes_per_env)).sum() * self._sim._B
                    enable_level_sweep &= bool(n_lane_steps <= groups_n_links.max() * get_gpu_core_count())

        rigid_config = dict(
            backend=gs.backend,
            para_level=self.sim._para_level,
            requires_grad=self.sim.options.requires_grad,
            use_hibernation=self._use_hibernation,
            batch_links_info=self._options.batch_links_info,
            batch_dofs_info=self._options.batch_dofs_info,
            batch_joints_info=self._options.batch_joints_info,
            enable_mujoco_compatibility=self._enable_mujoco_compatibility,
            enable_elliptic_friction=self._options.friction_cone == gs.friction_cone.elliptic,
            enable_signorini_contact=self._contact_resolution == gs.contact_resolution.signorini,
            enable_torsional_friction=self._options.enable_torsional_friction,
            enable_rolling_friction=self._options.enable_rolling_friction,
            enable_multi_contact=self._enable_multi_contact,
            enable_collision=self._enable_collision,
            enable_joint_limit=self._enable_joint_limit,
            box_box_detection=self._box_box_detection,
            sparse_solve=sparse_solve,
            enable_cone_free_hessian_reuse=enable_cone_free_hessian_reuse,
            integrator=self._integrator,
            solver_type=self._options.constraint_solver,
            broadphase_traversal=resolve_broadphase_traversal(
                self._options,
                self._enable_heterogeneous,
            ),
            # Parallelize init over (constraints, envs) when envs alone don't saturate the GPU.
            parallel_init=(
                gs.backend != gs.cpu
                and not self.sim.options.requires_grad
                and (is_generic or self.n_envs <= get_gpu_core_count())
            ),
            enable_level_sweep=enable_level_sweep,
            enable_tiled_island_seed=enable_tiled_island_seed,
            enable_cooperative_constraint_kernels=enable_cooperative_constraint_kernels,
            enable_cooperative_noslip=enable_cooperative_noslip,
            is_single_island=is_single_island,
            has_scalar_seed_factor=has_scalar_seed_factor,
            constraint_layout_batch_first=constraint_layout_batch_first,
        )

        # Prefer the monolith solver on CPU (always faster there, perf dispatch is a waste of effort)
        if gs.backend == gs.cpu or self.sim.options.requires_grad:
            rigid_config["prefer_decomposed_solver"] = 0

        # Per-DOF mass-block bounds (see dofs_mass_block_start in array_class.py), computed here because the tiled
        # factor arms below are sized for the largest block; _init_tree_fields uploads them.
        links_by_idx = {link.idx: link for link in self.links}
        dofs_mass_block_start = np.arange(self.n_dofs_, dtype=gs.np_int)
        for link in self.links:
            if link.n_dofs == 0:
                continue
            root_link = link
            node = link
            while node.parent_idx != -1:
                node = links_by_idx[node.parent_idx]
                if node.n_dofs > 0:
                    root_link = node
            dofs_mass_block_start[link.dof_start : link.dof_end] = root_link.dof_start
        dofs_mass_block_end = np.empty(self.n_dofs_, dtype=gs.np_int)
        for i_d in range(self.n_dofs_):
            dofs_mass_block_end[dofs_mass_block_start[i_d]] = i_d + 1
        dofs_mass_block_end = dofs_mass_block_end[dofs_mass_block_start]

        # Blocks form clean intervals: attach() is the only source of cross-entity blocks and rejects any layout that
        # would interleave foreign DOFs inside a merged one (see RigidEntity.attach).
        self._dofs_mass_block_start = dofs_mass_block_start
        self._dofs_mass_block_end = dofs_mass_block_end

        if self.n_entities:
            # The tiled and cooperative Cholesky kernels trade per-env serial work for cross-lane parallelism, so they
            # only help while envs alone do not already saturate the GPU. Above that env count one-thread-per-env keeps
            # every core busy and the scalar path wins; below it the parallel kernels hide latency by swapping warps.
            # The crossover is also hardware- and kernel-dependent, so the env threshold (GPU core count) is a heuristic
            # and a dynamic timer-based selection would be more accurate still.
            # Largest mass block in DOFs, the unit the per-block factor is sized for: merged entities can make a block
            # exceed any single entity (see attach).
            max_block_dofs = int((dofs_mass_block_end - dofs_mass_block_start).max()) if self.n_dofs else 0
            if gs.backend != gs.cpu:
                max_tiled_envs = get_gpu_core_count()
                envs_undersaturate = self.n_envs <= max_tiled_envs

                # Register tile of the mass-matrix Cholesky kernels and unit of the island tile caps, see
                # cholesky_tile_size_for in array_class.py.
                cholesky_tile_size = 32 if is_generic else array_class.cholesky_tile_size_for(self.n_dofs)

                # A shared tile holds the dofs it factors rounded up to the next size of get_gpu_shared_tile_sizes,
                # which bounds the sizes compiled while keeping the occupancy of the tightest fit. A mass block above
                # the largest size factors outside the shared tile, which leaves the tile size unused.
                shared_tile_sizes = get_gpu_shared_tile_sizes(2 if is_generic else 8)
                mass_matrix_fits_shared = max_block_dofs <= shared_tile_sizes[-1]
                tiled_n_dofs_per_block = min(
                    (size for size in shared_tile_sizes if size >= max_block_dofs), default=shared_tile_sizes[-1]
                )

                # The cooperative per-island solve stages one island's tile in shared memory, in size classes (see
                # island_tile_cap_first in array_class.py): the last cap is the shared tile of the dof count, or the
                # largest one where the dofs exceed it. An island holds at least one tree, so a class below the
                # smallest tree never holds an island: the first cap starts at that tree.
                island_tile_cap_last = min(
                    (size for size in shared_tile_sizes if size >= self.n_dofs), default=shared_tile_sizes[-1]
                )
                min_tree_dofs = trees_n_dofs[trees_n_dofs > 0].min() if self.n_dofs else 0
                island_tile_cap_first = cholesky_tile_size
                while island_tile_cap_first < min(min_tree_dofs, island_tile_cap_last):
                    island_tile_cap_first *= 2
                island_tile_cap_first = min(island_tile_cap_first, island_tile_cap_last)

                # The cooperative in-place LDL^T has no cap; the shared-memory tile is faster but capped. Same env logic
                # as the Hessian: tile from the largest block >= 8 DOFs, drop the env guard above the cap where the
                # scalar O(n_block_dofs^3) per-(block, env) factor is always worse. Outside performance mode every scene
                # holding dofs takes the tiled factor, whatever its block size and env count.
                enable_tiled_cholesky_mass_matrix = (
                    max_block_dofs > 0
                    if is_generic
                    else max_block_dofs >= 8 and (not mass_matrix_fits_shared or envs_undersaturate)
                )

                # Register-streaming tiled mass factor for the >shared-cap forward GPU path: factors each mass
                # block in registers via the same primitive as the Hessian, faster than and numerically matching the
                # cooperative LDL^T. Reuses cholesky_tile_size (always 32 here).
                enable_register_tiled_mass = (
                    enable_tiled_cholesky_mass_matrix and not mass_matrix_fits_shared and not self._requires_grad
                )

                # The shared-memory mass factor solves the smooth acceleration while it still holds the factor. A
                # differentiable scene keeps the separate solve, which kernel_manual_compute_qacc_bw reverses.
                enable_fused_smooth_acc_solve = (
                    enable_tiled_cholesky_mass_matrix and mass_matrix_fits_shared and not self._requires_grad
                )

                rigid_config.update(
                    enable_tiled_cholesky_mass_matrix=enable_tiled_cholesky_mass_matrix,
                    mass_matrix_fits_shared=mass_matrix_fits_shared,
                    enable_register_tiled_mass=enable_register_tiled_mass,
                    enable_fused_smooth_acc_solve=enable_fused_smooth_acc_solve,
                    cholesky_tile_size=cholesky_tile_size,
                    tiled_n_dofs_per_block=tiled_n_dofs_per_block,
                    island_tile_cap_first=island_tile_cap_first,
                    island_tile_cap_last=island_tile_cap_last,
                    # An island holds at most every dof of the scene, so below this bound the factor paths above the
                    # last cap (see func_island_assemble_factor_solve_tiled) are dead code and stay uncompiled.
                    has_island_above_tile_cap=self.n_dofs > island_tile_cap_last,
                )

                # One thread block assembles and factors an island above the last cap for the Newton solver (see
                # func_island_assemble_factor_solve_tiled), at a cost growing with the cube of its dof count whatever
                # the environment count, where the skyline factor of the CPU backend exploits the sparsity of the
                # contact graph.
                if (
                    self._options.constraint_solver == gs.constraint_solver.Newton
                    and self.n_dofs > 4 * island_tile_cap_last
                ):
                    gs.logger.warning(
                        f"The scene holds {self.n_dofs} dofs. A contact island wider than {4 * island_tile_cap_last} "
                        "dofs takes seconds per step with the Newton constraint solver on the GPU backend. If the "
                        "bodies can pile up into one island, use the CPU backend or the CG constraint solver."
                    )

                # Manually pin the solve arm only where the winner is determinable in advance AND confirmed across
                # CUDA + Metal; genuinely backend-dependent cases fall through to the per-step autotuner.
                if not enable_cooperative_constraint_kernels:
                    # No cooperative layout (n_envs > 8192 already saturates the GPU, or n_dofs < 16): the decomposed
                    # arm has nothing to exploit, so the scalar one-thread-per-env monolith is the clear winner.
                    rigid_config["prefer_decomposed_solver"] = 0

                # The autotuner picks between two numerically distinct arms by timing them as the simulation runs, so
                # which one runs at a given step follows the machine rather than the scene. Pinning the arm is what
                # makes a trajectory reproducible; the decomposed one takes whatever the static scene description
                # leaves open, every case that description settles in the monolith's favor being pinned above already.
                if gs.use_deterministic_algorithms and rigid_config.get("prefer_decomposed_solver", -1) == -1:
                    rigid_config["prefer_decomposed_solver"] = 1

        # Jacobi equilibration of the Newton system (see nt_jacobi in array_class.py): every factor, incremental
        # update and solve of every arm rides the scaled coordinates; the reference behaviour keeps the raw factor.
        # Enabled when the model's mass-diagonal spread bound (_jacobi_mass_spread_bound) exceeds what the working
        # precision's Cholesky factor conditions accurately; the double-precision threshold carries the mantissa
        # headroom over single, past any scene built from physical units.
        mass_spread_threshold = 1e4 if gs.qd_float == qd.f32 else 5e12
        rigid_config["enable_jacobi_equilibration"] = (
            not self._enable_mujoco_compatibility
            and rigid_config["solver_type"] == gs.constraint_solver.Newton
            and jacobi_mass_spread_bound(self._entities) > mass_spread_threshold
        )
        self.rigid_config = array_class.RigidSimStaticConfig(**rigid_config)

        if self.rigid_config.requires_grad:
            if self.rigid_config.use_hibernation:
                gs.raise_exception("Hibernation is not supported yet when requires_grad is True")
            if self.rigid_config.integrator != gs.integrator.approximate_implicitfast:
                gs.raise_exception(
                    "Only approximate_implicitfast integrator is supported yet when requires_grad is True."
                )
            if self._options.noslip_iterations > 0:
                gs.raise_exception("Noslip is not supported yet when requires_grad is True.")
            if self._options.enable_torsional_friction or self._options.enable_rolling_friction:
                gs.raise_exception("Torsional and rolling friction are not supported yet when requires_grad is True.")

    def _create_data(self) -> None:
        """Allocate the Rigid-owned data directly from explicit dimensions and configuration."""
        rigid_info = array_class.get_rigid_info(
            batch_size=self._B,
            requires_grad=self._requires_grad,
            n_dofs=self.n_dofs_,
            rigid_config=self.rigid_config,
            n_trees=self.n_trees_,
            n_roots=self.n_roots_,
            n_links=self.n_links_,
            substep_dt=self._substep_dt,
            n_equalities=self._n_equalities,
            n_candidate_equalities=self.n_candidate_equalities_,
            hibernation_thresh_vel=self._hibernation_thresh_vel,
            links_tree_root_idx=self._links_tree_root_idx,
            links_parent_idx=self._links_parent_idx,
            n_qs=self.n_qs_,
            n_geoms=self.n_geoms_,
            options=self._options,
            kinematic_only=False,
        )
        dofs_info = array_class.get_dofs_info(
            options=self._options,
            n_dofs=self.n_dofs_,
            batch_size=self._B,
        )
        dofs_state = array_class.get_dofs_state(
            requires_grad=self._requires_grad,
            n_dofs=self.n_dofs_,
            batch_size=self._B,
        )
        links_info = array_class.get_links_info(
            n_links=self.n_links_,
            options=self._options,
            batch_size=self._B,
        )
        links_state = array_class.get_links_state(
            requires_grad=self._requires_grad,
            n_links=self.n_links_,
            batch_size=self._B,
            n_links_actual=self.n_links,
            links=self.links,
        )
        joints_info = array_class.get_joints_info(
            options=self._options,
            n_joints=self.n_joints_,
            batch_size=self._B,
        )
        joints_state = array_class.get_joints_state(
            requires_grad=self._requires_grad,
            n_joints=self.n_joints_,
            batch_size=self._B,
        )
        entities_info = array_class.get_entities_info(n_entities=self.n_entities_)
        entities_state = array_class.get_entities_state(
            n_entities=self.n_entities_,
            batch_size=self._B,
        )
        vverts_info = array_class.get_vverts_info(n_vverts=self.n_vverts_)
        vverts_state = array_class.get_vverts_state(
            n_custom_vverts=self.n_custom_vverts_,
            batch_size=self._B,
        )
        vfaces_info = array_class.get_vfaces_info(n_vfaces=self.n_vfaces_)
        vgeoms_info = array_class.get_vgeoms_info(n_vgeoms=self.n_vgeoms_)
        vgeoms_state = array_class.get_vgeoms_state(
            n_vgeoms=self.n_vgeoms_,
            batch_size=self._B,
        )
        geoms_info = array_class.get_geoms_info(n_geoms=self.n_geoms_)
        geoms_state = array_class.get_geoms_state(
            rigid_config=self.rigid_config,
            n_geoms=self.n_geoms_,
            batch_size=self._B,
        )
        verts_info = array_class.get_verts_info(n_verts=self.n_verts_)
        faces_info = array_class.get_faces_info(n_faces=self.n_faces_)
        edges_info = array_class.get_edges_info(n_edges=self.n_edges_)
        free_verts_state = array_class.get_free_verts_state(
            n_free_verts=self.n_free_verts_,
            batch_size=self._B,
        )
        fixed_verts_state = array_class.get_fixed_verts_state(n_fixed_verts=self.n_fixed_verts_)
        equalities_info = array_class.get_equalities_info(
            n_candidate_equalities=self.n_candidate_equalities_,
            batch_size=self._B,
        )
        dyn_info = array_class.DynInfo(
            entities=entities_info,
            links=links_info,
            joints=joints_info,
            dofs=dofs_info,
            geoms=geoms_info,
            verts=verts_info,
            faces=faces_info,
            edges=edges_info,
            vverts=vverts_info,
            vfaces=vfaces_info,
            vgeoms=vgeoms_info,
            equalities=equalities_info,
        )
        dyn_state = array_class.DynState(
            entities=entities_state,
            links=links_state,
            joints=joints_state,
            dofs=dofs_state,
            geoms=geoms_state,
            free_verts=free_verts_state,
            fixed_verts=fixed_verts_state,
            vverts=vverts_state,
            vgeoms=vgeoms_state,
        )
        weight_scratch = array_class.get_weight_scratch(
            n_dofs=self.n_dofs_,
            batch_size=self._B,
        )
        dyn_state_adjoint_cache = None
        if self._requires_grad:
            dyn_state_adjoint_cache = array_class.get_dyn_state_adjoint_cache(
                batch_size=self._B,
                requires_grad=self._requires_grad,
                rigid_config=self.rigid_config,
                links=self.links,
                n_entities=self.n_entities_,
                n_links=self.n_links_,
                n_links_actual=self.n_links,
                n_joints=self.n_joints_,
                n_dofs=self.n_dofs_,
                n_geoms=self.n_geoms_,
                n_free_verts=self.n_free_verts_,
                n_fixed_verts=self.n_fixed_verts_,
                n_custom_vverts=self.n_custom_vverts_,
                n_vgeoms=self.n_vgeoms_,
            )
        rigid_adjoint_cache = array_class.get_rigid_adjoint_cache(
            requires_grad=self._requires_grad,
            substeps_local=self.sim.substeps_local,
            n_qs=self.n_qs_,
            batch_size=self._B,
            n_dofs=self.n_dofs_,
        )
        errno = array_class.V(dtype=gs.qd_int, shape=(self._B,))
        self.data.geoms_init_AABB = rigid_info.geoms_init_AABB
        self.data.dyn_state = dyn_state
        self.data.dyn_info = dyn_info
        self.data.rigid_info = rigid_info
        self.data.weight_scratch = weight_scratch
        self.data.rigid_adjoint_cache = rigid_adjoint_cache
        self.data.dyn_state_adjoint_cache = dyn_state_adjoint_cache
        self.data.errno = errno

        self._errno = self.data.errno
        self.rigid_info = self.data.rigid_info
        self._rigid_adjoint_cache = self.data.rigid_adjoint_cache
        self.dyn_info = self.data.dyn_info
        self.dyn_state = self.data.dyn_state
        if self._use_hibernation:
            self.n_awake_dofs = self.rigid_info.n_awake_dofs
        if self._requires_grad:
            self.dyn_state_adjoint_cache = self.data.dyn_state_adjoint_cache

        self._gravity = self.rigid_info.gravity
        self._gravity.from_numpy(np.tile(np.array(self._options.gravity, dtype=gs.np_float), (self._B, 1)))

    def _init_tree_fields(self, rigid_info: array_class.RigidInfo) -> None:
        """Initialize the fields describing the kinematic trees, the structure of the joint-space mass matrix
        included."""
        self._init_kinematic_tree_fields(rigid_info)

        # tree structure information
        mass_parent_mask = np.zeros((self.n_dofs_, self.n_dofs_), dtype=gs.np_float)
        for i_l in range(self.n_links):
            j_l = i_l
            while j_l != -1:
                for i_d, j_d in qd.ndrange(
                    (self.links[i_l].dof_start, self.links[i_l].dof_end),
                    (self.links[j_l].dof_start, self.links[j_l].dof_end),
                ):
                    mass_parent_mask[i_d, j_d] = 1.0
                j_l = self.links[j_l].parent_idx

        # Per-DOF mass-block bounds, computed by _build_static_config (which sizes the tiled factor arms from them).
        dofs_mass_block_start = self._dofs_mass_block_start.copy()
        dofs_mass_block_end = self._dofs_mass_block_end.copy()

        # An aligned free body whose only DOFs are its own free joint has a diagonal joint-space mass block, so zero its
        # within-link off-diagonal mask to make the assembled mass exactly diagonal (else ~1e-6 round-off once it
        # rotates) and the skyline envelope tighter. A DOF-bearing (articulated) descendant adds off-diagonal base
        # coupling, so the block must be exactly the root's own DOFs for the diagonalization to be valid.
        # Writing a center of mass or an inertia on such a link is rejected, so the block stays valid for good (see
        # _set_links_info).
        for link in self.links:
            # 'aligned' already implies a free joint; the block bounds must additionally be exactly the link's own DOFs
            # (no DOF-bearing ancestor or descendant), otherwise the coupled block is not diagonal.
            if (
                not link.aligned
                or dofs_mass_block_start[link.dof_start] != link.dof_start
                or dofs_mass_block_end[link.dof_start] != link.dof_end
            ):
                continue
            for i_d in range(link.dof_start, link.dof_end):
                for j_d in range(link.dof_start, link.dof_end):
                    if i_d != j_d:
                        mass_parent_mask[i_d, j_d] = 0.0
                dofs_mass_block_start[i_d] = i_d
                dofs_mass_block_end[i_d] = i_d + 1

        # Smallest dof structurally coupled to each dof by the mass matrix (itself when none lies below it), read by the
        # skyline envelope of the per-island solver (see dof_env_start_local in array_class.py).
        dofs_mass_envelope_start = ((mass_parent_mask + mass_parent_mask.T) > 0.5).argmax(axis=1).astype(gs.np_int)
        rigid_info.mass_parent_mask.from_numpy(mass_parent_mask)
        rigid_info.dofs_mass_envelope_start.from_numpy(dofs_mass_envelope_start)
        rigid_info.dofs_mass_block_start.from_numpy(dofs_mass_block_start)
        rigid_info.dofs_mass_block_end.from_numpy(dofs_mass_block_end)

    def _dispatch_heterogeneous_vgeoms(self, dyn_info: array_class.DynInfo) -> None:
        """
        Dispatch per-environment geom/vgeom ranges and inertial properties for heterogeneous links.

        Extends the base class (which handles vgeom-only dispatch) to also dispatch collision geom
        ranges and per-variant inertial properties. Per-variant inertial is pre-computed during
        link._build() from actual geom objects, using analytic formulas for primitives.
        """
        for link in self.links:
            if link._variant_vgeom_ranges is None:
                continue

            n_variants = len(link._variant_vgeom_ranges)
            variant_idx = _balanced_variant_mapping(n_variants, self._B)

            # Build per-env arrays from link's variant data
            geom_starts = np.array([link._variant_geom_ranges[v][0] for v in variant_idx], dtype=gs.np_int)
            geom_ends = np.array([link._variant_geom_ranges[v][1] for v in variant_idx], dtype=gs.np_int)
            vgeom_starts = np.array([link._variant_vgeom_ranges[v][0] for v in variant_idx], dtype=gs.np_int)
            vgeom_ends = np.array([link._variant_vgeom_ranges[v][1] for v in variant_idx], dtype=gs.np_int)

            # Build per-env inertial arrays from pre-computed per-variant inertial
            links_inertial_mass = np.array([link._variant_inertial[v][0] for v in variant_idx], dtype=gs.np_float)
            links_inertial_pos = np.array([link._variant_inertial[v][1] for v in variant_idx], dtype=gs.np_float)
            links_inertial_quat = np.array([link._variant_inertial[v][2] for v in variant_idx], dtype=gs.np_float)
            links_inertial_i = np.array([link._variant_inertial[v][3] for v in variant_idx], dtype=gs.np_float)

            # Update links_info with per-environment values
            # Note: when batch_links_info is True, the shape is (n_links, B)
            kernel_update_heterogeneous_link_info(
                link.idx,
                geom_starts,
                geom_ends,
                vgeom_starts,
                vgeom_ends,
                links_inertial_mass,
                links_inertial_pos,
                links_inertial_quat,
                links_inertial_i,
                dyn_info,
            )

            # Set active_envs on geoms — indicates which environments each geom is active in
            for geom in link.geoms:
                active_envs_mask = (geom_starts <= geom.idx) & (geom.idx < geom_ends)
                geom.active_envs_mask = torch.tensor(active_envs_mask, device=gs.device)
                (geom.active_envs_idx,) = np.where(active_envs_mask)

            # Set active_envs on vgeoms
            for vgeom in link.vgeoms:
                active_envs_mask = (vgeom_starts <= vgeom.idx) & (vgeom.idx < vgeom_ends)
                vgeom.active_envs_mask = torch.tensor(active_envs_mask, device=gs.device)
                (vgeom.active_envs_idx,) = np.where(active_envs_mask)

    def _init_link_fields(
        self,
        dyn_state: array_class.DynState,
        dyn_info: array_class.DynInfo,
        rigid_info: array_class.RigidInfo,
    ) -> None:
        # The base initialization ends by dispatching each heterogeneous variant's inertial per environment, which
        # must write last, so this runs first.
        if self.links:
            links = self.links
            kernel_init_link_dynamics(
                np.array([link.desc.invweight for link in links], dtype=gs.np_float),
                np.array([link.desc.inertial_pos for link in links], dtype=gs.np_float),
                np.array([link.desc.inertial_quat for link in links], dtype=gs.np_float),
                np.array([link.desc.inertia for link in links], dtype=gs.np_float),
                np.array([link.desc.mass for link in links], dtype=gs.np_float),
                dyn_info,
            )

        self._init_kinematic_link_fields(
            dyn_state,
            dyn_info,
            rigid_info,
            self._dispatch_heterogeneous_vgeoms,
            sanitize_sol_params,
        )

    def _init_vert_fields(self, dyn_info: array_class.DynInfo) -> None:
        if self.n_verts > 0:
            geoms = self.geoms
            kernel_init_vert_fields(
                np.concatenate([np.full(geom.n_verts, geom.idx) for geom in geoms], dtype=gs.np_int),
                np.concatenate(
                    [np.arange(geom.verts_state_start, geom.verts_state_start + geom.n_verts) for geom in geoms],
                    dtype=gs.np_int,
                ),
                np.concatenate([geom.init_verts for geom in geoms], dtype=gs.np_float),
                np.concatenate([geom.init_faces + geom.vert_start for geom in geoms], dtype=gs.np_int),
                np.concatenate([geom.init_edges + geom.vert_start for geom in geoms], dtype=gs.np_int),
                np.concatenate([geom.init_normals for geom in geoms], dtype=gs.np_float),
                np.concatenate([geom.init_center_pos for geom in geoms], dtype=gs.np_float),
                np.concatenate(
                    [np.full(geom.n_verts, geom.is_fixed and not geom.entity._batch_fixed_verts) for geom in geoms],
                    dtype=gs.np_bool,
                ),
                dyn_info,
                self.rigid_config,
            )

    def _init_dof_length(self, dyn_info: array_class.DynInfo) -> None:
        # Characteristic length of each dof (1 for translation, the body radius for rotation), used to weight dof
        # velocities in the hibernation rest test. Computed here, after geom dispatch, because a heterogeneous entity's
        # per-variant geoms (and hence body radius) are only assigned to environments at that point. Only needed when
        # hibernation is on.
        if not self._use_hibernation:
            return

        joints = self.joints
        if sum(joint.n_dofs for joint in joints) == 0:
            return

        # dofs_length is per-env only for a heterogeneous entity; broadcast the shared row across envs when dofs_info
        # is batched (always so for a heterogeneous entity, optionally for a homogeneous one).
        dof_length = np.concatenate([joint.dofs_length for joint in joints], axis=0)
        if self._options.batch_dofs_info and dof_length.ndim == 1:
            dof_length = np.broadcast_to(dof_length[:, None], (len(dof_length), self._B))
        dyn_info.dofs.dof_length.from_numpy(dof_length)

    def _init_geom_fields(
        self,
        dyn_state: array_class.DynState,
        dyn_info: array_class.DynInfo,
        rigid_info: array_class.RigidInfo,
    ) -> None:
        geoms_init_AABB = rigid_info.geoms_init_AABB
        if self.n_geoms > 0:
            geoms = self.geoms
            geoms_sol_params = np.array([geom.desc.sol_params for geom in geoms], dtype=gs.np_float)
            # A geom keeps the time constant the model states; the floor lands on the value a contact mixes out of its
            # two geoms (see func_set_contact_data), the value the solver actually consumes. Flooring per geom would
            # raise the mix of a pair that is already above the floor, distorting the stated stiffness with no
            # stability benefit.
            sanitize_sol_params(
                geoms_sol_params,
                self.sol_min_timeconst,
                self.sol_default_timeconst,
                floor_timeconst=False,
            )

            # Accurately compute the center of mass of each geometry if possible.
            # Note that the mean vertex position is a bad approximation, which is impeding the ability of MPR to
            # estimate the exact contact information.
            geoms_center = []
            for geom in geoms:
                tmesh = geom.mesh.trimesh
                if tmesh.is_watertight:
                    geoms_center.append(tmesh.center_mass)
                else:
                    # Still fallback to mean vertex position if no better option...
                    geoms_center.append(np.mean(tmesh.vertices, axis=0))

            # A geom is hollow when its own center lies in a cavity rather than inside its material (bowl, mug,
            # nut), i.e. its own SDF is positive at its center. This is a static property of the collision
            # geometry, precomputed here so the narrowphase never has to probe it at runtime. SPHERE/PLANE/TERRAIN
            # SDFs are analytic and convex geoms enclose their own center, so neither is ever hollow and probing
            # them would only force their SDF grid to be built.
            geoms_is_hollow = []
            for geom, center in zip(geoms, geoms_center):
                is_hollow = False
                if (
                    geom.type not in (gs.GEOM_TYPE.SPHERE, gs.GEOM_TYPE.PLANE, gs.GEOM_TYPE.TERRAIN)
                    and not geom.is_convex
                ):
                    sdf_grid = geom.desc.mesh.get_sdf_grid(geom.desc.sdf)
                    grid_pos = sdf_grid.T_mesh_to_sdf[:3, :3] @ center + sdf_grid.T_mesh_to_sdf[:3, 3]
                    cell = np.minimum(
                        np.maximum(np.floor(grid_pos).astype(gs.np_int), 0), np.subtract(geom.desc.sdf.n_nodes, 2)
                    )
                    frac = grid_pos - cell
                    corners = sdf_grid.value[cell[0] : cell[0] + 2, cell[1] : cell[1] + 2, cell[2] : cell[2] + 2]
                    weights_x, weights_y, weights_z = ([1.0 - frac[i], frac[i]] for i in range(3))
                    sd_center = np.einsum("i,j,k,ijk->", weights_x, weights_y, weights_z, corners)
                    is_hollow = sd_center > gs.EPS
                geoms_is_hollow.append(is_hollow)

            kernel_init_geom_fields(
                np.array([geom.link.idx for geom in geoms], dtype=gs.np_int),
                np.array([geom.vert_start for geom in geoms], dtype=gs.np_int),
                np.array([geom.face_start for geom in geoms], dtype=gs.np_int),
                np.array([geom.edge_start for geom in geoms], dtype=gs.np_int),
                np.array([geom.verts_state_start for geom in geoms], dtype=gs.np_int),
                np.array([geom.vert_end for geom in geoms], dtype=gs.np_int),
                np.array([geom.face_end for geom in geoms], dtype=gs.np_int),
                np.array([geom.edge_end for geom in geoms], dtype=gs.np_int),
                np.array([geom.verts_state_end for geom in geoms], dtype=gs.np_int),
                np.array([geom.init_pos for geom in geoms], dtype=gs.np_float),
                np.array(geoms_center, dtype=gs.np_float),
                np.array([geom.init_quat for geom in geoms], dtype=gs.np_float),
                np.array([geom.type for geom in geoms], dtype=gs.np_int),
                np.array([geom.desc.friction for geom in geoms], dtype=gs.np_float),
                np.array([geom.desc.friction_torsional for geom in geoms], dtype=gs.np_float),
                np.array([geom.desc.friction_rolling for geom in geoms], dtype=gs.np_float),
                geoms_sol_params,
                np.array([geom.data for geom in geoms], dtype=gs.np_float),
                np.array([geom.is_convex for geom in geoms], dtype=gs.np_bool),
                np.array([geom.needs_coup for geom in geoms], dtype=gs.np_int),
                np.array([geom.contype for geom in geoms], dtype=np.int32),
                np.array([geom.conaffinity for geom in geoms], dtype=np.int32),
                np.array([geom.coup_softness for geom in geoms], dtype=gs.np_float),
                np.array([geom.coup_friction for geom in geoms], dtype=gs.np_float),
                np.array([geom.coup_restitution for geom in geoms], dtype=gs.np_float),
                np.array([geom.is_fixed for geom in geoms], dtype=gs.np_bool),
                np.array([geom.metadata.get("decomposed", False) for geom in geoms], dtype=gs.np_bool),
                np.array(geoms_is_hollow, dtype=gs.np_bool),
                geoms_init_AABB,
                dyn_state,
                dyn_info,
                self.rigid_config,
            )
        return geoms_init_AABB

    def _init_entity_fields(
        self,
        dyn_state: array_class.DynState,
        dyn_info: array_class.DynInfo,
        rigid_info: array_class.RigidInfo,
    ) -> None:
        if self.entities:
            entities = self.entities
            kernel_init_entity_fields(
                np.array([entity.dof_start for entity in entities], dtype=gs.np_int),
                np.array([entity.dof_end for entity in entities], dtype=gs.np_int),
                np.array([entity.link_start for entity in entities], dtype=gs.np_int),
                np.array([entity.link_end for entity in entities], dtype=gs.np_int),
                np.array([entity.geom_start for entity in entities], dtype=gs.np_int),
                np.array([entity.geom_end for entity in entities], dtype=gs.np_int),
                np.array([entity.gravity_compensation for entity in entities], dtype=gs.np_float),
                np.array([entity.is_local_collision_mask for entity in entities], dtype=gs.np_bool),
                dyn_state,
                dyn_info,
                rigid_info,
                self.rigid_config,
            )

    def _init_equality_fields(self, dyn_info: array_class.DynInfo) -> None:
        if self.n_equalities > 0:
            equalities = self.equalities

            equalities_sol_params = np.array([equality.desc.sol_params for equality in equalities], dtype=gs.np_float)
            sanitize_sol_params(
                equalities_sol_params,
                self.sol_min_timeconst,
                self.sol_default_timeconst,
            )

            kernel_init_equality_fields(
                np.array([equality.type for equality in equalities], dtype=gs.np_int),
                np.array([equality.eq_obj1id for equality in equalities], dtype=gs.np_int),
                np.array([equality.eq_obj2id for equality in equalities], dtype=gs.np_int),
                np.array([equality.eq_data for equality in equalities], dtype=gs.np_float),
                np.array([equality.type for equality in equalities], dtype=gs.np_int),
                equalities_sol_params,
                dyn_info,
                self.rigid_config,
            )

    def _init_dynamics(self) -> None:
        """Initialize the collision geometry, equality constraints and rigid dynamics."""
        self._init_vert_fields(self.dyn_info)
        self.geoms_init_AABB = self._init_geom_fields(
            self.dyn_state,
            self.dyn_info,
            self.rigid_info,
        )
        self._init_equality_fields(self.dyn_info)
        self._init_dof_length(self.dyn_info)

        # Compute the pose of every link in the neutral configuration and update the pose of all the geometries,
        # including inactive variants for a given env: the step updates the active variant alone, so this is the only
        # pose the inactive ones get (see func_update_geoms_link). The collider built next relies on these updated poses
        # to determine whether some collision pairs must be filtered out.
        kernel_update_cartesian_space(
            self.dyn_state,
            self.dyn_info,
            self.rigid_info,
            self.rigid_config,
            force_update_all_geoms=True,
            is_backward=False,
        )

    def _initialize_inverse_weights(self) -> None:
        """Settle inverse weights and default rotor inertia after contact and constraint data exist."""
        self.refresh_invweight_and_meaninertia(force_update=False, in_place=True)

        # Fill in the default rotor inertia (see 'KinematicVariantDescription'), one variant per environment. It is
        # written to the armature field from the host once the parsed inverse weights stand, and a second refresh
        # recomputes the ones it changes: those of every degree of freedom and link of the kinematic trees holding a
        # defaulted joint. The parsed inverse weights stand everywhere else.
        dofs_idx, dofs_default, dofs_link = [], [], []
        for entity in self._entities:
            rotor_links = [
                link
                for link in entity.links
                if link.n_dofs == 1 and link.joints[0].type in (gs.JOINT_TYPE.REVOLUTE, gs.JOINT_TYPE.PRISMATIC)
            ]
            if not rotor_links:
                continue
            # Only a scene or robot description file yields a rotor link, and those morphs state a default armature.
            if entity.desc.variants:
                variants_default = np.array([variant.default_armature or 0.0 for variant in entity.desc.variants])
            else:
                variants_default = np.array([entity.main_morph.default_armature or 0.0])
            envs_default = variants_default[_balanced_variant_mapping(variants_default.size, self._B)]
            if (envs_default <= 0.0).all():
                continue
            dofs_idx.extend(link.dof_start for link in rotor_links)
            dofs_default.extend([envs_default] * len(rotor_links))
            dofs_link.extend(rotor_links)
        if dofs_idx:
            # Field layout, batch last, since the arrays are written back as fields.
            dofs_armature = qd_to_numpy(self.dyn_info.dofs.armature, transpose=False, copy=True)
            is_default = (np.atleast_2d(dofs_armature[dofs_idx].T) <= 0.0).all(axis=0)
            if is_default.any():
                dofs_idx = np.array(dofs_idx)[is_default]
                default_armature = np.stack(dofs_default, axis=1)[:, is_default]
                dofs_armature[dofs_idx] = default_armature.T if self._options.batch_dofs_info else default_armature[0]
                self.dyn_info.dofs.armature.from_numpy(dofs_armature)
                roots_idx = {link.root_idx for link, is_dof_default in zip(dofs_link, is_default) if is_dof_default}
                trees_links = [link for link in self.links if link.root_idx in roots_idx]
                dofs_invweight = qd_to_numpy(self.dyn_info.dofs.invweight, transpose=False, copy=True)
                dofs_invweight[[i_d for link in trees_links for i_d in range(link.dof_start, link.dof_end)]] = -1.0
                self.dyn_info.dofs.invweight.from_numpy(dofs_invweight)
                links_idx = [link.idx for link in trees_links]
                links_invweight = qd_to_numpy(self.dyn_info.links.invweight, transpose=False, copy=True)
                links_invweight[links_idx] = -1.0
                self.dyn_info.links.invweight.from_numpy(links_invweight)
                self.refresh_invweight_and_meaninertia(force_update=False, in_place=True)

        # The constraint solver decides it has converged from quantities summed over the whole scene, and every DOF of
        # a link contributes a cost of the order of the link's mass. A link whose mass is a tolerance-fraction of the
        # scene total therefore contributes less than the tolerance, and the solve stops while that link still carries
        # residual. The floor below follows the tolerance linearly, down to the resolution of the working precision in
        # kilograms. The scene total is summed before the loop below folds each link into its parent, after which every
        # entry holds the mass of a whole subtree, which is what the floor is compared against.
        links_subtree_mass = np.atleast_2d(qd_to_numpy(self.dyn_info.links.inertial_mass, transpose=True, copy=True))
        movable_links_idx = np.flatnonzero([link.n_dofs > 0 for link in self.links])
        if movable_links_idx.size:
            total_mass = links_subtree_mass[:, movable_links_idx].sum(axis=1)
            for i_l in reversed(range(self._n_links)):
                if self.links[i_l].parent_idx >= 0:
                    links_subtree_mass[:, self.links[i_l].parent_idx] += links_subtree_mass[:, i_l]
            movable_links_mass = links_subtree_mass[:, movable_links_idx]
            mass_floor = np.maximum(self._options.tolerance * total_mass, 0.2 * self._options.tolerance)[:, None]
            i_b_min, i_l_min = np.unravel_index((movable_links_mass / mass_floor).argmin(), movable_links_mass.shape)
            mass_min = movable_links_mass[i_b_min, i_l_min]
            if mass_min < mass_floor[i_b_min, 0]:
                link = self.links[movable_links_idx[i_l_min]]
                if gs.qd_float == qd.f32:
                    remedy = "Use 64-bit simulation precision or tighten the solver tolerance."
                else:
                    remedy = "Tighten the solver tolerance."
                gs.logger.warning(
                    f"Link '{link.name}' has mass {mass_min:.1e}, too small for the constraint solver to be "
                    f"numerically stable. {remedy} Note that increasing the solver iterations would not help."
                )

    # -------------------------- Kinematic model initialization -------------------------

    def _init_dof_fields(
        self,
        dyn_state: array_class.DynState,
        dyn_info: array_class.DynInfo,
        rigid_info: array_class.RigidInfo,
    ) -> None:
        joints = self.joints
        has_dofs = sum(joint.n_dofs for joint in joints) > 0
        if has_dofs:
            kernel_init_dof_fields(
                np.concatenate(
                    [(joint.link._entity_idx_in_solver,) * joint.n_dofs for joint in joints if joint.n_dofs],
                    dtype=gs.np_int,
                ),
                np.concatenate([joint.desc.dofs_motion_ang for joint in joints], dtype=gs.np_float),
                np.concatenate([joint.desc.dofs_motion_vel for joint in joints], dtype=gs.np_float),
                np.concatenate([joint.desc.dofs_limit for joint in joints], dtype=gs.np_float),
                np.concatenate([joint.desc.dofs_invweight for joint in joints], dtype=gs.np_float),
                np.concatenate([joint.desc.dofs_stiffness for joint in joints], dtype=gs.np_float),
                np.concatenate([joint.desc.dofs_damping for joint in joints], dtype=gs.np_float),
                np.concatenate([joint.desc.dofs_frictionloss for joint in joints], dtype=gs.np_float),
                np.concatenate([joint.desc.dofs_armature for joint in joints], dtype=gs.np_float),
                np.concatenate([joint.desc.dofs_act_gain for joint in joints], dtype=gs.np_float),
                np.concatenate([joint.desc.dofs_act_bias for joint in joints], dtype=gs.np_float),
                np.concatenate([joint.desc.dofs_force_range for joint in joints], dtype=gs.np_float),
                dyn_state,
                dyn_info,
                rigid_info,
                self.rigid_config,
            )

        dyn_state.dofs.force.fill(0)

    def _init_kinematic_tree_fields(self, rigid_info: array_class.RigidInfo) -> None:
        """Initialize the fields describing the kinematic roots and trees (see roots_link_idx and trees_root_idx in
        array_class.py).

        The links come parent first and each branch occupies a contiguous index range, so the dofs of a tree form one
        contiguous range and the trees are disjoint in dof space.
        """
        # A static link belongs to no tree, and a scene without any tree keeps its padded tree slot empty (see
        # trees_root_idx in array_class.py)
        links_tree_idx = np.full(self.n_links_, -1, dtype=gs.np_int)
        if self._n_roots:
            roots_link_idx, links_root_rank = np.unique(self._links_root_idx, return_inverse=True)
            roots_link_end = np.zeros(self._n_roots, dtype=gs.np_int)
            np.maximum.at(roots_link_end, links_root_rank, np.arange(1, self.n_links + 1, dtype=gs.np_int))
            rigid_info.roots_link_idx.from_numpy(roots_link_idx)
            rigid_info.links_root_end.from_numpy(roots_link_end[links_root_rank])
        if self._n_trees:
            links_n_dofs = np.array([link.n_dofs for link in self.links], dtype=gs.np_int)
            links_dof_start = np.array([link.dof_start for link in self.links], dtype=gs.np_int)
            links_dof_end = np.array([link.dof_end for link in self.links], dtype=gs.np_int)
            tree_links = np.flatnonzero(self._links_tree_root_idx >= 0)
            trees_root_idx, links_tree_rank = np.unique(self._links_tree_root_idx[tree_links], return_inverse=True)
            trees_n_dofs = np.zeros(self._n_trees, dtype=gs.np_int)
            np.add.at(trees_n_dofs, links_tree_rank, links_n_dofs[tree_links])
            # A dof-less link carries no dof range of its own
            dof_links = np.flatnonzero(links_n_dofs[tree_links])
            trees_dof_start = np.full(self._n_trees, self.n_dofs, dtype=gs.np_int)
            np.minimum.at(trees_dof_start, links_tree_rank[dof_links], links_dof_start[tree_links][dof_links])
            trees_dof_end = np.zeros(self._n_trees, dtype=gs.np_int)
            np.maximum.at(trees_dof_end, links_tree_rank[dof_links], links_dof_end[tree_links][dof_links])
            if (trees_dof_end - trees_dof_start != trees_n_dofs).any():
                gs.raise_exception("The dofs of a kinematic tree must be contiguous.")
            trees_link_end = np.zeros(self._n_trees, dtype=gs.np_int)
            np.maximum.at(trees_link_end, links_tree_rank, tree_links + 1)
            trees_n_links = np.zeros(self._n_trees, dtype=gs.np_int)
            np.add.at(trees_n_links, links_tree_rank, 1)
            trees_order = np.argsort(trees_dof_start)
            trees_rank = np.empty(self._n_trees, dtype=gs.np_int)
            trees_rank[trees_order] = np.arange(self._n_trees, dtype=gs.np_int)
            links_tree_idx[tree_links] = trees_rank[links_tree_rank]
            rigid_info.trees_root_idx.from_numpy(trees_root_idx[trees_order])
            rigid_info.trees_link_end.from_numpy(trees_link_end[trees_order])
            rigid_info.trees_n_links.from_numpy(trees_n_links[trees_order])
            rigid_info.trees_dof_start.from_numpy(trees_dof_start[trees_order])
            rigid_info.trees_n_dofs.from_numpy(trees_n_dofs[trees_order])
        else:
            rigid_info.trees_root_idx.fill(0)
            rigid_info.trees_link_end.fill(0)
        rigid_info.links_tree_idx.from_numpy(links_tree_idx)

        # Fill the level tables of every tree and root for the level sweep (see trees_levels_links_idx in
        # array_class.py). A scene without any tree keeps its padded tree slot at zero levels.
        if self._n_roots and self.rigid_config.enable_level_sweep:
            for links_group_idx, links_group_level, n_groups, groups_fields in (
                (
                    links_tree_idx,
                    self._links_tree_level,
                    self.n_trees_,
                    (
                        rigid_info.trees_level_start,
                        rigid_info.trees_n_levels,
                        rigid_info.trees_levels_links_idx,
                        None,
                        rigid_info.trees_levels_links_end,
                    ),
                ),
                (
                    links_root_rank,
                    self._links_root_level,
                    self._n_roots,
                    (
                        rigid_info.roots_level_start,
                        rigid_info.roots_n_levels,
                        rigid_info.roots_levels_links_idx,
                        rigid_info.roots_levels_links_start,
                        rigid_info.roots_levels_links_end,
                    ),
                ),
            ):
                group_links = np.arange(self.n_links, dtype=gs.np_int)[links_group_idx >= 0]
                links_group = links_group_idx[group_links]
                links_level = links_group_level[group_links]
                groups_n_levels = np.zeros(n_groups, dtype=gs.np_int)
                np.maximum.at(groups_n_levels, links_group, links_level + 1)
                groups_level_start = np.zeros(n_groups, dtype=gs.np_int)
                np.cumsum(np.bincount(links_group, minlength=n_groups)[:-1], out=groups_level_start[1:])
                # The links of every group, group after group, level after level, and each level in ascending order
                entries_order = np.lexsort((group_links, links_level, links_group))
                groups_levels_links_idx = group_links[entries_order]
                entries_group = links_group[entries_order]
                entries_level = links_level[entries_order]
                entries_idx = np.arange(len(groups_levels_links_idx), dtype=gs.np_int)
                # An entry ends its level where the next entry starts another level or another group
                is_level_last = np.ones(len(groups_levels_links_idx), dtype=bool)
                is_level_last[:-1] = (np.diff(entries_group) != 0) | (np.diff(entries_level) != 0)
                levels_last_entry = entries_idx[is_level_last]
                groups_levels_links_end = levels_last_entry[np.searchsorted(levels_last_entry, entries_idx)] + 1
                is_level_first = np.ones(len(groups_levels_links_idx), dtype=bool)
                is_level_first[1:] = is_level_last[:-1]
                levels_first_entry = entries_idx[is_level_first]
                groups_levels_links_start = levels_first_entry[
                    np.searchsorted(levels_first_entry, entries_idx, side="right") - 1
                ]
                for field, value in zip(
                    groups_fields,
                    (
                        groups_level_start,
                        groups_n_levels,
                        groups_levels_links_idx,
                        groups_levels_links_start,
                        groups_levels_links_end,
                    ),
                ):
                    if field is not None and len(value):
                        field.from_numpy(value)
            child_links = np.arange(self.n_links, dtype=gs.np_int)[self._links_parent_idx >= 0]
            links_child_start = np.zeros(self.n_links_ + 1, dtype=gs.np_int)
            np.cumsum(
                np.bincount(self._links_parent_idx[child_links], minlength=self.n_links_), out=links_child_start[1:]
            )
            rigid_info.links_child_start.from_numpy(links_child_start)
            if len(child_links):
                rigid_info.links_child_idx.from_numpy(
                    child_links[np.lexsort((-child_links, self._links_parent_idx[child_links]))]
                )

    def _init_kinematic_link_fields(
        self,
        dyn_state: array_class.DynState,
        dyn_info: array_class.DynInfo,
        rigid_info: array_class.RigidInfo,
        dispatch_vgeoms: Callable[[array_class.DynInfo], None],
        sanitize_sol_params: Callable[..., HostTensor],
    ) -> None:
        if self.links:
            links = self.links
            kernel_init_link_fields(
                np.array([link.parent_idx for link in links], dtype=gs.np_int),
                np.array([link.root_idx for link in links], dtype=gs.np_int),
                np.array([link.q_start for link in links], dtype=gs.np_int),
                np.array([link.dof_start for link in links], dtype=gs.np_int),
                np.array([link.joint_start for link in links], dtype=gs.np_int),
                np.array([link.q_end for link in links], dtype=gs.np_int),
                np.array([link.dof_end for link in links], dtype=gs.np_int),
                np.array([link.joint_end for link in links], dtype=gs.np_int),
                np.array([link._entity_idx_in_solver for link in links], dtype=gs.np_int),
                np.array([link.geom_start for link in links], dtype=gs.np_int),
                np.array([link.geom_end for link in links], dtype=gs.np_int),
                np.array([link.vgeom_start for link in links], dtype=gs.np_int),
                np.array([link.vgeom_end for link in links], dtype=gs.np_int),
                np.array([link.is_fixed for link in links], dtype=gs.np_bool),
                np.array([link.desc.pos for link in links], dtype=gs.np_float),
                np.array([link.desc.quat for link in links], dtype=gs.np_float),
                dyn_state,
                dyn_info,
                rigid_info,
                self.rigid_config,
            )

        if self.joints:
            joints = self.joints
            joints_sol_params = np.array([joint.desc.sol_params for joint in joints], dtype=gs.np_float)
            sanitize_sol_params(joints_sol_params, self.sol_min_timeconst, self.sol_default_timeconst)

            kernel_init_joint_fields(
                np.array([joint.q_start for joint in joints], dtype=gs.np_int),
                np.array([joint.dof_start for joint in joints], dtype=gs.np_int),
                np.array([joint.q_end for joint in joints], dtype=gs.np_int),
                np.array([joint.dof_end for joint in joints], dtype=gs.np_int),
                np.array([joint.type for joint in joints], dtype=gs.np_int),
                joints_sol_params,
                np.array([joint.pos for joint in joints], dtype=gs.np_float),
                dyn_info,
                self.rigid_config,
            )

        self._dispatch_heterogeneous_kinematics(dyn_state, dyn_info, sanitize_sol_params)

        # Set initial qpos
        qpos = rigid_info.qpos
        qpos0 = rigid_info.qpos0
        if self.n_qs > 0:
            init_qpos = np.tile(np.expand_dims(self.init_qpos, -1), (1, self._B))

            # Dispatch per-variant init_qpos for heterogeneous entities
            for entity in self.entities:
                if not entity._desc.variants:
                    continue
                n_variants = len(entity._desc.variants)
                variant_idx = _balanced_variant_mapping(n_variants, self._B)
                q_s, q_e = entity.q_start, entity.q_start + entity.n_qs
                for i_b in range(self._B):
                    init_qpos[q_s:q_e, i_b] = entity._desc.variants[variant_idx[i_b]].init_qpos

            qpos0.from_numpy(init_qpos)
            dofs_limit = np.broadcast_to(qd_to_numpy(dyn_info.dofs.limit, transpose=True), (self._B, self.n_dofs_, 2))
            is_init_qpos_out_of_bounds = False
            for joint in self.joints:
                if joint.type in (gs.JOINT_TYPE.REVOLUTE, gs.JOINT_TYPE.PRISMATIC):
                    is_init_qpos_out_of_bounds |= (dofs_limit[:, joint.dof_start, 0] > init_qpos[joint.q_start]).any()
                    is_init_qpos_out_of_bounds |= (init_qpos[joint.q_start] > dofs_limit[:, joint.dof_start, 1]).any()
            if is_init_qpos_out_of_bounds:
                gs.logger.warning("Neutral robot position (qpos0) exceeds joint limits.")
            qpos.from_numpy(init_qpos)

        # Dispatch heterogeneous variant vgeom ranges per-environment
        dispatch_vgeoms(dyn_info)

    def _dispatch_heterogeneous_kinematics(
        self,
        dyn_state: array_class.DynState,
        dyn_info: array_class.DynInfo,
        sanitize_sol_params: Callable[..., HostTensor],
    ) -> None:
        """Write the link frames and the joint and dof parameters of each heterogeneous variant to the envs carrying it.

        The init kernels broadcast those of the primary to every environment, while heterogeneous simulation batches
        the link, joint and dof info (see 'build').
        """
        entities = [entity for entity in self.entities if entity._desc.variants]
        if not entities:
            return

        # Field layout, batch second, since the arrays are written back as fields
        links_pos = qd_to_numpy(dyn_info.links.pos, transpose=False, copy=True)
        links_quat = qd_to_numpy(dyn_info.links.quat, transpose=False, copy=True)
        joints_pos = qd_to_numpy(dyn_info.joints.pos, transpose=False, copy=True)
        joints_sol_params = qd_to_numpy(dyn_info.joints.sol_params, transpose=False, copy=True)
        dofs_info = dyn_info.dofs
        dofs_fields = (
            (dofs_info.motion_ang, "dofs_motion_ang"),
            (dofs_info.motion_vel, "dofs_motion_vel"),
            (dofs_info.limit, "dofs_limit"),
            (dofs_info.stiffness, "dofs_stiffness"),
            (dofs_info.damping, "dofs_damping"),
            (dofs_info.frictionloss, "dofs_frictionloss"),
            (dofs_info.armature, "dofs_armature"),
            (dofs_info.act_gain, "dofs_act_gain"),
            (dofs_info.act_bias, "dofs_act_bias"),
            (dofs_info.force_range, "dofs_force_range"),
        )
        dofs_values = [qd_to_numpy(field, transpose=False, copy=True) for field, _ in dofs_fields]

        for entity in entities:
            variants_links = [entity._desc.links, *(variant.links for variant in entity._desc.variants[1:])]
            envs_variant_idx = _balanced_variant_mapping(len(variants_links), self._B)
            links_slice = slice(entity.link_start, entity.link_end)
            variants_links_pos = np.stack([[l_desc.pos for l_desc in links] for links in variants_links])
            variants_links_quat = np.stack([[l_desc.quat for l_desc in links] for links in variants_links])
            links_pos[links_slice] = variants_links_pos[envs_variant_idx].swapaxes(0, 1)
            links_quat[links_slice] = variants_links_quat[envs_variant_idx].swapaxes(0, 1)

            # A file morph drops its joints without degrees of freedom, so a fixed one may leave the entity without any
            if entity.n_joints == 0:
                continue
            variants_joints = [
                [asdict(j_desc) for l_desc in links for j_desc in l_desc.joints] for links in variants_links
            ]
            joints_slice = slice(entity.joint_start, entity.joint_end)
            variants_joints_pos = np.stack([[j_fields["pos"] for j_fields in joints] for joints in variants_joints])
            joints_pos[joints_slice] = variants_joints_pos[envs_variant_idx].swapaxes(0, 1)
            variants_sol_params = np.stack(
                [[j_fields["sol_params"] for j_fields in joints] for joints in variants_joints]
            )
            sanitize_sol_params(variants_sol_params, self.sol_min_timeconst, self.sol_default_timeconst)
            variants_sol_params = variants_sol_params.reshape(variants_sol_params.shape)
            joints_sol_params[joints_slice] = variants_sol_params[envs_variant_idx].swapaxes(0, 1)
            dofs_slice = slice(entity.dof_start, entity.dof_end)
            for dofs_value, (_, name) in zip(dofs_values, dofs_fields):
                variants_value = np.stack(
                    [np.concatenate([j_fields[name] for j_fields in joints]) for joints in variants_joints]
                )
                dofs_value[dofs_slice] = variants_value[envs_variant_idx].swapaxes(0, 1)

        dyn_info.links.pos.from_numpy(links_pos)
        dyn_info.links.quat.from_numpy(links_quat)
        dyn_info.joints.pos.from_numpy(joints_pos)
        dyn_info.joints.sol_params.from_numpy(joints_sol_params)
        for (field, _), dofs_value in zip(dofs_fields, dofs_values):
            field.from_numpy(dofs_value)

        # The state of a fixed root link is set at init alone (see kernel_init_link_fields), so it takes the frame of
        # the variant its environment carries too
        fixed_roots_idx = [entity.link_start for entity in entities if entity.base_link.is_fixed]
        if fixed_roots_idx:
            links_state_pos = qd_to_numpy(dyn_state.links.pos, transpose=False, copy=True)
            links_state_quat = qd_to_numpy(dyn_state.links.quat, transpose=False, copy=True)
            links_state_pos[fixed_roots_idx] = links_pos[fixed_roots_idx]
            links_state_quat[fixed_roots_idx] = links_quat[fixed_roots_idx]
            dyn_state.links.pos.from_numpy(links_state_pos)
            dyn_state.links.quat.from_numpy(links_state_quat)

    def _dispatch_kinematic_heterogeneous_vgeoms(self, dyn_info: array_class.DynInfo) -> None:
        """Override per-link vgeom ranges for heterogeneous variants, which rigid_dispatch_heterogeneous_vgeoms extends."""
        for link in self.links:
            if link._variant_vgeom_ranges is None:
                continue

            n_variants = len(link._variant_vgeom_ranges)
            variant_idx = _balanced_variant_mapping(n_variants, self._B)

            vgeom_starts = np.array([link._variant_vgeom_ranges[v][0] for v in variant_idx], dtype=gs.np_int)
            vgeom_ends = np.array([link._variant_vgeom_ranges[v][1] for v in variant_idx], dtype=gs.np_int)

            kernel_update_heterogeneous_links_vgeom(link.idx, vgeom_starts, vgeom_ends, dyn_info)

            for vgeom in link.vgeoms:
                active_envs_mask = (vgeom_starts <= vgeom.idx) & (vgeom.idx < vgeom_ends)
                vgeom.active_envs_mask = torch.tensor(active_envs_mask, device=gs.device)
                (vgeom.active_envs_idx,) = np.where(active_envs_mask)

    def _init_vvert_fields(self, dyn_info: array_class.DynInfo) -> None:
        if self.n_vverts == 0:
            return

        vgeoms = self.vgeoms
        vverts = np.concatenate([vg.init_vverts for vg in vgeoms], dtype=gs.np_float)
        vnormals = np.concatenate([vg.init_vnormals for vg in vgeoms], dtype=gs.np_float)
        vfaces = np.concatenate([vg.init_vfaces + vg._vvert_start for vg in vgeoms], dtype=gs.np_int)
        vverts_vgeom_idx = np.concatenate([np.full(vg.n_vverts, vg.idx) for vg in vgeoms], dtype=gs.np_int)
        vverts_state_idx = np.full(self.n_vverts, -1, dtype=gs.np_int)
        for entity in self.entities:
            if not entity._morph.enable_custom_vverts:
                continue
            entity_custom_offset = entity._custom_vvert_start - entity._vvert_start
            for vgeom in entity.vgeoms:
                local = np.arange(vgeom.n_vverts, dtype=gs.np_int)
                vverts_state_idx[vgeom._vvert_start + local] = vgeom._vvert_start + entity_custom_offset + local
        kernel_init_vvert_fields(
            vverts_vgeom_idx,
            vverts_state_idx,
            vverts,
            vfaces,
            vnormals,
            dyn_info,
            self.rigid_config,
        )

    def _init_vgeom_fields(self, dyn_info: array_class.DynInfo) -> None:
        if self.n_vgeoms > 0:
            vgeoms = self.vgeoms
            kernel_init_vgeom_fields(
                np.array([vgeom.link.idx for vgeom in vgeoms], dtype=gs.np_int),
                np.array([vgeom.vvert_start for vgeom in vgeoms], dtype=gs.np_int),
                np.array([vgeom.vface_start for vgeom in vgeoms], dtype=gs.np_int),
                np.array([vgeom.vvert_end for vgeom in vgeoms], dtype=gs.np_int),
                np.array([vgeom.vface_end for vgeom in vgeoms], dtype=gs.np_int),
                np.array([vgeom.init_pos for vgeom in vgeoms], dtype=gs.np_float),
                np.array([vgeom.init_quat for vgeom in vgeoms], dtype=gs.np_float),
                np.array([vgeom.desc.vmesh.color for vgeom in vgeoms], dtype=gs.np_float),
                dyn_info,
                self.rigid_config,
            )

    def _init_vfaces_raycast_mask(self, dyn_info: array_class.DynInfo) -> torch.Tensor:
        """Fit the static per-vface visual-raycasting opt-in mask; see the vfaces_raycast_mask property.

        Sized over the padded vgeom count so the gather also covers a solver holding no vgeom at all, whose vface
        fields still carry one padding slot: nothing opts in there, so every entry stays 0.
        """
        vgeom_enabled = torch.zeros(self.n_vgeoms_, dtype=gs.tc_bool, device=gs.device)
        for entity in self.entities:
            if not entity.material.use_visual_raycasting:
                continue
            for vgeom in entity.vgeoms:
                vgeom_enabled[vgeom.idx] = 1
        return vgeom_enabled[qd_to_torch(dyn_info.vfaces.vgeom_idx)]

    def _init_vverts_state(self, dyn_state: array_class.DynState, dyn_info: array_class.DynInfo) -> None:
        # Seed vverts_state.pos with one FK pass so opt-in entities show their initial pose from frame 0 even before
        # the user touches the buffer with set_vverts.
        if self.n_custom_vverts == 0:
            return
        opt_in_vgeoms_idx = np.array(
            [vgeom.idx for entity in self.entities if entity._morph.enable_custom_vverts for vgeom in entity.vgeoms],
            dtype=gs.np_int,
        )
        if opt_in_vgeoms_idx.size == 0:
            return
        kernel_update_vgeoms(dyn_state, dyn_info, self.rigid_config)
        kernel_update_vverts_for_vgeoms(opt_in_vgeoms_idx, dyn_state, dyn_info, self.rigid_config)

    # -------------------------- Rigid kinematic implementation --------------------------

    def _build_kinematic_model(self) -> None:
        """Build host-side entity topology without allocating the standalone backend."""
        self.n_envs = self.sim.n_envs
        self._B = self.sim._B
        self._para_level = self.sim._para_level

        for entity in self._entities:
            entity._build()

        # Resolve link reachability once; the frontend facade uses these indices for mutation notifications.
        self._articulated_links_idx = np.array([link.idx for link in self.links if not link.is_fixed])
        self._links_parent_idx = np.array([link.parent_idx for link in self.links])

        self._n_qs = self.n_qs
        self._n_dofs = self.n_dofs
        self._n_links = self.n_links
        self._n_joints = self.n_joints
        self._n_vgeoms = self.n_vgeoms
        self._n_vfaces = self.n_vfaces
        self._n_vverts = self.n_vverts
        self._n_custom_vverts = self.n_custom_vverts
        self._n_custom_vfaces = self.n_custom_vfaces
        self._n_entities = self.n_entities

        self._vgeoms = self.vgeoms
        self._links = self.links
        self._joints = self.joints

        base_links_idx = []
        for link in self.links:
            if link.parent_idx == -1 and link.is_fixed:
                base_links_idx.append(link.idx)
        for joint in self.joints:
            if joint.type == gs.JOINT_TYPE.FREE:
                base_links_idx.append(joint.link.idx)
        self.base_links_idx = torch.tensor(base_links_idx, dtype=gs.tc_int, device=gs.device)

        # World<-authored pose offset, applied by the relative get/set methods. Links carry the morph offset composed
        # with the base-link inertial alignment (identity for non-base links); geoms and visual geoms carry only the
        # morph offset, since alignment re-expresses them into the aligned frame. Heterogeneous entities vary the
        # base-link offset per environment (one offset per variant), so the link offset is stored per-env when any such
        # entity has divergent variant offsets, and without an environment axis otherwise. The relative getters
        # recompute the inverse on the fly.
        links_offset_per_env = any(
            not all(
                np.allclose(variant.offset_pos, entity._desc.variants[0].offset_pos)
                and np.allclose(variant.offset_quat, entity._desc.variants[0].offset_quat)
                for variant in entity._desc.variants
            )
            for entity in self._entities
        )
        offset_shape = (self._B, self.n_links) if links_offset_per_env else (self.n_links,)
        links_offset_pos = np.zeros((*offset_shape, 3), dtype=gs.np_float)
        links_offset_quat = np.tile(gu.identity_quat(), (*offset_shape, 1))
        vgeoms_offset_pos = np.zeros((self.n_vgeoms, 3), dtype=gs.np_float)
        vgeoms_offset_quat = np.tile(gu.identity_quat(), (self.n_vgeoms, 1))
        for entity in self._entities:
            if links_offset_per_env and entity._desc.variants:
                variant_idx = _balanced_variant_mapping(len(entity._desc.variants), self._B)
                links_offset_pos[np.arange(self._B), entity.base_link_idx] = np.stack(
                    [entity._desc.variants[v].offset_pos for v in variant_idx]
                )
                links_offset_quat[np.arange(self._B), entity.base_link_idx] = np.stack(
                    [entity._desc.variants[v].offset_quat for v in variant_idx]
                )
                vgeoms_ranges = entity.base_link._variant_vgeom_ranges
            else:
                # Each root link carries its own offset; children inherit it through the kinematic chain and stay
                # identity. Visual geoms get the morph offset conjugated into their own frame.
                for local_idx, link in enumerate(entity.links):
                    links_offset_pos[..., entity._link_start + local_idx, :] = link.desc.offset_pos
                    links_offset_quat[..., entity._link_start + local_idx, :] = link.desc.offset_quat
                vgeoms_ranges = None
            _fill_base_link_geom_offsets(vgeoms_offset_pos, vgeoms_offset_quat, entity, entity.vgeoms, vgeoms_ranges)
        # Per-link identity masks gate the relative get/set/backward paths: a relative set on a link whose offset is
        # identity is a plain passthrough, while a non-identity offset rewrites the world position or drops the
        # composition jacobian. Kept as host-side numpy (never used in kernels) so the gates avoid a GPU->host sync.
        self.links_offset_quat_is_identity = np.all(
            np.isclose(gu.quat_to_xyz(links_offset_quat), 0.0, atol=gs.EPS), axis=-1
        )
        self.links_offset_pos_is_identity = np.all(np.isclose(links_offset_pos, 0.0, atol=gs.EPS), axis=-1)
        if links_offset_per_env:
            self.links_offset_quat_is_identity = self.links_offset_quat_is_identity.all(axis=0)
            self.links_offset_pos_is_identity = self.links_offset_pos_is_identity.all(axis=0)
        # The link offset tensors are allocated even when every link is identity, so the getter kernels can take them
        # unconditionally.
        self.links_offset_pos = torch.from_numpy(links_offset_pos).to(device=gs.device, dtype=gs.tc_float)
        self.links_offset_quat = torch.from_numpy(links_offset_quat).to(device=gs.device, dtype=gs.tc_float)
        self.vgeoms_offset_pos = self.vgeoms_offset_quat = None
        if not (
            np.allclose(vgeoms_offset_pos, 0.0, atol=gs.EPS)
            and np.allclose(gu.quat_to_xyz(vgeoms_offset_quat), 0.0, atol=gs.EPS)
        ):
            self.vgeoms_offset_pos = torch.from_numpy(vgeoms_offset_pos).to(device=gs.device, dtype=gs.tc_float)
            self.vgeoms_offset_quat = torch.from_numpy(vgeoms_offset_quat).to(device=gs.device, dtype=gs.tc_float)

        self.n_qs_ = max(1, self.n_qs)
        self.n_dofs_ = max(1, self.n_dofs)
        self.n_links_ = max(1, self.n_links)
        self.n_joints_ = max(1, self.n_joints)
        self.n_vgeoms_ = max(1, self.n_vgeoms)
        self.n_vfaces_ = max(1, self.n_vfaces)
        self.n_vverts_ = max(1, self.n_vverts)
        self.n_custom_vverts_ = max(1, self.n_custom_vverts)
        self.n_custom_vfaces_ = max(1, self.n_custom_vfaces)
        self.n_entities_ = max(1, self.n_entities)

        # The kinematic roots and trees, see roots_link_idx and trees_root_idx in array_class.py. A link no dof moves,
        # its own or an ancestor's, is static and belongs to no tree. A moving link roots the tree of its parent, or its
        # own when its parent is static or absent. The parents precede the children.
        self._n_roots = len({link.root_idx for link in self.links})
        self.n_roots_ = max(1, self._n_roots)
        links_is_static = np.ones(self.n_links, dtype=bool)
        links_tree_root_idx = np.full(self.n_links, -1, dtype=gs.np_int)
        for link in self.links:
            is_parent_static = link.parent_idx == -1 or links_is_static[link.parent_idx]
            links_is_static[link.idx] = is_parent_static and link.n_dofs == 0
            if links_is_static[link.idx]:
                continue
            links_tree_root_idx[link.idx] = link.idx if is_parent_static else links_tree_root_idx[link.parent_idx]
        self._links_tree_root_idx = links_tree_root_idx
        self._n_trees = np.unique(links_tree_root_idx[links_tree_root_idx >= 0]).size
        self.n_trees_ = max(1, self._n_trees)

        # The level of a link in its root and in its tree is its number of ancestors below the root link of either (see
        # trees_levels_links_idx in array_class.py). Each pass moves every link one ancestor up, so the depth of a link
        # is the number of passes it takes to run out of parents.
        links_depth = np.zeros(self.n_links, dtype=gs.np_int)
        links_ancestor_idx = self._links_parent_idx
        while (links_ancestor_idx >= 0).any():
            has_ancestor = links_ancestor_idx >= 0
            links_depth += has_ancestor
            links_ancestor_idx = np.where(has_ancestor, self._links_parent_idx[links_ancestor_idx], -1)
        self._links_root_idx = np.array([link.root_idx for link in self.links], dtype=gs.np_int)
        self._links_root_level = links_depth - links_depth[self._links_root_idx]
        self._links_tree_level = np.where(links_tree_root_idx >= 0, links_depth - links_depth[links_tree_root_idx], -1)

        # Heterogeneous simulation stores the link, joint and dof info per environment, every variant stating its own
        # (see '_dispatch_heterogeneous_kinematics'). We must update options because the getters read from them.
        if self._enable_heterogeneous:
            self._options.batch_links_info = True
            self._options.batch_joints_info = True
            self._options.batch_dofs_info = True

    def _reset_kinematic_grad(self) -> None:
        # Zero every Quadrants-side adjoint buffer the solver owns so a subsequent backward pass does not accumulate on
        # top of stale residues from the previous one. The rigid/kinematic `.grad` pipeline writes through
        # `qd.atomic_add` (e.g. `kernel_get_state_grad`, the `substep_backward` chain), which is the correct
        # reverse-mode accumulation rule for a single backward pass but requires the buffers to start at zero. MPM's
        # `reset_grad` already zeros its fields explicitly; the rigid/kinematic solvers historically relied on users
        # rebuilding the scene per `loss.backward()`, which breaks training loops that call `scene.reset()` between
        # backward passes.
        if self._requires_grad:
            qd_zero_grad(self.dyn_state.links)
            qd_zero_grad(self.dyn_state.dofs)
            qd_zero_grad(self.dyn_state.joints)
            qd_zero_grad(self.rigid_info)
            # One flush for the zeroing batch; see qd_zero_grad in misc.py.
            if gs.use_zerocopy and gs.backend == gs.metal:
                torch.mps.synchronize()
        for entity in self._entities:
            entity.reset_grad()
        self._queried_states.clear()


@register_system(SimulatorEngine)
def create_rigid_dynamics(engine: SimulatorEngine) -> RigidDynamics | None:
    """Admit the Rigid owner only when the authored Scene contains a Rigid entity."""
    return (
        RigidDynamics(engine.scene)
        if any(isinstance(entity, RigidEntity) for entity in engine.scene.entities)
        else None
    )


@qd.kernel
def kernel_substep_wakeup_coupled_links(
    rigid_data: qd.template(),
    island_data: qd.template(),
    rigid_config: qd.template(),
):
    func_substep_wakeup_coupled_links(rigid_data, island_data, rigid_config)


@qd.kernel(fastcache=True)
def kernel_forward_dynamics(
    dyn_state: array_class.DynState,
    island_state: array_class.IslandState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    is_backward: qd.template(),
):
    func_forward_dynamics(dyn_state, island_state, dyn_info, rigid_info, rigid_config, is_backward)


@qd.kernel(fastcache=True)
def kernel_step_1(
    rigid_data: qd.template(),
    island_data: qd.template(),
    rigid_config: qd.template(),
    is_forward_pos_updated: qd.template(),
    is_forward_vel_updated: qd.template(),
    is_backward: qd.template(),
):
    func_step_1(
        rigid_data,
        island_data,
        rigid_config,
        is_forward_pos_updated,
        is_forward_vel_updated,
        is_backward,
    )


@qd.kernel(fastcache=True)
def kernel_step_2(
    rigid_data: qd.template(),
    rigid_config: qd.template(),
    is_backward: qd.template(),
):
    func_step_2(rigid_data, rigid_config, is_backward)
