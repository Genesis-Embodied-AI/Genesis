"""Manager System of the rigid solver, owning its data and exposing stage Actions composed around a host solve.

The initialization of the rigid data extends its kinematic fields through the implementation sections at the end of
this class.
"""

from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
import quadrants as qd
import torch

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
from genesis.constants import link_ref_frame
from genesis.engine.core import Data, System, register_system
from genesis.engine.entities import DroneEntity, RigidEntity
from genesis.engine.solvers.base_solver import kernel_set_gravity
from genesis.engine.solvers.rigid.kinematic_tools import balanced_variant_mapping as _balanced_variant_mapping
from genesis.engine.solvers.rigid.kinematic_tools import fill_base_link_geom_offsets as _fill_base_link_geom_offsets
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
    kernel_get_state,
    kernel_get_state_grad,
    kernel_get_terrain_height,
    kernel_set_dofs_act_bias,
    kernel_set_dofs_act_gain,
    kernel_set_dofs_armature,
    kernel_set_dofs_damping,
    kernel_set_dofs_force_grad,
    kernel_set_dofs_force_range,
    kernel_set_dofs_frictionloss,
    kernel_set_dofs_kp,
    kernel_set_dofs_kv,
    kernel_set_dofs_limit,
    kernel_set_dofs_position,
    kernel_set_dofs_stiffness,
    kernel_set_dofs_velocity,
    kernel_set_dofs_velocity_grad,
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
    kernel_set_links_pos_grad,
    kernel_set_links_quat,
    kernel_set_links_quat_grad,
    kernel_set_qpos,
    kernel_set_sol_params,
    kernel_set_state,
    kernel_set_vverts,
    kernel_update_drone_propeller_vgeoms,
    kernel_wake_up_entities_by_dofs,
    kernel_wake_up_entities_by_links,
    kernel_wake_up_entities_by_qs,
)
from genesis.engine.solvers.rigid.abd.diff import (
    func_integrate_dq_entity,
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
from genesis.engine.solvers.rigid.abd.manual_bw import (
    kernel_manual_compute_qacc_bw,
    kernel_manual_forward_kinematics_bw,
    kernel_manual_forward_velocity_bw,
)
from genesis.engine.solvers.rigid.abd.misc import (
    kernel_apply_links_external_wrench,
    kernel_apply_links_external_wrench_at_pos,
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
    sanitize_ref_frame,
    sanitize_sol_params,
)
from genesis.engine.states import RigidSolverState
from genesis.engine.states.cache import QueriedStates
from genesis.options.solvers import RigidOptions
from genesis.utils.misc import (
    DeprecationError,
    assign_indexed_tensor,
    broadcast_tensor,
    get_gpu_core_count,
    get_gpu_shared_tile_sizes,
    indices_to_mask,
    qd_to_numpy,
    qd_to_torch,
    qd_zero_grad,
    sanitize_indexed_tensor,
    tensor_to_array,
)

from genesis.engine.solvers.rigid.substep import func_step_1, func_step_2, func_substep_wakeup_coupled_links
from ..simulator_engine import SimulatorEngine

if TYPE_CHECKING:
    from genesis.engine.scene import Scene

TensorLike: TypeAlias = ArrayLike | torch.Tensor
IndexLike: TypeAlias = ArrayLike | torch.Tensor | None
HostTensor: TypeAlias = NDArray[np.generic] | torch.Tensor


IS_OLD_TORCH = tuple(map(int, torch.__version__.split(".")[:2])) < (2, 8)


class RigidSolverSystem(System):
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
        kinematics_scratch: array_class.KinematicsScratch
        weight_scratch: array_class.WeightScratch
        ik_scratch: array_class.IKScratch
        rigid_adjoint_cache: array_class.RigidAdjointCache
        dyn_state_adjoint_cache: array_class.DynState | None
        errno: qd.Tensor

    data: Data
    _TERRAIN_HEIGHT_QUERY_TILT_TOLERANCE = 1e-3

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
        from .rigid_island_system import RigidIslandSystem

        self.island_system = self.require(RigidIslandSystem)

    # -------------------------------- Actions --------------------------------

    @System.action(kind="host")
    def collect(self):
        return self.collect_model

    @System.action(kind="host")
    def initialize(self):
        return self.initialize_data

    @System.action(kind="host")
    def finalize(self):
        return self.finalize_data

    @System.action(kind="stage")
    def wake_coupled_links_stage(self):
        return func_substep_wakeup_coupled_links, self.data, self.island_system.data, self.rigid_config

    @System.action(kind="host")
    def wake_coupled_links_kernel(self):
        return kernel_substep_wakeup_coupled_links, self.data, self.island_system.data, self.rigid_config

    @System.action(kind="stage")
    def forward_dynamics_stage(self):
        return func_step_1, self.data, self.island_system.data, self.rigid_config

    @System.action(kind="host")
    def forward_dynamics_kernel(self):
        return kernel_step_1, self.data, self.island_system.data, self.rigid_config

    @System.action(kind="stage")
    def integrate_stage(self):
        return func_step_2, self.data, self.rigid_config

    @System.action(kind="host")
    def integrate_kernel(self):
        return kernel_step_2, self.data, self.rigid_config

    @System.action(kind="host")
    def coupling_set_qpos(self):
        return self.set_coupling_qpos

    @System.action(kind="host")
    def coupling_set_dofs_velocity(self):
        return self.set_coupling_dofs_velocity

    @System.action(kind="host")
    def coupling_apply_wrench(self):
        return self.apply_coupling_wrench

    # -------------------------------- Public get/set API --------------------------------

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
            self.island_system.data.island,
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
            links_idx = self._base_links_idx
        # A relative 'set_pos' adds 'R(authored_quat) @ offset_pos', which reads the current orientation; that
        # dependency is not propagated, so with a non-zero offset position the backward pass is only supported on links
        # without one. 'relative=False' sets the world position directly and is a plain passthrough.
        idx = links_idx if isinstance(links_idx, int) else slice(None)
        if relative and not self._links_offset_pos_is_identity[idx].all():
            gs.raise_exception(
                "Backward pass for 'set_pos' with 'relative=True' is only supported on links without an offset "
                "position (no inertial alignment shifting the link origin). Use 'relative=False' to set the world "
                "position."
            )
        pos_grad_, links_idx, envs_idx = self._sanitize_io_variables(
            pos_grad.unsqueeze(-2), links_idx, self.n_links, "links_idx", envs_idx, (3,), skip_allocation=True
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
            links_idx = self._base_links_idx
        # A relative 'set_quat' composes the orientation offset (a constant right-multiplication) and, when the offset
        # position is non-zero, also rewrites the world position from the input orientation. Neither jacobian is
        # propagated, so the backward pass is only supported when the targeted links carry no pose offset. With
        # 'relative=False' the world orientation is set directly, which is a plain passthrough and always differentiable.
        idx = links_idx if isinstance(links_idx, int) else slice(None)
        if relative and not (
            self._links_offset_quat_is_identity[idx].all() and self._links_offset_pos_is_identity[idx].all()
        ):
            gs.raise_exception(
                "Backward pass for 'set_quat' with 'relative=True' is only supported on links without a pose offset "
                "(no up-axis conversion or inertial alignment). Use 'relative=False' to set the world orientation."
            )
        quat_grad_, links_idx, envs_idx = self._sanitize_io_variables(
            quat_grad.unsqueeze(-2), links_idx, self.n_links, "links_idx", envs_idx, (4,), skip_allocation=True
        )
        if self.n_envs == 0:
            quat_grad_ = quat_grad_.unsqueeze(0)
        kernel_set_links_quat_grad(
            links_idx, envs_idx, quat_grad_, self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config
        )

    def set_dofs_velocity_grad(self, dofs_idx: IndexLike, envs_idx: IndexLike, velocity_grad: torch.Tensor) -> None:
        velocity_grad_, dofs_idx, envs_idx = self._sanitize_io_variables(
            velocity_grad, dofs_idx, self.n_dofs, "dofs_idx", envs_idx, skip_allocation=True
        )
        if self.n_envs == 0:
            velocity_grad_ = velocity_grad_.unsqueeze(0)
        kernel_set_dofs_velocity_grad(dofs_idx, envs_idx, velocity_grad_, self.dyn_state, self.rigid_config)

    def set_dofs_force_grad(self, dofs_idx: IndexLike, envs_idx: IndexLike, force_grad: torch.Tensor) -> None:
        force_grad_, dofs_idx, envs_idx = self._sanitize_io_variables(
            force_grad, dofs_idx, self.n_dofs, "dofs_idx", envs_idx, skip_allocation=True
        )
        if self.n_envs == 0:
            force_grad_ = force_grad_.unsqueeze(0)
        kernel_set_dofs_force_grad(dofs_idx, envs_idx, force_grad_, self.dyn_state, self.rigid_config)

    def get_terrain_height(self, positions: TensorLike, link_idx: int, envs_idx: IndexLike = None) -> torch.Tensor:
        terrain = self._links[link_idx].entity

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
            self.dyn_state,
            self.rigid_config,
            tilt_tolerance=self._TERRAIN_HEIGHT_QUERY_TILT_TOLERANCE,
            is_per_env=is_per_env,
        )

        if self.n_envs == 0:
            heights = heights[0]
        if is_single_position:
            heights = heights[..., 0]
        return heights

    def get_links_quat(
        self, links_idx: IndexLike = None, envs_idx: IndexLike = None, *, relative: bool = False
    ) -> torch.Tensor:
        tensor = qd_to_torch(self.dyn_state.links.quat, envs_idx, links_idx, transpose=True, copy=True)
        idx = links_idx if isinstance(links_idx, int) else slice(None)
        if relative and not self._links_offset_quat_is_identity[idx].all():
            offset_quat = _select_links_offset(self._links_offset_quat, links_idx, envs_idx)
            tensor = gu.transform_quat_by_quat(gu.inv_quat(offset_quat), tensor)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_joints_anchor_pos(self, joints_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        tensor = qd_to_torch(self.dyn_state.joints.xanchor, envs_idx, joints_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_joints_anchor_axis(self, joints_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        tensor = qd_to_torch(self.dyn_state.joints.xaxis, envs_idx, joints_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_vgeoms_pos(
        self, vgeoms_idx: IndexLike = None, envs_idx: IndexLike = None, *, relative: bool = False
    ) -> torch.Tensor:
        tensor = qd_to_torch(self.dyn_state.vgeoms.pos, envs_idx, vgeoms_idx, transpose=True, copy=True)
        if relative and self._vgeoms_offset_pos is not None:
            quat = qd_to_torch(self.dyn_state.vgeoms.quat, envs_idx, vgeoms_idx, transpose=True, copy=True)
            offset_pos = self._vgeoms_offset_pos if vgeoms_idx is None else self._vgeoms_offset_pos[vgeoms_idx]
            offset_quat = self._vgeoms_offset_quat if vgeoms_idx is None else self._vgeoms_offset_quat[vgeoms_idx]
            tensor -= _offset_world_shift(offset_pos, offset_quat, quat)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_vgeoms_quat(
        self, vgeoms_idx: IndexLike = None, envs_idx: IndexLike = None, *, relative: bool = False
    ) -> torch.Tensor:
        tensor = qd_to_torch(self.dyn_state.vgeoms.quat, envs_idx, vgeoms_idx, transpose=True, copy=True)
        if relative and self._vgeoms_offset_quat is not None:
            offset_quat = self._vgeoms_offset_quat if vgeoms_idx is None else self._vgeoms_offset_quat[vgeoms_idx]
            tensor = gu.transform_quat_by_quat(gu.inv_quat(offset_quat), tensor)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_links_ang(self, links_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        tensor = qd_to_torch(self.dyn_state.links.cd_ang, envs_idx, links_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_qpos(self, qs_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        tensor = qd_to_torch(self.qpos, envs_idx, qs_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_dofs_velocity(self, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        tensor = qd_to_torch(self.dyn_state.dofs.vel, envs_idx, dofs_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_dofs_position(self, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        """Read current DOF positions."""
        tensor = qd_to_torch(self.dyn_state.dofs.pos, envs_idx, dofs_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_dofs_limit(
        self, dofs_idx: IndexLike = None, envs_idx: IndexLike = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        tensor = qd_to_torch(self.dyn_info.dofs.limit, envs_idx, dofs_idx, transpose=True, copy=True)
        if self.n_envs == 0 and self._options.batch_dofs_info:
            tensor = tensor[0]
        return tensor[..., 0], tensor[..., 1]

    def update_vgeoms(self) -> None:
        kernel_update_vgeoms(self.dyn_state, self.dyn_info, self.rigid_config)

    def update_vverts_for_vgeoms(self, vgeoms_idx: IndexLike) -> None:
        """Refresh the vverts_state.pos slice for the requested vgeoms by re-running FK.

        Used by set_vverts(None, ...) and at scene-build time to initialize the custom buffer with a meaningful pose.
        The kernel is a no-op for vgeoms whose vverts have no state slot.
        """
        if self.n_custom_vverts == 0:
            return
        kernel_update_vverts_for_vgeoms(vgeoms_idx, self.dyn_state, self.dyn_info, self.rigid_config)

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
        if vverts is None:
            self.update_vverts_for_vgeoms(vgeoms_idx)
            return

        if gs.use_zerocopy:
            data = qd_to_torch(self.dyn_state.vverts.pos, transpose=True, copy=False)
            if isinstance(envs_idx, torch.Tensor) and envs_idx.dtype == torch.bool:
                pos_slice = data[:, custom_vvert_start:custom_vvert_end]
                if vverts.ndim == 3 and len(vverts) != len(pos_slice):
                    pos_slice.masked_scatter_(envs_idx[:, None, None], vverts.view_as(vverts))
                else:
                    vverts_b = broadcast_tensor(vverts, gs.tc_float, pos_slice.shape)
                    torch.where(envs_idx[:, None, None], vverts_b, pos_slice, out=pos_slice)
            else:
                vverts_mask = indices_to_mask(slice(custom_vvert_start, custom_vvert_end))
                pos_mask = (0, *vverts_mask) if self.n_envs == 0 else indices_to_mask(envs_idx, *vverts_mask)
                assign_indexed_tensor(data, pos_mask, vverts)
            return

        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        target_shape = (envs_idx.shape[0], custom_vvert_end - custom_vvert_start, 3)
        vverts = broadcast_tensor(vverts, gs.tc_float, target_shape, ("envs", "vverts", "xyz")).contiguous()
        kernel_set_vverts(custom_vvert_start, envs_idx, vverts, self.dyn_state, self.rigid_config)

    def get_vverts(self, custom_vvert_start: int, custom_vvert_end: int, envs_idx: IndexLike = None) -> torch.Tensor:
        """Return a copy of the vverts_state.pos slice for the given custom-vvert range.

        Shape: (len(envs_idx), custom_vvert_end - custom_vvert_start, 3). envs_idx=None returns every env.
        """
        tensor = qd_to_torch(
            self.dyn_state.vverts.pos, envs_idx, slice(custom_vvert_start, custom_vvert_end), transpose=True, copy=True
        )
        return tensor[0] if self.n_envs == 0 else tensor

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
        jacobian = self.kinematics_scratch.jacobian
        if local_point is None:
            kernel_get_jacobian_zero(
                link_idx, dof_start, jacobian, self.dyn_state, self.dyn_info, self.rigid_config, self._B
            )
        else:
            kernel_get_jacobian(
                link_idx, dof_start, local_point, jacobian, self.dyn_state, self.dyn_info, self.rigid_config, self._B
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
            self.kinematics_scratch.qpos_cache,
            self.dyn_state,
            self.dyn_info,
            self.rigid_info,
            self.rigid_config,
        )

        if self.n_envs == 0:
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
        n_links = len(links_idx)
        n_tgts = self._options.IK_max_targets
        if n_links > n_tgts:
            gs.raise_exception(f"Cannot solve for {n_links} targets. Raise 'IK_max_targets' above {n_tgts}.")

        ik_state, ik_fk, targets = self._get_ik_scratch()

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
            self.dyn_state,
            ik_state,
            ik_fk,
            targets,
            self.dyn_info,
            self.rigid_info,
            self.rigid_config,
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

    def _sanitize_io_variables(
        self,
        tensor: TensorLike,
        inputs_idx: IndexLike,
        input_size: int,
        idx_name: str,
        envs_idx: IndexLike = None,
        element_shape: tuple[int, ...] = (),
        *,
        batched: bool = True,
        skip_allocation: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        envs_idx_ = self._scene._sanitize_envs_idx(envs_idx) if batched else self._scene._envs_idx[:0]

        if self.n_envs == 0 or not batched:
            tensor_, (inputs_idx_,) = sanitize_indexed_tensor(
                tensor,
                gs.tc_float,
                (inputs_idx,),
                (-1, *element_shape),
                (input_size, *element_shape),
                (idx_name, *("" for _ in element_shape)),
                skip_allocation=skip_allocation,
            )
        else:
            tensor_, (envs_idx_, inputs_idx_) = sanitize_indexed_tensor(
                tensor,
                gs.tc_float,
                (envs_idx_, inputs_idx),
                (-1, -1, *element_shape),
                (self.n_envs, input_size, *element_shape),
                ("envs_idx", idx_name, *("" for _ in element_shape)),
                skip_allocation=skip_allocation,
            )

        return tensor_, inputs_idx_, envs_idx_

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
        if self._gravity is None:
            gs.raise_exception(f"{type(self).__name__} holds no body for a gravity to act on.")
        # Indices are not checked on purpose: deciding to raise means reading a comparison back from the device, which
        # syncs, and this may run every step.
        if gs.use_zerocopy:
            envs_mask = indices_to_mask(envs_idx)
            gravity_t = qd_to_torch(self._gravity, transpose=True, copy=False)
            assign_indexed_tensor(
                gravity_t, envs_mask, broadcast_tensor(gravity, gs.tc_float, gravity_t[envs_mask].shape)
            )
            if gs.backend == gs.metal:
                torch.mps.synchronize()
            return

        envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        gravity = broadcast_tensor(gravity, gs.tc_float, (len(envs_idx), 3), ("envs_idx", ""))
        kernel_set_gravity(gravity.contiguous(), envs_idx, self._gravity)

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
        if self._gravity is None:
            gs.raise_exception(f"{type(self).__name__} holds no body for a gravity to act on.")
        tensor = qd_to_torch(self._gravity, envs_idx, transpose=True, copy=True)
        return tensor[0] if self._sim.n_envs == 0 else tensor

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
        if links_idx is None:
            links_idx = self._base_links_idx

        # Without any offset position on the targeted links, the authored and internal link origins coincide, so a
        # relative set is just an absolute one.
        idx = links_idx if isinstance(links_idx, int) else slice(None)
        if relative and self._links_offset_pos_is_identity[idx].all():
            relative = False

        # Map a single base link's authored position to world here (keeping the current orientation) so the zero-copy
        # in-place write below still applies, both for an environment-uniform and a per-environment offset. Multi-link
        # relative sets are composed in the kernel branch instead.
        if relative and isinstance(links_idx, int):
            cur_quat = self.get_links_quat(links_idx, envs_idx, relative=False)[..., 0, :]
            offset_pos = _select_links_offset(self._links_offset_pos, links_idx, envs_idx)[..., 0, :]
            offset_quat = _select_links_offset(self._links_offset_quat, links_idx, envs_idx)[..., 0, :]
            pos = torch.as_tensor(pos, dtype=gs.tc_float, device=gs.device) + _offset_world_shift(
                offset_pos, offset_quat, cur_quat
            )
            relative = False

        # Zero-copy fast path: single base link, non-relative. Write the position buffer in place instead of
        # launching a kernel. The kernel path below handles relative or multi-link updates, as well as waking up
        # hibernated entities, which is required whenever hibernation is enabled.
        if gs.use_zerocopy and not relative and isinstance(links_idx, int) and not self._use_hibernation:
            link = self.links[links_idx]
            if link.is_fixed:
                data = qd_to_torch(self.dyn_state.links.pos, transpose=True, copy=False)
                target = data[:, links_idx]
            else:
                data = qd_to_torch(self.rigid_info.qpos, transpose=True, copy=False)
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
                mask = (0,) if self.n_envs == 0 else indices_to_mask(envs_idx)
                assign_indexed_tensor(target, mask, pos)
            if gs.backend == gs.metal:
                torch.mps.synchronize()
        else:
            pos, links_idx, envs_idx = self._sanitize_io_variables(
                pos, links_idx, self.n_links, "links_idx", envs_idx, (3,), skip_allocation=True
            )
            if self.n_envs == 0:
                pos = pos[None]

            if relative:
                # Compose the body-frame offset onto the authored position, keeping the current orientation, then set
                # the resulting world position absolutely.
                cur_quat = qd_to_torch(self.dyn_state.links.quat, envs_idx, links_idx, transpose=True, copy=True)
                offset_pos = _select_links_offset(self._links_offset_pos, links_idx, envs_idx)
                offset_quat = _select_links_offset(self._links_offset_quat, links_idx, envs_idx)
                pos = pos + _offset_world_shift(offset_pos, offset_quat, cur_quat)
                relative = False

            # Raise exception for fixed links with at least one geom and non-batched fixed vertices, except if setting
            # same location for all envs at once
            set_all_envs = torch.equal(torch.sort(envs_idx).values, self._scene._envs_idx)
            has_fixed_verts = any(
                link.is_fixed and link.geoms and not link.entity._batch_fixed_verts
                for link in (self.links[i_l] for i_l in links_idx)
            )
            if has_fixed_verts and not (set_all_envs and (torch.diff(pos, dim=0).abs() < gs.EPS).all()):
                gs.raise_exception(
                    "Specifying env-specific pos for fixed links with at least one geometry requires setting morph "
                    "option 'batch_fixed_verts=True'."
                )

            # Wake up hibernated entities before setting position (fixed links don't need wake-up)
            if self._use_hibernation and not all(self.links[i_l].is_fixed for i_l in links_idx):
                kernel_wake_up_entities_by_links(
                    links_idx,
                    envs_idx,
                    self.dyn_state,
                    self.island_system.data.island,
                    self.dyn_info,
                    self.rigid_info,
                    self.rigid_config,
                )

            kernel_set_links_pos(
                links_idx, envs_idx, pos, self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config
            )

        if not skip_forward:
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
        if links_idx is None:
            links_idx = self._base_links_idx

        # Without any pose offset on the targeted links, the authored and internal link origins coincide, so a relative
        # set is just an absolute one.
        idx = links_idx if isinstance(links_idx, int) else slice(None)
        if relative and (
            self._links_offset_quat_is_identity[idx].all() and self._links_offset_pos_is_identity[idx].all()
        ):
            relative = False

        # Reused by the int fast path and the kernel branch below.
        relative_pos_passthrough = relative and self._links_offset_pos_is_identity[idx].all()

        # Compose a single base link's authored orientation to world here so the zero-copy in-place write below still
        # applies, both for an environment-uniform and a per-environment offset. This only preserves the authored-frame
        # position when the offset position is identity; a non-zero offset position rotates with the orientation and
        # is handled (together with multi-link relative sets) in the kernel branch instead.
        if isinstance(links_idx, int) and relative_pos_passthrough:
            offset_quat = _select_links_offset(self._links_offset_quat, links_idx, envs_idx)[..., 0, :]
            quat = gu.transform_quat_by_quat(offset_quat, torch.as_tensor(quat, dtype=gs.tc_float, device=gs.device))
            relative = False

        # Zero-copy fast path: single base link, non-relative. Write the quaternion buffer in place instead of
        # launching a kernel. The kernel path below handles relative or multi-link updates, as well as waking up
        # hibernated entities, which is required whenever hibernation is enabled.
        if gs.use_zerocopy and not relative and isinstance(links_idx, int) and not self._use_hibernation:
            link = self.links[links_idx]
            if link.is_fixed:
                data = qd_to_torch(self.dyn_state.links.quat, transpose=True, copy=False)
                target = data[:, links_idx]
            else:
                data = qd_to_torch(self.rigid_info.qpos, transpose=True, copy=False)
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
                mask = (0,) if self.n_envs == 0 else indices_to_mask(envs_idx)
                assign_indexed_tensor(target, mask, quat)
            if gs.backend == gs.metal:
                torch.mps.synchronize()
        else:
            quat, links_idx, envs_idx = self._sanitize_io_variables(
                quat, links_idx, self.n_links, "links_idx", envs_idx, (4,), skip_allocation=True
            )
            if self.n_envs == 0:
                quat = quat[None]

            if relative:
                offset_quat = _select_links_offset(self._links_offset_quat, links_idx, envs_idx)
                if not relative_pos_passthrough:
                    # The offset position rotates with the orientation, so keep the authored-frame position fixed by
                    # rewriting the world position from the current authored position and the new authored orientation.
                    cur_pos = qd_to_torch(self.dyn_state.links.pos, envs_idx, links_idx, transpose=True, copy=True)
                    cur_quat = qd_to_torch(self.dyn_state.links.quat, envs_idx, links_idx, transpose=True, copy=True)
                    offset_pos = _select_links_offset(self._links_offset_pos, links_idx, envs_idx)
                    authored_pos = cur_pos - _offset_world_shift(offset_pos, offset_quat, cur_quat)
                    world_pos = authored_pos + gu.transform_by_quat(offset_pos, quat)
                    kernel_set_links_pos(
                        links_idx,
                        envs_idx,
                        world_pos,
                        self.dyn_state,
                        self.dyn_info,
                        self.rigid_info,
                        self.rigid_config,
                    )
                # Compose the offset onto the authored orientation, then set the resulting world orientation absolutely.
                quat = gu.transform_quat_by_quat(offset_quat, quat)
                relative = False

            set_all_envs = torch.equal(torch.sort(envs_idx).values, self._scene._envs_idx)
            has_fixed_verts = any(
                link.is_fixed and link.geoms and not link.entity._batch_fixed_verts
                for link in (self.links[i_l] for i_l in links_idx)
            )
            if has_fixed_verts and not (set_all_envs and (torch.diff(quat, dim=0).abs() < gs.EPS).all()):
                gs.raise_exception("Impossible to set env-specific quat for fixed links with at least one geometry.")

            # Wake up hibernated entities before setting quaternion (fixed links don't need wake-up)
            if self._use_hibernation and not all(self.links[i_l].is_fixed for i_l in links_idx):
                kernel_wake_up_entities_by_links(
                    links_idx,
                    envs_idx,
                    self.dyn_state,
                    self.island_system.data.island,
                    self.dyn_info,
                    self.rigid_info,
                    self.rigid_config,
                )

            kernel_set_links_quat(
                links_idx, envs_idx, quat, self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config
            )

        if not skip_forward:
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
        if not self._options.batch_links_info and envs_idx is not None:
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
            links_idx_ = range(self.n_links)
        elif not isinstance(links_idx, torch.Tensor) or gs.debug or links_idx.device.type == "cpu":
            (links_idx_,) = indices_to_mask(links_idx, keepdim=False, to_torch=False, boolean_mask=False)
            if isinstance(links_idx_, slice):
                links_idx_ = range(*links_idx_.indices(self.n_links))
            elif isinstance(links_idx_, torch.Tensor):
                links_idx_ = tensor_to_array(links_idx_)
            if np.ndim(links_idx_) == 0:
                links_idx_ = (links_idx_,)
        values, links_idx, envs_idx = self._sanitize_io_variables(
            values,
            links_idx,
            self.n_links,
            "links_idx",
            envs_idx,
            element_shape=shape,
            batched=self._options.batch_links_info,
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
            for link in self.links:
                anchor = link
                while anchor.parent_idx != -1 and all(joint.type == gs.JOINT_TYPE.FIXED for joint in anchor.joints):
                    anchor = self.links[anchor.parent_idx]
                links_anchor.append(anchor if anchor.aligned else None)
            values_idx_by_link = {i_l: i_col for i_col, i_l in enumerate(links_idx_)}
            for i_l in values_idx_by_link:
                anchor = links_anchor[i_l]
                if anchor is None:
                    continue
                links_idx_body = [link.idx for link in self.links if links_anchor[link.idx] is anchor]
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
                link = self.links[i_l]
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

        envs_idx = envs_idx if self._options.batch_links_info else self._scene._envs_idx
        if not self._options.batch_links_info or self.n_envs == 0:
            values = values[None]
        # The weights are solved again rather than scaled by the mass ratio: a weight follows the mass alone for a body
        # whose degrees of freedom carry no armature, which a model loaded from a file has.
        if name == "mass":
            kernel_set_links_mass(
                links_idx,
                envs_idx,
                values,
                self.weight_scratch.jac_row,
                self.weight_scratch.solve_out,
                self.dyn_state,
                self.island_system.data.island,
                self.dyn_info,
                self.rigid_info,
                self.rigid_config,
                scale_inertia,
                self._is_forward_pos_updated,
                self._is_forward_vel_updated,
            )
        elif name == "COM":
            kernel_set_links_COM(
                links_idx,
                envs_idx,
                values,
                self.weight_scratch.jac_row,
                self.weight_scratch.solve_out,
                self.dyn_state,
                self.island_system.data.island,
                self.dyn_info,
                self.rigid_info,
                self.rigid_config,
                self._is_forward_pos_updated,
                self._is_forward_vel_updated,
            )
        else:
            kernel_set_links_inertia(
                links_idx,
                envs_idx,
                values,
                self.weight_scratch.jac_row,
                self.weight_scratch.solve_out,
                self.dyn_state,
                self.island_system.data.island,
                self.dyn_info,
                self.rigid_info,
                self.rigid_config,
                refresh_position=self._is_forward_pos_updated,
            )

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
        # Neither the masses nor the links they are written to are read back for checking: a condition drawn from
        # values on the device forces a synchronization at every call. The mass is spread over the links given, so
        # they have to carry some between them. A selection already on the device is left as it is, for that reason.
        if not isinstance(links_idx, torch.Tensor):
            if links_idx is None:
                links = self.links
            elif isinstance(links_idx, slice):
                links = self.links[links_idx]
            elif isinstance(links_idx, (int, np.integer)):
                links = (self.links[links_idx],)
            else:
                links = [self.links[i_l] for i_l in links_idx]
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
        friction_ratio, geoms_idx, envs_idx = self._sanitize_io_variables(
            friction_ratio, geoms_idx, self.n_geoms, "geoms_idx", envs_idx, skip_allocation=True
        )
        if self.n_envs == 0:
            friction_ratio = friction_ratio[None]
        kernel_set_geoms_friction_ratio(geoms_idx, envs_idx, friction_ratio, self.dyn_state, self.rigid_config)

    def set_qpos(
        self,
        qpos: TensorLike,
        qs_idx: IndexLike = None,
        envs_idx: IndexLike = None,
        *,
        skip_forward: bool = False,
    ) -> None:
        if gs.use_zerocopy:
            data = qd_to_torch(self.rigid_info.qpos, transpose=True, copy=False)
            errno = qd_to_torch(self._errno, copy=False)
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
                mask = (0, *qs_mask) if self.n_envs == 0 else indices_to_mask(envs_idx, *qs_mask, boolean_mask=False)
                assign_indexed_tensor(data, mask, qpos)
                errno[envs_idx] = 0
                if mask and isinstance(mask[0], torch.Tensor):
                    envs_idx = mask[0].reshape((-1,))
            if gs.backend == gs.metal:
                torch.mps.synchronize()
        else:
            qpos, qs_idx, envs_idx = self._sanitize_io_variables(
                qpos, qs_idx, self.n_qs, "qs_idx", envs_idx, skip_allocation=True
            )
            if self.n_envs == 0:
                qpos = qpos[None]

            # Teleporting a sleeping body must revive it, otherwise the new pose is silently dropped. Nothing can be
            # hibernated while the scene is still being built.
            if self._use_hibernation and self.is_built:
                kernel_wake_up_entities_by_qs(
                    qs_idx,
                    envs_idx,
                    self.dyn_state,
                    self.island_system.data.island,
                    self.dyn_info,
                    self.rigid_info,
                    self.rigid_config,
                )

            kernel_set_qpos(qs_idx, envs_idx, qpos, self.rigid_info, self.rigid_config)
            kernel_set_zero(envs_idx, self._errno)

        if not skip_forward:
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
        sol_params_ = broadcast_tensor(sol_params, gs.tc_float, (7,), ("",))
        sol_params_ = sanitize_sol_params(sol_params_.clone(), self._sol_min_timeconst)
        kernel_set_global_sol_params(sol_params_, self.dyn_info, self.rigid_config)

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
        # Make sure that a single constraint type has been selected at once
        if sum(inputs_idx is not None for inputs_idx in (geoms_idx, joints_idx, eqs_idx)) > 1:
            gs.raise_exception("Cannot set more than one constraint type at once.")

        # Select the right input type
        if eqs_idx is not None:
            constraint_type = ConstraintType.EQUALITY
            idx_name = "eqs_idx"
            inputs_idx = eqs_idx
            inputs_length = self.n_equalities
            batched = True
        elif joints_idx is not None:
            constraint_type = ConstraintType.JOINT
            idx_name = "joints_idx"
            inputs_idx = joints_idx
            inputs_length = self.n_joints
            batched = self._options.batch_joints_info
        else:
            constraint_type = ConstraintType.GEOM
            idx_name = "geoms_idx"
            inputs_idx = geoms_idx
            inputs_length = self.n_geoms
            batched = False

        # Sanitize input arguments
        sol_params_, inputs_idx, envs_idx = self._sanitize_io_variables(
            sol_params, inputs_idx, inputs_length, idx_name, envs_idx, (7,), batched=batched, skip_allocation=True
        )
        # Geom values resolve an unspecified time constant to the default, then defer the floor to contact
        # assembly, which floors the mixed pair value (see func_set_contact_data), mirroring the build-time
        # sanitization; joint and equality values are consumed unmixed, so they are floored here.
        sol_params_ = sanitize_sol_params(
            sol_params_.clone(),
            self._sol_min_timeconst,
            self._sol_default_timeconst if constraint_type == ConstraintType.GEOM else None,
            floor_timeconst=constraint_type != ConstraintType.GEOM,
        )
        if self.n_envs == 0 and batched:
            sol_params_ = sol_params_[None]

        kernel_set_sol_params(inputs_idx, envs_idx, sol_params_, self.dyn_info, self.rigid_config, int(constraint_type))

    def _set_dofs_info(
        self,
        tensor_list: Sequence[TensorLike],
        dofs_idx: IndexLike,
        name: str,
        envs_idx: IndexLike = None,
    ) -> None:
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
            mask = indices_to_mask(*((envs_idx, dofs_idx) if self._options.batch_dofs_info else (dofs_idx,)))
            if name == "kp":
                # kp sets act_gain, act_bias[0] = 0, act_bias[1] = -kp (full PD reset)
                kp = torch.as_tensor(tensor_list[0], dtype=gs.tc_float, device=gs.device)
                gain = qd_to_torch(self.dyn_info.dofs.act_gain, transpose=True, copy=False)
                assign_indexed_tensor(gain, mask, kp)
                bias = qd_to_torch(self.dyn_info.dofs.act_bias, transpose=True, copy=False)
                bias[(*mask, ..., 0)] = 0.0
                assign_indexed_tensor(bias, (*mask, ..., 1), -kp)
            elif name == "kv":
                # kv sets act_bias[..., 2] = -kv
                kv = torch.as_tensor(tensor_list[0], dtype=gs.tc_float, device=gs.device)
                bias = qd_to_torch(self.dyn_info.dofs.act_bias, transpose=True, copy=False)
                assign_indexed_tensor(bias, (*mask, ..., 2), -kv)
            else:
                data = qd_to_torch(getattr(self.dyn_info.dofs, name), transpose=True, copy=False)
                num_values = len(tensor_list)
                for j, mask_j in enumerate(((*mask, ..., j) for j in range(num_values)) if num_values > 1 else (mask,)):
                    assign_indexed_tensor(data, mask_j, tensor_list[j])
            if gs.backend == gs.metal:
                torch.mps.synchronize()
            return

        tensor_list = list(tensor_list)
        for j, tensor in enumerate(tensor_list):
            tensor, dofs_idx, envs_idx_ = self._sanitize_io_variables(
                tensor,
                dofs_idx,
                self.n_dofs,
                "dofs_idx",
                envs_idx,
                batched=self._options.batch_dofs_info,
                skip_allocation=True,
            )
            if self.n_envs == 0 and self._options.batch_dofs_info:
                tensor = tensor[None]
            tensor_list[j] = tensor
        if name == "kp":
            kernel_set_dofs_kp(dofs_idx, envs_idx_, *tensor_list, self.dyn_info, self.rigid_config)
        elif name == "kv":
            kernel_set_dofs_kv(dofs_idx, envs_idx_, *tensor_list, self.dyn_info, self.rigid_config)
        elif name == "force_range":
            kernel_set_dofs_force_range(dofs_idx, envs_idx_, *tensor_list, self.dyn_info, self.rigid_config)
        elif name == "stiffness":
            kernel_set_dofs_stiffness(dofs_idx, envs_idx_, *tensor_list, self.dyn_info, self.rigid_config)
        elif name == "armature":
            kernel_set_dofs_armature(dofs_idx, envs_idx_, *tensor_list, self.dyn_info, self.rigid_config)
            self.refresh_invweight_and_meaninertia(envs_idx=envs_idx, force_update=True)
        elif name == "damping":
            kernel_set_dofs_damping(dofs_idx, envs_idx_, *tensor_list, self.dyn_info, self.rigid_config)
        elif name == "frictionloss":
            kernel_set_dofs_frictionloss(dofs_idx, envs_idx_, *tensor_list, self.dyn_info, self.rigid_config)
        elif name == "limit":
            kernel_set_dofs_limit(dofs_idx, envs_idx_, *tensor_list, self.dyn_info, self.rigid_config)
        elif name == "act_gain":
            kernel_set_dofs_act_gain(dofs_idx, envs_idx_, *tensor_list, self.dyn_info, self.rigid_config)
        elif name == "act_bias":
            kernel_set_dofs_act_bias(dofs_idx, envs_idx_, *tensor_list, self.dyn_info, self.rigid_config)
        else:
            gs.raise_exception(f"Invalid `name` {name}.")

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
        position, dofs_idx, envs_idx = self._sanitize_io_variables(
            position, dofs_idx, self.n_dofs, "dofs_idx", envs_idx, skip_allocation=True
        )
        if self.n_envs == 0:
            position = position[None]

        self._wake_dofs(dofs_idx, envs_idx)

        kernel_set_dofs_position(
            dofs_idx, envs_idx, position, self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config
        )

        if gs.use_zerocopy:
            errno = qd_to_torch(self._errno, copy=False)
            errno[envs_idx] = 0
            if gs.backend == gs.metal:
                torch.mps.synchronize()
        else:
            kernel_set_zero(envs_idx, self._errno)

        kernel_forward_kinematics_links_geoms(
            envs_idx, self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config
        )
        self._is_forward_pos_updated = True
        self._is_forward_vel_updated = True

    def _wake_dofs(self, dofs_idx: torch.Tensor, envs_idx: torch.Tensor) -> None:
        # Revive any hibernated entity owning these (already sanitized) dofs before a state is written to them;
        # forward dynamics and integration act only on awake dofs, so a state written to a sleeping body would
        # otherwise be silently dropped until it is woken by some other means.
        if self._use_hibernation:
            kernel_wake_up_entities_by_dofs(
                dofs_idx,
                envs_idx,
                self.dyn_state,
                self.island_system.data.island,
                self.dyn_info,
                self.rigid_info,
                self.rigid_config,
            )

    def set_dofs_velocity(
        self,
        velocity: TensorLike,
        dofs_idx: IndexLike = None,
        envs_idx: IndexLike = None,
        *,
        skip_forward: bool = False,
    ) -> None:
        # Wake the owning entities before delegating to the base setter, which re-sanitizes and applies the write.
        if self._use_hibernation:
            _, wake_dofs_idx, wake_envs_idx = self._sanitize_io_variables(
                velocity, dofs_idx, self.n_dofs, "dofs_idx", envs_idx, skip_allocation=True
            )
            self._wake_dofs(wake_dofs_idx, wake_envs_idx)
        self._set_kinematic_dofs_velocity(velocity, dofs_idx, envs_idx, skip_forward=skip_forward)

    def control_dofs_force(self, force: TensorLike, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> None:
        # A control target is consumed by the actuation pass of the next step, which wakes a sleeping link it actuates
        # (see func_torque_and_passive_force), so the target is written without waking anything here.
        if gs.use_zerocopy:
            mask = (0, *indices_to_mask(dofs_idx)) if self.n_envs == 0 else indices_to_mask(envs_idx, dofs_idx)
            ctrl_mode = qd_to_torch(self.dyn_state.dofs.ctrl_mode, transpose=True, copy=False)
            ctrl_mode[mask] = gs.CTRL_MODE.FORCE
            ctrl_force = qd_to_torch(self.dyn_state.dofs.ctrl_force, transpose=True, copy=False)
            assign_indexed_tensor(ctrl_force, mask, force)
            if gs.backend == gs.metal:
                torch.mps.synchronize()
            return

        force, dofs_idx, envs_idx = self._sanitize_io_variables(
            force, dofs_idx, self.n_dofs, "dofs_idx", envs_idx, skip_allocation=True
        )
        if self.n_envs == 0:
            force = force[None]

        kernel_control_dofs_force(dofs_idx, envs_idx, force, self.dyn_state, self.rigid_config)

    def control_dofs_velocity(
        self, velocity: TensorLike, dofs_idx: IndexLike = None, envs_idx: IndexLike = None
    ) -> None:
        if gs.use_zerocopy:
            mask = (0, *indices_to_mask(dofs_idx)) if self.n_envs == 0 else indices_to_mask(envs_idx, dofs_idx)
            ctrl_mode = qd_to_torch(self.dyn_state.dofs.ctrl_mode, transpose=True, copy=False)
            ctrl_mode[mask] = gs.CTRL_MODE.VELOCITY
            ctrl_pos = qd_to_torch(self.dyn_state.dofs.ctrl_pos, transpose=True, copy=False)
            ctrl_pos[mask] = 0.0
            ctrl_vel = qd_to_torch(self.dyn_state.dofs.ctrl_vel, transpose=True, copy=False)
            assign_indexed_tensor(ctrl_vel, mask, velocity)
            if gs.backend == gs.metal:
                torch.mps.synchronize()
            return

        velocity, dofs_idx, envs_idx = self._sanitize_io_variables(
            velocity, dofs_idx, self.n_dofs, "dofs_idx", envs_idx, skip_allocation=True
        )
        if self.n_envs == 0:
            velocity = velocity[None]

        kernel_control_dofs_velocity(dofs_idx, envs_idx, velocity, self.dyn_state, self.rigid_config)

    def control_dofs_position(
        self, position: TensorLike, dofs_idx: IndexLike = None, envs_idx: IndexLike = None
    ) -> None:
        if gs.use_zerocopy:
            mask = (0, *indices_to_mask(dofs_idx)) if self.n_envs == 0 else indices_to_mask(envs_idx, dofs_idx)
            ctrl_mode = qd_to_torch(self.dyn_state.dofs.ctrl_mode, transpose=True, copy=False)
            ctrl_mode[mask] = gs.CTRL_MODE.POSITION
            ctrl_pos = qd_to_torch(self.dyn_state.dofs.ctrl_pos, transpose=True, copy=False)
            assign_indexed_tensor(ctrl_pos, mask, position)
            ctrl_vel = qd_to_torch(self.dyn_state.dofs.ctrl_vel, transpose=True, copy=False)
            ctrl_vel[mask] = 0.0
            if gs.backend == gs.metal:
                torch.mps.synchronize()
            return

        position, dofs_idx, envs_idx = self._sanitize_io_variables(
            position, dofs_idx, self.n_dofs, "dofs_idx", envs_idx, skip_allocation=True
        )
        if self.n_envs == 0:
            position = position[None]

        kernel_control_dofs_position(dofs_idx, envs_idx, position, self.dyn_state, self.rigid_config)

    def control_dofs_position_velocity(
        self,
        position: TensorLike,
        velocity: TensorLike,
        dofs_idx: IndexLike = None,
        envs_idx: IndexLike = None,
    ) -> None:
        if gs.use_zerocopy:
            mask = (0, *indices_to_mask(dofs_idx)) if self.n_envs == 0 else indices_to_mask(envs_idx, dofs_idx)
            ctrl_mode = qd_to_torch(self.dyn_state.dofs.ctrl_mode, transpose=True, copy=False)
            ctrl_mode[mask] = gs.CTRL_MODE.POSITION
            ctrl_pos = qd_to_torch(self.dyn_state.dofs.ctrl_pos, transpose=True, copy=False)
            assign_indexed_tensor(ctrl_pos, mask, position)
            ctrl_vel = qd_to_torch(self.dyn_state.dofs.ctrl_vel, transpose=True, copy=False)
            assign_indexed_tensor(ctrl_vel, mask, velocity)
            if gs.backend == gs.metal:
                torch.mps.synchronize()
            return

        position, dofs_idx, _ = self._sanitize_io_variables(
            position, dofs_idx, self.n_dofs, "dofs_idx", envs_idx, skip_allocation=True
        )
        velocity, dofs_idx, envs_idx = self._sanitize_io_variables(
            velocity, dofs_idx, self.n_dofs, "dofs_idx", envs_idx, skip_allocation=True
        )
        if self.n_envs == 0:
            position = position[None]
            velocity = velocity[None]

        kernel_control_dofs_position_velocity(dofs_idx, envs_idx, position, velocity, self.dyn_state, self.rigid_config)

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
        if eqs_idx is not None:
            # Always batched
            tensor = qd_to_torch(self.dyn_info.equalities.sol_params, envs_idx, eqs_idx, transpose=True, copy=True)
            if self.n_envs == 0:
                tensor = tensor[0]
        elif joints_idx is not None:
            # Conditionally batched
            assert envs_idx is None
            # batch_shape = (envs_idx, joints_idx) if self._options.batch_joints_info else (joints_idx,) tensor =
            # qd_to_torch(self.dyn_info.joints.sol_params, *batch_shape, transpose=True)
            tensor = qd_to_torch(self.dyn_info.joints.sol_params, envs_idx, joints_idx, transpose=True, copy=True)
            if self.n_envs == 0 and self._options.batch_joints_info:
                tensor = tensor[0]
        else:  # geoms_idx is not None
            # Never batched
            assert envs_idx is None
            tensor = qd_to_torch(self.dyn_info.geoms.sol_params, geoms_idx, transpose=True, copy=True)
        return tensor

    def get_links_pos(
        self,
        links_idx: IndexLike = None,
        envs_idx: IndexLike = None,
        *,
        ref: link_ref_frame = link_ref_frame.link_origin,
        relative: bool = False,
    ) -> torch.Tensor:
        idx = links_idx if isinstance(links_idx, int) else slice(None)
        if not gs.use_zerocopy:
            _, links_idx, envs_idx = self._sanitize_io_variables(
                None, links_idx, self.n_links, "links_idx", envs_idx, (3,), skip_allocation=True
            )

        ref_frame = sanitize_ref_frame(ref)
        if ref_frame == link_ref_frame.root_COM:
            tensor = qd_to_torch(self.dyn_state.links.root_COM, envs_idx, links_idx, transpose=True, copy=True)
        elif ref_frame == link_ref_frame.link_COM:
            i_pos = qd_to_torch(self.dyn_state.links.i_pos, envs_idx, links_idx, transpose=True)
            root_COM = qd_to_torch(self.dyn_state.links.root_COM, envs_idx, links_idx, transpose=True)
            tensor = i_pos + root_COM
        else:
            tensor = qd_to_torch(self.dyn_state.links.pos, envs_idx, links_idx, transpose=True, copy=True)

        # The pose offset is defined on the link origin, so it is only stripped for the 'link_origin' reference.
        if relative and ref_frame == link_ref_frame.link_origin and not self._links_offset_pos_is_identity[idx].all():
            quat = qd_to_torch(self.dyn_state.links.quat, envs_idx, links_idx, transpose=True)
            offset_pos = _select_links_offset(self._links_offset_pos, links_idx, envs_idx)
            offset_quat = _select_links_offset(self._links_offset_quat, links_idx, envs_idx)
            tensor -= _offset_world_shift(offset_pos, offset_quat, quat)

        return tensor[0] if self.n_envs == 0 else tensor

    def get_links_vel(
        self,
        links_idx: IndexLike = None,
        envs_idx: IndexLike = None,
        *,
        ref: link_ref_frame = link_ref_frame.link_origin,
        relative: bool = False,
    ) -> torch.Tensor:
        ref_frame = sanitize_ref_frame(ref, has_root_com=False)
        idx = links_idx if isinstance(links_idx, int) else slice(None)
        # Only the 'link_origin' reference carries the offset, as in 'get_links_pos'.
        is_relative = (
            relative and ref_frame == link_ref_frame.link_origin and not self._links_offset_pos_is_identity[idx].all()
        )
        if gs.use_zerocopy:
            mask = indices_to_mask(envs_idx, links_idx)
            cd_vel = qd_to_torch(self.dyn_state.links.cd_vel, transpose=True)
            cd_ang = qd_to_torch(self.dyn_state.links.cd_ang, transpose=True)
            if ref_frame == link_ref_frame.link_COM:
                i_pos = qd_to_torch(self.dyn_state.links.i_pos, transpose=True)
                delta = i_pos[mask]
            else:
                pos = qd_to_torch(self.dyn_state.links.pos, transpose=True)
                root_COM = qd_to_torch(self.dyn_state.links.root_COM, transpose=True)
                delta = pos[mask] - root_COM[mask]
                if is_relative:
                    quat = qd_to_torch(self.dyn_state.links.quat, envs_idx, links_idx, transpose=True)
                    offset_pos = _select_links_offset(self._links_offset_pos, links_idx, envs_idx)
                    offset_quat = _select_links_offset(self._links_offset_quat, links_idx, envs_idx)
                    delta = delta - _offset_world_shift(offset_pos, offset_quat, quat)
            tensor = cd_vel[mask] + cd_ang[mask].cross(delta, dim=-1)
            return tensor[0] if self.n_envs == 0 else tensor

        _tensor, links_idx, envs_idx = self._sanitize_io_variables(
            None, links_idx, self.n_links, "links_idx", envs_idx, (3,)
        )
        tensor = _tensor[None] if self.n_envs == 0 else _tensor
        kernel_get_links_vel(
            links_idx,
            envs_idx,
            tensor,
            self._links_offset_pos,
            self._links_offset_quat,
            self.dyn_state,
            self.rigid_config,
            ref_frame,
            is_relative=is_relative,
        )
        return _tensor

    def get_links_acc(
        self, links_idx: IndexLike = None, envs_idx: IndexLike = None, *, relative: bool = False
    ) -> torch.Tensor:
        idx = links_idx if isinstance(links_idx, int) else slice(None)
        _tensor, links_idx, envs_idx = self._sanitize_io_variables(
            None, links_idx, self.n_links, "links_idx", envs_idx, (3,)
        )
        tensor = _tensor[None] if self.n_envs == 0 else _tensor
        kernel_get_links_acc(
            links_idx,
            envs_idx,
            tensor,
            self._links_offset_pos,
            self._links_offset_quat,
            self.dyn_state,
            self.rigid_config,
            is_relative=relative and not self._links_offset_pos_is_identity[idx].all(),
        )
        return _tensor

    def get_links_acc_ang(self, links_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        tensor = qd_to_torch(self.dyn_state.links.cacc_ang, envs_idx, links_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_links_root_COM(self, links_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        """
        Returns the center of mass (COM) of the entire kinematic tree to which the specified links belong.

        This corresponds to the global COM of each entity, assuming a single-rooted structure - that is, as long as no
        two successive links are connected by a free-floating joint (ie a joint that allows all 6 degrees of freedom).
        """
        tensor = qd_to_torch(self.dyn_state.links.root_COM, envs_idx, links_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_links_COM(self, links_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        """The center of mass (COM) of each link, as an offset from the origin of its local frame."""
        if not self._options.batch_links_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched links info.")
        tensor = qd_to_torch(self.dyn_info.links.inertial_pos, envs_idx, links_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 and self._options.batch_links_info else tensor

    def is_links_info_batched(self) -> bool:
        """Whether the inertial properties of a link are stored per environment rather than shared by the batch."""
        return self._options.batch_links_info

    def get_links_mass(self, links_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        """The mass of each link, as the solver currently uses it."""
        if not self._options.batch_links_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched links info.")
        tensor = qd_to_torch(self.dyn_info.links.inertial_mass, envs_idx, links_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 and self._options.batch_links_info else tensor

    def get_links_inertia(self, links_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        """The inertia matrix of each link, in its inertial frame, as the solver currently uses it."""
        if not self._options.batch_links_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched links info.")
        tensor = qd_to_torch(self.dyn_info.links.inertial_i, envs_idx, links_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 and self._options.batch_links_info else tensor

    def get_links_invweight(self, links_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        if not self._options.batch_links_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched links info.")
        tensor = qd_to_torch(self.dyn_info.links.invweight, envs_idx, links_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 and self._options.batch_links_info else tensor

    def get_geoms_friction_ratio(self, geoms_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        tensor = qd_to_torch(self.dyn_state.geoms.friction_ratio, envs_idx, geoms_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_geoms_pos(
        self, geoms_idx: IndexLike = None, envs_idx: IndexLike = None, *, relative: bool = False
    ) -> torch.Tensor:
        tensor = qd_to_torch(self.dyn_state.geoms.pos, envs_idx, geoms_idx, transpose=True, copy=True)
        if relative and self._geoms_offset_pos is not None:
            quat = qd_to_torch(self.dyn_state.geoms.quat, envs_idx, geoms_idx, transpose=True, copy=True)
            offset_pos = self._geoms_offset_pos if geoms_idx is None else self._geoms_offset_pos[geoms_idx]
            offset_quat = self._geoms_offset_quat if geoms_idx is None else self._geoms_offset_quat[geoms_idx]
            tensor -= _offset_world_shift(offset_pos, offset_quat, quat)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_geoms_quat(
        self, geoms_idx: IndexLike = None, envs_idx: IndexLike = None, *, relative: bool = False
    ) -> torch.Tensor:
        tensor = qd_to_torch(self.dyn_state.geoms.quat, envs_idx, geoms_idx, transpose=True, copy=True)
        if relative and self._geoms_offset_quat is not None:
            offset_quat = self._geoms_offset_quat if geoms_idx is None else self._geoms_offset_quat[geoms_idx]
            tensor = gu.transform_quat_by_quat(gu.inv_quat(offset_quat), tensor)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_dofs_control_force(self, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        _tensor, dofs_idx, envs_idx = self._sanitize_io_variables(None, dofs_idx, self.n_dofs, "dofs_idx", envs_idx)
        tensor = _tensor[None] if self.n_envs == 0 else _tensor
        kernel_get_dofs_control_force(dofs_idx, envs_idx, tensor, self.dyn_state, self.dyn_info, self.rigid_config)
        return _tensor

    def get_dofs_force(self, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        tensor = qd_to_torch(self.dyn_state.dofs.force, envs_idx, dofs_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_dofs_acc(self, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        tensor = qd_to_torch(self.dyn_state.dofs.acc, envs_idx, dofs_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_dofs_kp(self, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        if not self._options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        gain = qd_to_torch(self.dyn_info.dofs.act_gain, envs_idx, dofs_idx, transpose=True, copy=True)
        bias = qd_to_torch(self.dyn_info.dofs.act_bias, envs_idx, dofs_idx, transpose=True, copy=True)
        if self.n_envs == 0 and self._options.batch_dofs_info:
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
        if not self._options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        gain = qd_to_torch(self.dyn_info.dofs.act_gain, envs_idx, dofs_idx, transpose=True, copy=True)
        bias = qd_to_torch(self.dyn_info.dofs.act_bias, envs_idx, dofs_idx, transpose=True, copy=True)
        if self.n_envs == 0 and self._options.batch_dofs_info:
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
        if not self._options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        tensor = qd_to_torch(self.dyn_info.dofs.act_gain, envs_idx, dofs_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 and self._options.batch_dofs_info else tensor

    def get_dofs_act_bias(
        self, dofs_idx: IndexLike = None, envs_idx: IndexLike = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self._options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        tensor = qd_to_torch(self.dyn_info.dofs.act_bias, envs_idx, dofs_idx, transpose=True, copy=True)
        if self.n_envs == 0 and self._options.batch_dofs_info:
            tensor = tensor[0]
        return tensor[..., 0], tensor[..., 1], tensor[..., 2]

    def get_dofs_force_range(
        self, dofs_idx: IndexLike = None, envs_idx: IndexLike = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        tensor = qd_to_torch(self.dyn_info.dofs.force_range, envs_idx, dofs_idx, transpose=True, copy=True)
        if self.n_envs == 0 and self._options.batch_dofs_info:
            tensor = tensor[0]
        return tensor[..., 0], tensor[..., 1]

    def get_dofs_stiffness(self, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        if not self._options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        tensor = qd_to_torch(self.dyn_info.dofs.stiffness, envs_idx, dofs_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 and self._options.batch_dofs_info else tensor

    def get_dofs_invweight(self, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        if not self._options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        tensor = qd_to_torch(self.dyn_info.dofs.invweight, envs_idx, dofs_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 and self._options.batch_dofs_info else tensor

    def get_dofs_armature(self, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        if not self._options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        tensor = qd_to_torch(self.dyn_info.dofs.armature, envs_idx, dofs_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 and self._options.batch_dofs_info else tensor

    def get_dofs_damping(self, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        if not self._options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        tensor = qd_to_torch(self.dyn_info.dofs.damping, envs_idx, dofs_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 and self._options.batch_dofs_info else tensor

    def get_dofs_frictionloss(self, dofs_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        if not self._options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        tensor = qd_to_torch(self.dyn_info.dofs.frictionloss, envs_idx, dofs_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 and self._options.batch_dofs_info else tensor

    def get_mass_mat(
        self,
        dofs_idx: IndexLike = None,
        envs_idx: IndexLike = None,
        decompose: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        tensor = qd_to_torch(self.mass_mat_L if decompose else self.mass_mat, envs_idx, transpose=True, copy=True)
        if dofs_idx is not None:
            tensor = tensor[indices_to_mask(None, dofs_idx, dofs_idx)]
        if self.n_envs == 0:
            tensor = tensor[0]

        if decompose:
            mass_mat_D_inv = qd_to_torch(self.rigid_info.mass_mat_D_inv, envs_idx, dofs_idx, transpose=True, copy=True)
            if self.n_envs == 0:
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
        # Spatial velocity and inertia are both referenced to the tree COM, so the quadratic form expands as
        # 0.5 * m * |v|^2 + v . (w x (m * c)) + 0.5 * w . (I * w), with `cinr_pos` holding the first moment m * c.
        cd_ang = qd_to_torch(self.dyn_state.links.cd_ang, envs_idx, links_idx, transpose=True)
        cd_vel = qd_to_torch(self.dyn_state.links.cd_vel, envs_idx, links_idx, transpose=True)
        cinr_inertial = qd_to_torch(self.dyn_state.links.cinr_inertial, envs_idx, links_idx, transpose=True)
        cinr_pos = qd_to_torch(self.dyn_state.links.cinr_pos, envs_idx, links_idx, transpose=True)
        cinr_mass = qd_to_torch(self.dyn_state.links.cinr_mass, envs_idx, links_idx, transpose=True)
        translational = cinr_mass * torch.sum(cd_vel * cd_vel, dim=-1)
        coupling = torch.sum(cd_vel * torch.cross(cd_ang, cinr_pos, dim=-1), dim=-1)
        rotational = torch.sum(cd_ang * torch.matmul(cinr_inertial, cd_ang.unsqueeze(-1)).squeeze(-1), dim=-1)
        kinetic_energy = torch.sum(0.5 * (translational + rotational) + coupling, dim=-1)

        dofs_vel = qd_to_torch(self.dyn_state.dofs.vel, envs_idx, dofs_idx, transpose=True)
        armature = self.get_dofs_armature(dofs_idx, envs_idx if self._options.batch_dofs_info else None)
        kinetic_energy += 0.5 * torch.sum(armature * dofs_vel * dofs_vel, dim=-1)

        return kinetic_energy[0] if self.n_envs == 0 else kinetic_energy

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
        gravity = self.get_gravity(envs_idx=envs_idx)  # (3,) or (n_envs, 3)
        links_pos = self.get_links_pos(links_idx, envs_idx, ref=link_ref_frame.link_COM)  # (..., n_links, 3)
        links_envs_idx = envs_idx if self._options.batch_links_info else None
        links_mass = qd_to_torch(self.dyn_info.links.inertial_mass, links_envs_idx, links_idx, transpose=True)
        links_is_fixed = qd_to_torch(self.dyn_info.links.is_fixed, links_envs_idx, links_idx, transpose=True)
        links_mass = links_mass.masked_fill(links_is_fixed.to(torch.bool), 0.0)
        if self.n_envs == 0 and self._options.batch_links_info:
            links_mass = links_mass[0]

        # PE_i = m_i * g^T * p_i => PE = sum_i(m_i * (g . p_i))
        # g is (..., 3), links_pos is (..., n_links, 3) -> broadcast g to (..., 1, 3)
        g_dot_p = torch.sum(gravity.unsqueeze(-2) * links_pos, dim=-1)  # (..., n_links)
        potential_energy = -torch.sum(links_mass * g_dot_p, dim=-1)

        # `dofs.pos` holds the deflection `qpos - qpos0` that the spring force is proportional to, so the elastic
        # energy integrates directly against it whatever the neutral configuration.
        dofs_pos = qd_to_torch(self.dyn_state.dofs.pos, envs_idx, dofs_idx, transpose=True)
        stiffness = self.get_dofs_stiffness(dofs_idx, envs_idx if self._options.batch_dofs_info else None)
        spring_energy = 0.5 * torch.sum(stiffness * dofs_pos * dofs_pos, dim=-1)
        if self.n_envs == 0:
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
        return qd_to_torch(self.dyn_info.geoms.friction, geoms_idx, copy=True)

    def get_geoms_friction_torsional(self, geoms_idx: IndexLike = None) -> torch.Tensor:
        return qd_to_torch(self.dyn_info.geoms.friction_torsional, geoms_idx, copy=True)

    def get_geoms_friction_rolling(self, geoms_idx: IndexLike = None) -> torch.Tensor:
        return qd_to_torch(self.dyn_info.geoms.friction_rolling, geoms_idx, copy=True)

    def get_AABB(self, entities_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        aabb_min = qd_to_torch(self.dyn_state.geoms.aabb_min, envs_idx, transpose=True)
        aabb_max = qd_to_torch(self.dyn_state.geoms.aabb_max, envs_idx, transpose=True)

        aabb = torch.stack([aabb_min, aabb_max], dim=-2)

        if entities_idx is not None:
            entity_geom_starts = []
            entity_geom_ends = []
            for entity_idx in entities_idx:
                entity = self._entities[entity_idx]
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

        return aabb[0] if self.n_envs == 0 else aabb

    def set_geom_friction(self, friction: TensorLike, geoms_idx: IndexLike) -> None:
        kernel_set_geom_friction(geoms_idx, friction, self.dyn_info)

    def set_geom_friction_torsional(self, friction_torsional: TensorLike, geoms_idx: IndexLike) -> None:
        kernel_set_geom_friction_torsional(geoms_idx, friction_torsional, self.dyn_info)

    def set_geom_friction_rolling(self, friction_rolling: TensorLike, geoms_idx: IndexLike) -> None:
        kernel_set_geom_friction_rolling(geoms_idx, friction_rolling, self.dyn_info)

    def set_geoms_friction(self, friction: TensorLike, geoms_idx: IndexLike = None) -> None:
        friction, geoms_idx, _ = self._sanitize_io_variables(
            friction, geoms_idx, self.n_geoms, "geoms_idx", envs_idx=None, batched=False, skip_allocation=True
        )
        kernel_set_geoms_friction(geoms_idx, friction, self.dyn_info, self.rigid_config)

    def set_geoms_friction_torsional(self, friction_torsional: TensorLike, geoms_idx: IndexLike = None) -> None:
        friction_torsional, geoms_idx, _ = self._sanitize_io_variables(
            friction_torsional, geoms_idx, self.n_geoms, "geoms_idx", envs_idx=None, batched=False, skip_allocation=True
        )
        kernel_set_geoms_friction_torsional(geoms_idx, friction_torsional, self.dyn_info, self.rigid_config)

    def set_geoms_friction_rolling(self, friction_rolling: TensorLike, geoms_idx: IndexLike = None) -> None:
        friction_rolling, geoms_idx, _ = self._sanitize_io_variables(
            friction_rolling, geoms_idx, self.n_geoms, "geoms_idx", envs_idx=None, batched=False, skip_allocation=True
        )
        kernel_set_geoms_friction_rolling(geoms_idx, friction_rolling, self.dyn_info, self.rigid_config)

    def clear_external_force(self) -> None:
        if gs.use_zerocopy:
            for tensor in (self.dyn_state.links.cfrc_applied_ang, self.dyn_state.links.cfrc_applied_vel):
                out = qd_to_torch(tensor, copy=False)
                out.zero_()
            if gs.backend == gs.metal:
                torch.mps.synchronize()
            return

        kernel_clear_external_force(self.dyn_state, self.rigid_info, self.rigid_config)

    def update_drone_propeller_vgeoms(
        self,
        propellers_vgeom_idxs: TensorLike,
        propellers_revs: TensorLike,
        propellers_spin: TensorLike,
    ) -> None:
        kernel_update_drone_propeller_vgeoms(
            propellers_vgeom_idxs, propellers_revs, propellers_spin, self.dyn_state, self.rigid_info, self.rigid_config
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
        kernel_set_drone_rpm(
            propellers_link_idx, kf, km, propellers_rpm, propellers_spin, self.dyn_state, self.rigid_config, invert
        )

    def update_verts_for_geoms(self, geoms_idx: IndexLike) -> None:
        _, geoms_idx, _ = self._sanitize_io_variables(
            None, geoms_idx, self.n_geoms, "geoms_idx", envs_idx=None, skip_allocation=True
        )
        kernel_update_verts_for_geoms(geoms_idx, self.dyn_state, self.dyn_info, self.rigid_config)

    def get_fixed_collision_verts(self, verts_idx: IndexLike) -> torch.Tensor:
        """Read fixed collision vertices from the Rigid-owned state."""
        return qd_to_torch(self.dyn_state.fixed_verts.pos, verts_idx, copy=True)

    def get_free_collision_verts(self, verts_idx: IndexLike) -> torch.Tensor:
        """Read per-environment collision vertices from the Rigid-owned state."""
        tensor = qd_to_torch(self.dyn_state.free_verts.pos, None, verts_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_vgeom_transforms_batched(self, envs_idx: IndexLike = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Read visual-geometry transforms while preserving the physical batch dimension."""
        self.update_vgeoms()
        pos = qd_to_torch(self.dyn_state.vgeoms.pos, envs_idx, transpose=True, copy=True)
        quat = qd_to_torch(self.dyn_state.vgeoms.quat, envs_idx, transpose=True, copy=True)
        return pos, quat

    def get_error_envs_mask(self) -> torch.Tensor:
        return qd_to_torch(self._errno) > 0

    def _func_forward_dynamics(self) -> None:
        kernel_forward_dynamics(
            self.dyn_state,
            self.island_system.data.island,
            self.dyn_info,
            self.rigid_info,
            self.rigid_config,
            False,
        )

    def _func_update_acc(self) -> None:
        kernel_update_acc(self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config)

    def _func_integrate_dq_entity(
        self,
        dq: qd.template(),
        i_e: int,
        i_b: int,
        respect_joint_limit: bool,
    ) -> None:
        func_integrate_dq_entity(i_e, i_b, dq, self.dyn_info, self.rigid_info, self.rigid_config, respect_joint_limit)

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
                    self.island_system.data.island.is_hibernated, transpose=True, copy=False
                )
                islands_next_link_dst = qd_to_torch(
                    self.island_system.data.island.hibernated_next_link, transpose=True, copy=False
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
                    self.island_system.data.island,
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
        if force is None and torque is None:
            gs.raise_exception("Either 'force' or 'torque' must be specified.")
        if force is None and pos is not None:
            gs.raise_exception("'pos' requires 'force', as a torque acts the same wherever it is applied.")
        ref_frame = sanitize_ref_frame(ref, has_root_com=False)

        if pos is not None:
            pos, _, _ = self._sanitize_io_variables(
                pos, links_idx, self.n_links, "links_idx", envs_idx, (3,), skip_allocation=True
            )
        # An absent wrench component contributes nothing, which the kernel spells out as zeros.
        force = 0.0 if force is None else force
        torque = 0.0 if torque is None else torque
        force, _, _ = self._sanitize_io_variables(
            force, links_idx, self.n_links, "links_idx", envs_idx, (3,), skip_allocation=True
        )
        torque, links_idx, envs_idx = self._sanitize_io_variables(
            torque, links_idx, self.n_links, "links_idx", envs_idx, (3,), skip_allocation=True
        )
        if self.n_envs == 0:
            force, torque = force[None], torque[None]
            if pos is not None:
                pos = pos[None]

        # A wrench on a sleeping body must revive it, otherwise the input is silently dropped.
        if self._use_hibernation:
            kernel_wake_up_entities_by_links(
                links_idx,
                envs_idx,
                self.dyn_state,
                self.island_system.data.island,
                self.dyn_info,
                self.rigid_info,
                self.rigid_config,
            )

        if pos is None:
            kernel_apply_links_external_wrench(
                links_idx, envs_idx, force, torque, self.dyn_state, self.rigid_config, ref_frame, local
            )
        else:
            kernel_apply_links_external_wrench_at_pos(
                links_idx, envs_idx, pos, force, torque, self.dyn_state, self.rigid_config, ref_frame, local
            )

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

    def prepare_backward_replay(self, f: int) -> None:
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

    def begin_backward(self, f: int) -> None:
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

    def reverse_smooth_acceleration(self) -> None:
        """Reverse the unconstrained acceleration solve when no Constraint System exists."""
        kernel_manual_compute_qacc_bw(self.dyn_state, self.rigid_info, self.rigid_config)

    def finish_backward(self, f: int) -> None:
        """Reverse forward dynamics and restore forward execution mode."""
        envs_idx = self._scene._sanitize_envs_idx(None)
        kernel_copy_acc(f, self.dyn_state, self._rigid_adjoint_cache, self.rigid_config)

        kernel_forward_dynamics_without_qacc.grad(
            self.dyn_state,
            self.island_system.data.island,
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

    def get_links_net_contact_force(self, links_idx: IndexLike = None, envs_idx: IndexLike = None) -> torch.Tensor:
        tensor = qd_to_torch(
            self.dyn_state.links.contact_force,
            envs_idx,
            links_idx,
            transpose=True,
            copy=True,
        )
        return tensor[0] if self.n_envs == 0 else tensor

    # -------------------------------- Implementation --------------------------------

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
        self._geoms_offset_pos = self._geoms_offset_quat = None
        if not (
            np.allclose(geoms_offset_pos, 0.0, atol=gs.EPS)
            and np.allclose(gu.quat_to_xyz(geoms_offset_quat), 0.0, atol=gs.EPS)
        ):
            self._geoms_offset_pos = torch.from_numpy(geoms_offset_pos).to(device=gs.device, dtype=gs.tc_float)
            self._geoms_offset_quat = torch.from_numpy(geoms_offset_quat).to(device=gs.device, dtype=gs.tc_float)
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
        self._sol_min_timeconst = 2.0 * self._substep_dt
        self._sol_default_timeconst = (
            None
            if self._options.constraint_timeconst is None
            else max(self._options.constraint_timeconst, self._sol_min_timeconst)
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

    def finalize_data(self) -> None:
        """Finish initialization after island, contact and optional constraint owners have allocated their data."""
        self._finalize_dynamics()

    def set_coupling_qpos(self, qpos: TensorLike, qs_idx: IndexLike, skip_forward: bool) -> None:
        self.set_qpos(qpos, qs_idx=qs_idx, skip_forward=skip_forward)

    def set_coupling_dofs_velocity(self, velocity: TensorLike, dofs_idx: IndexLike, skip_forward: bool) -> None:
        self.set_dofs_velocity(velocity, dofs_idx=dofs_idx, skip_forward=skip_forward)

    def apply_coupling_wrench(self, force: TensorLike, torque: TensorLike, links_idx: IndexLike) -> None:
        self.apply_links_external_wrench(force, torque, links_idx=links_idx)

    def get_forward_update_flags(self) -> tuple[bool, bool]:
        return self._is_forward_pos_updated, self._is_forward_vel_updated

    def invalidate_forward_state(self) -> None:
        self._is_forward_pos_updated = False
        self._is_forward_vel_updated = False

    def save_adjoint_cache(self, f: int) -> None:
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

    def mark_forward_updated(self) -> None:
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
        kinematics_scratch = array_class.get_kinematics_scratch(
            entities=self.entities,
            batch_size=self._B,
            n_qs_actual=self.n_qs,
        )
        weight_scratch = array_class.get_weight_scratch(
            n_dofs=self.n_dofs_,
            batch_size=self._B,
        )
        ik_scratch = array_class.get_ik_scratch(
            options=self._options,
            entities=self.entities,
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
        self.data.kinematics_scratch = kinematics_scratch
        self.data.weight_scratch = weight_scratch
        self.data.ik_scratch = ik_scratch
        self.data.rigid_adjoint_cache = rigid_adjoint_cache
        self.data.dyn_state_adjoint_cache = dyn_state_adjoint_cache
        self.data.errno = errno

        self._errno = self.data.errno
        self.rigid_info = self.data.rigid_info
        self._rigid_adjoint_cache = self.data.rigid_adjoint_cache
        self.dyn_info = self.data.dyn_info
        self.dyn_state = self.data.dyn_state
        self.kinematics_scratch = self.data.kinematics_scratch
        if self._use_hibernation:
            self.n_awake_dofs = self.rigid_info.n_awake_dofs
        if self._requires_grad:
            self.dyn_state_adjoint_cache = self.data.dyn_state_adjoint_cache

        self._gravity = self.rigid_info.gravity
        self._gravity.from_numpy(np.tile(np.array(self._options.gravity, dtype=gs.np_float), (self._B, 1)))

    def _get_ik_scratch(self) -> array_class.IKScratch:
        return self.data.ik_scratch

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
                self._sol_min_timeconst,
                self._sol_default_timeconst,
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
                self._sol_min_timeconst,
                self._sol_default_timeconst,
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

    def _finalize_dynamics(self) -> None:
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
            sanitize_sol_params(joints_sol_params, self._sol_min_timeconst, self._sol_default_timeconst)

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
            sanitize_sol_params(variants_sol_params, self._sol_min_timeconst, self._sol_default_timeconst)
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
        self._base_links_idx = torch.tensor(base_links_idx, dtype=gs.tc_int, device=gs.device)

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
        self._links_offset_quat_is_identity = np.all(
            np.isclose(gu.quat_to_xyz(links_offset_quat), 0.0, atol=gs.EPS), axis=-1
        )
        self._links_offset_pos_is_identity = np.all(np.isclose(links_offset_pos, 0.0, atol=gs.EPS), axis=-1)
        if links_offset_per_env:
            self._links_offset_quat_is_identity = self._links_offset_quat_is_identity.all(axis=0)
            self._links_offset_pos_is_identity = self._links_offset_pos_is_identity.all(axis=0)
        # The link offset tensors are allocated even when every link is identity, so the getter kernels can take them
        # unconditionally.
        self._links_offset_pos = torch.from_numpy(links_offset_pos).to(device=gs.device, dtype=gs.tc_float)
        self._links_offset_quat = torch.from_numpy(links_offset_quat).to(device=gs.device, dtype=gs.tc_float)
        self._vgeoms_offset_pos = self._vgeoms_offset_quat = None
        if not (
            np.allclose(vgeoms_offset_pos, 0.0, atol=gs.EPS)
            and np.allclose(gu.quat_to_xyz(vgeoms_offset_quat), 0.0, atol=gs.EPS)
        ):
            self._vgeoms_offset_pos = torch.from_numpy(vgeoms_offset_pos).to(device=gs.device, dtype=gs.tc_float)
            self._vgeoms_offset_quat = torch.from_numpy(vgeoms_offset_quat).to(device=gs.device, dtype=gs.tc_float)

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

    def _set_kinematic_dofs_velocity(
        self,
        velocity: TensorLike | None,
        dofs_idx: IndexLike = None,
        envs_idx: IndexLike = None,
        *,
        skip_forward: bool = False,
    ) -> None:
        if gs.use_zerocopy:
            vel = qd_to_torch(self.dyn_state.dofs.vel, transpose=True, copy=False)
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
                    (0, *dofs_mask) if self.n_envs == 0 else indices_to_mask(envs_idx, *dofs_mask, boolean_mask=False)
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
            velocity, dofs_idx, envs_idx = self._sanitize_io_variables(
                velocity, dofs_idx, self.n_dofs, "dofs_idx", envs_idx, skip_allocation=True
            )
            if velocity is None:
                kernel_set_dofs_zero_velocity(dofs_idx, envs_idx, self.dyn_state, self.rigid_config)
            else:
                if self.n_envs == 0:
                    velocity = velocity[None]
                kernel_set_dofs_velocity(dofs_idx, envs_idx, velocity, self.dyn_state, self.rigid_config)

        if not skip_forward:
            if envs_idx.dtype == torch.bool:
                fn = kernel_masked_forward_velocity
            else:
                fn = kernel_forward_velocity
            fn(envs_idx, self.dyn_state, self.dyn_info, self.rigid_info, self.rigid_config, is_backward=False)
            self._is_forward_vel_updated = True
        else:
            self._is_forward_vel_updated = False

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
def create_rigid_solver_system(engine: SimulatorEngine) -> RigidSolverSystem | None:
    """Admit the Rigid owner only when the authored Scene contains a Rigid entity."""
    return (
        RigidSolverSystem(engine.scene)
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
