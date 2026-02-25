import math
from typing import TYPE_CHECKING, Literal

import quadrants as qd
import numpy as np
import torch

import genesis as gs
import genesis.utils.array_class as array_class
from genesis.engine.entities import DroneEntity, RigidEntity
from genesis.engine.entities.base_entity import Entity
from genesis.engine.states import QueriedStates, RigidSolverState
from genesis.options.solvers import RigidOptions
from genesis.utils.misc import (
    DeprecationError,
    qd_to_torch,
    qd_to_numpy,
    indices_to_mask,
    broadcast_tensor,
    sanitize_indexed_tensor,
    assign_indexed_tensor,
)
from genesis.utils.sdf import SDF

from ..base_solver import Solver
from .collider import Collider
from .constraint import ConstraintSolver, ConstraintSolverIsland
from .abd.misc import (
    func_add_safe_backward,
    func_apply_coupling_force,
    func_apply_link_external_force,
    func_apply_external_torque,
    func_apply_link_external_torque,
    func_atomic_add_if,
    func_check_index_range,
    func_clear_external_force,
    func_read_field_if,
    func_wakeup_entity_and_its_temp_island,
    func_write_field_if,
    func_write_and_read_field_if,
    kernel_init_invweight,
    kernel_init_meaninertia,
    kernel_init_dof_fields,
    kernel_init_link_fields,
    kernel_update_heterogeneous_link_info,
    kernel_init_joint_fields,
    kernel_init_vert_fields,
    kernel_init_vvert_fields,
    kernel_init_geom_fields,
    kernel_adjust_link_inertia,
    kernel_init_vgeom_fields,
    kernel_init_entity_fields,
    kernel_init_equality_fields,
    kernel_apply_links_external_force,
    kernel_apply_links_external_torque,
    kernel_update_geoms_render_T,
    kernel_update_vgeoms_render_T,
    kernel_bit_reduction,
    kernel_set_zero,
    kernel_clear_external_force,
)
from .abd.forward_kinematics import (
    func_aggregate_awake_entities,
    func_COM_links,
    func_COM_links_entity,
    func_forward_kinematics_entity,
    func_forward_kinematics_batch,
    func_forward_velocity_entity,
    func_forward_velocity_batch,
    func_forward_velocity,
    func_hibernate_entity_and_zero_dof_velocities,
    func_hibernate__for_all_awake_islands_either_hiberanate_or_update_aabb_sort_buffer,
    func_update_geoms_entity,
    func_update_geoms_batch,
    func_update_all_verts,
    func_update_cartesian_space,
    func_update_cartesian_space_entity,
    func_update_cartesian_space_batch,
    func_update_geoms,
    func_update_verts_for_geom,
    kernel_forward_kinematics_links_geoms,
    kernel_masked_forward_kinematics_links_geoms,
    kernel_forward_velocity,
    kernel_masked_forward_velocity,
    kernel_forward_kinematics_entity,
    kernel_update_geoms,
    kernel_update_verts_for_geoms,
    kernel_update_all_verts,
    kernel_update_geom_aabbs,
    kernel_update_vgeoms,
    kernel_update_cartesian_space,
)
from .abd.forward_dynamics import (
    func_actuation,
    func_bias_force,
    func_compute_mass_matrix,
    func_compute_qacc,
    func_factor_mass,
    func_forward_dynamics,
    func_solve_mass_entity,
    func_solve_mass_batch,
    func_solve_mass,
    func_torque_and_passive_force,
    func_update_acc,
    func_update_force,
    func_integrate,
    func_implicit_damping,
    func_vel_at_point,
    kernel_compute_mass_matrix,
    kernel_forward_dynamics,
    kernel_update_acc,
    kernel_compute_qacc,
    kernel_forward_dynamics_without_qacc,
    update_qacc_from_qvel_delta,
    update_qvel,
)
from .abd.accessor import (
    kernel_get_state,
    kernel_set_state,
    kernel_get_state_grad,
    kernel_set_links_pos,
    kernel_set_links_pos_grad,
    kernel_set_links_quat,
    kernel_set_links_quat_grad,
    kernel_set_links_mass_shift,
    kernel_set_links_COM_shift,
    kernel_set_links_inertial_mass,
    kernel_wake_up_entities_by_links,
    kernel_set_geoms_friction_ratio,
    kernel_set_qpos,
    kernel_set_global_sol_params,
    kernel_set_sol_params,
    kernel_set_dofs_kp,
    kernel_set_dofs_kv,
    kernel_set_dofs_force_range,
    kernel_set_dofs_stiffness,
    kernel_set_dofs_armature,
    kernel_set_dofs_damping,
    kernel_set_dofs_frictionloss,
    kernel_set_dofs_limit,
    kernel_set_dofs_velocity,
    kernel_set_dofs_velocity_grad,
    kernel_set_dofs_zero_velocity,
    kernel_set_dofs_position,
    kernel_control_dofs_force,
    kernel_control_dofs_velocity,
    kernel_control_dofs_position,
    kernel_control_dofs_position_velocity,
    kernel_get_links_vel,
    kernel_get_links_acc,
    kernel_get_dofs_control_force,
    kernel_set_drone_rpm,
    kernel_update_drone_propeller_vgeoms,
    kernel_set_geom_friction,
    kernel_set_geoms_friction,
)
from .abd.diff import (
    func_copy_cartesian_space,
    func_copy_next_to_curr,
    func_copy_next_to_curr_grad,
    func_integrate_dq_entity,
    func_is_grad_valid,
    func_load_adjoint_cache,
    func_save_adjoint_cache,
    kernel_save_adjoint_cache,
    kernel_prepare_backward_substep,
    kernel_begin_backward_substep,
    kernel_copy_acc,
)

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.engine.simulator import Simulator
    from genesis.engine.entities.rigid_entity import RigidJoint, RigidLink, RigidGeom, RigidVisGeom


IS_OLD_TORCH = tuple(map(int, torch.__version__.split(".")[:2])) < (2, 8)

# minimum constraint impedance
IMP_MIN = 0.0001
# maximum constraint impedance
IMP_MAX = 0.9999

# Minimum ratio between simulation timestep `_substep_dt` and time constant of constraints
TIME_CONSTANT_SAFETY_FACTOR = 2.0


def _sanitize_sol_params(
    sol_params,
    min_timeconst: float,
    default_timeconst: float | None = None,
):
    timeconst, dampratio, dmin, dmax, width, mid, power = sol_params.reshape((-1, 7)).T
    if default_timeconst is None:
        default_timeconst = min_timeconst
    if (timeconst < gs.EPS).any():
        gs.logger.debug(
            f"Constraint solver time constant not specified. Using default value (`{default_timeconst:0.6g}`)."
        )
    invalid_mask = (timeconst > gs.EPS) & (timeconst + gs.EPS < min_timeconst)
    if invalid_mask.any():
        gs.logger.warning(
            "Constraint solver time constant should be greater than 2*substep_dt. timeconst is changed from "
            f"`{min(timeconst[invalid_mask]):0.6g}` to `{min_timeconst:0.6g}`). Decrease simulation timestep or "
            "increase timeconst to avoid altering the original value."
        )
    timeconst[timeconst < gs.EPS] = default_timeconst
    timeconst[:] = timeconst.clip(min_timeconst)
    dampratio[:] = dampratio.clip(0.0)
    dmin[:] = dmin.clip(IMP_MIN, IMP_MAX)
    dmax[:] = dmax.clip(IMP_MIN, IMP_MAX)
    mid[:] = mid.clip(IMP_MIN, IMP_MAX)
    width[:] = width.clip(0.0)
    power[:] = power.clip(1)
    return sol_params


class RigidSolver(Solver):
    # override typing
    _entities: list[RigidEntity] = gs.List()

    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def __init__(self, scene: "Scene", sim: "Simulator", options: RigidOptions) -> None:
        super().__init__(scene, sim, options)

        # options
        self._enable_collision = options.enable_collision
        self._enable_multi_contact = options.enable_multi_contact
        self._enable_mujoco_compatibility = options.enable_mujoco_compatibility
        self._enable_joint_limit = options.enable_joint_limit
        self._enable_self_collision = options.enable_self_collision
        self._enable_neutral_collision = options.enable_neutral_collision
        self._enable_adjacent_collision = options.enable_adjacent_collision
        self._disable_constraint = options.disable_constraint
        self._max_collision_pairs = options.max_collision_pairs
        self._integrator = options.integrator
        self._box_box_detection = options.box_box_detection
        self._requires_grad = self._sim.options.requires_grad
        self._enable_heterogeneous = False  # Set to True when any entity has heterogeneous morphs

        self._use_contact_island = options.use_contact_island
        self._use_hibernation = options.use_hibernation and options.use_contact_island
        if options.use_hibernation and not options.use_contact_island:
            gs.logger.warning(
                "`use_hibernation` is set to False because `use_contact_island=False`. Please set "
                "`use_contact_island=True` if you want to use hibernation"
            )

        self._hibernation_thresh_vel = options.hibernation_thresh_vel
        self._hibernation_thresh_acc = options.hibernation_thresh_acc

        if options.contact_resolve_time is not None:
            gs.logger.warning(
                "Rigid option 'contact_resolve_time' is deprecated and will be remove in future release. Please "
                "use 'constraint_timeconst' instead."
            )
        self._sol_min_timeconst = TIME_CONSTANT_SAFETY_FACTOR * self._substep_dt
        self._sol_default_timeconst = max(options.constraint_timeconst, self._sol_min_timeconst)

        if (
            not self._disable_constraint
            and self._enable_collision
            and not options.use_gjk_collision
            and self._substep_dt < 0.002
        ):
            gs.logger.warning(
                "Using a simulation timestep smaller than 2ms is not recommended for 'use_gjk_collision=False' as "
                "it could lead to numerically unstable collision detection."
            )

        self._options = options

        self.collider = None
        self.constraint_solver = None

        self.qpos: qd.Template | qd.types.NDArray | None = None

        self._is_backward: bool = False
        self._is_forward_pos_updated: bool = False
        self._is_forward_vel_updated: bool = False

        self._queried_states = QueriedStates()

        self._ckpt = dict()

    def init_ckpt(self):
        pass

    def add_entity(self, idx, material, morph, surface, visualize_contact, name: str | None = None) -> Entity:
        # Handle heterogeneous morphs (list/tuple of morphs)
        morph_heterogeneous = []
        if isinstance(morph, (tuple, list)):
            morph, *morph_heterogeneous = morph
            self._enable_heterogeneous |= bool(morph_heterogeneous)

        if isinstance(morph, gs.morphs.Drone):
            EntityClass = DroneEntity
        else:
            EntityClass = RigidEntity

        morph._enable_mujoco_compatibility = self._enable_mujoco_compatibility

        entity = EntityClass(
            scene=self._scene,
            solver=self,
            material=material,
            morph=morph,
            surface=surface,
            idx=idx,
            idx_in_solver=self.n_entities,
            link_start=self.n_links,
            joint_start=self.n_joints,
            q_start=self.n_qs,
            dof_start=self.n_dofs,
            geom_start=self.n_geoms,
            cell_start=self.n_cells,
            vert_start=self.n_verts,
            free_verts_state_start=self.n_free_verts,
            fixed_verts_state_start=self.n_fixed_verts,
            face_start=self.n_faces,
            edge_start=self.n_edges,
            vgeom_start=self.n_vgeoms,
            vvert_start=self.n_vverts,
            vface_start=self.n_vfaces,
            visualize_contact=visualize_contact,
            morph_heterogeneous=morph_heterogeneous,
            name=name,
        )
        assert isinstance(entity, RigidEntity)
        self._entities.append(entity)

        return entity

    def build(self):
        super().build()
        self.n_envs = self.sim.n_envs
        self._B = self.sim._B
        self._para_level = self.sim._para_level

        for entity in self._entities:
            entity._build()

        self._n_qs = self.n_qs
        self._n_dofs = self.n_dofs
        self._n_links = self.n_links
        self._n_joints = self.n_joints
        self._n_geoms = self.n_geoms
        self._n_cells = self.n_cells
        self._n_verts = self.n_verts
        self._n_free_verts = self.n_free_verts
        self._n_fixed_verts = self.n_fixed_verts
        self._n_faces = self.n_faces
        self._n_edges = self.n_edges
        self._n_vgeoms = self.n_vgeoms
        self._n_vfaces = self.n_vfaces
        self._n_vverts = self.n_vverts
        self._n_entities = self.n_entities
        self._n_equalities = self.n_equalities

        self._geoms = self.geoms
        self._vgeoms = self.vgeoms
        self._links = self.links
        self._joints = self.joints
        self._equalities = self.equalities

        base_links_idx = []
        for link in self.links:
            if link.parent_idx == -1 and link.is_fixed:
                base_links_idx.append(link.idx)
        for joint in self.joints:
            if joint.type == gs.JOINT_TYPE.FREE:
                base_links_idx.append(joint.link.idx)
        self._base_links_idx = torch.tensor(base_links_idx, dtype=gs.tc_int, device=gs.device)

        # used for creating dummy fields for compilation to work
        self.n_qs_ = max(1, self.n_qs)
        self.n_dofs_ = max(1, self.n_dofs)
        self.n_links_ = max(1, self.n_links)
        self.n_joints_ = max(1, self.n_joints)
        self.n_geoms_ = max(1, self.n_geoms)
        self.n_cells_ = max(1, self.n_cells)
        self.n_verts_ = max(1, self.n_verts)
        self.n_faces_ = max(1, self.n_faces)
        self.n_edges_ = max(1, self.n_edges)
        self.n_vgeoms_ = max(1, self.n_vgeoms)
        self.n_vfaces_ = max(1, self.n_vfaces)
        self.n_vverts_ = max(1, self.n_vverts)
        self.n_entities_ = max(1, self.n_entities)
        self.n_free_verts_ = max(1, self.n_free_verts)
        self.n_fixed_verts_ = max(1, self.n_fixed_verts)
        self.n_candidate_equalities_ = max(1, self.n_equalities + self._options.max_dynamic_constraints)

        # batch_links_info is required when heterogeneous simulation is used.
        # We must update options because get_links_info reads from solver._options.batch_links_info.
        if self._enable_heterogeneous:
            self._options.batch_links_info = True

        static_rigid_sim_config = dict(
            backend=gs.backend,
            para_level=self.sim._para_level,
            requires_grad=self.sim.options.requires_grad,
            use_hibernation=self._use_hibernation,
            batch_links_info=self._options.batch_links_info,
            batch_dofs_info=self._options.batch_dofs_info,
            batch_joints_info=self._options.batch_joints_info,
            enable_heterogeneous=self._enable_heterogeneous,
            enable_mujoco_compatibility=self._enable_mujoco_compatibility,
            enable_multi_contact=self._enable_multi_contact,
            enable_collision=self._enable_collision,
            enable_joint_limit=self._enable_joint_limit,
            box_box_detection=self._box_box_detection,
            sparse_solve=self._options.sparse_solve,
            integrator=self._integrator,
            solver_type=self._options.constraint_solver,
        )

        if self.is_active:
            # TODO: These alternative tiled algorithms are designed to reduce the impact of latency. However, naive
            # implementation scales slightly better asymptotically than shared memory-based implementation because the
            # scheduler of modern GPUs is able to hides latency by swapping warps if the workload is sufficient. The
            # crossover threshold is both hardware and kernel-dependent. As a result, the optimal implementation should
            # be selected based on dynamic timer-based profiling instead of hard-coded heuristic.
            max_n_dofs_per_entity = max(entity.n_dofs for entity in self.entities) if self.entities else 0
            if gs.backend != gs.cpu:
                max_shared_mem = 32.0 if gs.backend == gs.metal else 48.0
                max_n_warps = int(math.sqrt(max_shared_mem * 1024 / (4 if gs.qd_float == qd.f32 else 8))) // 32
                max_n_threads = max_n_warps * 32

                enable_tiled_cholesky_mass_matrix = 8 <= max_n_dofs_per_entity <= max_n_threads and self.n_envs <= 16384
                enable_tiled_cholesky_hessian = 16 <= self.n_dofs <= max_n_threads and self.n_envs <= 16384
                tiled_n_dofs = min(max(math.ceil(self.n_dofs / 32), 1), max_n_warps) * 32
                tiled_n_dofs_per_entity = min(max(math.ceil(max_n_dofs_per_entity / 32), 1), max_n_warps) * 32

                static_rigid_sim_config.update(
                    enable_tiled_cholesky_mass_matrix=enable_tiled_cholesky_mass_matrix,
                    enable_tiled_cholesky_hessian=enable_tiled_cholesky_hessian,
                    tiled_n_dofs_per_entity=tiled_n_dofs_per_entity,
                    tiled_n_dofs=tiled_n_dofs,
                )

            # Add terms for static inner loops, use -1 if not requires_grad to avoid re-compilation
            if self.sim.options.requires_grad:
                static_rigid_sim_config.update(
                    max_n_geoms_per_entity=max(len(entity.geoms) for entity in self.entities) if self.links else 0,
                    max_n_links_per_entity=max(len(entity.links) for entity in self.entities) if self.entities else 0,
                    max_n_joints_per_link=max(len(link.joints) for link in self.links) if self.links else 0,
                    max_n_dofs_per_joint=max(joint.n_dofs for joint in self.joints) if self.joints else 0,
                    max_n_dofs_per_entity=max_n_dofs_per_entity,
                    max_n_dofs_per_link=max(link.n_dofs for link in self.links) if self.links else 0,
                    max_n_qs_per_link=max(link.n_qs for link in self.links) if self.links else 0,
                    n_entities=self._n_entities,
                    n_links=self._n_links,
                    n_geoms=self._n_geoms,
                )

        self._static_rigid_sim_config = array_class.StructRigidSimStaticConfig(**static_rigid_sim_config)

        if self._static_rigid_sim_config.use_hibernation:
            if gs.use_ndarray:
                gs.raise_exception(
                    "Hibernation is not yet supported with dynamic array mode. "
                    "Please set performance_mode=True or use_hibernation=False."
                )

        if self._static_rigid_sim_config.requires_grad:
            if self._static_rigid_sim_config.use_hibernation:
                gs.raise_exception("Hibernation is not supported yet when requires_grad is True")
            if self._static_rigid_sim_config.integrator != gs.integrator.approximate_implicitfast:
                gs.raise_exception(
                    "Only approximate_implicitfast integrator is supported yet when requires_grad is True."
                )
            from genesis.engine.couplers import SAPCoupler, IPCCoupler

            if isinstance(self.sim.coupler, (SAPCoupler, IPCCoupler)):
                gs.raise_exception(
                    f"{type(self.sim.coupler).__name__} is not supported yet when requires_grad is True."
                )

            if getattr(self._options, "noslip_iterations", 0) > 0:
                gs.raise_exception("Noslip is not supported yet when requires_grad is True.")

        # We initialize data even if the solver is not active because the coupler needs arguments like
        # rigid_solver.links_state, etc. regardless of the solver is active or not.
        self.data_manager = array_class.DataManager(self)
        self._errno = self.data_manager.errno

        self._rigid_global_info = self.data_manager.rigid_global_info
        self._rigid_adjoint_cache = self.data_manager.rigid_adjoint_cache
        if self._use_hibernation:
            self.n_awake_dofs = self._rigid_global_info.n_awake_dofs
            self.awake_dofs = self._rigid_global_info.awake_dofs
            self.n_awake_links = self._rigid_global_info.n_awake_links
            self.awake_links = self._rigid_global_info.awake_links
            self.n_awake_entities = self._rigid_global_info.n_awake_entities
            self.awake_entities = self._rigid_global_info.awake_entities
        if self._requires_grad:
            self.dofs_state_adjoint_cache = self.data_manager.dofs_state_adjoint_cache
            self.links_state_adjoint_cache = self.data_manager.links_state_adjoint_cache
            self.joints_state_adjoint_cache = self.data_manager.joints_state_adjoint_cache
            self.geoms_state_adjoint_cache = self.data_manager.geoms_state_adjoint_cache

        self._init_mass_mat()
        self._init_dof_fields()

        self._init_vert_fields()
        self._init_vvert_fields()
        self._init_geom_fields()
        self._init_vgeom_fields()
        self._init_link_fields()
        self._process_heterogeneous_link_info()
        self._init_entity_fields()
        self._init_equality_fields()

        self._init_envs_offset()

        self._init_invweight_and_meaninertia(force_update=False)
        self._func_update_geoms(self._scene._envs_idx, force_update_fixed_geoms=True)

        self._init_collider()
        self._init_constraint_solver()

        # FIXME: when the migration is finished, we will remove the about two lines
        self._func_vel_at_point = func_vel_at_point
        self._func_apply_coupling_force = func_apply_coupling_force

    def _init_invweight_and_meaninertia(self, envs_idx=None, *, force_update=True):
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

        # Compute state in neutral configuration at rest
        qpos = qd_to_torch(self.qpos0, envs_idx, transpose=True)
        if self.n_envs == 0:
            qpos = qpos[0]
        self.set_qpos(qpos, envs_idx=envs_idx if self.n_envs > 0 else None)

        # Compute mass matrix without any implicit damping terms
        # TODO: This kernel could be optimized to take `envs_idx` as input if performance is critical.
        kernel_compute_mass_matrix(
            links_state=self.links_state,
            links_info=self.links_info,
            dofs_state=self.dofs_state,
            dofs_info=self.dofs_info,
            entities_info=self.entities_info,
            rigid_global_info=self._rigid_global_info,
            static_rigid_sim_config=self._static_rigid_sim_config,
            decompose=True,
        )

        # Define some proxies for convenience
        mass_mat_D_inv = qd_to_numpy(self._rigid_global_info.mass_mat_D_inv)
        mass_mat_L = qd_to_numpy(self._rigid_global_info.mass_mat_L)
        offsets = qd_to_numpy(self.links_state.i_pos)
        cdof_ang = qd_to_numpy(self.dofs_state.cdof_ang)
        cdof_vel = qd_to_numpy(self.dofs_state.cdof_vel)
        links_joint_start = qd_to_numpy(self.links_info.joint_start)
        links_joint_end = qd_to_numpy(self.links_info.joint_end)
        links_dof_end = qd_to_numpy(self.links_info.dof_end)
        links_n_dofs = qd_to_numpy(self.links_info.n_dofs)
        links_parent_idx = qd_to_numpy(self.links_info.parent_idx)
        joints_type = qd_to_numpy(self.joints_info.type)
        joints_dof_start = qd_to_numpy(self.joints_info.dof_start)
        joints_n_dofs = qd_to_numpy(self.joints_info.n_dofs)

        links_invweight = np.zeros((len(envs_idx), self._n_links, 2), dtype=gs.np_float)
        dofs_invweight = np.zeros((len(envs_idx), self._n_dofs), dtype=gs.np_float)

        # TODO: Simple numpy-based for-loop for now as it is not performance critical
        for i_b_, i_b in enumerate(envs_idx):
            # Compute the inverted mass matrix efficiently
            mass_mat_L_inv = np.eye(self.n_dofs_)
            for i_d in range(self.n_dofs_):
                for j_d in range(i_d):
                    mass_mat_L_inv[i_d] -= mass_mat_L[i_d, j_d, i_b] * mass_mat_L_inv[j_d]
            mass_mat_inv = (mass_mat_L_inv * mass_mat_D_inv[:, i_b]) @ mass_mat_L_inv.T

            # Compute links invweight if necessary
            if i_b_ == 0 or self._options.batch_links_info:
                for i_l in range(self._n_links):
                    jacp = np.zeros((3, self._n_dofs))
                    jacr = np.zeros((3, self._n_dofs))

                    offset = offsets[i_l, i_b]

                    j_l = i_l
                    while j_l != -1:
                        link_n_dofs = links_n_dofs[j_l]
                        if self._options.batch_links_info:
                            link_n_dofs = link_n_dofs[i_b]
                        for i_d_ in range(link_n_dofs):
                            link_dof_end = links_dof_end[j_l]
                            if self._options.batch_links_info:
                                link_dof_end = link_dof_end[i_b]
                            i_d = link_dof_end - i_d_ - 1
                            jacp[:, i_d] = cdof_vel[i_d, i_b] + np.cross(cdof_ang[i_d, i_b], offset)
                            jacr[:, i_d] = cdof_ang[i_d, i_b]
                        link_parent_idx = links_parent_idx[j_l]
                        if self._options.batch_links_info:
                            link_parent_idx = link_parent_idx[i_b]
                        j_l = link_parent_idx

                    jac = np.concatenate((jacp, jacr), axis=0)

                    A = jac @ mass_mat_inv @ jac.T
                    A_diag = np.diag(A)

                    links_invweight[i_b_, i_l, 0] = A_diag[:3].mean()
                    links_invweight[i_b_, i_l, 1] = A_diag[3:].mean()

            # Compute dofs invweight
            if i_b_ == 0 or self._options.batch_dofs_info:
                for i_l in range(self._n_links):
                    link_joint_start = links_joint_start[i_l]
                    link_joint_end = links_joint_end[i_l]
                    if self._options.batch_links_info:
                        link_joint_start = link_joint_start[i_b]
                        link_joint_end = link_joint_end[i_b]
                    for i_j in range(link_joint_start, link_joint_end):
                        joint_type = joints_type[i_j]
                        if self._options.batch_joints_info:
                            joint_type = joint_type[i_b]
                        if joint_type == gs.JOINT_TYPE.FIXED:
                            continue

                        dof_start = joints_dof_start[i_j]
                        n_dofs = joints_n_dofs[i_j]
                        if self._options.batch_joints_info:
                            dof_start = dof_start[i_b]
                            n_dofs = n_dofs[i_b]
                        jac = np.zeros((n_dofs, self._n_dofs))
                        for i_d_ in range(n_dofs):
                            jac[i_d_, dof_start + i_d_] = 1.0

                        A = jac @ mass_mat_inv @ jac.T
                        A_diag = np.diag(A)

                        if joint_type == gs.JOINT_TYPE.FREE:
                            dofs_invweight[i_b_, dof_start : (dof_start + 3)] = A_diag[:3].mean()
                            dofs_invweight[i_b_, (dof_start + 3) : (dof_start + 6)] = A_diag[3:].mean()
                        elif joint_type == gs.JOINT_TYPE.SPHERICAL:
                            dofs_invweight[i_b_, dof_start : (dof_start + 3)] = A_diag[:3].mean()
                        else:  # REVOLUTE or PRISMATIC
                            dofs_invweight[i_b_, dof_start] = A_diag[0]

            # Stop there if not batched
            if not batched:
                break

        # Update links and dofs invweight if necessary
        if not self._options.batch_links_info:
            links_invweight = links_invweight[0]
        if not self._options.batch_dofs_info:
            dofs_invweight = dofs_invweight[0]
        kernel_init_invweight(
            envs_idx,
            links_invweight,
            dofs_invweight,
            links_info=self.links_info,
            dofs_info=self.dofs_info,
            force_update=force_update,
            rigid_global_info=self._rigid_global_info,
            static_rigid_sim_config=self._static_rigid_sim_config,
        )

        # Compute meaninertia from mass matrix
        kernel_init_meaninertia(
            envs_idx=envs_idx,
            rigid_global_info=self._rigid_global_info,
            entities_info=self.entities_info,
            static_rigid_sim_config=self._static_rigid_sim_config,
        )

    def _init_mass_mat(self):
        self.mass_mat = self._rigid_global_info.mass_mat
        self.mass_mat_L = self._rigid_global_info.mass_mat_L
        self.mass_mat_D_inv = self._rigid_global_info.mass_mat_D_inv
        self.mass_mat_mask = self._rigid_global_info.mass_mat_mask
        self.meaninertia = self._rigid_global_info.meaninertia

        self.mass_mat_mask.fill(True)

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
        self._rigid_global_info.mass_parent_mask.from_numpy(mass_parent_mask)

        self._rigid_global_info.gravity.from_numpy(self.gravity)

    def _init_dof_fields(self):
        self.dofs_info = self.data_manager.dofs_info
        self.dofs_state = self.data_manager.dofs_state

        joints = self.joints
        has_dofs = sum(joint.n_dofs for joint in joints) > 0
        if has_dofs:  # handle the case where there is a link with no dofs -- otherwise may cause invalid memory
            kernel_init_dof_fields(
                entity_idx=np.concatenate(
                    [(joint.link._entity_idx_in_solver,) * joint.n_dofs for joint in joints if joint.n_dofs],
                    dtype=gs.np_int,
                ),
                dofs_motion_ang=np.concatenate([joint.dofs_motion_ang for joint in joints], dtype=gs.np_float),
                dofs_motion_vel=np.concatenate([joint.dofs_motion_vel for joint in joints], dtype=gs.np_float),
                dofs_limit=np.concatenate([joint.dofs_limit for joint in joints], dtype=gs.np_float),
                dofs_invweight=np.concatenate([joint.dofs_invweight for joint in joints], dtype=gs.np_float),
                dofs_stiffness=np.concatenate([joint.dofs_stiffness for joint in joints], dtype=gs.np_float),
                dofs_damping=np.concatenate([joint.dofs_damping for joint in joints], dtype=gs.np_float),
                dofs_frictionloss=np.concatenate([joint.dofs_frictionloss for joint in joints], dtype=gs.np_float),
                dofs_armature=np.concatenate([joint.dofs_armature for joint in joints], dtype=gs.np_float),
                dofs_kp=np.concatenate([joint.dofs_kp for joint in joints], dtype=gs.np_float),
                dofs_kv=np.concatenate([joint.dofs_kv for joint in joints], dtype=gs.np_float),
                dofs_force_range=np.concatenate([joint.dofs_force_range for joint in joints], dtype=gs.np_float),
                dofs_info=self.dofs_info,
                dofs_state=self.dofs_state,
                rigid_global_info=self._rigid_global_info,
                static_rigid_sim_config=self._static_rigid_sim_config,
            )

        # just in case
        self.dofs_state.force.fill(0)

    def _init_link_fields(self):
        self.links_info = self.data_manager.links_info
        self.links_state = self.data_manager.links_state

        if self.links:
            links = self.links
            kernel_init_link_fields(
                links_parent_idx=np.array([link.parent_idx for link in links], dtype=gs.np_int),
                links_root_idx=np.array([link.root_idx for link in links], dtype=gs.np_int),
                links_q_start=np.array([link.q_start for link in links], dtype=gs.np_int),
                links_dof_start=np.array([link.dof_start for link in links], dtype=gs.np_int),
                links_joint_start=np.array([link.joint_start for link in links], dtype=gs.np_int),
                links_q_end=np.array([link.q_end for link in links], dtype=gs.np_int),
                links_dof_end=np.array([link.dof_end for link in links], dtype=gs.np_int),
                links_joint_end=np.array([link.joint_end for link in links], dtype=gs.np_int),
                links_invweight=np.array([link.invweight for link in links], dtype=gs.np_float),
                links_is_fixed=np.array([link.is_fixed for link in links], dtype=gs.np_bool),
                links_pos=np.array([link.pos for link in links], dtype=gs.np_float),
                links_quat=np.array([link.quat for link in links], dtype=gs.np_float),
                links_inertial_pos=np.array([link.inertial_pos for link in links], dtype=gs.np_float),
                links_inertial_quat=np.array([link.inertial_quat for link in links], dtype=gs.np_float),
                links_inertial_i=np.array([link.inertial_i for link in links], dtype=gs.np_float),
                links_inertial_mass=np.array([link.inertial_mass for link in links], dtype=gs.np_float),
                links_entity_idx=np.array([link._entity_idx_in_solver for link in links], dtype=gs.np_int),
                links_geom_start=np.array([link.geom_start for link in links], dtype=gs.np_int),
                links_geom_end=np.array([link.geom_end for link in links], dtype=gs.np_int),
                links_vgeom_start=np.array([link.vgeom_start for link in links], dtype=gs.np_int),
                links_vgeom_end=np.array([link.vgeom_end for link in links], dtype=gs.np_int),
                # Quadrants variables
                links_info=self.links_info,
                links_state=self.links_state,
                rigid_global_info=self._rigid_global_info,
                static_rigid_sim_config=self._static_rigid_sim_config,
            )

        self.joints_info = self.data_manager.joints_info
        self.joints_state = self.data_manager.joints_state

        if self.joints:
            # Make sure that the constraints parameters are valid
            joints = self.joints
            joints_sol_params = np.array([joint.sol_params for joint in joints], dtype=gs.np_float)
            _sanitize_sol_params(joints_sol_params, self._sol_min_timeconst, self._sol_default_timeconst)

            kernel_init_joint_fields(
                joints_type=np.array([joint.type for joint in joints], dtype=gs.np_int),
                joints_sol_params=joints_sol_params,
                joints_q_start=np.array([joint.q_start for joint in joints], dtype=gs.np_int),
                joints_dof_start=np.array([joint.dof_start for joint in joints], dtype=gs.np_int),
                joints_q_end=np.array([joint.q_end for joint in joints], dtype=gs.np_int),
                joints_dof_end=np.array([joint.dof_end for joint in joints], dtype=gs.np_int),
                joints_pos=np.array([joint.pos for joint in joints], dtype=gs.np_float),
                # Quadrants variables
                joints_info=self.joints_info,
                static_rigid_sim_config=self._static_rigid_sim_config,
            )

        # Check if the initial configuration is out-of-bounds
        self.qpos = self._rigid_global_info.qpos
        self.qpos0 = self._rigid_global_info.qpos0
        is_init_qpos_out_of_bounds = False
        if self.n_qs > 0:
            init_qpos = np.tile(np.expand_dims(self.init_qpos, -1), (1, self._B))
            self.qpos0.from_numpy(init_qpos)
            for joint in joints:
                if joint.type in (gs.JOINT_TYPE.REVOLUTE, gs.JOINT_TYPE.PRISMATIC):
                    is_init_qpos_out_of_bounds |= (joint.dofs_limit[0, 0] > init_qpos[joint.q_start]).any()
                    is_init_qpos_out_of_bounds |= (init_qpos[joint.q_start] > joint.dofs_limit[0, 1]).any()
                    # init_qpos[joint.q_start] = np.clip(init_qpos[joint.q_start], *joint.dofs_limit[0])
            self.qpos.from_numpy(init_qpos)
        if is_init_qpos_out_of_bounds:
            gs.logger.warning(
                "Neutral robot position (qpos0) exceeds joint limits."
                # "Clipping initial position too make sure it is valid."
            )

        # This is for IK use only
        # TODO: support IK with parallel envs
        # self._rigid_global_info.links_T = qd.Matrix.field(n=4, m=4, dtype=gs.qd_float, shape=self.n_links)
        self.links_T = self._rigid_global_info.links_T

    def _process_heterogeneous_link_info(self):
        """
        Process heterogeneous link info: dispatch geoms per environment and compute per-env inertial properties.
        This method is called after _init_link_fields to update the per-environment inertial properties
        for entities with heterogeneous morphs.
        """
        for entity in self._entities:
            # Skip non-heterogeneous entities
            if not entity._enable_heterogeneous:
                continue

            # Get the number of variants for this entity
            n_variants = len(entity.variants_geom_start)

            # Distribute variants across environments using balanced block assignment:
            # - If B >= n_variants: first B/n_variants environments get variant 0, next get variant 1, etc.
            # - If B < n_variants: each environment gets a different variant (some variants unused)
            if self._B >= n_variants:
                base = self._B // n_variants
                extra = self._B % n_variants  # first `extra` chunks get one more
                sizes = np.r_[np.full(extra, base + 1), np.full(n_variants - extra, base)]
                variant_idx = np.repeat(np.arange(n_variants), sizes)
            else:
                # Each environment gets a unique variant; variants beyond B are unused
                variant_idx = np.arange(self._B)

            # Get arrays from entity
            np_geom_start = np.array(entity.variants_geom_start, dtype=gs.np_int)
            np_geom_end = np.array(entity.variants_geom_end, dtype=gs.np_int)
            np_vgeom_start = np.array(entity.variants_vgeom_start, dtype=gs.np_int)
            np_vgeom_end = np.array(entity.variants_vgeom_end, dtype=gs.np_int)

            # Process each link in this heterogeneous entity (currently only single-link supported)
            for link in entity.links:
                i_l = link.idx

                # Build per-env arrays for geom/vgeom ranges
                links_geom_start = np_geom_start[variant_idx]
                links_geom_end = np_geom_end[variant_idx]
                links_vgeom_start = np_vgeom_start[variant_idx]
                links_vgeom_end = np_vgeom_end[variant_idx]

                # Build per-env arrays for inertial properties
                links_inertial_mass = np.array(
                    [entity.variants_inertial_mass[v] for v in variant_idx], dtype=gs.np_float
                )
                links_inertial_pos = np.array([entity.variants_inertial_pos[v] for v in variant_idx], dtype=gs.np_float)
                links_inertial_i = np.array([entity.variants_inertial_i[v] for v in variant_idx], dtype=gs.np_float)

                # Update links_info with per-environment values
                # Note: when batch_links_info is True, the shape is (n_links, B)
                kernel_update_heterogeneous_link_info(
                    i_l,
                    links_geom_start,
                    links_geom_end,
                    links_vgeom_start,
                    links_vgeom_end,
                    links_inertial_mass,
                    links_inertial_pos,
                    links_inertial_i,
                    self.links_info,
                )

                # Update active_envs_idx for geoms and vgeoms - indicates which environments each geom is active in
                for geom in link.geoms:
                    active_envs_mask = (links_geom_start <= geom.idx) & (geom.idx < links_geom_end)
                    geom.active_envs_mask = torch.tensor(active_envs_mask, device=gs.device)
                    (geom.active_envs_idx,) = np.where(active_envs_mask)

                for vgeom in link.vgeoms:
                    active_envs_mask = (links_vgeom_start <= vgeom.idx) & (vgeom.idx < links_vgeom_end)
                    vgeom.active_envs_mask = torch.tensor(active_envs_mask, device=gs.device)
                    (vgeom.active_envs_idx,) = np.where(active_envs_mask)

    def _init_vert_fields(self):
        # # collisioin geom
        self.verts_info = self.data_manager.verts_info
        self.faces_info = self.data_manager.faces_info
        self.edges_info = self.data_manager.edges_info
        self.free_verts_state = self.data_manager.free_verts_state
        self.fixed_verts_state = self.data_manager.fixed_verts_state

        if self.n_verts > 0:
            geoms = self.geoms
            kernel_init_vert_fields(
                verts=np.concatenate([geom.init_verts for geom in geoms], dtype=gs.np_float),
                faces=np.concatenate([geom.init_faces + geom.vert_start for geom in geoms], dtype=gs.np_int),
                edges=np.concatenate([geom.init_edges + geom.vert_start for geom in geoms], dtype=gs.np_int),
                normals=np.concatenate([geom.init_normals for geom in geoms], dtype=gs.np_float),
                verts_geom_idx=np.concatenate([np.full(geom.n_verts, geom.idx) for geom in geoms], dtype=gs.np_int),
                init_center_pos=np.concatenate([geom.init_center_pos for geom in geoms], dtype=gs.np_float),
                verts_state_idx=np.concatenate(
                    [np.arange(geom.verts_state_start, geom.verts_state_start + geom.n_verts) for geom in geoms],
                    dtype=gs.np_int,
                ),
                is_fixed=np.concatenate(
                    [np.full(geom.n_verts, geom.is_fixed and not geom.entity._batch_fixed_verts) for geom in geoms],
                    dtype=gs.np_bool,
                ),
                # Quadrants variables
                verts_info=self.verts_info,
                faces_info=self.faces_info,
                edges_info=self.edges_info,
                static_rigid_sim_config=self._static_rigid_sim_config,
            )

    def _init_vvert_fields(self):
        # visual geom
        self.vverts_info = self.data_manager.vverts_info
        self.vfaces_info = self.data_manager.vfaces_info
        if self.n_vverts > 0:
            vgeoms = self.vgeoms
            kernel_init_vvert_fields(
                vverts=np.concatenate([vgeom.init_vverts for vgeom in vgeoms], dtype=gs.np_float),
                vfaces=np.concatenate([vgeom.init_vfaces + vgeom.vvert_start for vgeom in vgeoms], dtype=gs.np_int),
                vnormals=np.concatenate([vgeom.init_vnormals for vgeom in vgeoms], dtype=gs.np_float),
                vverts_vgeom_idx=np.concatenate(
                    [np.full(vgeom.n_vverts, vgeom.idx) for vgeom in vgeoms], dtype=gs.np_int
                ),
                # Quadrants variables
                vverts_info=self.vverts_info,
                vfaces_info=self.vfaces_info,
                static_rigid_sim_config=self._static_rigid_sim_config,
            )

    def _init_geom_fields(self):
        self.geoms_info: array_class.GeomsInfo = self.data_manager.geoms_info
        self.geoms_state: array_class.GeomsState = self.data_manager.geoms_state
        self.geoms_init_AABB = self._rigid_global_info.geoms_init_AABB
        self._geoms_render_T = np.empty((self.n_geoms_, self._B, 4, 4), dtype=np.float32)

        if self.n_geoms > 0:
            # Make sure that the constraints parameters are valid
            geoms = self.geoms
            geoms_sol_params = np.array([geom.sol_params for geom in geoms], dtype=gs.np_float)
            _sanitize_sol_params(geoms_sol_params, self._sol_min_timeconst, self._sol_default_timeconst)

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

            kernel_init_geom_fields(
                geoms_pos=np.array([geom.init_pos for geom in geoms], dtype=gs.np_float),
                geoms_center=np.array(geoms_center, dtype=gs.np_float),
                geoms_quat=np.array([geom.init_quat for geom in geoms], dtype=gs.np_float),
                geoms_link_idx=np.array([geom.link.idx for geom in geoms], dtype=gs.np_int),
                geoms_type=np.array([geom.type for geom in geoms], dtype=gs.np_int),
                geoms_friction=np.array([geom.friction for geom in geoms], dtype=gs.np_float),
                geoms_sol_params=geoms_sol_params,
                geoms_vert_start=np.array([geom.vert_start for geom in geoms], dtype=gs.np_int),
                geoms_face_start=np.array([geom.face_start for geom in geoms], dtype=gs.np_int),
                geoms_edge_start=np.array([geom.edge_start for geom in geoms], dtype=gs.np_int),
                geoms_verts_state_start=np.array([geom.verts_state_start for geom in geoms], dtype=gs.np_int),
                geoms_vert_end=np.array([geom.vert_end for geom in geoms], dtype=gs.np_int),
                geoms_face_end=np.array([geom.face_end for geom in geoms], dtype=gs.np_int),
                geoms_edge_end=np.array([geom.edge_end for geom in geoms], dtype=gs.np_int),
                geoms_verts_state_end=np.array([geom.verts_state_end for geom in geoms], dtype=gs.np_int),
                geoms_data=np.array([geom.data for geom in geoms], dtype=gs.np_float),
                geoms_is_convex=np.array([geom.is_convex for geom in geoms], dtype=gs.np_bool),
                geoms_needs_coup=np.array([geom.needs_coup for geom in geoms], dtype=gs.np_int),
                geoms_contype=np.array([geom.contype for geom in geoms], dtype=np.int32),
                geoms_conaffinity=np.array([geom.conaffinity for geom in geoms], dtype=np.int32),
                geoms_coup_softness=np.array([geom.coup_softness for geom in geoms], dtype=gs.np_float),
                geoms_coup_friction=np.array([geom.coup_friction for geom in geoms], dtype=gs.np_float),
                geoms_coup_restitution=np.array([geom.coup_restitution for geom in geoms], dtype=gs.np_float),
                geoms_is_fixed=np.array([geom.is_fixed for geom in geoms], dtype=gs.np_bool),
                geoms_is_decomp=np.array([geom.metadata.get("decomposed", False) for geom in geoms], dtype=gs.np_bool),
                # Quadrants variables
                geoms_info=self.geoms_info,
                geoms_state=self.geoms_state,
                verts_info=self.verts_info,
                geoms_init_AABB=self.geoms_init_AABB,
                static_rigid_sim_config=self._static_rigid_sim_config,
            )

    def _init_vgeom_fields(self):
        self.vgeoms_info: array_class.VGeomsInfo = self.data_manager.vgeoms_info
        self.vgeoms_state: array_class.VGeomsState = self.data_manager.vgeoms_state
        self._vgeoms_render_T = np.empty((self.n_vgeoms_, self._B, 4, 4), dtype=np.float32)

        if self.n_vgeoms > 0:
            vgeoms = self.vgeoms
            kernel_init_vgeom_fields(
                vgeoms_pos=np.array([vgeom.init_pos for vgeom in vgeoms], dtype=gs.np_float),
                vgeoms_quat=np.array([vgeom.init_quat for vgeom in vgeoms], dtype=gs.np_float),
                vgeoms_link_idx=np.array([vgeom.link.idx for vgeom in vgeoms], dtype=gs.np_int),
                vgeoms_vvert_start=np.array([vgeom.vvert_start for vgeom in vgeoms], dtype=gs.np_int),
                vgeoms_vface_start=np.array([vgeom.vface_start for vgeom in vgeoms], dtype=gs.np_int),
                vgeoms_vvert_end=np.array([vgeom.vvert_end for vgeom in vgeoms], dtype=gs.np_int),
                vgeoms_vface_end=np.array([vgeom.vface_end for vgeom in vgeoms], dtype=gs.np_int),
                vgeoms_color=np.array([vgeom._color for vgeom in vgeoms], dtype=gs.np_float),
                # Quadrants variables
                vgeoms_info=self.vgeoms_info,
                static_rigid_sim_config=self._static_rigid_sim_config,
            )

    def _init_entity_fields(self):
        self.entities_info = self.data_manager.entities_info
        self.entities_state = self.data_manager.entities_state

        if self._entities:
            entities = self._entities
            kernel_init_entity_fields(
                entities_dof_start=np.array([entity.dof_start for entity in entities], dtype=gs.np_int),
                entities_dof_end=np.array([entity.dof_end for entity in entities], dtype=gs.np_int),
                entities_link_start=np.array([entity.link_start for entity in entities], dtype=gs.np_int),
                entities_link_end=np.array([entity.link_end for entity in entities], dtype=gs.np_int),
                entities_geom_start=np.array([entity.geom_start for entity in entities], dtype=gs.np_int),
                entities_geom_end=np.array([entity.geom_end for entity in entities], dtype=gs.np_int),
                entities_gravity_compensation=np.array(
                    [entity.gravity_compensation for entity in entities], dtype=gs.np_float
                ),
                entities_is_local_collision_mask=np.array(
                    [entity.is_local_collision_mask for entity in entities], dtype=gs.np_bool
                ),
                # Quadrants variables
                entities_info=self.entities_info,
                entities_state=self.entities_state,
                links_info=self.links_info,
                dofs_info=self.dofs_info,
                rigid_global_info=self._rigid_global_info,
                static_rigid_sim_config=self._static_rigid_sim_config,
            )

    def _init_equality_fields(self):
        self.equalities_info = self.data_manager.equalities_info
        if self.n_equalities > 0:
            equalities = self.equalities

            equalities_sol_params = np.array([equality.sol_params for equality in equalities], dtype=gs.np_float)
            _sanitize_sol_params(equalities_sol_params, self._sol_min_timeconst, self._sol_default_timeconst)

            kernel_init_equality_fields(
                equalities_type=np.array([equality.type for equality in equalities], dtype=gs.np_int),
                equalities_eq_obj1id=np.array([equality.eq_obj1id for equality in equalities], dtype=gs.np_int),
                equalities_eq_obj2id=np.array([equality.eq_obj2id for equality in equalities], dtype=gs.np_int),
                equalities_eq_data=np.array([equality.eq_data for equality in equalities], dtype=gs.np_float),
                equalities_eq_type=np.array([equality.type for equality in equalities], dtype=gs.np_int),
                equalities_sol_params=equalities_sol_params,
                # Quadrants variables
                equalities_info=self.equalities_info,
                static_rigid_sim_config=self._static_rigid_sim_config,
            )
            if self._use_contact_island:
                gs.logger.warn("contact island is not supported for equality constraints yet")

    def _init_envs_offset(self):
        self.envs_offset = self._rigid_global_info.envs_offset
        self.envs_offset.from_numpy(self._scene.envs_offset)

    def _init_collider(self):
        self.collider = Collider(self)

        if self.collider._collider_static_config.has_terrain:
            link_idx_ = next(
                i for i, _type in enumerate(qd_to_numpy(self.geoms_info.type)) if _type == gs.GEOM_TYPE.TERRAIN
            )
            link_idx = qd_to_numpy(self.geoms_info.link_idx, link_idx_, keepdim=False)
            entity_idx = qd_to_numpy(self.links_info.entity_idx, link_idx, keepdim=False)
            if self._options.batch_links_info:
                entity_idx = entity_idx[0]
            entity = self._entities[entity_idx]

            scale = np.asarray(entity.terrain_scale, dtype=gs.np_float)
            rc = np.array(entity.terrain_hf.shape, dtype=gs.np_int)
            hf = entity.terrain_hf.astype(gs.np_float, copy=False) * scale[1]
            xyz_maxmin = np.array(
                [rc[0] * scale[0], rc[1] * scale[0], hf.max(), 0, 0, hf.min() - 1.0],
                dtype=gs.np_float,
            )

            self.terrain_hf = qd.field(dtype=gs.qd_float, shape=hf.shape)
            self.terrain_rc = qd.field(dtype=gs.qd_int, shape=(2,))
            self.terrain_scale = qd.field(dtype=gs.qd_float, shape=(2,))
            self.terrain_xyz_maxmin = qd.field(dtype=gs.qd_float, shape=(6,))

            self.terrain_hf.from_numpy(hf)
            self.terrain_rc.from_numpy(rc)
            self.terrain_scale.from_numpy(scale)
            self.terrain_xyz_maxmin.from_numpy(xyz_maxmin)

    def _init_constraint_solver(self):
        if self._use_contact_island:
            self.constraint_solver = ConstraintSolverIsland(self)
        else:
            self.constraint_solver = ConstraintSolver(self)

    def substep(self, f):
        # from genesis.utils.tools import create_timer
        from genesis.engine.couplers import SAPCoupler

        if self._requires_grad and f == 0:
            kernel_save_adjoint_cache(
                f=f,
                dofs_state=self.dofs_state,
                rigid_global_info=self._rigid_global_info,
                rigid_adjoint_cache=self._rigid_adjoint_cache,
                static_rigid_sim_config=self._static_rigid_sim_config,
            )

        kernel_step_1(
            self.links_state,
            self.links_info,
            self.joints_state,
            self.joints_info,
            self.dofs_state,
            self.dofs_info,
            self.geoms_state,
            self.geoms_info,
            self.entities_state,
            self.entities_info,
            self._rigid_global_info,
            self._static_rigid_sim_config,
            self.constraint_solver.contact_island.contact_island_state,
            self._is_forward_pos_updated,
            self._is_forward_vel_updated,
            self._is_backward,
        )

        if isinstance(self.sim.coupler, SAPCoupler):
            update_qvel(
                self.dofs_state,
                self._rigid_global_info,
                self._static_rigid_sim_config,
                self._is_backward,
            )
        else:
            self._func_constraint_force()
            kernel_step_2(
                self.dofs_state,
                self.dofs_info,
                self.links_info,
                self.links_state,
                self.joints_info,
                self.joints_state,
                self.entities_state,
                self.entities_info,
                self.geoms_info,
                self.geoms_state,
                self.collider._collider_state,
                self._rigid_global_info,
                self._static_rigid_sim_config,
                self.constraint_solver.contact_island.contact_island_state,
                self._is_backward,
                self._errno,
            )
            self._is_forward_pos_updated = not self._enable_mujoco_compatibility
            self._is_forward_vel_updated = not self._enable_mujoco_compatibility
            if self._requires_grad:
                kernel_save_adjoint_cache(
                    f + 1,
                    self.dofs_state,
                    self._rigid_global_info,
                    self._rigid_adjoint_cache,
                    self._static_rigid_sim_config,
                )

    def get_error_envs_mask(self):
        return qd_to_torch(self._errno) > 0

    def check_errno(self):
        # TODO: Add some class ErrorCode(IntEnum) to manage error codes x)
        if gs.use_zerocopy:
            errno = np.bitwise_or.reduce(qd_to_numpy(self._errno))
        else:
            errno = kernel_bit_reduction(self._errno)

        if errno & array_class.ErrorCode.OVERFLOW_CANDIDATE_CONTACTS:
            max_collision_pairs_broad = self.collider._collider_info.max_collision_pairs_broad[None]
            gs.raise_exception(
                f"Exceeding max number of broad phase candidate contact pairs ({max_collision_pairs_broad}). "
                f"Please increase the value of RigidSolver's option 'multiplier_collision_broad_phase'."
            )
        if errno & array_class.ErrorCode.OVERFLOW_COLLISION_PAIRS:
            max_contact_pairs = self.collider._collider_info.max_contact_pairs[None]
            gs.raise_exception(
                f"Exceeding max number of contact pairs ({max_contact_pairs}). Please increase the value of "
                "RigidSolver's option 'max_collision_pairs'."
            )
        if errno & array_class.ErrorCode.INVALID_FORCE_NAN:
            gs.raise_exception("Invalid constraint forces causing 'nan'. Please decrease Rigid simulation timestep.")
        if errno & array_class.ErrorCode.INVALID_ACC_NAN:
            gs.raise_exception("Invalid accelerations causing 'nan'. Please decrease Rigid simulation timestep.")
        if errno & array_class.ErrorCode.OVERFLOW_HIBERNATION_ISLANDS:
            gs.raise_exception("Contact island buffer overflow. Please increase RigidOptions 'max_collision_pairs'.")

    def _kernel_detect_collision(self):
        self.collider.clear()
        self.collider.detection()

    def detect_collision(self, env_idx=0):
        # TODO: support batching
        self._kernel_detect_collision()

        n_collision = qd_to_numpy(self.collider._collider_state.n_contacts)[env_idx]
        collision_pairs = np.empty((n_collision, 2), dtype=np.int32)
        collision_pairs[:, 0] = qd_to_numpy(self.collider._collider_state.contact_data.geom_a)[:n_collision, env_idx]
        collision_pairs[:, 1] = qd_to_numpy(self.collider._collider_state.contact_data.geom_b)[:n_collision, env_idx]

        return collision_pairs

    def _func_constraint_force(self):
        if not self._disable_constraint:
            if self._use_contact_island:
                self.constraint_solver.clear()
            else:
                self.constraint_solver.add_equality_constraints()

        if self._enable_collision:
            self.collider.detection()

        if not self._disable_constraint:
            if self._use_contact_island:
                self.constraint_solver.add_constraints()
            else:
                self.constraint_solver.add_inequality_constraints()

            self.constraint_solver.resolve()

    def _func_forward_dynamics(self):
        kernel_forward_dynamics(
            self.links_state,
            self.links_info,
            self.dofs_state,
            self.dofs_info,
            self.joints_info,
            self.entities_state,
            self.entities_info,
            self.geoms_state,
            self._rigid_global_info,
            self._static_rigid_sim_config,
            self.constraint_solver.contact_island.contact_island_state,
        )

    def _func_update_acc(self):
        kernel_update_acc(
            self.dofs_state,
            self.links_info,
            self.links_state,
            self.entities_info,
            self._rigid_global_info,
            self._static_rigid_sim_config,
        )

    def _func_forward_kinematics_entity(self, i_e, envs_idx):
        kernel_forward_kinematics_entity(
            i_e,
            envs_idx,
            self.links_state,
            self.links_info,
            self.joints_state,
            self.joints_info,
            self.dofs_state,
            self.dofs_info,
            self.entities_info,
            self._rigid_global_info,
            self._static_rigid_sim_config,
        )

    def _func_integrate_dq_entity(self, dq, i_e, i_b, respect_joint_limit):
        func_integrate_dq_entity(
            dq,
            i_e,
            i_b,
            respect_joint_limit,
            links_info=self.links_info,
            joints_info=self.joints_info,
            dofs_info=self.dofs_info,
            entities_info=self.entities_info,
            rigid_global_info=self._rigid_global_info,
            static_rigid_sim_config=self._static_rigid_sim_config,
        )

    def _func_update_geoms(self, envs_idx, *, force_update_fixed_geoms=False):
        kernel_update_geoms(
            envs_idx,
            self.entities_info,
            self.geoms_info,
            self.geoms_state,
            self.links_state,
            self._rigid_global_info,
            self._static_rigid_sim_config,
            force_update_fixed_geoms,
        )

    def apply_links_external_force(
        self,
        force,
        links_idx=None,
        envs_idx=None,
        *,
        ref: Literal["link_origin", "link_com", "root_com"] = "link_origin",
        local: bool = False,
    ):
        """
        Apply some external linear force on a set of links.

        Parameters
        ----------
        force : array_like
            The force to apply.
        links_idx : None | array_like, optional
            The indices of the links on which to apply force. None to specify all links. Default to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        ref: "link_origin" | "link_com" | "root_com", optional
            The reference frame on which the linear force will be applied. "link_origin" refers to the origin of the
            link, "link_com" refers to the center of mass of the link, and "root_com" refers to the center of mass of
            the entire kinematic tree to which a link belong (see `get_links_root_COM` for details).
        local: bool, optional
            Whether the force is expressed in the local coordinates associated with the reference frame instead of
            world frame. Only supported for `ref="link_origin"` or `ref="link_com"`.
        """
        force, links_idx, envs_idx = self._sanitize_io_variables(
            force, links_idx, self.n_links, "links_idx", envs_idx, (3,), skip_allocation=True
        )
        if self.n_envs == 0:
            force = force[None]

        if ref == "root_com":
            if local:
                raise ValueError("'local=True' not compatible with ref='root_com'.")
            ref = 0
        elif ref == "link_com":
            ref = 1
        elif ref == "link_origin":
            ref = 2
        else:
            raise ValueError("'ref' must be either 'link_origin', 'link_com', or 'root_com'.")

        kernel_apply_links_external_force(
            force, links_idx, envs_idx, ref, 1 if local else 0, self.links_state, self._static_rigid_sim_config
        )

    def apply_links_external_torque(
        self,
        torque,
        links_idx=None,
        envs_idx=None,
        *,
        ref: Literal["link_origin", "link_com", "root_com"] = "link_origin",
        local: bool = False,
    ):
        """
        Apply some external torque on a set of links.

        Parameters
        ----------
        torque : array_like
            The torque to apply.
        links_idx : None | array_like, optional
            The indices of the links on which to apply torque. None to specify all links. Default to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        ref: "link_origin" | "link_com" | "root_com", optional
            The reference frame on which the torque will be applied. "link_origin" refers to the origin of the link,
            "link_com" refers to the center of mass of the link, and "root_com" refers to the center of mass of
            the entire kinematic tree to which a link belong (see `get_links_root_COM` for details). Note that this
            argument has no effect unless `local=True`.
        local: bool, optional
            Whether the torque is expressed in the local coordinates associated with the reference frame instead of
            world frame. Only supported for `ref="link_origin"` or `ref="link_com"`.
        """
        torque, links_idx, envs_idx = self._sanitize_io_variables(
            torque, links_idx, self.n_links, "links_idx", envs_idx, (3,), skip_allocation=True
        )
        if self.n_envs == 0:
            torque = torque[None]

        if ref == "root_com":
            if local:
                raise ValueError("'local=True' not compatible with ref='root_com'.")
            ref = 0
        elif ref == "link_com":
            ref = 1
        elif ref == "link_origin":
            ref = 2
        else:
            raise ValueError("'ref' must be either 'link_origin', 'link_com', or 'root_com'.")

        kernel_apply_links_external_torque(
            torque, links_idx, envs_idx, ref, 1 if local else 0, self.links_state, self._static_rigid_sim_config
        )

    def substep_pre_coupling(self, f):
        if self.is_active:
            # Skip rigid body computation when using IPCCoupler (IPC handles rigid simulation)
            from genesis.engine.couplers import IPCCoupler

            if isinstance(self.sim.coupler, IPCCoupler):
                # If any rigid entity is coupled to IPC, skip pre-coupling rigid simulation
                # The rigid simulation will be done in post-coupling phase instead
                if self.sim.coupler.has_any_rigid_coupling:
                    return

            # Run Genesis rigid simulation step for non-IPC couplers
            self.substep(f)

    def substep_pre_coupling_grad(self, f):
        # Change to backward mode
        self._is_backward = True

        # Run forward substep again to restore this step's information, this is needed because we do not store info
        # of every substep.
        kernel_prepare_backward_substep(
            f=f,
            links_state=self.links_state,
            links_info=self.links_info,
            joints_state=self.joints_state,
            joints_info=self.joints_info,
            dofs_state=self.dofs_state,
            dofs_info=self.dofs_info,
            geoms_state=self.geoms_state,
            geoms_info=self.geoms_info,
            entities_info=self.entities_info,
            rigid_global_info=self._rigid_global_info,
            dofs_state_adjoint_cache=self.dofs_state_adjoint_cache,
            links_state_adjoint_cache=self.links_state_adjoint_cache,
            joints_state_adjoint_cache=self.joints_state_adjoint_cache,
            geoms_state_adjoint_cache=self.geoms_state_adjoint_cache,
            rigid_adjoint_cache=self._rigid_adjoint_cache,
            static_rigid_sim_config=self._static_rigid_sim_config,
        )
        self.substep(f)

        # =================== Backward substep ======================
        envs_idx = self._scene._sanitize_envs_idx(None)
        if not self._enable_mujoco_compatibility:
            kernel_forward_velocity.grad(
                envs_idx=envs_idx,
                links_state=self.links_state,
                links_info=self.links_info,
                joints_info=self.joints_info,
                dofs_state=self.dofs_state,
                entities_info=self.entities_info,
                rigid_global_info=self._rigid_global_info,
                static_rigid_sim_config=self._static_rigid_sim_config,
                is_backward=True,
            )
            kernel_update_cartesian_space.grad(
                links_state=self.links_state,
                links_info=self.links_info,
                joints_state=self.joints_state,
                joints_info=self.joints_info,
                dofs_state=self.dofs_state,
                dofs_info=self.dofs_info,
                geoms_state=self.geoms_state,
                geoms_info=self.geoms_info,
                entities_info=self.entities_info,
                rigid_global_info=self._rigid_global_info,
                static_rigid_sim_config=self._static_rigid_sim_config,
                force_update_fixed_geoms=False,
                is_backward=True,
            )

        is_grad_valid = kernel_begin_backward_substep(
            f=f,
            links_state=self.links_state,
            links_info=self.links_info,
            joints_state=self.joints_state,
            joints_info=self.joints_info,
            dofs_state=self.dofs_state,
            dofs_info=self.dofs_info,
            geoms_state=self.geoms_state,
            geoms_info=self.geoms_info,
            entities_info=self.entities_info,
            rigid_global_info=self._rigid_global_info,
            dofs_state_adjoint_cache=self.dofs_state_adjoint_cache,
            links_state_adjoint_cache=self.links_state_adjoint_cache,
            joints_state_adjoint_cache=self.joints_state_adjoint_cache,
            geoms_state_adjoint_cache=self.geoms_state_adjoint_cache,
            rigid_adjoint_cache=self._rigid_adjoint_cache,
            static_rigid_sim_config=self._static_rigid_sim_config,
        )
        if not is_grad_valid:
            gs.raise_exception(f"Nan grad in qpos or dofs_vel found at step {self._sim.cur_step_global}")

        kernel_step_2.grad(
            dofs_state=self.dofs_state,
            dofs_info=self.dofs_info,
            links_info=self.links_info,
            links_state=self.links_state,
            joints_info=self.joints_info,
            joints_state=self.joints_state,
            entities_state=self.entities_state,
            entities_info=self.entities_info,
            geoms_info=self.geoms_info,
            geoms_state=self.geoms_state,
            collider_state=self.collider._collider_state,
            rigid_global_info=self._rigid_global_info,
            static_rigid_sim_config=self._static_rigid_sim_config,
            contact_island_state=self.constraint_solver.contact_island.contact_island_state,
            is_backward=True,
            errno=self._errno,
        )

        # We cannot use [kernel_forward_dynamics.grad] because we read [dofs_state.acc] and overwrite it in the kernel,
        # which is prohibited (https://docs.taichi-lang.org/docs/differentiable_programming#global-data-access-rules).
        # In [kernel_forward_dynamics], we read [acc] in [func_update_acc] and overwrite it in [kernel_compute_qacc].
        # As [kenrel_compute_qacc] is called at the end of [kernel_forward_dynamics], we first backpropagate through
        # [kernel_compute_qacc] and then restore the original [acc] from the adjoint cache. This copy operation
        # cannot be merged with [kernel_compute_qacc.grad] because .grad function itself is a standalone kernel.
        # We could possibly merge this small kernel later if (1) .grad function is regarded as a function instead of a
        # kernel, (2) we add another variable to store the new [acc] from [kernel_compute_qacc] and thus can avoid
        # the data access violation. However, both of these require major changes.
        kernel_compute_qacc.grad(
            dofs_state=self.dofs_state,
            entities_info=self.entities_info,
            rigid_global_info=self._rigid_global_info,
            static_rigid_sim_config=self._static_rigid_sim_config,
            is_backward=True,
        )
        kernel_copy_acc(
            f=f,
            dofs_state=self.dofs_state,
            rigid_adjoint_cache=self._rigid_adjoint_cache,
            static_rigid_sim_config=self._static_rigid_sim_config,
        )

        kernel_forward_dynamics_without_qacc.grad(
            links_state=self.links_state,
            links_info=self.links_info,
            dofs_state=self.dofs_state,
            dofs_info=self.dofs_info,
            joints_info=self.joints_info,
            entities_state=self.entities_state,
            entities_info=self.entities_info,
            geoms_state=self.geoms_state,
            rigid_global_info=self._rigid_global_info,
            static_rigid_sim_config=self._static_rigid_sim_config,
            contact_island_state=self.constraint_solver.contact_island.contact_island_state,
            is_backward=True,
        )

        # If it was the very first substep, we need to backpropagate through the initial update of the cartesian space
        if self._enable_mujoco_compatibility or self._sim.cur_substep_global == 0:
            kernel_forward_velocity.grad(
                envs_idx=envs_idx,
                links_state=self.links_state,
                links_info=self.links_info,
                joints_info=self.joints_info,
                dofs_state=self.dofs_state,
                entities_info=self.entities_info,
                rigid_global_info=self._rigid_global_info,
                static_rigid_sim_config=self._static_rigid_sim_config,
                is_backward=True,
            )
            kernel_update_cartesian_space.grad(
                links_state=self.links_state,
                links_info=self.links_info,
                joints_state=self.joints_state,
                joints_info=self.joints_info,
                dofs_state=self.dofs_state,
                dofs_info=self.dofs_info,
                geoms_state=self.geoms_state,
                geoms_info=self.geoms_info,
                entities_info=self.entities_info,
                rigid_global_info=self._rigid_global_info,
                static_rigid_sim_config=self._static_rigid_sim_config,
                force_update_fixed_geoms=False,
                is_backward=True,
            )

        # Change back to forward mode
        self._is_backward = False

    def substep_post_coupling(self, f):
        from genesis.engine.couplers import SAPCoupler, IPCCoupler

        if not self.is_active:
            return

        if isinstance(self.sim.coupler, SAPCoupler):
            update_qacc_from_qvel_delta(
                dofs_state=self.dofs_state,
                rigid_global_info=self._rigid_global_info,
                static_rigid_sim_config=self._static_rigid_sim_config,
                is_backward=self._is_backward,
            )
            kernel_step_2(
                dofs_state=self.dofs_state,
                dofs_info=self.dofs_info,
                links_info=self.links_info,
                links_state=self.links_state,
                joints_info=self.joints_info,
                joints_state=self.joints_state,
                entities_state=self.entities_state,
                entities_info=self.entities_info,
                geoms_info=self.geoms_info,
                geoms_state=self.geoms_state,
                collider_state=self.collider._collider_state,
                rigid_global_info=self._rigid_global_info,
                static_rigid_sim_config=self._static_rigid_sim_config,
                contact_island_state=self.constraint_solver.contact_island.contact_island_state,
                is_backward=self._is_backward,
                errno=self._errno,
            )
        elif isinstance(self.sim.coupler, IPCCoupler):
            # If any rigid entity is coupled to IPC, perform rigid simulation in post-coupling phase.
            # Collision exclusion for IPC-coupled links is handled in the collider at build time.
            if self.sim.coupler.has_any_rigid_coupling:
                self.substep(f)

    def substep_post_coupling_grad(self, f):
        pass

    def add_grad_from_state(self, state):
        if self.is_active:
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
                qpos_grad=qpos_grad,
                vel_grad=dofs_vel_grad,
                links_pos_grad=links_pos_grad,
                links_quat_grad=links_quat_grad,
                links_state=self.links_state,
                dofs_state=self.dofs_state,
                geoms_state=self.geoms_state,
                rigid_global_info=self._rigid_global_info,
                static_rigid_sim_config=self._static_rigid_sim_config,
            )

    def collect_output_grads(self):
        """
        Collect gradients from downstream queried states.
        """
        if self._sim.cur_step_global in self._queried_states:
            # one step could have multiple states
            assert len(self._queried_states[self._sim.cur_step_global]) == 1
            state = self._queried_states[self._sim.cur_step_global][0]
            self.add_grad_from_state(state)

    def reset_grad(self):
        for entity in self._entities:
            entity.reset_grad()
        self._queried_states.clear()

    def update_geoms_render_T(self):
        kernel_update_geoms_render_T(
            self._geoms_render_T,
            geoms_state=self.geoms_state,
            rigid_global_info=self._rigid_global_info,
            static_rigid_sim_config=self._static_rigid_sim_config,
        )

    def update_vgeoms_render_T(self):
        kernel_update_vgeoms_render_T(
            self._vgeoms_render_T,
            vgeoms_info=self.vgeoms_info,
            vgeoms_state=self.vgeoms_state,
            links_state=self.links_state,
            rigid_global_info=self._rigid_global_info,
            static_rigid_sim_config=self._static_rigid_sim_config,
        )

    def get_state(self, f=None):
        s_global = self.sim.cur_step_global
        if self.is_active:
            if s_global in self._queried_states:
                return self._queried_states[s_global][0]

            state = RigidSolverState(self._scene, s_global)

            kernel_get_state(
                qpos=state.qpos,
                vel=state.dofs_vel,
                acc=state.dofs_acc,
                links_pos=state.links_pos,
                links_quat=state.links_quat,
                i_pos_shift=state.i_pos_shift,
                mass_shift=state.mass_shift,
                friction_ratio=state.friction_ratio,
                links_state=self.links_state,
                dofs_state=self.dofs_state,
                geoms_state=self.geoms_state,
                rigid_global_info=self._rigid_global_info,
                static_rigid_sim_config=self._static_rigid_sim_config,
            )
            self._queried_states.append(state)
        else:
            state = None
        return state

    def set_state(self, f, state, envs_idx=None):
        if self.is_active:
            envs_idx = self._scene._sanitize_envs_idx(envs_idx)

            if gs.use_zerocopy:
                errno = qd_to_torch(self._errno, copy=False)
                errno[envs_idx] = 0
            else:
                kernel_set_zero(envs_idx, self._errno)

            kernel_set_state(
                qpos=state.qpos,
                dofs_vel=state.dofs_vel,
                dofs_acc=state.dofs_acc,
                links_pos=state.links_pos,
                links_quat=state.links_quat,
                i_pos_shift=state.i_pos_shift,
                mass_shift=state.mass_shift,
                friction_ratio=state.friction_ratio,
                envs_idx=envs_idx,
                links_state=self.links_state,
                dofs_state=self.dofs_state,
                geoms_state=self.geoms_state,
                rigid_global_info=self._rigid_global_info,
                static_rigid_sim_config=self._static_rigid_sim_config,
            )
            kernel_forward_kinematics_links_geoms(
                envs_idx,
                links_state=self.links_state,
                links_info=self.links_info,
                joints_state=self.joints_state,
                joints_info=self.joints_info,
                dofs_state=self.dofs_state,
                dofs_info=self.dofs_info,
                geoms_state=self.geoms_state,
                geoms_info=self.geoms_info,
                entities_info=self.entities_info,
                rigid_global_info=self._rigid_global_info,
                static_rigid_sim_config=self._static_rigid_sim_config,
            )
            self._is_forward_pos_updated = True
            self._is_forward_vel_updated = True

            self.collider.clear(envs_idx)
            self.constraint_solver.clear(envs_idx)

            for entity in self.entities:
                if isinstance(entity, DroneEntity):
                    entity._prev_prop_t = -1

    def process_input(self, in_backward=False):
        for entity in self._entities:
            entity.process_input(in_backward=in_backward)

    def process_input_grad(self):
        for entity in self._entities:
            entity.process_input_grad()

    def save_ckpt(self, ckpt_name):
        # Save ckpt only if we need gradients, because this operation is costly
        if self._requires_grad:
            if ckpt_name not in self._ckpt:
                self._ckpt[ckpt_name] = dict()

            self._ckpt[ckpt_name]["qpos"] = qd_to_numpy(self._rigid_adjoint_cache.qpos)
            self._ckpt[ckpt_name]["dofs_vel"] = qd_to_numpy(self._rigid_adjoint_cache.dofs_vel)
            self._ckpt[ckpt_name]["dofs_acc"] = qd_to_numpy(self._rigid_adjoint_cache.dofs_acc)

            for entity in self._entities:
                entity.save_ckpt(ckpt_name)

    def load_ckpt(self, ckpt_name):
        # Set first frame
        self._rigid_global_info.qpos.from_numpy(self._ckpt[ckpt_name]["qpos"][0])
        self.dofs_state.vel.from_numpy(self._ckpt[ckpt_name]["dofs_vel"][0])
        self.dofs_state.acc.from_numpy(self._ckpt[ckpt_name]["dofs_acc"][0])

        if not self._enable_mujoco_compatibility:
            kernel_update_cartesian_space(
                links_state=self.links_state,
                links_info=self.links_info,
                joints_state=self.joints_state,
                joints_info=self.joints_info,
                dofs_state=self.dofs_state,
                dofs_info=self.dofs_info,
                geoms_state=self.geoms_state,
                geoms_info=self.geoms_info,
                entities_info=self.entities_info,
                rigid_global_info=self._rigid_global_info,
                static_rigid_sim_config=self._static_rigid_sim_config,
                force_update_fixed_geoms=False,
                is_backward=False,
            )

        for entity in self._entities:
            entity.load_ckpt(ckpt_name)

    @property
    def is_active(self):
        return self.n_links > 0

    # ------------------------------------------------------------------------------------
    # ------------------------------------ control ---------------------------------------
    # ------------------------------------------------------------------------------------

    def _sanitize_io_variables(
        self,
        tensor: "np.typing.ArrayLike | None",
        inputs_idx: int | range | slice | tuple[int, ...] | list[int] | torch.Tensor | np.ndarray | None,
        input_size: int,
        idx_name: str,
        envs_idx: int | range | slice | tuple[int, ...] | list[int] | torch.Tensor | np.ndarray | None = None,
        element_shape: tuple[int, ...] | list[int] = (),
        *,
        batched: bool = True,
        skip_allocation: bool = False,
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        # Handling default arguments
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

    def set_links_pos(self, pos, links_idx=None, envs_idx=None):
        raise DeprecationError("This method has been removed. Please use 'set_base_links_pos' instead.")

    def set_base_links_pos(self, pos, links_idx=None, envs_idx=None, *, relative=False):
        if links_idx is None:
            links_idx = self._base_links_idx
        pos, links_idx, envs_idx = self._sanitize_io_variables(
            pos, links_idx, self.n_links, "links_idx", envs_idx, (3,), skip_allocation=True
        )
        if self.n_envs == 0:
            pos = pos[None]

        # FIXME: This check is too expensive
        # if not torch.isin(links_idx, self._base_links_idx).all():
        #     gs.raise_exception("`links_idx` contains at least one link that is not a base link.")

        # Raise exception for fixed links with at least one geom and non-batched fixed vertices, except if setting same
        # location for all envs at once
        set_all_envs = torch.equal(torch.sort(envs_idx).values, self._scene._envs_idx)
        has_fixed_verts = any(
            link.is_fixed and (link.geoms or link.vgeoms) and not link.entity._batch_fixed_verts
            for link in (self.links[i_l] for i_l in links_idx)
        )
        if has_fixed_verts and not (set_all_envs and (torch.diff(pos, dim=0).abs() < gs.EPS).all()):
            gs.raise_exception(
                "Specifying env-specific pos for fixed links with at least one geometry requires setting morph "
                "option 'batch_fixed_verts=True'."
            )

        # Wake up hibernated entities before setting position
        if self._options.use_hibernation:
            kernel_wake_up_entities_by_links(
                links_idx,
                envs_idx,
                links_info=self.links_info,
                links_state=self.links_state,
                entities_state=self.entities_state,
                entities_info=self.entities_info,
                dofs_state=self.dofs_state,
                geoms_state=self.geoms_state,
                rigid_global_info=self._rigid_global_info,
                static_rigid_sim_config=self._static_rigid_sim_config,
            )

        kernel_set_links_pos(
            relative,
            pos,
            links_idx,
            envs_idx,
            links_info=self.links_info,
            links_state=self.links_state,
            rigid_global_info=self._rigid_global_info,
            static_rigid_sim_config=self._static_rigid_sim_config,
        )

        kernel_forward_kinematics_links_geoms(
            envs_idx,
            links_state=self.links_state,
            links_info=self.links_info,
            joints_state=self.joints_state,
            joints_info=self.joints_info,
            dofs_state=self.dofs_state,
            dofs_info=self.dofs_info,
            geoms_state=self.geoms_state,
            geoms_info=self.geoms_info,
            entities_info=self.entities_info,
            rigid_global_info=self._rigid_global_info,
            static_rigid_sim_config=self._static_rigid_sim_config,
        )
        self._is_forward_pos_updated = True
        self._is_forward_vel_updated = True

    def set_base_links_pos_grad(self, links_idx, envs_idx, relative, pos_grad):
        if links_idx is None:
            links_idx = self._base_links_idx
        pos_grad_, links_idx, envs_idx = self._sanitize_io_variables(
            pos_grad.unsqueeze(-2), links_idx, self.n_links, "links_idx", envs_idx, (3,), skip_allocation=True
        )
        if self.n_envs == 0:
            pos_grad_ = pos_grad_.unsqueeze(0)
        kernel_set_links_pos_grad(
            relative,
            pos_grad_,
            links_idx,
            envs_idx,
            links_info=self.links_info,
            links_state=self.links_state,
            rigid_global_info=self._rigid_global_info,
            static_rigid_sim_config=self._static_rigid_sim_config,
        )

    def set_links_quat(self, quat, links_idx=None, envs_idx=None):
        raise DeprecationError("This method has been removed. Please use 'set_base_links_quat' instead.")

    def set_base_links_quat(self, quat, links_idx=None, envs_idx=None, *, relative=False):
        if links_idx is None:
            links_idx = self._base_links_idx
        quat, links_idx, envs_idx = self._sanitize_io_variables(
            quat, links_idx, self.n_links, "links_idx", envs_idx, (4,), skip_allocation=True
        )
        if self.n_envs == 0:
            quat = quat[None]

        # FIXME: This check is too expensive
        # if not torch.isin(links_idx, self._base_links_idx).all():
        #     gs.raise_exception("`links_idx` contains at least one link that is not a base link.")

        set_all_envs = torch.equal(torch.sort(envs_idx).values, self._scene._envs_idx)
        has_fixed_verts = any(
            link.is_fixed and (link.geoms or link.vgeoms) and not link.entity._batch_fixed_verts
            for link in (self.links[i_l] for i_l in links_idx)
        )
        if has_fixed_verts and not (set_all_envs and (torch.diff(quat, dim=0).abs() < gs.EPS).all()):
            gs.raise_exception("Impossible to set env-specific quat for fixed links with at least one geometry.")

        # Wake up hibernated entities before setting quaternion
        if self._options.use_hibernation:
            kernel_wake_up_entities_by_links(
                links_idx,
                envs_idx,
                links_info=self.links_info,
                links_state=self.links_state,
                entities_state=self.entities_state,
                entities_info=self.entities_info,
                dofs_state=self.dofs_state,
                geoms_state=self.geoms_state,
                rigid_global_info=self._rigid_global_info,
                static_rigid_sim_config=self._static_rigid_sim_config,
            )

        kernel_set_links_quat(
            relative,
            quat,
            links_idx,
            envs_idx,
            links_info=self.links_info,
            links_state=self.links_state,
            rigid_global_info=self._rigid_global_info,
            static_rigid_sim_config=self._static_rigid_sim_config,
        )

        kernel_forward_kinematics_links_geoms(
            envs_idx,
            links_state=self.links_state,
            links_info=self.links_info,
            joints_state=self.joints_state,
            joints_info=self.joints_info,
            dofs_state=self.dofs_state,
            dofs_info=self.dofs_info,
            geoms_state=self.geoms_state,
            geoms_info=self.geoms_info,
            entities_info=self.entities_info,
            rigid_global_info=self._rigid_global_info,
            static_rigid_sim_config=self._static_rigid_sim_config,
        )
        self._is_forward_pos_updated = True
        self._is_forward_vel_updated = True

    def set_base_links_quat_grad(self, links_idx, envs_idx, relative, quat_grad):
        if links_idx is None:
            links_idx = self._base_links_idx
        quat_grad_, links_idx, envs_idx = self._sanitize_io_variables(
            quat_grad.unsqueeze(-2), links_idx, self.n_links, "links_idx", envs_idx, (4,), skip_allocation=True
        )
        if self.n_envs == 0:
            quat_grad_ = quat_grad_.unsqueeze(0)
        assert relative == False, "Backward pass for relative quaternion is not supported yet."
        kernel_set_links_quat_grad(
            relative,
            quat_grad_,
            links_idx,
            envs_idx,
            links_info=self.links_info,
            links_state=self.links_state,
            rigid_global_info=self._rigid_global_info,
            static_rigid_sim_config=self._static_rigid_sim_config,
        )

    def set_links_mass_shift(self, mass, links_idx=None, envs_idx=None):
        mass, links_idx, envs_idx = self._sanitize_io_variables(
            mass, links_idx, self.n_links, "links_idx", envs_idx, skip_allocation=True
        )
        if self.n_envs == 0:
            mass = mass[None]
        kernel_set_links_mass_shift(
            mass,
            links_idx,
            envs_idx,
            links_state=self.links_state,
            static_rigid_sim_config=self._static_rigid_sim_config,
        )

    def set_links_COM_shift(self, com, links_idx=None, envs_idx=None):
        com, links_idx, envs_idx = self._sanitize_io_variables(
            com, links_idx, self.n_links, "links_idx", envs_idx, (3,), skip_allocation=True
        )
        if self.n_envs == 0:
            com = com[None]
        kernel_set_links_COM_shift(com, links_idx, envs_idx, self.links_state, self._static_rigid_sim_config)

    def set_links_inertial_mass(self, mass, links_idx=None, envs_idx=None):
        mass, links_idx, envs_idx = self._sanitize_io_variables(
            mass,
            links_idx,
            self.n_links,
            "links_idx",
            envs_idx,
            batched=self._options.batch_links_info,
            skip_allocation=True,
        )
        if self.n_envs == 0 and self._options.batch_links_info:
            mass = mass[None]
        kernel_set_links_inertial_mass(mass, links_idx, envs_idx, self.links_info, self._static_rigid_sim_config)

    def set_geoms_friction_ratio(self, friction_ratio, geoms_idx=None, envs_idx=None):
        friction_ratio, geoms_idx, envs_idx = self._sanitize_io_variables(
            friction_ratio, geoms_idx, self.n_geoms, "geoms_idx", envs_idx, skip_allocation=True
        )
        if self.n_envs == 0:
            friction_ratio = friction_ratio[None]
        kernel_set_geoms_friction_ratio(
            friction_ratio, geoms_idx, envs_idx, self.geoms_state, self._static_rigid_sim_config
        )

    def set_qpos(self, qpos, qs_idx=None, envs_idx=None, *, skip_forward=False):
        if gs.use_zerocopy:
            data = qd_to_torch(self._rigid_global_info.qpos, transpose=True, copy=False)
            errno = qd_to_torch(self._errno, copy=False)
            qs_mask = indices_to_mask(qs_idx)
            if (
                (not qs_mask or isinstance(qs_mask[0], slice))
                and isinstance(envs_idx, torch.Tensor)
                and envs_idx.dtype == torch.bool
            ):
                qs_data = data[(slice(None), *qs_mask)]
                if qpos.ndim == 2:
                    # Note that it is necessary to create a new temporary view because it will be modified in-place
                    qs_data.masked_scatter_(envs_idx[:, None], qpos.view_as(qpos))
                else:
                    qpos = broadcast_tensor(qpos, gs.tc_float, qs_data.shape)
                    torch.where(envs_idx[:, None], qpos, qs_data, out=qs_data)
                errno.masked_fill_(envs_idx, 0.0)
            else:
                mask = (0, *qs_mask) if self.n_envs == 0 else indices_to_mask(envs_idx, *qs_mask)
                assign_indexed_tensor(data, mask, qpos)
                errno[envs_idx] = 0
                if mask and isinstance(mask[0], torch.Tensor):
                    envs_idx = mask[0].reshape((-1,))
        else:
            qpos, qs_idx, envs_idx = self._sanitize_io_variables(
                qpos, qs_idx, self.n_qs, "qs_idx", envs_idx, skip_allocation=True
            )
            if self.n_envs == 0:
                qpos = qpos[None]
            kernel_set_qpos(qpos, qs_idx, envs_idx, self._rigid_global_info, self._static_rigid_sim_config)
            kernel_set_zero(envs_idx, self._errno)

        if self.collider is not None:
            self.collider.reset(envs_idx)
        if self.constraint_solver is not None:
            self.constraint_solver.reset(envs_idx)

        if not skip_forward:
            if not isinstance(envs_idx, torch.Tensor):
                envs_idx = self._scene._sanitize_envs_idx(envs_idx)
            if envs_idx.dtype == torch.bool:
                fn = kernel_masked_forward_kinematics_links_geoms
            else:
                fn = kernel_forward_kinematics_links_geoms
            fn(
                envs_idx,
                links_state=self.links_state,
                links_info=self.links_info,
                joints_state=self.joints_state,
                joints_info=self.joints_info,
                dofs_state=self.dofs_state,
                dofs_info=self.dofs_info,
                geoms_state=self.geoms_state,
                geoms_info=self.geoms_info,
                entities_info=self.entities_info,
                rigid_global_info=self._rigid_global_info,
                static_rigid_sim_config=self._static_rigid_sim_config,
            )
            self._is_forward_pos_updated = True
            self._is_forward_vel_updated = True
        else:
            self._is_forward_pos_updated = False
            self._is_forward_vel_updated = False

    def set_global_sol_params(self, sol_params):
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
        sol_params_ = _sanitize_sol_params(sol_params_.clone(), self._sol_min_timeconst)
        kernel_set_global_sol_params(
            sol_params_, self.geoms_info, self.joints_info, self.equalities_info, self._static_rigid_sim_config
        )

    def set_sol_params(self, sol_params, geoms_idx=None, envs_idx=None, *, joints_idx=None, eqs_idx=None):
        """
        Set constraint solver parameters.

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
            constraint_type = 2
            idx_name = "eqs_idx"
            inputs_idx = eqs_idx
            inputs_length = self.n_equalities
            batched = True
        elif joints_idx is not None:
            constraint_type = 1
            idx_name = "joints_idx"
            inputs_idx = joints_idx
            inputs_length = self.n_joints
            batched = self._options.batch_joints_info
        else:
            constraint_type = 0
            idx_name = "geoms_idx"
            inputs_idx = geoms_idx
            inputs_length = self.n_geoms
            batched = False

        # Sanitize input arguments
        sol_params_, inputs_idx, envs_idx = self._sanitize_io_variables(
            sol_params, inputs_idx, inputs_length, idx_name, envs_idx, (7,), batched=batched, skip_allocation=True
        )
        sol_params_ = _sanitize_sol_params(sol_params_.clone(), self._sol_min_timeconst)
        if self.n_envs == 0 and batched:
            sol_params_ = sol_params_[None]

        kernel_set_sol_params(
            constraint_type,
            sol_params_,
            inputs_idx,
            envs_idx,
            geoms_info=self.geoms_info,
            joints_info=self.joints_info,
            equalities_info=self.equalities_info,
            static_rigid_sim_config=self._static_rigid_sim_config,
        )

    def _set_dofs_info(self, tensor_list, dofs_idx, name, envs_idx=None):
        if gs.use_zerocopy and name in {"kp", "kv", "force_range", "stiffness", "damping", "frictionloss", "limit"}:
            mask = indices_to_mask(*((envs_idx, dofs_idx) if self._options.batch_dofs_info else (dofs_idx,)))
            data = qd_to_torch(getattr(self.dofs_info, name), transpose=True, copy=False)
            num_values = len(tensor_list)
            for j, mask_j in enumerate(((*mask, ..., j) for j in range(num_values)) if num_values > 1 else (mask,)):
                assign_indexed_tensor(data, mask_j, tensor_list[j])
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
            kernel_set_dofs_kp(*tensor_list, dofs_idx, envs_idx_, self.dofs_info, self._static_rigid_sim_config)
        elif name == "kv":
            kernel_set_dofs_kv(*tensor_list, dofs_idx, envs_idx_, self.dofs_info, self._static_rigid_sim_config)
        elif name == "force_range":
            kernel_set_dofs_force_range(
                *tensor_list, dofs_idx, envs_idx_, self.dofs_info, self._static_rigid_sim_config
            )
        elif name == "stiffness":
            kernel_set_dofs_stiffness(*tensor_list, dofs_idx, envs_idx_, self.dofs_info, self._static_rigid_sim_config)
        elif name == "armature":
            kernel_set_dofs_armature(*tensor_list, dofs_idx, envs_idx_, self.dofs_info, self._static_rigid_sim_config)
            qs_idx = torch.arange(self.n_qs, dtype=gs.tc_int, device=gs.device)
            qpos_cur = self.get_qpos(qs_idx=qs_idx, envs_idx=envs_idx)
            self._init_invweight_and_meaninertia(envs_idx=envs_idx, force_update=True)
            self.set_qpos(qpos_cur, qs_idx=qs_idx, envs_idx=envs_idx)
        elif name == "damping":
            kernel_set_dofs_damping(*tensor_list, dofs_idx, envs_idx_, self.dofs_info, self._static_rigid_sim_config)
        elif name == "frictionloss":
            kernel_set_dofs_frictionloss(
                *tensor_list, dofs_idx, envs_idx_, self.dofs_info, self._static_rigid_sim_config
            )
        elif name == "limit":
            kernel_set_dofs_limit(*tensor_list, dofs_idx, envs_idx_, self.dofs_info, self._static_rigid_sim_config)
        else:
            gs.raise_exception(f"Invalid `name` {name}.")

    def set_dofs_kp(self, kp, dofs_idx=None, envs_idx=None):
        self._set_dofs_info([kp], dofs_idx, "kp", envs_idx)

    def set_dofs_kv(self, kv, dofs_idx=None, envs_idx=None):
        self._set_dofs_info([kv], dofs_idx, "kv", envs_idx)

    def set_dofs_force_range(self, lower, upper, dofs_idx=None, envs_idx=None):
        self._set_dofs_info([lower, upper], dofs_idx, "force_range", envs_idx)

    def set_dofs_stiffness(self, stiffness, dofs_idx=None, envs_idx=None):
        self._set_dofs_info([stiffness], dofs_idx, "stiffness", envs_idx)

    def set_dofs_armature(self, armature, dofs_idx=None, envs_idx=None):
        self._set_dofs_info([armature], dofs_idx, "armature", envs_idx)

    def set_dofs_damping(self, damping, dofs_idx=None, envs_idx=None):
        self._set_dofs_info([damping], dofs_idx, "damping", envs_idx)

    def set_dofs_frictionloss(self, frictionloss, dofs_idx=None, envs_idx=None):
        self._set_dofs_info([frictionloss], dofs_idx, "frictionloss", envs_idx)

    def set_dofs_limit(self, lower, upper, dofs_idx=None, envs_idx=None):
        self._set_dofs_info([lower, upper], dofs_idx, "limit", envs_idx)

    def set_dofs_velocity(self, velocity, dofs_idx=None, envs_idx=None, *, skip_forward=False):
        if gs.use_zerocopy:
            vel = qd_to_torch(self.dofs_state.vel, transpose=True, copy=False)
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
                    if velocity.ndim == 2:
                        # Note that it is necessary to create a new temporary view because it will be modified in-place
                        dofs_vel.masked_scatter_(envs_idx[:, None], velocity.view_as(velocity))
                    else:
                        velocity = broadcast_tensor(velocity, gs.tc_float, dofs_vel.shape)
                        torch.where(envs_idx[:, None], velocity, dofs_vel, out=dofs_vel)
            else:
                mask = (0, *dofs_mask) if self.n_envs == 0 else indices_to_mask(envs_idx, *dofs_mask)
                if velocity is None:
                    vel[mask] = 0.0
                else:
                    assign_indexed_tensor(vel, mask, velocity)
                if mask and isinstance(mask[0], torch.Tensor):
                    envs_idx = mask[0].reshape((-1,))
            if not skip_forward and not isinstance(envs_idx, torch.Tensor):
                envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        else:
            velocity, dofs_idx, envs_idx = self._sanitize_io_variables(
                velocity, dofs_idx, self.n_dofs, "dofs_idx", envs_idx, skip_allocation=True
            )
            if velocity is None:
                kernel_set_dofs_zero_velocity(dofs_idx, envs_idx, self.dofs_state, self._static_rigid_sim_config)
            else:
                if self.n_envs == 0:
                    velocity = velocity[None]
                kernel_set_dofs_velocity(velocity, dofs_idx, envs_idx, self.dofs_state, self._static_rigid_sim_config)

        if not skip_forward:
            if envs_idx.dtype == torch.bool:
                fn = kernel_masked_forward_velocity
            else:
                fn = kernel_forward_velocity
            fn(
                envs_idx,
                links_state=self.links_state,
                links_info=self.links_info,
                joints_info=self.joints_info,
                dofs_state=self.dofs_state,
                entities_info=self.entities_info,
                rigid_global_info=self._rigid_global_info,
                static_rigid_sim_config=self._static_rigid_sim_config,
                is_backward=False,
            )
            self._is_forward_vel_updated = True
        else:
            self._is_forward_vel_updated = False

    def set_dofs_velocity_grad(self, dofs_idx, envs_idx, velocity_grad):
        velocity_grad_, dofs_idx, envs_idx = self._sanitize_io_variables(
            velocity_grad, dofs_idx, self.n_dofs, "dofs_idx", envs_idx, skip_allocation=True
        )
        if self.n_envs == 0:
            velocity_grad_ = velocity_grad_.unsqueeze(0)
        kernel_set_dofs_velocity_grad(
            velocity_grad_, dofs_idx, envs_idx, self.dofs_state, self._static_rigid_sim_config
        )

    def set_dofs_position(self, position, dofs_idx=None, envs_idx=None):
        position, dofs_idx, envs_idx = self._sanitize_io_variables(
            position, dofs_idx, self.n_dofs, "dofs_idx", envs_idx, skip_allocation=True
        )
        if self.n_envs == 0:
            position = position[None]
        kernel_set_dofs_position(
            position,
            dofs_idx,
            envs_idx,
            self.dofs_state,
            self.links_info,
            self.joints_info,
            self.entities_info,
            self._rigid_global_info,
            self._static_rigid_sim_config,
        )

        if gs.use_zerocopy:
            errno = qd_to_torch(self._errno, copy=False)
            errno[envs_idx] = 0
        else:
            kernel_set_zero(envs_idx, self._errno)

        self.collider.reset(envs_idx)
        self.constraint_solver.reset(envs_idx)

        kernel_forward_kinematics_links_geoms(
            envs_idx,
            links_state=self.links_state,
            links_info=self.links_info,
            joints_state=self.joints_state,
            joints_info=self.joints_info,
            dofs_state=self.dofs_state,
            dofs_info=self.dofs_info,
            geoms_state=self.geoms_state,
            geoms_info=self.geoms_info,
            entities_info=self.entities_info,
            rigid_global_info=self._rigid_global_info,
            static_rigid_sim_config=self._static_rigid_sim_config,
        )
        self._is_forward_pos_updated = True
        self._is_forward_vel_updated = True

    def control_dofs_force(self, force, dofs_idx=None, envs_idx=None):
        if gs.use_zerocopy:
            mask = (0, *indices_to_mask(dofs_idx)) if self.n_envs == 0 else indices_to_mask(envs_idx, dofs_idx)
            ctrl_mode = qd_to_torch(self.dofs_state.ctrl_mode, transpose=True, copy=False)
            ctrl_mode[mask] = gs.CTRL_MODE.FORCE
            ctrl_force = qd_to_torch(self.dofs_state.ctrl_force, transpose=True, copy=False)
            assign_indexed_tensor(ctrl_force, mask, force)
            return

        force, dofs_idx, envs_idx = self._sanitize_io_variables(
            force, dofs_idx, self.n_dofs, "dofs_idx", envs_idx, skip_allocation=True
        )
        if self.n_envs == 0:
            force = force[None]

        kernel_control_dofs_force(force, dofs_idx, envs_idx, self.dofs_state, self._static_rigid_sim_config)

    def control_dofs_velocity(self, velocity, dofs_idx=None, envs_idx=None):
        if gs.use_zerocopy:
            mask = (0, *indices_to_mask(dofs_idx)) if self.n_envs == 0 else indices_to_mask(envs_idx, dofs_idx)
            ctrl_mode = qd_to_torch(self.dofs_state.ctrl_mode, transpose=True, copy=False)
            ctrl_mode[mask] = gs.CTRL_MODE.VELOCITY
            ctrl_pos = qd_to_torch(self.dofs_state.ctrl_pos, transpose=True, copy=False)
            ctrl_pos[mask] = 0.0
            ctrl_vel = qd_to_torch(self.dofs_state.ctrl_vel, transpose=True, copy=False)
            assign_indexed_tensor(ctrl_vel, mask, velocity)
            return

        velocity, dofs_idx, envs_idx = self._sanitize_io_variables(
            velocity, dofs_idx, self.n_dofs, "dofs_idx", envs_idx, skip_allocation=True
        )
        if self.n_envs == 0:
            velocity = velocity[None]

        kernel_control_dofs_velocity(velocity, dofs_idx, envs_idx, self.dofs_state, self._static_rigid_sim_config)

    def control_dofs_position(self, position, dofs_idx=None, envs_idx=None):
        if gs.use_zerocopy:
            mask = (0, *indices_to_mask(dofs_idx)) if self.n_envs == 0 else indices_to_mask(envs_idx, dofs_idx)
            ctrl_mode = qd_to_torch(self.dofs_state.ctrl_mode, transpose=True, copy=False)
            ctrl_mode[mask] = gs.CTRL_MODE.POSITION
            ctrl_pos = qd_to_torch(self.dofs_state.ctrl_pos, transpose=True, copy=False)
            assign_indexed_tensor(ctrl_pos, mask, position)
            ctrl_vel = qd_to_torch(self.dofs_state.ctrl_vel, transpose=True, copy=False)
            ctrl_vel[mask] = 0.0
            return

        position, dofs_idx, envs_idx = self._sanitize_io_variables(
            position, dofs_idx, self.n_dofs, "dofs_idx", envs_idx, skip_allocation=True
        )
        if self.n_envs == 0:
            position = position[None]

        kernel_control_dofs_position(position, dofs_idx, envs_idx, self.dofs_state, self._static_rigid_sim_config)

    def control_dofs_position_velocity(self, position, velocity, dofs_idx=None, envs_idx=None):
        if gs.use_zerocopy:
            mask = (0, *indices_to_mask(dofs_idx)) if self.n_envs == 0 else indices_to_mask(envs_idx, dofs_idx)
            ctrl_mode = qd_to_torch(self.dofs_state.ctrl_mode, transpose=True, copy=False)
            ctrl_mode[mask] = gs.CTRL_MODE.POSITION
            ctrl_pos = qd_to_torch(self.dofs_state.ctrl_pos, transpose=True, copy=False)
            assign_indexed_tensor(ctrl_pos, mask, position)
            ctrl_vel = qd_to_torch(self.dofs_state.ctrl_vel, transpose=True, copy=False)
            assign_indexed_tensor(ctrl_vel, mask, velocity)
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

        kernel_control_dofs_position_velocity(
            position, velocity, dofs_idx, envs_idx, self.dofs_state, self._static_rigid_sim_config
        )

    def get_sol_params(self, geoms_idx=None, envs_idx=None, *, joints_idx=None, eqs_idx=None):
        """
        Get constraint solver parameters.
        """
        if eqs_idx is not None:
            # Always batched
            tensor = qd_to_torch(self.equalities_info.sol_params, envs_idx, eqs_idx, transpose=True, copy=True)
            if self.n_envs == 0:
                tensor = tensor[0]
        elif joints_idx is not None:
            # Conditionally batched
            assert envs_idx is None
            # batch_shape = (envs_idx, joints_idx) if self._options.batch_joints_info else (joints_idx,)
            # tensor = qd_to_torch(self.joints_info.sol_params, *batch_shape, transpose=True)
            tensor = qd_to_torch(self.joints_info.sol_params, envs_idx, joints_idx, transpose=True, copy=True)
            if self.n_envs == 0 and self._options.batch_joints_info:
                tensor = tensor[0]
        else:  # geoms_idx is not None
            # Never batched
            assert envs_idx is None
            tensor = qd_to_torch(self.geoms_info.sol_params, geoms_idx, transpose=True, copy=True)
        return tensor

    @staticmethod
    def _convert_ref_to_idx(ref: Literal["link_origin", "link_com", "root_com"]):
        if ref == "root_com":
            return 0
        elif ref == "link_com":
            return 1
        elif ref == "link_origin":
            return 2
        else:
            gs.raise_exception("'ref' must be either 'link_origin', 'link_com', or 'root_com'.")

    def get_links_pos(
        self,
        links_idx=None,
        envs_idx=None,
        *,
        ref: Literal["link_origin", "link_com", "root_com"] = "link_origin",
    ):
        if not gs.use_zerocopy:
            _, links_idx, envs_idx = self._sanitize_io_variables(
                None, links_idx, self.n_links, "links_idx", envs_idx, (3,), skip_allocation=True
            )

        ref = self._convert_ref_to_idx(ref)
        if ref == 0:
            tensor = qd_to_torch(self.links_state.root_COM, envs_idx, links_idx, transpose=True, copy=True)
        elif ref == 1:
            i_pos = qd_to_torch(self.links_state.i_pos, envs_idx, links_idx, transpose=True)
            root_COM = qd_to_torch(self.links_state.root_COM, envs_idx, links_idx, transpose=True)
            tensor = i_pos + root_COM
        elif ref == 2:
            tensor = qd_to_torch(self.links_state.pos, envs_idx, links_idx, transpose=True, copy=True)
        else:
            gs.raise_exception("'ref' must be either 'link_origin', 'link_com', or 'root_com'.")

        return tensor[0] if self.n_envs == 0 else tensor

    def get_links_quat(self, links_idx=None, envs_idx=None):
        tensor = qd_to_torch(self.links_state.quat, envs_idx, links_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_links_vel(
        self, links_idx=None, envs_idx=None, *, ref: Literal["link_origin", "link_com", "root_com"] = "link_origin"
    ):
        if gs.use_zerocopy:
            mask = (0, *indices_to_mask(links_idx)) if self.n_envs == 0 else indices_to_mask(envs_idx, links_idx)
            cd_vel = qd_to_torch(self.links_state.cd_vel, transpose=True)
            if ref == "root_com":
                return cd_vel[mask]
            cd_ang = qd_to_torch(self.links_state.cd_ang, transpose=True)
            if ref == "link_com":
                i_pos = qd_to_torch(self.links_state.i_pos, transpose=True)
                delta = i_pos[mask]
            else:
                pos = qd_to_torch(self.links_state.pos, transpose=True)
                root_COM = qd_to_torch(self.links_state.root_COM, transpose=True)
                delta = pos[mask] - root_COM[mask]
            return cd_vel[mask] + cd_ang[mask].cross(delta, dim=-1)

        _tensor, links_idx, envs_idx = self._sanitize_io_variables(
            None, links_idx, self.n_links, "links_idx", envs_idx, (3,)
        )
        tensor = _tensor[None] if self.n_envs == 0 else _tensor
        ref = self._convert_ref_to_idx(ref)
        kernel_get_links_vel(tensor, links_idx, envs_idx, ref, self.links_state, self._static_rigid_sim_config)
        return _tensor

    def get_links_ang(self, links_idx=None, envs_idx=None):
        tensor = qd_to_torch(self.links_state.cd_ang, envs_idx, links_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_links_acc(self, links_idx=None, envs_idx=None):
        _tensor, links_idx, envs_idx = self._sanitize_io_variables(
            None, links_idx, self.n_links, "links_idx", envs_idx, (3,)
        )
        tensor = _tensor[None] if self.n_envs == 0 else _tensor
        kernel_get_links_acc(
            tensor,
            links_idx,
            envs_idx,
            self.links_state,
            self._static_rigid_sim_config,
        )
        return _tensor

    def get_links_acc_ang(self, links_idx=None, envs_idx=None):
        tensor = qd_to_torch(self.links_state.cacc_ang, envs_idx, links_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_links_root_COM(self, links_idx=None, envs_idx=None):
        """
        Returns the center of mass (COM) of the entire kinematic tree to which the specified links belong.

        This corresponds to the global COM of each entity, assuming a single-rooted structure - that is, as long as no
        two successive links are connected by a free-floating joint (ie a joint that allows all 6 degrees of freedom).
        """
        tensor = qd_to_torch(self.links_state.root_COM, envs_idx, links_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_links_mass_shift(self, links_idx=None, envs_idx=None):
        tensor = qd_to_torch(self.links_state.mass_shift, envs_idx, links_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_links_COM_shift(self, links_idx=None, envs_idx=None):
        tensor = qd_to_torch(self.links_state.i_pos_shift, envs_idx, links_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_links_inertial_mass(self, links_idx=None, envs_idx=None):
        if self._options.batch_links_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched links info.")
        tensor = qd_to_torch(self.links_info.inertial_mass, envs_idx, links_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 and self._options.batch_links_info else tensor

    def get_links_invweight(self, links_idx=None, envs_idx=None):
        if self._options.batch_links_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched links info.")
        tensor = qd_to_torch(self.links_info.invweight, envs_idx, links_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 and self._options.batch_links_info else tensor

    def get_geoms_friction_ratio(self, geoms_idx=None, envs_idx=None):
        tensor = qd_to_torch(self.geoms_state.friction_ratio, envs_idx, geoms_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_geoms_pos(self, geoms_idx=None, envs_idx=None):
        tensor = qd_to_torch(self.geoms_state.pos, envs_idx, geoms_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_geoms_quat(self, geoms_idx=None, envs_idx=None):
        tensor = qd_to_torch(self.geoms_state.quat, envs_idx, geoms_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_qpos(self, qs_idx=None, envs_idx=None):
        tensor = qd_to_torch(self.qpos, envs_idx, qs_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_dofs_control_force(self, dofs_idx=None, envs_idx=None):
        _tensor, dofs_idx, envs_idx = self._sanitize_io_variables(None, dofs_idx, self.n_dofs, "dofs_idx", envs_idx)
        tensor = _tensor[None] if self.n_envs == 0 else _tensor
        kernel_get_dofs_control_force(
            tensor, dofs_idx, envs_idx, self.dofs_state, self.dofs_info, self._static_rigid_sim_config
        )
        return _tensor

    def get_dofs_force(self, dofs_idx=None, envs_idx=None):
        tensor = qd_to_torch(self.dofs_state.force, envs_idx, dofs_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_dofs_velocity(self, dofs_idx=None, envs_idx=None):
        tensor = qd_to_torch(self.dofs_state.vel, envs_idx, dofs_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_dofs_position(self, dofs_idx=None, envs_idx=None):
        tensor = qd_to_torch(self.dofs_state.pos, envs_idx, dofs_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_dofs_kp(self, dofs_idx=None, envs_idx=None):
        if not self._options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        tensor = qd_to_torch(self.dofs_info.kp, envs_idx, dofs_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 and self._options.batch_dofs_info else tensor

    def get_dofs_kv(self, dofs_idx=None, envs_idx=None):
        if not self._options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        tensor = qd_to_torch(self.dofs_info.kv, envs_idx, dofs_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 and self._options.batch_dofs_info else tensor

    def get_dofs_force_range(self, dofs_idx=None, envs_idx=None):
        if not self._options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        tensor = qd_to_torch(self.dofs_info.force_range, envs_idx, dofs_idx, transpose=True, copy=True)
        if self.n_envs == 0 and self._options.batch_dofs_info:
            tensor = tensor[0]
        return tensor[..., 0], tensor[..., 1]

    def get_dofs_limit(self, dofs_idx=None, envs_idx=None):
        if not self._options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        tensor = qd_to_torch(self.dofs_info.limit, envs_idx, dofs_idx, transpose=True, copy=True)
        if self.n_envs == 0 and self._options.batch_dofs_info:
            tensor = tensor[0]
        return tensor[..., 0], tensor[..., 1]

    def get_dofs_stiffness(self, dofs_idx=None, envs_idx=None):
        if not self._options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        tensor = qd_to_torch(self.dofs_info.stiffness, envs_idx, dofs_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 and self._options.batch_dofs_info else tensor

    def get_dofs_invweight(self, dofs_idx=None, envs_idx=None):
        if not self._options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        tensor = qd_to_torch(self.dofs_info.invweight, envs_idx, dofs_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 and self._options.batch_dofs_info else tensor

    def get_dofs_armature(self, dofs_idx=None, envs_idx=None):
        if not self._options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        tensor = qd_to_torch(self.dofs_info.armature, envs_idx, dofs_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 and self._options.batch_dofs_info else tensor

    def get_dofs_damping(self, dofs_idx=None, envs_idx=None):
        if not self._options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        tensor = qd_to_torch(self.dofs_info.damping, envs_idx, dofs_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 and self._options.batch_dofs_info else tensor

    def get_dofs_frictionloss(self, dofs_idx=None, envs_idx=None):
        if not self._options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        tensor = qd_to_torch(self.dofs_info.frictionloss, envs_idx, dofs_idx, transpose=True, copy=True)
        return tensor[0] if self.n_envs == 0 and self._options.batch_dofs_info else tensor

    def get_mass_mat(self, dofs_idx=None, envs_idx=None, decompose=False):
        tensor = qd_to_torch(self.mass_mat_L if decompose else self.mass_mat, envs_idx, transpose=True, copy=True)
        if dofs_idx is not None:
            tensor = tensor[indices_to_mask(None, dofs_idx, dofs_idx)]
        if self.n_envs == 0:
            tensor = tensor[0]

        if decompose:
            mass_mat_D_inv = qd_to_torch(
                self._rigid_global_info.mass_mat_D_inv, envs_idx, dofs_idx, transpose=True, copy=True
            )
            if self.n_envs == 0:
                mass_mat_D_inv = mass_mat_D_inv[0]
            return tensor, mass_mat_D_inv

        return tensor

    def get_geoms_friction(self, geoms_idx=None):
        return qd_to_torch(self.geoms_info.friction, geoms_idx, copy=True)

    def get_AABB(self, entities_idx=None, envs_idx=None):
        from genesis.engine.couplers import LegacyCoupler

        if not isinstance(self.sim.coupler, LegacyCoupler):
            gs.raise_exception("Method only supported when using 'LegacyCoupler' coupler type.")

        aabb_min = qd_to_torch(self.geoms_state.aabb_min, envs_idx, transpose=True)
        aabb_max = qd_to_torch(self.geoms_state.aabb_max, envs_idx, transpose=True)

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

    def set_geom_friction(self, friction, geoms_idx):
        kernel_set_geom_friction(geoms_idx, friction, self.geoms_info)

    def set_geoms_friction(self, friction, geoms_idx=None):
        friction, geoms_idx, _ = self._sanitize_io_variables(
            friction, geoms_idx, self.n_geoms, "geoms_idx", envs_idx=None, batched=False, skip_allocation=True
        )
        kernel_set_geoms_friction(friction, geoms_idx, self.geoms_info, self._static_rigid_sim_config)

    def add_weld_constraint(self, link1_idx, link2_idx, envs_idx=None):
        return self.constraint_solver.add_weld_constraint(link1_idx, link2_idx, envs_idx)

    def delete_weld_constraint(self, link1_idx, link2_idx, envs_idx=None):
        return self.constraint_solver.delete_weld_constraint(link1_idx, link2_idx, envs_idx)

    def get_weld_constraints(self, as_tensor: bool = True, to_torch: bool = True):
        return self.constraint_solver.get_weld_constraints(as_tensor, to_torch)

    def get_equality_constraints(self, as_tensor: bool = True, to_torch: bool = True):
        return self.constraint_solver.get_equality_constraints(as_tensor, to_torch)

    def clear_external_force(self):
        if gs.use_zerocopy:
            for tensor in (self.links_state.cfrc_applied_ang, self.links_state.cfrc_applied_vel):
                out = qd_to_torch(tensor, copy=False)
                out.zero_()
        else:
            kernel_clear_external_force(self.links_state, self._rigid_global_info, self._static_rigid_sim_config)

    def update_vgeoms(self):
        kernel_update_vgeoms(self.vgeoms_info, self.vgeoms_state, self.links_state, self._static_rigid_sim_config)

    @gs.assert_built
    def set_gravity(self, gravity, envs_idx=None):
        super().set_gravity(gravity, envs_idx)
        if hasattr(self, "_rigid_global_info"):
            self._rigid_global_info.gravity.copy_from(self._gravity)

    def update_drone_propeller_vgeoms(self, propellers_vgeom_idxs, propellers_revs, propellers_spin):
        kernel_update_drone_propeller_vgeoms(
            propellers_vgeom_idxs,
            propellers_revs,
            propellers_spin,
            self.vgeoms_state,
            self._rigid_global_info,
            self._static_rigid_sim_config,
        )

    def set_drone_rpm(self, propellers_link_idx, propellers_rpm, propellers_spin, KF, KM, invert):
        kernel_set_drone_rpm(
            propellers_link_idx,
            propellers_rpm,
            propellers_spin,
            KF,
            KM,
            invert,
            self.links_state,
            self._static_rigid_sim_config,
        )

    def update_verts_for_geoms(self, geoms_idx):
        _, geoms_idx, _ = self._sanitize_io_variables(
            None, geoms_idx, self.n_geoms, "geoms_idx", envs_idx=None, skip_allocation=True
        )
        kernel_update_verts_for_geoms(
            geoms_idx,
            self.geoms_state,
            self.geoms_info,
            self.verts_info,
            self.free_verts_state,
            self.fixed_verts_state,
            self._static_rigid_sim_config,
        )

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def links(self) -> list["RigidLink"]:
        if self.is_built:
            return self._links
        return gs.List(link for entity in self._entities for link in entity.links)

    @property
    def joints(self) -> list["RigidJoint"]:
        if self.is_built:
            return self._joints
        return gs.List(joint for entity in self._entities for joint in entity.joints)

    @property
    def geoms(self) -> list["RigidGeom"]:
        if self.is_built:
            return self._geoms
        return gs.List(geom for entity in self._entities for geom in entity.geoms)

    @property
    def vgeoms(self) -> list["RigidVisGeom"]:
        if self.is_built:
            return self._vgeoms
        return gs.List(vgeom for entity in self._entities for vgeom in entity.vgeoms)

    @property
    def n_links(self):
        if self.is_built:
            return self._n_links
        return len(self.links)

    @property
    def n_joints(self):
        if self.is_built:
            return self._n_joints
        return len(self.joints)

    @property
    def n_geoms(self):
        if self.is_built:
            return self._n_geoms
        return len(self.geoms)

    @property
    def n_cells(self):
        if self.is_built:
            return self._n_cells
        return sum(entity.n_cells for entity in self._entities)

    @property
    def n_vgeoms(self):
        if self.is_built:
            return self._n_vgeoms
        return len(self.vgeoms)

    @property
    def n_verts(self):
        if self.is_built:
            return self._n_verts
        return sum(entity.n_verts for entity in self._entities)

    @property
    def n_free_verts(self):
        if self.is_built:
            return self._n_free_verts
        return sum(link.n_verts if not link.is_fixed or link.entity._batch_fixed_verts else 0 for link in self.links)

    @property
    def n_fixed_verts(self):
        if self.is_built:
            return self._n_fixed_verts
        return sum(link.n_verts if link.is_fixed and not link.entity._batch_fixed_verts else 0 for link in self.links)

    @property
    def n_vverts(self):
        if self.is_built:
            return self._n_vverts
        return sum([entity.n_vverts for entity in self._entities])

    @property
    def n_faces(self):
        if self.is_built:
            return self._n_faces
        return sum([entity.n_faces for entity in self._entities])

    @property
    def n_vfaces(self):
        if self.is_built:
            return self._n_vfaces
        return sum([entity.n_vfaces for entity in self._entities])

    @property
    def n_edges(self):
        if self.is_built:
            return self._n_edges
        return sum([entity.n_edges for entity in self._entities])

    @property
    def n_qs(self):
        if self.is_built:
            return self._n_qs
        return sum([entity.n_qs for entity in self._entities])

    @property
    def n_dofs(self):
        if self.is_built:
            return self._n_dofs
        return sum(entity.n_dofs for entity in self._entities)

    @property
    def init_qpos(self):
        if self._entities:
            return np.concatenate([entity.init_qpos for entity in self._entities], dtype=gs.np_float)
        return np.array([], dtype=gs.np_float)

    @property
    def max_collision_pairs(self):
        return self._max_collision_pairs

    @property
    def n_equalities(self):
        if self.is_built:
            return self._n_equalities
        return sum(entity.n_equalities for entity in self._entities)

    @property
    def equalities(self):
        if self.is_built:
            return self._equalities
        return gs.List(equality for entity in self._entities for equality in entity.equalities)


@qd.kernel(fastcache=gs.use_fastcache)
def kernel_step_1(
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    joints_state: array_class.JointsState,
    joints_info: array_class.JointsInfo,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    entities_state: array_class.EntitiesState,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    contact_island_state: array_class.ContactIslandState,
    is_forward_pos_updated: qd.template(),
    is_forward_vel_updated: qd.template(),
    is_backward: qd.template(),
):
    if qd.static(not is_forward_pos_updated):
        func_update_cartesian_space(
            links_state=links_state,
            links_info=links_info,
            joints_state=joints_state,
            joints_info=joints_info,
            dofs_state=dofs_state,
            dofs_info=dofs_info,
            geoms_info=geoms_info,
            geoms_state=geoms_state,
            entities_info=entities_info,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
            force_update_fixed_geoms=False,
            is_backward=is_backward,
        )

    if qd.static(not is_forward_vel_updated):
        func_forward_velocity(
            entities_info=entities_info,
            links_info=links_info,
            links_state=links_state,
            joints_info=joints_info,
            dofs_state=dofs_state,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
            is_backward=is_backward,
        )

    func_forward_dynamics(
        links_state=links_state,
        links_info=links_info,
        dofs_state=dofs_state,
        dofs_info=dofs_info,
        joints_info=joints_info,
        entities_state=entities_state,
        entities_info=entities_info,
        geoms_state=geoms_state,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
        contact_island_state=contact_island_state,
        is_backward=is_backward,
    )


@qd.kernel(fastcache=gs.use_fastcache)
def kernel_step_2(
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    links_info: array_class.LinksInfo,
    links_state: array_class.LinksState,
    joints_info: array_class.JointsInfo,
    joints_state: array_class.JointsState,
    entities_state: array_class.EntitiesState,
    entities_info: array_class.EntitiesInfo,
    geoms_info: array_class.GeomsInfo,
    geoms_state: array_class.GeomsState,
    collider_state: array_class.ColliderState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    contact_island_state: array_class.ContactIslandState,
    is_backward: qd.template(),
    errno: array_class.V_ANNOTATION,
):
    # Position, Velocity and Acceleration data must be consistent when computing links acceleration, otherwise it
    # would not corresponds to anyting physical. There is no other way than doing this right before integration,
    # because the acceleration at the end of the step is unknown for now as it may change discontinuous between
    # before and after integration under the effect of external forces and constraints. This means that
    # acceleration data will be shifted one timestep in the past, but there isn't really any way around.
    func_update_acc(
        update_cacc=True,
        dofs_state=dofs_state,
        links_info=links_info,
        links_state=links_state,
        entities_info=entities_info,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
        is_backward=is_backward,
    )

    if qd.static(static_rigid_sim_config.integrator != gs.integrator.approximate_implicitfast):
        func_implicit_damping(
            dofs_state=dofs_state,
            dofs_info=dofs_info,
            entities_info=entities_info,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
            is_backward=is_backward,
        )

    func_integrate(
        dofs_state=dofs_state,
        links_info=links_info,
        joints_info=joints_info,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
        is_backward=is_backward,
    )

    if qd.static(static_rigid_sim_config.use_hibernation):
        func_hibernate__for_all_awake_islands_either_hiberanate_or_update_aabb_sort_buffer(
            dofs_state=dofs_state,
            entities_state=entities_state,
            entities_info=entities_info,
            links_state=links_state,
            geoms_state=geoms_state,
            collider_state=collider_state,
            unused__rigid_global_info=rigid_global_info,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
            contact_island_state=contact_island_state,
            errno=errno,
        )
        func_aggregate_awake_entities(
            entities_state=entities_state,
            entities_info=entities_info,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
        )

    if qd.static(not is_backward):
        func_copy_next_to_curr(
            dofs_state=dofs_state,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
            errno=errno,
        )

        if qd.static(not static_rigid_sim_config.enable_mujoco_compatibility):
            func_update_cartesian_space(
                links_state=links_state,
                links_info=links_info,
                joints_state=joints_state,
                joints_info=joints_info,
                dofs_state=dofs_state,
                dofs_info=dofs_info,
                geoms_info=geoms_info,
                geoms_state=geoms_state,
                entities_info=entities_info,
                rigid_global_info=rigid_global_info,
                static_rigid_sim_config=static_rigid_sim_config,
                force_update_fixed_geoms=False,
                is_backward=is_backward,
            )
            func_forward_velocity(
                entities_info=entities_info,
                links_info=links_info,
                links_state=links_state,
                joints_info=joints_info,
                dofs_state=dofs_state,
                rigid_global_info=rigid_global_info,
                static_rigid_sim_config=static_rigid_sim_config,
                is_backward=is_backward,
            )
