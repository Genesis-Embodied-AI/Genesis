from collections import defaultdict
from typing import TYPE_CHECKING, Literal

import gstaichi as ti
import numpy as np
import numpy.typing as npt
import torch

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
from genesis.engine.entities import AvatarEntity, DroneEntity, RigidEntity
from genesis.engine.entities.base_entity import Entity
from genesis.engine.states import QueriedStates, RigidSolverState
from genesis.options.solvers import RigidOptions
from genesis.utils import linalg as lu
from genesis.utils.misc import (
    DeprecationError,
    ti_to_torch,
    ti_to_numpy,
    indices_to_mask,
    broadcast_tensor,
    sanitize_indexed_tensor,
    assign_indexed_tensor,
)
from genesis.utils.sdf_decomp import SDF

from ..base_solver import Solver
from .collider_decomp import Collider
from .constraint_solver_decomp import ConstraintSolver
from .constraint_solver_decomp_island import ConstraintSolverIsland
from .rigid_solver_decomp_util import func_wakeup_entity_and_its_temp_island

if TYPE_CHECKING:
    import genesis.engine.solvers.rigid.array_class
    from genesis.engine.scene import Scene
    from genesis.engine.simulator import Simulator

# minimum constraint impedance
IMP_MIN = 0.0001
# maximum constraint impedance
IMP_MAX = 0.9999

# Minimum ratio between simulation timestep `_substep_dt` and time constant of constraints
TIME_CONSTANT_SAFETY_FACTOR = 2.0


def _sanitize_sol_params(
    sol_params: npt.NDArray[np.float64] | torch.Tensor, min_timeconst: float, default_timeconst: float | None = None
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
        self._enable_adjacent_collision = options.enable_adjacent_collision
        self._disable_constraint = options.disable_constraint
        self._max_collision_pairs = options.max_collision_pairs
        self._integrator = options.integrator
        self._box_box_detection = options.box_box_detection
        self._requires_grad = self._sim.options.requires_grad

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

        self._cur_step = -1

        self.qpos: ti.Template | ti.types.NDArray | None = None

        self._queried_states = QueriedStates()

        self._ckpt = dict()

    def init_ckpt(self):
        pass

    def add_entity(self, idx, material, morph, surface, visualize_contact) -> Entity:
        if isinstance(material, gs.materials.Avatar):
            EntityClass = AvatarEntity
            if visualize_contact:
                gs.raise_exception("AvatarEntity does not support 'visualize_contact=True'.")
        elif isinstance(morph, gs.morphs.Drone):
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

        # FIXME: AvatarSolver should not inherit from RigidSolver, not to mention that it is completely broken...
        is_rigid_solver = type(self) is RigidSolver
        if is_rigid_solver:
            static_rigid_sim_config = dict(
                para_level=self.sim._para_level,
                requires_grad=self.sim.options.requires_grad,
                use_hibernation=self._use_hibernation,
                batch_links_info=self._options.batch_links_info,
                batch_dofs_info=self._options.batch_dofs_info,
                batch_joints_info=self._options.batch_joints_info,
                enable_mujoco_compatibility=self._enable_mujoco_compatibility,
                enable_multi_contact=self._enable_multi_contact,
                enable_collision=self._enable_collision,
                enable_joint_limit=self._enable_joint_limit,
                box_box_detection=self._box_box_detection,
                sparse_solve=self._options.sparse_solve,
                integrator=self._integrator,
                solver_type=self._options.constraint_solver,
                is_backward=False,
            )
            # Add terms for static inner loops, use -1 if not requires_grad to avoid re-compilation
            if self.sim.options.requires_grad:
                static_rigid_sim_config.update(
                    max_n_geoms_per_entity=max(len(entity.geoms) for entity in self.entities) if self.links else 0,
                    max_n_links_per_entity=max(len(entity.links) for entity in self.entities) if self.entities else 0,
                    max_n_joints_per_link=max(len(link.joints) for link in self.links) if self.links else 0,
                    max_n_dofs_per_joint=max(joint.n_dofs for joint in self.joints) if self.joints else 0,
                    max_n_dofs_per_entity=max(entity.n_dofs for entity in self.entities) if self.entities else 0,
                    max_n_dofs_per_link=max(link.n_dofs for link in self.links) if self.links else 0,
                    max_n_qs_per_link=max(link.n_qs for link in self.links) if self.links else 0,
                    n_links=self._n_links,
                    n_geoms=self._n_geoms,
                )
            self._static_rigid_sim_config = array_class.StructRigidSimStaticConfig(**static_rigid_sim_config)
        else:
            self._static_rigid_sim_config = array_class.StructRigidSimStaticConfig(
                para_level=self.sim._para_level,
                requires_grad=self.sim.options.requires_grad,
                enable_collision=self._enable_collision,
                integrator=gs.integrator.approximate_implicitfast,
                solver_type=gs.constraint_solver.CG,
                is_backward=False,
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

        # when the migration is finished, we will remove the about two lines
        self._func_vel_at_point = func_vel_at_point
        self._func_apply_coupling_force = func_apply_coupling_force

        # For rigid solver, we initialize them even if the solver is not active because the coupler needs arguments like
        # rigid_solver.links_state, etc. regardless of the solver is active or not.
        if is_rigid_solver or self.is_active:
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
            self._init_entity_fields()
            self._init_equality_fields()

            self._init_envs_offset()
            self._init_sdf()
            self._init_collider()
            self._init_constraint_solver()

            self._init_invweight_and_meaninertia(force_update=False)
            self._func_update_geoms(self._scene._envs_idx, force_update_fixed_geoms=True)

    def _init_invweight_and_meaninertia(self, envs_idx=None, *, force_update=True):
        # Early return if no DoFs. This is essential to avoid segfault on CUDA.
        if self._n_dofs == 0:
            return

        # Handling default arguments
        batched = self._options.batch_dofs_info and self._options.batch_links_info
        if not batched and envs_idx is not None:
            gs.raise_exception(
                "Links and dofs must be batched to selectively update invweight and meaninertia for some environment."
            )
        envs_idx = self._scene._sanitize_envs_idx(envs_idx)

        # Compute state in neutral configuration at rest
        qpos = ti_to_torch(self.qpos0, envs_idx, transpose=True)
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
        mass_mat_D_inv = ti_to_numpy(self._rigid_global_info.mass_mat_D_inv)
        mass_mat_L = ti_to_numpy(self._rigid_global_info.mass_mat_L)
        offsets = ti_to_numpy(self.links_state.i_pos)
        cdof_ang = ti_to_numpy(self.dofs_state.cdof_ang)
        cdof_vel = ti_to_numpy(self.dofs_state.cdof_vel)
        links_joint_start = ti_to_numpy(self.links_info.joint_start)
        links_joint_end = ti_to_numpy(self.links_info.joint_end)
        links_dof_end = ti_to_numpy(self.links_info.dof_end)
        links_n_dofs = ti_to_numpy(self.links_info.n_dofs)
        links_parent_idx = ti_to_numpy(self.links_info.parent_idx)
        joints_type = ti_to_numpy(self.joints_info.type)
        joints_dof_start = ti_to_numpy(self.joints_info.dof_start)
        joints_n_dofs = ti_to_numpy(self.joints_info.n_dofs)

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
                for i_d, j_d in ti.ndrange(
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
                # taichi variables
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
                # taichi variables
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
                "Reference robot position exceeds joint limits."
                # "Clipping initial position too make sure it is valid."
            )

        # This is for IK use only
        # TODO: support IK with parallel envs
        # self._rigid_global_info.links_T = ti.Matrix.field(n=4, m=4, dtype=gs.ti_float, shape=self.n_links)
        self.links_T = self._rigid_global_info.links_T

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
                # taichi variables
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
                # taichi variables
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
                # taichi variables
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
                # taichi variables
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
                # taichi variables
                entities_info=self.entities_info,
                entities_state=self.entities_state,
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
                # taichi variables
                equalities_info=self.equalities_info,
                static_rigid_sim_config=self._static_rigid_sim_config,
            )
            if self._use_contact_island:
                gs.logger.warn("contact island is not supported for equality constraints yet")

    def _init_envs_offset(self):
        self.envs_offset = self._rigid_global_info.envs_offset
        self.envs_offset.from_numpy(self._scene.envs_offset)

    def _init_sdf(self):
        self.sdf = SDF(self)

    def _init_collider(self):
        self.collider = Collider(self)

        if self.collider._collider_static_config.has_terrain:
            link_idx_ = next(
                i for i, _type in enumerate(ti_to_numpy(self.geoms_info.type)) if _type == gs.GEOM_TYPE.TERRAIN
            )
            link_idx = ti_to_numpy(self.geoms_info.link_idx, link_idx_, keepdim=False)
            entity_idx = ti_to_numpy(self.links_info.entity_idx, link_idx, keepdim=False)
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

            self.terrain_hf = ti.field(dtype=gs.ti_float, shape=hf.shape)
            self.terrain_rc = ti.field(dtype=gs.ti_int, shape=(2,))
            self.terrain_scale = ti.field(dtype=gs.ti_float, shape=(2,))
            self.terrain_xyz_maxmin = ti.field(dtype=gs.ti_float, shape=(6,))

            self.terrain_hf.from_numpy(hf)
            self.terrain_rc.from_numpy(rc)
            self.terrain_scale.from_numpy(scale)
            self.terrain_xyz_maxmin.from_numpy(xyz_maxmin)

    def _init_constraint_solver(self):
        if self.links:
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
            links_state=self.links_state,
            links_info=self.links_info,
            joints_state=self.joints_state,
            joints_info=self.joints_info,
            dofs_state=self.dofs_state,
            dofs_info=self.dofs_info,
            geoms_state=self.geoms_state,
            geoms_info=self.geoms_info,
            entities_state=self.entities_state,
            entities_info=self.entities_info,
            rigid_global_info=self._rigid_global_info,
            static_rigid_sim_config=self._static_rigid_sim_config,
            contact_island_state=self.constraint_solver.contact_island.contact_island_state,
        )

        if isinstance(self.sim.coupler, SAPCoupler):
            update_qvel(
                dofs_state=self.dofs_state,
                rigid_global_info=self._rigid_global_info,
                static_rigid_sim_config=self._static_rigid_sim_config,
            )
        else:
            self._func_constraint_force()
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
            )
            if self._requires_grad:
                kernel_save_adjoint_cache(
                    f=f + 1,
                    dofs_state=self.dofs_state,
                    rigid_global_info=self._rigid_global_info,
                    rigid_adjoint_cache=self._rigid_adjoint_cache,
                    static_rigid_sim_config=self._static_rigid_sim_config,
                )

    def check_errno(self):
        # Note that errno must be evaluated BEFORE match because otherwise it will be evaluated for each case...
        # See official documentation: https://docs.python.org/3.10/reference/compound_stmts.html#overview
        if gs.use_zerocopy:
            errno = ti_to_torch(self._errno, copy=None).item()
        else:
            errno = kernel_get_errno(self._errno)
        match errno:
            case 1:
                max_collision_pairs_broad = self.collider._collider_info.max_collision_pairs_broad[None]
                gs.raise_exception(
                    f"Exceeding max number of broad phase candidate contact pairs ({max_collision_pairs_broad}). "
                    f"Please increase the value of RigidSolver's option 'multiplier_collision_broad_phase'."
                )
            case 2:
                max_contact_pairs = self.collider._collider_info.max_contact_pairs[None]
                gs.raise_exception(
                    f"Exceeding max number of contact pairs ({max_contact_pairs}). Please increase the value of "
                    "RigidSolver's option 'max_collision_pairs'."
                )
            case 3:
                gs.raise_exception("Invalid accelerations causing 'nan'. Please decrease Rigid simulation timestep.")

    def _kernel_detect_collision(self):
        self.collider.reset(cache_only=True)
        self.collider.clear()
        self.collider.detection()

    def detect_collision(self, env_idx=0):
        # TODO: support batching
        self._kernel_detect_collision()

        n_collision = ti_to_numpy(self.collider._collider_state.n_contacts)[env_idx]
        collision_pairs = np.empty((n_collision, 2), dtype=np.int32)
        collision_pairs[:, 0] = ti_to_numpy(self.collider._collider_state.contact_data.geom_a)[:n_collision, env_idx]
        collision_pairs[:, 1] = ti_to_numpy(self.collider._collider_state.contact_data.geom_b)[:n_collision, env_idx]

        return collision_pairs

    def _func_constraint_force(self):
        if not self._disable_constraint:
            if self._use_contact_island:
                self.constraint_solver.clear()
            else:
                self.constraint_solver.clear(cache_only=True)
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
        )

    def _func_update_acc(self):
        kernel_update_acc(
            dofs_state=self.dofs_state,
            links_info=self.links_info,
            links_state=self.links_state,
            entities_info=self.entities_info,
            rigid_global_info=self._rigid_global_info,
            static_rigid_sim_config=self._static_rigid_sim_config,
        )

    def _func_forward_kinematics_entity(self, i_e, envs_idx):
        kernel_forward_kinematics_entity(
            i_e,
            envs_idx,
            links_state=self.links_state,
            links_info=self.links_info,
            joints_state=self.joints_state,
            joints_info=self.joints_info,
            dofs_state=self.dofs_state,
            dofs_info=self.dofs_info,
            entities_info=self.entities_info,
            rigid_global_info=self._rigid_global_info,
            static_rigid_sim_config=self._static_rigid_sim_config,
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
            entities_info=self.entities_info,
            geoms_info=self.geoms_info,
            geoms_state=self.geoms_state,
            links_state=self.links_state,
            rigid_global_info=self._rigid_global_info,
            static_rigid_sim_config=self._static_rigid_sim_config,
            force_update_fixed_geoms=force_update_fixed_geoms,
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
                return

            # Run Genesis rigid simulation step
            self.substep(f)

    def substep_pre_coupling_grad(self, f):
        # Change to backward mode
        self._static_rigid_sim_config.is_backward = True

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
            )

        # Change back to forward mode
        self._static_rigid_sim_config.is_backward = False

    def substep_post_coupling(self, f):
        from genesis.engine.couplers import SAPCoupler, IPCCoupler

        if not self.is_active:
            return

        if isinstance(self.sim.coupler, SAPCoupler):
            update_qacc_from_qvel_delta(
                dofs_state=self.dofs_state,
                rigid_global_info=self._rigid_global_info,
                static_rigid_sim_config=self._static_rigid_sim_config,
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
            )
        elif isinstance(self.sim.coupler, IPCCoupler):
            # For IPCCoupler, perform full rigid body computation in post-coupling phase
            # This allows IPC to handle rigid bodies during the coupling phase
            # Temporarily disable ground collision if requested
            if self.sim.coupler.options.disable_genesis_ground_contact:
                original_enable_collision = self._enable_collision
                self._enable_collision = False
                self.substep(f)
                self._enable_collision = original_enable_collision
            else:
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

            self._errno[None] = 0
            self.collider.reset(envs_idx, cache_only=False)
            self.collider.clear(envs_idx)
            if self.constraint_solver is not None:
                self.constraint_solver.reset(envs_idx)
            self._cur_step = -1

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

            self._ckpt[ckpt_name]["qpos"] = ti_to_numpy(self._rigid_adjoint_cache.qpos)
            self._ckpt[ckpt_name]["dofs_vel"] = ti_to_numpy(self._rigid_adjoint_cache.dofs_vel)
            self._ckpt[ckpt_name]["dofs_acc"] = ti_to_numpy(self._rigid_adjoint_cache.dofs_acc)

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
        tensor: np.typing.ArrayLike | None,
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
            mask = (0, *indices_to_mask(qs_idx)) if self.n_envs == 0 else indices_to_mask(envs_idx, qs_idx)
            data = ti_to_torch(self._rigid_global_info.qpos, transpose=True, copy=False)
            assign_indexed_tensor(data, mask, qpos)
            if mask and isinstance(mask[0], torch.Tensor):
                envs_idx = mask[0].reshape((-1,))
        else:
            qpos, qs_idx, envs_idx = self._sanitize_io_variables(
                qpos, qs_idx, self.n_qs, "qs_idx", envs_idx, skip_allocation=True
            )
            if self.n_envs == 0:
                qpos = qpos[None]
            kernel_set_qpos(qpos, qs_idx, envs_idx, self._rigid_global_info, self._static_rigid_sim_config)

        self.collider.reset(envs_idx, cache_only=True)
        if not isinstance(envs_idx, torch.Tensor):
            envs_idx = self._scene._sanitize_envs_idx(envs_idx)
        if not skip_forward:
            self.collider.clear(envs_idx)
        if self.constraint_solver is not None:
            if self._use_contact_island:
                self.constraint_solver.reset(envs_idx)
            else:
                self.constraint_solver.reset(envs_idx, clear_contraints_info=not skip_forward)
        if not skip_forward:
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
            data = ti_to_torch(getattr(self.dofs_info, name), transpose=True, copy=False)
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
            kernel_set_dofs_kp(tensor_list[0], dofs_idx, envs_idx_, self.dofs_info, self._static_rigid_sim_config)
        elif name == "kv":
            kernel_set_dofs_kv(tensor_list[0], dofs_idx, envs_idx_, self.dofs_info, self._static_rigid_sim_config)
        elif name == "force_range":
            kernel_set_dofs_force_range(
                tensor_list[0], tensor_list[1], dofs_idx, envs_idx_, self.dofs_info, self._static_rigid_sim_config
            )
        elif name == "stiffness":
            kernel_set_dofs_stiffness(
                tensor_list[0], dofs_idx, envs_idx_, self.dofs_info, self._static_rigid_sim_config
            )
        elif name == "armature":
            kernel_set_dofs_armature(tensor_list[0], dofs_idx, envs_idx_, self.dofs_info, self._static_rigid_sim_config)
            qs_idx = torch.arange(self.n_qs, dtype=gs.tc_int, device=gs.device)
            qpos_cur = self.get_qpos(qs_idx=qs_idx, envs_idx=envs_idx)
            self._init_invweight_and_meaninertia(envs_idx=envs_idx, force_update=True)
            self.set_qpos(qpos_cur, qs_idx=qs_idx, envs_idx=envs_idx)
        elif name == "damping":
            kernel_set_dofs_damping(tensor_list[0], dofs_idx, envs_idx_, self.dofs_info, self._static_rigid_sim_config)
        elif name == "frictionloss":
            kernel_set_dofs_frictionloss(
                tensor_list[0], dofs_idx, envs_idx_, self.dofs_info, self._static_rigid_sim_config
            )
        elif name == "limit":
            kernel_set_dofs_limit(
                tensor_list[0], tensor_list[1], dofs_idx, envs_idx_, self.dofs_info, self._static_rigid_sim_config
            )
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
            vel = ti_to_torch(self.dofs_state.vel, transpose=True, copy=False)
            if velocity is None and isinstance(dofs_idx, slice) and isinstance(envs_idx, torch.Tensor):
                (vel := vel[:, dofs_idx]).scatter_(0, envs_idx[:, None].expand((-1, vel.shape[1])), 0.0)
            else:
                mask = (0, *indices_to_mask(dofs_idx)) if self.n_envs == 0 else indices_to_mask(envs_idx, dofs_idx)
                if velocity is None:
                    vel[mask] = 0.0
                else:
                    assign_indexed_tensor(vel, mask, velocity)
                if mask and isinstance(mask[0], torch.Tensor):
                    envs_idx = mask[0].reshape((-1,))
                elif not isinstance(envs_idx, torch.Tensor):
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
            kernel_forward_velocity(
                envs_idx,
                links_state=self.links_state,
                links_info=self.links_info,
                joints_info=self.joints_info,
                dofs_state=self.dofs_state,
                entities_info=self.entities_info,
                rigid_global_info=self._rigid_global_info,
                static_rigid_sim_config=self._static_rigid_sim_config,
            )

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

        self.collider.reset(envs_idx, cache_only=True)
        self.collider.clear(envs_idx)
        if self.constraint_solver is not None:
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

    def control_dofs_force(self, force, dofs_idx=None, envs_idx=None):
        if gs.use_zerocopy:
            mask = (0, *indices_to_mask(dofs_idx)) if self.n_envs == 0 else indices_to_mask(envs_idx, dofs_idx)
            ctrl_mode = ti_to_torch(self.dofs_state.ctrl_mode, transpose=True, copy=False)
            ctrl_mode[mask] = gs.CTRL_MODE.FORCE
            ctrl_force = ti_to_torch(self.dofs_state.ctrl_force, transpose=True, copy=False)
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
            ctrl_mode = ti_to_torch(self.dofs_state.ctrl_mode, transpose=True, copy=False)
            ctrl_mode[mask] = gs.CTRL_MODE.VELOCITY
            ctrl_pos = ti_to_torch(self.dofs_state.ctrl_pos, transpose=True, copy=False)
            ctrl_pos[mask] = 0.0
            ctrl_vel = ti_to_torch(self.dofs_state.ctrl_vel, transpose=True, copy=False)
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
            ctrl_mode = ti_to_torch(self.dofs_state.ctrl_mode, transpose=True, copy=False)
            ctrl_mode[mask] = gs.CTRL_MODE.POSITION
            ctrl_pos = ti_to_torch(self.dofs_state.ctrl_pos, transpose=True, copy=False)
            assign_indexed_tensor(ctrl_pos, mask, position)
            ctrl_vel = ti_to_torch(self.dofs_state.ctrl_vel, transpose=True, copy=False)
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
            ctrl_mode = ti_to_torch(self.dofs_state.ctrl_mode, transpose=True, copy=False)
            ctrl_mode[mask] = gs.CTRL_MODE.POSITION
            ctrl_pos = ti_to_torch(self.dofs_state.ctrl_pos, transpose=True, copy=False)
            assign_indexed_tensor(ctrl_pos, mask, position)
            ctrl_vel = ti_to_torch(self.dofs_state.ctrl_vel, transpose=True, copy=False)
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
            tensor = ti_to_torch(self.equalities_info.sol_params, envs_idx, eqs_idx, transpose=True)
            if self.n_envs == 0:
                tensor = tensor[0]
        elif joints_idx is not None:
            # Conditionally batched
            assert envs_idx is None
            # batch_shape = (envs_idx, joints_idx) if self._options.batch_joints_info else (joints_idx,)
            # tensor = ti_to_torch(self.joints_info.sol_params, *batch_shape, transpose=True)
            tensor = ti_to_torch(self.joints_info.sol_params, envs_idx, joints_idx, transpose=True)
            if self.n_envs == 0 and self._options.batch_joints_info:
                tensor = tensor[0]
        else:
            # Never batched
            assert envs_idx is None
            tensor = ti_to_torch(self.geoms_info.sol_params, geoms_idx, transpose=True)
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
        to_torch: bool = True,
    ):
        if not gs.use_zerocopy:
            _, links_idx, envs_idx = self._sanitize_io_variables(
                None, links_idx, self.n_links, "links_idx", envs_idx, (3,), skip_allocation=True
            )

        ref = self._convert_ref_to_idx(ref)
        if ref == 0:
            tensor = ti_to_torch(self.links_state.root_COM, envs_idx, links_idx, transpose=True)
        elif ref == 1:
            i_pos = ti_to_torch(self.links_state.i_pos, envs_idx, links_idx, transpose=True)
            root_COM = ti_to_torch(self.links_state.root_COM, envs_idx, links_idx, transpose=True)
            tensor = i_pos + root_COM
        elif ref == 2:
            tensor = ti_to_torch(self.links_state.pos, envs_idx, links_idx, transpose=True)
        else:
            gs.raise_exception("'ref' must be either 'link_origin', 'link_com', or 'root_com'.")

        return tensor[0] if self.n_envs == 0 else tensor

    def get_links_quat(self, links_idx=None, envs_idx=None, *, to_torch=True):
        tensor = ti_to_torch(self.links_state.quat, envs_idx, links_idx, transpose=True)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_links_vel(
        self, links_idx=None, envs_idx=None, *, ref: Literal["link_origin", "link_com", "root_com"] = "link_origin"
    ):
        if gs.use_zerocopy:
            mask = (0, *indices_to_mask(links_idx)) if self.n_envs == 0 else indices_to_mask(envs_idx, links_idx)
            cd_vel = ti_to_torch(self.links_state.cd_vel, transpose=True, copy=False)
            if ref == "root_com":
                return cd_vel[mask]
            cd_ang = ti_to_torch(self.links_state.cd_ang, transpose=True, copy=False)
            if ref == "link_com":
                i_pos = ti_to_torch(self.links_state.i_pos, transpose=True, copy=False)
                delta = i_pos[mask]
            else:
                pos = ti_to_torch(self.links_state.pos, transpose=True, copy=False)
                root_COM = ti_to_torch(self.links_state.root_COM, transpose=True, copy=False)
                delta = pos[mask] - root_COM[mask]
            return cd_vel[mask] + cd_ang[mask].cross(delta, dim=-1)

        _tensor, links_idx, envs_idx = self._sanitize_io_variables(
            None, links_idx, self.n_links, "links_idx", envs_idx, (3,)
        )
        tensor = _tensor[None] if self.n_envs == 0 else _tensor
        ref = self._convert_ref_to_idx(ref)
        kernel_get_links_vel(tensor, links_idx, envs_idx, ref, self.links_state, self._static_rigid_sim_config)
        return _tensor

    def get_links_ang(self, links_idx=None, envs_idx=None, *, to_torch=True):
        tensor = ti_to_torch(self.links_state.cd_ang, envs_idx, links_idx, transpose=True)
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

    def get_links_acc_ang(self, links_idx=None, envs_idx=None, *, to_torch=True):
        tensor = ti_to_torch(self.links_state.cacc_ang, envs_idx, links_idx, transpose=True)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_links_root_COM(self, links_idx=None, envs_idx=None, *, to_torch=True):
        """
        Returns the center of mass (COM) of the entire kinematic tree to which the specified links belong.

        This corresponds to the global COM of each entity, assuming a single-rooted structure - that is, as long as no
        two successive links are connected by a free-floating joint (ie a joint that allows all 6 degrees of freedom).
        """
        tensor = ti_to_torch(self.links_state.root_COM, envs_idx, links_idx, transpose=True)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_links_mass_shift(self, links_idx=None, envs_idx=None, *, to_torch=True):
        tensor = ti_to_torch(self.links_state.mass_shift, envs_idx, links_idx, transpose=True)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_links_COM_shift(self, links_idx=None, envs_idx=None, *, to_torch=True):
        tensor = ti_to_torch(self.links_state.i_pos_shift, envs_idx, links_idx, transpose=True)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_links_inertial_mass(self, links_idx=None, envs_idx=None):
        if self._options.batch_links_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched links info.")
        tensor = ti_to_torch(self.links_info.inertial_mass, envs_idx, links_idx, transpose=True)
        return tensor[0] if self.n_envs == 0 and self._options.batch_links_info else tensor

    def get_links_invweight(self, links_idx=None, envs_idx=None):
        if self._options.batch_links_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched links info.")
        tensor = ti_to_torch(self.links_info.invweight, envs_idx, links_idx, transpose=True)
        return tensor[0] if self.n_envs == 0 and self._options.batch_links_info else tensor

    def get_geoms_friction_ratio(self, geoms_idx=None, envs_idx=None):
        tensor = ti_to_torch(self.geoms_state.friction_ratio, envs_idx, geoms_idx, transpose=True)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_geoms_pos(self, geoms_idx=None, envs_idx=None):
        tensor = ti_to_torch(self.geoms_state.pos, envs_idx, geoms_idx, transpose=True)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_qpos(self, qs_idx=None, envs_idx=None):
        tensor = ti_to_torch(self.qpos, envs_idx, qs_idx, transpose=True)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_dofs_control_force(self, dofs_idx=None, envs_idx=None):
        _tensor, dofs_idx, envs_idx = self._sanitize_io_variables(None, dofs_idx, self.n_dofs, "dofs_idx", envs_idx)
        tensor = _tensor[None] if self.n_envs == 0 else _tensor
        kernel_get_dofs_control_force(
            tensor, dofs_idx, envs_idx, self.dofs_state, self.dofs_info, self._static_rigid_sim_config
        )
        return _tensor

    def get_dofs_force(self, dofs_idx=None, envs_idx=None):
        tensor = ti_to_torch(self.dofs_state.force, envs_idx, dofs_idx, transpose=True)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_dofs_velocity(self, dofs_idx=None, envs_idx=None):
        tensor = ti_to_torch(self.dofs_state.vel, envs_idx, dofs_idx, transpose=True)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_dofs_position(self, dofs_idx=None, envs_idx=None):
        tensor = ti_to_torch(self.dofs_state.pos, envs_idx, dofs_idx, transpose=True)
        return tensor[0] if self.n_envs == 0 else tensor

    def get_dofs_kp(self, dofs_idx=None, envs_idx=None):
        if not self._options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        tensor = ti_to_torch(self.dofs_info.kp, envs_idx, dofs_idx, transpose=True)
        return tensor[0] if self.n_envs == 0 and self._options.batch_dofs_info else tensor

    def get_dofs_kv(self, dofs_idx=None, envs_idx=None):
        if not self._options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        tensor = ti_to_torch(self.dofs_info.kv, envs_idx, dofs_idx, transpose=True)
        return tensor[0] if self.n_envs == 0 and self._options.batch_dofs_info else tensor

    def get_dofs_force_range(self, dofs_idx=None, envs_idx=None):
        if not self._options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        tensor = ti_to_torch(self.dofs_info.force_range, envs_idx, dofs_idx, transpose=True)
        if self.n_envs == 0 and self._options.batch_dofs_info:
            tensor = tensor[0]
        return tensor[..., 0], tensor[..., 1]

    def get_dofs_limit(self, dofs_idx=None, envs_idx=None):
        if not self._options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        tensor = ti_to_torch(self.dofs_info.limit, envs_idx, dofs_idx, transpose=True)
        if self.n_envs == 0 and self._options.batch_dofs_info:
            tensor = tensor[0]
        return tensor[..., 0], tensor[..., 1]

    def get_dofs_stiffness(self, dofs_idx=None, envs_idx=None):
        if not self._options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        tensor = ti_to_torch(self.dofs_info.stiffness, envs_idx, dofs_idx, transpose=True)
        return tensor[0] if self.n_envs == 0 and self._options.batch_dofs_info else tensor

    def get_dofs_invweight(self, dofs_idx=None, envs_idx=None):
        if not self._options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        tensor = ti_to_torch(self.dofs_info.invweight, envs_idx, dofs_idx, transpose=True)
        return tensor[0] if self.n_envs == 0 and self._options.batch_dofs_info else tensor

    def get_dofs_armature(self, dofs_idx=None, envs_idx=None):
        if not self._options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        tensor = ti_to_torch(self.dofs_info.armature, envs_idx, dofs_idx, transpose=True)
        return tensor[0] if self.n_envs == 0 and self._options.batch_dofs_info else tensor

    def get_dofs_damping(self, dofs_idx=None, envs_idx=None):
        if not self._options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        tensor = ti_to_torch(self.dofs_info.damping, envs_idx, dofs_idx, transpose=True)
        return tensor[0] if self.n_envs == 0 and self._options.batch_dofs_info else tensor

    def get_dofs_frictionloss(self, dofs_idx=None, envs_idx=None):
        if not self._options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        tensor = ti_to_torch(self.dofs_info.frictionloss, envs_idx, dofs_idx, transpose=True)
        return tensor[0] if self.n_envs == 0 and self._options.batch_dofs_info else tensor

    def get_mass_mat(self, dofs_idx=None, envs_idx=None, decompose=False):
        tensor = ti_to_torch(self.mass_mat_L if decompose else self.mass_mat, envs_idx, transpose=True)

        if dofs_idx is not None:
            if isinstance(dofs_idx, (slice, int, np.integer)) or (dofs_idx.ndim == 0):
                tensor = tensor[:, dofs_idx, dofs_idx]
                if tensor.ndim == 1:
                    tensor = tensor.reshape((-1, 1, 1))
            else:
                tensor = tensor[:, dofs_idx[:, None], dofs_idx]
        if self.n_envs == 0:
            tensor = tensor[0]

        if decompose:
            mass_mat_D_inv = ti_to_torch(self._rigid_global_info.mass_mat_D_inv, envs_idx, dofs_idx, transpose=True)
            if self.n_envs == 0:
                mass_mat_D_inv = mass_mat_D_inv[0]
            return tensor, mass_mat_D_inv

        return tensor

    def get_geoms_friction(self, geoms_idx=None):
        return ti_to_torch(self.geoms_info.friction, geoms_idx, None)

    def get_AABB(self, entities_idx=None, envs_idx=None):
        from genesis.engine.couplers import LegacyCoupler

        if not isinstance(self.sim.coupler, LegacyCoupler):
            gs.raise_exception("Method only supported when using 'LegacyCoupler' coupler type.")

        aabb_min = ti_to_torch(self.geoms_state.aabb_min, envs_idx, transpose=True)
        aabb_max = ti_to_torch(self.geoms_state.aabb_max, envs_idx, transpose=True)

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
                out = ti_to_torch(tensor, copy=False)
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

    def update_drone_propeller_vgeoms(self, n_propellers, propellers_vgeom_idxs, propellers_revs, propellers_spin):
        kernel_update_drone_propeller_vgeoms(
            n_propellers,
            propellers_vgeom_idxs,
            propellers_revs,
            propellers_spin,
            self.vgeoms_state,
            self._rigid_global_info,
            self._static_rigid_sim_config,
        )

    def set_drone_rpm(self, n_propellers, propellers_link_idx, propellers_rpm, propellers_spin, KF, KM, invert):
        kernel_set_drone_rpm(
            n_propellers,
            propellers_link_idx,
            propellers_rpm,
            propellers_spin,
            KF,
            KM,
            invert,
            self.links_state,
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
        )

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def links(self):
        if self.is_built:
            return self._links
        return gs.List(link for entity in self._entities for link in entity.links)

    @property
    def joints(self):
        if self.is_built:
            return self._joints
        return gs.List(joint for entity in self._entities for joint in entity.joints)

    @property
    def geoms(self):
        if self.is_built:
            return self._geoms
        return gs.List(geom for entity in self._entities for geom in entity.geoms)

    @property
    def vgeoms(self):
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


@ti.kernel
def update_qacc_from_qvel_delta(
    dofs_state: array_class.DofsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    n_dofs = dofs_state.ctrl_mode.shape[0]
    _B = dofs_state.ctrl_mode.shape[1]
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_0, i_b in ti.ndrange(1, _B) if ti.static(static_rigid_sim_config.use_hibernation) else ti.ndrange(n_dofs, _B):
        for i_1 in (
            (
                # Dynamic inner loop for forward pass
                range(rigid_global_info.n_awake_dofs[i_b])
                if ti.static(static_rigid_sim_config.use_hibernation)
                else range(1)
            )
            if ti.static(not static_rigid_sim_config.is_backward)
            else (
                # Static inner loop for backward pass
                ti.static(range(static_rigid_sim_config.max_n_awake_dofs))
                if ti.static(static_rigid_sim_config.use_hibernation)
                else ti.static(range(1))
            )
        ):
            if i_1 < (rigid_global_info.n_awake_dofs[i_b] if ti.static(static_rigid_sim_config.use_hibernation) else 1):
                i_d = (
                    rigid_global_info.awake_dofs[i_1, i_b]
                    if ti.static(static_rigid_sim_config.use_hibernation)
                    else i_0
                )
                dofs_state.acc[i_d, i_b] = (
                    dofs_state.vel[i_d, i_b] - dofs_state.vel_prev[i_d, i_b]
                ) / rigid_global_info.substep_dt[None]
                dofs_state.vel[i_d, i_b] = dofs_state.vel_prev[i_d, i_b]


@ti.kernel
def update_qvel(
    dofs_state: array_class.DofsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    _B = dofs_state.vel.shape[1]
    n_dofs = dofs_state.vel.shape[0]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_0, i_b in ti.ndrange(1, _B) if ti.static(static_rigid_sim_config.use_hibernation) else ti.ndrange(n_dofs, _B):
        for i_1 in (
            (
                # Dynamic inner loop for forward pass
                range(rigid_global_info.n_awake_dofs[i_b])
                if ti.static(static_rigid_sim_config.use_hibernation)
                else range(1)
            )
            if ti.static(not static_rigid_sim_config.is_backward)
            else (
                # Static inner loop for backward pass
                ti.static(range(static_rigid_sim_config.max_n_awake_dofs))
                if ti.static(static_rigid_sim_config.use_hibernation)
                else ti.static(range(1))
            )
        ):
            if i_1 < (rigid_global_info.n_awake_dofs[i_b] if ti.static(static_rigid_sim_config.use_hibernation) else 1):
                i_d = (
                    rigid_global_info.awake_dofs[i_1, i_b]
                    if ti.static(static_rigid_sim_config.use_hibernation)
                    else i_0
                )
                dofs_state.vel_prev[i_d, i_b] = dofs_state.vel[i_d, i_b]
                dofs_state.vel[i_d, i_b] = (
                    dofs_state.vel[i_d, i_b] + dofs_state.acc[i_d, i_b] * rigid_global_info.substep_dt[None]
                )


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_compute_mass_matrix(
    # taichi variables
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    decompose: ti.template(),
):
    func_compute_mass_matrix(
        implicit_damping=False,
        links_state=links_state,
        links_info=links_info,
        dofs_state=dofs_state,
        dofs_info=dofs_info,
        entities_info=entities_info,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
    )
    if decompose:
        func_factor_mass(
            implicit_damping=False,
            entities_info=entities_info,
            dofs_state=dofs_state,
            dofs_info=dofs_info,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
        )


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_init_invweight(
    envs_idx: ti.types.ndarray(),
    links_invweight: ti.types.ndarray(),
    dofs_invweight: ti.types.ndarray(),
    links_info: array_class.LinksInfo,
    dofs_info: array_class.DofsInfo,
    force_update: ti.template(),
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    EPS = rigid_global_info.EPS[None]

    if ti.static(static_rigid_sim_config.batch_links_info):
        ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
        for i_l, i_b_ in ti.ndrange(links_info.parent_idx.shape[0], envs_idx.shape[0]):
            i_b = envs_idx[i_b_]
            for j in ti.static(range(2)):
                if force_update or links_info.invweight[i_l, i_b][j] < EPS:
                    links_info.invweight[i_l, i_b][j] = links_invweight[i_b_, i_l, j]
    else:
        ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
        for i_l in range(links_info.parent_idx.shape[0]):
            for j in ti.static(range(2)):
                if force_update or links_info.invweight[i_l][j] < EPS:
                    links_info.invweight[i_l][j] = links_invweight[i_l, j]

    if ti.static(static_rigid_sim_config.batch_dofs_info):
        ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
        for i_d, i_b_ in ti.ndrange(dofs_info.dof_start.shape[0], envs_idx.shape[0]):
            i_b = envs_idx[i_b_]
            if force_update or dofs_info.invweight[i_d, i_b] < EPS:
                dofs_info.invweight[i_d, i_b] = dofs_invweight[i_b_, i_d]
    else:
        ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
        for i_d in range(dofs_info.dof_start.shape[0]):
            if force_update or dofs_info.invweight[i_d] < EPS:
                dofs_info.invweight[i_d] = dofs_invweight[i_d]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_init_meaninertia(
    envs_idx: ti.types.ndarray(),
    rigid_global_info: array_class.RigidGlobalInfo,
    entities_info: array_class.EntitiesInfo,
    static_rigid_sim_config: ti.template(),
):
    n_dofs = rigid_global_info.mass_mat.shape[0]
    n_entities = entities_info.n_links.shape[0]
    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_b_ in range(envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        if n_dofs > 0:
            rigid_global_info.meaninertia[i_b] = 0.0
            for i_e in range(n_entities):
                for i_d in range(entities_info.dof_start[i_e], entities_info.dof_end[i_e]):
                    I_d = [i_d, i_b] if ti.static(static_rigid_sim_config.batch_dofs_info) else i_d
                    rigid_global_info.meaninertia[i_b] += rigid_global_info.mass_mat[i_d, i_d, i_b]
                rigid_global_info.meaninertia[i_b] = rigid_global_info.meaninertia[i_b] / n_dofs
        else:
            rigid_global_info.meaninertia[i_b] = 1.0


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_init_dof_fields(
    # input np array
    dofs_motion_ang: ti.types.ndarray(),
    dofs_motion_vel: ti.types.ndarray(),
    dofs_limit: ti.types.ndarray(),
    dofs_invweight: ti.types.ndarray(),
    dofs_stiffness: ti.types.ndarray(),
    dofs_damping: ti.types.ndarray(),
    dofs_frictionloss: ti.types.ndarray(),
    dofs_armature: ti.types.ndarray(),
    dofs_kp: ti.types.ndarray(),
    dofs_kv: ti.types.ndarray(),
    dofs_force_range: ti.types.ndarray(),
    # taichi variables
    dofs_info: array_class.DofsInfo,
    dofs_state: array_class.DofsState,
    # we will use RigidGlobalInfo as typing after Hugh adds array_struct feature to gstaichi
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    n_dofs = dofs_state.ctrl_mode.shape[0]
    _B = dofs_state.ctrl_mode.shape[1]
    for I_d in ti.grouped(dofs_info.invweight):
        i_d = I_d[0]  # batching (if any) will be the second dim

        for j in ti.static(range(3)):
            dofs_info.motion_ang[I_d][j] = dofs_motion_ang[i_d, j]
            dofs_info.motion_vel[I_d][j] = dofs_motion_vel[i_d, j]

        for j in ti.static(range(2)):
            dofs_info.limit[I_d][j] = dofs_limit[i_d, j]
            dofs_info.force_range[I_d][j] = dofs_force_range[i_d, j]

        dofs_info.armature[I_d] = dofs_armature[i_d]
        dofs_info.invweight[I_d] = dofs_invweight[i_d]
        dofs_info.stiffness[I_d] = dofs_stiffness[i_d]
        dofs_info.damping[I_d] = dofs_damping[i_d]
        dofs_info.frictionloss[I_d] = dofs_frictionloss[i_d]
        dofs_info.kp[I_d] = dofs_kp[i_d]
        dofs_info.kv[I_d] = dofs_kv[i_d]

    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_d, i_b in ti.ndrange(n_dofs, _B):
        dofs_state.ctrl_mode[i_d, i_b] = gs.CTRL_MODE.FORCE
        dofs_state.ctrl_force[i_d, i_b] = gs.ti_float(0.0)

    if ti.static(static_rigid_sim_config.use_hibernation):
        ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
        for i_d, i_b in ti.ndrange(n_dofs, _B):
            dofs_state.hibernated[i_d, i_b] = False
            rigid_global_info.awake_dofs[i_d, i_b] = i_d

        ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
        for i_b in range(_B):
            rigid_global_info.n_awake_dofs[i_b] = n_dofs


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_init_link_fields(
    links_parent_idx: ti.types.ndarray(),
    links_root_idx: ti.types.ndarray(),
    links_q_start: ti.types.ndarray(),
    links_dof_start: ti.types.ndarray(),
    links_joint_start: ti.types.ndarray(),
    links_q_end: ti.types.ndarray(),
    links_dof_end: ti.types.ndarray(),
    links_joint_end: ti.types.ndarray(),
    links_invweight: ti.types.ndarray(),
    links_is_fixed: ti.types.ndarray(),
    links_pos: ti.types.ndarray(),
    links_quat: ti.types.ndarray(),
    links_inertial_pos: ti.types.ndarray(),
    links_inertial_quat: ti.types.ndarray(),
    links_inertial_i: ti.types.ndarray(),
    links_inertial_mass: ti.types.ndarray(),
    links_entity_idx: ti.types.ndarray(),
    # taichi variables
    links_info: array_class.LinksInfo,
    links_state: array_class.LinksState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    n_links = links_parent_idx.shape[0]
    _B = links_state.pos.shape[1]
    for I_l in ti.grouped(links_info.invweight):
        i_l = I_l[0]

        links_info.parent_idx[I_l] = links_parent_idx[i_l]
        links_info.root_idx[I_l] = links_root_idx[i_l]
        links_info.q_start[I_l] = links_q_start[i_l]
        links_info.joint_start[I_l] = links_joint_start[i_l]
        links_info.dof_start[I_l] = links_dof_start[i_l]
        links_info.q_end[I_l] = links_q_end[i_l]
        links_info.dof_end[I_l] = links_dof_end[i_l]
        links_info.joint_end[I_l] = links_joint_end[i_l]
        links_info.n_dofs[I_l] = links_dof_end[i_l] - links_dof_start[i_l]
        links_info.is_fixed[I_l] = links_is_fixed[i_l]
        links_info.entity_idx[I_l] = links_entity_idx[i_l]

        for j in ti.static(range(2)):
            links_info.invweight[I_l][j] = links_invweight[i_l, j]

        for j in ti.static(range(4)):
            links_info.quat[I_l][j] = links_quat[i_l, j]
            links_info.inertial_quat[I_l][j] = links_inertial_quat[i_l, j]

        for j in ti.static(range(3)):
            links_info.pos[I_l][j] = links_pos[i_l, j]
            links_info.inertial_pos[I_l][j] = links_inertial_pos[i_l, j]

        links_info.inertial_mass[I_l] = links_inertial_mass[i_l]
        for j1 in ti.static(range(3)):
            for j2 in ti.static(range(3)):
                links_info.inertial_i[I_l][j1, j2] = links_inertial_i[i_l, j1, j2]

    for i_l, i_b in ti.ndrange(n_links, _B):
        I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l

        # Update state for root fixed link. Their state will not be updated in forward kinematics later but can be manually changed by user.
        if links_info.parent_idx[I_l] == -1 and links_info.is_fixed[I_l]:
            for j in ti.static(range(4)):
                links_state.quat[i_l, i_b][j] = links_quat[i_l, j]

            for j in ti.static(range(3)):
                links_state.pos[i_l, i_b][j] = links_pos[i_l, j]

        for j in ti.static(range(3)):
            links_state.i_pos_shift[i_l, i_b][j] = 0.0
        links_state.mass_shift[i_l, i_b] = 0.0

    if ti.static(static_rigid_sim_config.use_hibernation):
        ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
        for i_l, i_b in ti.ndrange(n_links, _B):
            links_state.hibernated[i_l, i_b] = False
            rigid_global_info.awake_links[i_l, i_b] = i_l

        ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
        for i_b in range(_B):
            rigid_global_info.n_awake_links[i_b] = n_links


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_init_joint_fields(
    joints_type: ti.types.ndarray(),
    joints_sol_params: ti.types.ndarray(),
    joints_q_start: ti.types.ndarray(),
    joints_dof_start: ti.types.ndarray(),
    joints_q_end: ti.types.ndarray(),
    joints_dof_end: ti.types.ndarray(),
    joints_pos: ti.types.ndarray(),
    # taichi variables
    joints_info: array_class.JointsInfo,
    static_rigid_sim_config: ti.template(),
):
    for I_j in ti.grouped(joints_info.type):
        i_j = I_j[0]

        joints_info.type[I_j] = joints_type[i_j]
        joints_info.q_start[I_j] = joints_q_start[i_j]
        joints_info.dof_start[I_j] = joints_dof_start[i_j]
        joints_info.q_end[I_j] = joints_q_end[i_j]
        joints_info.dof_end[I_j] = joints_dof_end[i_j]
        joints_info.n_dofs[I_j] = joints_dof_end[i_j] - joints_dof_start[i_j]

        for j in ti.static(range(7)):
            joints_info.sol_params[I_j][j] = joints_sol_params[i_j, j]
        for j in ti.static(range(3)):
            joints_info.pos[I_j][j] = joints_pos[i_j, j]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_init_vert_fields(
    verts: ti.types.ndarray(),
    faces: ti.types.ndarray(),
    edges: ti.types.ndarray(),
    normals: ti.types.ndarray(),
    verts_geom_idx: ti.types.ndarray(),
    init_center_pos: ti.types.ndarray(),
    verts_state_idx: ti.types.ndarray(),
    is_fixed: ti.types.ndarray(),
    # taichi variables
    verts_info: array_class.VertsInfo,
    faces_info: array_class.FacesInfo,
    edges_info: array_class.EdgesInfo,
    static_rigid_sim_config: ti.template(),
):
    n_verts = verts.shape[0]
    n_faces = faces.shape[0]
    n_edges = edges.shape[0]
    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_v in range(n_verts):
        for j in ti.static(range(3)):
            verts_info.init_pos[i_v][j] = verts[i_v, j]
            verts_info.init_normal[i_v][j] = normals[i_v, j]
            verts_info.init_center_pos[i_v][j] = init_center_pos[i_v, j]

        verts_info.geom_idx[i_v] = verts_geom_idx[i_v]
        verts_info.verts_state_idx[i_v] = verts_state_idx[i_v]
        verts_info.is_fixed[i_v] = is_fixed[i_v]

    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_f in range(n_faces):
        for j in ti.static(range(3)):
            faces_info.verts_idx[i_f][j] = faces[i_f, j]
        faces_info.geom_idx[i_f] = verts_geom_idx[faces[i_f, 0]]

    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_ed in range(n_edges):
        edges_info.v0[i_ed] = edges[i_ed, 0]
        edges_info.v1[i_ed] = edges[i_ed, 1]
        # minus = verts_info.init_pos[edges[i_ed, 0]] - verts_info.init_pos[edges[i_ed, 1]]
        # edges_info.length[i_ed] = minus.norm()
        # FIXME: the line below does not work
        edges_info.length[i_ed] = (verts_info.init_pos[edges[i_ed, 0]] - verts_info.init_pos[edges[i_ed, 1]]).norm()


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_init_vvert_fields(
    vverts: ti.types.ndarray(),
    vfaces: ti.types.ndarray(),
    vnormals: ti.types.ndarray(),
    vverts_vgeom_idx: ti.types.ndarray(),
    # taichi variables
    vverts_info: array_class.VVertsInfo,
    vfaces_info: array_class.VFacesInfo,
    static_rigid_sim_config: ti.template(),
):
    n_vverts = vverts.shape[0]
    n_vfaces = vfaces.shape[0]
    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_vv in range(n_vverts):
        for j in ti.static(range(3)):
            vverts_info.init_pos[i_vv][j] = vverts[i_vv, j]
            vverts_info.init_vnormal[i_vv][j] = vnormals[i_vv, j]

        vverts_info.vgeom_idx[i_vv] = vverts_vgeom_idx[i_vv]

    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_vf in range(n_vfaces):
        for j in ti.static(range(3)):
            vfaces_info.vverts_idx[i_vf][j] = vfaces[i_vf, j]
        vfaces_info.vgeom_idx[i_vf] = vverts_vgeom_idx[vfaces[i_vf, 0]]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_init_geom_fields(
    geoms_pos: ti.types.ndarray(),
    geoms_center: ti.types.ndarray(),
    geoms_quat: ti.types.ndarray(),
    geoms_link_idx: ti.types.ndarray(),
    geoms_type: ti.types.ndarray(),
    geoms_friction: ti.types.ndarray(),
    geoms_sol_params: ti.types.ndarray(),
    geoms_vert_start: ti.types.ndarray(),
    geoms_face_start: ti.types.ndarray(),
    geoms_edge_start: ti.types.ndarray(),
    geoms_verts_state_start: ti.types.ndarray(),
    geoms_vert_end: ti.types.ndarray(),
    geoms_face_end: ti.types.ndarray(),
    geoms_edge_end: ti.types.ndarray(),
    geoms_verts_state_end: ti.types.ndarray(),
    geoms_data: ti.types.ndarray(),
    geoms_is_convex: ti.types.ndarray(),
    geoms_needs_coup: ti.types.ndarray(),
    geoms_contype: ti.types.ndarray(),
    geoms_conaffinity: ti.types.ndarray(),
    geoms_coup_softness: ti.types.ndarray(),
    geoms_coup_friction: ti.types.ndarray(),
    geoms_coup_restitution: ti.types.ndarray(),
    geoms_is_fixed: ti.types.ndarray(),
    geoms_is_decomp: ti.types.ndarray(),
    # taichi variables
    geoms_info: array_class.GeomsInfo,
    geoms_state: array_class.GeomsState,
    verts_info: array_class.VertsInfo,
    geoms_init_AABB: array_class.GeomsInitAABB,  # TODO: move to rigid global info
    static_rigid_sim_config: ti.template(),
):
    n_geoms = geoms_pos.shape[0]
    _B = geoms_state.friction_ratio.shape[1]
    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_g in range(n_geoms):
        for j in ti.static(range(3)):
            geoms_info.pos[i_g][j] = geoms_pos[i_g, j]
            geoms_info.center[i_g][j] = geoms_center[i_g, j]

        for j in ti.static(range(4)):
            geoms_info.quat[i_g][j] = geoms_quat[i_g, j]

        for j in ti.static(range(7)):
            geoms_info.data[i_g][j] = geoms_data[i_g, j]
            geoms_info.sol_params[i_g][j] = geoms_sol_params[i_g, j]

        geoms_info.vert_start[i_g] = geoms_vert_start[i_g]
        geoms_info.vert_end[i_g] = geoms_vert_end[i_g]
        geoms_info.vert_num[i_g] = geoms_vert_end[i_g] - geoms_vert_start[i_g]

        geoms_info.face_start[i_g] = geoms_face_start[i_g]
        geoms_info.face_end[i_g] = geoms_face_end[i_g]
        geoms_info.face_num[i_g] = geoms_face_end[i_g] - geoms_face_start[i_g]

        geoms_info.edge_start[i_g] = geoms_edge_start[i_g]
        geoms_info.edge_end[i_g] = geoms_edge_end[i_g]
        geoms_info.edge_num[i_g] = geoms_edge_end[i_g] - geoms_edge_start[i_g]

        geoms_info.verts_state_start[i_g] = geoms_verts_state_start[i_g]
        geoms_info.verts_state_end[i_g] = geoms_verts_state_end[i_g]

        geoms_info.link_idx[i_g] = geoms_link_idx[i_g]
        geoms_info.type[i_g] = geoms_type[i_g]
        geoms_info.friction[i_g] = geoms_friction[i_g]

        geoms_info.is_convex[i_g] = geoms_is_convex[i_g]
        geoms_info.needs_coup[i_g] = geoms_needs_coup[i_g]
        geoms_info.contype[i_g] = geoms_contype[i_g]
        geoms_info.conaffinity[i_g] = geoms_conaffinity[i_g]

        geoms_info.coup_softness[i_g] = geoms_coup_softness[i_g]
        geoms_info.coup_friction[i_g] = geoms_coup_friction[i_g]
        geoms_info.coup_restitution[i_g] = geoms_coup_restitution[i_g]

        geoms_info.is_fixed[i_g] = geoms_is_fixed[i_g]
        geoms_info.is_decomposed[i_g] = geoms_is_decomp[i_g]

        # compute init AABB.
        # Beware the ordering the this corners is critical and MUST NOT be changed as this order is used elsewhere
        # in the codebase, e.g. overlap estimation between two convex geometries using there bounding boxes.
        lower = gu.ti_vec3(ti.math.inf)
        upper = gu.ti_vec3(-ti.math.inf)
        for i_v in range(geoms_vert_start[i_g], geoms_vert_end[i_g]):
            lower = ti.min(lower, verts_info.init_pos[i_v])
            upper = ti.max(upper, verts_info.init_pos[i_v])
        geoms_init_AABB[i_g, 0] = ti.Vector([lower[0], lower[1], lower[2]], dt=gs.ti_float)
        geoms_init_AABB[i_g, 1] = ti.Vector([lower[0], lower[1], upper[2]], dt=gs.ti_float)
        geoms_init_AABB[i_g, 2] = ti.Vector([lower[0], upper[1], lower[2]], dt=gs.ti_float)
        geoms_init_AABB[i_g, 3] = ti.Vector([lower[0], upper[1], upper[2]], dt=gs.ti_float)
        geoms_init_AABB[i_g, 4] = ti.Vector([upper[0], lower[1], lower[2]], dt=gs.ti_float)
        geoms_init_AABB[i_g, 5] = ti.Vector([upper[0], lower[1], upper[2]], dt=gs.ti_float)
        geoms_init_AABB[i_g, 6] = ti.Vector([upper[0], upper[1], lower[2]], dt=gs.ti_float)
        geoms_init_AABB[i_g, 7] = ti.Vector([upper[0], upper[1], upper[2]], dt=gs.ti_float)

    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_g, i_b in ti.ndrange(n_geoms, _B):
        geoms_state.friction_ratio[i_g, i_b] = 1.0


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_adjust_link_inertia(
    link_idx: ti.i32,
    ratio: ti.f32,
    links_info: array_class.LinksInfo,
    static_rigid_sim_config: ti.template(),
):
    if ti.static(static_rigid_sim_config.batch_links_info):
        _B = links_info.root_idx.shape[1]
        for i_b in range(_B):
            for j in ti.static(range(2)):
                links_info.invweight[link_idx, i_b][j] /= ratio
            links_info.inertial_mass[link_idx, i_b] *= ratio
            for j1, j2 in ti.static(ti.ndrange(3, 3)):
                links_info.inertial_i[link_idx, i_b][j1, j2] *= ratio
    else:
        for j in ti.static(range(2)):
            links_info.invweight[link_idx][j] /= ratio
        links_info.inertial_mass[link_idx] *= ratio
        for j1, j2 in ti.static(ti.ndrange(3, 3)):
            links_info.inertial_i[link_idx][j1, j2] *= ratio


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_init_vgeom_fields(
    vgeoms_pos: ti.types.ndarray(),
    vgeoms_quat: ti.types.ndarray(),
    vgeoms_link_idx: ti.types.ndarray(),
    vgeoms_vvert_start: ti.types.ndarray(),
    vgeoms_vface_start: ti.types.ndarray(),
    vgeoms_vvert_end: ti.types.ndarray(),
    vgeoms_vface_end: ti.types.ndarray(),
    vgeoms_color: ti.types.ndarray(),
    # taichi variables
    vgeoms_info: array_class.VGeomsInfo,
    static_rigid_sim_config: ti.template(),
):
    n_vgeoms = vgeoms_pos.shape[0]
    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_vg in range(n_vgeoms):
        for j in ti.static(range(3)):
            vgeoms_info.pos[i_vg][j] = vgeoms_pos[i_vg, j]

        for j in ti.static(range(4)):
            vgeoms_info.quat[i_vg][j] = vgeoms_quat[i_vg, j]

        vgeoms_info.vvert_start[i_vg] = vgeoms_vvert_start[i_vg]
        vgeoms_info.vvert_end[i_vg] = vgeoms_vvert_end[i_vg]
        vgeoms_info.vvert_num[i_vg] = vgeoms_vvert_end[i_vg] - vgeoms_vvert_start[i_vg]

        vgeoms_info.vface_start[i_vg] = vgeoms_vface_start[i_vg]
        vgeoms_info.vface_end[i_vg] = vgeoms_vface_end[i_vg]
        vgeoms_info.vface_num[i_vg] = vgeoms_vface_end[i_vg] - vgeoms_vface_start[i_vg]

        vgeoms_info.link_idx[i_vg] = vgeoms_link_idx[i_vg]
        for j in ti.static(range(4)):
            vgeoms_info.color[i_vg][j] = vgeoms_color[i_vg, j]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_init_entity_fields(
    entities_dof_start: ti.types.ndarray(),
    entities_dof_end: ti.types.ndarray(),
    entities_link_start: ti.types.ndarray(),
    entities_link_end: ti.types.ndarray(),
    entities_geom_start: ti.types.ndarray(),
    entities_geom_end: ti.types.ndarray(),
    entities_gravity_compensation: ti.types.ndarray(),
    entities_is_local_collision_mask: ti.types.ndarray(),
    # taichi variables
    entities_info: array_class.EntitiesInfo,
    entities_state: array_class.EntitiesState,
    dofs_info: array_class.DofsInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    n_entities = entities_dof_start.shape[0]
    _B = entities_state.hibernated.shape[1]

    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_e in range(n_entities):
        entities_info.dof_start[i_e] = entities_dof_start[i_e]
        entities_info.dof_end[i_e] = entities_dof_end[i_e]
        entities_info.n_dofs[i_e] = entities_dof_end[i_e] - entities_dof_start[i_e]

        entities_info.link_start[i_e] = entities_link_start[i_e]
        entities_info.link_end[i_e] = entities_link_end[i_e]
        entities_info.n_links[i_e] = entities_link_end[i_e] - entities_link_start[i_e]

        entities_info.geom_start[i_e] = entities_geom_start[i_e]
        entities_info.geom_end[i_e] = entities_geom_end[i_e]
        entities_info.n_geoms[i_e] = entities_geom_end[i_e] - entities_geom_start[i_e]

        entities_info.gravity_compensation[i_e] = entities_gravity_compensation[i_e]
        entities_info.is_local_collision_mask[i_e] = entities_is_local_collision_mask[i_e]

        if ti.static(static_rigid_sim_config.batch_dofs_info):
            for i_d, i_b in ti.ndrange((entities_dof_start[i_e], entities_dof_end[i_e]), _B):
                dofs_info.dof_start[i_d, i_b] = entities_dof_start[i_e]
        else:
            for i_d in range(entities_dof_start[i_e], entities_dof_end[i_e]):
                dofs_info.dof_start[i_d] = entities_dof_start[i_e]

    if ti.static(static_rigid_sim_config.use_hibernation):
        ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
        for i_e, i_b in ti.ndrange(n_entities, _B):
            entities_state.hibernated[i_e, i_b] = False
            rigid_global_info.awake_entities[i_e, i_b] = i_e

        ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
        for i_b in range(_B):
            rigid_global_info.n_awake_entities[i_b] = n_entities


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_init_equality_fields(
    equalities_type: ti.types.ndarray(),
    equalities_eq_obj1id: ti.types.ndarray(),
    equalities_eq_obj2id: ti.types.ndarray(),
    equalities_eq_data: ti.types.ndarray(),
    equalities_eq_type: ti.types.ndarray(),
    equalities_sol_params: ti.types.ndarray(),
    # taichi variables
    equalities_info: array_class.EqualitiesInfo,
    static_rigid_sim_config: ti.template(),
):
    n_equalities = equalities_eq_obj1id.shape[0]
    _B = equalities_info.eq_obj1id.shape[1]

    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_eq, i_b in ti.ndrange(n_equalities, _B):
        equalities_info.eq_obj1id[i_eq, i_b] = equalities_eq_obj1id[i_eq]
        equalities_info.eq_obj2id[i_eq, i_b] = equalities_eq_obj2id[i_eq]
        equalities_info.eq_type[i_eq, i_b] = equalities_eq_type[i_eq]
        for j in ti.static(range(11)):
            equalities_info.eq_data[i_eq, i_b][j] = equalities_eq_data[i_eq, j]
        for j in ti.static(range(7)):
            equalities_info.sol_params[i_eq, i_b][j] = equalities_sol_params[i_eq, j]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_forward_dynamics(
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    joints_info: array_class.JointsInfo,
    entities_state: array_class.EntitiesState,
    entities_info: array_class.EntitiesInfo,
    geoms_state: array_class.GeomsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    contact_island_state: array_class.ContactIslandState,
):
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
    )


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_update_acc(
    dofs_state: array_class.DofsState,
    links_info: array_class.LinksInfo,
    links_state: array_class.LinksState,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    func_update_acc(
        update_cacc=True,
        dofs_state=dofs_state,
        links_info=links_info,
        links_state=links_state,
        entities_info=entities_info,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
    )


@ti.func
def func_vel_at_point(pos_world, link_idx, i_b, links_state: array_class.LinksState):
    """
    Velocity of a certain point on a rigid link.
    """
    vel_rot = links_state.cd_ang[link_idx, i_b].cross(pos_world - links_state.root_COM[link_idx, i_b])
    vel_lin = links_state.cd_vel[link_idx, i_b]
    return vel_rot + vel_lin


@ti.func
def func_compute_mass_matrix(
    implicit_damping: ti.template(),
    # taichi variables
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    # crb initialize
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_0, i_b in (
        ti.ndrange(1, links_state.pos.shape[1])
        if ti.static(static_rigid_sim_config.use_hibernation)
        else ti.ndrange(links_state.pos.shape[0], links_state.pos.shape[1])
    ):
        for i_1 in (
            (
                # Dynamic inner loop for forward pass
                range(rigid_global_info.n_awake_links[i_b])
                if ti.static(static_rigid_sim_config.use_hibernation)
                else range(1)
            )
            if ti.static(not static_rigid_sim_config.is_backward)
            else (
                # Static inner loop for backward pass
                ti.static(range(static_rigid_sim_config.max_n_awake_links))
                if ti.static(static_rigid_sim_config.use_hibernation)
                else ti.static(range(1))
            )
        ):
            if func_check_index_range(
                i_1, 0, rigid_global_info.n_awake_links[i_b], static_rigid_sim_config.use_hibernation
            ):
                i_l = (
                    rigid_global_info.awake_links[i_1, i_b]
                    if ti.static(static_rigid_sim_config.use_hibernation)
                    else i_0
                )

                links_state.crb_inertial[i_l, i_b] = links_state.cinr_inertial[i_l, i_b]
                links_state.crb_pos[i_l, i_b] = links_state.cinr_pos[i_l, i_b]
                links_state.crb_quat[i_l, i_b] = links_state.cinr_quat[i_l, i_b]
                links_state.crb_mass[i_l, i_b] = links_state.cinr_mass[i_l, i_b]

    # crb
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_0, i_b in (
        ti.ndrange(1, links_state.pos.shape[1])
        if ti.static(static_rigid_sim_config.use_hibernation)
        else ti.ndrange(entities_info.n_links.shape[0], links_state.pos.shape[1])
    ):
        for i_1 in (
            (
                # Dynamic inner loop for forward pass
                range(rigid_global_info.n_awake_entities[i_b])
                if ti.static(static_rigid_sim_config.use_hibernation)
                else ti.static(range(1))
            )
            if ti.static(not static_rigid_sim_config.is_backward)
            else (
                # Static inner loop for backward pass
                ti.static(range(static_rigid_sim_config.max_n_awake_entities))
                if ti.static(static_rigid_sim_config.use_hibernation)
                else ti.static(range(1))
            )
        ):
            if func_check_index_range(
                i_1, 0, rigid_global_info.n_awake_entities[i_b], static_rigid_sim_config.use_hibernation
            ):
                i_e = (
                    rigid_global_info.awake_entities[i_1, i_b]
                    if ti.static(static_rigid_sim_config.use_hibernation)
                    else i_0
                )

                for i in (
                    range(entities_info.n_links[i_e])
                    if ti.static(not static_rigid_sim_config.is_backward)
                    else ti.static(range(static_rigid_sim_config.max_n_links_per_entity))
                ):
                    if func_check_index_range(i, 0, entities_info.n_links[i_e], static_rigid_sim_config.is_backward):
                        i_l = entities_info.link_end[i_e] - 1 - i
                        I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l
                        i_p = links_info.parent_idx[I_l]

                        if i_p != -1:
                            func_atomic_add_if_backward_2d(
                                links_state.crb_inertial,
                                i_p,
                                i_b,
                                links_state.crb_inertial[i_l, i_b],
                                static_rigid_sim_config,
                            )
                            func_atomic_add_if_backward_2d(
                                links_state.crb_mass, i_p, i_b, links_state.crb_mass[i_l, i_b], static_rigid_sim_config
                            )
                            func_atomic_add_if_backward_2d(
                                links_state.crb_pos, i_p, i_b, links_state.crb_pos[i_l, i_b], static_rigid_sim_config
                            )
                            func_atomic_add_if_backward_2d(
                                links_state.crb_quat, i_p, i_b, links_state.crb_quat[i_l, i_b], static_rigid_sim_config
                            )

    # mass_mat
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_0, i_b in (
        ti.ndrange(1, links_state.pos.shape[1])
        if ti.static(static_rigid_sim_config.use_hibernation)
        else ti.ndrange(links_state.pos.shape[0], links_state.pos.shape[1])
    ):
        for i_1 in (
            (
                # Dynamic inner loop for forward pass
                range(rigid_global_info.n_awake_links[i_b])
                if ti.static(static_rigid_sim_config.use_hibernation)
                else range(1)
            )
            if ti.static(not static_rigid_sim_config.is_backward)
            else (
                # Static inner loop for backward pass
                ti.static(range(static_rigid_sim_config.max_n_awake_links))
                if ti.static(static_rigid_sim_config.use_hibernation)
                else ti.static(range(1))
            )
        ):
            if func_check_index_range(
                i_1, 0, rigid_global_info.n_awake_links[i_b], static_rigid_sim_config.use_hibernation
            ):
                i_l = (
                    rigid_global_info.awake_links[i_1, i_b]
                    if ti.static(static_rigid_sim_config.use_hibernation)
                    else i_0
                )
                I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l

                for i_d_ in (
                    range(links_info.dof_start[I_l], links_info.dof_end[I_l])
                    if ti.static(not static_rigid_sim_config.is_backward)
                    else ti.static(range(static_rigid_sim_config.max_n_dofs_per_link))
                ):
                    i_d = (
                        i_d_ if ti.static(not static_rigid_sim_config.is_backward) else links_info.dof_start[I_l] + i_d_
                    )

                    if func_check_index_range(
                        i_d, links_info.dof_start[I_l], links_info.dof_end[I_l], static_rigid_sim_config.is_backward
                    ):
                        dofs_state.f_ang[i_d, i_b], dofs_state.f_vel[i_d, i_b] = gu.inertial_mul(
                            links_state.crb_pos[i_l, i_b],
                            links_state.crb_inertial[i_l, i_b],
                            links_state.crb_mass[i_l, i_b],
                            dofs_state.cdof_vel[i_d, i_b],
                            dofs_state.cdof_ang[i_d, i_b],
                        )

    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_0, i_b in (
        ti.ndrange(1, links_state.pos.shape[1])
        if ti.static(static_rigid_sim_config.use_hibernation)
        else ti.ndrange(entities_info.n_links.shape[0], links_state.pos.shape[1])
    ):
        for i_1 in (
            (
                # Dynamic inner loop for forward pass
                range(rigid_global_info.n_awake_entities[i_b])
                if ti.static(static_rigid_sim_config.use_hibernation)
                else range(1)
            )
            if ti.static(not static_rigid_sim_config.is_backward)
            else (
                # Static inner loop for backward pass
                ti.static(range(static_rigid_sim_config.max_n_awake_entities))
                if ti.static(static_rigid_sim_config.use_hibernation)
                else ti.static(range(1))
            )
        ):
            if func_check_index_range(
                i_1, 0, rigid_global_info.n_awake_entities[i_b], static_rigid_sim_config.use_hibernation
            ):
                i_e = (
                    rigid_global_info.awake_entities[i_1, i_b]
                    if ti.static(static_rigid_sim_config.use_hibernation)
                    else i_0
                )

                for i_d_, j_d_ in (
                    (
                        # Dynamic inner loop for forward pass
                        ti.ndrange(
                            (entities_info.dof_start[i_e], entities_info.dof_end[i_e]),
                            (entities_info.dof_start[i_e], entities_info.dof_end[i_e]),
                        )
                    )
                    if ti.static(not static_rigid_sim_config.is_backward)
                    else (
                        # Static inner loop for backward pass
                        ti.static(
                            ti.ndrange(
                                static_rigid_sim_config.max_n_dofs_per_entity,
                                static_rigid_sim_config.max_n_dofs_per_entity,
                            )
                        )
                    )
                ):
                    i_d = (
                        i_d_
                        if ti.static(not static_rigid_sim_config.is_backward)
                        else entities_info.dof_start[i_e] + i_d_
                    )
                    j_d = (
                        j_d_
                        if ti.static(not static_rigid_sim_config.is_backward)
                        else entities_info.dof_start[i_e] + j_d_
                    )

                    if func_check_index_range(
                        i_d,
                        entities_info.dof_start[i_e],
                        entities_info.dof_end[i_e],
                        static_rigid_sim_config.is_backward,
                    ) and func_check_index_range(
                        j_d,
                        entities_info.dof_start[i_e],
                        entities_info.dof_end[i_e],
                        static_rigid_sim_config.is_backward,
                    ):
                        rigid_global_info.mass_mat[i_d, j_d, i_b] = (
                            dofs_state.f_ang[i_d, i_b].dot(dofs_state.cdof_ang[j_d, i_b])
                            + dofs_state.f_vel[i_d, i_b].dot(dofs_state.cdof_vel[j_d, i_b])
                        ) * rigid_global_info.mass_parent_mask[i_d, j_d]

                if ti.static(not static_rigid_sim_config.is_backward):
                    for i_d in range(entities_info.dof_start[i_e], entities_info.dof_end[i_e]):
                        for j_d in range(i_d + 1, entities_info.dof_end[i_e]):
                            rigid_global_info.mass_mat[i_d, j_d, i_b] = rigid_global_info.mass_mat[j_d, i_d, i_b]
                else:
                    for i_d_, j_d_ in ti.static(
                        ti.ndrange(
                            static_rigid_sim_config.max_n_dofs_per_entity,
                            static_rigid_sim_config.max_n_dofs_per_entity,
                        )
                    ):
                        i_d = entities_info.dof_start[i_e] + i_d_
                        j_d = entities_info.dof_start[i_e] + j_d_

                        if i_d < entities_info.dof_end[i_e] and j_d < entities_info.dof_end[i_e] and j_d > i_d:
                            rigid_global_info.mass_mat[i_d, j_d, i_b] = rigid_global_info.mass_mat[j_d, i_d, i_b]

    # Take into account motor armature
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_d, i_b in ti.ndrange(dofs_state.f_ang.shape[0], links_state.pos.shape[1]):
        I_d = [i_d, i_b] if ti.static(static_rigid_sim_config.batch_dofs_info) else i_d
        func_atomic_add_if_backward_3d(
            rigid_global_info.mass_mat, i_d, i_d, i_b, dofs_info.armature[I_d], static_rigid_sim_config
        )

    # Take into account first-order correction terms for implicit integration scheme right away
    if ti.static(implicit_damping):
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_d, i_b in ti.ndrange(dofs_state.f_ang.shape[0], links_state.pos.shape[1]):
            I_d = [i_d, i_b] if ti.static(static_rigid_sim_config.batch_dofs_info) else i_d
            rigid_global_info.mass_mat[i_d, i_d, i_b] += dofs_info.damping[I_d] * rigid_global_info.substep_dt[None]
            if (
                dofs_state.ctrl_mode[i_d, i_b] == gs.CTRL_MODE.POSITION
                or dofs_state.ctrl_mode[i_d, i_b] == gs.CTRL_MODE.VELOCITY
            ):
                # qM += d qfrc_actuator / d qvel
                rigid_global_info.mass_mat[i_d, i_d, i_b] += dofs_info.kv[I_d] * rigid_global_info.substep_dt[None]


@ti.func
def func_factor_mass(
    implicit_damping: ti.template(),
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    """
    Compute Cholesky decomposition (L^T @ D @ L) of mass matrix.
    """
    if ti.static(not static_rigid_sim_config.is_backward):
        _B = dofs_state.ctrl_mode.shape[1]
        n_entities = entities_info.n_links.shape[0]

        ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
        for i_0, i_b in (
            ti.ndrange(1, _B) if ti.static(static_rigid_sim_config.use_hibernation) else ti.ndrange(n_entities, _B)
        ):
            for i_1 in (
                (
                    # Dynamic inner loop for forward pass
                    range(rigid_global_info.n_awake_entities[i_b])
                    if ti.static(static_rigid_sim_config.use_hibernation)
                    else range(1)
                )
                if ti.static(not static_rigid_sim_config.is_backward)
                else (
                    # Static inner loop for backward pass
                    ti.static(range(static_rigid_sim_config.max_n_awake_entities))
                    if ti.static(static_rigid_sim_config.use_hibernation)
                    else ti.static(range(1))
                )
            ):
                if func_check_index_range(
                    i_1, 0, rigid_global_info.n_awake_entities[i_b], static_rigid_sim_config.use_hibernation
                ):
                    i_e = (
                        rigid_global_info.awake_entities[i_1, i_b]
                        if ti.static(static_rigid_sim_config.use_hibernation)
                        else i_0
                    )

                    if rigid_global_info.mass_mat_mask[i_e, i_b]:
                        entity_dof_start = entities_info.dof_start[i_e]
                        entity_dof_end = entities_info.dof_end[i_e]
                        n_dofs = entities_info.n_dofs[i_e]

                        for i_d_ in (
                            range(entity_dof_start, entity_dof_end)
                            if ti.static(not static_rigid_sim_config.is_backward)
                            else ti.static(range(static_rigid_sim_config.max_n_dofs_per_entity))
                        ):
                            i_d = (
                                i_d_
                                if ti.static(not static_rigid_sim_config.is_backward)
                                else entities_info.dof_start[i_e] + i_d_
                            )

                            if func_check_index_range(
                                i_d, entity_dof_start, entity_dof_end, static_rigid_sim_config.is_backward
                            ):
                                for j_d_ in (
                                    range(entity_dof_start, i_d + 1)
                                    if ti.static(not static_rigid_sim_config.is_backward)
                                    else ti.static(range(static_rigid_sim_config.max_n_dofs_per_entity))
                                ):
                                    j_d = (
                                        j_d_
                                        if ti.static(not static_rigid_sim_config.is_backward)
                                        else entities_info.dof_start[i_e] + j_d_
                                    )

                                    if func_check_index_range(
                                        j_d, entity_dof_start, i_d + 1, static_rigid_sim_config.is_backward
                                    ):
                                        rigid_global_info.mass_mat_L[i_d, j_d, i_b] = rigid_global_info.mass_mat[
                                            i_d, j_d, i_b
                                        ]

                                if ti.static(implicit_damping):
                                    I_d = [i_d, i_b] if ti.static(static_rigid_sim_config.batch_dofs_info) else i_d
                                    rigid_global_info.mass_mat_L[i_d, i_d, i_b] += (
                                        dofs_info.damping[I_d] * rigid_global_info.substep_dt[None]
                                    )
                                    if ti.static(static_rigid_sim_config.integrator == gs.integrator.implicitfast):
                                        if (dofs_state.ctrl_mode[i_d, i_b] == gs.CTRL_MODE.POSITION) or (
                                            dofs_state.ctrl_mode[i_d, i_b] == gs.CTRL_MODE.VELOCITY
                                        ):
                                            rigid_global_info.mass_mat_L[i_d, i_d, i_b] += (
                                                dofs_info.kv[I_d] * rigid_global_info.substep_dt[None]
                                            )

                        for i_d_ in (
                            range(n_dofs)
                            if ti.static(not static_rigid_sim_config.is_backward)
                            else ti.static(range(static_rigid_sim_config.max_n_dofs_per_entity))
                        ):
                            if func_check_index_range(i_d_, 0, n_dofs, static_rigid_sim_config.is_backward):
                                i_d = entity_dof_end - i_d_ - 1
                                rigid_global_info.mass_mat_D_inv[i_d, i_b] = (
                                    1.0 / rigid_global_info.mass_mat_L[i_d, i_d, i_b]
                                )

                                for j_d_ in (
                                    range(i_d - entity_dof_start)
                                    if ti.static(not static_rigid_sim_config.is_backward)
                                    else ti.static(range(static_rigid_sim_config.max_n_dofs_per_entity))
                                ):
                                    if func_check_index_range(
                                        j_d_, 0, i_d - entity_dof_start, static_rigid_sim_config.is_backward
                                    ):
                                        j_d = i_d - j_d_ - 1
                                        a = (
                                            rigid_global_info.mass_mat_L[i_d, j_d, i_b]
                                            * rigid_global_info.mass_mat_D_inv[i_d, i_b]
                                        )

                                        for k_d_ in (
                                            range(entity_dof_start, j_d + 1)
                                            if ti.static(not static_rigid_sim_config.is_backward)
                                            else ti.static(range(static_rigid_sim_config.max_n_dofs_per_entity))
                                        ):
                                            k_d = (
                                                k_d_
                                                if ti.static(not static_rigid_sim_config.is_backward)
                                                else entities_info.dof_start[i_e] + k_d_
                                            )
                                            if func_check_index_range(
                                                k_d, entity_dof_start, j_d + 1, static_rigid_sim_config.is_backward
                                            ):
                                                rigid_global_info.mass_mat_L[j_d, k_d, i_b] -= (
                                                    a * rigid_global_info.mass_mat_L[i_d, k_d, i_b]
                                                )
                                        rigid_global_info.mass_mat_L[i_d, j_d, i_b] = a

                                # FIXME: Diagonal coeffs of L are ignored in computations, so no need to update them.
                                rigid_global_info.mass_mat_L[i_d, i_d, i_b] = 1.0

    else:
        # Cholesky decomposition that has safe access pattern and robust handling of divide by zero for AD. Even though
        # it is logically equivalent to the above block, it shows slightly numerical difference in the result, and thus
        # it fails for a unit test ("test_urdf_rope"), while passing all the others. TODO: Investigate if we can fix this
        # and only use this block.

        # Assume this is the outermost loop
        ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL))
        for i_e, i_b in ti.ndrange(entities_info.n_links.shape[0], dofs_state.ctrl_mode.shape[1]):
            if rigid_global_info.mass_mat_mask[i_e, i_b]:
                entity_dof_start = entities_info.dof_start[i_e]
                entity_dof_end = entities_info.dof_end[i_e]
                n_dofs = entities_info.n_dofs[i_e]

                for i_d0 in (
                    range(n_dofs)
                    if ti.static(not static_rigid_sim_config.is_backward)
                    else ti.static(range(static_rigid_sim_config.max_n_dofs_per_entity))
                ):
                    if func_check_index_range(i_d0, 0, n_dofs, static_rigid_sim_config.is_backward):
                        i_d = entity_dof_start + i_d0
                        i_pr = (entity_dof_start + entity_dof_end - 1) - i_d
                        for j_d_ in (
                            range(entity_dof_start, i_d + 1)
                            if ti.static(not static_rigid_sim_config.is_backward)
                            else ti.static(range(static_rigid_sim_config.max_n_dofs_per_entity))
                        ):
                            j_d = (
                                j_d_
                                if ti.static(not static_rigid_sim_config.is_backward)
                                else (j_d_ + entities_info.dof_start[i_e])
                            )
                            j_pr = (entity_dof_start + entity_dof_end - 1) - j_d
                            if func_check_index_range(
                                j_d, entity_dof_start, i_d + 1, static_rigid_sim_config.is_backward
                            ):
                                rigid_global_info.mass_mat_L_bw[0, i_pr, j_pr, i_b] = rigid_global_info.mass_mat[
                                    i_d, j_d, i_b
                                ]
                                rigid_global_info.mass_mat_L_bw[0, j_pr, i_pr, i_b] = rigid_global_info.mass_mat[
                                    i_d, j_d, i_b
                                ]

                        if ti.static(implicit_damping):
                            I_d = [i_d, i_b] if ti.static(static_rigid_sim_config.batch_dofs_info) else i_d
                            rigid_global_info.mass_mat_L_bw[0, i_pr, i_pr, i_b] += (
                                dofs_info.damping[I_d] * rigid_global_info.substep_dt[None]
                            )
                            if ti.static(static_rigid_sim_config.integrator == gs.integrator.implicitfast):
                                if (
                                    dofs_state.ctrl_mode[i_d, i_b] == gs.CTRL_MODE.POSITION
                                    or dofs_state.ctrl_mode[i_d, i_b] == gs.CTRL_MODE.VELOCITY
                                ):
                                    rigid_global_info.mass_mat_L_bw[0, i_pr, i_pr, i_b] += (
                                        dofs_info.kv[I_d] * rigid_global_info.substep_dt[None]
                                    )

                # Cholesky-Banachiewicz algorithm (in the perturbed indices), access pattern is safe for autodiff
                # https://en.wikipedia.org/wiki/Cholesky_decomposition
                for p_i0 in (
                    range(n_dofs)
                    if ti.static(not static_rigid_sim_config.is_backward)
                    else ti.static(range(static_rigid_sim_config.max_n_dofs_per_entity))
                ):
                    for p_j0 in (
                        range(p_i0 + 1)
                        if ti.static(not static_rigid_sim_config.is_backward)
                        else ti.static(range(static_rigid_sim_config.max_n_dofs_per_entity))
                    ):
                        if func_check_index_range(
                            p_i0, 0, n_dofs, static_rigid_sim_config.is_backward
                        ) and func_check_index_range(p_j0, 0, p_i0 + 1, static_rigid_sim_config.is_backward):
                            # j_pr <= i_pr
                            i_pr = entity_dof_start + p_i0
                            j_pr = entity_dof_start + p_j0

                            sum = gs.ti_float(0.0)
                            for p_k0 in (
                                range(p_j0)
                                if ti.static(not static_rigid_sim_config.is_backward)
                                else ti.static(range(static_rigid_sim_config.max_n_dofs_per_entity))
                            ):
                                # k_pr < j_pr
                                if func_check_index_range(p_k0, 0, p_j0, static_rigid_sim_config.is_backward):
                                    k_pr = entity_dof_start + p_k0
                                    sum += (
                                        rigid_global_info.mass_mat_L_bw[1, i_pr, k_pr, i_b]
                                        * rigid_global_info.mass_mat_L_bw[1, j_pr, k_pr, i_b]
                                    )

                            a = rigid_global_info.mass_mat_L_bw[0, i_pr, j_pr, i_b] - sum
                            b = ti.math.clamp(rigid_global_info.mass_mat_L_bw[1, j_pr, j_pr, i_b], gs.EPS, ti.math.inf)
                            if p_i0 == p_j0:
                                rigid_global_info.mass_mat_L_bw[1, i_pr, j_pr, i_b] = ti.sqrt(
                                    ti.math.clamp(a, gs.EPS, ti.math.inf)
                                )
                            else:
                                rigid_global_info.mass_mat_L_bw[1, i_pr, j_pr, i_b] = a / b

                for i_d0 in (
                    range(n_dofs)
                    if ti.static(not static_rigid_sim_config.is_backward)
                    else ti.static(range(static_rigid_sim_config.max_n_dofs_per_entity))
                ):
                    for i_d1 in (
                        range(i_d0 + 1)
                        if ti.static(not static_rigid_sim_config.is_backward)
                        else ti.static(range(static_rigid_sim_config.max_n_dofs_per_entity))
                    ):
                        if func_check_index_range(
                            i_d0, 0, n_dofs, static_rigid_sim_config.is_backward
                        ) and func_check_index_range(i_d1, 0, i_d0 + 1, static_rigid_sim_config.is_backward):
                            i_d = entity_dof_start + i_d0
                            j_d = entity_dof_start + i_d1
                            i_pr = (entity_dof_start + entity_dof_end - 1) - i_d
                            j_pr = (entity_dof_start + entity_dof_end - 1) - j_d

                            a = rigid_global_info.mass_mat_L_bw[1, i_pr, i_pr, i_b]
                            rigid_global_info.mass_mat_L[i_d, j_d, i_b] = rigid_global_info.mass_mat_L_bw[
                                1, j_pr, i_pr, i_b
                            ] / ti.math.clamp(a, gs.EPS, ti.math.inf)

                            if i_d == j_d:
                                rigid_global_info.mass_mat_D_inv[i_d, i_b] = 1.0 / (
                                    ti.math.clamp(a**2, gs.EPS, ti.math.inf)
                                )


@ti.func
def func_solve_mass_batched(
    vec: array_class.V_ANNOTATION,
    out: array_class.V_ANNOTATION,
    out_bw: array_class.V_ANNOTATION,
    i_b: ti.int32,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    # This loop is considered an inner loop
    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_0 in (
        (
            # Dynamic inner loop for forward pass
            range(rigid_global_info.n_awake_entities[i_b])
            if ti.static(static_rigid_sim_config.use_hibernation)
            else range(entities_info.n_links.shape[0])
        )
        if ti.static(not static_rigid_sim_config.is_backward)
        else (
            # Static inner loop for backward pass
            ti.static(range(static_rigid_sim_config.max_n_awake_entities))
            if ti.static(static_rigid_sim_config.use_hibernation)
            else ti.static(range(static_rigid_sim_config.max_n_links_per_entity))
        )
    ):
        if func_check_index_range(i_0, 0, rigid_global_info.n_awake_entities[i_b], static_rigid_sim_config.is_backward):
            i_e = (
                rigid_global_info.awake_entities[i_0, i_b]
                if ti.static(static_rigid_sim_config.use_hibernation)
                else i_0
            )

            if rigid_global_info.mass_mat_mask[i_e, i_b]:
                entity_dof_start = entities_info.dof_start[i_e]
                entity_dof_end = entities_info.dof_end[i_e]
                n_dofs = entities_info.n_dofs[i_e]

                # Step 1: Solve w st. L^T @ w = y
                for i_d_ in (
                    range(n_dofs)
                    if ti.static(not static_rigid_sim_config.is_backward)
                    else ti.static(range(static_rigid_sim_config.max_n_dofs_per_entity))
                ):
                    if func_check_index_range(i_d_, 0, n_dofs, static_rigid_sim_config.is_backward):
                        i_d = entity_dof_end - i_d_ - 1
                        if ti.static(static_rigid_sim_config.is_backward):
                            out_bw[0, i_d, i_b] = vec[i_d, i_b]
                        else:
                            out[i_d, i_b] = vec[i_d, i_b]

                        for j_d_ in (
                            range(i_d + 1, entity_dof_end)
                            if ti.static(not static_rigid_sim_config.is_backward)
                            else ti.static(range(static_rigid_sim_config.max_n_dofs_per_entity))
                        ):
                            j_d = (
                                j_d_
                                if ti.static(not static_rigid_sim_config.is_backward)
                                else (j_d_ + entities_info.dof_start[i_e])
                            )
                            if func_check_index_range(
                                j_d, i_d + 1, entity_dof_end, static_rigid_sim_config.is_backward
                            ):
                                # Since we read out[j_d, i_b], and j_d > i_d, which means that out[j_d, i_b] is already
                                # finalized at this point, we don't need to care about AD mutation rule.
                                if ti.static(static_rigid_sim_config.is_backward):
                                    out_bw[0, i_d, i_b] += -(
                                        rigid_global_info.mass_mat_L[j_d, i_d, i_b] * out_bw[0, j_d, i_b]
                                    )
                                else:
                                    out[i_d, i_b] -= rigid_global_info.mass_mat_L[j_d, i_d, i_b] * out[j_d, i_b]

                # Step 2: z = D^{-1} w
                for i_d_ in (
                    range(entity_dof_start, entity_dof_end)
                    if ti.static(not static_rigid_sim_config.is_backward)
                    else ti.static(range(static_rigid_sim_config.max_n_dofs_per_entity))
                ):
                    i_d = (
                        i_d_
                        if ti.static(not static_rigid_sim_config.is_backward)
                        else (i_d_ + entities_info.dof_start[i_e])
                    )
                    if func_check_index_range(
                        i_d, entity_dof_start, entity_dof_end, static_rigid_sim_config.is_backward
                    ):
                        if ti.static(static_rigid_sim_config.is_backward):
                            out_bw[1, i_d, i_b] = out_bw[0, i_d, i_b] * rigid_global_info.mass_mat_D_inv[i_d, i_b]
                        else:
                            out[i_d, i_b] *= rigid_global_info.mass_mat_D_inv[i_d, i_b]

                # Step 3: Solve x st. L @ x = z
                for i_d_ in (
                    range(entity_dof_start, entity_dof_end)
                    if ti.static(not static_rigid_sim_config.is_backward)
                    else ti.static(range(static_rigid_sim_config.max_n_dofs_per_entity))
                ):
                    i_d = (
                        i_d_
                        if ti.static(not static_rigid_sim_config.is_backward)
                        else (i_d_ + entities_info.dof_start[i_e])
                    )
                    if func_check_index_range(
                        i_d, entity_dof_start, entity_dof_end, static_rigid_sim_config.is_backward
                    ):
                        curr_out = out[i_d, i_b]
                        if ti.static(static_rigid_sim_config.is_backward):
                            curr_out = out_bw[1, i_d, i_b]

                        for j_d_ in (
                            range(entity_dof_start, i_d)
                            if ti.static(not static_rigid_sim_config.is_backward)
                            else ti.static(range(static_rigid_sim_config.max_n_dofs_per_entity))
                        ):
                            j_d = (
                                j_d_
                                if ti.static(not static_rigid_sim_config.is_backward)
                                else (j_d_ + entities_info.dof_start[i_e])
                            )
                            if func_check_index_range(j_d, entity_dof_start, i_d, static_rigid_sim_config.is_backward):
                                if ti.static(static_rigid_sim_config.is_backward):
                                    curr_out += -(rigid_global_info.mass_mat_L[i_d, j_d, i_b] * out[j_d, i_b])
                                else:
                                    out[i_d, i_b] -= rigid_global_info.mass_mat_L[i_d, j_d, i_b] * out[j_d, i_b]

                        if ti.static(static_rigid_sim_config.is_backward):
                            out[i_d, i_b] = curr_out


@ti.func
def func_solve_mass(
    vec: array_class.V_ANNOTATION,
    out: array_class.V_ANNOTATION,
    out_bw: array_class.V_ANNOTATION,  # Should not be None if backward
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    # This loop must be the outermost loop to be differentiable
    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_b in range(out.shape[1]):
        func_solve_mass_batched(
            vec,
            out,
            out_bw,
            i_b,
            entities_info=entities_info,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
        )


# @@@@@@@@@ Composer starts here
# decomposed kernels should happen in the block below. This block will be handled by composer and composed into a single kernel
@ti.func
def func_forward_dynamics(
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    joints_info: array_class.JointsInfo,
    entities_state: array_class.EntitiesState,
    entities_info: array_class.EntitiesInfo,
    geoms_state: array_class.GeomsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    contact_island_state: array_class.ContactIslandState,
):
    func_compute_mass_matrix(
        implicit_damping=ti.static(static_rigid_sim_config.integrator == gs.integrator.approximate_implicitfast),
        links_state=links_state,
        links_info=links_info,
        dofs_state=dofs_state,
        dofs_info=dofs_info,
        entities_info=entities_info,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
    )
    func_factor_mass(
        implicit_damping=False,
        entities_info=entities_info,
        dofs_state=dofs_state,
        dofs_info=dofs_info,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
    )
    func_torque_and_passive_force(
        entities_state=entities_state,
        entities_info=entities_info,
        dofs_state=dofs_state,
        dofs_info=dofs_info,
        links_state=links_state,
        links_info=links_info,
        joints_info=joints_info,
        geoms_state=geoms_state,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
        contact_island_state=contact_island_state,
    )
    func_update_acc(
        update_cacc=False,
        dofs_state=dofs_state,
        links_info=links_info,
        links_state=links_state,
        entities_info=entities_info,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
    )
    func_update_force(
        links_state=links_state,
        links_info=links_info,
        entities_info=entities_info,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
    )
    # self._func_actuation()
    func_bias_force(
        dofs_state=dofs_state,
        links_state=links_state,
        links_info=links_info,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
    )
    func_compute_qacc(
        dofs_state=dofs_state,
        entities_info=entities_info,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
    )


@ti.kernel
def kernel_forward_dynamics_without_qacc(
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    joints_info: array_class.JointsInfo,
    entities_state: array_class.EntitiesState,
    entities_info: array_class.EntitiesInfo,
    geoms_state: array_class.GeomsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    contact_island_state: array_class.ContactIslandState,
):
    func_compute_mass_matrix(
        implicit_damping=ti.static(static_rigid_sim_config.integrator == gs.integrator.approximate_implicitfast),
        links_state=links_state,
        links_info=links_info,
        dofs_state=dofs_state,
        dofs_info=dofs_info,
        entities_info=entities_info,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
    )
    func_factor_mass(
        implicit_damping=False,
        entities_info=entities_info,
        dofs_state=dofs_state,
        dofs_info=dofs_info,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
    )
    func_torque_and_passive_force(
        entities_state=entities_state,
        entities_info=entities_info,
        dofs_state=dofs_state,
        dofs_info=dofs_info,
        links_state=links_state,
        links_info=links_info,
        joints_info=joints_info,
        geoms_state=geoms_state,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
        contact_island_state=contact_island_state,
    )
    func_update_acc(
        update_cacc=False,
        dofs_state=dofs_state,
        links_info=links_info,
        links_state=links_state,
        entities_info=entities_info,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
    )
    func_update_force(
        links_state=links_state,
        links_info=links_info,
        entities_info=entities_info,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
    )
    # self._func_actuation()
    func_bias_force(
        dofs_state=dofs_state,
        links_state=links_state,
        links_info=links_info,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
    )


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_clear_external_force(
    links_state: array_class.LinksState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    func_clear_external_force(
        links_state=links_state,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
    )


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_update_cartesian_space(
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    joints_state: array_class.JointsState,
    joints_info: array_class.JointsInfo,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    geoms_info: array_class.GeomsInfo,
    geoms_state: array_class.GeomsState,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    force_update_fixed_geoms: ti.template(),
):
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(links_state.pos.shape[1]):
        func_update_cartesian_space(
            i_b=i_b,
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
            force_update_fixed_geoms=force_update_fixed_geoms,
        )


@ti.func
def func_update_cartesian_space(
    i_b,
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    joints_state: array_class.JointsState,
    joints_info: array_class.JointsInfo,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    geoms_info: array_class.GeomsInfo,
    geoms_state: array_class.GeomsState,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    force_update_fixed_geoms: ti.template(),
):
    func_forward_kinematics(
        i_b,
        links_state=links_state,
        links_info=links_info,
        joints_state=joints_state,
        joints_info=joints_info,
        dofs_state=dofs_state,
        dofs_info=dofs_info,
        entities_info=entities_info,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
    )
    func_COM_links(
        i_b,
        links_state=links_state,
        links_info=links_info,
        joints_state=joints_state,
        joints_info=joints_info,
        dofs_state=dofs_state,
        dofs_info=dofs_info,
        entities_info=entities_info,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
    )
    func_update_geoms(
        i_b=i_b,
        entities_info=entities_info,
        geoms_info=geoms_info,
        geoms_state=geoms_state,
        links_state=links_state,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
        force_update_fixed_geoms=force_update_fixed_geoms,
    )


@ti.kernel(fastcache=gs.use_fastcache)
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
    static_rigid_sim_config: ti.template(),
    contact_island_state: array_class.ContactIslandState,
):
    if ti.static(static_rigid_sim_config.enable_mujoco_compatibility):
        _B = links_state.pos.shape[1]
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(_B):
            func_update_cartesian_space(
                i_b=i_b,
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
            )
            func_forward_velocity(
                i_b=i_b,
                entities_info=entities_info,
                links_info=links_info,
                links_state=links_state,
                joints_info=joints_info,
                dofs_state=dofs_state,
                rigid_global_info=rigid_global_info,
                static_rigid_sim_config=static_rigid_sim_config,
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
    )


@ti.func
def func_implicit_damping(
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    EPS = rigid_global_info.EPS[None]

    n_entities = entities_info.dof_start.shape[0]
    _B = dofs_state.ctrl_mode.shape[1]
    # Determine whether the mass matrix must be re-computed to take into account first-order correction terms.
    # Note that avoiding inverting the mass matrix twice would not only speed up simulation but also improving
    # numerical stability as computing post-damping accelerations from forces is not necessary anymore.
    if ti.static(
        not static_rigid_sim_config.enable_mujoco_compatibility
        or static_rigid_sim_config.integrator == gs.integrator.Euler
    ):
        for i_e, i_b in ti.ndrange(n_entities, _B):
            rigid_global_info.mass_mat_mask[i_e, i_b] = False

        ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
        for i_e, i_b in ti.ndrange(n_entities, _B):
            entity_dof_start = entities_info.dof_start[i_e]
            entity_dof_end = entities_info.dof_end[i_e]
            for i_d_ in (
                range(entity_dof_start, entity_dof_end)
                if ti.static(not static_rigid_sim_config.is_backward)
                else ti.static(range(static_rigid_sim_config.max_n_dofs_per_entity))
            ):
                i_d = (
                    i_d_ if ti.static(not static_rigid_sim_config.is_backward) else entities_info.dof_start[i_e] + i_d_
                )
                if i_d < entity_dof_end:
                    I_d = [i_d, i_b] if ti.static(static_rigid_sim_config.batch_dofs_info) else i_d
                    if dofs_info.damping[I_d] > EPS:
                        rigid_global_info.mass_mat_mask[i_e, i_b] = True
                    if ti.static(static_rigid_sim_config.integrator != gs.integrator.Euler):
                        if (
                            dofs_state.ctrl_mode[i_d, i_b] == gs.CTRL_MODE.POSITION
                            or dofs_state.ctrl_mode[i_d, i_b] == gs.CTRL_MODE.VELOCITY
                        ) and dofs_info.kv[I_d] > EPS:
                            rigid_global_info.mass_mat_mask[i_e, i_b] = True

    func_factor_mass(
        implicit_damping=True,
        entities_info=entities_info,
        dofs_state=dofs_state,
        dofs_info=dofs_info,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
    )
    func_solve_mass(
        vec=dofs_state.force,
        out=dofs_state.acc,
        out_bw=dofs_state.acc_bw,
        entities_info=entities_info,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
    )

    # Disable pre-computed factorization mask right away
    if ti.static(
        not static_rigid_sim_config.enable_mujoco_compatibility
        or static_rigid_sim_config.integrator == gs.integrator.Euler
    ):
        for i_e, i_b in ti.ndrange(n_entities, _B):
            rigid_global_info.mass_mat_mask[i_e, i_b] = True


@ti.kernel(fastcache=gs.use_fastcache)
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
    static_rigid_sim_config: ti.template(),
    contact_island_state: array_class.ContactIslandState,
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
    )

    if ti.static(static_rigid_sim_config.integrator != gs.integrator.approximate_implicitfast):
        func_implicit_damping(
            dofs_state=dofs_state,
            dofs_info=dofs_info,
            entities_info=entities_info,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
        )

    func_integrate(
        dofs_state=dofs_state,
        links_info=links_info,
        joints_info=joints_info,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
    )

    if ti.static(static_rigid_sim_config.use_hibernation):
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
        )
        func_aggregate_awake_entities(
            entities_state=entities_state,
            entities_info=entities_info,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
        )

    if ti.static(not static_rigid_sim_config.is_backward):
        func_copy_next_to_curr(
            dofs_state=dofs_state,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
        )

        if ti.static(not static_rigid_sim_config.enable_mujoco_compatibility):
            _B = links_state.pos.shape[1]
            ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
            for i_b in range(_B):
                func_update_cartesian_space(
                    i_b=i_b,
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
                )

                func_forward_velocity(
                    i_b=i_b,
                    entities_info=entities_info,
                    links_info=links_info,
                    links_state=links_state,
                    joints_info=joints_info,
                    dofs_state=dofs_state,
                    rigid_global_info=rigid_global_info,
                    static_rigid_sim_config=static_rigid_sim_config,
                )


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_forward_kinematics_links_geoms(
    envs_idx: ti.types.ndarray(),
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    joints_state: array_class.JointsState,
    joints_info: array_class.JointsInfo,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    for i_b_ in range(envs_idx.shape[0]):
        i_b = envs_idx[i_b_]

        func_update_cartesian_space(
            i_b=i_b,
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
            force_update_fixed_geoms=True,
        )
        func_forward_velocity(
            i_b=i_b,
            entities_info=entities_info,
            links_info=links_info,
            links_state=links_state,
            joints_info=joints_info,
            dofs_state=dofs_state,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
        )


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_forward_velocity(
    envs_idx: ti.types.ndarray(),
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    joints_info: array_class.JointsInfo,
    dofs_state: array_class.DofsState,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    for i_b_ in range(envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        func_forward_velocity(
            i_b=i_b,
            entities_info=entities_info,
            links_info=links_info,
            links_state=links_state,
            joints_info=joints_info,
            dofs_state=dofs_state,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
        )


@ti.func
def func_COM_links(
    i_b,
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
    for i_l_ in (
        (
            # Dynamic inner loop for forward pass
            range(rigid_global_info.n_awake_links[i_b])
            if ti.static(static_rigid_sim_config.use_hibernation)
            else range(links_info.root_idx.shape[0])
        )
        if ti.static(not static_rigid_sim_config.is_backward)
        else (
            # Static inner loop for backward pass
            ti.static(range(static_rigid_sim_config.max_n_awake_links))
            if ti.static(static_rigid_sim_config.use_hibernation)
            else ti.static(range(static_rigid_sim_config.n_links))
        )
    ):
        if func_check_index_range(
            i_l_, 0, rigid_global_info.n_awake_links[i_b], static_rigid_sim_config.use_hibernation
        ):
            i_l = (
                rigid_global_info.awake_links[i_l_, i_b] if ti.static(static_rigid_sim_config.use_hibernation) else i_l_
            )

            links_state.root_COM_bw[i_l, i_b].fill(0.0)
            links_state.mass_sum[i_l, i_b] = 0.0

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l_ in (
        (
            # Dynamic inner loop for forward pass
            range(rigid_global_info.n_awake_links[i_b])
            if ti.static(static_rigid_sim_config.use_hibernation)
            else range(links_info.root_idx.shape[0])
        )
        if ti.static(not static_rigid_sim_config.is_backward)
        else (
            # Static inner loop for backward pass
            ti.static(range(static_rigid_sim_config.max_n_awake_links))
            if ti.static(static_rigid_sim_config.use_hibernation)
            else ti.static(range(static_rigid_sim_config.n_links))
        )
    ):
        if func_check_index_range(
            i_l_, 0, rigid_global_info.n_awake_links[i_b], static_rigid_sim_config.use_hibernation
        ):
            i_l = (
                rigid_global_info.awake_links[i_l_, i_b] if ti.static(static_rigid_sim_config.use_hibernation) else i_l_
            )
            I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l

            mass = links_info.inertial_mass[I_l] + links_state.mass_shift[i_l, i_b]
            (
                links_state.i_pos_bw[i_l, i_b],
                links_state.i_quat[i_l, i_b],
            ) = gu.ti_transform_pos_quat_by_trans_quat(
                links_info.inertial_pos[I_l] + links_state.i_pos_shift[i_l, i_b],
                links_info.inertial_quat[I_l],
                links_state.pos[i_l, i_b],
                links_state.quat[i_l, i_b],
            )

            i_r = links_info.root_idx[I_l]
            links_state.mass_sum[i_r, i_b] += mass
            links_state.root_COM_bw[i_r, i_b] += mass * links_state.i_pos_bw[i_l, i_b]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l_ in (
        (
            # Dynamic inner loop for forward pass
            range(rigid_global_info.n_awake_links[i_b])
            if ti.static(static_rigid_sim_config.use_hibernation)
            else range(links_info.root_idx.shape[0])
        )
        if ti.static(not static_rigid_sim_config.is_backward)
        else (
            # Static inner loop for backward pass
            ti.static(range(static_rigid_sim_config.max_n_awake_links))
            if ti.static(static_rigid_sim_config.use_hibernation)
            else ti.static(range(static_rigid_sim_config.n_links))
        )
    ):
        if func_check_index_range(
            i_l_, 0, rigid_global_info.n_awake_links[i_b], static_rigid_sim_config.use_hibernation
        ):
            i_l = (
                rigid_global_info.awake_links[i_l_, i_b] if ti.static(static_rigid_sim_config.use_hibernation) else i_l_
            )
            I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l

            i_r = links_info.root_idx[I_l]
            if i_l == i_r:
                links_state.root_COM[i_l, i_b] = links_state.root_COM_bw[i_l, i_b] / links_state.mass_sum[i_l, i_b]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l_ in (
        (
            # Dynamic inner loop for forward pass
            range(rigid_global_info.n_awake_links[i_b])
            if ti.static(static_rigid_sim_config.use_hibernation)
            else range(links_info.root_idx.shape[0])
        )
        if ti.static(not static_rigid_sim_config.is_backward)
        else (
            # Static inner loop for backward pass
            ti.static(range(static_rigid_sim_config.max_n_awake_links))
            if ti.static(static_rigid_sim_config.use_hibernation)
            else ti.static(range(static_rigid_sim_config.n_links))
        )
    ):
        if func_check_index_range(
            i_l_, 0, rigid_global_info.n_awake_links[i_b], static_rigid_sim_config.use_hibernation
        ):
            i_l = (
                rigid_global_info.awake_links[i_l_, i_b] if ti.static(static_rigid_sim_config.use_hibernation) else i_l_
            )
            I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l

            i_r = links_info.root_idx[I_l]
            links_state.root_COM[i_l, i_b] = links_state.root_COM[i_r, i_b]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l_ in (
        (
            # Dynamic inner loop for forward pass
            range(rigid_global_info.n_awake_links[i_b])
            if ti.static(static_rigid_sim_config.use_hibernation)
            else range(links_info.root_idx.shape[0])
        )
        if ti.static(not static_rigid_sim_config.is_backward)
        else (
            # Static inner loop for backward pass
            ti.static(range(static_rigid_sim_config.max_n_awake_links))
            if ti.static(static_rigid_sim_config.use_hibernation)
            else ti.static(range(static_rigid_sim_config.n_links))
        )
    ):
        if func_check_index_range(
            i_l_, 0, rigid_global_info.n_awake_links[i_b], static_rigid_sim_config.use_hibernation
        ):
            i_l = (
                rigid_global_info.awake_links[i_l_, i_b] if ti.static(static_rigid_sim_config.use_hibernation) else i_l_
            )
            I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l

            i_r = links_info.root_idx[I_l]
            links_state.i_pos[i_l, i_b] = links_state.i_pos_bw[i_l, i_b] - links_state.root_COM[i_l, i_b]

            i_inertial = links_info.inertial_i[I_l]
            i_mass = links_info.inertial_mass[I_l] + links_state.mass_shift[i_l, i_b]
            (
                links_state.cinr_inertial[i_l, i_b],
                links_state.cinr_pos[i_l, i_b],
                links_state.cinr_quat[i_l, i_b],
                links_state.cinr_mass[i_l, i_b],
            ) = gu.ti_transform_inertia_by_trans_quat(
                i_inertial,
                i_mass,
                links_state.i_pos[i_l, i_b],
                links_state.i_quat[i_l, i_b],
                rigid_global_info.EPS[None],
            )

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l_ in (
        (
            # Dynamic inner loop for forward pass
            range(rigid_global_info.n_awake_links[i_b])
            if ti.static(static_rigid_sim_config.use_hibernation)
            else range(links_info.root_idx.shape[0])
        )
        if ti.static(not static_rigid_sim_config.is_backward)
        else (
            # Static inner loop for backward pass
            ti.static(range(static_rigid_sim_config.max_n_awake_links))
            if ti.static(static_rigid_sim_config.use_hibernation)
            else ti.static(range(static_rigid_sim_config.n_links))
        )
    ):
        if func_check_index_range(
            i_l_, 0, rigid_global_info.n_awake_links[i_b], static_rigid_sim_config.use_hibernation
        ):
            i_l = (
                rigid_global_info.awake_links[i_l_, i_b] if ti.static(static_rigid_sim_config.use_hibernation) else i_l_
            )
            I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l

            if links_info.n_dofs[I_l] > 0:
                i_p = links_info.parent_idx[I_l]

                _i_j = links_info.joint_start[I_l]
                _I_j = [_i_j, i_b] if ti.static(static_rigid_sim_config.batch_joints_info) else _i_j
                joint_type = joints_info.type[_I_j]

                p_pos = ti.Vector.zero(gs.ti_float, 3)
                p_quat = gu.ti_identity_quat()
                if i_p != -1:
                    p_pos = links_state.pos[i_p, i_b]
                    p_quat = links_state.quat[i_p, i_b]

                if joint_type == gs.JOINT_TYPE.FREE or (links_info.is_fixed[I_l] and i_p == -1):
                    links_state.j_pos[i_l, i_b] = links_state.pos[i_l, i_b]
                    links_state.j_quat[i_l, i_b] = links_state.quat[i_l, i_b]
                else:
                    (
                        links_state.j_pos_bw[i_l, 0, i_b],
                        links_state.j_quat_bw[i_l, 0, i_b],
                    ) = gu.ti_transform_pos_quat_by_trans_quat(links_info.pos[I_l], links_info.quat[I_l], p_pos, p_quat)

                    n_joints = links_info.joint_end[I_l] - links_info.joint_start[I_l]

                    for i_j_ in (
                        range(n_joints)
                        if ti.static(not static_rigid_sim_config.is_backward)
                        else ti.static(range(static_rigid_sim_config.max_n_joints_per_link))
                    ):
                        i_j = i_j_ + links_info.joint_start[I_l]

                        curr_i_j = 0 if ti.static(not static_rigid_sim_config.is_backward) else i_j_
                        next_i_j = 0 if ti.static(not static_rigid_sim_config.is_backward) else i_j_ + 1

                        if func_check_index_range(
                            i_j,
                            links_info.joint_start[I_l],
                            links_info.joint_end[I_l],
                            static_rigid_sim_config.is_backward,
                        ):
                            I_j = [i_j, i_b] if ti.static(static_rigid_sim_config.batch_joints_info) else i_j

                            (
                                links_state.j_pos_bw[i_l, next_i_j, i_b],
                                links_state.j_quat_bw[i_l, next_i_j, i_b],
                            ) = gu.ti_transform_pos_quat_by_trans_quat(
                                joints_info.pos[I_j],
                                gu.ti_identity_quat(),
                                links_state.j_pos_bw[i_l, curr_i_j, i_b],
                                links_state.j_quat_bw[i_l, curr_i_j, i_b],
                            )

                    i_j_ = 0 if ti.static(not static_rigid_sim_config.is_backward) else n_joints
                    links_state.j_pos[i_l, i_b] = links_state.j_pos_bw[i_l, i_j_, i_b]
                    links_state.j_quat[i_l, i_b] = links_state.j_quat_bw[i_l, i_j_, i_b]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l_ in (
        (
            # Dynamic inner loop for forward pass
            range(rigid_global_info.n_awake_links[i_b])
            if ti.static(static_rigid_sim_config.use_hibernation)
            else range(links_info.root_idx.shape[0])
        )
        if ti.static(not static_rigid_sim_config.is_backward)
        else (
            # Static inner loop for backward pass
            ti.static(range(static_rigid_sim_config.max_n_awake_links))
            if ti.static(static_rigid_sim_config.use_hibernation)
            else ti.static(range(static_rigid_sim_config.n_links))
        )
    ):
        if func_check_index_range(
            i_l_, 0, rigid_global_info.n_awake_links[i_b], static_rigid_sim_config.use_hibernation
        ):
            i_l = (
                rigid_global_info.awake_links[i_l_, i_b] if ti.static(static_rigid_sim_config.use_hibernation) else i_l_
            )
            I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l

            if links_info.n_dofs[I_l] > 0:
                for i_j_ in (
                    range(links_info.joint_start[I_l], links_info.joint_end[I_l])
                    if ti.static(not static_rigid_sim_config.is_backward)
                    else ti.static(range(static_rigid_sim_config.max_n_joints_per_link))
                ):
                    i_j = (
                        i_j_
                        if ti.static(not static_rigid_sim_config.is_backward)
                        else (i_j_ + links_info.joint_start[I_l])
                    )

                    if func_check_index_range(
                        i_j, links_info.joint_start[I_l], links_info.joint_end[I_l], static_rigid_sim_config.is_backward
                    ):
                        offset_pos = links_state.root_COM[i_l, i_b] - joints_state.xanchor[i_j, i_b]
                        I_j = [i_j, i_b] if ti.static(static_rigid_sim_config.batch_joints_info) else i_j
                        joint_type = joints_info.type[I_j]

                        dof_start = joints_info.dof_start[I_j]

                        EPS = rigid_global_info.EPS[None]
                        if joint_type == gs.JOINT_TYPE.REVOLUTE:
                            dofs_state.cdof_ang[dof_start, i_b] = joints_state.xaxis[i_j, i_b]
                            dofs_state.cdof_vel[dof_start, i_b] = joints_state.xaxis[i_j, i_b].cross(offset_pos)
                        elif joint_type == gs.JOINT_TYPE.PRISMATIC:
                            dofs_state.cdof_ang[dof_start, i_b] = ti.Vector.zero(gs.ti_float, 3)
                            dofs_state.cdof_vel[dof_start, i_b] = joints_state.xaxis[i_j, i_b]
                        elif joint_type == gs.JOINT_TYPE.SPHERICAL:
                            xmat_T = gu.ti_quat_to_R(links_state.quat[i_l, i_b], EPS).transpose()
                            for i in ti.static(range(3)):
                                dofs_state.cdof_ang[i + dof_start, i_b] = xmat_T[i, :]
                                dofs_state.cdof_vel[i + dof_start, i_b] = xmat_T[i, :].cross(offset_pos)
                        elif joint_type == gs.JOINT_TYPE.FREE:
                            for i in ti.static(range(3)):
                                dofs_state.cdof_ang[i + dof_start, i_b] = ti.Vector.zero(gs.ti_float, 3)
                                dofs_state.cdof_vel[i + dof_start, i_b] = ti.Vector.zero(gs.ti_float, 3)
                                dofs_state.cdof_vel[i + dof_start, i_b][i] = 1.0

                            xmat_T = gu.ti_quat_to_R(links_state.quat[i_l, i_b], EPS).transpose()
                            for i in ti.static(range(3)):
                                dofs_state.cdof_ang[i + dof_start + 3, i_b] = xmat_T[i, :]
                                dofs_state.cdof_vel[i + dof_start + 3, i_b] = xmat_T[i, :].cross(offset_pos)

                        for i_d_ in (
                            range(dof_start, joints_info.dof_end[I_j])
                            if ti.static(not static_rigid_sim_config.is_backward)
                            else ti.static(range(static_rigid_sim_config.max_n_dofs_per_joint))
                        ):
                            i_d = i_d_ if ti.static(not static_rigid_sim_config.is_backward) else (i_d_ + dof_start)
                            if func_check_index_range(
                                i_d, dof_start, joints_info.dof_end[I_j], static_rigid_sim_config.is_backward
                            ):
                                dofs_state.cdofvel_ang[i_d, i_b] = (
                                    dofs_state.cdof_ang[i_d, i_b] * dofs_state.vel[i_d, i_b]
                                )
                                dofs_state.cdofvel_vel[i_d, i_b] = (
                                    dofs_state.cdof_vel[i_d, i_b] * dofs_state.vel[i_d, i_b]
                                )


@ti.func
def func_forward_kinematics(
    i_b,
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
    for i_e_ in (
        (
            range(rigid_global_info.n_awake_entities[i_b])
            if ti.static(static_rigid_sim_config.use_hibernation)
            else range(entities_info.n_links.shape[0])
        )
        if ti.static(not static_rigid_sim_config.is_backward)
        else (
            ti.static(range(static_rigid_sim_config.max_n_awake_entities))
            if ti.static(static_rigid_sim_config.use_hibernation)
            else ti.static(range(static_rigid_sim_config.max_n_links_per_entity))
        )
    ):
        if func_check_index_range(
            i_e_, 0, rigid_global_info.n_awake_entities[i_b], static_rigid_sim_config.use_hibernation
        ):
            i_e = (
                rigid_global_info.awake_entities[i_e_, i_b]
                if ti.static(static_rigid_sim_config.use_hibernation)
                else i_e_
            )

            func_forward_kinematics_entity(
                i_e,
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
            )


@ti.func
def func_forward_velocity(
    i_b,
    entities_info: array_class.EntitiesInfo,
    links_info: array_class.LinksInfo,
    links_state: array_class.LinksState,
    joints_info: array_class.JointsInfo,
    dofs_state: array_class.DofsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    for i_e_ in (
        (
            # Dynamic inner loop for forward pass
            range(rigid_global_info.n_awake_entities[i_b])
            if ti.static(static_rigid_sim_config.use_hibernation)
            else range(entities_info.n_links.shape[0])
        )
        if ti.static(not static_rigid_sim_config.is_backward)
        else (
            # Static inner loop for backward pass
            ti.static(range(static_rigid_sim_config.max_n_awake_entities))
            if ti.static(static_rigid_sim_config.use_hibernation)
            else ti.static(range(static_rigid_sim_config.max_n_links_per_entity))
        )
    ):
        if func_check_index_range(
            i_e_, 0, rigid_global_info.n_awake_entities[i_b], static_rigid_sim_config.use_hibernation
        ):
            i_e = (
                rigid_global_info.awake_entities[i_e_, i_b]
                if ti.static(static_rigid_sim_config.use_hibernation)
                else i_e_
            )
            func_forward_velocity_entity(
                i_e=i_e,
                i_b=i_b,
                entities_info=entities_info,
                links_info=links_info,
                links_state=links_state,
                joints_info=joints_info,
                dofs_state=dofs_state,
                rigid_global_info=rigid_global_info,
                static_rigid_sim_config=static_rigid_sim_config,
            )


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_forward_kinematics_entity(
    i_e: ti.int32,
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
    for i_b_ in range(envs_idx.shape[0]):
        i_b = envs_idx[i_b_]

        func_forward_kinematics_entity(
            i_e,
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
        )


@ti.func
def func_forward_kinematics_entity(
    i_e,
    i_b,
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
    # Becomes static loop in backward pass, because we assume this loop is an inner loop
    for i_l_ in (
        range(entities_info.link_start[i_e], entities_info.link_end[i_e])
        if ti.static(not static_rigid_sim_config.is_backward)
        else ti.static(range(static_rigid_sim_config.max_n_links_per_entity))
    ):
        EPS = rigid_global_info.EPS[None]
        i_l = i_l_ if ti.static(not static_rigid_sim_config.is_backward) else (i_l_ + entities_info.link_start[i_e])

        if func_check_index_range(
            i_l, entities_info.link_start[i_e], entities_info.link_end[i_e], static_rigid_sim_config.is_backward
        ):
            I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l

            links_state.pos_bw[i_l, 0, i_b] = links_info.pos[I_l]
            links_state.quat_bw[i_l, 0, i_b] = links_info.quat[I_l]
            if links_info.parent_idx[I_l] != -1:
                parent_pos = links_state.pos[links_info.parent_idx[I_l], i_b]
                parent_quat = links_state.quat[links_info.parent_idx[I_l], i_b]
                links_state.pos_bw[i_l, 0, i_b] = parent_pos + gu.ti_transform_by_quat(links_info.pos[I_l], parent_quat)
                links_state.quat_bw[i_l, 0, i_b] = gu.ti_transform_quat_by_quat(links_info.quat[I_l], parent_quat)

            n_joints = links_info.joint_end[I_l] - links_info.joint_start[I_l]

            for i_j_ in (
                range(n_joints)
                if ti.static(not static_rigid_sim_config.is_backward)
                else ti.static(range(static_rigid_sim_config.max_n_joints_per_link))
            ):
                i_j = i_j_ + links_info.joint_start[I_l]

                curr_i_j = 0 if ti.static(not static_rigid_sim_config.is_backward) else i_j_
                next_i_j = 0 if ti.static(not static_rigid_sim_config.is_backward) else i_j_ + 1

                if func_check_index_range(
                    i_j, links_info.joint_start[I_l], links_info.joint_end[I_l], static_rigid_sim_config.is_backward
                ):
                    I_j = [i_j, i_b] if ti.static(static_rigid_sim_config.batch_joints_info) else i_j
                    joint_type = joints_info.type[I_j]
                    q_start = joints_info.q_start[I_j]
                    dof_start = joints_info.dof_start[I_j]
                    I_d = [dof_start, i_b] if ti.static(static_rigid_sim_config.batch_dofs_info) else dof_start

                    # compute axis and anchor
                    if joint_type == gs.JOINT_TYPE.FREE:
                        joints_state.xanchor[i_j, i_b] = ti.Vector(
                            [
                                rigid_global_info.qpos[q_start, i_b],
                                rigid_global_info.qpos[q_start + 1, i_b],
                                rigid_global_info.qpos[q_start + 2, i_b],
                            ]
                        )
                        joints_state.xaxis[i_j, i_b] = ti.Vector([0.0, 0.0, 1.0])
                    elif joint_type == gs.JOINT_TYPE.FIXED:
                        pass
                    else:
                        axis = ti.Vector([0.0, 0.0, 1.0], dt=gs.ti_float)
                        if joint_type == gs.JOINT_TYPE.REVOLUTE:
                            axis = dofs_info.motion_ang[I_d]
                        elif joint_type == gs.JOINT_TYPE.PRISMATIC:
                            axis = dofs_info.motion_vel[I_d]

                        joints_state.xanchor[i_j, i_b] = (
                            gu.ti_transform_by_quat(joints_info.pos[I_j], links_state.quat_bw[i_l, curr_i_j, i_b])
                            + links_state.pos_bw[i_l, curr_i_j, i_b]
                        )
                        joints_state.xaxis[i_j, i_b] = gu.ti_transform_by_quat(
                            axis, links_state.quat_bw[i_l, curr_i_j, i_b]
                        )

                    if joint_type == gs.JOINT_TYPE.FREE:
                        links_state.pos_bw[i_l, next_i_j, i_b] = ti.Vector(
                            [
                                rigid_global_info.qpos[q_start, i_b],
                                rigid_global_info.qpos[q_start + 1, i_b],
                                rigid_global_info.qpos[q_start + 2, i_b],
                            ],
                            dt=gs.ti_float,
                        )
                        links_state.quat_bw[i_l, next_i_j, i_b] = ti.Vector(
                            [
                                rigid_global_info.qpos[q_start + 3, i_b],
                                rigid_global_info.qpos[q_start + 4, i_b],
                                rigid_global_info.qpos[q_start + 5, i_b],
                                rigid_global_info.qpos[q_start + 6, i_b],
                            ],
                            dt=gs.ti_float,
                        )
                        xyz = gu.ti_quat_to_xyz(links_state.quat_bw[i_l, next_i_j, i_b], EPS)
                        for j in ti.static(range(3)):
                            dofs_state.pos[dof_start + j, i_b] = links_state.pos_bw[i_l, next_i_j, i_b][j]
                            dofs_state.pos[dof_start + 3 + j, i_b] = xyz[j]
                    elif joint_type == gs.JOINT_TYPE.FIXED:
                        pass
                    elif joint_type == gs.JOINT_TYPE.SPHERICAL:
                        qloc = ti.Vector(
                            [
                                rigid_global_info.qpos[q_start, i_b],
                                rigid_global_info.qpos[q_start + 1, i_b],
                                rigid_global_info.qpos[q_start + 2, i_b],
                                rigid_global_info.qpos[q_start + 3, i_b],
                            ],
                            dt=gs.ti_float,
                        )
                        xyz = gu.ti_quat_to_xyz(qloc, EPS)
                        for j in ti.static(range(3)):
                            dofs_state.pos[dof_start + j, i_b] = xyz[j]
                        links_state.quat_bw[i_l, next_i_j, i_b] = gu.ti_transform_quat_by_quat(
                            qloc, links_state.quat_bw[i_l, curr_i_j, i_b]
                        )
                        links_state.pos_bw[i_l, next_i_j, i_b] = joints_state.xanchor[
                            i_j, i_b
                        ] - gu.ti_transform_by_quat(joints_info.pos[I_j], links_state.quat_bw[i_l, next_i_j, i_b])
                    elif joint_type == gs.JOINT_TYPE.REVOLUTE:
                        axis = dofs_info.motion_ang[I_d]
                        dofs_state.pos[dof_start, i_b] = (
                            rigid_global_info.qpos[q_start, i_b] - rigid_global_info.qpos0[q_start, i_b]
                        )
                        qloc = gu.ti_rotvec_to_quat(axis * dofs_state.pos[dof_start, i_b], EPS)
                        links_state.quat_bw[i_l, next_i_j, i_b] = gu.ti_transform_quat_by_quat(
                            qloc, links_state.quat_bw[i_l, curr_i_j, i_b]
                        )
                        links_state.pos_bw[i_l, next_i_j, i_b] = joints_state.xanchor[
                            i_j, i_b
                        ] - gu.ti_transform_by_quat(joints_info.pos[I_j], links_state.quat_bw[i_l, next_i_j, i_b])
                    else:  # joint_type == gs.JOINT_TYPE.PRISMATIC:
                        dofs_state.pos[dof_start, i_b] = (
                            rigid_global_info.qpos[q_start, i_b] - rigid_global_info.qpos0[q_start, i_b]
                        )
                        links_state.pos_bw[i_l, next_i_j, i_b] = (
                            links_state.pos_bw[i_l, curr_i_j, i_b]
                            + joints_state.xaxis[i_j, i_b] * dofs_state.pos[dof_start, i_b]
                        )

            # Skip link pose update for fixed root links to let users manually overwrite them
            i_j_ = 0 if ti.static(not static_rigid_sim_config.is_backward) else n_joints
            if not (links_info.parent_idx[I_l] == -1 and links_info.is_fixed[I_l]):
                links_state.pos[i_l, i_b] = links_state.pos_bw[i_l, i_j_, i_b]
                links_state.quat[i_l, i_b] = links_state.quat_bw[i_l, i_j_, i_b]


@ti.func
def func_forward_velocity_entity(
    i_e,
    i_b,
    entities_info: array_class.EntitiesInfo,
    links_info: array_class.LinksInfo,
    links_state: array_class.LinksState,
    joints_info: array_class.JointsInfo,
    dofs_state: array_class.DofsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    for i_l_ in (
        range(entities_info.link_start[i_e], entities_info.link_end[i_e])
        if ti.static(not static_rigid_sim_config.is_backward)
        else ti.static(range(static_rigid_sim_config.max_n_links_per_entity))
    ):
        i_l = i_l_ if ti.static(not static_rigid_sim_config.is_backward) else (i_l_ + entities_info.link_start[i_e])

        if func_check_index_range(
            i_l, entities_info.link_start[i_e], entities_info.link_end[i_e], static_rigid_sim_config.is_backward
        ):
            I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l
            n_joints = links_info.joint_end[I_l] - links_info.joint_start[I_l]

            links_state.cd_vel_bw[i_l, 0, i_b] = ti.Vector.zero(gs.ti_float, 3)
            links_state.cd_ang_bw[i_l, 0, i_b] = ti.Vector.zero(gs.ti_float, 3)

            if links_info.parent_idx[I_l] != -1:
                links_state.cd_vel_bw[i_l, 0, i_b] = links_state.cd_vel[links_info.parent_idx[I_l], i_b]
                links_state.cd_ang_bw[i_l, 0, i_b] = links_state.cd_ang[links_info.parent_idx[I_l], i_b]

            for i_j_ in (
                range(n_joints)
                if ti.static(not static_rigid_sim_config.is_backward)
                else ti.static(range(static_rigid_sim_config.max_n_joints_per_link))
            ):
                i_j = i_j_ + links_info.joint_start[I_l]

                if func_check_index_range(
                    i_j, links_info.joint_start[I_l], links_info.joint_end[I_l], static_rigid_sim_config.is_backward
                ):
                    I_j = [i_j, i_b] if ti.static(static_rigid_sim_config.batch_joints_info) else i_j
                    joint_type = joints_info.type[I_j]
                    q_start = joints_info.q_start[I_j]
                    dof_start = joints_info.dof_start[I_j]

                    curr_i_j = 0 if ti.static(not static_rigid_sim_config.is_backward) else i_j_
                    next_i_j = 0 if ti.static(not static_rigid_sim_config.is_backward) else i_j_ + 1

                    if joint_type == gs.JOINT_TYPE.FREE:
                        for i_3 in ti.static(range(3)):
                            func_atomic_add_if_backward_3d(
                                links_state.cd_vel_bw,
                                i_l,
                                curr_i_j,
                                i_b,
                                dofs_state.cdof_vel[dof_start + i_3, i_b] * dofs_state.vel[dof_start + i_3, i_b],
                                static_rigid_sim_config,
                            )
                            func_atomic_add_if_backward_3d(
                                links_state.cd_ang_bw,
                                i_l,
                                curr_i_j,
                                i_b,
                                dofs_state.cdof_ang[dof_start + i_3, i_b] * dofs_state.vel[dof_start + i_3, i_b],
                                static_rigid_sim_config,
                            )

                        for i_3 in ti.static(range(3)):
                            (
                                dofs_state.cdofd_ang[dof_start + i_3, i_b],
                                dofs_state.cdofd_vel[dof_start + i_3, i_b],
                            ) = ti.Vector.zero(gs.ti_float, 3), ti.Vector.zero(gs.ti_float, 3)

                            (
                                dofs_state.cdofd_ang[dof_start + i_3 + 3, i_b],
                                dofs_state.cdofd_vel[dof_start + i_3 + 3, i_b],
                            ) = gu.motion_cross_motion(
                                links_state.cd_ang_bw[i_l, curr_i_j, i_b],
                                links_state.cd_vel_bw[i_l, curr_i_j, i_b],
                                dofs_state.cdof_ang[dof_start + i_3 + 3, i_b],
                                dofs_state.cdof_vel[dof_start + i_3 + 3, i_b],
                            )

                        links_state.cd_vel_bw[i_l, next_i_j, i_b] = links_state.cd_vel_bw[i_l, curr_i_j, i_b]
                        links_state.cd_ang_bw[i_l, next_i_j, i_b] = links_state.cd_ang_bw[i_l, curr_i_j, i_b]

                        for i_3 in ti.static(range(3)):
                            func_atomic_add_if_backward_3d(
                                links_state.cd_vel_bw,
                                i_l,
                                next_i_j,
                                i_b,
                                dofs_state.cdof_vel[dof_start + i_3 + 3, i_b]
                                * dofs_state.vel[dof_start + i_3 + 3, i_b],
                                static_rigid_sim_config,
                            )
                            func_atomic_add_if_backward_3d(
                                links_state.cd_ang_bw,
                                i_l,
                                next_i_j,
                                i_b,
                                dofs_state.cdof_ang[dof_start + i_3 + 3, i_b]
                                * dofs_state.vel[dof_start + i_3 + 3, i_b],
                                static_rigid_sim_config,
                            )

                    else:
                        for i_d_ in (
                            range(dof_start, joints_info.dof_end[I_j])
                            if ti.static(not static_rigid_sim_config.is_backward)
                            else ti.static(range(static_rigid_sim_config.max_n_dofs_per_joint))
                        ):
                            i_d = i_d_ if ti.static(not static_rigid_sim_config.is_backward) else (i_d_ + dof_start)
                            if func_check_index_range(
                                i_d, dof_start, joints_info.dof_end[I_j], static_rigid_sim_config.is_backward
                            ):
                                dofs_state.cdofd_ang[i_d, i_b], dofs_state.cdofd_vel[i_d, i_b] = gu.motion_cross_motion(
                                    links_state.cd_ang_bw[i_l, curr_i_j, i_b],
                                    links_state.cd_vel_bw[i_l, curr_i_j, i_b],
                                    dofs_state.cdof_ang[i_d, i_b],
                                    dofs_state.cdof_vel[i_d, i_b],
                                )

                        links_state.cd_vel_bw[i_l, next_i_j, i_b] = links_state.cd_vel_bw[i_l, curr_i_j, i_b]
                        links_state.cd_ang_bw[i_l, next_i_j, i_b] = links_state.cd_ang_bw[i_l, curr_i_j, i_b]

                        for i_d_ in (
                            range(dof_start, joints_info.dof_end[I_j])
                            if ti.static(not static_rigid_sim_config.is_backward)
                            else ti.static(range(static_rigid_sim_config.max_n_dofs_per_joint))
                        ):
                            i_d = i_d_ if ti.static(not static_rigid_sim_config.is_backward) else (i_d_ + dof_start)
                            if func_check_index_range(
                                i_d, dof_start, joints_info.dof_end[I_j], static_rigid_sim_config.is_backward
                            ):
                                func_atomic_add_if_backward_3d(
                                    links_state.cd_vel_bw,
                                    i_l,
                                    next_i_j,
                                    i_b,
                                    dofs_state.cdof_vel[i_d, i_b] * dofs_state.vel[i_d, i_b],
                                    static_rigid_sim_config,
                                )
                                func_atomic_add_if_backward_3d(
                                    links_state.cd_ang_bw,
                                    i_l,
                                    next_i_j,
                                    i_b,
                                    dofs_state.cdof_ang[i_d, i_b] * dofs_state.vel[i_d, i_b],
                                    static_rigid_sim_config,
                                )

            i_j_ = 0 if ti.static(not static_rigid_sim_config.is_backward) else n_joints
            links_state.cd_vel[i_l, i_b] = links_state.cd_vel_bw[i_l, i_j_, i_b]
            links_state.cd_ang[i_l, i_b] = links_state.cd_ang_bw[i_l, i_j_, i_b]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_update_geoms(
    envs_idx: ti.types.ndarray(),
    entities_info: array_class.EntitiesInfo,
    geoms_info: array_class.GeomsInfo,
    geoms_state: array_class.GeomsState,
    links_state: array_class.LinksState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    force_update_fixed_geoms: ti.template(),
):
    for i_b_ in range(envs_idx.shape[0]):
        i_b = envs_idx[i_b_]

        func_update_geoms(
            i_b,
            entities_info,
            geoms_info,
            geoms_state,
            links_state,
            rigid_global_info,
            static_rigid_sim_config,
            force_update_fixed_geoms,
        )


@ti.func
def func_update_geoms(
    i_b,
    entities_info: array_class.EntitiesInfo,
    geoms_info: array_class.GeomsInfo,
    geoms_state: array_class.GeomsState,
    links_state: array_class.LinksState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    force_update_fixed_geoms: ti.template(),
):
    """
    NOTE: this only update geom pose, not its verts and else.
    """
    for i_0 in (
        (
            # Dynamic inner loop for forward pass
            range(rigid_global_info.n_awake_entities[i_b])
            if ti.static(static_rigid_sim_config.use_hibernation)
            else range(geoms_info.pos.shape[0])
        )
        if ti.static(not static_rigid_sim_config.is_backward)
        else (
            # Static inner loop for backward pass
            ti.static(range(static_rigid_sim_config.max_n_awake_entities))
            if ti.static(static_rigid_sim_config.use_hibernation)
            else ti.static(range(static_rigid_sim_config.n_geoms))
        )
    ):
        i_e = rigid_global_info.awake_entities[i_0, i_b] if ti.static(static_rigid_sim_config.use_hibernation) else 0
        n_geoms = entities_info.geom_end[i_e] - entities_info.geom_start[i_e]

        for i_1 in (
            (
                # Dynamic inner loop for forward pass
                range(n_geoms)
                if ti.static(static_rigid_sim_config.use_hibernation)
                else range(1)
            )
            if ti.static(not static_rigid_sim_config.is_backward)
            else (
                # Static inner loop for backward pass
                ti.static(range(static_rigid_sim_config.max_n_geoms_per_entity))
                if ti.static(static_rigid_sim_config.use_hibernation)
                else ti.static(range(1))
            )
        ):
            i_g = i_1 + entities_info.geom_start[i_e] if ti.static(static_rigid_sim_config.use_hibernation) else i_0
            if func_check_index_range(
                i_g, entities_info.geom_start[i_e], entities_info.geom_end[i_e], static_rigid_sim_config.is_backward
            ):
                if force_update_fixed_geoms or not geoms_info.is_fixed[i_g]:
                    (
                        geoms_state.pos[i_g, i_b],
                        geoms_state.quat[i_g, i_b],
                    ) = gu.ti_transform_pos_quat_by_trans_quat(
                        geoms_info.pos[i_g],
                        geoms_info.quat[i_g],
                        links_state.pos[geoms_info.link_idx[i_g], i_b],
                        links_state.quat[geoms_info.link_idx[i_g], i_b],
                    )
                    geoms_state.verts_updated[i_g, i_b] = False


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_update_verts_for_geoms(
    geoms_idx: ti.types.ndarray(),
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    free_verts_state: array_class.VertsState,
    fixed_verts_state: array_class.VertsState,
):
    n_geoms = geoms_idx.shape[0]
    _B = geoms_state.verts_updated.shape[1]
    for i_g_, i_b in ti.ndrange(n_geoms, _B):
        i_g = geoms_idx[i_g_]
        func_update_verts_for_geom(i_g, i_b, geoms_state, geoms_info, verts_info, free_verts_state, fixed_verts_state)


@ti.func
def func_update_verts_for_geom(
    i_g: ti.i32,
    i_b: ti.i32,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    free_verts_state: array_class.VertsState,
    fixed_verts_state: array_class.VertsState,
):
    if not geoms_state.verts_updated[i_g, i_b]:
        i_v_start = geoms_info.vert_start[i_g]
        if verts_info.is_fixed[i_v_start]:
            for i_v in range(i_v_start, geoms_info.vert_end[i_g]):
                verts_state_idx = verts_info.verts_state_idx[i_v]
                fixed_verts_state.pos[verts_state_idx] = gu.ti_transform_by_trans_quat(
                    verts_info.init_pos[i_v], geoms_state.pos[i_g, i_b], geoms_state.quat[i_g, i_b]
                )
            _B = geoms_state.verts_updated.shape[1]
            for j_b in range(_B):
                geoms_state.verts_updated[i_g, j_b] = True
        else:
            for i_v in range(i_v_start, geoms_info.vert_end[i_g]):
                verts_state_idx = verts_info.verts_state_idx[i_v]
                free_verts_state.pos[verts_state_idx, i_b] = gu.ti_transform_by_trans_quat(
                    verts_info.init_pos[i_v], geoms_state.pos[i_g, i_b], geoms_state.quat[i_g, i_b]
                )
            geoms_state.verts_updated[i_g, i_b] = True


@ti.func
def func_update_all_verts(
    geoms_info: array_class.GeomsInfo,
    geoms_state: array_class.GeomsState,
    verts_info: array_class.VertsInfo,
    free_verts_state: array_class.VertsState,
    fixed_verts_state: array_class.VertsState,
):
    n_geoms, _B = geoms_state.pos.shape
    for i_g, i_b in ti.ndrange(n_geoms, _B):
        func_update_verts_for_geom(i_g, i_b, geoms_state, geoms_info, verts_info, free_verts_state, fixed_verts_state)


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_update_all_verts(
    geoms_info: array_class.GeomsInfo,
    geoms_state: array_class.GeomsState,
    verts_info: array_class.VertsInfo,
    free_verts_state: array_class.VertsState,
    fixed_verts_state: array_class.VertsState,
):
    func_update_all_verts(geoms_info, geoms_state, verts_info, free_verts_state, fixed_verts_state)


@ti.kernel
def kernel_update_geom_aabbs(
    geoms_state: array_class.GeomsState,
    geoms_init_AABB: array_class.GeomsInitAABB,
    static_rigid_sim_config: ti.template(),
):
    n_geoms = geoms_state.pos.shape[0]
    _B = geoms_state.pos.shape[1]
    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_g, i_b in ti.ndrange(n_geoms, _B):
        g_pos = geoms_state.pos[i_g, i_b]
        g_quat = geoms_state.quat[i_g, i_b]

        lower = gu.ti_vec3(ti.math.inf)
        upper = gu.ti_vec3(-ti.math.inf)
        for i_corner in ti.static(range(8)):
            corner_pos = gu.ti_transform_by_trans_quat(geoms_init_AABB[i_g, i_corner], g_pos, g_quat)
            lower = ti.min(lower, corner_pos)
            upper = ti.max(upper, corner_pos)

        geoms_state.aabb_min[i_g, i_b] = lower
        geoms_state.aabb_max[i_g, i_b] = upper


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_update_vgeoms(
    vgeoms_info: array_class.VGeomsInfo,
    vgeoms_state: array_class.VGeomsState,
    links_state: array_class.LinksState,
    static_rigid_sim_config: ti.template(),
):
    """
    Vgeoms are only for visualization purposes.
    """
    n_vgeoms = vgeoms_info.link_idx.shape[0]
    _B = links_state.pos.shape[1]
    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_g, i_b in ti.ndrange(n_vgeoms, _B):
        vgeoms_state.pos[i_g, i_b], vgeoms_state.quat[i_g, i_b] = gu.ti_transform_pos_quat_by_trans_quat(
            vgeoms_info.pos[i_g],
            vgeoms_info.quat[i_g],
            links_state.pos[vgeoms_info.link_idx[i_g], i_b],
            links_state.quat[vgeoms_info.link_idx[i_g], i_b],
        )


@ti.func
def func_hibernate__for_all_awake_islands_either_hiberanate_or_update_aabb_sort_buffer(
    dofs_state: array_class.DofsState,
    entities_state: array_class.EntitiesState,
    entities_info: array_class.EntitiesInfo,
    links_state: array_class.LinksState,
    geoms_state: array_class.GeomsState,
    collider_state: array_class.ColliderState,
    unused__rigid_global_info: array_class.RigidGlobalInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    contact_island_state: array_class.ContactIslandState,
):
    n_entities = entities_state.hibernated.shape[0]
    _B = entities_state.hibernated.shape[1]

    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_b in range(_B):
        for island_idx in range(contact_island_state.n_islands[i_b]):
            was_island_hibernated = contact_island_state.island_hibernated[island_idx, i_b]

            if not was_island_hibernated:
                are_all_entities_okay_for_hibernation = True
                entity_ref_range = contact_island_state.island_entity[island_idx, i_b]
                for i_entity_ref_offset_ in range(entity_ref_range.n):
                    entity_ref = entity_ref_range.start + i_entity_ref_offset_
                    entity_idx = contact_island_state.entity_id[entity_ref, i_b]

                    # Hibernated entities already have zero dofs_state.acc/vel
                    is_entity_hibernated = entities_state.hibernated[entity_idx, i_b]
                    if is_entity_hibernated:
                        continue

                    for i_d in range(entities_info.dof_start[entity_idx], entities_info.dof_end[entity_idx]):
                        max_acc = rigid_global_info.hibernation_thresh_acc[None]
                        max_vel = rigid_global_info.hibernation_thresh_vel[None]
                        if ti.abs(dofs_state.acc[i_d, i_b]) > max_acc or ti.abs(dofs_state.vel[i_d, i_b]) > max_vel:
                            are_all_entities_okay_for_hibernation = False
                            break

                    if not are_all_entities_okay_for_hibernation:
                        break

                if not are_all_entities_okay_for_hibernation:
                    # update collider sort_buffer with aabb extents along x-axis
                    for i_entity_ref_offset_ in range(entity_ref_range.n):
                        entity_ref = entity_ref_range.start + i_entity_ref_offset_
                        entity_idx = contact_island_state.entity_id[entity_ref, i_b]
                        for i_g in range(entities_info.geom_start[entity_idx], entities_info.geom_end[entity_idx]):
                            min_idx, min_val = geoms_state.min_buffer_idx[i_g, i_b], geoms_state.aabb_min[i_g, i_b][0]
                            max_idx, max_val = geoms_state.max_buffer_idx[i_g, i_b], geoms_state.aabb_max[i_g, i_b][0]
                            collider_state.sort_buffer.value[min_idx, i_b] = min_val
                            collider_state.sort_buffer.value[max_idx, i_b] = max_val
                else:
                    # perform hibernation
                    prev_entity_ref = entity_ref_range.start + entity_ref_range.n - 1
                    prev_entity_idx = contact_island_state.entity_id[prev_entity_ref, i_b]

                    for i_entity_ref_offset_ in range(entity_ref_range.n):
                        entity_ref = entity_ref_range.start + i_entity_ref_offset_
                        entity_idx = contact_island_state.entity_id[entity_ref, i_b]

                        func_hibernate_entity_and_zero_dof_velocities(
                            entity_idx,
                            i_b,
                            entities_state=entities_state,
                            entities_info=entities_info,
                            dofs_state=dofs_state,
                            links_state=links_state,
                            geoms_state=geoms_state,
                        )

                        # store entities in the hibernated islands by daisy chaining them
                        contact_island_state.entity_idx_to_next_entity_idx_in_hibernated_island[
                            prev_entity_idx, i_b
                        ] = entity_idx
                        prev_entity_idx = entity_idx


@ti.func
def func_aggregate_awake_entities(
    entities_state: array_class.EntitiesState,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):

    n_entities = entities_state.hibernated.shape[0]
    _B = entities_state.hibernated.shape[1]
    rigid_global_info.n_awake_entities.fill(0)
    rigid_global_info.n_awake_links.fill(0)
    rigid_global_info.n_awake_dofs.fill(0)
    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_e, i_b in ti.ndrange(n_entities, _B):
        if entities_state.hibernated[i_e, i_b] or entities_info.n_dofs[i_e] == 0:
            continue

        next_awake_entity_idx = ti.atomic_add(rigid_global_info.n_awake_entities[i_b], 1)
        rigid_global_info.awake_entities[next_awake_entity_idx, i_b] = i_e

        n_dofs = entities_info.n_dofs[i_e]
        entity_dofs_base_idx: ti.int32 = entities_info.dof_start[i_e]
        awake_dofs_base_idx = ti.atomic_add(rigid_global_info.n_awake_dofs[i_b], n_dofs)
        for i_d_ in range(n_dofs):
            rigid_global_info.awake_dofs[awake_dofs_base_idx + i_d_, i_b] = entity_dofs_base_idx + i_d_

        n_links = entities_info.n_links[i_e]
        entity_links_base_idx: ti.int32 = entities_info.link_start[i_e]
        awake_links_base_idx = ti.atomic_add(rigid_global_info.n_awake_links[i_b], n_links)
        for i_l_ in range(n_links):
            rigid_global_info.awake_links[awake_links_base_idx + i_l_, i_b] = entity_links_base_idx + i_l_


@ti.func
def func_hibernate_entity_and_zero_dof_velocities(
    i_e: int,
    i_b: int,
    entities_state: array_class.EntitiesState,
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    links_state: array_class.LinksState,
    geoms_state: array_class.GeomsState,
):
    """
    Mark RigidEnity, individual DOFs in DofsState, RigidLinks, and RigidGeoms as hibernated.

    Also, zero out DOF velocitities and accelerations.
    """
    entities_state.hibernated[i_e, i_b] = True

    for i_d in range(entities_info.dof_start[i_e], entities_info.dof_end[i_e]):
        dofs_state.hibernated[i_d, i_b] = True
        dofs_state.vel[i_d, i_b] = 0.0
        dofs_state.acc[i_d, i_b] = 0.0

    for i_l in range(entities_info.link_start[i_e], entities_info.link_end[i_e]):
        links_state.hibernated[i_l, i_b] = True

    for i_g in range(entities_info.geom_start[i_e], entities_info.geom_end[i_e]):
        geoms_state.hibernated[i_g, i_b] = True


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_apply_links_external_force(
    force: ti.types.ndarray(),
    links_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    ref: ti.template(),
    local: ti.template(),
    links_state: array_class.LinksState,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_l_, i_b_ in ti.ndrange(links_idx.shape[0], envs_idx.shape[0]):
        force_i = ti.Vector([force[i_b_, i_l_, 0], force[i_b_, i_l_, 1], force[i_b_, i_l_, 2]], dt=gs.ti_float)
        func_apply_link_external_force(force_i, links_idx[i_l_], envs_idx[i_b_], ref, local, links_state)


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_apply_links_external_torque(
    torque: ti.types.ndarray(),
    links_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    ref: ti.template(),
    local: ti.template(),
    links_state: array_class.LinksState,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_l_, i_b_ in ti.ndrange(links_idx.shape[0], envs_idx.shape[0]):
        torque_i = ti.Vector([torque[i_b_, i_l_, 0], torque[i_b_, i_l_, 1], torque[i_b_, i_l_, 2]], dt=gs.ti_float)
        func_apply_link_external_torque(torque_i, links_idx[i_l_], envs_idx[i_b_], ref, local, links_state)


@ti.func
def func_apply_coupling_force(pos, force, link_idx, env_idx, links_state: array_class.LinksState):
    torque = (pos - links_state.root_COM[link_idx, env_idx]).cross(force)
    links_state.cfrc_coupling_ang[link_idx, env_idx] -= torque
    links_state.cfrc_coupling_vel[link_idx, env_idx] -= force


@ti.func
def func_apply_link_external_force(
    force,
    link_idx,
    env_idx,
    ref: ti.template(),
    local: ti.template(),
    links_state: array_class.LinksState,
):
    torque = ti.Vector.zero(gs.ti_float, 3)
    if ti.static(ref == 1):  # link's CoM
        if ti.static(local == 1):
            force = gu.ti_transform_by_quat(force, links_state.i_quat[link_idx, env_idx])
        torque = links_state.i_pos[link_idx, env_idx].cross(force)
    if ti.static(ref == 2):  # link's origin
        if ti.static(local == 1):
            force = gu.ti_transform_by_quat(force, links_state.i_quat[link_idx, env_idx])
        torque = (links_state.pos[link_idx, env_idx] - links_state.root_COM[link_idx, env_idx]).cross(force)

    links_state.cfrc_applied_vel[link_idx, env_idx] -= force
    links_state.cfrc_applied_ang[link_idx, env_idx] -= torque


@ti.func
def func_apply_external_torque(self, torque, link_idx, env_idx):
    self.links_state.cfrc_applied_ang[link_idx, env_idx] -= torque


@ti.func
def func_apply_link_external_torque(
    torque,
    link_idx,
    env_idx,
    ref: ti.template(),
    local: ti.template(),
    links_state: array_class.LinksState,
):
    if ti.static(ref == 1 and local == 1):  # link's CoM
        torque = gu.ti_transform_by_quat(torque, links_state.i_quat[link_idx, env_idx])
    if ti.static(ref == 2 and local == 1):  # link's origin
        torque = gu.ti_transform_by_quat(torque, links_state.quat[link_idx, env_idx])

    links_state.cfrc_applied_ang[link_idx, env_idx] -= torque


@ti.func
def func_clear_external_force(
    links_state: array_class.LinksState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    _B = links_state.pos.shape[1]
    n_links = links_state.pos.shape[0]

    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_0, i_b in (
        ti.ndrange(1, _B) if ti.static(static_rigid_sim_config.use_hibernation) else ti.ndrange(n_links, _B)
    ):
        for i_1 in (
            range(rigid_global_info.n_awake_links[i_b])
            if ti.static(static_rigid_sim_config.use_hibernation)
            else range(1)
        ):
            i_l = rigid_global_info.awake_links[i_1, i_b] if ti.static(static_rigid_sim_config.use_hibernation) else i_0
            links_state.cfrc_applied_ang[i_l, i_b] = ti.Vector.zero(gs.ti_float, 3)
            links_state.cfrc_applied_vel[i_l, i_b] = ti.Vector.zero(gs.ti_float, 3)


@ti.func
def func_torque_and_passive_force(
    entities_state: array_class.EntitiesState,
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    joints_info: array_class.JointsInfo,
    geoms_state: array_class.GeomsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    contact_island_state: array_class.ContactIslandState,
):
    # compute force based on each dof's ctrl mode
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_e, i_b in ti.ndrange(entities_info.n_links.shape[0], dofs_state.ctrl_mode.shape[1]):
        wakeup = False
        EPS = rigid_global_info.EPS[None]

        for i_l_ in (
            range(entities_info.link_start[i_e], entities_info.link_end[i_e])
            if ti.static(not static_rigid_sim_config.is_backward)
            else ti.static(range(static_rigid_sim_config.max_n_links_per_entity))
        ):
            i_l = i_l_ if ti.static(not static_rigid_sim_config.is_backward) else (i_l_ + entities_info.link_start[i_e])

            if func_check_index_range(
                i_l, entities_info.link_start[i_e], entities_info.link_end[i_e], static_rigid_sim_config.is_backward
            ):
                I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l
                if links_info.n_dofs[I_l] > 0:
                    i_j = links_info.joint_start[I_l]
                    I_j = [i_j, i_b] if ti.static(static_rigid_sim_config.batch_joints_info) else i_j
                    joint_type = joints_info.type[I_j]

                    for i_d_ in (
                        range(links_info.dof_start[I_l], links_info.dof_end[I_l])
                        if ti.static(not static_rigid_sim_config.is_backward)
                        else ti.static(range(static_rigid_sim_config.max_n_dofs_per_link))
                    ):
                        i_d = (
                            i_d_
                            if ti.static(not static_rigid_sim_config.is_backward)
                            else (i_d_ + links_info.dof_start[I_l])
                        )

                        if func_check_index_range(
                            i_d, links_info.dof_start[I_l], links_info.dof_end[I_l], static_rigid_sim_config.is_backward
                        ):
                            I_d = [i_d, i_b] if ti.static(static_rigid_sim_config.batch_dofs_info) else i_d
                            force = gs.ti_float(0.0)
                            if dofs_state.ctrl_mode[i_d, i_b] == gs.CTRL_MODE.FORCE:
                                force = dofs_state.ctrl_force[i_d, i_b]
                            elif dofs_state.ctrl_mode[i_d, i_b] == gs.CTRL_MODE.VELOCITY:
                                force = dofs_info.kv[I_d] * (dofs_state.ctrl_vel[i_d, i_b] - dofs_state.vel[i_d, i_b])
                            elif dofs_state.ctrl_mode[i_d, i_b] == gs.CTRL_MODE.POSITION and not (
                                joint_type == gs.JOINT_TYPE.FREE and i_d >= links_info.dof_start[I_l] + 3
                            ):
                                force = dofs_info.kp[I_d] * (
                                    dofs_state.ctrl_pos[i_d, i_b] - dofs_state.pos[i_d, i_b]
                                ) + dofs_info.kv[I_d] * (dofs_state.ctrl_vel[i_d, i_b] - dofs_state.vel[i_d, i_b])

                            dofs_state.qf_applied[i_d, i_b] = ti.math.clamp(
                                force,
                                dofs_info.force_range[I_d][0],
                                dofs_info.force_range[I_d][1],
                            )

                            if ti.abs(force) > EPS:
                                wakeup = True

                    dof_start = links_info.dof_start[I_l]
                    if joint_type == gs.JOINT_TYPE.FREE and (
                        dofs_state.ctrl_mode[dof_start + 3, i_b] == gs.CTRL_MODE.POSITION
                        or dofs_state.ctrl_mode[dof_start + 4, i_b] == gs.CTRL_MODE.POSITION
                        or dofs_state.ctrl_mode[dof_start + 5, i_b] == gs.CTRL_MODE.POSITION
                    ):
                        xyz = ti.Vector(
                            [
                                dofs_state.pos[0 + 3 + dof_start, i_b],
                                dofs_state.pos[1 + 3 + dof_start, i_b],
                                dofs_state.pos[2 + 3 + dof_start, i_b],
                            ],
                            dt=gs.ti_float,
                        )

                        ctrl_xyz = ti.Vector(
                            [
                                dofs_state.ctrl_pos[0 + 3 + dof_start, i_b],
                                dofs_state.ctrl_pos[1 + 3 + dof_start, i_b],
                                dofs_state.ctrl_pos[2 + 3 + dof_start, i_b],
                            ],
                            dt=gs.ti_float,
                        )

                        quat = gu.ti_xyz_to_quat(xyz)
                        ctrl_quat = gu.ti_xyz_to_quat(ctrl_xyz)

                        q_diff = gu.ti_transform_quat_by_quat(ctrl_quat, gu.ti_inv_quat(quat))
                        rotvec = gu.ti_quat_to_rotvec(q_diff, EPS)

                        for j in ti.static(range(3)):
                            i_d = dof_start + 3 + j
                            I_d = [i_d, i_b] if ti.static(static_rigid_sim_config.batch_dofs_info) else i_d
                            force = dofs_info.kp[I_d] * rotvec[j] - dofs_info.kv[I_d] * dofs_state.vel[i_d, i_b]

                            dofs_state.qf_applied[i_d, i_b] = ti.math.clamp(
                                force, dofs_info.force_range[I_d][0], dofs_info.force_range[I_d][1]
                            )

                            if ti.abs(force) > EPS:
                                wakeup = True

        if ti.static(static_rigid_sim_config.use_hibernation):
            if entities_state.hibernated[i_e, i_b] and wakeup:
                # TODO: migrate this function
                func_wakeup_entity_and_its_temp_island(
                    i_e,
                    i_b,
                    entities_state,
                    entities_info,
                    dofs_state,
                    links_state,
                    geoms_state,
                    rigid_global_info,
                    contact_island_state,
                )

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_0, i_b in (
        ti.ndrange(1, dofs_state.ctrl_mode.shape[1])
        if ti.static(static_rigid_sim_config.use_hibernation)
        else ti.ndrange(dofs_state.ctrl_mode.shape[0], dofs_state.ctrl_mode.shape[1])
    ):
        for i_1 in (
            (
                # Dynamic inner for forward pass
                range(rigid_global_info.n_awake_dofs[i_b])
                if ti.static(static_rigid_sim_config.use_hibernation)
                else range(1)
            )
            if ti.static(not static_rigid_sim_config.is_backward)
            else (
                # Static inner for backward pass
                ti.static(range(static_rigid_sim_config.max_n_awake_dofs))
                if ti.static(static_rigid_sim_config.use_hibernation)
                else ti.static(range(1))
            )
        ):
            if func_check_index_range(
                i_1, 0, rigid_global_info.n_awake_dofs[i_b], static_rigid_sim_config.use_hibernation
            ):
                i_d = (
                    rigid_global_info.awake_dofs[i_1, i_b]
                    if ti.static(static_rigid_sim_config.use_hibernation)
                    else i_0
                )

                I_d = [i_d, i_b] if ti.static(static_rigid_sim_config.batch_dofs_info) else i_d
                dofs_state.qf_passive[i_d, i_b] = -dofs_info.damping[I_d] * dofs_state.vel[i_d, i_b]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_0, i_b in (
        ti.ndrange(1, dofs_state.ctrl_mode.shape[1])
        if ti.static(static_rigid_sim_config.use_hibernation)
        else ti.ndrange(links_info.root_idx.shape[0], dofs_state.ctrl_mode.shape[1])
    ):
        for i_1 in (
            (
                # Dynamic inner for forward pass
                range(rigid_global_info.n_awake_links[i_b])
                if ti.static(static_rigid_sim_config.use_hibernation)
                else range(1)
            )
            if ti.static(not static_rigid_sim_config.is_backward)
            else (
                # Static inner for backward pass
                ti.static(range(static_rigid_sim_config.max_n_awake_links))
                if ti.static(static_rigid_sim_config.use_hibernation)
                else ti.static(range(1))
            )
        ):
            if func_check_index_range(
                i_1, 0, rigid_global_info.n_awake_links[i_b], static_rigid_sim_config.use_hibernation
            ):
                i_l = (
                    rigid_global_info.awake_links[i_1, i_b]
                    if ti.static(static_rigid_sim_config.use_hibernation)
                    else i_0
                )
                I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l

                if links_info.n_dofs[I_l] > 0:
                    i_j = links_info.joint_start[I_l]
                    I_j = [i_j, i_b] if ti.static(static_rigid_sim_config.batch_joints_info) else i_j
                    joint_type = joints_info.type[I_j]

                    if joint_type != gs.JOINT_TYPE.FREE and joint_type != gs.JOINT_TYPE.FIXED:
                        q_start = links_info.q_start[I_l]
                        dof_start = links_info.dof_start[I_l]
                        dof_end = links_info.dof_end[I_l]

                        for j_d in (
                            range(dof_end - dof_start)
                            if ti.static(not static_rigid_sim_config.is_backward)
                            else ti.static(range(static_rigid_sim_config.max_n_dofs_per_link))
                        ):
                            if func_check_index_range(j_d, 0, dof_end - dof_start, static_rigid_sim_config.is_backward):
                                I_d = (
                                    [dof_start + j_d, i_b]
                                    if ti.static(static_rigid_sim_config.batch_dofs_info)
                                    else dof_start + j_d
                                )
                                func_atomic_add_if_backward_2d(
                                    dofs_state.qf_passive,
                                    dof_start + j_d,
                                    i_b,
                                    -rigid_global_info.qpos[q_start + j_d, i_b] * dofs_info.stiffness[I_d],
                                    static_rigid_sim_config,
                                )


@ti.func
def func_update_acc(
    update_cacc: ti.template(),
    dofs_state: array_class.DofsState,
    links_info: array_class.LinksInfo,
    links_state: array_class.LinksState,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    # Assume this is the outermost loop
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_0, i_b in (
        ti.ndrange(1, dofs_state.ctrl_mode.shape[1])
        if ti.static(static_rigid_sim_config.use_hibernation)
        else ti.ndrange(entities_info.n_links.shape[0], dofs_state.ctrl_mode.shape[1])
    ):
        for i_1 in (
            (
                # Dynamic inner loop for forward pass
                range(rigid_global_info.n_awake_entities[i_b])
                if ti.static(static_rigid_sim_config.use_hibernation)
                else range(1)
            )
            if ti.static(not static_rigid_sim_config.is_backward)
            else (
                # Static inner loop for backward pass
                ti.static(range(static_rigid_sim_config.max_n_awake_entities))
                if ti.static(static_rigid_sim_config.use_hibernation)
                else ti.static(range(1))
            )
        ):
            if func_check_index_range(
                i_1, 0, rigid_global_info.n_awake_entities[i_b], static_rigid_sim_config.use_hibernation
            ):
                i_e = (
                    rigid_global_info.awake_entities[i_1, i_b]
                    if ti.static(static_rigid_sim_config.use_hibernation)
                    else i_0
                )

                for i_l_ in (
                    range(entities_info.link_start[i_e], entities_info.link_end[i_e])
                    if ti.static(not static_rigid_sim_config.is_backward)
                    else ti.static(range(static_rigid_sim_config.max_n_links_per_entity))
                ):
                    i_l = (
                        i_l_
                        if ti.static(not static_rigid_sim_config.is_backward)
                        else (i_l_ + entities_info.link_start[i_e])
                    )

                    if func_check_index_range(
                        i_l,
                        entities_info.link_start[i_e],
                        entities_info.link_end[i_e],
                        static_rigid_sim_config.is_backward,
                    ):
                        I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l
                        i_p = links_info.parent_idx[I_l]

                        if i_p == -1:
                            links_state.cdd_vel[i_l, i_b] = -rigid_global_info.gravity[i_b] * (
                                1 - entities_info.gravity_compensation[i_e]
                            )
                            links_state.cdd_ang[i_l, i_b] = ti.Vector.zero(gs.ti_float, 3)
                            if ti.static(update_cacc):
                                links_state.cacc_lin[i_l, i_b] = ti.Vector.zero(gs.ti_float, 3)
                                links_state.cacc_ang[i_l, i_b] = ti.Vector.zero(gs.ti_float, 3)
                        else:
                            links_state.cdd_vel[i_l, i_b] = links_state.cdd_vel[i_p, i_b]
                            links_state.cdd_ang[i_l, i_b] = links_state.cdd_ang[i_p, i_b]
                            if ti.static(update_cacc):
                                links_state.cacc_lin[i_l, i_b] = links_state.cacc_lin[i_p, i_b]
                                links_state.cacc_ang[i_l, i_b] = links_state.cacc_ang[i_p, i_b]

                        for i_d_ in (
                            range(links_info.dof_start[I_l], links_info.dof_end[I_l])
                            if ti.static(not static_rigid_sim_config.is_backward)
                            else ti.static(range(static_rigid_sim_config.max_n_dofs_per_link))
                        ):
                            i_d = (
                                i_d_
                                if ti.static(not static_rigid_sim_config.is_backward)
                                else (i_d_ + links_info.dof_start[I_l])
                            )

                            if func_check_index_range(
                                i_d,
                                links_info.dof_start[I_l],
                                links_info.dof_end[I_l],
                                static_rigid_sim_config.is_backward,
                            ):
                                # cacc = cacc_parent + cdofdot * qvel + cdof * qacc
                                local_cdd_vel = dofs_state.cdofd_vel[i_d, i_b] * dofs_state.vel[i_d, i_b]
                                local_cdd_ang = dofs_state.cdofd_ang[i_d, i_b] * dofs_state.vel[i_d, i_b]

                                func_atomic_add_if_backward_2d(
                                    links_state.cdd_vel, i_l, i_b, local_cdd_vel, static_rigid_sim_config
                                )
                                func_atomic_add_if_backward_2d(
                                    links_state.cdd_ang, i_l, i_b, local_cdd_ang, static_rigid_sim_config
                                )
                                if ti.static(update_cacc):
                                    func_atomic_add_if_backward_2d(
                                        links_state.cacc_lin,
                                        i_l,
                                        i_b,
                                        local_cdd_vel + dofs_state.cdof_vel[i_d, i_b] * dofs_state.acc[i_d, i_b],
                                        static_rigid_sim_config,
                                    )
                                    func_atomic_add_if_backward_2d(
                                        links_state.cacc_ang,
                                        i_l,
                                        i_b,
                                        local_cdd_ang + dofs_state.cdof_ang[i_d, i_b] * dofs_state.acc[i_d, i_b],
                                        static_rigid_sim_config,
                                    )


@ti.func
def func_update_force(
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_0, i_b in (
        ti.ndrange(1, links_state.pos.shape[1])
        if ti.static(static_rigid_sim_config.use_hibernation)
        else ti.ndrange(links_info.root_idx.shape[0], links_state.pos.shape[1])
    ):
        for i_1 in (
            (
                # Dynamic inner loop for forward pass
                range(rigid_global_info.n_awake_links[i_b])
                if ti.static(static_rigid_sim_config.use_hibernation)
                else range(1)
            )
            if ti.static(not static_rigid_sim_config.is_backward)
            else (
                # Static inner loop for backward pass
                ti.static(range(static_rigid_sim_config.max_n_awake_links))
                if ti.static(static_rigid_sim_config.use_hibernation)
                else ti.static(range(1))
            )
        ):
            if func_check_index_range(
                i_1, 0, rigid_global_info.n_awake_links[i_b], static_rigid_sim_config.use_hibernation
            ):
                i_l = (
                    rigid_global_info.awake_links[i_1, i_b]
                    if ti.static(static_rigid_sim_config.use_hibernation)
                    else i_0
                )

                f1_ang, f1_vel = gu.inertial_mul(
                    links_state.cinr_pos[i_l, i_b],
                    links_state.cinr_inertial[i_l, i_b],
                    links_state.cinr_mass[i_l, i_b],
                    links_state.cdd_vel[i_l, i_b],
                    links_state.cdd_ang[i_l, i_b],
                )
                f2_ang, f2_vel = gu.inertial_mul(
                    links_state.cinr_pos[i_l, i_b],
                    links_state.cinr_inertial[i_l, i_b],
                    links_state.cinr_mass[i_l, i_b],
                    links_state.cd_vel[i_l, i_b],
                    links_state.cd_ang[i_l, i_b],
                )
                f3_ang, f3_vel = gu.motion_cross_force(
                    links_state.cd_ang[i_l, i_b], links_state.cd_vel[i_l, i_b], f2_ang, f2_vel
                )

                links_state.cfrc_vel[i_l, i_b] = (
                    f1_vel + f3_vel + links_state.cfrc_applied_vel[i_l, i_b] + links_state.cfrc_coupling_vel[i_l, i_b]
                )
                links_state.cfrc_ang[i_l, i_b] = (
                    f1_ang + f3_ang + links_state.cfrc_applied_ang[i_l, i_b] + links_state.cfrc_coupling_ang[i_l, i_b]
                )

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_0, i_b in (
        ti.ndrange(1, links_state.pos.shape[1])
        if ti.static(static_rigid_sim_config.use_hibernation)
        else ti.ndrange(entities_info.n_links.shape[0], links_state.pos.shape[1])
    ):
        for i_1 in (
            (
                # Dynamic inner loop for forward pass
                range(rigid_global_info.n_awake_entities[i_b])
                if ti.static(static_rigid_sim_config.use_hibernation)
                else range(1)
            )
            if ti.static(not static_rigid_sim_config.is_backward)
            else (
                # Static inner loop for backward pass
                ti.static(range(static_rigid_sim_config.max_n_awake_entities))
                if ti.static(static_rigid_sim_config.use_hibernation)
                else ti.static(range(1))
            )
        ):
            if func_check_index_range(
                i_1, 0, rigid_global_info.n_awake_entities[i_b], static_rigid_sim_config.use_hibernation
            ):
                i_e = (
                    rigid_global_info.awake_entities[i_1, i_b]
                    if ti.static(static_rigid_sim_config.use_hibernation)
                    else i_0
                )

                for i_l_ in (
                    range(entities_info.n_links[i_e])
                    if ti.static(not static_rigid_sim_config.is_backward)
                    else ti.static(range(static_rigid_sim_config.max_n_links_per_entity))
                ):
                    if func_check_index_range(i_l_, 0, entities_info.n_links[i_e], static_rigid_sim_config.is_backward):
                        i_l = entities_info.link_end[i_e] - 1 - i_l_
                        I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l
                        i_p = links_info.parent_idx[I_l]
                        if i_p != -1:
                            func_atomic_add_if_backward_2d(
                                links_state.cfrc_vel, i_p, i_b, links_state.cfrc_vel[i_l, i_b], static_rigid_sim_config
                            )
                            func_atomic_add_if_backward_2d(
                                links_state.cfrc_ang, i_p, i_b, links_state.cfrc_ang[i_l, i_b], static_rigid_sim_config
                            )

    # Clear coupling forces after use
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for I in ti.grouped(ti.ndrange(*links_state.cfrc_coupling_ang.shape)):
        links_state.cfrc_coupling_ang[I] = ti.Vector.zero(gs.ti_float, 3)
        links_state.cfrc_coupling_vel[I] = ti.Vector.zero(gs.ti_float, 3)


@ti.func
def func_actuation(self):
    if ti.static(self._use_hibernation):
        pass
    else:
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
        for i_l, i_b in ti.ndrange(self.n_links, self._B):
            I_l = [i_l, i_b] if ti.static(self._options.batch_links_info) else i_l
            for i_j in range(self.links_info.joint_start[I_l], self.links_info.joint_end[I_l]):
                I_j = [i_j, i_b] if ti.static(self._options.batch_joints_info) else i_j
                joint_type = self.joints_info.type[I_j]
                q_start = self.joints_info.q_start[I_j]

                if joint_type == gs.JOINT_TYPE.REVOLUTE or joint_type == gs.JOINT_TYPE.PRISMATIC:
                    gear = -1  # TODO
                    i_d = self.links_info.dof_start[I_l]
                    self.dofs_state.act_length[i_d, i_b] = gear * self.qpos[q_start, i_b]
                    self.dofs_state.qf_actuator[i_d, i_b] = self.dofs_state.act_length[i_d, i_b]
                else:
                    for i_d in range(self.links_info.dof_start[I_l], self.links_info.dof_end[I_l]):
                        self.dofs_state.act_length[i_d, i_b] = 0.0
                        self.dofs_state.qf_actuator[i_d, i_b] = self.dofs_state.act_length[i_d, i_b]


@ti.func
def func_bias_force(
    dofs_state: array_class.DofsState,
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_0, i_b in (
        ti.ndrange(1, dofs_state.ctrl_mode.shape[1])
        if ti.static(static_rigid_sim_config.use_hibernation)
        else ti.ndrange(links_info.root_idx.shape[0], dofs_state.ctrl_mode.shape[1])
    ):
        for i_1 in (
            (
                # Dynamic inner loop for forward pass
                range(rigid_global_info.n_awake_links[i_b])
                if ti.static(static_rigid_sim_config.use_hibernation)
                else range(1)
            )
            if ti.static(not static_rigid_sim_config.is_backward)
            else (
                # Static inner loop for backward pass
                ti.static(range(static_rigid_sim_config.max_n_awake_links))
                if ti.static(static_rigid_sim_config.use_hibernation)
                else ti.static(range(1))
            )
        ):
            if func_check_index_range(
                i_1, 0, rigid_global_info.n_awake_links[i_b], static_rigid_sim_config.use_hibernation
            ):
                i_l = (
                    rigid_global_info.awake_links[i_1, i_b]
                    if ti.static(static_rigid_sim_config.use_hibernation)
                    else i_0
                )
                I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l

                for i_d_ in (
                    range(links_info.dof_start[I_l], links_info.dof_end[I_l])
                    if ti.static(not static_rigid_sim_config.is_backward)
                    else ti.static(range(static_rigid_sim_config.max_n_dofs_per_link))
                ):
                    i_d = (
                        i_d_
                        if ti.static(not static_rigid_sim_config.is_backward)
                        else (i_d_ + links_info.dof_start[I_l])
                    )
                    if func_check_index_range(
                        i_d, links_info.dof_start[I_l], links_info.dof_end[I_l], static_rigid_sim_config.is_backward
                    ):
                        dofs_state.qf_bias[i_d, i_b] = dofs_state.cdof_ang[i_d, i_b].dot(
                            links_state.cfrc_ang[i_l, i_b]
                        ) + dofs_state.cdof_vel[i_d, i_b].dot(links_state.cfrc_vel[i_l, i_b])

                        dofs_state.force[i_d, i_b] = (
                            dofs_state.qf_passive[i_d, i_b]
                            - dofs_state.qf_bias[i_d, i_b]
                            + dofs_state.qf_applied[i_d, i_b]
                            # + self.dofs_state.qf_actuator[i_d, i_b]
                        )

                        dofs_state.qf_smooth[i_d, i_b] = dofs_state.force[i_d, i_b]


@ti.kernel
def kernel_compute_qacc(
    dofs_state: array_class.DofsState,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    func_compute_qacc(
        dofs_state=dofs_state,
        entities_info=entities_info,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
    )


@ti.func
def func_compute_qacc(
    dofs_state: array_class.DofsState,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    func_solve_mass(
        vec=dofs_state.force,
        out=dofs_state.acc_smooth,
        out_bw=dofs_state.acc_smooth_bw,
        entities_info=entities_info,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
    )

    # Assume this is the outermost loop
    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_0, i_b in (
        ti.ndrange(1, dofs_state.ctrl_mode.shape[1])
        if ti.static(static_rigid_sim_config.use_hibernation)
        else ti.ndrange(entities_info.n_links.shape[0], dofs_state.ctrl_mode.shape[1])
    ):
        for i_1 in (
            (
                # Dynamic inner loop for forward pass
                range(rigid_global_info.n_awake_entities[i_b])
                if ti.static(static_rigid_sim_config.use_hibernation)
                else range(1)
            )
            if ti.static(not static_rigid_sim_config.is_backward)
            else (
                # Static inner loop for backward pass
                ti.static(range(static_rigid_sim_config.max_n_awake_entities))
                if ti.static(static_rigid_sim_config.use_hibernation)
                else ti.static(range(1))
            )
        ):
            if func_check_index_range(
                i_1, 0, rigid_global_info.n_awake_entities[i_b], static_rigid_sim_config.use_hibernation
            ):
                i_e = (
                    rigid_global_info.awake_entities[i_1, i_b]
                    if ti.static(static_rigid_sim_config.use_hibernation)
                    else i_0
                )

                for i_d1_ in (
                    range(entities_info.n_dofs[i_e])
                    if ti.static(not static_rigid_sim_config.is_backward)
                    else ti.static(range(static_rigid_sim_config.max_n_dofs_per_entity))
                ):
                    i_d1 = entities_info.dof_start[i_e] + i_d1_
                    if func_check_index_range(
                        i_d1,
                        entities_info.dof_start[i_e],
                        entities_info.dof_end[i_e],
                        static_rigid_sim_config.is_backward,
                    ):
                        dofs_state.acc[i_d1, i_b] = dofs_state.acc_smooth[i_d1, i_b]


@ti.func
def func_integrate(
    dofs_state: array_class.DofsState,
    links_info: array_class.LinksInfo,
    joints_info: array_class.JointsInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_0, i_b in (
        (ti.ndrange(1, dofs_state.ctrl_mode.shape[1]))
        if ti.static(static_rigid_sim_config.use_hibernation)
        else (ti.ndrange(dofs_state.ctrl_mode.shape[0], dofs_state.ctrl_mode.shape[1]))
    ):
        for i_1 in (
            (
                # Dynamic inner loop for forward pass
                range(rigid_global_info.n_awake_dofs[i_b])
                if ti.static(static_rigid_sim_config.use_hibernation)
                else range(1)
            )
            if ti.static(not static_rigid_sim_config.is_backward)
            else (
                # Static inner loop for backward pass
                ti.static(range(static_rigid_sim_config.max_n_awake_dofs))
                if ti.static(static_rigid_sim_config.use_hibernation)
                else ti.static(range(1))
            )
        ):
            if func_check_index_range(
                i_1, 0, rigid_global_info.n_awake_dofs[i_b], static_rigid_sim_config.use_hibernation
            ):
                i_d = (
                    rigid_global_info.awake_dofs[i_1, i_b]
                    if ti.static(static_rigid_sim_config.use_hibernation)
                    else i_0
                )

                # Prevent nan propagation
                is_valid = True
                if ti.static(not static_rigid_sim_config.is_backward):
                    is_valid = ~ti.math.isnan(dofs_state.acc[i_d, i_b])
                if is_valid:
                    dofs_state.vel_next[i_d, i_b] = (
                        dofs_state.vel[i_d, i_b] + dofs_state.acc[i_d, i_b] * rigid_global_info.substep_dt[None]
                    )

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_0, i_b in (
        (ti.ndrange(1, dofs_state.ctrl_mode.shape[1]))
        if ti.static(static_rigid_sim_config.use_hibernation)
        else (ti.ndrange(links_info.root_idx.shape[0], dofs_state.ctrl_mode.shape[1]))
    ):
        for i_1 in (
            (
                # Dynamic inner loop for forward pass
                range(rigid_global_info.n_awake_links[i_b])
                if ti.static(static_rigid_sim_config.use_hibernation)
                else range(1)
            )
            if ti.static(not static_rigid_sim_config.is_backward)
            else (
                # Static inner loop for backward pass
                ti.static(range(static_rigid_sim_config.max_n_awake_links))
                if ti.static(static_rigid_sim_config.use_hibernation)
                else ti.static(range(1))
            )
        ):
            if func_check_index_range(
                i_1, 0, rigid_global_info.n_awake_links[i_b], static_rigid_sim_config.use_hibernation
            ):
                i_l = (
                    rigid_global_info.awake_links[i_1, i_b]
                    if ti.static(static_rigid_sim_config.use_hibernation)
                    else i_0
                )
                I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l
                if links_info.n_dofs[I_l] > 0:
                    EPS = rigid_global_info.EPS[None]
                    dof_start = links_info.dof_start[I_l]
                    q_start = links_info.q_start[I_l]
                    q_end = links_info.q_end[I_l]

                    i_j = links_info.joint_start[I_l]
                    I_j = [i_j, i_b] if ti.static(static_rigid_sim_config.batch_joints_info) else i_j
                    joint_type = joints_info.type[I_j]

                    if joint_type == gs.JOINT_TYPE.FREE:
                        pos = ti.Vector(
                            [
                                rigid_global_info.qpos[q_start, i_b],
                                rigid_global_info.qpos[q_start + 1, i_b],
                                rigid_global_info.qpos[q_start + 2, i_b],
                            ]
                        )
                        vel = ti.Vector(
                            [
                                dofs_state.vel_next[dof_start, i_b],
                                dofs_state.vel_next[dof_start + 1, i_b],
                                dofs_state.vel_next[dof_start + 2, i_b],
                            ]
                        )
                        # Backward pass requires atomic add
                        if ti.static(static_rigid_sim_config.is_backward):
                            pos += vel * rigid_global_info.substep_dt[None]
                        else:
                            pos = pos + vel * rigid_global_info.substep_dt[None]
                        for j in ti.static(range(3)):
                            rigid_global_info.qpos_next[q_start + j, i_b] = pos[j]
                    if joint_type == gs.JOINT_TYPE.SPHERICAL or joint_type == gs.JOINT_TYPE.FREE:
                        rot_offset = 3 if joint_type == gs.JOINT_TYPE.FREE else 0
                        rot0 = ti.Vector(
                            [
                                rigid_global_info.qpos[q_start + rot_offset + 0, i_b],
                                rigid_global_info.qpos[q_start + rot_offset + 1, i_b],
                                rigid_global_info.qpos[q_start + rot_offset + 2, i_b],
                                rigid_global_info.qpos[q_start + rot_offset + 3, i_b],
                            ]
                        )
                        ang = (
                            ti.Vector(
                                [
                                    dofs_state.vel_next[dof_start + rot_offset + 0, i_b],
                                    dofs_state.vel_next[dof_start + rot_offset + 1, i_b],
                                    dofs_state.vel_next[dof_start + rot_offset + 2, i_b],
                                ]
                            )
                            * rigid_global_info.substep_dt[None]
                        )
                        qrot = gu.ti_rotvec_to_quat(ang, EPS)
                        rot = gu.ti_transform_quat_by_quat(qrot, rot0)
                        for j in ti.static(range(4)):
                            rigid_global_info.qpos_next[q_start + j + rot_offset, i_b] = rot[j]
                    else:
                        for j_ in (
                            (range(q_end - q_start))
                            if ti.static(not static_rigid_sim_config.is_backward)
                            else (ti.static(range(static_rigid_sim_config.max_n_qs_per_link)))
                        ):
                            j = q_start + j_
                            if j < q_end:
                                rigid_global_info.qpos_next[j, i_b] = (
                                    rigid_global_info.qpos[j, i_b]
                                    + dofs_state.vel_next[dof_start + j_, i_b] * rigid_global_info.substep_dt[None]
                                )


@ti.func
def func_copy_next_to_curr(
    dofs_state: array_class.DofsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for I in ti.grouped(ti.ndrange(*dofs_state.vel.shape)):
        dofs_state.vel[I] = dofs_state.vel_next[I]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for I in ti.grouped(ti.ndrange(*rigid_global_info.qpos.shape)):
        rigid_global_info.qpos[I] = rigid_global_info.qpos_next[I]


@ti.func
def func_copy_next_to_curr_grad(
    f: ti.int32,
    dofs_state: array_class.DofsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    rigid_adjoint_cache: array_class.RigidAdjointCache,
    static_rigid_sim_config: ti.template(),
):
    n_dofs = dofs_state.vel.shape[0]
    n_qs = rigid_global_info.qpos.shape[0]
    _B = dofs_state.vel.shape[1]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_d, i_b in ti.ndrange(n_dofs, _B):
        dofs_state.vel_next.grad[i_d, i_b] = dofs_state.vel.grad[i_d, i_b]
        dofs_state.vel.grad[i_d, i_b] = 0.0
        dofs_state.vel[i_d, i_b] = rigid_adjoint_cache.dofs_vel[f, i_d, i_b]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_q, i_b in ti.ndrange(n_qs, _B):
        rigid_global_info.qpos_next.grad[i_q, i_b] = rigid_global_info.qpos.grad[i_q, i_b]
        rigid_global_info.qpos.grad[i_q, i_b] = 0.0
        rigid_global_info.qpos[i_q, i_b] = rigid_adjoint_cache.qpos[f, i_q, i_b]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_save_adjoint_cache(
    f: ti.int32,
    dofs_state: array_class.DofsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    rigid_adjoint_cache: array_class.RigidAdjointCache,
    static_rigid_sim_config: ti.template(),
):
    func_save_adjoint_cache(f, dofs_state, rigid_global_info, rigid_adjoint_cache, static_rigid_sim_config)


@ti.func
def func_save_adjoint_cache(
    f: ti.int32,
    dofs_state: array_class.DofsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    rigid_adjoint_cache: array_class.RigidAdjointCache,
    static_rigid_sim_config: ti.template(),
):
    n_dofs = dofs_state.vel.shape[0]
    n_qs = rigid_global_info.qpos.shape[0]
    _B = dofs_state.vel.shape[1]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_d, i_b in ti.ndrange(n_dofs, _B):
        rigid_adjoint_cache.dofs_vel[f, i_d, i_b] = dofs_state.vel[i_d, i_b]
        rigid_adjoint_cache.dofs_acc[f, i_d, i_b] = dofs_state.acc[i_d, i_b]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_q, i_b in ti.ndrange(n_qs, _B):
        rigid_adjoint_cache.qpos[f, i_q, i_b] = rigid_global_info.qpos[i_q, i_b]


@ti.func
def func_load_adjoint_cache(
    f: ti.int32,
    dofs_state: array_class.DofsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    rigid_adjoint_cache: array_class.RigidAdjointCache,
    static_rigid_sim_config: ti.template(),
):
    n_dofs = dofs_state.vel.shape[0]
    n_qs = rigid_global_info.qpos.shape[0]
    _B = dofs_state.vel.shape[1]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_d, i_b in ti.ndrange(n_dofs, _B):
        dofs_state.vel[i_d, i_b] = rigid_adjoint_cache.dofs_vel[f, i_d, i_b]
        dofs_state.acc[i_d, i_b] = rigid_adjoint_cache.dofs_acc[f, i_d, i_b]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_q, i_b in ti.ndrange(n_qs, _B):
        rigid_global_info.qpos[i_q, i_b] = rigid_adjoint_cache.qpos[f, i_q, i_b]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_prepare_backward_substep(
    f: ti.int32,
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    joints_state: array_class.JointsState,
    joints_info: array_class.JointsInfo,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    dofs_state_adjoint_cache: array_class.DofsState,
    links_state_adjoint_cache: array_class.LinksState,
    joints_state_adjoint_cache: array_class.JointsState,
    geoms_state_adjoint_cache: array_class.GeomsState,
    rigid_adjoint_cache: array_class.RigidAdjointCache,
    static_rigid_sim_config: ti.template(),
):
    # Load the current state from adjoint cache
    func_load_adjoint_cache(
        f=f,
        dofs_state=dofs_state,
        rigid_global_info=rigid_global_info,
        rigid_adjoint_cache=rigid_adjoint_cache,
        static_rigid_sim_config=static_rigid_sim_config,
    )

    # If mujoco compatibility is disabled, update the cartesian space and save the results to adjoint cache. This is
    # because the cartesian space is overwritten later by other kernels if mujoco compatibility was disabled.
    if not static_rigid_sim_config.enable_mujoco_compatibility:
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(links_state.pos.shape[1]):
            func_update_cartesian_space(
                i_b=i_b,
                links_state=links_state,
                links_info=links_info,
                joints_state=joints_state,
                joints_info=joints_info,
                dofs_state=dofs_state,
                dofs_info=dofs_info,
                geoms_state=geoms_state,
                geoms_info=geoms_info,
                entities_info=entities_info,
                rigid_global_info=rigid_global_info,
                static_rigid_sim_config=static_rigid_sim_config,
                force_update_fixed_geoms=False,
            )

        # FIXME: Parameter pruning for ndarray is buggy on this one. Inlining this function is "fixing" this issue.
        # Save results of [update_cartesian_space] to adjoint cache
        # func_copy_cartesian_space(
        #     src_dofs_state=dofs_state,
        #     src_links_state=links_state,
        #     src_joints_state=joints_state,
        #     src_geoms_state=geoms_state,
        #     dst_dofs_state=dofs_state_adjoint_cache,
        #     dst_links_state=links_state_adjoint_cache,
        #     dst_joints_state=joints_state_adjoint_cache,
        #     dst_geoms_state=geoms_state_adjoint_cache,
        #     static_rigid_sim_config=static_rigid_sim_config,
        # )

        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for I in ti.grouped(ti.ndrange(*dofs_state.pos.shape)):
            # pos, cdof_ang, cdof_vel, cdofvel_ang, cdofvel_vel, cdofd_ang, cdofd_vel
            dofs_state_adjoint_cache.pos[I] = dofs_state.pos[I]
            dofs_state_adjoint_cache.cdof_ang[I] = dofs_state.cdof_ang[I]
            dofs_state_adjoint_cache.cdof_vel[I] = dofs_state.cdof_vel[I]
            dofs_state_adjoint_cache.cdofvel_ang[I] = dofs_state.cdofvel_ang[I]
            dofs_state_adjoint_cache.cdofvel_vel[I] = dofs_state.cdofvel_vel[I]
            dofs_state_adjoint_cache.cdofd_ang[I] = dofs_state.cdofd_ang[I]
            dofs_state_adjoint_cache.cdofd_vel[I] = dofs_state.cdofd_vel[I]

        # links state
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for I in ti.grouped(ti.ndrange(*links_state.pos.shape)):
            # pos, quat, root_COM, mass_sum, i_pos, i_quat, cinr_inertial, cinr_pos, cinr_quat, cinr_mass, j_pos, j_quat,
            # cd_vel, cd_ang
            links_state_adjoint_cache.pos[I] = links_state.pos[I]
            links_state_adjoint_cache.quat[I] = links_state.quat[I]
            links_state_adjoint_cache.root_COM[I] = links_state.root_COM[I]
            links_state_adjoint_cache.mass_sum[I] = links_state.mass_sum[I]
            links_state_adjoint_cache.i_pos[I] = links_state.i_pos[I]
            links_state_adjoint_cache.i_quat[I] = links_state.i_quat[I]
            links_state_adjoint_cache.cinr_inertial[I] = links_state.cinr_inertial[I]
            links_state_adjoint_cache.cinr_pos[I] = links_state.cinr_pos[I]
            links_state_adjoint_cache.cinr_quat[I] = links_state.cinr_quat[I]
            links_state_adjoint_cache.cinr_mass[I] = links_state.cinr_mass[I]
            links_state_adjoint_cache.j_pos[I] = links_state.j_pos[I]
            links_state_adjoint_cache.j_quat[I] = links_state.j_quat[I]
            links_state_adjoint_cache.cd_vel[I] = links_state.cd_vel[I]
            links_state_adjoint_cache.cd_ang[I] = links_state.cd_ang[I]

        # joints state
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for I in ti.grouped(ti.ndrange(*joints_state.xanchor.shape)):
            # xanchor, xaxis
            joints_state_adjoint_cache.xanchor[I] = joints_state.xanchor[I]
            joints_state_adjoint_cache.xaxis[I] = joints_state.xaxis[I]

        # geoms state
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for I in ti.grouped(ti.ndrange(*geoms_state.pos.shape)):
            # pos, quat, verts_updated
            geoms_state_adjoint_cache.pos[I] = geoms_state.pos[I]
            geoms_state_adjoint_cache.quat[I] = geoms_state.quat[I]
            geoms_state_adjoint_cache.verts_updated[I] = geoms_state.verts_updated[I]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_begin_backward_substep(
    f: ti.int32,
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    joints_state: array_class.JointsState,
    joints_info: array_class.JointsInfo,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    dofs_state_adjoint_cache: array_class.DofsState,
    links_state_adjoint_cache: array_class.LinksState,
    joints_state_adjoint_cache: array_class.JointsState,
    geoms_state_adjoint_cache: array_class.GeomsState,
    rigid_adjoint_cache: array_class.RigidAdjointCache,
    static_rigid_sim_config: ti.template(),
) -> ti.i32:
    is_grad_valid = func_is_grad_valid(
        rigid_global_info=rigid_global_info,
        dofs_state=dofs_state,
        static_rigid_sim_config=static_rigid_sim_config,
    )
    if is_grad_valid:
        func_copy_next_to_curr_grad(
            f=f,
            dofs_state=dofs_state,
            rigid_global_info=rigid_global_info,
            rigid_adjoint_cache=rigid_adjoint_cache,
            static_rigid_sim_config=static_rigid_sim_config,
        )

        if not static_rigid_sim_config.enable_mujoco_compatibility:
            # FIXME: Parameter pruning for ndarray is buggy on this one. Inlining this function is "fixing" this issue.
            # Save results of [update_cartesian_space] to adjoint cache
            # func_copy_cartesian_space(
            #     src_dofs_state=dofs_state,
            #     src_links_state=links_state,
            #     src_joints_state=joints_state,
            #     src_geoms_state=geoms_state,
            #     dst_dofs_state=dofs_state_adjoint_cache,
            #     dst_links_state=links_state_adjoint_cache,
            #     dst_joints_state=joints_state_adjoint_cache,
            #     dst_geoms_state=geoms_state_adjoint_cache,
            #     static_rigid_sim_config=static_rigid_sim_config,
            # )

            ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
            for I in ti.grouped(ti.ndrange(*dofs_state.pos.shape)):
                # pos, cdof_ang, cdof_vel, cdofvel_ang, cdofvel_vel, cdofd_ang, cdofd_vel
                dofs_state_adjoint_cache.pos[I] = dofs_state.pos[I]
                dofs_state_adjoint_cache.cdof_ang[I] = dofs_state.cdof_ang[I]
                dofs_state_adjoint_cache.cdof_vel[I] = dofs_state.cdof_vel[I]
                dofs_state_adjoint_cache.cdofvel_ang[I] = dofs_state.cdofvel_ang[I]
                dofs_state_adjoint_cache.cdofvel_vel[I] = dofs_state.cdofvel_vel[I]
                dofs_state_adjoint_cache.cdofd_ang[I] = dofs_state.cdofd_ang[I]
                dofs_state_adjoint_cache.cdofd_vel[I] = dofs_state.cdofd_vel[I]

            # links state
            ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
            for I in ti.grouped(ti.ndrange(*links_state.pos.shape)):
                # pos, quat, root_COM, mass_sum, i_pos, i_quat, cinr_inertial, cinr_pos, cinr_quat, cinr_mass, j_pos, j_quat,
                # cd_vel, cd_ang
                links_state_adjoint_cache.pos[I] = links_state.pos[I]
                links_state_adjoint_cache.quat[I] = links_state.quat[I]
                links_state_adjoint_cache.root_COM[I] = links_state.root_COM[I]
                links_state_adjoint_cache.mass_sum[I] = links_state.mass_sum[I]
                links_state_adjoint_cache.i_pos[I] = links_state.i_pos[I]
                links_state_adjoint_cache.i_quat[I] = links_state.i_quat[I]
                links_state_adjoint_cache.cinr_inertial[I] = links_state.cinr_inertial[I]
                links_state_adjoint_cache.cinr_pos[I] = links_state.cinr_pos[I]
                links_state_adjoint_cache.cinr_quat[I] = links_state.cinr_quat[I]
                links_state_adjoint_cache.cinr_mass[I] = links_state.cinr_mass[I]
                links_state_adjoint_cache.j_pos[I] = links_state.j_pos[I]
                links_state_adjoint_cache.j_quat[I] = links_state.j_quat[I]
                links_state_adjoint_cache.cd_vel[I] = links_state.cd_vel[I]
                links_state_adjoint_cache.cd_ang[I] = links_state.cd_ang[I]

            # joints state
            ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
            for I in ti.grouped(ti.ndrange(*joints_state.xanchor.shape)):
                # xanchor, xaxis
                joints_state_adjoint_cache.xanchor[I] = joints_state.xanchor[I]
                joints_state_adjoint_cache.xaxis[I] = joints_state.xaxis[I]

            # geoms state
            ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
            for I in ti.grouped(ti.ndrange(*geoms_state.pos.shape)):
                # pos, quat, verts_updated
                geoms_state_adjoint_cache.pos[I] = geoms_state.pos[I]
                geoms_state_adjoint_cache.quat[I] = geoms_state.quat[I]
                geoms_state_adjoint_cache.verts_updated[I] = geoms_state.verts_updated[I]

    return is_grad_valid


@ti.func
def func_is_grad_valid(
    rigid_global_info: array_class.RigidGlobalInfo,
    dofs_state: array_class.DofsState,
    static_rigid_sim_config: ti.template(),
):
    is_valid = True
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for I in ti.grouped(ti.ndrange(*rigid_global_info.qpos.shape)):
        if ti.math.isnan(rigid_global_info.qpos.grad[I]):
            is_valid = False

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for I in ti.grouped(ti.ndrange(*dofs_state.vel.shape)):
        if ti.math.isnan(dofs_state.vel.grad[I]):
            is_valid = False

    return is_valid


@ti.func
def func_copy_cartesian_space(
    src_dofs_state: array_class.DofsState,
    src_links_state: array_class.LinksState,
    src_joints_state: array_class.JointsState,
    src_geoms_state: array_class.GeomsState,
    dst_dofs_state: array_class.DofsState,
    dst_links_state: array_class.LinksState,
    dst_joints_state: array_class.JointsState,
    dst_geoms_state: array_class.GeomsState,
    static_rigid_sim_config: ti.template(),
):
    # Copy outputs of [kernel_update_cartesian_space] among [dofs, links, joints, geoms] states. This is used to restore
    # the outputs that were overwritten if we disabled mujoco compatibility for backward pass.

    # dofs state
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for I in ti.grouped(ti.ndrange(*src_dofs_state.pos.shape)):
        # pos, cdof_ang, cdof_vel, cdofvel_ang, cdofvel_vel, cdofd_ang, cdofd_vel
        dst_dofs_state.pos[I] = src_dofs_state.pos[I]
        dst_dofs_state.cdof_ang[I] = src_dofs_state.cdof_ang[I]
        dst_dofs_state.cdof_vel[I] = src_dofs_state.cdof_vel[I]
        dst_dofs_state.cdofvel_ang[I] = src_dofs_state.cdofvel_ang[I]
        dst_dofs_state.cdofvel_vel[I] = src_dofs_state.cdofvel_vel[I]
        dst_dofs_state.cdofd_ang[I] = src_dofs_state.cdofd_ang[I]
        dst_dofs_state.cdofd_vel[I] = src_dofs_state.cdofd_vel[I]

    # links state
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for I in ti.grouped(ti.ndrange(*src_links_state.pos.shape)):
        # pos, quat, root_COM, mass_sum, i_pos, i_quat, cinr_inertial, cinr_pos, cinr_quat, cinr_mass, j_pos, j_quat,
        # cd_vel, cd_ang
        dst_links_state.pos[I] = src_links_state.pos[I]
        dst_links_state.quat[I] = src_links_state.quat[I]
        dst_links_state.root_COM[I] = src_links_state.root_COM[I]
        dst_links_state.mass_sum[I] = src_links_state.mass_sum[I]
        dst_links_state.i_pos[I] = src_links_state.i_pos[I]
        dst_links_state.i_quat[I] = src_links_state.i_quat[I]
        dst_links_state.cinr_inertial[I] = src_links_state.cinr_inertial[I]
        dst_links_state.cinr_pos[I] = src_links_state.cinr_pos[I]
        dst_links_state.cinr_quat[I] = src_links_state.cinr_quat[I]
        dst_links_state.cinr_mass[I] = src_links_state.cinr_mass[I]
        dst_links_state.j_pos[I] = src_links_state.j_pos[I]
        dst_links_state.j_quat[I] = src_links_state.j_quat[I]
        dst_links_state.cd_vel[I] = src_links_state.cd_vel[I]
        dst_links_state.cd_ang[I] = src_links_state.cd_ang[I]

    # joints state
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for I in ti.grouped(ti.ndrange(*src_joints_state.xanchor.shape)):
        # xanchor, xaxis
        dst_joints_state.xanchor[I] = src_joints_state.xanchor[I]
        dst_joints_state.xaxis[I] = src_joints_state.xaxis[I]

    # geoms state
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for I in ti.grouped(ti.ndrange(*src_geoms_state.pos.shape)):
        # pos, quat, verts_updated
        dst_geoms_state.pos[I] = src_geoms_state.pos[I]
        dst_geoms_state.quat[I] = src_geoms_state.quat[I]
        dst_geoms_state.verts_updated[I] = src_geoms_state.verts_updated[I]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_copy_acc(
    f: ti.int32,
    dofs_state: array_class.DofsState,
    rigid_adjoint_cache: array_class.RigidAdjointCache,
    static_rigid_sim_config: ti.template(),
):
    n_dofs = dofs_state.vel.shape[0]
    _B = dofs_state.vel.shape[1]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_d, i_b in ti.ndrange(n_dofs, _B):
        dofs_state.acc[i_d, i_b] = rigid_adjoint_cache.dofs_acc[f, i_d, i_b]


@ti.func
def func_integrate_dq_entity(
    dq,
    i_e,
    i_b,
    respect_joint_limit,
    links_info: array_class.LinksInfo,
    joints_info: array_class.JointsInfo,
    dofs_info: array_class.DofsInfo,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    EPS = rigid_global_info.EPS[None]

    for i_l in range(entities_info.link_start[i_e], entities_info.link_end[i_e]):
        I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l
        if links_info.n_dofs[I_l] == 0:
            continue

        i_j = links_info.joint_start[I_l]
        I_j = [i_j, i_b] if ti.static(static_rigid_sim_config.batch_joints_info) else i_j
        joint_type = joints_info.type[I_j]

        q_start = links_info.q_start[I_l]
        dof_start = links_info.dof_start[I_l]
        dq_start = links_info.dof_start[I_l] - entities_info.dof_start[i_e]

        if joint_type == gs.JOINT_TYPE.FREE:
            pos = ti.Vector(
                [
                    rigid_global_info.qpos[q_start, i_b],
                    rigid_global_info.qpos[q_start + 1, i_b],
                    rigid_global_info.qpos[q_start + 2, i_b],
                ]
            )
            dpos = ti.Vector([dq[dq_start, i_b], dq[dq_start + 1, i_b], dq[dq_start + 2, i_b]])
            pos = pos + dpos

            quat = ti.Vector(
                [
                    rigid_global_info.qpos[q_start + 3, i_b],
                    rigid_global_info.qpos[q_start + 4, i_b],
                    rigid_global_info.qpos[q_start + 5, i_b],
                    rigid_global_info.qpos[q_start + 6, i_b],
                ]
            )
            dquat = gu.ti_rotvec_to_quat(
                ti.Vector([dq[dq_start + 3, i_b], dq[dq_start + 4, i_b], dq[dq_start + 5, i_b]], dt=gs.ti_float), EPS
            )
            quat = gu.ti_transform_quat_by_quat(
                quat, dquat
            )  # Note that this order is different from integrateing vel. Here dq is w.r.t to world.

            for j in ti.static(range(3)):
                rigid_global_info.qpos[q_start + j, i_b] = pos[j]

            for j in ti.static(range(4)):
                rigid_global_info.qpos[q_start + j + 3, i_b] = quat[j]

        elif joint_type == gs.JOINT_TYPE.FIXED:
            pass

        else:
            for i_d_ in range(links_info.n_dofs[I_l]):
                rigid_global_info.qpos[q_start + i_d_, i_b] = (
                    rigid_global_info.qpos[q_start + i_d_, i_b] + dq[dq_start + i_d_, i_b]
                )

                if respect_joint_limit:
                    I_d = (
                        [dof_start + i_d_, i_b]
                        if ti.static(static_rigid_sim_config.batch_dofs_info)
                        else dof_start + i_d_
                    )
                    rigid_global_info.qpos[q_start + i_d_, i_b] = ti.math.clamp(
                        rigid_global_info.qpos[q_start + i_d_, i_b],
                        dofs_info.limit[I_d][0],
                        dofs_info.limit[I_d][1],
                    )


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_update_geoms_render_T(
    geoms_render_T: ti.types.ndarray(),
    geoms_state: array_class.GeomsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    EPS = rigid_global_info.EPS[None]

    n_geoms = geoms_state.pos.shape[0]
    _B = geoms_state.pos.shape[1]
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_g, i_b in ti.ndrange(n_geoms, _B):
        geom_T = gu.ti_trans_quat_to_T(
            geoms_state.pos[i_g, i_b] + rigid_global_info.envs_offset[i_b],
            geoms_state.quat[i_g, i_b],
            EPS,
        )
        for J in ti.static(ti.grouped(ti.static(ti.ndrange(4, 4)))):
            geoms_render_T[(i_g, i_b, *J)] = ti.cast(geom_T[J], ti.float32)


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_update_vgeoms_render_T(
    vgeoms_render_T: ti.types.ndarray(),
    vgeoms_info: array_class.VGeomsInfo,
    vgeoms_state: array_class.VGeomsState,
    links_state: array_class.LinksState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    EPS = rigid_global_info.EPS[None]

    n_vgeoms = vgeoms_info.link_idx.shape[0]
    _B = links_state.pos.shape[1]
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_g, i_b in ti.ndrange(n_vgeoms, _B):
        geom_T = gu.ti_trans_quat_to_T(
            vgeoms_state.pos[i_g, i_b] + rigid_global_info.envs_offset[i_b],
            vgeoms_state.quat[i_g, i_b],
            EPS,
        )
        for J in ti.static(ti.grouped(ti.ndrange(4, 4))):
            vgeoms_render_T[(i_g, i_b, *J)] = ti.cast(geom_T[J], ti.float32)


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_get_state(
    qpos: ti.types.ndarray(),
    vel: ti.types.ndarray(),
    acc: ti.types.ndarray(),
    links_pos: ti.types.ndarray(),
    links_quat: ti.types.ndarray(),
    i_pos_shift: ti.types.ndarray(),
    mass_shift: ti.types.ndarray(),
    friction_ratio: ti.types.ndarray(),
    links_state: array_class.LinksState,
    dofs_state: array_class.DofsState,
    geoms_state: array_class.GeomsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):

    n_qs = qpos.shape[1]
    n_dofs = vel.shape[1]
    n_links = links_pos.shape[1]
    n_geoms = friction_ratio.shape[1]
    _B = qpos.shape[0]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_q, i_b in ti.ndrange(n_qs, _B):
        qpos[i_b, i_q] = rigid_global_info.qpos[i_q, i_b]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_d, i_b in ti.ndrange(n_dofs, _B):
        vel[i_b, i_d] = dofs_state.vel[i_d, i_b]
        acc[i_b, i_d] = dofs_state.acc[i_d, i_b]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l, i_b in ti.ndrange(n_links, _B):
        for j in ti.static(range(3)):
            links_pos[i_b, i_l, j] = links_state.pos[i_l, i_b][j]
            i_pos_shift[i_b, i_l, j] = links_state.i_pos_shift[i_l, i_b][j]
        for j in ti.static(range(4)):
            links_quat[i_b, i_l, j] = links_state.quat[i_l, i_b][j]
        mass_shift[i_b, i_l] = links_state.mass_shift[i_l, i_b]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l, i_b in ti.ndrange(n_geoms, _B):
        friction_ratio[i_b, i_l] = geoms_state.friction_ratio[i_l, i_b]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_set_state(
    qpos: ti.types.ndarray(),
    dofs_vel: ti.types.ndarray(),
    dofs_acc: ti.types.ndarray(),
    links_pos: ti.types.ndarray(),
    links_quat: ti.types.ndarray(),
    i_pos_shift: ti.types.ndarray(),
    mass_shift: ti.types.ndarray(),
    friction_ratio: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    links_state: array_class.LinksState,
    dofs_state: array_class.DofsState,
    geoms_state: array_class.GeomsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):

    n_qs = qpos.shape[1]
    n_dofs = dofs_vel.shape[1]
    n_links = links_pos.shape[1]
    n_geoms = friction_ratio.shape[1]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_q, i_b_ in ti.ndrange(n_qs, envs_idx.shape[0]):
        rigid_global_info.qpos[i_q, envs_idx[i_b_]] = qpos[envs_idx[i_b_], i_q]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_d, i_b_ in ti.ndrange(n_dofs, envs_idx.shape[0]):
        dofs_state.vel[i_d, envs_idx[i_b_]] = dofs_vel[envs_idx[i_b_], i_d]
        dofs_state.acc[i_d, envs_idx[i_b_]] = dofs_acc[envs_idx[i_b_], i_d]
        dofs_state.ctrl_force[i_d, envs_idx[i_b_]] = gs.ti_float(0.0)
        dofs_state.ctrl_mode[i_d, envs_idx[i_b_]] = gs.CTRL_MODE.FORCE

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l, i_b_ in ti.ndrange(n_links, envs_idx.shape[0]):
        for j in ti.static(range(3)):
            links_state.pos[i_l, envs_idx[i_b_]][j] = links_pos[envs_idx[i_b_], i_l, j]
            links_state.i_pos_shift[i_l, envs_idx[i_b_]][j] = i_pos_shift[envs_idx[i_b_], i_l, j]
        for j in ti.static(range(4)):
            links_state.quat[i_l, envs_idx[i_b_]][j] = links_quat[envs_idx[i_b_], i_l, j]
        links_state.mass_shift[i_l, envs_idx[i_b_]] = mass_shift[envs_idx[i_b_], i_l]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l, i_b_ in ti.ndrange(n_geoms, envs_idx.shape[0]):
        geoms_state.friction_ratio[i_l, envs_idx[i_b_]] = friction_ratio[envs_idx[i_b_], i_l]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_get_state_grad(
    qpos_grad: ti.types.ndarray(),
    vel_grad: ti.types.ndarray(),
    links_pos_grad: ti.types.ndarray(),
    links_quat_grad: ti.types.ndarray(),
    links_state: array_class.LinksState,
    dofs_state: array_class.DofsState,
    geoms_state: array_class.GeomsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    n_qs = qpos_grad.shape[1]
    n_dofs = vel_grad.shape[1]
    n_links = links_pos_grad.shape[1]
    _B = qpos_grad.shape[0]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_q, i_b in ti.ndrange(n_qs, _B):
        rigid_global_info.qpos.grad[i_q, i_b] += qpos_grad[i_b, i_q]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_d, i_b in ti.ndrange(n_dofs, _B):
        dofs_state.vel.grad[i_d, i_b] += vel_grad[i_b, i_d]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l, i_b in ti.ndrange(n_links, _B):
        for j in ti.static(range(3)):
            links_state.pos.grad[i_l, i_b][j] += links_pos_grad[i_b, i_l, j]
        for j in ti.static(range(4)):
            links_state.quat.grad[i_l, i_b][j] += links_quat_grad[i_b, i_l, j]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_set_links_pos(
    relative: ti.i32,
    pos: ti.types.ndarray(),
    links_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    links_info: array_class.LinksInfo,
    links_state: array_class.LinksState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l_, i_b_ in ti.ndrange(links_idx.shape[0], envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        i_l = links_idx[i_l_]
        I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l

        if links_info.parent_idx[I_l] == -1 and links_info.is_fixed[I_l]:
            for j in ti.static(range(3)):
                links_state.pos[i_l, i_b][j] = pos[i_b_, i_l_, j]
            if relative:
                for j in ti.static(range(3)):
                    links_state.pos[i_l, i_b][j] = links_state.pos[i_l, i_b][j] + links_info.pos[I_l][j]
        else:
            q_start = links_info.q_start[I_l]
            for j in ti.static(range(3)):
                rigid_global_info.qpos[q_start + j, i_b] = pos[i_b_, i_l_, j]
            if relative:
                for j in ti.static(range(3)):
                    rigid_global_info.qpos[q_start + j, i_b] = (
                        rigid_global_info.qpos[q_start + j, i_b] + rigid_global_info.qpos0[q_start + j, i_b]
                    )


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_set_links_pos_grad(
    relative: ti.i32,
    pos_grad: ti.types.ndarray(),
    links_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    links_info: array_class.LinksInfo,
    links_state: array_class.LinksState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l_, i_b_ in ti.ndrange(links_idx.shape[0], envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        i_l = links_idx[i_l_]
        I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l

        if links_info.parent_idx[I_l] == -1 and links_info.is_fixed[I_l]:
            for j in ti.static(range(3)):
                pos_grad[i_b_, i_l_, j] = links_state.pos.grad[i_l, i_b][j]
                links_state.pos.grad[i_l, i_b][j] = 0.0
        else:
            q_start = links_info.q_start[I_l]
            for j in ti.static(range(3)):
                pos_grad[i_b_, i_l_, j] = rigid_global_info.qpos.grad[q_start + j, i_b]
                rigid_global_info.qpos.grad[q_start + j, i_b] = 0.0


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_set_links_quat(
    relative: ti.i32,
    quat: ti.types.ndarray(),
    links_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    links_info: array_class.LinksInfo,
    links_state: array_class.LinksState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l_, i_b_ in ti.ndrange(links_idx.shape[0], envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        i_l = links_idx[i_l_]
        I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l

        if relative:
            quat_ = ti.Vector(
                [
                    quat[i_b_, i_l_, 0],
                    quat[i_b_, i_l_, 1],
                    quat[i_b_, i_l_, 2],
                    quat[i_b_, i_l_, 3],
                ],
                dt=gs.ti_float,
            )
            if links_info.parent_idx[I_l] == -1 and links_info.is_fixed[I_l]:
                links_state.quat[i_l, i_b] = gu.ti_transform_quat_by_quat(links_info.quat[I_l], quat_)
            else:
                q_start = links_info.q_start[I_l]
                quat0 = ti.Vector(
                    [
                        rigid_global_info.qpos0[q_start + 3, i_b],
                        rigid_global_info.qpos0[q_start + 4, i_b],
                        rigid_global_info.qpos0[q_start + 5, i_b],
                        rigid_global_info.qpos0[q_start + 6, i_b],
                    ],
                    dt=gs.ti_float,
                )
                quat_ = gu.ti_transform_quat_by_quat(quat0, quat_)
                for j in ti.static(range(4)):
                    rigid_global_info.qpos[q_start + j + 3, i_b] = quat_[j]
        else:
            if links_info.parent_idx[I_l] == -1 and links_info.is_fixed[I_l]:
                for j in ti.static(range(4)):
                    links_state.quat[i_l, i_b][j] = quat[i_b_, i_l_, j]
            else:
                q_start = links_info.q_start[I_l]
                for j in ti.static(range(4)):
                    rigid_global_info.qpos[q_start + j + 3, i_b] = quat[i_b_, i_l_, j]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_set_links_quat_grad(
    relative: ti.i32,
    quat_grad: ti.types.ndarray(),
    links_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    links_info: array_class.LinksInfo,
    links_state: array_class.LinksState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l_, i_b_ in ti.ndrange(links_idx.shape[0], envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        i_l = links_idx[i_l_]
        I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l

        if links_info.parent_idx[I_l] == -1 and links_info.is_fixed[I_l]:
            for j in ti.static(range(4)):
                quat_grad[i_b_, i_l_, j] = links_state.quat.grad[i_l, i_b][j]
                links_state.quat.grad[i_l, i_b][j] = 0.0
        else:
            q_start = links_info.q_start[I_l]
            for j in ti.static(range(4)):
                quat_grad[i_b_, i_l_, j] = rigid_global_info.qpos.grad[q_start + j + 3, i_b]
                rigid_global_info.qpos.grad[q_start + j + 3, i_b] = 0.0


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_set_links_mass_shift(
    mass: ti.types.ndarray(),
    links_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    links_state: array_class.LinksState,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l_, i_b_ in ti.ndrange(links_idx.shape[0], envs_idx.shape[0]):
        links_state.mass_shift[links_idx[i_l_], envs_idx[i_b_]] = mass[i_b_, i_l_]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_set_links_COM_shift(
    com: ti.types.ndarray(),
    links_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    links_state: array_class.LinksState,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l_, i_b_ in ti.ndrange(links_idx.shape[0], envs_idx.shape[0]):
        for j in ti.static(range(3)):
            links_state.i_pos_shift[links_idx[i_l_], envs_idx[i_b_]][j] = com[i_b_, i_l_, j]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_set_links_inertial_mass(
    inertial_mass: ti.types.ndarray(),
    links_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    links_info: array_class.LinksInfo,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    if ti.static(static_rigid_sim_config.batch_links_info):
        for i_l_, i_b_ in ti.ndrange(links_idx.shape[0], envs_idx.shape[0]):
            links_info.inertial_mass[links_idx[i_l_], envs_idx[i_b_]] = inertial_mass[i_b_, i_l_]
    else:
        for i_l_ in range(links_idx.shape[0]):
            links_info.inertial_mass[links_idx[i_l_]] = inertial_mass[i_l_]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_set_geoms_friction_ratio(
    friction_ratio: ti.types.ndarray(),
    geoms_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    geoms_state: array_class.GeomsState,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_g_, i_b_ in ti.ndrange(geoms_idx.shape[0], envs_idx.shape[0]):
        geoms_state.friction_ratio[geoms_idx[i_g_], envs_idx[i_b_]] = friction_ratio[i_b_, i_g_]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_set_qpos(
    qpos: ti.types.ndarray(),
    qs_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_q_, i_b_ in ti.ndrange(qs_idx.shape[0], envs_idx.shape[0]):
        rigid_global_info.qpos[qs_idx[i_q_], envs_idx[i_b_]] = qpos[i_b_, i_q_]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_set_global_sol_params(
    sol_params: ti.types.ndarray(),
    geoms_info: array_class.GeomsInfo,
    joints_info: array_class.JointsInfo,
    equalities_info: array_class.EqualitiesInfo,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    n_geoms = geoms_info.sol_params.shape[0]
    n_joints = joints_info.sol_params.shape[0]
    n_equalities = equalities_info.sol_params.shape[0]
    _B = equalities_info.sol_params.shape[1]

    for i_g in range(n_geoms):
        for j in ti.static(range(7)):
            geoms_info.sol_params[i_g][j] = sol_params[j]

    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_j, i_b in ti.ndrange(n_joints, _B):
        I_j = [i_j, i_b] if ti.static(static_rigid_sim_config.batch_joints_info) else i_j
        for j in ti.static(range(7)):
            joints_info.sol_params[I_j][j] = sol_params[j]

    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_eq, i_b in ti.ndrange(n_equalities, _B):
        for j in ti.static(range(7)):
            equalities_info.sol_params[i_eq, i_b][j] = sol_params[j]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_set_sol_params(
    constraint_type: ti.template(),
    sol_params: ti.types.ndarray(),
    inputs_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    geoms_info: array_class.GeomsInfo,
    joints_info: array_class.JointsInfo,
    equalities_info: array_class.EqualitiesInfo,
    static_rigid_sim_config: ti.template(),
):
    if ti.static(constraint_type == 0):  # geometries
        ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
        for i_g_ in range(inputs_idx.shape[0]):
            for j in ti.static(range(7)):
                geoms_info.sol_params[inputs_idx[i_g_]][j] = sol_params[i_g_, j]
    elif ti.static(constraint_type == 1):  # joints
        ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
        if ti.static(static_rigid_sim_config.batch_joints_info):
            for i_j_, i_b_ in ti.ndrange(inputs_idx.shape[0], envs_idx.shape[0]):
                for j in ti.static(range(7)):
                    joints_info.sol_params[inputs_idx[i_j_], envs_idx[i_b_]][j] = sol_params[i_b_, i_j_, j]
        else:
            for i_j_ in range(inputs_idx.shape[0]):
                for j in ti.static(range(7)):
                    joints_info.sol_params[inputs_idx[i_j_]][j] = sol_params[i_j_, j]
    else:  # equalities
        ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
        for i_eq_, i_b_ in ti.ndrange(inputs_idx.shape[0], envs_idx.shape[0]):
            for j in ti.static(range(7)):
                equalities_info.sol_params[inputs_idx[i_eq_], envs_idx[i_b_]][j] = sol_params[i_b_, i_eq_, j]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_set_dofs_kp(
    kp: ti.types.ndarray(),
    dofs_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    dofs_info: array_class.DofsInfo,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    if ti.static(static_rigid_sim_config.batch_dofs_info):
        for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
            dofs_info.kp[dofs_idx[i_d_], envs_idx[i_b_]] = kp[i_b_, i_d_]
    else:
        for i_d_ in range(dofs_idx.shape[0]):
            dofs_info.kp[dofs_idx[i_d_]] = kp[i_d_]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_set_dofs_kv(
    kv: ti.types.ndarray(),
    dofs_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    dofs_info: array_class.DofsInfo,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    if ti.static(static_rigid_sim_config.batch_dofs_info):
        for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
            dofs_info.kv[dofs_idx[i_d_], envs_idx[i_b_]] = kv[i_b_, i_d_]
    else:
        for i_d_ in range(dofs_idx.shape[0]):
            dofs_info.kv[dofs_idx[i_d_]] = kv[i_d_]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_set_dofs_force_range(
    lower: ti.types.ndarray(),
    upper: ti.types.ndarray(),
    dofs_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    dofs_info: array_class.DofsInfo,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    if ti.static(static_rigid_sim_config.batch_dofs_info):
        for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
            dofs_info.force_range[dofs_idx[i_d_], envs_idx[i_b_]][0] = lower[i_b_, i_d_]
            dofs_info.force_range[dofs_idx[i_d_], envs_idx[i_b_]][1] = upper[i_b_, i_d_]
    else:
        for i_d_ in range(dofs_idx.shape[0]):
            dofs_info.force_range[dofs_idx[i_d_]][0] = lower[i_d_]
            dofs_info.force_range[dofs_idx[i_d_]][1] = upper[i_d_]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_set_dofs_stiffness(
    stiffness: ti.types.ndarray(),
    dofs_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    dofs_info: array_class.DofsInfo,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    if ti.static(static_rigid_sim_config.batch_dofs_info):
        for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
            dofs_info.stiffness[dofs_idx[i_d_], envs_idx[i_b_]] = stiffness[i_b_, i_d_]
    else:
        for i_d_ in range(dofs_idx.shape[0]):
            dofs_info.stiffness[dofs_idx[i_d_]] = stiffness[i_d_]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_set_dofs_armature(
    armature: ti.types.ndarray(),
    dofs_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    dofs_info: array_class.DofsInfo,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    if ti.static(static_rigid_sim_config.batch_dofs_info):
        for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
            dofs_info.armature[dofs_idx[i_d_], envs_idx[i_b_]] = armature[i_b_, i_d_]
    else:
        for i_d_ in range(dofs_idx.shape[0]):
            dofs_info.armature[dofs_idx[i_d_]] = armature[i_d_]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_set_dofs_damping(
    damping: ti.types.ndarray(),
    dofs_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    dofs_info: array_class.DofsInfo,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    if ti.static(static_rigid_sim_config.batch_dofs_info):
        for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
            dofs_info.damping[dofs_idx[i_d_], envs_idx[i_b_]] = damping[i_b_, i_d_]
    else:
        for i_d_ in range(dofs_idx.shape[0]):
            dofs_info.damping[dofs_idx[i_d_]] = damping[i_d_]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_set_dofs_frictionloss(
    frictionloss: ti.types.ndarray(),
    dofs_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    dofs_info: array_class.DofsInfo,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    if ti.static(static_rigid_sim_config.batch_dofs_info):
        for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
            dofs_info.frictionloss[dofs_idx[i_d_], envs_idx[i_b_]] = frictionloss[i_b_, i_d_]
    else:
        for i_d_ in range(dofs_idx.shape[0]):
            dofs_info.frictionloss[dofs_idx[i_d_]] = frictionloss[i_d_]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_set_dofs_limit(
    lower: ti.types.ndarray(),
    upper: ti.types.ndarray(),
    dofs_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    dofs_info: array_class.DofsInfo,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    if ti.static(static_rigid_sim_config.batch_dofs_info):
        for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
            dofs_info.limit[dofs_idx[i_d_], envs_idx[i_b_]][0] = lower[i_b_, i_d_]
            dofs_info.limit[dofs_idx[i_d_], envs_idx[i_b_]][1] = upper[i_b_, i_d_]
    else:
        for i_d_ in range(dofs_idx.shape[0]):
            dofs_info.limit[dofs_idx[i_d_]][0] = lower[i_d_]
            dofs_info.limit[dofs_idx[i_d_]][1] = upper[i_d_]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_set_dofs_velocity(
    velocity: ti.types.ndarray(),
    dofs_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    dofs_state: array_class.DofsState,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
        dofs_state.vel[dofs_idx[i_d_], envs_idx[i_b_]] = velocity[i_b_, i_d_]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_set_dofs_velocity_grad(
    velocity_grad: ti.types.ndarray(),
    dofs_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    dofs_state: array_class.DofsState,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
        velocity_grad[i_b_, i_d_] = dofs_state.vel.grad[dofs_idx[i_d_], envs_idx[i_b_]]
        dofs_state.vel.grad[dofs_idx[i_d_], envs_idx[i_b_]] = 0.0


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_set_dofs_zero_velocity(
    dofs_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    dofs_state: array_class.DofsState,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
        dofs_state.vel[dofs_idx[i_d_], envs_idx[i_b_]] = 0.0


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_set_dofs_position(
    position: ti.types.ndarray(),
    dofs_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    dofs_state: array_class.DofsState,
    links_info: array_class.LinksInfo,
    joints_info: array_class.JointsInfo,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    n_entities = entities_info.link_start.shape[0]

    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
        dofs_state.pos[dofs_idx[i_d_], envs_idx[i_b_]] = position[i_b_, i_d_]

    # Note that qpos must be updated, as dofs_state.pos is not used for actual IK.
    # TODO: Make this more efficient by only taking care of releavant qs/dofs.
    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_e, i_b_ in ti.ndrange(n_entities, envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        for i_l in range(entities_info.link_start[i_e], entities_info.link_end[i_e]):
            I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l
            if links_info.n_dofs[I_l] == 0:
                continue

            dof_start = links_info.dof_start[I_l]
            q_start = links_info.q_start[I_l]

            i_j = links_info.joint_start[I_l]
            I_j = [i_j, i_b] if ti.static(static_rigid_sim_config.batch_joints_info) else i_j
            joint_type = joints_info.type[I_j]

            if joint_type == gs.JOINT_TYPE.FIXED:
                pass
            elif joint_type == gs.JOINT_TYPE.FREE:
                xyz = ti.Vector(
                    [
                        dofs_state.pos[0 + 3 + dof_start, i_b],
                        dofs_state.pos[1 + 3 + dof_start, i_b],
                        dofs_state.pos[2 + 3 + dof_start, i_b],
                    ],
                    dt=gs.ti_float,
                )
                quat = gu.ti_xyz_to_quat(xyz)

                for j in ti.static(range(3)):
                    rigid_global_info.qpos[j + q_start, i_b] = dofs_state.pos[j + dof_start, i_b]

                for j in ti.static(range(4)):
                    rigid_global_info.qpos[j + 3 + q_start, i_b] = quat[j]
            elif joint_type == gs.JOINT_TYPE.SPHERICAL:
                xyz = ti.Vector(
                    [
                        dofs_state.pos[0 + dof_start, i_b],
                        dofs_state.pos[1 + dof_start, i_b],
                        dofs_state.pos[2 + dof_start, i_b],
                    ],
                    dt=gs.ti_float,
                )
                quat = gu.ti_xyz_to_quat(xyz)
                for i_q_ in ti.static(range(4)):
                    i_q = q_start + i_q_
                    rigid_global_info.qpos[i_q, i_b] = quat[i_q_]
            else:  # (gs.JOINT_TYPE.REVOLUTE, gs.JOINT_TYPE.PRISMATIC)
                for i_d_ in range(links_info.dof_end[I_l] - dof_start):
                    i_q = q_start + i_d_
                    i_d = dof_start + i_d_
                    rigid_global_info.qpos[i_q, i_b] = rigid_global_info.qpos0[i_q, i_b] + dofs_state.pos[i_d, i_b]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_control_dofs_force(
    force: ti.types.ndarray(),
    dofs_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    dofs_state: array_class.DofsState,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
        dofs_state.ctrl_mode[dofs_idx[i_d_], envs_idx[i_b_]] = gs.CTRL_MODE.FORCE
        dofs_state.ctrl_force[dofs_idx[i_d_], envs_idx[i_b_]] = force[i_b_, i_d_]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_control_dofs_velocity(
    velocity: ti.types.ndarray(),
    dofs_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    dofs_state: array_class.DofsState,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
        i_d = dofs_idx[i_d_]
        i_b = envs_idx[i_b_]

        dofs_state.ctrl_mode[i_d, i_b] = gs.CTRL_MODE.VELOCITY
        dofs_state.ctrl_vel[i_d, i_b] = velocity[i_b_, i_d_]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_control_dofs_position(
    position: ti.types.ndarray(),
    dofs_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    dofs_state: array_class.DofsState,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
        i_d = dofs_idx[i_d_]
        i_b = envs_idx[i_b_]

        dofs_state.ctrl_mode[i_d, i_b] = gs.CTRL_MODE.POSITION
        dofs_state.ctrl_pos[i_d, i_b] = position[i_b_, i_d_]
        dofs_state.ctrl_vel[i_d, i_b] = 0.0


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_control_dofs_position_velocity(
    position: ti.types.ndarray(),
    velocity: ti.types.ndarray(),
    dofs_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    dofs_state: array_class.DofsState,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
        i_d = dofs_idx[i_d_]
        i_b = envs_idx[i_b_]

        dofs_state.ctrl_mode[i_d, i_b] = gs.CTRL_MODE.POSITION
        dofs_state.ctrl_pos[i_d, i_b] = position[i_b_, i_d_]
        dofs_state.ctrl_vel[i_d, i_b] = velocity[i_b_, i_d_]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_get_links_vel(
    tensor: ti.types.ndarray(),
    links_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    ref: ti.template(),
    links_state: array_class.LinksState,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_l_, i_b_ in ti.ndrange(links_idx.shape[0], envs_idx.shape[0]):
        # This is the velocity in world coordinates expressed at global com-position
        vel = links_state.cd_vel[links_idx[i_l_], envs_idx[i_b_]]  # entity's CoM

        # Translate to get the velocity expressed at a different position if necessary link-position
        if ti.static(ref == 1):  # link's CoM
            vel = vel + links_state.cd_ang[links_idx[i_l_], envs_idx[i_b_]].cross(
                links_state.i_pos[links_idx[i_l_], envs_idx[i_b_]]
            )
        if ti.static(ref == 2):  # link's origin
            vel = vel + links_state.cd_ang[links_idx[i_l_], envs_idx[i_b_]].cross(
                links_state.pos[links_idx[i_l_], envs_idx[i_b_]] - links_state.root_COM[links_idx[i_l_], envs_idx[i_b_]]
            )

        for j in ti.static(range(3)):
            tensor[i_b_, i_l_, j] = vel[j]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_get_links_acc(
    tensor: ti.types.ndarray(),
    links_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    links_state: array_class.LinksState,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_l_, i_b_ in ti.ndrange(links_idx.shape[0], envs_idx.shape[0]):
        i_l = links_idx[i_l_]
        i_b = envs_idx[i_b_]

        # Compute links spatial acceleration expressed at links origin in world coordinates
        cpos = links_state.pos[i_l, i_b] - links_state.root_COM[i_l, i_b]
        acc_ang = links_state.cacc_ang[i_l, i_b]
        acc_lin = links_state.cacc_lin[i_l, i_b] + acc_ang.cross(cpos)

        # Compute links classical linear acceleration expressed at links origin in world coordinates
        ang = links_state.cd_ang[i_l, i_b]
        vel = links_state.cd_vel[i_l, i_b] + ang.cross(cpos)
        acc_classic_lin = acc_lin + ang.cross(vel)

        for j in ti.static(range(3)):
            tensor[i_b_, i_l_, j] = acc_classic_lin[j]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_get_dofs_control_force(
    tensor: ti.types.ndarray(),
    dofs_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    static_rigid_sim_config: ti.template(),
):
    # we need to compute control force here because this won't be computed until the next actual simulation step
    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
        i_d = dofs_idx[i_d_]
        i_b = envs_idx[i_b_]
        I_d = [i_d, i_b] if ti.static(static_rigid_sim_config.batch_dofs_info) else i_d
        force = gs.ti_float(0.0)
        if dofs_state.ctrl_mode[i_d, i_b] == gs.CTRL_MODE.FORCE:
            force = dofs_state.ctrl_force[i_d, i_b]
        elif dofs_state.ctrl_mode[i_d, i_b] == gs.CTRL_MODE.VELOCITY:
            force = dofs_info.kv[I_d] * (dofs_state.ctrl_vel[i_d, i_b] - dofs_state.vel[i_d, i_b])
        elif dofs_state.ctrl_mode[i_d, i_b] == gs.CTRL_MODE.POSITION:
            force = dofs_info.kp[I_d] * (dofs_state.ctrl_pos[i_d, i_b] - dofs_state.pos[i_d, i_b]) + dofs_info.kv[
                I_d
            ] * (dofs_state.ctrl_vel[i_d, i_b] - dofs_state.vel[i_d, i_b])
        tensor[i_b_, i_d_] = ti.math.clamp(
            force,
            dofs_info.force_range[I_d][0],
            dofs_info.force_range[I_d][1],
        )


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_set_drone_rpm(
    n_propellers: ti.i32,
    propellers_link_idx: ti.types.ndarray(),
    propellers_rpm: ti.types.ndarray(),
    propellers_spin: ti.types.ndarray(),
    KF: ti.float32,
    KM: ti.float32,
    invert: ti.i32,
    links_state: array_class.LinksState,
):
    """
    Set the RPM of propellers of a drone entity.

    This method should only be called by drone entities.
    """
    _B = propellers_rpm.shape[0]
    for i_b in range(_B):
        for i_prop in range(n_propellers):
            i_l = propellers_link_idx[i_prop]

            force = ti.Vector([0.0, 0.0, propellers_rpm[i_b, i_prop] ** 2 * KF], dt=gs.ti_float)
            torque = ti.Vector(
                [0.0, 0.0, propellers_rpm[i_b, i_prop] ** 2 * KM * propellers_spin[i_prop]], dt=gs.ti_float
            )
            if invert:
                torque = -torque

            func_apply_link_external_force(force, i_l, i_b, 1, 1, links_state)
            func_apply_link_external_torque(torque, i_l, i_b, 1, 1, links_state)


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_update_drone_propeller_vgeoms(
    n_propellers: ti.i32,
    propellers_vgeom_idxs: ti.types.ndarray(),
    propellers_revs: ti.types.ndarray(),
    propellers_spin: ti.types.ndarray(),
    vgeoms_state: array_class.VGeomsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    """
    Update the angle of the vgeom in the propellers of a drone entity.
    """
    EPS = rigid_global_info.EPS[None]
    _B = propellers_revs.shape[1]
    for i_pp, i_b in ti.ndrange(n_propellers, _B):
        i_vg = propellers_vgeom_idxs[i_pp]
        rad = (
            propellers_revs[i_pp, i_b] * propellers_spin[i_pp] * rigid_global_info.substep_dt[None] * ti.math.pi / 30.0
        )
        vgeoms_state.quat[i_vg, i_b] = gu.ti_transform_quat_by_quat(
            gu.ti_rotvec_to_quat(ti.Vector([0.0, 0.0, rad], dt=gs.ti_float), EPS),
            vgeoms_state.quat[i_vg, i_b],
        )


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_set_geom_friction(geoms_idx: ti.i32, friction: ti.f32, geoms_info: array_class.GeomsInfo):
    geoms_info.friction[geoms_idx] = friction


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_set_geoms_friction(
    friction: ti.types.ndarray(),
    geoms_idx: ti.types.ndarray(),
    geoms_info: array_class.GeomsInfo,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=ti.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_g_ in ti.ndrange(geoms_idx.shape[0]):
        geoms_info.friction[geoms_idx[i_g_]] = friction[i_g_]


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_get_errno(errno: array_class.V_ANNOTATION) -> ti.i32:
    return errno[None]


@ti.func
def func_atomic_add_if_backward_2d(
    field: array_class.V_ANNOTATION, i: ti.i32, j: ti.i32, value, static_rigid_sim_config: ti.template()
):
    # Use (expensive) atomic add in backward for differentiability -- when there is race condition on the field to write,
    # use atomic add directly.
    # https://docs.taichi-lang.org/docs/differentiable_programming#global-data-access-rules
    if ti.static(static_rigid_sim_config.is_backward):
        field[i, j] += value
    else:
        field[i, j] = field[i, j] + value


@ti.func
def func_atomic_add_if_backward_3d(
    field: array_class.V_ANNOTATION, i: ti.i32, j: ti.i32, k: ti.i32, value, static_rigid_sim_config: ti.template()
):
    # Use (expensive) atomic add in backward for differentiability -- when there is race condition on the field to write,
    # use atomic add directly.
    # https://docs.taichi-lang.org/docs/differentiable_programming#global-data-access-rules
    if ti.static(static_rigid_sim_config.is_backward):
        field[i, j, k] += value
    else:
        field[i, j, k] = field[i, j, k] + value


@ti.func
def func_check_index_range(idx: ti.i32, min: ti.i32, max: ti.i32, cond: ti.template()):
    # Conditionally check if the index is in the range [min, max) to save computational cost
    return (idx >= min and idx < max) if ti.static(cond) else True
