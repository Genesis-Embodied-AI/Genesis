from typing import Literal, TYPE_CHECKING
from dataclasses import dataclass

import numpy as np
import torch
import numpy.typing as npt
import taichi as ti

import genesis as gs
from genesis.engine.entities.base_entity import Entity
from genesis.options.solvers import RigidOptions
import genesis.utils.geom as gu
from genesis.utils import linalg as lu
from genesis.utils.misc import ti_field_to_torch, DeprecationError, ALLOCATE_TENSOR_WARNING
from genesis.engine.entities import AvatarEntity, DroneEntity, RigidEntity
from genesis.engine.states.solvers import RigidSolverState
from genesis.styles import colors, formats
import genesis.utils.array_class as array_class

from ..base_solver import Solver
from .collider_decomp import Collider
from .constraint_solver_decomp import ConstraintSolver
from .constraint_solver_decomp_island import ConstraintSolverIsland
from ....utils.sdf_decomp import SDF

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.engine.simulator import Simulator


# minimum constraint impedance
IMP_MIN = 0.0001
# maximum constraint impedance
IMP_MAX = 0.9999

# Minimum ratio between simulation timestep `_substep_dt` and time constant of constraints
TIME_CONSTANT_SAFETY_FACTOR = 2.0


def _sanitize_sol_params(
    sol_params: npt.NDArray[np.float64], min_timeconst: float, default_timeconst: float | None = None
):
    assert sol_params.shape[-1] == 7
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


@ti.data_oriented
class RigidSolver(Solver):
    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    @ti.data_oriented
    class StaticRigidSimConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

        # # store static arguments here
        # para_level: int = 0
        # use_hibernation: bool = False
        # use_contact_island: bool = False
        # batch_links_info: bool = False
        # batch_dofs_info: bool = False
        # batch_joints_info: bool = False
        # enable_mujoco_compatibility: bool = False
        # enable_multi_contact: bool = True
        # enable_self_collision: bool = True
        # enable_adjacent_collision: bool = False
        # enable_collision: bool = False
        # box_box_detection: bool = False
        # integrator: gs.integrator = gs.integrator.implicitfast
        # sparse_solve: bool = False
        # solver_type: gs.constraint_solver = gs.constraint_solver.CG
        # # dynamic properties
        # substep_dt: float = 0.01
        # iterations: int = 10
        # tolerance: float = 1e-6
        # ls_iterations: int = 10
        # ls_tolerance: float = 1e-6

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

        if not options.use_gjk_collision and self._substep_dt < 0.002:
            gs.logger.warning(
                "Using a simulation timestep smaller than 2ms is not recommended for 'use_gjk_collision=False' as it "
                "could lead to numerically unstable collision detection."
            )

        self._options = options

        self._cur_step = -1

    def add_entity(self, idx, material, morph, surface, visualize_contact) -> Entity:
        if isinstance(material, gs.materials.Avatar):
            EntityClass = AvatarEntity
            if visualize_contact:
                gs.raise_exception("AvatarEntity does not support 'visualize_contact=True'.")
        else:
            if isinstance(morph, gs.morphs.Drone):
                EntityClass = DroneEntity
            else:
                EntityClass = RigidEntity

        if morph.is_free:
            verts_state_start = self.n_free_verts
        else:
            verts_state_start = self.n_fixed_verts

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
            verts_state_start=verts_state_start,
            face_start=self.n_faces,
            edge_start=self.n_edges,
            vgeom_start=self.n_vgeoms,
            vvert_start=self.n_vverts,
            vface_start=self.n_vfaces,
            visualize_contact=visualize_contact,
        )
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

        self.n_equalities_candidate = max(1, self.n_equalities + self._options.max_dynamic_constraints)

        self._static_rigid_sim_config = self.StaticRigidSimConfig(
            para_level=self.sim._para_level,
            use_hibernation=getattr(self, "_use_hibernation", False),
            use_contact_island=getattr(self, "_use_contact_island", False),
            batch_links_info=getattr(self._options, "batch_links_info", False),
            batch_dofs_info=getattr(self._options, "batch_dofs_info", False),
            batch_joints_info=getattr(self._options, "batch_joints_info", False),
            enable_mujoco_compatibility=getattr(self, "_enable_mujoco_compatibility", False),
            enable_multi_contact=getattr(self, "_enable_multi_contact", True),
            enable_self_collision=getattr(self, "_enable_self_collision", True),
            enable_adjacent_collision=getattr(self, "_enable_adjacent_collision", False),
            enable_collision=getattr(self, "_enable_collision", False),
            box_box_detection=getattr(self, "_box_box_detection", False),
            integrator=getattr(self, "_integrator", gs.integrator.implicitfast),
            sparse_solve=getattr(self._options, "sparse_solve", False),
            solver_type=getattr(self._options, "constraint_solver", gs.constraint_solver.CG),
            # dynamic properties
            substep_dt=self._substep_dt,
            iterations=getattr(self._options, "iterations", 10),
            tolerance=getattr(self._options, "tolerance", 1e-6),
            ls_iterations=getattr(self._options, "ls_iterations", 10),
            ls_tolerance=getattr(self._options, "ls_tolerance", 1e-6),
            n_equalities=self._n_equalities,
            n_equalities_candidate=self.n_equalities_candidate,
        )

        # when the migration is finished, we will remove the about two lines
        self._func_vel_at_point = func_vel_at_point
        self._func_apply_external_force = func_apply_external_force

        if self.is_active():

            self.data_manager = array_class.DataManager(self)

            self._rigid_global_info = self.data_manager.rigid_global_info
            if self._use_hibernation:
                self.n_awake_dofs = self._rigid_global_info.n_awake_dofs
                self.awake_dofs = self._rigid_global_info.awake_dofs
                self.n_awake_links = self._rigid_global_info.n_awake_links
                self.awake_links = self._rigid_global_info.awake_links
                self.n_awake_entities = self._rigid_global_info.data_manager.n_awake_entities
                self.awake_entities = self._rigid_global_info.data_manager.awake_entities

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

            # Compute state in neutral configuration at rest
            kernel_forward_kinematics_links_geoms(
                self._scene._envs_idx,
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
            self._init_invweight()
            kernel_init_meaninertia(
                rigid_global_info=self._rigid_global_info,
                entities_info=self.entities_info,
                static_rigid_sim_config=self._static_rigid_sim_config,
            )

    def _init_invweight(self):
        # Early return if no DoFs. This is essential to avoid segfault on CUDA.
        if self._n_dofs == 0:
            return

        # Compute mass matrix without any implicit damping terms
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
        mass_mat_D_inv = self._rigid_global_info.mass_mat_D_inv.to_numpy()[:, 0]
        mass_mat_L = self._rigid_global_info.mass_mat_L.to_numpy()[:, :, 0]
        offsets = self.links_state.i_pos.to_numpy()[:, 0]
        cdof_ang = self.dofs_state.cdof_ang.to_numpy()[:, 0]
        cdof_vel = self.dofs_state.cdof_vel.to_numpy()[:, 0]
        links_joint_start = self.links_info.joint_start.to_numpy()
        links_joint_end = self.links_info.joint_end.to_numpy()
        links_dof_end = self.links_info.dof_end.to_numpy()
        links_n_dofs = self.links_info.n_dofs.to_numpy()
        links_parent_idx = self.links_info.parent_idx.to_numpy()
        joints_type = self.joints_info.type.to_numpy()
        joints_dof_start = self.joints_info.dof_start.to_numpy()
        joints_n_dofs = self.joints_info.n_dofs.to_numpy()
        if self._options.batch_links_info:
            links_joint_start = links_joint_start[:, 0]
            links_joint_end = links_joint_end[:, 0]
            links_dof_end = links_dof_end[:, 0]
            links_n_dofs = links_n_dofs[:, 0]
            links_parent_idx = links_parent_idx[:, 0]
        if self._options.batch_joints_info:
            joints_type = joints_type[:, 0]
            joints_dof_start = joints_dof_start[:, 0]
            joints_n_dofs = joints_n_dofs[:, 0]

        # Compute the inverted mass matrix efficiently
        mass_mat_L_inv = np.eye(self.n_dofs_)
        for i in range(self.n_dofs_):
            for j in range(i):
                mass_mat_L_inv[i] -= mass_mat_L[i, j] * mass_mat_L_inv[j]
        mass_mat_inv = (mass_mat_L_inv * mass_mat_D_inv) @ mass_mat_L_inv.T

        # Compute links invweight
        links_invweight = np.zeros((self._n_links, 2), dtype=gs.np_float)
        for i_l in range(self._n_links):
            jacp = np.zeros((3, self._n_dofs))
            jacr = np.zeros((3, self._n_dofs))

            offset = offsets[i_l]

            j_l = i_l
            while j_l != -1:
                for i_d_ in range(links_n_dofs[j_l]):
                    i_d = links_dof_end[j_l] - i_d_ - 1
                    jacp[:, i_d] = cdof_vel[i_d] + np.cross(cdof_ang[i_d], offset)
                    jacr[:, i_d] = cdof_ang[i_d]
                j_l = links_parent_idx[j_l]

            jac = np.concatenate((jacp, jacr), axis=0)

            A = jac @ mass_mat_inv @ jac.T
            A_diag = np.diag(A)

            links_invweight[i_l, 0] = A_diag[:3].mean()
            links_invweight[i_l, 1] = A_diag[3:].mean()

        # Compute dofs invweight
        dofs_invweight = np.zeros((self._n_dofs,), dtype=gs.np_float)
        for i_l in range(self._n_links):
            for i_j in range(links_joint_start[i_l], links_joint_end[i_l]):
                joint_type = joints_type[i_j]
                if joint_type == gs.JOINT_TYPE.FIXED:
                    continue

                dof_start = joints_dof_start[i_j]
                n_dofs = joints_n_dofs[i_j]
                jac = np.zeros((n_dofs, self._n_dofs))
                for i_d_ in range(n_dofs):
                    jac[i_d_, dof_start + i_d_] = 1.0

                A = jac @ mass_mat_inv @ jac.T
                A_diag = np.diag(A)

                if joint_type == gs.JOINT_TYPE.FREE:
                    dofs_invweight[dof_start : (dof_start + 3)] = A_diag[:3].mean()
                    dofs_invweight[(dof_start + 3) : (dof_start + 6)] = A_diag[3:].mean()
                elif joint_type == gs.JOINT_TYPE.SPHERICAL:
                    dofs_invweight[dof_start : (dof_start + 3)] = A_diag[:3].mean()
                else:  # REVOLUTE or PRISMATIC
                    dofs_invweight[dof_start] = A_diag[0]

        # Update links and dofs invweight for values that are not already pre-computed
        kernel_init_invweight(
            links_invweight,
            dofs_invweight,
            links_info=self.links_info,
            dofs_info=self.dofs_info,
        )

    def _batch_shape(self, shape=None, first_dim=False, B=None):
        if B is None:
            B = self._B

        if shape is None:
            return (B,)
        elif type(shape) in [list, tuple]:
            return (B,) + shape if first_dim else shape + (B,)
        else:
            return (B, shape) if first_dim else (shape, B)

    def _init_mass_mat(self):
        self.mass_mat = self._rigid_global_info.mass_mat
        self.mass_mat_L = self._rigid_global_info.mass_mat_L
        self.mass_mat_D_inv = self._rigid_global_info.mass_mat_D_inv
        self._mass_mat_mask = self._rigid_global_info._mass_mat_mask
        self.meaninertia = self._rigid_global_info.meaninertia
        # self.mass_mat = ti.field(dtype=gs.ti_float, shape=self._batch_shape((self.n_dofs_, self.n_dofs_)))
        # self.mass_mat_L = ti.field(dtype=gs.ti_float, shape=self._batch_shape((self.n_dofs_, self.n_dofs_)))
        # self.mass_mat_D_inv = ti.field(dtype=gs.ti_float, shape=self._batch_shape((self.n_dofs_,)))

        # self._mass_mat_mask = ti.field(dtype=gs.ti_int, shape=self._batch_shape(self.n_entities_))
        self._rigid_global_info._mass_mat_mask.fill(1)

        # self.meaninertia = ti.field(dtype=gs.ti_float, shape=self._batch_shape())

        # tree structure information
        mass_parent_mask = np.zeros((self.n_dofs_, self.n_dofs_), dtype=gs.np_float)

        for i in range(self.n_links):
            j = i
            while j != -1:
                for i_d, j_d in ti.ndrange(
                    (self.links[i].dof_start, self.links[i].dof_end), (self.links[j].dof_start, self.links[j].dof_end)
                ):
                    mass_parent_mask[i_d, j_d] = 1.0
                j = self.links[j].parent_idx

        # self.mass_parent_mask = ti.field(dtype=gs.ti_float, shape=(self.n_dofs_, self.n_dofs_))

        self._rigid_global_info.mass_parent_mask.from_numpy(mass_parent_mask)

        # just in case
        self._rigid_global_info.mass_mat_L.fill(0)
        self._rigid_global_info.mass_mat_D_inv.fill(0)
        self._rigid_global_info.meaninertia.fill(0)

        # self._rigid_global_info.mass_mat = self.mass_mat
        # self._rigid_global_info.mass_mat_L = self.mass_mat_L
        # self._rigid_global_info.mass_mat_D_inv = self.mass_mat_D_inv
        # self._rigid_global_info._mass_mat_mask = self._mass_mat_mask
        # self._rigid_global_info.meaninertia = self.meaninertia
        # self._rigid_global_info.mass_parent_mask = self.mass_parent_mask
        # self._rigid_global_info.gravity = self._gravity

        gravity = np.tile(self.sim.gravity, (self._B, 1))
        self._rigid_global_info.gravity.from_numpy(gravity)

    def _init_dof_fields(self):
        # if self._use_hibernation:
        #     # we are going to move n_awake_dofs and awake_dofs to _rigid_global_info completely after migration.
        #     # But right now, other kernels are still using self.n_awake_dofs and self.awake_dofs
        #     # so we need to keep them in self for now.
        #     self.n_awake_dofs = self._rigid_global_info.n_awake_dofs
        #     self.awake_dofs = self._rigid_global_info.awake_dofs

        # struct_dof_info = ti.types.struct(
        #     stiffness=gs.ti_float,
        #     invweight=gs.ti_float,
        #     armature=gs.ti_float,
        #     damping=gs.ti_float,
        #     motion_ang=gs.ti_vec3,
        #     motion_vel=gs.ti_vec3,
        #     limit=gs.ti_vec2,
        #     dof_start=gs.ti_int,  # dof_start of its entity
        #     kp=gs.ti_float,
        #     kv=gs.ti_float,
        #     force_range=gs.ti_vec2,
        # )

        # struct_dof_state = ti.types.struct(
        #     force=gs.ti_float,
        #     qf_bias=gs.ti_float,
        #     qf_passive=gs.ti_float,
        #     qf_actuator=gs.ti_float,
        #     qf_applied=gs.ti_float,
        #     act_length=gs.ti_float,
        #     pos=gs.ti_float,
        #     vel=gs.ti_float,
        #     acc=gs.ti_float,
        #     acc_smooth=gs.ti_float,
        #     qf_smooth=gs.ti_float,
        #     qf_constraint=gs.ti_float,
        #     cdof_ang=gs.ti_vec3,
        #     cdof_vel=gs.ti_vec3,
        #     cdofvel_ang=gs.ti_vec3,
        #     cdofvel_vel=gs.ti_vec3,
        #     cdofd_ang=gs.ti_vec3,
        #     cdofd_vel=gs.ti_vec3,
        #     f_vel=gs.ti_vec3,
        #     f_ang=gs.ti_vec3,
        #     ctrl_force=gs.ti_float,
        #     ctrl_pos=gs.ti_float,
        #     ctrl_vel=gs.ti_float,
        #     ctrl_mode=gs.ti_int,
        #     hibernated=gs.ti_int,  # Flag for dofs that converge into a static state (hibernation)
        # )
        # dofs_info_shape = self._batch_shape(self.n_dofs_) if self._options.batch_dofs_info else self.n_dofs_
        # self.dofs_info = struct_dof_info.field(shape=dofs_info_shape, needs_grad=False, layout=ti.Layout.SOA)
        # self.dofs_state = struct_dof_state.field(
        #     shape=self._batch_shape(self.n_dofs_), needs_grad=False, layout=ti.Layout.SOA
        # )

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

        joints = self.joints
        if joints:
            # Make sure that the constraints parameters are valid
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

        self.qpos0 = self._rigid_global_info.qpos0
        if self.n_qs > 0:
            init_qpos = self._batch_array(self.init_qpos)
            self.qpos0.from_numpy(init_qpos)

        # Check if the initial configuration is out-of-bounds
        self.qpos = self._rigid_global_info.qpos
        is_init_qpos_out_of_bounds = False
        if self.n_qs > 0:
            init_qpos = self._batch_array(self.init_qpos)
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
                is_free=np.concatenate([np.full(geom.n_verts, geom.is_free) for geom in geoms], dtype=gs.np_int),
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
        self.geoms_info = self.data_manager.geoms_info
        self.geoms_state = self.data_manager.geoms_state
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
                geoms_is_convex=np.array([geom.is_convex for geom in geoms], dtype=gs.np_int),
                geoms_needs_coup=np.array([geom.needs_coup for geom in geoms], dtype=gs.np_int),
                geoms_contype=np.array([geom.contype for geom in geoms], dtype=np.int32),
                geoms_conaffinity=np.array([geom.conaffinity for geom in geoms], dtype=np.int32),
                geoms_coup_softness=np.array([geom.coup_softness for geom in geoms], dtype=gs.np_float),
                geoms_coup_friction=np.array([geom.coup_friction for geom in geoms], dtype=gs.np_float),
                geoms_coup_restitution=np.array([geom.coup_restitution for geom in geoms], dtype=gs.np_float),
                geoms_is_free=np.array([geom.is_free for geom in geoms], dtype=gs.np_int),
                geoms_is_decomp=np.array([geom.metadata.get("decomposed", False) for geom in geoms], dtype=gs.np_int),
                # taichi variables
                geoms_info=self.geoms_info,
                geoms_state=self.geoms_state,
                verts_info=self.verts_info,
                geoms_init_AABB=self.geoms_init_AABB,
                static_rigid_sim_config=self._static_rigid_sim_config,
            )

    def _init_vgeom_fields(self):
        self.vgeoms_info = self.data_manager.vgeoms_info
        self.vgeoms_state = self.data_manager.vgeoms_state
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
        # if self._use_hibernation:
        #     self.n_awake_entities = ti.field(dtype=gs.ti_int, shape=self._B)
        #     self.awake_entities = ti.field(dtype=gs.ti_int, shape=self._batch_shape(self.n_entities_))

        # struct_entity_info = ti.types.struct(
        #     dof_start=gs.ti_int,
        #     dof_end=gs.ti_int,
        #     n_dofs=gs.ti_int,
        #     link_start=gs.ti_int,
        #     link_end=gs.ti_int,
        #     n_links=gs.ti_int,
        #     geom_start=gs.ti_int,
        #     geom_end=gs.ti_int,
        #     n_geoms=gs.ti_int,
        #     gravity_compensation=gs.ti_float,
        # )

        # struct_entity_state = ti.types.struct(
        #     hibernated=gs.ti_int,
        # )

        # self.entities_info = struct_entity_info.field(shape=self.n_entities, needs_grad=False, layout=ti.Layout.SOA)
        # self.entities_state = struct_entity_state.field(
        #     shape=self._batch_shape(self.n_entities), needs_grad=False, layout=ti.Layout.SOA
        # )

        self.entities_info = self.data_manager.entities_info
        self.entities_state = self.data_manager.entities_state

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
            _sanitize_sol_params(
                equalities_sol_params,
                self._sol_min_timeconst,
                self._sol_default_timeconst,
            )

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
            links_idx = self.geoms_info.link_idx.to_numpy()[self.geoms_info.type.to_numpy() == gs.GEOM_TYPE.TERRAIN]
            entity = self._entities[self.links_info.entity_idx.to_numpy()[links_idx[0]]]

            scale = entity.terrain_scale
            rc = np.array(entity.terrain_hf.shape, dtype=gs.np_int)
            hf = entity.terrain_hf.astype(gs.np_float, copy=False) * scale[1]
            xyz_maxmin = np.array(
                [rc[0] * scale[0], rc[1] * scale[0], hf.max(), 0, 0, hf.min() - 1.0],
                dtype=gs.np_float,
            )

            self.terrain_hf = ti.field(dtype=gs.ti_float, shape=hf.shape)
            self.terrain_rc = ti.field(dtype=gs.ti_int, shape=2)
            self.terrain_scale = ti.field(dtype=gs.ti_float, shape=2)
            self.terrain_xyz_maxmin = ti.field(dtype=gs.ti_float, shape=6)

            self.terrain_hf.from_numpy(hf)
            self.terrain_rc.from_numpy(rc)
            self.terrain_scale.from_numpy(scale)
            self.terrain_xyz_maxmin.from_numpy(xyz_maxmin)

    def _init_constraint_solver(self):
        if self._use_contact_island:
            self.constraint_solver = ConstraintSolverIsland(self)
        else:
            self.constraint_solver = ConstraintSolver(self)

    def substep(self):
        # from genesis.utils.tools import create_timer

        # timer = create_timer("rigid", level=1, ti_sync=True, skip_first_call=True)
        kernel_step_1(
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
        # timer.stamp("kernel_step_1")
        self._func_constraint_force()
        # timer.stamp("constraint_force")
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
        )
        # timer.stamp("kernel_step_2")

    def _kernel_detect_collision(self):
        self.collider.clear()
        self.collider.detection()

    def detect_collision(self, env_idx=0):
        # TODO: support batching
        self._kernel_detect_collision()
        n_collision = self.collider._collider_state.n_contacts.to_numpy()[env_idx]
        collision_pairs = np.empty((n_collision, 2), dtype=np.int32)
        collision_pairs[:, 0] = self.collider._collider_state.contact_data.geom_a.to_numpy()[:n_collision, env_idx]
        collision_pairs[:, 1] = self.collider._collider_state.contact_data.geom_b.to_numpy()[:n_collision, env_idx]
        return collision_pairs

    def _func_constraint_force(self):
        # from genesis.utils.tools import create_timer

        # timer = create_timer(name="constraint_force", level=2, ti_sync=True, skip_first_call=True)
        self._func_constraint_clear()
        # timer.stamp("constraint_solver.clear")

        if self._enable_collision:
            self.collider.detection()
            # timer.stamp("detection")

        if not self._disable_constraint:
            self.constraint_solver.handle_constraints()
        # timer.stamp("constraint_solver.handle_constraints")

    def _func_constraint_clear(self):
        self.constraint_solver.constraint_state.n_constraints.fill(0)
        self.constraint_solver.constraint_state.n_constraints_equality.fill(0)
        self.constraint_solver.constraint_state.n_constraints_frictionloss.fill(0)
        self.collider._collider_state.n_contacts.fill(0)

    def _func_forward_dynamics(self):
        kernel_forward_dynamics(
            links_state=self.links_state,
            links_info=self.links_info,
            dofs_state=self.dofs_state,
            dofs_info=self.dofs_info,
            joints_info=self.joints_info,
            entities_info=self.entities_info,
            rigid_global_info=self._rigid_global_info,
            static_rigid_sim_config=self._static_rigid_sim_config,
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

    def _func_update_geoms(self, envs_idx):
        kernel_update_geoms(
            envs_idx,
            entities_info=self.entities_info,
            geoms_info=self.geoms_info,
            geoms_state=self.geoms_state,
            links_state=self.links_state,
            rigid_global_info=self._rigid_global_info,
            static_rigid_sim_config=self._static_rigid_sim_config,
        )

    # TODO: we need to use a kernel to clear the constraints if hibernation is enabled
    # right now, a python-scope function is more convenient since .fill(0) only works on python scope for ndarray
    # @ti.kernel
    # def _func_constraint_clear(
    #     self_unused,
    #     links_state: array_class.LinksState,
    #     links_info: array_class.LinksInfo,
    #     collider_state: array_class.ColliderState,
    #     static_rigid_sim_config: ti.template(),
    # ):

    #     if static_rigid_sim_config.enable_collision:
    #         if ti.static(static_rigid_sim_config.use_hibernation):
    #             collider_state.n_contacts_hibernated.fill(0)
    #             _B = collider_state.n_contacts.shape[0]
    #             ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    #             for i_b in range(_B):
    #                 # Advect hibernated contacts
    #                 for i_c in range(collider_state.n_contacts[i_b]):
    #                     i_la = collider_state.contact_data[i_c, i_b].link_a
    #                     i_lb = collider_state.contact_data[i_c, i_b].link_b
    #                     I_la = [i_la, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_la
    #                     I_lb = [i_lb, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_lb

    #                     # Pair of hibernated-fixed links -> hibernated contact
    #                     # TODO: we should also include hibernated-hibernated links and wake up the whole contact island
    #                     # once a new collision is detected
    #                     if (links_state.hibernated[i_la, i_b] and links_info.is_fixed[I_lb]) or (
    #                         links_state.hibernated[i_lb, i_b] and links_info.is_fixed[I_la]
    #                     ):
    #                         i_c_hibernated = collider_state.n_contacts_hibernated[i_b]
    #                         if i_c != i_c_hibernated:
    #                             collider_state.contact_data[i_c_hibernated, i_b] = collider_state.contact_data[i_c, i_b]
    #                         collider_state.n_contacts_hibernated[i_b] = i_c_hibernated + 1

    #                 collider_state.n_contacts[i_b] = collider_state.n_contacts_hibernated[i_b]
    #         else:
    #             collider_state.n_contacts.fill(0)

    def _batch_array(self, arr, first_dim=False):
        if first_dim:
            return np.tile(np.expand_dims(arr, 0), self._batch_shape(arr.ndim * (1,), True))
        else:
            return np.tile(np.expand_dims(arr, -1), self._batch_shape(arr.ndim * (1,)))

    def _process_dim(self, tensor, envs_idx=None):
        if self.n_envs == 0:
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            else:
                gs.raise_exception(
                    f"Invalid input shape: {tensor.shape}. Expecting a 1D tensor for non-parallelized scene."
                )
        else:
            if tensor.ndim == 2:
                if envs_idx is not None:
                    if len(tensor) != len(envs_idx):
                        gs.raise_exception(
                            f"Invalid input shape: {tensor.shape}. 1st dimension of input does not match `envs_idx`."
                        )
                else:
                    if len(tensor) != self.n_envs:
                        gs.raise_exception(
                            f"Invalid input shape: {tensor.shape}. 1st dimension of input does not match `scene.n_envs`."
                        )
            else:
                gs.raise_exception(
                    f"Invalid input shape: {tensor.shape}. Expecting a 2D tensor for scene with parallelized envs."
                )
        return tensor

    def apply_links_external_force(
        self,
        force,
        links_idx=None,
        envs_idx=None,
        *,
        ref: Literal["link_origin", "link_com", "root_com"] = "link_origin",
        local: bool = False,
        unsafe: bool = False,
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
        force, links_idx, envs_idx = self._sanitize_2D_io_variables(
            force, links_idx, self.n_links, 3, envs_idx, idx_name="links_idx", skip_allocation=True, unsafe=unsafe
        )
        if self.n_envs == 0:
            force = force.unsqueeze(0)

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
        unsafe=False,
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
        torque, links_idx, envs_idx = self._sanitize_2D_io_variables(
            torque, links_idx, self.n_links, 3, envs_idx, idx_name="links_idx", skip_allocation=True, unsafe=unsafe
        )
        if self.n_envs == 0:
            torque = torque.unsqueeze(0)

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
        if self.is_active():
            self.substep()

    def substep_pre_coupling_grad(self, f):
        pass

    def substep_post_coupling(self, f):
        pass

    def substep_post_coupling_grad(self, f):
        pass

    def add_grad_from_state(self, state):
        pass

    def collect_output_grads(self):
        """
        Collect gradients from downstream queried states.
        """
        pass

    def reset_grad(self):
        pass

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

    def get_state(self, f):
        if self.is_active():
            state = RigidSolverState(self._scene)

            # qpos: ti.types.ndarray(),
            # vel: ti.types.ndarray(),
            # links_pos: ti.types.ndarray(),
            # links_quat: ti.types.ndarray(),
            # i_pos_shift: ti.types.ndarray(),
            # mass_shift: ti.types.ndarray(),
            # friction_ratio: ti.types.ndarray(),
            # links_state: array_class.LinksState,
            # dofs_state: array_class.DofsState,
            # geoms_state: array_class.GeomsState,
            # rigid_global_info: array_class.RigidGlobalInfo,
            # static_rigid_sim_config: ti.template(),

            kernel_get_state(
                qpos=state.qpos,
                vel=state.dofs_vel,
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
        else:
            state = None
        return state

    def set_state(self, f, state, envs_idx=None):
        if self.is_active():
            envs_idx = self._scene._sanitize_envs_idx(envs_idx)
            kernel_set_state(
                qpos=state.qpos,
                dofs_vel=state.dofs_vel,
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
            self.collider.reset(envs_idx)
            self.collider.clear(envs_idx)
            if self.constraint_solver is not None:
                self.constraint_solver.reset(envs_idx)
                self.constraint_solver.clear(envs_idx)
            self._cur_step = -1

    def process_input(self, in_backward=False):
        pass

    def process_input_grad(self):
        pass

    def save_ckpt(self, ckpt_name):
        pass

    def load_ckpt(self, ckpt_name):
        pass

    def is_active(self):
        return self.n_links > 0

    # ------------------------------------------------------------------------------------
    # ------------------------------------ control ---------------------------------------
    # ------------------------------------------------------------------------------------

    def _sanitize_1D_io_variables(
        self,
        tensor,
        inputs_idx,
        input_size,
        envs_idx,
        batched=True,
        idx_name="dofs_idx",
        *,
        skip_allocation=False,
        unsafe=False,
    ):
        # Handling default arguments
        if batched:
            envs_idx = self._scene._sanitize_envs_idx(envs_idx, unsafe=unsafe)
        else:
            envs_idx = torch.empty((0,), dtype=gs.tc_int, device=gs.device)

        if inputs_idx is None:
            inputs_idx = range(input_size)
        elif isinstance(inputs_idx, slice):
            inputs_idx = range(
                inputs_idx.start or 0,
                inputs_idx.stop if inputs_idx.stop is not None else input_size,
                inputs_idx.step or 1,
            )
        elif isinstance(inputs_idx, (int, np.integer)):
            inputs_idx = [inputs_idx]

        is_preallocated = tensor is not None
        if not is_preallocated and not skip_allocation:
            if batched and self.n_envs > 0:
                shape = self._batch_shape(len(inputs_idx), True, B=len(envs_idx))
            else:
                shape = (len(inputs_idx),)
            tensor = torch.empty(shape, dtype=gs.tc_float, device=gs.device)

        # Early return if unsafe
        if unsafe:
            return tensor, inputs_idx, envs_idx

        # Perform a bunch of sanity checks
        _inputs_idx = torch.as_tensor(inputs_idx, dtype=gs.tc_int, device=gs.device).contiguous()
        if _inputs_idx is not inputs_idx:
            gs.logger.debug(ALLOCATE_TENSOR_WARNING)
        _inputs_idx = torch.atleast_1d(_inputs_idx)
        if _inputs_idx.ndim != 1:
            gs.raise_exception(f"Expecting 1D tensor for `{idx_name}`.")
        if len(inputs_idx):
            inputs_start, inputs_end = min(inputs_idx), max(inputs_idx)
            if inputs_start < 0 or input_size <= inputs_end:
                gs.raise_exception(f"`{idx_name}` is out-of-range.")

        if is_preallocated:
            _tensor = torch.as_tensor(tensor, dtype=gs.tc_float, device=gs.device).contiguous()
            if _tensor is not tensor:
                gs.logger.debug(ALLOCATE_TENSOR_WARNING)
            tensor = _tensor.unsqueeze(0) if batched and self.n_envs and _tensor.ndim == 1 else _tensor

            if tensor.shape[-1] != len(inputs_idx):
                gs.raise_exception(f"Last dimension of the input tensor does not match length of `{idx_name}`.")

            if batched:
                if self.n_envs == 0:
                    if tensor.ndim != 1:
                        gs.raise_exception(
                            f"Invalid input shape: {tensor.shape}. Expecting a 1D tensor for non-parallelized scene."
                        )
                else:
                    if tensor.ndim == 2:
                        if tensor.shape[0] != len(envs_idx):
                            gs.raise_exception(
                                f"Invalid input shape: {tensor.shape}. First dimension of the input tensor does not match "
                                f"length ({len(envs_idx)}) of `envs_idx` (or `scene.n_envs` if `envs_idx` is None)."
                            )
                    else:
                        gs.raise_exception(
                            f"Invalid input shape: {tensor.shape}. Expecting a 2D tensor for scene with parallelized envs."
                        )
            else:
                if tensor.ndim != 1:
                    gs.raise_exception("Expecting 1D output tensor.")
        return tensor, _inputs_idx, envs_idx

    def _sanitize_2D_io_variables(
        self,
        tensor,
        inputs_idx,
        input_size,
        vec_size,
        envs_idx=None,
        batched=True,
        idx_name="links_idx",
        *,
        skip_allocation=False,
        unsafe=False,
    ):
        # Handling default arguments
        if batched:
            envs_idx = self._scene._sanitize_envs_idx(envs_idx, unsafe=unsafe)
        else:
            envs_idx = torch.empty((), dtype=gs.tc_int, device=gs.device)

        if inputs_idx is None:
            inputs_idx = range(input_size)
        elif isinstance(inputs_idx, slice):
            inputs_idx = range(
                inputs_idx.start or 0,
                inputs_idx.stop if inputs_idx.stop is not None else input_size,
                inputs_idx.step or 1,
            )
        elif isinstance(inputs_idx, (int, np.integer)):
            inputs_idx = [inputs_idx]

        is_preallocated = tensor is not None
        if not is_preallocated and not skip_allocation:
            if batched and self.n_envs > 0:
                shape = self._batch_shape((len(inputs_idx), vec_size), True, B=len(envs_idx))
            else:
                shape = (len(inputs_idx), vec_size)
            tensor = torch.empty(shape, dtype=gs.tc_float, device=gs.device)

        # Early return if unsafe
        if unsafe:
            return tensor, inputs_idx, envs_idx

        # Perform a bunch of sanity checks
        _inputs_idx = torch.as_tensor(inputs_idx, dtype=gs.tc_int, device=gs.device).contiguous()
        if _inputs_idx is not inputs_idx:
            gs.logger.debug(ALLOCATE_TENSOR_WARNING)
        _inputs_idx = torch.atleast_1d(_inputs_idx)
        if _inputs_idx.ndim != 1:
            gs.raise_exception(f"Expecting 1D tensor for `{idx_name}`.")
        inputs_start, inputs_end = min(inputs_idx), max(inputs_idx)
        if inputs_start < 0 or input_size <= inputs_end:
            gs.raise_exception(f"`{idx_name}` is out-of-range.")

        if is_preallocated:
            _tensor = torch.as_tensor(tensor, dtype=gs.tc_float, device=gs.device).contiguous()
            if _tensor is not tensor:
                gs.logger.debug(ALLOCATE_TENSOR_WARNING)
            tensor = _tensor.unsqueeze(0) if batched and self.n_envs and _tensor.ndim == 2 else _tensor

            if tensor.shape[-2] != len(inputs_idx):
                gs.raise_exception(f"Second last dimension of the input tensor does not match length of `{idx_name}`.")
            if tensor.shape[-1] != vec_size:
                gs.raise_exception(f"Last dimension of the input tensor must be {vec_size}.")

            if batched:
                if self.n_envs == 0:
                    if tensor.ndim != 2:
                        gs.raise_exception(
                            f"Invalid input shape: {tensor.shape}. Expecting a 2D tensor for non-parallelized scene."
                        )

                else:
                    if tensor.ndim == 3:
                        if tensor.shape[0] != len(envs_idx):
                            gs.raise_exception(
                                f"Invalid input shape: {tensor.shape}. First dimension of the input tensor does not match "
                                "length of `envs_idx` (or `scene.n_envs` if `envs_idx` is None)."
                            )
                    else:
                        gs.raise_exception(
                            f"Invalid input shape: {tensor.shape}. Expecting a 3D tensor for scene with parallelized envs."
                        )
            else:
                if tensor.ndim != 2:
                    gs.raise_exception("Expecting 2D input tensor.")
        return tensor, _inputs_idx, envs_idx

    def _get_qs_idx(self, qs_idx_local=None):
        return self._get_qs_idx_local(qs_idx_local) + self._q_start

    def set_links_pos(self, pos, links_idx=None, envs_idx=None, *, skip_forward=False, unsafe=False):
        raise DeprecationError("This method has been removed. Please use 'set_base_links_pos' instead.")

    def set_base_links_pos(
        self, pos, links_idx=None, envs_idx=None, *, relative=False, skip_forward=False, unsafe=False
    ):
        if links_idx is None:
            links_idx = self._base_links_idx
        pos, links_idx, envs_idx = self._sanitize_2D_io_variables(
            pos, links_idx, self.n_links, 3, envs_idx, idx_name="links_idx", skip_allocation=True, unsafe=unsafe
        )
        if self.n_envs == 0:
            pos = pos.unsqueeze(0)
        if not unsafe and not torch.isin(links_idx, self._base_links_idx).all():
            gs.raise_exception("`links_idx` contains at least one link that is not a base link.")
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

    def set_links_quat(self, quat, links_idx=None, envs_idx=None, *, skip_forward=False, unsafe=False):
        raise DeprecationError("This method has been removed. Please use 'set_base_links_quat' instead.")

    def set_base_links_quat(
        self, quat, links_idx=None, envs_idx=None, *, relative=False, skip_forward=False, unsafe=False
    ):
        if links_idx is None:
            links_idx = self._base_links_idx
        quat, links_idx, envs_idx = self._sanitize_2D_io_variables(
            quat, links_idx, self.n_links, 4, envs_idx, idx_name="links_idx", skip_allocation=True, unsafe=unsafe
        )
        if self.n_envs == 0:
            quat = quat.unsqueeze(0)
        if not unsafe and not torch.isin(links_idx, self._base_links_idx).all():
            gs.raise_exception("`links_idx` contains at least one link that is not a base link.")
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

    def set_links_mass_shift(self, mass, links_idx=None, envs_idx=None, *, unsafe=False):
        mass, links_idx, envs_idx = self._sanitize_1D_io_variables(
            mass, links_idx, self.n_links, envs_idx, idx_name="links_idx", skip_allocation=True, unsafe=unsafe
        )
        if self.n_envs == 0:
            mass = mass.unsqueeze(0)
        kernel_set_links_mass_shift(
            mass,
            links_idx,
            envs_idx,
            links_state=self.links_state,
            static_rigid_sim_config=self._static_rigid_sim_config,
        )

    def set_links_COM_shift(self, com, links_idx=None, envs_idx=None, *, unsafe=False):
        com, links_idx, envs_idx = self._sanitize_2D_io_variables(
            com, links_idx, self.n_links, 3, envs_idx, idx_name="links_idx", skip_allocation=True, unsafe=unsafe
        )
        if self.n_envs == 0:
            com = com.unsqueeze(0)
        kernel_set_links_COM_shift(com, links_idx, envs_idx, self.links_state, self._static_rigid_sim_config)

    def set_links_inertial_mass(self, mass, links_idx=None, envs_idx=None, *, unsafe=False):
        _, links_idx, envs_idx = self._sanitize_1D_io_variables(
            mass,
            links_idx,
            self.n_links,
            envs_idx,
            batched=self._options.batch_links_info,
            idx_name="links_idx",
            skip_allocation=True,
            unsafe=unsafe,
        )
        if self.n_envs == 0 and self._options.batch_links_info:
            mass = mass.unsqueeze(0)
        kernel_set_links_inertial_mass(mass, links_idx, envs_idx, self.links_info, self._static_rigid_sim_config)

    def set_links_invweight(self, invweight, links_idx=None, envs_idx=None, *, unsafe=False):
        _, links_idx, envs_idx = self._sanitize_2D_io_variables(
            invweight,
            links_idx,
            self.n_links,
            2,
            envs_idx,
            batched=self._options.batch_links_info,
            idx_name="links_idx",
            skip_allocation=True,
            unsafe=unsafe,
        )
        if self.n_envs == 0 and self._options.batch_links_info:
            invweight = invweight.unsqueeze(0)
        kernel_set_links_invweight(invweight, links_idx, envs_idx, self.links_info, self._static_rigid_sim_config)

    def set_geoms_friction_ratio(self, friction_ratio, geoms_idx=None, envs_idx=None, *, unsafe=False):
        friction_ratio, geoms_idx, envs_idx = self._sanitize_1D_io_variables(
            friction_ratio, geoms_idx, self.n_geoms, envs_idx, idx_name="geoms_idx", skip_allocation=True, unsafe=unsafe
        )
        if self.n_envs == 0:
            friction_ratio = friction_ratio.unsqueeze(0)
        kernel_set_geoms_friction_ratio(
            friction_ratio, geoms_idx, envs_idx, self.geoms_state, self._static_rigid_sim_config
        )

    def set_qpos(self, qpos, qs_idx=None, envs_idx=None, *, skip_forward=False, unsafe=False):
        qpos, qs_idx, envs_idx = self._sanitize_1D_io_variables(
            qpos, qs_idx, self.n_qs, envs_idx, idx_name="qs_idx", skip_allocation=True, unsafe=unsafe
        )
        if self.n_envs == 0:
            qpos = qpos.unsqueeze(0)
        kernel_set_qpos(qpos, qs_idx, envs_idx, self._rigid_global_info, self._static_rigid_sim_config)
        self.collider.reset(envs_idx)
        self.collider.clear(envs_idx)
        if self.constraint_solver is not None:
            self.constraint_solver.reset(envs_idx)
            self.constraint_solver.clear(envs_idx)
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

    def set_global_sol_params(self, sol_params, *, unsafe=False):
        """
        Set constraint solver parameters.

        Reference: https://mujoco.readthedocs.io/en/latest/modeling.html#solver-parameters

        Parameters
        ----------
        sol_params: Tuple[float] | List[float] | np.ndarray | torch.tensor
            array of length 7 in which each element corresponds to
            (timeconst, dampratio, dmin, dmax, width, mid, power)
        """
        # Sanitize input arguments
        if not unsafe:
            _sol_params = torch.as_tensor(sol_params, dtype=gs.tc_float, device=gs.device).contiguous()
            if _sol_params is not sol_params:
                gs.logger.debug(ALLOCATE_TENSOR_WARNING)
            sol_params = _sol_params

        # Make sure that the constraints parameters are within range
        _sanitize_sol_params(sol_params, self._sol_min_timeconst)

        kernel_set_global_sol_params(
            sol_params, self.geoms_info, self.joints_info, self.equalities_info, self._static_rigid_sim_config
        )

    def set_sol_params(self, sol_params, geoms_idx=None, envs_idx=None, *, joints_idx=None, eqs_idx=None, unsafe=False):
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
            raise gs.raise_exception("Cannot set more than one constraint type at once.")

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
        sol_params_, inputs_idx, envs_idx = self._sanitize_2D_io_variables(
            sol_params,
            inputs_idx,
            inputs_length,
            7,
            envs_idx,
            batched=batched,
            idx_name=idx_name,
            skip_allocation=True,
            unsafe=unsafe,
        )

        # Make sure that the constraints parameters are within range
        sol_params = sol_params_.clone() if sol_params_ is sol_params else sol_params_
        _sanitize_sol_params(sol_params, self._sol_min_timeconst)

        if batched and self.n_envs == 0:
            sol_params = sol_params.unsqueeze(0)
        kernel_set_sol_params(
            constraint_type,
            sol_params,
            inputs_idx,
            envs_idx,
            geoms_info=self.geoms_info,
            joints_info=self.joints_info,
            equalities_info=self.equalities_info,
            static_rigid_sim_config=self._static_rigid_sim_config,
        )

    def _set_dofs_info(self, tensor_list, dofs_idx, name, envs_idx=None, *, unsafe=False):
        tensor_list = list(tensor_list)
        for i, tensor in enumerate(tensor_list):
            tensor_list[i], dofs_idx, envs_idx = self._sanitize_1D_io_variables(
                tensor,
                dofs_idx,
                self.n_dofs,
                envs_idx,
                batched=self._options.batch_dofs_info,
                skip_allocation=True,
                unsafe=unsafe,
            )
        if name == "kp":
            kernel_set_dofs_kp(tensor_list[0], dofs_idx, envs_idx, self.dofs_info, self._static_rigid_sim_config)
        elif name == "kv":
            kernel_set_dofs_kv(tensor_list[0], dofs_idx, envs_idx, self.dofs_info, self._static_rigid_sim_config)
        elif name == "force_range":
            kernel_set_dofs_force_range(
                tensor_list[0], tensor_list[1], dofs_idx, envs_idx, self.dofs_info, self._static_rigid_sim_config
            )
        elif name == "stiffness":
            kernel_set_dofs_stiffness(tensor_list[0], dofs_idx, envs_idx, self.dofs_info, self._static_rigid_sim_config)
        elif name == "invweight":
            kernel_set_dofs_invweight(tensor_list[0], dofs_idx, envs_idx, self.dofs_info, self._static_rigid_sim_config)
        elif name == "armature":
            kernel_set_dofs_armature(tensor_list[0], dofs_idx, envs_idx, self.dofs_info, self._static_rigid_sim_config)
        elif name == "damping":
            kernel_set_dofs_damping(tensor_list[0], dofs_idx, envs_idx, self.dofs_info, self._static_rigid_sim_config)
        elif name == "limit":
            kernel_set_dofs_limit(
                tensor_list[0], tensor_list[1], dofs_idx, envs_idx, self.dofs_info, self._static_rigid_sim_config
            )
        else:
            gs.raise_exception(f"Invalid `name` {name}.")

    def set_dofs_kp(self, kp, dofs_idx=None, envs_idx=None, *, unsafe=False):
        self._set_dofs_info([kp], dofs_idx, "kp", envs_idx, unsafe=unsafe)

    def set_dofs_kv(self, kv, dofs_idx=None, envs_idx=None, *, unsafe=False):
        self._set_dofs_info([kv], dofs_idx, "kv", envs_idx, unsafe=unsafe)

    def set_dofs_force_range(self, lower, upper, dofs_idx=None, envs_idx=None, *, unsafe=False):
        self._set_dofs_info([lower, upper], dofs_idx, "force_range", envs_idx, unsafe=unsafe)

    def set_dofs_stiffness(self, stiffness, dofs_idx=None, envs_idx=None, *, unsafe=False):
        self._set_dofs_info([stiffness], dofs_idx, "stiffness", envs_idx, unsafe=unsafe)

    def set_dofs_invweight(self, invweight, dofs_idx=None, envs_idx=None, *, unsafe=False):
        self._set_dofs_info([invweight], dofs_idx, "invweight", envs_idx, unsafe=unsafe)

    def set_dofs_armature(self, armature, dofs_idx=None, envs_idx=None, *, unsafe=False):
        self._set_dofs_info([armature], dofs_idx, "armature", envs_idx, unsafe=unsafe)

    def set_dofs_damping(self, damping, dofs_idx=None, envs_idx=None, *, unsafe=False):
        self._set_dofs_info([damping], dofs_idx, "damping", envs_idx)

    def set_dofs_limit(self, lower, upper, dofs_idx=None, envs_idx=None, *, unsafe=False):
        self._set_dofs_info([lower, upper], dofs_idx, "limit", envs_idx, unsafe=unsafe)

    def set_dofs_velocity(self, velocity, dofs_idx=None, envs_idx=None, *, skip_forward=False, unsafe=False):
        velocity, dofs_idx, envs_idx = self._sanitize_1D_io_variables(
            velocity, dofs_idx, self.n_dofs, envs_idx, skip_allocation=True, unsafe=unsafe
        )

        if velocity is None:
            kernel_set_dofs_zero_velocity(dofs_idx, envs_idx, self.dofs_state, self._static_rigid_sim_config)
        else:
            if self.n_envs == 0:
                velocity = velocity.unsqueeze(0)
            kernel_set_dofs_velocity(velocity, dofs_idx, envs_idx, self.dofs_state, self._static_rigid_sim_config)

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

    def set_dofs_position(self, position, dofs_idx=None, envs_idx=None, *, skip_forward=False, unsafe=False):
        position, dofs_idx, envs_idx = self._sanitize_1D_io_variables(
            position, dofs_idx, self.n_dofs, envs_idx, skip_allocation=True, unsafe=unsafe
        )
        if self.n_envs == 0:
            position = position.unsqueeze(0)
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
        self.collider.reset(envs_idx)
        self.collider.clear(envs_idx)
        if self.constraint_solver is not None:
            self.constraint_solver.reset(envs_idx)
            self.constraint_solver.clear(envs_idx)
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

    def control_dofs_force(self, force, dofs_idx=None, envs_idx=None, *, unsafe=False):
        force, dofs_idx, envs_idx = self._sanitize_1D_io_variables(
            force, dofs_idx, self.n_dofs, envs_idx, skip_allocation=True, unsafe=unsafe
        )
        if self.n_envs == 0:
            force = force.unsqueeze(0)
        kernel_control_dofs_force(force, dofs_idx, envs_idx, self.dofs_state, self._static_rigid_sim_config)

    def control_dofs_velocity(self, velocity, dofs_idx=None, envs_idx=None, *, unsafe=False):
        velocity, dofs_idx, envs_idx = self._sanitize_1D_io_variables(
            velocity, dofs_idx, self.n_dofs, envs_idx, skip_allocation=True, unsafe=unsafe
        )
        if self.n_envs == 0:
            velocity = velocity.unsqueeze(0)
        kernel_control_dofs_velocity(velocity, dofs_idx, envs_idx, self.dofs_state, self._static_rigid_sim_config)

    def control_dofs_position(self, position, dofs_idx=None, envs_idx=None, *, unsafe=False):
        position, dofs_idx, envs_idx = self._sanitize_1D_io_variables(
            position, dofs_idx, self.n_dofs, envs_idx, skip_allocation=True, unsafe=unsafe
        )
        if self.n_envs == 0:
            position = position.unsqueeze(0)
        kernel_control_dofs_position(position, dofs_idx, envs_idx, self.dofs_state, self._static_rigid_sim_config)

    def get_sol_params(self, geoms_idx=None, envs_idx=None, *, joints_idx=None, eqs_idx=None, unsafe=False):
        """
        Get constraint solver parameters.
        """
        if eqs_idx is not None:
            if not unsafe:
                assert envs_idx is None
            tensor = ti_field_to_torch(self.equalities_info.sol_params, None, eqs_idx, transpose=True, unsafe=unsafe)
            if self.n_envs == 0:
                tensor = tensor.squeeze(0)
        elif joints_idx is not None:
            tensor = ti_field_to_torch(self.joints_info.sol_params, envs_idx, joints_idx, transpose=True, unsafe=unsafe)
            if self.n_envs == 0 and self._options.batch_joints_info:
                tensor = tensor.squeeze(0)
        else:
            if not unsafe:
                assert envs_idx is None
            tensor = ti_field_to_torch(self.geoms_info.sol_params, geoms_idx, transpose=True, unsafe=unsafe)
        return tensor

    def get_links_pos(self, links_idx=None, envs_idx=None, *, unsafe=False):
        tensor = ti_field_to_torch(self.links_state.pos, envs_idx, links_idx, transpose=True, unsafe=unsafe)
        return tensor.squeeze(0) if self.n_envs == 0 else tensor

    def get_links_quat(self, links_idx=None, envs_idx=None, *, unsafe=False):
        tensor = ti_field_to_torch(self.links_state.quat, envs_idx, links_idx, transpose=True, unsafe=unsafe)
        return tensor.squeeze(0) if self.n_envs == 0 else tensor

    def get_links_vel(
        self,
        links_idx=None,
        envs_idx=None,
        *,
        ref: Literal["link_origin", "link_com", "root_com"] = "link_origin",
        unsafe: bool = False,
    ):
        _tensor, links_idx, envs_idx = self._sanitize_2D_io_variables(
            None, links_idx, self.n_links, 3, envs_idx, idx_name="links_idx", unsafe=unsafe
        )
        tensor = _tensor.unsqueeze(0) if self.n_envs == 0 else _tensor
        if ref == "root_com":
            ref = 0
        elif ref == "link_com":
            ref = 1
        elif ref == "link_origin":
            ref = 2
        else:
            raise ValueError("'ref' must be either 'link_origin', 'link_com', or 'root_com'.")
        kernel_get_links_vel(tensor, links_idx, envs_idx, ref, self.links_state, self._static_rigid_sim_config)
        return _tensor

    def get_links_ang(self, links_idx=None, envs_idx=None, *, unsafe=False):
        tensor = ti_field_to_torch(self.links_state.cd_ang, envs_idx, links_idx, transpose=True, unsafe=unsafe)
        return tensor.squeeze(0) if self.n_envs == 0 else tensor

    def get_links_acc(self, links_idx=None, envs_idx=None, *, mimick_imu=False, unsafe=False):
        _tensor, links_idx, envs_idx = self._sanitize_2D_io_variables(
            None, links_idx, self.n_links, 3, envs_idx, idx_name="links_idx", unsafe=unsafe
        )
        tensor = _tensor.unsqueeze(0) if self.n_envs == 0 else _tensor
        kernel_get_links_acc(
            mimick_imu,
            tensor,
            links_idx,
            envs_idx,
            self.links_state,
            self._rigid_global_info,
            self._static_rigid_sim_config,
        )
        return _tensor

    def get_links_acc_ang(self, links_idx=None, envs_idx=None, *, unsafe=False):
        tensor = ti_field_to_torch(self.links_state.cacc_ang, envs_idx, links_idx, transpose=True, unsafe=unsafe)
        return tensor.squeeze(0) if self.n_envs == 0 else tensor

    def get_links_root_COM(self, links_idx=None, envs_idx=None, *, unsafe=False):
        """
        Returns the center of mass (COM) of the entire kinematic tree to which the specified links belong.

        This corresponds to the global COM of each entity, assuming a single-rooted structure — that is, as long as no
        two successive links are connected by a free-floating joint (ie a joint that allows all 6 degrees of freedom).
        """
        tensor = ti_field_to_torch(self.links_state.COM, envs_idx, links_idx, transpose=True, unsafe=unsafe)
        return tensor.squeeze(0) if self.n_envs == 0 else tensor

    def get_links_mass_shift(self, links_idx=None, envs_idx=None, *, unsafe=False):
        tensor = ti_field_to_torch(self.links_state.mass_shift, envs_idx, links_idx, transpose=True, unsafe=unsafe)
        return tensor.squeeze(0) if self.n_envs == 0 else tensor

    def get_links_COM_shift(self, links_idx=None, envs_idx=None, *, unsafe=False):
        tensor = ti_field_to_torch(self.links_state.i_pos_shift, envs_idx, links_idx, transpose=True, unsafe=unsafe)
        return tensor.squeeze(0) if self.n_envs == 0 else tensor

    def get_links_inertial_mass(self, links_idx=None, envs_idx=None, *, unsafe=False):
        if not unsafe and self._options.batch_links_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched links info.")
        tensor = ti_field_to_torch(self.links_info.inertial_mass, envs_idx, links_idx, transpose=True, unsafe=unsafe)
        return tensor.squeeze(0) if self.n_envs == 0 and self._options.batch_links_info else tensor

    def get_links_invweight(self, links_idx=None, envs_idx=None, *, unsafe=False):
        if not unsafe and self._options.batch_links_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched links info.")
        tensor = ti_field_to_torch(self.links_info.invweight, envs_idx, links_idx, transpose=True, unsafe=unsafe)
        return tensor.squeeze(0) if self.n_envs == 0 and self._options.batch_links_info else tensor

    def get_geoms_friction_ratio(self, geoms_idx=None, envs_idx=None, *, unsafe=False):
        tensor = ti_field_to_torch(self.geoms_state.friction_ratio, envs_idx, geoms_idx, transpose=True, unsafe=unsafe)
        return tensor.squeeze(0) if self.n_envs == 0 else tensor

    def get_geoms_pos(self, geoms_idx=None, envs_idx=None, *, unsafe=False):
        tensor = ti_field_to_torch(self.geoms_state.pos, envs_idx, geoms_idx, transpose=True, unsafe=unsafe)
        return tensor.squeeze(0) if self.n_envs == 0 else tensor

    def get_qpos(self, qs_idx=None, envs_idx=None, *, unsafe=False):
        tensor = ti_field_to_torch(self.qpos, envs_idx, qs_idx, transpose=True, unsafe=unsafe)
        return tensor.squeeze(0) if self.n_envs == 0 else tensor

    def get_dofs_control_force(self, dofs_idx=None, envs_idx=None, *, unsafe=False):
        _tensor, dofs_idx, envs_idx = self._sanitize_1D_io_variables(
            None, dofs_idx, self.n_dofs, envs_idx, unsafe=unsafe
        )
        tensor = _tensor.unsqueeze(0) if self.n_envs == 0 else _tensor
        kernel_get_dofs_control_force(
            tensor, dofs_idx, envs_idx, self.dofs_state, self.dofs_info, self._static_rigid_sim_config
        )
        return _tensor

    def get_dofs_force(self, dofs_idx=None, envs_idx=None, *, unsafe=False):
        tensor = ti_field_to_torch(self.dofs_state.force, envs_idx, dofs_idx, transpose=True, unsafe=unsafe)
        return tensor.squeeze(0) if self.n_envs == 0 else tensor

    def get_dofs_velocity(self, dofs_idx=None, envs_idx=None, *, unsafe=False):
        tensor = ti_field_to_torch(self.dofs_state.vel, envs_idx, dofs_idx, transpose=True, unsafe=unsafe)
        return tensor.squeeze(0) if self.n_envs == 0 else tensor

    def get_dofs_position(self, dofs_idx=None, envs_idx=None, *, unsafe=False):
        tensor = ti_field_to_torch(self.dofs_state.pos, envs_idx, dofs_idx, transpose=True, unsafe=unsafe)
        return tensor.squeeze(0) if self.n_envs == 0 else tensor

    def get_dofs_kp(self, dofs_idx=None, envs_idx=None, *, unsafe=False):
        if not unsafe and not self._options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        tensor = ti_field_to_torch(self.dofs_info.kp, envs_idx, dofs_idx, transpose=True, unsafe=unsafe)
        return tensor.squeeze(0) if self.n_envs == 0 and self._options.batch_dofs_info else tensor

    def get_dofs_kv(self, dofs_idx=None, envs_idx=None, *, unsafe=False):
        if not unsafe and not self._options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        tensor = ti_field_to_torch(self.dofs_info.kv, envs_idx, dofs_idx, transpose=True, unsafe=unsafe)
        return tensor.squeeze(0) if self.n_envs == 0 and self._options.batch_dofs_info else tensor

    def get_dofs_force_range(self, dofs_idx=None, envs_idx=None, *, unsafe=False):
        if not unsafe and not self._options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        tensor = ti_field_to_torch(self.dofs_info.force_range, envs_idx, dofs_idx, transpose=True, unsafe=unsafe)
        if self.n_envs == 0 and self._options.batch_dofs_info:
            tensor = tensor.squeeze(0)
        return tensor[..., 0], tensor[..., 1]

    def get_dofs_limit(self, dofs_idx=None, envs_idx=None, *, unsafe=False):
        if not unsafe and not self._options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        tensor = ti_field_to_torch(self.dofs_info.limit, envs_idx, dofs_idx, transpose=True, unsafe=unsafe)
        if self.n_envs == 0 and self._options.batch_dofs_info:
            tensor = tensor.squeeze(0)
        return tensor[..., 0], tensor[..., 1]

    def get_dofs_stiffness(self, dofs_idx=None, envs_idx=None, *, unsafe=False):
        if not unsafe and not self._options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        tensor = ti_field_to_torch(self.dofs_info.stiffness, envs_idx, dofs_idx, transpose=True, unsafe=unsafe)
        return tensor.squeeze(0) if self.n_envs == 0 and self._options.batch_dofs_info else tensor

    def get_dofs_invweight(self, dofs_idx=None, envs_idx=None, *, unsafe=False):
        if not unsafe and not self._options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        tensor = ti_field_to_torch(self.dofs_info.invweight, envs_idx, dofs_idx, transpose=True, unsafe=unsafe)
        return tensor.squeeze(0) if self.n_envs == 0 and self._options.batch_dofs_info else tensor

    def get_dofs_armature(self, dofs_idx=None, envs_idx=None, *, unsafe=False):
        if not unsafe and not self._options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        tensor = ti_field_to_torch(self.dofs_info.armature, envs_idx, dofs_idx, transpose=True, unsafe=unsafe)
        return tensor.squeeze(0) if self.n_envs == 0 and self._options.batch_dofs_info else tensor

    def get_dofs_damping(self, dofs_idx=None, envs_idx=None, *, unsafe=False):
        if not unsafe and not self._options.batch_dofs_info and envs_idx is not None:
            gs.raise_exception("`envs_idx` cannot be specified for non-batched dofs info.")
        tensor = ti_field_to_torch(self.dofs_info.damping, envs_idx, dofs_idx, transpose=True, unsafe=unsafe)
        return tensor.squeeze(0) if self.n_envs == 0 and self._options.batch_dofs_info else tensor

    def get_mass_mat(self, dofs_idx=None, envs_idx=None, decompose=False, *, unsafe=False):
        tensor = ti_field_to_torch(
            self.mass_mat_L if decompose else self.mass_mat, envs_idx, transpose=True, unsafe=unsafe
        )

        if dofs_idx is not None:
            if isinstance(dofs_idx, (slice, int, np.integer)) or (dofs_idx.ndim == 0):
                tensor = tensor[:, dofs_idx, dofs_idx]
                if tensor.ndim == 1:
                    tensor = tensor.reshape((-1, 1, 1))
            else:
                tensor = tensor[:, dofs_idx.unsqueeze(1), dofs_idx]
        if self.n_envs == 0:
            tensor = tensor.squeeze(0)

        if decompose:
            mass_mat_D_inv = ti_field_to_torch(
                self._rigid_global_info.mass_mat_D_inv, envs_idx, dofs_idx, transpose=True, unsafe=unsafe
            )
            if self.n_envs == 0:
                mass_mat_D_inv = mass_mat_D_inv.squeeze(0)
            return tensor, mass_mat_D_inv

        return tensor

    def get_geoms_friction(self, geoms_idx=None, *, unsafe=False):
        return ti_field_to_torch(self.geoms_info.friction, geoms_idx, None, unsafe=unsafe)

    def set_geom_friction(self, friction, geoms_idx):
        kernel_set_geom_friction(geoms_idx, friction, self.geoms_info)

    def set_geoms_friction(self, friction, geoms_idx=None, *, unsafe=False):
        friction, geoms_idx, _ = self._sanitize_1D_io_variables(
            friction,
            geoms_idx,
            self.n_geoms,
            None,
            batched=False,
            idx_name="geoms_idx",
            skip_allocation=True,
            unsafe=unsafe,
        )
        kernel_set_geoms_friction(friction, geoms_idx, self.geoms_info, self._static_rigid_sim_config)

    def add_weld_constraint(self, link1_idx, link2_idx, envs_idx=None, *, unsafe=False):
        return self.constraint_solver.add_weld_constraint(link1_idx, link2_idx, envs_idx, unsafe=unsafe)

    def delete_weld_constraint(self, link1_idx, link2_idx, envs_idx=None, *, unsafe=False):
        return self.constraint_solver.delete_weld_constraint(link1_idx, link2_idx, envs_idx, unsafe=unsafe)

    def get_weld_constraints(self, as_tensor: bool = True, to_torch: bool = True):
        return self.constraint_solver.get_weld_constraints(as_tensor, to_torch)

    def get_equality_constraints(self, as_tensor: bool = True, to_torch: bool = True):
        return self.constraint_solver.get_equality_constraints(as_tensor, to_torch)

    def clear_external_force(self):
        kernel_clear_external_force(self.links_state, self._rigid_global_info, self._static_rigid_sim_config)

    def update_vgeoms(self):
        kernel_update_vgeoms(self.vgeoms_info, self.vgeoms_state, self.links_state, self._static_rigid_sim_config)

    @gs.assert_built
    def set_gravity(self, gravity, envs_idx=None):
        super().set_gravity(gravity, envs_idx)
        if hasattr(self, "_rigid_global_info"):
            self._rigid_global_info.gravity.copy_from(self._gravity)

    def rigid_entity_inverse_kinematics(
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
        rigid_entity,
    ):
        # Inverse kinematics logic moved from rigid_entity to here, because it incurs circular import.
        kernel_rigid_entity_inverse_kinematics(
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
            rigid_entity,
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

    def update_drone_propeller_vgeoms(self, n_propellers, propellers_vgeom_idxs, propellers_revs, propellers_spin):
        kernel_update_drone_propeller_vgeoms(
            n_propellers,
            propellers_vgeom_idxs,
            propellers_revs,
            propellers_spin,
            self.vgeoms_state,
            self._static_rigid_sim_config,
        )

    def set_drone_rpm(self, n_propellers, propellers_link_idxs, propellers_rpm, propellers_spin, KF, KM, invert):
        kernel_set_drone_rpm(
            n_propellers,
            propellers_link_idxs,
            propellers_rpm,
            propellers_spin,
            KF,
            KM,
            invert,
            self.links_state,
        )

    def update_verts_for_geom(self, i_g):
        kernel_update_verts_for_geom(
            i_g,
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
        return sum(entity.n_verts if entity.is_free else 0 for entity in self._entities)

    @property
    def n_fixed_verts(self):
        if self.is_built:
            return self._n_fixed_verts
        return sum(entity.n_verts if not entity.is_free else 0 for entity in self._entities)

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
        if len(self._entities) == 0:
            return np.array([])
        return np.concatenate([entity.init_qpos for entity in self._entities], dtype=gs.np_float)

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
def kernel_compute_mass_matrix(
    # taichi variables
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    decompose: ti.i32,
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


@ti.kernel
def kernel_init_invweight(
    links_invweight: ti.types.ndarray(),
    dofs_invweight: ti.types.ndarray(),
    # taichi variables
    links_info: array_class.LinksInfo,
    dofs_info: array_class.DofsInfo,
):
    for I in ti.grouped(links_info.parent_idx):
        for j in ti.static(range(2)):
            if links_info.invweight[I][j] < gs.EPS:
                links_info.invweight[I][j] = links_invweight[I[0], j]

    for I in ti.grouped(dofs_info.dof_start):
        if dofs_info.invweight[I] < gs.EPS:
            dofs_info.invweight[I] = dofs_invweight[I[0]]


@ti.kernel
def kernel_init_meaninertia(
    # taichi variables
    rigid_global_info: array_class.RigidGlobalInfo,
    entities_info: array_class.EntitiesInfo,
    static_rigid_sim_config: ti.template(),
):
    n_dofs = rigid_global_info.mass_mat.shape[0]
    _B = rigid_global_info.mass_mat.shape[2]
    n_entities = entities_info.n_links.shape[0]
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_b in range(_B):
        if n_dofs > 0:
            rigid_global_info.meaninertia[i_b] = 0.0
            for i_e in range(n_entities):
                for i_d in range(entities_info.dof_start[i_e], entities_info.dof_end[i_e]):
                    I_d = [i_d, i_b] if ti.static(static_rigid_sim_config.batch_dofs_info) else i_d
                    rigid_global_info.meaninertia[i_b] += rigid_global_info.mass_mat[i_d, i_d, i_b]
                rigid_global_info.meaninertia[i_b] = rigid_global_info.meaninertia[i_b] / n_dofs
        else:
            rigid_global_info.meaninertia[i_b] = 1.0


@ti.kernel
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
    # we will use RigidGlobalInfo as typing after Hugh adds array_struct feature to taichi
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    n_dofs = dofs_state.ctrl_mode.shape[0]
    _B = dofs_state.ctrl_mode.shape[1]
    for I in ti.grouped(dofs_info.invweight):
        i = I[0]  # batching (if any) will be the second dim

        for j in ti.static(range(3)):
            dofs_info.motion_ang[I][j] = dofs_motion_ang[i, j]
            dofs_info.motion_vel[I][j] = dofs_motion_vel[i, j]

        for j in ti.static(range(2)):
            dofs_info.limit[I][j] = dofs_limit[i, j]
            dofs_info.force_range[I][j] = dofs_force_range[i, j]

        dofs_info.armature[I] = dofs_armature[i]
        dofs_info.invweight[I] = dofs_invweight[i]
        dofs_info.stiffness[I] = dofs_stiffness[i]
        dofs_info.damping[I] = dofs_damping[i]
        dofs_info.frictionloss[I] = dofs_frictionloss[i]
        dofs_info.kp[I] = dofs_kp[i]
        dofs_info.kv[I] = dofs_kv[i]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i, b in ti.ndrange(n_dofs, _B):
        dofs_state.ctrl_mode[i, b] = gs.CTRL_MODE.FORCE

    if ti.static(static_rigid_sim_config.use_hibernation):
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
        for i, b in ti.ndrange(n_dofs, _B):
            dofs_state.hibernated[i, b] = False
            rigid_global_info.awake_dofs[i, b] = i

        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
        for b in range(_B):
            rigid_global_info.n_awake_dofs[b] = n_dofs


@ti.kernel
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
    for I in ti.grouped(links_info.invweight):
        i = I[0]

        links_info.parent_idx[I] = links_parent_idx[i]
        links_info.root_idx[I] = links_root_idx[i]
        links_info.q_start[I] = links_q_start[i]
        links_info.joint_start[I] = links_joint_start[i]
        links_info.dof_start[I] = links_dof_start[i]
        links_info.q_end[I] = links_q_end[i]
        links_info.dof_end[I] = links_dof_end[i]
        links_info.joint_end[I] = links_joint_end[i]
        links_info.n_dofs[I] = links_dof_end[i] - links_dof_start[i]
        links_info.is_fixed[I] = links_is_fixed[i]
        links_info.entity_idx[I] = links_entity_idx[i]

        for j in ti.static(range(2)):
            links_info.invweight[I][j] = links_invweight[i, j]

        for j in ti.static(range(4)):
            links_info.quat[I][j] = links_quat[i, j]
            links_info.inertial_quat[I][j] = links_inertial_quat[i, j]

        for j in ti.static(range(3)):
            links_info.pos[I][j] = links_pos[i, j]
            links_info.inertial_pos[I][j] = links_inertial_pos[i, j]

        links_info.inertial_mass[I] = links_inertial_mass[i]
        for j1 in ti.static(range(3)):
            for j2 in ti.static(range(3)):
                links_info.inertial_i[I][j1, j2] = links_inertial_i[i, j1, j2]

    for i, b in ti.ndrange(n_links, _B):
        I = [i, b] if ti.static(static_rigid_sim_config.batch_links_info) else i

        # Update state for root fixed link. Their state will not be updated in forward kinematics later but can be manually changed by user.
        if links_info.parent_idx[I] == -1 and links_info.is_fixed[I]:
            for j in ti.static(range(4)):
                links_state.quat[i, b][j] = links_quat[i, j]

            for j in ti.static(range(3)):
                links_state.pos[i, b][j] = links_pos[i, j]

        for j in ti.static(range(3)):
            links_state.i_pos_shift[i, b][j] = 0.0
        links_state.mass_shift[i, b] = 0.0

    if ti.static(static_rigid_sim_config.use_hibernation):
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
        for i, b in ti.ndrange(n_links, _B):
            links_state.hibernated[i, b] = False
            rigid_global_info.awake_links[i, b] = i

        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
        for b in range(_B):
            rigid_global_info.n_awake_links[b] = n_links


@ti.kernel
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
    for I in ti.grouped(joints_info.type):
        i = I[0]

        joints_info.type[I] = joints_type[i]
        joints_info.q_start[I] = joints_q_start[i]
        joints_info.dof_start[I] = joints_dof_start[i]
        joints_info.q_end[I] = joints_q_end[i]
        joints_info.dof_end[I] = joints_dof_end[i]
        joints_info.n_dofs[I] = joints_dof_end[i] - joints_dof_start[i]

        for j in ti.static(range(7)):
            joints_info.sol_params[I][j] = joints_sol_params[i, j]
        for j in ti.static(range(3)):
            joints_info.pos[I][j] = joints_pos[i, j]


@ti.kernel
def kernel_init_vert_fields(
    verts: ti.types.ndarray(),
    faces: ti.types.ndarray(),
    edges: ti.types.ndarray(),
    normals: ti.types.ndarray(),
    verts_geom_idx: ti.types.ndarray(),
    init_center_pos: ti.types.ndarray(),
    verts_state_idx: ti.types.ndarray(),
    is_free: ti.types.ndarray(),
    # taichi variables
    verts_info: array_class.VertsInfo,
    faces_info: array_class.FacesInfo,
    edges_info: array_class.EdgesInfo,
    static_rigid_sim_config: ti.template(),
):
    n_verts = verts.shape[0]
    n_faces = faces.shape[0]
    n_edges = edges.shape[0]
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i in range(n_verts):
        for j in ti.static(range(3)):
            verts_info.init_pos[i][j] = verts[i, j]
            verts_info.init_normal[i][j] = normals[i, j]
            verts_info.init_center_pos[i][j] = init_center_pos[i, j]

        verts_info.geom_idx[i] = verts_geom_idx[i]
        verts_info.verts_state_idx[i] = verts_state_idx[i]
        verts_info.is_free[i] = is_free[i]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i in range(n_faces):
        for j in ti.static(range(3)):
            faces_info.verts_idx[i][j] = faces[i, j]
        faces_info.geom_idx[i] = verts_geom_idx[faces[i, 0]]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i in range(n_edges):
        edges_info.v0[i] = edges[i, 0]
        edges_info.v1[i] = edges[i, 1]
        # minus = verts_info.init_pos[edges[i, 0]] - verts_info.init_pos[edges[i, 1]]
        # edges_info.length[i] = minus.norm()
        # FIXME: the line below does not work
        edges_info.length[i] = (verts_info.init_pos[edges[i, 0]] - verts_info.init_pos[edges[i, 1]]).norm()


@ti.kernel
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
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i in range(n_vverts):
        for j in ti.static(range(3)):
            vverts_info.init_pos[i][j] = vverts[i, j]
            vverts_info.init_vnormal[i][j] = vnormals[i, j]

        vverts_info.vgeom_idx[i] = vverts_vgeom_idx[i]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i in range(n_vfaces):
        for j in ti.static(range(3)):
            vfaces_info.vverts_idx[i][j] = vfaces[i, j]
        vfaces_info.vgeom_idx[i] = vverts_vgeom_idx[vfaces[i, 0]]


@ti.kernel
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
    geoms_is_free: ti.types.ndarray(),
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
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i in range(n_geoms):
        for j in ti.static(range(3)):
            geoms_info.pos[i][j] = geoms_pos[i, j]
            geoms_info.center[i][j] = geoms_center[i, j]

        for j in ti.static(range(4)):
            geoms_info.quat[i][j] = geoms_quat[i, j]

        for j in ti.static(range(7)):
            geoms_info.data[i][j] = geoms_data[i, j]
            geoms_info.sol_params[i][j] = geoms_sol_params[i, j]

        geoms_info.vert_start[i] = geoms_vert_start[i]
        geoms_info.vert_end[i] = geoms_vert_end[i]
        geoms_info.vert_num[i] = geoms_vert_end[i] - geoms_vert_start[i]

        geoms_info.face_start[i] = geoms_face_start[i]
        geoms_info.face_end[i] = geoms_face_end[i]
        geoms_info.face_num[i] = geoms_face_end[i] - geoms_face_start[i]

        geoms_info.edge_start[i] = geoms_edge_start[i]
        geoms_info.edge_end[i] = geoms_edge_end[i]
        geoms_info.edge_num[i] = geoms_edge_end[i] - geoms_edge_start[i]

        geoms_info.verts_state_start[i] = geoms_verts_state_start[i]
        geoms_info.verts_state_end[i] = geoms_verts_state_end[i]

        geoms_info.link_idx[i] = geoms_link_idx[i]
        geoms_info.type[i] = geoms_type[i]
        geoms_info.friction[i] = geoms_friction[i]

        geoms_info.is_convex[i] = geoms_is_convex[i]
        geoms_info.needs_coup[i] = geoms_needs_coup[i]
        geoms_info.contype[i] = geoms_contype[i]
        geoms_info.conaffinity[i] = geoms_conaffinity[i]

        geoms_info.coup_softness[i] = geoms_coup_softness[i]
        geoms_info.coup_friction[i] = geoms_coup_friction[i]
        geoms_info.coup_restitution[i] = geoms_coup_restitution[i]

        geoms_info.is_free[i] = geoms_is_free[i]
        geoms_info.is_decomposed[i] = geoms_is_decomp[i]

        # compute init AABB.
        # Beware the ordering the this corners is critical and MUST NOT be changed as this order is used elsewhere
        # in the codebase, e.g. overlap estimation between two convex geometries using there bounding boxes.
        lower = gu.ti_vec3(ti.math.inf)
        upper = gu.ti_vec3(-ti.math.inf)
        for i_v in range(geoms_vert_start[i], geoms_vert_end[i]):
            lower = ti.min(lower, verts_info.init_pos[i_v])
            upper = ti.max(upper, verts_info.init_pos[i_v])
        geoms_init_AABB[i, 0] = ti.Vector([lower[0], lower[1], lower[2]], dt=gs.ti_float)
        geoms_init_AABB[i, 1] = ti.Vector([lower[0], lower[1], upper[2]], dt=gs.ti_float)
        geoms_init_AABB[i, 2] = ti.Vector([lower[0], upper[1], lower[2]], dt=gs.ti_float)
        geoms_init_AABB[i, 3] = ti.Vector([lower[0], upper[1], upper[2]], dt=gs.ti_float)
        geoms_init_AABB[i, 4] = ti.Vector([upper[0], lower[1], lower[2]], dt=gs.ti_float)
        geoms_init_AABB[i, 5] = ti.Vector([upper[0], lower[1], upper[2]], dt=gs.ti_float)
        geoms_init_AABB[i, 6] = ti.Vector([upper[0], upper[1], lower[2]], dt=gs.ti_float)
        geoms_init_AABB[i, 7] = ti.Vector([upper[0], upper[1], upper[2]], dt=gs.ti_float)

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_g, i_b in ti.ndrange(n_geoms, _B):
        geoms_state.friction_ratio[i_g, i_b] = 1.0


@ti.kernel
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


@ti.kernel
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
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i in range(n_vgeoms):
        for j in ti.static(range(3)):
            vgeoms_info.pos[i][j] = vgeoms_pos[i, j]

        for j in ti.static(range(4)):
            vgeoms_info.quat[i][j] = vgeoms_quat[i, j]

        vgeoms_info.vvert_start[i] = vgeoms_vvert_start[i]
        vgeoms_info.vvert_end[i] = vgeoms_vvert_end[i]
        vgeoms_info.vvert_num[i] = vgeoms_vvert_end[i] - vgeoms_vvert_start[i]

        vgeoms_info.vface_start[i] = vgeoms_vface_start[i]
        vgeoms_info.vface_end[i] = vgeoms_vface_end[i]
        vgeoms_info.vface_num[i] = vgeoms_vface_end[i] - vgeoms_vface_start[i]

        vgeoms_info.link_idx[i] = vgeoms_link_idx[i]
        for j in ti.static(range(4)):
            vgeoms_info.color[i][j] = vgeoms_color[i, j]


@ti.kernel
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

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i in range(n_entities):
        entities_info.dof_start[i] = entities_dof_start[i]
        entities_info.dof_end[i] = entities_dof_end[i]
        entities_info.n_dofs[i] = entities_dof_end[i] - entities_dof_start[i]

        entities_info.link_start[i] = entities_link_start[i]
        entities_info.link_end[i] = entities_link_end[i]
        entities_info.n_links[i] = entities_link_end[i] - entities_link_start[i]

        entities_info.geom_start[i] = entities_geom_start[i]
        entities_info.geom_end[i] = entities_geom_end[i]
        entities_info.n_geoms[i] = entities_geom_end[i] - entities_geom_start[i]

        entities_info.gravity_compensation[i] = entities_gravity_compensation[i]
        entities_info.is_local_collision_mask[i] = entities_is_local_collision_mask[i]

        if ti.static(static_rigid_sim_config.batch_dofs_info):
            for i_d, i_b in ti.ndrange((entities_dof_start[i], entities_dof_end[i]), _B):
                dofs_info.dof_start[i_d, i_b] = entities_dof_start[i]
        else:
            for i_d in range(entities_dof_start[i], entities_dof_end[i]):
                dofs_info.dof_start[i_d] = entities_dof_start[i]

    if ti.static(static_rigid_sim_config.use_hibernation):
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
        for i, b in ti.ndrange(n_entities, _B):
            entities_state.hibernated[i, b] = False
            rigid_global_info.awake_entities[i, b] = i

        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
        for b in range(_B):
            rigid_global_info.n_awake_entities[b] = n_entities


@ti.kernel
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

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i, b in ti.ndrange(n_equalities, _B):
        equalities_info.eq_obj1id[i, b] = equalities_eq_obj1id[i]
        equalities_info.eq_obj2id[i, b] = equalities_eq_obj2id[i]
        equalities_info.eq_type[i, b] = equalities_eq_type[i]
        for j in ti.static(range(11)):
            equalities_info.eq_data[i, b][j] = equalities_eq_data[i, j]
        for j in ti.static(range(7)):
            equalities_info.sol_params[i, b][j] = equalities_sol_params[i, j]


@ti.kernel
def kernel_forward_dynamics(
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    joints_info: array_class.JointsInfo,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    func_forward_dynamics(
        links_state=links_state,
        links_info=links_info,
        dofs_state=dofs_state,
        dofs_info=dofs_info,
        joints_info=joints_info,
        entities_info=entities_info,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
    )


@ti.kernel
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
    vel_rot = links_state.cd_ang[link_idx, i_b].cross(pos_world - links_state.COM[link_idx, i_b])
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
    _B = links_state.pos.shape[1]
    n_links = links_state.pos.shape[0]
    n_entities = entities_info.n_links.shape[0]
    n_dofs = dofs_state.f_ang.shape[0]

    if ti.static(static_rigid_sim_config.use_hibernation):
        # crb initialize
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(_B):
            for i_l_ in range(rigid_global_info.n_awake_links[i_b]):
                i_l = rigid_global_info.awake_links[i_l_, i_b]
                links_state.crb_inertial[i_l, i_b] = links_state.cinr_inertial[i_l, i_b]
                links_state.crb_pos[i_l, i_b] = links_state.cinr_pos[i_l, i_b]
                links_state.crb_quat[i_l, i_b] = links_state.cinr_quat[i_l, i_b]
                links_state.crb_mass[i_l, i_b] = links_state.cinr_mass[i_l, i_b]

        # crb
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(_B):
            for i_e_ in range(rigid_global_info.n_awake_entities[i_b]):
                i_e = rigid_global_info.awake_entities[i_e_, i_b]
                for i in range(entities_info.n_links[i_e]):
                    i_l = entities_info.link_end[i_e] - 1 - i
                    I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l
                    i_p = links_info.parent_idx[I_l]

                    if i_p != -1:
                        links_state.crb_inertial[i_p, i_b] = (
                            links_state.crb_inertial[i_p, i_b] + links_state.crb_inertial[i_l, i_b]
                        )
                        links_state.crb_mass[i_p, i_b] = links_state.crb_mass[i_p, i_b] + links_state.crb_mass[i_l, i_b]

                        links_state.crb_pos[i_p, i_b] = links_state.crb_pos[i_p, i_b] + links_state.crb_pos[i_l, i_b]
                        links_state.crb_quat[i_p, i_b] = links_state.crb_quat[i_p, i_b] + links_state.crb_quat[i_l, i_b]

        # mass_mat
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(_B):
            for i_l_ in range(rigid_global_info.n_awake_links[i_b]):
                i_l = rigid_global_info.awake_links[i_l_, i_b]
                I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l
                for i_d in range(links_info.dof_start[I_l], links_info.dof_end[I_l]):
                    dofs_state.f_ang[i_d, i_b], dofs_state.f_vel[i_d, i_b] = gu.inertial_mul(
                        links_state.crb_pos[i_l, i_b],
                        links_state.crb_inertial[i_l, i_b],
                        links_state.crb_mass[i_l, i_b],
                        dofs_state.cdof_vel[i_d, i_b],
                        dofs_state.cdof_ang[i_d, i_b],
                    )

        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
        for i_b in range(_B):
            for i_e_ in range(rigid_global_info.n_awake_entities[i_b]):
                i_e = rigid_global_info.awake_entities[i_e_, i_b]
                for i_d in range(entities_info.dof_start[i_e], entities_info.dof_end[i_e]):
                    for j_d in range(entities_info.dof_start[i_e], entities_info.dof_end[i_e]):
                        rigid_global_info.mass_mat[i_d, j_d, i_b] = (
                            dofs_state.f_ang[i_d, i_b].dot(dofs_state.cdof_ang[j_d, i_b])
                            + dofs_state.f_vel[i_d, i_b].dot(dofs_state.cdof_vel[j_d, i_b])
                        ) * rigid_global_info.mass_parent_mask[i_d, j_d]

                # FIXME: Updating the lower-part of the mass matrix is irrelevant
                for i_d in range(entities_info.dof_start[i_e], entities_info.dof_end[i_e]):
                    for j_d in range(i_d + 1, entities_info.dof_end[i_e]):
                        rigid_global_info.mass_mat[i_d, j_d, i_b] = rigid_global_info.mass_mat[j_d, i_d, i_b]

                # Take into account motor armature
                for i_d in range(entities_info.dof_start[i_e], entities_info.dof_end[i_e]):
                    I_d = [i_d, i_b] if ti.static(static_rigid_sim_config.batch_dofs_info) else i_d
                    rigid_global_info.mass_mat[i_d, i_d, i_b] = (
                        rigid_global_info.mass_mat[i_d, i_d, i_b] + dofs_info.armature[I_d]
                    )

                # Take into account first-order correction terms for implicit integration scheme right away
                if ti.static(implicit_damping):
                    for i_d in range(entities_info.dof_start[i_e], entities_info.dof_end[i_e]):
                        I_d = [i_d, i_b] if ti.static(static_rigid_sim_config.batch_dofs_info) else i_d
                        rigid_global_info.mass_mat[i_d, i_d, i_b] += (
                            dofs_info.damping[I_d] * static_rigid_sim_config.substep_dt
                        )
                        if (dofs_state.ctrl_mode[i_d, i_b] == gs.CTRL_MODE.POSITION) or (
                            dofs_state.ctrl_mode[i_d, i_b] == gs.CTRL_MODE.VELOCITY
                        ):
                            # qM += d qfrc_actuator / d qvel
                            rigid_global_info.mass_mat[i_d, i_d, i_b] += (
                                dofs_info.kv[I_d] * static_rigid_sim_config.substep_dt
                            )
    else:
        # crb initialize
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_l, i_b in ti.ndrange(n_links, _B):
            links_state.crb_inertial[i_l, i_b] = links_state.cinr_inertial[i_l, i_b]
            links_state.crb_pos[i_l, i_b] = links_state.cinr_pos[i_l, i_b]
            links_state.crb_quat[i_l, i_b] = links_state.cinr_quat[i_l, i_b]
            links_state.crb_mass[i_l, i_b] = links_state.cinr_mass[i_l, i_b]

        # crb
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_e, i_b in ti.ndrange(n_entities, _B):
            for i in range(entities_info.n_links[i_e]):
                i_l = entities_info.link_end[i_e] - 1 - i
                I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l
                i_p = links_info.parent_idx[I_l]

                if i_p != -1:
                    links_state.crb_inertial[i_p, i_b] = (
                        links_state.crb_inertial[i_p, i_b] + links_state.crb_inertial[i_l, i_b]
                    )
                    links_state.crb_mass[i_p, i_b] = links_state.crb_mass[i_p, i_b] + links_state.crb_mass[i_l, i_b]

                    links_state.crb_pos[i_p, i_b] = links_state.crb_pos[i_p, i_b] + links_state.crb_pos[i_l, i_b]
                    links_state.crb_quat[i_p, i_b] = links_state.crb_quat[i_p, i_b] + links_state.crb_quat[i_l, i_b]

        # mass_mat
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_l, i_b in ti.ndrange(n_links, _B):
            I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l
            for i_d in range(links_info.dof_start[I_l], links_info.dof_end[I_l]):
                dofs_state.f_ang[i_d, i_b], dofs_state.f_vel[i_d, i_b] = gu.inertial_mul(
                    links_state.crb_pos[i_l, i_b],
                    links_state.crb_inertial[i_l, i_b],
                    links_state.crb_mass[i_l, i_b],
                    dofs_state.cdof_vel[i_d, i_b],
                    dofs_state.cdof_ang[i_d, i_b],
                )

        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
        for i_e, i_b in ti.ndrange(n_entities, _B):
            for i_d, j_d in ti.ndrange(
                (entities_info.dof_start[i_e], entities_info.dof_end[i_e]),
                (entities_info.dof_start[i_e], entities_info.dof_end[i_e]),
            ):
                rigid_global_info.mass_mat[i_d, j_d, i_b] = (
                    dofs_state.f_ang[i_d, i_b].dot(dofs_state.cdof_ang[j_d, i_b])
                    + dofs_state.f_vel[i_d, i_b].dot(dofs_state.cdof_vel[j_d, i_b])
                ) * rigid_global_info.mass_parent_mask[i_d, j_d]

            # FIXME: Updating the lower-part of the mass matrix is irrelevant
            for i_d in range(entities_info.dof_start[i_e], entities_info.dof_end[i_e]):
                for j_d in range(i_d + 1, entities_info.dof_end[i_e]):
                    rigid_global_info.mass_mat[i_d, j_d, i_b] = rigid_global_info.mass_mat[j_d, i_d, i_b]

        # Take into account motor armature
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_d, i_b in ti.ndrange(n_dofs, _B):
            I_d = [i_d, i_b] if ti.static(static_rigid_sim_config.batch_dofs_info) else i_d
            rigid_global_info.mass_mat[i_d, i_d, i_b] = (
                rigid_global_info.mass_mat[i_d, i_d, i_b] + dofs_info.armature[I_d]
            )

        # Take into account first-order correction terms for implicit integration scheme right away
        if ti.static(implicit_damping):
            ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
            for i_d, i_b in ti.ndrange(n_dofs, _B):
                I_d = [i_d, i_b] if ti.static(static_rigid_sim_config.batch_dofs_info) else i_d
                rigid_global_info.mass_mat[i_d, i_d, i_b] += dofs_info.damping[I_d] * static_rigid_sim_config.substep_dt
                if (dofs_state.ctrl_mode[i_d, i_b] == gs.CTRL_MODE.POSITION) or (
                    dofs_state.ctrl_mode[i_d, i_b] == gs.CTRL_MODE.VELOCITY
                ):
                    # qM += d qfrc_actuator / d qvel
                    rigid_global_info.mass_mat[i_d, i_d, i_b] += dofs_info.kv[I_d] * static_rigid_sim_config.substep_dt


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
    _B = dofs_state.ctrl_mode.shape[1]
    n_entities = entities_info.n_links.shape[0]

    if ti.static(static_rigid_sim_config.use_hibernation):
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
        for i_b in range(_B):
            for i_e_ in range(rigid_global_info.n_awake_entities[i_b]):
                i_e = rigid_global_info.awake_entities[i_e_, i_b]

                if rigid_global_info._mass_mat_mask[i_e, i_b] == 1:
                    entity_dof_start = entities_info.dof_start[i_e]
                    entity_dof_end = entities_info.dof_end[i_e]
                    n_dofs = entities_info.n_dofs[i_e]

                    for i_d in range(entity_dof_start, entity_dof_end):
                        for j_d in range(entity_dof_start, i_d + 1):
                            rigid_global_info.mass_mat_L[i_d, j_d, i_b] = rigid_global_info.mass_mat[i_d, j_d, i_b]

                        if ti.static(implicit_damping):
                            I_d = [i_d, i_b] if ti.static(static_rigid_sim_config.batch_dofs_info) else i_d
                            rigid_global_info.mass_mat_L[i_d, i_d, i_b] += (
                                dofs_info.damping[I_d] * static_rigid_sim_config.substep_dt
                            )
                            if ti.static(static_rigid_sim_config.integrator == gs.integrator.implicitfast):
                                if (dofs_state.ctrl_mode[i_d, i_b] == gs.CTRL_MODE.POSITION) or (
                                    dofs_state.ctrl_mode[i_d, i_b] == gs.CTRL_MODE.VELOCITY
                                ):
                                    rigid_global_info.mass_mat_L[i_d, i_d, i_b] += (
                                        dofs_info.kv[I_d] * static_rigid_sim_config.substep_dt
                                    )

                    for i_d_ in range(n_dofs):
                        i_d = entity_dof_end - i_d_ - 1
                        rigid_global_info.mass_mat_D_inv[i_d, i_b] = 1.0 / rigid_global_info.mass_mat_L[i_d, i_d, i_b]

                        for j_d_ in range(i_d - entity_dof_start):
                            j_d = i_d - j_d_ - 1
                            a = rigid_global_info.mass_mat_L[i_d, j_d, i_b] * rigid_global_info.mass_mat_D_inv[i_d, i_b]
                            for k_d in range(entity_dof_start, j_d + 1):
                                rigid_global_info.mass_mat_L[j_d, k_d, i_b] -= (
                                    a * rigid_global_info.mass_mat_L[i_d, k_d, i_b]
                                )
                            rigid_global_info.mass_mat_L[i_d, j_d, i_b] = a

                        # FIXME: Diagonal coeffs of L are ignored in computations, so no need to update them.
                        rigid_global_info.mass_mat_L[i_d, i_d, i_b] = 1.0
    else:
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
        for i_e, i_b in ti.ndrange(n_entities, _B):
            if rigid_global_info._mass_mat_mask[i_e, i_b] == 1:
                entity_dof_start = entities_info.dof_start[i_e]
                entity_dof_end = entities_info.dof_end[i_e]
                n_dofs = entities_info.n_dofs[i_e]

                for i_d in range(entity_dof_start, entity_dof_end):
                    for j_d in range(entity_dof_start, i_d + 1):
                        rigid_global_info.mass_mat_L[i_d, j_d, i_b] = rigid_global_info.mass_mat[i_d, j_d, i_b]

                    if ti.static(implicit_damping):
                        I_d = [i_d, i_b] if ti.static(static_rigid_sim_config.batch_dofs_info) else i_d
                        rigid_global_info.mass_mat_L[i_d, i_d, i_b] += (
                            dofs_info.damping[I_d] * static_rigid_sim_config.substep_dt
                        )
                        if ti.static(static_rigid_sim_config.integrator == gs.integrator.implicitfast):
                            if (dofs_state.ctrl_mode[i_d, i_b] == gs.CTRL_MODE.POSITION) or (
                                dofs_state.ctrl_mode[i_d, i_b] == gs.CTRL_MODE.VELOCITY
                            ):
                                rigid_global_info.mass_mat_L[i_d, i_d, i_b] += (
                                    dofs_info.kv[I_d] * static_rigid_sim_config.substep_dt
                                )

                for i_d_ in range(n_dofs):
                    i_d = entity_dof_end - i_d_ - 1
                    rigid_global_info.mass_mat_D_inv[i_d, i_b] = 1.0 / rigid_global_info.mass_mat_L[i_d, i_d, i_b]

                    for j_d_ in range(i_d - entity_dof_start):
                        j_d = i_d - j_d_ - 1
                        a = rigid_global_info.mass_mat_L[i_d, j_d, i_b] * rigid_global_info.mass_mat_D_inv[i_d, i_b]
                        for k_d in range(entity_dof_start, j_d + 1):
                            rigid_global_info.mass_mat_L[j_d, k_d, i_b] -= (
                                a * rigid_global_info.mass_mat_L[i_d, k_d, i_b]
                            )
                        rigid_global_info.mass_mat_L[i_d, j_d, i_b] = a

                    # FIXME: Diagonal coeffs of L are ignored in computations, so no need to update them.
                    rigid_global_info.mass_mat_L[i_d, i_d, i_b] = 1.0


@ti.func
def func_solve_mass_batched(
    vec: array_class.V_ANNOTATION,
    out: array_class.V_ANNOTATION,
    i_b: ti.int32,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):

    n_entities = entities_info.n_links.shape[0]
    if ti.static(static_rigid_sim_config.use_hibernation):
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
        for i_e_ in range(rigid_global_info.n_awake_entities[i_b]):
            i_e = rigid_global_info.awake_entities[i_e_, i_b]

            if rigid_global_info._mass_mat_mask[i_e, i_b] == 1:
                entity_dof_start = entities_info.dof_start[i_e]
                entity_dof_end = entities_info.dof_end[i_e]
                n_dofs = entities_info.n_dofs[i_e]

                # Step 1: Solve w st. L^T @ w = y
                for i_d_ in range(n_dofs):
                    i_d = entity_dof_end - i_d_ - 1
                    out[i_d, i_b] = vec[i_d, i_b]
                    for j_d in range(i_d + 1, entity_dof_end):
                        out[i_d, i_b] -= rigid_global_info.mass_mat_L[j_d, i_d, i_b] * out[j_d, i_b]

                # Step 2: z = D^{-1} w
                for i_d in range(entity_dof_start, entity_dof_end):
                    out[i_d, i_b] *= rigid_global_info.mass_mat_D_inv[i_d, i_b]

                # Step 3: Solve x st. L @ x = z
                for i_d in range(entity_dof_start, entity_dof_end):
                    for j_d in range(entity_dof_start, i_d):
                        out[i_d, i_b] -= rigid_global_info.mass_mat_L[i_d, j_d, i_b] * out[j_d, i_b]
    else:
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
        for i_e in range(n_entities):
            if rigid_global_info._mass_mat_mask[i_e, i_b] == 1:
                entity_dof_start = entities_info.dof_start[i_e]
                entity_dof_end = entities_info.dof_end[i_e]
                n_dofs = entities_info.n_dofs[i_e]

                # Step 1: Solve w st. L^T @ w = y
                for i_d_ in range(n_dofs):
                    i_d = entity_dof_end - i_d_ - 1
                    out[i_d, i_b] = vec[i_d, i_b]
                    for j_d in range(i_d + 1, entity_dof_end):
                        out[i_d, i_b] -= rigid_global_info.mass_mat_L[j_d, i_d, i_b] * out[j_d, i_b]

                # Step 2: z = D^{-1} w
                for i_d in range(entity_dof_start, entity_dof_end):
                    out[i_d, i_b] *= rigid_global_info.mass_mat_D_inv[i_d, i_b]

                # Step 3: Solve x st. L @ x = z
                for i_d in range(entity_dof_start, entity_dof_end):
                    for j_d in range(entity_dof_start, i_d):
                        out[i_d, i_b] -= rigid_global_info.mass_mat_L[i_d, j_d, i_b] * out[j_d, i_b]


@ti.func
def func_solve_mass(
    vec: array_class.V_ANNOTATION,
    out: array_class.V_ANNOTATION,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    _B = out.shape[1]
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_b in range(_B):
        func_solve_mass_batched(
            vec,
            out,
            i_b,
            entities_info=entities_info,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
        )


@ti.kernel
def kernel_rigid_entity_inverse_kinematics(
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
    rigid_entity: ti.template(),
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
                rigid_global_info.qpos[i_q + rigid_entity._q_start, i_b] = init_qpos[i_b, i_q]

        for i_error in range(n_error_dims):
            rigid_entity._IK_err_pose_best[i_error, i_b] = 1e4

        solved = False
        for i_sample in range(max_samples):
            for _ in range(max_solver_iters):
                # run FK to update link states using current q
                func_forward_kinematics_entity(
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
                )
                # compute error
                solved = True
                for i_ee in range(n_links):
                    i_l_ee = links_idx[i_ee]

                    tgt_pos_i = ti.Vector([poss[i_ee, i_b, 0], poss[i_ee, i_b, 1], poss[i_ee, i_b, 2]])
                    err_pos_i = tgt_pos_i - links_state.pos[i_l_ee, i_b]
                    for k in range(3):
                        err_pos_i[k] *= pos_mask[k] * link_pos_mask[i_ee]
                    if err_pos_i.norm() > pos_tol:
                        solved = False

                    tgt_quat_i = ti.Vector(
                        [quats[i_ee, i_b, 0], quats[i_ee, i_b, 1], quats[i_ee, i_b, 2], quats[i_ee, i_b, 3]]
                    )
                    err_rot_i = gu.ti_quat_to_rotvec(
                        gu.ti_transform_quat_by_quat(gu.ti_inv_quat(links_state.quat[i_l_ee, i_b]), tgt_quat_i)
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
                        i_l_ee, i_b, ti.Vector.zero(gs.ti_float, 3), pos_mask, rot_mask
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

                for i in range(rigid_entity.n_dofs):  # IK_delta_qpos = IK_jacobian_T @ IK_vec
                    rigid_entity._IK_delta_qpos[i, i_b] = 0
                for i in range(n_dofs):
                    for j in range(n_error_dims):
                        i_ = dofs_idx[
                            i
                        ]  # NOTE: IK_delta_qpos uses the original indexing instead of the effective n_dofs
                        rigid_entity._IK_delta_qpos[i_, i_b] += (
                            rigid_entity._IK_jacobian_T[i, j, i_b] * rigid_entity._IK_vec[j, i_b]
                        )

                for i in range(rigid_entity.n_dofs):
                    rigid_entity._IK_delta_qpos[i, i_b] = ti.math.clamp(
                        rigid_entity._IK_delta_qpos[i, i_b], -max_step_size, max_step_size
                    )

                # update q
                func_integrate_dq_entity(
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
                func_forward_kinematics_entity(
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
                )
                solved = True
                for i_ee in range(n_links):
                    i_l_ee = links_idx[i_ee]

                    tgt_pos_i = ti.Vector([poss[i_ee, i_b, 0], poss[i_ee, i_b, 1], poss[i_ee, i_b, 2]])
                    err_pos_i = tgt_pos_i - links_state.pos[i_l_ee, i_b]
                    for k in range(3):
                        err_pos_i[k] *= pos_mask[k] * link_pos_mask[i_ee]
                    if err_pos_i.norm() > pos_tol:
                        solved = False

                    tgt_quat_i = ti.Vector(
                        [quats[i_ee, i_b, 0], quats[i_ee, i_b, 1], quats[i_ee, i_b, 2], quats[i_ee, i_b, 3]]
                    )
                    err_rot_i = gu.ti_quat_to_rotvec(
                        gu.ti_transform_quat_by_quat(gu.ti_inv_quat(links_state.quat[i_l_ee, i_b]), tgt_quat_i)
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
                    for _i_l in range(n_links_by_dofs):
                        i_l = links_idx_by_dofs[_i_l]
                        I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l

                        for i_j in range(links_info.joint_start[I_l], links_info.joint_end[I_l]):
                            I_j = [i_j, i_b] if ti.static(static_rigid_sim_config.batch_joints_info) else i_j

                            I_dof_start = (
                                [joints_info.dof_start[I_j], i_b]
                                if ti.static(static_rigid_sim_config.batch_dofs_info)
                                else joints_info.dof_start[I_j]
                            )
                            q_start = joints_info.q_start[I_j]
                            dof_limit = dofs_info.limit[I_dof_start]

                            if joints_info.type[I_j] == gs.JOINT_TYPE.FREE:
                                pass

                            elif (
                                joints_info.type[I_j] == gs.JOINT_TYPE.REVOLUTE
                                or joints_info.type[I_j] == gs.JOINT_TYPE.PRISMATIC
                            ):
                                if ti.math.isinf(dof_limit[0]) or ti.math.isinf(dof_limit[1]):
                                    pass
                                else:
                                    rigid_global_info.qpos[q_start, i_b] = dof_limit[0] + ti.random() * (
                                        dof_limit[1] - dof_limit[0]
                                    )
                else:
                    pass  # When respect_joint_limit=False, we can simply continue from the last solution

        # restore original qpos and link state
        for i_q in range(rigid_entity.n_qs):
            rigid_global_info.qpos[i_q + rigid_entity._q_start, i_b] = rigid_entity._IK_qpos_orig[i_q, i_b]
        func_forward_kinematics_entity(
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
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
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
        entities_info=entities_info,
        dofs_state=dofs_state,
        dofs_info=dofs_info,
        links_info=links_info,
        joints_info=joints_info,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
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
    func_forward_velocity(
        i_b,
        entities_info=entities_info,
        links_info=links_info,
        links_state=links_state,
        joints_info=joints_info,
        dofs_state=dofs_state,
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
    )


@ti.kernel
def kernel_step_1(
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
            )

    func_forward_dynamics(
        links_state=links_state,
        links_info=links_info,
        dofs_state=dofs_state,
        dofs_info=dofs_info,
        joints_info=joints_info,
        entities_info=entities_info,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
    )


@ti.func
def func_implicit_damping(
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):

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
            rigid_global_info._mass_mat_mask[i_e, i_b] = 0

        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
        for i_e, i_b in ti.ndrange(n_entities, _B):
            entity_dof_start = entities_info.dof_start[i_e]
            entity_dof_end = entities_info.dof_end[i_e]
            for i_d in range(entity_dof_start, entity_dof_end):
                I_d = [i_d, i_b] if ti.static(static_rigid_sim_config.batch_dofs_info) else i_d
                if dofs_info.damping[I_d] > gs.EPS:
                    rigid_global_info._mass_mat_mask[i_e, i_b] = 1
                if ti.static(static_rigid_sim_config.integrator != gs.integrator.Euler):
                    if (
                        (dofs_state.ctrl_mode[i_d, i_b] == gs.CTRL_MODE.POSITION)
                        or (dofs_state.ctrl_mode[i_d, i_b] == gs.CTRL_MODE.VELOCITY)
                    ) and dofs_info.kv[I_d] > gs.EPS:
                        rigid_global_info._mass_mat_mask[i_e, i_b] = 1

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
            rigid_global_info._mass_mat_mask[i_e, i_b] = 1


@ti.kernel
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
        func_hibernate(
            dofs_state=dofs_state,
            entities_state=entities_state,
            entities_info=entities_info,
            links_state=links_state,
            geoms_state=geoms_state,
            collider_state=collider_state,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
        )
        func_aggregate_awake_entities(
            entities_state=entities_state,
            entities_info=entities_info,
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
            )


@ti.kernel
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

    n_links = links_info.root_idx.shape[0]
    if ti.static(static_rigid_sim_config.use_hibernation):
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_l_ in range(rigid_global_info.n_awake_links[i_b]):
            i_l = rigid_global_info.awake_links[i_l_, i_b]

            links_state.COM[i_l, i_b].fill(0.0)
            links_state.mass_sum[i_l, i_b] = 0.0

        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_l_ in range(rigid_global_info.n_awake_links[i_b]):
            i_l = rigid_global_info.awake_links[i_l_, i_b]
            I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l

            mass = links_info.inertial_mass[I_l] + links_state.mass_shift[i_l, i_b]
            (
                links_state.i_pos[i_l, i_b],
                links_state.i_quat[i_l, i_b],
            ) = gu.ti_transform_pos_quat_by_trans_quat(
                links_info.inertial_pos[I_l] + links_state.i_pos_shift[i_l, i_b],
                links_info.inertial_quat[I_l],
                links_state.pos[i_l, i_b],
                links_state.quat[i_l, i_b],
            )

            i_r = links_info.root_idx[I_l]
            links_state.mass_sum[i_r, i_b] += mass
            links_state.COM[i_r, i_b] += mass * links_state.i_pos[i_l, i_b]

        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_l_ in range(rigid_global_info.n_awake_links[i_b]):
            i_l = rigid_global_info.awake_links[i_l_, i_b]
            I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l

            i_r = links_info.root_idx[I_l]
            if i_l == i_r:
                links_state.COM[i_l, i_b] = links_state.COM[i_l, i_b] / links_state.mass_sum[i_l, i_b]

        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_l_ in range(rigid_global_info.n_awake_links[i_b]):
            i_l = rigid_global_info.awake_links[i_l_, i_b]
            I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l

            i_r = links_info.root_idx[I_l]
            links_state.COM[i_l, i_b] = links_state.COM[i_r, i_b]

        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_l_ in range(rigid_global_info.n_awake_links[i_b]):
            i_l = rigid_global_info.awake_links[i_l_, i_b]
            I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l

            i_r = links_info.root_idx[I_l]
            links_state.i_pos[i_l, i_b] = links_state.i_pos[i_l, i_b] - links_state.COM[i_l, i_b]

            i_inertial = links_info.inertial_i[I_l]
            i_mass = links_info.inertial_mass[I_l] + links_state.mass_shift[i_l, i_b]
            (
                links_state.cinr_inertial[i_l, i_b],
                links_state.cinr_pos[i_l, i_b],
                links_state.cinr_quat[i_l, i_b],
                links_state.cinr_mass[i_l, i_b],
            ) = gu.ti_transform_inertia_by_trans_quat(
                i_inertial, i_mass, links_state.i_pos[i_l, i_b], links_state.i_quat[i_l, i_b]
            )

        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_l_ in range(rigid_global_info.n_awake_links[i_b]):
            i_l = rigid_global_info.awake_links[i_l_, i_b]
            I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l
            if links_info.n_dofs[I_l] == 0:
                continue

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
                    links_state.j_pos[i_l, i_b],
                    links_state.j_quat[i_l, i_b],
                ) = gu.ti_transform_pos_quat_by_trans_quat(links_info.pos[I_l], links_info.quat[I_l], p_pos, p_quat)

                for i_j in range(links_info.joint_start[I_l], links_info.joint_end[I_l]):
                    I_j = [i_j, i_b] if ti.static(static_rigid_sim_config.batch_joints_info) else i_j

                    (
                        links_state.j_pos[i_l, i_b],
                        links_state.j_quat[i_l, i_b],
                    ) = gu.ti_transform_pos_quat_by_trans_quat(
                        joints_info.pos[I_j],
                        gu.ti_identity_quat(),
                        links_state.j_pos[i_l, i_b],
                        links_state.j_quat[i_l, i_b],
                    )

        # cdof_fn
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_l_ in range(rigid_global_info.n_awake_links[i_b]):
            i_l = rigid_global_info.awake_links[i_l_, i_b]
            I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l
            if links_info.n_dofs[I_l] == 0:
                continue

            i_j = links_info.joint_start[I_l]
            I_j = [i_j, i_b] if ti.static(static_rigid_sim_config.batch_joints_info) else i_j
            joint_type = joints_info.type[I_j]

            if joint_type == gs.JOINT_TYPE.FREE:
                for i_d in range(links_info.dof_start[I_l], links_info.dof_end[I_l]):
                    I_d = [i_d, i_b] if ti.static(static_rigid_sim_config.batch_dofs_info) else i_d
                    dofs_state.cdof_vel[i_d, i_b] = dofs_info.motion_vel[I_d]
                    dofs_state.cdof_ang[i_d, i_b] = gu.ti_transform_by_quat(
                        dofs_info.motion_ang[I_d], links_state.j_quat[i_l, i_b]
                    )

                    offset_pos = links_state.COM[i_l, i_b] - links_state.j_pos[i_l, i_b]
                    (
                        dofs_state.cdof_ang[i_d, i_b],
                        dofs_state.cdof_vel[i_d, i_b],
                    ) = gu.ti_transform_motion_by_trans_quat(
                        dofs_state.cdof_ang[i_d, i_b],
                        dofs_state.cdof_vel[i_d, i_b],
                        offset_pos,
                        gu.ti_identity_quat(),
                    )

                    dofs_state.cdofvel_ang[i_d, i_b] = dofs_state.cdof_ang[i_d, i_b] * dofs_state.vel[i_d, i_b]
                    dofs_state.cdofvel_vel[i_d, i_b] = dofs_state.cdof_vel[i_d, i_b] * dofs_state.vel[i_d, i_b]

            elif joint_type == gs.JOINT_TYPE.FIXED:
                pass
            else:
                for i_d in range(links_info.dof_start[I_l], links_info.dof_end[I_l]):
                    I_d = [i_d, i_b] if ti.static(static_rigid_sim_config.batch_dofs_info) else i_d
                    motion_vel = dofs_info.motion_vel[I_d]
                    motion_ang = dofs_info.motion_ang[I_d]

                    dofs_state.cdof_ang[i_d, i_b] = gu.ti_transform_by_quat(motion_ang, links_state.j_quat[i_l, i_b])
                    dofs_state.cdof_vel[i_d, i_b] = gu.ti_transform_by_quat(motion_vel, links_state.j_quat[i_l, i_b])

                    offset_pos = links_state.COM[i_l, i_b] - links_state.j_pos[i_l, i_b]
                    (
                        dofs_state.cdof_ang[i_d, i_b],
                        dofs_state.cdof_vel[i_d, i_b],
                    ) = gu.ti_transform_motion_by_trans_quat(
                        dofs_state.cdof_ang[i_d, i_b],
                        dofs_state.cdof_vel[i_d, i_b],
                        offset_pos,
                        gu.ti_identity_quat(),
                    )

                    dofs_state.cdofvel_ang[i_d, i_b] = dofs_state.cdof_ang[i_d, i_b] * dofs_state.vel[i_d, i_b]
                    dofs_state.cdofvel_vel[i_d, i_b] = dofs_state.cdof_vel[i_d, i_b] * dofs_state.vel[i_d, i_b]
    else:
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_l in range(n_links):
            links_state.COM[i_l, i_b].fill(0.0)
            links_state.mass_sum[i_l, i_b] = 0.0

        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_l in range(n_links):
            I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l

            mass = links_info.inertial_mass[I_l] + links_state.mass_shift[i_l, i_b]
            (
                links_state.i_pos[i_l, i_b],
                links_state.i_quat[i_l, i_b],
            ) = gu.ti_transform_pos_quat_by_trans_quat(
                links_info.inertial_pos[I_l] + links_state.i_pos_shift[i_l, i_b],
                links_info.inertial_quat[I_l],
                links_state.pos[i_l, i_b],
                links_state.quat[i_l, i_b],
            )

            i_r = links_info.root_idx[I_l]
            links_state.mass_sum[i_r, i_b] += mass
            links_state.COM[i_r, i_b] += mass * links_state.i_pos[i_l, i_b]

        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_l in range(n_links):
            I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l

            i_r = links_info.root_idx[I_l]
            if i_l == i_r:
                if links_state.mass_sum[i_l, i_b] > 0.0:
                    links_state.COM[i_l, i_b] = links_state.COM[i_l, i_b] / links_state.mass_sum[i_l, i_b]

        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_l in range(n_links):
            I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l

            i_r = links_info.root_idx[I_l]
            links_state.COM[i_l, i_b] = links_state.COM[i_r, i_b]

        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_l in range(n_links):
            I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l

            i_r = links_info.root_idx[I_l]
            links_state.i_pos[i_l, i_b] = links_state.i_pos[i_l, i_b] - links_state.COM[i_l, i_b]

            i_inertial = links_info.inertial_i[I_l]
            i_mass = links_info.inertial_mass[I_l] + links_state.mass_shift[i_l, i_b]
            (
                links_state.cinr_inertial[i_l, i_b],
                links_state.cinr_pos[i_l, i_b],
                links_state.cinr_quat[i_l, i_b],
                links_state.cinr_mass[i_l, i_b],
            ) = gu.ti_transform_inertia_by_trans_quat(
                i_inertial, i_mass, links_state.i_pos[i_l, i_b], links_state.i_quat[i_l, i_b]
            )

        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_l in range(n_links):
            I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l
            if links_info.n_dofs[I_l] == 0:
                continue

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
                    links_state.j_pos[i_l, i_b],
                    links_state.j_quat[i_l, i_b],
                ) = gu.ti_transform_pos_quat_by_trans_quat(links_info.pos[I_l], links_info.quat[I_l], p_pos, p_quat)

                for i_j in range(links_info.joint_start[I_l], links_info.joint_end[I_l]):
                    I_j = [i_j, i_b] if ti.static(static_rigid_sim_config.batch_joints_info) else i_j

                    (
                        links_state.j_pos[i_l, i_b],
                        links_state.j_quat[i_l, i_b],
                    ) = gu.ti_transform_pos_quat_by_trans_quat(
                        joints_info.pos[I_j],
                        gu.ti_identity_quat(),
                        links_state.j_pos[i_l, i_b],
                        links_state.j_quat[i_l, i_b],
                    )

        # cdof_fn
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_l in range(n_links):
            I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l
            if links_info.n_dofs[I_l] == 0:
                continue

            for i_j in range(links_info.joint_start[I_l], links_info.joint_end[I_l]):
                offset_pos = links_state.COM[i_l, i_b] - joints_state.xanchor[i_j, i_b]
                I_j = [i_j, i_b] if ti.static(static_rigid_sim_config.batch_joints_info) else i_j
                joint_type = joints_info.type[I_j]

                dof_start = joints_info.dof_start[I_j]

                if joint_type == gs.JOINT_TYPE.REVOLUTE:
                    dofs_state.cdof_ang[dof_start, i_b] = joints_state.xaxis[i_j, i_b]
                    dofs_state.cdof_vel[dof_start, i_b] = joints_state.xaxis[i_j, i_b].cross(offset_pos)
                elif joint_type == gs.JOINT_TYPE.PRISMATIC:
                    dofs_state.cdof_ang[dof_start, i_b] = ti.Vector.zero(gs.ti_float, 3)
                    dofs_state.cdof_vel[dof_start, i_b] = joints_state.xaxis[i_j, i_b]
                elif joint_type == gs.JOINT_TYPE.SPHERICAL:
                    xmat_T = gu.ti_quat_to_R(links_state.quat[i_l, i_b]).transpose()
                    for i in ti.static(range(3)):
                        dofs_state.cdof_ang[i + dof_start, i_b] = xmat_T[i, :]
                        dofs_state.cdof_vel[i + dof_start, i_b] = xmat_T[i, :].cross(offset_pos)
                elif joint_type == gs.JOINT_TYPE.FREE:
                    for i in ti.static(range(3)):
                        dofs_state.cdof_ang[i + dof_start, i_b] = ti.Vector.zero(gs.ti_float, 3)
                        dofs_state.cdof_vel[i + dof_start, i_b] = ti.Vector.zero(gs.ti_float, 3)
                        dofs_state.cdof_vel[i + dof_start, i_b][i] = 1.0

                    xmat_T = gu.ti_quat_to_R(links_state.quat[i_l, i_b]).transpose()
                    for i in ti.static(range(3)):
                        dofs_state.cdof_ang[i + dof_start + 3, i_b] = xmat_T[i, :]
                        dofs_state.cdof_vel[i + dof_start + 3, i_b] = xmat_T[i, :].cross(offset_pos)

                for i_d in range(dof_start, joints_info.dof_end[I_j]):
                    dofs_state.cdofvel_ang[i_d, i_b] = dofs_state.cdof_ang[i_d, i_b] * dofs_state.vel[i_d, i_b]
                    dofs_state.cdofvel_vel[i_d, i_b] = dofs_state.cdof_vel[i_d, i_b] * dofs_state.vel[i_d, i_b]


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
    n_entities = entities_info.n_links.shape[0]
    if ti.static(static_rigid_sim_config.use_hibernation):
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_e_ in range(rigid_global_info.n_awake_entities[i_b]):
            i_e = rigid_global_info.awake_entities[i_e_, i_b]
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
    else:
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_e in range(n_entities):
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
    n_entities = entities_info.n_links.shape[0]
    if ti.static(static_rigid_sim_config.use_hibernation):
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_e_ in range(rigid_global_info.n_awake_entities[i_b]):
            i_e = rigid_global_info.awake_entities[i_e_, i_b]
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
    else:
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_e in range(n_entities):
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


@ti.kernel
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
    for i_l in range(entities_info.link_start[i_e], entities_info.link_end[i_e]):
        I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l

        pos = links_info.pos[I_l]
        quat = links_info.quat[I_l]
        if links_info.parent_idx[I_l] != -1:
            parent_pos = links_state.pos[links_info.parent_idx[I_l], i_b]
            parent_quat = links_state.quat[links_info.parent_idx[I_l], i_b]
            pos = parent_pos + gu.ti_transform_by_quat(pos, parent_quat)
            quat = gu.ti_transform_quat_by_quat(quat, parent_quat)

        for i_j in range(links_info.joint_start[I_l], links_info.joint_end[I_l]):
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

                joints_state.xanchor[i_j, i_b] = gu.ti_transform_by_quat(joints_info.pos[I_j], quat) + pos
                joints_state.xaxis[i_j, i_b] = gu.ti_transform_by_quat(axis, quat)

            if joint_type == gs.JOINT_TYPE.FREE:
                pos = ti.Vector(
                    [
                        rigid_global_info.qpos[q_start, i_b],
                        rigid_global_info.qpos[q_start + 1, i_b],
                        rigid_global_info.qpos[q_start + 2, i_b],
                    ],
                    dt=gs.ti_float,
                )
                quat = ti.Vector(
                    [
                        rigid_global_info.qpos[q_start + 3, i_b],
                        rigid_global_info.qpos[q_start + 4, i_b],
                        rigid_global_info.qpos[q_start + 5, i_b],
                        rigid_global_info.qpos[q_start + 6, i_b],
                    ],
                    dt=gs.ti_float,
                )
                quat = gu.ti_normalize(quat)
                xyz = gu.ti_quat_to_xyz(quat)
                for i in ti.static(range(3)):
                    dofs_state.pos[dof_start + i, i_b] = pos[i]
                    dofs_state.pos[dof_start + 3 + i, i_b] = xyz[i]
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
                xyz = gu.ti_quat_to_xyz(qloc)
                for i in ti.static(range(3)):
                    dofs_state.pos[dof_start + i, i_b] = xyz[i]
                quat = gu.ti_transform_quat_by_quat(qloc, quat)
                pos = joints_state.xanchor[i_j, i_b] - gu.ti_transform_by_quat(joints_info.pos[I_j], quat)
            elif joint_type == gs.JOINT_TYPE.REVOLUTE:
                axis = dofs_info.motion_ang[I_d]
                dofs_state.pos[dof_start, i_b] = (
                    rigid_global_info.qpos[q_start, i_b] - rigid_global_info.qpos0[q_start, i_b]
                )
                qloc = gu.ti_rotvec_to_quat(axis * dofs_state.pos[dof_start, i_b])
                quat = gu.ti_transform_quat_by_quat(qloc, quat)
                pos = joints_state.xanchor[i_j, i_b] - gu.ti_transform_by_quat(joints_info.pos[I_j], quat)
            else:  # joint_type == gs.JOINT_TYPE.PRISMATIC:
                dofs_state.pos[dof_start, i_b] = (
                    rigid_global_info.qpos[q_start, i_b] - rigid_global_info.qpos0[q_start, i_b]
                )
                pos = pos + joints_state.xaxis[i_j, i_b] * dofs_state.pos[dof_start, i_b]

        # Skip link pose update for fixed root links to let users manually overwrite them
        if not (links_info.parent_idx[I_l] == -1 and links_info.is_fixed[I_l]):
            links_state.pos[i_l, i_b] = pos
            links_state.quat[i_l, i_b] = quat


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
    for i_l in range(entities_info.link_start[i_e], entities_info.link_end[i_e]):
        I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l

        cvel_vel = ti.Vector.zero(gs.ti_float, 3)
        cvel_ang = ti.Vector.zero(gs.ti_float, 3)
        if links_info.parent_idx[I_l] != -1:
            cvel_vel = links_state.cd_vel[links_info.parent_idx[I_l], i_b]
            cvel_ang = links_state.cd_ang[links_info.parent_idx[I_l], i_b]

        for i_j in range(links_info.joint_start[I_l], links_info.joint_end[I_l]):
            I_j = [i_j, i_b] if ti.static(static_rigid_sim_config.batch_joints_info) else i_j
            joint_type = joints_info.type[I_j]
            q_start = joints_info.q_start[I_j]
            dof_start = joints_info.dof_start[I_j]

            if joint_type == gs.JOINT_TYPE.FREE:
                for i_3 in ti.static(range(3)):
                    cvel_vel = (
                        cvel_vel + dofs_state.cdof_vel[dof_start + i_3, i_b] * dofs_state.vel[dof_start + i_3, i_b]
                    )
                    cvel_ang = (
                        cvel_ang + dofs_state.cdof_ang[dof_start + i_3, i_b] * dofs_state.vel[dof_start + i_3, i_b]
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
                        cvel_ang,
                        cvel_vel,
                        dofs_state.cdof_ang[dof_start + i_3 + 3, i_b],
                        dofs_state.cdof_vel[dof_start + i_3 + 3, i_b],
                    )

                for i_3 in ti.static(range(3)):
                    cvel_vel = (
                        cvel_vel
                        + dofs_state.cdof_vel[dof_start + i_3 + 3, i_b] * dofs_state.vel[dof_start + i_3 + 3, i_b]
                    )
                    cvel_ang = (
                        cvel_ang
                        + dofs_state.cdof_ang[dof_start + i_3 + 3, i_b] * dofs_state.vel[dof_start + i_3 + 3, i_b]
                    )

            else:
                for i_d in range(dof_start, joints_info.dof_end[I_j]):
                    dofs_state.cdofd_ang[i_d, i_b], dofs_state.cdofd_vel[i_d, i_b] = gu.motion_cross_motion(
                        cvel_ang,
                        cvel_vel,
                        dofs_state.cdof_ang[i_d, i_b],
                        dofs_state.cdof_vel[i_d, i_b],
                    )
                for i_d in range(dof_start, joints_info.dof_end[I_j]):
                    cvel_vel = cvel_vel + dofs_state.cdof_vel[i_d, i_b] * dofs_state.vel[i_d, i_b]
                    cvel_ang = cvel_ang + dofs_state.cdof_ang[i_d, i_b] * dofs_state.vel[i_d, i_b]

        links_state.cd_vel[i_l, i_b] = cvel_vel
        links_state.cd_ang[i_l, i_b] = cvel_ang


@ti.kernel
def kernel_update_geoms(
    envs_idx: ti.types.ndarray(),
    entities_info: array_class.EntitiesInfo,
    geoms_info: array_class.GeomsInfo,
    geoms_state: array_class.GeomsState,
    links_state: array_class.LinksState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
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
):
    """
    NOTE: this only update geom pose, not its verts and else.
    """
    n_geoms = geoms_info.pos.shape[0]
    if ti.static(static_rigid_sim_config.use_hibernation):
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
        for i_e_ in range(rigid_global_info.n_awake_entities[i_b]):
            i_e = rigid_global_info.awake_entities[i_e_, i_b]
            for i_g in range(entities_info.geom_start[i_e], entities_info.geom_end[i_e]):
                (
                    geoms_state.pos[i_g, i_b],
                    geoms_state.quat[i_g, i_b],
                ) = gu.ti_transform_pos_quat_by_trans_quat(
                    geoms_info.pos[i_g],
                    geoms_info.quat[i_g],
                    links_state.pos[geoms_info.link_idx[i_g], i_b],
                    links_state.quat[geoms_info.link_idx[i_g], i_b],
                )

                geoms_state.verts_updated[i_g, i_b] = 0
    else:
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
        for i_g in range(n_geoms):
            (
                geoms_state.pos[i_g, i_b],
                geoms_state.quat[i_g, i_b],
            ) = gu.ti_transform_pos_quat_by_trans_quat(
                geoms_info.pos[i_g],
                geoms_info.quat[i_g],
                links_state.pos[geoms_info.link_idx[i_g], i_b],
                links_state.quat[geoms_info.link_idx[i_g], i_b],
            )

            geoms_state.verts_updated[i_g, i_b] = 0


@ti.kernel
def kernel_update_verts_for_geom(
    i_g: ti.i32,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    free_verts_state: array_class.FreeVertsState,
    fixed_verts_state: array_class.FixedVertsState,
):
    _B = geoms_state.verts_updated.shape[1]
    for i_b in range(_B):
        func_update_verts_for_geom(i_g, i_b, geoms_state, geoms_info, verts_info, free_verts_state, fixed_verts_state)


@ti.func
def func_update_verts_for_geom(
    i_g: ti.i32,
    i_b: ti.i32,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    free_verts_state: array_class.FreeVertsState,
    fixed_verts_state: array_class.FixedVertsState,
):
    if not geoms_state.verts_updated[i_g, i_b]:
        if geoms_info.is_free[i_g]:
            for i_v in range(geoms_info.vert_start[i_g], geoms_info.vert_end[i_g]):
                verts_state_idx = verts_info.verts_state_idx[i_v]
                free_verts_state.pos[verts_state_idx, i_b] = gu.ti_transform_by_trans_quat(
                    verts_info.init_pos[i_v], geoms_state.pos[i_g, i_b], geoms_state.quat[i_g, i_b]
                )
            geoms_state.verts_updated[i_g, i_b] = 1
        elif i_b == 0:
            for i_v in range(geoms_info.vert_start[i_g], geoms_info.vert_end[i_g]):
                verts_state_idx = verts_info.verts_state_idx[i_v]
                fixed_verts_state.pos[verts_state_idx] = gu.ti_transform_by_trans_quat(
                    verts_info.init_pos[i_v], geoms_state.pos[i_g, i_b], geoms_state.quat[i_g, i_b]
                )
            geoms_state.verts_updated[i_g, 0] = 1


@ti.func
def func_update_all_verts(self):
    ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
    for i_v, i_b in ti.ndrange(self.n_verts, self._B):
        g_pos = self.geoms_state.pos[self.verts_info.geom_idx[i_v], i_b]
        g_quat = self.geoms_state.quat[self.verts_info.geom_idx[i_v], i_b]
        verts_state_idx = self.verts_info.verts_state_idx[i_v]
        if self.verts_info.is_free[i_v]:
            self.free_verts_state.pos[verts_state_idx, i_b] = gu.ti_transform_by_trans_quat(
                self.verts_info.init_pos[i_v], g_pos, g_quat
            )
        elif i_b == 0:
            self.fixed_verts_state.pos[verts_state_idx] = gu.ti_transform_by_trans_quat(
                self.verts_info.init_pos[i_v], g_pos, g_quat
            )


@ti.kernel
def kernel_update_geom_aabbs(
    geoms_state: array_class.GeomsState,
    geoms_init_AABB: array_class.GeomsInitAABB,
    static_rigid_sim_config: ti.template(),
):
    n_geoms = geoms_state.pos.shape[0]
    _B = geoms_state.pos.shape[1]
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
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


@ti.kernel
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
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_g, i_b in ti.ndrange(n_vgeoms, _B):
        vgeoms_state.pos[i_g, i_b], vgeoms_state.quat[i_g, i_b] = gu.ti_transform_pos_quat_by_trans_quat(
            vgeoms_info.pos[i_g],
            vgeoms_info.quat[i_g],
            links_state.pos[vgeoms_info.link_idx[i_g], i_b],
            links_state.quat[vgeoms_info.link_idx[i_g], i_b],
        )


@ti.func
def func_hibernate(
    dofs_state,
    entities_state,
    entities_info,
    links_state,
    geoms_state,
    collider_state,
    rigid_global_info,
    static_rigid_sim_config: ti.template(),
):

    n_entities = entities_state.hibernated.shape[0]
    _B = entities_state.hibernated.shape[1]
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_e, i_b in ti.ndrange(n_entities, _B):
        if (
            not entities_state.hibernated[i_e, i_b] and entities_info.n_dofs[i_e] > 0
        ):  # We do not hibernate fixed entity
            hibernate = True
            for i_d in range(entities_info.dof_start[i_e], entities_info.dof_end[i_e]):
                if (
                    ti.abs(dofs_state.acc[i_d, i_b]) > static_rigid_sim_config.hibernation_thresh_acc
                    or ti.abs(dofs_state.vel[i_d, i_b]) > static_rigid_sim_config.hibernation_thresh_vel
                ):
                    hibernate = False
                    break

            if hibernate:
                func_hibernate_entity(
                    i_e,
                    i_b,
                    entities_state=entities_state,
                    entities_info=entities_info,
                    dofs_state=dofs_state,
                    links_state=links_state,
                    geoms_state=geoms_state,
                )
            else:
                # update collider sort_buffer
                for i_g in range(entities_info.geom_start[i_e], entities_info.geom_end[i_e]):
                    collider_state.sort_buffer.value[geoms_state.min_buffer_idx[i_g, i_b], i_b] = geoms_state.aabb_min[
                        i_g, i_b
                    ][0]
                    collider_state.sort_buffer.value[geoms_state.max_buffer_idx[i_g, i_b], i_b] = geoms_state.aabb_max[
                        i_g, i_b
                    ][0]


@ti.func
def func_aggregate_awake_entities(
    entities_state,
    entities_info,
    rigid_global_info,
    static_rigid_sim_config: ti.template(),
):

    n_entities = entities_state.hibernated.shape[0]
    _B = entities_state.hibernated.shape[1]
    rigid_global_info.n_awake_entities.fill(0)
    rigid_global_info.n_awake_links.fill(0)
    rigid_global_info.n_awake_dofs.fill(0)
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_e, i_b in ti.ndrange(n_entities, _B):
        if entities_state.hibernated[i_e, i_b] or entities_info.n_dofs[i_e] == 0:
            continue
        n_awake_entities = ti.atomic_add(rigid_global_info.n_awake_entities[i_b], 1)
        rigid_global_info.awake_entities[n_awake_entities, i_b] = i_e

        for i_d in range(entities_info.dof_start[i_e], entities_info.dof_end[i_e]):
            n_awake_dofs = ti.atomic_add(rigid_global_info.n_awake_dofs[i_b], 1)
            rigid_global_info.awake_dofs[n_awake_dofs, i_b] = i_d

        for i_l in range(entities_info.link_start[i_e], entities_info.link_end[i_e]):
            n_awake_links = ti.atomic_add(rigid_global_info.n_awake_links[i_b], 1)
            rigid_global_info.awake_links[n_awake_links, i_b] = i_l


@ti.func
def func_hibernate_entity(
    i_e,
    i_b,
    entities_state,
    entities_info,
    dofs_state,
    links_state,
    geoms_state,
):

    entities_state.hibernated[i_e, i_b] = True

    for i_d in range(entities_info.dof_start[i_e], entities_info.dof_end[i_e]):
        dofs_state.hibernated[i_d, i_b] = True
        dofs_state.vel[i_d, i_b] = 0.0
        dofs_state.acc[i_d, i_b] = 0.0

    for i_l in range(entities_info.link_start[i_e], entities_info.link_end[i_e]):
        links_state.hibernated[i_l, i_b] = True

    for i_g in range(entities_info.geom_start[i_e], entities_info.geom_end[i_e]):
        geoms_state.hibernated[i_g, i_b] = True


@ti.func
def func_wakeup_entity(
    i_e,
    i_b,
    entities_state: array_class.EntitiesState,
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    links_state: array_class.LinksState,
    rigid_global_info: array_class.RigidGlobalInfo,
):
    if entities_state.hibernated[i_e, i_b]:
        entities_state.hibernated[i_e, i_b] = False
        n_awake_entities = ti.atomic_add(rigid_global_info.n_awake_entities[i_b], 1)
        rigid_global_info.awake_entities[n_awake_entities, i_b] = i_e

        for i_d in range(entities_info.dof_start[i_e], entities_info.dof_end[i_e]):
            dofs_state.hibernated[i_d, i_b] = False
            n_awake_dofs = ti.atomic_add(rigid_global_info.n_awake_dofs[i_b], 1)
            rigid_global_info.awake_dofs[n_awake_dofs, i_b] = i_d

        for i_l in range(entities_info.link_start[i_e], entities_info.link_end[i_e]):
            links_state.hibernated[i_l, i_b] = False
            n_awake_links = ti.atomic_add(rigid_global_info.n_awake_links[i_b], 1)
            rigid_global_info.awake_links[n_awake_links, i_b] = i_l


@ti.kernel
def kernel_apply_links_external_force(
    force: ti.types.ndarray(),
    links_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    ref: ti.template(),
    local: ti.template(),
    links_state: array_class.LinksState,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_l_, i_b_ in ti.ndrange(links_idx.shape[0], envs_idx.shape[0]):
        force_i = ti.Vector([force[i_b_, i_l_, 0], force[i_b_, i_l_, 1], force[i_b_, i_l_, 2]], dt=gs.ti_float)
        func_apply_link_external_force(force_i, links_idx[i_l_], envs_idx[i_b_], ref, local, links_state)


@ti.kernel
def kernel_apply_links_external_torque(
    torque: ti.types.ndarray(),
    links_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    ref: ti.template(),
    local: ti.template(),
    links_state: array_class.LinksState,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_l_, i_b_ in ti.ndrange(links_idx.shape[0], envs_idx.shape[0]):
        torque_i = ti.Vector([torque[i_b_, i_l_, 0], torque[i_b_, i_l_, 1], torque[i_b_, i_l_, 2]], dt=gs.ti_float)
        func_apply_link_external_torque(torque_i, links_idx[i_l_], envs_idx[i_b_], ref, local, links_state)


@ti.func
def func_apply_external_force(pos, force, link_idx, env_idx, links_state: array_class.LinksState):
    torque = (pos - links_state.COM[link_idx, env_idx]).cross(force)
    links_state.cfrc_applied_ang[link_idx, env_idx] -= torque
    links_state.cfrc_applied_vel[link_idx, env_idx] -= force


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
        torque = (links_state.pos[link_idx, env_idx] - links_state.COM[link_idx, env_idx]).cross(force)

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

    if ti.static(static_rigid_sim_config.use_hibernation):
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
        for i_b in range(_B):
            for i_l_ in range(rigid_global_info.n_awake_links[i_b]):
                i_l = rigid_global_info.awake_links[i_l_, i_b]
                links_state.cfrc_applied_ang[i_l, i_b] = ti.Vector.zero(gs.ti_float, 3)
                links_state.cfrc_applied_vel[i_l, i_b] = ti.Vector.zero(gs.ti_float, 3)
    else:
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
        for i_l, i_b in ti.ndrange(n_links, _B):
            links_state.cfrc_applied_ang[i_l, i_b] = ti.Vector.zero(gs.ti_float, 3)
            links_state.cfrc_applied_vel[i_l, i_b] = ti.Vector.zero(gs.ti_float, 3)


@ti.func
def func_torque_and_passive_force(
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    links_info: array_class.LinksInfo,
    joints_info: array_class.JointsInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    n_entities = entities_info.n_links.shape[0]
    _B = dofs_state.ctrl_mode.shape[1]
    n_dofs = dofs_state.ctrl_mode.shape[0]
    n_links = links_info.root_idx.shape[0]

    # compute force based on each dof's ctrl mode
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_e, i_b in ti.ndrange(n_entities, _B):
        wakeup = False
        for i_l in range(entities_info.link_start[i_e], entities_info.link_end[i_e]):
            I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l
            if links_info.n_dofs[I_l] == 0:
                continue

            i_j = links_info.joint_start[I_l]
            I_j = [i_j, i_b] if ti.static(static_rigid_sim_config.batch_joints_info) else i_j
            joint_type = joints_info.type[I_j]

            for i_d in range(links_info.dof_start[I_l], links_info.dof_end[I_l]):
                I_d = [i_d, i_b] if ti.static(static_rigid_sim_config.batch_dofs_info) else i_d
                force = gs.ti_float(0.0)
                if dofs_state.ctrl_mode[i_d, i_b] == gs.CTRL_MODE.FORCE:
                    force = dofs_state.ctrl_force[i_d, i_b]
                elif dofs_state.ctrl_mode[i_d, i_b] == gs.CTRL_MODE.VELOCITY:
                    force = dofs_info.kv[I_d] * (dofs_state.ctrl_vel[i_d, i_b] - dofs_state.vel[i_d, i_b])
                elif dofs_state.ctrl_mode[i_d, i_b] == gs.CTRL_MODE.POSITION and not (
                    joint_type == gs.JOINT_TYPE.FREE and i_d >= links_info.dof_start[I_l] + 3
                ):
                    force = (
                        dofs_info.kp[I_d] * (dofs_state.ctrl_pos[i_d, i_b] - dofs_state.pos[i_d, i_b])
                        - dofs_info.kv[I_d] * dofs_state.vel[i_d, i_b]
                    )

                dofs_state.qf_applied[i_d, i_b] = ti.math.clamp(
                    force,
                    dofs_info.force_range[I_d][0],
                    dofs_info.force_range[I_d][1],
                )

                if ti.abs(force) > gs.EPS:
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
                rotvec = gu.ti_quat_to_rotvec(q_diff)

                for j in ti.static(range(3)):
                    i_d = dof_start + 3 + j
                    I_d = [i_d, i_b] if ti.static(static_rigid_sim_config.batch_dofs_info) else i_d
                    force = dofs_info.kp[I_d] * rotvec[j] - dofs_info.kv[I_d] * dofs_state.vel[i_d, i_b]

                    dofs_state.qf_applied[i_d, i_b] = ti.math.clamp(
                        force, dofs_info.force_range[I_d][0], dofs_info.force_range[I_d][1]
                    )

                    if ti.abs(force) > gs.EPS:
                        wakeup = True

        if ti.static(static_rigid_sim_config.use_hibernation):
            if wakeup:
                func_wakeup_entity(i_e, i_b)

    if ti.static(static_rigid_sim_config.use_hibernation):
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(rigid_global_info._B):
            for i_d_ in range(rigid_global_info.n_awake_dofs[i_b]):
                i_d = rigid_global_info.awake_dofs[i_d_, i_b]
                I_d = [i_d, i_b] if ti.static(static_rigid_sim_config.batch_dofs_info) else i_d

                dofs_state.qf_passive[i_d, i_b] = -dofs_info.damping[I_d] * dofs_state.vel[i_d, i_b]

        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(rigid_global_info._B):
            for i_l_ in range(rigid_global_info.n_awake_links[i_b]):
                i_l = rigid_global_info.awake_links[i_l_, i_b]
                I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l
                if links_info.n_dofs[I_l] == 0:
                    continue

                i_j = links_info.joint_start[I_l]
                I_j = [i_j, i_b] if ti.static(static_rigid_sim_config.batch_joints_info) else i_j
                joint_type = joints_info.type[I_j]

                if joint_type != gs.JOINT_TYPE.FREE and joint_type != gs.JOINT_TYPE.FIXED:
                    dof_start = links_info.dof_start[I_l]
                    q_start = links_info.q_start[I_l]
                    q_end = links_info.q_end[I_l]

                    for j_d in range(q_end - q_start):
                        I_d = (
                            [dof_start + j_d, i_b]
                            if ti.static(static_rigid_sim_config.batch_dofs_info)
                            else dof_start + j_d
                        )
                        dofs_state.qf_passive[dof_start + j_d, i_b] += (
                            -rigid_global_info.qpos[q_start + j_d, i_b] * dofs_info.stiffness[I_d]
                        )
    else:
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_d, i_b in ti.ndrange(n_dofs, _B):
            I_d = [i_d, i_b] if ti.static(static_rigid_sim_config.batch_dofs_info) else i_d
            dofs_state.qf_passive[i_d, i_b] = -dofs_info.damping[I_d] * dofs_state.vel[i_d, i_b]

        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_l, i_b in ti.ndrange(n_links, _B):
            I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l
            if links_info.n_dofs[I_l] == 0:
                continue

            i_j = links_info.joint_start[I_l]
            I_j = [i_j, i_b] if ti.static(static_rigid_sim_config.batch_joints_info) else i_j
            joint_type = joints_info.type[I_j]

            if joint_type != gs.JOINT_TYPE.FREE and joint_type != gs.JOINT_TYPE.FIXED:
                dof_start = links_info.dof_start[I_l]
                q_start = links_info.q_start[I_l]
                q_end = links_info.q_end[I_l]

                for j_d in range(q_end - q_start):
                    I_d = (
                        [dof_start + j_d, i_b]
                        if ti.static(static_rigid_sim_config.batch_dofs_info)
                        else dof_start + j_d
                    )
                    dofs_state.qf_passive[dof_start + j_d, i_b] += (
                        -rigid_global_info.qpos[q_start + j_d, i_b] * dofs_info.stiffness[I_d]
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
    _B = dofs_state.ctrl_mode.shape[1]
    n_links = links_info.root_idx.shape[0]
    n_entities = entities_info.n_links.shape[0]

    if ti.static(static_rigid_sim_config.use_hibernation):
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(_B):
            for i_e_ in range(rigid_global_info.n_awake_entities[i_b]):
                i_e = rigid_global_info.awake_entities[i_e_, i_b]
                for i_l in range(entities_info.link_start[i_e], entities_info.link_end[i_e]):
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

                    for i_d in range(links_info.dof_start[I_l], links_info.dof_end[I_l]):
                        local_cdd_vel = dofs_state.cdofd_vel[i_d, i_b] * dofs_state.vel[i_d, i_b]
                        local_cdd_ang = dofs_state.cdofd_ang[i_d, i_b] * dofs_state.vel[i_d, i_b]
                        links_state.cdd_vel[i_l, i_b] = links_state.cdd_vel[i_l, i_b] + local_cdd_vel
                        links_state.cdd_ang[i_l, i_b] = links_state.cdd_ang[i_l, i_b] + local_cdd_ang
                        if ti.static(update_cacc):
                            links_state.cacc_lin[i_l, i_b] = (
                                links_state.cacc_lin[i_l, i_b]
                                + local_cdd_vel
                                + dofs_state.cdof_vel[i_d, i_b] * dofs_state.acc[i_d, i_b]
                            )
                            links_state.cacc_ang[i_l, i_b] = (
                                links_state.cacc_ang[i_l, i_b]
                                + local_cdd_ang
                                + dofs_state.cdof_ang[i_d, i_b] * dofs_state.acc[i_d, i_b]
                            )
    else:
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_e, i_b in ti.ndrange(n_entities, _B):
            for i_l in range(entities_info.link_start[i_e], entities_info.link_end[i_e]):
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

                for i_d in range(links_info.dof_start[I_l], links_info.dof_end[I_l]):
                    # cacc = cacc_parent + cdofdot * qvel + cdof * qacc
                    local_cdd_vel = dofs_state.cdofd_vel[i_d, i_b] * dofs_state.vel[i_d, i_b]
                    local_cdd_ang = dofs_state.cdofd_ang[i_d, i_b] * dofs_state.vel[i_d, i_b]
                    links_state.cdd_vel[i_l, i_b] = links_state.cdd_vel[i_l, i_b] + local_cdd_vel
                    links_state.cdd_ang[i_l, i_b] = links_state.cdd_ang[i_l, i_b] + local_cdd_ang
                    if ti.static(update_cacc):
                        links_state.cacc_lin[i_l, i_b] = (
                            links_state.cacc_lin[i_l, i_b]
                            + local_cdd_vel
                            + dofs_state.cdof_vel[i_d, i_b] * dofs_state.acc[i_d, i_b]
                        )
                        links_state.cacc_ang[i_l, i_b] = (
                            links_state.cacc_ang[i_l, i_b]
                            + local_cdd_ang
                            + dofs_state.cdof_ang[i_d, i_b] * dofs_state.acc[i_d, i_b]
                        )


@ti.func
def func_update_force(
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    _B = links_state.pos.shape[1]
    n_links = links_info.root_idx.shape[0]
    n_entities = entities_info.n_links.shape[0]

    if ti.static(static_rigid_sim_config.use_hibernation):
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(_B):
            for i_l_ in range(rigid_global_info.n_awake_links[i_b]):
                i_l = rigid_global_info.awake_links[i_l_, i_b]

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
                f2_ang, f2_vel = gu.motion_cross_force(
                    links_state.cd_ang[i_l, i_b], links_state.cd_vel[i_l, i_b], f2_ang, f2_vel
                )

                links_state.cfrc_vel[i_l, i_b] = f1_vel + f2_vel + links_state.cfrc_applied_vel[i_l, i_b]
                links_state.cfrc_ang[i_l, i_b] = f1_ang + f2_ang + links_state.cfrc_applied_ang[i_l, i_b]

        for i_b in range(_B):
            for i_e_ in range(rigid_global_info.n_awake_entities[i_b]):
                i_e = rigid_global_info.awake_entities[i_e_, i_b]
                for i_l in range(entities_info.link_end[i_e] - 1 - entities_info.link_start[i_e]):
                    i_l = entities_info.link_end[i_e] - 1 - i
                    I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l
                    i_p = links_info.parent_idx[I_l]
                    if i_p != -1:
                        links_state.cfrc_vel[i_p, i_b] = links_state.cfrc_vel[i_p, i_b] + links_state.cfrc_vel[i_l, i_b]
                        links_state.cfrc_ang[i_p, i_b] = links_state.cfrc_ang[i_p, i_b] + links_state.cfrc_ang[i_l, i_b]
    else:
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_l, i_b in ti.ndrange(n_links, _B):
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
            f2_ang, f2_vel = gu.motion_cross_force(
                links_state.cd_ang[i_l, i_b], links_state.cd_vel[i_l, i_b], f2_ang, f2_vel
            )

            links_state.cfrc_vel[i_l, i_b] = f1_vel + f2_vel + links_state.cfrc_applied_vel[i_l, i_b]
            links_state.cfrc_ang[i_l, i_b] = f1_ang + f2_ang + links_state.cfrc_applied_ang[i_l, i_b]

        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_e, i_b in ti.ndrange(n_entities, _B):
            for i in range(entities_info.n_links[i_e]):
                i_l = entities_info.link_end[i_e] - 1 - i
                I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l
                i_p = links_info.parent_idx[I_l]
                if i_p != -1:
                    links_state.cfrc_vel[i_p, i_b] = links_state.cfrc_vel[i_p, i_b] + links_state.cfrc_vel[i_l, i_b]
                    links_state.cfrc_ang[i_p, i_b] = links_state.cfrc_ang[i_p, i_b] + links_state.cfrc_ang[i_l, i_b]


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
    _B = dofs_state.ctrl_mode.shape[1]
    n_links = links_info.root_idx.shape[0]

    if ti.static(static_rigid_sim_config.use_hibernation):
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(_B):
            for i_l_ in range(rigid_global_info.n_awake_links[i_b]):
                i_l = rigid_global_info.awake_links[i_l_, i_b]
                I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l

                for i_d in range(links_info.dof_start[I_l], links_info.dof_end[I_l]):
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

    else:
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_l, i_b in ti.ndrange(n_links, _B):
            I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l

            for i_d in range(links_info.dof_start[I_l], links_info.dof_end[I_l]):
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


@ti.func
def func_compute_qacc(
    dofs_state: array_class.DofsState,
    entities_info: array_class.EntitiesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    _B = dofs_state.ctrl_mode.shape[1]
    n_entities = entities_info.n_links.shape[0]

    func_solve_mass(
        vec=dofs_state.force,
        out=dofs_state.acc_smooth,
        entities_info=entities_info,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
    )

    if ti.static(static_rigid_sim_config.use_hibernation):
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
        for i_b in ti.range(_B):
            for i_e_ in range(rigid_global_info.n_awake_entities[i_b]):
                i_e = rigid_global_info.awake_entities[i_e_, i_b]
                for i_d1_ in range(entities_info.n_dofs[i_e]):
                    i_d1 = entities_info.dof_start[i_e] + i_d1_
                    dofs_state.acc[i_d1, i_b] = dofs_state.acc_smooth[i_d1, i_b]
    else:
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
        for i_e, i_b in ti.ndrange(n_entities, _B):
            for i_d1_ in range(entities_info.n_dofs[i_e]):
                i_d1 = entities_info.dof_start[i_e] + i_d1_
                dofs_state.acc[i_d1, i_b] = dofs_state.acc_smooth[i_d1, i_b]


@ti.func
def func_integrate(
    dofs_state: array_class.DofsState,
    links_info: array_class.LinksInfo,
    joints_info: array_class.JointsInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):

    _B = dofs_state.ctrl_mode.shape[1]
    n_dofs = dofs_state.ctrl_mode.shape[0]
    n_links = links_info.root_idx.shape[0]
    if ti.static(static_rigid_sim_config.use_hibernation):
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(_B):
            for i_d_ in range(rigid_global_info.n_awake_dofs[i_b]):
                i_d = rigid_global_info.awake_dofs[i_d_, i_b]
                dofs_state.vel[i_d, i_b] = (
                    dofs_state.vel[i_d, i_b] + dofs_state.acc[i_d, i_b] * static_rigid_sim_config.substep_dt
                )

        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(_B):
            for i_l_ in range(rigid_global_info.n_awake_links[i_b]):
                i_l = rigid_global_info.awake_links[i_l_, i_b]
                I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l

                for i_j in range(links_info.joint_start[I_l], links_info.joint_end[I_l]):
                    dof_start = joints_info.dof_start[I_j]
                    q_start = joints_info.q_start[I_j]
                    q_end = joints_info.q_end[I_j]

                    I_j = [i_j, i_b] if ti.static(static_rigid_sim_config.batch_joints_info) else i_j
                    joint_type = joints_info.type[I_j]

                    if joint_type == gs.JOINT_TYPE.FREE:
                        rot = ti.Vector(
                            [
                                rigid_global_info.qpos[q_start + 3, i_b],
                                rigid_global_info.qpos[q_start + 4, i_b],
                                rigid_global_info.qpos[q_start + 5, i_b],
                                rigid_global_info.qpos[q_start + 6, i_b],
                            ]
                        )
                        ang = (
                            ti.Vector(
                                [
                                    dofs_state.vel[dof_start + 3, i_b],
                                    dofs_state.vel[dof_start + 4, i_b],
                                    dofs_state.vel[dof_start + 5, i_b],
                                ]
                            )
                            * static_rigid_sim_config.substep_dt
                        )
                        qrot = gu.ti_rotvec_to_quat(ang)
                        rot = gu.ti_transform_quat_by_quat(qrot, rot)
                        pos = ti.Vector(
                            [
                                rigid_global_info.qpos[q_start, i_b],
                                rigid_global_info.qpos[q_start + 1, i_b],
                                rigid_global_info.qpos[q_start + 2, i_b],
                            ]
                        )
                        vel = ti.Vector(
                            [
                                dofs_state.vel[dof_start, i_b],
                                dofs_state.vel[dof_start + 1, i_b],
                                dofs_state.vel[dof_start + 2, i_b],
                            ]
                        )
                        pos = pos + vel * static_rigid_sim_config.substep_dt
                        for j in ti.static(range(3)):
                            rigid_global_info.qpos[q_start + j, i_b] = pos[j]
                        for j in ti.static(range(4)):
                            rigid_global_info.qpos[q_start + j + 3, i_b] = rot[j]
                    elif joint_type == gs.JOINT_TYPE.FIXED:
                        pass
                    elif joint_type == gs.JOINT_TYPE.SPHERICAL:
                        rot = ti.Vector(
                            [
                                rigid_global_info.qpos[q_start + 0, i_b],
                                rigid_global_info.qpos[q_start + 1, i_b],
                                rigid_global_info.qpos[q_start + 2, i_b],
                                rigid_global_info.qpos[q_start + 3, i_b],
                            ]
                        )
                        ang = (
                            ti.Vector(
                                [
                                    dofs_state.vel[dof_start + 3, i_b],
                                    dofs_state.vel[dof_start + 4, i_b],
                                    dofs_state.vel[dof_start + 5, i_b],
                                ]
                            )
                            * static_rigid_sim_config.substep_dt
                        )
                        qrot = gu.ti_rotvec_to_quat(ang)
                        rot = gu.ti_transform_quat_by_quat(qrot, rot)
                        for j in ti.static(range(4)):
                            rigid_global_info.qpos[q_start + j, i_b] = rot[j]

                    else:
                        for j in range(q_end - q_start):
                            rigid_global_info.qpos[q_start + j, i_b] = (
                                rigid_global_info.qpos[q_start + j, i_b]
                                + dofs_state.vel[dof_start + j, i_b] * static_rigid_sim_config.substep_dt
                            )

    else:
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_d, i_b in ti.ndrange(n_dofs, _B):
            dofs_state.vel[i_d, i_b] = (
                dofs_state.vel[i_d, i_b] + dofs_state.acc[i_d, i_b] * static_rigid_sim_config.substep_dt
            )

        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_l, i_b in ti.ndrange(n_links, _B):
            I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l
            if links_info.n_dofs[I_l] == 0:
                continue

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
                        dofs_state.vel[dof_start, i_b],
                        dofs_state.vel[dof_start + 1, i_b],
                        dofs_state.vel[dof_start + 2, i_b],
                    ]
                )
                pos = pos + vel * static_rigid_sim_config.substep_dt
                for j in ti.static(range(3)):
                    rigid_global_info.qpos[q_start + j, i_b] = pos[j]
            if joint_type == gs.JOINT_TYPE.SPHERICAL or joint_type == gs.JOINT_TYPE.FREE:
                rot_offset = 3 if joint_type == gs.JOINT_TYPE.FREE else 0
                rot = ti.Vector(
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
                            dofs_state.vel[dof_start + rot_offset + 0, i_b],
                            dofs_state.vel[dof_start + rot_offset + 1, i_b],
                            dofs_state.vel[dof_start + rot_offset + 2, i_b],
                        ]
                    )
                    * static_rigid_sim_config.substep_dt
                )
                qrot = gu.ti_rotvec_to_quat(ang)
                rot = gu.ti_transform_quat_by_quat(qrot, rot)
                for j in ti.static(range(4)):
                    rigid_global_info.qpos[q_start + j + rot_offset, i_b] = rot[j]
            else:
                for j in range(q_end - q_start):
                    rigid_global_info.qpos[q_start + j, i_b] = (
                        rigid_global_info.qpos[q_start + j, i_b]
                        + dofs_state.vel[dof_start + j, i_b] * static_rigid_sim_config.substep_dt
                    )


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
                ti.Vector([dq[dq_start + 3, i_b], dq[dq_start + 4, i_b], dq[dq_start + 5, i_b]])
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


@ti.kernel
def kernel_update_geoms_render_T(
    geoms_render_T: ti.types.ndarray(),
    geoms_state: array_class.GeomsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    n_geoms = geoms_state.pos.shape[0]
    _B = geoms_state.pos.shape[1]
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_g, i_b in ti.ndrange(n_geoms, _B):
        geom_T = gu.ti_trans_quat_to_T(
            geoms_state.pos[i_g, i_b] + rigid_global_info.envs_offset[i_b],
            geoms_state.quat[i_g, i_b],
        )
        for i, j in ti.static(ti.ndrange(4, 4)):
            geoms_render_T[i_g, i_b, i, j] = ti.cast(geom_T[i, j], ti.float32)


@ti.kernel
def kernel_update_vgeoms_render_T(
    vgeoms_render_T: ti.types.ndarray(),
    vgeoms_info: array_class.VGeomsInfo,
    vgeoms_state: array_class.VGeomsState,
    links_state: array_class.LinksState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    n_vgeoms = vgeoms_info.link_idx.shape[0]
    _B = links_state.pos.shape[1]
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_g, i_b in ti.ndrange(n_vgeoms, _B):
        geom_T = gu.ti_trans_quat_to_T(
            vgeoms_state.pos[i_g, i_b] + rigid_global_info.envs_offset[i_b],
            vgeoms_state.quat[i_g, i_b],
        )
        for i, j in ti.static(ti.ndrange(4, 4)):
            vgeoms_render_T[i_g, i_b, i, j] = ti.cast(geom_T[i, j], ti.float32)


@ti.kernel
def kernel_get_state(
    qpos: ti.types.ndarray(),
    vel: ti.types.ndarray(),
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

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l, i_b in ti.ndrange(n_links, _B):
        for i in ti.static(range(3)):
            links_pos[i_b, i_l, i] = links_state.pos[i_l, i_b][i]
            i_pos_shift[i_b, i_l, i] = links_state.i_pos_shift[i_l, i_b][i]
        for i in ti.static(range(4)):
            links_quat[i_b, i_l, i] = links_state.quat[i_l, i_b][i]
        mass_shift[i_b, i_l] = links_state.mass_shift[i_l, i_b]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l, i_b in ti.ndrange(n_geoms, _B):
        friction_ratio[i_b, i_l] = geoms_state.friction_ratio[i_l, i_b]


@ti.kernel
def kernel_set_state(
    qpos: ti.types.ndarray(),
    dofs_vel: ti.types.ndarray(),
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

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l, i_b_ in ti.ndrange(n_links, envs_idx.shape[0]):
        for i in ti.static(range(3)):
            links_state.pos[i_l, envs_idx[i_b_]][i] = links_pos[envs_idx[i_b_], i_l, i]
            links_state.i_pos_shift[i_l, envs_idx[i_b_]][i] = i_pos_shift[envs_idx[i_b_], i_l, i]
        for i in ti.static(range(4)):
            links_state.quat[i_l, envs_idx[i_b_]][i] = links_quat[envs_idx[i_b_], i_l, i]
        links_state.mass_shift[i_l, envs_idx[i_b_]] = mass_shift[envs_idx[i_b_], i_l]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l, i_b_ in ti.ndrange(n_geoms, envs_idx.shape[0]):
        geoms_state.friction_ratio[i_l, envs_idx[i_b_]] = friction_ratio[envs_idx[i_b_], i_l]


@ti.kernel
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
            for i in ti.static(range(3)):
                links_state.pos[i_l, i_b][i] = pos[i_b_, i_l_, i]
            if relative:
                for i in ti.static(range(3)):
                    links_state.pos[i_l, i_b][i] = links_state.pos[i_l, i_b][i] + links_info.pos[I_l][i]
        else:
            q_start = links_info.q_start[I_l]
            for i in ti.static(range(3)):
                rigid_global_info.qpos[q_start + i, i_b] = pos[i_b_, i_l_, i]
            if relative:
                for i in ti.static(range(3)):
                    rigid_global_info.qpos[q_start + i, i_b] = (
                        rigid_global_info.qpos[q_start + i, i_b] + rigid_global_info.qpos0[q_start + i, i_b]
                    )


@ti.kernel
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
                for i in ti.static(range(4)):
                    rigid_global_info.qpos[q_start + i + 3, i_b] = quat_[i]
        else:
            if links_info.parent_idx[I_l] == -1 and links_info.is_fixed[I_l]:
                for i in ti.static(range(4)):
                    links_state.quat[i_l, i_b][i] = quat[i_b_, i_l_, i]
            else:
                q_start = links_info.q_start[I_l]
                for i in ti.static(range(4)):
                    rigid_global_info.qpos[q_start + i + 3, i_b] = quat[i_b_, i_l_, i]


@ti.kernel
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


@ti.kernel
def kernel_set_links_COM_shift(
    com: ti.types.ndarray(),
    links_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    links_state: array_class.LinksState,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_l_, i_b_ in ti.ndrange(links_idx.shape[0], envs_idx.shape[0]):
        for i in ti.static(range(3)):
            links_state.i_pos_shift[links_idx[i_l_], envs_idx[i_b_]][i] = com[i_b_, i_l_, i]


@ti.kernel
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


@ti.kernel
def kernel_set_links_invweight(
    invweight: ti.types.ndarray(),
    links_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    links_info: array_class.LinksInfo,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    if ti.static(static_rigid_sim_config.batch_links_info):
        for i_l_, i_b_, j in ti.ndrange(links_idx.shape[0], envs_idx.shape[0], 2):
            links_info.invweight[links_idx[i_l_], envs_idx[i_b_]][j] = invweight[i_b_, i_l_, j]
    else:
        for i_l_, j in ti.ndrange(links_idx.shape[0], 2):
            links_info.invweight[links_idx[i_l_]][j] = invweight[i_l_, j]


@ti.kernel
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


@ti.kernel
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


@ti.kernel
def kernel_set_global_sol_params(
    sol_params: ti.types.ndarray(),
    geoms_info: array_class.GeomsInfo,
    joints_info: array_class.JointsInfo,
    equalities_info: array_class.EqualitiesInfo,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
    n_geoms = geoms_info.sol_params.shape[0]
    n_joints = joints_info.sol_params.shape[0]
    n_equalities = equalities_info.sol_params.shape[0]
    _B = equalities_info.sol_params.shape[1]

    for i_g in range(n_geoms):
        for i in ti.static(range(7)):
            geoms_info.sol_params[i_g][i] = sol_params[i]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_j, i_b in ti.ndrange(n_joints, _B):
        I_j = [i_j, i_b] if ti.static(static_rigid_sim_config.batch_joints_info) else i_j
        for i in ti.static(range(7)):
            joints_info.sol_params[I_j][i] = sol_params[i]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_eq, i_b in ti.ndrange(n_equalities, _B):
        for i in ti.static(range(7)):
            equalities_info.sol_params[i_eq, i_b][i] = sol_params[i]


@ti.kernel
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
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
        for i_g_ in range(inputs_idx.shape[0]):
            for i in ti.static(range(7)):
                geoms_info.sol_params[inputs_idx[i_g_]][i] = sol_params[i_g_, i]
    elif ti.static(constraint_type == 1):  # joints
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
        if ti.static(static_rigid_sim_config.batch_joints_info):
            for i_j_, i_b_ in ti.ndrange(inputs_idx.shape[0], envs_idx.shape[0]):
                for i in ti.static(range(7)):
                    joints_info.sol_params[inputs_idx[i_j_], envs_idx[i_b_]][i] = sol_params[i_b_, i_j_, i]
        else:
            for i_j_ in range(inputs_idx.shape[0]):
                for i in ti.static(range(7)):
                    joints_info.sol_params[inputs_idx[i_j_]][i] = sol_params[i_j_, i]
    else:  # equalities
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
        for i_eq_, i_b_ in ti.ndrange(inputs_idx.shape[0], envs_idx.shape[0]):
            for i in ti.static(range(7)):
                equalities_info.sol_params[inputs_idx[i_eq_], envs_idx[i_b_]][i] = sol_params[i_b_, i_eq_, i]


@ti.kernel
def kernel_set_dofs_kp(
    kp: ti.types.ndarray(),
    dofs_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    dofs_info: array_class.DofsInfo,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
    if ti.static(static_rigid_sim_config.batch_dofs_info):
        for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
            dofs_info.kp[dofs_idx[i_d_], envs_idx[i_b_]] = kp[i_b_, i_d_]
    else:
        for i_d_ in range(dofs_idx.shape[0]):
            dofs_info.kp[dofs_idx[i_d_]] = kp[i_d_]


@ti.kernel
def kernel_set_dofs_kv(
    kv: ti.types.ndarray(),
    dofs_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    dofs_info: array_class.DofsInfo,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
    if ti.static(static_rigid_sim_config.batch_dofs_info):
        for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
            dofs_info.kv[dofs_idx[i_d_], envs_idx[i_b_]] = kv[i_b_, i_d_]
    else:
        for i_d_ in range(dofs_idx.shape[0]):
            dofs_info.kv[dofs_idx[i_d_]] = kv[i_d_]


@ti.kernel
def kernel_set_dofs_force_range(
    lower: ti.types.ndarray(),
    upper: ti.types.ndarray(),
    dofs_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    dofs_info: array_class.DofsInfo,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
    if ti.static(static_rigid_sim_config.batch_dofs_info):
        for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
            dofs_info.force_range[dofs_idx[i_d_], envs_idx[i_b_]][0] = lower[i_b_, i_d_]
            dofs_info.force_range[dofs_idx[i_d_], envs_idx[i_b_]][1] = upper[i_b_, i_d_]
    else:
        for i_d_ in range(dofs_idx.shape[0]):
            dofs_info.force_range[dofs_idx[i_d_]][0] = lower[i_d_]
            dofs_info.force_range[dofs_idx[i_d_]][1] = upper[i_d_]


@ti.kernel
def kernel_set_dofs_stiffness(
    stiffness: ti.types.ndarray(),
    dofs_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    dofs_info: array_class.DofsInfo,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
    if ti.static(static_rigid_sim_config.batch_dofs_info):
        for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
            dofs_info.stiffness[dofs_idx[i_d_], envs_idx[i_b_]] = stiffness[i_b_, i_d_]
    else:
        for i_d_ in range(dofs_idx.shape[0]):
            dofs_info.stiffness[dofs_idx[i_d_]] = stiffness[i_d_]


@ti.kernel
def kernel_set_dofs_invweight(
    invweight: ti.types.ndarray(),
    dofs_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    dofs_info: array_class.DofsInfo,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
    if ti.static(static_rigid_sim_config.batch_dofs_info):
        for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
            dofs_info.invweight[dofs_idx[i_d_], envs_idx[i_b_]] = invweight[i_b_, i_d_]
    else:
        for i_d_ in range(dofs_idx.shape[0]):
            dofs_info.invweight[dofs_idx[i_d_]] = invweight[i_d_]


@ti.kernel
def kernel_set_dofs_armature(
    armature: ti.types.ndarray(),
    dofs_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    dofs_info: array_class.DofsInfo,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
    if ti.static(static_rigid_sim_config.batch_dofs_info):
        for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
            dofs_info.armature[dofs_idx[i_d_], envs_idx[i_b_]] = armature[i_b_, i_d_]
    else:
        for i_d_ in range(dofs_idx.shape[0]):
            dofs_info.armature[dofs_idx[i_d_]] = armature[i_d_]


@ti.kernel
def kernel_set_dofs_damping(
    damping: ti.types.ndarray(),
    dofs_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    dofs_info: array_class.DofsInfo,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
    if ti.static(static_rigid_sim_config.batch_dofs_info):
        for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
            dofs_info.damping[dofs_idx[i_d_], envs_idx[i_b_]] = damping[i_b_, i_d_]
    else:
        for i_d_ in range(dofs_idx.shape[0]):
            dofs_info.damping[dofs_idx[i_d_]] = damping[i_d_]


@ti.kernel
def kernel_set_dofs_limit(
    lower: ti.types.ndarray(),
    upper: ti.types.ndarray(),
    dofs_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    dofs_info: array_class.DofsInfo,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
    if ti.static(static_rigid_sim_config.batch_dofs_info):
        for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
            dofs_info.limit[dofs_idx[i_d_], envs_idx[i_b_]][0] = lower[i_b_, i_d_]
            dofs_info.limit[dofs_idx[i_d_], envs_idx[i_b_]][1] = upper[i_b_, i_d_]
    else:
        for i_d_ in range(dofs_idx.shape[0]):
            dofs_info.limit[dofs_idx[i_d_]][0] = lower[i_d_]
            dofs_info.limit[dofs_idx[i_d_]][1] = upper[i_d_]


@ti.kernel
def kernel_set_dofs_velocity(
    velocity: ti.types.ndarray(),
    dofs_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    dofs_state: array_class.DofsState,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
        dofs_state.vel[dofs_idx[i_d_], envs_idx[i_b_]] = velocity[i_b_, i_d_]


@ti.kernel
def kernel_set_dofs_zero_velocity(
    dofs_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    dofs_state: array_class.DofsState,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
        dofs_state.vel[dofs_idx[i_d_], envs_idx[i_b_]] = 0.0


@ti.kernel
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

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
        dofs_state.pos[dofs_idx[i_d_], envs_idx[i_b_]] = position[i_b_, i_d_]

    # also need to update qpos, as dofs_state.pos is not used for actual IK
    # TODO: make this more efficient by only taking care of releavant qs/dofs

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
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

            if joint_type == gs.JOINT_TYPE.FREE:
                xyz = ti.Vector(
                    [
                        dofs_state.pos[0 + 3 + dof_start, i_b],
                        dofs_state.pos[1 + 3 + dof_start, i_b],
                        dofs_state.pos[2 + 3 + dof_start, i_b],
                    ],
                    dt=gs.ti_float,
                )
                quat = gu.ti_xyz_to_quat(xyz)

                for i_q in ti.static(range(3)):
                    rigid_global_info.qpos[i_q + q_start, i_b] = dofs_state.pos[i_q + dof_start, i_b]

                for i_q in ti.static(range(4)):
                    rigid_global_info.qpos[i_q + 3 + q_start, i_b] = quat[i_q]
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
                    rigid_global_info.qpos[i_q, i_b] = quat[i_q - q_start]
            else:
                for i_q in range(q_start, links_info.q_end[I_l]):
                    rigid_global_info.qpos[i_q, i_b] = dofs_state.pos[dof_start + i_q - q_start, i_b]


@ti.kernel
def kernel_control_dofs_force(
    force: ti.types.ndarray(),
    dofs_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    dofs_state: array_class.DofsState,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
        dofs_state.ctrl_mode[dofs_idx[i_d_], envs_idx[i_b_]] = gs.CTRL_MODE.FORCE
        dofs_state.ctrl_force[dofs_idx[i_d_], envs_idx[i_b_]] = force[i_b_, i_d_]


@ti.kernel
def kernel_control_dofs_velocity(
    velocity: ti.types.ndarray(),
    dofs_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    dofs_state: array_class.DofsState,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
        dofs_state.ctrl_mode[dofs_idx[i_d_], envs_idx[i_b_]] = gs.CTRL_MODE.VELOCITY
        dofs_state.ctrl_vel[dofs_idx[i_d_], envs_idx[i_b_]] = velocity[i_b_, i_d_]


@ti.kernel
def kernel_control_dofs_position(
    position: ti.types.ndarray(),
    dofs_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    dofs_state: array_class.DofsState,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
        dofs_state.ctrl_mode[dofs_idx[i_d_], envs_idx[i_b_]] = gs.CTRL_MODE.POSITION
        dofs_state.ctrl_pos[dofs_idx[i_d_], envs_idx[i_b_]] = position[i_b_, i_d_]


@ti.kernel
def kernel_get_links_vel(
    tensor: ti.types.ndarray(),
    links_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    ref: ti.template(),
    links_state: array_class.LinksState,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
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
                links_state.pos[links_idx[i_l_], envs_idx[i_b_]] - links_state.COM[links_idx[i_l_], envs_idx[i_b_]]
            )

        for i in ti.static(range(3)):
            tensor[i_b_, i_l_, i] = vel[i]


@ti.kernel
def kernel_get_links_acc(
    mimick_imu: ti.i32,
    tensor: ti.types.ndarray(),
    links_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    links_state: array_class.LinksState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_l_, i_b_ in ti.ndrange(links_idx.shape[0], envs_idx.shape[0]):
        i_l = links_idx[i_l_]
        i_b = envs_idx[i_b_]

        # Compute links spatial acceleration expressed at links origin in world coordinates
        cpos = links_state.pos[i_l, i_b] - links_state.COM[i_l, i_b]
        acc_ang = links_state.cacc_ang[i_l, i_b]
        acc_lin = links_state.cacc_lin[i_l, i_b] + acc_ang.cross(cpos)

        # Compute links classical linear acceleration expressed at links origin in world coordinates
        ang = links_state.cd_ang[i_l, i_b]
        vel = links_state.cd_vel[i_l, i_b] + ang.cross(cpos)
        acc_classic_lin = acc_lin + ang.cross(vel)

        # Mimick IMU accelerometer signal if requested
        if mimick_imu:
            # Subtract gravity
            acc_classic_lin -= rigid_global_info.gravity[i_b]

            # Move the resulting linear acceleration in local links frame
            acc_classic_lin = gu.ti_inv_transform_by_quat(acc_classic_lin, links_state.quat[i_l, i_b])

        for i in ti.static(range(3)):
            tensor[i_b_, i_l_, i] = acc_classic_lin[i]


@ti.kernel
def kernel_get_dofs_control_force(
    tensor: ti.types.ndarray(),
    dofs_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    static_rigid_sim_config: ti.template(),
):
    # we need to compute control force here because this won't be computed until the next actual simulation step
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
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
            force = (
                dofs_info.kp[I_d] * (dofs_state.ctrl_pos[i_d, i_b] - dofs_state.pos[i_d, i_b])
                - dofs_info.kv[I_d] * dofs_state.vel[i_d, i_b]
            )
        tensor[i_b_, i_d_] = ti.math.clamp(
            force,
            dofs_info.force_range[I_d][0],
            dofs_info.force_range[I_d][1],
        )


@ti.kernel
def kernel_set_drone_rpm(
    n_propellers: ti.i32,
    propellers_link_idxs: ti.types.ndarray(),
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
    _B = propellers_rpm.shape[1]
    for i_b in range(_B):
        for i_prop in range(n_propellers):
            i_l = propellers_link_idxs[i_prop]

            force = ti.Vector([0.0, 0.0, propellers_rpm[i_prop, i_b] ** 2 * KF], dt=gs.ti_float)
            torque = ti.Vector(
                [0.0, 0.0, propellers_rpm[i_prop, i_b] ** 2 * KM * propellers_spin[i_prop]], dt=gs.ti_float
            )
            if invert:
                torque = -torque

            func_apply_link_external_force(force, i_l, i_b, 1, 1, links_state)
            func_apply_link_external_torque(torque, i_l, i_b, 1, 1, links_state)


@ti.kernel
def kernel_update_drone_propeller_vgeoms(
    n_propellers: ti.i32,
    propellers_vgeom_idxs: ti.types.ndarray(),
    propellers_revs: ti.types.ndarray(),
    propellers_spin: ti.types.ndarray(),
    vgeoms_state: array_class.VGeomsState,
    static_rigid_sim_config: ti.template(),
):
    """
    Update the angle of the vgeom in the propellers of a drone entity.
    """
    _B = propellers_revs.shape[1]
    for i, b in ti.ndrange(n_propellers, _B):
        rad = propellers_revs[i, b] * propellers_spin[i] * static_rigid_sim_config.substep_dt * np.pi / 30
        vgeoms_state.quat[propellers_vgeom_idxs[i], b] = gu.ti_transform_quat_by_quat(
            gu.ti_rotvec_to_quat(ti.Vector([0.0, 0.0, rad], dt=gs.ti_float)),
            vgeoms_state.quat[propellers_vgeom_idxs[i], b],
        )


@ti.kernel
def kernel_set_geom_friction(geoms_idx: ti.i32, friction: ti.f32, geoms_info: array_class.GeomsInfo):
    geoms_info.friction[geoms_idx] = friction


@ti.kernel
def kernel_set_geoms_friction(
    friction: ti.types.ndarray(),
    geoms_idx: ti.types.ndarray(),
    geoms_info: array_class.GeomsInfo,
    static_rigid_sim_config: ti.template(),
):
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_g_ in ti.ndrange(geoms_idx.shape[0]):
        geoms_info.friction[geoms_idx[i_g_]] = friction[i_g_]
