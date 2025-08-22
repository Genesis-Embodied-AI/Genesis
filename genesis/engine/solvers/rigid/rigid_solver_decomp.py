from dataclasses import dataclass
from typing import Callable, Literal, TYPE_CHECKING

import gstaichi as ti
import numpy as np
import numpy.typing as npt
import torch

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.array_class as array_class

from genesis.engine.entities import AvatarEntity, DroneEntity, RigidEntity
from genesis.engine.entities.base_entity import Entity
from genesis.engine.solvers.rigid.contact_island import ContactIsland
from genesis.engine.states.solvers import RigidSolverState
from genesis.options.solvers import RigidOptions
from genesis.styles import colors, formats
from genesis.utils import linalg as lu
from genesis.utils.misc import ti_field_to_torch, DeprecationError, ALLOCATE_TENSOR_WARNING

from ....utils.sdf_decomp import SDF
from ..base_solver import Solver
from .constraint_solver_decomp import ConstraintSolver
from .constraint_solver_decomp_island import ConstraintSolverIsland
from .contact_island import INVALID_NEXT_HIBERNATED_ENTITY_IDX
from .collider_decomp import Collider
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
    def maybe_pure_local(fn: Callable) -> Callable:
        from .... import maybe_pure
        return maybe_pure(fn)

    # override typing
    _entities: list[RigidEntity] = gs.List()

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
        elif isinstance(morph, gs.morphs.Drone):
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

        self.n_equalities_candidate = max(1, self.n_equalities + self._options.max_dynamic_constraints)

        # Note optional hibernation_threshold_acc/vel params at the bottom of the initialization list.
        # This is caused by this code being also run by AvatarSolver, which inherits from this class
        # but does not have all the attributes of the base class.
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
            hibernation_thresh_acc=getattr(self, "_hibernation_thresh_acc", 0.0),
            hibernation_thresh_vel=getattr(self, "_hibernation_thresh_vel", 0.0),
        )

        # when the migration is finished, we will remove the about two lines
        from . import rigid_solver_decomp_kernels
        self._func_vel_at_point = rigid_solver_decomp_kernels.func_vel_at_point
        self._func_apply_external_force = rigid_solver_decomp_kernels.func_apply_external_force

        if self.is_active():
            self.data_manager = array_class.DataManager(self)

            self._rigid_global_info = self.data_manager.rigid_global_info
            if self._use_hibernation:
                self.n_awake_dofs = self._rigid_global_info.n_awake_dofs
                self.awake_dofs = self._rigid_global_info.awake_dofs
                self.n_awake_links = self._rigid_global_info.n_awake_links
                self.awake_links = self._rigid_global_info.awake_links
                self.n_awake_entities = self._rigid_global_info.n_awake_entities
                self.awake_entities = self._rigid_global_info.awake_entities

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
            rigid_solver_decomp_kernels.kernel_forward_kinematics_links_geoms(
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
            rigid_solver_decomp_kernels.kernel_init_meaninertia(
                rigid_global_info=self._rigid_global_info,
                entities_info=self.entities_info,
                static_rigid_sim_config=self._static_rigid_sim_config,
            )

    def _init_invweight(self):
        from . import rigid_solver_decomp_kernels
        # Early return if no DoFs. This is essential to avoid segfault on CUDA.
        if self._n_dofs == 0:
            return

        # Compute mass matrix without any implicit damping terms
        rigid_solver_decomp_kernels.kernel_compute_mass_matrix(
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
        from . import rigid_solver_decomp_kernels
        rigid_solver_decomp_kernels.kernel_init_invweight(
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
        from . import rigid_solver_decomp_kernels
        if has_dofs:  # handle the case where there is a link with no dofs -- otherwise may cause invalid memory
            rigid_solver_decomp_kernels.kernel_init_dof_fields(
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
        from . import rigid_solver_decomp_kernels
        rigid_solver_decomp_kernels.kernel_init_link_fields(
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
        from . import rigid_solver_decomp_kernels
        if joints:
            # Make sure that the constraints parameters are valid
            joints_sol_params = np.array([joint.sol_params for joint in joints], dtype=gs.np_float)
            _sanitize_sol_params(joints_sol_params, self._sol_min_timeconst, self._sol_default_timeconst)

            rigid_solver_decomp_kernels.kernel_init_joint_fields(
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

        from . import rigid_solver_decomp_kernels
        if self.n_verts > 0:
            geoms = self.geoms
            rigid_solver_decomp_kernels.kernel_init_vert_fields(
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
        from . import rigid_solver_decomp_kernels
        if self.n_vverts > 0:
            vgeoms = self.vgeoms
            rigid_solver_decomp_kernels.kernel_init_vvert_fields(
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
        from . import rigid_solver_decomp_kernels

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

            rigid_solver_decomp_kernels.kernel_init_geom_fields(
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
        self.vgeoms_info: array_class.VGeomsInfo = self.data_manager.vgeoms_info
        self.vgeoms_state: array_class.VGeomsState = self.data_manager.vgeoms_state
        self._vgeoms_render_T = np.empty((self.n_vgeoms_, self._B, 4, 4), dtype=np.float32)

        from . import rigid_solver_decomp_kernels
        if self.n_vgeoms > 0:
            vgeoms = self.vgeoms
            rigid_solver_decomp_kernels.kernel_init_vgeom_fields(
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
        from . import rigid_solver_decomp_kernels
        rigid_solver_decomp_kernels.kernel_init_entity_fields(
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
        from . import rigid_solver_decomp_kernels
        if self.n_equalities > 0:
            equalities = self.equalities

            equalities_sol_params = np.array([equality.sol_params for equality in equalities], dtype=gs.np_float)
            _sanitize_sol_params(
                equalities_sol_params,
                self._sol_min_timeconst,
                self._sol_default_timeconst,
            )

            rigid_solver_decomp_kernels.kernel_init_equality_fields(
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
        from genesis.engine.couplers import SAPCoupler

        # timer = create_timer("rigid", level=1, ti_sync=True, skip_first_call=True)
        from . import rigid_solver_decomp_kernels
        rigid_solver_decomp_kernels.kernel_step_1(
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
        # timer.stamp("kernel_step_1")

        if isinstance(self.sim.coupler, SAPCoupler):
            self.update_qvel()
        else:
            self._func_constraint_force()
            # timer.stamp("constraint_force")
            rigid_solver_decomp_kernels.kernel_step_2(
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
        from . import rigid_solver_decomp_kernels
        rigid_solver_decomp_kernels.kernel_forward_dynamics(
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
        from . import rigid_solver_decomp_kernels
        rigid_solver_decomp_kernels.kernel_update_acc(
            dofs_state=self.dofs_state,
            links_info=self.links_info,
            links_state=self.links_state,
            entities_info=self.entities_info,
            rigid_global_info=self._rigid_global_info,
            static_rigid_sim_config=self._static_rigid_sim_config,
        )

    def _func_forward_kinematics_entity(self, i_e, envs_idx):
        from . import rigid_solver_decomp_kernels
        rigid_solver_decomp_kernels.kernel_forward_kinematics_entity(
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
        from . import rigid_solver_decomp_kernels
        rigid_solver_decomp_kernels.func_integrate_dq_entity(
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
        from . import rigid_solver_decomp_kernels
        rigid_solver_decomp_kernels.kernel_update_geoms(
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

        from . import rigid_solver_decomp_kernels
        rigid_solver_decomp_kernels.kernel_apply_links_external_force(
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

        from . import rigid_solver_decomp_kernels
        rigid_solver_decomp_kernels.kernel_apply_links_external_torque(
            torque, links_idx, envs_idx, ref, 1 if local else 0, self.links_state, self._static_rigid_sim_config
        )

    @ti.kernel
    def update_qvel(self):
        if ti.static(self._use_hibernation):
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_b in range(self._B):
                for i_d_ in range(self.n_awake_dofs[i_b]):
                    i_d = self.awake_dofs[i_d_, i_b]
                    self.dofs_state.vel_prev[i_d, i_b] = self.dofs_state.vel[i_d, i_b]
                    self.dofs_state.vel[i_d, i_b] = (
                        self.dofs_state.vel[i_d, i_b] + self.dofs_state.acc[i_d, i_b] * self._substep_dt
                    )
        else:
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_d, i_b in ti.ndrange(self.n_dofs, self._B):
                self.dofs_state.vel_prev[i_d, i_b] = self.dofs_state.vel[i_d, i_b]
                self.dofs_state.vel[i_d, i_b] = (
                    self.dofs_state.vel[i_d, i_b] + self.dofs_state.acc[i_d, i_b] * self._substep_dt
                )

    @ti.kernel
    def update_qacc_from_qvel_delta(self):
        if ti.static(self._use_hibernation):
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_b in range(self._B):
                for i_d_ in range(self.n_awake_dofs[i_b]):
                    i_d = self.awake_dofs[i_d_, i_b]
                    self.dofs_state.acc[i_d, i_b] = (
                        self.dofs_state.vel[i_d, i_b] - self.dofs_state.vel_prev[i_d, i_b]
                    ) / self._substep_dt
                    self.dofs_state.vel[i_d, i_b] = self.dofs_state.vel_prev[i_d, i_b]
        else:
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_d, i_b in ti.ndrange(self.n_dofs, self._B):
                self.dofs_state.acc[i_d, i_b] = (
                    self.dofs_state.vel[i_d, i_b] - self.dofs_state.vel_prev[i_d, i_b]
                ) / self._substep_dt
                self.dofs_state.vel[i_d, i_b] = self.dofs_state.vel_prev[i_d, i_b]

    def substep_pre_coupling(self, f):
        if self.is_active():
            self.substep()

    def substep_pre_coupling_grad(self, f):
        pass

    def substep_post_coupling(self, f):
        from genesis.engine.couplers import SAPCoupler
        from . import rigid_solver_decomp_kernels
        if self.is_active() and isinstance(self.sim.coupler, SAPCoupler):
            self.update_qacc_from_qvel_delta()
            rigid_solver_decomp_kernels.kernel_step_2(
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
        from . import rigid_solver_decomp_kernels
        rigid_solver_decomp_kernels.kernel_update_geoms_render_T(
            self._geoms_render_T,
            geoms_state=self.geoms_state,
            rigid_global_info=self._rigid_global_info,
            static_rigid_sim_config=self._static_rigid_sim_config,
        )

    def update_vgeoms_render_T(self):
        from . import rigid_solver_decomp_kernels
        rigid_solver_decomp_kernels.kernel_update_vgeoms_render_T(
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

            from . import rigid_solver_decomp_kernels
            rigid_solver_decomp_kernels.kernel_get_state(
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
        from . import rigid_solver_decomp_kernels
        if self.is_active():
            envs_idx = self._scene._sanitize_envs_idx(envs_idx)
            rigid_solver_decomp_kernels.kernel_set_state(
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
            rigid_solver_decomp_kernels.kernel_forward_kinematics_links_geoms(
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
        from . import rigid_solver_decomp_kernels
        if links_idx is None:
            links_idx = self._base_links_idx
        pos, links_idx, envs_idx = self._sanitize_2D_io_variables(
            pos, links_idx, self.n_links, 3, envs_idx, idx_name="links_idx", skip_allocation=True, unsafe=unsafe
        )
        if self.n_envs == 0:
            pos = pos.unsqueeze(0)
        if not unsafe and not torch.isin(links_idx, self._base_links_idx).all():
            gs.raise_exception("`links_idx` contains at least one link that is not a base link.")
        rigid_solver_decomp_kernels.kernel_set_links_pos(
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
            rigid_solver_decomp_kernels.kernel_forward_kinematics_links_geoms(
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
        from . import rigid_solver_decomp_kernels
        if links_idx is None:
            links_idx = self._base_links_idx
        quat, links_idx, envs_idx = self._sanitize_2D_io_variables(
            quat, links_idx, self.n_links, 4, envs_idx, idx_name="links_idx", skip_allocation=True, unsafe=unsafe
        )
        if self.n_envs == 0:
            quat = quat.unsqueeze(0)
        if not unsafe and not torch.isin(links_idx, self._base_links_idx).all():
            gs.raise_exception("`links_idx` contains at least one link that is not a base link.")
        rigid_solver_decomp_kernels.kernel_set_links_quat(
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
            rigid_solver_decomp_kernels.kernel_forward_kinematics_links_geoms(
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
        from . import rigid_solver_decomp_kernels
        mass, links_idx, envs_idx = self._sanitize_1D_io_variables(
            mass, links_idx, self.n_links, envs_idx, idx_name="links_idx", skip_allocation=True, unsafe=unsafe
        )
        if self.n_envs == 0:
            mass = mass.unsqueeze(0)
        rigid_solver_decomp_kernels.kernel_set_links_mass_shift(
            mass,
            links_idx,
            envs_idx,
            links_state=self.links_state,
            static_rigid_sim_config=self._static_rigid_sim_config,
        )

    def set_links_COM_shift(self, com, links_idx=None, envs_idx=None, *, unsafe=False):
        from . import rigid_solver_decomp_kernels
        com, links_idx, envs_idx = self._sanitize_2D_io_variables(
            com, links_idx, self.n_links, 3, envs_idx, idx_name="links_idx", skip_allocation=True, unsafe=unsafe
        )
        if self.n_envs == 0:
            com = com.unsqueeze(0)
        rigid_solver_decomp_kernels.kernel_set_links_COM_shift(com, links_idx, envs_idx, self.links_state, self._static_rigid_sim_config)

    def set_links_inertial_mass(self, mass, links_idx=None, envs_idx=None, *, unsafe=False):
        from . import rigid_solver_decomp_kernels
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
        rigid_solver_decomp_kernels.kernel_set_links_inertial_mass(mass, links_idx, envs_idx, self.links_info, self._static_rigid_sim_config)

    def set_links_invweight(self, invweight, links_idx=None, envs_idx=None, *, unsafe=False):
        from . import rigid_solver_decomp_kernels
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
        rigid_solver_decomp_kernels.kernel_set_links_invweight(invweight, links_idx, envs_idx, self.links_info, self._static_rigid_sim_config)

    def set_geoms_friction_ratio(self, friction_ratio, geoms_idx=None, envs_idx=None, *, unsafe=False):
        from . import rigid_solver_decomp_kernels
        friction_ratio, geoms_idx, envs_idx = self._sanitize_1D_io_variables(
            friction_ratio, geoms_idx, self.n_geoms, envs_idx, idx_name="geoms_idx", skip_allocation=True, unsafe=unsafe
        )
        if self.n_envs == 0:
            friction_ratio = friction_ratio.unsqueeze(0)
        rigid_solver_decomp_kernels.kernel_set_geoms_friction_ratio(
            friction_ratio, geoms_idx, envs_idx, self.geoms_state, self._static_rigid_sim_config
        )

    def set_qpos(self, qpos, qs_idx=None, envs_idx=None, *, skip_forward=False, unsafe=False):
        from . import rigid_solver_decomp_kernels
        qpos, qs_idx, envs_idx = self._sanitize_1D_io_variables(
            qpos, qs_idx, self.n_qs, envs_idx, idx_name="qs_idx", skip_allocation=True, unsafe=unsafe
        )
        if self.n_envs == 0:
            qpos = qpos.unsqueeze(0)
        rigid_solver_decomp_kernels.kernel_set_qpos(qpos, qs_idx, envs_idx, self._rigid_global_info, self._static_rigid_sim_config)
        self.collider.reset(envs_idx)
        self.collider.clear(envs_idx)
        if self.constraint_solver is not None:
            self.constraint_solver.reset(envs_idx)
            self.constraint_solver.clear(envs_idx)
        if not skip_forward:
            rigid_solver_decomp_kernels.kernel_forward_kinematics_links_geoms(
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
        from . import rigid_solver_decomp_kernels
        # Sanitize input arguments
        if not unsafe:
            _sol_params = torch.as_tensor(sol_params, dtype=gs.tc_float, device=gs.device).contiguous()
            if _sol_params is not sol_params:
                gs.logger.debug(ALLOCATE_TENSOR_WARNING)
            sol_params = _sol_params

        # Make sure that the constraints parameters are within range
        _sanitize_sol_params(sol_params, self._sol_min_timeconst)

        rigid_solver_decomp_kernels.kernel_set_global_sol_params(
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
        from . import rigid_solver_decomp_kernels
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
        rigid_solver_decomp_kernels.kernel_set_sol_params(
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
        from . import rigid_solver_decomp_kernels
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
            rigid_solver_decomp_kernels.kernel_set_dofs_kp(tensor_list[0], dofs_idx, envs_idx, self.dofs_info, self._static_rigid_sim_config)
        elif name == "kv":
            rigid_solver_decomp_kernels.kernel_set_dofs_kv(tensor_list[0], dofs_idx, envs_idx, self.dofs_info, self._static_rigid_sim_config)
        elif name == "force_range":
            rigid_solver_decomp_kernels.kernel_set_dofs_force_range(
                tensor_list[0], tensor_list[1], dofs_idx, envs_idx, self.dofs_info, self._static_rigid_sim_config
            )
        elif name == "stiffness":
            rigid_solver_decomp_kernels.kernel_set_dofs_stiffness(tensor_list[0], dofs_idx, envs_idx, self.dofs_info, self._static_rigid_sim_config)
        elif name == "invweight":
            rigid_solver_decomp_kernels.kernel_set_dofs_invweight(tensor_list[0], dofs_idx, envs_idx, self.dofs_info, self._static_rigid_sim_config)
        elif name == "armature":
            rigid_solver_decomp_kernels.kernel_set_dofs_armature(tensor_list[0], dofs_idx, envs_idx, self.dofs_info, self._static_rigid_sim_config)
        elif name == "damping":
            rigid_solver_decomp_kernels.kernel_set_dofs_damping(tensor_list[0], dofs_idx, envs_idx, self.dofs_info, self._static_rigid_sim_config)
        elif name == "frictionloss":
            rigid_solver_decomp_kernels.kernel_set_dofs_frictionloss(
                tensor_list[0], dofs_idx, envs_idx, self.dofs_info, self._static_rigid_sim_config
            )
        elif name == "limit":
            rigid_solver_decomp_kernels.kernel_set_dofs_limit(
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

    def set_dofs_frictionloss(self, frictionloss, dofs_idx=None, envs_idx=None, *, unsafe=False):
        self._set_dofs_info([frictionloss], dofs_idx, "frictionloss", envs_idx, unsafe=unsafe)

    def set_dofs_limit(self, lower, upper, dofs_idx=None, envs_idx=None, *, unsafe=False):
        self._set_dofs_info([lower, upper], dofs_idx, "limit", envs_idx, unsafe=unsafe)

    def set_dofs_velocity(self, velocity, dofs_idx=None, envs_idx=None, *, skip_forward=False, unsafe=False):
        from . import rigid_solver_decomp_kernels
        velocity, dofs_idx, envs_idx = self._sanitize_1D_io_variables(
            velocity, dofs_idx, self.n_dofs, envs_idx, skip_allocation=True, unsafe=unsafe
        )

        if velocity is None:
            rigid_solver_decomp_kernels.kernel_set_dofs_zero_velocity(dofs_idx, envs_idx, self.dofs_state, self._static_rigid_sim_config)
        else:
            if self.n_envs == 0:
                velocity = velocity.unsqueeze(0)
            rigid_solver_decomp_kernels.kernel_set_dofs_velocity(velocity, dofs_idx, envs_idx, self.dofs_state, self._static_rigid_sim_config)

        if not skip_forward:
            rigid_solver_decomp_kernels.kernel_forward_kinematics_links_geoms(
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
        from . import rigid_solver_decomp_kernels
        position, dofs_idx, envs_idx = self._sanitize_1D_io_variables(
            position, dofs_idx, self.n_dofs, envs_idx, skip_allocation=True, unsafe=unsafe
        )
        if self.n_envs == 0:
            position = position.unsqueeze(0)
        rigid_solver_decomp_kernels.kernel_set_dofs_position(
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
            rigid_solver_decomp_kernels.kernel_forward_kinematics_links_geoms(
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
        from . import rigid_solver_decomp_kernels
        force, dofs_idx, envs_idx = self._sanitize_1D_io_variables(
            force, dofs_idx, self.n_dofs, envs_idx, skip_allocation=True, unsafe=unsafe
        )
        if self.n_envs == 0:
            force = force.unsqueeze(0)
        rigid_solver_decomp_kernels.kernel_control_dofs_force(force, dofs_idx, envs_idx, self.dofs_state, self._static_rigid_sim_config)

    def control_dofs_velocity(self, velocity, dofs_idx=None, envs_idx=None, *, unsafe=False):
        from . import rigid_solver_decomp_kernels
        velocity, dofs_idx, envs_idx = self._sanitize_1D_io_variables(
            velocity, dofs_idx, self.n_dofs, envs_idx, skip_allocation=True, unsafe=unsafe
        )
        if self.n_envs == 0:
            velocity = velocity.unsqueeze(0)
        has_gains = rigid_solver_decomp_kernels.kernel_control_dofs_velocity(
            velocity, dofs_idx, envs_idx, self.dofs_state, self.dofs_info, self._static_rigid_sim_config
        )
        if not unsafe and not has_gains:
            raise gs.raise_exception(
                "Please set control gains kp,kv using `get_dofs_kp`,`get_dofs_kv` prior to calling this method."
            )

    def control_dofs_position(self, position, dofs_idx=None, envs_idx=None, *, unsafe=False):
        from . import rigid_solver_decomp_kernels
        position, dofs_idx, envs_idx = self._sanitize_1D_io_variables(
            position, dofs_idx, self.n_dofs, envs_idx, skip_allocation=True, unsafe=unsafe
        )
        if self.n_envs == 0:
            position = position.unsqueeze(0)
        has_gains = rigid_solver_decomp_kernels.kernel_control_dofs_position(
            position, dofs_idx, envs_idx, self.dofs_state, self.dofs_info, self._static_rigid_sim_config
        )
        if not unsafe and not has_gains:
            raise gs.raise_exception(
                "Please set control gains kp,kv using `get_dofs_kp`,`get_dofs_kv` prior to calling this method."
            )

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
        from . import rigid_solver_decomp_kernels
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
        rigid_solver_decomp_kernels.kernel_get_links_vel(tensor, links_idx, envs_idx, ref, self.links_state, self._static_rigid_sim_config)
        return _tensor

    def get_links_ang(self, links_idx=None, envs_idx=None, *, unsafe=False):
        tensor = ti_field_to_torch(self.links_state.cd_ang, envs_idx, links_idx, transpose=True, unsafe=unsafe)
        return tensor.squeeze(0) if self.n_envs == 0 else tensor

    def get_links_acc(self, links_idx=None, envs_idx=None, *, unsafe=False):
        from . import rigid_solver_decomp_kernels
        _tensor, links_idx, envs_idx = self._sanitize_2D_io_variables(
            None, links_idx, self.n_links, 3, envs_idx, idx_name="links_idx", unsafe=unsafe
        )
        tensor = _tensor.unsqueeze(0) if self.n_envs == 0 else _tensor
        rigid_solver_decomp_kernels.kernel_get_links_acc(
            tensor,
            links_idx,
            envs_idx,
            self.links_state,
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
        from . import rigid_solver_decomp_kernels
        _tensor, dofs_idx, envs_idx = self._sanitize_1D_io_variables(
            None, dofs_idx, self.n_dofs, envs_idx, unsafe=unsafe
        )
        tensor = _tensor.unsqueeze(0) if self.n_envs == 0 else _tensor
        rigid_solver_decomp_kernels.kernel_get_dofs_control_force(
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
        from . import rigid_solver_decomp_kernels
        rigid_solver_decomp_kernels.kernel_set_geom_friction(geoms_idx, friction, self.geoms_info)

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
        rigid_solver_decomp_kernels.kernel_set_geoms_friction(friction, geoms_idx, self.geoms_info, self._static_rigid_sim_config)

    def add_weld_constraint(self, link1_idx, link2_idx, envs_idx=None, *, unsafe=False):
        return self.constraint_solver.add_weld_constraint(link1_idx, link2_idx, envs_idx, unsafe=unsafe)

    def delete_weld_constraint(self, link1_idx, link2_idx, envs_idx=None, *, unsafe=False):
        return self.constraint_solver.delete_weld_constraint(link1_idx, link2_idx, envs_idx, unsafe=unsafe)

    def get_weld_constraints(self, as_tensor: bool = True, to_torch: bool = True):
        return self.constraint_solver.get_weld_constraints(as_tensor, to_torch)

    def get_equality_constraints(self, as_tensor: bool = True, to_torch: bool = True):
        return self.constraint_solver.get_equality_constraints(as_tensor, to_torch)

    def clear_external_force(self):
        from . import rigid_solver_decomp_kernels
        rigid_solver_decomp_kernels.kernel_clear_external_force(self.links_state, self._rigid_global_info, self._static_rigid_sim_config)

    def update_vgeoms(self):
        from . import rigid_solver_decomp_kernels
        rigid_solver_decomp_kernels.kernel_update_vgeoms(self.vgeoms_info, self.vgeoms_state, self.links_state, self._static_rigid_sim_config)

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
        from . import rigid_solver_decomp_kernels
        # Inverse kinematics logic moved from rigid_entity to here, because it incurs circular import.
        rigid_solver_decomp_kernels.kernel_rigid_entity_inverse_kinematics(
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
        from . import rigid_solver_decomp_kernels
        rigid_solver_decomp_kernels.kernel_update_drone_propeller_vgeoms(
            n_propellers,
            propellers_vgeom_idxs,
            propellers_revs,
            propellers_spin,
            self.vgeoms_state,
            self._static_rigid_sim_config,
        )

    def set_drone_rpm(self, n_propellers, propellers_link_idxs, propellers_rpm, propellers_spin, KF, KM, invert):
        from . import rigid_solver_decomp_kernels
        rigid_solver_decomp_kernels.kernel_set_drone_rpm(
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
        from . import rigid_solver_decomp_kernels
        rigid_solver_decomp_kernels.kernel_update_verts_for_geom(
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
