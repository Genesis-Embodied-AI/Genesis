from typing import Literal, TYPE_CHECKING

import numpy as np
import torch
import numpy.typing as npt
import taichi as ti

import genesis as gs
from genesis.engine.entities.base_entity import Entity
from genesis.options.solvers import RigidOptions
import genesis.utils.geom as gu
from genesis.utils.misc import ti_field_to_torch, DeprecationError, ALLOCATE_TENSOR_WARNING
from genesis.engine.entities import AvatarEntity, DroneEntity, RigidEntity
from genesis.engine.states.solvers import RigidSolverState
from genesis.styles import colors, formats

from ..base_solver import Solver
from .collider_decomp import Collider
from .constraint_solver_decomp import ConstraintSolver
from .constraint_solver_decomp_island import ConstraintSolverIsland
from .sdf_decomp import SDF

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
    sol_params: npt.NDArray[np.float64], min_timeconst: float, global_timeconst: float | None = None
):
    assert sol_params.shape[-1] == 7
    timeconst, dampratio, dmin, dmax, width, mid, power = sol_params.reshape((-1, 7)).T
    if global_timeconst is not None:
        timeconst[:] = global_timeconst
    if (timeconst < gs.EPS).any():
        # We deliberately set timeconst to be zero for urdf and meshes so that it can fall back to 2*dt
        gs.logger.debug(f"Constraint solver time constant not specified. Using minimum value (`{min_timeconst:0.6g}`).")
    if ((timeconst > gs.EPS) & (timeconst + gs.EPS < min_timeconst)).any():
        gs.logger.warning(
            "Constraint solver time constant should be greater than 2*subste_dt. timeconst is changed from "
            f"`{timeconst.min():0.6g}` to `{min_timeconst:0.6g}`). Decrease simulation timestep or "
            "increase timeconst to avoid altering the original value."
        )
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

    def __init__(self, scene: "Scene", sim: "Simulator", options: RigidOptions) -> None:
        super().__init__(scene, sim, options)

        if self._substep_dt < 0.002:
            gs.logger.warning(
                "Using a simulation timestep smaller than 2ms is not recommended as it could lead to numerically "
                "unstable collision detection."
            )

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

        self._sol_min_timeconst = TIME_CONSTANT_SAFETY_FACTOR * self._substep_dt
        self._sol_global_timeconst = options.constraint_timeconst

        if options.contact_resolve_time is not None:
            self._sol_global_timeconst = options.contact_resolve_time
            gs.logger.warning(
                "Rigid option 'contact_resolve_time' is deprecated and will be remove in future release. Please use "
                "'constraint_timeconst' instead."
            )

        self._options = options

        self._cur_step = -1

    def add_entity(self, idx, material, morph, surface, visualize_contact) -> Entity:
        if isinstance(material, gs.materials.Avatar):
            EntityClass = AvatarEntity
            if visualize_contact:
                gs.raise_exception("AvatarEntity does not support visualize_contact")
        else:
            if isinstance(morph, gs.morphs.Drone):
                EntityClass = DroneEntity
            else:
                EntityClass = RigidEntity

        if morph.is_free:
            verts_state_start = self.n_free_verts
        else:
            verts_state_start = self.n_fixed_verts

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

        self.n_equalities_candidate = max(1, self.n_equalities + self._options.max_dynamic_constraints)

        if self.is_active():
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
            self._kernel_forward_kinematics_links_geoms(self._scene._envs_idx)

            self._init_invweight()
            self._kernel_init_meaninertia()

    def _init_invweight(self):
        # Early return if no DoFs. This is essential to avoid segfault on CUDA.
        if self._n_dofs == 0:
            return

        # Compute mass matrix without any implicit damping terms
        self._kernel_compute_mass_matrix()

        # Define some proxies for convenience
        mass_mat = self.mass_mat.to_numpy()[:, :, 0]
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

            A = jac @ np.linalg.inv(mass_mat) @ jac.T
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

                A = jac @ np.linalg.inv(mass_mat) @ jac.T
                A_diag = np.diag(A)

                if joint_type == gs.JOINT_TYPE.FREE:
                    dofs_invweight[dof_start : (dof_start + 3)] = A_diag[:3].mean()
                    dofs_invweight[(dof_start + 3) : (dof_start + 6)] = A_diag[3:].mean()
                elif joint_type == gs.JOINT_TYPE.SPHERICAL:
                    dofs_invweight[dof_start : (dof_start + 3)] = A_diag[:3].mean()
                else:  # REVOLUTE or PRISMATIC
                    dofs_invweight[dof_start] = A_diag[0]

        # Update links and dofs invweight for values that are not already pre-computed
        self._kernel_init_invweight(links_invweight, dofs_invweight)

    @ti.kernel
    def _kernel_compute_mass_matrix(self):
        self._func_compute_mass_matrix(implicit_damping=False)

    @ti.kernel
    def _kernel_init_invweight(
        self,
        links_invweight: ti.types.ndarray(),
        dofs_invweight: ti.types.ndarray(),
    ):
        for I in ti.grouped(self.links_info):
            for j in ti.static(range(2)):
                if self.links_info[I].invweight[j] < gs.EPS:
                    self.links_info[I].invweight[j] = links_invweight[I[0], j]

        for I in ti.grouped(self.dofs_info):
            if self.dofs_info[I].invweight < gs.EPS:
                self.dofs_info[I].invweight = dofs_invweight[I[0]]

    @ti.kernel
    def _kernel_init_meaninertia(self):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i_b in range(self._B):
            if self.n_dofs > 0:
                self.meaninertia[i_b] = 0.0
                for i_e in range(self.n_entities):
                    e_info = self.entities_info[i_e]
                    for i_d in range(e_info.dof_start, e_info.dof_end):
                        I_d = [i_d, i_b] if ti.static(self._options.batch_dofs_info) else i_d
                        self.meaninertia[i_b] += self.mass_mat[i_d, i_d, i_b]
                    self.meaninertia[i_b] = self.meaninertia[i_b] / self.n_dofs
            else:
                self.meaninertia[i_b] = 1.0

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
        self.entity_max_dofs = max([entity.n_dofs for entity in self._entities])

        self.mass_mat = ti.field(dtype=gs.ti_float, shape=self._batch_shape((self.n_dofs_, self.n_dofs_)))
        self.mass_mat_L = ti.field(dtype=gs.ti_float, shape=self._batch_shape((self.n_dofs_, self.n_dofs_)))
        self.mass_mat_D_inv = ti.field(dtype=gs.ti_float, shape=self._batch_shape((self.n_dofs_,)))

        self._mass_mat_mask = ti.field(dtype=gs.ti_int, shape=self._batch_shape(self.n_entities_))
        self._mass_mat_mask.fill(1)

        self.meaninertia = ti.field(dtype=gs.ti_float, shape=self._batch_shape())

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

        self.mass_parent_mask = ti.field(dtype=gs.ti_float, shape=(self.n_dofs_, self.n_dofs_))
        self.mass_parent_mask.from_numpy(mass_parent_mask)

        # just in case
        self.mass_mat_L.fill(0)
        self.mass_mat_D_inv.fill(0)
        self.meaninertia.fill(0)

    def _init_dof_fields(self):
        if self._use_hibernation:
            self.n_awake_dofs = ti.field(dtype=gs.ti_int, shape=self._B)
            self.awake_dofs = ti.field(dtype=gs.ti_int, shape=self._batch_shape(self.n_dofs_))

        struct_dof_info = ti.types.struct(
            stiffness=gs.ti_float,
            invweight=gs.ti_float,
            armature=gs.ti_float,
            damping=gs.ti_float,
            motion_ang=gs.ti_vec3,
            motion_vel=gs.ti_vec3,
            limit=gs.ti_vec2,
            dof_start=gs.ti_int,  # dof_start of its entity
            kp=gs.ti_float,
            kv=gs.ti_float,
            force_range=gs.ti_vec2,
        )

        struct_dof_state = ti.types.struct(
            force=gs.ti_float,
            qf_bias=gs.ti_float,
            qf_passive=gs.ti_float,
            qf_actuator=gs.ti_float,
            qf_applied=gs.ti_float,
            act_length=gs.ti_float,
            pos=gs.ti_float,
            vel=gs.ti_float,
            acc=gs.ti_float,
            acc_smooth=gs.ti_float,
            qf_smooth=gs.ti_float,
            qf_constraint=gs.ti_float,
            cdof_ang=gs.ti_vec3,
            cdof_vel=gs.ti_vec3,
            cdofvel_ang=gs.ti_vec3,
            cdofvel_vel=gs.ti_vec3,
            cdofd_ang=gs.ti_vec3,
            cdofd_vel=gs.ti_vec3,
            f_vel=gs.ti_vec3,
            f_ang=gs.ti_vec3,
            ctrl_force=gs.ti_float,
            ctrl_pos=gs.ti_float,
            ctrl_vel=gs.ti_float,
            ctrl_mode=gs.ti_int,
            hibernated=gs.ti_int,  # Flag for dofs that converge into a static state (hibernation)
        )
        dofs_info_shape = self._batch_shape(self.n_dofs_) if self._options.batch_dofs_info else self.n_dofs_
        self.dofs_info = struct_dof_info.field(shape=dofs_info_shape, needs_grad=False, layout=ti.Layout.SOA)
        self.dofs_state = struct_dof_state.field(
            shape=self._batch_shape(self.n_dofs_), needs_grad=False, layout=ti.Layout.SOA
        )

        joints = self.joints
        has_dofs = sum(joint.n_dofs for joint in joints) > 0
        if has_dofs:  # handle the case where there is a link with no dofs -- otherwise may cause invalid memory
            self._kernel_init_dof_fields(
                dofs_motion_ang=np.concatenate([joint.dofs_motion_ang for joint in joints], dtype=gs.np_float),
                dofs_motion_vel=np.concatenate([joint.dofs_motion_vel for joint in joints], dtype=gs.np_float),
                dofs_limit=np.concatenate([joint.dofs_limit for joint in joints], dtype=gs.np_float),
                dofs_invweight=np.concatenate([joint.dofs_invweight for joint in joints], dtype=gs.np_float),
                dofs_stiffness=np.concatenate([joint.dofs_stiffness for joint in joints], dtype=gs.np_float),
                dofs_damping=np.concatenate([joint.dofs_damping for joint in joints], dtype=gs.np_float),
                dofs_armature=np.concatenate([joint.dofs_armature for joint in joints], dtype=gs.np_float),
                dofs_kp=np.concatenate([joint.dofs_kp for joint in joints], dtype=gs.np_float),
                dofs_kv=np.concatenate([joint.dofs_kv for joint in joints], dtype=gs.np_float),
                dofs_force_range=np.concatenate([joint.dofs_force_range for joint in joints], dtype=gs.np_float),
            )

        # just in case
        self.dofs_state.force.fill(0)

    @ti.kernel
    def _kernel_init_dof_fields(
        self,
        dofs_motion_ang: ti.types.ndarray(),
        dofs_motion_vel: ti.types.ndarray(),
        dofs_limit: ti.types.ndarray(),
        dofs_invweight: ti.types.ndarray(),
        dofs_stiffness: ti.types.ndarray(),
        dofs_damping: ti.types.ndarray(),
        dofs_armature: ti.types.ndarray(),
        dofs_kp: ti.types.ndarray(),
        dofs_kv: ti.types.ndarray(),
        dofs_force_range: ti.types.ndarray(),
    ):
        for I in ti.grouped(self.dofs_info):
            i = I[0]  # batching (if any) will be the second dim

            for j in ti.static(range(3)):
                self.dofs_info[I].motion_ang[j] = dofs_motion_ang[i, j]
                self.dofs_info[I].motion_vel[j] = dofs_motion_vel[i, j]

            for j in ti.static(range(2)):
                self.dofs_info[I].limit[j] = dofs_limit[i, j]
                self.dofs_info[I].force_range[j] = dofs_force_range[i, j]

            self.dofs_info[I].armature = dofs_armature[i]
            self.dofs_info[I].invweight = dofs_invweight[i]
            self.dofs_info[I].stiffness = dofs_stiffness[i]
            self.dofs_info[I].damping = dofs_damping[i]
            self.dofs_info[I].kp = dofs_kp[i]
            self.dofs_info[I].kv = dofs_kv[i]

        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i, b in ti.ndrange(self.n_dofs, self._B):
            self.dofs_state[i, b].ctrl_mode = gs.CTRL_MODE.FORCE

        if ti.static(self._use_hibernation):
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
            for i, b in ti.ndrange(self.n_dofs, self._B):
                self.dofs_state[i, b].hibernated = False
                self.awake_dofs[i, b] = i

            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
            for b in range(self._B):
                self.n_awake_dofs[b] = self.n_dofs

    def _init_link_fields(self):
        if self._use_hibernation:
            self.n_awake_links = ti.field(dtype=gs.ti_int, shape=self._B)
            self.awake_links = ti.field(dtype=gs.ti_int, shape=self._batch_shape(self.n_links_))

        struct_link_info = ti.types.struct(
            parent_idx=gs.ti_int,
            root_idx=gs.ti_int,
            q_start=gs.ti_int,
            dof_start=gs.ti_int,
            joint_start=gs.ti_int,
            q_end=gs.ti_int,
            dof_end=gs.ti_int,
            joint_end=gs.ti_int,
            n_dofs=gs.ti_int,
            pos=gs.ti_vec3,
            quat=gs.ti_vec4,
            invweight=gs.ti_vec2,
            is_fixed=gs.ti_int,
            inertial_pos=gs.ti_vec3,
            inertial_quat=gs.ti_vec4,
            inertial_i=gs.ti_mat3,
            inertial_mass=gs.ti_float,
            entity_idx=gs.ti_int,  # entity.idx_in_solver
        )
        struct_link_state = ti.types.struct(
            cinr_inertial=gs.ti_mat3,
            cinr_pos=gs.ti_vec3,
            cinr_quat=gs.ti_vec4,
            cinr_mass=gs.ti_float,
            crb_inertial=gs.ti_mat3,
            crb_pos=gs.ti_vec3,
            crb_quat=gs.ti_vec4,
            crb_mass=gs.ti_float,
            cdd_vel=gs.ti_vec3,
            cdd_ang=gs.ti_vec3,
            pos=gs.ti_vec3,
            quat=gs.ti_vec4,
            i_pos=gs.ti_vec3,
            i_quat=gs.ti_vec4,
            j_pos=gs.ti_vec3,
            j_quat=gs.ti_vec4,
            j_vel=gs.ti_vec3,
            j_ang=gs.ti_vec3,
            # cd
            cd_ang=gs.ti_vec3,
            cd_vel=gs.ti_vec3,
            root_COM=gs.ti_vec3,
            mass_sum=gs.ti_float,
            COM=gs.ti_vec3,
            mass_shift=gs.ti_float,
            i_pos_shift=gs.ti_vec3,
            # COM-based link's spatial acceleration
            cacc_ang=gs.ti_vec3,
            cacc_lin=gs.ti_vec3,
            # COM-based total forces
            cfrc_ang=gs.ti_vec3,
            cfrc_vel=gs.ti_vec3,
            # COM-based external forces explictly applied by the user (i.e. not resulting from any kind of constraints)
            cfrc_applied_ang=gs.ti_vec3,
            cfrc_applied_vel=gs.ti_vec3,
            # net force from external contacts
            contact_force=gs.ti_vec3,
            # Flag for links that converge into a static state (hibernation)
            hibernated=gs.ti_int,
        )

        links_info_shape = self._batch_shape(self.n_links) if self._options.batch_links_info else self.n_links
        self.links_info = struct_link_info.field(shape=links_info_shape, needs_grad=False, layout=ti.Layout.SOA)
        self.links_state = struct_link_state.field(
            shape=self._batch_shape(self.n_links), needs_grad=False, layout=ti.Layout.SOA
        )

        links = self.links
        self._kernel_init_link_fields(
            links_parent_idx=np.array([link.parent_idx for link in links], dtype=gs.np_int),
            links_root_idx=np.array([link.root_idx for link in links], dtype=gs.np_int),
            links_q_start=np.array([link.q_start for link in links], dtype=gs.np_int),
            links_dof_start=np.array([link.dof_start for link in links], dtype=gs.np_int),
            links_joint_start=np.array([link.joint_start for link in links], dtype=gs.np_int),
            links_q_end=np.array([link.q_end for link in links], dtype=gs.np_int),
            links_dof_end=np.array([link.dof_end for link in links], dtype=gs.np_int),
            links_joint_end=np.array([link.joint_end for link in links], dtype=gs.np_int),
            links_invweight=np.array([link.invweight for link in links], dtype=gs.np_float),
            links_is_fixed=np.array([link.is_fixed for link in links], dtype=gs.np_int),
            links_pos=np.array([link.pos for link in links], dtype=gs.np_float),
            links_quat=np.array([link.quat for link in links], dtype=gs.np_float),
            links_inertial_pos=np.array([link.inertial_pos for link in links], dtype=gs.np_float),
            links_inertial_quat=np.array([link.inertial_quat for link in links], dtype=gs.np_float),
            links_inertial_i=np.array([link.inertial_i for link in links], dtype=gs.np_float),
            links_inertial_mass=np.array([link.inertial_mass for link in links], dtype=gs.np_float),
            links_entity_idx=np.array([link._entity_idx_in_solver for link in links], dtype=gs.np_int),
        )

        struct_joint_info = ti.types.struct(
            type=gs.ti_int,
            sol_params=gs.ti_vec7,
            q_start=gs.ti_int,
            dof_start=gs.ti_int,
            q_end=gs.ti_int,
            dof_end=gs.ti_int,
            n_dofs=gs.ti_int,
            pos=gs.ti_vec3,
        )
        struct_joint_state = ti.types.struct(
            xanchor=gs.ti_vec3,
            xaxis=gs.ti_vec3,
        )

        # Field size cannot be zero,
        joints_info_shape = self._batch_shape(self.n_joints_) if self._options.batch_joints_info else self.n_joints_
        self.joints_info = struct_joint_info.field(shape=joints_info_shape, needs_grad=False, layout=ti.Layout.SOA)
        self.joints_state = struct_joint_state.field(
            shape=self._batch_shape(self.n_joints_), needs_grad=False, layout=ti.Layout.SOA
        )

        joints = self.joints
        if joints:
            # Make sure that the constraints parameters are valid
            joints_sol_params = np.array([joint.sol_params for joint in joints], dtype=gs.np_float)
            _sanitize_sol_params(joints_sol_params, self._sol_min_timeconst, self._sol_global_timeconst)

            self._kernel_init_joint_fields(
                joints_type=np.array([joint.type for joint in joints], dtype=gs.np_int),
                joints_sol_params=joints_sol_params,
                joints_q_start=np.array([joint.q_start for joint in joints], dtype=gs.np_int),
                joints_dof_start=np.array([joint.dof_start for joint in joints], dtype=gs.np_int),
                joints_q_end=np.array([joint.q_end for joint in joints], dtype=gs.np_int),
                joints_dof_end=np.array([joint.dof_end for joint in joints], dtype=gs.np_int),
                joints_pos=np.array([joint.pos for joint in joints], dtype=gs.np_float),
            )

        self.qpos0 = ti.field(dtype=gs.ti_float, shape=self._batch_shape(self.n_qs_))
        if self.n_qs > 0:
            init_qpos = self._batch_array(self.init_qpos.astype(gs.np_float))
            self.qpos0.from_numpy(init_qpos)

        # Check if the initial configuration is out-of-bounds
        self.qpos = ti.field(dtype=gs.ti_float, shape=self._batch_shape(self.n_qs_))
        is_init_qpos_out_of_bounds = False
        if self.n_qs > 0:
            init_qpos = self._batch_array(self.init_qpos.astype(gs.np_float))
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
        self.links_T = ti.Matrix.field(n=4, m=4, dtype=gs.ti_float, shape=self.n_links)

    @ti.kernel
    def _kernel_init_link_fields(
        self,
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
    ):
        for I in ti.grouped(self.links_info):
            i = I[0]

            self.links_info[I].parent_idx = links_parent_idx[i]
            self.links_info[I].root_idx = links_root_idx[i]
            self.links_info[I].q_start = links_q_start[i]
            self.links_info[I].joint_start = links_joint_start[i]
            self.links_info[I].dof_start = links_dof_start[i]
            self.links_info[I].q_end = links_q_end[i]
            self.links_info[I].dof_end = links_dof_end[i]
            self.links_info[I].joint_end = links_joint_end[i]
            self.links_info[I].n_dofs = links_dof_end[i] - links_dof_start[i]
            self.links_info[I].is_fixed = links_is_fixed[i]
            self.links_info[I].entity_idx = links_entity_idx[i]

            for j in ti.static(range(2)):
                self.links_info[I].invweight[j] = links_invweight[i, j]

            for j in ti.static(range(4)):
                self.links_info[I].quat[j] = links_quat[i, j]
                self.links_info[I].inertial_quat[j] = links_inertial_quat[i, j]

            for j in ti.static(range(3)):
                self.links_info[I].pos[j] = links_pos[i, j]
                self.links_info[I].inertial_pos[j] = links_inertial_pos[i, j]

            self.links_info[I].inertial_mass = links_inertial_mass[i]
            for j1 in ti.static(range(3)):
                for j2 in ti.static(range(3)):
                    self.links_info[I].inertial_i[j1, j2] = links_inertial_i[i, j1, j2]

        for i, b in ti.ndrange(self.n_links, self._B):
            I = [i, b] if ti.static(self._options.batch_links_info) else i

            # Update state for root fixed link. Their state will not be updated in forward kinematics later but can be manually changed by user.
            if self.links_info[I].parent_idx == -1 and self.links_info[I].is_fixed:
                for j in ti.static(range(4)):
                    self.links_state[i, b].quat[j] = links_quat[i, j]

                for j in ti.static(range(3)):
                    self.links_state[i, b].pos[j] = links_pos[i, j]

            for j in ti.static(range(3)):
                self.links_state[i, b].i_pos_shift[j] = 0.0
            self.links_state[i, b].mass_shift = 0.0

        if ti.static(self._use_hibernation):
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
            for i, b in ti.ndrange(self.n_links, self._B):
                self.links_state[i, b].hibernated = False
                self.awake_links[i, b] = i

            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
            for b in range(self._B):
                self.n_awake_links[b] = self.n_links

    @ti.kernel
    def _kernel_init_joint_fields(
        self,
        joints_type: ti.types.ndarray(),
        joints_sol_params: ti.types.ndarray(),
        joints_q_start: ti.types.ndarray(),
        joints_dof_start: ti.types.ndarray(),
        joints_q_end: ti.types.ndarray(),
        joints_dof_end: ti.types.ndarray(),
        joints_pos: ti.types.ndarray(),
    ):
        for I in ti.grouped(self.joints_info):
            i = I[0]

            self.joints_info[I].type = joints_type[i]
            self.joints_info[I].q_start = joints_q_start[i]
            self.joints_info[I].dof_start = joints_dof_start[i]
            self.joints_info[I].q_end = joints_q_end[i]
            self.joints_info[I].dof_end = joints_dof_end[i]
            self.joints_info[I].n_dofs = joints_dof_end[i] - joints_dof_start[i]

            for j in ti.static(range(7)):
                self.joints_info[I].sol_params[j] = joints_sol_params[i, j]
            for j in ti.static(range(3)):
                self.joints_info[I].pos[j] = joints_pos[i, j]

    def _init_vert_fields(self):
        # collisioin geom
        struct_vert_info = ti.types.struct(
            init_pos=gs.ti_vec3,
            init_normal=gs.ti_vec3,
            geom_idx=gs.ti_int,
            init_center_pos=gs.ti_vec3,
            verts_state_idx=gs.ti_int,
            is_free=gs.ti_int,
        )
        struct_face_info = ti.types.struct(
            verts_idx=gs.ti_ivec3,
            geom_idx=gs.ti_int,
        )
        struct_edge_info = ti.types.struct(
            v0=gs.ti_int,
            v1=gs.ti_int,
            length=gs.ti_float,
        )
        struct_vert_state = ti.types.struct(
            pos=gs.ti_vec3,
        )

        self.verts_info = struct_vert_info.field(shape=(self.n_verts_), needs_grad=False, layout=ti.Layout.SOA)
        self.faces_info = struct_face_info.field(shape=(self.n_faces_), needs_grad=False, layout=ti.Layout.SOA)
        self.edges_info = struct_edge_info.field(shape=(self.n_edges_), needs_grad=False, layout=ti.Layout.SOA)

        if self.n_free_verts > 0:
            self.free_verts_state = struct_vert_state.field(
                shape=self._batch_shape(self.n_free_verts), needs_grad=False, layout=ti.Layout.SOA
            )
        self.fixed_verts_state = struct_vert_state.field(
            shape=(max(1, self.n_fixed_verts),), needs_grad=False, layout=ti.Layout.SOA
        )

        if self.n_verts > 0:
            geoms = self.geoms
            self._kernel_init_vert_fields(
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
            )

    def _init_vvert_fields(self):
        # visual geom
        struct_vvert_info = ti.types.struct(
            init_pos=gs.ti_vec3,
            init_vnormal=gs.ti_vec3,
            vgeom_idx=gs.ti_int,
        )
        struct_vface_info = ti.types.struct(
            vverts_idx=gs.ti_ivec3,
            vgeom_idx=gs.ti_int,
        )

        self.vverts_info = struct_vvert_info.field(shape=(self.n_vverts_), needs_grad=False, layout=ti.Layout.SOA)
        self.vfaces_info = struct_vface_info.field(shape=(self.n_vfaces_), needs_grad=False, layout=ti.Layout.SOA)

        if self.n_vverts > 0:
            vgeoms = self.vgeoms
            self._kernel_init_vvert_fields(
                vverts=np.concatenate([vgeom.init_vverts for vgeom in vgeoms], dtype=gs.np_float),
                vfaces=np.concatenate([vgeom.init_vfaces + vgeom.vvert_start for vgeom in vgeoms], dtype=gs.np_int),
                vnormals=np.concatenate([vgeom.init_vnormals for vgeom in vgeoms], dtype=gs.np_float),
                vverts_vgeom_idx=np.concatenate(
                    [np.full(vgeom.n_vverts, vgeom.idx) for vgeom in vgeoms], dtype=gs.np_int
                ),
            )

    @ti.kernel
    def _kernel_init_vert_fields(
        self,
        verts: ti.types.ndarray(),
        faces: ti.types.ndarray(),
        edges: ti.types.ndarray(),
        normals: ti.types.ndarray(),
        verts_geom_idx: ti.types.ndarray(),
        init_center_pos: ti.types.ndarray(),
        verts_state_idx: ti.types.ndarray(),
        is_free: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i in range(self.n_verts):
            for j in ti.static(range(3)):
                self.verts_info[i].init_pos[j] = verts[i, j]
                self.verts_info[i].init_normal[j] = normals[i, j]
                self.verts_info[i].init_center_pos[j] = init_center_pos[i, j]

            self.verts_info[i].geom_idx = verts_geom_idx[i]
            self.verts_info[i].verts_state_idx = verts_state_idx[i]
            self.verts_info[i].is_free = is_free[i]

        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i in range(self.n_faces):
            for j in ti.static(range(3)):
                self.faces_info[i].verts_idx[j] = faces[i, j]
            self.faces_info[i].geom_idx = verts_geom_idx[faces[i, 0]]

        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i in range(self.n_edges):
            self.edges_info[i].v0 = edges[i, 0]
            self.edges_info[i].v1 = edges[i, 1]
            self.edges_info[i].length = (
                self.verts_info[edges[i, 0]].init_pos - self.verts_info[edges[i, 1]].init_pos
            ).norm()

    @ti.kernel
    def _kernel_init_vvert_fields(
        self,
        vverts: ti.types.ndarray(),
        vfaces: ti.types.ndarray(),
        vnormals: ti.types.ndarray(),
        vverts_vgeom_idx: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i in range(self.n_vverts):
            for j in ti.static(range(3)):
                self.vverts_info[i].init_pos[j] = vverts[i, j]
                self.vverts_info[i].init_vnormal[j] = vnormals[i, j]

            self.vverts_info[i].vgeom_idx = vverts_vgeom_idx[i]

        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i in range(self.n_vfaces):
            for j in ti.static(range(3)):
                self.vfaces_info[i].vverts_idx[j] = vfaces[i, j]
            self.vfaces_info[i].vgeom_idx = vverts_vgeom_idx[vfaces[i, 0]]

    def _init_geom_fields(self):
        struct_geom_info = ti.types.struct(
            pos=gs.ti_vec3,
            center=gs.ti_vec3,
            quat=gs.ti_vec4,
            data=gs.ti_vec7,
            link_idx=gs.ti_int,
            type=gs.ti_int,
            friction=gs.ti_float,
            sol_params=gs.ti_vec7,
            vert_num=gs.ti_int,
            vert_start=gs.ti_int,
            vert_end=gs.ti_int,
            verts_state_start=gs.ti_int,
            verts_state_end=gs.ti_int,
            face_num=gs.ti_int,
            face_start=gs.ti_int,
            face_end=gs.ti_int,
            edge_num=gs.ti_int,
            edge_start=gs.ti_int,
            edge_end=gs.ti_int,
            is_convex=gs.ti_int,
            contype=ti.i32,
            conaffinity=ti.i32,
            is_free=gs.ti_int,
            is_decomposed=gs.ti_int,
            needs_coup=gs.ti_int,
            coup_friction=gs.ti_float,
            coup_softness=gs.ti_float,
            coup_restitution=gs.ti_float,
        )
        struct_geom_state = ti.types.struct(
            pos=gs.ti_vec3,
            quat=gs.ti_vec4,
            aabb_min=gs.ti_vec3,
            aabb_max=gs.ti_vec3,
            verts_updated=gs.ti_int,
            min_buffer_idx=gs.ti_int,
            max_buffer_idx=gs.ti_int,
            hibernated=gs.ti_int,
            friction_ratio=gs.ti_float,
        )

        self.geoms_info = struct_geom_info.field(shape=self.n_geoms_, needs_grad=False, layout=ti.Layout.SOA)
        self.geoms_init_AABB = ti.Vector.field(
            3, dtype=gs.ti_float, shape=(self.n_geoms_, 8)
        )  # stores 8 corners of AABB
        self.geoms_state = struct_geom_state.field(
            shape=self._batch_shape(self.n_geoms_), needs_grad=False, layout=ti.Layout.SOA
        )
        self._geoms_render_T = np.empty((self.n_geoms_, self._B, 4, 4), order="F", dtype=np.float32)

        if self.n_geoms > 0:
            # Make sure that the constraints parameters are valid
            geoms = self.geoms
            geoms_sol_params = np.array([geom.sol_params for geom in geoms], dtype=gs.np_float)
            _sanitize_sol_params(geoms_sol_params, self._sol_min_timeconst, self._sol_global_timeconst)

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

            self._kernel_init_geom_fields(
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
            )

    @ti.kernel
    def _kernel_init_geom_fields(
        self,
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
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i in range(self.n_geoms):
            for j in ti.static(range(3)):
                self.geoms_info[i].pos[j] = geoms_pos[i, j]
                self.geoms_info[i].center[j] = geoms_center[i, j]

            for j in ti.static(range(4)):
                self.geoms_info[i].quat[j] = geoms_quat[i, j]

            for j in ti.static(range(7)):
                self.geoms_info[i].data[j] = geoms_data[i, j]
                self.geoms_info[i].sol_params[j] = geoms_sol_params[i, j]

            self.geoms_info[i].vert_start = geoms_vert_start[i]
            self.geoms_info[i].vert_end = geoms_vert_end[i]
            self.geoms_info[i].vert_num = geoms_vert_end[i] - geoms_vert_start[i]

            self.geoms_info[i].face_start = geoms_face_start[i]
            self.geoms_info[i].face_end = geoms_face_end[i]
            self.geoms_info[i].face_num = geoms_face_end[i] - geoms_face_start[i]

            self.geoms_info[i].edge_start = geoms_edge_start[i]
            self.geoms_info[i].edge_end = geoms_edge_end[i]
            self.geoms_info[i].edge_num = geoms_edge_end[i] - geoms_edge_start[i]

            self.geoms_info[i].verts_state_start = geoms_verts_state_start[i]
            self.geoms_info[i].verts_state_end = geoms_verts_state_end[i]

            self.geoms_info[i].link_idx = geoms_link_idx[i]
            self.geoms_info[i].type = geoms_type[i]
            self.geoms_info[i].friction = geoms_friction[i]

            self.geoms_info[i].is_convex = geoms_is_convex[i]
            self.geoms_info[i].needs_coup = geoms_needs_coup[i]
            self.geoms_info[i].contype = geoms_contype[i]
            self.geoms_info[i].conaffinity = geoms_conaffinity[i]

            self.geoms_info[i].coup_softness = geoms_coup_softness[i]
            self.geoms_info[i].coup_friction = geoms_coup_friction[i]
            self.geoms_info[i].coup_restitution = geoms_coup_restitution[i]

            self.geoms_info[i].is_free = geoms_is_free[i]
            self.geoms_info[i].is_decomposed = geoms_is_decomp[i]

            # compute init AABB.
            # Beware the ordering the this corners is critical and MUST NOT be changed as this order is used elsewhere
            # in the codebase, e.g. overlap estimation between two convex geometries using there bounding boxes.
            lower = gu.ti_vec3(ti.math.inf)
            upper = gu.ti_vec3(-ti.math.inf)
            for i_v in range(geoms_vert_start[i], geoms_vert_end[i]):
                lower = ti.min(lower, self.verts_info[i_v].init_pos)
                upper = ti.max(upper, self.verts_info[i_v].init_pos)
            self.geoms_init_AABB[i, 0] = ti.Vector([lower[0], lower[1], lower[2]], dt=gs.ti_float)
            self.geoms_init_AABB[i, 1] = ti.Vector([lower[0], lower[1], upper[2]], dt=gs.ti_float)
            self.geoms_init_AABB[i, 2] = ti.Vector([lower[0], upper[1], lower[2]], dt=gs.ti_float)
            self.geoms_init_AABB[i, 3] = ti.Vector([lower[0], upper[1], upper[2]], dt=gs.ti_float)
            self.geoms_init_AABB[i, 4] = ti.Vector([upper[0], lower[1], lower[2]], dt=gs.ti_float)
            self.geoms_init_AABB[i, 5] = ti.Vector([upper[0], lower[1], upper[2]], dt=gs.ti_float)
            self.geoms_init_AABB[i, 6] = ti.Vector([upper[0], upper[1], lower[2]], dt=gs.ti_float)
            self.geoms_init_AABB[i, 7] = ti.Vector([upper[0], upper[1], upper[2]], dt=gs.ti_float)

        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i_g, i_b in ti.ndrange(self.n_geoms, self._B):
            self.geoms_state[i_g, i_b].friction_ratio = 1.0

    @ti.kernel
    def _kernel_adjust_link_inertia(
        self,
        link_idx: ti.i32,
        ratio: ti.f32,
    ):
        if ti.static(self._options.batch_links_info):
            for i_b in range(self._B):
                for j in ti.static(range(2)):
                    self.links_info[link_idx, i_b].invweight[j] /= ratio
                self.links_info[link_idx, i_b].inertial_mass *= ratio
                for j1, j2 in ti.static(ti.ndrange(3, 3)):
                    self.links_info[link_idx, i_b].inertial_i[j1, j2] *= ratio
        else:
            for i_b in range(self._B):
                for j in ti.static(range(2)):
                    self.links_info[link_idx].invweight[j] /= ratio
                self.links_info[link_idx].inertial_mass *= ratio
                for j1, j2 in ti.static(ti.ndrange(3, 3)):
                    self.links_info[link_idx].inertial_i[j1, j2] *= ratio

    def _init_vgeom_fields(self):
        struct_vgeom_info = ti.types.struct(
            pos=gs.ti_vec3,
            quat=gs.ti_vec4,
            link_idx=gs.ti_int,
            vvert_num=gs.ti_int,
            vvert_start=gs.ti_int,
            vvert_end=gs.ti_int,
            vface_num=gs.ti_int,
            vface_start=gs.ti_int,
            vface_end=gs.ti_int,
        )
        struct_vgeom_state = ti.types.struct(
            pos=gs.ti_vec3,
            quat=gs.ti_vec4,
        )
        self.vgeoms_info = struct_vgeom_info.field(shape=self.n_vgeoms_, needs_grad=False, layout=ti.Layout.SOA)
        self.vgeoms_state = struct_vgeom_state.field(
            shape=self._batch_shape(self.n_vgeoms_), needs_grad=False, layout=ti.Layout.SOA
        )
        self._vgeoms_render_T = np.empty((self.n_vgeoms_, self._B, 4, 4), order="F", dtype=np.float32)

        if self.n_vgeoms > 0:
            vgeoms = self.vgeoms
            self._kernel_init_vgeom_fields(
                vgeoms_pos=np.array([vgeom.init_pos for vgeom in vgeoms], dtype=gs.np_float),
                vgeoms_quat=np.array([vgeom.init_quat for vgeom in vgeoms], dtype=gs.np_float),
                vgeoms_link_idx=np.array([vgeom.link.idx for vgeom in vgeoms], dtype=gs.np_int),
                vgeoms_vvert_start=np.array([vgeom.vvert_start for vgeom in vgeoms], dtype=gs.np_int),
                vgeoms_vface_start=np.array([vgeom.vface_start for vgeom in vgeoms], dtype=gs.np_int),
                vgeoms_vvert_end=np.array([vgeom.vvert_end for vgeom in vgeoms], dtype=gs.np_int),
                vgeoms_vface_end=np.array([vgeom.vface_end for vgeom in vgeoms], dtype=gs.np_int),
            )

    @ti.kernel
    def _kernel_init_vgeom_fields(
        self,
        vgeoms_pos: ti.types.ndarray(),
        vgeoms_quat: ti.types.ndarray(),
        vgeoms_link_idx: ti.types.ndarray(),
        vgeoms_vvert_start: ti.types.ndarray(),
        vgeoms_vface_start: ti.types.ndarray(),
        vgeoms_vvert_end: ti.types.ndarray(),
        vgeoms_vface_end: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i in range(self.n_vgeoms):
            for j in ti.static(range(3)):
                self.vgeoms_info[i].pos[j] = vgeoms_pos[i, j]

            for j in ti.static(range(4)):
                self.vgeoms_info[i].quat[j] = vgeoms_quat[i, j]

            self.vgeoms_info[i].vvert_start = vgeoms_vvert_start[i]
            self.vgeoms_info[i].vvert_end = vgeoms_vvert_end[i]
            self.vgeoms_info[i].vvert_num = vgeoms_vvert_end[i] - vgeoms_vvert_start[i]

            self.vgeoms_info[i].vface_start = vgeoms_vface_start[i]
            self.vgeoms_info[i].vface_end = vgeoms_vface_end[i]
            self.vgeoms_info[i].vface_num = vgeoms_vface_end[i] - vgeoms_vface_start[i]

            self.vgeoms_info[i].link_idx = vgeoms_link_idx[i]

    def _init_entity_fields(self):
        if self._use_hibernation:
            self.n_awake_entities = ti.field(dtype=gs.ti_int, shape=self._B)
            self.awake_entities = ti.field(dtype=gs.ti_int, shape=self._batch_shape(self.n_entities_))

        struct_entity_info = ti.types.struct(
            dof_start=gs.ti_int,
            dof_end=gs.ti_int,
            n_dofs=gs.ti_int,
            link_start=gs.ti_int,
            link_end=gs.ti_int,
            n_links=gs.ti_int,
            geom_start=gs.ti_int,
            geom_end=gs.ti_int,
            n_geoms=gs.ti_int,
            gravity_compensation=gs.ti_float,
        )

        struct_entity_state = ti.types.struct(
            hibernated=gs.ti_int,
        )

        self.entities_info = struct_entity_info.field(shape=self.n_entities, needs_grad=False, layout=ti.Layout.SOA)
        self.entities_state = struct_entity_state.field(
            shape=self._batch_shape(self.n_entities), needs_grad=False, layout=ti.Layout.SOA
        )

        entities = self._entities
        self._kernel_init_entity_fields(
            entities_dof_start=np.array([entity.dof_start for entity in entities], dtype=gs.np_int),
            entities_dof_end=np.array([entity.dof_end for entity in entities], dtype=gs.np_int),
            entities_link_start=np.array([entity.link_start for entity in entities], dtype=gs.np_int),
            entities_link_end=np.array([entity.link_end for entity in entities], dtype=gs.np_int),
            entities_geom_start=np.array([entity.geom_start for entity in entities], dtype=gs.np_int),
            entities_geom_end=np.array([entity.geom_end for entity in entities], dtype=gs.np_int),
            entities_gravity_compensation=np.array(
                [entity.gravity_compensation for entity in entities], dtype=gs.np_float
            ),
        )

    @ti.kernel
    def _kernel_init_entity_fields(
        self,
        entities_dof_start: ti.types.ndarray(),
        entities_dof_end: ti.types.ndarray(),
        entities_link_start: ti.types.ndarray(),
        entities_link_end: ti.types.ndarray(),
        entities_geom_start: ti.types.ndarray(),
        entities_geom_end: ti.types.ndarray(),
        entities_gravity_compensation: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i in range(self.n_entities):
            self.entities_info[i].dof_start = entities_dof_start[i]
            self.entities_info[i].dof_end = entities_dof_end[i]
            self.entities_info[i].n_dofs = entities_dof_end[i] - entities_dof_start[i]

            self.entities_info[i].link_start = entities_link_start[i]
            self.entities_info[i].link_end = entities_link_end[i]
            self.entities_info[i].n_links = entities_link_end[i] - entities_link_start[i]

            self.entities_info[i].geom_start = entities_geom_start[i]
            self.entities_info[i].geom_end = entities_geom_end[i]
            self.entities_info[i].n_geoms = entities_geom_end[i] - entities_geom_start[i]

            self.entities_info[i].gravity_compensation = entities_gravity_compensation[i]

            if ti.static(self._options.batch_dofs_info):
                for i_d, i_b in ti.ndrange((entities_dof_start[i], entities_dof_end[i]), self._B):
                    self.dofs_info[i_d, i_b].dof_start = entities_dof_start[i]
            else:
                for i_d in range(entities_dof_start[i], entities_dof_end[i]):
                    self.dofs_info[i_d].dof_start = entities_dof_start[i]

        if ti.static(self._use_hibernation):
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
            for i, b in ti.ndrange(self.n_entities, self._B):
                self.entities_state[i, b].hibernated = False
                self.awake_entities[i, b] = i

            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
            for b in range(self._B):
                self.n_awake_entities[b] = self.n_entities

    def _init_equality_fields(self):
        struct_equality_info = ti.types.struct(
            eq_obj1id=gs.ti_int,
            eq_obj2id=gs.ti_int,
            eq_data=gs.ti_vec11,
            eq_type=gs.ti_int,
            sol_params=gs.ti_vec7,
        )
        self.equalities_info = struct_equality_info.field(
            shape=self._batch_shape(self.n_equalities_candidate), needs_grad=False, layout=ti.Layout.SOA
        )
        if self.n_equalities > 0:
            equalities = self.equalities

            equalities_sol_params = np.array([equality.sol_params for equality in equalities], dtype=gs.np_float)
            _sanitize_sol_params(
                equalities_sol_params,
                self._sol_min_timeconst,
                self._sol_global_timeconst,
            )

            self._kernel_init_equality_fields(
                equalities_type=np.array([equality.type for equality in equalities], dtype=gs.np_int),
                equalities_eq_obj1id=np.array([equality.eq_obj1id for equality in equalities], dtype=gs.np_int),
                equalities_eq_obj2id=np.array([equality.eq_obj2id for equality in equalities], dtype=gs.np_int),
                equalities_eq_data=np.array([equality.eq_data for equality in equalities], dtype=gs.np_float),
                equalities_eq_type=np.array([equality.type for equality in equalities], dtype=gs.np_int),
                equalities_sol_params=equalities_sol_params,
            )
            if self._use_contact_island:
                gs.logger.warn("contact island is not supported for equality constraints yet")

    @ti.kernel
    def _kernel_init_equality_fields(
        self,
        equalities_type: ti.types.ndarray(),
        equalities_eq_obj1id: ti.types.ndarray(),
        equalities_eq_obj2id: ti.types.ndarray(),
        equalities_eq_data: ti.types.ndarray(),
        equalities_eq_type: ti.types.ndarray(),
        equalities_sol_params: ti.types.ndarray(),
    ):

        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i, b in ti.ndrange(self.n_equalities, self._B):
            self.equalities_info[i, b].eq_obj1id = equalities_eq_obj1id[i]
            self.equalities_info[i, b].eq_obj2id = equalities_eq_obj2id[i]
            self.equalities_info[i, b].eq_type = equalities_eq_type[i]
            for j in ti.static(range(11)):
                self.equalities_info[i, b].eq_data[j] = equalities_eq_data[i, j]
            for j in ti.static(range(7)):
                self.equalities_info[i, b].sol_params[j] = equalities_sol_params[i, j]

    def _init_envs_offset(self):
        self.envs_offset = ti.Vector.field(3, dtype=gs.ti_float, shape=self._B)
        self.envs_offset.from_numpy(self._scene.envs_offset.astype(gs.np_float))

    def _init_sdf(self):
        self.sdf = SDF(self)

    def _init_collider(self):
        self.collider = Collider(self)

        if self.collider._has_terrain:
            links_idx = self.geoms_info.link_idx.to_numpy()[self.geoms_info.type.to_numpy() == gs.GEOM_TYPE.TERRAIN]
            entity = self._entities[self.links_info.entity_idx.to_numpy()[links_idx[0]]]

            scale = entity.terrain_scale.astype(gs.np_float)
            rc = np.array(entity.terrain_hf.shape, dtype=gs.np_int)
            hf = entity.terrain_hf.astype(gs.np_float) * scale[1]
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

    @ti.func
    def _func_vel_at_point(self, pos_world, link_idx, i_b):
        """
        Velocity of a certain point on a rigid link.
        """
        link_state = self.links_state[link_idx, i_b]
        vel_rot = link_state.cd_ang.cross(pos_world - link_state.COM)
        vel_lin = link_state.cd_vel
        return vel_rot + vel_lin

    @ti.func
    def _func_compute_mass_matrix(self, implicit_damping: ti.template()):
        if ti.static(self._use_hibernation):
            # crb initialize
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_b in range(self._B):
                for i_l_ in range(self.n_awake_links[i_b]):
                    i_l = self.awake_links[i_l_, i_b]
                    self.links_state[i_l, i_b].crb_inertial = self.links_state[i_l, i_b].cinr_inertial
                    self.links_state[i_l, i_b].crb_pos = self.links_state[i_l, i_b].cinr_pos
                    self.links_state[i_l, i_b].crb_quat = self.links_state[i_l, i_b].cinr_quat
                    self.links_state[i_l, i_b].crb_mass = self.links_state[i_l, i_b].cinr_mass

            # crb
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_b in range(self._B):
                for i_e_ in range(self.n_awake_entities[i_b]):
                    i_e = self.awake_entities[i_e_, i_b]
                    for i in range(self.entities_info[i_e].n_links):
                        i_l = self.entities_info[i_e].link_end - 1 - i
                        I_l = [i_l, i_b] if ti.static(self._options.batch_links_info) else i_l
                        i_p = self.links_info[I_l].parent_idx

                        if i_p != -1:
                            self.links_state[i_p, i_b].crb_inertial = (
                                self.links_state[i_p, i_b].crb_inertial + self.links_state[i_l, i_b].crb_inertial
                            )
                            self.links_state[i_p, i_b].crb_mass = (
                                self.links_state[i_p, i_b].crb_mass + self.links_state[i_l, i_b].crb_mass
                            )

                            self.links_state[i_p, i_b].crb_pos = (
                                self.links_state[i_p, i_b].crb_pos + self.links_state[i_l, i_b].crb_pos
                            )
                            self.links_state[i_p, i_b].crb_quat = (
                                self.links_state[i_p, i_b].crb_quat + self.links_state[i_l, i_b].crb_quat
                            )

            # mass_mat
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_b in range(self._B):
                for i_l_ in range(self.n_awake_links[i_b]):
                    i_l = self.awake_links[i_l_, i_b]
                    I_l = [i_l, i_b] if ti.static(self._options.batch_links_info) else i_l
                    l_info = self.links_info[I_l]
                    for i_d in range(l_info.dof_start, l_info.dof_end):
                        self.dofs_state[i_d, i_b].f_ang, self.dofs_state[i_d, i_b].f_vel = gu.inertial_mul(
                            self.links_state[i_l, i_b].crb_pos,
                            self.links_state[i_l, i_b].crb_inertial,
                            self.links_state[i_l, i_b].crb_mass,
                            self.dofs_state[i_d, i_b].cdof_vel,
                            self.dofs_state[i_d, i_b].cdof_ang,
                        )

            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
            for i_b in range(self._B):
                for i_e_ in range(self.n_awake_entities[i_b]):
                    i_e = self.awake_entities[i_e_, i_b]
                    e_info = self.entities_info[i_e]
                    for i_d in range(e_info.dof_start, e_info.dof_end):
                        for j_d in range(e_info.dof_start, e_info.dof_end):
                            self.mass_mat[i_d, j_d, i_b] = (
                                self.dofs_state[i_d, i_b].f_ang.dot(self.dofs_state[j_d, i_b].cdof_ang)
                                + self.dofs_state[i_d, i_b].f_vel.dot(self.dofs_state[j_d, i_b].cdof_vel)
                            ) * self.mass_parent_mask[i_d, j_d]

                    # FIXME: Updating the lower-part of the mass matrix is irrelevant
                    for i_d in range(e_info.dof_start, e_info.dof_end):
                        for j_d in range(i_d + 1, e_info.dof_end):
                            self.mass_mat[i_d, j_d, i_b] = self.mass_mat[j_d, i_d, i_b]

                    # Take into account motor armature
                    for i_d in range(e_info.dof_start, e_info.dof_end):
                        I_d = [i_d, i_b] if ti.static(self._options.batch_dofs_info) else i_d
                        self.mass_mat[i_d, i_d, i_b] = self.mass_mat[i_d, i_d, i_b] + self.dofs_info[I_d].armature

                    # Take into account first-order correction terms for implicit integration scheme right away
                    if ti.static(implicit_damping):
                        for i_d in range(e_info.dof_start, e_info.dof_end):
                            I_d = [i_d, i_b] if ti.static(self._options.batch_dofs_info) else i_d
                            self.mass_mat[i_d, i_d, i_b] += self.dofs_info[I_d].damping * self._substep_dt
                            if (self.dofs_state[i_d, i_b].ctrl_mode == gs.CTRL_MODE.POSITION) or (
                                self.dofs_state[i_d, i_b].ctrl_mode == gs.CTRL_MODE.VELOCITY
                            ):
                                # qM += d qfrc_actuator / d qvel
                                self.mass_mat[i_d, i_d, i_b] += self.dofs_info[I_d].kv * self._substep_dt
        else:
            # crb initialize
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_l, i_b in ti.ndrange(self.n_links, self._B):
                self.links_state[i_l, i_b].crb_inertial = self.links_state[i_l, i_b].cinr_inertial
                self.links_state[i_l, i_b].crb_pos = self.links_state[i_l, i_b].cinr_pos
                self.links_state[i_l, i_b].crb_quat = self.links_state[i_l, i_b].cinr_quat
                self.links_state[i_l, i_b].crb_mass = self.links_state[i_l, i_b].cinr_mass

            # crb
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_e, i_b in ti.ndrange(self.n_entities, self._B):
                for i in range(self.entities_info[i_e].n_links):
                    i_l = self.entities_info[i_e].link_end - 1 - i
                    I_l = [i_l, i_b] if ti.static(self._options.batch_links_info) else i_l
                    i_p = self.links_info[I_l].parent_idx

                    if i_p != -1:
                        self.links_state[i_p, i_b].crb_inertial = (
                            self.links_state[i_p, i_b].crb_inertial + self.links_state[i_l, i_b].crb_inertial
                        )
                        self.links_state[i_p, i_b].crb_mass = (
                            self.links_state[i_p, i_b].crb_mass + self.links_state[i_l, i_b].crb_mass
                        )

                        self.links_state[i_p, i_b].crb_pos = (
                            self.links_state[i_p, i_b].crb_pos + self.links_state[i_l, i_b].crb_pos
                        )
                        self.links_state[i_p, i_b].crb_quat = (
                            self.links_state[i_p, i_b].crb_quat + self.links_state[i_l, i_b].crb_quat
                        )

            # mass_mat
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_l, i_b in ti.ndrange(self.n_links, self._B):
                I_l = [i_l, i_b] if ti.static(self._options.batch_links_info) else i_l
                l_info = self.links_info[I_l]
                for i_d in range(l_info.dof_start, l_info.dof_end):
                    self.dofs_state[i_d, i_b].f_ang, self.dofs_state[i_d, i_b].f_vel = gu.inertial_mul(
                        self.links_state[i_l, i_b].crb_pos,
                        self.links_state[i_l, i_b].crb_inertial,
                        self.links_state[i_l, i_b].crb_mass,
                        self.dofs_state[i_d, i_b].cdof_vel,
                        self.dofs_state[i_d, i_b].cdof_ang,
                    )

            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
            for i_e, i_b in ti.ndrange(self.n_entities, self._B):
                e_info = self.entities_info[i_e]
                for i_d, j_d in ti.ndrange((e_info.dof_start, e_info.dof_end), (e_info.dof_start, e_info.dof_end)):
                    self.mass_mat[i_d, j_d, i_b] = (
                        self.dofs_state[i_d, i_b].f_ang.dot(self.dofs_state[j_d, i_b].cdof_ang)
                        + self.dofs_state[i_d, i_b].f_vel.dot(self.dofs_state[j_d, i_b].cdof_vel)
                    ) * self.mass_parent_mask[i_d, j_d]

                # FIXME: Updating the lower-part of the mass matrix is irrelevant
                for i_d in range(e_info.dof_start, e_info.dof_end):
                    for j_d in range(i_d + 1, e_info.dof_end):
                        self.mass_mat[i_d, j_d, i_b] = self.mass_mat[j_d, i_d, i_b]

            # Take into account motor armature
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_d, i_b in ti.ndrange(self.n_dofs, self._B):
                I_d = [i_d, i_b] if ti.static(self._options.batch_dofs_info) else i_d
                self.mass_mat[i_d, i_d, i_b] = self.mass_mat[i_d, i_d, i_b] + self.dofs_info[I_d].armature

            # Take into account first-order correction terms for implicit integration scheme right away
            if ti.static(implicit_damping):
                ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
                for i_d, i_b in ti.ndrange(self.n_dofs, self._B):
                    I_d = [i_d, i_b] if ti.static(self._options.batch_dofs_info) else i_d
                    self.mass_mat[i_d, i_d, i_b] += self.dofs_info[I_d].damping * self._substep_dt
                    if (self.dofs_state[i_d, i_b].ctrl_mode == gs.CTRL_MODE.POSITION) or (
                        self.dofs_state[i_d, i_b].ctrl_mode == gs.CTRL_MODE.VELOCITY
                    ):
                        # qM += d qfrc_actuator / d qvel
                        self.mass_mat[i_d, i_d, i_b] += self.dofs_info[I_d].kv * self._substep_dt

    @ti.func
    def _func_factor_mass(self, implicit_damping: ti.template()):
        """
        Compute Cholesky decomposition (L^T @ D @ L) of mass matrix.
        """
        if ti.static(self._use_hibernation):
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
            for i_b in range(self._B):
                for i_e_ in range(self.n_awake_entities[i_b]):
                    i_e = self.awake_entities[i_e_, i_b]

                    if self._mass_mat_mask[i_e, i_b] == 1:
                        entity_dof_start = self.entities_info[i_e].dof_start
                        entity_dof_end = self.entities_info[i_e].dof_end
                        n_dofs = self.entities_info[i_e].n_dofs

                        for i_d in range(entity_dof_start, entity_dof_end):
                            for j_d in range(entity_dof_start, i_d + 1):
                                self.mass_mat_L[i_d, j_d, i_b] = self.mass_mat[i_d, j_d, i_b]

                            if ti.static(implicit_damping):
                                I_d = [i_d, i_b] if ti.static(self._options.batch_dofs_info) else i_d
                                self.mass_mat_L[i_d, i_d, i_b] += self.dofs_info[I_d].damping * self._substep_dt
                                if ti.static(self._integrator == gs.integrator.implicitfast):
                                    if (self.dofs_state[i_d, i_b].ctrl_mode == gs.CTRL_MODE.POSITION) or (
                                        self.dofs_state[i_d, i_b].ctrl_mode == gs.CTRL_MODE.VELOCITY
                                    ):
                                        self.mass_mat_L[i_d, i_d, i_b] += self.dofs_info[I_d].kv * self._substep_dt

                        for i_d_ in range(n_dofs):
                            i_d = entity_dof_end - i_d_ - 1
                            self.mass_mat_D_inv[i_d, i_b] = 1.0 / self.mass_mat_L[i_d, i_d, i_b]

                            for j_d_ in range(i_d - entity_dof_start):
                                j_d = i_d - j_d_ - 1
                                a = self.mass_mat_L[i_d, j_d, i_b] * self.mass_mat_D_inv[i_d, i_b]
                                for k_d in range(entity_dof_start, j_d + 1):
                                    self.mass_mat_L[j_d, k_d, i_b] -= a * self.mass_mat_L[i_d, k_d, i_b]
                                self.mass_mat_L[i_d, j_d, i_b] = a

                            # FIXME: Diagonal coeffs of L are ignored in computations, so no need to update them.
                            self.mass_mat_L[i_d, i_d, i_b] = 1.0
        else:
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
            for i_e, i_b in ti.ndrange(self.n_entities, self._B):
                if self._mass_mat_mask[i_e, i_b] == 1:
                    entity_dof_start = self.entities_info[i_e].dof_start
                    entity_dof_end = self.entities_info[i_e].dof_end
                    n_dofs = self.entities_info[i_e].n_dofs

                    for i_d in range(entity_dof_start, entity_dof_end):
                        for j_d in range(entity_dof_start, i_d + 1):
                            self.mass_mat_L[i_d, j_d, i_b] = self.mass_mat[i_d, j_d, i_b]

                        if ti.static(implicit_damping):
                            I_d = [i_d, i_b] if ti.static(self._options.batch_dofs_info) else i_d
                            self.mass_mat_L[i_d, i_d, i_b] += self.dofs_info[I_d].damping * self._substep_dt
                            if ti.static(self._integrator == gs.integrator.implicitfast):
                                if (self.dofs_state[i_d, i_b].ctrl_mode == gs.CTRL_MODE.POSITION) or (
                                    self.dofs_state[i_d, i_b].ctrl_mode == gs.CTRL_MODE.VELOCITY
                                ):
                                    self.mass_mat_L[i_d, i_d, i_b] += self.dofs_info[I_d].kv * self._substep_dt

                    for i_d_ in range(n_dofs):
                        i_d = entity_dof_end - i_d_ - 1
                        self.mass_mat_D_inv[i_d, i_b] = 1.0 / self.mass_mat_L[i_d, i_d, i_b]

                        for j_d_ in range(i_d - entity_dof_start):
                            j_d = i_d - j_d_ - 1
                            a = self.mass_mat_L[i_d, j_d, i_b] * self.mass_mat_D_inv[i_d, i_b]
                            for k_d in range(entity_dof_start, j_d + 1):
                                self.mass_mat_L[j_d, k_d, i_b] -= a * self.mass_mat_L[i_d, k_d, i_b]
                            self.mass_mat_L[i_d, j_d, i_b] = a

                        # FIXME: Diagonal coeffs of L are ignored in computations, so no need to update them.
                        self.mass_mat_L[i_d, i_d, i_b] = 1.0

    @ti.func
    def _func_solve_mass_batched(self, vec, out, i_b):
        if ti.static(self._use_hibernation):
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
            for i_e_ in range(self.n_awake_entities[i_b]):
                i_e = self.awake_entities[i_e_, i_b]

                if self._mass_mat_mask[i_e, i_b] == 1:
                    entity_dof_start = self.entities_info[i_e].dof_start
                    entity_dof_end = self.entities_info[i_e].dof_end
                    n_dofs = self.entities_info[i_e].n_dofs

                    # Step 1: Solve w st. L^T @ w = y
                    for i_d_ in range(n_dofs):
                        i_d = entity_dof_end - i_d_ - 1
                        out[i_d, i_b] = vec[i_d, i_b]
                        for j_d in range(i_d + 1, entity_dof_end):
                            out[i_d, i_b] -= self.mass_mat_L[j_d, i_d, i_b] * out[j_d, i_b]

                    # Step 2: z = D^{-1} w
                    for i_d in range(entity_dof_start, entity_dof_end):
                        out[i_d, i_b] *= self.mass_mat_D_inv[i_d, i_b]

                    # Step 3: Solve x st. L @ x = z
                    for i_d in range(entity_dof_start, entity_dof_end):
                        for j_d in range(entity_dof_start, i_d):
                            out[i_d, i_b] -= self.mass_mat_L[i_d, j_d, i_b] * out[j_d, i_b]
        else:
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
            for i_e in range(self.n_entities):
                if self._mass_mat_mask[i_e, i_b] == 1:
                    entity_dof_start = self.entities_info[i_e].dof_start
                    entity_dof_end = self.entities_info[i_e].dof_end
                    n_dofs = self.entities_info[i_e].n_dofs

                    # Step 1: Solve w st. L^T @ w = y
                    for i_d_ in range(n_dofs):
                        i_d = entity_dof_end - i_d_ - 1
                        out[i_d, i_b] = vec[i_d, i_b]
                        for j_d in range(i_d + 1, entity_dof_end):
                            out[i_d, i_b] -= self.mass_mat_L[j_d, i_d, i_b] * out[j_d, i_b]

                    # Step 2: z = D^{-1} w
                    for i_d in range(entity_dof_start, entity_dof_end):
                        out[i_d, i_b] *= self.mass_mat_D_inv[i_d, i_b]

                    # Step 3: Solve x st. L @ x = z
                    for i_d in range(entity_dof_start, entity_dof_end):
                        for j_d in range(entity_dof_start, i_d):
                            out[i_d, i_b] -= self.mass_mat_L[i_d, j_d, i_b] * out[j_d, i_b]

    @ti.func
    def _func_solve_mass(self, vec, out):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i_b in range(self._B):
            self._func_solve_mass_batched(vec, out, i_b)

    @ti.kernel
    def _kernel_forward_dynamics(self):
        self._func_forward_dynamics()

    @ti.kernel
    def _kernel_update_acc(self):
        self._func_update_acc(update_cacc=True)

    # @@@@@@@@@ Composer starts here
    # decomposed kernels should happen in the block below. This block will be handled by composer and composed into a single kernel
    @ti.func
    def _func_forward_dynamics(self):
        self._func_compute_mass_matrix(
            implicit_damping=ti.static(self._integrator == gs.integrator.approximate_implicitfast)
        )
        self._func_factor_mass(implicit_damping=False)
        self._func_torque_and_passive_force()
        self._func_update_acc(update_cacc=False)
        self._func_update_force()
        # self._func_actuation()
        self._func_bias_force()
        self._func_compute_qacc()

    @ti.kernel
    def _kernel_clear_external_force(self):
        self._func_clear_external_force()

    def substep(self):
        from genesis.utils.tools import create_timer

        timer = create_timer("rigid", level=1, ti_sync=True, skip_first_call=True)
        self._kernel_step_1()
        timer.stamp("kernel_step_1")

        # constraint force
        self._func_constraint_force()
        timer.stamp("constraint_force")

        self._kernel_step_2()
        timer.stamp("kernel_step_2")

    @ti.kernel
    def _kernel_step_1(self):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(self._B):
            self._func_forward_kinematics(i_b)
            self._func_COM_links(i_b)
            self._func_forward_velocity(i_b)
            self._func_update_geoms(i_b)

        self._func_forward_dynamics()

    @ti.func
    def _func_implicit_damping(self):
        # Determine whether the mass matrix must be re-computed to take into account first-order correction terms.
        # Note that avoiding inverting the mass matrix twice would not only speed up simulation but also improving
        # numerical stability as computing post-damping accelerations from forces is not necessary anymore.
        if ti.static(not self._enable_mujoco_compatibility or self._integrator == gs.integrator.Euler):
            self._mass_mat_mask.fill(0)
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
            for i_e, i_b in ti.ndrange(self.n_entities, self._B):
                entity_dof_start = self.entities_info[i_e].dof_start
                entity_dof_end = self.entities_info[i_e].dof_end
                for i_d in range(entity_dof_start, entity_dof_end):
                    I_d = [i_d, i_b] if ti.static(self._options.batch_dofs_info) else i_d
                    if self.dofs_info[I_d].damping > gs.EPS:
                        self._mass_mat_mask[i_e, i_b] = 1
                    if ti.static(self._integrator != gs.integrator.Euler):
                        if (
                            (self.dofs_state[i_d, i_b].ctrl_mode == gs.CTRL_MODE.POSITION)
                            or (self.dofs_state[i_d, i_b].ctrl_mode == gs.CTRL_MODE.VELOCITY)
                        ) and self.dofs_info[I_d].kv > gs.EPS:
                            self._mass_mat_mask[i_e, i_b] = 1

        self._func_factor_mass(implicit_damping=True)
        self._func_solve_mass(self.dofs_state.force, self.dofs_state.acc)

        # Disable pre-computed factorization mask right away
        if ti.static(not self._enable_mujoco_compatibility or self._integrator == gs.integrator.Euler):
            self._mass_mat_mask.fill(1)

    @ti.kernel
    def _kernel_step_2(self):
        # Position, Velocity and Acceleration data must be consistent when computing links acceleration, otherwise it
        # would not corresponds to anyting physical. There is no other way than doing this right before integration,
        # because the acceleration at the end of the step is unknown for now as it may change discontinuous between
        # before and after integration under the effect of external forces and constraints. This means that
        # acceleration data will be shifted one timestep in the past, but there isn't really any way around.
        self._func_update_acc(update_cacc=True)

        if ti.static(self._integrator != gs.integrator.approximate_implicitfast):
            self._func_implicit_damping()

        self._func_integrate()

        if ti.static(self._use_hibernation):
            self._func_hibernate()
            self._func_aggregate_awake_entities()

    def _kernel_detect_collision(self):
        self.collider.clear()
        self.collider.detection()

    # @@@@@@@@@ Composer ends here

    def detect_collision(self, env_idx=0):
        # TODO: support batching
        self._kernel_detect_collision()
        n_collision = self.collider.n_contacts.to_numpy()[env_idx]
        collision_pairs = np.empty((n_collision, 2), dtype=np.int32)
        collision_pairs[:, 0] = self.collider.contact_data.geom_a.to_numpy()[:n_collision, env_idx]
        collision_pairs[:, 1] = self.collider.contact_data.geom_b.to_numpy()[:n_collision, env_idx]
        return collision_pairs

    @ti.kernel
    def _kernel_forward_kinematics_links_geoms(self, envs_idx: ti.types.ndarray()):
        for i_b in envs_idx:
            self._func_forward_kinematics(i_b)
            self._func_COM_links(i_b)
            self._func_forward_velocity(i_b)
            self._func_update_geoms(i_b)

    def _func_constraint_force(self):
        from genesis.utils.tools import create_timer

        timer = create_timer(name="constraint_force", level=2, ti_sync=True, skip_first_call=True)
        if self._enable_collision or self._enable_joint_limit or self.n_equalities > 0:
            self._func_constraint_clear()
            timer.stamp("constraint_solver.clear")

        if self._enable_collision:
            self.collider.detection()
            timer.stamp("detection")

        if not self._disable_constraint:
            self.constraint_solver.handle_constraints()
        timer.stamp("constraint_solver.handle_constraints")

    @ti.kernel
    def _func_constraint_clear(self):
        self.constraint_solver.n_constraints.fill(0)
        if ti.static(not self._use_contact_island):
            self.constraint_solver.n_constraints_equality.fill(0)

        if self._enable_collision:
            if ti.static(self._use_hibernation):
                self.n_contacts_hibernated.fill(0)

                ti.loop_config(serialize=self._solver._para_level < gs.PARA_LEVEL.ALL)
                for i_b in range(self._B):
                    # Advect hibernated contacts
                    for i_c in range(self.n_contacts[i_b]):
                        i_la = self.contact_data[i_c, i_b].link_a
                        i_lb = self.contact_data[i_c, i_b].link_b
                        I_la = [i_la, i_b] if ti.static(self._solver._options.batch_links_info) else i_la
                        I_lb = [i_lb, i_b] if ti.static(self._solver._options.batch_links_info) else i_lb

                        # Pair of hibernated-fixed links -> hibernated contact
                        # TODO: we should also include hibernated-hibernated links and wake up the whole contact island
                        # once a new collision is detected
                        if (
                            self._solver.links_state[i_la, i_b].hibernated and self._solver.links_info[I_lb].is_fixed
                        ) or (
                            self._solver.links_state[i_lb, i_b].hibernated and self._solver.links_info[I_la].is_fixed
                        ):
                            i_c_hibernated = self.n_contacts_hibernated[i_b]
                            if i_c != i_c_hibernated:
                                self.contact_data[i_c_hibernated, i_b] = self.contact_data[i_c, i_b]
                            self.n_contacts_hibernated[i_b] = i_c_hibernated + 1

                    self.n_contacts[i_b] = self.n_contacts_hibernated[i_b]
            else:
                self.collider.n_contacts.fill(0)

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

    @ti.func
    def _func_COM_links(self, i_b):
        if ti.static(self._use_hibernation):
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_l_ in range(self.n_awake_links[i_b]):
                i_l = self.awake_links[i_l_, i_b]

                self.links_state[i_l, i_b].root_COM = ti.Vector.zero(gs.ti_float, 3)
                self.links_state[i_l, i_b].mass_sum = 0.0

            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_l_ in range(self.n_awake_links[i_b]):
                i_l = self.awake_links[i_l_, i_b]
                I_l = [i_l, i_b] if ti.static(self._options.batch_links_info) else i_l

                l = self.links_state[i_l, i_b]
                l_info = self.links_info[I_l]
                mass = l_info.inertial_mass + l.mass_shift
                (
                    self.links_state[i_l, i_b].i_pos,
                    self.links_state[i_l, i_b].i_quat,
                ) = gu.ti_transform_pos_quat_by_trans_quat(
                    l_info.inertial_pos + l.i_pos_shift, l_info.inertial_quat, l.pos, l.quat
                )

                i_r = self.links_info[I_l].root_idx
                ti.atomic_add(self.links_state[i_r, i_b].mass_sum, mass)

                COM = mass * self.links_state[i_l, i_b].i_pos
                ti.atomic_add(self.links_state[i_r, i_b].root_COM, COM)

            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_l_ in range(self.n_awake_links[i_b]):
                i_l = self.awake_links[i_l_, i_b]
                I_l = [i_l, i_b] if ti.static(self._options.batch_links_info) else i_l

                i_r = self.links_info[I_l].root_idx
                if i_l == i_r:
                    self.links_state[i_l, i_b].root_COM = (
                        self.links_state[i_l, i_b].root_COM / self.links_state[i_l, i_b].mass_sum
                    )

            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_l_ in range(self.n_awake_links[i_b]):
                i_l = self.awake_links[i_l_, i_b]
                I_l = [i_l, i_b] if ti.static(self._options.batch_links_info) else i_l

                i_r = self.links_info[I_l].root_idx
                self.links_state[i_l, i_b].root_COM = self.links_state[i_r, i_b].root_COM

            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_l_ in range(self.n_awake_links[i_b]):
                i_l = self.awake_links[i_l_, i_b]
                I_l = [i_l, i_b] if ti.static(self._options.batch_links_info) else i_l

                l = self.links_state[i_l, i_b]
                l_info = self.links_info[I_l]

                i_r = self.links_info[I_l].root_idx
                self.links_state[i_l, i_b].COM = self.links_state[i_r, i_b].root_COM
                self.links_state[i_l, i_b].i_pos = self.links_state[i_l, i_b].i_pos - self.links_state[i_l, i_b].COM

                i_inertial = l_info.inertial_i
                i_mass = l_info.inertial_mass + l.mass_shift
                (
                    self.links_state[i_l, i_b].cinr_inertial,
                    self.links_state[i_l, i_b].cinr_pos,
                    self.links_state[i_l, i_b].cinr_quat,
                    self.links_state[i_l, i_b].cinr_mass,
                ) = gu.ti_transform_inertia_by_trans_quat(
                    i_inertial, i_mass, self.links_state[i_l, i_b].i_pos, self.links_state[i_l, i_b].i_quat
                )

            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_l_ in range(self.n_awake_links[i_b]):
                i_l = self.awake_links[i_l_, i_b]
                I_l = [i_l, i_b] if ti.static(self._options.batch_links_info) else i_l
                l_info = self.links_info[I_l]
                if l_info.n_dofs == 0:
                    continue

                i_p = l_info.parent_idx

                _i_j = self.links_info[I_l].joint_start
                _I_j = [_i_j, i_b] if ti.static(self._options.batch_joints_info) else _i_j
                joint_type = self.joints_info[_I_j].type

                p_pos = ti.Vector.zero(gs.ti_float, 3)
                p_quat = gu.ti_identity_quat()
                if i_p != -1:
                    p_pos = self.links_state[i_p, i_b].pos
                    p_quat = self.links_state[i_p, i_b].quat

                if joint_type == gs.JOINT_TYPE.FREE or (l_info.is_fixed and i_p == -1):
                    self.links_state[i_l, i_b].j_pos = self.links_state[i_l, i_b].pos
                    self.links_state[i_l, i_b].j_quat = self.links_state[i_l, i_b].quat
                else:
                    (
                        self.links_state[i_l, i_b].j_pos,
                        self.links_state[i_l, i_b].j_quat,
                    ) = gu.ti_transform_pos_quat_by_trans_quat(l_info.pos, l_info.quat, p_pos, p_quat)

                    for i_j in range(l_info.joint_start, l_info.joint_end):
                        I_j = [i_j, i_b] if ti.static(self._options.batch_joints_info) else i_j
                        j_info = self.joints_info[I_j]

                        (
                            self.links_state[i_l, i_b].j_pos,
                            self.links_state[i_l, i_b].j_quat,
                        ) = gu.ti_transform_pos_quat_by_trans_quat(
                            j_info.pos,
                            gu.ti_identity_quat(),
                            self.links_state[i_l, i_b].j_pos,
                            self.links_state[i_l, i_b].j_quat,
                        )

            # cdof_fn
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_l_ in range(self.n_awake_links[i_b]):
                i_l = self.awake_links[i_l_, i_b]
                I_l = [i_l, i_b] if ti.static(self._options.batch_links_info) else i_l
                l_info = self.links_info[I_l]
                if l_info.n_dofs == 0:
                    continue

                i_j = l_info.joint_start
                I_j = [i_j, i_b] if ti.static(self._options.batch_joints_info) else i_j
                joint_type = self.joints_info[I_j].type

                if joint_type == gs.JOINT_TYPE.FREE:
                    for i_d in range(l_info.dof_start, l_info.dof_end):
                        I_d = [i_d, i_b] if ti.static(self._options.batch_dofs_info) else i_d
                        self.dofs_state[i_d, i_b].cdof_vel = self.dofs_info[I_d].motion_vel
                        self.dofs_state[i_d, i_b].cdof_ang = gu.ti_transform_by_quat(
                            self.dofs_info[I_d].motion_ang, self.links_state[i_l, i_b].j_quat
                        )

                        offset_pos = self.links_state[i_l, i_b].COM - self.links_state[i_l, i_b].j_pos
                        (
                            self.dofs_state[i_d, i_b].cdof_ang,
                            self.dofs_state[i_d, i_b].cdof_vel,
                        ) = gu.ti_transform_motion_by_trans_quat(
                            self.dofs_state[i_d, i_b].cdof_ang,
                            self.dofs_state[i_d, i_b].cdof_vel,
                            offset_pos,
                            gu.ti_identity_quat(),
                        )

                        self.dofs_state[i_d, i_b].cdofvel_ang = (
                            self.dofs_state[i_d, i_b].cdof_ang * self.dofs_state[i_d, i_b].vel
                        )
                        self.dofs_state[i_d, i_b].cdofvel_vel = (
                            self.dofs_state[i_d, i_b].cdof_vel * self.dofs_state[i_d, i_b].vel
                        )

                elif joint_type == gs.JOINT_TYPE.FIXED:
                    pass
                else:
                    for i_d in range(l_info.dof_start, l_info.dof_end):
                        I_d = [i_d, i_b] if ti.static(self._options.batch_dofs_info) else i_d
                        motion_vel = self.dofs_info[I_d].motion_vel
                        motion_ang = self.dofs_info[I_d].motion_ang

                        self.dofs_state[i_d, i_b].cdof_ang = gu.ti_transform_by_quat(
                            motion_ang, self.links_state[i_l, i_b].j_quat
                        )
                        self.dofs_state[i_d, i_b].cdof_vel = gu.ti_transform_by_quat(
                            motion_vel, self.links_state[i_l, i_b].j_quat
                        )

                        offset_pos = self.links_state[i_l, i_b].COM - self.links_state[i_l, i_b].j_pos
                        (
                            self.dofs_state[i_d, i_b].cdof_ang,
                            self.dofs_state[i_d, i_b].cdof_vel,
                        ) = gu.ti_transform_motion_by_trans_quat(
                            self.dofs_state[i_d, i_b].cdof_ang,
                            self.dofs_state[i_d, i_b].cdof_vel,
                            offset_pos,
                            gu.ti_identity_quat(),
                        )

                        self.dofs_state[i_d, i_b].cdofvel_ang = (
                            self.dofs_state[i_d, i_b].cdof_ang * self.dofs_state[i_d, i_b].vel
                        )
                        self.dofs_state[i_d, i_b].cdofvel_vel = (
                            self.dofs_state[i_d, i_b].cdof_vel * self.dofs_state[i_d, i_b].vel
                        )
        else:
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_l in range(self.n_links):
                self.links_state[i_l, i_b].root_COM = ti.Vector.zero(gs.ti_float, 3)
                self.links_state[i_l, i_b].mass_sum = 0.0

            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_l in range(self.n_links):
                I_l = [i_l, i_b] if ti.static(self._options.batch_links_info) else i_l

                l = self.links_state[i_l, i_b]
                l_info = self.links_info[I_l]
                mass = l_info.inertial_mass + l.mass_shift
                (
                    self.links_state[i_l, i_b].i_pos,
                    self.links_state[i_l, i_b].i_quat,
                ) = gu.ti_transform_pos_quat_by_trans_quat(
                    l_info.inertial_pos + l.i_pos_shift, l_info.inertial_quat, l.pos, l.quat
                )

                i_r = self.links_info[I_l].root_idx
                ti.atomic_add(self.links_state[i_r, i_b].mass_sum, mass)

                COM = mass * self.links_state[i_l, i_b].i_pos
                ti.atomic_add(self.links_state[i_r, i_b].root_COM, COM)

            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_l in range(self.n_links):
                I_l = [i_l, i_b] if ti.static(self._options.batch_links_info) else i_l

                i_r = self.links_info[I_l].root_idx
                if i_l == i_r:
                    if self.links_state[i_l, i_b].mass_sum > 0.0:
                        self.links_state[i_l, i_b].root_COM = (
                            self.links_state[i_l, i_b].root_COM / self.links_state[i_l, i_b].mass_sum
                        )

            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_l in range(self.n_links):
                I_l = [i_l, i_b] if ti.static(self._options.batch_links_info) else i_l

                i_r = self.links_info[I_l].root_idx
                self.links_state[i_l, i_b].root_COM = self.links_state[i_r, i_b].root_COM

            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_l in range(self.n_links):
                I_l = [i_l, i_b] if ti.static(self._options.batch_links_info) else i_l

                l = self.links_state[i_l, i_b]
                l_info = self.links_info[I_l]

                i_r = self.links_info[I_l].root_idx
                self.links_state[i_l, i_b].COM = self.links_state[i_r, i_b].root_COM
                self.links_state[i_l, i_b].i_pos = self.links_state[i_l, i_b].i_pos - self.links_state[i_l, i_b].COM

                i_inertial = l_info.inertial_i
                i_mass = l_info.inertial_mass + l.mass_shift
                (
                    self.links_state[i_l, i_b].cinr_inertial,
                    self.links_state[i_l, i_b].cinr_pos,
                    self.links_state[i_l, i_b].cinr_quat,
                    self.links_state[i_l, i_b].cinr_mass,
                ) = gu.ti_transform_inertia_by_trans_quat(
                    i_inertial, i_mass, self.links_state[i_l, i_b].i_pos, self.links_state[i_l, i_b].i_quat
                )

            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_l in range(self.n_links):
                I_l = [i_l, i_b] if ti.static(self._options.batch_links_info) else i_l
                l_info = self.links_info[I_l]
                if l_info.n_dofs == 0:
                    continue
                i_p = l_info.parent_idx

                _i_j = l_info.joint_start
                _I_j = [_i_j, i_b] if ti.static(self._options.batch_joints_info) else _i_j
                joint_type = self.joints_info[_I_j].type

                p_pos = ti.Vector.zero(gs.ti_float, 3)
                p_quat = gu.ti_identity_quat()
                if i_p != -1:
                    p_pos = self.links_state[i_p, i_b].pos
                    p_quat = self.links_state[i_p, i_b].quat

                if joint_type == gs.JOINT_TYPE.FREE or (l_info.is_fixed and i_p == -1):
                    self.links_state[i_l, i_b].j_pos = self.links_state[i_l, i_b].pos
                    self.links_state[i_l, i_b].j_quat = self.links_state[i_l, i_b].quat
                else:
                    (
                        self.links_state[i_l, i_b].j_pos,
                        self.links_state[i_l, i_b].j_quat,
                    ) = gu.ti_transform_pos_quat_by_trans_quat(l_info.pos, l_info.quat, p_pos, p_quat)

                    for i_j in range(l_info.joint_start, l_info.joint_end):
                        I_j = [i_j, i_b] if ti.static(self._options.batch_joints_info) else i_j
                        j_info = self.joints_info[I_j]

                        (
                            self.links_state[i_l, i_b].j_pos,
                            self.links_state[i_l, i_b].j_quat,
                        ) = gu.ti_transform_pos_quat_by_trans_quat(
                            j_info.pos,
                            gu.ti_identity_quat(),
                            self.links_state[i_l, i_b].j_pos,
                            self.links_state[i_l, i_b].j_quat,
                        )

            # cdof_fn
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_l in range(self.n_links):
                I_l = [i_l, i_b] if ti.static(self._options.batch_links_info) else i_l
                l_info = self.links_info[I_l]

                for i_j in range(l_info.joint_start, l_info.joint_end):
                    offset_pos = self.links_state[i_l, i_b].COM - self.joints_state[i_j, i_b].xanchor
                    I_j = [i_j, i_b] if ti.static(self._options.batch_joints_info) else i_j
                    j_info = self.joints_info[I_j]
                    joint_type = j_info.type

                    dof_start = j_info.dof_start

                    if joint_type == gs.JOINT_TYPE.REVOLUTE:
                        self.dofs_state[dof_start, i_b].cdof_ang = self.joints_state[i_j, i_b].xaxis
                        self.dofs_state[dof_start, i_b].cdof_vel = self.joints_state[i_j, i_b].xaxis.cross(offset_pos)
                    elif joint_type == gs.JOINT_TYPE.PRISMATIC:
                        self.dofs_state[dof_start, i_b].cdof_ang = ti.Vector.zero(gs.ti_float, 3)
                        self.dofs_state[dof_start, i_b].cdof_vel = self.joints_state[i_j, i_b].xaxis
                    elif joint_type == gs.JOINT_TYPE.SPHERICAL:
                        xmat_T = gu.ti_quat_to_R(self.links_state[i_l, i_b].quat).transpose()
                        for i in ti.static(range(3)):
                            self.dofs_state[i + dof_start, i_b].cdof_ang = xmat_T[i, :]
                            self.dofs_state[i + dof_start, i_b].cdof_vel = xmat_T[i, :].cross(offset_pos)
                    elif joint_type == gs.JOINT_TYPE.FREE:
                        for i in ti.static(range(3)):
                            self.dofs_state[i + dof_start, i_b].cdof_ang = ti.Vector.zero(gs.ti_float, 3)
                            self.dofs_state[i + dof_start, i_b].cdof_vel = ti.Vector.zero(gs.ti_float, 3)
                            self.dofs_state[i + dof_start, i_b].cdof_vel[i] = 1.0

                        xmat_T = gu.ti_quat_to_R(self.links_state[i_l, i_b].quat).transpose()
                        for i in ti.static(range(3)):
                            self.dofs_state[i + dof_start + 3, i_b].cdof_ang = xmat_T[i, :]
                            self.dofs_state[i + dof_start + 3, i_b].cdof_vel = xmat_T[i, :].cross(offset_pos)

                    for i_d in range(dof_start, j_info.dof_end):
                        self.dofs_state[i_d, i_b].cdofvel_ang = (
                            self.dofs_state[i_d, i_b].cdof_ang * self.dofs_state[i_d, i_b].vel
                        )
                        self.dofs_state[i_d, i_b].cdofvel_vel = (
                            self.dofs_state[i_d, i_b].cdof_vel * self.dofs_state[i_d, i_b].vel
                        )

    @ti.func
    def _func_forward_kinematics(self, i_b):
        if ti.static(self._use_hibernation):
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_e_ in range(self.n_awake_entities[i_b]):
                i_e = self.awake_entities[i_e_, i_b]
                self._func_forward_kinematics_entity(i_e, i_b)
        else:
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_e in range(self.n_entities):
                self._func_forward_kinematics_entity(i_e, i_b)

    @ti.func
    def _func_forward_velocity(self, i_b):
        if ti.static(self._use_hibernation):
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_e_ in range(self.n_awake_entities[i_b]):
                i_e = self.awake_entities[i_e_, i_b]
                self._func_forward_velocity_entity(i_e, i_b)
        else:
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_e in range(self.n_entities):
                self._func_forward_velocity_entity(i_e, i_b)

    @ti.func
    def _func_forward_kinematics_entity(self, i_e, i_b):
        for i_l in range(self.entities_info[i_e].link_start, self.entities_info[i_e].link_end):
            I_l = [i_l, i_b] if ti.static(self._options.batch_links_info) else i_l
            l_info = self.links_info[I_l]

            pos = l_info.pos
            quat = l_info.quat
            if l_info.parent_idx != -1:
                parent_pos = self.links_state[l_info.parent_idx, i_b].pos
                parent_quat = self.links_state[l_info.parent_idx, i_b].quat
                pos = parent_pos + gu.ti_transform_by_quat(pos, parent_quat)
                quat = gu.ti_transform_quat_by_quat(quat, parent_quat)

            for i_j in range(l_info.joint_start, l_info.joint_end):
                I_j = [i_j, i_b] if ti.static(self._options.batch_joints_info) else i_j
                j_info = self.joints_info[I_j]
                joint_type = j_info.type
                q_start = j_info.q_start
                dof_start = j_info.dof_start
                I_d = [dof_start, i_b] if ti.static(self._options.batch_dofs_info) else dof_start

                # compute axis and anchor
                if joint_type == gs.JOINT_TYPE.FREE:
                    self.joints_state[i_j, i_b].xanchor = ti.Vector(
                        [self.qpos[q_start, i_b], self.qpos[q_start + 1, i_b], self.qpos[q_start + 2, i_b]]
                    )
                    self.joints_state[i_j, i_b].xaxis = ti.Vector([0.0, 0.0, 1.0])
                elif joint_type == gs.JOINT_TYPE.FIXED:
                    pass
                else:
                    axis = ti.Vector([0.0, 0.0, 1.0], dt=gs.ti_float)
                    if joint_type == gs.JOINT_TYPE.REVOLUTE:
                        axis = self.dofs_info[I_d].motion_ang
                    elif joint_type == gs.JOINT_TYPE.PRISMATIC:
                        axis = self.dofs_info[I_d].motion_vel

                    self.joints_state[i_j, i_b].xanchor = gu.ti_transform_by_quat(j_info.pos, quat) + pos
                    self.joints_state[i_j, i_b].xaxis = gu.ti_transform_by_quat(axis, quat)

                if joint_type == gs.JOINT_TYPE.FREE:
                    pos = ti.Vector(
                        [self.qpos[q_start, i_b], self.qpos[q_start + 1, i_b], self.qpos[q_start + 2, i_b]],
                        dt=gs.ti_float,
                    )
                    quat = ti.Vector(
                        [
                            self.qpos[q_start + 3, i_b],
                            self.qpos[q_start + 4, i_b],
                            self.qpos[q_start + 5, i_b],
                            self.qpos[q_start + 6, i_b],
                        ],
                        dt=gs.ti_float,
                    )
                    quat = gu.ti_normalize(quat)
                    xyz = gu.ti_quat_to_xyz(quat)
                    for i in ti.static(range(3)):
                        self.dofs_state[dof_start + i, i_b].pos = pos[i]
                        self.dofs_state[dof_start + 3 + i, i_b].pos = xyz[i]
                elif joint_type == gs.JOINT_TYPE.FIXED:
                    pass
                elif joint_type == gs.JOINT_TYPE.SPHERICAL:
                    qloc = ti.Vector(
                        [
                            self.qpos[q_start, i_b],
                            self.qpos[q_start + 1, i_b],
                            self.qpos[q_start + 2, i_b],
                            self.qpos[q_start + 3, i_b],
                        ],
                        dt=gs.ti_float,
                    )
                    xyz = gu.ti_quat_to_xyz(qloc)
                    for i in ti.static(range(3)):
                        self.dofs_state[dof_start + i, i_b].pos = xyz[i]
                    quat = gu.ti_transform_quat_by_quat(qloc, quat)
                    pos = self.joints_state[i_j, i_b].xanchor - gu.ti_transform_by_quat(j_info.pos, quat)
                elif joint_type == gs.JOINT_TYPE.REVOLUTE:
                    axis = self.dofs_info[I_d].motion_ang
                    self.dofs_state[dof_start, i_b].pos = self.qpos[q_start, i_b] - self.qpos0[q_start, i_b]
                    qloc = gu.ti_rotvec_to_quat(axis * self.dofs_state[dof_start, i_b].pos)
                    quat = gu.ti_transform_quat_by_quat(qloc, quat)
                    pos = self.joints_state[i_j, i_b].xanchor - gu.ti_transform_by_quat(j_info.pos, quat)
                else:  # joint_type == gs.JOINT_TYPE.PRISMATIC:
                    self.dofs_state[dof_start, i_b].pos = self.qpos[q_start, i_b] - self.qpos0[q_start, i_b]
                    pos = pos + self.joints_state[i_j, i_b].xaxis * self.dofs_state[dof_start, i_b].pos

            # Skip link pose update for fixed root links to let users manually overwrite them
            if not (l_info.parent_idx == -1 and l_info.is_fixed):
                self.links_state[i_l, i_b].pos = pos
                self.links_state[i_l, i_b].quat = quat

    @ti.func
    def _func_forward_velocity_entity(self, i_e, i_b):
        for i_l in range(self.entities_info[i_e].link_start, self.entities_info[i_e].link_end):
            I_l = [i_l, i_b] if ti.static(self._options.batch_links_info) else i_l
            l_info = self.links_info[I_l]

            cvel_vel = ti.Vector.zero(gs.ti_float, 3)
            cvel_ang = ti.Vector.zero(gs.ti_float, 3)
            if l_info.parent_idx != -1:
                cvel_vel = self.links_state[l_info.parent_idx, i_b].cd_vel
                cvel_ang = self.links_state[l_info.parent_idx, i_b].cd_ang

            for i_j in range(l_info.joint_start, l_info.joint_end):
                I_j = [i_j, i_b] if ti.static(self._options.batch_joints_info) else i_j
                j_info = self.joints_info[I_j]
                joint_type = j_info.type
                q_start = j_info.q_start
                dof_start = j_info.dof_start

                if joint_type == gs.JOINT_TYPE.FREE:
                    for i_3 in ti.static(range(3)):
                        cvel_vel = (
                            cvel_vel
                            + self.dofs_state[dof_start + i_3, i_b].cdof_vel * self.dofs_state[dof_start + i_3, i_b].vel
                        )
                        cvel_ang = (
                            cvel_ang
                            + self.dofs_state[dof_start + i_3, i_b].cdof_ang * self.dofs_state[dof_start + i_3, i_b].vel
                        )

                    for i_3 in ti.static(range(3)):
                        (
                            self.dofs_state[dof_start + i_3, i_b].cdofd_ang,
                            self.dofs_state[dof_start + i_3, i_b].cdofd_vel,
                        ) = ti.Vector.zero(gs.ti_float, 3), ti.Vector.zero(gs.ti_float, 3)

                        (
                            self.dofs_state[dof_start + i_3 + 3, i_b].cdofd_ang,
                            self.dofs_state[dof_start + i_3 + 3, i_b].cdofd_vel,
                        ) = gu.motion_cross_motion(
                            cvel_ang,
                            cvel_vel,
                            self.dofs_state[dof_start + i_3 + 3, i_b].cdof_ang,
                            self.dofs_state[dof_start + i_3 + 3, i_b].cdof_vel,
                        )

                    for i_3 in ti.static(range(3)):
                        cvel_vel = (
                            cvel_vel
                            + self.dofs_state[dof_start + i_3 + 3, i_b].cdof_vel
                            * self.dofs_state[dof_start + i_3 + 3, i_b].vel
                        )
                        cvel_ang = (
                            cvel_ang
                            + self.dofs_state[dof_start + i_3 + 3, i_b].cdof_ang
                            * self.dofs_state[dof_start + i_3 + 3, i_b].vel
                        )

                else:
                    for i_d in range(dof_start, j_info.dof_end):
                        self.dofs_state[i_d, i_b].cdofd_ang, self.dofs_state[i_d, i_b].cdofd_vel = (
                            gu.motion_cross_motion(
                                cvel_ang,
                                cvel_vel,
                                self.dofs_state[i_d, i_b].cdof_ang,
                                self.dofs_state[i_d, i_b].cdof_vel,
                            )
                        )
                    for i_d in range(dof_start, j_info.dof_end):
                        cvel_vel = cvel_vel + self.dofs_state[i_d, i_b].cdof_vel * self.dofs_state[i_d, i_b].vel
                        cvel_ang = cvel_ang + self.dofs_state[i_d, i_b].cdof_ang * self.dofs_state[i_d, i_b].vel

            self.links_state[i_l, i_b].cd_vel = cvel_vel
            self.links_state[i_l, i_b].cd_ang = cvel_ang

    @ti.func
    def _func_update_geoms(self, i_b):
        """
        NOTE: this only update geom pose, not its verts and else.
        """
        if ti.static(self._use_hibernation):
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
            for i_e_ in range(self.n_awake_entities[i_b]):
                i_e = self.awake_entities[i_e_, i_b]
                e_info = self.entities_info[i_e]
                for i_g in range(e_info.geom_start, e_info.geom_end):
                    g_info = self.geoms_info[i_g]

                    l_state = self.links_state[g_info.link_idx, i_b]
                    (
                        self.geoms_state[i_g, i_b].pos,
                        self.geoms_state[i_g, i_b].quat,
                    ) = gu.ti_transform_pos_quat_by_trans_quat(g_info.pos, g_info.quat, l_state.pos, l_state.quat)

                    self.geoms_state[i_g, i_b].verts_updated = 0
        else:
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
            for i_g in range(self.n_geoms):
                g_info = self.geoms_info[i_g]

                l_state = self.links_state[g_info.link_idx, i_b]
                (
                    self.geoms_state[i_g, i_b].pos,
                    self.geoms_state[i_g, i_b].quat,
                ) = gu.ti_transform_pos_quat_by_trans_quat(g_info.pos, g_info.quat, l_state.pos, l_state.quat)

                self.geoms_state[i_g, i_b].verts_updated = 0

    @ti.func
    def _func_update_verts_for_geom(self, i_g, i_b):
        g_state = self.geoms_state[i_g, i_b]
        if not g_state.verts_updated:
            g_info = self.geoms_info[i_g]
            if g_info.is_free:
                for i_v in range(g_info.vert_start, g_info.vert_end):
                    verts_state_idx = self.verts_info[i_v].verts_state_idx
                    self.free_verts_state[verts_state_idx, i_b].pos = gu.ti_transform_by_trans_quat(
                        self.verts_info[i_v].init_pos, g_state.pos, g_state.quat
                    )
                self.geoms_state[i_g, i_b].verts_updated = 1
            elif i_b == 0:
                for i_v in range(g_info.vert_start, g_info.vert_end):
                    verts_state_idx = self.verts_info[i_v].verts_state_idx
                    self.fixed_verts_state[verts_state_idx].pos = gu.ti_transform_by_trans_quat(
                        self.verts_info[i_v].init_pos, g_state.pos, g_state.quat
                    )
                self.geoms_state[i_g, 0].verts_updated = 1

    @ti.func
    def _func_update_all_verts(self):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i_v, i_b in ti.ndrange(self.n_verts, self._B):
            g_state = self.geoms_state[self.verts_info[i_v].geom_idx, i_b]
            verts_state_idx = self.verts_info[i_v].verts_state_idx
            if self.verts_info[i_v].is_free:
                self.free_verts_state[verts_state_idx, i_b].pos = gu.ti_transform_by_trans_quat(
                    self.verts_info[i_v].init_pos, g_state.pos, g_state.quat
                )
            elif i_b == 0:
                self.fixed_verts_state[verts_state_idx].pos = gu.ti_transform_by_trans_quat(
                    self.verts_info[i_v].init_pos, g_state.pos, g_state.quat
                )

    @ti.func
    def _func_update_geom_aabbs(self):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i_g, i_b in ti.ndrange(self.n_geoms, self._B):
            g_state = self.geoms_state[i_g, i_b]

            lower = gu.ti_vec3(ti.math.inf)
            upper = gu.ti_vec3(-ti.math.inf)
            for i_corner in ti.static(range(8)):
                corner_pos = gu.ti_transform_by_trans_quat(
                    self.geoms_init_AABB[i_g, i_corner], g_state.pos, g_state.quat
                )
                lower = ti.min(lower, corner_pos)
                upper = ti.max(upper, corner_pos)

            self.geoms_state[i_g, i_b].aabb_min = lower
            self.geoms_state[i_g, i_b].aabb_max = upper

    @ti.kernel
    def _kernel_update_vgeoms(self):
        """
        Vgeoms are only for visualization purposes.
        """
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i_g, i_b in ti.ndrange(self.n_vgeoms, self._B):
            g_info = self.vgeoms_info[i_g]
            l = self.links_state[g_info.link_idx, i_b]
            self.vgeoms_state[i_g, i_b].pos, self.vgeoms_state[i_g, i_b].quat = gu.ti_transform_pos_quat_by_trans_quat(
                g_info.pos, g_info.quat, l.pos, l.quat
            )

    @ti.func
    def _func_hibernate(self):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i_e, i_b in ti.ndrange(self.n_entities, self._B):
            if (
                not self.entities_state[i_e, i_b].hibernated and self.entities_info[i_e].n_dofs > 0
            ):  # We do not hibernate fixed entity
                hibernate = True
                for i_d in range(self.entities_info[i_e].dof_start, self.entities_info[i_e].dof_end):
                    if (
                        ti.abs(self.dofs_state[i_d, i_b].acc) > self._hibernation_thresh_acc
                        or ti.abs(self.dofs_state[i_d, i_b].vel) > self._hibernation_thresh_vel
                    ):
                        hibernate = False
                        break

                if hibernate:
                    self._func_hibernate_entity(i_e, i_b)
                else:
                    # update collider sort_buffer
                    for i_g in range(self.entities_info[i_e].geom_start, self.entities_info[i_e].geom_end):
                        self.collider.sort_buffer[self.geoms_state[i_g, i_b].min_buffer_idx, i_b].value = (
                            self.geoms_state[i_g, i_b].aabb_min[0]
                        )
                        self.collider.sort_buffer[self.geoms_state[i_g, i_b].max_buffer_idx, i_b].value = (
                            self.geoms_state[i_g, i_b].aabb_max[0]
                        )

    @ti.func
    def _func_aggregate_awake_entities(self):
        self.n_awake_entities.fill(0)
        self.n_awake_links.fill(0)
        self.n_awake_dofs.fill(0)
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i_e, i_b in ti.ndrange(self.n_entities, self._B):
            if self.entities_state[i_e, i_b].hibernated or self.entities_info[i_e].n_dofs == 0:
                continue
            n_awake_entities = ti.atomic_add(self.n_awake_entities[i_b], 1)
            self.awake_entities[n_awake_entities, i_b] = i_e

            for i_d in range(self.entities_info[i_e].dof_start, self.entities_info[i_e].dof_end):
                n_awake_dofs = ti.atomic_add(self.n_awake_dofs[i_b], 1)
                self.awake_dofs[n_awake_dofs, i_b] = i_d

            for i_l in range(self.entities_info[i_e].link_start, self.entities_info[i_e].link_end):
                n_awake_links = ti.atomic_add(self.n_awake_links[i_b], 1)
                self.awake_links[n_awake_links, i_b] = i_l

    @ti.func
    def _func_hibernate_entity(self, i_e, i_b):
        e_info = self.entities_info[i_e]

        self.entities_state[i_e, i_b].hibernated = True

        for i_d in range(e_info.dof_start, e_info.dof_end):
            self.dofs_state[i_d, i_b].hibernated = True
            self.dofs_state[i_d, i_b].vel = 0.0
            self.dofs_state[i_d, i_b].acc = 0.0

        for i_l in range(e_info.link_start, e_info.link_end):
            self.links_state[i_l, i_b].hibernated = True

        for i_g in range(e_info.geom_start, e_info.geom_end):
            self.geoms_state[i_g, i_b].hibernated = True

    @ti.func
    def _func_wakeup_entity(self, i_e, i_b):
        if self.entities_state[i_e, i_b].hibernated:
            self.entities_state[i_e, i_b].hibernated = False
            n_awake_entities = ti.atomic_add(self.n_awake_entities[i_b], 1)
            self.awake_entities[n_awake_entities, i_b] = i_e

            e_info = self.entities_info[i_e]

            for i_d in range(e_info.dof_start, e_info.dof_end):
                self.dofs_state[i_d, i_b].hibernated = False
                n_awake_dofs = ti.atomic_add(self.n_awake_dofs[i_b], 1)
                self.awake_dofs[n_awake_dofs, i_b] = i_d

            for i_l in range(e_info.link_start, e_info.link_end):
                self.links_state[i_l, i_b].hibernated = False
                n_awake_links = ti.atomic_add(self.n_awake_links[i_b], 1)
                self.awake_links[n_awake_links, i_b] = i_l

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

        self._kernel_apply_links_external_force(force, links_idx, envs_idx, ref, 1 if local else 0)

    @ti.kernel
    def _kernel_apply_links_external_force(
        self,
        force: ti.types.ndarray(),
        links_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
        ref: ti.template(),
        local: ti.template(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i_l_, i_b_ in ti.ndrange(links_idx.shape[0], envs_idx.shape[0]):
            force_i = ti.Vector([force[i_b_, i_l_, 0], force[i_b_, i_l_, 1], force[i_b_, i_l_, 2]], dt=gs.ti_float)
            self._func_apply_link_external_force(force_i, links_idx[i_l_], envs_idx[i_b_], ref, local)

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

        self._kernel_apply_links_external_torque(torque, links_idx, envs_idx, ref, 1 if local else 0)

    @ti.kernel
    def _kernel_apply_links_external_torque(
        self,
        torque: ti.types.ndarray(),
        links_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
        ref: ti.template(),
        local: ti.template(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i_l_, i_b_ in ti.ndrange(links_idx.shape[0], envs_idx.shape[0]):
            torque_i = ti.Vector([torque[i_b_, i_l_, 0], torque[i_b_, i_l_, 1], torque[i_b_, i_l_, 2]], dt=gs.ti_float)
            self._func_apply_link_external_torque(torque_i, links_idx[i_l_], envs_idx[i_b_], ref, local)

    @ti.func
    def _func_apply_external_force(self, pos, force, link_idx, env_idx):
        torque = (pos - self.links_state[link_idx, env_idx].COM).cross(force)
        self.links_state[link_idx, env_idx].cfrc_applied_ang -= torque
        self.links_state[link_idx, env_idx].cfrc_applied_vel -= force

    @ti.func
    def _func_apply_link_external_force(self, force, link_idx, env_idx, ref: ti.template(), local: ti.template()):
        torque = ti.Vector.zero(gs.ti_float, 3)
        if ti.static(ref == 1):  # link's CoM
            if ti.static(local == 1):
                force = gu.ti_transform_by_quat(force, self.links_state[link_idx, env_idx].i_quat)
            torque = self.links_state[link_idx, env_idx].i_pos.cross(force)
        if ti.static(ref == 2):  # link's origin
            if ti.static(local == 1):
                force = gu.ti_transform_by_quat(force, self.links_state[link_idx, env_idx].quat)
            torque = (self.links_state[link_idx, env_idx].pos - self.links_state[link_idx, env_idx].COM).cross(force)

        self.links_state[link_idx, env_idx].cfrc_applied_vel -= force
        self.links_state[link_idx, env_idx].cfrc_applied_ang -= torque

    @ti.func
    def _func_apply_external_torque(self, torque, link_idx, env_idx):
        self.links_state[link_idx, env_idx].cfrc_applied_ang -= torque

    @ti.func
    def _func_apply_link_external_torque(self, torque, link_idx, env_idx, ref: ti.template(), local: ti.template()):
        if ti.static(ref == 1 and local == 1):  # link's CoM
            torque = gu.ti_transform_by_quat(torque, self.links_state[link_idx, env_idx].i_quat)
        if ti.static(ref == 2 and local == 1):  # link's origin
            torque = gu.ti_transform_by_quat(torque, self.links_state[link_idx, env_idx].quat)

        self.links_state[link_idx, env_idx].cfrc_applied_ang -= torque

    @ti.func
    def _func_clear_external_force(self):
        if ti.static(self._use_hibernation):
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
            for i_b in range(self._B):
                for i_l_ in range(self.n_awake_links[i_b]):
                    i_l = self.awake_links[i_l_, i_b]
                    self.links_state[i_l, i_b].cfrc_applied_ang = ti.Vector.zero(gs.ti_float, 3)
                    self.links_state[i_l, i_b].cfrc_applied_vel = ti.Vector.zero(gs.ti_float, 3)
        else:
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
            for i_l, i_b in ti.ndrange(self.n_links, self._B):
                self.links_state[i_l, i_b].cfrc_applied_ang = ti.Vector.zero(gs.ti_float, 3)
                self.links_state[i_l, i_b].cfrc_applied_vel = ti.Vector.zero(gs.ti_float, 3)

    @ti.func
    def _func_torque_and_passive_force(self):
        # compute force based on each dof's ctrl mode
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
        for i_e, i_b in ti.ndrange(self.n_entities, self._B):
            wakeup = False
            for i_l in range(self.entities_info[i_e].link_start, self.entities_info[i_e].link_end):
                I_l = [i_l, i_b] if ti.static(self._options.batch_links_info) else i_l
                l_info = self.links_info[I_l]
                if l_info.n_dofs == 0:
                    continue

                i_j = l_info.joint_start
                I_j = [i_j, i_b] if ti.static(self._options.batch_joints_info) else i_j
                joint_type = self.joints_info[I_j].type

                for i_d in range(l_info.dof_start, l_info.dof_end):
                    I_d = [i_d, i_b] if ti.static(self._options.batch_dofs_info) else i_d
                    force = gs.ti_float(0.0)
                    if self.dofs_state[i_d, i_b].ctrl_mode == gs.CTRL_MODE.FORCE:
                        force = self.dofs_state[i_d, i_b].ctrl_force
                    elif self.dofs_state[i_d, i_b].ctrl_mode == gs.CTRL_MODE.VELOCITY:
                        force = self.dofs_info[I_d].kv * (
                            self.dofs_state[i_d, i_b].ctrl_vel - self.dofs_state[i_d, i_b].vel
                        )
                    elif self.dofs_state[i_d, i_b].ctrl_mode == gs.CTRL_MODE.POSITION and not (
                        joint_type == gs.JOINT_TYPE.FREE and i_d >= l_info.dof_start + 3
                    ):
                        force = (
                            self.dofs_info[I_d].kp
                            * (self.dofs_state[i_d, i_b].ctrl_pos - self.dofs_state[i_d, i_b].pos)
                            - self.dofs_info[I_d].kv * self.dofs_state[i_d, i_b].vel
                        )

                    self.dofs_state[i_d, i_b].qf_applied = ti.math.clamp(
                        force,
                        self.dofs_info[I_d].force_range[0],
                        self.dofs_info[I_d].force_range[1],
                    )

                    if ti.abs(force) > gs.EPS:
                        wakeup = True

                dof_start = l_info.dof_start
                if joint_type == gs.JOINT_TYPE.FREE and (
                    self.dofs_state[dof_start + 3, i_b].ctrl_mode == gs.CTRL_MODE.POSITION
                    or self.dofs_state[dof_start + 4, i_b].ctrl_mode == gs.CTRL_MODE.POSITION
                    or self.dofs_state[dof_start + 5, i_b].ctrl_mode == gs.CTRL_MODE.POSITION
                ):
                    xyz = ti.Vector(
                        [
                            self.dofs_state[0 + 3 + dof_start, i_b].pos,
                            self.dofs_state[1 + 3 + dof_start, i_b].pos,
                            self.dofs_state[2 + 3 + dof_start, i_b].pos,
                        ],
                        dt=gs.ti_float,
                    )

                    ctrl_xyz = ti.Vector(
                        [
                            self.dofs_state[0 + 3 + dof_start, i_b].ctrl_pos,
                            self.dofs_state[1 + 3 + dof_start, i_b].ctrl_pos,
                            self.dofs_state[2 + 3 + dof_start, i_b].ctrl_pos,
                        ],
                        dt=gs.ti_float,
                    )

                    quat = gu.ti_xyz_to_quat(xyz)
                    ctrl_quat = gu.ti_xyz_to_quat(ctrl_xyz)

                    q_diff = gu.ti_transform_quat_by_quat(ctrl_quat, gu.ti_inv_quat(quat))
                    rotvec = gu.ti_quat_to_rotvec(q_diff)

                    for j in ti.static(range(3)):
                        i_d = dof_start + 3 + j
                        I_d = [i_d, i_b] if ti.static(self._options.batch_dofs_info) else i_d
                        force = (
                            self.dofs_info[I_d].kp * rotvec[j] - self.dofs_info[I_d].kv * self.dofs_state[i_d, i_b].vel
                        )

                        self.dofs_state[i_d, i_b].qf_applied = ti.math.clamp(
                            force, self.dofs_info[I_d].force_range[0], self.dofs_info[I_d].force_range[1]
                        )

                        if ti.abs(force) > gs.EPS:
                            wakeup = True

            if ti.static(self._use_hibernation):
                if wakeup:
                    self._func_wakeup_entity(i_e, i_b)

        if ti.static(self._use_hibernation):
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_b in range(self._B):
                for i_d_ in range(self.n_awake_dofs[i_b]):
                    i_d = self.awake_dofs[i_d_, i_b]
                    I_d = [i_d, i_b] if ti.static(self._options.batch_dofs_info) else i_d

                    self.dofs_state[i_d, i_b].qf_passive = -self.dofs_info[I_d].damping * self.dofs_state[i_d, i_b].vel

            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_b in range(self._B):
                for i_l_ in range(self.n_awake_links[i_b]):
                    i_l = self.awake_links[i_l_, i_b]
                    I_l = [i_l, i_b] if ti.static(self._options.batch_links_info) else i_l
                    l_info = self.links_info[I_l]
                    if l_info.n_dofs == 0:
                        continue

                    i_j = l_info.joint_start
                    I_j = [i_j, i_b] if ti.static(self._options.batch_joints_info) else i_j
                    joint_type = self.joints_info[I_j].type

                    if joint_type != gs.JOINT_TYPE.FREE and joint_type != gs.JOINT_TYPE.FIXED:
                        dof_start = l_info.dof_start
                        q_start = l_info.q_start
                        q_end = l_info.q_end

                        for j_d in range(q_end - q_start):
                            I_d = (
                                [dof_start + j_d, i_b] if ti.static(self._options.batch_dofs_info) else dof_start + j_d
                            )
                            self.dofs_state[dof_start + j_d, i_b].qf_passive += (
                                -self.qpos[q_start + j_d, i_b] * self.dofs_info[I_d].stiffness
                            )
        else:
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_d, i_b in ti.ndrange(self.n_dofs, self._B):
                I_d = [i_d, i_b] if ti.static(self._options.batch_dofs_info) else i_d
                self.dofs_state[i_d, i_b].qf_passive = -self.dofs_info[I_d].damping * self.dofs_state[i_d, i_b].vel

            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_l, i_b in ti.ndrange(self.n_links, self._B):
                I_l = [i_l, i_b] if ti.static(self._options.batch_links_info) else i_l
                l_info = self.links_info[I_l]
                if l_info.n_dofs == 0:
                    continue

                i_j = l_info.joint_start
                I_j = [i_j, i_b] if ti.static(self._options.batch_joints_info) else i_j
                joint_type = self.joints_info[I_j].type

                if joint_type != gs.JOINT_TYPE.FREE and joint_type != gs.JOINT_TYPE.FIXED:
                    dof_start = l_info.dof_start
                    q_start = l_info.q_start
                    q_end = l_info.q_end

                    for j_d in range(q_end - q_start):
                        I_d = [dof_start + j_d, i_b] if ti.static(self._options.batch_dofs_info) else dof_start + j_d
                        self.dofs_state[dof_start + j_d, i_b].qf_passive += (
                            -self.qpos[q_start + j_d, i_b] * self.dofs_info[I_d].stiffness
                        )

    @ti.func
    def _func_update_acc(self, update_cacc: ti.template()):
        if ti.static(self._use_hibernation):
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_b in range(self._B):
                for i_e_ in range(self.n_awake_entities[i_b]):
                    i_e = self.awake_entities[i_e_, i_b]
                    e_info = self.entities_info[i_e]
                    for i_l in range(e_info.link_start, e_info.link_end):
                        I_l = [i_l, i_b] if ti.static(self._options.batch_links_info) else i_l
                        i_p = self.links_info[I_l].parent_idx

                        if i_p == -1:
                            self.links_state[i_l, i_b].cdd_vel = -self._gravity[None] * (
                                1 - e_info.gravity_compensation
                            )
                            self.links_state[i_l, i_b].cdd_ang = ti.Vector.zero(gs.ti_float, 3)
                            if ti.static(update_cacc):
                                self.links_state[i_l, i_b].cacc_lin = ti.Vector.zero(gs.ti_float, 3)
                                self.links_state[i_l, i_b].cacc_ang = ti.Vector.zero(gs.ti_float, 3)
                        else:
                            self.links_state[i_l, i_b].cdd_vel = self.links_state[i_p, i_b].cdd_vel
                            self.links_state[i_l, i_b].cdd_ang = self.links_state[i_p, i_b].cdd_ang
                            if ti.static(update_cacc):
                                self.links_state[i_l, i_b].cacc_lin = self.links_state[i_p, i_b].cacc_lin
                                self.links_state[i_l, i_b].cacc_ang = self.links_state[i_p, i_b].cacc_ang

                        for i_d in range(self.links_info[I_l].dof_start, self.links_info[I_l].dof_end):
                            local_cdd_vel = self.dofs_state[i_d, i_b].cdofd_vel * self.dofs_state[i_d, i_b].vel
                            local_cdd_ang = self.dofs_state[i_d, i_b].cdofd_ang * self.dofs_state[i_d, i_b].vel
                            self.links_state[i_l, i_b].cdd_vel = self.links_state[i_l, i_b].cdd_vel + local_cdd_vel
                            self.links_state[i_l, i_b].cdd_ang = self.links_state[i_l, i_b].cdd_ang + local_cdd_ang
                            if ti.static(update_cacc):
                                self.links_state[i_l, i_b].cacc_lin = (
                                    self.links_state[i_l, i_b].cacc_lin
                                    + local_cdd_vel
                                    + self.dofs_state[i_d, i_b].cdof_vel * self.dofs_state[i_d, i_b].acc
                                )
                                self.links_state[i_l, i_b].cacc_ang = (
                                    self.links_state[i_l, i_b].cacc_ang
                                    + local_cdd_ang
                                    + self.dofs_state[i_d, i_b].cdof_ang * self.dofs_state[i_d, i_b].acc
                                )
        else:
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_e, i_b in ti.ndrange(self.n_entities, self._B):
                e_info = self.entities_info[i_e]
                for i_l in range(e_info.link_start, e_info.link_end):
                    I_l = [i_l, i_b] if ti.static(self._options.batch_links_info) else i_l
                    i_p = self.links_info[I_l].parent_idx

                    if i_p == -1:
                        self.links_state[i_l, i_b].cdd_vel = -self._gravity[None] * (1 - e_info.gravity_compensation)
                        self.links_state[i_l, i_b].cdd_ang = ti.Vector.zero(gs.ti_float, 3)
                        if ti.static(update_cacc):
                            self.links_state[i_l, i_b].cacc_lin = ti.Vector.zero(gs.ti_float, 3)
                            self.links_state[i_l, i_b].cacc_ang = ti.Vector.zero(gs.ti_float, 3)
                    else:
                        self.links_state[i_l, i_b].cdd_vel = self.links_state[i_p, i_b].cdd_vel
                        self.links_state[i_l, i_b].cdd_ang = self.links_state[i_p, i_b].cdd_ang
                        if ti.static(update_cacc):
                            self.links_state[i_l, i_b].cacc_lin = self.links_state[i_p, i_b].cacc_lin
                            self.links_state[i_l, i_b].cacc_ang = self.links_state[i_p, i_b].cacc_ang

                    for i_d in range(self.links_info[I_l].dof_start, self.links_info[I_l].dof_end):
                        # cacc = cacc_parent + cdofdot * qvel + cdof * qacc
                        local_cdd_vel = self.dofs_state[i_d, i_b].cdofd_vel * self.dofs_state[i_d, i_b].vel
                        local_cdd_ang = self.dofs_state[i_d, i_b].cdofd_ang * self.dofs_state[i_d, i_b].vel
                        self.links_state[i_l, i_b].cdd_vel = self.links_state[i_l, i_b].cdd_vel + local_cdd_vel
                        self.links_state[i_l, i_b].cdd_ang = self.links_state[i_l, i_b].cdd_ang + local_cdd_ang
                        if ti.static(update_cacc):
                            self.links_state[i_l, i_b].cacc_lin = (
                                self.links_state[i_l, i_b].cacc_lin
                                + local_cdd_vel
                                + self.dofs_state[i_d, i_b].cdof_vel * self.dofs_state[i_d, i_b].acc
                            )
                            self.links_state[i_l, i_b].cacc_ang = (
                                self.links_state[i_l, i_b].cacc_ang
                                + local_cdd_ang
                                + self.dofs_state[i_d, i_b].cdof_ang * self.dofs_state[i_d, i_b].acc
                            )

    @ti.func
    def _func_update_force(self):
        if ti.static(self._use_hibernation):
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_b in range(self._B):
                for i_l_ in range(self.n_awake_links[i_b]):
                    i_l = self.awake_links[i_l_, i_b]

                    f1_ang, f1_vel = gu.inertial_mul(
                        self.links_state[i_l, i_b].cinr_pos,
                        self.links_state[i_l, i_b].cinr_inertial,
                        self.links_state[i_l, i_b].cinr_mass,
                        self.links_state[i_l, i_b].cdd_vel,
                        self.links_state[i_l, i_b].cdd_ang,
                    )
                    f2_ang, f2_vel = gu.inertial_mul(
                        self.links_state[i_l, i_b].cinr_pos,
                        self.links_state[i_l, i_b].cinr_inertial,
                        self.links_state[i_l, i_b].cinr_mass,
                        self.links_state[i_l, i_b].cd_vel,
                        self.links_state[i_l, i_b].cd_ang,
                    )
                    f2_ang, f2_vel = gu.motion_cross_force(
                        self.links_state[i_l, i_b].cd_ang, self.links_state[i_l, i_b].cd_vel, f2_ang, f2_vel
                    )

                    self.links_state[i_l, i_b].cfrc_vel = f1_vel + f2_vel + self.links_state[i_l, i_b].cfrc_applied_vel
                    self.links_state[i_l, i_b].cfrc_ang = f1_ang + f2_ang + self.links_state[i_l, i_b].cfrc_applied_ang

            for i_b in range(self._B):
                for i_e_ in range(self.n_awake_entities[i_b]):
                    i_e = self.awake_entities[i_e_, i_b]
                    e_info = self.entities_info[i_e]
                    for i in range(e_info.n_links):
                        i_l = e_info.link_end - 1 - i
                        I_l = [i_l, i_b] if ti.static(self._options.batch_links_info) else i_l
                        i_p = self.links_info[I_l].parent_idx
                        if i_p != -1:
                            self.links_state[i_p, i_b].cfrc_vel = (
                                self.links_state[i_p, i_b].cfrc_vel + self.links_state[i_l, i_b].cfrc_vel
                            )
                            self.links_state[i_p, i_b].cfrc_ang = (
                                self.links_state[i_p, i_b].cfrc_ang + self.links_state[i_l, i_b].cfrc_ang
                            )
        else:
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_l, i_b in ti.ndrange(self.n_links, self._B):
                f1_ang, f1_vel = gu.inertial_mul(
                    self.links_state[i_l, i_b].cinr_pos,
                    self.links_state[i_l, i_b].cinr_inertial,
                    self.links_state[i_l, i_b].cinr_mass,
                    self.links_state[i_l, i_b].cdd_vel,
                    self.links_state[i_l, i_b].cdd_ang,
                )
                f2_ang, f2_vel = gu.inertial_mul(
                    self.links_state[i_l, i_b].cinr_pos,
                    self.links_state[i_l, i_b].cinr_inertial,
                    self.links_state[i_l, i_b].cinr_mass,
                    self.links_state[i_l, i_b].cd_vel,
                    self.links_state[i_l, i_b].cd_ang,
                )
                f2_ang, f2_vel = gu.motion_cross_force(
                    self.links_state[i_l, i_b].cd_ang, self.links_state[i_l, i_b].cd_vel, f2_ang, f2_vel
                )

                self.links_state[i_l, i_b].cfrc_vel = f1_vel + f2_vel + self.links_state[i_l, i_b].cfrc_applied_vel
                self.links_state[i_l, i_b].cfrc_ang = f1_ang + f2_ang + self.links_state[i_l, i_b].cfrc_applied_ang

            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_e, i_b in ti.ndrange(self.n_entities, self._B):
                e_info = self.entities_info[i_e]
                for i in range(e_info.n_links):
                    i_l = e_info.link_end - 1 - i
                    I_l = [i_l, i_b] if ti.static(self._options.batch_links_info) else i_l
                    i_p = self.links_info[I_l].parent_idx
                    if i_p != -1:
                        self.links_state[i_p, i_b].cfrc_vel = (
                            self.links_state[i_p, i_b].cfrc_vel + self.links_state[i_l, i_b].cfrc_vel
                        )
                        self.links_state[i_p, i_b].cfrc_ang = (
                            self.links_state[i_p, i_b].cfrc_ang + self.links_state[i_l, i_b].cfrc_ang
                        )

    @ti.func
    def _func_actuation(self):
        if ti.static(self._use_hibernation):
            pass
        else:
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_l, i_b in ti.ndrange(self.n_links, self._B):
                I_l = [i_l, i_b] if ti.static(self._options.batch_links_info) else i_l
                for i_j in range(l_info.joint_start, l_info.joint_end):
                    I_j = [i_j, i_b] if ti.static(self._options.batch_joints_info) else i_j
                    joint_type = self.joints_info[I_j].type
                    q_start = self.joints_info[I_j].q_start

                    if joint_type == gs.JOINT_TYPE.REVOLUTE or joint_type == gs.JOINT_TYPE.PRISMATIC:
                        gear = -1  # TODO
                        i_d = self.links_info[I_l].dof_start
                        self.dofs_state[i_d, i_b].act_length = gear * self.qpos[q_start, i_b]
                        self.dofs_state[i_d, i_b].qf_actuator = self.dofs_state[i_d, i_b].act_length
                    else:
                        for i_d in range(self.links_info[I_l].dof_start, self.links_info[I_l].dof_end):
                            self.dofs_state[i_d, i_b].act_length = 0.0
                            self.dofs_state[i_d, i_b].qf_actuator = self.dofs_state[i_d, i_b].act_length

    @ti.func
    def _func_bias_force(self):
        if ti.static(self._use_hibernation):
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_b in range(self._B):
                for i_l_ in range(self.n_awake_links[i_b]):
                    i_l = self.awake_links[i_l_, i_b]
                    I_l = [i_l, i_b] if ti.static(self._options.batch_links_info) else i_l
                    l_info = self.links_info[I_l]

                    for i_d in range(l_info.dof_start, l_info.dof_end):
                        self.dofs_state[i_d, i_b].qf_bias = self.dofs_state[i_d, i_b].cdof_ang.dot(
                            self.links_state[i_l, i_b].cfrc_ang
                        ) + self.dofs_state[i_d, i_b].cdof_vel.dot(self.links_state[i_l, i_b].cfrc_vel)

                        self.dofs_state[i_d, i_b].force = (
                            self.dofs_state[i_d, i_b].qf_passive
                            - self.dofs_state[i_d, i_b].qf_bias
                            + self.dofs_state[i_d, i_b].qf_applied
                            # + self.dofs_state[i_d, i_b].qf_actuator
                        )

                        self.dofs_state[i_d, i_b].qf_smooth = self.dofs_state[i_d, i_b].force

        else:
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_l, i_b in ti.ndrange(self.n_links, self._B):
                I_l = [i_l, i_b] if ti.static(self._options.batch_links_info) else i_l
                l_info = self.links_info[I_l]

                for i_d in range(l_info.dof_start, l_info.dof_end):
                    self.dofs_state[i_d, i_b].qf_bias = self.dofs_state[i_d, i_b].cdof_ang.dot(
                        self.links_state[i_l, i_b].cfrc_ang
                    ) + self.dofs_state[i_d, i_b].cdof_vel.dot(self.links_state[i_l, i_b].cfrc_vel)

                    self.dofs_state[i_d, i_b].force = (
                        self.dofs_state[i_d, i_b].qf_passive
                        - self.dofs_state[i_d, i_b].qf_bias
                        + self.dofs_state[i_d, i_b].qf_applied
                        # + self.dofs_state[i_d, i_b].qf_actuator
                    )

                    self.dofs_state[i_d, i_b].qf_smooth = self.dofs_state[i_d, i_b].force

    @ti.func
    def _func_compute_qacc(self):
        self._func_solve_mass(self.dofs_state.force, self.dofs_state.acc_smooth)

        if ti.static(self._use_hibernation):
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
            for i_d1_, i_b in ti.ndrange(self.entity_max_dofs, self._B):
                for i_e_ in range(self.n_awake_entities[i_b]):
                    i_e = self.awake_entities[i_e_, i_b]
                    e_info = self.entities_info[i_e]
                    if i_d1_ < e_info.n_dofs:
                        self.dofs_state[i_d1, i_b].acc = self.dofs_state[i_d1, i_b].acc_smooth
        else:
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
            for i_e, i_d1_, i_b in ti.ndrange(self.n_entities, self.entity_max_dofs, self._B):
                e_info = self.entities_info[i_e]
                if i_d1_ < e_info.n_dofs:
                    i_d1 = e_info.dof_start + i_d1_
                    self.dofs_state[i_d1, i_b].acc = self.dofs_state[i_d1, i_b].acc_smooth

    @ti.func
    def _func_integrate(self):
        if ti.static(self._use_hibernation):
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_b in range(self._B):
                for i_d_ in range(self.n_awake_dofs[i_b]):
                    i_d = self.awake_dofs[i_d_, i_b]
                    self.dofs_state[i_d, i_b].vel = (
                        self.dofs_state[i_d, i_b].vel + self.dofs_state[i_d, i_b].acc * self._substep_dt
                    )

            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_b in range(self._B):
                for i_l_ in range(self.n_awake_links[i_b]):
                    i_l = self.awake_links[i_l_, i_b]
                    I_l = [i_l, i_b] if ti.static(self._options.batch_links_info) else i_l

                    for i_j in range(self.links_info[I_l].joint_start, self.links_info[I_l].joint_end):
                        dof_start = self.joints_info[I_j].dof_start
                        q_start = self.joints_info[I_j].q_start
                        q_end = self.joints_info[I_j].q_end

                        I_j = [i_j, i_b] if ti.static(self._options.batch_joints_info) else i_j
                        joint_type = self.joints_info[I_j].joint_type

                        if joint_type == gs.JOINT_TYPE.FREE:
                            rot = ti.Vector(
                                [
                                    self.qpos[q_start + 3, i_b],
                                    self.qpos[q_start + 4, i_b],
                                    self.qpos[q_start + 5, i_b],
                                    self.qpos[q_start + 6, i_b],
                                ]
                            )
                            ang = (
                                ti.Vector(
                                    [
                                        self.dofs_state[dof_start + 3, i_b].vel,
                                        self.dofs_state[dof_start + 4, i_b].vel,
                                        self.dofs_state[dof_start + 5, i_b].vel,
                                    ]
                                )
                                * self._substep_dt
                            )
                            qrot = gu.ti_rotvec_to_quat(ang)
                            rot = gu.ti_transform_quat_by_quat(qrot, rot)
                            pos = ti.Vector(
                                [self.qpos[q_start, i_b], self.qpos[q_start + 1, i_b], self.qpos[q_start + 2, i_b]]
                            )
                            vel = ti.Vector(
                                [
                                    self.dofs_state[dof_start, i_b].vel,
                                    self.dofs_state[dof_start + 1, i_b].vel,
                                    self.dofs_state[dof_start + 2, i_b].vel,
                                ]
                            )
                            pos = pos + vel * self._substep_dt
                            for j in ti.static(range(3)):
                                self.qpos[q_start + j, i_b] = pos[j]
                            for j in ti.static(range(4)):
                                self.qpos[q_start + j + 3, i_b] = rot[j]
                        elif joint_type == gs.JOINT_TYPE.FIXED:
                            pass
                        elif joint_type == gs.JOINT_TYPE.SPHERICAL:
                            rot = ti.Vector(
                                [
                                    self.qpos[q_start + 0, i_b],
                                    self.qpos[q_start + 1, i_b],
                                    self.qpos[q_start + 2, i_b],
                                    self.qpos[q_start + 3, i_b],
                                ]
                            )
                            ang = (
                                ti.Vector(
                                    [
                                        self.dofs_state[dof_start + 3, i_b].vel,
                                        self.dofs_state[dof_start + 4, i_b].vel,
                                        self.dofs_state[dof_start + 5, i_b].vel,
                                    ]
                                )
                                * self._substep_dt
                            )
                            qrot = gu.ti_rotvec_to_quat(ang)
                            rot = gu.ti_transform_quat_by_quat(qrot, rot)
                            for j in ti.static(range(4)):
                                self.qpos[q_start + j, i_b] = rot[j]

                        else:
                            for j in range(q_end - q_start):
                                self.qpos[q_start + j, i_b] = (
                                    self.qpos[q_start + j, i_b]
                                    + self.dofs_state[dof_start + j, i_b].vel * self._substep_dt
                                )

        else:
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_d, i_b in ti.ndrange(self.n_dofs, self._B):
                self.dofs_state[i_d, i_b].vel = (
                    self.dofs_state[i_d, i_b].vel + self.dofs_state[i_d, i_b].acc * self._substep_dt
                )

            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_l, i_b in ti.ndrange(self.n_links, self._B):
                I_l = [i_l, i_b] if ti.static(self._options.batch_links_info) else i_l
                l_info = self.links_info[I_l]
                if l_info.n_dofs == 0:
                    continue

                dof_start = l_info.dof_start
                q_start = l_info.q_start
                q_end = l_info.q_end

                i_j = l_info.joint_start
                I_j = [i_j, i_b] if ti.static(self._options.batch_joints_info) else i_j
                joint_type = self.joints_info[I_j].type

                if joint_type == gs.JOINT_TYPE.FREE:
                    pos = ti.Vector([self.qpos[q_start, i_b], self.qpos[q_start + 1, i_b], self.qpos[q_start + 2, i_b]])
                    vel = ti.Vector(
                        [
                            self.dofs_state[dof_start, i_b].vel,
                            self.dofs_state[dof_start + 1, i_b].vel,
                            self.dofs_state[dof_start + 2, i_b].vel,
                        ]
                    )
                    pos = pos + vel * self._substep_dt
                    for j in ti.static(range(3)):
                        self.qpos[q_start + j, i_b] = pos[j]
                if joint_type == gs.JOINT_TYPE.SPHERICAL or joint_type == gs.JOINT_TYPE.FREE:
                    rot_offset = 3 if joint_type == gs.JOINT_TYPE.FREE else 0
                    rot = ti.Vector(
                        [
                            self.qpos[q_start + rot_offset + 0, i_b],
                            self.qpos[q_start + rot_offset + 1, i_b],
                            self.qpos[q_start + rot_offset + 2, i_b],
                            self.qpos[q_start + rot_offset + 3, i_b],
                        ]
                    )
                    ang = (
                        ti.Vector(
                            [
                                self.dofs_state[dof_start + rot_offset + 0, i_b].vel,
                                self.dofs_state[dof_start + rot_offset + 1, i_b].vel,
                                self.dofs_state[dof_start + rot_offset + 2, i_b].vel,
                            ]
                        )
                        * self._substep_dt
                    )
                    qrot = gu.ti_rotvec_to_quat(ang)
                    rot = gu.ti_transform_quat_by_quat(qrot, rot)
                    for j in ti.static(range(4)):
                        self.qpos[q_start + j + rot_offset, i_b] = rot[j]
                else:
                    for j in range(q_end - q_start):
                        self.qpos[q_start + j, i_b] = (
                            self.qpos[q_start + j, i_b] + self.dofs_state[dof_start + j, i_b].vel * self._substep_dt
                        )

    @ti.func
    def _func_integrate_dq_entity(self, dq, i_e, i_b, respect_joint_limit):
        e_info = self.entities_info[i_e]
        for i_l in range(e_info.link_start, e_info.link_end):
            I_l = [i_l, i_b] if ti.static(self._options.batch_links_info) else i_l
            l_info = self.links_info[I_l]
            if l_info.n_dofs == 0:
                continue

            i_j = l_info.joint_start
            I_j = [i_j, i_b] if ti.static(self._options.batch_joints_info) else i_j
            joint_type = self.joints_info[I_j].type

            q_start = l_info.q_start
            dof_start = l_info.dof_start
            dq_start = l_info.dof_start - e_info.dof_start

            if joint_type == gs.JOINT_TYPE.FREE:
                pos = ti.Vector([self.qpos[q_start, i_b], self.qpos[q_start + 1, i_b], self.qpos[q_start + 2, i_b]])
                dpos = ti.Vector([dq[dq_start, i_b], dq[dq_start + 1, i_b], dq[dq_start + 2, i_b]])
                pos = pos + dpos

                quat = ti.Vector(
                    [
                        self.qpos[q_start + 3, i_b],
                        self.qpos[q_start + 4, i_b],
                        self.qpos[q_start + 5, i_b],
                        self.qpos[q_start + 6, i_b],
                    ]
                )
                dquat = gu.ti_rotvec_to_quat(
                    ti.Vector([dq[dq_start + 3, i_b], dq[dq_start + 4, i_b], dq[dq_start + 5, i_b]])
                )
                quat = gu.ti_transform_quat_by_quat(
                    quat, dquat
                )  # Note that this order is different from integrateing vel. Here dq is w.r.t to world.

                for j in ti.static(range(3)):
                    self.qpos[q_start + j, i_b] = pos[j]

                for j in ti.static(range(4)):
                    self.qpos[q_start + j + 3, i_b] = quat[j]

            elif joint_type == gs.JOINT_TYPE.FIXED:
                pass

            else:
                for i_d_ in range(l_info.n_dofs):
                    self.qpos[q_start + i_d_, i_b] = self.qpos[q_start + i_d_, i_b] + dq[dq_start + i_d_, i_b]

                    if respect_joint_limit:
                        I_d = [dof_start + i_d_, i_b] if ti.static(self._options.batch_dofs_info) else dof_start + i_d_
                        self.qpos[q_start + i_d_, i_b] = ti.math.clamp(
                            self.qpos[q_start + i_d_, i_b],
                            self.dofs_info[I_d].limit[0],
                            self.dofs_info[I_d].limit[1],
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

    @ti.kernel
    def _kernel_update_geoms_render_T(self, geoms_render_T: ti.types.ndarray()):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i_g, i_b in ti.ndrange(self.n_geoms, self._B):
            geom_T = gu.ti_trans_quat_to_T(
                self.geoms_state[i_g, i_b].pos + self.envs_offset[i_b],
                self.geoms_state[i_g, i_b].quat,
            )
            for i, j in ti.static(ti.ndrange(4, 4)):
                geoms_render_T[i_g, i_b, i, j] = ti.cast(geom_T[i, j], ti.float32)

    def update_geoms_render_T(self):
        self._kernel_update_geoms_render_T(self._geoms_render_T)

    @ti.kernel
    def _kernel_update_vgeoms_render_T(self, vgeoms_render_T: ti.types.ndarray()):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i_g, i_b in ti.ndrange(self.n_vgeoms, self._B):
            geom_T = gu.ti_trans_quat_to_T(
                self.vgeoms_state[i_g, i_b].pos + self.envs_offset[i_b],
                self.vgeoms_state[i_g, i_b].quat,
            )
            for i, j in ti.static(ti.ndrange(4, 4)):
                vgeoms_render_T[i_g, i_b, i, j] = ti.cast(geom_T[i, j], ti.float32)

    def update_vgeoms_render_T(self):
        self._kernel_update_vgeoms_render_T(self._vgeoms_render_T)

    def get_state(self, f):
        if self.is_active():
            state = RigidSolverState(self._scene)
            self._kernel_get_state(
                state.qpos,
                state.dofs_vel,
                state.links_pos,
                state.links_quat,
                state.i_pos_shift,
                state.mass_shift,
                state.friction_ratio,
            )
        else:
            state = None
        return state

    @ti.kernel
    def _kernel_get_state(
        self,
        qpos: ti.types.ndarray(),
        vel: ti.types.ndarray(),
        links_pos: ti.types.ndarray(),
        links_quat: ti.types.ndarray(),
        i_pos_shift: ti.types.ndarray(),
        mass_shift: ti.types.ndarray(),
        friction_ratio: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
        for i_q, i_b in ti.ndrange(self.n_qs, self._B):
            qpos[i_b, i_q] = self.qpos[i_q, i_b]

        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
        for i_d, i_b in ti.ndrange(self.n_dofs, self._B):
            vel[i_b, i_d] = self.dofs_state[i_d, i_b].vel

        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
        for i_l, i_b in ti.ndrange(self.n_links, self._B):
            for i in ti.static(range(3)):
                links_pos[i_b, i_l, i] = self.links_state[i_l, i_b].pos[i]
                i_pos_shift[i_b, i_l, i] = self.links_state[i_l, i_b].i_pos_shift[i]
            for i in ti.static(range(4)):
                links_quat[i_b, i_l, i] = self.links_state[i_l, i_b].quat[i]
            mass_shift[i_b, i_l] = self.links_state[i_l, i_b].mass_shift

        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
        for i_l, i_b in ti.ndrange(self.n_geoms, self._B):
            friction_ratio[i_b, i_l] = self.geoms_state[i_l, i_b].friction_ratio

    def set_state(self, f, state, envs_idx=None):
        if self.is_active():
            envs_idx = self._sanitize_envs_idx(envs_idx)
            self._kernel_set_state(
                state.qpos,
                state.dofs_vel,
                state.links_pos,
                state.links_quat,
                state.i_pos_shift,
                state.mass_shift,
                state.friction_ratio,
                envs_idx,
            )
            self._kernel_forward_kinematics_links_geoms(envs_idx)
            self.collider.reset(envs_idx)
            self.collider.clear(envs_idx)
            if self.constraint_solver is not None:
                self.constraint_solver.reset(envs_idx)
                self.constraint_solver.clear(envs_idx)
            self._cur_step = -1

    @ti.kernel
    def _kernel_set_state(
        self,
        qpos: ti.types.ndarray(),
        dofs_vel: ti.types.ndarray(),
        links_pos: ti.types.ndarray(),
        links_quat: ti.types.ndarray(),
        i_pos_shift: ti.types.ndarray(),
        mass_shift: ti.types.ndarray(),
        friction_ratio: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
        for i_q, i_b_ in ti.ndrange(self.n_qs, envs_idx.shape[0]):
            self.qpos[i_q, envs_idx[i_b_]] = qpos[envs_idx[i_b_], i_q]

        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
        for i_d, i_b_ in ti.ndrange(self.n_dofs, envs_idx.shape[0]):
            self.dofs_state[i_d, envs_idx[i_b_]].vel = dofs_vel[envs_idx[i_b_], i_d]

        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
        for i_l, i_b_ in ti.ndrange(self.n_links, envs_idx.shape[0]):
            for i in ti.static(range(3)):
                self.links_state[i_l, envs_idx[i_b_]].pos[i] = links_pos[envs_idx[i_b_], i_l, i]
                self.links_state[i_l, envs_idx[i_b_]].i_pos_shift[i] = i_pos_shift[envs_idx[i_b_], i_l, i]
            for i in ti.static(range(4)):
                self.links_state[i_l, envs_idx[i_b_]].quat[i] = links_quat[envs_idx[i_b_], i_l, i]
            self.links_state[i_l, envs_idx[i_b_]].mass_shift = mass_shift[envs_idx[i_b_], i_l]

        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
        for i_l, i_b_ in ti.ndrange(self.n_geoms, envs_idx.shape[0]):
            self.geoms_state[i_l, envs_idx[i_b_]].friction_ratio = friction_ratio[envs_idx[i_b_], i_l]

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

    def _sanitize_envs_idx(self, envs_idx, *, unsafe=False):
        # Handling default argument and special cases
        if envs_idx is None:
            return self._scene._envs_idx

        if self.n_envs == 0:
            gs.raise_exception("`envs_idx` is not supported for non-parallelized scene.")

        if isinstance(envs_idx, slice):
            return self._scene._envs_idx[envs_idx]
        if isinstance(envs_idx, int):
            return self._scene._envs_idx[[envs_idx]]

        # Early return if unsafe
        if unsafe:
            return envs_idx

        # Perform a bunch of sanity checks
        _envs_idx = torch.atleast_1d(torch.as_tensor(envs_idx, dtype=gs.tc_int, device=gs.device)).contiguous()
        if _envs_idx is not envs_idx:
            gs.logger.debug(ALLOCATE_TENSOR_WARNING)

        if _envs_idx.ndim != 1:
            gs.raise_exception("Expecting a 1D tensor for `envs_idx`.")

        if (_envs_idx < 0).any() or (_envs_idx >= self.n_envs).any():
            gs.raise_exception("`envs_idx` exceeds valid range.")

        return _envs_idx

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
            envs_idx = self._sanitize_envs_idx(envs_idx, unsafe=unsafe)
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
        elif isinstance(inputs_idx, int):
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
                                "length of `envs_idx` (or `scene.n_envs` if `envs_idx` is None)."
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
            envs_idx = self._sanitize_envs_idx(envs_idx, unsafe=unsafe)
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
        elif isinstance(inputs_idx, int):
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
        self._kernel_set_links_pos(relative, pos, links_idx, envs_idx)
        if not skip_forward:
            self._kernel_forward_kinematics_links_geoms(envs_idx)

    @ti.kernel
    def _kernel_set_links_pos(
        self,
        relative: ti.i32,
        pos: ti.types.ndarray(),
        links_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i_l_, i_b_ in ti.ndrange(links_idx.shape[0], envs_idx.shape[0]):
            i_b = envs_idx[i_b_]
            i_l = links_idx[i_l_]
            I_l = [i_l, i_b] if ti.static(self._options.batch_links_info) else i_l

            if self.links_info[I_l].parent_idx == -1 and self.links_info[I_l].is_fixed:
                for i in ti.static(range(3)):
                    self.links_state[i_l, i_b].pos[i] = pos[i_b_, i_l_, i]
                if relative:
                    for i in ti.static(range(3)):
                        self.links_state[i_l, i_b].pos[i] = (
                            self.links_state[i_l, i_b].pos[i] + self.links_info[I_l].pos[i]
                        )
            else:
                q_start = self.links_info[I_l].q_start
                for i in ti.static(range(3)):
                    self.qpos[q_start + i, i_b] = pos[i_b_, i_l_, i]
                if relative:
                    for i in ti.static(range(3)):
                        self.qpos[q_start + i, i_b] = self.qpos[q_start + i, i_b] + self.qpos0[q_start + i, i_b]

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
        self._kernel_set_links_quat(relative, quat, links_idx, envs_idx)
        if not skip_forward:
            self._kernel_forward_kinematics_links_geoms(envs_idx)

    @ti.kernel
    def _kernel_set_links_quat(
        self,
        relative: ti.i32,
        quat: ti.types.ndarray(),
        links_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i_l_, i_b_ in ti.ndrange(links_idx.shape[0], envs_idx.shape[0]):
            i_b = envs_idx[i_b_]
            i_l = links_idx[i_l_]
            I_l = [i_l, i_b] if ti.static(self._options.batch_links_info) else i_l

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
                if self.links_info[I_l].parent_idx == -1 and self.links_info[I_l].is_fixed:
                    self.links_state[i_l, i_b].quat = gu.ti_transform_quat_by_quat(self.links_info[I_l].quat, quat_)
                else:
                    q_start = self.links_info[I_l].q_start
                    quat0 = ti.Vector(
                        [
                            self.qpos0[q_start + 3, i_b],
                            self.qpos0[q_start + 4, i_b],
                            self.qpos0[q_start + 5, i_b],
                            self.qpos0[q_start + 6, i_b],
                        ],
                        dt=gs.ti_float,
                    )
                    quat_ = gu.ti_transform_quat_by_quat(quat0, quat_)
                    for i in ti.static(range(4)):
                        self.qpos[q_start + i + 3, i_b] = quat_[i]
            else:
                if self.links_info[I_l].parent_idx == -1 and self.links_info[I_l].is_fixed:
                    for i in ti.static(range(4)):
                        self.links_state[i_l, i_b].quat[i] = quat[i_b_, i_l_, i]
                else:
                    q_start = self.links_info[I_l].q_start
                    for i in ti.static(range(4)):
                        self.qpos[q_start + i + 3, i_b] = quat[i_b_, i_l_, i]

    def set_links_mass_shift(self, mass, links_idx=None, envs_idx=None, *, unsafe=False):
        mass, links_idx, envs_idx = self._sanitize_1D_io_variables(
            mass, links_idx, self.n_links, envs_idx, idx_name="links_idx", skip_allocation=True, unsafe=unsafe
        )
        if self.n_envs == 0:
            mass = mass.unsqueeze(0)
        self._kernel_set_links_mass_shift(mass, links_idx, envs_idx)

    @ti.kernel
    def _kernel_set_links_mass_shift(
        self,
        mass: ti.types.ndarray(),
        links_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i_l_, i_b_ in ti.ndrange(links_idx.shape[0], envs_idx.shape[0]):
            self.links_state[links_idx[i_l_], envs_idx[i_b_]].mass_shift = mass[i_b_, i_l_]

    def set_links_COM_shift(self, com, links_idx=None, envs_idx=None, *, unsafe=False):
        com, links_idx, envs_idx = self._sanitize_2D_io_variables(
            com, links_idx, self.n_links, 3, envs_idx, idx_name="links_idx", skip_allocation=True, unsafe=unsafe
        )
        if self.n_envs == 0:
            com = com.unsqueeze(0)
        self._kernel_set_links_COM_shift(com, links_idx, envs_idx)

    @ti.kernel
    def _kernel_set_links_COM_shift(
        self,
        com: ti.types.ndarray(),
        links_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i_l_, i_b_ in ti.ndrange(links_idx.shape[0], envs_idx.shape[0]):
            for i in ti.static(range(3)):
                self.links_state[links_idx[i_l_], envs_idx[i_b_]].i_pos_shift[i] = com[i_b_, i_l_, i]

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
        self._kernel_set_links_inertial_mass(mass, links_idx, envs_idx)

    @ti.kernel
    def _kernel_set_links_inertial_mass(
        self,
        inertial_mass: ti.types.ndarray(),
        links_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        if ti.static(self._options.batch_links_info):
            for i_l_, i_b_ in ti.ndrange(links_idx.shape[0], envs_idx.shape[0]):
                self.links_info[links_idx[i_l_], envs_idx[i_b_]].inertial_mass = inertial_mass[i_b_, i_l_]
        else:
            for i_l_ in range(links_idx.shape[0]):
                self.links_info[links_idx[i_l_]].inertial_mass = inertial_mass[i_l_]

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
        self._kernel_set_links_invweight(invweight, links_idx, envs_idx)

    @ti.kernel
    def _kernel_set_links_invweight(
        self,
        invweight: ti.types.ndarray(),
        links_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        if ti.static(self._options.batch_links_info):
            for i_l_, i_b_, j in ti.ndrange(links_idx.shape[0], envs_idx.shape[0], 2):
                self.links_info[links_idx[i_l_], envs_idx[i_b_]].invweight[j] = invweight[i_b_, i_l_, j]
        else:
            for i_l_, j in ti.ndrange(links_idx.shape[0], 2):
                self.links_info[links_idx[i_l_]].invweight[j] = invweight[i_l_, j]

    def set_geoms_friction_ratio(self, friction_ratio, geoms_idx=None, envs_idx=None, *, unsafe=False):
        friction_ratio, geoms_idx, envs_idx = self._sanitize_1D_io_variables(
            friction_ratio, geoms_idx, self.n_geoms, envs_idx, idx_name="geoms_idx", skip_allocation=True, unsafe=unsafe
        )
        self._kernel_set_geoms_friction_ratio(friction_ratio, geoms_idx, envs_idx)

    @ti.kernel
    def _kernel_set_geoms_friction_ratio(
        self,
        friction_ratio: ti.types.ndarray(),
        geoms_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i_g_, i_b_ in ti.ndrange(geoms_idx.shape[0], envs_idx.shape[0]):
            self.geoms_state[geoms_idx[i_g_], envs_idx[i_b_]].friction_ratio = friction_ratio[i_b_, i_g_]

    def set_qpos(self, qpos, qs_idx=None, envs_idx=None, *, skip_forward=False, unsafe=False):
        qpos, qs_idx, envs_idx = self._sanitize_1D_io_variables(
            qpos, qs_idx, self.n_qs, envs_idx, idx_name="qs_idx", skip_allocation=True, unsafe=unsafe
        )
        if self.n_envs == 0:
            qpos = qpos.unsqueeze(0)
        self._kernel_set_qpos(qpos, qs_idx, envs_idx)
        self.collider.reset(envs_idx)
        self.collider.clear(envs_idx)
        if self.constraint_solver is not None:
            self.constraint_solver.reset(envs_idx)
            self.constraint_solver.clear(envs_idx)
        if not skip_forward:
            self._kernel_forward_kinematics_links_geoms(envs_idx)

    @ti.kernel
    def _kernel_set_qpos(
        self,
        qpos: ti.types.ndarray(),
        qs_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i_q_, i_b_ in ti.ndrange(qs_idx.shape[0], envs_idx.shape[0]):
            self.qpos[qs_idx[i_q_], envs_idx[i_b_]] = qpos[i_b_, i_q_]

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

        self._kernel_set_global_sol_params(sol_params)

    @ti.kernel
    def _kernel_set_global_sol_params(self, sol_params: ti.types.ndarray()):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i_g in range(self.n_geoms):
            for i in ti.static(range(7)):
                self.geoms_info[i_g].sol_params[i] = sol_params[i]

        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i_j, i_b in ti.ndrange(self.n_joints, self._B):
            I_j = [i_j, i_b] if ti.static(self._options.batch_joints_info) else i_j
            for i in ti.static(range(7)):
                self.joints_info[I_j].sol_params[i] = sol_params[i]

        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i_eq, i_b in ti.ndrange(self.n_equalities, self._B):
            for i in ti.static(range(7)):
                self.equalities_info[i_eq, i_b].sol_params[i] = sol_params[i]

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
        self._kernel_set_sol_params(constraint_type, sol_params, inputs_idx, envs_idx)

    @ti.kernel
    def _kernel_set_sol_params(
        self,
        constraint_type: ti.template(),
        sol_params: ti.types.ndarray(),
        inputs_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        if ti.static(constraint_type == 0):  # geometries
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
            for i_g_ in range(inputs_idx.shape[0]):
                for i in ti.static(range(7)):
                    self.geoms_info[inputs_idx[i_g_]].sol_params[i] = sol_params[i_g_, i]
        elif ti.static(constraint_type == 1):  # joints
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
            if ti.static(self._options.batch_joints_info):
                for i_j_, i_b_ in ti.ndrange(inputs_idx.shape[0], envs_idx.shape[0]):
                    for i in ti.static(range(7)):
                        self.joints_info[inputs_idx[i_j_], envs_idx[i_b_]].sol_params[i] = sol_params[i_b_, i_j_, i]
            else:
                for i_j_ in range(inputs_idx.shape[0]):
                    for i in ti.static(range(7)):
                        self.joints_info[inputs_idx[i_j_]].sol_params[i] = sol_params[i_j_, i]
        else:  # equalities
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
            for i_eq_, i_b_ in ti.ndrange(inputs_idx.shape[0], envs_idx.shape[0]):
                for i in ti.static(range(7)):
                    self.equalities_info[inputs_idx[i_eq_], envs_idx[i_b_]].sol_params[i] = sol_params[i_b_, i_eq_, i]

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
            self._kernel_set_dofs_kp(tensor_list[0], dofs_idx, envs_idx)
        elif name == "kv":
            self._kernel_set_dofs_kv(tensor_list[0], dofs_idx, envs_idx)
        elif name == "force_range":
            self._kernel_set_dofs_force_range(tensor_list[0], tensor_list[1], dofs_idx, envs_idx)
        elif name == "stiffness":
            self._kernel_set_dofs_stiffness(tensor_list[0], dofs_idx, envs_idx)
        elif name == "invweight":
            self._kernel_set_dofs_invweight(tensor_list[0], dofs_idx, envs_idx)
        elif name == "armature":
            self._kernel_set_dofs_armature(tensor_list[0], dofs_idx, envs_idx)
        elif name == "damping":
            self._kernel_set_dofs_damping(tensor_list[0], dofs_idx, envs_idx)
        elif name == "limit":
            self._kernel_set_dofs_limit(tensor_list[0], tensor_list[1], dofs_idx, envs_idx)
        else:
            gs.raise_exception(f"Invalid `name` {name}.")

    def set_dofs_kp(self, kp, dofs_idx=None, envs_idx=None, *, unsafe=False):
        self._set_dofs_info([kp], dofs_idx, "kp", envs_idx, unsafe=unsafe)

    @ti.kernel
    def _kernel_set_dofs_kp(
        self,
        kp: ti.types.ndarray(),
        dofs_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        if ti.static(self._options.batch_dofs_info):
            for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
                self.dofs_info[dofs_idx[i_d_], envs_idx[i_b_]].kp = kp[i_b_, i_d_]
        else:
            for i_d_ in range(dofs_idx.shape[0]):
                self.dofs_info[dofs_idx[i_d_]].kp = kp[i_d_]

    def set_dofs_kv(self, kv, dofs_idx=None, envs_idx=None, *, unsafe=False):
        self._set_dofs_info([kv], dofs_idx, "kv", envs_idx, unsafe=unsafe)

    @ti.kernel
    def _kernel_set_dofs_kv(
        self,
        kv: ti.types.ndarray(),
        dofs_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        if ti.static(self._options.batch_dofs_info):
            for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
                self.dofs_info[dofs_idx[i_d_], envs_idx[i_b_]].kv = kv[i_b_, i_d_]
        else:
            for i_d_ in range(dofs_idx.shape[0]):
                self.dofs_info[dofs_idx[i_d_]].kv = kv[i_d_]

    def set_dofs_force_range(self, lower, upper, dofs_idx=None, envs_idx=None, *, unsafe=False):
        self._set_dofs_info([lower, upper], dofs_idx, "force_range", envs_idx, unsafe=unsafe)

    @ti.kernel
    def _kernel_set_dofs_force_range(
        self,
        lower: ti.types.ndarray(),
        upper: ti.types.ndarray(),
        dofs_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        if ti.static(self._options.batch_dofs_info):
            for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
                self.dofs_info[dofs_idx[i_d_], envs_idx[i_b_]].force_range[0] = lower[i_b_, i_d_]
                self.dofs_info[dofs_idx[i_d_], envs_idx[i_b_]].force_range[1] = upper[i_b_, i_d_]
        else:
            for i_d_ in range(dofs_idx.shape[0]):
                self.dofs_info[dofs_idx[i_d_]].force_range[0] = lower[i_d_]
                self.dofs_info[dofs_idx[i_d_]].force_range[1] = upper[i_d_]

    def set_dofs_stiffness(self, stiffness, dofs_idx=None, envs_idx=None, *, unsafe=False):
        self._set_dofs_info([stiffness], dofs_idx, "stiffness", envs_idx, unsafe=unsafe)

    @ti.kernel
    def _kernel_set_dofs_stiffness(
        self,
        stiffness: ti.types.ndarray(),
        dofs_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        if ti.static(self._options.batch_dofs_info):
            for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
                self.dofs_info[dofs_idx[i_d_], envs_idx[i_b_]].stiffness = stiffness[i_b_, i_d_]
        else:
            for i_d_ in range(dofs_idx.shape[0]):
                self.dofs_info[dofs_idx[i_d_]].stiffness = stiffness[i_d_]

    def set_dofs_invweight(self, invweight, dofs_idx=None, envs_idx=None, *, unsafe=False):
        self._set_dofs_info([invweight], dofs_idx, "invweight", envs_idx, unsafe=unsafe)

    @ti.kernel
    def _kernel_set_dofs_invweight(
        self,
        invweight: ti.types.ndarray(),
        dofs_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        if ti.static(self._options.batch_dofs_info):
            for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
                self.dofs_info[dofs_idx[i_d_], envs_idx[i_b_]].invweight = invweight[i_b_, i_d_]
        else:
            for i_d_ in range(dofs_idx.shape[0]):
                self.dofs_info[dofs_idx[i_d_]].invweight = invweight[i_d_]

    def set_dofs_armature(self, armature, dofs_idx=None, envs_idx=None, *, unsafe=False):
        self._set_dofs_info([armature], dofs_idx, "armature", envs_idx, unsafe=unsafe)

    @ti.kernel
    def _kernel_set_dofs_armature(
        self,
        armature: ti.types.ndarray(),
        dofs_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        if ti.static(self._options.batch_dofs_info):
            for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
                self.dofs_info[dofs_idx[i_d_], envs_idx[i_b_]].armature = armature[i_b_, i_d_]
        else:
            for i_d_ in range(dofs_idx.shape[0]):
                self.dofs_info[dofs_idx[i_d_]].armature = armature[i_d_]

    def set_dofs_damping(self, damping, dofs_idx=None, envs_idx=None, *, unsafe=False):
        self._set_dofs_info([damping], dofs_idx, "damping", envs_idx)

    @ti.kernel
    def _kernel_set_dofs_damping(
        self,
        damping: ti.types.ndarray(),
        dofs_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        if ti.static(self._options.batch_dofs_info):
            for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
                self.dofs_info[dofs_idx[i_d_], envs_idx[i_b_]].damping = damping[i_b_, i_d_]
        else:
            for i_d_ in range(dofs_idx.shape[0]):
                self.dofs_info[dofs_idx[i_d_]].damping = damping[i_d_]

    def set_dofs_limit(self, lower, upper, dofs_idx=None, envs_idx=None, *, unsafe=False):
        self._set_dofs_info([lower, upper], dofs_idx, "limit", envs_idx, unsafe=unsafe)

    @ti.kernel
    def _kernel_set_dofs_limit(
        self,
        lower: ti.types.ndarray(),
        upper: ti.types.ndarray(),
        dofs_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        if ti.static(self._options.batch_dofs_info):
            for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
                self.dofs_info[dofs_idx[i_d_], envs_idx[i_b_]].limit[0] = lower[i_b_, i_d_]
                self.dofs_info[dofs_idx[i_d_], envs_idx[i_b_]].limit[1] = upper[i_b_, i_d_]
        else:
            for i_d_ in range(dofs_idx.shape[0]):
                self.dofs_info[dofs_idx[i_d_]].limit[0] = lower[i_d_]
                self.dofs_info[dofs_idx[i_d_]].limit[1] = upper[i_d_]

    def set_dofs_velocity(self, velocity, dofs_idx=None, envs_idx=None, *, skip_forward=False, unsafe=False):
        velocity, dofs_idx, envs_idx = self._sanitize_1D_io_variables(
            velocity, dofs_idx, self.n_dofs, envs_idx, skip_allocation=True, unsafe=unsafe
        )

        if velocity is None:
            self._kernel_set_dofs_zero_velocity(dofs_idx, envs_idx)
        else:
            if self.n_envs == 0:
                velocity = velocity.unsqueeze(0)
            self._kernel_set_dofs_velocity(velocity, dofs_idx, envs_idx)

        if not skip_forward:
            self._kernel_forward_kinematics_links_geoms(envs_idx)

    @ti.kernel
    def _kernel_set_dofs_velocity(
        self,
        velocity: ti.types.ndarray(),
        dofs_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
            self.dofs_state[dofs_idx[i_d_], envs_idx[i_b_]].vel = velocity[i_b_, i_d_]

    @ti.kernel
    def _kernel_set_dofs_zero_velocity(
        self,
        dofs_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
            self.dofs_state[dofs_idx[i_d_], envs_idx[i_b_]].vel = 0.0

    def set_dofs_position(self, position, dofs_idx=None, envs_idx=None, *, skip_forward=False, unsafe=False):
        position, dofs_idx, envs_idx = self._sanitize_1D_io_variables(
            position, dofs_idx, self.n_dofs, envs_idx, skip_allocation=True, unsafe=unsafe
        )
        if self.n_envs == 0:
            position = position.unsqueeze(0)
        self._kernel_set_dofs_position(position, dofs_idx, envs_idx)
        self.collider.reset(envs_idx)
        self.collider.clear(envs_idx)
        if self.constraint_solver is not None:
            self.constraint_solver.reset(envs_idx)
            self.constraint_solver.clear(envs_idx)
        if not skip_forward:
            self._kernel_forward_kinematics_links_geoms(envs_idx)

    @ti.kernel
    def _kernel_set_dofs_position(
        self,
        position: ti.types.ndarray(),
        dofs_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
            self.dofs_state[dofs_idx[i_d_], envs_idx[i_b_]].pos = position[i_b_, i_d_]

        # also need to update qpos, as dofs_state.pos is not used for actual IK
        # TODO: make this more efficient by only taking care of releavant qs/dofs
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i_e, i_b_ in ti.ndrange(self.n_entities, envs_idx.shape[0]):
            i_b = envs_idx[i_b_]
            for i_l in range(self.entities_info[i_e].link_start, self.entities_info[i_e].link_end):
                I_l = [i_l, i_b] if ti.static(self._options.batch_links_info) else i_l
                l_info = self.links_info[I_l]
                if l_info.n_dofs == 0:
                    continue

                dof_start = l_info.dof_start
                q_start = l_info.q_start

                i_j = l_info.joint_start
                I_j = [i_j, i_b] if ti.static(self._options.batch_joints_info) else i_j
                joint_type = self.joints_info[I_j].type

                if joint_type == gs.JOINT_TYPE.FREE:
                    xyz = ti.Vector(
                        [
                            self.dofs_state[0 + 3 + dof_start, i_b].pos,
                            self.dofs_state[1 + 3 + dof_start, i_b].pos,
                            self.dofs_state[2 + 3 + dof_start, i_b].pos,
                        ],
                        dt=gs.ti_float,
                    )
                    quat = gu.ti_xyz_to_quat(xyz)

                    for i_q in ti.static(range(3)):
                        self.qpos[i_q + q_start, i_b] = self.dofs_state[i_q + dof_start, i_b].pos

                    for i_q in ti.static(range(4)):
                        self.qpos[i_q + 3 + q_start, i_b] = quat[i_q]
                elif joint_type == gs.JOINT_TYPE.SPHERICAL:
                    xyz = ti.Vector(
                        [
                            self.dofs_state[0 + dof_start, i_b].pos,
                            self.dofs_state[1 + dof_start, i_b].pos,
                            self.dofs_state[2 + dof_start, i_b].pos,
                        ],
                        dt=gs.ti_float,
                    )
                    quat = gu.ti_xyz_to_quat(xyz)
                    for i_q_ in ti.static(range(4)):
                        i_q = q_start + i_q_
                        self.qpos[i_q, i_b] = quat[i_q - q_start]
                else:
                    for i_q in range(q_start, l_info.q_end):
                        self.qpos[i_q, i_b] = self.dofs_state[dof_start + i_q - q_start, i_b].pos

    def control_dofs_force(self, force, dofs_idx=None, envs_idx=None, *, unsafe=False):
        force, dofs_idx, envs_idx = self._sanitize_1D_io_variables(
            force, dofs_idx, self.n_dofs, envs_idx, skip_allocation=True, unsafe=unsafe
        )
        if self.n_envs == 0:
            force = force.unsqueeze(0)
        self._kernel_control_dofs_force(force, dofs_idx, envs_idx)

    @ti.kernel
    def _kernel_control_dofs_force(
        self,
        force: ti.types.ndarray(),
        dofs_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
            self.dofs_state[dofs_idx[i_d_], envs_idx[i_b_]].ctrl_mode = gs.CTRL_MODE.FORCE
            self.dofs_state[dofs_idx[i_d_], envs_idx[i_b_]].ctrl_force = force[i_b_, i_d_]

    def control_dofs_velocity(self, velocity, dofs_idx=None, envs_idx=None, *, unsafe=False):
        velocity, dofs_idx, envs_idx = self._sanitize_1D_io_variables(
            velocity, dofs_idx, self.n_dofs, envs_idx, skip_allocation=True, unsafe=unsafe
        )
        if self.n_envs == 0:
            velocity = velocity.unsqueeze(0)
        self._kernel_control_dofs_velocity(velocity, dofs_idx, envs_idx)

    @ti.kernel
    def _kernel_control_dofs_velocity(
        self,
        velocity: ti.types.ndarray(),
        dofs_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
            self.dofs_state[dofs_idx[i_d_], envs_idx[i_b_]].ctrl_mode = gs.CTRL_MODE.VELOCITY
            self.dofs_state[dofs_idx[i_d_], envs_idx[i_b_]].ctrl_vel = velocity[i_b_, i_d_]

    def control_dofs_position(self, position, dofs_idx=None, envs_idx=None, *, unsafe=False):
        position, dofs_idx, envs_idx = self._sanitize_1D_io_variables(
            position, dofs_idx, self.n_dofs, envs_idx, skip_allocation=True, unsafe=unsafe
        )
        if self.n_envs == 0:
            position = position.unsqueeze(0)
        self._kernel_control_dofs_position(position, dofs_idx, envs_idx)

    @ti.kernel
    def _kernel_control_dofs_position(
        self,
        position: ti.types.ndarray(),
        dofs_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
            self.dofs_state[dofs_idx[i_d_], envs_idx[i_b_]].ctrl_mode = gs.CTRL_MODE.POSITION
            self.dofs_state[dofs_idx[i_d_], envs_idx[i_b_]].ctrl_pos = position[i_b_, i_d_]

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
        self._kernel_get_links_vel(tensor, links_idx, envs_idx, ref)
        return _tensor

    @ti.kernel
    def _kernel_get_links_vel(
        self,
        tensor: ti.types.ndarray(),
        links_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
        ref: ti.template(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i_l_, i_b_ in ti.ndrange(links_idx.shape[0], envs_idx.shape[0]):
            l_state = self.links_state[links_idx[i_l_], envs_idx[i_b_]]

            # This is the velocity in world coordinates expressed at global com-position
            vel = l_state.cd_vel  # entity's CoM

            # Translate to get the velocity expressed at a different position if necessary link-position
            if ti.static(ref == 1):  # link's CoM
                vel = vel + l_state.cd_ang.cross(l_state.i_pos)
            if ti.static(ref == 2):  # link's origin
                vel = vel + l_state.cd_ang.cross(l_state.pos - l_state.COM)

            for i in ti.static(range(3)):
                tensor[i_b_, i_l_, i] = vel[i]

    def get_links_ang(self, links_idx=None, envs_idx=None, *, unsafe=False):
        tensor = ti_field_to_torch(self.links_state.cd_ang, envs_idx, links_idx, transpose=True, unsafe=unsafe)
        return tensor.squeeze(0) if self.n_envs == 0 else tensor

    def get_links_acc(self, links_idx=None, envs_idx=None, *, mimick_imu=False, unsafe=False):
        _tensor, links_idx, envs_idx = self._sanitize_2D_io_variables(
            None, links_idx, self.n_links, 3, envs_idx, idx_name="links_idx", unsafe=unsafe
        )
        tensor = _tensor.unsqueeze(0) if self.n_envs == 0 else _tensor
        self._kernel_get_links_acc(mimick_imu, tensor, links_idx, envs_idx)
        return _tensor

    @ti.kernel
    def _kernel_get_links_acc(
        self,
        mimick_imu: ti.i32,
        tensor: ti.types.ndarray(),
        links_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i_l_, i_b_ in ti.ndrange(links_idx.shape[0], envs_idx.shape[0]):
            i_l = links_idx[i_l_]
            i_b = envs_idx[i_b_]

            # Compute links spatial acceleration expressed at links origin in world coordinates
            cpos = self.links_state[i_l, i_b].pos - self.links_state[i_l, i_b].COM
            acc_ang = self.links_state[i_l, i_b].cacc_ang
            acc_lin = self.links_state[i_l, i_b].cacc_lin + acc_ang.cross(cpos)

            # Compute links classical linear acceleration expressed at links origin in world coordinates
            ang = self.links_state[i_l, i_b].cd_ang
            vel = self.links_state[i_l, i_b].cd_vel + ang.cross(cpos)
            acc_classic_lin = acc_lin + ang.cross(vel)

            # Mimick IMU accelerometer signal if requested
            if mimick_imu:
                # Subtract gravity
                acc_classic_lin -= self._gravity[None]

                # Move the resulting linear acceleration in local links frame
                acc_classic_lin = gu.ti_inv_transform_by_quat(acc_classic_lin, self.links_state[i_l, i_b].quat)

            for i in ti.static(range(3)):
                tensor[i_b_, i_l_, i] = acc_classic_lin[i]

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
        self._kernel_get_dofs_control_force(tensor, dofs_idx, envs_idx)
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

    @ti.kernel
    def _kernel_get_dofs_control_force(
        self,
        tensor: ti.types.ndarray(),
        dofs_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        # we need to compute control force here because this won't be computed until the next actual simulation step
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
            i_d = dofs_idx[i_d_]
            i_b = envs_idx[i_b_]
            I_d = [i_d, i_b] if ti.static(self._options.batch_dofs_info) else i_d
            force = gs.ti_float(0.0)
            if self.dofs_state[i_d, i_b].ctrl_mode == gs.CTRL_MODE.FORCE:
                force = self.dofs_state[i_d, i_b].ctrl_force
            elif self.dofs_state[i_d, i_b].ctrl_mode == gs.CTRL_MODE.VELOCITY:
                force = self.dofs_info[I_d].kv * (self.dofs_state[i_d, i_b].ctrl_vel - self.dofs_state[i_d, i_b].vel)
            elif self.dofs_state[i_d, i_b].ctrl_mode == gs.CTRL_MODE.POSITION:
                force = (
                    self.dofs_info[I_d].kp * (self.dofs_state[i_d, i_b].ctrl_pos - self.dofs_state[i_d, i_b].pos)
                    - self.dofs_info[I_d].kv * self.dofs_state[i_d, i_b].vel
                )
            tensor[i_b_, i_d_] = ti.math.clamp(
                force,
                self.dofs_info[I_d].force_range[0],
                self.dofs_info[I_d].force_range[1],
            )

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
            if isinstance(dofs_idx, (slice, int)) or (dofs_idx.ndim == 0):
                tensor = tensor[:, dofs_idx, dofs_idx]
                if tensor.ndim == 1:
                    tensor = tensor.reshape((-1, 1, 1))
            else:
                tensor = tensor[:, dofs_idx.unsqueeze(1), dofs_idx]
        if self.n_envs == 0:
            tensor = tensor.squeeze(0)

        if decompose:
            mass_mat_D_inv = ti_field_to_torch(self.mass_mat_D_inv, envs_idx, dofs_idx, transpose=True, unsafe=unsafe)
            if self.n_envs == 0:
                mass_mat_D_inv = mass_mat_D_inv.squeeze(0)
            return tensor, mass_mat_D_inv

        return tensor

    @ti.kernel
    def _kernel_set_drone_rpm(
        self,
        n_propellers: ti.i32,
        propellers_link_idxs: ti.types.ndarray(),
        propellers_rpm: ti.types.ndarray(),
        propellers_spin: ti.types.ndarray(),
        KF: ti.float32,
        KM: ti.float32,
        invert: ti.i32,
    ):
        """
        Set the RPM of propellers of a drone entity.

        This method should only be called by drone entities.
        """
        for i_b in range(self._B):
            for i_prop in range(n_propellers):
                i_l = propellers_link_idxs[i_prop]

                force = ti.Vector([0.0, 0.0, propellers_rpm[i_prop, i_b] ** 2 * KF], dt=gs.ti_float)
                torque = ti.Vector(
                    [0.0, 0.0, propellers_rpm[i_prop, i_b] ** 2 * KM * propellers_spin[i_prop]], dt=gs.ti_float
                )
                if invert:
                    torque = -torque

                self._func_apply_link_external_force(force, i_l, i_b, 1, 1)
                self._func_apply_link_external_torque(torque, i_l, i_b, 1, 1)

    @ti.kernel
    def _update_drone_propeller_vgeoms(
        self,
        n_propellers: ti.i32,
        propellers_vgeom_idxs: ti.types.ndarray(),
        propellers_revs: ti.types.ndarray(),
        propellers_spin: ti.types.ndarray(),
    ):
        """
        Update the angle of the vgeom in the propellers of a drone entity.
        """
        for i, b in ti.ndrange(n_propellers, self._B):
            rad = propellers_revs[i, b] * propellers_spin[i] * self._substep_dt * np.pi / 30
            self.vgeoms_state[propellers_vgeom_idxs[i], b].quat = gu.ti_transform_quat_by_quat(
                gu.ti_rotvec_to_quat(ti.Vector([0.0, 0.0, rad], dt=gs.ti_float)),
                self.vgeoms_state[propellers_vgeom_idxs[i], b].quat,
            )

    def get_geoms_friction(self, geoms_idx=None, *, unsafe=False):
        return ti_field_to_torch(self.geoms_info.friction, geoms_idx, None, unsafe=unsafe)

    def set_geom_friction(self, friction, geoms_idx):
        self._kernel_set_geom_friction(geoms_idx, friction)

    @ti.kernel
    def _kernel_set_geom_friction(self, geoms_idx: ti.i32, friction: ti.f32):
        self.geoms_info[geoms_idx].friction = friction

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
        self._kernel_set_geoms_friction(friction, geoms_idx)

    @ti.kernel
    def _kernel_set_geoms_friction(
        self,
        friction: ti.types.ndarray(),
        geoms_idx: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i_g_ in ti.ndrange(geoms_idx.shape[0]):
            self.geoms_info[geoms_idx[i_g_]].friction = friction[i_g_]

    def add_weld_constraint(self, link1_idx, link2_idx, envs_idx=None, *, unsafe=False):
        _, link1_idx, _ = self._sanitize_1D_io_variables(
            None, link1_idx, self.n_links, envs_idx, idx_name="links_idx", skip_allocation=True, unsafe=unsafe
        )
        _, link2_idx, envs_idx = self._sanitize_1D_io_variables(
            None, link2_idx, self.n_links, envs_idx, idx_name="links_idx", skip_allocation=True, unsafe=unsafe
        )
        self._kernel_add_weld_constraint(link1_idx, link2_idx, envs_idx)

    @ti.kernel
    def _kernel_add_weld_constraint(
        self,
        link1_idx: ti.types.ndarray(),
        link2_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i_b_ in ti.ndrange(envs_idx.shape[0]):
            i_b = envs_idx[i_b_]
            if self.constraint_solver.ti_n_equalities[i_b] >= self.n_equalities_candidate:
                self.constraint_solver.ti_n_equalities[i_b] = self.n_equalities_candidate - 1
                print(
                    f"{colors.YELLOW}[Genesis] [00:00:00] [WARNING] Too many constraints, delete the last one."
                    f"{formats.RESET}"
                )
            i_e = self.constraint_solver.ti_n_equalities[i_b]

            l1 = link1_idx[i_b]
            l2 = link2_idx[i_b]

            shared_pos = self.links_state[l1, i_b].pos
            pos1 = gu.ti_inv_transform_by_trans_quat(
                shared_pos, self.links_state[l1, i_b].pos, self.links_state[l1, i_b].quat
            )
            pos2 = gu.ti_inv_transform_by_trans_quat(
                shared_pos, self.links_state[l2, i_b].pos, self.links_state[l2, i_b].quat
            )

            self.equalities_info[i_e, i_b].eq_type = gs.ti_int(gs.EQUALITY_TYPE.WELD)
            self.equalities_info[i_e, i_b].eq_obj1id = l1
            self.equalities_info[i_e, i_b].eq_obj2id = l2

            for i_3 in ti.static(range(3)):
                self.equalities_info[i_e, i_b].eq_data[i_3 + 3] = pos1[i_3]
                self.equalities_info[i_e, i_b].eq_data[i_3] = pos2[i_3]

            relpose = gu.ti_quat_mul(gu.ti_inv_quat(self.links_state[l1, i_b].quat), self.links_state[l2, i_b].quat)

            self.equalities_info[i_e, i_b].eq_data[6] = relpose[0]
            self.equalities_info[i_e, i_b].eq_data[7] = relpose[1]
            self.equalities_info[i_e, i_b].eq_data[8] = relpose[2]
            self.equalities_info[i_e, i_b].eq_data[9] = relpose[3]

            self.equalities_info[i_e, i_b].eq_data[10] = 1.0
            self.equalities_info[i_e, i_b].sol_params = ti.Vector(
                [2 * self._substep_dt, 1.0e00, 9.0e-01, 9.5e-01, 1.0e-03, 5.0e-01, 2.0e00]
            )

            self.constraint_solver.ti_n_equalities[i_b] = self.constraint_solver.ti_n_equalities[i_b] + 1

    def delete_weld_constraint(self, link1_idx, link2_idx, envs_idx=None, *, unsafe=False):
        _, link1_idx, _ = self._sanitize_1D_io_variables(
            None, link1_idx, self.n_links, envs_idx, idx_name="links_idx", skip_allocation=True, unsafe=unsafe
        )
        _, link2_idx, envs_idx = self._sanitize_1D_io_variables(
            None, link2_idx, self.n_links, envs_idx, idx_name="links_idx", skip_allocation=True, unsafe=unsafe
        )
        self._kernel_delete_weld_constraint(link1_idx, link2_idx, envs_idx)

    @ti.kernel
    def _kernel_delete_weld_constraint(
        self,
        link1_idx: ti.types.ndarray(),
        link2_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i_b_ in ti.ndrange(envs_idx.shape[0]):
            i_b = envs_idx[i_b_]
            for i_e in range(self.n_equalities, self.constraint_solver.ti_n_equalities[i_b]):
                if (
                    self.equalities_info[i_e, i_b].eq_type == gs.EQUALITY_TYPE.WELD
                    and self.equalities_info[i_e, i_b].eq_obj1id == link1_idx[i_b]
                    and self.equalities_info[i_e, i_b].eq_obj2id == link2_idx[i_b]
                ):
                    if i_e < self.constraint_solver.ti_n_equalities[i_b] - 1:
                        self.equalities_info[i_e, i_b] = self.equalities_info[
                            self.constraint_solver.ti_n_equalities[i_b] - 1, i_b
                        ]
                    self.constraint_solver.ti_n_equalities[i_b] = self.constraint_solver.ti_n_equalities[i_b] - 1

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
        return np.concatenate([entity.init_qpos for entity in self._entities])

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
