import numpy as np
import taichi as ti
import torch

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.entities import AvatarEntity, DroneEntity, RigidEntity
from genesis.engine.states.solvers import RigidSolverState

from ..base_solver import Solver
from .collider_decomp import Collider
from .constraint_solver_decomp import ConstraintSolver
from .constraint_solver_decomp_island import ConstraintSolverIsland
from .sdf_decomp import SDF


@ti.data_oriented
class RigidSolver(Solver):
    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def __init__(self, scene, sim, options):
        super().__init__(scene, sim, options)

        # options
        self._enable_collision = options.enable_collision
        self._enable_joint_limit = options.enable_joint_limit
        self._enable_self_collision = options.enable_self_collision
        self._max_collision_pairs = options.max_collision_pairs
        self._integrator = options.integrator
        self._box_box_detection = options.box_box_detection

        self._use_contact_island = options.use_contact_island
        self._use_hibernation = options.use_hibernation and options.use_contact_island
        if options.use_hibernation and not options.use_contact_island:
            gs.logger.warning(
                "`use_hibernation` is set to False because `use_contact_island=False`. Please turn on `use_contact_island` if you want to use hibernation"
            )

        self._hibernation_thresh_vel = options.hibernation_thresh_vel
        self._hibernation_thresh_acc = options.hibernation_thresh_acc

        if options.contact_resolve_time is None:
            self._sol_contact_resolve_time = 2 * self._substep_dt
        else:
            self._sol_contact_resolve_time = options.contact_resolve_time

        self._options = options

        self._cur_step = -1

    def add_entity(self, idx, material, morph, surface, visualize_contact):
        if isinstance(material, gs.materials.Avatar):
            entity_class = AvatarEntity
            if visualize_contact:
                gs.raise_exception("AvatarEntity does not support visualize_contact")
        else:
            if isinstance(morph, gs.morphs.Drone):
                entity_class = DroneEntity
            else:
                entity_class = RigidEntity

        if morph.is_free:
            verts_state_start = self.n_free_verts
        else:
            verts_state_start = self.n_fixed_verts

        entity = entity_class(
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
        self.n_equalities_ = max(1, self.n_equalities)

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

            # run complete FK once to update geoms state and mass matrix
            self._kernel_forward_kinematics_links_geoms()

            self._init_invweight()

    def _init_invweight(self):
        self._kernel_forward_dynamics()

        cdof_ang = self.dofs_state.cdof_ang.to_numpy()[:, 0, :]
        cdof_vel = self.dofs_state.cdof_vel.to_numpy()[:, 0, :]

        mass_mat_inv = self.mass_mat_inv.to_numpy()[:, :, 0]
        dof_start = self.links_info.dof_start.to_numpy()
        dof_end = self.links_info.dof_end.to_numpy()
        n_dofs = self.links_info.n_dofs.to_numpy()
        parent_idx = self.links_info.parent_idx.to_numpy()
        if self._options.batch_links_info:
            dof_start = dof_start[:, 0]
            dof_end = dof_end[:, 0]
            n_dofs = n_dofs[:, 0]
            parent_idx = parent_idx[:, 0]

        offsets = self.links_state.i_pos.to_numpy()[:, 0, :]

        invweight = np.zeros([self._n_links], dtype=gs.np_float)

        for i_link in range(self._n_links):
            if n_dofs[i_link] > 0:
                jacp = np.zeros([self._n_dofs, 3])
                jacr = np.zeros([self._n_dofs, 3])

                offset = offsets[i_link]

                this_link = i_link
                while this_link >= 0:
                    for i_d_ in range(dof_end[this_link] - dof_start[this_link]):
                        i_d = dof_end[this_link] - i_d_ - 1
                        jacr[i_d] = cdof_ang[i_d]

                        tmp = np.cross(cdof_ang[i_d], offset)
                        jacp[i_d] = cdof_vel[i_d] + tmp

                    this_link = parent_idx[this_link]

                jac = np.concatenate([jacp, jacr], 1)

                A = jac.T @ mass_mat_inv @ jac

                invweight[i_link] = (A[0, 0] + A[1, 1] + A[2, 2]) / 3

        for i_link in range(self._n_links):
            if n_dofs[i_link] == 0:
                next_link = parent_idx[i_link]
                while n_dofs[next_link] == 0 and next_link >= 0:
                    next_link = parent_idx[next_link]
                if next_link >= 0:
                    invweight[i_link] = invweight[next_link]
                else:
                    for i in range(self._n_links):
                        if n_dofs[i] > 0:
                            invweight[i_link] = invweight[i]
                            break

        self._kernel_init_invweight(invweight)

    @ti.kernel
    def _kernel_init_invweight(
        self,
        invweight: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for I in ti.grouped(self.links_info):
            if self.links_info[I].invweight < 0:
                self.links_info[I].invweight = invweight[I[0]]

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
        self.mass_mat_U = ti.field(dtype=gs.ti_float, shape=self._batch_shape((self.n_dofs_, self.n_dofs_)))
        self.mass_mat_y = ti.field(dtype=gs.ti_float, shape=self._batch_shape((self.n_dofs_, self.n_dofs_)))
        self.mass_mat_inv = ti.field(dtype=gs.ti_float, shape=self._batch_shape((self.n_dofs_, self.n_dofs_)))

        # tree structure information
        mass_parent_mask = np.zeros((self.n_dofs_, self.n_dofs_), dtype=gs.np_float)

        for i in range(self.n_links):
            j = i
            while j > -1:
                for i_d in range(self.joints[i].dof_start, self.joints[i].dof_end):
                    for j_d in range(self.joints[j].dof_start, self.joints[j].dof_end):
                        mass_parent_mask[i_d, j_d] = 1.0
                j = self.links[j].parent_idx

        self.mass_parent_mask = ti.field(dtype=gs.ti_float, shape=(self.n_dofs_, self.n_dofs_))
        self.mass_parent_mask.from_numpy(mass_parent_mask)

        # just in case
        self.mass_mat_L.fill(0)
        self.mass_mat_U.fill(0)
        self.mass_mat_y.fill(0)
        self.mass_mat_inv.fill(0)

    def _init_dof_fields(self):
        if self._use_hibernation:
            self.n_awake_dofs = ti.field(dtype=gs.ti_int, shape=self._B)
            self.awake_dofs = ti.field(dtype=gs.ti_int, shape=self._batch_shape(self.n_dofs_))

        struct_dof_info = ti.types.struct(
            stiffness=gs.ti_float,
            sol_params=gs.ti_vec7,
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
        is_nonempty = np.concatenate([joint.dofs_motion_ang for joint in joints], dtype=gs.np_float).shape[0] > 0
        if is_nonempty:  # handle the case where there is a link with no dofs -- otherwise may cause invalid memory
            self._kernel_init_dof_fields(
                dofs_motion_ang=np.concatenate([joint.dofs_motion_ang for joint in joints], dtype=gs.np_float),
                dofs_motion_vel=np.concatenate([joint.dofs_motion_vel for joint in joints], dtype=gs.np_float),
                dofs_limit=np.concatenate([joint.dofs_limit for joint in joints], dtype=gs.np_float),
                dofs_invweight=np.concatenate([joint.dofs_invweight for joint in joints], dtype=gs.np_float),
                dofs_stiffness=np.concatenate([joint.dofs_stiffness for joint in joints], dtype=gs.np_float),
                dofs_sol_params=np.concatenate([joint.dofs_sol_params for joint in joints], dtype=gs.np_float),
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
        dofs_sol_params: ti.types.ndarray(),
        dofs_damping: ti.types.ndarray(),
        dofs_armature: ti.types.ndarray(),
        dofs_kp: ti.types.ndarray(),
        dofs_kv: ti.types.ndarray(),
        dofs_force_range: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for I in ti.grouped(self.dofs_info):
            i = I[0]  # batching (if any) will be the second dim

            for j in ti.static(range(3)):
                self.dofs_info[I].motion_ang[j] = dofs_motion_ang[i, j]
                self.dofs_info[I].motion_vel[j] = dofs_motion_vel[i, j]

            for j in ti.static(range(2)):
                self.dofs_info[I].limit[j] = dofs_limit[i, j]
                self.dofs_info[I].force_range[j] = dofs_force_range[i, j]

            for j in ti.static(range(7)):
                self.dofs_info[I].sol_params[j] = dofs_sol_params[i, j]

            self.dofs_info[I].sol_params[0] = self._sol_contact_resolve_time

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
            q_end=gs.ti_int,
            dof_end=gs.ti_int,
            n_dofs=gs.ti_int,
            pos=gs.ti_vec3,
            quat=gs.ti_vec4,
            joint_type=gs.ti_int,
            invweight=gs.ti_float,
            is_fixed=gs.ti_int,
            joint_pos=gs.ti_vec3,
            joint_quat=gs.ti_vec4,
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
            ang=gs.ti_vec3,
            vel=gs.ti_vec3,
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
            # cfrc_flat
            cfrc_flat_ang=gs.ti_vec3,
            cfrc_flat_vel=gs.ti_vec3,
            # COM-based external force
            cfrc_ext_ang=gs.ti_vec3,
            cfrc_ext_vel=gs.ti_vec3,
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
        joints = self.joints
        self._kernel_init_link_fields(
            links_parent_idx=np.array([link.parent_idx for link in links], dtype=gs.np_int),
            links_root_idx=np.array([link.root_idx for link in links], dtype=gs.np_int),
            links_q_start=np.array([joint.q_start for joint in joints], dtype=gs.np_int),
            links_dof_start=np.array([joint.dof_start for joint in joints], dtype=gs.np_int),
            links_q_end=np.array([joint.q_end for joint in joints], dtype=gs.np_int),
            links_dof_end=np.array([joint.dof_end for joint in joints], dtype=gs.np_int),
            links_joint_type=np.array([joint.type for joint in joints], dtype=gs.np_int),
            links_invweight=np.array([link.invweight for link in links], dtype=gs.np_float),
            links_is_fixed=np.array([link.is_fixed for link in links], dtype=gs.np_int),
            links_pos=np.array([link.pos for link in links], dtype=gs.np_float),
            links_quat=np.array([link.quat for link in links], dtype=gs.np_float),
            links_inertial_pos=np.array([link.inertial_pos for link in links], dtype=gs.np_float),
            links_inertial_quat=np.array([link.inertial_quat for link in links], dtype=gs.np_float),
            links_inertial_i=np.array([link.inertial_i for link in links], dtype=gs.np_float),
            links_inertial_mass=np.array([link.inertial_mass for link in links], dtype=gs.np_float),
            links_joint_pos=np.array([joint.pos for joint in joints], dtype=gs.np_float),
            links_joint_quat=np.array([joint.quat for joint in joints], dtype=gs.np_float),
            links_entity_idx=np.array([link._entity_idx_in_solver for link in links], dtype=gs.np_int),
        )

        self.qpos = ti.field(dtype=gs.ti_float, shape=self._batch_shape(self.n_qs_))
        self.qpos0 = ti.field(dtype=gs.ti_float, shape=self._batch_shape(self.n_qs_))
        if self.n_qs > 0:
            init_qpos = self._batch_array(self.init_qpos.astype(gs.np_float))
            self.qpos.from_numpy(init_qpos)
            self.qpos0.from_numpy(init_qpos)

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
        links_q_end: ti.types.ndarray(),
        links_dof_end: ti.types.ndarray(),
        links_joint_type: ti.types.ndarray(),
        links_invweight: ti.types.ndarray(),
        links_is_fixed: ti.types.ndarray(),
        links_pos: ti.types.ndarray(),
        links_quat: ti.types.ndarray(),
        links_inertial_pos: ti.types.ndarray(),
        links_inertial_quat: ti.types.ndarray(),
        links_inertial_i: ti.types.ndarray(),
        links_inertial_mass: ti.types.ndarray(),
        links_joint_pos: ti.types.ndarray(),
        links_joint_quat: ti.types.ndarray(),
        links_entity_idx: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for I in ti.grouped(self.links_info):
            i = I[0]

            self.links_info[I].parent_idx = links_parent_idx[i]
            self.links_info[I].root_idx = links_root_idx[i]
            self.links_info[I].q_start = links_q_start[i]
            self.links_info[I].dof_start = links_dof_start[i]
            self.links_info[I].q_end = links_q_end[i]
            self.links_info[I].dof_end = links_dof_end[i]
            self.links_info[I].n_dofs = links_dof_end[i] - links_dof_start[i]
            self.links_info[I].joint_type = links_joint_type[i]
            self.links_info[I].invweight = links_invweight[i]
            self.links_info[I].is_fixed = links_is_fixed[i]
            self.links_info[I].entity_idx = links_entity_idx[i]

            for j in ti.static(range(4)):
                self.links_info[I].quat[j] = links_quat[i, j]
                self.links_info[I].joint_quat[j] = links_joint_quat[i, j]
                self.links_info[I].inertial_quat[j] = links_inertial_quat[i, j]

            for j in ti.static(range(3)):
                self.links_info[I].pos[j] = links_pos[i, j]
                self.links_info[I].joint_pos[j] = links_joint_pos[i, j]
                self.links_info[I].inertial_pos[j] = links_inertial_pos[i, j]

            self.links_info[I].inertial_mass = links_inertial_mass[i]
            for j1 in ti.static(range(3)):
                for j2 in ti.static(range(3)):
                    self.links_info[I].inertial_i[j1, j2] = links_inertial_i[i, j1, j2]

        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
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
            needs_coup=gs.ti_int,
            coup_friction=gs.ti_float,
            coup_softness=gs.ti_float,
            coup_restitution=gs.ti_float,
            is_free=gs.ti_int,
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
        self._geoms_render_T = np.empty((self.n_geoms_, self._B, 4, 4), dtype=gs.np_float)

        if self.n_geoms > 0:
            geoms = self.geoms
            self._kernel_init_geom_fields(
                geoms_pos=np.array([geom.init_pos for geom in geoms], dtype=gs.np_float),
                geoms_quat=np.array([geom.init_quat for geom in geoms], dtype=gs.np_float),
                geoms_link_idx=np.array([geom.link.idx for geom in geoms], dtype=gs.np_int),
                geoms_type=np.array([geom.type for geom in geoms], dtype=gs.np_int),
                geoms_friction=np.array([geom.friction for geom in geoms], dtype=gs.np_float),
                geoms_sol_params=np.array([geom.sol_params for geom in geoms], dtype=gs.np_float),
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
                geoms_coup_softness=np.array([geom.coup_softness for geom in geoms], dtype=gs.np_float),
                geoms_coup_friction=np.array([geom.coup_friction for geom in geoms], dtype=gs.np_float),
                geoms_coup_restitution=np.array([geom.coup_restitution for geom in geoms], dtype=gs.np_float),
                geoms_is_free=np.array([geom.is_free for geom in geoms], dtype=gs.np_int),
            )

    @ti.kernel
    def _kernel_init_geom_fields(
        self,
        geoms_pos: ti.types.ndarray(),
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
        geoms_coup_softness: ti.types.ndarray(),
        geoms_coup_friction: ti.types.ndarray(),
        geoms_coup_restitution: ti.types.ndarray(),
        geoms_is_free: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i in range(self.n_geoms):
            for j in ti.static(range(3)):
                self.geoms_info[i].pos[j] = geoms_pos[i, j]

            for j in ti.static(range(4)):
                self.geoms_info[i].quat[j] = geoms_quat[i, j]

            for j in ti.static(range(7)):
                self.geoms_info[i].data[j] = geoms_data[i, j]
                self.geoms_info[i].sol_params[j] = geoms_sol_params[i, j]
            self.geoms_info[i].sol_params[0] = self._sol_contact_resolve_time

            self.geoms_info[i].sol_params[0] = ti.max(self.geoms_info[i].sol_params[0], self._substep_dt * 2)

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

            self.geoms_info[i].coup_softness = geoms_coup_softness[i]
            self.geoms_info[i].coup_friction = geoms_coup_friction[i]
            self.geoms_info[i].coup_restitution = geoms_coup_restitution[i]

            self.geoms_info[i].is_free = geoms_is_free[i]

            # compute init AABB
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

            # compute geom center
            self.geoms_info[i].center = ti.Vector.zero(gs.ti_float, 3)
            for i_v in range(self.geoms_info[i].vert_start, self.geoms_info[i].vert_end):
                pos = self.verts_info[i_v].init_pos
                self.geoms_info[i].center += pos

            self.geoms_info[i].center /= self.geoms_info[i].vert_end - self.geoms_info[i].vert_start

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
                self.links_info[link_idx, i_b].invweight /= ratio
                self.links_info[link_idx, i_b].inertial_mass *= ratio
                for j1, j2 in ti.ndrange(3, 3):
                    self.links_info[link_idx, i_b].inertial_i[j1, j2] *= ratio
        else:
            for i_b in range(self._B):
                self.links_info[link_idx].invweight /= ratio
                self.links_info[link_idx].inertial_mass *= ratio
                for j1, j2 in ti.ndrange(3, 3):
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
        self._vgeoms_render_T = np.empty((self.n_vgeoms_, self._B, 4, 4), dtype=gs.np_float)

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
                for i_b in range(self._B):
                    for i_d in range(entities_dof_start[i], entities_dof_end[i]):
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
            equality_type=gs.ti_int,
            link1_idx=gs.ti_int,
            link2_idx=gs.ti_int,
            anchor1_pos=gs.ti_vec3,
            anchor2_pos=gs.ti_vec3,
            rel_pose=gs.ti_vec4,
            torque_scale=gs.ti_float,
            entity_idx=gs.ti_int,
            sol_params=gs.ti_vec7,
        )
        self.equality_info = struct_equality_info.field(
            shape=self.n_equalities_, needs_grad=False, layout=ti.Layout.SOA
        )
        if self.n_equalities > 0:
            equalities = self.equalities
            self._kernel_init_equality_fields(
                equalities_type=np.array([equality.type for equality in equalities], dtype=gs.np_int),
                equalities_link1_idx=np.array([equality.link1_idx for equality in equalities], dtype=gs.np_int),
                equalities_link2_idx=np.array([equality.link2_idx for equality in equalities], dtype=gs.np_int),
                equalities_anchor1_pos=np.array([equality.anchor1_pos for equality in equalities], dtype=gs.np_float),
                equalities_anchor2_pos=np.array([equality.anchor2_pos for equality in equalities], dtype=gs.np_float),
                equalities_rel_pose=np.array([equality.rel_pose for equality in equalities], dtype=gs.np_float),
                equalities_torque_scale=np.array([equality.torque_scale for equality in equalities], dtype=gs.np_float),
                equalities_sol_params=np.array([equality.sol_params for equality in equalities], dtype=gs.np_float),
            )
            if self._use_contact_island:
                gs.logger.warn("contact island is not supported for equality constraints yet")

    @ti.kernel
    def _kernel_init_equality_fields(
        self,
        equalities_type: ti.types.ndarray(),
        equalities_link1_idx: ti.types.ndarray(),
        equalities_link2_idx: ti.types.ndarray(),
        equalities_anchor1_pos: ti.types.ndarray(),
        equalities_anchor2_pos: ti.types.ndarray(),
        equalities_rel_pose: ti.types.ndarray(),
        equalities_torque_scale: ti.types.ndarray(),
        equalities_sol_params: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i in range(self.n_equalities):
            self.equality_info[i].equality_type = equalities_type[i]
            self.equality_info[i].link1_idx = equalities_link1_idx[i]
            self.equality_info[i].link2_idx = equalities_link2_idx[i]
            for j in ti.static(range(3)):
                self.equality_info[i].anchor1_pos[j] = equalities_anchor1_pos[i, j]
                self.equality_info[i].anchor2_pos[j] = equalities_anchor2_pos[i, j]
            for j in ti.static(range(4)):
                self.equality_info[i].rel_pose[j] = equalities_rel_pose[i, j]
            self.equality_info[i].torque_scale = equalities_torque_scale[i]
            for j in ti.static(range(7)):
                self.equality_info[i].sol_params[j] = equalities_sol_params[i, j]

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
        vel_rot = link_state.ang.cross(pos_world - link_state.pos)
        vel_lin = link_state.vel
        return vel_rot + vel_lin

    @ti.func
    def _func_compute_mass_matrix(self):
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

                    for i_d in range(e_info.dof_start, e_info.dof_end):
                        I_d = [i_d, i_b] if ti.static(self._options.batch_dofs_info) else i_d
                        self.mass_mat[i_d, i_d, i_b] = (
                            self.mass_mat[i_d, i_d, i_b]
                            + self.dofs_info[I_d].armature
                            + self.dofs_info[I_d].damping * self._substep_dt
                        )
                        for j_d in range(i_d + 1, e_info.dof_end):
                            self.mass_mat[i_d, j_d, i_b] = self.mass_mat[j_d, i_d, i_b]

            if ti.static(self._integrator == gs.integrator.approximate_implicitfast):
                # Compute implicit derivative for control force
                # qDeriv += d qfrc_actuator / d qvel
                ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
                for i_d, i_b in ti.ndrange(self.n_dofs, self._B):
                    I_d = [i_d, i_b] if ti.static(self._options.batch_dofs_info) else i_d

                    if self.dofs_state[i_d, i_b].ctrl_mode == gs.CTRL_MODE.FORCE:
                        pass

                    elif self.dofs_state[i_d, i_b].ctrl_mode == gs.CTRL_MODE.VELOCITY:
                        self.mass_mat[i_d, i_d, i_b] = (
                            self.mass_mat[i_d, i_d, i_b] + self.dofs_info[I_d].kv * self._substep_dt
                        )

                    elif self.dofs_state[i_d, i_b].ctrl_mode == gs.CTRL_MODE.POSITION:
                        self.mass_mat[i_d, i_d, i_b] = (
                            self.mass_mat[i_d, i_d, i_b] + self.dofs_info[I_d].kv * self._substep_dt
                        )

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
            for i_b in range(self._B):
                for i_l in range(self.n_links):
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
                for i_d in range(e_info.dof_start, e_info.dof_end):
                    for j_d in range(e_info.dof_start, e_info.dof_end):
                        self.mass_mat[i_d, j_d, i_b] = (
                            self.dofs_state[i_d, i_b].f_ang.dot(self.dofs_state[j_d, i_b].cdof_ang)
                            + self.dofs_state[i_d, i_b].f_vel.dot(self.dofs_state[j_d, i_b].cdof_vel)
                        ) * self.mass_parent_mask[i_d, j_d]

                for i_d in range(e_info.dof_start, e_info.dof_end):
                    I_d = [i_d, i_b] if ti.static(self._options.batch_dofs_info) else i_d
                    self.mass_mat[i_d, i_d, i_b] = (
                        self.mass_mat[i_d, i_d, i_b]
                        + self.dofs_info[I_d].armature
                        + self.dofs_info[I_d].damping * self._substep_dt
                    )
                    for j_d in range(i_d + 1, e_info.dof_end):
                        self.mass_mat[i_d, j_d, i_b] = self.mass_mat[j_d, i_d, i_b]

            if ti.static(self._integrator == gs.integrator.approximate_implicitfast):
                # Compute implicit derivative for control force
                # qDeriv += d qfrc_actuator / d qvel
                ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
                for i_d, i_b in ti.ndrange(self.n_dofs, self._B):
                    I_d = [i_d, i_b] if ti.static(self._options.batch_dofs_info) else i_d

                    if self.dofs_state[i_d, i_b].ctrl_mode == gs.CTRL_MODE.FORCE:
                        pass

                    elif self.dofs_state[i_d, i_b].ctrl_mode == gs.CTRL_MODE.VELOCITY:
                        self.mass_mat[i_d, i_d, i_b] = (
                            self.mass_mat[i_d, i_d, i_b] + self.dofs_info[I_d].kv * self._substep_dt
                        )

                    elif self.dofs_state[i_d, i_b].ctrl_mode == gs.CTRL_MODE.POSITION:
                        self.mass_mat[i_d, i_d, i_b] = (
                            self.mass_mat[i_d, i_d, i_b] + self.dofs_info[I_d].kv * self._substep_dt
                        )

    @ti.func
    def _func_inv_mass(self):
        """
        Inverse via LU decomposition
        """
        if ti.static(self._use_hibernation):
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
            for i_b in range(self._B):
                for i_e_ in range(self.n_awake_entities[i_b]):
                    i_e = self.awake_entities[i_e_, i_b]

                    entity_dof_start = self.entities_info[i_e].dof_start
                    entity_dof_end = self.entities_info[i_e].dof_end
                    for i in range(entity_dof_start, entity_dof_end):
                        self.mass_mat_L[i, i, i_b] = 1
                        for j in range(i, entity_dof_end):
                            tmp = self.mass_mat[i, j, i_b]
                            for k in range(entity_dof_start, i):
                                tmp -= self.mass_mat_L[i, k, i_b] * self.mass_mat_U[k, j, i_b]
                            self.mass_mat_U[i, j, i_b] = tmp

                        for j in range(i + 1, entity_dof_end):
                            tmp = self.mass_mat[j, i, i_b]
                            for k in range(entity_dof_start, i):
                                tmp -= self.mass_mat_L[j, k, i_b] * self.mass_mat_U[k, i, i_b]
                            self.mass_mat_L[j, i, i_b] = tmp / self.mass_mat_U[i, i, i_b]

            # k can also be parallel
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
            for k_, i_b in ti.ndrange(self.entity_max_dofs, self._B):
                for i_e_ in range(self.n_awake_entities[i_b]):
                    i_e = self.awake_entities[i_e_, i_b]

                    if k_ < self.entities_info[i_e].n_dofs:
                        entity_dof_start = self.entities_info[i_e].dof_start
                        entity_dof_end = self.entities_info[i_e].dof_end
                        k = entity_dof_start + k_

                        # forward substitution
                        for i in range(entity_dof_start, entity_dof_end):
                            tmp = gs.ti_float(0.0)
                            for j in range(entity_dof_start, i):
                                tmp += self.mass_mat_L[i, j, i_b] * self.mass_mat_y[k, j, i_b]
                            if i == k:
                                self.mass_mat_y[k, i, i_b] = 1 - tmp
                            else:
                                self.mass_mat_y[k, i, i_b] = -tmp

                        # backward substitution
                        for i_ in range(self.entities_info[i_e].n_dofs):
                            i = entity_dof_end - 1 - i_
                            tmp = gs.ti_float(0.0)
                            for j in range(i + 1, entity_dof_end):
                                tmp += self.mass_mat_U[i, j, i_b] * self.mass_mat_inv[j, k, i_b]
                            self.mass_mat_inv[i, k, i_b] = (self.mass_mat_y[k, i, i_b] - tmp) / self.mass_mat_U[
                                i, i, i_b
                            ]
        else:
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
            for i_b, i_e in ti.ndrange(self._B, self.n_entities):
                entity_dof_start = self.entities_info[i_e].dof_start
                entity_dof_end = self.entities_info[i_e].dof_end

                for i in range(entity_dof_start, entity_dof_end):
                    self.mass_mat_L[i, i, i_b] = 1
                    for j in range(i, entity_dof_end):
                        tmp = self.mass_mat[i, j, i_b]
                        for k in range(entity_dof_start, i):
                            tmp -= self.mass_mat_L[i, k, i_b] * self.mass_mat_U[k, j, i_b]
                        self.mass_mat_U[i, j, i_b] = tmp

                    for j in range(i + 1, entity_dof_end):
                        tmp = self.mass_mat[j, i, i_b]
                        for k in range(entity_dof_start, i):
                            tmp -= self.mass_mat_L[j, k, i_b] * self.mass_mat_U[k, i, i_b]
                        self.mass_mat_L[j, i, i_b] = tmp / self.mass_mat_U[i, i, i_b]

            # k can also be parallel
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
            for i_e, k_, i_b in ti.ndrange(self.n_entities, self.entity_max_dofs, self._B):
                if k_ < self.entities_info[i_e].n_dofs:
                    entity_dof_start = self.entities_info[i_e].dof_start
                    entity_dof_end = self.entities_info[i_e].dof_end
                    k = entity_dof_start + k_

                    # forward substitution
                    for i in range(entity_dof_start, entity_dof_end):
                        tmp = gs.ti_float(0.0)
                        for j in range(entity_dof_start, i):
                            tmp += self.mass_mat_L[i, j, i_b] * self.mass_mat_y[k, j, i_b]
                        if i == k:
                            self.mass_mat_y[k, i, i_b] = 1 - tmp
                        else:
                            self.mass_mat_y[k, i, i_b] = -tmp

                    # backward substitution
                    for i_ in range(self.entities_info[i_e].n_dofs):
                        i = entity_dof_end - 1 - i_
                        tmp = gs.ti_float(0.0)
                        for j in range(i + 1, entity_dof_end):
                            tmp += self.mass_mat_U[i, j, i_b] * self.mass_mat_inv[j, k, i_b]
                        self.mass_mat_inv[i, k, i_b] = (self.mass_mat_y[k, i, i_b] - tmp) / self.mass_mat_U[i, i, i_b]

    @ti.kernel
    def _kernel_forward_dynamics(self):
        self._func_forward_dynamics()

    # @@@@@@@@@ Composer starts here
    # decomposed kernels should happen in the block below. This block will be handled by composer and composed into a single kernel
    @ti.func
    def _func_forward_dynamics(self):
        self._func_compute_mass_matrix()
        self._func_inv_mass()
        self._func_torque_and_passive_force()
        self._func_system_update_acc(False)
        self._func_system_update_force()
        self._func_inverse_link_force()
        # self._func_actuation()
        self._func_bias_force()
        self._func_compute_qacc()
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
        self._func_forward_dynamics()

    @ti.func
    def _func_implicit_damping(self):
        # TODO: hibernate
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
        for i_d, i_b in ti.ndrange(self.n_dofs, self._B):
            I_d = [i_d, i_b] if ti.static(self._options.batch_dofs_info) else i_d

            if self.dofs_state[i_d, i_b].ctrl_mode == gs.CTRL_MODE.FORCE:
                pass

            elif self.dofs_state[i_d, i_b].ctrl_mode == gs.CTRL_MODE.VELOCITY:
                self.mass_mat[i_d, i_d, i_b] = self.mass_mat[i_d, i_d, i_b] + self.dofs_info[I_d].kv * self._substep_dt

            elif self.dofs_state[i_d, i_b].ctrl_mode == gs.CTRL_MODE.POSITION:
                self.mass_mat[i_d, i_d, i_b] = self.mass_mat[i_d, i_d, i_b] + self.dofs_info[I_d].kv * self._substep_dt

            self.dofs_state[i_d, i_b].force += self.dofs_state[i_d, i_b].qf_constraint

        self._func_inv_mass()

        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i_e, i_d1_, i_b in ti.ndrange(self.n_entities, self.entity_max_dofs, self._B):
            e_info = self.entities_info[i_e]
            if i_d1_ < e_info.n_dofs:
                i_d1 = e_info.dof_start + i_d1_
                acc = gs.ti_float(0.0)

                for i_d2 in range(e_info.dof_start, e_info.dof_end):
                    acc += self.mass_mat_inv[i_d1, i_d2, i_b] * self.dofs_state[i_d2, i_b].force
                self.dofs_state[i_d1, i_b].acc = acc

    @ti.kernel
    def _kernel_step_2(self):
        if ti.static(self._integrator == gs.integrator.implicitfast):
            self._func_implicit_damping()
        self._func_integrate()

        self._func_forward_kinematics()
        self._func_transform_COM()
        self._func_update_geoms()

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
    def _kernel_forward_kinematics_links_geoms(self):
        self._func_forward_kinematics()
        self._func_transform_COM()
        self._func_update_geoms()

    def _func_constraint_force(self):
        from genesis.utils.tools import create_timer

        timer = create_timer(name="constraint_force", level=2, ti_sync=True, skip_first_call=True)
        if self._enable_collision or self._enable_joint_limit:
            self.constraint_solver.clear()
        timer.stamp("constraint_solver.clear")

        if self._enable_collision:
            self.collider.clear()
            timer.stamp("collider.clear")
            self.collider.detection()
            timer.stamp("detection")

        self.constraint_solver.handle_constraints()
        timer.stamp("constraint_solver.handle_constraints")

        # if self._enable_collision:
        #     self.constraint_solver.contact_island.construct()

        #     self.constraint_solver.add_collision_constraints()
        #     timer.stamp('collision_constraints')

        # if self._enable_joint_limit:
        #     self.constraint_solver.add_joint_limit_constraints()
        #     timer.stamp('joint_limit_constraints')

        # if self._enable_collision or self._enable_joint_limit:
        #     self.constraint_solver.resolve()
        # timer.stamp('solve')

    def _batch_array(self, arr, first_dim=False):
        if first_dim:
            return np.tile(np.expand_dims(arr, 0), self._batch_shape(arr.ndim * (1,), True))
        else:
            return np.tile(np.expand_dims(arr, -1), self._batch_shape(arr.ndim * (1,)))

    def _process_dim(self, tensor, envs_idx=None):
        if self.n_envs == 0:
            if tensor.ndim == 1:
                tensor = tensor[None, :]
            else:
                gs.raise_exception(
                    f"Invalid input shape: {tensor.shape}. Expecting a 1D tensor for non-parallelized scene."
                )
        else:
            if tensor.ndim == 2:
                if envs_idx is not None:
                    if tensor.shape[0] != len(envs_idx):
                        gs.raise_exception(
                            f"Invalid input shape: {tensor.shape}. 1st dimension of input does not match `envs_idx`."
                        )
                else:
                    if tensor.shape[0] != self.n_envs:
                        gs.raise_exception(
                            f"Invalid input shape: {tensor.shape}. 1st dimension of input does not match `scene.n_envs`."
                        )
            else:
                gs.raise_exception(
                    f"Invalid input shape: {tensor.shape}. Expecting a 2D tensor for scene with parallelized envs."
                )
        return tensor

    @ti.func
    def _func_COM_links(self):
        if ti.static(self._use_hibernation):
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_b in range(self._B):
                for i_l_ in range(self.n_awake_links[i_b]):
                    i_l = self.awake_links[i_l_, i_b]

                    self.links_state[i_l, i_b].root_COM = ti.Vector.zero(gs.ti_float, 3)
                    self.links_state[i_l, i_b].mass_sum = 0.0

            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_b in range(self._B):
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
            for i_b in range(self._B):
                for i_l_ in range(self.n_awake_links[i_b]):
                    i_l = self.awake_links[i_l_, i_b]
                    I_l = [i_l, i_b] if ti.static(self._options.batch_links_info) else i_l

                    i_r = self.links_info[I_l].root_idx
                    if i_l == i_r:
                        self.links_state[i_l, i_b].root_COM = (
                            self.links_state[i_l, i_b].root_COM / self.links_state[i_l, i_b].mass_sum
                        )

            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_b in range(self._B):
                for i_l_ in range(self.n_awake_links[i_b]):
                    i_l = self.awake_links[i_l_, i_b]
                    I_l = [i_l, i_b] if ti.static(self._options.batch_links_info) else i_l

                    i_r = self.links_info[I_l].root_idx
                    self.links_state[i_l, i_b].root_COM = self.links_state[i_r, i_b].root_COM

            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_b in range(self._B):
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
            for i_b in range(self._B):
                for i_l_ in range(self.n_awake_links[i_b]):
                    i_l = self.awake_links[i_l_, i_b]
                    I_l = [i_l, i_b] if ti.static(self._options.batch_links_info) else i_l

                    l_info = self.links_info[I_l]
                    i_p = l_info.parent_idx

                    p_pos = ti.Vector.zero(gs.ti_float, 3)
                    p_quat = gu.ti_identity_quat()

                    if i_p != -1:
                        p_pos = self.links_state[i_p, i_b].pos
                        p_quat = self.links_state[i_p, i_b].quat

                    if l_info.joint_type == gs.JOINT_TYPE.FREE or (l_info.is_fixed and i_p == -1):
                        self.links_state[i_l, i_b].j_pos = self.links_state[i_l, i_b].pos
                        self.links_state[i_l, i_b].j_quat = self.links_state[i_l, i_b].quat
                    else:
                        (
                            self.links_state[i_l, i_b].j_pos,
                            self.links_state[i_l, i_b].j_quat,
                        ) = gu.ti_transform_pos_quat_by_trans_quat(l_info.pos, l_info.quat, p_pos, p_quat)

                        (
                            self.links_state[i_l, i_b].j_pos,
                            self.links_state[i_l, i_b].j_quat,
                        ) = gu.ti_transform_pos_quat_by_trans_quat(
                            l_info.joint_pos,
                            l_info.joint_quat,
                            self.links_state[i_l, i_b].j_pos,
                            self.links_state[i_l, i_b].j_quat,
                        )

            # cdof_fn
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_b in range(self._B):
                for i_l_ in range(self.n_awake_links[i_b]):
                    i_l = self.awake_links[i_l_, i_b]
                    I_l = [i_l, i_b] if ti.static(self._options.batch_links_info) else i_l

                    l_info = self.links_info[I_l]

                    if l_info.joint_type == gs.JOINT_TYPE.FREE:
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

                    elif l_info.joint_type == gs.JOINT_TYPE.FIXED:
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
            for i_b in range(self._B):
                for i_l in range(self.n_links):
                    self.links_state[i_l, i_b].root_COM = ti.Vector.zero(gs.ti_float, 3)
                    self.links_state[i_l, i_b].mass_sum = 0.0

            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_b in range(self._B):
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
            for i_b in range(self._B):
                for i_l in range(self.n_links):
                    I_l = [i_l, i_b] if ti.static(self._options.batch_links_info) else i_l

                    i_r = self.links_info[I_l].root_idx
                    if i_l == i_r:
                        self.links_state[i_l, i_b].root_COM = (
                            self.links_state[i_l, i_b].root_COM / self.links_state[i_l, i_b].mass_sum
                        )

            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_b in range(self._B):
                for i_l in range(self.n_links):
                    I_l = [i_l, i_b] if ti.static(self._options.batch_links_info) else i_l

                    i_r = self.links_info[I_l].root_idx
                    self.links_state[i_l, i_b].root_COM = self.links_state[i_r, i_b].root_COM

            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_b in range(self._B):
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
            for i_b in range(self._B):
                for i_l in range(self.n_links):
                    I_l = [i_l, i_b] if ti.static(self._options.batch_links_info) else i_l

                    l_info = self.links_info[I_l]
                    i_p = l_info.parent_idx

                    p_pos = ti.Vector.zero(gs.ti_float, 3)
                    p_quat = gu.ti_identity_quat()

                    if i_p != -1:
                        p_pos = self.links_state[i_p, i_b].pos
                        p_quat = self.links_state[i_p, i_b].quat

                    if l_info.joint_type == gs.JOINT_TYPE.FREE or (l_info.is_fixed and i_p == -1):
                        self.links_state[i_l, i_b].j_pos = self.links_state[i_l, i_b].pos
                        self.links_state[i_l, i_b].j_quat = self.links_state[i_l, i_b].quat
                    else:
                        (
                            self.links_state[i_l, i_b].j_pos,
                            self.links_state[i_l, i_b].j_quat,
                        ) = gu.ti_transform_pos_quat_by_trans_quat(l_info.pos, l_info.quat, p_pos, p_quat)

                        (
                            self.links_state[i_l, i_b].j_pos,
                            self.links_state[i_l, i_b].j_quat,
                        ) = gu.ti_transform_pos_quat_by_trans_quat(
                            l_info.joint_pos,
                            l_info.joint_quat,
                            self.links_state[i_l, i_b].j_pos,
                            self.links_state[i_l, i_b].j_quat,
                        )

            # cdof_fn
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_b in range(self._B):
                for i_l in range(self.n_links):
                    I_l = [i_l, i_b] if ti.static(self._options.batch_links_info) else i_l

                    l_info = self.links_info[I_l]

                    if l_info.joint_type == gs.JOINT_TYPE.FREE:
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

                    elif l_info.joint_type == gs.JOINT_TYPE.FIXED:
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

    @ti.func
    def _func_COM_cd(self):
        if ti.static(self._use_hibernation):
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
            for i_b in range(self._B):
                for i_e_ in range(self.n_awake_entities[i_b]):
                    i_e = self.awake_entities[i_e_, i_b]

                    e_info = self.entities_info[i_e]
                    for i_l in range(e_info.link_start, e_info.link_end):
                        l_info = self.links_info[i_l]
                        i_p = l_info.parent_idx

                        cd_vel = ti.Vector.zero(gs.ti_float, 3)
                        cd_ang = ti.Vector.zero(gs.ti_float, 3)
                        if i_p == -1:
                            for i_d in range(l_info.dof_start, l_info.dof_end):
                                cd_vel += self.dofs_state[i_d, i_b].cdofvel_vel
                                cd_ang += self.dofs_state[i_d, i_b].cdofvel_ang
                        else:
                            cd_vel = self.links_state[i_p, i_b].cd_vel
                            cd_ang = self.links_state[i_p, i_b].cd_ang
                            for i_d in range(l_info.dof_start, l_info.dof_end):
                                cd_vel += self.dofs_state[i_d, i_b].cdofvel_vel
                                cd_ang += self.dofs_state[i_d, i_b].cdofvel_ang
                        self.links_state[i_l, i_b].cd_vel = cd_vel
                        self.links_state[i_l, i_b].cd_ang = cd_ang
        else:
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
            for i_e, i_b in ti.ndrange(self.n_entities, self._B):
                e_info = self.entities_info[i_e]
                for i_l in range(e_info.link_start, e_info.link_end):
                    I_l = [i_l, i_b] if ti.static(self._options.batch_links_info) else i_l
                    l_info = self.links_info[I_l]
                    i_p = l_info.parent_idx

                    cd_vel = ti.Vector.zero(gs.ti_float, 3)
                    cd_ang = ti.Vector.zero(gs.ti_float, 3)
                    if i_p == -1:
                        for i_d in range(l_info.dof_start, l_info.dof_end):
                            cd_vel += self.dofs_state[i_d, i_b].cdofvel_vel
                            cd_ang += self.dofs_state[i_d, i_b].cdofvel_ang
                    else:
                        cd_vel = self.links_state[i_p, i_b].cd_vel
                        cd_ang = self.links_state[i_p, i_b].cd_ang
                        for i_d in range(l_info.dof_start, l_info.dof_end):
                            cd_vel += self.dofs_state[i_d, i_b].cdofvel_vel
                            cd_ang += self.dofs_state[i_d, i_b].cdofvel_ang
                    self.links_state[i_l, i_b].cd_vel = cd_vel
                    self.links_state[i_l, i_b].cd_ang = cd_ang

    @ti.func
    def _func_COM_cdofd(self):
        if ti.static(self._use_hibernation):
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_b in range(self._B):
                for i_l_ in range(self.n_awake_links[i_b]):
                    i_l = self.awake_links[i_l_, i_b]
                    I_l = [i_l, i_b] if ti.static(self._options.batch_links_info) else i_l

                    l_info = self.links_info[I_l]

                    if l_info.joint_type == gs.JOINT_TYPE.FREE:
                        cd_ang = ti.Vector.zero(gs.ti_float, 3)
                        cd_vel = ti.Vector.zero(gs.ti_float, 3)

                        for i_d in range(l_info.dof_start, l_info.dof_start + 3):
                            cd_ang = cd_ang + self.dofs_state[i_d, i_b].cdofvel_ang
                            cd_vel = cd_vel + self.dofs_state[i_d, i_b].cdofvel_vel

                        for i_d in range(l_info.dof_start, l_info.dof_start + 3):
                            self.dofs_state[i_d, i_b].cdofd_ang = ti.Vector.zero(gs.ti_float, 3)
                            self.dofs_state[i_d, i_b].cdofd_vel = ti.Vector.zero(gs.ti_float, 3)

                        for i_d in range(l_info.dof_start + 3, l_info.dof_start + 6):
                            (
                                self.dofs_state[i_d, i_b].cdofd_ang,
                                self.dofs_state[i_d, i_b].cdofd_vel,
                            ) = gu.motion_cross_motion(
                                cd_ang,
                                cd_vel,
                                self.dofs_state[i_d, i_b].cdof_ang,
                                self.dofs_state[i_d, i_b].cdof_vel,
                            )

                    elif l_info.joint_type == gs.JOINT_TYPE.FIXED:
                        pass

                    else:
                        for i_d in range(l_info.dof_start, l_info.dof_end):
                            (
                                self.dofs_state[i_d, i_b].cdofd_ang,
                                self.dofs_state[i_d, i_b].cdofd_vel,
                            ) = gu.motion_cross_motion(
                                self.links_state[i_l, i_b].cd_ang,
                                self.links_state[i_l, i_b].cd_vel,
                                self.dofs_state[i_d, i_b].cdof_ang,
                                self.dofs_state[i_d, i_b].cdof_vel,
                            )

        else:
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_b in range(self._B):
                for i_l in range(self.n_links):
                    I_l = [i_l, i_b] if ti.static(self._options.batch_links_info) else i_l

                    l_info = self.links_info[I_l]

                    if l_info.joint_type == gs.JOINT_TYPE.FREE:
                        cd_ang = ti.Vector.zero(gs.ti_float, 3)
                        cd_vel = ti.Vector.zero(gs.ti_float, 3)

                        for i_d in range(l_info.dof_start, l_info.dof_start + 3):
                            cd_ang = cd_ang + self.dofs_state[i_d, i_b].cdofvel_ang
                            cd_vel = cd_vel + self.dofs_state[i_d, i_b].cdofvel_vel

                        for i_d in range(l_info.dof_start, l_info.dof_start + 3):
                            self.dofs_state[i_d, i_b].cdofd_ang = ti.Vector.zero(gs.ti_float, 3)
                            self.dofs_state[i_d, i_b].cdofd_vel = ti.Vector.zero(gs.ti_float, 3)

                        for i_d in range(l_info.dof_start + 3, l_info.dof_start + 6):
                            (
                                self.dofs_state[i_d, i_b].cdofd_ang,
                                self.dofs_state[i_d, i_b].cdofd_vel,
                            ) = gu.motion_cross_motion(
                                cd_ang,
                                cd_vel,
                                self.dofs_state[i_d, i_b].cdof_ang,
                                self.dofs_state[i_d, i_b].cdof_vel,
                            )

                    elif l_info.joint_type == gs.JOINT_TYPE.FIXED:
                        pass

                    else:
                        for i_d in range(l_info.dof_start, l_info.dof_end):
                            (
                                self.dofs_state[i_d, i_b].cdofd_ang,
                                self.dofs_state[i_d, i_b].cdofd_vel,
                            ) = gu.motion_cross_motion(
                                self.links_state[i_l, i_b].cd_ang,
                                self.links_state[i_l, i_b].cd_vel,
                                self.dofs_state[i_d, i_b].cdof_ang,
                                self.dofs_state[i_d, i_b].cdof_vel,
                            )

    @ti.func
    def _func_transform_COM(self):
        self._func_COM_links()
        self._func_COM_cd()
        self._func_COM_cdofd()

    @ti.func
    def _func_forward_kinematics(self):
        if ti.static(self._use_hibernation):
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_b in range(self._B):
                for i_e_ in range(self.n_awake_entities[i_b]):
                    i_e = self.awake_entities[i_e_, i_b]
                    self._func_forward_kinematics_entity(i_e, i_b)
        else:
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_b in range(self._B):
                for i_e in range(self.n_entities):
                    self._func_forward_kinematics_entity(i_e, i_b)

    @ti.func
    def _func_forward_kinematics_entity(self, i_e, i_b):
        # calculate_j
        for i_l in range(self.entities_info[i_e].link_start, self.entities_info[i_e].link_end):
            I_l = [i_l, i_b] if ti.static(self._options.batch_links_info) else i_l
            l_info = self.links_info[I_l]

            if l_info.joint_type == gs.JOINT_TYPE.FREE:
                for i_q in ti.static(range(3)):
                    self.links_state[i_l, i_b].j_pos[i_q] = self.qpos[i_q + l_info.q_start, i_b]
                    self.links_state[i_l, i_b].j_ang[i_q] = self.dofs_state[i_q + 3 + l_info.dof_start, i_b].vel
                    self.links_state[i_l, i_b].j_vel[i_q] = self.dofs_state[i_q + l_info.dof_start, i_b].vel
                for i_q in ti.static(range(4)):
                    self.links_state[i_l, i_b].j_quat[i_q] = self.qpos[i_q + 3 + l_info.q_start, i_b]

                xyz = gu.ti_quat_to_xyz(self.links_state[i_l, i_b].j_quat)
                for i_q in ti.static(range(3)):
                    self.dofs_state[i_q + l_info.dof_start, i_b].pos = self.qpos[i_q + l_info.q_start, i_b]
                    self.dofs_state[i_q + 3 + l_info.dof_start, i_b].pos = xyz[i_q]

            elif l_info.joint_type == gs.JOINT_TYPE.FIXED:
                self.links_state[i_l, i_b].j_pos = ti.Vector.zero(gs.ti_float, 3)
                self.links_state[i_l, i_b].j_quat = gu.ti_identity_quat()
                self.links_state[i_l, i_b].j_ang = ti.Vector.zero(gs.ti_float, 3)
                self.links_state[i_l, i_b].j_vel = ti.Vector.zero(gs.ti_float, 3)

            else:
                self.dofs_state[l_info.dof_start, i_b].pos = self.qpos[l_info.q_start, i_b]
                I_dof_start = [l_info.dof_start, i_b] if ti.static(self._options.batch_dofs_info) else l_info.dof_start
                dof_info = self.dofs_info[I_dof_start]
                self.links_state[i_l, i_b].j_pos = dof_info.motion_vel * (
                    self.qpos[l_info.q_start, i_b] - self.qpos0[l_info.q_start, i_b]
                )
                self.links_state[i_l, i_b].j_quat = gu.ti_rotvec_to_quat(
                    dof_info.motion_ang * (self.qpos[l_info.q_start, i_b] - self.qpos0[l_info.q_start, i_b])
                )
                self.links_state[i_l, i_b].j_ang = dof_info.motion_ang * self.dofs_state[l_info.dof_start, i_b].vel
                self.links_state[i_l, i_b].j_vel = dof_info.motion_vel * self.dofs_state[l_info.dof_start, i_b].vel

                for i_d in range(l_info.dof_start + 1, l_info.dof_end):
                    I_d = [i_d, i_b] if ti.static(self._options.batch_dofs_info) else i_d
                    dof_info = self.dofs_info[I_d]
                    qi = l_info.q_start + i_d - l_info.dof_start
                    self.dofs_state[i_d, i_b].pos = self.qpos[qi, i_b]
                    ji_pos = dof_info.motion_vel * self.qpos[qi, i_b]
                    ji_quat = gu.ti_rotvec_to_quat(dof_info.motion_ang * self.qpos[qi, i_b])

                    (
                        self.links_state[i_l, i_b].j_pos,
                        self.links_state[i_l, i_b].j_quat,
                    ) = gu.ti_transform_pos_quat_by_trans_quat(
                        ji_pos, ji_quat, self.links_state[i_l, i_b].j_pos, self.links_state[i_l, i_b].j_quat
                    )

                    ji_ang = dof_info.motion_ang * self.dofs_state[i_d, i_b].vel
                    ji_vel = dof_info.motion_vel * self.dofs_state[i_d, i_b].vel

                    self.links_state[i_l, i_b].j_ang = self.links_state[i_l, i_b].j_ang + gu.ti_transform_by_quat(
                        ji_ang, ji_quat
                    )

                    self.links_state[i_l, i_b].j_vel = self.links_state[i_l, i_b].j_vel + gu.ti_transform_by_quat(
                        ji_vel + ji_pos.cross(ji_ang), ji_quat
                    )

            if l_info.joint_type != gs.JOINT_TYPE.FREE:
                anchor_pos, anchor_quat = gu.ti_transform_pos_quat_by_trans_quat(
                    l_info.joint_pos,
                    l_info.joint_quat,
                    ti.Vector.zero(gs.ti_float, 3),
                    self.links_state[i_l, i_b].j_quat,
                )

                self.links_state[i_l, i_b].j_pos = self.links_state[i_l, i_b].j_pos + l_info.joint_pos - anchor_pos
                (
                    self.links_state[i_l, i_b].j_pos,
                    self.links_state[i_l, i_b].j_quat,
                ) = gu.ti_transform_pos_quat_by_trans_quat(
                    self.links_state[i_l, i_b].j_pos, self.links_state[i_l, i_b].j_quat, l_info.pos, l_info.quat
                )

        # joint_to_world
        for i_l in range(self.entities_info[i_e].link_start, self.entities_info[i_e].link_end):
            l_state = self.links_state[i_l, i_b]
            I_l = [i_l, i_b] if ti.static(self._options.batch_links_info) else i_l
            l_info = self.links_info[I_l]
            i_p = l_info.parent_idx

            if i_p == -1:  # root link
                if not l_info.is_fixed:
                    self.links_state[i_l, i_b].pos = l_state.j_pos
                    self.links_state[i_l, i_b].quat = gu.ti_normalize(l_state.j_quat)

                self.links_state[i_l, i_b].ang = gu.ti_transform_by_quat(l_state.j_ang, l_state.quat)
                self.links_state[i_l, i_b].vel = l_state.j_vel

            else:
                p = self.links_state[i_p, i_b]
                self.links_state[i_l, i_b].pos, quat = gu.ti_transform_pos_quat_by_trans_quat(
                    self.links_state[i_l, i_b].j_pos, self.links_state[i_l, i_b].j_quat, p.pos, p.quat
                )
                self.links_state[i_l, i_b].quat = gu.ti_normalize(quat)

                self.links_state[i_l, i_b].ang = p.ang + gu.ti_transform_by_quat(
                    self.links_state[i_l, i_b].j_ang, self.links_state[i_l, i_b].quat
                )
                self.links_state[i_l, i_b].vel = (
                    p.vel
                    + p.ang.cross(self.links_state[i_l, i_b].pos - p.pos)
                    + gu.ti_transform_by_quat(self.links_state[i_l, i_b].j_vel, self.links_state[i_l, i_b].quat)
                )

    @ti.func
    def _func_update_geoms(self):
        """
        NOTE: this only update geom pose, not its verts and else.
        """
        if ti.static(self._use_hibernation):
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
            for i_b in range(self._B):
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
            for i_g, i_b in ti.ndrange(self.n_geoms, self._B):
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
            for i_corner in range(8):
                corner_pos = gu.ti_transform_by_trans_quat(
                    self.geoms_init_AABB[i_g, i_corner], g_state.pos, g_state.quat
                )
                lower = ti.min(lower, corner_pos)
                upper = ti.max(upper, corner_pos)

            self.geoms_state[i_g, i_b].aabb_min = lower
            self.geoms_state[i_g, i_b].aabb_max = upper

    @ti.func
    def _func_update_geom_aabbs_tight(self):
        """
        Tight AABB recomputed using all vertices, but much slower
        """
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i_g, i_b in ti.ndrange(self.n_geoms, self._B):
            self._func_update_verts_for_geom(i_g, i_b)

            g_info = self.geoms_info[i_g]
            if g_info.is_free:
                lower = self.free_verts_state[g_info.verts_state_start, i_b].pos
                upper = self.free_verts_state[g_info.verts_state_start, i_b].pos

                for i_v in range(g_info.verts_state_start, g_info.verts_state_end):
                    lower = ti.min(lower, self.free_verts_state[i_v, i_b].pos)
                    upper = ti.max(upper, self.free_verts_state[i_v, i_b].pos)

                self.geoms_state[i_g, i_b].aabb_min = lower
                self.geoms_state[i_g, i_b].aabb_max = upper
            else:
                lower = self.fixed_verts_state[g_info.verts_state_start].pos
                upper = self.fixed_verts_state[g_info.verts_state_start].pos

                for i_v in range(g_info.verts_state_start, g_info.verts_state_end):
                    lower = ti.min(lower, self.fixed_verts_state[i_v].pos)
                    upper = ti.max(upper, self.fixed_verts_state[i_v].pos)

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

    def apply_links_external_force(self, force, links_idx, envs_idx=None):
        force, links_idx, envs_idx = self._validate_2D_io_variables(force, links_idx, 3, envs_idx, idx_name="links_idx")

        self._kernel_apply_links_external_force(force, links_idx, envs_idx)

    @ti.kernel
    def _kernel_apply_links_external_force(
        self,
        force: ti.types.ndarray(),
        links_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i_l_, i_b_ in ti.ndrange(links_idx.shape[0], envs_idx.shape[0]):
            for i in ti.static(range(3)):
                self.links_state[links_idx[i_l_], envs_idx[i_b_]].cfrc_ext_vel[i] -= force[i_b_, i_l_, i]

    def apply_links_external_torque(self, torque, links_idx, envs_idx=None):
        torque, links_idx, envs_idx = self._validate_2D_io_variables(
            torque, links_idx, 3, envs_idx, idx_name="links_idx"
        )

        self._kernel_apply_links_external_torque(torque, links_idx, envs_idx)

    @ti.kernel
    def _kernel_apply_links_external_torque(
        self,
        torque: ti.types.ndarray(),
        links_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i_l_, i_b_ in ti.ndrange(links_idx.shape[0], envs_idx.shape[0]):
            for i in ti.static(range(3)):
                self.links_state[links_idx[i_l_], envs_idx[i_b_]].cfrc_ext_ang[i] -= torque[i_b_, i_l_, i]

    @ti.func
    def _func_apply_external_force(self, pos, force, link_idx, batch_idx):
        torque = (pos - self.links_state[link_idx, batch_idx].COM).cross(force)
        self.links_state[link_idx, batch_idx].cfrc_ext_ang -= torque
        self.links_state[link_idx, batch_idx].cfrc_ext_vel -= force

    @ti.func
    def _func_apply_external_torque(self, torque, link_idx, batch_idx):
        self.links_state[link_idx, batch_idx].cfrc_ext_ang -= torque

    @ti.func
    def _func_apply_external_force_link_frame(self, pos, force, link_idx, batch_idx):
        pos = gu.ti_transform_by_trans_quat(
            pos, self.links_state[link_idx, batch_idx].pos, self.links_state[link_idx, batch_idx].quat
        )
        force = gu.ti_transform_by_quat(force, self.links_state[link_idx, batch_idx].quat)
        self._func_apply_external_force(pos, force, link_idx, batch_idx)

    @ti.func
    def _func_apply_external_torque_link_frame(self, torque, link_idx, batch_idx):
        torque = gu.ti_transform_by_quat(torque, self.links_state[link_idx, batch_idx].quat)
        self._func_apply_external_torque(torque, link_idx, batch_idx)

    @ti.func
    def _func_apply_external_force_link_inertial_frame(self, pos, force, link_idx, batch_idx):
        link_I = [link_idx, batch_idx] if ti.static(self._options.batch_links_info) else link_idx
        pos = gu.ti_transform_by_trans_quat(
            pos, self.links_info[link_I].inertial_pos, self.links_info[link_I].inertial_quat
        )
        force = gu.ti_transform_by_quat(force, self.links_info[link_I].inertial_quat)
        self._func_apply_external_force_link_frame(pos, force, link_idx, batch_idx)

    @ti.func
    def _func_apply_external_torque_link_inertial_frame(self, torque, link_idx, batch_idx):
        link_I = [link_idx, batch_idx] if ti.static(self._options.batch_links_info) else link_idx
        torque = gu.ti_transform_by_quat(torque, self.links_info[link_I].inertial_quat)
        self._func_apply_external_torque_link_frame(torque, link_idx, batch_idx)

    @ti.func
    def _func_clear_external_force(self):
        if ti.static(self._use_hibernation):
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
            for i_b in range(self._B):
                for i_l_ in range(self.n_awake_links[i_b]):
                    i_l = self.awake_links[i_l_, i_b]
                    self.links_state[i_l, i_b].cfrc_ext_ang = ti.Vector.zero(gs.ti_float, 3)
                    self.links_state[i_l, i_b].cfrc_ext_vel = ti.Vector.zero(gs.ti_float, 3)
        else:
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
            for i_l, i_b in ti.ndrange(self.n_links, self._B):
                self.links_state[i_l, i_b].cfrc_ext_ang = ti.Vector.zero(gs.ti_float, 3)
                self.links_state[i_l, i_b].cfrc_ext_vel = ti.Vector.zero(gs.ti_float, 3)

    @ti.func
    def _func_torque_and_passive_force(self):
        # compute force based on each dof's ctrl mode
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
        for i_e, i_b in ti.ndrange(self.n_entities, self._B):
            wakeup = False
            for i_l in range(self.entities_info[i_e].link_start, self.entities_info[i_e].link_end):
                force = gs.ti_float(0.0)
                I_l = [i_l, i_b] if ti.static(self._options.batch_links_info) else i_l
                l_info = self.links_info[I_l]
                for i_d in range(l_info.dof_start, l_info.dof_end):
                    I_d = [i_d, i_b] if ti.static(self._options.batch_dofs_info) else i_d
                    if self.dofs_state[i_d, i_b].ctrl_mode == gs.CTRL_MODE.FORCE:
                        force = self.dofs_state[i_d, i_b].ctrl_force
                    elif self.dofs_state[i_d, i_b].ctrl_mode == gs.CTRL_MODE.VELOCITY:
                        force = self.dofs_info[I_d].kv * (
                            self.dofs_state[i_d, i_b].ctrl_vel - self.dofs_state[i_d, i_b].vel
                        )
                    elif self.dofs_state[i_d, i_b].ctrl_mode == gs.CTRL_MODE.POSITION and not (
                        l_info.joint_type == gs.JOINT_TYPE.FREE and i_d >= l_info.dof_start + 3
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

                ds = l_info.dof_start
                if l_info.joint_type == gs.JOINT_TYPE.FREE and (
                    self.dofs_state[ds + 3, i_b].ctrl_mode == gs.CTRL_MODE.POSITION
                    or self.dofs_state[ds + 4, i_b].ctrl_mode == gs.CTRL_MODE.POSITION
                    or self.dofs_state[ds + 5, i_b].ctrl_mode == gs.CTRL_MODE.POSITION
                ):
                    xyz = ti.Vector(
                        [
                            self.dofs_state[0 + 3 + l_info.dof_start, i_b].pos,
                            self.dofs_state[1 + 3 + l_info.dof_start, i_b].pos,
                            self.dofs_state[2 + 3 + l_info.dof_start, i_b].pos,
                        ],
                        dt=gs.ti_float,
                    )

                    ctrl_xyz = ti.Vector(
                        [
                            self.dofs_state[0 + 3 + l_info.dof_start, i_b].ctrl_pos,
                            self.dofs_state[1 + 3 + l_info.dof_start, i_b].ctrl_pos,
                            self.dofs_state[2 + 3 + l_info.dof_start, i_b].ctrl_pos,
                        ],
                        dt=gs.ti_float,
                    )

                    quat = gu.ti_xyz_to_quat(xyz)
                    ctrl_quat = gu.ti_xyz_to_quat(ctrl_xyz)

                    q_diff = gu.ti_transform_quat_by_quat(ctrl_quat, gu.ti_inv_quat(quat))
                    rotvec = gu.ti_quat_to_rotvec(q_diff)

                    for i_d in range(l_info.dof_start + 3, l_info.dof_end):
                        I_d = [i_d, i_b] if ti.static(self._options.batch_dofs_info) else i_d
                        force = (
                            self.dofs_info[I_d].kp * rotvec[i_d - l_info.dof_start - 3]
                            - self.dofs_info[I_d].kv * self.dofs_state[i_d, i_b].vel
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
                for i_l_ in range(self.n_awake_links[i_b]):
                    i_l = self.awake_links[i_l_, i_b]
                    I_l = [i_l, i_b] if ti.static(self._options.batch_links_info) else i_l
                    l_info = self.links_info[I_l]
                    if (
                        l_info.joint_type == gs.JOINT_TYPE.REVOLUTE
                        or l_info.joint_type == gs.JOINT_TYPE.PRISMATIC
                        or l_info.joint_type == gs.JOINT_TYPE.PLANAR
                        or l_info.joint_type == gs.JOINT_TYPE.SPHERICAL
                    ):
                        dof_start = l_info.dof_start
                        q_start = l_info.q_start
                        q_end = l_info.q_end

                        for j_d in range(q_end - q_start):
                            I_d = (
                                [dof_start + j_d, i_b] if ti.static(self._options.batch_dofs_info) else dof_start + j_d
                            )
                            self.dofs_state[dof_start + j_d, i_b].qf_passive = (
                                -self.qpos[q_start + j_d, i_b] * self.dofs_info[I_d].stiffness
                            )

            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_b in range(self._B):
                for i_d_ in range(self.n_awake_dofs[i_b]):
                    i_d = self.awake_dofs[i_d_, i_b]
                    I_d = [i_d, i_b] if ti.static(self._options.batch_dofs_info) else i_d

                    self.dofs_state[i_d, i_b].qf_passive += -self.dofs_info[I_d].damping * self.dofs_state[i_d, i_b].vel
        else:
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_l, i_b in ti.ndrange(self.n_links, self._B):
                I_l = [i_l, i_b] if ti.static(self._options.batch_links_info) else i_l
                l_info = self.links_info[I_l]
                if (
                    l_info.joint_type == gs.JOINT_TYPE.REVOLUTE
                    or l_info.joint_type == gs.JOINT_TYPE.PRISMATIC
                    or l_info.joint_type == gs.JOINT_TYPE.PLANAR
                    or l_info.joint_type == gs.JOINT_TYPE.SPHERICAL
                ):
                    dof_start = l_info.dof_start
                    q_start = l_info.q_start
                    q_end = l_info.q_end

                    for j_d in range(q_end - q_start):
                        I_d = [dof_start + j_d, i_b] if ti.static(self._options.batch_dofs_info) else dof_start + j_d
                        self.dofs_state[dof_start + j_d, i_b].qf_passive = (
                            -self.qpos[q_start + j_d, i_b] * self.dofs_info[I_d].stiffness
                        )

            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_d, i_b in ti.ndrange(self.n_dofs, self._B):
                I_d = [i_d, i_b] if ti.static(self._options.batch_dofs_info) else i_d
                self.dofs_state[i_d, i_b].qf_passive += -self.dofs_info[I_d].damping * self.dofs_state[i_d, i_b].vel

    @ti.func
    def _func_system_update_acc(self, for_sensor):
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
                            if for_sensor:
                                self.links_state[i_l, i_b].cdd_vel = ti.Vector.zero(gs.ti_float, 3)
                            else:
                                self.links_state[i_l, i_b].cdd_vel = -self._gravity[None] * (
                                    1 - e_info.gravity_compensation
                                )
                            self.links_state[i_l, i_b].cdd_ang = ti.Vector.zero(gs.ti_float, 3)
                        else:
                            self.links_state[i_l, i_b].cdd_vel = self.links_state[i_p, i_b].cdd_vel
                            self.links_state[i_l, i_b].cdd_ang = self.links_state[i_p, i_b].cdd_ang
                            # dimension

                        map_sum_vel = ti.Vector.zero(gs.ti_float, 3)
                        map_sum_ang = ti.Vector.zero(gs.ti_float, 3)

                        for i_d in range(self.links_info[I_l].dof_start, self.links_info[I_l].dof_end):
                            map_sum_vel = (
                                map_sum_vel + self.dofs_state[i_d, i_b].cdofd_vel * self.dofs_state[i_d, i_b].vel
                            )
                            map_sum_ang = (
                                map_sum_ang + self.dofs_state[i_d, i_b].cdofd_ang * self.dofs_state[i_d, i_b].vel
                            )

                            if for_sensor:
                                map_sum_ang = (
                                    map_sum_vel + self.dofs_state[i_d, i_b].cdof_vel * self.dofs_state[i_d, i_b].acc
                                )
                                map_sum_vel = (
                                    map_sum_ang + self.dofs_state[i_d, i_b].cdofd_ang * self.dofs_state[i_d, i_b].acc
                                )

                        self.links_state[i_l, i_b].cdd_vel = self.links_state[i_l, i_b].cdd_vel + map_sum_vel
                        self.links_state[i_l, i_b].cdd_ang = self.links_state[i_l, i_b].cdd_ang + map_sum_ang
        else:
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_e, i_b in ti.ndrange(self.n_entities, self._B):
                e_info = self.entities_info[i_e]
                for i_l in range(e_info.link_start, e_info.link_end):
                    I_l = [i_l, i_b] if ti.static(self._options.batch_links_info) else i_l

                    i_p = self.links_info[I_l].parent_idx
                    if i_p == -1:
                        if for_sensor:
                            self.links_state[i_l, i_b].cdd_vel = ti.Vector.zero(gs.ti_float, 3)
                        else:
                            self.links_state[i_l, i_b].cdd_vel = -self._gravity[None] * (
                                1 - e_info.gravity_compensation
                            )
                        self.links_state[i_l, i_b].cdd_ang = ti.Vector.zero(gs.ti_float, 3)
                    else:
                        self.links_state[i_l, i_b].cdd_vel = self.links_state[i_p, i_b].cdd_vel
                        self.links_state[i_l, i_b].cdd_ang = self.links_state[i_p, i_b].cdd_ang
                        # dimension

                    map_sum_vel = ti.Vector.zero(gs.ti_float, 3)
                    map_sum_ang = ti.Vector.zero(gs.ti_float, 3)

                    for i_d in range(self.links_info[I_l].dof_start, self.links_info[I_l].dof_end):
                        map_sum_vel = map_sum_vel + self.dofs_state[i_d, i_b].cdofd_vel * self.dofs_state[i_d, i_b].vel
                        map_sum_ang = map_sum_ang + self.dofs_state[i_d, i_b].cdofd_ang * self.dofs_state[i_d, i_b].vel

                        if for_sensor:
                            map_sum_ang = (
                                map_sum_vel + self.dofs_state[i_d, i_b].cdof_vel * self.dofs_state[i_d, i_b].acc
                            )
                            map_sum_vel = (
                                map_sum_ang + self.dofs_state[i_d, i_b].cdofd_ang * self.dofs_state[i_d, i_b].acc
                            )

                    self.links_state[i_l, i_b].cdd_vel = self.links_state[i_l, i_b].cdd_vel + map_sum_vel
                    self.links_state[i_l, i_b].cdd_ang = self.links_state[i_l, i_b].cdd_ang + map_sum_ang

    @ti.func
    def _func_system_update_force(self):
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

                    self.links_state[i_l, i_b].cfrc_flat_vel = f1_vel + f2_vel
                    self.links_state[i_l, i_b].cfrc_flat_ang = f1_ang + f2_ang

                    self.links_state[i_l, i_b].cfrc_flat_vel = (
                        self.links_state[i_l, i_b].cfrc_ext_vel + self.links_state[i_l, i_b].cfrc_flat_vel
                    )
                    self.links_state[i_l, i_b].cfrc_flat_ang = (
                        self.links_state[i_l, i_b].cfrc_ext_ang + self.links_state[i_l, i_b].cfrc_flat_ang
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

                self.links_state[i_l, i_b].cfrc_flat_vel = f1_vel + f2_vel
                self.links_state[i_l, i_b].cfrc_flat_ang = f1_ang + f2_ang

                self.links_state[i_l, i_b].cfrc_flat_vel = (
                    self.links_state[i_l, i_b].cfrc_ext_vel + self.links_state[i_l, i_b].cfrc_flat_vel
                )
                self.links_state[i_l, i_b].cfrc_flat_ang = (
                    self.links_state[i_l, i_b].cfrc_ext_ang + self.links_state[i_l, i_b].cfrc_flat_ang
                )

    @ti.func
    def _func_inverse_link_force(self):
        if ti.static(self._use_hibernation):
            for i_b in range(self._B):
                for i_e_ in range(self.n_awake_entities[i_b]):
                    i_e = self.awake_entities[i_e_, i_b]
                    e_info = self.entities_info[i_e]
                    for i in range(e_info.n_links):
                        i_l = e_info.link_end - 1 - i
                        I_l = [i_l, i_b] if ti.static(self._options.batch_links_info) else i_l
                        i_p = self.links_info[I_l].parent_idx
                        if i_p != -1:
                            self.links_state[i_p, i_b].cfrc_flat_vel = (
                                self.links_state[i_p, i_b].cfrc_flat_vel + self.links_state[i_l, i_b].cfrc_flat_vel
                            )
                            self.links_state[i_p, i_b].cfrc_flat_ang = (
                                self.links_state[i_p, i_b].cfrc_flat_ang + self.links_state[i_l, i_b].cfrc_flat_ang
                            )

        else:
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_e, i_b in ti.ndrange(self.n_entities, self._B):
                e_info = self.entities_info[i_e]
                for i in range(e_info.n_links):
                    i_l = e_info.link_end - 1 - i
                    I_l = [i_l, i_b] if ti.static(self._options.batch_links_info) else i_l
                    i_p = self.links_info[I_l].parent_idx
                    if i_p != -1:
                        self.links_state[i_p, i_b].cfrc_flat_vel = (
                            self.links_state[i_p, i_b].cfrc_flat_vel + self.links_state[i_l, i_b].cfrc_flat_vel
                        )
                        self.links_state[i_p, i_b].cfrc_flat_ang = (
                            self.links_state[i_p, i_b].cfrc_flat_ang + self.links_state[i_l, i_b].cfrc_flat_ang
                        )

    @ti.func
    def _func_actuation(self):
        if ti.static(self._use_hibernation):
            pass
        else:
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_l, i_b in ti.ndrange(self.n_links, self._B):
                joint_type = self.links_info[i_l].joint_type
                I_l = [i_l, i_b] if ti.static(self._options.batch_links_info) else i_l
                q_start = self.links_info[I_l].q_start

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
                    dof_start = l_info.dof_start
                    dof_end = l_info.dof_end

                    for i_d in range(dof_start, dof_end):
                        self.dofs_state[i_d, i_b].qf_bias = self.dofs_state[i_d, i_b].cdof_ang.dot(
                            self.links_state[i_l, i_b].cfrc_flat_ang
                        ) + self.dofs_state[i_d, i_b].cdof_vel.dot(self.links_state[i_l, i_b].cfrc_flat_vel)

                        self.dofs_state[i_d, i_b].force = (
                            self.dofs_state[i_d, i_b].qf_passive
                            - self.dofs_state[i_d, i_b].qf_bias
                            + self.dofs_state[i_d, i_b].qf_applied
                        )
                        # + self.dofs_state[i_d, i_b].qf_actuator \

        else:
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_l, i_b in ti.ndrange(self.n_links, self._B):
                I_l = [i_l, i_b] if ti.static(self._options.batch_links_info) else i_l

                l_info = self.links_info[I_l]
                dof_start = l_info.dof_start
                dof_end = l_info.dof_end

                for i_d in range(dof_start, dof_end):
                    self.dofs_state[i_d, i_b].qf_bias = self.dofs_state[i_d, i_b].cdof_ang.dot(
                        self.links_state[i_l, i_b].cfrc_flat_ang
                    ) + self.dofs_state[i_d, i_b].cdof_vel.dot(self.links_state[i_l, i_b].cfrc_flat_vel)

                    self.dofs_state[i_d, i_b].force = (
                        self.dofs_state[i_d, i_b].qf_passive
                        - self.dofs_state[i_d, i_b].qf_bias
                        + self.dofs_state[i_d, i_b].qf_applied
                    )
                    # + self.dofs_state[i_d, i_b].qf_actuator \

    @ti.func
    def _func_compute_qacc(self):
        if ti.static(self._use_hibernation):
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
            for i_d1_, i_b in ti.ndrange(self.entity_max_dofs, self._B):
                for i_e_ in range(self.n_awake_entities[i_b]):
                    i_e = self.awake_entities[i_e_, i_b]
                    e_info = self.entities_info[i_e]
                    if i_d1_ < e_info.n_dofs:
                        i_d1 = e_info.dof_start + i_d1_
                        acc = gs.ti_float(0.0)
                        for i_d2 in range(e_info.dof_start, e_info.dof_end):
                            acc += self.mass_mat_inv[i_d1, i_d2, i_b] * self.dofs_state[i_d2, i_b].force
                        self.dofs_state[i_d1, i_b].acc = acc
                        self.dofs_state[i_d1, i_b].acc_smooth = acc

        else:
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
            for i_e, i_d1_, i_b in ti.ndrange(self.n_entities, self.entity_max_dofs, self._B):
                e_info = self.entities_info[i_e]
                if i_d1_ < e_info.n_dofs:
                    i_d1 = e_info.dof_start + i_d1_
                    acc = gs.ti_float(0.0)
                    for i_d2 in range(e_info.dof_start, e_info.dof_end):
                        acc += self.mass_mat_inv[i_d1, i_d2, i_b] * self.dofs_state[i_d2, i_b].force
                    self.dofs_state[i_d1, i_b].acc = acc
                    self.dofs_state[i_d1, i_b].acc_smooth = acc

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
                    joint_type = self.links_info[I_l].joint_type
                    dof_start = self.links_info[I_l].dof_start
                    q_start = self.links_info[I_l].q_start
                    q_end = self.links_info[I_l].q_end

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
                    else:
                        for j in range(q_end - q_start):
                            self.qpos[q_start + j, i_b] = (
                                self.qpos[q_start + j, i_b] + self.dofs_state[dof_start + j, i_b].vel * self._substep_dt
                            )

        else:
            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_d, i_b in ti.ndrange(self.n_dofs, self._B):
                self.dofs_state[i_d, i_b].vel = (
                    self.dofs_state[i_d, i_b].vel + self.dofs_state[i_d, i_b].acc * self._substep_dt
                )

            # for i_b in range(self._B):
            #     is_exploded = False
            #     for i_d in range(self.n_dofs):
            #         self.dofs_state[i_d, i_b].vel = self.dofs_state[i_d, i_b].vel + self.dofs_state[i_d, i_b].acc * self._substep_dt
            #         if ti.abs(self.dofs_state[i_d, i_b].acc) > 1e10:
            #             is_exploded = True

            #     if is_exploded:
            #         print("WARNING! acc is too large, set to zero", i_b)
            #         for i_d in range(self.n_dofs):
            #             self.dofs_state[i_d, i_b].vel = gs.ti_float(0.)

            ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
            for i_l, i_b in ti.ndrange(self.n_links, self._B):
                I_l = [i_l, i_b] if ti.static(self._options.batch_links_info) else i_l
                joint_type = self.links_info[I_l].joint_type
                dof_start = self.links_info[I_l].dof_start
                q_start = self.links_info[I_l].q_start
                q_end = self.links_info[I_l].q_end

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
                    for j in ti.static(range(4)):
                        self.qpos[q_start + j + 3, i_b] = rot[j]
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
            q_start = l_info.q_start
            dof_start = l_info.dof_start
            dq_start = l_info.dof_start - e_info.dof_start

            if l_info.joint_type == gs.JOINT_TYPE.FREE:
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

            elif l_info.joint_type == gs.JOINT_TYPE.FIXED:
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
            for i, j in ti.ndrange(4, 4):
                geoms_render_T[i_g, i_b, i, j] = geom_T[i, j]

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
            for i, j in ti.ndrange(4, 4):
                vgeoms_render_T[i_g, i_b, i, j] = geom_T[i, j]

    def update_vgeoms_render_T(self):
        self._kernel_update_vgeoms_render_T(self._vgeoms_render_T)

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
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
        for i_q, i_b in ti.ndrange(self.n_qs, self._B):
            self.qpos[i_q, i_b] = qpos[i_b, i_q]

        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
        for i_d, i_b in ti.ndrange(self.n_dofs, self._B):
            self.dofs_state[i_d, i_b].vel = dofs_vel[i_b, i_d]

        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
        for i_l, i_b in ti.ndrange(self.n_links, self._B):
            for i in ti.static(range(3)):
                self.links_state[i_l, i_b].pos[i] = links_pos[i_b, i_l, i]
                self.links_state[i_l, i_b].i_pos_shift[i] = i_pos_shift[i_b, i_l, i]
            for i in ti.static(range(4)):
                self.links_state[i_l, i_b].quat[i] = links_quat[i_b, i_l, i]
            self.links_state[i_l, i_b].mass_shift = mass_shift[i_b, i_l]

        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.ALL)
        for i_l, i_b in ti.ndrange(self.n_geoms, self._B):
            self.geoms_state[i_l, i_b].friction_ratio = friction_ratio[i_b, i_l]

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

    def set_state(self, f, state):
        if self.is_active():
            self._kernel_set_state(
                state.qpos,
                state.dofs_vel,
                state.links_pos,
                state.links_quat,
                state.i_pos_shift,
                state.mass_shift,
                state.friction_ratio,
            )
            self._kernel_forward_kinematics_links_geoms()
            self.collider.reset()
            if self.constraint_solver is not None:
                self.constraint_solver.reset()
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

    def _get_envs_idx(self, envs_idx):
        if envs_idx is None:
            envs_idx = self._scene._envs_idx

        else:
            if self.n_envs == 0:
                gs.raise_exception("`envs_idx` is not supported for non-parallelized scene.")

            envs_idx = torch.as_tensor(envs_idx, dtype=gs.tc_int, device=gs.device).contiguous()

            if envs_idx.ndim != 1:
                gs.raise_exception("Expecting a 1D tensor for `envs_idx`.")

            if (envs_idx < 0).any() or (envs_idx >= self.n_envs).any():
                gs.raise_exception("`envs_idx` exceeds valid range.")

        return envs_idx

    def _validate_1D_io_variables(self, tensor, inputs_idx, envs_idx=None, batched=True, idx_name="dofs_idx"):
        inputs_idx = torch.as_tensor(inputs_idx, dtype=gs.tc_int, device=gs.device).contiguous()
        if inputs_idx.ndim != 1:
            gs.raise_exception(f"Expecting 1D tensor for `{idx_name}`.")

        if tensor is not None:
            tensor = torch.as_tensor(tensor, dtype=gs.tc_float, device=gs.device).contiguous()
            if tensor.shape[-1] != len(inputs_idx):
                gs.raise_exception(f"Last dimension of the input tensor does not match length of `{idx_name}`.")
        else:
            if batched and self.n_envs > 0:
                if envs_idx is not None:
                    B = len(envs_idx)
                else:
                    B = self.n_envs
                tensor = torch.empty(self._batch_shape(len(inputs_idx), True, B=B), dtype=gs.tc_float, device=gs.device)
            else:
                tensor = torch.empty(len(inputs_idx), dtype=gs.tc_float, device=gs.device)

        if batched:
            envs_idx = self._get_envs_idx(envs_idx)

            if self.n_envs == 0:
                if tensor.ndim == 1:
                    tensor = tensor[None, :]
                else:
                    gs.raise_exception(
                        f"Invalid input shape: {tensor.shape}. Expecting a 1D tensor for non-parallelized scene."
                    )

            else:
                if tensor.ndim == 2:
                    if tensor.shape[0] != len(envs_idx):
                        gs.raise_exception(
                            f"Invalid input shape: {tensor.shape}. First dimension of the input tensor does not match length of `envs_idx` (or `scene.n_envs` if `envs_idx` is None)."
                        )
                else:
                    tensor = tensor.repeat(len(envs_idx), 1)
                    gs.logger.warning(f"Input tensor is converted to {tensor.shape} for an additional batch dimension")

            return tensor, inputs_idx, envs_idx

        else:
            if tensor.ndim != 1:
                gs.raise_exception("Expecting 1D input tensor.")

            return tensor, inputs_idx

    def _validate_2D_io_variables(
        self, tensor, inputs_idx, vec_size, envs_idx=None, batched=True, idx_name="links_idx"
    ):
        inputs_idx = torch.as_tensor(inputs_idx, dtype=gs.tc_int, device=gs.device).contiguous()
        if inputs_idx.ndim != 1:
            gs.raise_exception(f"Expecting 1D tensor for `{idx_name}`.")

        if tensor is not None:
            tensor = torch.as_tensor(tensor, dtype=gs.tc_float, device=gs.device).contiguous()
            if tensor.shape[-2] != len(inputs_idx):
                gs.raise_exception(f"Second last dimension of the input tensor does not match length of `{idx_name}`.")
            if tensor.shape[-1] != vec_size:
                gs.raise_exception(f"Last dimension of the input tensor must be {vec_size}.")

        else:
            if batched and self.n_envs > 0:
                if envs_idx is not None:
                    B = len(envs_idx)
                else:
                    B = self.n_envs
                tensor = torch.empty(
                    self._batch_shape((len(inputs_idx), vec_size), True, B=B), dtype=gs.tc_float, device=gs.device
                )
            else:
                tensor = torch.empty((len(inputs_idx), vec_size), dtype=gs.tc_float, device=gs.device)

        if batched:
            envs_idx = self._get_envs_idx(envs_idx)

            if self.n_envs == 0:
                if tensor.ndim == 2:
                    tensor = tensor[None, :]
                else:
                    gs.raise_exception(
                        f"Invalid input shape: {tensor.shape}. Expecting a 2D tensor for non-parallelized scene."
                    )

            else:
                if tensor.ndim == 3:
                    if tensor.shape[0] != len(envs_idx):
                        gs.raise_exception(
                            f"Invalid input shape: {tensor.shape}. First dimension of the input tensor does not match length of `envs_idx` (or `scene.n_envs` if `envs_idx` is None)."
                        )
                else:
                    gs.raise_exception(
                        f"Invalid input shape: {tensor.shape}. Expecting a 3D tensor for scene with parallelized envs."
                    )

            return tensor, inputs_idx, envs_idx

        else:
            if tensor.ndim != 2:
                gs.raise_exception("Expecting 2D input tensor.")

            return tensor, inputs_idx

    def _get_qs_idx(self, qs_idx_local=None):
        return self._get_qs_idx_local(qs_idx_local) + self._q_start

    def set_links_pos(self, pos, links_idx, envs_idx=None):
        """
        Only effetive for base links.
        """
        pos, links_idx, envs_idx = self._validate_2D_io_variables(pos, links_idx, 3, envs_idx, idx_name="links_idx")

        self._kernel_set_links_pos(
            pos,
            links_idx,
            envs_idx,
        )
        self._kernel_forward_kinematics_links_geoms()

    @ti.kernel
    def _kernel_set_links_pos(
        self,
        pos: ti.types.ndarray(),
        links_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i_l_, i_b_ in ti.ndrange(links_idx.shape[0], envs_idx.shape[0]):
            i_l = links_idx[i_l_]
            I_l = [i_l, i_b_] if ti.static(self._options.batch_links_info) else i_l
            if self.links_info[I_l].is_fixed:  # change links_state directly as the link's pose is not contained in qpos
                for i in ti.static(range(3)):
                    self.links_state[i_l, envs_idx[i_b_]].pos[i] = pos[i_b_, i_l_, i]

            else:  # free base link's pose is reflected in qpos, and links_state will be computed automatically
                q_start = self.links_info[I_l].q_start
                for i in ti.static(range(3)):
                    self.qpos[q_start + i, envs_idx[i_b_]] = pos[i_b_, i_l_, i]

    def set_links_quat(self, quat, links_idx, envs_idx=None):
        """
        Only effetive for base links.
        """
        quat, links_idx, envs_idx = self._validate_2D_io_variables(quat, links_idx, 4, envs_idx, idx_name="links_idx")

        self._kernel_set_links_quat(
            quat,
            links_idx,
            envs_idx,
        )
        self._kernel_forward_kinematics_links_geoms()

    @ti.kernel
    def _kernel_set_links_quat(
        self,
        quat: ti.types.ndarray(),
        links_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i_l_, i_b_ in ti.ndrange(links_idx.shape[0], envs_idx.shape[0]):
            i_l = links_idx[i_l_]
            I_l = [i_l, i_b_] if ti.static(self._options.batch_links_info) else i_l
            if self.links_info[I_l].is_fixed:  # change links_state directly as the link's pose is not contained in qpos
                for i in ti.static(range(4)):
                    self.links_state[i_l, envs_idx[i_b_]].quat[i] = quat[i_b_, i_l_, i]

            else:  # free base link's pose is reflected in qpos, and links_state will be computed automatically
                q_start = self.links_info[I_l].q_start
                for i in ti.static(range(4)):
                    self.qpos[q_start + i + 3, envs_idx[i_b_]] = quat[i_b_, i_l_, i]

    def set_links_mass_shift(self, mass, links_idx, envs_idx=None):
        mass, links_idx, envs_idx = self._validate_1D_io_variables(mass, links_idx, envs_idx, idx_name="links_idx")

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

    def set_links_COM_shift(self, com, links_idx, envs_idx=None):
        com, links_idx, envs_idx = self._validate_2D_io_variables(com, links_idx, 3, envs_idx, idx_name="links_idx")

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

    def _set_links_info(self, tensor, links_idx, name, envs_idx=None):
        if self._options.batch_links_info:
            tensor, links_idx, envs_idx = self._validate_1D_io_variables(
                tensor, links_idx, envs_idx, idx_name="links_idx"
            )
        else:
            tensor, links_idx = self._validate_1D_io_variables(tensor, links_idx, idx_name="links_idx", batched=False)
            envs_idx = torch.empty(())

        if name == "invweight":
            self._kernel_set_links_invweight(tensor, links_idx, envs_idx)
        elif name == "inertial_mass":
            self._kernel_set_links_inertial_mass(tensor, links_idx, envs_idx)
        else:
            gs.raise_exception(f"Invalid `name` {name}.")

    def set_links_inertial_mass(self, invweight, links_idx, envs_idx=None):
        self._set_links_info(invweight, links_idx, "inertial_mass", envs_idx)

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

    def set_links_invweight(self, invweight, links_idx, envs_idx=None):
        self._set_links_info(invweight, links_idx, "invweight", envs_idx)

    @ti.kernel
    def _kernel_set_links_invweight(
        self,
        invweight: ti.types.ndarray(),
        links_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        if ti.static(self._options.batch_links_info):
            for i_l_, i_b_ in ti.ndrange(links_idx.shape[0], envs_idx.shape[0]):
                self.links_info[links_idx[i_l_], envs_idx[i_b_]].invweight = invweight[i_b_, i_l_]
        else:
            for i_l_ in range(links_idx.shape[0]):
                self.links_info[links_idx[i_l_]].invweight = invweight[i_l_]

    def set_geoms_friction_ratio(self, friction_ratio, geoms_idx, envs_idx=None):
        friction_ratio, geoms_idx, envs_idx = self._validate_1D_io_variables(
            friction_ratio, geoms_idx, envs_idx, idx_name="geoms_idx"
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

    def set_qpos(self, qpos, qs_idx, envs_idx=None):
        qpos, qs_idx, envs_idx = self._validate_1D_io_variables(qpos, qs_idx, envs_idx, idx_name="qs_idx")

        self._kernel_set_qpos(
            qpos,
            qs_idx,
            envs_idx,
        )
        self._kernel_forward_kinematics_links_geoms()

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

    def set_global_sol_params(self, sol_params):
        """
        Solver parameters (timeconst, dampratio, dmin, dmax, width, mid, power).
        Reference: https://mujoco.readthedocs.io/en/latest/modeling.html#solver-parameters
        """
        assert len(sol_params) == 7
        self._kernel_set_global_sol_params(sol_params)

    @ti.kernel
    def _kernel_set_global_sol_params(self, sol_params: ti.types.ndarray()):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i in range(self.n_geoms):
            for j in ti.static(range(7)):
                self.geoms_info[i].sol_params[j] = sol_params[j]

        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i, b in ti.ndrange(self.n_dofs, self._B):
            I = [i, b] if ti.static(self._options.batch_dofs_info) else i
            for j in ti.static(range(7)):
                self.dofs_info[I].sol_params[j] = sol_params[j]

            self.dofs_info[I].sol_params[0] = self._substep_dt * 2

    def _set_dofs_info(self, tensor_list, dofs_idx, name, envs_idx=None):
        if self._options.batch_dofs_info:
            for i, tensor in enumerate(tensor_list):
                if i == (len(tensor_list) - 1):
                    tensor_list[i], dofs_idx, envs_idx = self._validate_1D_io_variables(tensor, dofs_idx, envs_idx)
                else:
                    tensor_list[i], _, _ = self._validate_1D_io_variables(tensor, dofs_idx, envs_idx)
        else:
            for i, tensor in enumerate(tensor_list):
                if i == (len(tensor_list) - 1):
                    tensor_list[i], _ = self._validate_1D_io_variables(tensor, dofs_idx, batched=False)
                else:
                    tensor_list[i], dofs_idx = self._validate_1D_io_variables(tensor, dofs_idx, batched=False)
            envs_idx = torch.empty(())

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

    def set_dofs_kp(self, kp, dofs_idx, envs_idx=None):
        self._set_dofs_info([kp], dofs_idx, "kp", envs_idx)

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

    def set_dofs_kv(self, kv, dofs_idx, envs_idx=None):
        self._set_dofs_info([kv], dofs_idx, "kv", envs_idx)

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

    def set_dofs_force_range(self, lower, upper, dofs_idx, envs_idx=None):
        self._set_dofs_info([lower, upper], dofs_idx, "force_range", envs_idx)

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

    def set_dofs_stiffness(self, stiffness, dofs_idx, envs_idx=None):
        self._set_dofs_info([stiffness], dofs_idx, "stiffness", envs_idx)

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

    def set_dofs_invweight(self, invweight, dofs_idx, envs_idx=None):
        self._set_dofs_info([invweight], dofs_idx, "invweight", envs_idx)

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

    def set_dofs_armature(self, armature, dofs_idx, envs_idx=None):
        self._set_dofs_info([armature], dofs_idx, "armature", envs_idx)

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

    def set_dofs_damping(self, damping, dofs_idx, envs_idx=None):
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

    def set_dofs_limit(self, lower, upper, dofs_idx, envs_idx=None):
        self._set_dofs_info([lower, upper], dofs_idx, "limit", envs_idx)

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

    def set_dofs_velocity(self, velocity, dofs_idx, envs_idx=None):
        velocity, dofs_idx, envs_idx = self._validate_1D_io_variables(velocity, dofs_idx, envs_idx)
        self._kernel_set_dofs_velocity(
            velocity,
            dofs_idx,
            envs_idx,
        )
        self._kernel_forward_kinematics_links_geoms()

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

    def set_dofs_position(self, position, dofs_idx, envs_idx=None):
        position, dofs_idx, envs_idx = self._validate_1D_io_variables(position, dofs_idx, envs_idx)
        self._kernel_set_dofs_position(
            position,
            dofs_idx,
            envs_idx,
        )
        self._kernel_forward_kinematics_links_geoms()

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

                if l_info.joint_type == gs.JOINT_TYPE.FREE:
                    xyz = ti.Vector(
                        [
                            self.dofs_state[0 + 3 + l_info.dof_start, i_b].pos,
                            self.dofs_state[1 + 3 + l_info.dof_start, i_b].pos,
                            self.dofs_state[2 + 3 + l_info.dof_start, i_b].pos,
                        ],
                        dt=gs.ti_float,
                    )
                    quat = gu.ti_xyz_to_quat(xyz)

                    for i_q in ti.static(range(3)):
                        self.qpos[i_q + l_info.q_start, i_b] = self.dofs_state[i_q + l_info.dof_start, i_b].pos

                    for i_q in ti.static(range(4)):
                        self.qpos[i_q + 3 + l_info.q_start, i_b] = quat[i_q]
                else:
                    for i_q in range(l_info.q_start, l_info.q_end):
                        self.qpos[i_q, i_b] = self.dofs_state[l_info.dof_start + i_q - l_info.q_start, i_b].pos

    def control_dofs_force(self, force, dofs_idx, envs_idx=None):
        force, dofs_idx, envs_idx = self._validate_1D_io_variables(force, dofs_idx, envs_idx)
        self._kernel_control_dofs_force(
            force,
            dofs_idx,
            envs_idx,
        )

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

    def control_dofs_velocity(self, velocity, dofs_idx, envs_idx=None):
        velocity, dofs_idx, envs_idx = self._validate_1D_io_variables(velocity, dofs_idx, envs_idx)
        self._kernel_control_dofs_velocity(
            velocity,
            dofs_idx,
            envs_idx,
        )

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

    def control_dofs_position(self, position, dofs_idx, envs_idx=None):
        position, dofs_idx, envs_idx = self._validate_1D_io_variables(position, dofs_idx, envs_idx)
        self._kernel_control_dofs_position(
            position,
            dofs_idx,
            envs_idx,
        )

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

    def get_links_pos(self, links_idx, envs_idx=None):
        tensor, links_idx, envs_idx = self._validate_2D_io_variables(None, links_idx, 3, envs_idx, idx_name="links_idx")

        self._kernel_get_links_pos(tensor, links_idx, envs_idx)

        if self.n_envs == 0:
            tensor = tensor.squeeze(0)
        return tensor

    @ti.kernel
    def _kernel_get_links_pos(
        self,
        tensor: ti.types.ndarray(),
        links_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i_l_, i_b_ in ti.ndrange(links_idx.shape[0], envs_idx.shape[0]):
            for i in ti.static(range(3)):
                tensor[i_b_, i_l_, i] = self.links_state[links_idx[i_l_], envs_idx[i_b_]].pos[i]

    def get_links_quat(self, links_idx, envs_idx=None):
        tensor, links_idx, envs_idx = self._validate_2D_io_variables(None, links_idx, 4, envs_idx, idx_name="links_idx")

        self._kernel_get_links_quat(tensor, links_idx, envs_idx)

        if self.n_envs == 0:
            tensor = tensor.squeeze(0)
        return tensor

    @ti.kernel
    def _kernel_get_links_quat(
        self,
        tensor: ti.types.ndarray(),
        links_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i_l_, i_b_ in ti.ndrange(links_idx.shape[0], envs_idx.shape[0]):
            for i in ti.static(range(4)):
                tensor[i_b_, i_l_, i] = self.links_state[links_idx[i_l_], envs_idx[i_b_]].quat[i]

    def get_links_vel(self, links_idx, envs_idx=None):
        tensor, links_idx, envs_idx = self._validate_2D_io_variables(None, links_idx, 3, envs_idx, idx_name="links_idx")

        self._kernel_get_links_vel(tensor, links_idx, envs_idx)

        if self.n_envs == 0:
            tensor = tensor.squeeze(0)
        return tensor

    @ti.kernel
    def _kernel_get_links_vel(
        self,
        tensor: ti.types.ndarray(),
        links_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i_l_, i_b_ in ti.ndrange(links_idx.shape[0], envs_idx.shape[0]):
            for i in ti.static(range(3)):
                tensor[i_b_, i_l_, i] = self.links_state[links_idx[i_l_], envs_idx[i_b_]].vel[i]

    def get_links_ang(self, links_idx, envs_idx=None):
        tensor, links_idx, envs_idx = self._validate_2D_io_variables(None, links_idx, 3, envs_idx, idx_name="links_idx")

        self._kernel_get_links_ang(tensor, links_idx, envs_idx)

        if self.n_envs == 0:
            tensor = tensor.squeeze(0)
        return tensor

    @ti.kernel
    def _kernel_get_links_ang(
        self,
        tensor: ti.types.ndarray(),
        links_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i_l_, i_b_ in ti.ndrange(links_idx.shape[0], envs_idx.shape[0]):
            for i in ti.static(range(3)):
                tensor[i_b_, i_l_, i] = self.links_state[links_idx[i_l_], envs_idx[i_b_]].ang[i]

    @ti.kernel
    def _kernel_inverse_dynamics_for_sensors(self):
        self._func_system_update_acc(True)
        self._func_system_update_force()
        self._func_inverse_link_force()

    def get_links_acc(self, links_idx, envs_idx=None):
        tensor, links_idx, envs_idx = self._validate_2D_io_variables(None, links_idx, 3, envs_idx, idx_name="links_idx")
        self._kernel_inverse_dynamics_for_sensors()
        self._kernel_get_links_acc(tensor, links_idx, envs_idx)

        if self.n_envs == 0:
            tensor = tensor.squeeze(0)
        return tensor

    @ti.kernel
    def _kernel_get_links_acc(
        self,
        tensor: ti.types.ndarray(),
        links_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i_l_, i_b_ in ti.ndrange(links_idx.shape[0], envs_idx.shape[0]):
            i_l = links_idx[i_l_]
            i_b = envs_idx[i_b_]

            quat = gu.ti_inv_quat(self.links_state[i_l, i_b].j_quat)
            dpos = self.links_state[i_l, i_b].pos - self.links_state[i_l, i_b].COM
            acc = gu.ti_transform_by_quat(  # gravitational component is included in cdd_vel already
                self.links_state[i_l, i_b].cdd_vel - dpos.cross(self.links_state[i_l, i_b].cdd_ang),
                quat,
            )
            ang = gu.ti_transform_by_quat(self.links_state[i_l, i_b].cd_ang, quat)
            lin = gu.ti_transform_by_quat(
                self.links_state[i_l, i_b].cd_vel - dpos.cross(self.links_state[i_l, i_b].cd_ang),
                quat,
            )
            correction = ang.cross(lin)  # centrifugal
            final_acc = acc + correction

            final_acc = self.links_state[i_l, i_b].cdd_vel
            for i in range(3):
                tensor[i_b_, i_l_, i] = final_acc[i]

    def get_links_COM(self, links_idx, envs_idx=None):
        tensor, links_idx, envs_idx = self._validate_2D_io_variables(None, links_idx, 3, envs_idx, idx_name="links_idx")

        self._kernel_get_links_COM(tensor, links_idx, envs_idx)

        if self.n_envs == 0:
            tensor = tensor.squeeze(0)
        return tensor

    @ti.kernel
    def _kernel_get_links_COM(
        self,
        tensor: ti.types.ndarray(),
        links_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i_l_, i_b_ in ti.ndrange(links_idx.shape[0], envs_idx.shape[0]):
            for i in ti.static(range(3)):
                tensor[i_b_, i_l_, i] = self.links_state[links_idx[i_l_], envs_idx[i_b_]].COM[i]

    def get_links_mass_shift(self, links_idx, envs_idx=None):
        tensor, links_idx, envs_idx = self._validate_1D_io_variables(None, links_idx, envs_idx, idx_name="links_idx")

        self._kernel_get_links_mass_shift(tensor, links_idx, envs_idx)

        if self.n_envs == 0:
            tensor = tensor.squeeze(0)
        return tensor

    @ti.kernel
    def _kernel_get_links_mass_shift(
        self,
        tensor: ti.types.ndarray(),
        links_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i_l_, i_b_ in ti.ndrange(links_idx.shape[0], envs_idx.shape[0]):
            tensor[i_b_, i_l_] = self.links_state[links_idx[i_l_], envs_idx[i_b_]].mass_shift

    def get_links_COM_shift(self, links_idx, envs_idx=None):
        tensor, links_idx, envs_idx = self._validate_2D_io_variables(None, links_idx, 3, envs_idx, idx_name="links_idx")
        self._kernel_get_links_COM_shift(tensor, links_idx, envs_idx)

        if self.n_envs == 0:
            tensor = tensor.squeeze(0)
        return tensor

    @ti.kernel
    def _kernel_get_links_COM_shift(
        self,
        tensor: ti.types.ndarray(),
        links_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i_l_, i_b_ in ti.ndrange(links_idx.shape[0], envs_idx.shape[0]):
            for i in ti.static(range(3)):
                tensor[i_b_, i_l_, i] = self.links_state[links_idx[i_l_], envs_idx[i_b_]].i_pos_shift[i]

    def _get_links_info(self, links_idx, name, envs_idx=None):
        if self._options.batch_links_info:
            tensor, links_idx, envs_idx = self._validate_1D_io_variables(
                None, links_idx, envs_idx, idx_name="links_idx"
            )
        else:
            tensor, links_idx = self._validate_1D_io_variables(None, links_idx, idx_name="links_idx", batched=False)
            envs_idx = torch.empty(())

        if name == "invweight":
            self._kernel_get_links_invweight(tensor, links_idx, envs_idx)
            return tensor
        elif name == "inertial_mass":
            self._kernel_get_links_inertial_mass(tensor, links_idx, envs_idx)
            return tensor
        else:
            gs.raise_exception(f"Invalid `name` {name}.")

    def get_links_inertial_mass(self, links_idx, envs_idx=None):
        return self._get_links_info(links_idx, "inertial_mass", envs_idx)

    @ti.kernel
    def _kernel_get_links_inertial_mass(
        self,
        tensor: ti.types.ndarray(),
        links_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        if ti.static(self._options.batch_links_info):
            for i_l_, i_b_ in ti.ndrange(links_idx.shape[0], envs_idx.shape[0]):
                tensor[i_b_, i_l_] = self.links_info[links_idx[i_l_], envs_idx[i_b_]].inertial_mass
        else:
            for i_l_ in range(links_idx.shape[0]):
                tensor[i_l_] = self.links_info[links_idx[i_l_]].inertial_mass

    def get_links_invweight(self, links_idx, envs_idx=None):
        return self._get_links_info(links_idx, "invweight", envs_idx)

    @ti.kernel
    def _kernel_get_links_invweight(
        self,
        tensor: ti.types.ndarray(),
        links_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        if ti.static(self._options.batch_links_info):
            for i_l_, i_b_ in ti.ndrange(links_idx.shape[0], envs_idx.shape[0]):
                tensor[i_b_, i_l_] = self.links_info[links_idx[i_l_], envs_idx[i_b_]].invweight
        else:
            for i_l_ in range(links_idx.shape[0]):
                tensor[i_l_] = self.links_info[links_idx[i_l_]].invweight

    def get_geoms_friction_ratio(self, geoms_idx, envs_idx=None):
        tensor, geoms_idx, envs_idx = self._validate_1D_io_variables(None, geoms_idx, envs_idx, idx_name="geoms_idx")

        self._kernel_get_geoms_friction_ratio(tensor, geoms_idx, envs_idx)

        if self.n_envs == 0:
            tensor = tensor.squeeze(0)
        return tensor

    @ti.kernel
    def _kernel_get_geoms_friction_ratio(
        self,
        tensor: ti.types.ndarray(),
        geoms_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i_g_, i_b_ in ti.ndrange(geoms_idx.shape[0], envs_idx.shape[0]):
            tensor[i_b_, i_g_] = self.geoms_state[geoms_idx[i_g_], envs_idx[i_b_]].friction_ratio

    def get_geoms_pos(self, geoms_idx, envs_idx=None):
        tensor, geoms_idx, envs_idx = self._validate_2D_io_variables(None, geoms_idx, 3, envs_idx, idx_name="geoms_idx")

        self._kernel_get_geoms_pos(tensor, geoms_idx, envs_idx)

        if self.n_envs == 0:
            tensor = tensor.squeeze(0)
        return tensor

    @ti.kernel
    def _kernel_get_geoms_pos(
        self,
        tensor: ti.types.ndarray(),
        geoms_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i_g_, i_b_ in ti.ndrange(geoms_idx.shape[0], envs_idx.shape[0]):
            for i in ti.static(range(3)):
                tensor[i_b_, i_g_, i] = self.geoms_state[geoms_idx[i_g_], envs_idx[i_b_]].pos[i]

    def get_qpos(self, qs_idx, envs_idx=None):
        tensor, qs_idx, envs_idx = self._validate_1D_io_variables(None, qs_idx, envs_idx, idx_name="qs_idx")

        self._kernel_get_qpos(tensor, qs_idx, envs_idx)

        if self.n_envs == 0:
            tensor = tensor.squeeze(0)
        return tensor

    @ti.kernel
    def _kernel_get_qpos(
        self,
        tensor: ti.types.ndarray(),
        qs_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i_q_, i_b_ in ti.ndrange(qs_idx.shape[0], envs_idx.shape[0]):
            tensor[i_b_, i_q_] = self.qpos[qs_idx[i_q_], envs_idx[i_b_]]

    def get_dofs_control_force(self, dofs_idx, envs_idx=None):
        return self._get_dofs_state(dofs_idx, "control_force", envs_idx)

    def get_dofs_force(self, dofs_idx, envs_idx=None):
        return self._get_dofs_state(dofs_idx, "force", envs_idx)

    def get_dofs_velocity(self, dofs_idx, envs_idx=None):
        return self._get_dofs_state(dofs_idx, "velocity", envs_idx)

    def get_dofs_position(self, dofs_idx, envs_idx=None):
        return self._get_dofs_state(dofs_idx, "position", envs_idx)

    def _get_dofs_state(self, dofs_idx, name, envs_idx=None):
        tensor, dofs_idx, envs_idx = self._validate_1D_io_variables(None, dofs_idx, envs_idx)

        if name == "control_force":
            self._kernel_get_dofs_control_force(tensor, dofs_idx, envs_idx)
        elif name == "force":
            self._kernel_get_dofs_force(tensor, dofs_idx, envs_idx)
        elif name == "velocity":
            self._kernel_get_dofs_velocity(tensor, dofs_idx, envs_idx)
        elif name == "position":
            self._kernel_get_dofs_position(tensor, dofs_idx, envs_idx)
        else:
            gs.raise_exception("Invalid `name`.")

        if self.n_envs == 0:
            tensor = tensor.squeeze(0)
        return tensor

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

    @ti.kernel
    def _kernel_get_dofs_force(
        self,
        tensor: ti.types.ndarray(),
        dofs_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
            tensor[i_b_, i_d_] = self.dofs_state[dofs_idx[i_d_], envs_idx[i_b_]].force

    @ti.kernel
    def _kernel_get_dofs_velocity(
        self,
        tensor: ti.types.ndarray(),
        dofs_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
            tensor[i_b_, i_d_] = self.dofs_state[dofs_idx[i_d_], envs_idx[i_b_]].vel

    @ti.kernel
    def _kernel_get_dofs_position(
        self,
        tensor: ti.types.ndarray(),
        dofs_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
            tensor[i_b_, i_d_] = self.dofs_state[dofs_idx[i_d_], envs_idx[i_b_]].pos

    def get_dofs_kp(self, dofs_idx, envs_idx=None):
        return self._get_dofs_info(dofs_idx, "kp", envs_idx)

    def get_dofs_kv(self, dofs_idx, envs_idx=None):
        return self._get_dofs_info(dofs_idx, "kv", envs_idx)

    def get_dofs_force_range(self, dofs_idx, envs_idx=None):
        return self._get_dofs_info(dofs_idx, "force_range", envs_idx)

    def get_dofs_limit(self, dofs_idx, envs_idx=None):
        return self._get_dofs_info(dofs_idx, "limit", envs_idx)

    def get_dofs_stiffness(self, dofs_idx, envs_idx=None):
        return self._get_dofs_info(dofs_idx, "stiffness", envs_idx)

    def get_dofs_invweight(self, dofs_idx, envs_idx=None):
        return self._get_dofs_info(dofs_idx, "invweight", envs_idx)

    def get_dofs_armature(self, dofs_idx, envs_idx=None):
        return self._get_dofs_info(dofs_idx, "armature", envs_idx)

    def get_dofs_damping(self, dofs_idx, envs_idx=None):
        return self._get_dofs_info(dofs_idx, "damping", envs_idx)

    def _get_dofs_info(self, dofs_idx, name, envs_idx=None):
        if self._options.batch_dofs_info:
            tensor, dofs_idx, envs_idx = self._validate_1D_io_variables(None, dofs_idx, envs_idx)
        else:
            tensor, dofs_idx = self._validate_1D_io_variables(None, dofs_idx, batched=False)
            envs_idx = torch.empty(())

        if name == "kp":
            self._kernel_get_dofs_kp(tensor, dofs_idx, envs_idx)
            return tensor

        elif name == "kv":
            self._kernel_get_dofs_kv(tensor, dofs_idx, envs_idx)
            return tensor

        elif name == "force_range":
            lower = torch.empty_like(tensor)
            upper = torch.empty_like(tensor)
            self._kernel_get_dofs_force_range(lower, upper, dofs_idx, envs_idx)
            return lower, upper

        elif name == "limit":
            lower = torch.empty_like(tensor)
            upper = torch.empty_like(tensor)
            self._kernel_get_dofs_limit(lower, upper, dofs_idx, envs_idx)
            return lower, upper

        elif name == "stiffness":
            self._kernel_get_dofs_stiffness(tensor, dofs_idx, envs_idx)
            return tensor

        elif name == "invweight":
            self._kernel_get_dofs_invweight(tensor, dofs_idx, envs_idx)
            return tensor

        elif name == "armature":
            self._kernel_get_dofs_armature(tensor, dofs_idx, envs_idx)
            return tensor

        elif name == "damping":
            self._kernel_get_dofs_damping(tensor, dofs_idx, envs_idx)
            return tensor

        else:
            gs.raise_exception()

    @ti.kernel
    def _kernel_get_dofs_kp(
        self,
        tensor: ti.types.ndarray(),
        dofs_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        if ti.static(self._options.batch_dofs_info):
            for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
                tensor[i_b_, i_d_] = self.dofs_info[dofs_idx[i_d_], envs_idx[i_b_]].kp
        else:
            for i_d_ in range(dofs_idx.shape[0]):
                tensor[i_d_] = self.dofs_info[dofs_idx[i_d_]].kp

    @ti.kernel
    def _kernel_get_dofs_kv(
        self,
        tensor: ti.types.ndarray(),
        dofs_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        if ti.static(self._options.batch_dofs_info):
            for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
                tensor[i_b_, i_d_] = self.dofs_info[dofs_idx[i_d_], envs_idx[i_b_]].kv
        else:
            for i_d_ in range(dofs_idx.shape[0]):
                tensor[i_d_] = self.dofs_info[dofs_idx[i_d_]].kv

    @ti.kernel
    def _kernel_get_dofs_force_range(
        self,
        lower: ti.types.ndarray(),
        upper: ti.types.ndarray(),
        dofs_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        if ti.static(self._options.batch_dofs_info):
            for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
                lower[i_b_, i_d_] = self.dofs_info[dofs_idx[i_d_], envs_idx[i_b_]].force_range[0]
                upper[i_b_, i_d_] = self.dofs_info[dofs_idx[i_d_], envs_idx[i_b_]].force_range[1]
        else:
            for i_d_ in range(dofs_idx.shape[0]):
                lower[i_d_] = self.dofs_info[dofs_idx[i_d_]].force_range[0]
                upper[i_d_] = self.dofs_info[dofs_idx[i_d_]].force_range[1]

    @ti.kernel
    def _kernel_get_dofs_limit(
        self,
        lower: ti.types.ndarray(),
        upper: ti.types.ndarray(),
        dofs_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        if ti.static(self._options.batch_dofs_info):
            for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
                lower[i_b_, i_d_] = self.dofs_info[dofs_idx[i_d_], envs_idx[i_b_]].limit[0]
                upper[i_b_, i_d_] = self.dofs_info[dofs_idx[i_d_], envs_idx[i_b_]].limit[1]
        else:
            for i_d_ in range(dofs_idx.shape[0]):
                lower[i_d_] = self.dofs_info[dofs_idx[i_d_]].limit[0]
                upper[i_d_] = self.dofs_info[dofs_idx[i_d_]].limit[1]

    @ti.kernel
    def _kernel_get_dofs_stiffness(
        self,
        tensor: ti.types.ndarray(),
        dofs_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        if ti.static(self._options.batch_dofs_info):
            for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
                tensor[i_b_, i_d_] = self.dofs_info[dofs_idx[i_d_], envs_idx[i_b_]].stiffness
        else:
            for i_d_ in range(dofs_idx.shape[0]):
                tensor[i_d_] = self.dofs_info[dofs_idx[i_d_]].stiffness

    @ti.kernel
    def _kernel_get_dofs_invweight(
        self,
        tensor: ti.types.ndarray(),
        dofs_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        if ti.static(self._options.batch_dofs_info):
            for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
                tensor[i_b_, i_d_] = self.dofs_info[dofs_idx[i_d_], envs_idx[i_b_]].invweight
        else:
            for i_d_ in range(dofs_idx.shape[0]):
                tensor[i_d_] = self.dofs_info[dofs_idx[i_d_]].invweight

    @ti.kernel
    def _kernel_get_dofs_armature(
        self,
        tensor: ti.types.ndarray(),
        dofs_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        if ti.static(self._options.batch_dofs_info):
            for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
                tensor[i_b_, i_d_] = self.dofs_info[dofs_idx[i_d_], envs_idx[i_b_]].armature
        else:
            for i_d_ in range(dofs_idx.shape[0]):
                tensor[i_d_] = self.dofs_info[dofs_idx[i_d_]].armature

    @ti.kernel
    def _kernel_get_dofs_damping(
        self,
        tensor: ti.types.ndarray(),
        dofs_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        if ti.static(self._options.batch_dofs_info):
            for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
                tensor[i_b_, i_d_] = self.dofs_info[dofs_idx[i_d_], envs_idx[i_b_]].damping
        else:
            for i_d_ in range(dofs_idx.shape[0]):
                tensor[i_d_] = self.dofs_info[dofs_idx[i_d_]].damping

    @ti.kernel
    def _kernel_set_drone_rpm(
        self,
        n_propellers: ti.i32,
        COM_link_idx: ti.i32,
        propellers_link_idxs: ti.types.ndarray(),
        propellers_rpm: ti.types.ndarray(),
        propellers_spin: ti.types.ndarray(),
        KF: ti.float32,
        KM: ti.float32,
        invert: ti.i32,
    ):
        """
        Set the RPM of propellers of a drone entity.
        Should only be called by drone entities.
        """
        for b in range(self._B):
            torque = 0.0
            for i in range(n_propellers):
                force_i = propellers_rpm[i, b] ** 2 * KF
                torque += propellers_rpm[i, b] ** 2 * KM * propellers_spin[i]
                self._func_apply_external_force_link_inertial_frame(
                    ti.Vector([0.0, 0.0, 0.0]), ti.Vector([0.0, 0.0, force_i]), propellers_link_idxs[i], b
                )

            if invert:
                torque = -torque

            self._func_apply_external_torque_link_inertial_frame(ti.Vector([0.0, 0.0, torque]), COM_link_idx, b)

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
                gu.ti_rotvec_to_quat(ti.Vector([0.0, 0.0, rad])),
                self.vgeoms_state[propellers_vgeom_idxs[i], b].quat,
            )

    def get_geoms_friction(self, geoms_idx):
        tensor, geoms_idx = self._validate_1D_io_variables(None, geoms_idx, batched=False, idx_name="geoms_idx")

        self._kernel_get_geoms_friction(tensor, geoms_idx)

        if self.n_envs == 0:
            tensor = tensor.squeeze(0)
        return tensor

    @ti.kernel
    def _kernel_get_geoms_friction(
        self,
        tensor: ti.types.ndarray(),
        geoms_idx: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._para_level < gs.PARA_LEVEL.PARTIAL)
        for i_g_ in ti.ndrange(geoms_idx.shape[0]):
            tensor[i_g_] = self.geoms_info[geoms_idx[i_g_]].friction

    def set_geom_friction(self, friction, geoms_idx):
        self._kernel_set_geom_friction(geoms_idx, friction)

    @ti.kernel
    def _kernel_set_geom_friction(self, geoms_idx: ti.i32, friction: ti.f32):
        self.geoms_info[geoms_idx].friction = friction

    def set_geoms_friction(self, friction, geoms_idx):
        friction, geoms_idx = self._validate_1D_io_variables(friction, geoms_idx, batched=False, idx_name="geoms_idx")

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

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def links(self):
        if self.is_built:
            return self._links
        else:
            links = gs.List()
            for entity in self._entities:
                links += entity.links
            return links

    @property
    def joints(self):
        if self.is_built:
            return self._joints
        else:
            joints = gs.List()
            for entity in self._entities:
                joints += entity.joints
            return joints

    @property
    def geoms(self):
        if self.is_built:
            return self._geoms
        else:
            geoms = gs.List()
            for entity in self._entities:
                geoms += entity.geoms
            return geoms

    @property
    def vgeoms(self):
        if self.is_built:
            return self._vgeoms
        else:
            vgeoms = gs.List()
            for entity in self._entities:
                vgeoms += entity.vgeoms
            return vgeoms

    @property
    def n_links(self):
        if self.is_built:
            return self._n_links
        else:
            return len(self.links)

    @property
    def n_joints(self):
        if self.is_built:
            return self._n_joints
        else:
            return len(self.joints)

    @property
    def n_geoms(self):
        if self.is_built:
            return self._n_geoms
        else:
            return len(self.geoms)

    @property
    def n_cells(self):
        if self.is_built:
            return self._n_cells
        else:
            return sum([entity.n_cells for entity in self._entities])

    @property
    def n_vgeoms(self):
        if self.is_built:
            return self._n_vgeoms
        else:
            return len(self.vgeoms)

    @property
    def n_verts(self):
        if self.is_built:
            return self._n_verts
        else:
            return sum([entity.n_verts for entity in self._entities])

    @property
    def n_free_verts(self):
        if self.is_built:
            return self._n_free_verts
        else:
            return sum([entity.n_verts if entity.is_free else 0 for entity in self._entities])

    @property
    def n_fixed_verts(self):
        if self.is_built:
            return self._n_fixed_verts
        else:
            return sum([entity.n_verts if not entity.is_free else 0 for entity in self._entities])

    @property
    def n_vverts(self):
        if self.is_built:
            return self._n_vverts
        else:
            return sum([entity.n_vverts for entity in self._entities])

    @property
    def n_faces(self):
        if self.is_built:
            return self._n_faces
        else:
            return sum([entity.n_faces for entity in self._entities])

    @property
    def n_vfaces(self):
        if self.is_built:
            return self._n_vfaces
        else:
            return sum([entity.n_vfaces for entity in self._entities])

    @property
    def n_edges(self):
        if self.is_built:
            return self._n_edges
        else:
            return sum([entity.n_edges for entity in self._entities])

    @property
    def n_qs(self):
        if self.is_built:
            return self._n_qs
        else:
            return sum([entity.n_qs for entity in self._entities])

    @property
    def n_dofs(self):
        if self.is_built:
            return self._n_dofs
        else:
            return sum([entity.n_dofs for entity in self._entities])

    @property
    def init_qpos(self):
        if len(self._entities) == 0:
            return np.array([])
        else:
            return np.concatenate([entity.init_qpos for entity in self._entities])

    @property
    def max_collision_pairs(self):
        return self._max_collision_pairs

    @property
    def n_equalities(self):
        if self.is_built:
            return self._n_equalities
        else:
            return sum([entity.n_equalities for entity in self._entities])

    @property
    def equalities(self):
        if self.is_built:
            return self._equalities
        else:
            equalities = gs.List()
            for entity in self._entities:
                equalities += entity.equalities
            return equalities
