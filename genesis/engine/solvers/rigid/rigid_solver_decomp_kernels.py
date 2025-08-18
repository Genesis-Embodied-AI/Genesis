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
import gstaichi as ti
from .... import maybe_pure


@maybe_pure
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


@maybe_pure
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


@maybe_pure
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


@maybe_pure
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
    # we will use RigidGlobalInfo as typing after Hugh adds array_struct feature to gstaichi
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


@maybe_pure
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


@maybe_pure
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


@maybe_pure
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


@maybe_pure
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


@maybe_pure
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


@maybe_pure
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


@maybe_pure
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


@maybe_pure
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


@maybe_pure
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


@maybe_pure
@ti.kernel
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
    contact_island_state: array_class.ContactIslandState,  # ContactIsland
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


@maybe_pure
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


@maybe_pure
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
                rigid_global_info.qpos[i_q + rigid_entity._q_start, i_b] = init_qpos[i_b_, i_q]

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


@maybe_pure
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


@maybe_pure
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


@maybe_pure
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
    contact_island_state: array_class.ContactIslandState,  # ContactIsland
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
            static_rigid_sim_config=static_rigid_sim_config,
            contact_island_state=contact_island_state,
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


@maybe_pure
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


@maybe_pure
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


@maybe_pure
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


@maybe_pure
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


@maybe_pure
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


@maybe_pure
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
def func_hibernate__for_all_awake_islands_either_hiberanate_or_update_aabb_sort_buffer(
    dofs_state: array_class.DofsState,
    entities_state: array_class.EntitiesState,
    entities_info: array_class.EntitiesInfo,
    links_state: array_class.LinksState,
    geoms_state: array_class.GeomsState,
    collider_state: array_class.ColliderState,
    unused__rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    contact_island_state: array_class.ContactIslandState,  # ContactIsland,
) -> None:

    n_entities = entities_state.hibernated.shape[0]
    _B = entities_state.hibernated.shape[1]

    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_b in range(_B):
        for island_idx in range(ci.n_islands[i_b]):
            was_island_hibernated = ci.island_hibernated[island_idx, i_b]

            if not was_island_hibernated:
                are_all_entities_okay_for_hibernation = True
                entity_ref_range = ci.island_entity[island_idx, i_b]
                for i_entity_ref_offset_ in range(entity_ref_range.n):
                    entity_ref = entity_ref_range.start + i_entity_ref_offset_
                    entity_idx = ci.entity_id[entity_ref, i_b]

                    # Hibernated entities already have zero dofs_state.acc/vel
                    is_entity_hibernated = entities_state.hibernated[entity_idx, i_b]
                    if is_entity_hibernated:
                        continue

                    for i_d in range(entities_info.dof_start[entity_idx], entities_info.dof_end[entity_idx]):
                        max_acc = static_rigid_sim_config.hibernation_thresh_acc
                        max_vel = static_rigid_sim_config.hibernation_thresh_vel
                        if ti.abs(dofs_state.acc[i_d, i_b]) > max_acc or ti.abs(dofs_state.vel[i_d, i_b]) > max_vel:
                            are_all_entities_okay_for_hibernation = False
                            break

                    if not are_all_entities_okay_for_hibernation:
                        break

                if not are_all_entities_okay_for_hibernation:
                    # update collider sort_buffer with aabb extents along x-axis
                    for i_entity_ref_offset_ in range(entity_ref_range.n):
                        entity_ref = entity_ref_range.start + i_entity_ref_offset_
                        entity_idx = ci.entity_id[entity_ref, i_b]
                        for i_g in range(entities_info.geom_start[entity_idx], entities_info.geom_end[entity_idx]):
                            min_idx, min_val = geoms_state.min_buffer_idx[i_g, i_b], geoms_state.aabb_min[i_g, i_b][0]
                            max_idx, max_val = geoms_state.max_buffer_idx[i_g, i_b], geoms_state.aabb_max[i_g, i_b][0]
                            collider_state.sort_buffer.value[min_idx, i_b] = min_val
                            collider_state.sort_buffer.value[max_idx, i_b] = max_val
                else:
                    # perform hibernation
                    prev_entity_ref = entity_ref_range.start + entity_ref_range.n - 1
                    prev_entity_idx = ci.entity_id[prev_entity_ref, i_b]

                    for i_entity_ref_offset_ in range(entity_ref_range.n):
                        entity_ref = entity_ref_range.start + i_entity_ref_offset_
                        entity_idx = ci.entity_id[entity_ref, i_b]

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
                        ci.entity_idx_to_next_entity_idx_in_hibernated_island[prev_entity_idx, i_b] = entity_idx
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
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_e, i_b in ti.ndrange(n_entities, _B):
        if entities_state.hibernated[i_e, i_b] or entities_info.n_dofs[i_e] == 0:
            continue

        next_awake_entity_idx = ti.atomic_add(rigid_global_info.n_awake_entities[i_b], 1)
        rigid_global_info.awake_entities[next_awake_entity_idx, i_b] = i_e

        n_dofs = entities_info.n_dofs[i_e]
        entity_dofs_base_idx: ti.int32 = entities_info.dof_start[i_e]
        awake_dofs_base_idx = ti.atomic_add(rigid_global_info.n_awake_dofs[i_b], n_dofs)
        for i in range(n_dofs):
            rigid_global_info.awake_dofs[awake_dofs_base_idx + i, i_b] = entity_dofs_base_idx + i

        n_links = entities_info.n_links[i_e]
        entity_links_base_idx: ti.int32 = entities_info.link_start[i_e]
        awake_links_base_idx = ti.atomic_add(rigid_global_info.n_awake_links[i_b], n_links)
        for i in range(n_links):
            rigid_global_info.awake_links[awake_links_base_idx + i, i_b] = entity_links_base_idx + i


@ti.func
def func_hibernate_entity_and_zero_dof_velocities(
    i_e: int,
    i_b: int,
    entities_state: array_class.EntitiesState,
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    links_state: array_class.LinksState,
    geoms_state: array_class.GeomsState,
) -> None:
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


@maybe_pure
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


@maybe_pure
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

        if ti.static(static_rigid_sim_config.use_hibernation) and entities_state.hibernated[i_e, i_b] and wakeup:
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

    if ti.static(static_rigid_sim_config.use_hibernation):
        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(_B):
            for i_d_ in range(rigid_global_info.n_awake_dofs[i_b]):
                i_d = rigid_global_info.awake_dofs[i_d_, i_b]
                I_d = [i_d, i_b] if ti.static(static_rigid_sim_config.batch_dofs_info) else i_d

                dofs_state.qf_passive[i_d, i_b] = -dofs_info.damping[I_d] * dofs_state.vel[i_d, i_b]

        ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(_B):
            for i_l_ in range(rigid_global_info.n_awake_links[i_b]):
                i_l = rigid_global_info.awake_links[i_l_, i_b]
                I_l = [i_l, i_b] if ti.static(static_rigid_sim_config.batch_links_info) else i_l
                if links_info.n_dofs[I_l] == 0:
                    continue

                i_j = links_info.joint_start[I_l]
                I_j = [i_j, i_b] if ti.static(static_rigid_sim_config.batch_joints_info) else i_j
                joint_type = joints_info.type[I_j]

                if joint_type != gs.JOINT_TYPE.FREE and joint_type != gs.JOINT_TYPE.FIXED:
                    q_start = links_info.q_start[I_l]
                    dof_start = links_info.dof_start[I_l]
                    dof_end = links_info.dof_end[I_l]

                    for j_d in range(dof_end - dof_start):
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
                q_start = links_info.q_start[I_l]
                dof_start = links_info.dof_start[I_l]
                dof_end = links_info.dof_end[I_l]

                for j_d in range(dof_end - dof_start):
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
                for i in range(entities_info.n_links[i_e]):
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
        for i_b in range(_B):
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
                    I_j = [i_j, i_b] if ti.static(static_rigid_sim_config.batch_joints_info) else i_j
                    dof_start = joints_info.dof_start[I_j]
                    q_start = joints_info.q_start[I_j]
                    q_end = joints_info.q_end[I_j]

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


@maybe_pure
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


@maybe_pure
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


@maybe_pure
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


@maybe_pure
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


@maybe_pure
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


@maybe_pure
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


@maybe_pure
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


@maybe_pure
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


@maybe_pure
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


@maybe_pure
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


@maybe_pure
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


@maybe_pure
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


@maybe_pure
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


@maybe_pure
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


@maybe_pure
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


@maybe_pure
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


@maybe_pure
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


@maybe_pure
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


@maybe_pure
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


@maybe_pure
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


@maybe_pure
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


@maybe_pure
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


@maybe_pure
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


@maybe_pure
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


@maybe_pure
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


@maybe_pure
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


@maybe_pure
@ti.kernel
def kernel_control_dofs_velocity(
    velocity: ti.types.ndarray(),
    dofs_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    static_rigid_sim_config: ti.template(),
) -> ti.i32:
    has_gains = gs.ti_bool(False)
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
        i_d = dofs_idx[i_d_]
        i_b = envs_idx[i_b_]

        I_d = [i_d, i_b] if ti.static(static_rigid_sim_config.batch_dofs_info) else i_d
        dofs_state.ctrl_mode[i_d, i_b] = gs.CTRL_MODE.VELOCITY
        dofs_state.ctrl_vel[i_d, i_b] = velocity[i_b_, i_d_]
        if (dofs_info.kp[I_d] > gs.EPS) | (dofs_info.kv[I_d] > gs.EPS):
            has_gains = True
    return has_gains


@maybe_pure
@ti.kernel
def kernel_control_dofs_position(
    position: ti.types.ndarray(),
    dofs_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    dofs_state: array_class.DofsState,
    dofs_info: array_class.DofsInfo,
    static_rigid_sim_config: ti.template(),
) -> ti.i32:
    has_gains = gs.ti_bool(False)
    ti.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_d_, i_b_ in ti.ndrange(dofs_idx.shape[0], envs_idx.shape[0]):
        i_d = dofs_idx[i_d_]
        i_b = envs_idx[i_b_]

        I_d = [i_d, i_b] if ti.static(static_rigid_sim_config.batch_dofs_info) else i_d
        dofs_state.ctrl_mode[i_d, i_b] = gs.CTRL_MODE.POSITION
        dofs_state.ctrl_pos[i_d, i_b] = position[i_b_, i_d_]
        if (dofs_info.kp[I_d] > gs.EPS) | (dofs_info.kv[I_d] > gs.EPS):
            has_gains = True
    return has_gains


@maybe_pure
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


@maybe_pure
@ti.kernel
def kernel_get_links_acc(
    tensor: ti.types.ndarray(),
    links_idx: ti.types.ndarray(),
    envs_idx: ti.types.ndarray(),
    links_state: array_class.LinksState,
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

        for i in ti.static(range(3)):
            tensor[i_b_, i_l_, i] = acc_classic_lin[i]


@maybe_pure
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


@maybe_pure
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


@maybe_pure
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


@maybe_pure
@ti.kernel
def kernel_set_geom_friction(geoms_idx: ti.i32, friction: ti.f32, geoms_info: array_class.GeomsInfo):
    geoms_info.friction[geoms_idx] = friction


@maybe_pure
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
