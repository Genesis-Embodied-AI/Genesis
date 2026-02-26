"""
Narrow-phase collision detection functions.

This module contains SDF-based contact detection, convex-convex contact,
terrain detection, box-box contact, and multi-contact search algorithms.
"""

import sys
from enum import IntEnum

import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
import genesis.utils.sdf as sdf

from . import capsule_contact, diff_gjk, gjk, mpr
from .box_contact import (
    func_box_box_contact,
    func_plane_box_contact,
)
from .contact import (
    func_add_contact,
    func_add_diff_contact_input,
    func_compute_tolerance,
    func_contact_orthogonals,
    func_rotate_frame,
    func_set_contact,
)
from .utils import func_point_in_geom_aabb


class CCD_ALGORITHM_CODE(IntEnum):
    """Convex collision detection algorithm codes."""

    # Our MPR (with SDF)
    MPR = 0
    # MuJoCo MPR
    MJ_MPR = 1
    # Our GJK
    GJK = 2
    # MuJoCo GJK
    MJ_GJK = 3


@qd.func
def func_contact_sphere_sdf(
    i_ga,
    i_gb,
    i_b,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    collider_static_config: qd.template(),
    sdf_info: array_class.SDFInfo,
):
    is_col = False
    penetration = gs.qd_float(0.0)
    normal = qd.Vector.zero(gs.qd_float, 3)
    contact_pos = qd.Vector.zero(gs.qd_float, 3)

    sphere_center = geoms_state.pos[i_ga, i_b]
    sphere_radius = geoms_info.data[i_ga][0]

    center_to_b_dist = sdf.sdf_func_world(geoms_state, geoms_info, sdf_info, sphere_center, i_gb, i_b)
    if center_to_b_dist < sphere_radius:
        is_col = True
        normal = sdf.sdf_func_normal_world(
            geoms_state, geoms_info, rigid_global_info, collider_static_config, sdf_info, sphere_center, i_gb, i_b
        )
        penetration = sphere_radius - center_to_b_dist
        contact_pos = sphere_center - (sphere_radius - 0.5 * penetration) * normal

    return is_col, normal, penetration, contact_pos


@qd.func
def func_contact_vertex_sdf(
    i_ga,
    i_gb,
    i_b,
    ga_pos: qd.types.vector(3, dtype=gs.qd_float),
    ga_quat: qd.types.vector(4, dtype=gs.qd_float),
    gb_pos: qd.types.vector(3, dtype=gs.qd_float),
    gb_quat: qd.types.vector(4, dtype=gs.qd_float),
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    collider_static_config: qd.template(),
    sdf_info: array_class.SDFInfo,
):
    is_col = False
    penetration = gs.qd_float(0.0)
    normal = qd.Vector.zero(gs.qd_float, 3)
    contact_pos = qd.Vector.zero(gs.qd_float, 3)

    for i_v in range(geoms_info.vert_start[i_ga], geoms_info.vert_end[i_ga]):
        vertex_pos = gu.qd_transform_by_trans_quat(verts_info.init_pos[i_v], ga_pos, ga_quat)
        if func_point_in_geom_aabb(geoms_state, i_gb, i_b, vertex_pos):
            new_penetration = -sdf.sdf_func_world_local(geoms_info, sdf_info, vertex_pos, i_gb, gb_pos, gb_quat)
            if new_penetration > penetration:
                is_col = True
                contact_pos = vertex_pos
                penetration = new_penetration

    if is_col:
        # Compute contact normal only once, and only in case of contact
        normal = sdf.sdf_func_normal_world_local(
            geoms_info, rigid_global_info, collider_static_config, sdf_info, contact_pos, i_gb, gb_pos, gb_quat
        )

        # The contact point must be offsetted by half the penetration depth
        contact_pos = contact_pos + 0.5 * penetration * normal

    return is_col, normal, penetration, contact_pos


@qd.func
def func_contact_edge_sdf(
    i_ga,
    i_gb,
    i_b,
    ga_pos: qd.types.vector(3, dtype=gs.qd_float),
    ga_quat: qd.types.vector(4, dtype=gs.qd_float),
    gb_pos: qd.types.vector(3, dtype=gs.qd_float),
    gb_quat: qd.types.vector(4, dtype=gs.qd_float),
    geoms_state: array_class.GeomsState,  # For AABB only
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    edges_info: array_class.EdgesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    collider_static_config: qd.template(),
    sdf_info: array_class.SDFInfo,
):
    EPS = rigid_global_info.EPS[None]

    is_col = False
    penetration = gs.qd_float(0.0)
    normal = qd.Vector.zero(gs.qd_float, 3)
    contact_pos = qd.Vector.zero(gs.qd_float, 3)

    ga_sdf_cell_size = sdf_info.geoms_info.sdf_cell_size[i_ga]

    for i_e in range(geoms_info.edge_start[i_ga], geoms_info.edge_end[i_ga]):
        cur_length = edges_info.length[i_e]
        if cur_length > ga_sdf_cell_size:
            i_v0 = edges_info.v0[i_e]
            i_v1 = edges_info.v1[i_e]

            p_0 = gu.qd_transform_by_trans_quat(verts_info.init_pos[i_v0], ga_pos, ga_quat)
            p_1 = gu.qd_transform_by_trans_quat(verts_info.init_pos[i_v1], ga_pos, ga_quat)
            vec_01 = gu.qd_normalize(p_1 - p_0, EPS)

            sdf_grad_0_b = sdf.sdf_func_grad_world_local(
                geoms_info, rigid_global_info, collider_static_config, sdf_info, p_0, i_gb, gb_pos, gb_quat
            )
            sdf_grad_1_b = sdf.sdf_func_grad_world_local(
                geoms_info, rigid_global_info, collider_static_config, sdf_info, p_1, i_gb, gb_pos, gb_quat
            )

            # check if the edge on a is facing towards mesh b
            sdf_grad_0_a = sdf.sdf_func_grad_world_local(
                geoms_info, rigid_global_info, collider_static_config, sdf_info, p_0, i_ga, ga_pos, ga_quat
            )
            sdf_grad_1_a = sdf.sdf_func_grad_world_local(
                geoms_info, rigid_global_info, collider_static_config, sdf_info, p_1, i_ga, ga_pos, ga_quat
            )
            normal_edge_0 = sdf_grad_0_a - sdf_grad_0_a.dot(vec_01) * vec_01
            normal_edge_1 = sdf_grad_1_a - sdf_grad_1_a.dot(vec_01) * vec_01

            if normal_edge_0.dot(sdf_grad_0_b) < 0 or normal_edge_1.dot(sdf_grad_1_b) < 0:
                # check if closest point is between the two points
                if sdf_grad_0_b.dot(vec_01) < 0 and sdf_grad_1_b.dot(vec_01) > 0:
                    while cur_length > ga_sdf_cell_size:
                        p_mid = 0.5 * (p_0 + p_1)
                        if (
                            sdf.sdf_func_grad_world_local(
                                geoms_info,
                                rigid_global_info,
                                collider_static_config,
                                sdf_info,
                                p_mid,
                                i_gb,
                                gb_pos,
                                gb_quat,
                            ).dot(vec_01)
                            < 0
                        ):
                            p_0 = p_mid
                        else:
                            p_1 = p_mid
                        cur_length = 0.5 * cur_length

                    p = 0.5 * (p_0 + p_1)
                    new_penetration = -sdf.sdf_func_world_local(geoms_info, sdf_info, p, i_gb, gb_pos, gb_quat)

                    if new_penetration > penetration:
                        is_col = True
                        normal = sdf.sdf_func_normal_world_local(
                            geoms_info, rigid_global_info, collider_static_config, sdf_info, p, i_gb, gb_pos, gb_quat
                        )
                        contact_pos = p
                        penetration = new_penetration

    # The contact point must be offsetted by half the penetration depth, for consistency with MPR
    contact_pos = contact_pos + 0.5 * penetration * normal

    return is_col, normal, penetration, contact_pos


@qd.func
def func_contact_convex_convex_sdf(
    i_ga,
    i_gb,
    i_b,
    i_va_ws,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    collider_info: array_class.ColliderInfo,
    collider_static_config: qd.template(),
    sdf_info: array_class.SDFInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    enable_edge_detection_fallback: qd.template(),
):
    EPS = rigid_global_info.EPS[None]

    gb_vert_start = geoms_info.vert_start[i_gb]
    ga_pos = geoms_state.pos[i_ga, i_b]
    ga_quat = geoms_state.quat[i_ga, i_b]
    gb_pos = geoms_state.pos[i_gb, i_b]
    gb_quat = geoms_state.quat[i_gb, i_b]

    is_col = False
    penetration = gs.qd_float(0.0)
    normal = qd.Vector.zero(gs.qd_float, 3)
    contact_pos = qd.Vector.zero(gs.qd_float, 3)

    i_va = i_va_ws
    if i_va == -1:
        # start traversing on the vertex graph with a smart initial vertex
        pos_vb = gu.qd_transform_by_trans_quat(verts_info.init_pos[gb_vert_start], gb_pos, gb_quat)
        i_va = sdf.sdf_func_find_closest_vert(geoms_state, geoms_info, sdf_info, pos_vb, i_ga, i_b)
    i_v_closest = i_va
    pos_v_closest = gu.qd_transform_by_trans_quat(verts_info.init_pos[i_v_closest], ga_pos, ga_quat)
    sd_v_closest = sdf.sdf_func_world(geoms_state, geoms_info, sdf_info, pos_v_closest, i_gb, i_b)

    while True:
        for i_neighbor_ in range(
            collider_info.vert_neighbor_start[i_va],
            collider_info.vert_neighbor_start[i_va] + collider_info.vert_n_neighbors[i_va],
        ):
            i_neighbor = collider_info.vert_neighbors[i_neighbor_]
            pos_neighbor = gu.qd_transform_by_trans_quat(verts_info.init_pos[i_neighbor], ga_pos, ga_quat)
            sd_neighbor = sdf.sdf_func_world(geoms_state, geoms_info, sdf_info, pos_neighbor, i_gb, i_b)
            if sd_neighbor < sd_v_closest - 1e-5:  # 1e-5 (0.01mm) to avoid endless loop due to numerical instability
                i_v_closest = i_neighbor
                sd_v_closest = sd_neighbor
                pos_v_closest = pos_neighbor

        if i_v_closest == i_va:  # no better neighbor
            break
        else:
            i_va = i_v_closest

    # i_va is the deepest vertex
    pos_a = pos_v_closest
    if sd_v_closest < 0.0:
        is_col = True
        normal = sdf.sdf_func_normal_world(
            geoms_state, geoms_info, rigid_global_info, collider_static_config, sdf_info, pos_a, i_gb, i_b
        )
        penetration = -sd_v_closest
        contact_pos = pos_a + 0.5 * penetration * normal
    elif enable_edge_detection_fallback:  # check edge surrounding it
        for i_neighbor_ in range(
            collider_info.vert_neighbor_start[i_va],
            collider_info.vert_neighbor_start[i_va] + collider_info.vert_n_neighbors[i_va],
        ):
            i_neighbor = collider_info.vert_neighbors[i_neighbor_]

            p_0 = pos_v_closest
            p_1 = gu.qd_transform_by_trans_quat(verts_info.init_pos[i_neighbor], ga_pos, ga_quat)
            vec_01 = gu.qd_normalize(p_1 - p_0, EPS)

            sdf_grad_0_b = sdf.sdf_func_grad_world(
                geoms_state, geoms_info, rigid_global_info, collider_static_config, sdf_info, p_0, i_gb, i_b
            )
            sdf_grad_1_b = sdf.sdf_func_grad_world(
                geoms_state, geoms_info, rigid_global_info, collider_static_config, sdf_info, p_1, i_gb, i_b
            )

            # check if the edge on a is facing towards mesh b (I am not 100% sure about this, subject to removal)
            sdf_grad_0_a = sdf.sdf_func_grad_world(
                geoms_state, geoms_info, rigid_global_info, collider_static_config, sdf_info, p_0, i_ga, i_b
            )
            sdf_grad_1_a = sdf.sdf_func_grad_world(
                geoms_state, geoms_info, rigid_global_info, collider_static_config, sdf_info, p_1, i_ga, i_b
            )
            normal_edge_0 = sdf_grad_0_a - sdf_grad_0_a.dot(vec_01) * vec_01
            normal_edge_1 = sdf_grad_1_a - sdf_grad_1_a.dot(vec_01) * vec_01

            if normal_edge_0.dot(sdf_grad_0_b) < 0 or normal_edge_1.dot(sdf_grad_1_b) < 0:
                # check if closest point is between the two points
                if sdf_grad_0_b.dot(vec_01) < 0 and sdf_grad_1_b.dot(vec_01) > 0:
                    cur_length = (p_1 - p_0).norm()
                    ga_sdf_cell_size = sdf_info.geoms_info.sdf_cell_size[i_ga]
                    while cur_length > ga_sdf_cell_size:
                        p_mid = 0.5 * (p_0 + p_1)
                        side = sdf.sdf_func_grad_world(
                            geoms_state,
                            geoms_info,
                            rigid_global_info,
                            collider_static_config,
                            sdf_info,
                            p_mid,
                            i_gb,
                            i_b,
                        ).dot(vec_01)
                        if side < 0:
                            p_0 = p_mid
                        else:
                            p_1 = p_mid

                        cur_length = 0.5 * cur_length

                    p = 0.5 * (p_0 + p_1)

                    new_penetration = -sdf.sdf_func_world(geoms_state, geoms_info, sdf_info, p, i_gb, i_b)

                    if new_penetration > 0.0:
                        is_col = True
                        normal = sdf.sdf_func_normal_world(
                            geoms_state, geoms_info, rigid_global_info, collider_static_config, sdf_info, p, i_gb, i_b
                        )
                        contact_pos = p
                        penetration = new_penetration
                        break

    return is_col, normal, penetration, contact_pos, i_va


@qd.func
def func_contact_mpr_terrain(
    i_ga,
    i_gb,
    i_b,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    geoms_init_AABB: array_class.GeomsInitAABB,
    static_rigid_sim_config: qd.template(),
    collider_state: array_class.ColliderState,
    collider_info: array_class.ColliderInfo,
    collider_static_config: qd.template(),
    mpr_state: array_class.MPRState,
    mpr_info: array_class.MPRInfo,
    support_field_info: array_class.SupportFieldInfo,
    errno: array_class.V_ANNOTATION,
):
    ga_pos, ga_quat = geoms_state.pos[i_ga, i_b], geoms_state.quat[i_ga, i_b]
    gb_pos, gb_quat = geoms_state.pos[i_gb, i_b], geoms_state.quat[i_gb, i_b]
    margin = gs.qd_float(0.0)

    is_return = False
    tolerance = func_compute_tolerance(i_ga, i_gb, i_b, collider_info.mc_tolerance[None], geoms_info, geoms_init_AABB)

    if not is_return:
        # Transform to terrain's frame (using local variables, not modifying global state)
        ga_pos_terrain_frame, ga_quat_terrain_frame = gu.qd_transform_pos_quat_by_trans_quat(
            ga_pos - gb_pos,
            ga_quat,
            qd.Vector.zero(gs.qd_float, 3),
            gu.qd_inv_quat(gb_quat),
        )
        gb_pos_terrain_frame = qd.Vector.zero(gs.qd_float, 3)
        gb_quat_terrain_frame = gu.qd_identity_quat()
        center_a = gu.qd_transform_by_trans_quat(geoms_info.center[i_ga], ga_pos_terrain_frame, ga_quat_terrain_frame)

        for i_axis, i_m in qd.ndrange(3, 2):
            direction = qd.Vector.zero(gs.qd_float, 3)
            if i_m == 0:
                direction[i_axis] = 1.0
            else:
                direction[i_axis] = -1.0
            v1 = mpr.support_driver(
                geoms_info,
                collider_state,
                collider_static_config,
                support_field_info,
                direction,
                i_ga,
                i_b,
                ga_pos_terrain_frame,
                ga_quat_terrain_frame,
            )
            collider_state.xyz_max_min[3 * i_m + i_axis, i_b] = v1[i_axis]

        for i in qd.static(range(3)):
            collider_state.prism[i, i_b][2] = collider_info.terrain_xyz_maxmin[5]

            if (
                collider_info.terrain_xyz_maxmin[i] < collider_state.xyz_max_min[i + 3, i_b] - margin
                or collider_info.terrain_xyz_maxmin[i + 3] > collider_state.xyz_max_min[i, i_b] + margin
            ):
                is_return = True

        if not is_return:
            sh = collider_info.terrain_scale[0]
            r_min = gs.qd_int(qd.floor((collider_state.xyz_max_min[3, i_b] - collider_info.terrain_xyz_maxmin[3]) / sh))
            r_max = gs.qd_int(qd.ceil((collider_state.xyz_max_min[0, i_b] - collider_info.terrain_xyz_maxmin[3]) / sh))
            c_min = gs.qd_int(qd.floor((collider_state.xyz_max_min[4, i_b] - collider_info.terrain_xyz_maxmin[4]) / sh))
            c_max = gs.qd_int(qd.ceil((collider_state.xyz_max_min[1, i_b] - collider_info.terrain_xyz_maxmin[4]) / sh))

            r_min = qd.max(0, r_min)
            c_min = qd.max(0, c_min)
            r_max = qd.min(collider_info.terrain_rc[0] - 1, r_max)
            c_max = qd.min(collider_info.terrain_rc[1] - 1, c_max)

            n_con = 0
            for r in range(r_min, r_max):
                nvert = 0
                for c in range(c_min, c_max + 1):
                    for i in range(2):
                        if n_con < qd.static(collider_static_config.n_contacts_per_pair):
                            nvert = nvert + 1
                            func_add_prism_vert(
                                sh * (r + i) + collider_info.terrain_xyz_maxmin[3],
                                sh * c + collider_info.terrain_xyz_maxmin[4],
                                collider_info.terrain_hf[r + i, c] + margin,
                                i_b,
                                collider_state,
                            )
                            if nvert > 2 and (
                                collider_state.prism[3, i_b][2] >= collider_state.xyz_max_min[5, i_b]
                                or collider_state.prism[4, i_b][2] >= collider_state.xyz_max_min[5, i_b]
                                or collider_state.prism[5, i_b][2] >= collider_state.xyz_max_min[5, i_b]
                            ):
                                center_b = qd.Vector.zero(gs.qd_float, 3)
                                for i_p in qd.static(range(6)):
                                    center_b = center_b + collider_state.prism[i_p, i_b]
                                center_b = center_b / 6.0

                                is_col, normal, penetration, contact_pos = mpr.func_mpr_contact_from_centers(
                                    geoms_info,
                                    static_rigid_sim_config,
                                    collider_state,
                                    collider_static_config,
                                    mpr_state,
                                    mpr_info,
                                    support_field_info,
                                    i_ga,
                                    i_gb,
                                    i_b,
                                    center_a,
                                    center_b,
                                    ga_pos_terrain_frame,
                                    ga_quat_terrain_frame,
                                    gb_pos_terrain_frame,
                                    gb_quat_terrain_frame,
                                )
                                if is_col:
                                    normal = gu.qd_transform_by_quat(normal, gb_quat)
                                    contact_pos = gu.qd_transform_by_quat(contact_pos, gb_quat)
                                    contact_pos = contact_pos + gb_pos

                                    valid = True
                                    i_c = collider_state.n_contacts[i_b]
                                    for j in range(n_con):
                                        if (
                                            contact_pos - collider_state.contact_data.pos[i_c - j - 1, i_b]
                                        ).norm() < tolerance:
                                            valid = False
                                            break

                                    if valid:
                                        func_add_contact(
                                            i_ga,
                                            i_gb,
                                            normal,
                                            contact_pos,
                                            penetration,
                                            i_b,
                                            geoms_state,
                                            geoms_info,
                                            collider_state,
                                            collider_info,
                                            errno,
                                        )
                                        n_con = n_con + 1


@qd.func
def func_add_prism_vert(
    x,
    y,
    z,
    i_b,
    collider_state: array_class.ColliderState,
):
    collider_state.prism[0, i_b] = collider_state.prism[1, i_b]
    collider_state.prism[1, i_b] = collider_state.prism[2, i_b]
    collider_state.prism[3, i_b] = collider_state.prism[4, i_b]
    collider_state.prism[4, i_b] = collider_state.prism[5, i_b]

    collider_state.prism[2, i_b][0] = x
    collider_state.prism[5, i_b][0] = x
    collider_state.prism[2, i_b][1] = y
    collider_state.prism[5, i_b][1] = y
    collider_state.prism[5, i_b][2] = z


@qd.func
def func_convex_convex_contact(
    i_ga,
    i_gb,
    i_b,
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    geoms_init_AABB: array_class.GeomsInitAABB,
    verts_info: array_class.VertsInfo,
    faces_info: array_class.FacesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    collider_state: array_class.ColliderState,
    collider_info: array_class.ColliderInfo,
    collider_static_config: qd.template(),
    mpr_state: array_class.MPRState,
    mpr_info: array_class.MPRInfo,
    gjk_state: array_class.GJKState,
    gjk_info: array_class.GJKInfo,
    gjk_static_config: qd.template(),
    support_field_info: array_class.SupportFieldInfo,
    # FIXME: Passing nested data structure as input argument is not supported for now.
    diff_contact_input: array_class.DiffContactInput,
    errno: array_class.V_ANNOTATION,
):
    if not (geoms_info.type[i_ga] == gs.GEOM_TYPE.PLANE and geoms_info.type[i_gb] == gs.GEOM_TYPE.BOX):
        EPS = rigid_global_info.EPS[None]

        # Disabling multi-contact for pairs of decomposed geoms would speed up simulation but may cause physical
        # instabilities in the few cases where multiple contact points are actually need. Increasing the tolerance
        # criteria to get rid of redundant contact points seems to be a better option.
        multi_contact = (
            static_rigid_sim_config.enable_multi_contact
            # and not (self._solver.geoms_info[i_ga].is_decomposed and self._solver.geoms_info[i_gb].is_decomposed)
            and geoms_info.type[i_ga] != gs.GEOM_TYPE.SPHERE
            and geoms_info.type[i_ga] != gs.GEOM_TYPE.ELLIPSOID
            and geoms_info.type[i_gb] != gs.GEOM_TYPE.SPHERE
            and geoms_info.type[i_gb] != gs.GEOM_TYPE.ELLIPSOID
        )

        tolerance = func_compute_tolerance(
            i_ga, i_gb, i_b, collider_info.mc_tolerance[None], geoms_info, geoms_init_AABB
        )
        diff_pos_tolerance = func_compute_tolerance(
            i_ga, i_gb, i_b, collider_info.diff_pos_tolerance[None], geoms_info, geoms_init_AABB
        )
        diff_normal_tolerance = collider_info.diff_normal_tolerance[None]

        # Load original geometry state into thread-local variables
        # These are the UNPERTURBED states used as reference point for each independent perturbation
        ga_pos_original = geoms_state.pos[i_ga, i_b]
        ga_quat_original = geoms_state.quat[i_ga, i_b]
        gb_pos_original = geoms_state.pos[i_gb, i_b]
        gb_quat_original = geoms_state.quat[i_gb, i_b]

        # Current (possibly perturbed) state - initialized to original, updated during perturbations
        ga_pos_current = ga_pos_original
        ga_quat_current = ga_quat_original
        gb_pos_current = gb_pos_original
        gb_quat_current = gb_quat_original

        # Pre-allocate some buffers
        # Note that the variables post-fixed with _0 are the values of these
        # variables for contact 0 (used for multi-contact).
        is_col_0 = False
        penetration_0 = gs.qd_float(0.0)
        normal_0 = qd.Vector.zero(gs.qd_float, 3)
        contact_pos_0 = qd.Vector.zero(gs.qd_float, 3)

        # Whether narrowphase detected a contact.
        is_col = False
        penetration = gs.qd_float(0.0)
        normal = qd.Vector.zero(gs.qd_float, 3)
        contact_pos = qd.Vector.zero(gs.qd_float, 3)

        n_con = gs.qd_int(0)
        axis_0 = qd.Vector.zero(gs.qd_float, 3)
        axis_1 = qd.Vector.zero(gs.qd_float, 3)
        qrot = qd.Vector.zero(gs.qd_float, 4)

        i_pair = collider_info.collision_pair_idx[(i_gb, i_ga) if i_ga > i_gb else (i_ga, i_gb)]
        for i_detection in range(5):
            prefer_gjk = (
                collider_static_config.ccd_algorithm == CCD_ALGORITHM_CODE.GJK
                or collider_static_config.ccd_algorithm == CCD_ALGORITHM_CODE.MJ_GJK
            )

            # Apply perturbations to thread-local state
            if multi_contact and is_col_0:
                # Perturbation axis must not be aligned with the principal axes of inertia the geometry,
                # otherwise it would be more sensitive to ill-conditioning.
                axis = (2 * (i_detection % 2) - 1) * axis_0 + (1 - 2 * ((i_detection // 2) % 2)) * axis_1
                qrot = gu.qd_rotvec_to_quat(collider_info.mc_perturbation[None] * axis, EPS)

                # Apply perturbation starting from original state
                ga_pos_current, ga_quat_current = func_rotate_frame(
                    pos=ga_pos_original, quat=ga_quat_original, contact_pos=contact_pos_0, qrot=qrot
                )
                gb_pos_current, gb_quat_current = func_rotate_frame(
                    pos=gb_pos_original, quat=gb_quat_original, contact_pos=contact_pos_0, qrot=gu.qd_inv_quat(qrot)
                )

            if (multi_contact and is_col_0) or (i_detection == 0):
                if geoms_info.type[i_ga] == gs.GEOM_TYPE.CAPSULE and geoms_info.type[i_gb] == gs.GEOM_TYPE.CAPSULE:
                    is_col, normal, contact_pos, penetration = capsule_contact.func_capsule_capsule_contact(
                        i_ga=i_ga,
                        i_gb=i_gb,
                        ga_pos=ga_pos_current,
                        ga_quat=ga_quat_current,
                        gb_pos=gb_pos_current,
                        gb_quat=gb_quat_current,
                        geoms_info=geoms_info,
                        rigid_global_info=rigid_global_info,
                    )
                elif (
                    geoms_info.type[i_ga] == gs.GEOM_TYPE.SPHERE and geoms_info.type[i_gb] == gs.GEOM_TYPE.CAPSULE
                ) or (geoms_info.type[i_ga] == gs.GEOM_TYPE.CAPSULE and geoms_info.type[i_gb] == gs.GEOM_TYPE.SPHERE):
                    is_col, normal, contact_pos, penetration = capsule_contact.func_sphere_capsule_contact(
                        i_ga=i_ga,
                        i_gb=i_gb,
                        ga_pos=ga_pos_current,
                        ga_quat=ga_quat_current,
                        gb_pos=gb_pos_current,
                        gb_quat=gb_quat_current,
                        geoms_info=geoms_info,
                        rigid_global_info=rigid_global_info,
                    )
                elif geoms_info.type[i_ga] == gs.GEOM_TYPE.PLANE:
                    plane_dir = qd.Vector(
                        [geoms_info.data[i_ga][0], geoms_info.data[i_ga][1], geoms_info.data[i_ga][2]], dt=gs.qd_float
                    )
                    plane_dir = gu.qd_transform_by_quat(plane_dir, ga_quat_current)
                    normal = -plane_dir.normalized()

                    v1 = mpr.support_driver(
                        geoms_info,
                        collider_state,
                        collider_static_config,
                        support_field_info,
                        normal,
                        i_gb,
                        i_b,
                        gb_pos_current,
                        gb_quat_current,
                    )
                    penetration = normal.dot(v1 - ga_pos_current)
                    contact_pos = v1 - 0.5 * penetration * normal
                    is_col = penetration > 0.0
                else:
                    ### MPR, MJ_MPR
                    if qd.static(
                        collider_static_config.ccd_algorithm in (CCD_ALGORITHM_CODE.MPR, CCD_ALGORITHM_CODE.MJ_MPR)
                    ):
                        # Try using MPR before anything else
                        is_mpr_updated = False
                        normal_ws = collider_state.contact_cache.normal[i_pair, i_b]
                        is_mpr_guess_direction_available = (qd.abs(normal_ws) > EPS).any()
                        for i_mpr in range(2):
                            if i_mpr == 1:
                                # Try without warm-start if no contact was detected using it.
                                # When penetration depth is very shallow, MPR may wrongly classify two geometries as not
                                # in contact while they actually are. This helps to improve contact persistence without
                                # increasing much the overall computational cost since the fallback should not be
                                # triggered very often.
                                if qd.static(not static_rigid_sim_config.enable_mujoco_compatibility):
                                    if (i_detection == 0) and not is_col and is_mpr_guess_direction_available:
                                        normal_ws = qd.Vector.zero(gs.qd_float, 3)
                                        is_mpr_guess_direction_available = False
                                        is_mpr_updated = False

                            if not is_mpr_updated:
                                is_col, normal, penetration, contact_pos = mpr.func_mpr_contact(
                                    geoms_info,
                                    geoms_init_AABB,
                                    rigid_global_info,
                                    static_rigid_sim_config,
                                    collider_state,
                                    collider_static_config,
                                    mpr_state,
                                    mpr_info,
                                    support_field_info,
                                    i_ga,
                                    i_gb,
                                    i_b,
                                    normal_ws,
                                    ga_pos_current,
                                    ga_quat_current,
                                    gb_pos_current,
                                    gb_quat_current,
                                )
                                is_mpr_updated = True

                        # Fallback on GJK if collision is detected by MPR if the initial penetration is already quite
                        # large, and either no collision direction was cached or the geometries have large overlap. This
                        # contact information provided by MPR may be unreliable in these cases.
                        if qd.static(collider_static_config.ccd_algorithm == CCD_ALGORITHM_CODE.MPR):
                            if penetration > tolerance:
                                prefer_gjk = not is_mpr_guess_direction_available or (
                                    collider_info.mc_tolerance[None] * penetration
                                    >= collider_info.mpr_to_gjk_overlap_ratio[None] * tolerance
                                )

                    ### GJK, MJ_GJK
                    if qd.static(collider_static_config.ccd_algorithm != CCD_ALGORITHM_CODE.MJ_MPR):
                        if prefer_gjk:
                            if qd.static(static_rigid_sim_config.requires_grad):
                                diff_gjk.func_gjk_contact(
                                    links_state,
                                    links_info,
                                    geoms_state,
                                    geoms_info,
                                    geoms_init_AABB,
                                    verts_info,
                                    faces_info,
                                    rigid_global_info,
                                    static_rigid_sim_config,
                                    collider_state,
                                    collider_static_config,
                                    gjk_state,
                                    gjk_info,
                                    support_field_info,
                                    diff_contact_input,
                                    i_ga,
                                    i_gb,
                                    i_b,
                                    ga_pos_current,
                                    ga_quat_current,
                                    gb_pos_current,
                                    gb_quat_current,
                                    diff_pos_tolerance,
                                    diff_normal_tolerance,
                                )
                            else:
                                gjk.func_gjk_contact(
                                    geoms_state,
                                    geoms_info,
                                    verts_info,
                                    faces_info,
                                    rigid_global_info,
                                    static_rigid_sim_config,
                                    collider_state,
                                    collider_static_config,
                                    gjk_state,
                                    gjk_info,
                                    gjk_static_config,
                                    support_field_info,
                                    i_ga,
                                    i_gb,
                                    i_b,
                                    ga_pos_current,
                                    ga_quat_current,
                                    gb_pos_current,
                                    gb_quat_current,
                                )

                            is_col = gjk_state.is_col[i_b] == 1
                            penetration = gjk_state.penetration[i_b]
                            n_contacts = gjk_state.n_contacts[i_b]

                            if is_col:
                                if qd.static(static_rigid_sim_config.requires_grad):
                                    for i_c in range(n_contacts):
                                        func_add_diff_contact_input(
                                            i_ga,
                                            i_gb,
                                            i_b,
                                            i_c,
                                            gjk_state,
                                            collider_state,
                                            collider_info,
                                        )
                                        func_add_contact(
                                            i_ga,
                                            i_gb,
                                            gjk_state.normal[i_b, i_c],
                                            gjk_state.contact_pos[i_b, i_c],
                                            gjk_state.diff_penetration[i_b, i_c],
                                            i_b,
                                            geoms_state,
                                            geoms_info,
                                            collider_state,
                                            collider_info,
                                            errno,
                                        )
                                    break
                                else:
                                    if gjk_state.multi_contact_flag[i_b]:
                                        # Since we already found multiple contact points, add the discovered contact
                                        # points and stop multi-contact search.
                                        for i_c in range(n_contacts):
                                            # Ignore contact points if the number of contacts exceeds the limit.
                                            if i_c < qd.static(collider_static_config.n_contacts_per_pair):
                                                contact_pos = gjk_state.contact_pos[i_b, i_c]
                                                normal = gjk_state.normal[i_b, i_c]
                                                if qd.static(static_rigid_sim_config.requires_grad):
                                                    penetration = gjk_state.diff_penetration[i_b, i_c]
                                                func_add_contact(
                                                    i_ga,
                                                    i_gb,
                                                    normal,
                                                    contact_pos,
                                                    penetration,
                                                    i_b,
                                                    geoms_state,
                                                    geoms_info,
                                                    collider_state,
                                                    collider_info,
                                                    errno,
                                                )

                                        break
                                    else:
                                        contact_pos = gjk_state.contact_pos[i_b, 0]
                                        normal = gjk_state.normal[i_b, 0]

            if i_detection == 0:
                is_col_0, normal_0, penetration_0, contact_pos_0 = is_col, normal, penetration, contact_pos
                if is_col_0:
                    func_add_contact(
                        i_ga,
                        i_gb,
                        normal_0,
                        contact_pos_0,
                        penetration_0,
                        i_b,
                        geoms_state,
                        geoms_info,
                        collider_state,
                        collider_info,
                        errno,
                    )
                    if multi_contact:
                        # Perturb geom_a around two orthogonal axes to find multiple contacts
                        axis_0, axis_1 = func_contact_orthogonals(
                            i_ga,
                            i_gb,
                            normal,
                            i_b,
                            links_state,
                            links_info,
                            geoms_state,
                            geoms_info,
                            geoms_init_AABB,
                            rigid_global_info,
                            static_rigid_sim_config,
                        )
                        n_con = 1

                    if qd.static(
                        collider_static_config.ccd_algorithm in (CCD_ALGORITHM_CODE.MPR, CCD_ALGORITHM_CODE.GJK)
                    ):
                        collider_state.contact_cache.normal[i_pair, i_b] = normal
                else:
                    # Clear collision normal cache if not in contact
                    collider_state.contact_cache.normal[i_pair, i_b] = qd.Vector.zero(gs.qd_float, 3)
            elif multi_contact and is_col:
                # For perturbed iterations (i_detection > 0), correct contact position and normal. This applies to all
                # collision methods when multi-contact is enabled, except mujoco compatible.
                #
                # 1. Project the contact point on both geometries
                # 2. Revert the effect of small rotation
                # 3. Update contact point
                if qd.static(
                    collider_static_config.ccd_algorithm not in (CCD_ALGORITHM_CODE.MJ_MPR, CCD_ALGORITHM_CODE.MJ_GJK)
                ):
                    contact_point_a = (
                        gu.qd_transform_by_quat(
                            (contact_pos - 0.5 * penetration * normal) - contact_pos_0,
                            gu.qd_inv_quat(qrot),
                        )
                        + contact_pos_0
                    )
                    contact_point_b = (
                        gu.qd_transform_by_quat(
                            (contact_pos + 0.5 * penetration * normal) - contact_pos_0,
                            qrot,
                        )
                        + contact_pos_0
                    )
                    contact_pos = 0.5 * (contact_point_a + contact_point_b)

                    # First-order correction of the normal direction.
                    # The way the contact normal gets twisted by applying perturbation of geometry poses is
                    # unpredictable as it depends on the final portal discovered by MPR. Alternatively, let compute
                    # the minimal rotation that makes the corrected twisted normal as closed as possible to the
                    # original one, up to the scale of the perturbation, then apply first-order Taylor expansion of
                    # Rodrigues' rotation formula.
                    twist_rotvec = qd.math.clamp(
                        normal.cross(normal_0),
                        -collider_info.mc_perturbation[None],
                        collider_info.mc_perturbation[None],
                    )
                    normal = normal + twist_rotvec.cross(normal)

                    # Make sure that the penetration is still positive before adding contact point.
                    # Note that adding some negative tolerance improves physical stability by encouraging persistent
                    # contact points and thefore more continuous contact forces, without changing the mean-field
                    # dynamics since zero-penetration contact points should not induce any force.
                    penetration = normal.dot(contact_point_b - contact_point_a)
                    if qd.static(collider_static_config.ccd_algorithm == CCD_ALGORITHM_CODE.MJ_GJK):
                        # Only change penetration to the initial one, because the normal vector could change abruptly
                        # under MuJoCo's GJK-EPA.
                        penetration = penetration_0

                # Discard contact point is repeated
                repeated = False
                for i_c in range(n_con):
                    if not repeated:
                        idx_prev = collider_state.n_contacts[i_b] - 1 - i_c
                        prev_contact = collider_state.contact_data.pos[idx_prev, i_b]
                        if (contact_pos - prev_contact).norm() < tolerance:
                            repeated = True

                if not repeated:
                    if penetration > -tolerance:
                        penetration = qd.max(penetration, 0.0)
                        func_add_contact(
                            i_ga,
                            i_gb,
                            normal,
                            contact_pos,
                            penetration,
                            i_b,
                            geoms_state,
                            geoms_info,
                            collider_state,
                            collider_info,
                            errno,
                        )
                        n_con = n_con + 1


@qd.kernel(fastcache=gs.use_fastcache)
def func_narrow_phase_convex_vs_convex(
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    geoms_init_AABB: array_class.GeomsInitAABB,
    verts_info: array_class.VertsInfo,
    faces_info: array_class.FacesInfo,
    edges_info: array_class.EdgesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    collider_state: array_class.ColliderState,
    collider_info: array_class.ColliderInfo,
    collider_static_config: qd.template(),
    mpr_state: array_class.MPRState,
    mpr_info: array_class.MPRInfo,
    gjk_state: array_class.GJKState,
    gjk_info: array_class.GJKInfo,
    gjk_static_config: qd.template(),
    sdf_info: array_class.SDFInfo,
    support_field_info: array_class.SupportFieldInfo,
    diff_contact_input: array_class.DiffContactInput,
    errno: array_class.V_ANNOTATION,
):
    _B = collider_state.active_buffer.shape[1]

    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        for i_pair in range(collider_state.n_broad_pairs[i_b]):
            i_ga = collider_state.broad_collision_pairs[i_pair, i_b][0]
            i_gb = collider_state.broad_collision_pairs[i_pair, i_b][1]

            if geoms_info.type[i_ga] > geoms_info.type[i_gb]:
                i_ga, i_gb = i_gb, i_ga

            if (
                geoms_info.is_convex[i_ga]
                and geoms_info.is_convex[i_gb]
                and not geoms_info.type[i_gb] == gs.GEOM_TYPE.TERRAIN
                and not (
                    static_rigid_sim_config.box_box_detection
                    and geoms_info.type[i_ga] == gs.GEOM_TYPE.BOX
                    and geoms_info.type[i_gb] == gs.GEOM_TYPE.BOX
                )
            ):
                if not (geoms_info.type[i_ga] == gs.GEOM_TYPE.PLANE and geoms_info.type[i_gb] == gs.GEOM_TYPE.BOX):
                    func_convex_convex_contact(
                        i_ga=i_ga,
                        i_gb=i_gb,
                        i_b=i_b,
                        links_state=links_state,
                        links_info=links_info,
                        geoms_state=geoms_state,
                        geoms_info=geoms_info,
                        geoms_init_AABB=geoms_init_AABB,
                        verts_info=verts_info,
                        faces_info=faces_info,
                        rigid_global_info=rigid_global_info,
                        static_rigid_sim_config=static_rigid_sim_config,
                        collider_state=collider_state,
                        collider_info=collider_info,
                        collider_static_config=collider_static_config,
                        mpr_state=mpr_state,
                        mpr_info=mpr_info,
                        gjk_state=gjk_state,
                        gjk_info=gjk_info,
                        gjk_static_config=gjk_static_config,
                        support_field_info=support_field_info,
                        # FIXME: Passing nested data structure as input argument is not supported for now.
                        diff_contact_input=diff_contact_input,
                        errno=errno,
                    )


@qd.kernel(fastcache=gs.use_fastcache)
def func_narrow_phase_diff_convex_vs_convex(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    static_rigid_sim_config: qd.template(),
    collider_state: array_class.ColliderState,
    collider_info: array_class.ColliderInfo,
    gjk_info: array_class.GJKInfo,
    # FIXME: Passing nested data structure as input argument is not supported for now.
    diff_contact_input: array_class.DiffContactInput,
):
    # Compute reference contacts
    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL)
    for i_c, i_b in qd.ndrange(collider_state.contact_data.pos.shape[0], collider_state.active_buffer.shape[1]):
        if i_c < collider_state.n_contacts[i_b]:
            ref_id = collider_state.diff_contact_input.ref_id[i_b, i_c]
            is_ref = i_c == ref_id
            i_ga = collider_state.diff_contact_input.geom_a[i_b, i_c]
            i_gb = collider_state.diff_contact_input.geom_b[i_b, i_c]

            if is_ref:
                ref_penetration = -1.0
                contact_pos, contact_normal, penetration, weight = diff_gjk.func_differentiable_contact(
                    geoms_state, diff_contact_input, gjk_info, i_ga, i_gb, i_b, i_c, ref_penetration
                )
                collider_state.diff_contact_input.ref_penetration[i_b, i_c] = penetration

                func_set_contact(
                    i_ga,
                    i_gb,
                    contact_normal,
                    contact_pos,
                    penetration * weight,
                    i_b,
                    i_c,
                    geoms_state,
                    geoms_info,
                    collider_state,
                    collider_info,
                )

    # Compute other contacts
    for i_c, i_b in qd.ndrange(collider_state.contact_data.pos.shape[0], collider_state.active_buffer.shape[1]):
        if i_c < collider_state.n_contacts[i_b]:
            ref_id = collider_state.diff_contact_input.ref_id[i_b, i_c]
            is_ref = i_c == ref_id
            i_ga = collider_state.diff_contact_input.geom_a[i_b, i_c]
            i_gb = collider_state.diff_contact_input.geom_b[i_b, i_c]

            if not is_ref:
                ref_penetration = collider_state.diff_contact_input.ref_penetration[i_b, ref_id]
                contact_pos, contact_normal, penetration, weight = diff_gjk.func_differentiable_contact(
                    geoms_state, diff_contact_input, gjk_info, i_ga, i_gb, i_b, i_c, ref_penetration
                )

                func_set_contact(
                    i_ga,
                    i_gb,
                    contact_normal,
                    contact_pos,
                    penetration * weight,
                    i_b,
                    i_c,
                    geoms_state,
                    geoms_info,
                    collider_state,
                    collider_info,
                )


@qd.kernel(fastcache=gs.use_fastcache)
def func_narrow_phase_convex_specializations(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    geoms_init_AABB: array_class.GeomsInitAABB,
    verts_info: array_class.VertsInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    collider_state: array_class.ColliderState,
    collider_info: array_class.ColliderInfo,
    collider_static_config: qd.template(),
    errno: array_class.V_ANNOTATION,
):
    _B = collider_state.active_buffer.shape[1]
    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        for i_pair in range(collider_state.n_broad_pairs[i_b]):
            i_ga = collider_state.broad_collision_pairs[i_pair, i_b][0]
            i_gb = collider_state.broad_collision_pairs[i_pair, i_b][1]

            if geoms_info.type[i_ga] > geoms_info.type[i_gb]:
                i_ga, i_gb = i_gb, i_ga

            if geoms_info.type[i_ga] == gs.GEOM_TYPE.PLANE and geoms_info.type[i_gb] == gs.GEOM_TYPE.BOX:
                func_plane_box_contact(
                    i_ga,
                    i_gb,
                    i_b,
                    geoms_state,
                    geoms_info,
                    geoms_init_AABB,
                    verts_info,
                    static_rigid_sim_config,
                    collider_state,
                    collider_info,
                    collider_static_config,
                    errno,
                )

            if qd.static(static_rigid_sim_config.box_box_detection):
                if geoms_info.type[i_ga] == gs.GEOM_TYPE.BOX and geoms_info.type[i_gb] == gs.GEOM_TYPE.BOX:
                    func_box_box_contact(
                        i_ga,
                        i_gb,
                        i_b,
                        geoms_state,
                        geoms_info,
                        collider_state,
                        collider_info,
                        rigid_global_info,
                        collider_static_config,
                        errno,
                    )


@qd.kernel(fastcache=gs.use_fastcache)
def func_narrow_phase_any_vs_terrain(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    geoms_init_AABB: array_class.GeomsInitAABB,
    static_rigid_sim_config: qd.template(),
    collider_state: array_class.ColliderState,
    collider_info: array_class.ColliderInfo,
    collider_static_config: qd.template(),
    mpr_state: array_class.MPRState,
    mpr_info: array_class.MPRInfo,
    support_field_info: array_class.SupportFieldInfo,
    errno: array_class.V_ANNOTATION,
):
    """
    NOTE: for a single non-batched scene with a lot of collisioin pairs, it will be faster if we also parallelize over `self.n_collision_pairs`. However, parallelize over both B and collisioin_pairs (instead of only over B) leads to significantly slow performance for batched scene. We can treat B=0 and B>0 separately, but we will end up with messier code.
    Therefore, for a big non-batched scene, users are encouraged to simply use `gs.cpu` backend.
    Updated NOTE & TODO: For a HUGE scene with numerous bodies, it's also reasonable to run on GPU. Let's save this for later.
    Update2: Now we use n_broad_pairs instead of n_collision_pairs, so we probably need to think about how to handle non-batched large scene better.
    """
    _B = collider_state.active_buffer.shape[1]
    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        for i_pair in range(collider_state.n_broad_pairs[i_b]):
            i_ga = collider_state.broad_collision_pairs[i_pair, i_b][0]
            i_gb = collider_state.broad_collision_pairs[i_pair, i_b][1]

            if qd.static(collider_static_config.has_terrain):
                if geoms_info.type[i_ga] == gs.GEOM_TYPE.TERRAIN:
                    i_ga, i_gb = i_gb, i_ga

                if geoms_info.type[i_gb] == gs.GEOM_TYPE.TERRAIN:
                    func_contact_mpr_terrain(
                        i_ga,
                        i_gb,
                        i_b,
                        geoms_state,
                        geoms_info,
                        geoms_init_AABB,
                        static_rigid_sim_config,
                        collider_state,
                        collider_info,
                        collider_static_config,
                        mpr_state,
                        mpr_info,
                        support_field_info,
                        errno,
                    )


@qd.kernel(fastcache=gs.use_fastcache)
def func_narrow_phase_nonconvex_vs_nonterrain(
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    geoms_init_AABB: array_class.GeomsInitAABB,
    verts_info: array_class.VertsInfo,
    edges_info: array_class.EdgesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    collider_state: array_class.ColliderState,
    collider_info: array_class.ColliderInfo,
    collider_static_config: qd.template(),
    sdf_info: array_class.SDFInfo,
    errno: array_class.V_ANNOTATION,
):
    """
    NOTE: for a single non-batched scene with a lot of collisioin pairs, it will be faster if we also parallelize over `self.n_collision_pairs`. However, parallelize over both B and collisioin_pairs (instead of only over B) leads to significantly slow performance for batched scene. We can treat B=0 and B>0 separately, but we will end up with messier code.
    Therefore, for a big non-batched scene, users are encouraged to simply use `gs.cpu` backend.
    Updated NOTE & TODO: For a HUGE scene with numerous bodies, it's also reasonable to run on GPU. Let's save this for later.
    Update2: Now we use n_broad_pairs instead of n_collision_pairs, so we probably need to think about how to handle non-batched large scene better.
    """
    EPS = rigid_global_info.EPS[None]

    _B = collider_state.active_buffer.shape[1]
    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_b in range(_B):
        for i_pair in range(collider_state.n_broad_pairs[i_b]):
            i_ga = collider_state.broad_collision_pairs[i_pair, i_b][0]
            i_gb = collider_state.broad_collision_pairs[i_pair, i_b][1]

            if qd.static(collider_static_config.has_nonconvex_nonterrain):
                if (
                    not (geoms_info.is_convex[i_ga] and geoms_info.is_convex[i_gb])
                    and geoms_info.type[i_gb] != gs.GEOM_TYPE.TERRAIN
                ):
                    is_col = False
                    tolerance = func_compute_tolerance(
                        i_ga, i_gb, i_b, collider_info.mc_tolerance[None], geoms_info, geoms_init_AABB
                    )
                    for i in range(2):
                        if i == 1:
                            i_ga, i_gb = i_gb, i_ga

                        # initial point
                        is_col_i = False
                        normal_i = qd.Vector.zero(gs.qd_float, 3)
                        contact_pos_i = qd.Vector.zero(gs.qd_float, 3)
                        if not is_col:
                            ga_pos = geoms_state.pos[i_ga, i_b]
                            ga_quat = geoms_state.quat[i_ga, i_b]
                            gb_pos = geoms_state.pos[i_gb, i_b]
                            gb_quat = geoms_state.quat[i_gb, i_b]
                            is_col_i, normal_i, penetration_i, contact_pos_i = func_contact_vertex_sdf(
                                i_ga,
                                i_gb,
                                i_b,
                                ga_pos,
                                ga_quat,
                                gb_pos,
                                gb_quat,
                                geoms_state,
                                geoms_info,
                                verts_info,
                                rigid_global_info,
                                collider_static_config,
                                sdf_info,
                            )
                            if is_col_i:
                                func_add_contact(
                                    i_ga,
                                    i_gb,
                                    normal_i,
                                    contact_pos_i,
                                    penetration_i,
                                    i_b,
                                    geoms_state,
                                    geoms_info,
                                    collider_state,
                                    collider_info,
                                    errno,
                                )

                        if qd.static(static_rigid_sim_config.enable_multi_contact):
                            if not is_col and is_col_i:
                                ga_pos_original, ga_quat_original = (
                                    geoms_state.pos[i_ga, i_b],
                                    geoms_state.quat[i_ga, i_b],
                                )
                                gb_pos_original, gb_quat_original = (
                                    geoms_state.pos[i_gb, i_b],
                                    geoms_state.quat[i_gb, i_b],
                                )

                                # Perturb geom_a around two orthogonal axes to find multiple contacts
                                axis_0, axis_1 = func_contact_orthogonals(
                                    i_ga,
                                    i_gb,
                                    normal_i,
                                    i_b,
                                    links_state,
                                    links_info,
                                    geoms_state,
                                    geoms_info,
                                    geoms_init_AABB,
                                    rigid_global_info,
                                    static_rigid_sim_config,
                                )

                                n_con = 1
                                for i_rot in range(1, 5):
                                    axis = (2 * (i_rot % 2) - 1) * axis_0 + (1 - 2 * ((i_rot // 2) % 2)) * axis_1

                                    qrot = gu.qd_rotvec_to_quat(collider_info.mc_perturbation[None] * axis, EPS)

                                    # Apply perturbations to local variables (no global state modification)
                                    ga_pos_perturbed, ga_quat_perturbed = func_rotate_frame(
                                        ga_pos_original, ga_quat_original, contact_pos_i, qrot
                                    )
                                    gb_pos_perturbed, gb_quat_perturbed = func_rotate_frame(
                                        gb_pos_original, gb_quat_original, contact_pos_i, gu.qd_inv_quat(qrot)
                                    )

                                    is_col, normal, penetration, contact_pos = func_contact_vertex_sdf(
                                        i_ga,
                                        i_gb,
                                        i_b,
                                        ga_pos_perturbed,
                                        ga_quat_perturbed,
                                        gb_pos_perturbed,
                                        gb_quat_perturbed,
                                        geoms_state,
                                        geoms_info,
                                        verts_info,
                                        rigid_global_info,
                                        collider_static_config,
                                        sdf_info,
                                    )

                                    if is_col:
                                        if qd.static(not static_rigid_sim_config.enable_mujoco_compatibility):
                                            # 1. Project the contact point on both geometries
                                            # 2. Revert the effect of small rotation
                                            # 3. Update contact point
                                            contact_point_a = (
                                                gu.qd_transform_by_quat(
                                                    (contact_pos - 0.5 * penetration * normal) - contact_pos_i,
                                                    gu.qd_inv_quat(qrot),
                                                )
                                                + contact_pos_i
                                            )
                                            contact_point_b = (
                                                gu.qd_transform_by_quat(
                                                    (contact_pos + 0.5 * penetration * normal) - contact_pos_i,
                                                    qrot,
                                                )
                                                + contact_pos_i
                                            )
                                            contact_pos = 0.5 * (contact_point_a + contact_point_b)

                                            # First-order correction of the normal direction
                                            twist_rotvec = qd.math.clamp(
                                                normal.cross(normal_i),
                                                -collider_info.mc_perturbation[None],
                                                collider_info.mc_perturbation[None],
                                            )
                                            normal = normal + twist_rotvec.cross(normal)

                                            # Make sure that the penetration is still positive
                                            penetration = normal.dot(contact_point_b - contact_point_a)

                                        # Discard contact point is repeated
                                        repeated = False
                                        for i_c in range(n_con):
                                            if not repeated:
                                                idx_prev = collider_state.n_contacts[i_b] - 1 - i_c
                                                prev_contact = collider_state.contact_data.pos[idx_prev, i_b]
                                                if (contact_pos - prev_contact).norm() < tolerance:
                                                    repeated = True

                                        if not repeated:
                                            if penetration > -tolerance:
                                                penetration = qd.max(penetration, 0.0)
                                                func_add_contact(
                                                    i_ga,
                                                    i_gb,
                                                    normal,
                                                    contact_pos,
                                                    penetration,
                                                    i_b,
                                                    geoms_state,
                                                    geoms_info,
                                                    collider_state,
                                                    collider_info,
                                                    errno,
                                                )
                                                n_con = n_con + 1

                        if not is_col:  # check edge-edge if vertex-face is not detected
                            # Extract current poses for initial collision detection
                            ga_pos = geoms_state.pos[i_ga, i_b]
                            ga_quat = geoms_state.quat[i_ga, i_b]
                            gb_pos = geoms_state.pos[i_gb, i_b]
                            gb_quat = geoms_state.quat[i_gb, i_b]

                            is_col, normal, penetration, contact_pos = func_contact_edge_sdf(
                                i_ga,
                                i_gb,
                                i_b,
                                ga_pos,
                                ga_quat,
                                gb_pos,
                                gb_quat,
                                geoms_state,
                                geoms_info,
                                verts_info,
                                edges_info,
                                rigid_global_info,
                                collider_static_config,
                                sdf_info,
                            )
                            if is_col:
                                func_add_contact(
                                    i_ga,
                                    i_gb,
                                    normal,
                                    contact_pos,
                                    penetration,
                                    i_b,
                                    geoms_state,
                                    geoms_info,
                                    collider_state,
                                    collider_info,
                                    errno,
                                )
