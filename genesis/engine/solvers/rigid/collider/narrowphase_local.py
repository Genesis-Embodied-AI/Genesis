"""
Thread-local state functions for narrow-phase collision detection.

This module provides thread-local versions of collision detection functions that use
per-thread copies of geometry state instead of modifying shared global state. This
enables race-free parallelization across collision pairs within the same environment.

The key difference from the original narrowphase.py functions is that geometry perturbations
for multi-contact detection are performed on thread-local copies (stored in registers),
preventing intra-environment race conditions when multiple threads process different
collision pairs involving the same geometry.
"""

import taichi as ti

import genesis as gs
from genesis.engine.solvers.rigid.collider import box_contact, contact, diff_gjk, gjk, mpr
from genesis.utils import array_class, geom_utils as gu


@ti.func
def func_rotate_frame_local(
    pos: ti.types.vector(3, dtype=gs.ti_float),
    quat: ti.types.vector(4, dtype=gs.ti_float),
    contact_pos: ti.types.vector(3, dtype=gs.ti_float),
    qrot: ti.types.vector(4, dtype=gs.ti_float),
) -> ti.types.struct(
    pos=ti.types.vector(3, dtype=gs.ti_float),
    quat=ti.types.vector(4, dtype=gs.ti_float),
):
    """
    Thread-local version of func_rotate_frame.

    Instead of modifying geoms_state in place, this function takes thread-local
    pos/quat and returns the updated values.

    Args:
        pos: Thread-local geometry position (28 bytes)
        quat: Thread-local geometry quaternion (28 bytes)
        contact_pos: Contact position to rotate around
        qrot: Rotation quaternion for perturbation

    Returns:
        Struct containing updated pos and quat (56 bytes total)
    """
    # Update quaternion
    new_quat = gu.ti_transform_quat_by_quat(quat, qrot)

    # Update position
    rel = contact_pos - pos
    vec = gu.ti_transform_by_quat(rel, qrot)
    vec = vec - rel
    new_pos = pos - vec

    # Return struct with both updated values
    return ti.Struct(pos=new_pos, quat=new_quat)


@ti.func
def func_convex_convex_contact_local(
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
    edges_info: array_class.EdgesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    collider_state: array_class.ColliderState,
    collider_info: array_class.ColliderInfo,
    collider_static_config: ti.template(),
    mpr_state: array_class.MPRState,
    mpr_info: array_class.MPRInfo,
    gjk_state: array_class.GJKState,
    gjk_info: array_class.GJKInfo,
    gjk_static_config: ti.template(),
    sdf_info: array_class.SDFInfo,
    support_field_info: array_class.SupportFieldInfo,
    diff_contact_input: array_class.DiffContactInput,
    errno: array_class.V_ANNOTATION,
):
    """
    Thread-local version of func_convex_convex_contact.

    This function uses thread-local copies of geometry state for multi-contact perturbations,
    preventing race conditions when multiple threads process collisions involving the same
    geometry in the same environment.

    Key changes from original:
    1. Load geometry state into thread-local variables at the start
    2. Apply perturbations to thread-local copies using func_rotate_frame_local
    3. Temporarily write thread-local state to global memory before calling collision algos
    4. Restore original state after each detection iteration
    5. Only the final confirmed contacts affect global state

    Memory usage: 112 bytes per thread (56 bytes each for geom_a and geom_b state)
    """
    if geoms_info.type[i_ga] == gs.GEOM_TYPE.PLANE and geoms_info.type[i_gb] == gs.GEOM_TYPE.BOX:
        # Plane-box collision doesn't use perturbations, so call original function
        if ti.static(sys.platform == "darwin"):
            box_contact.func_plane_box_contact(
                i_ga=i_ga,
                i_gb=i_gb,
                i_b=i_b,
                geoms_state=geoms_state,
                geoms_info=geoms_info,
                geoms_init_AABB=geoms_init_AABB,
                verts_info=verts_info,
                static_rigid_sim_config=static_rigid_sim_config,
                collider_state=collider_state,
                collider_info=collider_info,
                collider_static_config=collider_static_config,
                errno=errno,
            )
    else:
        EPS = rigid_global_info.EPS[None]

        multi_contact = (
            static_rigid_sim_config.enable_multi_contact
            and geoms_info.type[i_ga] != gs.GEOM_TYPE.SPHERE
            and geoms_info.type[i_ga] != gs.GEOM_TYPE.ELLIPSOID
            and geoms_info.type[i_gb] != gs.GEOM_TYPE.SPHERE
            and geoms_info.type[i_gb] != gs.GEOM_TYPE.ELLIPSOID
        )

        tolerance = contact.func_compute_tolerance(
            i_ga, i_gb, i_b, collider_info.mc_tolerance[None], geoms_info, geoms_init_AABB
        )
        diff_pos_tolerance = contact.func_compute_tolerance(
            i_ga, i_gb, i_b, collider_info.diff_pos_tolerance[None], geoms_info, geoms_init_AABB
        )
        diff_normal_tolerance = collider_info.diff_normal_tolerance[None]

        # CRITICAL: Backup original geometry state into thread-local variables (56 bytes per geom = 112 bytes total)
        # These live in registers and are private to this thread
        # We keep the ORIGINAL unperturbed state here and restore from it after each perturbation
        ga_pos_original = geoms_state.pos[i_ga, i_b]
        ga_quat_original = geoms_state.quat[i_ga, i_b]
        gb_pos_original = geoms_state.pos[i_gb, i_b]
        gb_quat_original = geoms_state.quat[i_gb, i_b]

        # Pre-allocate buffers
        is_col_0 = False
        penetration_0 = gs.ti_float(0.0)
        normal_0 = ti.Vector.zero(gs.ti_float, 3)
        contact_pos_0 = ti.Vector.zero(gs.ti_float, 3)

        is_col = False
        penetration = gs.ti_float(0.0)
        normal = ti.Vector.zero(gs.ti_float, 3)
        contact_pos = ti.Vector.zero(gs.ti_float, 3)

        n_con = gs.ti_int(0)
        axis_0 = ti.Vector.zero(gs.ti_float, 3)
        axis_1 = ti.Vector.zero(gs.ti_float, 3)
        qrot = ti.Vector.zero(gs.ti_float, 4)

        i_pair = collider_info.collision_pair_idx[(i_gb, i_ga) if i_ga > i_gb else (i_ga, i_gb)]

        for i_detection in range(5):
            prefer_gjk = (
                collider_static_config.ccd_algorithm == CCD_ALGORITHM_CODE.GJK
                or collider_static_config.ccd_algorithm == CCD_ALGORITHM_CODE.MJ_GJK
            )

            # Apply perturbations to global memory for collision detection
            # We start from the original (unperturbed) state and apply the perturbation
            if multi_contact and is_col_0:
                axis = (2 * (i_detection % 2) - 1) * axis_0 + (1 - 2 * ((i_detection // 2) % 2)) * axis_1
                qrot = gu.ti_rotvec_to_quat(collider_info.mc_perturbation[None] * axis, EPS)

                # Apply perturbation to global state using thread-local rotation function
                # Start from original unperturbed state
                ga_result = func_rotate_frame_local(ga_pos_original, ga_quat_original, contact_pos_0, qrot)
                geoms_state.pos[i_ga, i_b] = ga_result.pos
                geoms_state.quat[i_ga, i_b] = ga_result.quat

                gb_result = func_rotate_frame_local(
                    gb_pos_original, gb_quat_original, contact_pos_0, gu.ti_inv_quat(qrot)
                )
                geoms_state.pos[i_gb, i_b] = gb_result.pos
                geoms_state.quat[i_gb, i_b] = gb_result.quat

            if (multi_contact and is_col_0) or (i_detection == 0):
                if geoms_info.type[i_ga] == gs.GEOM_TYPE.PLANE:
                    # Plane collision detection
                    plane_dir = ti.Vector(
                        [geoms_info.data[i_ga][0], geoms_info.data[i_ga][1], geoms_info.data[i_ga][2]], dt=gs.ti_float
                    )
                    plane_dir = gu.ti_transform_by_quat(plane_dir, geoms_state.quat[i_ga, i_b])
                    normal = -plane_dir.normalized()

                    v1 = mpr.support_driver(
                        geoms_state,
                        geoms_info,
                        collider_state,
                        collider_info,
                        collider_static_config,
                        support_field_info,
                        normal,
                        i_gb,
                        i_b,
                    )
                    penetration = normal.dot(v1 - geoms_state.pos[i_ga, i_b])
                    contact_pos = v1 - 0.5 * penetration * normal
                    is_col = penetration > 0.0
                else:
                    # MPR collision detection
                    if ti.static(
                        collider_static_config.ccd_algorithm in (CCD_ALGORITHM_CODE.MPR, CCD_ALGORITHM_CODE.MJ_MPR)
                    ):
                        is_mpr_updated = False
                        normal_ws = collider_state.contact_cache.normal[i_pair, i_b]
                        is_mpr_guess_direction_available = (ti.abs(normal_ws) > EPS).any()

                        for i_mpr in range(2):
                            if i_mpr == 1:
                                if ti.static(not static_rigid_sim_config.enable_mujoco_compatibility):
                                    if (i_detection == 0) and not is_col and is_mpr_guess_direction_available:
                                        normal_ws = ti.Vector.zero(gs.ti_float, 3)
                                        is_mpr_guess_direction_available = False
                                        is_mpr_updated = False

                            if not is_mpr_updated:
                                is_col, normal, penetration, contact_pos = mpr.func_mpr_contact(
                                    geoms_state,
                                    geoms_info,
                                    geoms_init_AABB,
                                    rigid_global_info,
                                    static_rigid_sim_config,
                                    collider_state,
                                    collider_info,
                                    collider_static_config,
                                    mpr_state,
                                    mpr_info,
                                    support_field_info,
                                    i_ga,
                                    i_gb,
                                    i_b,
                                    normal_ws,
                                )
                                is_mpr_updated = True

                        # Fallback to GJK for deep penetrations
                        if ti.static(collider_static_config.ccd_algorithm == CCD_ALGORITHM_CODE.MPR):
                            if penetration > tolerance:
                                prefer_gjk = not is_mpr_guess_direction_available or (
                                    collider_info.mc_tolerance[None] * penetration
                                    >= collider_info.mpr_to_gjk_overlap_ratio[None] * tolerance
                                )

                    # GJK collision detection
                    if ti.static(collider_static_config.ccd_algorithm != CCD_ALGORITHM_CODE.MJ_MPR):
                        if prefer_gjk:
                            if ti.static(static_rigid_sim_config.requires_grad):
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
                                )

                            is_col = gjk_state.is_col[i_b] == 1
                            penetration = gjk_state.penetration[i_b]
                            n_contacts = gjk_state.n_contacts[i_b]

                            if is_col:
                                if ti.static(static_rigid_sim_config.requires_grad):
                                    for i_c in range(n_contacts):
                                        contact.func_add_diff_contact_input(
                                            i_ga,
                                            i_gb,
                                            i_b,
                                            i_c,
                                            gjk_state,
                                            collider_state,
                                            collider_info,
                                        )
                                        contact.func_add_contact(
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
                                        for i_c in range(n_contacts):
                                            if i_c < ti.static(collider_static_config.n_contacts_per_pair):
                                                contact_pos = gjk_state.contact_pos[i_b, i_c]
                                                normal = gjk_state.normal[i_b, i_c]
                                                if ti.static(static_rigid_sim_config.requires_grad):
                                                    penetration = gjk_state.diff_penetration[i_b, i_c]
                                                contact.func_add_contact(
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

            # Restore original geometry state after each perturbation iteration
            # This is done at the END of each iteration, just like the original code
            if multi_contact and is_col_0:
                geoms_state.pos[i_ga, i_b] = ga_pos_original
                geoms_state.quat[i_ga, i_b] = ga_quat_original
                geoms_state.pos[i_gb, i_b] = gb_pos_original
                geoms_state.quat[i_gb, i_b] = gb_quat_original

            # First detection: save results and add contact if found
            if i_detection == 0:
                is_col_0, normal_0, penetration_0, contact_pos_0 = is_col, normal, penetration, contact_pos
                if is_col_0:
                    contact.func_add_contact(
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
                        # Compute perturbation axes for subsequent detections
                        axis_0, axis_1 = contact.func_contact_orthogonals(
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

                    if ti.static(
                        collider_static_config.ccd_algorithm in (CCD_ALGORITHM_CODE.MPR, CCD_ALGORITHM_CODE.GJK)
                    ):
                        collider_state.contact_cache.normal[i_pair, i_b] = normal
                else:
                    collider_state.contact_cache.normal[i_pair, i_b] = ti.Vector.zero(gs.ti_float, 3)

            # Subsequent detections: correct contact position and add if valid
            elif multi_contact and is_col_0 > 0 and is_col > 0:
                if ti.static(collider_static_config.ccd_algorithm in (CCD_ALGORITHM_CODE.MPR, CCD_ALGORITHM_CODE.GJK)):
                    # Project contact points and correct for perturbation
                    contact_point_a = (
                        gu.ti_transform_by_quat(
                            (contact_pos - 0.5 * penetration * normal) - contact_pos_0,
                            gu.ti_inv_quat(qrot),
                        )
                        + contact_pos_0
                    )
                    contact_point_b = (
                        gu.ti_transform_by_quat(
                            (contact_pos + 0.5 * penetration * normal) - contact_pos_0,
                            qrot,
                        )
                        + contact_pos_0
                    )
                    contact_pos = 0.5 * (contact_point_a + contact_point_b)

                    # First-order correction of normal direction
                    twist_rotvec = ti.math.clamp(
                        normal.cross(normal_0),
                        -collider_info.mc_perturbation[None],
                        collider_info.mc_perturbation[None],
                    )
                    normal = normal + twist_rotvec.cross(normal)
                    penetration = normal.dot(contact_point_b - contact_point_a)

                if ti.static(collider_static_config.ccd_algorithm == CCD_ALGORITHM_CODE.MJ_GJK):
                    penetration = penetration_0

                # Check for duplicate contacts
                repeated = False
                for i_c in range(n_con):
                    if not repeated:
                        idx_prev = collider_state.n_contacts[i_b] - 1 - i_c
                        prev_contact = collider_state.contact_data.pos[idx_prev, i_b]
                        if (contact_pos - prev_contact).norm() < tolerance:
                            repeated = True

                # Add contact if not repeated and penetration is positive
                if not repeated:
                    if penetration > -tolerance:
                        penetration = ti.max(penetration, 0.0)
                        contact.func_add_contact(
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


# Import the CCD algorithm enum for use in static branches
from genesis.engine.solvers.rigid.collider.narrowphase import CCD_ALGORITHM_CODE

# Import sys for platform check
import sys
