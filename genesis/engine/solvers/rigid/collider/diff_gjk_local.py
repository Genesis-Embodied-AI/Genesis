"""
Thread-local versions of differentiable GJK collision detection functions.

This module provides versions of differentiable GJK functions that accept pos/quat
as direct parameters instead of reading from/writing to geoms_state, enabling
race-free multi-contact detection when parallelizing across collision pairs within
the same environment.
"""

import gstaichi as ti

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.solvers.rigid.collider import diff_gjk, epa, gjk as GJK, gjk_local, multi_contact
from genesis.engine.solvers.rigid.collider.contact_local import func_rotate_frame_local
from genesis.utils import array_class


@ti.func
def func_gjk_contact_local(
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    geoms_init_AABB: array_class.GeomsInitAABB,
    verts_info: array_class.VertsInfo,
    faces_info: array_class.FacesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    collider_state: array_class.ColliderState,
    collider_static_config: ti.template(),
    gjk_state: array_class.GJKState,
    gjk_info: array_class.GJKInfo,
    support_field_info: array_class.SupportFieldInfo,
    # FIXME: Passing nested data structure as input argument is not supported for now.
    diff_contact_input: array_class.DiffContactInput,
    i_ga,
    i_gb,
    i_b,
    ga_pos: ti.types.vector(3, dtype=gs.ti_float),
    ga_quat: ti.types.vector(4, dtype=gs.ti_float),
    gb_pos: ti.types.vector(3, dtype=gs.ti_float),
    gb_quat: ti.types.vector(4, dtype=gs.ti_float),
    pos_tol,
    normal_tol,
):
    """
    Thread-local version of func_gjk_contact.

    Detect multiple possible contact points between two geometries using GJK and EPA algorithms,
    and compute weights of the contact points for differentiability.

    Instead of modifying geoms_state directly, this function uses thread-local pos/quat variables
    for geometry perturbations, enabling race-free parallelization across collision pairs.

    Args:
        ... (standard GJK parameters)
        ga_pos: Thread-local position of geometry A
        ga_quat: Thread-local quaternion of geometry A
        gb_pos: Thread-local position of geometry B
        gb_quat: Thread-local quaternion of geometry B
        pos_tol: Position tolerance for duplicate contact detection
        normal_tol: Normal tolerance for duplicate contact detection

    For detailed algorithm description, see func_gjk_contact in diff_gjk.py.
    """
    EPS = rigid_global_info.EPS[None]

    # Clear the cache to prepare for this GJK-EPA run.
    GJK.clear_cache(gjk_state, i_b)

    gjk_state.n_diff_contact_input[i_b] = 0

    # Thread-local copies for perturbation (stored in registers, not global state)
    ga_pos_local = ga_pos
    ga_quat_local = ga_quat
    gb_pos_local = gb_pos
    gb_quat_local = gb_quat

    # Axis to rotate the geometry for perturbation
    axis_0 = ti.Vector.zero(gs.ti_float, 3)
    axis_1 = ti.Vector.zero(gs.ti_float, 3)

    # Default contact point and penetration in the original configuration
    default_contact_pos = gs.ti_vec3(0.0, 0.0, 0.0)
    default_penetration = gs.ti_float(0.0)
    found_default_epa = False

    # 4 (small) + 4 (large) perturbated configurations
    num_perturb = 8

    ### Detect multiple possible contact points and gather the non-differentiable contact data.
    for i in range(1 + num_perturb):
        # First iteration: Detect contact in default configuration.
        # 2,3,4,5: Detect contacts in slightly perturbed configuration.
        # 6,7,8,9: Detect contacts in more perturbed configuration.
        if i > 0:
            # Reset to the default configuration
            ga_pos_local = ga_pos
            ga_quat_local = ga_quat
            gb_pos_local = gb_pos
            gb_quat_local = gb_quat

            # Perturbation axis must not be aligned with the principal axes of inertia the geometry,
            # otherwise it would be more sensitive to ill-conditioning.
            axis = (2 * (i % 2) - 1) * axis_0 + (1 - 2 * ((i // 2) % 2)) * axis_1
            rotang = 1e-2 * (100 ** ((i - 1) // 4))
            qrot = gu.ti_rotvec_to_quat(rotang * axis, EPS)

            # Apply perturbation to local variables
            result_a = func_rotate_frame_local(ga_pos_local, ga_quat_local, default_contact_pos, qrot)
            ga_pos_local = result_a.pos
            ga_quat_local = result_a.quat

            result_b = func_rotate_frame_local(gb_pos_local, gb_quat_local, default_contact_pos, gu.ti_inv_quat(qrot))
            gb_pos_local = result_b.pos
            gb_quat_local = result_b.quat

        gjk_flag = gjk_local.func_safe_gjk_local(
            geoms_info,
            verts_info,
            rigid_global_info,
            static_rigid_sim_config,
            collider_state,
            collider_static_config,
            gjk_state,
            gjk_info,
            support_field_info,
            i_ga,
            i_gb,
            ga_pos_local,
            ga_quat_local,
            gb_pos_local,
            gb_quat_local,
            i_b,
        )

        if gjk_flag == GJK.GJK_RETURN_CODE.INTERSECT:
            # Initialize polytope
            gjk_state.polytope.nverts[i_b] = 0
            gjk_state.polytope.nfaces[i_b] = 0
            gjk_state.polytope.nfaces_map[i_b] = 0
            gjk_state.polytope.horizon_nedges[i_b] = 0

            # Construct the initial polytope from the GJK simplex
            epa.func_safe_epa_init(
                gjk_state,
                gjk_info,
                i_ga,
                i_gb,
                i_b,
            )

            if i == 0:
                # In default configuration, we use the extended EPA algorithm to find multiple contact points.
                max_epa_iter = gjk_info.epa_max_iterations[None]
                while max_epa_iter > 0:
                    i_f, num_iter = func_extended_epa_local(
                        geoms_info,
                        verts_info,
                        rigid_global_info,
                        static_rigid_sim_config,
                        collider_state,
                        collider_static_config,
                        gjk_state,
                        gjk_info,
                        support_field_info,
                        i_ga,
                        i_gb,
                        i_b,
                        ga_pos_local,
                        ga_quat_local,
                        gb_pos_local,
                        gb_quat_local,
                        max_epa_iter,
                    )
                    max_epa_iter -= num_iter

                    if i_f == -1:
                        break

                    # Mark the face as visited
                    gjk_state.polytope_faces.visited[i_b, i_f] = 1

                    # Compute penetration depth
                    witness1 = gjk_state.witness.point_obj1[i_b, 0]
                    witness2 = gjk_state.witness.point_obj2[i_b, 0]

                    normal = witness2 - witness1
                    penetration = normal.norm()

                    # If the penetration depth is larger than the (default EPA depth + eps), we can ignore this contact
                    # because the weight of the contact point would be 0.
                    if found_default_epa and (
                        penetration > default_penetration + gjk_info.diff_contact_eps_distance[None]
                    ):
                        continue

                    # Add input data for differentiable contact detection
                    func_add_diff_contact_input_local(
                        geoms_info,
                        verts_info,
                        static_rigid_sim_config,
                        collider_state,
                        collider_static_config,
                        gjk_state,
                        gjk_info,
                        support_field_info,
                        i_ga,
                        i_gb,
                        i_b,
                        ga_pos_local,
                        ga_quat_local,
                        gb_pos_local,
                        gb_quat_local,
                        i_f,
                    )

                    if not found_default_epa:
                        # If the default contact is already numerically unstable, we do not add any contact point.
                        if gjk_state.diff_contact_input.valid[i_b, 0] == 0:
                            gjk_state.n_diff_contact_input[i_b] = 0
                            break
                        default_contact_pos = 0.5 * (witness1 + witness2)
                        default_penetration = penetration

                        axis_0, axis_1 = diff_gjk.func_contact_orthogonals(
                            i_ga,
                            i_gb,
                            normal / penetration,
                            i_b,
                            links_state,
                            links_info,
                            geoms_state,
                            geoms_info,
                            geoms_init_AABB,
                            rigid_global_info,
                        )

                        found_default_epa = True

                    # Break the loop if we found enough contact points for default configuration. As we can find at most
                    # 8 contact points for perturbed configurations, we can find at most max_contacts_per_pair - 8
                    # contact points for default configuration.
                    if gjk_state.n_diff_contact_input[i_b] >= (gjk_info.max_contacts_per_pair[None] - num_perturb):
                        break

                # If we failed to find the default contact point, we do not add any contact point.
                if not found_default_epa:
                    break
            else:
                i_f = epa_local.func_safe_epa_local(
                    geoms_info,
                    verts_info,
                    rigid_global_info,
                    static_rigid_sim_config,
                    collider_state,
                    collider_static_config,
                    gjk_state,
                    gjk_info,
                    support_field_info,
                    i_ga,
                    i_gb,
                    ga_pos_local,
                    ga_quat_local,
                    gb_pos_local,
                    gb_quat_local,
                    i_b,
                )
                if i_f == -1:
                    continue

                # Add input data for differentiable contact detection
                # Use original (non-perturbed) positions for storing local vertex positions
                func_add_diff_contact_input_local(
                    geoms_info,
                    verts_info,
                    static_rigid_sim_config,
                    collider_state,
                    collider_static_config,
                    gjk_state,
                    gjk_info,
                    support_field_info,
                    i_ga,
                    i_gb,
                    i_b,
                    ga_pos,
                    ga_quat,
                    gb_pos,
                    gb_quat,
                    i_f,
                )

        elif i == 0:
            # If there was no intersection at the default configuration, we do not add any contact point.
            break

    ### Compute the differentiable contact data from the non-differentiable data.
    n_contacts = 0
    for i_c in range(gjk_state.n_diff_contact_input[i_b]):
        # We ignore the contact point if it is not numerically stable.
        if gjk_state.diff_contact_input.valid[i_b, i_c] == 0:
            continue

        # Compute the differentiable contact data.
        ref_penetration = -1.0
        if i_c > 0:
            ref_penetration = default_penetration
        contact_pos, contact_normal, penetration, weight = diff_gjk.func_differentiable_contact(
            geoms_state, diff_contact_input, gjk_info, i_ga, i_gb, i_b, i_c, ref_penetration
        )
        if i_c == 0:
            default_penetration = penetration

        # We ignore the contact point if the weight is 0.
        if weight == 0.0:
            if i_c == 0:
                # This will not happen, but we keep this for safety.
                break
            else:
                continue

        diff_penetration = penetration * weight

        # Check if there is any duplicate contact point.
        duplicate_id = -1
        for i_c2 in range(n_contacts):
            prev_contact_pos = gjk_state.contact_pos[i_b, i_c2]
            prev_contact_normal = gjk_state.normal[i_b, i_c2]

            if (contact_pos - prev_contact_pos).norm() > pos_tol:
                continue
            if contact_normal.dot(prev_contact_normal) < (1.0 - normal_tol):
                continue
            duplicate_id = i_c2
            break

        insert_id = n_contacts
        if duplicate_id != -1:
            # If it is duplicate and the prev. penetration depth is smaller, we replace the duplicate contact point.
            if gjk_state.diff_penetration[i_b, duplicate_id] < diff_penetration:
                insert_id = duplicate_id
            else:
                continue

        if i_c > 0 and insert_id == 0:
            # We keep the first contact point as the reference contact point.
            continue

        # Update the differentiable contact data.
        gjk_state.contact_pos[i_b, insert_id] = contact_pos
        gjk_state.normal[i_b, insert_id] = contact_normal
        gjk_state.diff_penetration[i_b, insert_id] = diff_penetration

        # Update the non-differentiable contact data for the backward pass.
        gjk_state.diff_contact_input.local_pos1_a[i_b, insert_id] = gjk_state.diff_contact_input.local_pos1_a[i_b, i_c]
        gjk_state.diff_contact_input.local_pos1_b[i_b, insert_id] = gjk_state.diff_contact_input.local_pos1_b[i_b, i_c]
        gjk_state.diff_contact_input.local_pos1_c[i_b, insert_id] = gjk_state.diff_contact_input.local_pos1_c[i_b, i_c]
        gjk_state.diff_contact_input.local_pos2_a[i_b, insert_id] = gjk_state.diff_contact_input.local_pos2_a[i_b, i_c]
        gjk_state.diff_contact_input.local_pos2_b[i_b, insert_id] = gjk_state.diff_contact_input.local_pos2_b[i_b, i_c]
        gjk_state.diff_contact_input.local_pos2_c[i_b, insert_id] = gjk_state.diff_contact_input.local_pos2_c[i_b, i_c]
        gjk_state.diff_contact_input.w_local_pos1[i_b, insert_id] = gjk_state.diff_contact_input.w_local_pos1[i_b, i_c]
        gjk_state.diff_contact_input.w_local_pos2[i_b, insert_id] = gjk_state.diff_contact_input.w_local_pos2[i_b, i_c]
        gjk_state.diff_contact_input.ref_id[i_b, insert_id] = 0
        if insert_id == 0:
            gjk_state.diff_contact_input.ref_penetration[i_b, insert_id] = penetration

        if insert_id == n_contacts:
            n_contacts += 1

        if n_contacts >= gjk_info.max_contacts_per_pair[None]:
            break

    gjk_state.n_contacts[i_b] = n_contacts
    gjk_state.is_col[i_b] = n_contacts > 0
    gjk_state.multi_contact_flag[i_b] = True


@ti.func
def func_extended_epa_local(
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    collider_state: array_class.ColliderState,
    collider_static_config: ti.template(),
    gjk_state: array_class.GJKState,
    gjk_info: array_class.GJKInfo,
    support_field_info: array_class.SupportFieldInfo,
    i_ga,
    i_gb,
    i_b,
    pos_a: ti.types.vector(3, dtype=gs.ti_float),
    quat_a: ti.types.vector(4, dtype=gs.ti_float),
    pos_b: ti.types.vector(3, dtype=gs.ti_float),
    quat_b: ti.types.vector(4, dtype=gs.ti_float),
    max_iter,
):
    """
    Thread-local version of func_extended_epa.

    Extended version of safe EPA algorithm to find multiple possible contact points
    for differentiable contact detection. Uses thread-local pos/quat instead of
    reading from geoms_state.

    Returns:
        nearest_i_f: Face index of the nearest face (-1 if none found)
        k: Number of iterations performed
    """
    tolerance = gjk_info.tolerance[None]
    nearest_i_f = gs.ti_int(-1)

    discrete = GJK.func_is_discrete_geoms(geoms_info, i_ga, i_gb)
    if discrete:
        # If the objects are discrete, we do not use tolerance.
        tolerance = rigid_global_info.EPS[None]

    k = 0
    converged = False
    while k < max_iter:
        k += 1

        # Find the polytope face with the smallest distance to the origin
        lower2 = gjk_info.FLOAT_MAX_SQ[None]
        nearest_i_f = -1

        for i in range(gjk_state.polytope.nfaces_map[i_b]):
            i_f = gjk_state.polytope_faces_map[i_b, i]
            if gjk_state.polytope_faces.visited[i_b, i_f] == 1:
                continue

            face_dist2 = gjk_state.polytope_faces.dist2[i_b, i_f]
            if face_dist2 < lower2:
                lower2 = face_dist2
                nearest_i_f = i_f

        if nearest_i_f == -1:
            break

        # Find a new support point w from the nearest face's normal
        lower = ti.sqrt(lower2)
        dir = gjk_state.polytope_faces.normal[i_b, nearest_i_f]
        wi = func_epa_support_local(
            geoms_info,
            verts_info,
            static_rigid_sim_config,
            collider_state,
            collider_static_config,
            gjk_state,
            gjk_info,
            support_field_info,
            i_ga,
            i_gb,
            i_b,
            pos_a,
            quat_a,
            pos_b,
            quat_b,
            dir,
            1.0,
        )
        w = gjk_state.polytope_verts.mink[i_b, wi]

        # The upper bound of depth at k-th iteration
        upper = w.dot(dir)

        # If the upper bound and lower bound are close enough, we can stop the algorithm
        if (upper - lower) < tolerance:
            converged = True
            break

        if discrete:
            repeated = False
            for i in range(gjk_state.polytope.nverts[i_b]):
                if i == wi:
                    continue
                elif (
                    gjk_state.polytope_verts.id1[i_b, i] == gjk_state.polytope_verts.id1[i_b, wi]
                    and gjk_state.polytope_verts.id2[i_b, i] == gjk_state.polytope_verts.id2[i_b, wi]
                ):
                    # The vertex w is already in the polytope, so we do not need to add it again.
                    repeated = True
                    break
            if repeated:
                nearest_i_f = -1
                break

        gjk_state.polytope.horizon_w[i_b] = w

        # Compute horizon
        horizon_flag = epa.func_epa_horizon(gjk_state, gjk_info, i_b, nearest_i_f)

        if horizon_flag:
            # There was an error in the horizon construction, so the horizon edge is not a closed loop.
            nearest_i_f = -1
            break

        if gjk_state.polytope.horizon_nedges[i_b] < 3:
            # Should not happen, because at least three edges should be in the horizon from one deleted face.
            nearest_i_f = -1
            break

        # Check if the memory space is enough for attaching new faces
        nfaces = gjk_state.polytope.nfaces[i_b]
        nedges = gjk_state.polytope.horizon_nedges[i_b]
        if nfaces + nedges >= gjk_info.polytope_max_faces[None]:
            # If the polytope is full, we cannot insert new faces
            nearest_i_f = -1
            break

        # Attach the new faces
        attach_flag = GJK.RETURN_CODE.SUCCESS
        for i in range(nedges):
            # Face id of the current face to attach
            i_f0 = nfaces + i
            # Face id of the next face to attach
            i_f1 = nfaces + (i + 1) % nedges

            horizon_i_f = gjk_state.polytope_horizon_data.face_idx[i_b, i]
            horizon_i_e = gjk_state.polytope_horizon_data.edge_idx[i_b, i]

            horizon_v1 = gjk_state.polytope_faces.verts_idx[i_b, horizon_i_f][horizon_i_e]
            horizon_v2 = gjk_state.polytope_faces.verts_idx[i_b, horizon_i_f][(horizon_i_e + 1) % 3]

            # Change the adjacent face index of the existing face
            gjk_state.polytope_faces.adj_idx[i_b, horizon_i_f][horizon_i_e] = i_f0

            # Attach the new face.
            # If this if the first face, will be adjacent to the face that will be attached last.
            adj_i_f_0 = i_f0 - 1 if (i > 0) else nfaces + nedges - 1
            adj_i_f_1 = horizon_i_f
            adj_i_f_2 = i_f1

            attach_flag = epa.func_safe_attach_face_to_polytope(
                gjk_state,
                gjk_info,
                i_b,
                wi,
                horizon_v2,
                horizon_v1,
                adj_i_f_2,  # Previous face id
                adj_i_f_1,
                adj_i_f_0,  # Next face id
            )
            if attach_flag != GJK.RETURN_CODE.SUCCESS:
                # Unrecoverable numerical issue
                break

            # Store face in the map
            nfaces_map = gjk_state.polytope.nfaces_map[i_b]
            gjk_state.polytope_faces_map[i_b, nfaces_map] = i_f0
            gjk_state.polytope_faces.map_idx[i_b, i_f0] = nfaces_map
            gjk_state.polytope.nfaces_map[i_b] += 1

        if attach_flag != GJK.RETURN_CODE.SUCCESS:
            nearest_i_f = -1
            break

        # Clear the horizon data for the next iteration
        gjk_state.polytope.horizon_nedges[i_b] = 0

        if (gjk_state.polytope.nfaces_map[i_b] == 0) or (nearest_i_f == -1):
            # No face candidate left
            nearest_i_f = -1
            break

    if converged:
        # Remove the last vertex from the polytope, because it was added because of this boundary face
        gjk_state.polytope.nverts[i_b] -= 1
    else:
        nearest_i_f = -1

    if nearest_i_f != -1:
        # Nearest face found
        dist2 = gjk_state.polytope_faces.dist2[i_b, nearest_i_f]
        flag = epa.func_safe_epa_witness(gjk_state, gjk_info, i_ga, i_gb, i_b, nearest_i_f)
        if flag == GJK.RETURN_CODE.SUCCESS:
            gjk_state.n_witness[i_b] = 1
            gjk_state.distance[i_b] = -ti.sqrt(dist2)
        else:
            # Failed to compute witness points, so the objects are not colliding
            gjk_state.n_witness[i_b] = 0
            gjk_state.distance[i_b] = 0.0
            nearest_i_f = -1
    else:
        # No face found, so the objects are not colliding
        gjk_state.n_witness[i_b] = 0
        gjk_state.distance[i_b] = 0.0

    return nearest_i_f, k


@ti.func
def func_epa_support_local(
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    static_rigid_sim_config: ti.template(),
    collider_state: array_class.ColliderState,
    collider_static_config: ti.template(),
    gjk_state: array_class.GJKState,
    gjk_info: array_class.GJKInfo,
    support_field_info: array_class.SupportFieldInfo,
    i_ga,
    i_gb,
    i_b,
    pos_a: ti.types.vector(3, dtype=gs.ti_float),
    quat_a: ti.types.vector(4, dtype=gs.ti_float),
    pos_b: ti.types.vector(3, dtype=gs.ti_float),
    quat_b: ti.types.vector(4, dtype=gs.ti_float),
    dir,
    dir_scale,
):
    """
    Thread-local version of func_epa_support.

    Find a support point for EPA and add it to the polytope.
    Uses thread-local pos/quat instead of reading from geoms_state.

    Returns:
        i_v: Index of the newly added vertex in the polytope
    """
    obj1, obj2, localpos1, localpos2, id1, id2, mink = gjk_local.func_support_local(
        geoms_info,
        verts_info,
        static_rigid_sim_config,
        collider_state,
        collider_static_config,
        gjk_state,
        gjk_info,
        support_field_info,
        i_ga,
        i_gb,
        i_b,
        dir * dir_scale,
        pos_a,
        quat_a,
        pos_b,
        quat_b,
        False,
    )

    i_v = gjk_state.polytope.nverts[i_b]
    gjk_state.polytope_verts.obj1[i_b, i_v] = obj1
    gjk_state.polytope_verts.obj2[i_b, i_v] = obj2
    gjk_state.polytope_verts.local_obj1[i_b, i_v] = localpos1
    gjk_state.polytope_verts.local_obj2[i_b, i_v] = localpos2
    gjk_state.polytope_verts.mink[i_b, i_v] = mink
    gjk_state.polytope_verts.id1[i_b, i_v] = id1
    gjk_state.polytope_verts.id2[i_b, i_v] = id2
    gjk_state.polytope.nverts[i_b] += 1

    return i_v


@ti.func
def func_add_diff_contact_input_local(
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    static_rigid_sim_config: ti.template(),
    collider_state: array_class.ColliderState,
    collider_static_config: ti.template(),
    gjk_state: array_class.GJKState,
    gjk_info: array_class.GJKInfo,
    support_field_info: array_class.SupportFieldInfo,
    i_ga,
    i_gb,
    i_b,
    pos_a: ti.types.vector(3, dtype=gs.ti_float),
    quat_a: ti.types.vector(4, dtype=gs.ti_float),
    pos_b: ti.types.vector(3, dtype=gs.ti_float),
    quat_b: ti.types.vector(4, dtype=gs.ti_float),
    i_f,
):
    """
    Thread-local version of func_add_diff_contact_input.

    Prepare the (non-differentiable) contact data that can be used for computing
    the differentiable contact data later. Uses thread-local pos/quat instead of
    reading from geoms_state.
    """
    n = gjk_state.n_diff_contact_input[i_b]

    i_v1 = gjk_state.polytope_faces.verts_idx[i_b, i_f][0]
    i_v2 = gjk_state.polytope_faces.verts_idx[i_b, i_f][1]
    i_v3 = gjk_state.polytope_faces.verts_idx[i_b, i_f][2]

    # Define the face (possibly) on the boundary of the Minkowski difference in the default configuration
    mink1 = gs.ti_vec3(0.0, 0.0, 0.0)
    mink2 = gs.ti_vec3(0.0, 0.0, 0.0)
    mink3 = gs.ti_vec3(0.0, 0.0, 0.0)

    for i in ti.static(range(3)):
        curr_i_v = i_v1
        if i == 1:
            curr_i_v = i_v2
        elif i == 2:
            curr_i_v = i_v3

        mink = diff_gjk.func_compute_minkowski_point(
            pos_a,
            quat_a,
            pos_b,
            quat_b,
            gjk_state.polytope_verts.local_obj1[i_b, curr_i_v],
            gjk_state.polytope_verts.local_obj2[i_b, curr_i_v],
        )
        if i == 0:
            mink1 = mink
        elif i == 1:
            mink2 = mink
        elif i == 2:
            mink3 = mink

    ### Check validity of this contact. The contact is valid if the contact information could be computed in numerically
    ### stable way in both the forward and backward pass.
    # (a) Check if the face is degenerate.
    normal = diff_gjk.func_plane_normal(mink1, mink2, mink3)
    normal_norm = normal.norm()
    is_face_degenerate = normal_norm < gjk_info.diff_contact_min_normal_norm[None]

    # (b) Check if the origin is very close to the face (which means very small penetration depth).
    proj_o = diff_gjk.func_project_origin_to_plane(mink1, mink2, mink3, normal)
    origin_dist = proj_o.norm()
    is_origin_close_to_face = origin_dist < gjk_info.diff_contact_min_penetration[None]

    ### Orient the face normal, so that it points to the other side of the origin.
    face_center = (mink1 + mink2 + mink3) / 3.0
    if normal_norm > gjk_info.FLOAT_MIN[None]:
        normal = normal.normalized()
    if normal.dot(face_center) < 0.0:
        normal = -normal

    ### Compute the support point along the face normal.
    obj1, obj2, localpos1, localpos2, id1, id2, mink = gjk_local.func_support_local(
        geoms_info,
        verts_info,
        static_rigid_sim_config,
        collider_state,
        collider_static_config,
        gjk_state,
        gjk_info,
        support_field_info,
        i_ga,
        i_gb,
        i_b,
        normal,
        pos_a,
        quat_a,
        pos_b,
        quat_b,
        False,
    )

    gjk_state.diff_contact_input.local_pos1_a[i_b, n] = gjk_state.polytope_verts.local_obj1[i_b, i_v1]
    gjk_state.diff_contact_input.local_pos1_b[i_b, n] = gjk_state.polytope_verts.local_obj1[i_b, i_v2]
    gjk_state.diff_contact_input.local_pos1_c[i_b, n] = gjk_state.polytope_verts.local_obj1[i_b, i_v3]
    gjk_state.diff_contact_input.local_pos2_a[i_b, n] = gjk_state.polytope_verts.local_obj2[i_b, i_v1]
    gjk_state.diff_contact_input.local_pos2_b[i_b, n] = gjk_state.polytope_verts.local_obj2[i_b, i_v2]
    gjk_state.diff_contact_input.local_pos2_c[i_b, n] = gjk_state.polytope_verts.local_obj2[i_b, i_v3]
    gjk_state.diff_contact_input.w_local_pos1[i_b, n] = localpos1
    gjk_state.diff_contact_input.w_local_pos2[i_b, n] = localpos2
    gjk_state.diff_contact_input.valid[i_b, n] = not (is_face_degenerate or is_origin_close_to_face)
    gjk_state.n_diff_contact_input[i_b] += 1
