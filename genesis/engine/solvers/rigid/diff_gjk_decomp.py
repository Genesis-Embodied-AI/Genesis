import gstaichi as ti
import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.array_class as array_class
import genesis.engine.solvers.rigid.gjk_decomp as GJK


@ti.func
def func_gjk_contact(
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    geoms_init_AABB: array_class.GeomsInitAABB,
    verts_info: array_class.VertsInfo,
    faces_info: array_class.FacesInfo,
    static_rigid_sim_config: ti.template(),
    collider_state: array_class.ColliderState,
    collider_static_config: ti.template(),
    gjk_state: array_class.GJKState,
    gjk_static_config: ti.template(),
    support_field_info: array_class.SupportFieldInfo,
    support_field_static_config: ti.template(),
    i_ga,
    i_gb,
    i_b,
    pos_tol,
    normal_tol,
):
    """
    Detect multiple possible contact points between two geometries using GJK and EPA algorithms.

    We first run the GJK algorithm to find the minimum distance between the two geometries. If the distance is
    smaller than the collision epsilon, we consider the geometries colliding. If they are colliding, we run the extended
    EPA algorithm in the original configuration and the original EPA algorithm for multiple perturbed configurations.
    In this way, we can find multiple possible contact points for differentiable contact detection.
    """
    # Clear the cache to prepare for this GJK-EPA run.
    GJK.clear_cache(gjk_state, i_b)

    gjk_state.n_diff_contact_data[i_b] = 0

    # Backup state before local perturbation
    ga_pos, ga_quat = geoms_state.pos[i_ga, i_b], geoms_state.quat[i_ga, i_b]
    gb_pos, gb_quat = geoms_state.pos[i_gb, i_b], geoms_state.quat[i_gb, i_b]

    # Axis to rotate the geometry for perturbation
    axis_0 = ti.Vector.zero(gs.ti_float, 3)
    axis_1 = ti.Vector.zero(gs.ti_float, 3)

    # Default contact point and penetration in the original configuration
    default_contact_pos = gs.ti_vec3(0.0, 0.0, 0.0)
    default_penetration = gs.ti_float(0.0)
    found_default_epa = False

    # 4 (small) + 4 (large) perturbated configurations
    num_perturb = 8

    for i in range(1 + num_perturb):
        # First iteration: Detect contact in default configuration.
        # 2,3,4,5: Detect contacts in slightly perturbed configuration.
        # 6,7,8,9: Detect contacts in more perturbed configuration.
        if i > 0:
            # Convert back to the default configuration
            geoms_state.pos[i_ga, i_b], geoms_state.quat[i_ga, i_b] = ga_pos, ga_quat
            geoms_state.pos[i_gb, i_b], geoms_state.quat[i_gb, i_b] = gb_pos, gb_quat

            # Perturbation axis must not be aligned with the principal axes of inertia the geometry,
            # otherwise it would be more sensitive to ill-conditionning.
            axis = (2 * (i % 2) - 1) * axis_0 + (1 - 2 * ((i // 2) % 2)) * axis_1
            rotang = 1e-2 * (100 ** ((i - 1) // 4))
            qrot = gu.ti_rotvec_to_quat(rotang * axis)
            func_rotate_frame(i_ga, default_contact_pos, qrot, i_b, geoms_state, geoms_info)
            func_rotate_frame(i_gb, default_contact_pos, gu.ti_inv_quat(qrot), i_b, geoms_state, geoms_info)

        gjk_flag = GJK.func_safe_gjk(
            geoms_state,
            geoms_info,
            verts_info,
            static_rigid_sim_config,
            collider_state,
            collider_static_config,
            gjk_state,
            gjk_static_config,
            support_field_info,
            support_field_static_config,
            i_ga,
            i_gb,
            i_b,
        )

        if gjk_flag == GJK.GJK_RETURN_CODE.INTERSECT:
            # Initialize polytope
            gjk_state.polytope.nverts[i_b] = 0
            gjk_state.polytope.nfaces[i_b] = 0
            gjk_state.polytope.nfaces_map[i_b] = 0
            gjk_state.polytope.horizon_nedges[i_b] = 0

            # Construct the initial polytope from the GJK simplex
            GJK.func_safe_epa_init(
                gjk_state,
                gjk_static_config,
                i_ga,
                i_gb,
                i_b,
            )

            if i == 0:
                # In default configuration, we use the extended EPA algorithm to find multiple contact points.
                max_epa_iter = gjk_static_config.epa_max_iterations
                while max_epa_iter > 0:
                    i_f, num_iter = func_extended_epa(
                        geoms_state,
                        geoms_info,
                        verts_info,
                        static_rigid_sim_config,
                        collider_state,
                        collider_static_config,
                        gjk_state,
                        gjk_static_config,
                        support_field_info,
                        support_field_static_config,
                        i_ga,
                        i_gb,
                        i_b,
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
                        penetration > default_penetration + gjk_static_config.diff_contact_eps_distance
                    ):
                        continue

                    # Add face information for differentiable operation later
                    func_add_diff_contact_data(
                        geoms_state,
                        geoms_info,
                        verts_info,
                        static_rigid_sim_config,
                        collider_state,
                        collider_static_config,
                        gjk_state,
                        gjk_static_config,
                        support_field_info,
                        support_field_static_config,
                        i_ga,
                        i_gb,
                        i_b,
                        i_f,
                    )

                    if not found_default_epa:
                        # Very small contact, ignore it
                        if penetration < gjk_static_config.FLOAT_MIN:
                            gjk_state.n_diff_contact_data[i_b] = 0
                            break
                        default_contact_pos = 0.5 * (witness1 + witness2)
                        default_penetration = penetration

                        axis_0, axis_1 = func_contact_orthogonals(
                            i_ga,
                            i_gb,
                            normal / penetration,
                            i_b,
                            links_state,
                            links_info,
                            geoms_state,
                            geoms_info,
                            geoms_init_AABB,
                        )

                        found_default_epa = True

                    # Break the loop if we found enough contact points for default configuration. As we can find at most
                    # 8 contact points for perturbed configurations, we can find at most max_contacts_per_pair - 8
                    # contact points for default configuration.
                    if gjk_state.n_diff_contact_data[i_b] >= (gjk_static_config.max_contacts_per_pair - num_perturb):
                        break

                # If we failed to find the default contact point, we do not add any contact point.
                if not found_default_epa:
                    break
            else:
                i_f = GJK.func_safe_epa(
                    geoms_state,
                    geoms_info,
                    verts_info,
                    static_rigid_sim_config,
                    collider_state,
                    collider_static_config,
                    gjk_state,
                    gjk_static_config,
                    support_field_info,
                    support_field_static_config,
                    i_ga,
                    i_gb,
                    i_b,
                )
                if i_f == -1:
                    continue

                # Convert back to the default configuration
                geoms_state.pos[i_ga, i_b], geoms_state.quat[i_ga, i_b] = ga_pos, ga_quat
                geoms_state.pos[i_gb, i_b], geoms_state.quat[i_gb, i_b] = gb_pos, gb_quat

                # Add face information for differentiable operation later
                func_add_diff_contact_data(
                    geoms_state,
                    geoms_info,
                    verts_info,
                    static_rigid_sim_config,
                    collider_state,
                    collider_static_config,
                    gjk_state,
                    gjk_static_config,
                    support_field_info,
                    support_field_static_config,
                    i_ga,
                    i_gb,
                    i_b,
                    i_f,
                )

        elif i == 0:
            # If there was no intersection at the default configuration, we do not add any contact point.
            break

    # Convert back to the default configuration
    geoms_state.pos[i_ga, i_b], geoms_state.quat[i_ga, i_b] = ga_pos, ga_quat
    geoms_state.pos[i_gb, i_b], geoms_state.quat[i_gb, i_b] = gb_pos, gb_quat

    # Compute the final contact points and normals.
    n_contacts = 0
    for i_c in range(gjk_state.n_diff_contact_data[i_b]):
        ref_penetration = -1.0
        if i_c > 0:
            ref_penetration = default_penetration
        contact_pos, contact_normal, penetration, weight, flag = func_differentiable_contact(
            geoms_state, gjk_state.diff_contact_data, gjk_static_config, i_ga, i_gb, i_b, i_c, ref_penetration
        )
        if i_c == 0:
            default_penetration = penetration
        if flag == GJK.RETURN_CODE.SUCCESS and weight > 0.0:
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

            # Update the contact information
            gjk_state.contact_pos[i_b, insert_id] = contact_pos
            gjk_state.normal[i_b, insert_id] = contact_normal
            gjk_state.diff_penetration[i_b, insert_id] = diff_penetration

            gjk_state.diff_contact_data.localpos1a[i_b, insert_id] = gjk_state.diff_contact_data.localpos1a[i_b, i_c]
            gjk_state.diff_contact_data.localpos1b[i_b, insert_id] = gjk_state.diff_contact_data.localpos1b[i_b, i_c]
            gjk_state.diff_contact_data.localpos1c[i_b, insert_id] = gjk_state.diff_contact_data.localpos1c[i_b, i_c]
            gjk_state.diff_contact_data.localpos2a[i_b, insert_id] = gjk_state.diff_contact_data.localpos2a[i_b, i_c]
            gjk_state.diff_contact_data.localpos2b[i_b, insert_id] = gjk_state.diff_contact_data.localpos2b[i_b, i_c]
            gjk_state.diff_contact_data.localpos2c[i_b, insert_id] = gjk_state.diff_contact_data.localpos2c[i_b, i_c]
            gjk_state.diff_contact_data.w_localpos1[i_b, insert_id] = gjk_state.diff_contact_data.w_localpos1[i_b, i_c]
            gjk_state.diff_contact_data.w_localpos2[i_b, insert_id] = gjk_state.diff_contact_data.w_localpos2[i_b, i_c]
            gjk_state.diff_contact_data.normal[i_b, insert_id] = gjk_state.diff_contact_data.normal[i_b, i_c]
            gjk_state.diff_contact_data.ref_id[i_b, insert_id] = 0

            if insert_id == n_contacts:
                n_contacts += 1

            if n_contacts >= gjk_static_config.max_contacts_per_pair:
                break
        elif i_c == 0:
            # If the first contact point is not valid, we do not add any contact point.
            break

    gjk_state.n_contacts[i_b] = n_contacts
    gjk_state.is_col[i_b] = 1 if n_contacts > 0 else 0
    gjk_state.multi_contact_flag[i_b] = 1


@ti.func
def func_extended_epa(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    static_rigid_sim_config: ti.template(),
    collider_state: array_class.ColliderState,
    collider_static_config: ti.template(),
    gjk_state: array_class.GJKState,
    gjk_static_config: ti.template(),
    support_field_info: array_class.SupportFieldInfo,
    support_field_static_config: ti.template(),
    i_ga,
    i_gb,
    i_b,
    max_iter,
):
    """
    Extended version of safe EPA algorithm to find multiple possible contact points for differentiable contact detection.

    While the original safe EPA algorithm finds the farthest boundary face on the Minkowski difference, this function
    can be used to find nearly-farthest boundary faces on the Minkowski difference. When the configurations of the
    objects are slightly perturbed, the farthest boundary face could change to one of the nearly-farthest boundary faces.
    Therefore, we use this function to find such nearly-farthest boundary faces for differentiability.
    """
    tolerance = gjk_static_config.tolerance
    nearest_i_f = gs.ti_int(-1)

    discrete = GJK.func_is_discrete_geoms(geoms_info, i_ga, i_gb, i_b)
    if discrete:
        # If the objects are discrete, we do not use tolerance.
        tolerance = gs.EPS

    k = 0
    converged = False
    while k < max_iter:
        k += 1

        # Find the polytope face with the smallest distance to the origin
        lower2 = gjk_static_config.FLOAT_MAX_SQ
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
        wi = GJK.func_epa_support(
            geoms_state,
            geoms_info,
            verts_info,
            static_rigid_sim_config,
            collider_state,
            collider_static_config,
            gjk_state,
            gjk_static_config,
            support_field_info,
            support_field_static_config,
            i_ga,
            i_gb,
            i_b,
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
        horizon_flag = GJK.func_epa_horizon(gjk_state, gjk_static_config, i_b, nearest_i_f)

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
        if nfaces + nedges >= gjk_static_config.polytope_max_faces:
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

            attach_flag = GJK.func_safe_attach_face_to_polytope(
                gjk_state,
                gjk_static_config,
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
        flag = GJK.func_safe_epa_witness(gjk_state, gjk_static_config, i_ga, i_gb, i_b, nearest_i_f)
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
def func_add_diff_contact_data(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    static_rigid_sim_config: ti.template(),
    collider_state: array_class.ColliderState,
    collider_static_config: ti.template(),
    gjk_state: array_class.GJKState,
    gjk_static_config: ti.template(),
    support_field_info: array_class.SupportFieldInfo,
    support_field_static_config: ti.template(),
    i_ga,
    i_gb,
    i_b,
    i_f,
):
    n = gjk_state.n_diff_contact_data[i_b]

    i_v1 = gjk_state.polytope_faces.verts_idx[i_b, i_f][0]
    i_v2 = gjk_state.polytope_faces.verts_idx[i_b, i_f][1]
    i_v3 = gjk_state.polytope_faces.verts_idx[i_b, i_f][2]

    # Find support point in the default configuration
    mink1 = gs.ti_vec3(0.0, 0.0, 0.0)
    mink2 = gs.ti_vec3(0.0, 0.0, 0.0)
    mink3 = gs.ti_vec3(0.0, 0.0, 0.0)

    for i in ti.static(range(3)):
        curr_i_v = i_v1
        if i == 1:
            curr_i_v = i_v2
        elif i == 2:
            curr_i_v = i_v3

        mink = func_compute_minkowski_point(
            geoms_state.pos[i_ga, i_b],
            geoms_state.quat[i_ga, i_b],
            geoms_state.pos[i_gb, i_b],
            geoms_state.quat[i_gb, i_b],
            gjk_state.polytope_verts.local_obj1[i_b, curr_i_v],
            gjk_state.polytope_verts.local_obj2[i_b, curr_i_v],
        )
        if i == 0:
            mink1 = mink
        elif i == 1:
            mink2 = mink
        elif i == 2:
            mink3 = mink

    normal, flag = GJK.func_plane_normal(gjk_static_config, mink1, mink2, mink3)
    if flag == GJK.RETURN_CODE.SUCCESS:
        face_center = (mink1 + mink2 + mink3) / 3.0

        min_bdist = gjk_static_config.FLOAT_MAX
        min_bdist_normal = gs.ti_vec3(0.0, 0.0, 0.0)
        min_bdist_w_localpos1 = gs.ti_vec3(0.0, 0.0, 0.0)
        min_bdist_w_localpos2 = gs.ti_vec3(0.0, 0.0, 0.0)

        # Find the support points in the [normal, -normal] direction
        for i in range(2):
            d = normal
            if i == 1:
                d = -normal

            obj1, obj2, localpos1, localpos2, id1, id2, mink = GJK.func_support(
                geoms_state,
                geoms_info,
                verts_info,
                static_rigid_sim_config,
                collider_state,
                collider_static_config,
                gjk_state,
                gjk_static_config,
                support_field_info,
                support_field_static_config,
                i_ga,
                i_gb,
                i_b,
                d,
                False,
            )

            # Compute boundary signed distance in this direction
            bdist = mink.dot(d) - face_center.dot(d)
            if bdist < min_bdist:
                min_bdist = bdist
                min_bdist_normal = d
                min_bdist_w_localpos1 = localpos1
                min_bdist_w_localpos2 = localpos2

        gjk_state.diff_contact_data.localpos1a[i_b, n] = gjk_state.polytope_verts.local_obj1[i_b, i_v1]
        gjk_state.diff_contact_data.localpos1b[i_b, n] = gjk_state.polytope_verts.local_obj1[i_b, i_v2]
        gjk_state.diff_contact_data.localpos1c[i_b, n] = gjk_state.polytope_verts.local_obj1[i_b, i_v3]
        gjk_state.diff_contact_data.localpos2a[i_b, n] = gjk_state.polytope_verts.local_obj2[i_b, i_v1]
        gjk_state.diff_contact_data.localpos2b[i_b, n] = gjk_state.polytope_verts.local_obj2[i_b, i_v2]
        gjk_state.diff_contact_data.localpos2c[i_b, n] = gjk_state.polytope_verts.local_obj2[i_b, i_v3]
        gjk_state.diff_contact_data.vid1[i_b, n][0] = gjk_state.polytope_verts.id1[i_b, i_v1]
        gjk_state.diff_contact_data.vid1[i_b, n][1] = gjk_state.polytope_verts.id1[i_b, i_v2]
        gjk_state.diff_contact_data.vid1[i_b, n][2] = gjk_state.polytope_verts.id1[i_b, i_v3]
        gjk_state.diff_contact_data.vid2[i_b, n][0] = gjk_state.polytope_verts.id2[i_b, i_v1]
        gjk_state.diff_contact_data.vid2[i_b, n][1] = gjk_state.polytope_verts.id2[i_b, i_v2]
        gjk_state.diff_contact_data.vid2[i_b, n][2] = gjk_state.polytope_verts.id2[i_b, i_v3]
        gjk_state.diff_contact_data.w_localpos1[i_b, n] = min_bdist_w_localpos1
        gjk_state.diff_contact_data.w_localpos2[i_b, n] = min_bdist_w_localpos2
        gjk_state.diff_contact_data.normal[i_b, n] = min_bdist_normal
        gjk_state.n_diff_contact_data[i_b] += 1


@ti.func
def func_contact_orthogonals(
    i_ga,
    i_gb,
    normal: ti.types.vector(3),
    i_b,
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    geoms_init_AABB: array_class.GeomsInitAABB,
):
    axis_0 = ti.Vector.zero(gs.ti_float, 3)
    axis_1 = ti.Vector.zero(gs.ti_float, 3)

    # The reference geometry is the one that will have the largest impact on the position of
    # the contact point. Basically, the smallest one between the two, which can be approximated
    # by the volume of their respective bounding box.
    i_g = i_gb
    if geoms_info.type[i_ga] != gs.GEOM_TYPE.PLANE:
        size_ga = geoms_init_AABB[i_ga, 7]
        volume_ga = size_ga[0] * size_ga[1] * size_ga[2]
        size_gb = geoms_init_AABB[i_gb, 7]
        volume_gb = size_gb[0] * size_gb[1] * size_gb[2]
        i_g = i_ga if volume_ga < volume_gb else i_gb

    # Compute orthogonal basis mixing principal inertia axes of geometry with contact normal
    i_l = geoms_info.link_idx[i_g]
    rot = gu.ti_quat_to_R(links_state.i_quat[i_l, i_b])
    axis_idx = gs.ti_int(0)
    axis_angle_max = gs.ti_float(0.0)
    for i in ti.static(range(3)):
        axis_angle = ti.abs(rot[:, i].dot(normal))
        if axis_angle > axis_angle_max:
            axis_angle_max = axis_angle
            axis_idx = i
    axis_idx = (axis_idx + 1) % 3
    axis_0 = rot[:, axis_idx]
    axis_0 = (axis_0 - normal.dot(axis_0) * normal).normalized()
    axis_1 = normal.cross(axis_0)

    return axis_0, axis_1


@ti.func
def func_rotate_frame(
    i_g,
    contact_pos: ti.types.vector(3),
    qrot: ti.types.vector(4),
    i_b,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
):
    geoms_state.quat[i_g, i_b] = gu.ti_transform_quat_by_quat(geoms_state.quat[i_g, i_b], qrot)

    rel = contact_pos - geoms_state.pos[i_g, i_b]
    vec = gu.ti_transform_by_quat(rel, qrot)
    vec = vec - rel
    geoms_state.pos[i_g, i_b] = geoms_state.pos[i_g, i_b] - vec


@ti.func
def func_differentiable_contact(
    geoms_state: array_class.GeomsState,
    diff_contact_data: array_class.DiffContactData,
    gjk_static_config: ti.template(),
    i_ga,
    i_gb,
    i_b,
    i_c,
    ref_penetration,
):
    """
    Compute the contact normal, penetration, and point for contact [i_c] in a differentiable way.

    The gradients flow through the position and quaternion stored in the geoms_state.
    """
    flag = GJK.RETURN_CODE.SUCCESS
    eps_B = gjk_static_config.diff_contact_eps_boundary
    eps_D = gjk_static_config.diff_contact_eps_distance
    eps_A = gjk_static_config.diff_contact_eps_affine

    # Result
    contact_pos = gs.ti_vec3(0.0, 0.0, 0.0)
    contact_normal = gs.ti_vec3(0.0, 0.0, 0.0)
    penetration = gs.ti_float(0.0)
    weight = gs.ti_float(0.0)

    # Transformations of the geometries
    trans1 = geoms_state.pos[i_ga, i_b]
    trans2 = geoms_state.pos[i_gb, i_b]
    quat1 = geoms_state.quat[i_ga, i_b]
    quat2 = geoms_state.quat[i_gb, i_b]

    # Local positions of the vertices that form the contact
    local_pos1a = diff_contact_data.localpos1a[i_b, i_c]
    local_pos1b = diff_contact_data.localpos1b[i_b, i_c]
    local_pos1c = diff_contact_data.localpos1c[i_b, i_c]
    local_pos2a = diff_contact_data.localpos2a[i_b, i_c]
    local_pos2b = diff_contact_data.localpos2b[i_b, i_c]
    local_pos2c = diff_contact_data.localpos2c[i_b, i_c]

    # Normal of the contact used as a reference for the normal selection
    ref_normal = diff_contact_data.normal[i_b, i_c]

    # Support points of the contact
    w_localpos1 = diff_contact_data.w_localpos1[i_b, i_c]
    w_localpos2 = diff_contact_data.w_localpos2[i_b, i_c]

    # Compute global positions of the vertices
    pos1a = gu.ti_transform_by_trans_quat(local_pos1a, trans1, quat1)
    pos1b = gu.ti_transform_by_trans_quat(local_pos1b, trans1, quat1)
    pos1c = gu.ti_transform_by_trans_quat(local_pos1c, trans1, quat1)
    pos2a = gu.ti_transform_by_trans_quat(local_pos2a, trans2, quat2)
    pos2b = gu.ti_transform_by_trans_quat(local_pos2b, trans2, quat2)
    pos2c = gu.ti_transform_by_trans_quat(local_pos2c, trans2, quat2)

    # Compute the vertices on the Minkowski difference
    mink1 = pos1a - pos2a
    mink2 = pos1b - pos2b
    mink3 = pos1c - pos2c

    # Project the origin onto the affine plane of the face
    proj_o, flag = GJK.func_project_origin_to_plane(gjk_static_config, mink1, mink2, mink3)

    if flag == GJK.RETURN_CODE.SUCCESS:
        _lambda = GJK.func_triangle_affine_coords(proj_o, mink1, mink2, mink3)

        if ti.math.isnan(_lambda[0]) or ti.math.isnan(_lambda[1]) or ti.math.isnan(_lambda[2]):
            flag = GJK.RETURN_CODE.FAIL
        elif ti.math.isinf(_lambda[0]) or ti.math.isinf(_lambda[1]) or ti.math.isinf(_lambda[2]):
            flag = GJK.RETURN_CODE.FAIL
        else:
            # Check validity of affine coordinates through reprojection
            proj_o_lambda = mink1 * _lambda[0] + mink2 * _lambda[1] + mink3 * _lambda[2]
            reprojection_error = (proj_o - proj_o_lambda).norm()

            # Take into account the face magnitude, as the error is relative to the face size.
            max_edge_len_inv = ti.rsqrt(
                max(
                    (mink1 - mink2).norm_sqr(),
                    (mink2 - mink3).norm_sqr(),
                    (mink3 - mink1).norm_sqr(),
                    gjk_static_config.FLOAT_MIN_SQ,
                )
            )
            rel_reprojection_error = reprojection_error * max_edge_len_inv
            if rel_reprojection_error > gjk_static_config.polytope_max_reprojection_error:
                flag = GJK.RETURN_CODE.FAIL

        if flag == GJK.RETURN_CODE.SUCCESS:
            # Point on geom 1
            w1 = pos1a * _lambda[0] + pos1b * _lambda[1] + pos1c * _lambda[2]
            # Point on geom 2
            w2 = pos2a * _lambda[0] + pos2b * _lambda[1] + pos2c * _lambda[2]

            # Contact position, normal, and penetration depth
            contact_pos = 0.5 * (w1 + w2)
            contact_normal = w2 - w1
            penetration = contact_normal.norm()
            if penetration > gjk_static_config.FLOAT_MIN:
                contact_normal = contact_normal / penetration

                # Compute weight for the penetration depth ---> Differentiable
                # Boundary weight: Compute boundary signed distance to the face
                _normal, flag = GJK.func_plane_normal(gjk_static_config, mink1, mink2, mink3)

                if flag == GJK.RETURN_CODE.SUCCESS:
                    if _normal.dot(ref_normal) < 0.0:
                        _normal = -_normal

                    face_center = (mink1 + mink2 + mink3) / 3.0

                    w_pos1 = gu.ti_transform_by_trans_quat(w_localpos1, trans1, quat1)
                    w_pos2 = gu.ti_transform_by_trans_quat(w_localpos2, trans2, quat2)
                    w = w_pos1 - w_pos2

                    bsdist = ti.max(w.dot(_normal) - face_center.dot(_normal), 0.0)
                    boundary_weight = 1.0 - ti.math.clamp(bsdist / eps_B, 0.0, 1.0)

                    # Distance weight
                    distance_weight = 1.0
                    if ref_penetration >= 0.0:
                        distance_weight = 1.0 - ti.math.clamp((penetration - ref_penetration) / eps_D, 0.0, 1.0)

                    # Affine weight
                    affine_weight_0 = 1.0 - ti.math.clamp(ti.max(0.0 - _lambda[0], _lambda[0] - 1.0) / eps_A, 0.0, 1.0)
                    affine_weight_1 = 1.0 - ti.math.clamp(ti.max(0.0 - _lambda[1], _lambda[1] - 1.0) / eps_A, 0.0, 1.0)
                    affine_weight_2 = 1.0 - ti.math.clamp(ti.max(0.0 - _lambda[2], _lambda[2] - 1.0) / eps_A, 0.0, 1.0)
                    affine_weight = (affine_weight_0 + affine_weight_1 + affine_weight_2) / 3.0

                    # Compute final weight
                    weight = affine_weight * distance_weight * boundary_weight
            else:
                flag = GJK.RETURN_CODE.FAIL

    return contact_pos, contact_normal, penetration, weight, flag


@ti.func
def func_compute_minkowski_point(
    ga_pos: ti.types.vector(3),
    ga_quat: ti.types.vector(4),
    gb_pos: ti.types.vector(3),
    gb_quat: ti.types.vector(4),
    va: ti.types.vector(3),
    vb: ti.types.vector(3),
):
    # Transform the points to the global frame
    va_ = gu.ti_transform_by_trans_quat(va, ga_pos, ga_quat)
    vb_ = gu.ti_transform_by_trans_quat(vb, gb_pos, gb_quat)
    return va_ - vb_
