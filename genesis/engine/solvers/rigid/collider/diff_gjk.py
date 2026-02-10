import gstaichi as ti
import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.array_class as array_class


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
    rigid_global_info: array_class.RigidGlobalInfo,
):
    EPS = rigid_global_info.EPS[None]

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
    rot = gu.ti_quat_to_R(links_state.i_quat[i_l, i_b], EPS)
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


# ------------------------------ Differentiable functions ------------------------------------
# These functions have minimal number of branches to align backward pass with forward pass.
# --------------------------------------------------------------------------------------------
@ti.func
def func_differentiable_contact(
    geoms_state: array_class.GeomsState,
    diff_contact_input: array_class.DiffContactInput,
    gjk_info: array_class.GJKInfo,
    i_ga,
    i_gb,
    i_b,
    i_c,
    ref_penetration,
):
    """
    Compute the contact normal, penetration, and point for contact [i_c] from the corresponding [diff_contact_input]
    in a differentiable way. The gradients flow through the position and quaternion stored in the [geoms_state].
    """
    eps_B = gjk_info.diff_contact_eps_boundary[None]
    eps_D = gjk_info.diff_contact_eps_distance[None]

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
    local_pos1_a = diff_contact_input.local_pos1_a[i_b, i_c]
    local_pos1_b = diff_contact_input.local_pos1_b[i_b, i_c]
    local_pos1_c = diff_contact_input.local_pos1_c[i_b, i_c]
    local_pos2_a = diff_contact_input.local_pos2_a[i_b, i_c]
    local_pos2_b = diff_contact_input.local_pos2_b[i_b, i_c]
    local_pos2_c = diff_contact_input.local_pos2_c[i_b, i_c]

    # Support points of the contact
    w_local_pos1 = diff_contact_input.w_local_pos1[i_b, i_c]
    w_local_pos2 = diff_contact_input.w_local_pos2[i_b, i_c]

    # Compute global positions of the vertices
    pos1a = gu.ti_transform_by_trans_quat(local_pos1_a, trans1, quat1)
    pos1b = gu.ti_transform_by_trans_quat(local_pos1_b, trans1, quat1)
    pos1c = gu.ti_transform_by_trans_quat(local_pos1_c, trans1, quat1)
    pos2a = gu.ti_transform_by_trans_quat(local_pos2_a, trans2, quat2)
    pos2b = gu.ti_transform_by_trans_quat(local_pos2_b, trans2, quat2)
    pos2c = gu.ti_transform_by_trans_quat(local_pos2_c, trans2, quat2)

    # Compute the vertices on the Minkowski difference
    mink1 = pos1a - pos2a
    mink2 = pos1b - pos2b
    mink3 = pos1c - pos2c
    face_center = (mink1 + mink2 + mink3) / 3.0

    ### Compute the witness points on the two geometries.
    # Compute the normal of the face
    normal = func_plane_normal(mink1, mink2, mink3)

    # Project the origin onto the affine plane of the face: This operation is guaranteed to be numerically stable, as
    # the normal length is guaranteed to be larger than the minimum normal norm in [gjk_info].
    proj_o = func_project_origin_to_plane(mink1, mink2, mink3, normal)

    # Compute the affine coordinates of the origin's projection on the face: This operation is also guaranteed to be
    # numerically stable, as the normal length is guaranteed to be larger than the minimum normal norm in
    # [gjk_info].
    _lambda = func_triangle_affine_coords(mink1, mink2, mink3, normal, proj_o)

    # Point on geom 1
    w1 = pos1a * _lambda[0] + pos1b * _lambda[1] + pos1c * _lambda[2]
    # Point on geom 2
    w2 = pos2a * _lambda[0] + pos2b * _lambda[1] + pos2c * _lambda[2]

    ### Compute contact position, normal, and penetration depth. These operations are guaranteed to be numerically stable,
    ### especially the normalization of the contact normal, as the penetration depth is guaranteed to be larger than the
    ### minimum penetration depth in [gjk_info].
    contact_pos = 0.5 * (w1 + w2)
    contact_normal = (w2 - w1).normalized()
    penetration = (w2 - w1).norm()

    ### Compute weight of the contact point.
    face_normal = normal
    if normal.dot(face_center) < 0.0:
        face_normal = -normal
    face_normal = face_normal.normalized()

    w_pos1 = gu.ti_transform_by_trans_quat(w_local_pos1, trans1, quat1)
    w_pos2 = gu.ti_transform_by_trans_quat(w_local_pos2, trans2, quat2)
    w = w_pos1 - w_pos2

    # Boundary weight
    bsdist = ti.max(w.dot(face_normal) - face_center.dot(face_normal), 0.0)
    boundary_weight = 1.0 - ti.math.clamp(bsdist / eps_B, 0.0, 1.0)

    # Distance weight
    distance_weight = 1.0
    if ref_penetration >= 0.0:
        distance_weight = 1.0 - ti.math.clamp((penetration - ref_penetration) / eps_D, 0.0, 1.0)

    # Affine weight: Theoretically we need this, but in practice it could cause instability
    # FIXME: Can we stabilize it?
    # affine_weight_0 = 1.0 - ti.math.clamp(ti.max(0.0 - _lambda[0], _lambda[0] - 1.0) / eps_A, 0.0, 1.0)
    # affine_weight_1 = 1.0 - ti.math.clamp(ti.max(0.0 - _lambda[1], _lambda[1] - 1.0) / eps_A, 0.0, 1.0)
    # affine_weight_2 = 1.0 - ti.math.clamp(ti.max(0.0 - _lambda[2], _lambda[2] - 1.0) / eps_A, 0.0, 1.0)
    # affine_weight = (affine_weight_0 + affine_weight_1 + affine_weight_2) / 3.0
    affine_weight = 1.0

    # Compute final weight
    weight = affine_weight * distance_weight * boundary_weight

    return contact_pos, contact_normal, penetration, weight


@ti.func
def func_plane_normal(v1, v2, v3):
    """
    Compute the normal of the plane defined by three points. The length of the normal corresponds to the two times the
    area of the triangle.
    """
    d21 = v2 - v1
    d31 = v3 - v1
    normal = d21.cross(d31)
    return normal


@ti.func
def func_project_origin_to_plane(v1, v2, v3, normal):
    """
    Project the origin onto the plane defined by the simplex vertices.

    @ normal: The face normal computed as (v2 - v1) x (v3 - v1). Its length should be guaranteed to be larger than the
    minimum normal norm in [gjk_info], but we do not check it here.
    """
    # Since normal norm is guaranteed to be larger than sqrt(10 * EPS), [nn] is guaranteed to be larger than 10 * EPS.
    v = v1
    nv = normal.dot(v)
    nn = normal.norm_sqr()
    return normal * (nv / nn)


@ti.func
def func_triangle_affine_coords(v1, v2, v3, normal, point):
    """
    Compute the affine coordinates of the point with respect to the triangle.

    @ normal: The face normal computed as (v2 - v1) x (v3 - v1). Its length should be guaranteed to be larger than the
    minimum normal norm in [gjk_info], but we do not check it here.
    @ point: The point on the plane that we want to compute the affine coordinates of.
    """
    # Since normal norm is guaranteed to be larger than sqrt(10 * EPS), [nn] is guaranteed to be larger than 10 * EPS.
    nn = normal.norm_sqr()
    inv_nn = 1.0 / nn

    return gs.ti_vec3(
        (v2 - point).cross(v3 - point).dot(normal) * inv_nn,
        (v3 - point).cross(v1 - point).dot(normal) * inv_nn,
        (v1 - point).cross(v2 - point).dot(normal) * inv_nn,
    )


from genesis.utils.deprecated_module_wrapper import create_virtual_deprecated_module

create_virtual_deprecated_module(__name__, "genesis.engine.solvers.rigid.diff_gjk_decomp")
