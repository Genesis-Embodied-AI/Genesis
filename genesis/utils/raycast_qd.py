import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu
from genesis.engine.bvh import STACK_SIZE
from genesis.engine.solvers.rigid.rigid_solver import func_update_all_verts


# FIXME: get_triangle_vertices/bvh_ray_cast/update_aabbs duplicate their visual counterparts below. The two paths
# differ only in the leaf-data fetch (fixed/free verts split vs vverts_state_idx + FK fallback) and in the dataclass
# shapes of geoms_state vs vgeoms_state. Quadrants does not currently support a qd.func arg that accepts either of
# two dataclasses with different field sets, so we cannot factor the BVH traversal into a single shared kernel; the
# kernel-call argument-fusion step strictly matches the annotated dataclass shape and rejects either union typing or
# qd.template() for dataclass-typed args. Until Quadrants gains generic-dataclass dispatch, the visual variants below
# stay as parallel copies.


@qd.func
def get_triangle_vertices(
    i_f: int,
    i_b: int,
    faces_info: array_class.FacesInfo,
    verts_info: array_class.VertsInfo,
    fixed_verts_state: array_class.VertsState,
    free_verts_state: array_class.VertsState,
):
    """
    Get the three vertices of a triangle in world space.

    Returns
    -------
    tri_vertices : qd.Matrix
        3x3 matrix where each column is a vertex position.
    """
    tri_vertices = qd.Matrix.zero(gs.qd_float, 3, 3)
    for i in qd.static(range(3)):
        i_v = faces_info.verts_idx[i_f][i]
        i_fv = verts_info.verts_state_idx[i_v]
        if verts_info.is_fixed[i_v]:
            tri_vertices[:, i] = fixed_verts_state.pos[i_fv]
        else:
            tri_vertices[:, i] = free_verts_state.pos[i_fv, i_b]
    return tri_vertices


@qd.func
def bvh_ray_cast(
    ray_start,
    ray_dir,
    max_range,
    i_b,
    bvh_nodes: qd.template(),
    bvh_morton_codes: qd.template(),
    faces_info: array_class.FacesInfo,
    verts_info: array_class.VertsInfo,
    fixed_verts_state: array_class.VertsState,
    free_verts_state: array_class.VertsState,
    eps,
):
    """
    Cast a ray through a BVH and find the closest intersection.

    Returns
    -------
    hit_face : gs.qd_int
        index of the hit triangle (-1 if no hit)
    hit_distance : gs.qd_float
        distance to hit point (unchanged max_range if no hit)
    hit_normal : qd.math.vec3
        normal vector at hit point (zero vector if no hit)
    """
    n_triangles = faces_info.verts_idx.shape[0]

    hit_face = -1
    closest_distance = gs.qd_float(max_range)
    hit_normal = qd.math.vec3(0.0, 0.0, 0.0)

    # Stack for non-recursive BVH traversal
    node_stack = qd.Vector.zero(gs.qd_int, qd.static(STACK_SIZE))
    node_stack[0] = 0  # Start at root node
    stack_idx = 1

    while stack_idx > 0:
        stack_idx -= 1
        node_idx = node_stack[stack_idx]

        node = bvh_nodes[i_b, node_idx]

        # Check if ray hits the node's bounding box
        aabb_t = ray_aabb_intersection(ray_start, ray_dir, node.bound.min, node.bound.max, eps)

        if aabb_t >= 0.0 and aabb_t < closest_distance:
            if node.left == -1:  # Leaf node
                # Get original triangle/face index
                sorted_leaf_idx = node_idx - (n_triangles - 1)
                i_f = qd.cast(bvh_morton_codes[i_b, sorted_leaf_idx][1], gs.qd_int)

                # Get triangle vertices
                tri_vertices = get_triangle_vertices(
                    i_f, i_b, faces_info, verts_info, fixed_verts_state, free_verts_state
                )
                v0, v1, v2 = tri_vertices[:, 0], tri_vertices[:, 1], tri_vertices[:, 2]

                # Perform ray-triangle intersection
                hit_result = ray_triangle_intersection(ray_start, ray_dir, v0, v1, v2, eps)

                if hit_result.w > 0.0 and hit_result.x < closest_distance and hit_result.x >= 0.0:
                    closest_distance = hit_result.x
                    hit_face = i_f
                    # Compute triangle normal
                    edge1 = v1 - v0
                    edge2 = v2 - v0
                    hit_normal = edge1.cross(edge2).normalized()
            else:  # Internal node
                # Push children onto stack
                if stack_idx < qd.static(STACK_SIZE - 2):
                    node_stack[stack_idx] = node.left
                    node_stack[stack_idx + 1] = node.right
                    stack_idx += 2

    return hit_face, closest_distance, hit_normal


@qd.func
def ray_triangle_intersection(
    ray_start: gs.qd_vec3,
    ray_dir: gs.qd_vec3,
    v0: gs.qd_vec3,
    v1: gs.qd_vec3,
    v2: gs.qd_vec3,
    eps: float,
):
    """
    Moller-Trumbore ray-triangle intersection.

    Returns
    -------
    result : qd.math.vec4
        (t, u, v, hit) where hit=1.0 if intersection found, 0.0 otherwise
    """
    result = qd.Vector.zero(gs.qd_float, 4)

    edge1 = v1 - v0
    edge2 = v2 - v0

    # Begin calculating determinant - also used to calculate u parameter
    h = ray_dir.cross(edge2)
    a = edge1.dot(h)

    # Check all conditions in sequence without early returns
    valid = True

    t = gs.qd_float(0.0)
    u = gs.qd_float(0.0)
    v = gs.qd_float(0.0)
    f = gs.qd_float(0.0)
    s = qd.Vector.zero(gs.qd_float, 3)
    q = qd.Vector.zero(gs.qd_float, 3)

    # If determinant is near zero, ray lies in plane of triangle
    if qd.abs(a) < eps:
        valid = False

    if valid:
        f = gs.qd_float(1.0) / a
        s = ray_start - v0
        u = f * s.dot(h)

        if u < 0.0 or u > 1.0:
            valid = False

    if valid:
        q = s.cross(edge1)
        v = f * ray_dir.dot(q)

        if v < 0.0 or u + v > 1.0:
            valid = False

    if valid:
        # At this stage we can compute t to find out where the intersection point is on the line
        t = f * edge2.dot(q)

        # Ray intersection
        if t <= eps:
            valid = False

    if valid:
        result = qd.math.vec4(t, u, v, gs.qd_float(1.0))

    return result


@qd.func
def ray_aabb_intersection(
    ray_start: gs.qd_vec3,
    ray_dir: gs.qd_vec3,
    aabb_min: gs.qd_vec3,
    aabb_max: gs.qd_vec3,
    eps: float,
):
    """
    Fast ray-AABB intersection test.
    Returns the t value of intersection, or -1.0 if no intersection.
    """
    result = -1.0

    # Use the slab method for ray-AABB intersection
    sign = qd.select(ray_dir >= 0.0, 1.0, -1.0)
    ray_dir = sign * qd.max(qd.abs(ray_dir), eps)
    inv_dir = 1.0 / ray_dir

    t1 = (aabb_min - ray_start) * inv_dir
    t2 = (aabb_max - ray_start) * inv_dir

    tmin = qd.min(t1, t2)
    tmax = qd.max(t1, t2)

    t_near = qd.max(tmin.x, tmin.y, tmin.z, gs.qd_float(0.0))
    t_far = qd.min(tmax.x, tmax.y, tmax.z)

    # Check if ray intersects AABB
    if t_near <= t_far:
        result = t_near

    return result


@qd.func
def update_aabbs(
    free_verts_state: array_class.VertsState,
    fixed_verts_state: array_class.VertsState,
    verts_info: array_class.VertsInfo,
    faces_info: array_class.FacesInfo,
    aabb_state: qd.template(),
):
    for i_b, i_f in qd.ndrange(free_verts_state.pos.shape[1], faces_info.verts_idx.shape[0]):
        aabb_state.aabbs[i_b, i_f].min.fill(qd.math.inf)
        aabb_state.aabbs[i_b, i_f].max.fill(-qd.math.inf)

        for i in qd.static(range(3)):
            i_v = faces_info.verts_idx[i_f][i]
            i_fv = verts_info.verts_state_idx[i_v]
            if verts_info.is_fixed[i_v]:
                pos_v = fixed_verts_state.pos[i_fv]
                aabb_state.aabbs[i_b, i_f].min = qd.min(aabb_state.aabbs[i_b, i_f].min, pos_v)
                aabb_state.aabbs[i_b, i_f].max = qd.max(aabb_state.aabbs[i_b, i_f].max, pos_v)
            else:
                pos_v = free_verts_state.pos[i_fv, i_b]
                aabb_state.aabbs[i_b, i_f].min = qd.min(aabb_state.aabbs[i_b, i_f].min, pos_v)
                aabb_state.aabbs[i_b, i_f].max = qd.max(aabb_state.aabbs[i_b, i_f].max, pos_v)


@qd.kernel
def kernel_update_verts_and_aabbs(
    geoms_info: array_class.GeomsInfo,
    geoms_state: array_class.GeomsState,
    verts_info: array_class.VertsInfo,
    faces_info: array_class.FacesInfo,
    free_verts_state: array_class.VertsState,
    fixed_verts_state: array_class.VertsState,
    static_rigid_sim_config: qd.template(),
    aabb_state: qd.template(),
):
    func_update_all_verts(
        geoms_state, geoms_info, verts_info, free_verts_state, fixed_verts_state, static_rigid_sim_config
    )
    update_aabbs(free_verts_state, fixed_verts_state, verts_info, faces_info, aabb_state)


# =========================================== Visual Mesh Raycasting ===========================================


@qd.func
def get_visual_vvert_pos(
    i_vv: int,
    i_b: int,
    vverts_info: array_class.VVertsInfo,
    vverts_state: array_class.VVertsState,
    vgeoms_state: array_class.VGeomsState,
):
    """
    Return the world-space position of a visual vertex, branching between the custom buffer and FK on the fly.

    Opt-in entities (morph.enable_custom_vverts=True) own a slot in vverts_state.pos referenced by vverts_state_idx.
    Non-opt-in entities (vverts_state_idx<0) have no slot; their position is recomputed by transforming the rest-pose
    init_pos with the owning vgeom's current pose.
    """
    pos = qd.math.vec3(0.0, 0.0, 0.0)
    i_state = vverts_info.vverts_state_idx[i_vv]
    if i_state >= 0:
        pos = vverts_state.pos[i_state, i_b]
    else:
        i_vg = vverts_info.vgeom_idx[i_vv]
        pos = gu.qd_transform_by_trans_quat(
            vverts_info.init_pos[i_vv], vgeoms_state.pos[i_vg, i_b], vgeoms_state.quat[i_vg, i_b]
        )
    return pos


@qd.func
def get_visual_triangle_vertices(
    i_f: int,
    i_b: int,
    vverts_info: array_class.VVertsInfo,
    vverts_state: array_class.VVertsState,
    vfaces_info: array_class.VFacesInfo,
    vgeoms_state: array_class.VGeomsState,
):
    """Get the three vertices of a triangle from the visual mesh in world space."""
    tri_vertices = qd.Matrix.zero(gs.qd_float, 3, 3)
    for i in qd.static(range(3)):
        i_vv = vfaces_info.vverts_idx[i_f][i]
        tri_vertices[:, i] = get_visual_vvert_pos(i_vv, i_b, vverts_info, vverts_state, vgeoms_state)
    return tri_vertices


@qd.func
def bvh_ray_cast_visual(
    ray_start,
    ray_dir,
    max_range,
    i_b,
    bvh_nodes: qd.template(),
    bvh_morton_codes: qd.template(),
    vverts_info: array_class.VVertsInfo,
    vverts_state: array_class.VVertsState,
    vfaces_info: array_class.VFacesInfo,
    vgeoms_state: array_class.VGeomsState,
    eps,
):
    """Cast a single ray against the visual-mesh BVH; returns (hit_face, distance, normal)."""
    n_triangles = vfaces_info.vverts_idx.shape[0]

    hit_face = -1
    closest_distance = gs.qd_float(max_range)
    hit_normal = qd.math.vec3(0.0, 0.0, 0.0)

    node_stack = qd.Vector.zero(gs.qd_int, qd.static(STACK_SIZE))
    node_stack[0] = 0
    stack_idx = 1

    while stack_idx > 0:
        stack_idx -= 1
        node_idx = node_stack[stack_idx]
        node = bvh_nodes[i_b, node_idx]

        aabb_t = ray_aabb_intersection(ray_start, ray_dir, node.bound.min, node.bound.max, eps)

        if aabb_t >= 0.0 and aabb_t < closest_distance:
            if node.left == -1:
                sorted_leaf_idx = node_idx - (n_triangles - 1)
                i_f = qd.cast(bvh_morton_codes[i_b, sorted_leaf_idx][1], gs.qd_int)

                tri_vertices = get_visual_triangle_vertices(
                    i_f, i_b, vverts_info, vverts_state, vfaces_info, vgeoms_state
                )
                v0, v1, v2 = tri_vertices[:, 0], tri_vertices[:, 1], tri_vertices[:, 2]

                hit_result = ray_triangle_intersection(ray_start, ray_dir, v0, v1, v2, eps)

                if hit_result.w > 0.0 and hit_result.x < closest_distance and hit_result.x >= 0.0:
                    closest_distance = hit_result.x
                    hit_face = i_f
                    edge1 = v1 - v0
                    edge2 = v2 - v0
                    hit_normal = edge1.cross(edge2).normalized()
            else:
                if stack_idx < qd.static(STACK_SIZE - 2):
                    node_stack[stack_idx] = node.left
                    node_stack[stack_idx + 1] = node.right
                    stack_idx += 2

    return hit_face, closest_distance, hit_normal


@qd.func
def update_visual_aabbs(
    vverts_info: array_class.VVertsInfo,
    vverts_state: array_class.VVertsState,
    vfaces_info: array_class.VFacesInfo,
    vgeoms_state: array_class.VGeomsState,
    face_mask: qd.types.ndarray(),
    aabb_state: qd.template(),
):
    """Update per-vface AABBs from the visual mesh. face_mask gates inclusion: 0 keeps the AABB inverted
    (unhittable) so vfaces from entities not opted into raycasting are skipped by ray queries."""
    _B = vgeoms_state.pos.shape[1]
    n_vfaces = vfaces_info.vverts_idx.shape[0]
    for i_b, i_f in qd.ndrange(_B, n_vfaces):
        aabb_state.aabbs[i_b, i_f].min.fill(qd.math.inf)
        aabb_state.aabbs[i_b, i_f].max.fill(-qd.math.inf)
        if face_mask[i_f] != 0:
            for i in qd.static(range(3)):
                i_vv = vfaces_info.vverts_idx[i_f][i]
                pos_v = get_visual_vvert_pos(i_vv, i_b, vverts_info, vverts_state, vgeoms_state)
                aabb_state.aabbs[i_b, i_f].min = qd.min(aabb_state.aabbs[i_b, i_f].min, pos_v)
                aabb_state.aabbs[i_b, i_f].max = qd.max(aabb_state.aabbs[i_b, i_f].max, pos_v)


@qd.kernel
def kernel_update_visual_aabbs(
    vverts_info: array_class.VVertsInfo,
    vverts_state: array_class.VVertsState,
    vfaces_info: array_class.VFacesInfo,
    vgeoms_state: array_class.VGeomsState,
    face_mask: qd.types.ndarray(),
    aabb_state: qd.template(),
):
    update_visual_aabbs(vverts_info, vverts_state, vfaces_info, vgeoms_state, face_mask, aabb_state)


# FIXME: Fastcache is not supported because of 'bvh_nodes', 'bvh_morton_codes'.
@qd.kernel(fastcache=False)
def kernel_cast_ray(
    fixed_verts_state: array_class.VertsState,
    free_verts_state: array_class.VertsState,
    verts_info: array_class.VertsInfo,
    faces_info: array_class.FacesInfo,
    bvh_nodes: qd.template(),
    bvh_morton_codes: qd.template(),
    ray_start: qd.types.ndarray(ndim=1),  # (3,)
    ray_direction: qd.types.ndarray(ndim=1),  # (3,)
    max_range: float,
    envs_idx: qd.types.ndarray(ndim=1),  # [n_envs]
    rigid_global_info: array_class.RigidGlobalInfo,
    result: array_class.RaycastResult,
    eps: float,
):
    """
    Cast a single ray against each env's BVH in parallel.

    Per-env: the ray is shifted by -envs_offset[i_b] (each BVH is in env-local coordinates) and the closest hit on
    that env is written to result[i_b]; envs not in envs_idx are left as no-hit (geom_idx == -1, distance == +inf).
    Aggregation across envs is intentionally out of scope, because cross-env reduction has no use beyond the viewer.
    """
    ray_start_world = qd.math.vec3(ray_start[0], ray_start[1], ray_start[2])
    ray_direction_world = qd.math.vec3(ray_direction[0], ray_direction[1], ray_direction[2])

    for i_b in range(result.geom_idx.shape[0]):
        result.distance[i_b] = qd.math.inf
        result.geom_idx[i_b] = -1
        result.hit_point[i_b] = qd.math.vec3(0.0, 0.0, 0.0)
        result.normal[i_b] = qd.math.vec3(0.0, 0.0, 0.0)

    for i_b_ in range(envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        env_offset = rigid_global_info.envs_offset[i_b]
        cur_hit_face, cur_distance, cur_hit_normal = bvh_ray_cast(
            ray_start=ray_start_world - env_offset,
            ray_dir=ray_direction_world,
            max_range=max_range,
            i_b=i_b,
            bvh_nodes=bvh_nodes,
            bvh_morton_codes=bvh_morton_codes,
            faces_info=faces_info,
            verts_info=verts_info,
            fixed_verts_state=fixed_verts_state,
            free_verts_state=free_verts_state,
            eps=eps,
        )
        if cur_hit_face >= 0:
            result.distance[i_b] = cur_distance
            result.geom_idx[i_b] = faces_info.geom_idx[cur_hit_face]
            result.normal[i_b] = cur_hit_normal
            result.hit_point[i_b] = ray_start_world + cur_distance * ray_direction_world


@qd.func
def write_ray_hit(
    hit_face: int,
    hit_distance: float,
    ray_start_world,
    ray_direction_world,
    ray_dir_local,
    i_b: int,
    i_s: int,
    i_p_sensor: int,
    i_p_offset: int,
    i_p_dist: int,
    is_world_frame: qd.types.ndarray(ndim=1),
    no_hit_values: qd.types.ndarray(ndim=1),
    output_hits: qd.types.ndarray(ndim=2),
    eps: float,
    is_merge: qd.template(),
):
    """Common post-BVH write block for both collision and visual cast kernels.

    `is_merge` is a compile-time flag. When False the function writes a value into every output slot (hit or
    no_hit_value), initializing the cache. When True the function only writes when it found a closer hit than what
    is already in the cache, so multiple BVH casts can be composed by chaining calls (first with is_merge=False,
    subsequent with is_merge=True) into the same output buffer with no scratch storage.
    """
    if hit_face >= 0 and (not is_merge or hit_distance < output_hits[i_p_dist, i_b]):
        # Store distance at: cache_offset + (num_points_in_sensor * 3) + point_idx_in_sensor
        output_hits[i_p_dist, i_b] = hit_distance

        hit_point = qd.math.vec3(0.0, 0.0, 0.0)
        if is_world_frame[i_s]:
            hit_point = ray_start_world + hit_distance * ray_direction_world
        else:
            # Local frame output along provided local ray direction
            hit_point = hit_distance * gu.qd_normalize(ray_dir_local, eps)
        # Store points at: cache_offset + point_idx_in_sensor * 3
        output_hits[i_p_offset + i_p_sensor * 3 + 0, i_b] = hit_point.x
        output_hits[i_p_offset + i_p_sensor * 3 + 1, i_b] = hit_point.y
        output_hits[i_p_offset + i_p_sensor * 3 + 2, i_b] = hit_point.z
    elif not is_merge:
        # No hit
        output_hits[i_p_offset + i_p_sensor * 3 + 0, i_b] = 0.0
        output_hits[i_p_offset + i_p_sensor * 3 + 1, i_b] = 0.0
        output_hits[i_p_offset + i_p_sensor * 3 + 2, i_b] = 0.0
        output_hits[i_p_dist, i_b] = no_hit_values[i_s]


@qd.kernel
def kernel_cast_rays(
    fixed_verts_state: array_class.VertsState,
    free_verts_state: array_class.VertsState,
    verts_info: array_class.VertsInfo,
    faces_info: array_class.FacesInfo,
    bvh_nodes: qd.template(),
    bvh_morton_codes: qd.template(),  # maps sorted leaves to original triangle indices
    links_pos: qd.types.ndarray(ndim=3),  # [n_env, n_sensors, 3]
    links_quat: qd.types.ndarray(ndim=3),  # [n_env, n_sensors, 4]
    ray_starts: qd.types.ndarray(ndim=2),  # [n_points, 3]
    ray_directions: qd.types.ndarray(ndim=2),  # [n_points, 3]
    max_ranges: qd.types.ndarray(ndim=1),  # [n_sensors]
    no_hit_values: qd.types.ndarray(ndim=1),  # [n_sensors]
    is_world_frame: qd.types.ndarray(ndim=1),  # [n_sensors]
    points_to_sensor_idx: qd.types.ndarray(ndim=1),  # [n_points]
    sensor_cache_offsets: qd.types.ndarray(ndim=1),  # [n_sensors] - cache start index for each sensor
    sensor_point_offsets: qd.types.ndarray(ndim=1),  # [n_sensors] - point start index for each sensor
    sensor_point_counts: qd.types.ndarray(ndim=1),  # [n_sensors] - number of points for each sensor
    output_hits: qd.types.ndarray(ndim=2),  # [total_cache_size, n_env]
    eps: float,
    is_merge: qd.template(),
):
    """Cast rays against a collision-mesh BVH, accelerated by a BVH. See write_ray_hit for `is_merge` semantics.

    The result `output_hits` is a 2D array of shape (total_cache_size, n_env) where in the first dimension each
    sensor's data is stored as [sensor_points (n_points * 3), sensor_ranges (n_points)].
    """
    n_points = ray_starts.shape[0]
    for i_p, i_b in qd.ndrange(n_points, output_hits.shape[-1]):
        i_s = points_to_sensor_idx[i_p]

        link_pos = qd.math.vec3(links_pos[i_b, i_s, 0], links_pos[i_b, i_s, 1], links_pos[i_b, i_s, 2])
        link_quat = qd.math.vec4(
            links_quat[i_b, i_s, 0], links_quat[i_b, i_s, 1], links_quat[i_b, i_s, 2], links_quat[i_b, i_s, 3]
        )

        ray_start_local = qd.math.vec3(ray_starts[i_p, 0], ray_starts[i_p, 1], ray_starts[i_p, 2])
        ray_start_world = gu.qd_transform_by_trans_quat(ray_start_local, link_pos, link_quat)

        ray_dir_local = qd.math.vec3(ray_directions[i_p, 0], ray_directions[i_p, 1], ray_directions[i_p, 2])
        ray_direction_world = gu.qd_normalize(gu.qd_transform_by_quat(ray_dir_local, link_quat), eps)

        hit_face, hit_distance, _hit_normal = bvh_ray_cast(
            ray_start=ray_start_world,
            ray_dir=ray_direction_world,
            max_range=max_ranges[i_s],
            i_b=i_b,
            bvh_nodes=bvh_nodes,
            bvh_morton_codes=bvh_morton_codes,
            faces_info=faces_info,
            verts_info=verts_info,
            fixed_verts_state=fixed_verts_state,
            free_verts_state=free_verts_state,
            eps=eps,
        )

        i_p_sensor = i_p - sensor_point_offsets[i_s]
        i_p_offset = sensor_cache_offsets[i_s]
        i_p_dist = i_p_offset + sensor_point_counts[i_s] * 3 + i_p_sensor
        write_ray_hit(
            hit_face,
            hit_distance,
            ray_start_world,
            ray_direction_world,
            ray_dir_local,
            i_b,
            i_s,
            i_p_sensor,
            i_p_offset,
            i_p_dist,
            is_world_frame,
            no_hit_values,
            output_hits,
            eps,
            is_merge,
        )


@qd.kernel
def kernel_cast_rays_visual(
    vverts_info: array_class.VVertsInfo,
    vverts_state: array_class.VVertsState,
    vfaces_info: array_class.VFacesInfo,
    vgeoms_state: array_class.VGeomsState,
    bvh_nodes: qd.template(),
    bvh_morton_codes: qd.template(),
    links_pos: qd.types.ndarray(ndim=3),
    links_quat: qd.types.ndarray(ndim=3),
    ray_starts: qd.types.ndarray(ndim=2),
    ray_directions: qd.types.ndarray(ndim=2),
    max_ranges: qd.types.ndarray(ndim=1),
    no_hit_values: qd.types.ndarray(ndim=1),
    is_world_frame: qd.types.ndarray(ndim=1),
    points_to_sensor_idx: qd.types.ndarray(ndim=1),
    sensor_cache_offsets: qd.types.ndarray(ndim=1),
    sensor_point_offsets: qd.types.ndarray(ndim=1),
    sensor_point_counts: qd.types.ndarray(ndim=1),
    output_hits: qd.types.ndarray(ndim=2),
    eps: float,
    is_merge: qd.template(),
):
    """Visual-mesh variant of kernel_cast_rays."""
    n_points = ray_starts.shape[0]
    for i_p, i_b in qd.ndrange(n_points, output_hits.shape[-1]):
        i_s = points_to_sensor_idx[i_p]

        link_pos = qd.math.vec3(links_pos[i_b, i_s, 0], links_pos[i_b, i_s, 1], links_pos[i_b, i_s, 2])
        link_quat = qd.math.vec4(
            links_quat[i_b, i_s, 0], links_quat[i_b, i_s, 1], links_quat[i_b, i_s, 2], links_quat[i_b, i_s, 3]
        )

        ray_start_local = qd.math.vec3(ray_starts[i_p, 0], ray_starts[i_p, 1], ray_starts[i_p, 2])
        ray_start_world = gu.qd_transform_by_trans_quat(ray_start_local, link_pos, link_quat)

        ray_dir_local = qd.math.vec3(ray_directions[i_p, 0], ray_directions[i_p, 1], ray_directions[i_p, 2])
        ray_direction_world = gu.qd_normalize(gu.qd_transform_by_quat(ray_dir_local, link_quat), eps)

        hit_face, hit_distance, _hit_normal = bvh_ray_cast_visual(
            ray_start=ray_start_world,
            ray_dir=ray_direction_world,
            max_range=max_ranges[i_s],
            i_b=i_b,
            bvh_nodes=bvh_nodes,
            bvh_morton_codes=bvh_morton_codes,
            vverts_info=vverts_info,
            vverts_state=vverts_state,
            vfaces_info=vfaces_info,
            vgeoms_state=vgeoms_state,
            eps=eps,
        )

        i_p_sensor = i_p - sensor_point_offsets[i_s]
        i_p_offset = sensor_cache_offsets[i_s]
        i_p_dist = i_p_offset + sensor_point_counts[i_s] * 3 + i_p_sensor
        write_ray_hit(
            hit_face,
            hit_distance,
            ray_start_world,
            ray_direction_world,
            ray_dir_local,
            i_b,
            i_s,
            i_p_sensor,
            i_p_offset,
            i_p_dist,
            is_world_frame,
            no_hit_values,
            output_hits,
            eps,
            is_merge,
        )
