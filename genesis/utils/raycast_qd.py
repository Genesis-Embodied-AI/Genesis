import math
from typing import TYPE_CHECKING

import quadrants as qd
import numpy as np

import genesis as gs
import genesis.utils.array_class as array_class
from genesis.engine.bvh import AABB, LBVH, STACK_SIZE
from genesis.engine.solvers.rigid.rigid_solver import func_update_all_verts
from genesis.utils.misc import qd_to_numpy
from genesis.utils.raycast import RayHit

if TYPE_CHECKING:
    from genesis.engine.scene import Scene


@qd.func
def get_triangle_vertices(
    i_f: gs.qd_int,
    i_b: gs.qd_int,
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
    ray_start: gs.qd_vec3,
    ray_dir: gs.qd_vec3,
    max_range: gs.qd_float,
    i_b: gs.qd_int,
    bvh_nodes: qd.template(),
    bvh_morton_codes: qd.template(),
    faces_info: array_class.FacesInfo,
    verts_info: array_class.VertsInfo,
    fixed_verts_state: array_class.VertsState,
    free_verts_state: array_class.VertsState,
    eps: gs.qd_float,
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
    eps: gs.qd_float,
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
        f = 1.0 / a
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
        result = qd.math.vec4(t, u, v, 1.0)

    return result


@qd.func
def ray_aabb_intersection(
    ray_start: gs.qd_vec3,
    ray_dir: gs.qd_vec3,
    aabb_min: gs.qd_vec3,
    aabb_max: gs.qd_vec3,
    eps: gs.qd_float,
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

    t_near = qd.max(tmin.x, tmin.y, tmin.z, 0.0)
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
        geoms_info, geoms_state, verts_info, free_verts_state, fixed_verts_state, static_rigid_sim_config
    )
    update_aabbs(
        free_verts_state,
        fixed_verts_state,
        verts_info,
        faces_info,
        aabb_state,
    )


@qd.kernel(fastcache=gs.use_fastcache)
def kernel_cast_ray(
    fixed_verts_state: array_class.VertsState,
    free_verts_state: array_class.VertsState,
    verts_info: array_class.VertsInfo,
    faces_info: array_class.FacesInfo,
    bvh_nodes: qd.template(),
    bvh_morton_codes: qd.template(),
    ray_start: qd.types.ndarray(ndim=1),  # (3,)
    ray_direction: qd.types.ndarray(ndim=1),  # (3,)
    max_range: gs.qd_float,
    envs_idx: qd.types.ndarray(ndim=1),  # [n_envs]
    result: array_class.RaycastResult,
    eps: gs.qd_float,
):
    """
    Quadrants kernel for casting a single ray.

    This loops over all environments in envs_idx and stores the closest hit in result.
    """
    # Setup ray
    ray_start_world = qd.math.vec3(ray_start[0], ray_start[1], ray_start[2])
    ray_direction_world = qd.math.vec3(ray_direction[0], ray_direction[1], ray_direction[2])

    # Initialize result with no hit
    result.distance[None] = qd.math.nan
    result.geom_idx[None] = -1
    result.hit_point[None] = qd.math.vec3(0.0, 0.0, 0.0)
    result.normal[None] = qd.math.vec3(0.0, 0.0, 0.0)
    result.env_idx[None] = -1

    closest_distance = max_range
    hit_face = -1
    hit_env_idx = -1
    hit_normal = qd.math.vec3(0.0, 0.0, 0.0)

    for i_b_ in range(envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        cur_hit_face, cur_distance, cur_hit_normal = bvh_ray_cast(
            ray_start=ray_start_world,
            ray_dir=ray_direction_world,
            max_range=closest_distance,
            i_b=i_b,
            bvh_nodes=bvh_nodes,
            bvh_morton_codes=bvh_morton_codes,
            faces_info=faces_info,
            verts_info=verts_info,
            fixed_verts_state=fixed_verts_state,
            free_verts_state=free_verts_state,
            eps=eps,
        )

        # Update global closest if this environment had a closer hit
        if cur_hit_face >= 0 and cur_distance < closest_distance:
            closest_distance = cur_distance
            hit_face = cur_hit_face
            hit_env_idx = i_b
            hit_normal = cur_hit_normal

    # Store result
    if hit_face >= 0:
        result.distance[None] = closest_distance
        # Find which geom this face belongs to
        i_g = faces_info.geom_idx[hit_face]
        result.geom_idx[None] = i_g
        # Compute hit point
        hit_point = ray_start_world + closest_distance * ray_direction_world
        result.hit_point[None] = hit_point
        # Store normal
        result.normal[None] = hit_normal
        result.env_idx[None] = hit_env_idx


class Raycaster:
    """
    BVH-accelerated raycaster. Currently only supports single-ray casting.
    """

    def __init__(self, scene: "Scene"):
        self.result = array_class.get_viewer_raycast_result()

        self.scene = scene
        self.solver = scene.sim.rigid_solver

        self.envs_idx = scene._envs_idx

        # Build the BVH structure for rendered environments.
        n_faces = self.solver.faces_info.geom_idx.shape[0]

        if n_faces == 0:
            gs.logger.warning("No faces found in scene, viewer raycasting will not work.")
            self.aabb = None
            self.bvh = None
            return

        self.aabb = AABB(n_batches=len(self.envs_idx), n_aabbs=n_faces)
        self.bvh = LBVH(
            self.aabb,
            max_n_query_result_per_aabb=0,  # Not used for ray queries
            n_radix_sort_groups=min(64, n_faces),
        )

        self.update()

    def _raycast_from_result(self, result: array_class.RaycastResult) -> "RayHit | None":
        distance = float(qd_to_numpy(result.distance))
        if math.isnan(distance):
            return None

        geom_idx = int(qd_to_numpy(result.geom_idx))
        position = qd_to_numpy(result.hit_point)
        normal = qd_to_numpy(result.normal)

        # Get the geom object from the solver
        geom = None
        if self.solver is not None and 0 <= geom_idx < len(self.solver.geoms):
            geom = self.solver.geoms[geom_idx]

        return RayHit(distance, position, normal, geom)

    def update(self):
        """Update the BVH structure with current geometry state."""
        if self.bvh is None:
            return

        # Update vertex positions and AABBs
        kernel_update_verts_and_aabbs(
            geoms_info=self.solver.geoms_info,
            geoms_state=self.solver.geoms_state,
            verts_info=self.solver.verts_info,
            faces_info=self.solver.faces_info,
            free_verts_state=self.solver.free_verts_state,
            fixed_verts_state=self.solver.fixed_verts_state,
            static_rigid_sim_config=self.solver._static_rigid_sim_config,
            aabb_state=self.aabb,
        )

        # Rebuild BVH
        self.bvh.build()

    def cast(
        self, ray_origin: np.ndarray, ray_direction: np.ndarray, max_range: float = 1000.0, envs_idx=None
    ) -> RayHit | None:
        """
        Cast a single ray against all rendered environments and return the closest hit.

        Parameters
        ----------
        ray_origin : np.ndarray, shape (3,)
            The origin point of the ray in world coordinates.
        ray_direction : np.ndarray, shape (3,)
            The normalized direction vector of the ray.
        max_range : float, optional
            Maximum distance to check for intersections. Default is 1000.0.
        envs_idx : np.ndarray, shape (n_envs,), optional
            Indices of environments to consider for raycasting. If None, use all environments.

        Returns
        -------
        RayHit | None
            A tuple containing distance, position, normal, and geom.
        """
        kernel_cast_ray(
            self.solver.fixed_verts_state,
            self.solver.free_verts_state,
            self.solver.verts_info,
            self.solver.faces_info,
            self.bvh.nodes,
            self.bvh.morton_codes,
            np.ascontiguousarray(ray_origin, dtype=gs.np_float),
            np.ascontiguousarray(ray_direction, dtype=gs.np_float),
            max_range,
            envs_idx if envs_idx is not None else self.envs_idx,
            self.result,
            gs.EPS,
        )
        return self._raycast_from_result(self.result)
