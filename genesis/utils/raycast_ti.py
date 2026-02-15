from typing import TYPE_CHECKING

import quadrants as ti
import numpy as np

import genesis as gs
import genesis.utils.array_class as array_class
from genesis.engine.bvh import AABB, LBVH, STACK_SIZE
from genesis.engine.solvers.rigid.rigid_solver import func_update_all_verts
from genesis.utils.raycast import RayHit

if TYPE_CHECKING:
    from genesis.engine.scene import Scene


NO_HIT_DISTANCE = -1.0


@ti.func
def get_triangle_vertices(
    i_f: ti.i32,
    i_b: ti.i32,
    faces_info: array_class.FacesInfo,
    verts_info: array_class.VertsInfo,
    fixed_verts_state: array_class.VertsState,
    free_verts_state: array_class.VertsState,
):
    """
    Get the three vertices of a triangle in world space.

    Returns
    -------
    tri_vertices : ti.Matrix
        3x3 matrix where each column is a vertex position.
    """
    tri_vertices = ti.Matrix.zero(gs.ti_float, 3, 3)
    for i in ti.static(range(3)):
        i_v = faces_info.verts_idx[i_f][i]
        i_fv = verts_info.verts_state_idx[i_v]
        if verts_info.is_fixed[i_v]:
            tri_vertices[:, i] = fixed_verts_state.pos[i_fv]
        else:
            tri_vertices[:, i] = free_verts_state.pos[i_fv, i_b]
    return tri_vertices


@ti.func
def bvh_ray_cast(
    ray_start: ti.types.vector(3, ti.f32),
    ray_dir: ti.types.vector(3, ti.f32),
    max_range: ti.f32,
    i_b: ti.i32,
    bvh_nodes: ti.template(),
    bvh_morton_codes: ti.template(),
    faces_info: array_class.FacesInfo,
    verts_info: array_class.VertsInfo,
    fixed_verts_state: array_class.VertsState,
    free_verts_state: array_class.VertsState,
):
    """
    Cast a ray through a BVH and find the closest intersection.

    Returns
    -------
    hit_face : ti.i32
        index of the hit triangle (-1 if no hit)
    hit_distance : ti.f32
        distance to hit point (unchanged max_range if no hit)
    hit_normal : ti.math.vec3
        normal vector at hit point (zero vector if no hit)
    """
    n_triangles = faces_info.verts_idx.shape[0]

    hit_face = -1
    closest_distance = gs.ti_float(max_range)
    hit_normal = ti.math.vec3(0.0, 0.0, 0.0)

    # Stack for non-recursive BVH traversal
    node_stack = ti.Vector.zero(ti.i32, STACK_SIZE)
    node_stack[0] = 0  # Start at root node
    stack_idx = 1

    while stack_idx > 0:
        stack_idx -= 1
        node_idx = node_stack[stack_idx]

        node = bvh_nodes[i_b, node_idx]

        # Check if ray hits the node's bounding box
        aabb_t = ray_aabb_intersection(ray_start, ray_dir, node.bound.min, node.bound.max)

        if aabb_t >= 0.0 and aabb_t < closest_distance:
            if node.left == -1:  # Leaf node
                # Get original triangle/face index
                sorted_leaf_idx = node_idx - (n_triangles - 1)
                i_f = ti.cast(bvh_morton_codes[i_b, sorted_leaf_idx][1], ti.i32)

                # Get triangle vertices
                tri_vertices = get_triangle_vertices(
                    i_f, i_b, faces_info, verts_info, fixed_verts_state, free_verts_state
                )
                v0, v1, v2 = tri_vertices[:, 0], tri_vertices[:, 1], tri_vertices[:, 2]

                # Perform ray-triangle intersection
                hit_result = ray_triangle_intersection(ray_start, ray_dir, v0, v1, v2)

                if hit_result.w > 0.0 and hit_result.x < closest_distance and hit_result.x >= 0.0:
                    closest_distance = hit_result.x
                    hit_face = i_f
                    # Compute triangle normal
                    edge1 = v1 - v0
                    edge2 = v2 - v0
                    hit_normal = edge1.cross(edge2).normalized()
            else:  # Internal node
                # Push children onto stack
                if stack_idx < ti.static(STACK_SIZE - 2):
                    node_stack[stack_idx] = node.left
                    node_stack[stack_idx + 1] = node.right
                    stack_idx += 2

    return hit_face, closest_distance, hit_normal


@ti.func
def ray_triangle_intersection(
    ray_start: ti.types.vector(3, ti.f32),
    ray_dir: ti.types.vector(3, ti.f32),
    v0: ti.types.vector(3, ti.f32),
    v1: ti.types.vector(3, ti.f32),
    v2: ti.types.vector(3, ti.f32),
):
    """
    Moller-Trumbore ray-triangle intersection.

    Returns
    -------
    result : ti.math.vec4
        (t, u, v, hit) where hit=1.0 if intersection found, 0.0 otherwise
    """
    result = ti.Vector.zero(gs.ti_float, 4)

    edge1 = v1 - v0
    edge2 = v2 - v0

    # Begin calculating determinant - also used to calculate u parameter
    h = ray_dir.cross(edge2)
    a = edge1.dot(h)

    # Check all conditions in sequence without early returns
    valid = True

    t = gs.ti_float(0.0)
    u = gs.ti_float(0.0)
    v = gs.ti_float(0.0)
    f = gs.ti_float(0.0)
    s = ti.Vector.zero(gs.ti_float, 3)
    q = ti.Vector.zero(gs.ti_float, 3)

    # If determinant is near zero, ray lies in plane of triangle
    if ti.abs(a) < gs.EPS:
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
        if t <= gs.EPS:
            valid = False

    if valid:
        result = ti.math.vec4(t, u, v, 1.0)

    return result


@ti.func
def ray_aabb_intersection(
    ray_start: ti.types.vector(3, ti.f32),
    ray_dir: ti.types.vector(3, ti.f32),
    aabb_min: ti.types.vector(3, ti.f32),
    aabb_max: ti.types.vector(3, ti.f32),
):
    """
    Fast ray-AABB intersection test.
    Returns the t value of intersection, or -1.0 if no intersection.
    """
    result = -1.0

    # Use the slab method for ray-AABB intersection
    sign = ti.select(ray_dir >= 0.0, 1.0, -1.0)
    ray_dir = sign * ti.max(ti.abs(ray_dir), gs.EPS)
    inv_dir = 1.0 / ray_dir

    t1 = (aabb_min - ray_start) * inv_dir
    t2 = (aabb_max - ray_start) * inv_dir

    tmin = ti.min(t1, t2)
    tmax = ti.max(t1, t2)

    t_near = ti.max(tmin.x, tmin.y, tmin.z, 0.0)
    t_far = ti.min(tmax.x, tmax.y, tmax.z)

    # Check if ray intersects AABB
    if t_near <= t_far:
        result = t_near

    return result


@ti.func
def update_aabbs(
    free_verts_state: array_class.VertsState,
    fixed_verts_state: array_class.VertsState,
    verts_info: array_class.VertsInfo,
    faces_info: array_class.FacesInfo,
    aabb_state: ti.template(),
):
    for i_b, i_f in ti.ndrange(free_verts_state.pos.shape[1], faces_info.verts_idx.shape[0]):
        aabb_state.aabbs[i_b, i_f].min.fill(ti.math.inf)
        aabb_state.aabbs[i_b, i_f].max.fill(-ti.math.inf)

        for i in ti.static(range(3)):
            i_v = faces_info.verts_idx[i_f][i]
            i_fv = verts_info.verts_state_idx[i_v]
            if verts_info.is_fixed[i_v]:
                pos_v = fixed_verts_state.pos[i_fv]
                aabb_state.aabbs[i_b, i_f].min = ti.min(aabb_state.aabbs[i_b, i_f].min, pos_v)
                aabb_state.aabbs[i_b, i_f].max = ti.max(aabb_state.aabbs[i_b, i_f].max, pos_v)
            else:
                pos_v = free_verts_state.pos[i_fv, i_b]
                aabb_state.aabbs[i_b, i_f].min = ti.min(aabb_state.aabbs[i_b, i_f].min, pos_v)
                aabb_state.aabbs[i_b, i_f].max = ti.max(aabb_state.aabbs[i_b, i_f].max, pos_v)


@ti.kernel
def kernel_update_verts_and_aabbs(
    geoms_info: array_class.GeomsInfo,
    geoms_state: array_class.GeomsState,
    verts_info: array_class.VertsInfo,
    faces_info: array_class.FacesInfo,
    free_verts_state: array_class.VertsState,
    fixed_verts_state: array_class.VertsState,
    static_rigid_sim_config: ti.template(),
    aabb_state: ti.template(),
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


@ti.kernel(fastcache=gs.use_fastcache)
def kernel_cast_ray(
    fixed_verts_state: array_class.VertsState,
    free_verts_state: array_class.VertsState,
    verts_info: array_class.VertsInfo,
    faces_info: array_class.FacesInfo,
    bvh_nodes: ti.template(),
    bvh_morton_codes: ti.template(),
    ray_start: ti.types.ndarray(ndim=1),  # (3,)
    ray_direction: ti.types.ndarray(ndim=1),  # (3,)
    max_range: ti.f32,
    envs_idx: ti.types.ndarray(ndim=1),  # [n_envs]
    result: array_class.RaycastResult,
):
    """
    Taichi kernel for casting a single ray.

    This loops over all environments in envs_idx and stores the closest hit in result.
    """
    # Setup ray
    ray_start_world = ti.math.vec3(ray_start[0], ray_start[1], ray_start[2])
    ray_direction_world = ti.math.vec3(ray_direction[0], ray_direction[1], ray_direction[2])

    # Initialize result with no hit
    result.distance[None] = NO_HIT_DISTANCE
    result.geom_idx[None] = -1
    result.hit_point[None] = ti.math.vec3(0.0, 0.0, 0.0)
    result.normal[None] = ti.math.vec3(0.0, 0.0, 0.0)
    result.env_idx[None] = -1

    closest_distance = max_range
    hit_face = -1
    hit_env_idx = -1
    hit_normal = ti.math.vec3(0.0, 0.0, 0.0)

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
        distance = float(result.distance.to_numpy())
        if distance < NO_HIT_DISTANCE + gs.EPS:
            return None

        geom_idx = int(result.geom_idx.to_numpy())
        position = result.hit_point.to_numpy()
        normal = result.normal.to_numpy()

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
            fixed_verts_state=self.solver.fixed_verts_state,
            free_verts_state=self.solver.free_verts_state,
            verts_info=self.solver.verts_info,
            faces_info=self.solver.faces_info,
            bvh_nodes=self.bvh.nodes,
            bvh_morton_codes=self.bvh.morton_codes,
            ray_start=np.ascontiguousarray(ray_origin, dtype=gs.np_float),
            ray_direction=np.ascontiguousarray(ray_direction, dtype=gs.np_float),
            max_range=max_range,
            envs_idx=envs_idx if envs_idx is not None else self.envs_idx,
            result=self.result,
        )
        return self._raycast_from_result(self.result)
