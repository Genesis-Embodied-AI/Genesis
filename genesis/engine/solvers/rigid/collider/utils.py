import quadrants as ti
import genesis.utils.array_class as array_class
from genesis.constants import GEOM_TYPE
from genesis import utils as gu
import genesis as gs


@ti.func
def func_closest_points_on_segments(
    seg_a_p1,
    seg_a_p2,
    seg_b_p1,
    seg_b_p2,
    EPS,
):
    """
    Compute closest points on two line segments using analytical solution.

    References
    ----------
    Real-Time Collision Detection by Christer Ericson, Chapter 5.1.9
    """
    segment_a_dir = seg_a_p2 - seg_a_p1
    segment_b_dir = seg_b_p2 - seg_b_p1
    vec_between_segment_origins = seg_a_p1 - seg_b_p1

    a_squared_len = segment_a_dir.dot(segment_a_dir)
    dot_product_dir = segment_a_dir.dot(segment_b_dir)
    b_squared_len = segment_b_dir.dot(segment_b_dir)
    d = segment_a_dir.dot(vec_between_segment_origins)
    e = segment_b_dir.dot(vec_between_segment_origins)

    denom = a_squared_len * b_squared_len - dot_product_dir * dot_product_dir

    s = gs.ti_float(0.0)
    t = gs.ti_float(0.0)

    if denom < EPS:
        # Segments are parallel or one/both are degenerate
        s = 0.0
        if b_squared_len > EPS:
            t = ti.math.clamp(e / b_squared_len, 0.0, 1.0)
        else:
            t = 0.0
    else:
        # General case: solve for optimal parameters
        s = (dot_product_dir * e - b_squared_len * d) / denom
        t = (a_squared_len * e - dot_product_dir * d) / denom

        s = ti.math.clamp(s, 0.0, 1.0)

        # Recompute t for clamped s
        t = ti.math.clamp((dot_product_dir * s + e) / b_squared_len if b_squared_len > EPS else 0.0, 0.0, 1.0)

        # Recompute s for clamped t (ensures we're on segment boundaries)
        s_new = ti.math.clamp((dot_product_dir * t - d) / a_squared_len if a_squared_len > EPS else 0.0, 0.0, 1.0)

        # Use refined s if it improves the solution
        if a_squared_len > EPS:
            s = s_new

    seg_a_closest = seg_a_p1 + s * segment_a_dir
    seg_b_closest = seg_b_p1 + t * segment_b_dir

    return seg_a_closest, seg_b_closest


@ti.func
def func_det3(
    v1,
    v2,
    v3,
):
    """
    Compute the determinant of a 3x3 matrix M = [v1 | v2 | v3].
    """
    return (
        v1[0] * (v2[1] * v3[2] - v2[2] * v3[1])
        - v1[1] * (v2[0] * v3[2] - v2[2] * v3[0])
        + v1[2] * (v2[0] * v3[1] - v2[1] * v3[0])
    )


@ti.func
def func_is_geom_aabbs_overlap(geoms_state: array_class.GeomsState, i_ga, i_gb, i_b):
    return not (
        (geoms_state.aabb_max[i_ga, i_b] <= geoms_state.aabb_min[i_gb, i_b]).any()
        or (geoms_state.aabb_min[i_ga, i_b] >= geoms_state.aabb_max[i_gb, i_b]).any()
    )


@ti.func
def func_is_discrete_geom(
    geoms_info: array_class.GeomsInfo,
    i_g,
):
    """
    Check if the given geom is a discrete geometry.
    """
    geom_type = geoms_info.type[i_g]
    return geom_type == GEOM_TYPE.MESH or geom_type == GEOM_TYPE.BOX


@ti.func
def func_is_discrete_geoms(
    geoms_info: array_class.GeomsInfo,
    i_ga,
    i_gb,
):
    """
    Check if the given geoms are discrete geometries.
    """
    return func_is_discrete_geom(geoms_info, i_ga) and func_is_discrete_geom(geoms_info, i_gb)


@ti.func
def func_is_equal_vec(a, b, eps):
    """
    Check if two vectors are equal within a small tolerance.
    """
    return (ti.abs(a - b) < eps).all()
