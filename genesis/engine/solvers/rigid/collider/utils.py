import quadrants as ti
import genesis.utils.array_class as array_class
from genesis.constants import GEOM_TYPE


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
