import numpy as np

from .. import util

from ..constants import log
from ..constants import tol_path as tol
from ..constants import res_path as res

# floating point zero
_TOL_ZERO = 1e-12


def arc_center(points, return_normal=True, return_angle=True):
    """
    Given three points on a 2D or 3D arc find the center,
    radius, normal, and angular span.

    Parameters
    ---------
    points : (3, dimension) float
      Points in space, where dimension is either 2 or 3
    return_normal : bool
      If True calculate the 3D normal unit vector
    return_angle : bool
      If True calculate the start and stop angle and span

    Returns
    ---------
    result : dict
      Contains arc center and other keys:
       'center' : (d,) float, cartesian center of the arc
       'radius' : float, radius of the arc
       'normal' : (3,) float, the plane normal.
       'angles' : (2,) float, angle of start and end in radians
       'span'   : float, angle swept by the arc in radians
    """
    points = np.asanyarray(points, dtype=np.float64)

    # get the non-unit vectors of the three points
    vectors = points[[2, 0, 1]] - points[[1, 2, 0]]
    # we need both the squared row sum and the non-squared
    abc2 = np.dot(vectors**2, [1] * points.shape[1])
    # same as np.linalg.norm(vectors, axis=1)
    abc = np.sqrt(abc2)

    # perform radius calculation scaled to shortest edge
    # to avoid precision issues with small or large arcs
    scale = abc.min()
    # get the edge lengths scaled to the smallest
    edges = abc / scale
    # half the total length of the edges
    half = edges.sum() / 2.0
    # check the denominator for the radius calculation
    denom = half * np.prod(half - edges)
    if denom < tol.merge:
        raise ValueError("arc is colinear!")
    # find the radius and scale back after the operation
    radius = scale * ((np.prod(edges) / 4.0) / np.sqrt(denom))

    # use a barycentric approach to get the center
    ba2 = (abc2[[1, 2, 0, 0, 2, 1, 0, 1, 2]] * [1, 1, -1, 1, 1, -1, 1, 1, -1]).reshape((3, 3)).sum(axis=1) * abc2
    center = points.T.dot(ba2) / ba2.sum()

    if tol.strict:
        # all points should be at the calculated radius from center
        assert util.allclose(np.linalg.norm(points - center, axis=1), radius)

    # start with initial results
    result = {"center": center, "radius": radius}

    if return_normal:
        if points.shape == (3, 2):
            # for 2D arcs still use the cross product so that
            # the sign of the normal vector is consistent
            result["normal"] = util.unitize(np.cross(np.append(-vectors[1], 0), np.append(vectors[2], 0)))
        else:
            # otherwise just take the cross product
            result["normal"] = util.unitize(np.cross(-vectors[1], vectors[2]))

    if return_angle:
        # vectors from points on arc to center point
        vector = util.unitize(points - center)
        edge_direction = np.diff(points, axis=0)
        # find the angle between the first and last vector
        dot = np.dot(*vector[[0, 2]])
        if dot < (_TOL_ZERO - 1):
            angle = np.pi
        elif dot > 1 - _TOL_ZERO:
            angle = 0.0
        else:
            angle = np.arccos(dot)
        # if the angle is nonzero and vectors are opposite direction
        # it means we have a long arc rather than the short path
        if abs(angle) > _TOL_ZERO and np.dot(*edge_direction) < 0.0:
            angle = (np.pi * 2) - angle
        # convoluted angle logic
        angles = np.arctan2(*vector[:, :2].T[::-1]) + np.pi * 2
        angles_sorted = np.sort(angles[[0, 2]])
        reverse = angles_sorted[0] < angles[1] < angles_sorted[1]
        angles_sorted = angles_sorted[:: (1 - int(not reverse) * 2)]
        result["angles"] = angles_sorted
        result["span"] = angle

    return result


def discretize_arc(points, close=False, scale=1.0):
    """
    Returns a version of a three point arc consisting of
    line segments.

    Parameters
    ---------
    points : (3, d) float
      Points on the arc where d in [2,3]
    close :  boolean
      If True close the arc into a circle
    scale : float
      What is the approximate overall drawing scale
      Used to establish order of magnitude for precision

    Returns
    ---------
    discrete : (m, d) float
      Connected points in space
    """
    # make sure points are (n, 3)
    points, is_2D = util.stack_3D(points, return_2D=True)
    # find the center of the points
    try:
        # try to find the center from the arc points
        center_info = arc_center(points)
    except BaseException:
        # if we hit an exception return a very bad but
        # technically correct discretization of the arc
        if is_2D:
            return points[:, :2]
        return points

    center, R, N, angle = (center_info["center"], center_info["radius"], center_info["normal"], center_info["span"])

    # if requested, close arc into a circle
    if close:
        angle = np.pi * 2

    # the number of facets, based on the angle criteria
    count_a = angle / res.seg_angle
    count_l = ((R * angle)) / (res.seg_frac * scale)

    # figure out the number of line segments
    count = np.max([count_a, count_l])
    # force at LEAST 4 points for the arc
    # otherwise the endpoints will diverge
    count = np.clip(count, 4, np.inf)
    count = int(np.ceil(count))

    V1 = util.unitize(points[0] - center)
    V2 = util.unitize(np.cross(-N, V1))
    t = np.linspace(0, angle, count)

    discrete = np.tile(center, (count, 1))
    discrete += R * np.cos(t).reshape((-1, 1)) * V1
    discrete += R * np.sin(t).reshape((-1, 1)) * V2

    # do an in-process check to make sure result endpoints
    # match the endpoints of the source arc
    if tol.strict and not close:
        arc_dist = util.row_norm(points[[0, -1]] - discrete[[0, -1]])
        arc_ok = (arc_dist < tol.merge).all()
        if not arc_ok:
            log.warning("failed to discretize arc (endpoint_distance=%s R=%s)", str(arc_dist), R)
            log.warning("Failed arc points: %s", str(points))
            raise ValueError("Arc endpoints diverging!")
    discrete = discrete[:, : (3 - is_2D)]

    return discrete


def to_threepoint(center, radius, angles=None):
    """
    For 2D arcs, given a center and radius convert them to three
    points on the arc.

    Parameters
    -----------
    center : (2,) float
      Center point on the plane
    radius : float
      Radius of arc
    angles : (2,) float
      Angles in radians for start and end angle
      if not specified, will default to (0.0, pi)

    Returns
    ----------
    three : (3, 2) float
      Arc control points
    """
    # if no angles provided assume we want a half circle
    if angles is None:
        angles = [0.0, np.pi]
    # force angles to float64
    angles = np.asanyarray(angles, dtype=np.float64)
    if angles.shape != (2,):
        raise ValueError("angles must be (2,)!")
    # provide the wrap around
    if angles[1] < angles[0]:
        angles[1] += np.pi * 2

    center = np.asanyarray(center, dtype=np.float64)
    if center.shape != (2,):
        raise ValueError("only valid on 2D arcs!")

    # turn the angles of [start, end]
    # into [start, middle, end]
    angles = np.array([angles[0], angles.mean(), angles[1]], dtype=np.float64)
    # turn angles into (3, 2) points
    three = (np.column_stack((np.cos(angles), np.sin(angles))) * radius) + center

    return three
