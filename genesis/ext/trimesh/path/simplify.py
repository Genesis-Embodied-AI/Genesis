import numpy as np

import copy
import collections

from . import arc
from . import entities

from .. import util

from ..nsphere import fit_nsphere

from ..constants import log
from ..constants import tol_path as tol


def fit_circle_check(points, scale, prior=None, final=False, verbose=False):
    """
    Fit a circle, and reject the fit if:
    * the radius is larger than tol.radius_min*scale or tol.radius_max*scale
    * any segment spans more than tol.seg_angle
    * any segment is longer than tol.seg_frac*scale
    * the fit deviates by more than tol.radius_frac*radius
    * the segments on the ends deviate from tangent by more than tol.tangent

    Parameters
    ---------
    points :  (n, d)
      List of points which represent a path
    prior :  (center, radius) tuple
      Best guess or None if unknown
    scale : float
      What is the overall scale of the set of points
    verbose : bool
     Output log.debug messages for the reasons
     for fit rejection only suggested for manual debugging

    Returns
    -----------
    if fit is acceptable:
        (center, radius) tuple
    else:
        None
    """
    # an arc needs at least three points
    if len(points) < 3:
        return None
    # make sure our points are a numpy array
    points = np.asanyarray(points, dtype=np.float64)

    # do a least squares fit on the points
    C, R, r_deviation = fit_nsphere(points, prior=prior)

    # check to make sure radius is between min and max allowed
    if not tol.radius_min < (R / scale) < tol.radius_max:
        if verbose:
            log.debug("circle fit error: R %f", R / scale)
        return None

    # check point radius error
    r_error = r_deviation / R
    if r_error > tol.radius_frac:
        if verbose:
            log.debug("circle fit error: fit %s", str(r_error))
        return None

    vectors = np.diff(points, axis=0)
    segment = util.row_norm(vectors)

    # approximate angle in radians, segments are linear length
    # not arc length but this is close and avoids a cosine
    angle = segment / R
    if (angle > tol.seg_angle).any():
        if verbose:
            log.debug("circle fit error: angle %s", str(angle))
        return None

    if final and (angle > tol.seg_angle_min).sum() < 3:
        log.debug("final: angle %s", str(angle))
        return None

    # check segment length as a fraction of drawing scale
    scaled = segment / scale

    if (scaled > tol.seg_frac).any():
        if verbose:
            log.debug("circle fit error: segment %s", str(scaled))
        return None

    # check to make sure the line segments on the ends are actually
    # tangent with the candidate circle fit
    mid_pt = points[[0, -2]] + (vectors[[0, -1]] * 0.5)
    radial = util.unitize(mid_pt - C)
    ends = util.unitize(vectors[[0, -1]])
    tangent = np.abs(np.arccos(util.diagonal_dot(radial, ends)))
    tangent = np.abs(tangent - np.pi / 2).max()

    if tangent > tol.tangent:
        if verbose:
            log.debug("circle fit error: tangent %f", np.degrees(tangent))
        return None

    result = {"center": C, "radius": R}

    return result


def is_circle(points, scale, verbose=False):
    """
    Given a set of points, quickly determine if they represent
    a circle or not.

    Parameters
    -------------
    points : (n,2 ) float
      Points in space
    scale : float
      Scale of overall drawing
    verbose : bool
      Print all fit messages or not

    Returns
    -------------
    control: (3,2) float, points in space, OR
              None, if not a circle
    """

    # make sure input is a numpy array
    points = np.asanyarray(points)
    scale = float(scale)

    # can only be a circle if the first and last point are the
    # same (AKA is a closed path)
    if np.linalg.norm(points[0] - points[-1]) > tol.merge:
        return None

    box = np.ptp(points, axis=0)
    # the bounding box size of the points
    # check aspect ratio as an early exit if the path is not a circle
    aspect = np.divide(*box)
    if np.abs(aspect - 1.0) > tol.aspect_frac:
        return None

    # fit a circle with tolerance checks
    CR = fit_circle_check(points, scale=scale)
    if CR is None:
        return None

    # return the circle as three control points
    control = arc.to_threepoint(**CR)
    return control


def merge_colinear(points, scale):
    """
    Given a set of points representing a path in space,
    merge points which are colinear.

    Parameters
    ----------
    points : (n, dimension) float
      Points in space
    scale : float
      Scale of drawing for precision

    Returns
    ----------
    merged : (j, d) float
      Points with colinear and duplicate
      points merged, where (j < n)
    """
    points = np.asanyarray(points, dtype=np.float64)
    scale = float(scale)

    if len(points.shape) != 2 or points.shape[1] != 2:
        raise ValueError("only for 2D points!")

    # if there's less than 3 points nothing to merge
    if len(points) < 3:
        return points.copy()

    # the vector from one point to the next
    direction = points[1:] - points[:-1]
    # the length of the direction vector
    direction_norm = util.row_norm(direction)
    # make sure points don't have zero length
    direction_ok = direction_norm > tol.merge

    # remove duplicate points
    points = np.vstack((points[0], points[1:][direction_ok]))
    direction = direction[direction_ok]
    direction_norm = direction_norm[direction_ok]

    # create a vector between every other point, then turn it perpendicular
    # if we have points A B C D
    # and direction vectors A-B, B-C, etc
    # these will be perpendicular to the vectors A-C, B-D, etc
    perp = (points[2:] - points[:-2]).T[::-1].T
    perp[:, 0] *= -1
    perp_norm = util.row_norm(perp)
    perp_nonzero = perp_norm > tol.merge
    perp[perp_nonzero] /= perp_norm[perp_nonzero].reshape((-1, 1))

    # find the projection of each direction vector
    # onto the perpendicular vector
    projection = np.abs(util.diagonal_dot(perp, direction[:-1]))

    projection_ratio = np.max((projection / direction_norm[1:], projection / direction_norm[:-1]), axis=0)

    mask = np.ones(len(points), dtype=bool)
    # since we took diff, we need to offset by one
    mask[1:-1][projection_ratio < 1e-4 * scale] = False

    merged = points[mask]
    return merged


def resample_spline(points, smooth=0.001, count=None, degree=3):
    """
    Resample a path in space, smoothing along a b-spline.

    Parameters
    -----------
    points : (n, dimension) float
      Points in space
    smooth : float
      Smoothing distance
    count :  int or None
      Number of samples desired in output
    degree : int
      Degree of spline polynomial

    Returns
    ---------
    resampled : (count, dimension) float
      Points in space
    """
    from scipy.interpolate import splprep, splev

    if count is None:
        count = len(points)
    points = np.asanyarray(points)
    closed = np.linalg.norm(points[0] - points[-1]) < tol.merge

    tpl = splprep(points.T, s=smooth, k=degree)[0]
    i = np.linspace(0.0, 1.0, count)
    resampled = np.column_stack(splev(i, tpl))

    if closed:
        shared = resampled[[0, -1]].mean(axis=0)
        resampled[0] = shared
        resampled[-1] = shared

    return resampled


def points_to_spline_entity(points, smooth=None, count=None):
    """
    Create a spline entity from a curve in space

    Parameters
    -----------
    points : (n, dimension) float
      Points in space
    smooth : float
      Smoothing distance
    count :  int or None
      Number of samples desired in result

    Returns
    ---------
    entity : entities.BSpline
      Entity object with points indexed at zero
    control : (m, dimension) float
      New vertices for entity
    """

    from scipy.interpolate import splprep

    if count is None:
        count = len(points)
    if smooth is None:
        smooth = 0.002

    points = np.asanyarray(points, dtype=np.float64)
    closed = np.linalg.norm(points[0] - points[-1]) < tol.merge

    knots, control, degree = splprep(points.T, s=smooth)[0]
    control = np.transpose(control)
    index = np.arange(len(control))

    if closed:
        control[0] = control[[0, -1]].mean(axis=0)
        control = control[:-1]
        index[-1] = index[0]

    entity = entities.BSpline(points=index, knots=knots, closed=closed)

    return entity, control


def simplify_basic(drawing, process=False, **kwargs):
    """
    Merge colinear segments and fit circles.

    Parameters
    -----------
    drawing : Path2D
      Source geometry, will not be modified

    Returns
    -----------
    simplified : Path2D
      Original path but with some closed line-loops converted to circles
    """

    if any(entity.__class__.__name__ != "Line" for entity in drawing.entities):
        log.debug("Skipping path containing entities other than `Line`")
        return drawing

    # we are going to do a bookkeeping to avoid having
    # to recompute literally everything when simplification is ran
    cache = copy.deepcopy(drawing._cache)

    # store new values
    vertices_new = collections.deque()
    entities_new = collections.deque()

    # avoid thrashing cache in loop
    scale = drawing.scale

    # loop through (n, 2) closed paths
    for discrete in drawing.discrete:
        # check to see if the closed entity is a circle
        circle = is_circle(discrete, scale=scale)
        if circle is not None:
            # the points are circular enough for our high standards
            # so replace them with a closed Arc entity
            entities_new.append(entities.Arc(points=np.arange(3) + len(vertices_new), closed=True))
            vertices_new.extend(circle)
        else:
            # not a circle, so clean up colinear segments
            # then save it as a single line entity
            points = merge_colinear(discrete, scale=scale)
            # references for new vertices
            indexes = np.arange(len(points)) + len(vertices_new)
            # discrete curves are always closed
            indexes[-1] = indexes[0]
            # append new vertices and entity
            entities_new.append(entities.Line(points=indexes))
            vertices_new.extend(points)

    # create the new drawing object
    simplified = type(drawing)(
        entities=entities_new, vertices=vertices_new, metadata=copy.deepcopy(drawing.metadata), process=process
    )
    # we have changed every path to a single closed entity
    # either a closed arc, or a closed line
    # so all closed paths are now represented by a single entity
    cache.cache.update(
        {
            "paths": np.arange(len(entities_new)).reshape((-1, 1)),
            "path_valid": np.ones(len(entities_new), dtype=bool),
            "dangling": np.array([]),
        }
    )

    # force recompute of exact bounds
    if "bounds" in cache.cache:
        cache.cache.pop("bounds")

    simplified._cache = cache
    # set the cache ID so it won't dump when a value is requested
    simplified._cache.id_set()

    return simplified


def simplify_spline(path, smooth=None, verbose=False):
    """
    Replace discrete curves with b-spline or Arc and
    return the result as a new Path2D object.

    Parameters
    ------------
    path : trimesh.path.Path2D
      Input geometry
    smooth : float
      Distance to smooth

    Returns
    ------------
    simplified : Path2D
      Consists of Arc and BSpline entities
    """

    new_vertices = []
    new_entities = []
    scale = path.scale

    for discrete in path.discrete:
        circle = is_circle(discrete, scale=scale, verbose=verbose)
        if circle is not None:
            # the points are circular enough for our high standards
            # so replace them with a closed Arc entity
            new_entities.append(entities.Arc(points=np.arange(3) + len(new_vertices), closed=True))
            new_vertices.extend(circle)
            continue

        # entities for this path
        entity, vertices = points_to_spline_entity(discrete, smooth=smooth)
        # reindex returned control points
        entity.points += len(new_vertices)
        # save entity and vertices
        new_vertices.extend(vertices)
        new_entities.append(entity)

    # create the Path2D object for the result
    simplified = type(path)(entities=new_entities, vertices=new_vertices)

    return simplified
