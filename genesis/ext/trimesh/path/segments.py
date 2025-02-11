"""
segments.py
--------------

Deal with (n, 2, 3) line segments.
"""

import numpy as np

from .. import util
from .. import grouping
from .. import geometry
from .. import interval
from .. import transformations

from ..constants import tol


def segments_to_parameters(segments):
    """
    For 3D line segments defined by two points, turn
    them in to an origin defined as the closest point along
    the line to the zero origin as well as a direction vector
    and start and end parameter.

    Parameters
    ------------
    segments : (n, 2, 3) float
       Line segments defined by start and end points

    Returns
    --------------
    origins : (n, 3) float
       Point on line closest to [0, 0, 0]
    vectors : (n, 3) float
       Unit line directions
    parameters : (n, 2) float
       Start and end distance pairs for each line
    """
    segments = np.asanyarray(segments, dtype=np.float64)
    if not util.is_shape(segments, (-1, 2, (2, 3))):
        raise ValueError("incorrect segment shape!", segments.shape)

    # make the initial origin one of the end points
    endpoint = segments[:, 0]
    vectors = segments[:, 1] - endpoint
    vectors_norm = util.row_norm(vectors)
    vectors /= vectors_norm.reshape((-1, 1))

    # find the point along the line nearest the origin
    offset = util.diagonal_dot(endpoint, vectors)
    # points nearest [0, 0, 0] will be our new origin
    origins = endpoint + (offset.reshape((-1, 1)) * -vectors)

    # parametric start and end of line segment
    parameters = np.column_stack((offset, offset + vectors_norm))
    # make sure signs are consistent
    vectors, signs = util.vector_hemisphere(vectors, return_sign=True)
    parameters *= signs.reshape((-1, 1))

    return origins, vectors, parameters


def parameters_to_segments(origins, vectors, parameters):
    """
    Convert a parametric line segment representation to
    a two point line segment representation

    Parameters
    ------------
    origins : (n, 3) float
       Line origin point
    vectors : (n, 3) float
       Unit line directions
    parameters : (n, 2) float
       Start and end distance pairs for each line

    Returns
    --------------
    segments : (n, 2, 3) float
       Line segments defined by start and end points
    """
    # don't copy input
    origins = np.asanyarray(origins, dtype=np.float64)
    vectors = np.asanyarray(vectors, dtype=np.float64)
    parameters = np.asanyarray(parameters, dtype=np.float64)

    # turn the segments into a reshapable 2D array
    segments = np.hstack((origins + vectors * parameters[:, :1], origins + vectors * parameters[:, 1:]))

    return segments.reshape((-1, 2, origins.shape[1]))


def colinear_pairs(segments, radius=0.01, angle=0.01, length=None):
    """
    Find pairs of segments which are colinear.

    Parameters
    -------------
    segments : (n, 2, (2, 3)) float
      Two or three dimensional line segments
    radius : float
      Maximum radius line origins can differ
      and be considered colinear
    angle : float
      Maximum angle in radians segments can
      differ and still be considered colinear
    length : None or float
      If specified, will additionally require
      that pairs have a *vertex* within this distance.

    Returns
    ------------
    pairs : (m, 2) int
      Indexes of segments which are colinear
    """
    from scipy import spatial

    # convert segments to parameterized origins
    # which are the closest point on the line to
    # the actual zero- origin
    origins, vectors, param = segments_to_parameters(segments)

    # create a kdtree for origins
    tree = spatial.cKDTree(origins)

    # find origins closer than specified radius
    pairs = tree.query_pairs(r=radius, output_type="ndarray")

    # calculate angles between pairs
    angles = geometry.vector_angle(vectors[pairs])

    # angles can be within tolerance of 180 degrees or 0.0 degrees
    angle_ok = np.logical_or(util.isclose(angles, np.pi, atol=angle), util.isclose(angles, 0.0, atol=angle))

    # apply angle threshold
    colinear = pairs[angle_ok]

    # if length is specified check endpoint proximity
    if length is not None:
        a, b = param[colinear.T]
        distance = np.abs(np.column_stack([a[:, :1] - b, a[:, 1:] - b])).min(axis=1)
        identical = distance < length
        # remove non- identical pairs
        colinear = colinear[identical]

    return colinear


def split(segments, points, atol=1e-5):
    """
    Find any points that lie on a segment (not an endpoint)
    and then split that segment into two segments.

    We are basically going to find the distance between
    point and both segment vertex, and see if it is with
    tolerance of the segment length.

    Parameters
    --------------
    segments : (n, 2, (2, 3) float
      Line segments in space
    points : (n, (2, 3)) float
      Points in space
    atol : float
      Absolute tolerance for distances

    Returns
    -------------
    split : (n, 2, (3 | 3) float
      Line segments in space, split at vertices
    """

    points = np.asanyarray(points, dtype=np.float64)
    segments = np.asanyarray(segments, dtype=np.float64)
    # reshape to a flat 2D (n, dimension) array
    seg_flat = segments.reshape((-1, segments.shape[2]))

    # find the length of every segment
    length = ((segments[:, 0, :] - segments[:, 1, :]) ** 2).sum(axis=1) ** 0.5

    # a mask to remove segments we split at the end
    keep = np.ones(len(segments), dtype=bool)
    # append new segments to a list
    new_seg = []

    # loop through every point
    for p in points:
        # note that you could probably get a speedup
        # by using scipy.spatial.distance.cdist here

        # find the distance from point to every segment endpoint
        pair = ((seg_flat - p) ** 2).sum(axis=1).reshape((-1, 2)) ** 0.5
        # point is on a segment if it is not on a vertex
        # and the sum length is equal to the actual segment length
        on_seg = np.logical_and(
            util.isclose(length, pair.sum(axis=1), atol=atol), ~util.isclose(pair, 0.0, atol=atol).any(axis=1)
        )

        # if we have any points on the segment split it in twain
        if on_seg.any():
            # remove the original segment
            keep = np.logical_and(keep, ~on_seg)
            # split every segment that this point lies on
            for seg in segments[on_seg]:
                new_seg.append([p, seg[0]])
                new_seg.append([p, seg[1]])

    if len(new_seg) > 0:
        return np.vstack((segments[keep], new_seg))
    else:
        return segments


def unique(segments, digits=5):
    """
    Find unique non-zero line segments.

    Parameters
    ------------
    segments : (n, 2, (2|3)) float
      Line segments in space
    digits : int
      How many digits to consider when merging vertices

    Returns
    -----------
    unique : (m, 2, (2|3)) float
      Segments with duplicates merged
    """
    segments = np.asanyarray(segments, dtype=np.float64)

    # find segments as unique indexes so we can find duplicates
    inverse = grouping.unique_rows(segments.reshape((-1, segments.shape[2])), digits=digits)[1].reshape((-1, 2))
    # make sure rows are sorted
    inverse.sort(axis=1)
    # remove segments where both indexes are the same
    mask = np.zeros(len(segments), dtype=bool)
    # only include the first occurrence of a segment
    mask[grouping.unique_rows(inverse)[0]] = True
    # remove segments that are zero-length
    mask[inverse[:, 0] == inverse[:, 1]] = False
    # apply the unique mask
    unique = segments[mask]

    return unique


def overlap(origins, vectors, params):
    """
    Find the overlap of two parallel line segments.

    Parameters
    ------------
    origins : (2, 3) float
       Origin points of lines in space
    vectors : (2, 3) float
       Unit direction vectors of lines
    params : (2, 2) float
       Two (start, end) distance pairs

    Returns
    ------------
    length : float
       Overlapping length
    overlap : (n, 2, 3) float
       Line segments for overlapping distance
    """
    # copy inputs and make sure shape is correct
    origins = np.array(origins).reshape((2, 3))
    vectors = np.array(vectors).reshape((2, 3))
    params = np.array(params).reshape((2, 2))

    if tol.strict:
        # convert input to parameters before flipping
        # to make sure we didn't screw it up
        truth = parameters_to_segments(origins, vectors, params)

    # this function only works on parallel lines
    dot = np.dot(*vectors)
    assert np.isclose(np.abs(dot), 1.0, atol=0.01)

    # if two vectors are reversed
    if dot < 0.0:
        # reverse direction vector
        vectors[1] *= -1.0
        # negate parameters
        params[1] *= -1.0

    if tol.strict:
        # do a check to make sure our reversal didn't
        # inadvertently give us incorrect segments
        assert np.allclose(truth, parameters_to_segments(origins, vectors, params))

    # merge the parameter ranges
    ok, new_range = interval.intersection(*params)

    if not ok:
        return 0.0, np.array([])

    # create the overlapping segment pairs (2, 2, 3)
    segments = np.array([o + v * new_range.reshape((-1, 1)) for o, v in zip(origins, vectors)])
    # get the length of the new range
    length = np.ptp(new_range)

    return length, segments


def extrude(segments, height, double_sided=False):
    """
    Extrude 2D line segments into 3D triangles.

    Parameters
    -------------
    segments : (n, 2, 2) float
      2D line segments
    height : float
      Distance to extrude along Z
    double_sided : bool
      If true, return 4 triangles per segment

    Returns
    -------------
    vertices : (n, 3) float
      Vertices in space
    faces : (n, 3) int
      Indices of vertices forming triangles
    """
    segments = np.asanyarray(segments, dtype=np.float64)
    if not util.is_shape(segments, (-1, 2, 2)):
        raise ValueError("segments shape incorrect")

    # we are creating two vertices  triangles for every 2D line segment
    # on the segments of the 2D triangulation
    vertices = np.tile(segments.reshape((-1, 2)), 2).reshape((-1, 2))
    vertices = np.column_stack((vertices, np.tile([0, height, 0, height], len(segments))))
    faces = np.tile([3, 1, 2, 2, 1, 0], (len(segments), 1))
    faces += np.arange(len(segments)).reshape((-1, 1)) * 4
    faces = faces.reshape((-1, 3))

    if double_sided:
        # stack so they will render from the back
        faces = np.vstack((faces, np.fliplr(faces)))

    return vertices, faces


def length(segments, summed=True):
    """
    Extrude 2D line segments into 3D triangles.

    Parameters
    -------------
    segments : (n, 2, 2) float
      2D line segments
    height : float
      Distance to extrude along Z
    double_sided : bool
      If true, return 4 triangles per segment

    Returns
    -------------
    vertices : (n, 3) float
      Vertices in space
    faces : (n, 3) int
      Indices of vertices forming triangles
    """
    segments = np.asanyarray(segments)
    norms = util.row_norm(segments[:, 0, :] - segments[:, 1, :])
    if summed:
        return norms.sum()
    return norms


def resample(segments, maxlen, return_index=False, return_count=False):
    """
    Resample line segments until no segment
    is longer than maxlen.

    Parameters
    -------------
    segments : (n, 2, 2) float
      2D line segments
    maxlen : float
      The maximum length of a line segment
    return_index : bool
      Return the index of the source segment
    return_count : bool
      Return how many segments each original was split into

    Returns
    -------------
    resampled : (m, 2, 3) float
      Line segments where no segment is longer than maxlen
    index : (m,) int
      [OPTIONAL] The index of segments resampled came from
    count : (n,) int
      [OPTIONAL] The count of the original segments
    """
    # check arguments
    maxlen = float(maxlen)
    segments = np.array(segments, dtype=np.float64)

    # shortcut for endpoints
    pt1 = segments[:, 0]
    pt2 = segments[:, 1]
    # vector between endpoints
    vec = pt2 - pt1
    # the integer number of times a segment needs to be split
    splits = np.ceil(util.row_norm(vec) / maxlen).astype(np.int64)

    # save resulting segments
    result = []
    # save index of original segment
    index = []

    tile = np.tile
    # generate the line indexes ahead of time
    stacks = util.stack_lines(np.arange(splits.max() + 1))

    # loop through each count of unique splits needed
    for split in np.unique(splits):
        # get a mask of which segments need to be split
        mask = splits == split
        # the vector for each incremental length
        increment = vec[mask] / split
        # stack the increment vector into the shape needed
        v = tile(increment, split + 1).reshape((-1, 3)) * tile(np.arange(split + 1), len(increment)).reshape((-1, 1))
        # stack the origin points correctly
        o = tile(pt1[mask], split + 1).reshape((-1, 3))
        # now get each segment as an (split, 3) polyline
        poly = (o + v).reshape((-1, split + 1, 3))
        # save the resulting segments
        # magical slicing is equivalent to:
        # > [p[stack] for p in poly]
        result.extend(poly[:, stacks[:split]])

        if return_index:
            # get the original index from the mask
            index_original = np.nonzero(mask)[0].reshape((-1, 1))
            # save one entry per split segment
            index.append((np.ones((len(poly), split), dtype=np.int64) * index_original).ravel())
        if tol.strict:
            # check to make sure every start and end point
            # from the reconstructed result corresponds
            for original, recon in zip(segments[mask], poly):
                assert np.allclose(original[0], recon[0])
                assert np.allclose(original[-1], recon[-1])
            # make sure stack slicing was OK
            assert np.allclose(util.stack_lines(np.arange(split + 1)), stacks[:split])

    # stack into (n, 2, 3) segments
    result = [np.concatenate(result)]

    if tol.strict:
        # make sure resampled segments have the same length as input
        assert np.isclose(length(segments), length(result[0]), atol=1e-3)

    # stack additional return options
    if return_index:
        # stack original indexes
        index = np.concatenate(index)
        if tol.strict:
            # index should correspond to result
            assert len(index) == len(result[0])
            # every segment should be represented
            assert set(index) == set(range(len(segments)))
        result.append(index)

    if return_count:
        result.append(splits)

    if len(result) == 1:
        return result[0]
    return result


def to_svg(segments, digits=4, matrix=None, merge=True):
    """
    Convert (n, 2, 2) line segments to an SVG path string.

    Parameters
    ------------
    segments : (n, 2, 2) float
      Line segments to convert
    digits : int
      Number of digits to include in SVG string
    matrix : None or (3, 3) float
      Homogeneous 2D transformation to apply before export

    Returns
    -----------
    path : str
      SVG path string with one line per segment
      IE: 'M 0.1 0.2 L 10 12'
    """
    segments = np.array(segments, copy=True)
    if not util.is_shape(segments, (-1, 2, 2)):
        raise ValueError("only for (n, 2, 2) segments!")

    # create the array to export
    # apply 2D transformation if passed
    if matrix is not None:
        segments = transformations.transform_points(segments.reshape((-1, 2)), matrix=matrix).reshape((-1, 2, 2))

    if merge:
        # remove duplicate and zero-length segments
        segments = unique(segments, digits=digits)

    # create the format string for a single line segment
    base = "M_ _L_ _".replace("_", "{:0." + str(int(digits)) + "f}")
    # create one large format string then apply points
    result = (base * len(segments)).format(*segments.ravel())
    return result
