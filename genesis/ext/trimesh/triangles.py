"""
triangles.py
-------------

Functions for dealing with triangle soups in (n, 3, 3) float form.
"""

import numpy as np

from . import util

from .util import unitize, diagonal_dot
from .points import point_plane_distance
from .constants import tol


def cross(triangles):
    """
    Returns the cross product of two edges from input triangles

    Parameters
    --------------
    triangles: (n, 3, 3) float
      Vertices of triangles

    Returns
    --------------
    crosses : (n, 3) float
      Cross product of two edge vectors
    """
    vectors = np.diff(triangles, axis=1)
    crosses = np.cross(vectors[:, 0], vectors[:, 1])
    return crosses


def area(triangles=None, crosses=None, sum=False):
    """
    Calculates the sum area of input triangles

    Parameters
    ----------
    triangles : (n, 3, 3) float
      Vertices of triangles
    crosses : (n, 3) float or None
      As a speedup don't re- compute cross products
    sum : bool
      Return summed area or individual triangle area

    Returns
    ----------
    area : (n,) float or float
      Individual or summed area depending on `sum` argument
    """
    if crosses is None:
        crosses = cross(triangles)
    areas = np.sqrt((crosses**2).sum(axis=1)) / 2.0
    if sum:
        return areas.sum()
    return areas


def normals(triangles=None, crosses=None):
    """
    Calculates the normals of input triangles

    Parameters
    ------------
    triangles : (n, 3, 3) float
      Vertex positions
    crosses : (n, 3) float
      Cross products of edge vectors

    Returns
    ------------
    normals : (m, 3) float
      Normal vectors
    valid : (n,) bool
      Was the face nonzero area or not
    """
    if crosses is None:
        crosses = cross(triangles)
    # unitize the cross product vectors
    unit, valid = unitize(crosses, check_valid=True)
    return unit, valid


def angles(triangles):
    """
    Calculates the angles of input triangles.

    Parameters
    ------------
    triangles : (n, 3, 3) float
      Vertex positions

    Returns
    ------------
    angles : (n, 3) float
      Angles at vertex positions in radians
      Degenerate angles will be returned as zero
    """
    # don't copy triangles
    triangles = np.asanyarray(triangles, dtype=np.float64)

    # get a unit vector for each edge of the triangle
    u = unitize(triangles[:, 1] - triangles[:, 0])
    v = unitize(triangles[:, 2] - triangles[:, 0])
    w = unitize(triangles[:, 2] - triangles[:, 1])

    # run the cosine and per-row dot product
    result = np.zeros((len(triangles), 3), dtype=np.float64)
    # clip to make sure we don't float error past 1.0
    result[:, 0] = np.arccos(np.clip(diagonal_dot(u, v), -1, 1))
    result[:, 1] = np.arccos(np.clip(diagonal_dot(-u, w), -1, 1))
    # the third angle is just the remaining
    result[:, 2] = np.pi - result[:, 0] - result[:, 1]

    # a triangle with any zero angles is degenerate
    # so set all of the angles to zero in that case
    result[(result < tol.merge).any(axis=1), :] = 0.0

    return result


def all_coplanar(triangles):
    """
    Check to see if a list of triangles are all coplanar

    Parameters
    ----------------
    triangles: (n, 3, 3) float
      Vertices of triangles

    Returns
    ---------------
    all_coplanar : bool
      True if all triangles are coplanar
    """
    triangles = np.asanyarray(triangles, dtype=np.float64)
    if not util.is_shape(triangles, (-1, 3, 3)):
        raise ValueError("Triangles must be (n, 3, 3)!")

    test_normal = normals(triangles)[0]
    test_vertex = triangles[0][0]
    distances = point_plane_distance(
        points=triangles[1:].reshape((-1, 3)), plane_normal=test_normal, plane_origin=test_vertex
    )
    all_coplanar = np.all(np.abs(distances) < tol.zero)
    return all_coplanar


def any_coplanar(triangles):
    """
    For a list of triangles if the FIRST triangle is coplanar
    with ANY of the following triangles, return True.
    Otherwise, return False.
    """
    triangles = np.asanyarray(triangles, dtype=np.float64)
    if not util.is_shape(triangles, (-1, 3, 3)):
        raise ValueError("Triangles must be (n, 3, 3)!")

    test_normal = normals(triangles)[0]
    test_vertex = triangles[0][0]
    distances = point_plane_distance(
        points=triangles[1:].reshape((-1, 3)), plane_normal=test_normal, plane_origin=test_vertex
    )
    any_coplanar = np.any(np.all(np.abs(distances.reshape((-1, 3)) < tol.zero), axis=1))
    return any_coplanar


def mass_properties(triangles, crosses=None, density=None, center_mass=None, skip_inertia=False):
    """
    Calculate the mass properties of a group of triangles.

    Implemented from:
    http://www.geometrictools.com/Documentation/PolyhedralMassProperties.pdf

    Parameters
    ----------
    triangles : (n, 3, 3) float
      Triangle vertices in space
    crosses : (n,) float
      Optional cross products of triangles
    density : float
      Optional override for density
    center_mass :  (3,) float
      Optional override for center mass
    skip_inertia : bool
      if True will not return moments matrix

    Returns
    ---------
    info : dict
      Mass properties
    """
    triangles = np.asanyarray(triangles, dtype=np.float64)
    if not util.is_shape(triangles, (-1, 3, 3)):
        raise ValueError("Triangles must be (n, 3, 3)!")

    if crosses is None:
        crosses = cross(triangles)
    if density is None:
        density = 1.0

    # these are the subexpressions of the integral
    # this is equvilant but 7x faster than triangles.sum(axis=1)
    f1 = triangles[:, 0, :] + triangles[:, 1, :] + triangles[:, 2, :]

    # for the the first vertex of every triangle:
    # triangles[:,0,:] will give rows like [[x0, y0, z0], ...]

    # for the x coordinates of every triangle
    # triangles[:,:,0] will give rows like [[x0, x1, x2], ...]
    f2 = (
        triangles[:, 0, :] ** 2
        + triangles[:, 1, :] ** 2
        + triangles[:, 0, :] * triangles[:, 1, :]
        + triangles[:, 2, :] * f1
    )
    f3 = (
        (triangles[:, 0, :] ** 3)
        + (triangles[:, 0, :] ** 2) * (triangles[:, 1, :])
        + (triangles[:, 0, :]) * (triangles[:, 1, :] ** 2)
        + (triangles[:, 1, :] ** 3)
        + (triangles[:, 2, :] * f2)
    )
    g0 = f2 + (triangles[:, 0, :] + f1) * triangles[:, 0, :]
    g1 = f2 + (triangles[:, 1, :] + f1) * triangles[:, 1, :]
    g2 = f2 + (triangles[:, 2, :] + f1) * triangles[:, 2, :]
    integral = np.zeros((10, len(f1)))
    integral[0] = crosses[:, 0] * f1[:, 0]
    integral[1:4] = (crosses * f2).T
    integral[4:7] = (crosses * f3).T
    for i in range(3):
        triangle_i = np.mod(i + 1, 3)
        integral[i + 7] = crosses[:, i] * (
            (triangles[:, 0, triangle_i] * g0[:, i])
            + (triangles[:, 1, triangle_i] * g1[:, i])
            + (triangles[:, 2, triangle_i] * g2[:, i])
        )

    coefficients = 1.0 / np.array([6, 24, 24, 24, 60, 60, 60, 120, 120, 120], dtype=np.float64)
    integrated = integral.sum(axis=1) * coefficients

    volume = integrated[0]

    if center_mass is None:
        if np.abs(volume) < tol.zero:
            # if there is no volume set center of mass to the origin
            center_mass = np.zeros(3, dtype=np.float64)
        else:
            # otherwise get it from the integration
            center_mass = integrated[1:4] / volume

    result = {"density": density, "mass": density * volume, "volume": volume, "center_mass": center_mass}

    if skip_inertia:
        return result

    inertia = np.zeros((3, 3))
    inertia[0, 0] = integrated[5] + integrated[6] - (volume * (center_mass[[1, 2]] ** 2).sum())
    inertia[1, 1] = integrated[4] + integrated[6] - (volume * (center_mass[[0, 2]] ** 2).sum())
    inertia[2, 2] = integrated[4] + integrated[5] - (volume * (center_mass[[0, 1]] ** 2).sum())
    inertia[0, 1] = -(integrated[7] - (volume * np.prod(center_mass[[0, 1]])))
    inertia[1, 2] = -(integrated[8] - (volume * np.prod(center_mass[[1, 2]])))
    inertia[0, 2] = -(integrated[9] - (volume * np.prod(center_mass[[0, 2]])))
    inertia[2, 0] = inertia[0, 2]
    inertia[2, 1] = inertia[1, 2]
    inertia[1, 0] = inertia[0, 1]
    inertia *= density
    result["inertia"] = inertia

    return result


def windings_aligned(triangles, normals_compare):
    """
    Given a list of triangles and a list of normals determine if the
    two are aligned

    Parameters
    ----------
    triangles : (n, 3, 3) float
      Vertex locations in space
    normals_compare : (n, 3) float
      List of normals to compare

    Returns
    ----------
    aligned : (n,) bool
      Are normals aligned with triangles
    """
    triangles = np.asanyarray(triangles, dtype=np.float64)
    if not util.is_shape(triangles, (-1, 3, 3), allow_zeros=True):
        raise ValueError("triangles must have shape (n, 3, 3), got %s" % str(triangles.shape))
    normals_compare = np.asanyarray(normals_compare, dtype=np.float64)

    calculated, valid = normals(triangles)
    if normals_compare.shape == (3,):
        # single comparison vector case
        difference = np.dot(calculated, normals_compare)
    else:
        # multiple comparison case
        difference = diagonal_dot(calculated, normals_compare[valid])

    aligned = np.zeros(len(triangles), dtype=bool)
    aligned[valid] = difference > 0.0

    return aligned


def bounds_tree(triangles):
    """
    Given a list of triangles, create an r-tree for broad- phase
    collision detection

    Parameters
    ---------
    triangles : (n, 3, 3) float
      Triangles in space

    Returns
    ---------
    tree : rtree.Rtree
      One node per triangle
    """
    triangles = np.asanyarray(triangles, dtype=np.float64)
    if not util.is_shape(triangles, (-1, 3, 3)):
        raise ValueError("Triangles must be (n, 3, 3)!")

    # the (n,6) interleaved bounding box for every triangle
    triangle_bounds = np.column_stack((triangles.min(axis=1), triangles.max(axis=1)))
    tree = util.bounds_tree(triangle_bounds)
    return tree


def nondegenerate(triangles, areas=None, height=None):
    """
    Find all triangles which have an oriented bounding box
    where both of the two sides is larger than a specified height.

    Degenerate triangles can be when:
    1) Two of the three vertices are colocated
    2) All three vertices are unique but colinear


    Parameters
    ----------
    triangles : (n, 3, 3) float
      Triangles in space
    height : float
      Minimum edge length of a triangle to keep

    Returns
    ----------
    nondegenerate : (n,) bool
      True if a triangle meets required minimum height
    """
    triangles = np.asanyarray(triangles, dtype=np.float64)
    if not util.is_shape(triangles, (-1, 3, 3)):
        raise ValueError("Triangles must be (n, 3, 3)!")

    if height is None:
        height = tol.merge

    # if both edges of the triangles OBB are longer than tol.merge
    # we declare them to be nondegenerate
    ok = (extents(triangles=triangles, areas=areas) > height).all(axis=1)

    return ok


def extents(triangles, areas=None):
    """
    Return the 2D bounding box size of each triangle.

    Parameters
    ----------
    triangles : (n, 3, 3) float
      Triangles in space
    areas : (n,) float
      Optional area of input triangles

    Returns
    ----------
    box :  (n, 2) float
      The size of each triangle's 2D oriented bounding box
    """
    triangles = np.asanyarray(triangles, dtype=np.float64)
    if not util.is_shape(triangles, (-1, 3, 3)):
        raise ValueError("Triangles must be (n, 3, 3)!")

    if areas is None:
        areas = area(triangles=triangles, sum=False)

    # the edge vectors which define the triangle
    a = triangles[:, 1] - triangles[:, 0]
    b = triangles[:, 2] - triangles[:, 0]

    # length of the edge vectors
    length_a = (a**2).sum(axis=1) ** 0.5
    length_b = (b**2).sum(axis=1) ** 0.5

    # which edges are acceptable length
    nonzero_a = length_a > tol.merge
    nonzero_b = length_b > tol.merge

    # find the two heights of the triangle
    # essentially this is the side length of an
    # oriented bounding box, per triangle
    box = np.zeros((len(triangles), 2), dtype=np.float64)
    box[:, 0][nonzero_a] = (areas[nonzero_a] * 2) / length_a[nonzero_a]
    box[:, 1][nonzero_b] = (areas[nonzero_b] * 2) / length_b[nonzero_b]

    return box


def barycentric_to_points(triangles, barycentric):
    """
    Convert a list of barycentric coordinates on a list of triangles
    to cartesian points.

    Parameters
    ------------
    triangles : (n, 3, 3) float
      Triangles in space
    barycentric : (n, 2) float
      Barycentric coordinates

    Returns
    -----------
    points : (m, 3) float
      Points in space
    """
    barycentric = np.asanyarray(barycentric, dtype=np.float64)
    triangles = np.asanyarray(triangles, dtype=np.float64)

    if not util.is_shape(triangles, (-1, 3, 3)):
        raise ValueError("Triangles must be (n, 3, 3)!")
    if barycentric.shape == (2,):
        barycentric = np.ones((len(triangles), 2), dtype=np.float64) * barycentric
    if util.is_shape(barycentric, (len(triangles), 2)):
        barycentric = np.column_stack((barycentric, 1.0 - barycentric.sum(axis=1)))
    elif not util.is_shape(barycentric, (len(triangles), 3)):
        raise ValueError("Barycentric shape incorrect!")

    barycentric /= barycentric.sum(axis=1).reshape((-1, 1))
    points = (triangles * barycentric.reshape((-1, 3, 1))).sum(axis=1)

    return points


def points_to_barycentric(triangles, points, method="cramer"):
    """
    Find the barycentric coordinates of points relative to triangles.

    The Cramer's rule solution implements:
        http://blackpawn.com/texts/pointinpoly

    The cross product solution implements:
        https://www.cs.ubc.ca/~heidrich/Papers/JGT.05.pdf


    Parameters
    -----------
    triangles : (n, 3, 3) float
      Triangles vertices in space
    points : (n, 3) float
      Point in space associated with a triangle
    method :  str
      Which method to compute the barycentric coordinates with:
        - 'cross': uses a method using cross products, roughly 2x slower but
                  different numerical robustness properties
        - anything else: uses a cramer's rule solution

    Returns
    -----------
    barycentric : (n, 3) float
      Barycentric coordinates of each point
    """

    def method_cross():
        n = np.cross(edge_vectors[:, 0], edge_vectors[:, 1])
        denominator = diagonal_dot(n, n)

        barycentric = np.zeros((len(triangles), 3), dtype=np.float64)
        barycentric[:, 2] = diagonal_dot(np.cross(edge_vectors[:, 0], w), n) / denominator
        barycentric[:, 1] = diagonal_dot(np.cross(w, edge_vectors[:, 1]), n) / denominator
        barycentric[:, 0] = 1 - barycentric[:, 1] - barycentric[:, 2]
        return barycentric

    def method_cramer():
        dot00 = diagonal_dot(edge_vectors[:, 0], edge_vectors[:, 0])
        dot01 = diagonal_dot(edge_vectors[:, 0], edge_vectors[:, 1])
        dot02 = diagonal_dot(edge_vectors[:, 0], w)
        dot11 = diagonal_dot(edge_vectors[:, 1], edge_vectors[:, 1])
        dot12 = diagonal_dot(edge_vectors[:, 1], w)

        inverse_denominator = 1.0 / (dot00 * dot11 - dot01 * dot01)

        barycentric = np.zeros((len(triangles), 3), dtype=np.float64)
        barycentric[:, 2] = (dot00 * dot12 - dot01 * dot02) * inverse_denominator
        barycentric[:, 1] = (dot11 * dot02 - dot01 * dot12) * inverse_denominator
        barycentric[:, 0] = 1 - barycentric[:, 1] - barycentric[:, 2]
        return barycentric

    # establish that input triangles and points are sane
    triangles = np.asanyarray(triangles, dtype=np.float64)
    points = np.asanyarray(points, dtype=np.float64)
    if not util.is_shape(triangles, (-1, 3, 3)):
        raise ValueError("triangles shape incorrect")
    if not util.is_shape(points, (len(triangles), 3)):
        raise ValueError("triangles and points must correspond")

    edge_vectors = triangles[:, 1:] - triangles[:, :1]
    w = points - triangles[:, 0].reshape((-1, 3))

    if method == "cross":
        return method_cross()
    return method_cramer()


def closest_point(triangles, points):
    """
    Return the closest point on the surface of each triangle for a
    list of corresponding points.

    Implements the method from "Real Time Collision Detection" and
    use the same variable names as "ClosestPtPointTriangle" to avoid
    being any more confusing.


    Parameters
    ----------
    triangles : (n, 3, 3) float
      Triangle vertices in space
    points : (n, 3) float
      Points in space

    Returns
    ----------
    closest : (n, 3) float
      Point on each triangle closest to each point
    """

    # check input triangles and points
    triangles = np.asanyarray(triangles, dtype=np.float64)
    points = np.asanyarray(points, dtype=np.float64)
    if not util.is_shape(triangles, (-1, 3, 3)):
        raise ValueError("triangles shape incorrect")
    if not util.is_shape(points, (len(triangles), 3)):
        raise ValueError("need same number of triangles and points!")

    # store the location of the closest point
    result = np.zeros_like(points)
    # which points still need to be handled
    remain = np.ones(len(points), dtype=bool)

    # if we dot product this against a (n, 3)
    # it is equivalent but faster than array.sum(axis=1)
    ones = [1.0, 1.0, 1.0]

    # get the three points of each triangle
    # use the same notation as RTCD to avoid confusion
    a = triangles[:, 0, :]
    b = triangles[:, 1, :]
    c = triangles[:, 2, :]

    # check if P is in vertex region outside A
    ab = b - a
    ac = c - a
    ap = points - a
    # this is a faster equivalent of:
    # diagonal_dot(ab, ap)
    d1 = np.dot(ab * ap, ones)
    d2 = np.dot(ac * ap, ones)

    # is the point at A
    is_a = np.logical_and(d1 < tol.zero, d2 < tol.zero)
    if any(is_a):
        result[is_a] = a[is_a]
        remain[is_a] = False

    # check if P in vertex region outside B
    bp = points - b
    d3 = np.dot(ab * bp, ones)
    d4 = np.dot(ac * bp, ones)

    # do the logic check
    is_b = (d3 > -tol.zero) & (d4 <= d3) & remain
    if any(is_b):
        result[is_b] = b[is_b]
        remain[is_b] = False

    # check if P in edge region of AB, if so return projection of P onto A
    vc = (d1 * d4) - (d3 * d2)
    is_ab = (vc < tol.zero) & (d1 > -tol.zero) & (d3 < tol.zero) & remain
    if any(is_ab):
        v = (d1[is_ab] / (d1[is_ab] - d3[is_ab])).reshape((-1, 1))
        result[is_ab] = a[is_ab] + (v * ab[is_ab])
        remain[is_ab] = False

    # check if P in vertex region outside C
    cp = points - c
    d5 = np.dot(ab * cp, ones)
    d6 = np.dot(ac * cp, ones)
    is_c = (d6 > -tol.zero) & (d5 <= d6) & remain
    if any(is_c):
        result[is_c] = c[is_c]
        remain[is_c] = False

    # check if P in edge region of AC, if so return projection of P onto AC
    vb = (d5 * d2) - (d1 * d6)
    is_ac = (vb < tol.zero) & (d2 > -tol.zero) & (d6 < tol.zero) & remain
    if any(is_ac):
        w = (d2[is_ac] / (d2[is_ac] - d6[is_ac])).reshape((-1, 1))
        result[is_ac] = a[is_ac] + w * ac[is_ac]
        remain[is_ac] = False

    # check if P in edge region of BC, if so return projection of P onto BC
    va = (d3 * d6) - (d5 * d4)
    is_bc = (va < tol.zero) & ((d4 - d3) > -tol.zero) & ((d5 - d6) > -tol.zero) & remain
    if any(is_bc):
        d43 = d4[is_bc] - d3[is_bc]
        w = (d43 / (d43 + (d5[is_bc] - d6[is_bc]))).reshape((-1, 1))
        result[is_bc] = b[is_bc] + w * (c[is_bc] - b[is_bc])
        remain[is_bc] = False

    # any remaining points must be inside face region
    if any(remain):
        # point is inside face region
        denom = 1.0 / (va[remain] + vb[remain] + vc[remain])
        v = (vb[remain] * denom).reshape((-1, 1))
        w = (vc[remain] * denom).reshape((-1, 1))
        # compute Q through its barycentric coordinates
        result[remain] = a[remain] + (ab[remain] * v) + (ac[remain] * w)

    return result


def to_kwargs(triangles):
    """
    Convert a list of triangles to the kwargs for the Trimesh
    constructor.

    Parameters
    ---------
    triangles : (n, 3, 3) float
      Triangles in space

    Returns
    ---------
    kwargs : dict
      Keyword arguments for the trimesh.Trimesh constructor
      Includes keys 'vertices' and 'faces'

    Examples
    ---------
    >>> mesh = trimesh.Trimesh(**trimesh.triangles.to_kwargs(triangles))
    """
    triangles = np.asanyarray(triangles, dtype=np.float64)
    if not util.is_shape(triangles, (-1, 3, 3)):
        raise ValueError("Triangles must be (n, 3, 3)!")

    vertices = triangles.reshape((-1, 3))
    faces = np.arange(len(vertices)).reshape((-1, 3))
    kwargs = {"vertices": vertices, "faces": faces}

    return kwargs
