import numpy as np

from shapely import ops
from shapely.geometry import Polygon

from .. import bounds
from .. import graph
from .. import geometry
from .. import grouping

from ..constants import log
from ..constants import tol_path as tol
from ..transformations import transform_points

from .simplify import fit_circle_check
from .traversal import resample_path

try:
    import networkx as nx
except BaseException as E:
    # create a dummy module which will raise the ImportError
    # or other exception only when someone tries to use networkx
    from ..exceptions import ExceptionWrapper

    nx = ExceptionWrapper(E)
try:
    from rtree import Rtree
except BaseException as E:
    # create a dummy module which will raise the ImportError
    from ..exceptions import ExceptionWrapper

    Rtree = ExceptionWrapper(E)


def enclosure_tree(polygons):
    """
    Given a list of shapely polygons with only exteriors,
    find which curves represent the exterior shell or root curve
    and which represent holes which penetrate the exterior.

    This is done with an R-tree for rough overlap detection,
    and then exact polygon queries for a final result.

    Parameters
    -----------
    polygons : (n,) shapely.geometry.Polygon
       Polygons which only have exteriors and may overlap

    Returns
    -----------
    roots : (m,) int
        Index of polygons which are root
    contains : networkx.DiGraph
       Edges indicate a polygon is
       contained by another polygon
    """
    tree = Rtree()
    # nodes are indexes in polygons
    contains = nx.DiGraph()
    for i, polygon in enumerate(polygons):
        # if a polygon is None it means creation
        # failed due to weird geometry so ignore it
        if polygon is None or len(polygon.bounds) != 4:
            continue
        # insert polygon bounds into rtree
        tree.insert(i, polygon.bounds)
        # make sure every valid polygon has a node
        contains.add_node(i)

    # loop through every polygon
    for i in contains.nodes():
        polygon = polygons[i]
        # we first query for bounding box intersections from the R-tree
        for j in tree.intersection(polygon.bounds):
            # if we are checking a polygon against itself continue
            if i == j:
                continue
            # do a more accurate polygon in polygon test
            # for the enclosure tree information
            if polygons[i].contains(polygons[j]):
                contains.add_edge(i, j)
            elif polygons[j].contains(polygons[i]):
                contains.add_edge(j, i)

    # a root or exterior curve has an even number of parents
    # wrap in dict call to avoid networkx view
    degree = dict(contains.in_degree())
    # convert keys and values to numpy arrays
    indexes = np.array(list(degree.keys()))
    degrees = np.array(list(degree.values()))
    # roots are curves with an even inward degree (parent count)
    roots = indexes[(degrees % 2) == 0]
    # if there are multiple nested polygons split the graph
    # so the contains logic returns the individual polygons
    if len(degrees) > 0 and degrees.max() > 1:
        # collect new edges for graph
        edges = []

        # order the roots so they are sorted by degree
        roots = roots[np.argsort([degree[r] for r in roots])]
        # find edges of subgraph for each root and children
        for root in roots:
            children = indexes[degrees == degree[root] + 1]
            edges.extend(contains.subgraph(np.append(children, root)).edges())
        # stack edges into new directed graph
        contains = nx.from_edgelist(edges, nx.DiGraph())
        # if roots have no children add them anyway
        contains.add_nodes_from(roots)

    return roots, contains


def edges_to_polygons(edges, vertices):
    """
    Given an edge list of indices and associated vertices
    representing lines, generate a list of polygons.

    Parameters
    -----------
    edges : (n, 2) int
      Indexes of vertices which represent lines
    vertices : (m, 2) float
      Vertices in 2D space

    Returns
    ----------
    polygons : (p,) shapely.geometry.Polygon
      Polygon objects with interiors
    """

    assert isinstance(vertices, np.ndarray)

    # create closed polygon objects
    polygons = []
    # loop through a sequence of ordered traversals
    for dfs in graph.traversals(edges, mode="dfs"):
        try:
            # try to recover polygons before they are more complicated
            repaired = repair_invalid(Polygon(vertices[dfs]))
            # if it returned a multipolygon extend into a flat list
            if hasattr(repaired, "geoms"):
                polygons.extend(repaired.geoms)
            else:
                polygons.append(repaired)
        except ValueError:
            continue

    # if there is only one polygon, just return it
    if len(polygons) == 1:
        return polygons

    # find which polygons contain which other polygons
    roots, tree = enclosure_tree(polygons)

    # generate list of polygons with proper interiors
    complete = []
    for root in roots:
        interior = list(tree[root].keys())
        shell = polygons[root].exterior.coords
        holes = [polygons[i].exterior.coords for i in interior]
        complete.append(Polygon(shell=shell, holes=holes))
    return complete


def polygons_obb(polygons):
    """
    Find the OBBs for a list of shapely.geometry.Polygons
    """
    rectangles = [None] * len(polygons)
    transforms = [None] * len(polygons)
    for i, p in enumerate(polygons):
        transforms[i], rectangles[i] = polygon_obb(p)
    return np.array(transforms), np.array(rectangles)


def polygon_obb(polygon):
    """
    Find the oriented bounding box of a Shapely polygon.

    The OBB is always aligned with an edge of the convex hull of the polygon.

    Parameters
    -------------
    polygons : shapely.geometry.Polygon
      Input geometry

    Returns
    -------------
    transform : (3, 3) float
      Transformation matrix
      which will move input polygon from its original position
      to the first quadrant where the AABB is the OBB
    extents : (2,) float
      Extents of transformed polygon
    """
    if hasattr(polygon, "exterior"):
        points = np.asanyarray(polygon.exterior.coords)
    elif isinstance(polygon, np.ndarray):
        points = polygon
    else:
        raise ValueError("polygon or points must be provided")

    transform, extents = bounds.oriented_bounds_2D(points)

    if tol.strict:
        moved = transform_points(points=points, matrix=transform)
        assert np.allclose(-extents / 2.0, moved.min(axis=0))
        assert np.allclose(extents / 2.0, moved.max(axis=0))

    return transform, extents


def transform_polygon(polygon, matrix):
    """
    Transform a polygon by a a 2D homogeneous transform.

    Parameters
    -------------
    polygon : shapely.geometry.Polygon
      2D polygon to be transformed.
    matrix  : (3, 3) float
      2D homogeneous transformation.

    Returns
    --------------
    result : shapely.geometry.Polygon
      Polygon transformed by matrix.
    """
    matrix = np.asanyarray(matrix, dtype=np.float64)

    if hasattr(polygon, "geoms"):
        result = [transform_polygon(p, t) for p, t in zip(polygon, matrix)]
        return result
    # transform the outer shell
    shell = transform_points(np.array(polygon.exterior.coords), matrix)[:, :2]
    # transform the interiors
    holes = [transform_points(np.array(i.coords), matrix)[:, :2] for i in polygon.interiors]
    # create a new polygon with the result
    result = Polygon(shell=shell, holes=holes)
    return result


def polygon_bounds(polygon, matrix=None):
    """
    Get the transformed axis aligned bounding box of a
    shapely Polygon object.

    Parameters
    ------------
    polygon : shapely.geometry.Polygon
      Polygon pre-transform
    matrix : (3, 3) float or None.
      Homogeneous transform moving polygon in space

    Returns
    ------------
    bounds : (2, 2) float
      Axis aligned bounding box of transformed polygon.
    """
    if matrix is not None:
        assert matrix.shape == (3, 3)
        points = transform_points(points=np.array(polygon.exterior.coords), matrix=matrix)
    else:
        points = np.array(polygon.exterior.coords)

    bounds = np.array([points.min(axis=0), points.max(axis=0)])
    assert bounds.shape == (2, 2)
    return bounds


def plot(polygon=None, show=True, axes=None, **kwargs):
    """
    Plot a shapely polygon using matplotlib.

    Parameters
    ------------
    polygon : shapely.geometry.Polygon
      Polygon to be plotted
    show : bool
      If True will display immediately
    **kwargs
      Passed to plt.plot
    """
    import matplotlib.pyplot as plt

    def plot_single(single):
        axes.plot(*single.exterior.xy, **kwargs)
        for interior in single.interiors:
            axes.plot(*interior.xy, **kwargs)

    # make aspect ratio non-stupid
    if axes is None:
        axes = plt.axes()
    axes.set_aspect("equal", "datalim")

    if polygon.__class__.__name__ == "MultiPolygon":
        [plot_single(i) for i in polygon.geoms]
    elif hasattr(polygon, "__iter__"):
        [plot_single(i) for i in polygon]
    elif polygon is not None:
        plot_single(polygon)

    if show:
        plt.show()

    return axes


def resample_boundaries(polygon, resolution, clip=None):
    """
    Return a version of a polygon with boundaries resampled
    to a specified resolution.

    Parameters
    -------------
    polygon : shapely.geometry.Polygon
      Source geometry
    resolution : float
      Desired distance between points on boundary
    clip : (2,) int
      Upper and lower bounds to clip
      number of samples to avoid exploding count

    Returns
    ------------
    kwargs : dict
     Keyword args for a Polygon constructor `Polygon(**kwargs)`
    """

    def resample_boundary(boundary):
        # add a polygon.exterior or polygon.interior to
        # the deque after resampling based on our resolution
        count = boundary.length / resolution
        count = int(np.clip(count, *clip))
        return resample_path(boundary.coords, count=count)

    if clip is None:
        clip = [8, 200]
    # create a sequence of [(n,2)] points
    kwargs = {"shell": resample_boundary(polygon.exterior), "holes": []}
    for interior in polygon.interiors:
        kwargs["holes"].append(resample_boundary(interior))

    return kwargs


def stack_boundaries(boundaries):
    """
    Stack the boundaries of a polygon into a single
    (n, 2) list of vertices.

    Parameters
    ------------
    boundaries : dict
      With keys 'shell', 'holes'

    Returns
    ------------
    stacked : (n, 2) float
      Stacked vertices
    """
    if len(boundaries["holes"]) == 0:
        return boundaries["shell"]
    result = np.vstack((boundaries["shell"], np.vstack(boundaries["holes"])))
    return result


def medial_axis(polygon, resolution=None, clip=None):
    """
    Given a shapely polygon, find the approximate medial axis
    using a voronoi diagram of evenly spaced points on the
    boundary of the polygon.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
      The source geometry
    resolution : float
      Distance between each sample on the polygon boundary
    clip : None, or (2,) int
      Clip sample count to min of clip[0] and max of clip[1]

    Returns
    ----------
    edges : (n, 2) int
      Vertex indices representing line segments
      on the polygon's medial axis
    vertices : (m, 2) float
      Vertex positions in space
    """
    # a circle will have a single point medial axis
    if len(polygon.interiors) == 0:
        # what is the approximate scale of the polygon
        scale = np.ptp(np.reshape(polygon.bounds, (2, 2)), axis=0).max()
        # a (center, radius, error) tuple
        fit = fit_circle_check(polygon.exterior.coords, scale=scale)
        # is this polygon in fact a circle
        if fit is not None:
            # return an edge that has the center as the midpoint
            epsilon = np.clip(fit["radius"] / 500, 1e-5, np.inf)
            vertices = np.array([fit["center"] + [0, epsilon], fit["center"] - [0, epsilon]], dtype=np.float64)
            # return a single edge to avoid consumers needing to special case
            edges = np.array([[0, 1]], dtype=np.int64)
            return edges, vertices

    from scipy.spatial import Voronoi
    from shapely import vectorized

    if resolution is None:
        resolution = np.ptp(np.reshape(polygon.bounds, (2, 2)), axis=0).max() / 100

    # get evenly spaced points on the polygons boundaries
    samples = resample_boundaries(polygon=polygon, resolution=resolution, clip=clip)
    # stack the boundary into a (m,2) float array
    samples = stack_boundaries(samples)
    # create the voronoi diagram on 2D points
    voronoi = Voronoi(samples)
    # which voronoi vertices are contained inside the polygon
    contains = vectorized.contains(polygon, *voronoi.vertices.T)
    # ridge vertices of -1 are outside, make sure they are False
    contains = np.append(contains, False)
    # make sure ridge vertices is numpy array
    ridge = np.asanyarray(voronoi.ridge_vertices, dtype=np.int64)
    # only take ridges where every vertex is contained
    edges = ridge[contains[ridge].all(axis=1)]

    # now we need to remove uncontained vertices
    contained = np.unique(edges)
    mask = np.zeros(len(voronoi.vertices), dtype=np.int64)
    mask[contained] = np.arange(len(contained))

    # mask voronoi vertices
    vertices = voronoi.vertices[contained]
    # re-index edges
    edges_final = mask[edges]

    if tol.strict:
        # make sure we didn't screw up indexes
        assert np.ptp((vertices[edges_final] - voronoi.vertices[edges])) < 1e-5

    return edges_final, vertices


def polygon_hash(polygon):
    """
    Return a vector containing values representitive of
    a particular polygon.

    Parameters
    ---------
    polygon : shapely.geometry.Polygon
      Input geometry

    Returns
    ---------
    hashed: (6), float
      Representitive values representing input polygon
    """
    result = np.array(
        [
            len(polygon.interiors),
            polygon.convex_hull.area,
            polygon.convex_hull.length,
            polygon.area,
            polygon.length,
            polygon.exterior.length,
        ],
        dtype=np.float64,
    )
    return result


def random_polygon(segments=8, radius=1.0):
    """
    Generate a random polygon with a maximum number of sides and approximate radius.

    Parameters
    ---------
    segments : int
      The maximum number of sides the random polygon will have
    radius : float
      The approximate radius of the polygon desired

    Returns
    ---------
    polygon : shapely.geometry.Polygon
      Geometry object with random exterior and no interiors.
    """
    angles = np.sort(np.cumsum(np.random.random(segments) * np.pi * 2) % (np.pi * 2))
    radii = np.random.random(segments) * radius
    points = np.column_stack((np.cos(angles), np.sin(angles))) * radii.reshape((-1, 1))
    points = np.vstack((points, points[0]))
    polygon = Polygon(points).buffer(0.0)
    if hasattr(polygon, "geoms"):
        return polygon.geoms[0]
    return polygon


def polygon_scale(polygon):
    """
    For a Polygon object return the diagonal length of the AABB.

    Parameters
    ------------
    polygon : shapely.geometry.Polygon
      Source geometry

    Returns
    ------------
    scale : float
      Length of AABB diagonal
    """
    extents = np.ptp(np.reshape(polygon.bounds, (2, 2)), axis=0)
    scale = (extents**2).sum() ** 0.5

    return scale


def paths_to_polygons(paths, scale=None):
    """
    Given a sequence of connected points turn them into
    valid shapely Polygon objects.

    Parameters
    -----------
    paths : (n,) sequence
      Of (m, 2) float closed paths
    scale: float
      Approximate scale of drawing for precision

    Returns
    -----------
    polys : (p,) list
      Filled with Polygon or None

    """
    polygons = [None] * len(paths)
    for i, path in enumerate(paths):
        if len(path) < 4:
            # since the first and last vertices are identical in
            # a closed loop a 4 vertex path is the minimum for
            # non-zero area
            continue
        try:
            polygons[i] = repair_invalid(Polygon(path), scale)
        except ValueError:
            # raised if a polygon is unrecoverable
            continue
        except BaseException:
            log.error("unrecoverable polygon", exc_info=True)
    polygons = np.array(polygons)
    return polygons


def sample(polygon, count, factor=1.5, max_iter=10):
    """
    Use rejection sampling to generate random points inside a
    polygon.

    Parameters
    -----------
    polygon : shapely.geometry.Polygon
      Polygon that will contain points
    count : int
      Number of points to return
    factor : float
      How many points to test per loop
    max_iter : int
      Maximum number of intersection checks is:
      > count * factor * max_iter

    Returns
    -----------
    hit : (n, 2) float
      Random points inside polygon
      where n <= count
    """
    # do batch point-in-polygon queries
    from shapely import vectorized

    # get size of bounding box
    bounds = np.reshape(polygon.bounds, (2, 2))
    extents = np.ptp(bounds, axis=0)

    # how many points to check per loop iteration
    per_loop = int(count * factor)

    # start with some rejection sampling
    points = bounds[0] + extents * np.random.random((per_loop, 2))
    # do the point in polygon test and append resulting hits
    mask = vectorized.contains(polygon, *points.T)
    hit = [points[mask]]
    hit_count = len(hit[0])
    # if our first non-looping check got enough samples exit
    if hit_count >= count:
        return hit[0][:count]

    # if we have to do iterations loop here slowly
    for _ in range(max_iter):
        # generate points inside polygons AABB
        points = np.random.random((per_loop, 2))
        points = (points * extents) + bounds[0]
        # do the point in polygon test and append resulting hits
        mask = vectorized.contains(polygon, *points.T)
        hit.append(points[mask])
        # keep track of how many points we've collected
        hit_count += len(hit[-1])
        # if we have enough points exit the loop
        if hit_count > count:
            break

    # stack the hits into an (n,2) array and truncate
    hit = np.vstack(hit)[:count]

    return hit


def repair_invalid(polygon, scale=None, rtol=0.5):
    """
    Given a shapely.geometry.Polygon, attempt to return a
    valid version of the polygon through buffering tricks.

    Parameters
    -----------
    polygon : shapely.geometry.Polygon
      Source geometry
    rtol : float
      How close does a perimeter have to be
    scale : float or None
      For numerical precision reference

    Returns
    ----------
    repaired : shapely.geometry.Polygon
      Repaired polygon

    Raises
    ----------
    ValueError
      If polygon can't be repaired
    """
    if hasattr(polygon, "is_valid") and polygon.is_valid:
        return polygon

    # basic repair involves buffering the polygon outwards
    # this will fix a subset of problems.
    basic = polygon.buffer(tol.zero)
    # if it returned multiple polygons check the largest
    if hasattr(basic, "geoms"):
        basic = basic.geoms[np.argmax([i.area for i in basic.geoms])]

    # check perimeter of result against original perimeter
    if basic.is_valid and np.isclose(basic.length, polygon.length, rtol=rtol):
        return basic

    if scale is None:
        distance = 0.002 * np.ptp(np.reshape(polygon.bounds, (2, 2)), axis=0).mean()
    else:
        distance = 0.002 * scale

    # if there are no interiors, we can work with just the exterior
    # ring, which is often more reliable
    if len(polygon.interiors) == 0:
        # try buffering the exterior of the polygon
        # the interior will be offset by -tol.buffer
        rings = polygon.exterior.buffer(distance).interiors
        if len(rings) == 1:
            # reconstruct a single polygon from the interior ring
            recon = Polygon(shell=rings[0]).buffer(distance)
            # check perimeter of result against original perimeter
            if recon.is_valid and np.isclose(recon.length, polygon.length, rtol=rtol):
                return recon

        # try de-deuplicating the outside ring
        points = np.array(polygon.exterior.coords)
        # remove any segments shorter than tol.merge
        # this is a little risky as if it was discretized more
        # finely than 1-e8 it may remove detail
        unique = np.append(True, (np.diff(points, axis=0) ** 2).sum(axis=1) ** 0.5 > 1e-8)
        # make a new polygon with result
        dedupe = Polygon(shell=points[unique])
        # check result
        if dedupe.is_valid and np.isclose(dedupe.length, polygon.length, rtol=rtol):
            return dedupe

    # buffer and unbuffer the whole polygon
    buffered = polygon.buffer(distance).buffer(-distance)
    # if it returned multiple polygons check the largest
    if hasattr(buffered, "geoms"):
        areas = np.array([b.area for b in buffered.geoms])
        return buffered.geoms[areas.argmax()]

    # check perimeter of result against original perimeter
    if buffered.is_valid and np.isclose(buffered.length, polygon.length, rtol=rtol):
        log.debug("Recovered invalid polygon through double buffering")
        return buffered

    raise ValueError("unable to recover polygon!")


def projected(mesh, normal, origin=None, ignore_sign=True, rpad=1e-5, apad=None, tol_dot=0.01, max_regions=200):
    """
    Project a mesh onto a plane and then extract the polygon
    that outlines the mesh projection on that plane.

    Note that this will ignore back-faces, which is only
    relevant if the source mesh isn't watertight.

    Also padding: this generates a result by unioning the
    polygons of multiple connected regions, which requires
    the polygons be padded by a distance so that a polygon
    union produces a single coherent result. This distance
    is calculated as: `apad + (rpad * scale)`

    Parameters
    ----------
    mesh : trimesh.Trimesh
      Source geometry
    check : bool
      If True make sure is flat
    normal : (3,) float
      Normal to extract flat pattern along
    origin : None or (3,) float
      Origin of plane to project mesh onto
    ignore_sign : bool
      Allow a projection from the normal vector in
      either direction: this provides a substantial speedup
      on watertight meshes where the direction is irrelevant
      but if you have a triangle soup and want to discard
      backfaces you should set this to False.
    rpad : float
      Proportion to pad polygons by before unioning
      and then de-padding result by to avoid zero-width gaps.
    apad : float
      Absolute padding to pad polygons by before unioning
      and then de-padding result by to avoid zero-width gaps.
    tol_dot : float
      Tolerance for discarding on-edge triangles.
    max_regions : int
      Raise an exception if the mesh has more than this
      number of disconnected regions to fail quickly before
      unioning.

    Returns
    ----------
    projected : shapely.geometry.Polygon or None
      Outline of source mesh

    Raises
    ---------
    ValueError
      If max_regions is exceeded
    """
    # make sure normal is a unitized copy
    normal = np.array(normal, dtype=np.float64)
    normal /= np.linalg.norm(normal)

    # the projection of each face normal onto facet normal
    dot_face = np.dot(normal, mesh.face_normals.T)
    if ignore_sign:
        # for watertight mesh speed up projection by handling side with less faces
        # check if face lies on front or back of normal
        front = dot_face > tol_dot
        back = dot_face < -tol_dot
        # divide the mesh into front facing section and back facing parts
        # and discard the faces perpendicular to the axis.
        # since we are doing a unary_union later we can use the front *or*
        # the back so we use which ever one has fewer triangles
        # we want the largest nonzero group
        count = np.array([front.sum(), back.sum()])
        if count.min() == 0:
            # if one of the sides has zero faces we need the other
            pick = count.argmax()
        else:
            # otherwise use the normal direction with the fewest faces
            pick = count.argmin()
        # use the picked side
        side = [front, back][pick]
    else:
        # if explicitly asked to care about the sign
        # only handle the front side of normal
        side = dot_face > tol_dot

    # subset the adjacency pairs to ones which have both faces included
    # on the side we are currently looking at
    adjacency_check = side[mesh.face_adjacency].all(axis=1)
    adjacency = mesh.face_adjacency[adjacency_check]

    # a sequence of face indexes that are connected
    face_groups = graph.connected_components(adjacency, nodes=np.nonzero(side)[0])

    # if something is goofy we may end up with thousands of
    # regions that do nothing except hang for an hour then segfault
    if len(face_groups) > max_regions:
        raise ValueError("too many disconnected groups!")

    # reshape edges into shape length of faces for indexing
    edges = mesh.edges_sorted.reshape((-1, 6))
    # transform from the mesh frame in 3D to the XY plane
    to_2D = geometry.plane_transform(origin=origin, normal=normal)
    # transform mesh vertices to 2D and clip the zero Z
    vertices_2D = transform_points(mesh.vertices, to_2D)[:, :2]

    polygons = []
    for faces in face_groups:
        # index edges by face then shape back to individual edges
        edge = edges[faces].reshape((-1, 2))
        # edges that occur only once are on the boundary
        group = grouping.group_rows(edge, require_count=1)
        # turn each region into polygons
        polygons.extend(edges_to_polygons(edges=edge[group], vertices=vertices_2D))

    padding = 0.0
    if apad is not None:
        # set padding by absolute value
        padding += float(apad)
    if rpad is not None:
        # get the 2D scale as the longest side of the AABB
        scale = np.ptp(vertices_2D, axis=0).max()
        # apply the scale-relative padding
        padding += float(rpad) * scale

    # some types of errors will lead to a bajillion disconnected
    # regions and the union will take forever to fail
    # so exit here early
    if len(polygons) > max_regions:
        raise ValueError("too many disconnected groups!")

    # if there is only one region we don't need to run a union
    elif len(polygons) == 1:
        return polygons[0]
    elif len(polygons) == 0:
        return None
    else:
        # inflate each polygon before unioning to remove zero-size
        # gaps then deflate the result after unioning by the same amount
        # note the following provides a 25% speedup but needs
        # more testing to see if it deflates to a decent looking
        # result:
        # polygon = ops.unary_union(
        #    [p.buffer(padding,
        #              join_style=2,
        #              mitre_limit=1.5)
        #     for p in polygons]).buffer(-padding)
        polygon = ops.unary_union([p.buffer(padding) for p in polygons]).buffer(-padding)
    return polygon


def second_moments(polygon, return_centered=False):
    """
    Calculate the second moments of area of a polygon
    from the boundary.

    Parameters
    ------------
    polygon : shapely.geometry.Polygon
      Closed polygon.
    return_centered : bool
      Get second moments for a frame with origin at the centroid
      and perform a principal axis transformation.

    Returns
    ----------
    moments : (3,) float
      The values of `[Ixx, Iyy, Ixy]`
    principal_moments : (2,) float
      Principal second moments of inertia: `[Imax, Imin]`
      Only returned if `centered`.
    alpha : float
      Angle by which the polygon needs to be rotated, so the
      principal axis align with the X and Y axis.
      Only returned if `centered`.
    transform : (3, 3) float
      Transformation matrix which rotates the polygon by alpha.
      Only returned if `centered`.
    """

    transform = np.eye(3)
    if return_centered:
        # calculate centroid and move polygon
        transform[:2, 2] = -np.array(polygon.centroid.coords)
        polygon = transform_polygon(polygon, transform)

    # start with the exterior
    coords = np.array(polygon.exterior.coords)
    # shorthand the coordinates
    x1, y1 = np.vstack((coords[-1], coords[:-1])).T
    x2, y2 = coords.T
    # do vectorized operations
    v = x1 * y2 - x2 * y1
    Ixx = np.sum(v * (y1 * y1 + y1 * y2 + y2 * y2)) / 12.0
    Iyy = np.sum(v * (x1 * x1 + x1 * x2 + x2 * x2)) / 12.0
    Ixy = np.sum(v * (x1 * y2 + 2 * x1 * y1 + 2 * x2 * y2 + x2 * y1)) / 24.0

    for interior in polygon.interiors:
        coords = np.array(interior.coords)
        # shorthand the coordinates
        x1, y1 = np.vstack((coords[-1], coords[:-1])).T
        x2, y2 = coords.T
        # do vectorized operations
        v = x1 * y2 - x2 * y1
        Ixx -= np.sum(v * (y1 * y1 + y1 * y2 + y2 * y2)) / 12.0
        Iyy -= np.sum(v * (x1 * x1 + x1 * x2 + x2 * x2)) / 12.0
        Ixy -= np.sum(v * (x1 * y2 + 2 * x1 * y1 + 2 * x2 * y2 + x2 * y1)) / 24.0

    moments = [Ixx, Iyy, Ixy]

    if not return_centered:
        return moments

    # get the principal moments
    root = np.sqrt(((Iyy - Ixx) / 2.0) ** 2 + Ixy**2)
    Imax = (Ixx + Iyy) / 2.0 + root
    Imin = (Ixx + Iyy) / 2.0 - root
    principal_moments = [Imax, Imin]

    # do the principal axis transform
    if np.isclose(Ixy, 0.0, atol=1e-12):
        alpha = 0
    elif np.isclose(Ixx, Iyy):
        # prevent division by 0
        alpha = 0.25 * np.pi
    else:
        alpha = 0.5 * np.arctan(2.0 * Ixy / (Ixx - Iyy))

    # construct transformation matrix
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)

    transform[0, 0] = cos_alpha
    transform[1, 1] = cos_alpha
    transform[0, 1] = -sin_alpha
    transform[1, 0] = sin_alpha

    return moments, principal_moments, alpha, transform
