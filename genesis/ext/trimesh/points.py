"""
points.py
-------------

Functions dealing with (n, d) points.
"""

import copy

import numpy as np

from .parent import Geometry3D
from .geometry import plane_transform
from .constants import tol
from .visual.color import VertexColor


from . import util
from . import caching
from . import grouping
from . import transformations


def point_plane_distance(points, plane_normal, plane_origin=None):
    """
    The minimum perpendicular distance of a point to a plane.

    Parameters
    -----------
    points : (n, 3) float
      Points in space
    plane_normal : (3,) float
      Unit normal vector
    plane_origin : (3,) float
      Plane origin in space

    Returns
    ------------
    distances : (n,) float
      Distance from point to plane
    """
    points = np.asanyarray(points, dtype=np.float64)
    if plane_origin is None:
        w = points
    else:
        w = points - plane_origin
    distances = np.dot(plane_normal, w.T) / np.linalg.norm(plane_normal)
    return distances


def major_axis(points):
    """
    Returns an approximate vector representing the major
    axis of the passed points.

    Parameters
    -------------
    points : (n, dimension) float
      Points in space

    Returns
    -------------
    axis : (dimension,) float
      Vector along approximate major axis
    """
    U, S, V = np.linalg.svd(points)
    axis = util.unitize(np.dot(S, V))
    return axis


def plane_fit(points):
    """
    Fit a plane to points using SVD.

    Parameters
    ---------
    points : (n, 3) float or (p, n, 3,) float
      3D points in space
      Second option allows to simultaneously compute
      p centroids and normals

    Returns
    ---------
    C : (3,) float or (p, 3,) float
      Point on the plane
    N : (3,) float or (p, 3,) float
      Unit normal vector of plane
    """
    # make sure input is numpy array
    points = np.asanyarray(points, dtype=np.float64)
    assert points.ndim == 2 or points.ndim == 3
    # with only one point set, np.dot is faster
    if points.ndim == 2:
        # make the plane origin the mean of the points
        C = points.mean(axis=0)
        # points offset by the plane origin
        x = points - C[None, :]
        # create a (3, 3) matrix
        M = np.dot(x.T, x)
    else:
        # make the plane origin the mean of the points
        C = points.mean(axis=1)
        # points offset by the plane origin
        x = points - C[:, None, :]
        # create a (p, 3, 3) matrix
        M = np.einsum("pnd, pnm->pdm", x, x)
    # run SVD
    N = np.linalg.svd(M)[0][..., -1]
    # return the centroid(s) and normal(s)
    return C, N


def radial_sort(points, origin, normal, start=None):
    """
    Sorts a set of points radially (by angle) around an
    axis specified by origin and normal vector.

    Parameters
    --------------
    points : (n, 3) float
      Points in space
    origin : (3,)  float
      Origin to sort around
    normal : (3,)  float
      Vector to sort around
    start : (3,) float
      Vector to specify start position in counter-clockwise
      order viewing in direction of normal, MUST not be
      parallel with normal

    Returns
    --------------
    ordered : (n, 3) float
      Same as input points but reordered
    """

    # create two axis perpendicular to each other and
    # the normal and project the points onto them
    if start is None:
        axis0 = [normal[0], normal[2], -normal[1]]
        axis1 = np.cross(normal, axis0)
    else:
        normal, start = util.unitize([normal, start])
        if np.abs(1 - np.abs(np.dot(normal, start))) < tol.zero:
            raise ValueError("start must not parallel with normal")
        axis0 = np.cross(start, normal)
        axis1 = np.cross(axis0, normal)
    vectors = points - origin
    # calculate the angles of the points on the axis
    angles = np.arctan2(np.dot(vectors, axis0), np.dot(vectors, axis1))
    # return the points sorted by angle
    return points[angles.argsort()[::-1]]


def project_to_plane(points, plane_normal, plane_origin, transform=None, return_transform=False, return_planar=True):
    """
    Project (n, 3) points onto a plane.

    Parameters
    -----------
    points : (n, 3) float
      Points in space.
    plane_normal : (3,) float
      Unit normal vector of plane
    plane_origin : (3,)
      Origin point of plane
    transform : None or (4, 4) float
      Homogeneous transform, if specified, normal+origin are overridden
    return_transform : bool
      Returns the (4, 4) matrix used or not
    return_planar : bool
      Return (n, 2) points rather than (n, 3) points
    """

    if np.all(np.abs(plane_normal) < tol.zero):
        raise NameError("Normal must be nonzero!")

    if transform is None:
        transform = plane_transform(plane_origin, plane_normal)

    transformed = transformations.transform_points(points, transform)
    transformed = transformed[:, 0 : (3 - int(return_planar))]

    if return_transform:
        polygon_to_3D = np.linalg.inv(transform)
        return transformed, polygon_to_3D
    return transformed


def remove_close(points, radius):
    """
    Given an (n, m) array of points return a subset of
    points where no point is closer than radius.

    Parameters
    ------------
    points : (n, dimension) float
      Points in space
    radius : float
      Minimum radius between result points

    Returns
    ------------
    culled : (m, dimension) float
      Points in space
    mask : (n,) bool
      Which points from the original points were returned
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    # get the index of every pair of points closer than our radius
    pairs = tree.query_pairs(radius, output_type="ndarray")

    # how often each vertex index appears in a pair
    # this is essentially a cheaply computed "vertex degree"
    # in the graph that we could construct for connected points
    count = np.bincount(pairs.ravel(), minlength=len(points))

    # for every pair we know we have to remove one of them
    # which of the two options we pick can have a large impact
    # on how much over-culling we end up doing
    column = count[pairs].argmax(axis=1)

    # take the value in each row with the highest degree
    # there is probably better numpy slicing you could do here
    highest = pairs.ravel()[column + 2 * np.arange(len(column))]

    # mask the vertices by index
    mask = np.ones(len(points), dtype=bool)
    mask[highest] = False

    if tol.strict:
        # verify we actually did what we said we'd do
        test = cKDTree(points[mask])
        assert len(test.query_pairs(radius)) == 0

    return points[mask], mask


def k_means(points, k, **kwargs):
    """
    Find k centroids that attempt to minimize the k- means problem:
    https://en.wikipedia.org/wiki/Metric_k-center

    Parameters
    ----------
    points:  (n, d) float
      Points in space
    k : int
      Number of centroids to compute
    **kwargs : dict
      Passed directly to scipy.cluster.vq.kmeans

    Returns
    ----------
    centroids : (k, d) float
      Points in some space
    labels: (n) int
      Indexes for which points belong to which centroid
    """
    from scipy.cluster.vq import kmeans
    from scipy.spatial import cKDTree

    points = np.asanyarray(points, dtype=np.float64)
    points_std = points.std(axis=0)
    points_std[points_std < tol.zero] = 1
    whitened = points / points_std
    centroids_whitened, distortion = kmeans(whitened, k, **kwargs)
    centroids = centroids_whitened * points_std

    # find which centroid each point is closest to
    tree = cKDTree(centroids)
    labels = tree.query(points, k=1)[1]

    return centroids, labels


def tsp(points, start=0):
    """
    Find an ordering of points where each is visited and
    the next point is the closest in euclidean distance,
    and if there are multiple points with equal distance
    go to an arbitrary one.

    Assumes every point is visitable from every other point,
    i.e. the travelling salesman problem on a fully connected
    graph. It is not a MINIMUM traversal; rather it is a
    "not totally goofy traversal, quickly." On random points
    this traversal is often ~20x shorter than random ordering,
    and executes on 1000 points in around 29ms on a 2014 i7.

    Parameters
    ---------------
    points : (n, dimension) float
      ND points in space
    start : int
      The index of points we should start at

    Returns
    ---------------
    traversal : (n,) int
      Ordered traversal visiting every point
    distances : (n - 1,) float
      The euclidean distance between points in traversal
    """
    # points should be float
    points = np.asanyarray(points, dtype=np.float64)

    if len(points.shape) != 2:
        raise ValueError("points must be (n, dimension)!")

    # start should be an index
    start = int(start)

    # a mask of unvisited points by index
    unvisited = np.ones(len(points), dtype=bool)
    unvisited[start] = False

    # traversal of points by index
    traversal = np.zeros(len(points), dtype=np.int64) - 1
    traversal[0] = start
    # list of distances
    distances = np.zeros(len(points) - 1, dtype=np.float64)
    # a mask of indexes in order
    index_mask = np.arange(len(points), dtype=np.int64)

    # in the loop we want to call distances.sum(axis=1)
    # a lot and it's actually kind of slow for "reasons"
    # dot products with ones is equivalent and ~2x faster
    sum_ones = np.ones(points.shape[1])

    # loop through all points
    for i in range(len(points) - 1):
        # which point are we currently on
        current = points[traversal[i]]

        # do NlogN distance query
        # use dot instead of .sum(axis=1) or np.linalg.norm
        # as it is faster, also don't square root here
        dist = np.dot((points[unvisited] - current) ** 2, sum_ones)

        # minimum distance index
        min_index = dist.argmin()
        # successor is closest unvisited point
        successor = index_mask[unvisited][min_index]
        # update the mask
        unvisited[successor] = False
        # store the index to the traversal
        traversal[i + 1] = successor
        # store the distance
        distances[i] = dist[min_index]

    # we were comparing distance^2 so take square root
    distances **= 0.5

    return traversal, distances


def plot_points(points, show=True):
    """
    Plot an (n, 3) list of points using matplotlib

    Parameters
    -------------
    points : (n, 3) float
      Points in space
    show : bool
      If False, will not show until plt.show() is called
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # NOQA

    points = np.asanyarray(points, dtype=np.float64)

    if len(points.shape) != 2:
        raise ValueError("Points must be (n, 2|3)!")

    if points.shape[1] == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(*points.T)
    elif points.shape[1] == 2:
        plt.scatter(*points.T)
    else:
        raise ValueError("points not 2D/3D: {}".format(points.shape))

    if show:
        plt.show()


class PointCloud(Geometry3D):
    """
    Hold 3D points in an object which can be visualized
    in a scene.
    """

    def __init__(self, vertices, colors=None, metadata=None, **kwargs):
        """
        Load an array of points into a PointCloud object.

        Parameters
        -------------
        vertices : (n, 3) float
          Points in space
        colors : (n, 4) uint8 or None
          RGBA colors for each point
        metadata : dict or None
          Metadata about points
        """
        self._data = caching.DataStore()
        self._cache = caching.Cache(self._data.__hash__)
        self.metadata = {}

        if metadata is not None:
            self.metadata.update(metadata)

        # load vertices
        self.vertices = vertices

        if "vertex_colors" in kwargs and colors is None:
            colors = kwargs["vertex_colors"]

        # save visual data to vertex color object
        self.visual = VertexColor(colors=colors, obj=self)

    def __setitem__(self, *args, **kwargs):
        return self.vertices.__setitem__(*args, **kwargs)

    def __getitem__(self, *args, **kwargs):
        return self.vertices.__getitem__(*args, **kwargs)

    @property
    def shape(self):
        """
        Get the shape of the pointcloud

        Returns
        ----------
        shape : (2,) int
          Shape of vertex array
        """
        return self.vertices.shape

    @property
    def is_empty(self):
        """
        Are there any vertices defined or not.

        Returns
        ----------
        empty : bool
          True if no vertices defined
        """
        return len(self.vertices) == 0

    def copy(self):
        """
        Safely get a copy of the current point cloud.

        Copied objects will have emptied caches to avoid memory
        issues and so may be slow on initial operations until
        caches are regenerated.

        Current object will *not* have its cache cleared.

        Returns
        ---------
        copied : trimesh.PointCloud
          Copy of current point cloud
        """
        copied = PointCloud(vertices=None)

        # copy vertex and face data
        copied._data.data = copy.deepcopy(self._data.data)
        # get metadata
        copied.metadata = copy.deepcopy(self.metadata)

        # make sure cache is set from here
        copied._cache.clear()

        return copied

    def hash(self):
        """
        Get a hash of the current vertices.

        Returns
        ----------
        hash : str
          Hash of self.vertices
        """
        return self._data.__hash__()

    def crc(self):
        """
        Get a CRC hash of the current vertices.

        Returns
        ----------
        crc : int
          Hash of self.vertices
        """
        return self._data.crc()

    def merge_vertices(self):
        """
        Merge vertices closer than tol.merge (default: 1e-8)
        """
        # run unique rows
        unique, inverse = grouping.unique_rows(self.vertices)

        # apply unique mask to vertices
        self.vertices = self.vertices[unique]

        # apply unique mask to colors
        if self.colors is not None and len(self.colors) == len(inverse):
            self.colors = self.colors[unique]

    def apply_transform(self, transform):
        """
        Apply a homogeneous transformation to the PointCloud
        object in- place.

        Parameters
        --------------
        transform : (4, 4) float
          Homogeneous transformation to apply to PointCloud
        """
        self.vertices = transformations.transform_points(self.vertices, matrix=transform)
        return self

    @property
    def bounds(self):
        """
        The axis aligned bounds of the PointCloud

        Returns
        ------------
        bounds : (2, 3) float
          Minimum, Maximum verteex
        """
        return np.array([self.vertices.min(axis=0), self.vertices.max(axis=0)])

    @property
    def extents(self):
        """
        The size of the axis aligned bounds

        Returns
        ------------
        extents : (3,) float
          Edge length of axis aligned bounding box
        """
        return np.ptp(self.bounds, axis=0)

    @property
    def centroid(self):
        """
        The mean vertex position

        Returns
        ------------
        centroid : (3,) float
          Mean vertex position
        """
        return self.vertices.mean(axis=0)

    @property
    def vertices(self):
        """
        Vertices of the PointCloud

        Returns
        ------------
        vertices : (n, 3) float
          Points in the PointCloud
        """
        return self._data["vertices"]

    @vertices.setter
    def vertices(self, data):
        if data is None:
            self._data["vertices"] = None
        else:
            # we want to copy data for new object
            data = np.array(data, dtype=np.float64, copy=True)
            if not util.is_shape(data, (-1, 3)):
                raise ValueError("Point clouds must be (n, 3)!")
            self._data["vertices"] = data

    @property
    def colors(self):
        """
        Stored per- point color

        Returns
        ----------
        colors : (len(self.vertices), 4) np.uint8
          Per- point RGBA color
        """
        return self.visual.vertex_colors

    @colors.setter
    def colors(self, data):
        self.visual.vertex_colors = data

    @caching.cache_decorator
    def kdtree(self):
        """
        Return a scipy.spatial.cKDTree of the vertices of the mesh.
        Not cached as this lead to observed memory issues and segfaults.

        Returns
        ---------
        tree : scipy.spatial.cKDTree
          Contains mesh.vertices
        """

        from scipy.spatial import cKDTree

        tree = cKDTree(self.vertices.view(np.ndarray))
        return tree

    @caching.cache_decorator
    def convex_hull(self):
        """
        A convex hull of every point.

        Returns
        -------------
        convex_hull : trimesh.Trimesh
          A watertight mesh of the hull of the points
        """
        from . import convex

        return convex.convex_hull(self.vertices)

    def scene(self):
        """
        A scene containing just the PointCloud

        Returns
        ----------
        scene : trimesh.Scene
          Scene object containing this PointCloud
        """
        from .scene.scene import Scene

        return Scene(self)

    def show(self, **kwargs):
        """
        Open a viewer window displaying the current PointCloud
        """
        self.scene().show(**kwargs)

    def export(self, file_obj=None, file_type=None, **kwargs):
        """
        Export the current pointcloud to a file object.
        If file_obj is a filename, file will be written there.
        Supported formats are xyz
        Parameters
        ------------
        file_obj: open writeable file object
          str, file name where to save the pointcloud
          None, if you would like this function to return the export blob
        file_type: str
          Which file type to export as.
          If file name is passed this is not required
        """
        from .exchange.export import export_mesh

        return export_mesh(self, file_obj=file_obj, file_type=file_type, **kwargs)

    def query(self, input_points, **kwargs):
        """
        Find the the closest points and associated attributes from this PointCloud.
        Parameters
        ------------
        input_points : (n, 3) float
          Input query points
        kwargs : dict
          Arguments for proximity.query_from_points
        result : proximity.NearestQueryResult
            Result of the query.
        """
        from .proximity import query_from_points

        return query_from_points(self.vertices, input_points, self.kdtree, **kwargs)

    def __add__(self, other):
        if len(other.colors) == len(self.colors) == 0:
            colors = None
        else:
            # preserve colors
            # if one point cloud has no color property use black
            other_colors = [[0, 0, 0, 255]] * len(other.vertices) if len(other.colors) == 0 else other.colors
            self_colors = [[0, 0, 0, 255]] * len(self.vertices) if len(self.colors) == 0 else self.colors
            colors = np.vstack((self_colors, other_colors))
        return PointCloud(vertices=np.vstack((self.vertices, other.vertices)), colors=colors)
