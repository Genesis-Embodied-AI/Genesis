"""
path.py
-----------

A module designed to work with vector paths such as
those stored in a DXF or SVG file.
"""

import numpy as np

import copy
import warnings
import collections

from hashlib import sha256

from ..points import plane_fit
from ..geometry import plane_transform
from ..visual import to_rgba
from ..constants import log
from ..constants import tol_path as tol

from .util import concatenate

from .. import parent
from .. import util
from .. import units
from .. import bounds
from .. import caching
from .. import grouping
from .. import exceptions
from .. import transformations as tf

from . import raster
from . import simplify
from . import creation  # NOQA
from . import segments  # NOQA
from . import traversal

from .exchange.export import export_path

# now import things which require non-minimal install of Trimesh
# create a dummy module which will raise the ImportError
# or other exception only when someone tries to use that function
try:
    from . import repair
except BaseException as E:
    repair = exceptions.ExceptionWrapper(E)
try:
    from . import polygons
except BaseException as E:
    polygons = exceptions.ExceptionWrapper(E)
try:
    from scipy.spatial import cKDTree
except BaseException as E:
    cKDTree = exceptions.ExceptionWrapper(E)
try:
    from shapely.geometry import Polygon
except BaseException as E:
    Polygon = exceptions.ExceptionWrapper(E)
try:
    import networkx as nx
except BaseException as E:
    nx = exceptions.ExceptionWrapper(E)


class Path(parent.Geometry):
    """
    A Path object consists of vertices and entities. Vertices
    are a simple (n, dimension) float array of points in space.

    Entities are a list of objects representing geometric
    primitives, such as Lines, Arcs, BSpline, etc. All entities
    reference vertices by index, so any transform applied to the
    simple vertex array is applied to the entity.
    """

    def __init__(self, entities=None, vertices=None, metadata=None, process=True, colors=None, **kwargs):
        """
        Instantiate a path object.

        Parameters
        -----------
        entities : (m,) trimesh.path.entities.Entity
          Contains geometric entities
        vertices : (n, dimension) float
          The vertices referenced by entities
        metadata : dict
          Any metadata about the path
        process :  bool
          Run simple cleanup or not
        """

        self.entities = entities
        self.vertices = vertices

        # assign each color to each entity
        self.colors = colors
        # collect metadata into new dictionary
        self.metadata = dict()
        if metadata.__class__.__name__ == "dict":
            self.metadata.update(metadata)

        # cache will dump whenever self.crc changes
        self._cache = caching.Cache(id_function=self.__hash__)

        if process:
            # literally nothing will work if vertices
            # aren't merged properly
            self.merge_vertices()

    def __repr__(self):
        """
        Print a quick summary of the number of vertices and entities.
        """
        return "<trimesh.{}(vertices.shape={}, len(entities)={})>".format(
            type(self).__name__, self.vertices.shape, len(self.entities)
        )

    def process(self):
        """
        Apply basic cleaning functions to the Path object in- place.
        """
        with self._cache:
            for func in self._process_functions():
                func()
        return self

    @property
    def colors(self):
        """
        Colors are stored per-entity.

        Returns
        ------------
        colors : (len(entities), 4) uint8
          RGBA colors for each entity
        """
        # start with default colors
        raw = [e.color for e in self.entities]
        if not any(c is not None for c in raw):
            return None

        colors = np.array([to_rgba(c) for c in raw])
        # don't allow parts of the color array to be written
        colors.flags["WRITEABLE"] = False
        return colors

    @colors.setter
    def colors(self, values):
        """
        Set the color for every entity in the Path.

        Parameters
        ------------
        values : (len(entities), 4) uint8
          Color of each entity
        """
        # if not set return
        if values is None:
            return
        # make sure colors are RGBA
        colors = to_rgba(values)
        if len(colors) != len(self.entities):
            raise ValueError("colors must be per-entity!")
        # otherwise assign each color to the entity
        for c, e in zip(colors, self.entities):
            e.color = c

    @property
    def vertices(self):
        return self._vertices

    @vertices.setter
    def vertices(self, values):
        self._vertices = caching.tracked_array(values, dtype=np.float64)

    @property
    def entities(self):
        """
        The actual entities making up the path.

        Returns
        -----------
        entities : (n,) trimesh.path.entities.Entity
          Entities such as Line, Arc, or BSpline curves
        """
        return self._entities

    @entities.setter
    def entities(self, values):
        if values is None:
            self._entities = np.array([])
        else:
            self._entities = np.asanyarray(values)

    @property
    def layers(self):
        """
        Get a list of the layer for every entity.

        Returns
        ---------
        layers : (len(entities), ) any
          Whatever is stored in each `entity.layer`
        """
        # layer is a required meta-property for entities
        return [e.layer for e in self.entities]

    def __hash__(self):
        """
        A hash of the current vertices and entities.

        Returns
        ------------
        hash : long int
          Appended hashes
        """
        # get the hash of the trackedarray vertices
        hashable = [hex(self.vertices.__hash__()).encode("utf-8")]
        # get the bytes for each entity
        hashable.extend(e._bytes() for e in self.entities)
        # hash the combined result
        return caching.hash_fast(b"".join(hashable))

    @caching.cache_decorator
    def paths(self):
        """
        Sequence of closed paths, encoded by entity index.

        Returns
        ---------
        paths : (n,) sequence of (*,) int
          Referencing self.entities
        """
        paths = traversal.closed_paths(self.entities, self.vertices)
        return paths

    @caching.cache_decorator
    def dangling(self):
        """
        List of entities that aren't included in a closed path

        Returns
        ----------
        dangling : (n,) int
          Index of self.entities
        """
        if len(self.paths) == 0:
            return np.arange(len(self.entities))
        else:
            included = np.hstack(self.paths)
        dangling = np.setdiff1d(np.arange(len(self.entities)), included)
        return dangling

    @caching.cache_decorator
    def kdtree(self):
        """
        A KDTree object holding the vertices of the path.

        Returns
        ----------
        kdtree : scipy.spatial.cKDTree
          Object holding self.vertices
        """
        kdtree = cKDTree(self.vertices.view(np.ndarray))
        return kdtree

    @property
    def scale(self):
        """
        What is a representitive number that reflects the magnitude
        of the world holding the paths, for numerical comparisons.

        Returns
        ----------
        scale : float
          Approximate size of the world holding this path
        """
        # use vertices peak-peak rather than exact extents
        scale = float((np.ptp(self.vertices, axis=0) ** 2).sum() ** 0.5)
        return scale

    @caching.cache_decorator
    def length(self):
        """
        The total discretized length of every entity.

        Returns
        --------
        length : float
          Summed length of every entity
        """
        length = float(sum(i.length(self.vertices) for i in self.entities))
        return length

    @caching.cache_decorator
    def bounds(self):
        """
        Return the axis aligned bounding box of the current path.

        Returns
        ----------
        bounds : (2, dimension) float
          AABB with (min, max) coordinates
        """
        # get the exact bounds of each entity
        # some entities (aka 3- point Arc) have bounds that can't
        # be generated from just bound box of vertices

        points = np.array([e.bounds(self.vertices) for e in self.entities], dtype=np.float64)

        # flatten bound extrema into (n, dimension) array
        points = points.reshape((-1, self.vertices.shape[1]))
        # get the max and min of all bounds
        return np.array([points.min(axis=0), points.max(axis=0)], dtype=np.float64)

    @caching.cache_decorator
    def centroid(self):
        """
        Return the centroid of axis aligned bounding box enclosing
        all entities of the path object.

        Returns
        -----------
        centroid : (d,) float
          Approximate centroid of the path
        """
        centroid = self.bounds.mean(axis=0)
        return centroid

    @property
    def extents(self):
        """
        The size of the axis aligned bounding box.

        Returns
        ---------
        extents : (dimension,) float
          Edge length of AABB
        """
        return np.ptp(self.bounds, axis=0)

    @property
    def units(self):
        """
        If there are units defined in self.metadata return them.

        Returns
        -----------
        units : str
          Current unit system
        """
        if "units" in self.metadata:
            return self.metadata["units"]
        else:
            return None

    @units.setter
    def units(self, units):
        self.metadata["units"] = units

    def convert_units(self, desired, guess=False):
        """
        Convert the units of the current drawing in place.

        Parameters
        -----------
        desired : str
          Unit system to convert to
        guess : bool
          If True will attempt to guess units
        """
        units._convert_units(self, desired=desired, guess=guess)

    def explode(self):
        """
        Turn every multi- segment entity into single segment
        entities in- place.
        """
        new_entities = []
        for entity in self.entities:
            # explode into multiple entities
            new_entities.extend(entity.explode())
        # avoid setter and assign new entities
        self._entities = np.array(new_entities)
        # explicitly clear cache
        self._cache.clear()

    def fill_gaps(self, distance=0.025):
        """
        Find vertices without degree 2 and try to connect to
        other vertices. Operations are done in-place.

        Parameters
        ----------
        distance : float
          Connect vertices up to this distance
        """
        repair.fill_gaps(self, distance=distance)

    @property
    def is_closed(self):
        """
        Are all entities connected to other entities.

        Returns
        -----------
        closed : bool
          Every entity is connected at its ends
        """
        closed = all(i == 2 for i in dict(self.vertex_graph.degree()).values())

        return closed

    @property
    def is_empty(self):
        """
        Are any entities defined for the current path.

        Returns
        ----------
        empty : bool
          True if no entities are defined
        """
        return len(self.entities) == 0

    @caching.cache_decorator
    def vertex_graph(self):
        """
        Return a networkx.Graph object for the entity connectivity

        graph : networkx.Graph
          Holds vertex indexes
        """
        graph, closed = traversal.vertex_graph(self.entities)
        return graph

    @caching.cache_decorator
    def vertex_nodes(self):
        """
        Get a list of which vertex indices are nodes,
        which are either endpoints or points where the
        entity makes a direction change.

        Returns
        --------------
        nodes : (n, 2) int
          Indexes of self.vertices which are nodes
        """
        nodes = np.vstack([e.nodes for e in self.entities])
        return nodes

    def apply_transform(self, transform):
        """
        Apply a transformation matrix to the current path in- place

        Parameters
        -----------
        transform : (d+1, d+1) float
          Homogeneous transformations for vertices
        """
        dimension = self.vertices.shape[1]
        transform = np.asanyarray(transform, dtype=np.float64)

        if transform.shape != (dimension + 1, dimension + 1):
            raise ValueError("transform is incorrect shape!")
        elif np.abs(transform - np.eye(dimension + 1)).max() < 1e-8:
            # if we've been passed an identity matrix do nothing
            return self

        # make sure cache is up to date
        self._cache.verify()
        # new cache to transfer items
        cache = {}
        # apply transform to discretized paths
        if "discrete" in self._cache.cache:
            cache["discrete"] = [tf.transform_points(d, matrix=transform) for d in self.discrete]

        # things we can just straight up copy
        # as they are topological not geometric
        for key in [
            "root",
            "paths",
            "path_valid",
            "dangling",
            "vertex_graph",
            "enclosure",
            "enclosure_shell",
            "enclosure_directed",
        ]:
            # if they're in cache save them from the purge
            if key in self._cache.cache:
                cache[key] = self._cache.cache[key]

        # transform vertices in place
        self.vertices = tf.transform_points(self.vertices, matrix=transform)
        # explicitly clear the cache
        self._cache.clear()
        self._cache.id_set()

        # populate the things we wangled
        self._cache.cache.update(cache)
        return self

    def apply_layer(self, name):
        """
        Apply a layer name to every entity in the path.

        Parameters
        ------------
        name : str
          Apply layer name to every entity
        """
        for e in self.entities:
            e.layer = name

    def rezero(self):
        """
        Translate so that every vertex is positive in the current
        mesh is positive.

        Returns
        -----------
        matrix : (dimension + 1, dimension + 1) float
          Homogeneous transformations that was applied
          to the current Path object.
        """
        # transform to the lower left corner
        matrix = tf.translation_matrix(-self.bounds[0])
        # cleanly apply trransformation matrix
        self.apply_transform(matrix)

        return matrix

    def merge_vertices(self, digits=None):
        """
        Merges vertices which are identical and replace references
        by altering `self.entities` and `self.vertices`

        Parameters
        --------------
        digits : None, or int
          How many digits to consider when merging vertices
        """
        if len(self.vertices) == 0:
            return
        if digits is None:
            digits = util.decimal_to_digits(tol.merge * self.scale, min_digits=1)

        unique, inverse = grouping.unique_rows(self.vertices, digits=digits)
        self.vertices = self.vertices[unique]

        entities_ok = np.ones(len(self.entities), dtype=bool)

        for index, entity in enumerate(self.entities):
            # what kind of entity are we dealing with
            kind = type(entity).__name__

            # entities that don't need runs merged
            # don't screw up control- point- knot relationship
            if kind in "BSpline Bezier Text":
                entity.points = inverse[entity.points]
                continue
            # if we merged duplicate vertices, the entity may
            # have multiple references to the same vertex
            points = grouping.merge_runs(inverse[entity.points])
            # if there are three points and two are identical fix it
            if kind == "Line":
                if len(points) == 3 and points[0] == points[-1]:
                    points = points[:2]
                elif len(points) < 2:
                    # lines need two or more vertices
                    entities_ok[index] = False
            elif kind == "Arc" and len(points) != 3:
                # three point arcs need three points
                entities_ok[index] = False

            # store points in entity
            entity.points = points

        # remove degenerate entities
        self.entities = self.entities[entities_ok]

    def replace_vertex_references(self, mask):
        """
        Replace the vertex index references in every entity.

        Parameters
        ------------
        mask : (len(self.vertices), ) int
          Contains new vertex indexes

        Notes
        ------------
        entity.points in self.entities
          Replaced by mask[entity.points]
        """
        for entity in self.entities:
            entity.points = mask[entity.points]

    def remove_entities(self, entity_ids):
        """
        Remove entities by index.

        Parameters
        -----------
        entity_ids : (n,) int
          Indexes of self.entities to remove
        """
        if len(entity_ids) == 0:
            return
        keep = np.ones(len(self.entities), dtype=bool)
        keep[entity_ids] = False
        self.entities = self.entities[keep]

    def remove_invalid(self):
        """
        Remove entities which declare themselves invalid

        Notes
        ----------
        self.entities: shortened
        """
        valid = np.array([i.is_valid for i in self.entities], dtype=bool)
        self.entities = self.entities[valid]

    def remove_duplicate_entities(self):
        """
        Remove entities that are duplicated

        Notes
        -------
        self.entities: length same or shorter
        """
        entity_hashes = np.array([hash(i) for i in self.entities])
        unique, inverse = grouping.unique_rows(entity_hashes)
        if len(unique) != len(self.entities):
            self.entities = self.entities[unique]

    @caching.cache_decorator
    def referenced_vertices(self):
        """
        Which vertices are referenced by an entity.

        Returns
        -----------
        referenced_vertices: (n,) int, indexes of self.vertices
        """
        # no entities no reference
        if len(self.entities) == 0:
            return np.array([], dtype=np.int64)
        referenced = np.concatenate([e.points for e in self.entities])
        referenced = np.unique(referenced.astype(np.int64))

        return referenced

    def remove_unreferenced_vertices(self):
        """
        Removes all vertices which aren't used by an entity.

        Notes
        ---------
        self.vertices : reordered and shortened
        self.entities : entity.points references updated
        """

        unique = self.referenced_vertices

        mask = np.ones(len(self.vertices), dtype=np.int64) * -1
        mask[unique] = np.arange(len(unique), dtype=np.int64)

        self.replace_vertex_references(mask=mask)
        self.vertices = self.vertices[unique]

    def discretize_path(self, path):
        """
        Given a list of entities, return a list of connected points.

        Parameters
        -----------
        path: (n,) int
          Indexes of self.entities

        Returns
        -----------
        discrete : (m, dimension)
          Linear segment path.
        """
        discrete = traversal.discretize_path(self.entities, self.vertices, path, scale=self.scale)
        return discrete

    @caching.cache_decorator
    def discrete(self):
        """
        A sequence of connected vertices in space, corresponding to
        self.paths.

        Returns
        ---------
        discrete : (len(self.paths),)
            A sequence of (m*, dimension) float
        """
        discrete = [self.discretize_path(i) for i in self.paths]
        return discrete

    def export(self, file_obj=None, file_type=None, **kwargs):
        """
        Export the path to a file object or return data.

        Parameters
        ---------------
        file_obj : None, str, or file object
          File object or string to export to
        file_type : None or str
          Type of file: dxf, dict, svg

        Returns
        ---------------
        exported : bytes or str
          Exported as specified type
        """
        return export_path(self, file_type=file_type, file_obj=file_obj, **kwargs)

    def to_dict(self):
        export_dict = self.export(file_type="dict")
        return export_dict

    def copy(self):
        """
        Get a copy of the current mesh

        Returns
        ---------
        copied : Path object
          Copy of self
        """

        metadata = {}
        # grab all the keys into a list so if something is added
        # in another thread it probably doesn't stomp on our loop
        for key in list(self.metadata.keys()):
            try:
                metadata[key] = copy.deepcopy(self.metadata[key])
            except RuntimeError:
                # multiple threads
                log.warning("key {} changed during copy".format(key))

        # copy the core data
        copied = type(self)(
            entities=copy.deepcopy(self.entities),
            vertices=copy.deepcopy(self.vertices),
            metadata=metadata,
            process=False,
        )

        cache = {}
        # try to copy the cache over to the new object
        try:
            # save dict keys before doing slow iteration
            keys = list(self._cache.cache.keys())
            # run through each key and copy into new cache
            for k in keys:
                cache[k] = copy.deepcopy(self._cache.cache[k])
        except RuntimeError:
            # if we have multiple threads this may error and is NBD
            log.debug("unable to copy cache")
        except BaseException:
            # catch and log errors we weren't expecting
            log.error("unable to copy cache", exc_info=True)
        copied._cache.cache = cache
        copied._cache.id_set()

        return copied

    def scene(self):
        """
        Get a scene object containing the current Path3D object.

        Returns
        --------
        scene: trimesh.scene.Scene object containing current path
        """
        from ..scene import Scene

        scene = Scene(self)
        return scene

    def __add__(self, other):
        """
        Concatenate two Path objects by appending vertices and
        reindexing point references.

        Parameters
        -----------
        other: Path object

        Returns
        -----------
        concat: Path object, appended from self and other
        """
        concat = concatenate([self, other])
        return concat


class Path3D(Path):
    """
    Hold multiple vector curves (lines, arcs, splines, etc) in 3D.
    """

    def _process_functions(self):
        return [self.merge_vertices, self.remove_duplicate_entities, self.remove_unreferenced_vertices]

    def to_planar(self, to_2D=None, normal=None, check=True):
        """
        Check to see if current vectors are all coplanar.

        If they are, return a Path2D and a transform which will
        transform the 2D representation back into 3 dimensions

        Parameters
        -----------
        to_2D: (4,4) float
            Homogeneous transformation matrix to apply,
            If not passed a plane will be fitted to vertices.
        normal: (3,) float, or None
           Approximate normal of direction of plane
           If to_2D is not specified sign
           will be applied to fit plane normal
        check:  bool
            If True: Raise a ValueError if
            points aren't coplanar

        Returns
        -----------
        planar : trimesh.path.Path2D
                   Current path transformed onto plane
        to_3D :  (4,4) float
                   Homeogenous transformations to move planar
                   back into 3D space
        """
        # which vertices are actually referenced
        referenced = self.referenced_vertices
        # if nothing is referenced return an empty path
        if len(referenced) == 0:
            return Path2D(), np.eye(4)

        # no explicit transform passed
        if to_2D is None:
            # fit a plane to our vertices
            C, N = plane_fit(self.vertices[referenced])
            # apply the normal sign hint
            if normal is not None:
                normal = np.asanyarray(normal, dtype=np.float64)
                if normal.shape == (3,):
                    N *= np.sign(np.dot(N, normal))
                    N = normal
                else:
                    log.debug("passed normal not used: {}".format(normal.shape))
            # create a transform from fit plane to XY
            to_2D = plane_transform(origin=C, normal=N)

        # make sure we've extracted a transform
        to_2D = np.asanyarray(to_2D, dtype=np.float64)
        if to_2D.shape != (4, 4):
            raise ValueError("unable to create transform!")

        # transform all vertices to 2D plane
        flat = tf.transform_points(self.vertices, to_2D)

        # Z values of vertices which are referenced
        heights = flat[referenced][:, 2]
        # points are not on a plane because Z varies
        if np.ptp(heights) > tol.planar:
            # since Z is inconsistent set height to zero
            height = 0.0
            if check:
                raise ValueError("points are not flat!")
        else:
            # if the points were planar store the height
            height = heights.mean()

        # the transform from 2D to 3D
        to_3D = np.linalg.inv(to_2D)

        # if the transform didn't move the path to
        # exactly Z=0 adjust it so the returned transform does
        if np.abs(height) > tol.planar:
            # adjust to_3D transform by height
            adjust = tf.translation_matrix([0, 0, height])
            # apply the height adjustment to_3D
            to_3D = np.dot(to_3D, adjust)

        # copy metadata to new object
        metadata = copy.deepcopy(self.metadata)
        # store transform we used to move it onto the plane
        metadata["to_3D"] = to_3D

        # create the Path2D with the same entities
        # and XY values of vertices projected onto the plane
        planar = Path2D(entities=copy.deepcopy(self.entities), vertices=flat[:, :2], metadata=metadata, process=False)

        return planar, to_3D

    def show(self, **kwargs):
        """
        Show the current Path3D object.
        """
        scene = self.scene()
        return scene.show(**kwargs)


class Path2D(Path):
    """
    Hold multiple vector curves (lines, arcs, splines, etc) in 3D.
    """

    def show(self, annotations=True):
        """
        Plot the current Path2D object using matplotlib.
        """
        if self.is_closed:
            self.plot_discrete(show=True, annotations=annotations)
        else:
            self.plot_entities(show=True, annotations=annotations)

    def _process_functions(self):
        """
        Return a list of functions to clean up a Path2D
        """
        return [self.merge_vertices, self.remove_duplicate_entities, self.remove_unreferenced_vertices]

    def apply_obb(self):
        """
        Transform the current path so that its OBB is axis aligned
        and OBB center is at the origin.

        Returns
        -----------
        obb : (3, 3) float
          Homogeneous transformation matrix
        """
        matrix = self.obb
        self.apply_transform(matrix)
        return matrix

    def apply_scale(self, scale):
        """
        Apply a 2D scale to the current Path2D.

        Parameters
        -------------
        scale : float or (2,) float
          Scale to apply in-place.
        """
        matrix = np.eye(3)
        matrix[:2, :2] *= scale
        return self.apply_transform(matrix)

    @caching.cache_decorator
    def obb(self):
        """
        Get a transform that centers and aligns the OBB of the
        referenced vertices with the XY axis.

        Returns
        -----------
        obb : (3, 3) float
          Homogeneous transformation matrix
        """
        matrix = bounds.oriented_bounds_2D(self.vertices[self.referenced_vertices])[0]
        return matrix

    def rasterize(self, pitch=None, origin=None, resolution=None, fill=True, width=None, **kwargs):
        """
        Rasterize a Path2D object into a boolean image ("mode 1").

        Parameters
        ------------
        pitch : float or (2,) float
          Length(s) in model space of pixel edges
        origin : (2,) float
          Origin position in model space
        resolution : (2,) int
          Resolution in pixel space
        fill : bool
          If True will return closed regions as filled
        width : int
          If not None will draw outline this wide (pixels)

        Returns
        ------------
        raster : PIL.Image object, mode 1
          Rasterized version of closed regions.
        """
        image = raster.rasterize(self, pitch=pitch, origin=origin, resolution=resolution, fill=fill, width=width)
        return image

    def sample(self, count, **kwargs):
        """
        Use rejection sampling to generate random points inside a
        polygon.

        Parameters
        -----------
        count : int
          Number of points to return
          If there are multiple bodies, there will
          be up to count * bodies points returned
        factor : float
          How many points to test per loop
          IE, count * factor
        max_iter : int,
          Maximum number of intersection loops
          to run, total points sampled is
          count * factor * max_iter

        Returns
        -----------
        hit : (n, 2) float
          Random points inside polygon
        """

        poly = self.polygons_full
        if len(poly) == 0:
            samples = np.array([])
        elif len(poly) == 1:
            samples = polygons.sample(poly[0], count=count, **kwargs)
        else:
            samples = util.vstack_empty([polygons.sample(i, count=count, **kwargs) for i in poly])

        return samples

    @property
    def body_count(self):
        """
        Returns a count of the number of unconnected polygons that
        may contain other curves but aren't contained themselves.

        Returns
        ---------
        body_count : int
          Number of unconnected independent polygons.
        """
        return len(self.root)

    def to_3D(self, transform=None):
        """
        Convert 2D path to 3D path on the XY plane.

        Parameters
        -------------
        transform : (4, 4) float
          If passed, will transform vertices.
          If not passed and 'to_3D' is in self.metadata
          that transform will be used.

        Returns
        -----------
        path_3D : Path3D
          3D version of current path
        """
        # if there is a stored 'to_3D' transform in metadata use it
        if transform is None and "to_3D" in self.metadata:
            transform = self.metadata["to_3D"]

        # copy vertices and stack with zeros from (n, 2) to (n, 3)
        vertices = np.column_stack((copy.deepcopy(self.vertices), np.zeros(len(self.vertices))))
        if transform is not None:
            vertices = tf.transform_points(vertices, transform)
        # make sure everything is deep copied
        path_3D = Path3D(
            entities=copy.deepcopy(self.entities), vertices=vertices, metadata=copy.deepcopy(self.metadata)
        )
        return path_3D

    @caching.cache_decorator
    def polygons_closed(self):
        """
        Cycles in the vertex graph, as shapely.geometry.Polygons.
        These are polygon objects for every closed circuit, with no notion
        of whether a polygon is a hole or an area. Every polygon in this
        list will have an exterior, but NO interiors.

        Returns
        ---------
        polygons_closed: (n,) list of shapely.geometry.Polygon objects
        """
        # will attempt to recover invalid garbage geometry
        # and will be None if geometry is unrecoverable
        polys = polygons.paths_to_polygons(self.discrete)
        return polys

    @caching.cache_decorator
    def polygons_full(self):
        """
        A list of shapely.geometry.Polygon objects with interiors created
        by checking which closed polygons enclose which other polygons.

        Returns
        ---------
        full : (len(self.root),) shapely.geometry.Polygon
            Polygons containing interiors
        """
        # pre- allocate the list to avoid indexing problems
        full = [None] * len(self.root)
        # store the graph to avoid cache thrashing
        enclosure = self.enclosure_directed
        # store closed polygons to avoid cache hits
        closed = self.polygons_closed

        # loop through root curves
        for i, root in enumerate(self.root):
            # a list of multiple Polygon objects that
            # are fully contained by the root curve
            children = [closed[child] for child in enclosure[root].keys()]
            # all polygons_closed are CCW, so for interiors reverse them
            holes = [np.array(p.exterior.coords)[::-1] for p in children]
            # a single Polygon object
            shell = closed[root].exterior
            # create a polygon with interiors
            full[i] = polygons.repair_invalid(Polygon(shell=shell, holes=holes))
        # so we can use advanced indexing
        full = np.array(full)

        return full

    @caching.cache_decorator
    def area(self):
        """
        Return the area of the polygons interior.

        Returns
        ---------
        area : float
          Total area of polygons minus interiors
        """
        area = float(sum(i.area for i in self.polygons_full))
        return area

    def extrude(self, height, **kwargs):
        """
        Extrude the current 2D path into a 3D mesh.

        Parameters
        ----------
        height: float, how far to extrude the profile
        kwargs: passed directly to meshpy.triangle.build:
                triangle.build(mesh_info,
                               verbose=False,
                               refinement_func=None,
                               attributes=False,
                               volume_constraints=True,
                               max_volume=None,
                               allow_boundary_steiner=True,
                               allow_volume_steiner=True,
                               quality_meshing=True,
                               generate_edges=None,
                               generate_faces=False,
                               min_angle=None)
        Returns
        --------
        mesh: trimesh object representing extruded polygon
        """
        from ..primitives import Extrusion

        result = [Extrusion(polygon=i, height=height, **kwargs) for i in self.polygons_full]
        if len(result) == 1:
            return result[0]
        return result

    def triangulate(self, **kwargs):
        """
        Create a region- aware triangulation of the 2D path.

        Parameters
        -------------
        **kwargs : dict
          Passed to `trimesh.creation.triangulate_polygon`

        Returns
        -------------
        vertices : (n, 2) float
          2D vertices of triangulation
        faces : (n, 3) int
          Indexes of vertices for triangles
        """
        from ..creation import triangulate_polygon

        # append vertices and faces into sequence
        v_seq = []
        f_seq = []

        # loop through polygons with interiors
        for polygon in self.polygons_full:
            v, f = triangulate_polygon(polygon, **kwargs)
            v_seq.append(v)
            f_seq.append(f)

        return util.append_faces(v_seq, f_seq)

    def medial_axis(self, resolution=None, clip=None):
        """
        Find the approximate medial axis based
        on a voronoi diagram of evenly spaced points on the
        boundary of the polygon.

        Parameters
        ----------
        resolution : None or float
          Distance between each sample on the polygon boundary
        clip : None, or (2,) float
          Min, max number of samples

        Returns
        ----------
        medial : Path2D object
          Contains only medial axis of Path
        """
        if resolution is None:
            resolution = self.scale / 1000.0

        # convert the edges to Path2D kwargs
        from .exchange.misc import edges_to_path

        # edges and vertices
        edge_vert = [polygons.medial_axis(i, resolution, clip) for i in self.polygons_full]
        # create a Path2D object for each region
        medials = [Path2D(**edges_to_path(edges=e, vertices=v)) for e, v in edge_vert]

        # get a single Path2D of medial axis
        medial = concatenate(medials)

        return medial

    def connected_paths(self, path_id, include_self=False):
        """
        Given an index of self.paths find other paths which
        overlap with that path.

        Parameters
        -----------
        path_id : int
          Index of self.paths
        include_self : bool
          Should the result include path_id or not

        Returns
        -----------
        path_ids :  (n, ) int
          Indexes of self.paths that overlap input path_id
        """
        if len(self.root) == 1:
            path_ids = np.arange(len(self.polygons_closed))
        else:
            path_ids = list(nx.node_connected_component(self.enclosure, path_id))
        if include_self:
            return np.array(path_ids)
        return np.setdiff1d(path_ids, [path_id])

    def simplify(self, **kwargs):
        """
        Return a version of the current path with colinear segments
        merged, and circles entities replacing segmented circular paths.

        Returns
        ---------
        simplified : Path2D object
        """
        return simplify.simplify_basic(self, **kwargs)

    def simplify_spline(self, smooth=0.0002, verbose=False):
        """
        Convert paths into b-splines.

        Parameters
        -----------
        smooth : float
          How much the spline should smooth the curve
        verbose : bool
          Print detailed log messages

        Returns
        ------------
        simplified : Path2D
          Discrete curves replaced with splines
        """
        return simplify.simplify_spline(self, smooth=smooth, verbose=verbose)

    def split(self, **kwargs):
        """
        If the current Path2D consists of n 'root' curves,
        split them into a list of n Path2D objects

        Returns
        ----------
        split:  (n,) list of Path2D objects
          Each connected region and interiors
        """
        return traversal.split(self)

    def plot_discrete(self, show=False, annotations=True):
        """
        Plot the closed curves of the path.
        """
        import matplotlib.pyplot as plt

        axis = plt.gca()
        axis.set_aspect("equal", "datalim")

        for i, points in enumerate(self.discrete):
            color = ["g", "k"][i in self.root]
            axis.plot(*points.T, color=color)

        if annotations:
            for e in self.entities:
                if not hasattr(e, "plot"):
                    continue
                e.plot(self.vertices)

        if show:
            plt.show()
        return axis

    def plot_entities(self, show=False, annotations=True, color=None):
        """
        Plot the entities of the path with no notion of topology.

        Parameters
        ------------
        show : bool
          Open a window immediately or not
        annotations : bool
          Call an entities custom plot function.
        color : str
          Override entity colors and make them all this color.
        """
        import matplotlib.pyplot as plt

        # keep plot axis scaled the same
        axis = plt.gca()
        axis.set_aspect("equal", "datalim")
        # hardcode a format for each entity type
        eformat = {
            "Line0": {"color": "g", "linewidth": 1},
            "Line1": {"color": "y", "linewidth": 1},
            "Arc0": {"color": "r", "linewidth": 1},
            "Arc1": {"color": "b", "linewidth": 1},
            "Bezier0": {"color": "k", "linewidth": 1},
            "Bezier1": {"color": "k", "linewidth": 1},
            "BSpline0": {"color": "m", "linewidth": 1},
            "BSpline1": {"color": "m", "linewidth": 1},
        }
        for entity in self.entities:
            # if the entity has it's own plot method use it
            if annotations and hasattr(entity, "plot"):
                entity.plot(self.vertices)
                continue
            # otherwise plot the discrete curve
            discrete = entity.discrete(self.vertices)
            # a unique key for entities
            e_key = entity.__class__.__name__ + str(int(entity.closed))

            fmt = eformat[e_key].copy()
            if color is not None:
                # passed color will override other options
                fmt["color"] = color
            elif hasattr(entity, "color"):
                # if entity has specified color use it
                fmt["color"] = entity.color
            axis.plot(*discrete.T, **fmt)
        if show:
            plt.show()

    @property
    def identifier(self):
        """
        A unique identifier for the path.

        Returns
        ---------
        identifier : (5,) float
          Unique identifier
        """
        hasher = polygons.polygon_hash
        target = self.polygons_full
        if len(target) == 1:
            return hasher(self.polygons_full[0])
        elif len(target) == 0:
            return np.zeros(5)

        return np.sum([hasher(p) for p in target], axis=1)

    @caching.cache_decorator
    def identifier_hash(self):
        """
        Return a hash of the identifier.

        Returns
        ----------
        hashed : (64,) str
          SHA256 hash of the identifier vector.
        """
        as_int = (self.identifier * 1e4).astype(np.int64)
        return sha256(as_int.tobytes(order="C")).hexdigest()

    @property
    def identifier_md5(self):
        warnings.warn(
            "`geom.identifier_md5` is deprecated and will "
            + "be removed in October 2023: replace "
            + "with `geom.identifier_hash`",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return self.identifier_hash

    @property
    def path_valid(self):
        """
        Returns
        ----------
        path_valid : (n,) bool
          Indexes of self.paths self.polygons_closed
          which are valid polygons.
        """
        valid = np.array([i is not None for i in self.polygons_closed], dtype=bool)
        return valid

    @caching.cache_decorator
    def root(self):
        """
        Which indexes of self.paths/self.polygons_closed
        are root curves, also known as 'shell' or 'exterior.

        Returns
        ---------
        root : (n,) int
          List of indexes
        """
        populate = self.enclosure_directed  # NOQA
        return self._cache["root"]

    @caching.cache_decorator
    def enclosure(self):
        """
        Undirected graph object of polygon enclosure.

        Returns
        -----------
        enclosure : networkx.Graph
          Enclosure graph of self.polygons by index.
        """
        with self._cache:
            undirected = self.enclosure_directed.to_undirected()
        return undirected

    @caching.cache_decorator
    def enclosure_directed(self):
        """
        Directed graph of polygon enclosure.

        Returns
        ----------
        enclosure_directed : networkx.DiGraph
          Directed graph: child nodes are fully
          contained by their parent node.
        """
        root, enclosure = polygons.enclosure_tree(self.polygons_closed)
        self._cache["root"] = root
        return enclosure

    @caching.cache_decorator
    def enclosure_shell(self):
        """
        A dictionary of path indexes which are 'shell' paths, and values
        of 'hole' paths.

        Returns
        ----------
        corresponding : dict
          {index of self.paths of shell : [indexes of holes]}
        """
        pairs = [(r, self.connected_paths(r, include_self=False)) for r in self.root]
        # OrderedDict to maintain corresponding order
        corresponding = collections.OrderedDict(pairs)
        return corresponding
