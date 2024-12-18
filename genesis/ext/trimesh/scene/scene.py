import collections

import numpy as np

from .. import caching, convex, grouping, inertia, transformations, units, util
from ..exchange import export
from ..parent import Geometry3D
from ..util import unique_name
from . import cameras, lighting
from .transforms import SceneGraph


class Scene(Geometry3D):
    """
    A simple scene graph which can be rendered directly via
    pyglet/openGL or through other endpoints such as a
    raytracer. Meshes are added by name, which can then be
    moved by updating transform in the transform tree.
    """

    def __init__(
        self,
        geometry=None,
        base_frame="world",
        metadata=None,
        graph=None,
        camera=None,
        lights=None,
        camera_transform=None,
    ):
        """
        Create a new Scene object.

        Parameters
        -------------
        geometry : Trimesh, Path2D, Path3D PointCloud or list
          Geometry to initially add to the scene
        base_frame : str or hashable
          Name of base frame
        metadata : dict
          Any metadata about the scene
        graph : TransformForest or None
          A passed transform graph to use
        camera : Camera or None
          A passed camera to use
        lights : [trimesh.scene.lighting.Light] or None
          A passed lights to use
        camera_transform : (4, 4) float or None
          Camera transform in the base frame
        """
        # mesh name : Trimesh object
        self.geometry = collections.OrderedDict()

        # create a new graph
        self.graph = SceneGraph(base_frame=base_frame)

        # create our cache
        self._cache = caching.Cache(id_function=self.__hash__)

        # add passed geometry to scene
        self.add_geometry(geometry)

        # hold metadata about the scene
        self.metadata = {}
        if isinstance(metadata, dict):
            self.metadata.update(metadata)

        if graph is not None:
            # if we've been passed a graph override the default
            self.graph = graph

        self.camera = camera
        self.lights = lights

        if camera is not None and camera_transform is not None:
            self.camera_transform = camera_transform

    def apply_transform(self, transform):
        """
        Apply a transform to all children of the base frame
        without modifying any geometry.

        Parameters
        --------------
        transform : (4, 4)
          Homogeneous transformation matrix.
        """
        base = self.graph.base_frame
        for child in self.graph.transforms.children[base]:
            combined = np.dot(transform, self.graph[child][0])
            self.graph.update(frame_from=base, frame_to=child, matrix=combined)
        return self

    def add_geometry(
        self, geometry, node_name=None, geom_name=None, parent_node_name=None, transform=None, metadata=None
    ):
        """
        Add a geometry to the scene.

        If the mesh has multiple transforms defined in its
        metadata, they will all be copied into the
        TransformForest of the current scene automatically.

        Parameters
        ----------
        geometry : Trimesh, Path2D, Path3D PointCloud or list
          Geometry to initially add to the scene
        node_name : None or str
          Name of the added node.
        geom_name : None or str
          Name of the added geometry.
        parent_node_name : None or str
          Name of the parent node in the graph.
        transform : None or (4, 4) float
          Transform that applies to the added node.
        metadata : None or dict
          Optional metadata for the node.

        Returns
        ----------
        node_name : str
          Name of single node in self.graph (passed in) or None if
          node was not added (eg. geometry was null or a Scene).
        """

        if geometry is None:
            return
        # PointCloud objects will look like a sequence
        elif util.is_sequence(geometry):
            # if passed a sequence add all elements
            return [
                self.add_geometry(
                    geometry=value,
                    node_name=node_name,
                    geom_name=geom_name,
                    parent_node_name=parent_node_name,
                    transform=transform,
                    metadata=metadata,
                )
                for value in geometry
            ]
        elif isinstance(geometry, dict):
            # if someone passed us a dict of geometry
            return {k: self.add_geometry(geometry=v, geom_name=k, metadata=metadata) for k, v in geometry.items()}

        elif isinstance(geometry, Scene):
            # concatenate current scene with passed scene
            concat = self + geometry
            # replace geometry in-place
            self.geometry.clear()
            self.geometry.update(concat.geometry)
            # replace graph data with concatenated graph
            self.graph.transforms = concat.graph.transforms
            return

        if not hasattr(geometry, "vertices"):
            util.log.debug("unknown type ({}) added to scene!".format(type(geometry).__name__))
            return

        # get or create a name to reference the geometry by
        if geom_name is not None:
            # if name is passed use it
            name = geom_name
        elif "name" in geometry.metadata:
            # if name is in metadata use it
            name = geometry.metadata["name"]
        elif "file_name" in geometry.metadata:
            name = geometry.metadata["file_name"]
        else:
            # try to create a simple name
            name = "geometry_" + str(len(self.geometry))

        # if its already taken use our unique name logic
        name = unique_name(start=name, contains=self.geometry.keys())
        # save the geometry reference
        self.geometry[name] = geometry

        # create a unique node name if not passed
        if node_name is None:
            # if the name of the geometry is also a transform node
            # which graph nodes already exist
            existing = self.graph.transforms.node_data.keys()
            # find a name that isn't contained already starting
            # at the name we have
            node_name = unique_name(name, existing)
            assert node_name not in existing

        if transform is None:
            # create an identity transform from parent_node
            transform = np.eye(4)

        self.graph.update(
            frame_to=node_name,
            frame_from=parent_node_name,
            matrix=transform,
            geometry=name,
            geometry_flags={"visible": True},
            metadata=metadata,
        )

        return node_name

    def delete_geometry(self, names):
        """
        Delete one more multiple geometries from the scene and also
        remove any node in the transform graph which references it.

        Parameters
        --------------
        name : hashable
          Name that references self.geometry
        """
        # make sure we have a set we can check
        if util.is_string(names):
            names = [names]
        names = set(names)

        # remove the geometry reference from relevant nodes
        self.graph.remove_geometries(names)
        # remove the geometries from our geometry store
        [self.geometry.pop(name, None) for name in names]

    def strip_visuals(self):
        """
        Strip visuals from every Trimesh geometry
        and set them to an empty `ColorVisuals`.
        """
        from ..visual.color import ColorVisuals

        for geometry in self.geometry.values():
            if util.is_instance_named(geometry, "Trimesh"):
                geometry.visual = ColorVisuals(mesh=geometry)

    def __hash__(self):
        """
        Return information about scene which is hashable.

        Returns
        ---------
        hashable : str
          Data which can be hashed.
        """
        # avoid accessing attribute in tight loop
        geometry = self.geometry
        # start with the last modified time of the scene graph
        hashable = [hex(self.graph.transforms.__hash__())]
        # take the re-hex string of the hash
        hashable.extend(hex(geometry[k].__hash__()) for k in geometry.keys())
        return caching.hash_fast("".join(hashable).encode("utf-8"))

    @property
    def is_empty(self):
        """
        Does the scene have anything in it.

        Returns
        ----------
        is_empty: bool, True if nothing is in the scene
        """

        is_empty = len(self.geometry) == 0
        return is_empty

    @property
    def is_valid(self):
        """
        Is every geometry connected to the root node.

        Returns
        -----------
        is_valid : bool
          Does every geometry have a transform
        """
        if len(self.geometry) == 0:
            return True

        try:
            referenced = {self.graph[i][1] for i in self.graph.nodes_geometry}
        except BaseException:
            # if connectivity to world frame is broken return false
            return False

        # every geometry is referenced
        ok = referenced == set(self.geometry.keys())

        return ok

    @caching.cache_decorator
    def bounds_corners(self):
        """
        Get the post-transform AABB for each node
        which has geometry defined.

        Returns
        -----------
        corners : dict
          Bounds for each node with vertices:
           {node_name : (2, 3) float}
        """
        # collect AABB for each geometry
        corners = {}
        # collect vertices for every mesh
        vertices = {k: m.vertices for k, m in self.geometry.items() if hasattr(m, "vertices") and len(m.vertices) > 0}
        # handle 2D geometries
        vertices.update({k: np.column_stack((v, np.zeros(len(v)))) for k, v in vertices.items() if v.shape[1] == 2})

        # loop through every node with geometry
        for node_name in self.graph.nodes_geometry:
            # access the transform and geometry name from node
            transform, geometry_name = self.graph[node_name]
            # will be None if no vertices for this node
            points = vertices.get(geometry_name)
            # skip empty geometries
            if points is None:
                continue
            # apply just the rotation to skip N multiplies
            dot = np.dot(transform[:3, :3], points.T)
            # append the AABB with translation applied after
            corners[node_name] = np.array([dot.min(axis=1) + transform[:3, 3], dot.max(axis=1) + transform[:3, 3]])
        return corners

    @caching.cache_decorator
    def bounds(self):
        """
        Return the overall bounding box of the scene.

        Returns
        --------
        bounds : (2, 3) float or None
          Position of [min, max] bounding box
          Returns None if no valid bounds exist
        """
        bounds_corners = self.bounds_corners
        if len(bounds_corners) == 0:
            return None
        # combine each geometry node AABB into a larger list
        corners = np.vstack(list(self.bounds_corners.values()))
        return np.array([corners.min(axis=0), corners.max(axis=0)], dtype=np.float64)

    @caching.cache_decorator
    def extents(self):
        """
        Return the axis aligned box size of the current scene.

        Returns
        ----------
        extents : (3,) float
          Bounding box sides length
        """
        return np.diff(self.bounds, axis=0).reshape(-1)

    @caching.cache_decorator
    def scale(self):
        """
        The approximate scale of the mesh

        Returns
        -----------
        scale : float
          The mean of the bounding box edge lengths
        """
        scale = (self.extents**2).sum() ** 0.5
        return scale

    @caching.cache_decorator
    def centroid(self):
        """
        Return the center of the bounding box for the scene.

        Returns
        --------
        centroid : (3) float
          Point for center of bounding box
        """
        centroid = np.mean(self.bounds, axis=0)
        return centroid

    @caching.cache_decorator
    def center_mass(self):
        """
        Find the center of mass for every instance in the scene.

        Returns
        ------------
        center_mass : (3,) float
          The center of mass of the scene
        """
        # get the center of mass and volume for each geometry
        center_mass = {k: m.center_mass for k, m in self.geometry.items() if hasattr(m, "center_mass")}
        mass = {k: m.mass for k, m in self.geometry.items() if hasattr(m, "mass")}

        # get the geometry name and transform for each instance
        graph = self.graph
        instance = [graph[n] for n in graph.nodes_geometry]

        # get the transformed center of mass for each instance
        transformed = np.array(
            [np.dot(mat, np.append(center_mass[g], 1))[:3] for mat, g in instance if g in center_mass], dtype=np.float64
        )
        # weight the center of mass locations by volume
        weights = np.array([mass[g] for _, g in instance], dtype=np.float64)
        weights /= weights.sum()
        return (transformed * weights.reshape((-1, 1))).sum(axis=0)

    @caching.cache_decorator
    def moment_inertia(self):
        """
        Return the moment of inertia of the current scene with
        respect to the center of mass of the current scene.

        Returns
        ------------
        inertia : (3, 3) float
          Inertia with respect to cartesian axis at `scene.center_mass`
        """
        return inertia.scene_inertia(scene=self, transform=transformations.translation_matrix(self.center_mass))

    def moment_inertia_frame(self, transform):
        """
        Return the moment of inertia of the current scene relative
        to a transform from the base frame.

        Parameters
        transform : (4, 4) float
          Homogeneous transformation matrix.

        Returns
        -------------
        inertia : (3, 3) float
          Inertia tensor at requested frame.
        """
        return inertia.scene_inertia(scene=self, transform=transform)

    @caching.cache_decorator
    def area(self):
        """
        What is the summed area of every geometry which
        has area.

        Returns
        ------------
        area : float
          Summed area of every instanced geometry
        """
        # get the area of every geometry that has an area property
        areas = {n: g.area for n, g in self.geometry.items() if hasattr(g, "area")}
        # sum the area including instancing
        return sum((areas.get(self.graph[n][1], 0.0) for n in self.graph.nodes_geometry), 0.0)

    @caching.cache_decorator
    def volume(self):
        """
        What is the summed volume of every geometry which
        has volume

        Returns
        ------------
        volume : float
          Summed area of every instanced geometry
        """
        # get the area of every geometry that has a volume attribute
        volume = {n: g.volume for n, g in self.geometry.items() if hasattr(g, "area")}
        # sum the area including instancing
        return sum((volume.get(self.graph[n][1], 0.0) for n in self.graph.nodes_geometry), 0.0)

    @caching.cache_decorator
    def triangles(self):
        """
        Return a correctly transformed polygon soup of the
        current scene.

        Returns
        ----------
        triangles : (n, 3, 3) float
          Triangles in space
        """
        triangles = collections.deque()
        triangles_node = collections.deque()

        for node_name in self.graph.nodes_geometry:
            # which geometry does this node refer to
            transform, geometry_name = self.graph[node_name]

            # get the actual potential mesh instance
            geometry = self.geometry[geometry_name]
            if not hasattr(geometry, "triangles"):
                continue
            # append the (n, 3, 3) triangles to a sequence
            triangles.append(
                transformations.transform_points(geometry.triangles.copy().reshape((-1, 3)), matrix=transform)
            )
            # save the node names for each triangle
            triangles_node.append(np.tile(node_name, len(geometry.triangles)))
        # save the resulting nodes to the cache
        self._cache["triangles_node"] = np.hstack(triangles_node)
        triangles = np.vstack(triangles).reshape((-1, 3, 3))
        return triangles

    @caching.cache_decorator
    def triangles_node(self):
        """
        Which node of self.graph does each triangle come from.

        Returns
        ---------
        triangles_index : (len(self.triangles),)
          Node name for each triangle
        """
        populate = self.triangles  # NOQA
        return self._cache["triangles_node"]

    @caching.cache_decorator
    def geometry_identifiers(self):
        """
        Look up geometries by identifier hash

        Returns
        ---------
        identifiers : dict
          {Identifier hash: key in self.geometry}
        """
        identifiers = {mesh.identifier_hash: name for name, mesh in self.geometry.items()}
        return identifiers

    @caching.cache_decorator
    def duplicate_nodes(self):
        """
        Return a sequence of node keys of identical meshes.

        Will include meshes with different geometry but identical
        spatial hashes as well as meshes repeated by self.nodes.

        Returns
        -----------
        duplicates : (m) sequence
          Keys of self.graph that represent identical geometry
        """
        # if there is no geometry we can have no duplicate nodes
        if len(self.geometry) == 0:
            return []

        # geometry name : hash of mesh
        hashes = {k: int(m.identifier_hash, 16) for k, m in self.geometry.items() if hasattr(m, "identifier_hash")}

        # bring into local scope for loop
        graph = self.graph
        # get a hash for each node name
        # scene.graph node name : hashed geometry
        node_hash = {node: hashes.get(graph[node][1]) for node in graph.nodes_geometry}

        # collect node names for each hash key
        duplicates = collections.defaultdict(list)
        # use a slightly off-label list comprehension
        # for debatable function call overhead avoidance
        [duplicates[hashed].append(node) for node, hashed in node_hash.items() if hashed is not None]

        # we only care about the values keys are garbage
        return list(duplicates.values())

    def deduplicated(self):
        """
        Return a new scene where each unique geometry is only
        included once and transforms are discarded.

        Returns
        -------------
        dedupe : Scene
          One copy of each unique geometry from scene
        """
        # collect geometry
        geometry = {}
        # loop through groups of identical nodes
        for group in self.duplicate_nodes:
            # get the name of the geometry
            name = self.graph[group[0]][1]
            # collect our unique collection of geometry
            geometry[name] = self.geometry[name]

        return Scene(geometry)

    def set_camera(self, angles=None, distance=None, center=None, resolution=None, fov=None):
        """
        Create a camera object for self.camera, and add
        a transform to self.graph for it.

        If arguments are not passed sane defaults will be figured
        out which show the mesh roughly centered.

        Parameters
        -----------
        angles : (3,) float
          Initial euler angles in radians
        distance : float
          Distance from centroid
        center : (3,) float
          Point camera should be center on
        camera : Camera object
          Object that stores camera parameters
        """

        if fov is None:
            fov = np.array([60, 45])

        # if no geometry nothing to set camera to
        if len(self.geometry) == 0:
            self._camera = cameras.Camera(fov=fov)
            self.graph[self._camera.name] = np.eye(4)
            return self._camera
        # set with no rotation by default
        if angles is None:
            angles = np.zeros(3)

        rotation = transformations.euler_matrix(*angles)
        transform = cameras.look_at(self.bounds, fov=fov, rotation=rotation, distance=distance, center=center)

        if hasattr(self, "_camera") and self._camera is not None:
            self._camera.fov = fov
            if resolution is not None:
                self._camera.resolution = resolution
        else:
            # create a new camera object
            self._camera = cameras.Camera(fov=fov, resolution=resolution)

        self.graph[self._camera.name] = transform

        return self._camera

    @property
    def camera_transform(self):
        """
        Get camera transform in the base frame.

        Returns
        -------
        camera_transform : (4, 4) float
          Camera transform in the base frame
        """
        return self.graph[self.camera.name][0]

    @camera_transform.setter
    def camera_transform(self, matrix):
        """
        Set the camera transform in the base frame

        Parameters
        ----------
        camera_transform : (4, 4) float
          Camera transform in the base frame
        """
        self.graph[self.camera.name] = matrix

    def camera_rays(self):
        """
        Calculate the trimesh.scene.Camera origin and ray
        direction vectors. Returns one ray per pixel as set
        in camera.resolution

        Returns
        --------------
        origin: (n, 3) float
          Ray origins in space
        vectors: (n, 3) float
          Ray direction unit vectors in world coordinates
        pixels : (n, 2) int
          Which pixel does each ray correspond to in an image
        """
        # get the unit vectors of the camera
        vectors, pixels = self.camera.to_rays()
        # find our scene's transform for the camera
        transform = self.camera_transform
        # apply the rotation to the unit ray direction vectors
        vectors = transformations.transform_points(vectors, transform, translate=False)
        # camera origin is single point so extract from
        origins = np.ones_like(vectors) * transformations.translation_from_matrix(transform)
        return origins, vectors, pixels

    @property
    def camera(self):
        """
        Get the single camera for the scene. If not manually
        set one will abe automatically generated.

        Returns
        ----------
        camera : trimesh.scene.Camera
          Camera object defined for the scene
        """
        # no camera set for the scene yet
        if not self.has_camera:
            # will create a camera with everything in view
            return self.set_camera()
        assert self._camera is not None

        return self._camera

    @camera.setter
    def camera(self, camera):
        """
        Set a camera object for the Scene.

        Parameters
        -----------
        camera : trimesh.scene.Camera
          Camera object for the scene
        """
        if camera is None:
            return
        self._camera = camera

    @property
    def has_camera(self):
        return hasattr(self, "_camera") and self._camera is not None

    @property
    def lights(self):
        """
        Get a list of the lights in the scene. If nothing is
        set it will generate some automatically.

        Returns
        -------------
        lights : [trimesh.scene.lighting.Light]
          Lights in the scene.
        """
        if not hasattr(self, "_lights") or self._lights is None:
            # do some automatic lighting
            lights, transforms = lighting.autolight(self)
            # assign the transforms to the scene graph
            for L, T in zip(lights, transforms):
                self.graph[L.name] = T
            # set the lights
            self._lights = lights
        return self._lights

    @lights.setter
    def lights(self, lights):
        """
        Assign a list of light objects to the scene

        Parameters
        --------------
        lights : [trimesh.scene.lighting.Light]
          Lights in the scene.
        """
        self._lights = lights

    def rezero(self):
        """
        Move the current scene so that the AABB of the whole
        scene is centered at the origin.

        Does this by changing the base frame to a new, offset
        base frame.
        """
        if self.is_empty or np.allclose(self.centroid, 0.0):
            # early exit since what we want already exists
            return

        # the transformation to move the overall scene to AABB centroid
        matrix = np.eye(4)
        matrix[:3, 3] = -self.centroid

        # we are going to change the base frame
        new_base = str(self.graph.base_frame) + "_I"
        self.graph.update(frame_from=new_base, frame_to=self.graph.base_frame, matrix=matrix)
        self.graph.base_frame = new_base

    def dump(self, concatenate=False):
        """
        Append all meshes in scene freezing transforms.

        Parameters
        ------------
        concatenate : bool
          If True, concatenate results into single mesh

        Returns
        ----------
        dumped : (n,) Trimesh, Path2D, Path3D, PointCloud
          Depending on what the scene contains. If `concatenate`
          then some geometry may be dropped if it doesn't match.
        """

        result = []
        for node_name in self.graph.nodes_geometry:
            transform, geometry_name = self.graph[node_name]
            # get a copy of the geometry
            current = self.geometry[geometry_name].copy()

            # if the geometry is 2D see if we have to upgrade to 3D
            if hasattr(current, "to_3D"):
                # check to see if the scene is transforming the path out of plane
                check = util.isclose(transform, util._IDENTITY, atol=1e-8)
                check[:2, :3] = True
                if not check.all():
                    # transform moves in 3D so we put this on the Z=0 plane
                    current = current.to_3D()
                else:
                    # transform moves in 2D so clip off the last row and column
                    transform = transform[:3, :3]

            # move the geometry vertices into the requested frame
            current.apply_transform(transform)
            current.metadata["name"] = geometry_name
            current.metadata["node"] = node_name

            # save to our list of meshes
            result.append(current)

        if concatenate:
            # if scene has mixed geometry this may drop some of it
            return util.concatenate(result)

        return np.array(result)

    def subscene(self, node):
        """
        Get part of a scene that succeeds a specified node.

        Parameters
        ------------
        node : any
          Hashable key in `scene.graph`

        Returns
        -----------
        subscene : Scene
          Partial scene generated from current.
        """
        # get every node that is a successor to specified node
        # this includes `node`
        graph = self.graph
        nodes = graph.transforms.successors(node)
        # get every edge that has an included node
        edges = [e for e in graph.to_edgelist() if e[0] in nodes]

        # create a scene graph when
        graph = SceneGraph(base_frame=node)
        graph.from_edgelist(edges)

        geometry_names = set([e[2]["geometry"] for e in edges if "geometry" in e[2]])
        geometry = {k: self.geometry[k] for k in geometry_names}
        result = Scene(geometry=geometry, graph=graph)
        return result

    @caching.cache_decorator
    def convex_hull(self):
        """
        The convex hull of the whole scene

        Returns
        ---------
        hull: Trimesh object, convex hull of all meshes in scene
        """
        points = util.vstack_empty([m.vertices for m in self.dump()])
        hull = convex.convex_hull(points)
        return hull

    def export(self, file_obj=None, file_type=None, **kwargs):
        """
        Export a snapshot of the current scene.

        Parameters
        ----------
        file_obj : str, file-like, or None
          File object to export to
        file_type : str or None
          What encoding to use for meshes
          IE: dict, dict64, stl

        Returns
        ----------
        export : bytes
          Only returned if file_obj is None
        """
        return export.export_scene(scene=self, file_obj=file_obj, file_type=file_type, **kwargs)

    def save_image(self, resolution=None, **kwargs):
        """
        Get a PNG image of a scene.

        Parameters
        -----------
        resolution : (2,) int
          Resolution to render image
        **kwargs
          Passed to SceneViewer constructor

        Returns
        -----------
        png : bytes
          Render of scene as a PNG
        """
        from ..viewer.windowed import render_scene

        png = render_scene(scene=self, resolution=resolution, **kwargs)
        return png

    @property
    def units(self):
        """
        Get the units for every model in the scene, and
        raise a ValueError if there are mixed units.

        Returns
        -----------
        units : str
          Units for every model in the scene
        """
        existing = [i.units for i in self.geometry.values()]

        if any(existing[0] != e for e in existing):
            # if all of our geometry doesn't have the same units already
            # this function will only do some hot nonsense
            raise ValueError("models in scene have inconsistent units!")

        return existing[0]

    @units.setter
    def units(self, value):
        """
        Set the units for every model in the scene without
        converting any units just setting the tag.

        Parameters
        ------------
        value : str
          Value to set every geometry unit value to
        """
        for m in self.geometry.values():
            m.units = value

    def convert_units(self, desired, guess=False):
        """
        If geometry has units defined convert them to new units.

        Returns a new scene with geometries and transforms scaled.

        Parameters
        ----------
        desired : str
          Desired final unit system: 'inches', 'mm', etc.
        guess : bool
          Is the converter allowed to guess scale when models
          don't have it specified in their metadata.

        Returns
        ----------
        scaled : trimesh.Scene
          Copy of scene with scaling applied and units set
          for every model
        """
        # if there is no geometry do nothing
        if len(self.geometry) == 0:
            return self.copy()

        current = self.units
        if current is None:
            # will raise ValueError if not in metadata
            # and not allowed to guess
            current = units.units_from_metadata(self, guess=guess)

        # find the float conversion
        scale = units.unit_conversion(current=current, desired=desired)

        # exit early if our current units are the same as desired units
        if np.isclose(scale, 1.0):
            result = self.copy()
        else:
            result = self.scaled(scale=scale)

        # apply the units to every geometry of the scaled result
        result.units = desired

        return result

    def explode(self, vector=None, origin=None):
        """
        Explode a scene around a point and vector.

        Parameters
        -----------
        vector : (3,) float or float
           Explode radially around a direction vector or spherically
        origin : (3,) float
          Point to explode around
        """
        if origin is None:
            origin = self.centroid
        if vector is None:
            vector = self.scale / 25.0

        vector = np.asanyarray(vector, dtype=np.float64)
        origin = np.asanyarray(origin, dtype=np.float64)

        for node_name in self.graph.nodes_geometry:
            transform, geometry_name = self.graph[node_name]
            centroid = self.geometry[geometry_name].centroid
            # transform centroid into nodes location
            centroid = np.dot(transform, np.append(centroid, 1))[:3]

            if vector.shape == ():
                # case where our vector is a single number
                offset = (centroid - origin) * vector
            elif np.shape(vector) == (3,):
                projected = np.dot(vector, (centroid - origin))
                offset = vector * projected
            else:
                raise ValueError("explode vector wrong shape!")

            # original transform is read-only
            T_new = transform.copy()
            T_new[:3, 3] += offset
            self.graph[node_name] = T_new

    def scaled(self, scale):
        """
        Return a copy of the current scene, with meshes and scene
        transforms scaled to the requested factor.

        Parameters
        -----------
        scale : float or (3,) float
          Factor to scale meshes and transforms

        Returns
        -----------
        scaled : trimesh.Scene
          A copy of the current scene but scaled
        """
        # convert 2D geometries to 3D for 3D scaling factors
        scale_is_3D = isinstance(scale, (list, tuple, np.ndarray)) and len(scale) == 3

        if scale_is_3D and np.all(np.asarray(scale) == scale[0]):
            # scale is uniform
            scale = float(scale[0])
            scale_is_3D = False
        elif not scale_is_3D:
            scale = float(scale)

        # result is a copy
        result = self.copy()

        if scale_is_3D:
            # Copy all geometries that appear multiple times in the scene,
            # such that no two nodes share the same geometry.
            # This is required since the non-uniform scaling will most likely
            # affect the same geometry in different poses differently.
            # Note, that this is not needed in the case of uniform scaling.
            for geom_name in result.graph.geometry_nodes:
                nodes_with_geom = result.graph.geometry_nodes[geom_name]
                if len(nodes_with_geom) > 1:
                    geom = result.geometry[geom_name]
                    for n in nodes_with_geom:
                        p = result.graph.transforms.parents[n]
                        result.add_geometry(
                            geometry=geom.copy(),
                            geom_name=geom_name,
                            node_name=n,
                            parent_node_name=p,
                            transform=result.graph.transforms.edge_data[(p, n)].get("matrix", None),
                            metadata=result.graph.transforms.edge_data[(p, n)].get("metadata", None),
                        )
                    result.delete_geometry(geom_name)

            # Convert all 2D paths to 3D paths
            for geom_name in result.geometry:
                if result.geometry[geom_name].vertices.shape[1] == 2:
                    result.geometry[geom_name] = result.geometry[geom_name].to_3D()

            # Scale all geometries by un-doing their local rotations first
            for key in result.graph.nodes_geometry:
                T, geom_name = result.graph.get(key)
                # transform from graph should be read-only
                T = T.copy()
                T[:3, 3] = 0.0

                # Get geometry transform w.r.t. base frame
                result.geometry[geom_name].apply_transform(T).apply_scale(scale).apply_transform(np.linalg.inv(T))

            # Scale all transformations in the scene graph
            edge_data = result.graph.transforms.edge_data
            for uv in edge_data:
                if "matrix" in edge_data[uv]:
                    props = edge_data[uv]
                    T = edge_data[uv]["matrix"].copy()
                    T[:3, 3] *= scale
                    props["matrix"] = T
                    result.graph.update(frame_from=uv[0], frame_to=uv[1], **props)
            # Clear cache
            result.graph.transforms._cache = {}
            result.graph.transforms._modified = gs.UID()
            result.graph._cache.clear()
        else:
            # matrix for 2D scaling
            scale_2D = np.eye(3) * scale
            # matrix for 3D scaling
            scale_3D = np.eye(4) * scale

            # preallocate transforms and geometries
            nodes = np.array(self.graph.nodes_geometry)
            transforms = np.zeros((len(nodes), 4, 4))
            geometries = [None] * len(nodes)

            # collect list of transforms
            for i, node in enumerate(nodes):
                transforms[i], geometries[i] = self.graph[node]

            # remove all existing transforms
            result.graph.clear()

            for group in grouping.group(geometries):
                # hashable reference to self.geometry
                geometry = geometries[group[0]]
                # original transform from world to geometry
                original = transforms[group[0]]
                # transform for geometry
                new_geom = np.dot(scale_3D, original)

                if result.geometry[geometry].vertices.shape[1] == 2:
                    # if our scene is 2D only scale in 2D
                    result.geometry[geometry].apply_transform(scale_2D)
                else:
                    # otherwise apply the full transform
                    result.geometry[geometry].apply_transform(new_geom)

                for node, T in zip(nodes[group], transforms[group]):
                    # generate the new transforms
                    transform = util.multi_dot([scale_3D, T, np.linalg.inv(new_geom)])
                    # apply scale to translation
                    transform[:3, 3] *= scale
                    # update scene with new transforms
                    result.graph.update(frame_to=node, matrix=transform, geometry=geometry)
        return result

    def copy(self):
        """
        Return a deep copy of the current scene

        Returns
        ----------
        copied : trimesh.Scene
          Copy of the current scene
        """
        # use the geometries copy method to
        # allow them to handle references to unpickle-able objects
        geometry = {n: g.copy() for n, g in self.geometry.items()}

        if not hasattr(self, "_camera") or self._camera is None:
            # if no camera set don't include it
            camera = None
        else:
            # otherwise get a copy of the camera
            camera = self.camera.copy()
        # create a new scene with copied geometry and graph
        copied = Scene(geometry=geometry, graph=self.graph.copy(), metadata=self.metadata.copy(), camera=camera)
        return copied

    def show(self, viewer=None, **kwargs):
        """
        Display the current scene.

        Parameters
        -----------
        viewer : Union[str, callable, None]
          What kind of viewer to use, such as
          'gl' to open a pyglet window, 'notebook'
          for a jupyter notebook or None
        kwargs : dict
          Includes `smooth`, which will turn
          on or off automatic smooth shading
        """

        if viewer is None:
            # check to see if we are in a notebook or not
            from ..viewer import in_notebook

            if in_notebook():
                viewer = "notebook"
            else:
                viewer = "gl"

        if viewer == "gl":
            # this imports pyglet, and will raise an ImportError
            # if pyglet is not available
            from ..viewer import SceneViewer

            return SceneViewer(self, **kwargs)
        elif viewer == "notebook":
            from ..viewer import scene_to_notebook

            return scene_to_notebook(self, **kwargs)
        elif callable(viewer):
            # if a callable method like a custom class
            # constructor was passed run using that
            return viewer(self, **kwargs)
        else:
            raise ValueError('viewer must be "gl", "notebook", callable, or None')

    def __add__(self, other):
        """
        Concatenate the current scene with another scene or mesh.

        Parameters
        ------------
        other : trimesh.Scene, trimesh.Trimesh, trimesh.Path
           Other object to append into the result scene

        Returns
        ------------
        appended : trimesh.Scene
           Scene with geometry from both scenes
        """
        result = append_scenes([self, other], common=[self.graph.base_frame])
        return result


def split_scene(geometry, **kwargs):
    """
    Given a geometry, list of geometries, or a Scene
    return them as a single Scene object.

    Parameters
    ----------
    geometry : splittable

    Returns
    ---------
    scene: trimesh.Scene
    """
    # already a scene, so return it
    if util.is_instance_named(geometry, "Scene"):
        return geometry

    # a list of things
    if util.is_sequence(geometry):
        metadata = {}
        for g in geometry:
            try:
                metadata.update(g.metadata)
            except BaseException:
                continue
        return Scene(geometry, metadata=metadata)

    # a single geometry so we are going to split
    split = []
    metadata = {}
    for g in util.make_sequence(geometry):
        split.extend(g.split(**kwargs))
        metadata.update(g.metadata)

    # if there is only one geometry in the mesh
    # name it from the file name
    if len(split) == 1 and "file_name" in metadata:
        split = {metadata["file_name"]: split[0]}

    scene = Scene(split, metadata=metadata)

    return scene


def append_scenes(iterable, common=None, base_frame="world"):
    """
    Concatenate multiple scene objects into one scene.

    Parameters
    -------------
    iterable : (n,) Trimesh or Scene
       Geometries that should be appended
    common : (n,) str
       Nodes that shouldn't be remapped
    base_frame : str
       Base frame of the resulting scene

    Returns
    ------------
    result : trimesh.Scene
       Scene containing all geometry
    """
    if isinstance(iterable, Scene):
        return iterable

    if common is None:
        common = [base_frame]

    # save geometry in dict
    geometry = {}
    # save transforms as edge tuples
    edges = []

    # nodes which shouldn't be remapped
    common = set(common)
    # nodes which are consumed and need to be remapped
    consumed = set()

    def node_remap(node):
        """
        Remap node to new name if necessary

        Parameters
        -------------
        node : hashable
           Node name in original scene

        Returns
        -------------
        name : hashable
           Node name in concatenated scene
        """

        # if we've already remapped a node use it
        if node in map_node:
            return map_node[node]

        # if a node is consumed and isn't one of the nodes
        # we're going to hold common between scenes remap it
        if node not in common and node in consumed:
            # generate a name not in consumed
            name = node + util.unique_id()
            map_node[node] = name
            node = name

        # keep track of which nodes have been used
        # in the current scene
        current.add(node)
        return node

    # loop through every geometry
    for s in iterable:
        # allow Trimesh/Path2D geometry to be passed
        if hasattr(s, "scene"):
            s = s.scene()
        # if we don't have a scene raise an exception
        if not isinstance(s, Scene):
            raise ValueError("{} is not a scene!".format(type(s).__name__))

        # remap geometries if they have been consumed
        map_geom = {}
        for k, v in s.geometry.items():
            # if a geometry already exists add a UUID to the name
            name = unique_name(start=k, contains=geometry.keys())
            # store name mapping
            map_geom[k] = name
            # store geometry with new name
            geometry[name] = v

        # remap nodes and edges so duplicates won't
        # stomp all over each other
        map_node = {}
        # the nodes used in this scene
        current = set()
        for a, b, attr in s.graph.to_edgelist():
            # remap node names from local names
            a, b = node_remap(a), node_remap(b)
            # remap geometry keys
            # if key is not in map_geom it means one of the scenes
            # referred to geometry that doesn't exist
            # rather than crash here we ignore it as the user
            # possibly intended to add in geometries back later
            if "geometry" in attr and attr["geometry"] in map_geom:
                attr["geometry"] = map_geom[attr["geometry"]]
            # save the new edge
            edges.append((a, b, attr))
        # mark nodes from current scene as consumed
        consumed.update(current)

    # add all data to a new scene
    result = Scene(base_frame=base_frame)
    result.graph.from_edgelist(edges)
    result.geometry.update(geometry)

    return result
