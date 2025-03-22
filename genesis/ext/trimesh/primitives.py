"""
primitives.py
----------------

Subclasses of Trimesh objects that are parameterized as primitives.

Useful because you can move boxes and spheres around
and then use trimesh operations on them at any point.
"""

import numpy as np
import abc

from . import util
from . import sample
from . import caching
from . import inertia
from . import creation
from . import triangles
from . import transformations as tf

from .base import Trimesh
from .constants import log, tol

# immutable identity matrix for checks
_IDENTITY = np.eye(4)
_IDENTITY.flags.writeable = False


class _Primitive(Trimesh):
    """
    Geometric _Primitives which are a subclass of Trimesh.
    Mesh is generated lazily when vertices or faces are requested.
    """

    # ignore superclass copy directives
    __copy__ = None
    __deepcopy__ = None

    def __init__(self):
        # run the Trimesh constructor with no arguments
        super(_Primitive, self).__init__()

        # remove any data
        self._data.clear()
        self._validate = False

        # make sure any cached numpy arrays have
        # set `array.flags.writable = False`
        self._cache.force_immutable = True

    def __repr__(self):
        return "<trimesh.primitives.{}>".format(type(self).__name__)

    @property
    def faces(self):
        stored = self._cache["faces"]
        if util.is_shape(stored, (-1, 3)):
            return stored
        self._create_mesh()
        return self._cache["faces"]

    @faces.setter
    def faces(self, values):
        log.warning("primitive faces are immutable: not setting!")

    @property
    def vertices(self):
        stored = self._cache["vertices"]
        if util.is_shape(stored, (-1, 3)):
            return stored

        self._create_mesh()
        return self._cache["vertices"]

    @vertices.setter
    def vertices(self, values):
        if values is not None:
            log.warning("primitive vertices are immutable: not setting!")

    @property
    def face_normals(self):
        # we need to avoid the logic in the superclass that
        # is specific to the data model prioritizing faces
        stored = self._cache["face_normals"]
        if util.is_shape(stored, (-1, 3)):
            return stored
        # just calculate if not stored
        unit, valid = triangles.normals(self.triangles)
        normals = np.zeros((len(valid), 3))
        normals[valid] = unit
        # store and return
        self._cache["face_normals"] = normals
        return normals

    @face_normals.setter
    def face_normals(self, values):
        if values is not None:
            log.warning("Primitive face normals are immutable!")

    @property
    def transform(self):
        """
        The transform of the Primitive object.

        Returns
        -------------
        transform : (4, 4) float
          Homogeneous transformation matrix
        """
        return self.primitive.transform

    @abc.abstractmethod
    def to_dict(self):
        """
        Should be implemented by each primitive.
        """
        raise NotImplementedError()

    def copy(self, **kwargs):
        """
        Return a copy of the Primitive object.

        Returns
        -------------
        copied : object
          Copy of current primitive
        """
        # get the constructor arguments
        kwargs.update(self.to_dict())
        # remove the type indicator, i.e. `Cylinder`
        kwargs.pop("kind")
        # create a new object with kwargs
        primitive_copy = type(self)(**kwargs)
        # copy metadata
        primitive_copy.metadata = self.metadata.copy()

        for k, v in self._data.data.items():
            if k not in primitive_copy._data:
                primitive_copy._data[k] = v

        return primitive_copy

    def to_mesh(self, **kwargs):
        """
        Return a copy of the Primitive object as a Trimesh.

        Parameters
        -----------
        kwargs : dict
          Passed to the Trimesh object constructor.

        Returns
        ------------
        mesh : trimesh.Trimesh
          Tessellated version of the primitive.
        """
        result = Trimesh(
            vertices=self.vertices.copy(),
            faces=self.faces.copy(),
            face_normals=self.face_normals.copy(),
            process=kwargs.pop("process", False),
            **kwargs,
        )
        return result

    def apply_transform(self, matrix):
        """
        Apply a transform to the current primitive by
        applying a new transform on top of existing
        `self.primitive.transform`. If the matrix
        contains scaling it will change parameters
        like `radius` or `height` automatically.

        Parameters
        ------------
        matrix: (4, 4) float
          Homogeneous transformation
        """
        matrix = np.asanyarray(matrix, order="C", dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError("matrix must be `(4, 4)`!")
        if util.allclose(matrix, _IDENTITY, 1e-8):
            # identity matrix is a no-op
            return self

        prim = self.primitive
        # copy the current transform
        current = prim.transform.copy()
        # see if matrix has scaling from the matrix
        scale = np.linalg.det(matrix[:3, :3]) ** (1.0 / 3.0)

        # the objects we handle re-scaling for
        # note that `Extrusion` is NOT supported
        kinds = (Box, Cylinder, Capsule, Sphere)
        if isinstance(self, kinds) and abs(scale - 1.0) > 1e-8:
            # scale the primitive attributes
            if hasattr(prim, "height"):
                prim.height *= scale
            if hasattr(prim, "radius"):
                prim.radius *= scale
            if hasattr(prim, "extents"):
                prim.extents *= scale
            # scale the translation of the current matrix
            current[:3, 3] *= scale
            # apply new matrix, rescale, translate, current
            updated = util.multi_dot([matrix, tf.scale_matrix(1.0 / scale), current])
        else:
            # without scaling just multiply
            updated = np.dot(matrix, current)

        # make sure matrix is a rigid transform
        if not tf.is_rigid(updated):
            raise ValueError("Couldn't produce rigid transform!")

        # apply the new matrix
        self.primitive.transform = updated

        return self

    def _create_mesh(self):
        raise ValueError("Primitive doesn't define mesh creation!")


class _PrimitiveAttributes(object):
    """
    Hold the mutable data which defines a primitive.
    """

    def __init__(self, parent, defaults, kwargs, mutable=True):
        """
        Hold the attributes for a Primitive.

        Parameters
        ------------
        parent : _Primitive
          Parent object reference.
        defaults : dict
          The default values for this primitive type.
        kwargs : dict
          User-passed values, i.e. {'radius': 10.0}
        """
        # store actual data in parent object
        self._data = parent._data
        # default values define the keys
        self._defaults = defaults
        # store a reference to the parent ubject
        self._parent = parent
        # start with a copy of all default objects
        self._data.update(defaults)
        # store whether this data is mutable after creation
        self._mutable = mutable
        # assign the keys passed by the user only if
        # they are a property of this primitive
        for key, default in defaults.items():
            value = kwargs.get(key, None)
            if value is not None:
                # convert passed data into type of defaults
                self._data[key] = util.convert_like(value, default)
        # make sure stored values are immutable after setting
        if not self._mutable:
            self._data.mutable = False

    @property
    def __doc__(self):
        # this is generated dynamically as the format
        # operation can be surprisingly slow and most
        # people never call it
        import pprint

        doc = (
            "Store the attributes of a {name} object.\n\n"
            + "When these values are changed, the mesh geometry will \n"
            + "automatically be updated to reflect the new values.\n\n"
            + "Available properties and their default values are:\n {defaults}"
            + "\n\nExample\n---------------\n"
            + "p = trimesh.primitives.{name}()\n"
            + "p.primitive.radius = 10\n"
            + "\n"
        ).format(name=self._parent.__class__.__name__, defaults=pprint.pformat(self._defaults, width=-1)[1:-1])
        return doc

    def __getattr__(self, key):
        if key.startswith("_"):
            return super(_PrimitiveAttributes, self).__getattr__(key)
        elif key == "center":
            # this whole __getattr__ is a little hacky
            return self._data["transform"][:3, 3]
        elif key in self._defaults:
            return util.convert_like(self._data[key], self._defaults[key])
        raise AttributeError("primitive object has no attribute '{}' ".format(key))

    def __setattr__(self, key, value):
        if key.startswith("_"):
            return super(_PrimitiveAttributes, self).__setattr__(key, value)
        elif key == "center":
            value = np.array(value, dtype=np.float64)
            transform = np.eye(4)
            transform[:3, 3] = value
            self._data["transform"] = transform
            return
        elif key in self._defaults:
            if self._mutable:
                self._data[key] = util.convert_like(value, self._defaults[key])
            else:
                raise ValueError("Primitive is configured as immutable! Cannot set attribute!")
        else:
            keys = list(self._defaults.keys())
            raise ValueError("Only default attributes {} can be set!".format(keys))

    def __dir__(self):
        result = sorted(dir(type(self)) + list(self._defaults.keys()))
        return result


class Cylinder(_Primitive):

    def __init__(self, radius=1.0, height=1.0, transform=None, sections=32, mutable=True):
        """
        Create a Cylinder Primitive, a subclass of Trimesh.

        Parameters
        -------------
        radius : float
          Radius of cylinder
        height : float
          Height of cylinder
        transform : (4, 4) float
          Homogeneous transformation matrix
        sections : int
          Number of facets in circle.
        mutable : bool
          Are extents and transform mutable after creation.
        """
        super(Cylinder, self).__init__()

        defaults = {"height": 10.0, "radius": 1.0, "transform": np.eye(4), "sections": 32}
        self.primitive = _PrimitiveAttributes(
            self,
            defaults=defaults,
            kwargs={"height": height, "radius": radius, "transform": transform, "sections": sections},
            mutable=mutable,
        )

    @caching.cache_decorator
    def volume(self):
        """
        The analytic volume of the cylinder primitive.

        Returns
        ---------
        volume : float
          Volume of the cylinder
        """
        volume = (np.pi * self.primitive.radius**2) * self.primitive.height
        return volume

    @caching.cache_decorator
    def moment_inertia(self):
        """
        The analytic inertia tensor of the cylinder primitive.

        Returns
        ----------
        tensor: (3, 3) float
          3D inertia tensor
        """

        tensor = inertia.cylinder_inertia(
            mass=self.volume,
            radius=self.primitive.radius,
            height=self.primitive.height,
            transform=self.primitive.transform,
        )
        return tensor

    @caching.cache_decorator
    def direction(self):
        """
        The direction of the cylinder's axis.

        Returns
        --------
        axis: (3,) float, vector along the cylinder axis
        """
        axis = np.dot(self.primitive.transform, [0, 0, 1, 0])[:3]
        return axis

    @property
    def segment(self):
        """
        A line segment which if inflated by cylinder radius
        would represent the cylinder primitive.

        Returns
        -------------
        segment : (2, 3) float
          Points representing a single line segment
        """
        # half the height
        half = self.primitive.height / 2.0
        # apply the transform to the Z- aligned segment
        points = np.dot(self.primitive.transform, np.transpose([[0, 0, -half, 1], [0, 0, half, 1]])).T[:, :3]
        return points

    def to_dict(self):
        """
        Get a copy of the current Cylinder primitive as
        a JSON-serializable dict that matches the schema
        in `trimesh/resources/schema/cylinder.schema.json`

        Returns
        ----------
        as_dict : dict
          Serializable data for this primitive.
        """
        return {
            "kind": "cylinder",
            "transform": self.primitive.transform.tolist(),
            "radius": float(self.primitive.radius),
            "height": float(self.primitive.height),
        }

    def buffer(self, distance):
        """
        Return a cylinder primitive which covers the source
        cylinder by distance: radius is inflated by distance
        height by twice the distance.

        Parameters
        ------------
        distance : float
          Distance to inflate cylinder radius and height

        Returns
        -------------
        buffered : Cylinder
         Cylinder primitive inflated by distance
        """
        distance = float(distance)
        buffered = Cylinder(
            height=self.primitive.height + distance * 2,
            radius=self.primitive.radius + distance,
            transform=self.primitive.transform.copy(),
        )
        return buffered

    def _create_mesh(self):
        log.debug("creating mesh for Cylinder primitive")
        mesh = creation.cylinder(
            radius=self.primitive.radius,
            height=self.primitive.height,
            sections=self.primitive.sections,
            transform=self.primitive.transform,
        )

        self._cache["vertices"] = mesh.vertices
        self._cache["faces"] = mesh.faces
        self._cache["face_normals"] = mesh.face_normals


class Capsule(_Primitive):

    def __init__(self, radius=1.0, height=10.0, transform=None, sections=32, mutable=True):
        """
        Create a Capsule Primitive, a subclass of Trimesh.

        Parameters
        ----------
        radius : float
          Radius of cylinder
        height : float
          Height of cylinder
        transform : (4, 4) float
          Transformation matrix
        sections : int
          Number of facets in circle
        mutable : bool
          Are extents and transform mutable after creation.
        """
        super(Capsule, self).__init__()

        defaults = {"height": 1.0, "radius": 1.0, "transform": np.eye(4), "sections": 32}
        self.primitive = _PrimitiveAttributes(
            self,
            defaults=defaults,
            kwargs={"height": height, "radius": radius, "transform": transform, "sections": sections},
            mutable=mutable,
        )

    @property
    def transform(self):
        return self.primitive.transform

    def to_dict(self):
        """
        Get a copy of the current Capsule primitive as
        a JSON-serializable dict that matches the schema
        in `trimesh/resources/schema/capsule.schema.json`

        Returns
        ----------
        as_dict : dict
          Serializable data for this primitive.
        """
        return {
            "kind": "capsule",
            "transform": self.primitive.transform.tolist(),
            "height": float(self.primitive.height),
            "radius": float(self.primitive.radius),
        }

    @caching.cache_decorator
    def direction(self):
        """
        The direction of the capsule's axis.

        Returns
        --------
        axis : (3,) float
          Vector along the cylinder axis
        """
        axis = np.dot(self.primitive.transform, [0, 0, 1, 0])[:3]
        return axis

    def _create_mesh(self):
        log.debug("creating mesh for `Capsule` primitive")

        mesh = creation.capsule(radius=self.primitive.radius, height=self.primitive.height)
        mesh.apply_transform(self.primitive.transform)

        self._cache["vertices"] = mesh.vertices
        self._cache["faces"] = mesh.faces
        self._cache["face_normals"] = mesh.face_normals


class Sphere(_Primitive):

    def __init__(self, radius=1.0, center=None, transform=None, subdivisions=3, mutable=True):
        """
        Create a Sphere Primitive, a subclass of Trimesh.

        Parameters
        ----------
        radius : float
          Radius of sphere
        center : None or (3,) float
          Center of sphere.
        transform : None or (4, 4) float
          Full homogeneous transform. Pass `center` OR `transform.
        subdivisions : int
          Number of subdivisions for icosphere.
        mutable : bool
          Are extents and transform mutable after creation.
        """

        super(Sphere, self).__init__()

        defaults = {"radius": 1.0, "transform": np.eye(4), "subdivisions": 3}
        constructor = {"radius": float(radius), "subdivisions": int(subdivisions)}
        # center is a helper method for "transform"
        # since a sphere is rotationally symmetric
        if center is not None:
            if transform is not None:
                raise ValueError("only one of `center` and `transform` may be passed!")
            translate = np.eye(4)
            translate[:3, 3] = center
            constructor["transform"] = translate
        elif transform is not None:
            constructor["transform"] = transform

        # create the attributes object
        self.primitive = _PrimitiveAttributes(self, defaults=defaults, kwargs=constructor, mutable=mutable)

    @property
    def center(self):
        return self.primitive.center

    @center.setter
    def center(self, value):
        self.primitive.center = value

    def to_dict(self):
        """
        Get a copy of the current Sphere primitive as
        a JSON-serializable dict that matches the schema
        in `trimesh/resources/schema/sphere.schema.json`

        Returns
        ----------
        as_dict : dict
          Serializable data for this primitive.
        """
        return {
            "kind": "sphere",
            "transform": self.primitive.transform.tolist(),
            "radius": float(self.primitive.radius),
        }

    @property
    def bounds(self):
        # no docstring so will inherit Trimesh docstring
        # return exact bounds from primitive center and radius (rather than faces)
        # self.extents will also use this information
        bounds = np.array(
            [self.primitive.center - self.primitive.radius, self.primitive.center + self.primitive.radius]
        )
        return bounds

    @property
    def bounding_box_oriented(self):
        # for a sphere the oriented bounding box is the same as the axis aligned
        # bounding box, and a sphere is the absolute slowest case for the OBB calculation
        # as it is a convex surface with a ton of face normals that all need to
        # be checked
        return self.bounding_box

    @caching.cache_decorator
    def area(self):
        """
        Surface area of the current sphere primitive.

        Returns
        --------
        area: float, surface area of the sphere Primitive
        """

        area = 4.0 * np.pi * (self.primitive.radius**2)
        return area

    @caching.cache_decorator
    def volume(self):
        """
        Volume of the current sphere primitive.

        Returns
        --------
        volume: float, volume of the sphere Primitive
        """

        volume = (4.0 * np.pi * (self.primitive.radius**3)) / 3.0
        return volume

    @caching.cache_decorator
    def moment_inertia(self):
        """
        The analytic inertia tensor of the sphere primitive.

        Returns
        ----------
        tensor: (3, 3) float
          3D inertia tensor.
        """
        return inertia.sphere_inertia(mass=self.volume, radius=self.primitive.radius)

    def _create_mesh(self):
        log.debug("creating mesh for Sphere primitive")
        unit = creation.icosphere(subdivisions=self.primitive.subdivisions, radius=self.primitive.radius)

        # apply the center offset here
        self._cache["vertices"] = unit.vertices + self.primitive.center
        self._cache["faces"] = unit.faces
        self._cache["face_normals"] = unit.face_normals


class Box(_Primitive):
    def __init__(self, extents=None, transform=None, bounds=None, mutable=True):
        """
        Create a Box Primitive as a subclass of Trimesh

        Parameters
        ----------
        extents : Optional[ndarray] (3,) float
          Length of each side of the 3D box.
        transform : Optional[ndarray] (4, 4) float
          Homogeneous transformation matrix for box center.
        bounds : Optional[ndarray] (2, 3) float
          Axis aligned bounding box, if passed extents and
          transform will be derived from this.
        mutable : bool
          Are extents and transform mutable after creation.
        """
        super(Box, self).__init__()
        defaults = {"transform": np.eye(4), "extents": np.ones(3)}

        if bounds is not None:
            # validate the multiple forms of input available here
            if extents is not None or transform is not None:
                raise ValueError("if `bounds` is passed `extents` and `transform` must not be!")
            bounds = np.array(bounds, dtype=np.float64)
            if bounds.shape != (2, 3):
                raise ValueError("`bounds` must be (2, 3) float")
            # create extents from AABB
            extents = np.ptp(bounds, axis=0)
            # translate to the center of the box
            transform = np.eye(4)
            transform[:3, 3] = bounds[0] + extents / 2.0

        self.primitive = _PrimitiveAttributes(
            self, defaults=defaults, kwargs={"extents": extents, "transform": transform}, mutable=mutable
        )

    def to_dict(self):
        """
        Get a copy of the current Box primitive as
        a JSON-serializable dict that matches the schema
        in `trimesh/resources/schema/box.schema.json`

        Returns
        ----------
        as_dict : dict
          Serializable data for this primitive.
        """
        return {
            "kind": "box",
            "transform": self.primitive.transform.tolist(),
            "extents": self.primitive.extents.tolist(),
        }

    @property
    def transform(self):
        return self.primitive.transform

    def sample_volume(self, count):
        """
        Return random samples from inside the volume of the box.

        Parameters
        -------------
        count : int
          Number of samples to return

        Returns
        ----------
        samples : (count, 3) float
          Points inside the volume
        """
        samples = sample.volume_rectangular(
            extents=self.primitive.extents, count=count, transform=self.primitive.transform
        )
        return samples

    def sample_grid(self, count=None, step=None):
        """
        Return a 3D grid which is contained by the box.
        Samples are either 'step' distance apart, or there are
        'count' samples per box side.

        Parameters
        -----------
        count : int or (3,) int
          If specified samples are spaced with np.linspace
        step : float or (3,) float
          If specified samples are spaced with np.arange

        Returns
        -----------
        grid : (n, 3) float
          Points inside the box
        """

        if count is not None and step is not None:
            raise ValueError("only step OR count can be specified!")

        # create pre- transform bounds from extents
        bounds = np.array([-self.primitive.extents, self.primitive.extents]) * 0.5

        if step is not None:
            grid = util.grid_arange(bounds, step=step)
        elif count is not None:
            grid = util.grid_linspace(bounds, count=count)
        else:
            raise ValueError("either count or step must be specified!")

        transformed = tf.transform_points(grid, matrix=self.primitive.transform)
        return transformed

    @property
    def is_oriented(self):
        """
        Returns whether or not the current box is rotated at all.
        """
        if util.is_shape(self.primitive.transform, (4, 4)):
            return not np.allclose(self.primitive.transform[0:3, 0:3], np.eye(3))
        else:
            return False

    @caching.cache_decorator
    def volume(self):
        """
        Volume of the box Primitive.

        Returns
        --------
        volume : float
          Volume of box.
        """
        volume = float(np.prod(self.primitive.extents))
        return volume

    def _create_mesh(self):
        log.debug("creating mesh for Box primitive")
        box = creation.box(extents=self.primitive.extents, transform=self.primitive.transform)

        self._cache.cache.update(box._cache.cache)
        self._cache["vertices"] = box.vertices
        self._cache["faces"] = box.faces
        self._cache["face_normals"] = box.face_normals

    def as_outline(self):
        """
        Return a Path3D containing the outline of the box.

        Returns
        -----------
        outline : trimesh.path.Path3D
          Outline of box primitive
        """
        # do the import in function to keep soft dependency
        from .path.creation import box_outline

        # return outline with same size as primitive
        return box_outline(extents=self.primitive.extents, transform=self.primitive.transform)


class Extrusion(_Primitive):
    def __init__(self, polygon=None, transform=None, height=1.0, mutable=True):
        """
        Create an Extrusion primitive, which
        is a subclass of Trimesh.

        Parameters
        ----------
        polygon : shapely.geometry.Polygon
          Polygon to extrude
        transform : (4, 4) float
          Transform to apply after extrusion
        height : float
          Height to extrude polygon by
        mutable : bool
          Are extents and transform mutable after creation.
        """
        # do the import here, fail early if Shapely isn't installed
        from shapely.geometry import Point

        # run the Trimesh init
        super(Extrusion, self).__init__()
        # set default values
        defaults = {"polygon": Point([0, 0]).buffer(1.0), "transform": np.eye(4), "height": 1.0}

        self.primitive = _PrimitiveAttributes(
            self,
            defaults=defaults,
            kwargs={"transform": transform, "polygon": polygon, "height": height},
            mutable=mutable,
        )

    @caching.cache_decorator
    def area(self):
        """
        The surface area of the primitive extrusion.

        Calculated from polygon and height to avoid mesh creation.

        Returns
        ----------
        area: float
          Surface area of 3D extrusion
        """
        # area of the sides of the extrusion
        area = abs(self.primitive.height * self.primitive.polygon.length)
        # area of the two caps of the extrusion
        area += self.primitive.polygon.area * 2
        return area

    @caching.cache_decorator
    def volume(self):
        """
        The volume of the Extrusion primitive.
        Calculated from polygon and height to avoid mesh creation.

        Returns
        ----------
        volume : float
          Volume of 3D extrusion
        """
        # height may be negative
        volume = abs(self.primitive.polygon.area * self.primitive.height)
        return volume

    @caching.cache_decorator
    def direction(self):
        """
        Based on the extrudes transform what is the
        vector along which the polygon will be extruded.

        Returns
        ---------
        direction : (3,) float
          Unit direction vector
        """
        # only consider rotation and signed height
        direction = np.dot(self.primitive.transform[:3, :3], [0.0, 0.0, np.sign(self.primitive.height)])
        return direction

    @property
    def origin(self):
        """
        Based on the extrude transform what is the
        origin of the plane it is extruded from.

        Returns
        -----------
        origin : (3,) float
          Origin of extrusion plane
        """
        return self.primitive.transform[:3, 3]

    @property
    def transform(self):
        return self.primitive.transform

    @caching.cache_decorator
    def bounding_box_oriented(self):
        # no docstring for inheritance
        # calculate OBB using 2D polygon and known axis
        from . import bounds

        # find the 2D bounding box using the polygon
        to_origin, box = bounds.oriented_bounds_2D(self.primitive.polygon.exterior.coords)
        #  3D extents
        extents = np.append(box, abs(self.primitive.height))
        # calculate to_3D transform from 2D obb
        rotation_Z = np.linalg.inv(tf.planar_matrix_to_3D(to_origin))
        rotation_Z[2, 3] = self.primitive.height / 2.0
        # combine the 2D OBB transformation with the 2D projection transform
        to_3D = np.dot(self.primitive.transform, rotation_Z)
        obb = Box(transform=to_3D, extents=extents, mutable=False)
        return obb

    def slide(self, distance):
        """
        Alter the transform of the current extrusion to slide it
        along its extrude_direction vector

        Parameters
        -----------
        distance : float
          Distance along self.extrude_direction to move
        """
        distance = float(distance)
        translation = np.eye(4)
        translation[2, 3] = distance
        new_transform = np.dot(self.primitive.transform.copy(), translation.copy())
        self.primitive.transform = new_transform

    def buffer(self, distance, distance_height=None, **kwargs):
        """
        Return a new Extrusion object which is expanded in profile
        and in height by a specified distance.

        Parameters
        --------------
        distance : float
          Distance to buffer polygon
        distance_height : float
          Distance to buffer above and below extrusion
        kwargs : dict
          Passed to Extrusion constructor

        Returns
        ----------
        buffered : primitives.Extrusion
          Extrusion object with new values
        """
        distance = float(distance)
        # if not specified use same distance for everything
        if distance_height is None:
            distance_height = distance

        # start with current height
        height = self.primitive.height
        # if current height is negative offset by negative amount
        height += np.sign(height) * 2.0 * distance_height

        # create a new extrusion with a buffered polygon
        # use type(self) vs Extrusion to handle subclasses
        buffered = type(self)(
            transform=self.primitive.transform.copy(),
            polygon=self.primitive.polygon.buffer(distance),
            height=height,
            **kwargs,
        )

        # slide the stock along the axis
        buffered.slide(-np.sign(height) * distance_height)

        return buffered

    def to_dict(self):
        """
        Get a copy of the current Extrusion primitive as
        a JSON-serializable dict that matches the schema
        in `trimesh/resources/schema/extrusion.schema.json`

        Returns
        ----------
        as_dict : dict
          Serializable data for this primitive.
        """
        return {
            "kind": "extrusion",
            "polygon": self.primitive.polygon.wkt,
            "transform": self.primitive.transform.tolist(),
            "height": float(self.primitive.height),
        }

    def _create_mesh(self):
        log.debug("creating mesh for Extrusion primitive")
        # extrude the polygon along Z
        mesh = creation.extrude_polygon(
            polygon=self.primitive.polygon, height=self.primitive.height, transform=self.primitive.transform
        )

        # check volume here in unit tests
        if tol.strict and mesh.volume < 0.0:
            raise ValueError("matrix inverted mesh!")

        # cache mesh geometry in the primitive
        self._cache["vertices"] = mesh.vertices
        self._cache["faces"] = mesh.faces
