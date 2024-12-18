"""
parent.py
-------------

The base class for Trimesh, PointCloud, and Scene objects
"""

import abc
import warnings

import numpy as np

from . import caching
from . import transformations as tf
from .util import ABC


class Geometry(ABC):
    """
    `Geometry` is the parent class for all geometry.

    By decorating a method with `abc.abstractmethod` it means
    the objects that inherit from `Geometry` MUST implement
    those methods.
    """

    @abc.abstractproperty
    def bounds(self):
        pass

    @abc.abstractproperty
    def extents(self):
        pass

    @abc.abstractmethod
    def apply_transform(self, matrix):
        pass

    @abc.abstractmethod
    def is_empty(self):
        pass

    def crc(self):
        """
        DEPRECATED OCTOBER 2023 : Use `hash(geometry)`

        Get a hash of the current geometry.

        Returns
        ---------
        hash : int
          Hash of current graph and geometry.
        """
        warnings.warn(
            "`geometry.crc()` is deprecated and will "
            + "be removed in October 2023: replace "
            + "with `geometry.__hash__()` or `hash(geometry)`",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return self.__hash__()

    def hash(self):
        """
        DEPRECATED OCTOBER 2023 : Use `hash(geometry)`

        Get a hash of the current geometry.

        Returns
        ---------
        hash : int
          Hash of current graph and geometry.
        """
        warnings.warn(
            "`geometry.hash()` is deprecated and will "
            + "be removed in October 2023: replace "
            + "with `geometry.__hash__()` or `hash(geometry)`",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return self.__hash__()

    def md5(self):
        """
        DEPRECATED OCTOBER 2023 : Use `hash(geometry)`

        Get a hash of the current geometry.

        Returns
        ---------
        hash : int
          Hash of current graph and geometry.
        """
        warnings.warn(
            "`geometry.md5()` is deprecated and will "
            + "be removed in October 2023: replace "
            + "with `geometry.__hash__()` or `hash(geometry)`",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return self.__hash__()

    def __hash__(self):
        """
        Get a hash of the current geometry.

        Returns
        ---------
        hash : int
          Hash of current graph and geometry.
        """
        return self._data.__hash__()

    @abc.abstractmethod
    def copy(self):
        pass

    @abc.abstractmethod
    def show(self):
        pass

    @abc.abstractmethod
    def __add__(self, other):
        pass

    @abc.abstractmethod
    def export(self, file_obj, file_type=None):
        pass

    def __repr__(self):
        """
        Print quick summary of the current geometry without
        computing properties.

        Returns
        -----------
        repr : str
          Human readable quick look at the geometry.
        """
        elements = []
        if hasattr(self, "vertices"):
            # for Trimesh and PointCloud
            elements.append("vertices.shape={}".format(self.vertices.shape))
        if hasattr(self, "faces"):
            # for Trimesh
            elements.append("faces.shape={}".format(self.faces.shape))
        if hasattr(self, "geometry") and isinstance(self.geometry, dict):
            # for Scene
            elements.append("len(geometry)={}".format(len(self.geometry)))
        if "Voxel" in type(self).__name__:
            # for VoxelGrid objects
            elements.append(str(self.shape)[1:-1])
        return "<trimesh.{}({})>".format(type(self).__name__, ", ".join(elements))

    def apply_translation(self, translation):
        """
        Translate the current mesh.

        Parameters
        ----------
        translation : (3,) float
          Translation in XYZ
        """
        translation = np.asanyarray(translation, dtype=np.float64)
        if translation.shape == (2,):
            # create a planar matrix if we were passed a 2D offset
            return self.apply_transform(tf.planar_matrix(offset=translation))
        elif translation.shape != (3,):
            raise ValueError("Translation must be (3,) or (2,)!")

        # manually create a translation matrix
        matrix = np.eye(4)
        matrix[:3, 3] = translation
        return self.apply_transform(matrix)

    def apply_scale(self, scaling):
        """
        Scale the mesh.

        Parameters
        ----------
        scaling : float or (3,) float
          Scale factor to apply to the mesh
        """
        matrix = tf.scale_and_translate(scale=scaling)
        # apply_transform will work nicely even on negative scales
        return self.apply_transform(matrix)

    def __radd__(self, other):
        """
        Concatenate the geometry allowing concatenation with
        built in `sum()` function:
          `sum(Iterable[trimesh.Trimesh])`

        Parameters
        ------------
        other : Geometry
          Geometry or 0

        Returns
        ----------
        concat : Geometry
          Geometry of combined result
        """

        if other == 0:
            # adding 0 to a geometry never makes sense
            return self
        # otherwise just use the regular add function
        return self.__add__(type(self)(other))


class Geometry3D(Geometry):
    """
    The `Geometry3D` object is the parent object of geometry objects
    which are three dimensional, including Trimesh, PointCloud,
    and Scene objects.
    """

    @caching.cache_decorator
    def bounding_box(self):
        """
        An axis aligned bounding box for the current mesh.

        Returns
        ----------
        aabb : trimesh.primitives.Box
          Box object with transform and extents defined
          representing the axis aligned bounding box of the mesh
        """
        from . import primitives

        transform = np.eye(4)
        # translate to center of axis aligned bounds
        transform[:3, 3] = self.bounds.mean(axis=0)

        aabb = primitives.Box(transform=transform, extents=self.extents, mutable=False)
        return aabb

    @caching.cache_decorator
    def bounding_box_oriented(self):
        """
        An oriented bounding box for the current mesh.

        Returns
        ---------
        obb : trimesh.primitives.Box
          Box object with transform and extents defined
          representing the minimum volume oriented
          bounding box of the mesh
        """
        from . import primitives, bounds

        to_origin, extents = bounds.oriented_bounds(self)
        obb = primitives.Box(transform=np.linalg.inv(to_origin), extents=extents, mutable=False)
        return obb

    @caching.cache_decorator
    def bounding_sphere(self):
        """
        A minimum volume bounding sphere for the current mesh.

        Note that the Sphere primitive returned has an unpadded
        exact `sphere_radius` so while the distance of every vertex
        of the current mesh from sphere_center will be less than
        sphere_radius, the faceted sphere primitive may not
        contain every vertex.

        Returns
        --------
        minball : trimesh.primitives.Sphere
          Sphere primitive containing current mesh
        """
        from . import primitives, nsphere

        center, radius = nsphere.minimum_nsphere(self)
        minball = primitives.Sphere(center=center, radius=radius, mutable=False)
        return minball

    @caching.cache_decorator
    def bounding_cylinder(self):
        """
        A minimum volume bounding cylinder for the current mesh.

        Returns
        --------
        mincyl : trimesh.primitives.Cylinder
          Cylinder primitive containing current mesh
        """
        from . import primitives, bounds

        kwargs = bounds.minimum_cylinder(self)
        mincyl = primitives.Cylinder(mutable=False, **kwargs)
        return mincyl

    @caching.cache_decorator
    def bounding_primitive(self):
        """
        The minimum volume primitive (box, sphere, or cylinder) that
        bounds the mesh.

        Returns
        ---------
        bounding_primitive : object
          Smallest primitive which bounds the mesh:
          trimesh.primitives.Sphere
          trimesh.primitives.Box
          trimesh.primitives.Cylinder
        """
        options = [self.bounding_box_oriented, self.bounding_sphere, self.bounding_cylinder]
        volume_min = np.argmin([i.volume for i in options])
        bounding_primitive = options[volume_min]
        return bounding_primitive

    def apply_obb(self):
        """
        Apply the oriented bounding box transform to the current mesh.

        This will result in a mesh with an AABB centered at the
        origin and the same dimensions as the OBB.

        Returns
        ----------
        matrix : (4, 4) float
          Transformation matrix that was applied
          to mesh to move it into OBB frame
        """
        matrix = self.bounding_box_oriented.primitive.transform
        matrix = np.linalg.inv(matrix)
        self.apply_transform(matrix)
        return matrix
