"""
entities.py
--------------

Basic geometric primitives which only store references to
vertex indices rather than vertices themselves.
"""

from copy import deepcopy

import numpy as np

from .. import util
from ..util import ABC
from .arc import arc_center, discretize_arc
from .curve import discretize_bezier, discretize_bspline


class Entity(ABC):

    def __init__(self, points, closed=None, layer=None, metadata=None, color=None, **kwargs):
        # points always reference vertex indices and are int
        self.points = np.asanyarray(points, dtype=np.int64)
        # save explicit closed
        if closed is not None:
            self.closed = closed
        # save the passed layer
        if layer is not None:
            self.layer = layer
        if metadata is not None:
            self.metadata.update(metadata)

        self._cache = {}

        # save the passed color
        self.color = color
        # save any other kwargs for general use
        self.kwargs = kwargs

    @property
    def metadata(self):
        """
        Get any metadata about the entity.

        Returns
        ---------
        metadata : dict
          Bag of properties.
        """
        if not hasattr(self, "_metadata"):
            self._metadata = {}
        # note that we don't let a new dict be assigned
        return self._metadata

    @property
    def layer(self):
        """
        Set the layer the entity resides on as a shortcut
        to putting it in the entity metadata.

        Returns
        ----------
        layer : any
          Hashable layer identifier.
        """
        return self.metadata.get("layer")

    @layer.setter
    def layer(self, value):
        """
        Set the current layer of the entity.

        Returns
        ----------
        layer : any
          Hashable layer indicator
        """
        self.metadata["layer"] = value

    def to_dict(self):
        """
        Returns a dictionary with all of the information
        about the entity.

        Returns
        -----------
        as_dict : dict
          Has keys 'type', 'points', 'closed'
        """
        return {"type": self.__class__.__name__, "points": self.points.tolist(), "closed": self.closed}

    @property
    def closed(self):
        """
        If the first point is the same as the end point
        the entity is closed

        Returns
        -----------
        closed : bool
          Is the entity closed or not?
        """
        closed = len(self.points) > 2 and self.points[0] == self.points[-1]
        return closed

    @property
    def nodes(self):
        """
        Returns an (n,2) list of nodes, or vertices on the path.
        Note that this brief class function assumes that all of the
        reference points are on the path which is true for lines and
        three point arcs.

        If you were to define another class where that wasn't the case
        (for example, the control points of a bezier curve),
        you would need to implement an entity- specific version of this
        function.

        The purpose of having a list of nodes is so that they can then be
        added as edges to a graph so we can use functions to check
        connectivity, extract paths, etc.

        The slicing on this function is essentially just tiling points
        so the first and last vertices aren't repeated. Example:

        self.points = [0,1,2]
        returns:      [[0,1], [1,2]]
        """
        return np.column_stack((self.points, self.points)).reshape(-1)[1:-1].reshape((-1, 2))

    @property
    def end_points(self):
        """
        Returns the first and last points. Also note that if you
        define a new entity class where the first and last vertices
        in self.points aren't the endpoints of the curve you need to
        implement this function for your class.

        Returns
        -------------
        ends : (2,) int
          Indices of the two end points of the entity
        """
        return self.points[[0, -1]]

    @property
    def is_valid(self):
        """
        Is the current entity valid.

        Returns
        -----------
        valid : bool
          Is the current entity well formed
        """
        return True

    def reverse(self, direction=-1):
        """
        Reverse the current entity in place.

        Parameters
        ----------------
        direction : int
          If positive will not touch direction
          If negative will reverse self.points
        """
        if direction < 0:
            self._direction = -1
        else:
            self._direction = 1

    def _orient(self, curve):
        """
        Reverse a curve if a flag is set.

        Parameters
        --------------
        curve : (n, dimension) float
          Curve made up of line segments in space

        Returns
        ------------
        orient : (n, dimension) float
          Original curve, but possibly reversed
        """
        if hasattr(self, "_direction") and self._direction < 0:
            return curve[::-1]
        return curve

    def bounds(self, vertices):
        """
        Return the AABB of the current entity.

        Parameters
        -----------
        vertices : (n, dimension) float
          Vertices in space

        Returns
        -----------
        bounds : (2, dimension) float
          Coordinates of AABB, in (min, max) form
        """
        bounds = np.array([vertices[self.points].min(axis=0), vertices[self.points].max(axis=0)])
        return bounds

    def length(self, vertices):
        """
        Return the total length of the entity.

        Parameters
        --------------
        vertices : (n, dimension) float
          Vertices in space

        Returns
        ---------
        length : float
          Total length of entity
        """
        diff = np.diff(self.discrete(vertices), axis=0) ** 2
        length = (np.dot(diff, [1] * vertices.shape[1]) ** 0.5).sum()
        return length

    def explode(self):
        """
        Split the entity into multiple entities.

        Returns
        ------------
        explode : list of Entity
          Current entity split into multiple entities.
        """
        return [self.copy()]

    def copy(self):
        """
        Return a copy of the current entity.

        Returns
        ------------
        copied : Entity
          Copy of current entity
        """
        copied = deepcopy(self)
        # only copy metadata if set
        if hasattr(self, "_metadata"):
            copied._metadata = deepcopy(self._metadata)
            # check for very annoying subtle copy failures
            assert id(copied._metadata) != gs.UID(self._metadata)
        assert id(copied.points) != gs.UID(self.points)
        return copied

    def __hash__(self):
        """
        Return a hash that represents the current entity.

        Returns
        ----------
        hashed : int
            Hash of current class name, points, and closed
        """
        return hash(self._bytes())

    def _bytes(self):
        """
        Get hashable bytes that define the current entity.

        Returns
        ------------
        data : bytes
          Hashable data defining the current entity
        """
        # give consistent ordering of points for hash
        if self.points[0] > self.points[-1]:
            return self.__class__.__name__.encode("utf-8") + self.points.tobytes()
        else:
            return self.__class__.__name__.encode("utf-8") + self.points[::-1].tobytes()


class Text(Entity):
    """
    Text to annotate a 2D or 3D path.
    """

    def __init__(
        self, origin, text, height=None, vector=None, normal=None, align=None, layer=None, color=None, metadata=None
    ):
        """
        An entity for text labels.

        Parameters
        --------------
        origin : int
          Index of a single vertex for text origin
        text : str
          The text to label
        height : float or None
          The height of text
        vector : int or None
          An vertex index for which direction text
          is written along unitized: vector - origin
        normal : int or None
          A vertex index for the plane normal:
          vector is along unitized: normal - origin
        align : (2,) str or None
          Where to draw from for [horizontal, vertical]:
              'center', 'left', 'right'
        """
        # where is text placed
        self.origin = origin
        # what direction is the text pointing
        self.vector = vector
        # what is the normal of the text plane
        self.normal = normal
        # how high is the text entity
        self.height = height
        # what layer is the entity on
        if layer is not None:
            self.layer = layer

        if metadata is not None:
            self.metadata.update(metadata)

        # what color is the entity
        self.color = color

        # None or (2,) str
        if align is None:
            # if not set make everything centered
            align = ["center", "center"]
        elif util.is_string(align):
            # if only one is passed set for both
            # horizontal and vertical
            align = [align, align]
        elif len(align) != 2:
            # otherwise raise rror
            raise ValueError("align must be (2,) str")

        self.align = align

        # make sure text is a string
        if hasattr(text, "decode"):
            self.text = text.decode("utf-8")
        else:
            self.text = str(text)

    @property
    def origin(self):
        """
        The origin point of the text.

        Returns
        -----------
        origin : int
          Index of vertices
        """
        return self.points[0]

    @origin.setter
    def origin(self, value):
        value = int(value)
        if not hasattr(self, "points") or np.ptp(self.points) == 0:
            self.points = np.ones(3, dtype=np.int64) * value
        else:
            self.points[0] = value

    @property
    def vector(self):
        """
        A point representing the text direction
        along the vector: vertices[vector] - vertices[origin]

        Returns
        ----------
        vector : int
          Index of vertex
        """
        return self.points[1]

    @vector.setter
    def vector(self, value):
        if value is None:
            return
        self.points[1] = int(value)

    @property
    def normal(self):
        """
        A point representing the plane normal along the
        vector: vertices[normal] - vertices[origin]

        Returns
        ------------
        normal : int
          Index of vertex
        """
        return self.points[2]

    @normal.setter
    def normal(self, value):
        if value is None:
            return
        self.points[2] = int(value)

    def plot(self, vertices, show=False):
        """
        Plot the text using matplotlib.

        Parameters
        --------------
        vertices : (n, 2) float
          Vertices in space
        show : bool
          If True, call plt.show()
        """
        if vertices.shape[1] != 2:
            raise ValueError("only for 2D points!")

        import matplotlib.pyplot as plt

        # get rotation angle in degrees
        angle = np.degrees(self.angle(vertices))

        # TODO: handle text size better
        plt.text(*vertices[self.origin], s=self.text, rotation=angle, ha=self.align[0], va=self.align[1], size=18)

        if show:
            plt.show()

    def angle(self, vertices):
        """
        If Text is 2D, get the rotation angle in radians.

        Parameters
        -----------
        vertices : (n, 2) float
          Vertices in space referenced by self.points

        Returns
        ---------
        angle : float
          Rotation angle in radians
        """

        if vertices.shape[1] != 2:
            raise ValueError("angle only valid for 2D points!")

        # get the vector from origin
        direction = vertices[self.vector] - vertices[self.origin]
        # get the rotation angle in radians
        angle = np.arctan2(*direction[::-1])

        return angle

    def length(self, vertices):
        return 0.0

    def discrete(self, *args, **kwargs):
        return np.array([])

    @property
    def closed(self):
        return False

    @property
    def is_valid(self):
        return True

    @property
    def nodes(self):
        return np.array([])

    @property
    def end_points(self):
        return np.array([])

    def _bytes(self):
        data = b"".join([b"Text", self.points.tobytes(), self.text.encode("utf-8")])
        return data


class Line(Entity):
    """
    A line or poly-line entity
    """

    def discrete(self, vertices, scale=1.0):
        """
        Discretize into a world- space path.

        Parameters
        ------------
        vertices: (n, dimension) float
          Points in space
        scale : float
          Size of overall scene for numerical comparisons

        Returns
        -------------
        discrete: (m, dimension) float
          Path in space composed of line segments
        """
        return self._orient(vertices[self.points])

    @property
    def is_valid(self):
        """
        Is the current entity valid.

        Returns
        -----------
        valid : bool
          Is the current entity well formed
        """
        valid = np.any((self.points - self.points[0]) != 0)
        return valid

    def explode(self):
        """
        If the current Line entity consists of multiple line
        break it up into n Line entities.

        Returns
        ----------
        exploded: (n,) Line entities
        """
        # copy over the current layer
        layer = self.layer
        points = np.column_stack((self.points, self.points)).ravel()[1:-1].reshape((-1, 2))
        exploded = [Line(i, layer=layer) for i in points]
        return exploded

    def _bytes(self):
        # give consistent ordering of points for hash
        if self.points[0] > self.points[-1]:
            return b"Line" + self.points.tobytes()
        else:
            return b"Line" + self.points[::-1].tobytes()


class Arc(Entity):

    @property
    def closed(self):
        """
        A boolean flag for whether the arc is closed (a circle) or not.

        Returns
        ----------
        closed : bool
          If set True, Arc will be a closed circle
        """
        return getattr(self, "_closed", False)

    @closed.setter
    def closed(self, value):
        """
        Set the Arc to be closed or not, without
        changing the control points

        Parameters
        ------------
        value : bool
          Should this Arc be a closed circle or not
        """
        self._closed = bool(value)

    @property
    def is_valid(self):
        """
        Is the current Arc entity valid.

        Returns
        -----------
        valid : bool
          Does the current Arc have exactly 3 control points
        """
        return len(np.unique(self.points)) == 3

    def _bytes(self):
        # give consistent ordering of points for hash
        order = int(self.points[0] > self.points[-1]) * 2 - 1
        return b"Arc" + bytes(self.closed) + self.points[::order].tobytes()

    def length(self, vertices):
        """
        Return the arc length of the 3-point arc.

        Parameter
        ----------
        vertices : (n, d) float
          Vertices for overall drawing.

        Returns
        -----------
        length : float
          Length of arc.
        """
        # find the actual radius and angle span
        if self.closed:
            # we don't need the angular span as
            # it's indicated as a closed circle
            fit = self.center(vertices, return_normal=False, return_angle=False)
            return np.pi * fit["radius"] * 4
        # get the angular span of the circular arc
        fit = self.center(vertices, return_normal=False, return_angle=True)
        return fit["span"] * fit["radius"] * 2

    def discrete(self, vertices, scale=1.0):
        """
        Discretize the arc entity into line sections.

        Parameters
        ------------
        vertices : (n, dimension) float
            Points in space
        scale : float
            Size of overall scene for numerical comparisons

        Returns
        -------------
        discrete : (m, dimension) float
          Path in space made up of line segments
        """

        return self._orient(discretize_arc(vertices[self.points], close=self.closed, scale=scale))

    def center(self, vertices, **kwargs):
        """
        Return the center information about the arc entity.

        Parameters
        -------------
        vertices : (n, dimension) float
          Vertices in space

        Returns
        -------------
        info : dict
          With keys: 'radius', 'center'
        """
        return arc_center(vertices[self.points], **kwargs)

    def bounds(self, vertices):
        """
        Return the AABB of the arc entity.

        Parameters
        -----------
        vertices: (n, dimension) float
          Vertices in space

        Returns
        -----------
        bounds : (2, dimension) float
          Coordinates of AABB in (min, max) form
        """
        if util.is_shape(vertices, (-1, 2)) and self.closed:
            # if we have a closed arc (a circle), we can return the actual bounds
            # this only works in two dimensions, otherwise this would return the
            # AABB of an sphere
            info = self.center(vertices, return_normal=False, return_angle=False)
            bounds = np.array([info["center"] - info["radius"], info["center"] + info["radius"]], dtype=np.float64)
        else:
            # since the AABB of a partial arc is hard, approximate
            # the bounds by just looking at the discrete values
            discrete = self.discrete(vertices)
            bounds = np.array([discrete.min(axis=0), discrete.max(axis=0)], dtype=np.float64)
        return bounds


class Curve(Entity):
    """
    The parent class for all wild curves in space.
    """

    @property
    def nodes(self):
        # a point midway through the curve
        mid = self.points[len(self.points) // 2]
        return [[self.points[0], mid], [mid, self.points[-1]]]


class Bezier(Curve):
    """
    An open or closed Bezier curve
    """

    def discrete(self, vertices, scale=1.0, count=None):
        """
        Discretize the Bezier curve.

        Parameters
        -------------
        vertices : (n, 2) or (n, 3) float
          Points in space
        scale : float
          Scale of overall drawings (for precision)
        count : int
          Number of segments to return

        Returns
        -------------
        discrete : (m, 2) or (m, 3) float
          Curve as line segments
        """
        return self._orient(discretize_bezier(vertices[self.points], count=count, scale=scale))


class BSpline(Curve):
    """
    An open or closed B- Spline.
    """

    def __init__(self, points, knots, layer=None, metadata=None, color=None, **kwargs):
        self.points = np.asanyarray(points, dtype=np.int64)
        self.knots = np.asanyarray(knots, dtype=np.float64)
        if layer is not None:
            self.layer = layer
        if metadata is not None:
            self.metadata.update(metadata)
        self._cache = {}
        self.kwargs = kwargs
        self.color = color

    def discrete(self, vertices, count=None, scale=1.0):
        """
        Discretize the B-Spline curve.

        Parameters
        -------------
        vertices : (n, 2) or (n, 3) float
          Points in space
        scale : float
          Scale of overall drawings (for precision)
        count : int
          Number of segments to return

        Returns
        -------------
        discrete : (m, 2) or (m, 3) float
          Curve as line segments
        """
        discrete = discretize_bspline(control=vertices[self.points], knots=self.knots, count=count, scale=scale)
        return self._orient(discrete)

    def _bytes(self):
        # give consistent ordering of points for hash
        if self.points[0] > self.points[-1]:
            return b"BSpline" + self.knots.tobytes() + self.points.tobytes()
        else:
            return b"BSpline" + self.knots[::-1].tobytes() + self.points[::-1].tobytes()

    def to_dict(self):
        """
        Returns a dictionary with all of the information
        about the entity.
        """
        return {
            "type": self.__class__.__name__,
            "points": self.points.tolist(),
            "knots": self.knots.tolist(),
            "closed": self.closed,
        }
