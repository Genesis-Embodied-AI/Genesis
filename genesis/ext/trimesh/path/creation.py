import numpy as np

from . import arc
from .entities import Line, Arc
from ..geometry import plane_transform

from .. import util
from .. import transformations


def circle_pattern(pattern_radius, circle_radius, count, center=None, angle=None, **kwargs):
    """
    Create a Path2D representing a circle pattern.

    Parameters
    ------------
    pattern_radius : float
      Radius of circle centers
    circle_radius : float
      The radius of each circle
    count : int
      Number of circles in the pattern
    center : (2,) float
      Center of pattern
    angle :  float
      If defined pattern will span this angle
      If None, pattern will be evenly spaced

    Returns
    -------------
    pattern : trimesh.path.Path2D
      Path containing circular pattern
    """
    from .path import Path2D

    if angle is None:
        angles = np.linspace(0.0, np.pi * 2.0, count + 1)[:-1]
    elif isinstance(angle, float) or isinstance(angle, int):
        angles = np.linspace(0.0, angle, count)
    else:
        raise ValueError("angle must be float or int!")

    if center is None:
        center = [0.0, 0.0]

    # centers of circles
    centers = np.column_stack((np.cos(angles), np.sin(angles))) * pattern_radius

    vert = []
    ents = []
    for circle_center in centers:
        # (3,3) center points of arc
        three = arc.to_threepoint(angles=[0, np.pi], center=circle_center, radius=circle_radius)
        # add a single circle entity
        ents.append(Arc(points=np.arange(3) + len(vert), closed=True))
        # keep flat array by extend instead of append
        vert.extend(three)

    # translate vertices to pattern center
    vert = np.array(vert) + center
    pattern = Path2D(entities=ents, vertices=vert, **kwargs)
    return pattern


def circle(radius, center=None, **kwargs):
    """
    Create a Path2D containing circle with the specified
    radius.

    Parameters
    --------------
    radius : float
      The radius of the circle
    center : None or (2,) float
      Center of the circle, origin by default
    ** kwargs : dict
      Passed to trimesh.path.Path2D constructor

    Returns
    -------------
    circle : Path2D
      Path containing specified circle
    """
    from .path import Path2D

    if center is None:
        center = [0.0, 0.0]
    else:
        center = np.asanyarray(center, dtype=np.float64)
    # make sure radius is a float
    radius = float(radius)

    # (3, 2) float, points on arc
    three = arc.to_threepoint(angles=[0, np.pi], center=center, radius=radius)
    # generate the path object
    result = Path2D(entities=[Arc(points=np.arange(3), closed=True)], vertices=three, **kwargs)

    return result


def rectangle(bounds, **kwargs):
    """
    Create a Path2D containing a single or multiple rectangles
    with the specified bounds.

    Parameters
    --------------
    bounds : (2, 2) float, or (m, 2, 2) float
      Minimum XY, Maximum XY

    Returns
    -------------
    rect : Path2D
      Path containing specified rectangles
    """
    from .path import Path2D

    # data should be float
    bounds = np.asanyarray(bounds, dtype=np.float64)

    # bounds are extents, re- shape to origin- centered rectangle
    if bounds.shape == (2,):
        half = np.abs(bounds) / 2.0
        bounds = np.array([-half, half])

    # should have one bounds or multiple bounds
    if not (util.is_shape(bounds, (2, 2)) or util.is_shape(bounds, (-1, 2, 2))):
        raise ValueError("bounds must be (m, 2, 2) or (2, 2)")

    # hold Line objects
    lines = []
    # hold (n, 2) cartesian points
    vertices = []

    # loop through each rectangle
    for lower, upper in bounds.reshape((-1, 2, 2)):
        lines.append(Line((np.arange(5) % 4) + len(vertices)))
        vertices.extend([lower, [upper[0], lower[1]], upper, [lower[0], upper[1]]])

    # create the Path2D with specified rectangles
    rect = Path2D(entities=lines, vertices=vertices, **kwargs)

    return rect


def box_outline(extents=None, transform=None, **kwargs):
    """
    Return a cuboid.

    Parameters
    ------------
    extents : float, or (3,) float
      Edge lengths
    transform: (4, 4) float
      Transformation matrix
    **kwargs:
        passed to Trimesh to create box

    Returns
    ------------
    geometry : trimesh.Path3D
      Path outline of a cuboid geometry
    """
    from .exchange.load import load_path

    # create vertices for the box
    vertices = [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1]
    vertices = np.array(vertices, order="C", dtype=np.float64).reshape((-1, 3))
    vertices -= 0.5

    # resize the vertices based on passed size
    if extents is not None:
        extents = np.asanyarray(extents, dtype=np.float64)
        if extents.shape != (3,):
            raise ValueError("Extents must be (3,)!")
        vertices *= extents

    # apply transform if passed
    if transform is not None:
        vertices = transformations.transform_points(vertices, transform)

    # vertex indices
    indices = [0, 1, 3, 2, 0, 4, 5, 7, 6, 4, 0, 2, 6, 7, 3, 1, 5]
    outline = load_path(vertices[indices])

    return outline


def grid(side, count=5, transform=None, plane_origin=None, plane_normal=None, include_circle=True, sections_circle=32):
    """
    Create a Path3D for a grid visualization of a plane.

    Parameters
    -----------
    side : float
      Length of half of a grid side
    count : int
      Number of grid lines per grid half
    transform : None or (4, 4) float
      Transformation matrix to move grid location.
      Takes precedence over plane_origin if both are passed.
    plane_origin : None or (3,) float
      Plane origin
    plane_normal : None or (3,) float
      Unit normal vector
    include_circle : bool
      Include a circular pattern inside the grid
    sections_circle : int
      How many sections should the smallest circle have

    Returns
    ----------
    grid : trimesh.path.Path3D
      Path containing grid plane visualization
    """
    from .path import Path3D

    # change full side length to half-side
    side = float(side)
    # make sure count is an integer
    count = int(count)
    # get a spaced sequence of radius
    radii = np.linspace(0.0, side, count + 1)[1:]
    # what's the maximum radius
    rmax = radii[-1]

    # keep a count of the current vertex count
    current = 0
    # collect vertices and entities
    vertices = []
    entities = []
    for r in radii:
        if include_circle:
            # scale the section count by radius
            circle_res = int((r / radii[0]) * sections_circle)
            # generate a circule pattern
            theta = np.linspace(0.0, np.pi * 2, circle_res)
            circle = np.column_stack((np.cos(theta), np.sin(theta))) * r
            # append the circle pattern
            vertices.append(circle)
            entities.append(Line(points=np.arange(len(circle)) + current))
            # keep the vertex count correct
            current += len(circle)
        # generate a series of grid lines
        vertices.append(
            [[-rmax, r], [rmax, r], [-rmax, -r], [rmax, -r], [r, -rmax], [r, rmax], [-r, -rmax], [-r, rmax]]
        )
        # append an entity per grid line
        for i in [0, 2, 4, 6]:
            entities.append(Line(points=np.arange(2) + current + i))
        current += len(vertices[-1])

    # add the middle lines which were skipped
    vertices.append([[0, rmax], [0, -rmax], [-rmax, 0], [rmax, 0]])
    entities.append(Line(points=np.arange(2) + current))
    entities.append(Line(points=np.arange(2) + current + 2))
    # stack vertices into clean (n, 3) float
    vertices = np.vstack(vertices)

    # if plane was passed instead of transform create the matrix here
    if transform is None and plane_origin is not None and plane_normal is not None:
        transform = np.linalg.inv(plane_transform(origin=plane_origin, normal=plane_normal))

    # stack vertices to 3D
    vertices = np.column_stack((vertices, np.zeros(len(vertices))))
    # apply transform if passed
    if transform is not None:
        vertices = transformations.transform_points(vertices, matrix=transform)
    # combine result into a Path3D object
    grid_path = Path3D(entities=entities, vertices=vertices)
    return grid_path
