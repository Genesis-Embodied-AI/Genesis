"""
raster.py
------------

Turn 2D vector paths into raster images using `pillow`
"""

import numpy as np

try:
    # keep pillow as a soft dependency
    from PIL import Image, ImageChops, ImageDraw
except BaseException as E:
    from .. import exceptions

    # re-raise the useful exception when called
    _handle = exceptions.ExceptionWrapper(E)
    Image = _handle
    ImageDraw = _handle
    ImageChops = _handle


def rasterize(path, pitch=None, origin=None, resolution=None, fill=True, width=None):
    """
    Rasterize a Path2D object into a boolean image ("mode 1").

    Parameters
    ------------
    path : Path2D
      Original geometry
    pitch : float or (2,) float
      Length(s) in model space of pixel edges
    origin : (2,) float
      Origin position in model space
    resolution : (2,) int
      Resolution in pixel space
    fill :  bool
      If True will return closed regions as filled
    width : int
      If not None will draw outline this wide in pixels

    Returns
    ------------
    raster : PIL.Image
      Rasterized version of input as `mode 1` image
    """

    if pitch is None:
        pitch = path.extents.max() / 2048

    if origin is None:
        origin = path.bounds[0] - (pitch * 2.0)

    # check inputs
    pitch = np.asanyarray(pitch, dtype=np.float64)
    origin = np.asanyarray(origin, dtype=np.float64)

    # if resolution is None make it larger than path
    if resolution is None:
        span = np.ptp(np.vstack((path.bounds, origin)), axis=0)
        resolution = np.ceil(span / pitch) + 2
    # get resolution as a (2,) int tuple
    resolution = np.asanyarray(resolution, dtype=np.int64)
    resolution = tuple(resolution.tolist())

    # convert all discrete paths to pixel space
    discrete = [((i - origin) / pitch).round().astype(np.int64) for i in path.discrete]

    # the path indexes that are exteriors
    # needed to know what to fill/empty but expensive
    roots = path.root
    enclosure = path.enclosure_directed

    # draw the exteriors
    result = Image.new(mode="1", size=resolution)
    draw = ImageDraw.Draw(result)

    # if a width is specified draw the outline
    if width is not None:
        width = int(width)
        for coords in discrete:
            draw.line(coords.flatten().tolist(), fill=1, width=width)
        # if we are not sampling the polygon exit
        if not fill:
            return result

    # roots are ordered by degree
    # so we draw the outermost one first
    # and then go in as we progress
    for root in roots:
        # draw the exterior
        draw.polygon(discrete[root].flatten().tolist(), fill=1)
        # draw the interior children
        for child in enclosure[root]:
            draw.polygon(discrete[child].flatten().tolist(), fill=0)

    return result
