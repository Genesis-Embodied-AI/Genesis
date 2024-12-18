import numpy as np

from .. import util
from ..points import PointCloud


def load_xyz(file_obj, delimiter=None, **kwargs):
    """
    Load an XYZ file into a PointCloud.

    Parameters
    ------------
    file_obj : an open file-like object
      Source data, ASCII XYZ
    delimiter : None or string
      Characters used to separate the columns of the file
      If not passed will use whitespace or commas

    Returns
    ----------
    kwargs : dict
      Data which can be passed to PointCloud constructor
    """
    # read the whole file into memory as a string
    raw = util.decode_text(file_obj.read()).strip()
    # get the first line to look at
    first = raw[: raw.find("\n")].strip()
    # guess the column count by looking at the first line
    columns = len(first.split())
    if columns < 3:
        raise ValueError("not enough columns in xyz file!")

    if delimiter is None and "," in first:
        # if no delimiter passed and file has commas
        delimiter = ","
    if delimiter is not None:
        # replace delimiter with whitespace so split works
        raw = raw.replace(delimiter, " ")

    # use string splitting to get array
    array = np.array(raw.split(), dtype=np.float64)
    # reshape to column count
    # if file has different numbers of values
    # per row this will fail as it should
    data = array.reshape((-1, columns))

    # start with no colors
    colors = None
    # vertices are the first three columns
    vertices = data[:, :3]
    if columns == 6:
        # RGB colors
        colors = np.array(data[:, 3:], dtype=np.uint8)
        colors = np.concatenate((colors, np.ones((len(data), 1), dtype=np.uint8) * 255), axis=1)
    elif columns >= 7:
        # extract RGBA colors
        colors = np.array(data[:, 3:8], dtype=np.uint8)
    # add extracted colors and vertices to kwargs
    kwargs.update({"vertices": vertices, "colors": colors})

    return kwargs


def export_xyz(cloud, write_colors=True, delimiter=None):
    """
    Export a PointCloud object to an XYZ format string.

    Parameters
    -------------
    cloud : trimesh.PointCloud
      Geometry in space
    write_colors : bool
      Write colors or not
    delimiter : None or str
      What to separate columns with

    Returns
    --------------
    export : str
      Pointcloud in XYZ format
    """
    if not isinstance(cloud, PointCloud):
        raise ValueError("object must be PointCloud")

    # compile data into a blob
    data = cloud.vertices
    if write_colors and hasattr(cloud, "colors") and cloud.colors is not None:
        # stack colors and  vertices
        data = np.hstack((data, cloud.colors))

    # if delimiter not passed use whitepace
    if delimiter is None:
        delimiter = " "
    # stack blob into XYZ format
    export = util.array_to_string(data, col_delim=delimiter)

    return export


_xyz_loaders = {"xyz": load_xyz}
_xyz_exporters = {"xyz": export_xyz}
