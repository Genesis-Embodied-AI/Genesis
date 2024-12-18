import os

from .dxf import _dxf_loaders
from .svg_io import svg_to_path
from ..path import Path

from . import misc
from ... import util


def load_path(file_obj, file_type=None, **kwargs):
    """
    Load a file to a Path file_object.

    Parameters
    -----------
    file_obj : One of the following:
         - Path, Path2D, or Path3D file_objects
         - open file file_object (dxf or svg)
         - file name (dxf or svg)
         - shapely.geometry.Polygon
         - shapely.geometry.MultiLineString
         - dict with kwargs for Path constructor
         - (n,2,(2|3)) float, line segments
    file_type : str
        Type of file is required if file
        file_object passed.

    Returns
    ---------
    path : Path, Path2D, Path3D file_object
        Data as a native trimesh Path file_object
    """
    # avoid a circular import
    from ...exchange.load import load_kwargs

    # record how long we took
    tic = util.now()

    if isinstance(file_obj, Path):
        # we have been passed a Path file_object so
        # do nothing and return the passed file_object
        return file_obj
    elif util.is_file(file_obj):
        # for open file file_objects use loaders
        kwargs.update(path_loaders[file_type](file_obj, file_type=file_type))
    elif util.is_string(file_obj):
        # strings passed are evaluated as file file_objects
        with open(file_obj, "rb") as f:
            # get the file type from the extension
            file_type = os.path.splitext(file_obj)[-1][1:].lower()
            # call the loader
            kwargs.update(path_loaders[file_type](f, file_type=file_type))
    elif util.is_instance_named(file_obj, ["Polygon", "MultiPolygon"]):
        # convert from shapely polygons to Path2D
        kwargs.update(misc.polygon_to_path(file_obj))
    elif util.is_instance_named(file_obj, "MultiLineString"):
        # convert from shapely LineStrings to Path2D
        kwargs.update(misc.linestrings_to_path(file_obj))
    elif isinstance(file_obj, dict):
        # load as kwargs
        return load_kwargs(file_obj)
    elif util.is_sequence(file_obj):
        # load as lines in space
        kwargs.update(misc.lines_to_path(file_obj))
    else:
        raise ValueError("Not a supported object type!")

    result = load_kwargs(kwargs)
    util.log.debug("loaded {} in {:0.4f}s".format(str(result), util.now() - tic))

    return result


def path_formats():
    """
    Get a list of supported path formats.

    Returns
    ------------
    loaders : list of str
        Extensions of loadable formats, ie:
        ['svg', 'dxf']
    """
    return set(path_loaders.keys())


path_loaders = {"svg": svg_to_path}
path_loaders.update(_dxf_loaders)
