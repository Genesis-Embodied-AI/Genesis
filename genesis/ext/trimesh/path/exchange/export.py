import os
from ... import util

from . import dxf
from . import svg_io


def export_path(path, file_type=None, file_obj=None, **kwargs):
    """
    Export a Path object to a file- like object, or to a filename

    Parameters
    ---------
    file_obj:  None, str, or file object
      A filename string or a file-like object
    file_type: None or str
      File type, e.g.: 'svg', 'dxf'
    kwargs : passed to loader

    Returns
    ---------
    exported : str or bytes
      Data exported
    """
    # if file object is a string it is probably a file path
    # so we can split the extension to set the file type
    if util.is_string(file_obj):
        file_type = util.split_extension(file_obj)

    # run the export
    export = _path_exporters[file_type](path, **kwargs)
    # if we've been passed files write the data
    _write_export(export=export, file_obj=file_obj)

    return export


def export_dict(path):
    """
    Export a path as a dict of kwargs for the Path constructor.
    """
    export_entities = [e.to_dict() for e in path.entities]
    export_object = {"entities": export_entities, "vertices": path.vertices.tolist()}
    return export_object


def _write_export(export, file_obj=None):
    """
    Write a string to a file.
    If file_obj isn't specified, return the string

    Parameters
    ---------
    export: a string of the export data
    file_obj: a file-like object or a filename
    """

    if file_obj is None:
        return export

    if hasattr(file_obj, "write"):
        out_file = file_obj
    else:
        # expand user and relative paths
        file_path = os.path.abspath(os.path.expanduser(file_obj))
        out_file = open(file_path, "wb")
    try:
        out_file.write(export)
    except TypeError:
        out_file.write(export.encode("utf-8"))

    out_file.close()

    return export


_path_exporters = {"dxf": dxf.export_dxf, "svg": svg_io.export_svg, "dict": export_dict}
