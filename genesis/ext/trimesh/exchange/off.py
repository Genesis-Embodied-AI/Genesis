import re
import numpy as np

from .. import util
from ..geometry import triangulate_quads


def load_off(file_obj, **kwargs):
    """
    Load an OFF file into the kwargs for a Trimesh constructor.

    Parameters
    ----------
    file_obj : file object
      Contains an OFF file

    Returns
    ----------
    loaded : dict
      kwargs for Trimesh constructor
    """
    text = file_obj.read()
    # will magically survive weird encoding sometimes
    # comment strip will handle all cases of commenting
    text = util.comment_strip(util.decode_text(text)).strip()

    # split the first key
    _, header, raw = re.split("(COFF|OFF)", text, maxsplit=1)
    if header.upper() not in ["OFF", "COFF"]:
        raise NameError("Not an OFF file! Header was: `{}`".format(header))

    # split into lines and remove whitespace
    splits = [i.strip() for i in str.splitlines(str(raw))]
    # remove empty lines
    splits = [i for i in splits if len(i) > 0]

    # the first non-comment line should be the counts
    header = np.array(splits[0].split(), dtype=np.int64)
    vertex_count, face_count = header[:2]

    vertices = np.array([i.split()[:3] for i in splits[1 : vertex_count + 1]], dtype=np.float64)

    # will fail if incorrect number of vertices loaded
    vertices = vertices.reshape((vertex_count, 3))

    # get lines with face data
    faces = [i.split() for i in splits[vertex_count + 1 : vertex_count + face_count + 1]]
    # the first value is count
    faces = [line[1 : int(line[0]) + 1] for line in faces]

    faces = triangulate_quads(faces)
    # save data as kwargs for a trimesh.Trimesh
    kwargs = {"vertices": vertices, "faces": faces}

    return kwargs


def export_off(mesh, digits=10):
    """
    Export a mesh as an OFF file, a simple text format

    Parameters
    -----------
    mesh : trimesh.Trimesh
      Geometry to export
    digits : int
      Number of digits to include on floats

    Returns
    -----------
    export : str
      OFF format output
    """
    # make sure specified digits is an int
    digits = int(digits)
    # prepend a 3 (face count) to each face
    faces_stacked = np.column_stack((np.ones(len(mesh.faces)) * 3, mesh.faces)).astype(np.int64)
    export = "OFF\n"
    # the header is vertex count, face count, another number
    export += str(len(mesh.vertices)) + " " + str(len(mesh.faces)) + " 0\n"
    export += util.array_to_string(mesh.vertices, col_delim=" ", row_delim="\n", digits=digits) + "\n"
    export += util.array_to_string(faces_stacked, col_delim=" ", row_delim="\n")
    return export


_off_loaders = {"off": load_off}
_off_exporters = {"off": export_off}
