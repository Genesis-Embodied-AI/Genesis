from .. import util

import numpy as np


class HeaderError(Exception):
    # the exception raised if an STL file object doesn't match its header
    pass


# define a numpy datatype for the data section of a binary STL file
# everything in STL is always Little Endian
# this works natively on Little Endian systems, but blows up on Big Endians
# so we always specify byteorder
_stl_dtype = np.dtype([("normals", "<f4", (3)), ("vertices", "<f4", (3, 3)), ("attributes", "<u2")])
# define a numpy datatype for the header of a binary STL file
_stl_dtype_header = np.dtype([("header", np.void, 80), ("face_count", "<u4")])


def load_stl(file_obj, **kwargs):
    """
    Load an STL file from a file object.

    Parameters
    ----------
    file_obj : open file-like object
      Containing STL data

    Returns
    ----------
    loaded : dict
      kwargs for a Trimesh constructor with keys:
      vertices:     (n,3) float, vertices
      faces:        (m,3) int, indexes of vertices
      face_normals: (m,3) float, normal vector of each face
    """
    # save start of file obj
    file_pos = file_obj.tell()
    try:
        # check the file for a header which matches the file length
        # if that is true, it is almost certainly a binary STL file
        # if the header doesn't match the file length a HeaderError will be
        # raised
        return load_stl_binary(file_obj)
    except HeaderError:
        # move the file back to where it was initially
        file_obj.seek(file_pos)
        # try to load the file as an ASCII STL
        # if the header doesn't match the file length
        # HeaderError will be raised
        return load_stl_ascii(file_obj)


def load_stl_binary(file_obj):
    """
    Load a binary STL file from a file object.

    Parameters
    ----------
    file_obj : open file- like object
      Containing STL data

    Returns
    ----------
    loaded: kwargs for a Trimesh constructor with keys:
              vertices:     (n,3) float, vertices
              faces:        (m,3) int, indexes of vertices
              face_normals: (m,3) float, normal vector of each face
    """
    # the header is always 84 bytes long, we just reference the dtype.itemsize
    # to be explicit about where that magical number comes from
    header_length = _stl_dtype_header.itemsize
    header_data = file_obj.read(header_length)
    if len(header_data) < header_length:
        raise HeaderError("Binary STL shorter than a fixed header!")

    try:
        header = np.frombuffer(header_data, dtype=_stl_dtype_header)
    except BaseException:
        raise HeaderError("Binary header incorrect type")

    try:
        # save the header block as a string
        # there could be any garbage in there so wrap in try
        metadata = {"header": util.decode_text(bytes(header["header"][0])).strip()}
    except BaseException:
        metadata = {}

    # now we check the length from the header versus the length of the file
    # data_start should always be position 84, but hard coding that felt ugly
    data_start = file_obj.tell()
    # this seeks to the end of the file
    # position 0, relative to the end of the file 'whence=2'
    file_obj.seek(0, 2)
    # we save the location of the end of the file and seek back to where we
    # started from
    data_end = file_obj.tell()
    file_obj.seek(data_start)

    # the binary format has a rigidly defined structure, and if the length
    # of the file doesn't match the header, the loaded version is almost
    # certainly going to be garbage.
    len_data = data_end - data_start
    len_expected = header["face_count"] * _stl_dtype.itemsize

    # this check is to see if this really is a binary STL file.
    # if we don't do this and try to load a file that isn't structured properly
    # we will be producing garbage or crashing hard
    # so it's much better to raise an exception here.
    if len_data != len_expected:
        raise HeaderError("Binary STL has incorrect length in header: {} vs {}".format(len_data, len_expected))

    blob = np.frombuffer(file_obj.read(), dtype=_stl_dtype)

    # return empty geometry if there are no vertices
    if not len(blob["vertices"]):
        return {"geometry": {}}

    # all of our vertices will be loaded in order
    # so faces are just sequential indices reshaped.
    faces = np.arange(header["face_count"][0] * 3).reshape((-1, 3))

    # there are two bytes per triangle saved for anything
    # which is sometimes used for face color
    result = {
        "vertices": blob["vertices"].reshape((-1, 3)),
        "face_normals": blob["normals"].reshape((-1, 3)),
        "faces": faces,
        "face_attributes": {"stl": blob["attributes"]},
        "metadata": metadata,
    }
    return result


def load_stl_ascii(file_obj):
    """
    Load an ASCII STL file from a file object.

    Parameters
    ----------
    file_obj : open file- like object
      Containing input data

    Returns
    ----------
    loaded : dict
      kwargs for a Trimesh constructor with keys:
      vertices:     (n,3) float, vertices
      faces:        (m,3) int, indexes of vertices
      face_normals: (m,3) float, normal vector of each face
    """

    # read all text into one string
    raw = util.decode_text(file_obj.read()).strip().lower()

    # split into solid body
    kwargs = {}
    solids = raw.split("endsolid")
    for solid in solids:
        # get just the vertices
        vertex_text = solid.split("vertex")
        vertices = np.fromstring(
            " ".join(line[: line.find("\n")] for line in vertex_text[1:]), sep=" ", dtype=np.float64
        )
        if len(vertices) < 3:
            continue
        if len(vertices) % 3 != 0:
            raise ValueError("incorrect number of vertices")

        # reshape vertices to final 3D shape
        vertices = vertices.reshape((-1, 3))
        faces = np.arange(len(vertices)).reshape((-1, 3))

        # try to extract the face normals the same way
        face_normals = None
        try:
            normal_text = solid.split("normal")
            normals = np.fromstring(
                " ".join(line[: line.find("\n")] for line in normal_text[1:]), sep=" ", dtype=np.float64
            )
            if len(normals) == len(vertices):
                face_normals = normals.reshape((-1, 3))
        except BaseException:
            util.log.warning("failed to extract face_normals", exc_info=True)

        try:
            # try to extract the name from the header
            text = vertex_text[0]
            # find the keyword for the header format:
            #    `solid {name}`
            index = text.find("solid")
            if index < 0:
                raise ValueError("missing `solid` keyword")
            # clip to the first newline after the `solid`
            name = text[index + 6 :].strip().split("\n", 1)[0].strip()
        except BaseException:
            # will be filled in by unique_name
            name = None

        # make sure geometry has a unique name
        name = util.unique_name(name, kwargs)
        kwargs[name] = {"vertices": vertices.reshape((-1, 3)), "face_normals": face_normals, "faces": faces}

    if len(kwargs) == 1:
        return next(iter(kwargs.values()))

    return {"geometry": kwargs}


def export_stl(mesh):
    """
    Convert a Trimesh object into a binary STL file.

    Parameters
    ---------
    mesh: Trimesh object

    Returns
    ---------
    export: bytes, representing mesh in binary STL form
    """
    header = np.zeros(1, dtype=_stl_dtype_header)
    if hasattr(mesh, "faces"):
        header["face_count"] = len(mesh.faces)
    export = header.tobytes()

    if hasattr(mesh, "faces"):
        packed = np.zeros(len(mesh.faces), dtype=_stl_dtype)
        packed["normals"] = mesh.face_normals
        packed["vertices"] = mesh.triangles
        export += packed.tobytes()

    return export


def export_stl_ascii(mesh):
    """
    Convert a Trimesh object into an ASCII STL file.

    Parameters
    ---------
    mesh : trimesh.Trimesh

    Returns
    ---------
    export : str
        Mesh represented as an ASCII STL file
    """

    # move all the data that's going into the STL file into one array
    blob = np.zeros((len(mesh.faces), 4, 3))
    blob[:, 0, :] = mesh.face_normals
    blob[:, 1:, :] = mesh.triangles

    # create a lengthy format string for the data section of the file
    format_string = "facet normal {} {} {}\nouter loop\n"
    format_string += "vertex {} {} {}\n" * 3
    format_string += "endloop\nendfacet\n"
    format_string *= len(mesh.faces)

    # concatenate the header, data, and footer
    export = "solid \n"
    export += format_string.format(*blob.reshape(-1))
    export += "endsolid"

    return export


_stl_loaders = {"stl": load_stl, "stl_ascii": load_stl}
