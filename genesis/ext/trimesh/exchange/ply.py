import numpy as np

from string import Template

import tempfile
import subprocess
import collections

from .. import util
from .. import visual
from .. import grouping
from .. import resources

from ..geometry import triangulate_quads
from ..constants import log

# from ply specification, and additional dtypes found in the wild
_dtypes = {
    "char": "i1",
    "uchar": "u1",
    "short": "i2",
    "ushort": "u2",
    "int": "i4",
    "int8": "i1",
    "int16": "i2",
    "int32": "i4",
    "int64": "i8",
    "uint": "u4",
    "uint8": "u1",
    "uint16": "u2",
    "uint32": "u4",
    "uint64": "u8",
    "float": "f4",
    "float16": "f2",
    "float32": "f4",
    "float64": "f8",
    "double": "f8",
}

# Inverse of the above dict, collisions on numpy type were removed
_inverse_dtypes = {
    "i1": "char",
    "u1": "uchar",
    "i2": "short",
    "u2": "ushort",
    "i4": "int",
    "i8": "int64",
    "u4": "uint",
    "u8": "uint64",
    "f4": "float",
    "f2": "float16",
    "f8": "double",
}


def _numpy_type_to_ply_type(_numpy_type):
    """
    Returns the closest ply equivalent of a numpy type

    Parameters
    ---------
    _numpy_type : a numpy datatype

    Returns
    ---------
    ply_type : string
    """
    return _inverse_dtypes[_numpy_type.str[1:]]


def load_ply(file_obj, resolver=None, fix_texture=True, prefer_color=None, *args, **kwargs):
    """
    Load a PLY file from an open file object.

    Parameters
    ---------
    file_obj : an open file- like object
      Source data, ASCII or binary PLY
    resolver : trimesh.visual.resolvers.Resolver
      Object which can resolve assets
    fix_texture : bool
      If True, will re- index vertices and faces
      so vertices with different UV coordinates
      are disconnected.
    prefer_color : None, 'vertex', or 'face'
      Which kind of color to prefer if both defined

    Returns
    ---------
    mesh_kwargs : dict
      Data which can be passed to
      Trimesh constructor, eg: a = Trimesh(**mesh_kwargs)
    """

    # OrderedDict which is populated from the header
    elements, is_ascii, image_name = _parse_header(file_obj)

    # functions will fill in elements from file_obj
    if is_ascii:
        _ply_ascii(elements, file_obj)
    else:
        _ply_binary(elements, file_obj)

    # try to load the referenced image
    image = None
    try:
        # soft dependency
        import PIL.Image

        # if an image name is passed try to load it
        if image_name is not None:
            data = resolver.get(image_name)
            image = PIL.Image.open(util.wrap_as_stream(data))
    except ImportError:
        log.debug("textures require `pip install pillow`")
    except BaseException:
        log.warning("unable to load image!", exc_info=True)

    # translate loaded PLY elements to kwargs
    kwargs = _elements_to_kwargs(image=image, elements=elements, fix_texture=fix_texture, prefer_color=prefer_color)

    return kwargs


def _add_attributes_to_dtype(dtype, attributes):
    """
    Parses attribute datatype to populate a numpy dtype list

    Parameters
    ----------
    dtype : list of numpy datatypes
      operated on in place
    attributes : dict
      contains all the attributes to parse

    Returns
    ----------
    dtype : list of numpy datatypes
    """
    for name, data in attributes.items():
        if data.ndim == 1:
            dtype.append((name, data.dtype))
        else:
            attribute_dtype = data.dtype if len(data.dtype) == 0 else data.dtype[0]
            dtype.append(("{}_count".format(name), "u1"))
            dtype.append((name, _numpy_type_to_ply_type(attribute_dtype), data.shape[1]))
    return dtype


def _add_attributes_to_header(header, attributes):
    """
    Parses attributes in to ply header entries

    Parameters
    ----------
    header : list of ply header entries
      operated on in place
    attributes : dict
      contains all the attributes to parse

    Returns
    ----------
    header : list
      Contains ply header entries
    """
    for name, data in attributes.items():
        if data.ndim == 1:
            header.append("property {} {}\n".format(_numpy_type_to_ply_type(data.dtype), name))
        else:
            header.append("property list uchar {} {}\n".format(_numpy_type_to_ply_type(data.dtype), name))
    return header


def _add_attributes_to_data_array(data_array, attributes):
    """
    Parses attribute data in to a custom array, assumes datatype has been defined
    appropriately

    Parameters
    ----------
    data_array : numpy array with custom datatype
      datatype reflects all the data to be stored for a given ply element
    attributes : dict
      contains all the attributes to parse

    Returns
    ----------
    data_array : numpy array with custom datatype
    """
    for name, data in attributes.items():
        if data.ndim > 1:
            data_array["{}_count".format(name)] = data.shape[1] * np.ones(data.shape[0])
        data_array[name] = data
    return data_array


def _assert_attributes_valid(attributes):
    """
    Asserts that a set of attributes is valid for PLY export.

    Parameters
    ----------
    attributes : dict
      Contains the attributes to validate

    Raises
    --------
    ValueError
      If passed attributes aren't valid.
    """
    for data in attributes.values():
        if data.ndim not in [1, 2]:
            raise ValueError("PLY attributes are limited to 1 or 2 dimensions")
        # Inelegant test for structured arrays, reference:
        # https://numpy.org/doc/stable/user/basics.rec.html
        if data.dtype.names is not None:
            raise ValueError("PLY attributes must be of a single datatype")


def export_ply(mesh, encoding="binary", vertex_normal=None, include_attributes=True):
    """
    Export a mesh in the PLY format.

    Parameters
    ----------
    mesh : trimesh.Trimesh
      Mesh to export.
    encoding : str
      PLY encoding: 'ascii' or 'binary_little_endian'
    vertex_normal : None or include vertex normals

    Returns
    ----------
    export : bytes of result
    """
    # evaluate input args
    # allow a shortcut for binary
    if encoding == "binary":
        encoding = "binary_little_endian"
    elif encoding not in ["binary_little_endian", "ascii"]:
        raise ValueError("encoding must be binary or ascii")
    # if vertex normals aren't specifically asked for
    # only export them if they are stored in cache
    if vertex_normal is None:
        vertex_normal = "vertex_normals" in mesh._cache

    # if we want to include mesh attributes in the export
    if include_attributes:
        if hasattr(mesh, "vertex_attributes"):
            # make sure to export texture coordinates as well
            if hasattr(mesh, "visual") and hasattr(mesh.visual, "uv"):
                mesh.vertex_attributes["s"] = mesh.visual.uv[:, 0]
                mesh.vertex_attributes["t"] = mesh.visual.uv[:, 1]
            _assert_attributes_valid(mesh.vertex_attributes)
        if hasattr(mesh, "face_attributes"):
            _assert_attributes_valid(mesh.face_attributes)

    # custom numpy dtypes for exporting
    dtype_face = [("count", "<u1"), ("index", "<i4", (3))]
    dtype_vertex = [("vertex", "<f4", (3))]
    # will be appended to main dtype if needed
    dtype_vertex_normal = ("normals", "<f4", (3))
    dtype_color = ("rgba", "<u1", (4))

    # get template strings in dict
    templates = resources.get("templates/ply.json", decode_json=True)
    # start collecting elements into a string for the header
    header = [templates["intro"]]

    # check if scene has geometry
    if hasattr(mesh, "vertices"):
        header.append(templates["vertex"])

        num_vertices = len(mesh.vertices)
        # if we're exporting vertex normals add them
        # to the header and dtype
        if vertex_normal:
            header.append(templates["vertex_normal"])
            dtype_vertex.append(dtype_vertex_normal)

        # if mesh has a vertex color add it to the header
        if mesh.visual.kind == "vertex":
            header.append(templates["color"])
            dtype_vertex.append(dtype_color)

        if include_attributes and hasattr(mesh, "vertex_attributes"):
            _add_attributes_to_header(header, mesh.vertex_attributes)
            _add_attributes_to_dtype(dtype_vertex, mesh.vertex_attributes)

        # create and populate the custom dtype for vertices
        vertex = np.zeros(num_vertices, dtype=dtype_vertex)
        vertex["vertex"] = mesh.vertices
        if vertex_normal:
            vertex["normals"] = mesh.vertex_normals
        if mesh.visual.kind == "vertex" and len(mesh.visual.vertex_colors):
            vertex["rgba"] = mesh.visual.vertex_colors

        if include_attributes and hasattr(mesh, "vertex_attributes"):
            _add_attributes_to_data_array(vertex, mesh.vertex_attributes)
    else:
        num_vertices = 0

    header_params = {"vertex_count": num_vertices, "encoding": encoding}

    if hasattr(mesh, "faces"):
        header.append(templates["face"])
        if mesh.visual.kind == "face" and encoding != "ascii":
            header.append(templates["color"])
            dtype_face.append(dtype_color)

        if include_attributes and hasattr(mesh, "face_attributes"):
            _add_attributes_to_header(header, mesh.face_attributes)
            _add_attributes_to_dtype(dtype_face, mesh.face_attributes)

        # put mesh face data into custom dtype to export
        faces = np.zeros(len(mesh.faces), dtype=dtype_face)
        faces["count"] = 3
        faces["index"] = mesh.faces
        if mesh.visual.kind == "face" and encoding != "ascii":
            faces["rgba"] = mesh.visual.face_colors
        header_params["face_count"] = len(mesh.faces)

        if include_attributes and hasattr(mesh, "face_attributes"):
            _add_attributes_to_data_array(faces, mesh.face_attributes)

    header.append(templates["outro"])
    export = Template("".join(header)).substitute(header_params).encode("utf-8")

    if encoding == "binary_little_endian":
        if hasattr(mesh, "vertices"):
            export += vertex.tobytes()
        if hasattr(mesh, "faces"):
            export += faces.tobytes()
    elif encoding == "ascii":
        export_data = util.structured_array_to_string(vertex, col_delim=" ", row_delim="\n")
        if hasattr(mesh, "faces"):
            export_data += "\n"
            export_data += util.structured_array_to_string(faces, col_delim=" ", row_delim="\n")
        export += export_data.encode("utf-8")
    else:
        raise ValueError("encoding must be ascii or binary!")

    return export


def _parse_header(file_obj):
    """
    Read the ASCII header of a PLY file, and leave the file object
    at the position of the start of data but past the header.

    Parameters
    -----------
    file_obj : open file object
      Positioned at the start of the file

    Returns
    -----------
    elements : collections.OrderedDict
      Fields and data types populated
    is_ascii : bool
      Whether the data is ASCII or binary
    image_name : None or str
      File name of TextureFile
    """

    if "ply" not in str(file_obj.readline()).lower():
        raise ValueError("Not a ply file!")

    # collect the encoding: binary or ASCII
    encoding = file_obj.readline().decode("utf-8").strip().lower()
    is_ascii = "ascii" in encoding

    # big or little endian
    endian = ["<", ">"][int("big" in encoding)]
    elements = collections.OrderedDict()

    # store file name of TextureFiles in the header
    image_name = None

    while True:
        raw = file_obj.readline()
        if raw is None:
            raise ValueError("Header not terminated properly!")
        raw = raw.decode("utf-8").strip()
        line = raw.split()

        # we're done
        if "end_header" in line:
            break

        # elements are groups of properties
        if "element" in line[0]:
            # we got a new element so add it
            name, length = line[1:]
            elements[name] = {"length": int(length), "properties": collections.OrderedDict()}
        # a property is a member of an element
        elif "property" in line[0]:
            # is the property a simple single value, like:
            # `propert float x`
            if len(line) == 3:
                dtype, field = line[1:]
                elements[name]["properties"][str(field)] = endian + _dtypes[dtype]
            # is the property a painful list, like:
            # `property list uchar int vertex_indices`
            elif "list" in line[1]:
                dtype_count, dtype, field = line[2:]
                elements[name]["properties"][str(field)] = (
                    endian + _dtypes[dtype_count] + ", ($LIST,)" + endian + _dtypes[dtype]
                )
        # referenced as a file name
        elif "texturefile" in raw.lower():
            # textures come listed like:
            # `comment TextureFile fuze_uv.jpg`
            index = raw.lower().index("texturefile") + 11
            # use the value from raw to preserve whitespace
            image_name = raw[index:].strip()

    return elements, is_ascii, image_name


def _elements_to_kwargs(elements, fix_texture, image, prefer_color=None):
    """
    Given an elements data structure, extract the keyword
    arguments that a Trimesh object constructor will expect.

    Parameters
    ------------
    elements : OrderedDict object
      With fields and data loaded
    fix_texture : bool
      If True, will re- index vertices and faces
      so vertices with different UV coordinates
      are disconnected.
    image : PIL.Image
      Image to be viewed
    prefer_color : None, 'vertex', or 'face'
      Which kind of color to prefer if both defined

    Returns
    -----------
    kwargs : dict
      Keyword arguments for Trimesh constructor
    """
    # store the raw ply structure as an internal key in metadata
    kwargs = {"metadata": {"_ply_raw": elements}}

    if "vertex" in elements and elements["vertex"]["length"]:
        vertices = np.column_stack([elements["vertex"]["data"][i] for i in "xyz"])
        if not util.is_shape(vertices, (-1, 3)):
            raise ValueError("Vertices were not (n,3)!")
    else:
        # return empty geometry if there are no vertices
        kwargs["geometry"] = {}
        return kwargs

    try:
        vertex_normals = np.column_stack([elements["vertex"]["data"][j] for j in ("nx", "ny", "nz")])
        if len(vertex_normals) == len(vertices):
            kwargs["vertex_normals"] = vertex_normals
    except BaseException:
        pass

    if "face" in elements and elements["face"]["length"]:
        face_data = elements["face"]["data"]
    else:
        # some PLY files only include vertices
        face_data = None
        faces = None

    # what keys do in-the-wild exporters use for vertices
    index_names = ["vertex_index", "vertex_indices"]
    texcoord = None

    if util.is_shape(face_data, (-1, (3, 4))):
        faces = face_data
    elif isinstance(face_data, dict):
        # get vertex indexes
        for i in index_names:
            if i in face_data:
                faces = face_data[i]
                break
        # if faces have UV coordinates defined use them
        if "texcoord" in face_data:
            texcoord = face_data["texcoord"]

    elif isinstance(face_data, np.ndarray):
        face_blob = elements["face"]["data"]
        # some exporters set this name to 'vertex_index'
        # and some others use 'vertex_indices' but we really
        # don't care about the name unless there are multiple
        if len(face_blob.dtype.names) == 1:
            name = face_blob.dtype.names[0]
        elif len(face_blob.dtype.names) > 1:
            # loop through options
            for i in face_blob.dtype.names:
                if i in index_names:
                    name = i
                    break
        # get faces
        faces = face_blob[name]["f1"]

        try:
            texcoord = face_blob["texcoord"]["f1"]
        except (ValueError, KeyError):
            # accessing numpy arrays with named fields
            # incorrectly is a ValueError
            pass

    if faces is not None:
        shape = np.shape(faces)
        if len(shape) != 2:
            # we may have mixed quads and triangles handle them with function
            faces = triangulate_quads(faces)

        if texcoord is None:
            # ply has no clear definition of how texture coordinates are stored,
            # unfortunately there are many common names that we need to try
            texcoord_names = [("texture_u", "texture_v"), ("u", "v"), ("s", "t")]
            for names in texcoord_names:
                # If texture coordinates are defined with vertices
                try:
                    t_u = elements["vertex"]["data"][names[0]]
                    t_v = elements["vertex"]["data"][names[1]]
                    texcoord = np.stack((t_u[faces.reshape(-1)], t_v[faces.reshape(-1)]), axis=-1).reshape(
                        (faces.shape[0], -1)
                    )
                    # stop trying once succeeded
                    break
                except (ValueError, KeyError):
                    # if the fields didn't exist
                    pass

        shape = np.shape(faces)

        # PLY stores texture coordinates per-face which is
        # slightly annoying, as we have to then figure out
        # which vertices have the same position but different UV
        if texcoord is not None and len(shape) == 2 and texcoord.shape == (faces.shape[0], faces.shape[1] * 2):

            # vertices with the same position but different
            # UV coordinates can't be merged without it
            # looking like it went through a woodchipper
            # in- the- wild PLY comes with things merged that
            # probably shouldn't be so disconnect vertices
            if fix_texture:
                # do import here
                from ..visual.texture import unmerge_faces

                # reshape to correspond with flattened faces
                uv_all = texcoord.reshape((-1, 2))
                # UV coordinates defined for every triangle have
                # duplicates which can be merged so figure out
                # which UV coordinates are the same here
                unique, inverse = grouping.unique_rows(uv_all)

                # use the indices of faces and face textures
                # to only merge vertices where the position
                # AND uv coordinate are the same
                faces, mask_v, mask_vt = unmerge_faces(faces, inverse.reshape(faces.shape))
                # apply the mask to get resulting vertices
                vertices = vertices[mask_v]
                # apply the mask to get UV coordinates
                uv = uv_all[unique][mask_vt]
            else:
                # don't alter vertices, UV will look like crap
                # if it was exported with vertices merged
                uv = np.zeros((len(vertices), 2))
                uv[faces.reshape(-1)] = texcoord.reshape((-1, 2))

            # create the visuals object for the texture
            kwargs["visual"] = visual.texture.TextureVisuals(uv=uv, image=image)
        elif texcoord is not None:
            # create a texture with an empty material
            from ..visual.texture import TextureVisuals

            uv = np.zeros((len(vertices), 2))
            uv[faces.reshape(-1)] = texcoord.reshape((-1, 2))
            kwargs["visual"] = TextureVisuals(uv=uv)
        # faces were not none so assign them
        kwargs["faces"] = faces
    # kwargs for Trimesh or PointCloud
    kwargs["vertices"] = vertices

    # if both vertex and face color are defined pick the one
    if "face" in elements:
        kwargs["face_colors"] = _element_colors(elements["face"])
    if "vertex" in elements:
        kwargs["vertex_colors"] = _element_colors(elements["vertex"])

    return kwargs


def _element_colors(element):
    """
    Given an element, try to extract RGBA color from
    properties and return them as an (n,3|4) array.

    Parameters
    -------------
    element : dict
      Containing color keys

    Returns
    ------------
    colors : (n, 3) or (n, 4) float
      Colors extracted from the element
    signal : float
      Estimate of range
    """
    keys = ["red", "green", "blue", "alpha"]
    candidate_colors = [element["data"][i] for i in keys if i in element["properties"]]
    if len(candidate_colors) >= 3:
        return np.column_stack(candidate_colors)
    return None


def _load_element_different(properties, data):
    """
    Load elements which include lists of different lengths
    based on the element's property-definitions.

    Parameters
    ------------
    properties : dict
      Property definitions encoded in a dict where the property name is the key
      and the property data type the value.
    data : array
      Data rows for this element.
    """
    edata = {k: [] for k in properties.keys()}
    for row in data:
        start = 0
        for name, dt in properties.items():
            length = 1
            if "$LIST" in dt:
                dt = dt.split("($LIST,)")[-1]
                # the first entry in a list-property is the number of elements
                # in the list
                length = int(row[start])
                # skip the first entry (the length), when reading the data
                start += 1
            end = start + length
            edata[name].append(row[start:end].astype(dt))
            # start next property at the end of this one
            start = end

    # if the shape of any array is (n, 1) we want to
    # squeeze/concatenate it into (n,)
    squeeze = {k: np.array(v, dtype="object") for k, v in edata.items()}
    # squeeze and convert any clean 2D arrays
    squeeze.update({k: v.squeeze().astype(edata[k][0].dtype) for k, v in squeeze.items() if len(v.shape) == 2})

    return squeeze


def _load_element_single(properties, data):
    """
    Load element data with lists of a single length
    based on the element's property-definitions.

    Parameters
    ------------
    properties : dict
      Property definitions encoded in a dict where
      the property name is the key and the property
      data type the value.
    data : array
      Data rows for this element, if the data contains
      list-properties all lists belonging to one property
      must have the same length.
    """

    first = data[0]
    columns = {}
    current = 0
    for name, dt in properties.items():
        # if the current index has gone past the number
        # of items we actually have exit the loop early
        if current >= len(first):
            break
        if "$LIST" in dt:
            dtype = dt.split("($LIST,)")[-1]
            # the first entry in a list-property
            # is the number of elements in the list

            length = int(first[current])
            columns[name] = data[:, current + 1 : current + 1 + length].astype(dtype)
            # offset by length of array plus one for each uint index
            current += length + 1
        else:
            columns[name] = data[:, current : current + 1].astype(dt)
            current += 1

    return columns


def _ply_ascii(elements, file_obj):
    """
    Load data from an ASCII PLY file into an existing elements data structure.

    Parameters
    ------------
    elements : OrderedDict
      Populated from the file header, data will
      be added in-place to this object
    file_obj : file-like-object
      Current position at the start
      of the data section (past the header).
    """

    # get the file contents as a string
    text = str(file_obj.read().decode("utf-8"))
    # split by newlines
    lines = str.splitlines(text)
    # get each line as an array split by whitespace
    array = [np.fromstring(i, sep=" ") for i in lines]
    # store the line position in the file
    row_pos = 0

    # loop through data we need
    for key, values in elements.items():
        # if the element is empty ignore it
        if "length" not in values or values["length"] == 0:
            continue
        data = array[row_pos : row_pos + values["length"]]
        row_pos += values["length"]
        # try stacking the data, which simplifies column-wise access. this is only
        # possible, if all rows have the same length.
        try:
            data = np.vstack(data)
            col_count_equal = True
        except ValueError:
            col_count_equal = False

        # number of list properties in this element
        list_count = sum(1 for dt in values["properties"].values() if "$LIST" in dt)
        if col_count_equal and list_count <= 1:
            # all rows have the same length and we only have at most one list
            # property where all entries have the same length. this means we can
            # use the quick numpy-based loading.
            element_data = _load_element_single(values["properties"], data)
        else:
            # there are lists of differing lengths. we need to fall back to loading
            # the data by iterating all rows and checking for list-lengths. this is
            # slower than the variant above.
            element_data = _load_element_different(values["properties"], data)

        elements[key]["data"] = element_data


def _ply_binary(elements, file_obj):
    """
    Load the data from a binary PLY file into the elements data structure.

    Parameters
    ------------
    elements : OrderedDict
      Populated from the file header.
      Object will be modified to add data by this function.
    file_obj : open file object
      With current position at the start
      of the data section (past the header)
    """

    def populate_listsize(file_obj, elements):
        """
        Given a set of elements populated from the header if there are any
        list properties seek in the file the length of the list.

        Note that if you have a list where each instance is different length
        (if for example you mixed triangles and quads) this won't work at all
        """
        p_start = file_obj.tell()
        p_current = file_obj.tell()
        elem_pop = []
        for element_key, element in elements.items():
            props = element["properties"]
            prior_data = ""
            for k, dtype in props.items():
                prop_pop = []
                if "$LIST" in dtype:
                    # every list field has two data types:
                    # the list length (single value), and the list data (multiple)
                    # here we are only reading the single value for list length
                    field_dtype = np.dtype(dtype.split(",")[0])
                    if len(prior_data) == 0:
                        offset = 0
                    else:
                        offset = np.dtype(prior_data).itemsize
                    file_obj.seek(p_current + offset)
                    blob = file_obj.read(field_dtype.itemsize)
                    if len(blob) == 0:
                        # no data was read for property
                        prop_pop.append(k)
                        break
                    size = np.frombuffer(blob, dtype=field_dtype)[0]
                    props[k] = props[k].replace("$LIST", str(size))
                prior_data += props[k] + ","
            if len(prop_pop) > 0:
                # if a property was empty remove it
                for pop in prop_pop:
                    props.pop(pop)
                # if we've removed all properties from
                # an element remove the element later
                if len(props) == 0:
                    elem_pop.append(element_key)
                    continue
            # get the size of the items in bytes
            itemsize = np.dtype(", ".join(props.values())).itemsize
            # offset the file based on read size
            p_current += element["length"] * itemsize
        # move the file back to where we found it
        file_obj.seek(p_start)
        # if there were elements without properties remove them
        for pop in elem_pop:
            elements.pop(pop)

    def populate_data(file_obj, elements):
        """
        Given the data type and field information from the header,
        read the data and add it to a 'data' field in the element.
        """
        for key in elements.keys():
            items = list(elements[key]["properties"].items())
            dtype = np.dtype(items)
            data = file_obj.read(elements[key]["length"] * dtype.itemsize)
            try:
                elements[key]["data"] = np.frombuffer(data, dtype=dtype)
            except BaseException:
                log.warning("PLY failed to populate: {}".format(key))
                elements[key]["data"] = None
        return elements

    def _elements_size(elements):
        """
        Given an elements data structure populated from the header,
        calculate how long the file should be if it is intact.
        """
        size = 0
        for element in elements.values():
            dtype = np.dtype(",".join(element["properties"].values()))
            size += element["length"] * dtype.itemsize
        return size

    # some elements are passed where the list dimensions
    # are not included in the header, so this function goes
    # into the meat of the file and grabs the list dimensions
    # before we to the main data read as a single operation
    populate_listsize(file_obj, elements)

    # how many bytes are left in the file
    size_file = util.distance_to_end(file_obj)
    # how many bytes should the data structure described by
    # the header take up
    size_elements = _elements_size(elements)

    # if the number of bytes is not the same the file is probably corrupt
    if size_file != size_elements:
        raise ValueError("PLY is unexpected length!")

    # with everything populated and a reasonable confidence the file
    # is intact, read the data fields described by the header
    populate_data(file_obj, elements)


def export_draco(mesh, bits=28):
    """
    Export a mesh using Google's Draco compressed format.

    Only works if draco_encoder is in your PATH:
    https://github.com/google/draco

    Parameters
    ----------
    mesh : Trimesh object
      Mesh to export
    bits : int
      Bits of quantization for position
      tol.merge=1e-8 is roughly 25 bits

    Returns
    ----------
    data : str or bytes
      DRC file bytes
    """
    with tempfile.NamedTemporaryFile(suffix=".ply") as temp_ply:
        temp_ply.write(export_ply(mesh))
        temp_ply.flush()
        with tempfile.NamedTemporaryFile(suffix=".drc") as encoded:
            subprocess.check_output([draco_encoder, "-qp", str(int(bits)), "-i", temp_ply.name, "-o", encoded.name])
            encoded.seek(0)
            data = encoded.read()
    return data


def load_draco(file_obj, **kwargs):
    """
    Load a mesh from Google's Draco format.

    Parameters
    ----------
    file_obj : file- like object
      Contains data

    Returns
    ----------
    kwargs : dict
      Keyword arguments to construct a Trimesh object
    """

    with tempfile.NamedTemporaryFile(suffix=".drc") as temp_drc:
        temp_drc.write(file_obj.read())
        temp_drc.flush()

        with tempfile.NamedTemporaryFile(suffix=".ply") as temp_ply:
            subprocess.check_output([draco_decoder, "-i", temp_drc.name, "-o", temp_ply.name])
            temp_ply.seek(0)
            kwargs = load_ply(temp_ply)
    return kwargs


_ply_loaders = {"ply": load_ply}
_ply_exporters = {"ply": export_ply}

draco_encoder = util.which("draco_encoder")
draco_decoder = util.which("draco_decoder")
if draco_decoder is not None:
    _ply_loaders["drc"] = load_draco
if draco_encoder is not None:
    _ply_exporters["drc"] = export_draco
