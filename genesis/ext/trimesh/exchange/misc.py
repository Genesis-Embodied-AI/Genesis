import json

from .. import util


def load_dict(data, **kwargs):
    """
    Load multiple input types into kwargs for a Trimesh constructor.
    Tries to extract keys:
    'faces'
    'vertices'
    'face_normals'
    'vertex_normals'

    Parameters
    ----------
    data : dict
    accepts multiple forms
          -dict: has keys for vertices and faces as (n,3) numpy arrays
          -dict: has keys for vertices/faces (n,3) arrays encoded as dicts/base64
                 with trimesh.util.array_to_encoded/trimesh.util.encoded_to_array
          -str:  json blob as dict with either straight array or base64 values
          -file object: json blob of dict
    file_type: not used

    Returns
    -----------
    loaded: dict with keys
            -vertices: (n,3) float
            -faces:    (n,3) int
            -face_normals: (n,3) float (optional)
    """
    if data is None:
        raise ValueError("data passed to load_dict was None!")
    if util.is_instance_named(data, "Trimesh"):
        return data
    if util.is_string(data):
        if "{" not in data:
            raise ValueError("Object is not a JSON encoded dictionary!")
        data = json.loads(data.decode("utf-8"))
    elif util.is_file(data):
        data = json.load(data)

    # what shape should the data be to be usable
    mesh_data = {
        "vertices": (-1, 3),
        "faces": (-1, (3, 4)),
        "face_normals": (-1, 3),
        "face_colors": (-1, (3, 4)),
        "vertex_normals": (-1, 3),
        "vertex_colors": (-1, (3, 4)),
    }

    # now go through data structure and if anything is encoded as base64
    # pull it back into numpy arrays
    if isinstance(data, dict):
        loaded = {}
        data = util.decode_keys(data, "utf-8")
        for key, shape in mesh_data.items():
            if key in data:
                loaded[key] = util.encoded_to_array(data[key])
                if not util.is_shape(loaded[key], shape):
                    raise ValueError("Shape of %s is %s, not %s!", key, str(loaded[key].shape), str(shape))
        if len(key) == 0:
            raise ValueError("Unable to extract any mesh data!")
        return loaded
    else:
        raise ValueError("%s object passed to dict loader!", data.__class__.__name__)


def load_meshio(file_obj, file_type=None, **kwargs):
    """
    Load a meshio-supported file into the kwargs for a Trimesh
    constructor.


    Parameters
    ----------
    file_obj : file object
      Contains a meshio file
    file_type : str
      File extension, aka 'vtk'

    Returns
    ----------
    loaded : dict
      kwargs for Trimesh constructor
    """
    # trimesh "file types" are really filename extensions
    file_formats = meshio.extension_to_filetypes["." + file_type]
    # load_meshio gets passed and io.BufferedReader
    # not all readers can cope with that
    # e.g., the ones that use h5m underneath
    # in that case use the associated file name instead
    mesh = None
    for file_format in file_formats:
        try:
            mesh = meshio.read(file_obj.name, file_format=file_format)
            break
        except BaseException:
            util.log.debug("failed to load", exc_info=True)
    if mesh is None:
        raise ValueError("Failed to load file!")

    # save data as kwargs for a trimesh.Trimesh
    result = {}
    # pass kwargs to mesh constructor
    result.update(kwargs)
    # add vertices
    result["vertices"] = mesh.points
    try:
        # add faces
        result["faces"] = mesh.get_cells_type("triangle")
    except BaseException:
        util.log.warning("unable to get faces", exc_info=True)
        result["faces"] = []

    return result


_misc_loaders = {"dict": load_dict, "dict64": load_dict, "json": load_dict}

try:
    import meshio

    # add meshio loaders here
    _meshio_loaders = {k[1:]: load_meshio for k in meshio.extension_to_filetypes.keys()}
    _misc_loaders.update(_meshio_loaders)
except BaseException:
    _meshio_loaders = {}
