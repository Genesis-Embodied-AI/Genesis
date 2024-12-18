import os
import json
import numpy as np

from .. import util
from .. import resolvers

from ..base import Trimesh
from ..parent import Geometry
from ..points import PointCloud
from ..scene.scene import Scene, append_scenes
from ..util import log, now
from ..exceptions import ExceptionWrapper

from . import misc
from .xyz import _xyz_loaders
from .ply import _ply_loaders
from .stl import _stl_loaders
from .dae import _collada_loaders
from .obj import _obj_loaders
from .off import _off_loaders
from .misc import _misc_loaders
from .gltf import _gltf_loaders
from .xaml import _xaml_loaders
from .binvox import _binvox_loaders
from .threemf import _three_loaders
from .openctm import _ctm_loaders
from .threedxml import _threedxml_loaders


try:
    from ..path.exchange.load import load_path, path_formats
except BaseException as E:
    # save a traceback to see why path didn't import
    load_path = ExceptionWrapper(E)
    # no path formats available

    def path_formats():
        return set()


def mesh_formats():
    """
    Get a list of mesh formats available to load.

    Returns
    -----------
    loaders : list
      Extensions of available mesh loaders,
      i.e. 'stl', 'ply', etc.
    """
    # filter out exceptionmodule loaders
    return set(k for k, v in mesh_loaders.items() if not isinstance(v, ExceptionWrapper))


def available_formats():
    """
    Get a list of all available loaders

    Returns
    -----------
    loaders : list
        Extensions of available loaders
        i.e. 'stl', 'ply', 'dxf', etc.
    """
    # set
    loaders = mesh_formats()
    loaders.update(path_formats())
    loaders.update(compressed_loaders.keys())

    return loaders


def load(file_obj, file_type=None, resolver=None, force=None, **kwargs):
    """
    Load a mesh or vectorized path into objects like
    Trimesh, Path2D, Path3D, Scene

    Parameters
    -----------
    file_obj : str, or file- like object
      The source of the data to be loadeded
    file_type: str
      What kind of file type do we have (eg: 'stl')
    resolver : trimesh.visual.Resolver
      Object to load referenced assets like materials and textures
    force : None or str
      For 'mesh': try to coerce scenes into a single mesh
      For 'scene': try to coerce everything into a scene
    kwargs : dict
      Passed to geometry __init__

    Returns
    ---------
    geometry : Trimesh, Path2D, Path3D, Scene
      Loaded geometry as trimesh classes
    """
    # check to see if we're trying to load something
    # that is already a native trimesh Geometry subclass
    if isinstance(file_obj, Geometry):
        log.info("Load called on %s object, returning input", file_obj.__class__.__name__)
        return file_obj

    # parse the file arguments into clean loadable form
    (
        file_obj,  # file- like object
        file_type,  # str, what kind of file
        metadata,  # dict, any metadata from file name
        opened,  # bool, did we open the file ourselves
        resolver,  # object to load referenced resources
    ) = parse_file_args(file_obj=file_obj, file_type=file_type, resolver=resolver)

    try:
        if isinstance(file_obj, dict):
            # if we've been passed a dict treat it as kwargs
            kwargs.update(file_obj)
            loaded = load_kwargs(kwargs)
        elif file_type in path_formats():
            # path formats get loaded with path loader
            loaded = load_path(file_obj, file_type=file_type, **kwargs)
        elif file_type in mesh_loaders:
            # mesh loaders use mesh loader
            loaded = load_mesh(file_obj, file_type=file_type, resolver=resolver, **kwargs)
        elif file_type in compressed_loaders:
            # for archives, like ZIP files
            loaded = load_compressed(file_obj, file_type=file_type, **kwargs)
        elif file_type in voxel_loaders:
            loaded = voxel_loaders[file_type](file_obj, file_type=file_type, resolver=resolver, **kwargs)
        else:
            if file_type in ["svg", "dxf"]:
                # call the dummy function to raise the import error
                # this prevents the exception from being super opaque
                load_path()
            else:
                raise ValueError("File type: %s not supported" % file_type)
    finally:
        # close any opened files even if we crashed out
        if opened:
            file_obj.close()

    # add load metadata ('file_name') to each loaded geometry
    for i in util.make_sequence(loaded):
        i.metadata.update(metadata)

    # if we opened the file in this function ourselves from a
    # file name clean up after ourselves by closing it
    if opened:
        file_obj.close()

    # combine a scene into a single mesh
    if force == "mesh" and isinstance(loaded, Scene):
        return util.concatenate(loaded.dump())
    if force == "scene" and not isinstance(loaded, Scene):
        return Scene(loaded)

    return loaded


def load_mesh(file_obj, file_type=None, resolver=None, **kwargs):
    """
    Load a mesh file into a Trimesh object

    Parameters
    -----------
    file_obj : str or file object
      File name or file with mesh data
    file_type : str or None
      Which file type, e.g. 'stl'
    kwargs : dict
      Passed to Trimesh constructor

    Returns
    ----------
    mesh : trimesh.Trimesh or trimesh.Scene
      Loaded geometry data
    """

    # parse the file arguments into clean loadable form
    (
        file_obj,  # file- like object
        file_type,  # str, what kind of file
        metadata,  # dict, any metadata from file name
        opened,  # bool, did we open the file ourselves
        resolver,  # object to load referenced resources
    ) = parse_file_args(file_obj=file_obj, file_type=file_type, resolver=resolver)

    try:
        # make sure we keep passed kwargs to loader
        # but also make sure loader keys override passed keys
        loader = mesh_loaders[file_type]
        tic = now()
        results = loader(file_obj, file_type=file_type, resolver=resolver, **kwargs)
        if not isinstance(results, list):
            results = [results]

        loaded = []
        for result in results:
            kwargs.update(result)
            loaded.append(load_kwargs(kwargs))
            loaded[-1].metadata.update(metadata)
        if len(loaded) == 1:
            loaded = loaded[0]
        # show the repr for loaded, loader used, and time
        log.debug("loaded {} using `{}` in {:0.4f}s".format(str(loaded), loader.__name__, now() - tic))
    finally:
        # if we failed to load close file
        if opened:
            file_obj.close()

    return loaded


def load_compressed(file_obj, file_type=None, resolver=None, mixed=False, **kwargs):
    """
    Given a compressed archive load all the geometry that
    we can from it.

    Parameters
    ----------
    file_obj : open file-like object
      Containing compressed data
    file_type : str
      Type of the archive file
    mixed : bool
      If False, for archives containing both 2D and 3D
      data will only load the 3D data into the Scene.

    Returns
    ----------
    scene : trimesh.Scene
      Geometry loaded in to a Scene object
    """

    # parse the file arguments into clean loadable form
    (
        file_obj,  # file- like object
        file_type,  # str, what kind of file
        metadata,  # dict, any metadata from file name
        opened,  # bool, did we open the file ourselves
        resolver,  # object to load referenced resources
    ) = parse_file_args(file_obj=file_obj, file_type=file_type, resolver=resolver)

    try:
        # a dict of 'name' : file-like object
        files = util.decompress(file_obj=file_obj, file_type=file_type)
        # store loaded geometries as a list
        geometries = []

        # so loaders can access textures/etc
        resolver = resolvers.ZipResolver(files)

        # try to save the files with meaningful metadata
        if "file_path" in metadata:
            archive_name = metadata["file_path"]
        else:
            archive_name = "archive"

        # populate our available formats
        if mixed:
            available = available_formats()
        else:
            # all types contained in ZIP archive
            contains = set(util.split_extension(n).lower() for n in files.keys())
            # if there are no mesh formats available
            if contains.isdisjoint(mesh_formats()):
                available = path_formats()
            else:
                available = mesh_formats()

        meta_archive = {}
        for name, data in files.items():
            try:
                # only load formats that we support
                compressed_type = util.split_extension(name).lower()

                # if file has metadata type include it
                if compressed_type in "yaml":
                    import yaml

                    meta_archive[name] = yaml.safe_load(data)
                elif compressed_type in "json":
                    import json

                    meta_archive[name] = json.loads(data)

                if compressed_type not in available:
                    # don't raise an exception, just try the next one
                    continue
                # store the file name relative to the archive
                metadata["file_name"] = archive_name + "/" + os.path.basename(name)
                # load the individual geometry
                loaded = load(file_obj=data, file_type=compressed_type, resolver=resolver, metadata=metadata, **kwargs)

                # some loaders return multiple geometries
                if util.is_sequence(loaded):
                    # if the loader has returned a list of meshes
                    geometries.extend(loaded)
                else:
                    # if the loader has returned a single geometry
                    geometries.append(loaded)
            except BaseException:
                log.debug("failed to load file in zip", exc_info=True)

    finally:
        # if we opened the file in this function
        # clean up after ourselves
        if opened:
            file_obj.close()

    # append meshes or scenes into a single Scene object
    result = append_scenes(geometries)

    # append any archive metadata files
    if isinstance(result, Scene):
        result.metadata.update(meta_archive)

    return result


def load_remote(url, **kwargs):
    """
    Load a mesh at a remote URL into a local trimesh object.

    This must be called explicitly rather than automatically
    from trimesh.load to ensure users don't accidentally make
    network requests.

    Parameters
    ------------
    url : string
      URL containing mesh file
    **kwargs : passed to `load`

    Returns
    ------------
    loaded : Trimesh, Path, Scene
      Loaded result
    """
    # import here to keep requirement soft
    import requests

    # download the mesh
    response = requests.get(url)
    # wrap as file object
    file_obj = util.wrap_as_stream(response.content)

    # so loaders can access textures/etc
    resolver = resolvers.WebResolver(url)

    try:
        # if we have a bunch of query parameters the type
        # will be wrong so try to clean up the URL
        # urllib is Python 3 only
        import urllib

        # remove the url-safe encoding then split off query params
        file_type = urllib.parse.unquote(url).split("?", 1)[0].split("/")[-1].strip()
    except BaseException:
        # otherwise just use the last chunk of URL
        file_type = url.split("/")[-1].split("?", 1)[0]

    # actually load the data from the retrieved bytes
    loaded = load(file_obj=file_obj, file_type=file_type, resolver=resolver, **kwargs)
    return loaded


def load_kwargs(*args, **kwargs):
    """
    Load geometry from a properly formatted dict or kwargs
    """

    def handle_scene():
        """
        Load a scene from our kwargs.

        class:      Scene
        geometry:   dict, name: Trimesh kwargs
        graph:      list of dict, kwargs for scene.graph.update
        base_frame: str, base frame of graph
        """
        graph = kwargs.get("graph", None)
        geometry = {k: load_kwargs(v) for k, v in kwargs["geometry"].items()}

        if graph is not None:
            scene = Scene()
            scene.geometry.update(geometry)
            for k in graph:
                if isinstance(k, dict):
                    scene.graph.update(**k)
                elif util.is_sequence(k) and len(k) == 3:
                    scene.graph.update(k[1], k[0], **k[2])
        else:
            scene = Scene(geometry)

        if "base_frame" in kwargs:
            scene.graph.base_frame = kwargs["base_frame"]
        metadata = kwargs.get("metadata")
        if isinstance(metadata, dict):
            scene.metadata.update(kwargs["metadata"])
        elif isinstance(metadata, str):
            # some ways someone might have encoded a string
            # note that these aren't evaluated until we
            # actually call the lambda in the loop
            candidates = [lambda: json.loads(metadata), lambda: json.loads(metadata.replace("'", '"'))]
            for c in candidates:
                try:
                    scene.metadata.update(c())
                    break
                except BaseException:
                    pass
        elif metadata is not None:
            log.warning("unloadable metadata")

        return scene

    def handle_mesh():
        """
        Handle the keyword arguments for a Trimesh object
        """
        # if they've been serialized as a dict
        if isinstance(kwargs["vertices"], dict) or isinstance(kwargs["faces"], dict):
            return Trimesh(**misc.load_dict(kwargs))
        # otherwise just load that puppy
        return Trimesh(**kwargs)

    def handle_export():
        """
        Handle an exported mesh.
        """
        data, file_type = kwargs["data"], kwargs["file_type"]
        if not isinstance(data, dict):
            data = util.wrap_as_stream(data)
        k = mesh_loaders[file_type](data, file_type=file_type)
        return Trimesh(**k)

    def handle_path():
        from ..path import Path2D, Path3D

        shape = np.shape(kwargs["vertices"])
        if len(shape) < 2:
            return Path2D()
        if shape[1] == 2:
            return Path2D(**kwargs)
        elif shape[1] == 3:
            return Path3D(**kwargs)
        else:
            raise ValueError("Vertices must be 2D or 3D!")

    def handle_pointcloud():
        return PointCloud(**kwargs)

    # if we've been passed a single dict instead of kwargs
    # substitute the dict for kwargs
    if len(kwargs) == 0 and len(args) == 1 and isinstance(args[0], dict):
        kwargs = args[0]

    # (function, tuple of expected keys)
    # order is important
    handlers = (
        (handle_scene, ("geometry",)),
        (handle_mesh, ("vertices", "faces")),
        (handle_path, ("entities", "vertices")),
        (handle_pointcloud, ("vertices",)),
        (handle_export, ("file_type", "data")),
    )

    # filter out keys with a value of None
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    # loop through handler functions and expected key
    for func, expected in handlers:
        if all(i in kwargs for i in expected):
            # all expected kwargs exist
            handler = func
            # exit the loop as we found one
            break
    else:
        raise ValueError("unable to determine type: {}".format(kwargs.keys()))

    return handler()


def parse_file_args(file_obj, file_type, resolver=None, **kwargs):
    """
    Given a file_obj and a file_type try to magically convert
    arguments to a file-like object and a lowercase string of
    file type.

    Parameters
    -----------
    file_obj : str
      if string represents a file path, returns:
        file_obj:   an 'rb' opened file object of the path
        file_type:  the extension from the file path

     if string is NOT a path, but has JSON-like special characters:
        file_obj:   the same string passed as file_obj
        file_type:  set to 'json'

     if string is a valid-looking URL
        file_obj: an open 'rb' file object with retrieved data
        file_type: from the extension

     if string is none of those:
        raise ValueError as we can't do anything with input

     if file like object:
        ValueError will be raised if file_type is None
        file_obj:  same as input
        file_type: same as input

     if other object: like a shapely.geometry.Polygon, etc:
        file_obj:  same as input
        file_type: if None initially, set to the class name
                    (in lower case), otherwise passed through

    file_type : str
         type of file and handled according to above

    Returns
    -----------
    file_obj : file-like object
      Contains data
    file_type : str
      Lower case of the type of file (eg 'stl', 'dae', etc)
    metadata : dict
      Any metadata gathered
    opened : bool
      Did we open the file or not
    resolver : trimesh.visual.Resolver
      Resolver to load other assets
    """
    metadata = {}
    opened = False
    if "metadata" in kwargs and isinstance(kwargs["metadata"], dict):
        metadata.update(kwargs["metadata"])

    if util.is_pathlib(file_obj):
        # convert pathlib objects to string
        file_obj = str(file_obj.absolute())

    if util.is_file(file_obj) and file_type is None:
        raise ValueError("file_type must be set for file objects!")
    if util.is_string(file_obj):
        try:
            # os.path.isfile will return False incorrectly
            # if we don't give it an absolute path
            file_path = os.path.expanduser(file_obj)
            file_path = os.path.abspath(file_path)
            exists = os.path.isfile(file_path)
        except BaseException:
            exists = False

        # file obj is a string which exists on filesystm
        if exists:
            # if not passed create a resolver to find other files
            if resolver is None:
                resolver = resolvers.FilePathResolver(file_path)
            # save the file name and path to metadata
            metadata["file_path"] = file_path
            metadata["file_name"] = os.path.basename(file_obj)
            # if file_obj is a path that exists use extension as file_type
            if file_type is None:
                file_type = util.split_extension(file_path, special=["tar.gz", "tar.bz2"])
            # actually open the file
            file_obj = open(file_path, "rb")
            opened = True
        else:
            if "{" in file_obj:
                # if a dict bracket is in the string, its probably a straight
                # JSON
                file_type = "json"
            elif "https://" in file_obj or "http://" in file_obj:
                # we've been passed a URL, warn to use explicit function
                # and don't do network calls via magical pipeline
                raise ValueError("use load_remote to load URL: {}".format(file_obj))
            elif file_type is None:
                raise ValueError("string is not a file: {}".format(file_obj))

    if file_type is None:
        file_type = file_obj.__class__.__name__

    if util.is_string(file_type) and "." in file_type:
        # if someone has passed the whole filename as the file_type
        # use the file extension as the file_type
        if "file_path" not in metadata:
            metadata["file_path"] = file_type
        metadata["file_name"] = os.path.basename(file_type)
        file_type = util.split_extension(file_type)
        if resolver is None and os.path.exists(file_type):
            resolver = resolvers.FilePathResolver(file_type)

    # all our stored extensions reference in lower case
    file_type = file_type.lower()

    # if we still have no resolver try using file_obj name
    if resolver is None and hasattr(file_obj, "name") and file_obj.name is not None and len(file_obj.name) > 0:
        resolver = resolvers.FilePathResolver(file_obj.name)

    return file_obj, file_type, metadata, opened, resolver


# loader functions for compressed extensions
compressed_loaders = {"zip": load_compressed, "tar.bz2": load_compressed, "tar.gz": load_compressed}

# map file_type to loader function
mesh_loaders = {}
mesh_loaders.update(_misc_loaders)
mesh_loaders.update(_stl_loaders)
mesh_loaders.update(_ctm_loaders)
mesh_loaders.update(_ply_loaders)
mesh_loaders.update(_obj_loaders)
mesh_loaders.update(_off_loaders)
mesh_loaders.update(_collada_loaders)
mesh_loaders.update(_gltf_loaders)
mesh_loaders.update(_xaml_loaders)
mesh_loaders.update(_threedxml_loaders)
mesh_loaders.update(_three_loaders)
mesh_loaders.update(_xyz_loaders)

# collect loaders which return voxel types
voxel_loaders = {}
voxel_loaders.update(_binvox_loaders)
