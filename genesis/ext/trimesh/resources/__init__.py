import os
import json

from ..util import decode_text, wrap_as_stream

# find the current absolute path to this directory
_pwd = os.path.expanduser(os.path.abspath(os.path.dirname(__file__)))

# once resources are loaded cache them
_cache = {}


def get(name, decode=True, decode_json=False, as_stream=False):
    """
    Get a resource from the `trimesh/resources` folder.

    Parameters
    -------------
    name : str
      File path relative to `trimesh/resources`
    decode : bool
      Whether or not to decode result as UTF-8
    decode_json : bool
      Run `json.loads` on resource if True.
    as_stream : bool
      Return as a file-like object

    Returns
    -------------
    resource : str, bytes, or decoded JSON
      File data
    """
    # key by name and decode
    cache_key = (name, bool(decode), bool(decode_json), bool(as_stream))
    cached = _cache.get(cache_key)
    if hasattr(cached, "seek"):
        cached.seek(0)
    if cached is not None:
        return cached

    # get the resource using relative names
    with open(os.path.join(_pwd, name), "rb") as f:
        resource = f.read()

    # make sure we return it as a string if asked
    if decode:
        # will decode into text if possibly
        resource = decode_text(resource)

    if decode_json:
        resource = json.loads(resource)
    elif as_stream:
        resource = wrap_as_stream(resource)

    # store for later access
    _cache[cache_key] = resource

    return resource


def get_schema(name):
    """
    Load a schema and evaluate the referenced files.

    Parameters
    ------------
    name : str
      Filename of schema.

    Returns
    ----------
    schema : dict
      Loaded and resolved schema.
    """
    from ..schemas import resolve
    from ..resolvers import FilePathResolver

    # get a resolver for our base path
    resolver = FilePathResolver(os.path.join(_pwd, "schema", name))
    # recursively load $ref keys
    schema = resolve(json.loads(decode_text(resolver.get(name))), resolver=resolver)
    return schema
