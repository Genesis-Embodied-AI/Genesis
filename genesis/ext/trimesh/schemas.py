"""
schemas.py
-------------

Tools for dealing with schemas, particularly JSONschema
"""

import json

from .util import decode_text


def resolve(item, resolver):
    """
    Given a JSON Schema containing `$ref` keys recursively
    evaluate to find and replace referenced files with their
    actual values using trimesh.resolvers.Resolver objects.

    Parameters
    ---------------
    item : any
      JSON schema including `$ref` to other files
    resolver : trimesh.visual.resolver.Resolver
      Resolver to fetch referenced assets

    Returns
    ----------
    result : any
      JSONSchema with references replaced
    """
    if isinstance(item, list):
        # run the resolver on every list item
        return [resolve(i, resolver) for i in item]
    elif isinstance(item, dict):
        if "$ref" in item:
            # if we have a reference to a file pop the key
            # and update the dict with the reference in-place
            raw = decode_text(resolver.get(item.pop("$ref")))
            item.update(json.loads(raw))
            # run the resolver on the dict again
            resolve(item, resolver)
        else:
            # make sure all keys are evaluated
            for i in item.values():
                resolve(i, resolver)
    return item
