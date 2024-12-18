"""
boolean.py
-------------

Do boolean operations on meshes using either Blender or OpenSCAD.
"""

from . import interfaces


def difference(meshes, engine=None, **kwargs):
    """
    Compute the boolean difference between a mesh an n other meshes.

    Parameters
    ----------
    meshes : list of trimesh.Trimesh
      Meshes to be processed
    engine : str
      Which backend to use, i.e. 'blender' or 'scad'

    Returns
    ----------
    difference : a - (other meshes), **kwargs for a Trimesh
    """
    result = _engines[engine](meshes, operation="difference", **kwargs)
    return result


def union(meshes, engine=None, **kwargs):
    """
    Compute the boolean union between a mesh an n other meshes.

    Parameters
    ----------
    meshes : list of trimesh.Trimesh
      Meshes to be processed
    engine : str
      Which backend to use, i.e. 'blender' or 'scad'

    Returns
    ----------
    union : a + (other meshes), **kwargs for a Trimesh
    """
    result = _engines[engine](meshes, operation="union", **kwargs)
    return result


def intersection(meshes, engine=None, **kwargs):
    """
    Compute the boolean intersection between a mesh an n other meshes.

    Parameters
    ----------
    meshes : list of trimesh.Trimesh
      Meshes to be processed
    engine : str
      Which backend to use, i.e. 'blender' or 'scad'

    Returns
    ----------
    intersection : **kwargs for a Trimesh object of the
                    volume that is contained by all meshes
    """
    result = _engines[engine](meshes, operation="intersection", **kwargs)
    return result


def boolean_automatic(meshes, operation, **kwargs):
    """
    Automatically pick an engine for booleans based on availability.

    Parameters
    --------------
    meshes : list of Trimesh
      Meshes to be booleaned
    operation : str
      Type of boolean, i.e. 'union', 'intersection', 'difference'

    Returns
    ---------------
    result : trimesh.Trimesh
      Result of boolean operation
    """
    if interfaces.blender.exists:
        result = interfaces.blender.boolean(meshes, operation, **kwargs)
    elif interfaces.scad.exists:
        result = interfaces.scad.boolean(meshes, operation, **kwargs)
    else:
        raise ValueError("No backends available for boolean operations!")
    return result


# which backend boolean engines
_engines = {
    None: boolean_automatic,
    "auto": boolean_automatic,
    "scad": interfaces.scad.boolean,
    "blender": interfaces.blender.boolean,
}
