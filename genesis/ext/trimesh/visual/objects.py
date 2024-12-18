"""
objects.py
--------------

Deal with objects which hold visual properties, like
ColorVisuals and TextureVisuals.
"""

import numpy as np

from .material import pack
from .texture import TextureVisuals
from .color import ColorVisuals, color_to_uv


def create_visual(**kwargs):
    """
    Create Visuals object from keyword arguments.

    Parameters
    -----------
    face_colors : (n, 3|4) uint8
      Face colors
    vertex_colors : (n, 3|4) uint8
      Vertex colors
    mesh : trimesh.Trimesh
      Mesh object

    Returns
    ----------
    visuals : ColorVisuals
      Visual object created from arguments
    """
    return ColorVisuals(**kwargs)


def concatenate(visuals, *args):
    """
    Concatenate multiple visual objects.

    Parameters
    ----------
    visuals : ColorVisuals or list
      Visuals to concatenate
    *args :  ColorVisuals or list
      More visuals to concatenate

    Returns
    ----------
    concat : Visuals
      If all are color
    """
    # get a flat list of Visuals objects
    if len(args) > 0:
        visuals = np.append(visuals, args)
    else:
        visuals = np.array(visuals)

    # if there are any texture visuals convert all to texture
    if any(v.kind == "texture" for v in visuals):
        # first collect materials and UV coordinates
        mats = []
        uvs = []
        for v in visuals:
            if v.kind == "texture":
                mats.append(v.material)
                if v.uv is None:
                    # otherwise use zeros
                    uvs.append(np.zeros((len(v.mesh.vertices), 2)) + 0.5)
                else:
                    # if uvs are of correct shape use them
                    uvs.append(v.uv)

            else:
                # create a material and UV coordinates from vertex colors
                color_mat, color_uv = color_to_uv(vertex_colors=v.vertex_colors)
                mats.append(color_mat)
                uvs.append(color_uv)
        # pack the materials and UV coordinates into one
        new_mat, new_uv = pack(materials=mats, uvs=uvs)
        return TextureVisuals(material=new_mat, uv=new_uv)

    # convert all visuals to the kind of the first
    kind = visuals[0].kind
    if kind == "face":
        colors = np.vstack([v.face_colors for v in visuals])
        return ColorVisuals(face_colors=colors)
    elif kind == "vertex":
        colors = np.vstack([v.vertex_colors for v in visuals])
        return ColorVisuals(vertex_colors=colors)

    return ColorVisuals()
