"""
xaml.py
---------

Load 3D XAMl files, an export option from Solidworks.
"""

import numpy as np

import collections

from .. import util
from .. import visual
from .. import transformations as tf


def load_XAML(file_obj, *args, **kwargs):
    """
    Load a 3D XAML file.

    Parameters
    ----------
    file_obj : file object
      Open XAML file.

    Returns
    ----------
    result : dict
      Kwargs for a Trimesh constructor.
    """

    def element_to_color(element):
        """
        Turn an XML element into a (4,) np.uint8 RGBA color
        """
        if element is None:
            return visual.DEFAULT_COLOR
        hexcolor = int(element.attrib["Color"].replace("#", ""), 16)
        opacity = float(element.attrib["Opacity"])
        rgba = [(hexcolor >> 16) & 0xFF, (hexcolor >> 8) & 0xFF, (hexcolor & 0xFF), opacity * 0xFF]
        rgba = np.array(rgba, dtype=np.uint8)
        return rgba

    def element_to_transform(element):
        """
        Turn an XML element into a (4,4) np.float64
        transformation matrix.
        """
        try:
            matrix = next(element.iter(tag=ns + "MatrixTransform3D")).attrib["Matrix"]
            matrix = np.array(matrix.split(), dtype=np.float64).reshape((4, 4)).T
            return matrix
        except StopIteration:
            # this will be raised if the MatrixTransform3D isn't in the passed
            # elements tree
            return np.eye(4)

    # read the file and parse XML
    file_data = file_obj.read()
    root = etree.XML(file_data)

    # the XML namespace
    ns = root.tag.split("}")[0] + "}"

    # the linked lists our results are going in
    vertices = []
    faces = []
    colors = []
    normals = []

    # iterate through the element tree
    # the GeometryModel3D tag contains a material and geometry
    for geometry in root.iter(tag=ns + "GeometryModel3D"):

        # get the diffuse and specular colors specified in the material
        color_search = ".//{ns}{color}Material/*/{ns}SolidColorBrush"
        diffuse = geometry.find(color_search.format(ns=ns, color="Diffuse"))
        specular = geometry.find(color_search.format(ns=ns, color="Specular"))

        # convert the element into a (4,) np.uint8 RGBA color
        diffuse = element_to_color(diffuse)
        specular = element_to_color(specular)

        # to get the final transform of a component we'll have to traverse
        # all the way back to the root node and save transforms we find
        current = geometry
        transforms = collections.deque()
        # when the root node is reached its parent will be None and we stop
        while current is not None:
            # element.find will only return elements that are direct children
            # of the current element as opposed to element.iter,
            # which will return any depth of child
            transform_element = current.find(ns + "ModelVisual3D.Transform")
            if transform_element is not None:
                # we are traversing the tree backwards, so append new
                # transforms to the left of the deque
                transforms.appendleft(element_to_transform(transform_element))
            # we are going from the lowest level of the tree to the highest
            # this avoids having to traverse any branches that don't have
            # geometry
            current = current.getparent()

        if len(transforms) == 0:
            # no transforms in the tree mean an identity matrix
            transform = np.eye(4)
        elif len(transforms) == 1:
            # one transform in the tree we can just use
            transform = transforms.pop()
        else:
            # multiple transforms we apply all of them in order
            transform = util.multi_dot(transforms)

        # iterate through the contained mesh geometry elements
        for g in geometry.iter(tag=ns + "MeshGeometry3D"):
            c_normals = np.array(g.attrib["Normals"].replace(",", " ").split(), dtype=np.float64).reshape((-1, 3))

            c_vertices = np.array(g.attrib["Positions"].replace(",", " ").split(), dtype=np.float64).reshape((-1, 3))
            # bake in the transform as we're saving
            c_vertices = tf.transform_points(c_vertices, transform)

            c_faces = np.array(g.attrib["TriangleIndices"].replace(",", " ").split(), dtype=np.int64).reshape((-1, 3))

            # save data to a sequence
            vertices.append(c_vertices)
            faces.append(c_faces)
            colors.append(np.tile(diffuse, (len(c_faces), 1)))
            normals.append(c_normals)

    # compile the results into clean numpy arrays
    result = dict()
    result["vertices"], result["faces"] = util.append_faces(vertices, faces)
    result["face_colors"] = np.vstack(colors)
    result["vertex_normals"] = np.vstack(normals)

    return result


try:
    from lxml import etree

    _xaml_loaders = {"xaml": load_XAML}
except BaseException as E:
    # create a dummy module which will raise the ImportError
    # or other exception only when someone tries to use networkx
    from ..exceptions import ExceptionWrapper

    _xaml_loaders = {"xaml": ExceptionWrapper(E)}
