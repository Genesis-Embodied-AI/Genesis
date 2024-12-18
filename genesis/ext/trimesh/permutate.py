"""
permutate.py
-------------

Randomly deform meshes in different ways.
"""

import numpy as np

from . import transformations
from . import triangles as triangles_module
from . import util


def transform(mesh, translation_scale=1000.0):
    """
    Return a permutated variant of a mesh by randomly reording faces
    and rotatating + translating a mesh by a random matrix.

    Parameters
    ----------
    mesh : trimesh.Trimesh
      Mesh, will not be altered by this function

    Returns
    ----------
    permutated : trimesh.Trimesh
      Mesh with same faces as input mesh but reordered
      and rigidly transformed in space.
    """
    # rotate and translate randomly
    matrix = transformations.random_rotation_matrix(translate=translation_scale)

    # randomly re-order triangles
    triangles = np.random.permutation(mesh.triangles).reshape((-1, 3))
    # apply rigid transform
    triangles = transformations.transform_points(triangles, matrix)

    # extract the class from the input object
    mesh_type = util.type_named(mesh, "Trimesh")
    # generate a new mesh from the permutated data
    permutated = mesh_type(**triangles_module.to_kwargs(triangles.reshape((-1, 3, 3))))

    return permutated


def noise(mesh, magnitude=None):
    """
    Add gaussian noise to every vertex of a mesh, making
    no effort to maintain topology or sanity.

    Parameters
    ----------
    mesh : trimesh.Trimesh
      Input geometry, will not be altered
    magnitude : float
      What is the maximum distance per axis we can displace a vertex.
      If None, value defaults to (mesh.scale / 100.0)

    Returns
    ----------
    permutated : trimesh.Trimesh
      Input mesh with noise applied
    """
    if magnitude is None:
        magnitude = mesh.scale / 100.0

    random = (np.random.random(mesh.vertices.shape) - 0.5) * magnitude
    vertices_noise = mesh.vertices.copy() + random

    # make sure we've re- ordered faces randomly
    triangles = np.random.permutation(vertices_noise[mesh.faces])

    mesh_type = util.type_named(mesh, "Trimesh")
    permutated = mesh_type(**triangles_module.to_kwargs(triangles))

    return permutated


def tessellation(mesh):
    """
    Subdivide each face of a mesh into three faces with the new vertex
    randomly placed inside the old face.

    This produces a mesh with exactly the same surface area and volume
    but with different tessellation.

    Parameters
    ------------
    mesh : trimesh.Trimesh
      Input geometry

    Returns
    ----------
    permutated : trimesh.Trimesh
      Mesh with remeshed facets
    """
    # create random barycentric coordinates for each face
    # pad all coordinates by a small amount to bias new vertex towards center
    barycentric = np.random.random(mesh.faces.shape) + 0.05
    barycentric /= barycentric.sum(axis=1).reshape((-1, 1))

    # create one new vertex somewhere in a face
    vertex_face = (barycentric.reshape((-1, 3, 1)) * mesh.triangles).sum(axis=1)
    vertex_face_id = np.arange(len(vertex_face)) + len(mesh.vertices)

    # new vertices are the old vertices stacked on the vertices in the faces
    vertices = np.vstack((mesh.vertices, vertex_face))
    # there are three new faces per old face, and we maintain correct winding
    faces = np.vstack(
        (
            np.column_stack((mesh.faces[:, [0, 1]], vertex_face_id)),
            np.column_stack((mesh.faces[:, [1, 2]], vertex_face_id)),
            np.column_stack((mesh.faces[:, [2, 0]], vertex_face_id)),
        )
    )
    # make sure the order of the faces is permutated
    faces = np.random.permutation(faces)

    mesh_type = util.type_named(mesh, "Trimesh")
    permutated = mesh_type(vertices=vertices, faces=faces)
    return permutated


class Permutator:

    def __init__(self, mesh):
        """
        A convenience object to get permutated versions of a mesh.
        """
        self._mesh = mesh

    def transform(self, translation_scale=1000):
        return transform(self._mesh, translation_scale=translation_scale)

    def noise(self, magnitude=None):
        return noise(self._mesh, magnitude)

    def tessellation(self):
        return tessellation(self._mesh)


try:
    # copy the function docstrings to the helper object
    Permutator.noise.__doc__ = noise.__doc__
    Permutator.transform.__doc__ = transform.__doc__
    Permutator.tessellation.__doc__ = tessellation.__doc__
except AttributeError:
    # no docstrings in Python2
    pass
