import copy

import numpy as np

from .base import Visuals
from . import color

from .. import util
from .. import caching
from .. import grouping

from .material import SimpleMaterial, PBRMaterial, empty_material  # NOQA


class TextureVisuals(Visuals):
    def __init__(self, uv=None, material=None, image=None, face_materials=None):
        """
        Store a single material and per-vertex UV coordinates
        for a mesh.

        If passed UV coordinates and a single image it will
        create a SimpleMaterial for the image.

        Parameters
        --------------
        uv : (n, 2) float
          UV coordinates for the mesh
        material : Material
          Store images and properties
        image : PIL.Image
          Can be passed to automatically create material
        """

        # store values we care about enough to hash
        self.vertex_attributes = caching.DataStore()
        # cache calculated values
        self._cache = caching.Cache(self.vertex_attributes.__hash__)

        # should be (n, 2) float
        self.uv = uv

        if material is None:
            if image is None:
                self.material = empty_material()
            else:
                # if an image is passed create a SimpleMaterial
                self.material = SimpleMaterial(image=image)
        else:
            # if passed assign
            self.material = material

        self.face_materials = face_materials

    def _verify_hash(self):
        """
        Dump the cache if anything in self.vertex_attributes
        has changed.
        """
        self._cache.verify()

    @property
    def kind(self):
        """
        Return the type of visual data stored

        Returns
        ----------
        kind : str
          What type of visuals are defined
        """
        return "texture"

    @property
    def defined(self):
        """
        Check if any data is stored

        Returns
        ----------
        defined : bool
          Are UV coordinates and images set?
        """
        ok = self.material is not None
        return ok

    def __hash__(self):
        """
        Get a CRC of the stored data.

        Returns
        --------------
        crc : int
          Hash of items in self.vertex_attributes
        """
        return self.vertex_attributes.__hash__()

    @property
    def uv(self):
        """
        Get the stored UV coordinates.

        Returns
        ------------
        uv : (n, 2) float or None
          Pixel position per-vertex.
        """
        return self.vertex_attributes.get("uv", None)

    @uv.setter
    def uv(self, values):
        """
        Set the UV coordinates.

        Parameters
        --------------
        values : (n, 2) float or None
          Pixel locations on a texture per- vertex
        """
        if values is None:
            self.vertex_attributes.pop("uv")
        else:
            self.vertex_attributes["uv"] = np.asanyarray(values, dtype=np.float64)

    def copy(self, uv=None):
        """
        Return a copy of the current TextureVisuals object.

        Returns
        ----------
        copied : TextureVisuals
          Contains the same information in a new object
        """
        if uv is None:
            uv = self.uv
        if uv is not None:
            uv = uv.copy()
        copied = TextureVisuals(uv=uv, material=self.material.copy(), face_materials=copy.copy(self.face_materials))

        return copied

    def to_color(self):
        """
        Convert textured visuals to a ColorVisuals with vertex
        color calculated from texture.

        Returns
        -----------
        vis : trimesh.visuals.ColorVisuals
          Contains vertex color from texture
        """
        # find the color at each UV coordinate
        colors = self.material.to_color(self.uv)
        # create ColorVisuals from result
        vis = color.ColorVisuals(vertex_colors=colors)
        return vis

    def face_subset(self, face_index):
        """
        Get a copy of
        """
        if self.uv is not None:
            indices = np.unique(self.mesh.faces[face_index].flatten())
            return self.copy(self.uv[indices])
        else:
            return self.copy()

    def update_vertices(self, mask):
        """
        Apply a mask to remove or duplicate vertex properties.

        Parameters
        ------------
        mask : (len(vertices),) bool or (n,) int
          Mask which can be used like: `vertex_attribute[mask]`
        """
        # collect updated masked values
        updates = {}
        for key, value in self.vertex_attributes.items():
            # DataStore will convert None to zero-length array
            if len(value) == 0:
                continue
            try:
                # store the update
                updates[key] = value[mask]
            except BaseException:
                # usual reason is an incorrect size or index
                util.log.warning("failed to update visual: `{}`".format(key))
        # clear all values from the vertex attributes
        self.vertex_attributes.clear()
        # apply the updated values
        self.vertex_attributes.update(updates)

    def update_faces(self, mask):
        """
        Apply a mask to remove or duplicate face properties,
        not applicable to texture visuals.
        """

    def concatenate(self, others):
        """
        Concatenate this TextureVisuals object with others
        and return the result without modifying this visual.

        Parameters
        -----------
        others : (n,) Visuals
          Other visual objects to concatenate

        Returns
        -----------
        concatenated : TextureVisuals
          Concatenated visual objects
        """
        from .objects import concatenate

        return concatenate(self, others)


def unmerge_faces(faces, *args, **kwargs):
    """
    Textured meshes can come with faces referencing vertex
    indices (`v`) and an array the same shape which references
    vertex texture indices (`vt`) and sometimes even normal (`vn`).

    Vertex locations with different values of any of these can't
    be considered the "same" vertex, and for our simple data
    model we need to not combine these vertices.

    Parameters
    -------------
    faces : (n, d) int
      References vertex indices
    *args : (n, d) int
      Various references of corresponding values
      This is usually UV coordinates or normal indexes
    maintain_faces : bool
      Do not alter original faces and return no-op masks.

    Returns
    -------------
    new_faces : (m, d) int
      New faces for masked vertices
    mask_v : (p,) int
      A mask to apply to vertices
    mask_* : (p,) int
      A mask to apply to vt array to get matching UV coordinates
      Returns as many of these as args were passed
    """
    # unfortunately Python2 doesn't let us put named kwargs
    # after an `*args` sequence so we have to do this ugly get
    maintain_faces = kwargs.get("maintain_faces", False)

    # don't alter faces
    if maintain_faces:
        # start with not altering faces at all
        result = [faces]
        # find the maximum index referenced by faces
        max_idx = faces.max()
        # add a vertex mask which is just ordered
        result.append(np.arange(max_idx + 1))

        # now given the order is fixed do our best on the rest of the order
        for arg in args:
            # create a mask of the attribute-vertex mapping
            # note that these might conflict since we're not unmerging
            masks = np.zeros((3, max_idx + 1), dtype=np.int64)
            # set the mask using the unmodified face indexes
            for i, f, a in zip(range(3), faces.T, arg.T):
                masks[i][f] = a
            # find the most commonly occurring attribute (i.e. UV coordinate)
            # and use that index note that this is doing a float conversion
            # and then median before converting back to int: could also do this as
            # a column diff and sort but this seemed easier and is fast enough
            result.append(np.median(masks, axis=0).astype(np.int64))

        return result

    # stack into pairs of (vertex index, texture index)
    stackable = [np.asanyarray(faces).reshape(-1)]
    # append multiple args to the correlated stack
    # this is usually UV coordinates (vt) and normals (vn)
    for arg in args:
        stackable.append(np.asanyarray(arg).reshape(-1))

    # unify them into rows of a numpy array
    stack = np.column_stack(stackable)
    # find unique pairs: we're trying to avoid merging
    # vertices that have the same position but different
    # texture coordinates
    unique, inverse = grouping.unique_rows(stack)

    # only take the unique pairs
    pairs = stack[unique]
    # try to maintain original vertex order
    order = pairs[:, 0].argsort()
    # apply the order to the pairs
    pairs = pairs[order]

    # we re-ordered the vertices to try to maintain
    # the original vertex order as much as possible
    # so to reconstruct the faces we need to remap
    remap = np.zeros(len(order), dtype=np.int64)
    remap[order] = np.arange(len(order))

    # the faces are just the inverse with the new order
    new_faces = remap[inverse].reshape((-1, faces.shape[1]))

    # the mask for vertices and masks for other args
    result = [new_faces]
    result.extend(pairs.T)

    return result


def power_resize(image, resample=1, square=False):
    """
    Resize a PIL image so every dimension is a power of two.

    Parameters
    ------------
    image : PIL.Image
      Input image
    resample : int
      Passed to Image.resize
    square : bool
      If True, upsize to a square image

    Returns
    -------------
    resized : PIL.Image
      Input image resized
    """
    # what is the current resolution of the image in pixels
    size = np.array(image.size, dtype=np.int64)
    # what is the resolution of the image upsized to the nearest
    # power of two on each axis: allow rectangular textures
    new_size = (2 ** np.ceil(np.log2(size))).astype(np.int64)

    # make every dimension the largest
    if square:
        new_size = np.ones(2, dtype=np.int64) * new_size.max()

    # if we're not powers of two upsize
    if (size != new_size).any():
        return image.resize(new_size, resample=resample)

    return image.copy()
