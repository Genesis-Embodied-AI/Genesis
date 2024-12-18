"""
material.py
-------------

Store visual materials as objects.
"""

import abc
import copy
import numpy as np

from . import color
from .. import util
from .. import exceptions

from ..constants import tol

# epsilon for comparing floating point
_eps = 1e-5


class Material(util.ABC):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("must be subclassed!")

    @abc.abstractmethod
    def __hash__(self):
        raise NotImplementedError("must be subclassed!")

    @abc.abstractproperty
    def main_color(self):
        """
        The "average" color of this material.

        Returns
        ---------
        color : (4,) uint8
          Average color of this material.
        """

    @property
    def name(self):
        if hasattr(self, "_name"):
            return self._name
        return "material_0"

    @name.setter
    def name(self, value):
        self._name = value

    def copy(self):
        return copy.deepcopy(self)


class SimpleMaterial(Material):
    """
    Hold a single image texture.
    """

    def __init__(self, image=None, diffuse=None, ambient=None, specular=None, glossiness=None, **kwargs):

        # save image
        self.image = image

        # save material colors as RGBA
        self.ambient = color.to_rgba(ambient)
        self.diffuse = color.to_rgba(diffuse)
        self.specular = color.to_rgba(specular)

        # save Ns
        self.glossiness = glossiness

        # save other keyword arguments
        self.kwargs = kwargs

    def to_color(self, uv):
        return color.uv_to_color(uv, self.image)

    def to_obj(self, name=None):
        """
        Convert the current material to an OBJ format
        material.

        Parameters
        -----------
        name : str or None
          Name to apply to the material

        Returns
        -----------
        tex_name : str
          Name of material
        mtl_name : str
          Name of mtl file in files
        files : dict
          Data as {file name : bytes}
        """
        # material parameters as 0.0-1.0 RGB
        Ka = color.to_float(self.ambient)[:3]
        Kd = color.to_float(self.diffuse)[:3]
        Ks = color.to_float(self.specular)[:3]

        if name is None:
            name = self.name

        # create an MTL file
        mtl = [
            "newmtl {}".format(name),
            "Ka {:0.8f} {:0.8f} {:0.8f}".format(*Ka),
            "Kd {:0.8f} {:0.8f} {:0.8f}".format(*Kd),
            "Ks {:0.8f} {:0.8f} {:0.8f}".format(*Ks),
            "Ns {:0.8f}".format(self.glossiness),
        ]

        # collect the OBJ data into files
        data = {}

        if self.image is not None:
            image_type = self.image.format
            # what is the name of the export image to save
            if image_type is None:
                image_type = "png"
            image_name = "{}.{}".format(name, image_type.lower())
            # save the reference to the image
            mtl.append("map_Kd {}".format(image_name))

            # save the image texture as bytes in the original format
            f_obj = util.BytesIO()
            self.image.save(fp=f_obj, format=image_type)
            f_obj.seek(0)
            data[image_name] = f_obj.read()

        data["{}.mtl".format(name)] = "\n".join(mtl).encode("utf-8")

        return data, name

    def __hash__(self):
        """
        Provide a hash of the material so we can detect
        duplicates.

        Returns
        ------------
        hash : int
          Hash of image and parameters
        """
        if hasattr(self.image, "tobytes"):
            # start with hash of raw image bytes
            hashed = hash(self.image.tobytes())
        else:
            # otherwise start with zero
            hashed = 0
        # we will add additional parameters with
        # an in-place xor of the additional value
        # if stored as numpy arrays add parameters
        if hasattr(self.ambient, "tobytes"):
            hashed ^= hash(self.ambient.tobytes())
        if hasattr(self.diffuse, "tobytes"):
            hashed ^= hash(self.diffuse.tobytes())
        if hasattr(self.specular, "tobytes"):
            hashed ^= hash(self.specular.tobytes())
        if isinstance(self.glossiness, float):
            hashed ^= hash(int(self.glossiness * 1000))
        return hashed

    @property
    def main_color(self):
        """
        Return the most prominent color.
        """
        return self.diffuse

    @property
    def glossiness(self):
        if hasattr(self, "_glossiness"):
            return self._glossiness
        return 1.0

    @glossiness.setter
    def glossiness(self, value):
        if value is None:
            return
        self._glossiness = float(value)

    def to_pbr(self):
        """
        Convert the current simple material to a
        PBR material.

        Returns
        ------------
        pbr : PBRMaterial
          Contains material information in PBR format.
        """
        # convert specular exponent to roughness
        roughness = (2 / (self.glossiness + 2)) ** (1.0 / 4.0)

        return PBRMaterial(roughnessFactor=roughness, baseColorTexture=self.image, baseColorFactor=self.diffuse)


class MultiMaterial(Material):
    def __init__(self, materials=None, **kwargs):
        """
        Wrapper for a list of Materials.

        Parameters
        ----------
        materials : Optional[List[Material]]
            List of materials with which the container to be initialized.
        """
        if materials is None:
            self.materials = []
        else:
            self.materials = materials

    def to_pbr(self):
        """
        TODO : IMPLEMENT
        """
        pbr = [m for m in self.materials if isinstance(m, PBRMaterial)]
        if len(pbr) == 0:
            return PBRMaterial()
        return pbr[0]

    def __hash__(self):
        """
        Provide a hash of the multi material so we can detect
        duplicates.

        Returns
        ------------
        hash : int
          Xor hash of the contained materials.
        """
        hashed = int(np.bitwise_xor.reduce([hash(m) for m in self.materials]))

        return hashed

    def __iter__(self):
        return iter(self.materials)

    def __next__(self):
        return next(self.materials)

    def __len__(self):
        return len(self.materials)

    @property
    def main_color(self):
        """
        The "average" color of this material.

        Returns
        ---------
        color : (4,) uint8
          Average color of this material.
        """

    def add(self, material):
        """
        Adds new material to the container.

        Parameters
        ----------
        material : Material
            The material to be added.
        """
        self.materials.append(material)

    def get(self, idx):
        """
        Get material by index.

        Parameters
        ----------
        idx : int
            Index of the material to be retrieved.

        Returns
        -------
            The material on the given index.
        """
        return self.materials[idx]


class PBRMaterial(Material):
    """
    Create a material for physically based rendering as
    specified by GLTF 2.0:
    https://git.io/fhkPZ

    Parameters with `Texture` in them must be PIL.Image objects
    """

    def __init__(
        self,
        name=None,
        emissiveFactor=None,
        emissiveTexture=None,
        baseColorFactor=None,
        metallicFactor=None,
        roughnessFactor=None,
        normalTexture=None,
        occlusionTexture=None,
        baseColorTexture=None,
        metallicRoughnessTexture=None,
        doubleSided=False,
        alphaMode=None,
        alphaCutoff=None,
        **kwargs,
    ):

        # store values in an internal dict
        self._data = {}

        # (3,) float
        self.emissiveFactor = emissiveFactor
        # (3,) or (4,) float with RGBA colors
        self.baseColorFactor = baseColorFactor

        # float
        self.metallicFactor = metallicFactor
        self.roughnessFactor = roughnessFactor
        self.alphaCutoff = alphaCutoff

        # PIL image
        self.normalTexture = normalTexture
        self.emissiveTexture = emissiveTexture
        self.occlusionTexture = occlusionTexture
        self.baseColorTexture = baseColorTexture
        self.metallicRoughnessTexture = metallicRoughnessTexture

        # bool
        self.doubleSided = doubleSided

        # str
        self.name = name
        self.alphaMode = alphaMode

        if len(kwargs) > 0:
            util.log.debug("unsupported material keys: {}".format(", ".join(kwargs.keys())))

    @property
    def emissiveFactor(self):
        """
        The factors for the emissive color of the material.
        This value defines linear multipliers for the sampled
        texels of the emissive texture.

        Returns
        -----------
        emissiveFactor : (3,) float
           Ech element in the array MUST be greater than
           or equal to 0 and less than or equal to 1.
        """
        return self._data.get("emissiveFactor")

    @emissiveFactor.setter
    def emissiveFactor(self, value):
        if value is None:
            # passing none effectively removes value
            self._data.pop("emissiveFactor", None)
        else:
            # non-None values must be a floating point
            emissive = np.array(value, dtype=np.float64).reshape(3)
            if emissive.min() < -_eps or emissive.max() > (1 + _eps):
                raise ValueError("all factors must be between 0.0-1.0")
            self._data["emissiveFactor"] = emissive

    @property
    def alphaMode(self):
        """
        The material alpha rendering mode enumeration
        specifying the interpretation of the alpha value of
        the base color.

        Returns
        -----------
        alphaMode : str
          One of 'OPAQUE', 'MASK', 'BLEND'
        """
        return self._data.get("alphaMode")

    @alphaMode.setter
    def alphaMode(self, value):
        if value is None:
            # passing none effectively removes value
            self._data.pop("alphaMode", None)
        else:
            # non-None values must be one of three values
            value = str(value).upper().strip()
            if value not in ["OPAQUE", "MASK", "BLEND"]:
                raise ValueError("incorrect alphaMode: %s", value)
            self._data["alphaMode"] = value

    @property
    def alphaCutoff(self):
        """
        Specifies the cutoff threshold when in MASK alpha mode.
        If the alpha value is greater than or equal to this value
        then it is rendered as fully opaque, otherwise, it is rendered
        as fully transparent. A value greater than 1.0 will render
        the entire material as fully transparent. This value MUST be
        ignored for other alpha modes. When alphaMode is not defined,
        this value MUST NOT be defined.

        Returns
        -----------
        alphaCutoff : float
          Value of cutoff.
        """
        return self._data.get("alphaCutoff")

    @alphaCutoff.setter
    def alphaCutoff(self, value):
        if value is None:
            # passing none effectively removes value
            self._data.pop("alphaCutoff", None)
        else:
            self._data["alphaCutoff"] = float(value)

    @property
    def doubleSided(self):
        """
        Specifies whether the material is double sided.

        Returns
        -----------
        doubleSided : bool
          Specifies whether the material is double sided.
        """
        return self._data.get("doubleSided")

    @doubleSided.setter
    def doubleSided(self, value):
        if value is None:
            # passing none effectively removes value
            self._data.pop("doubleSided", None)
        else:
            self._data["doubleSided"] = bool(value)

    @property
    def metallicFactor(self):
        """
        The factor for the metalness of the material. This value
        defines a linear multiplier for the sampled metalness values
        of the metallic-roughness texture.


        Returns
        -----------
        metallicFactor : float
          How metally is the material
        """
        return self._data.get("metallicFactor")

    @metallicFactor.setter
    def metallicFactor(self, value):
        if value is None:
            # passing none effectively removes value
            self._data.pop("metallicFactor", None)
        else:
            self._data["metallicFactor"] = float(value)

    @property
    def roughnessFactor(self):
        """
        The factor for the roughness of the material. This value
        defines a linear multiplier for the sampled roughness values
        of the metallic-roughness texture.

        Returns
        -----------
        roughnessFactor : float
          Roughness of material.
        """
        return self._data.get("roughnessFactor")

    @roughnessFactor.setter
    def roughnessFactor(self, value):
        if value is None:
            # passing none effectively removes value
            self._data.pop("roughnessFactor", None)
        else:
            self._data["roughnessFactor"] = float(value)

    @property
    def baseColorFactor(self):
        """
        The factors for the base color of the material. This
        value defines linear multipliers for the sampled texels
        of the base color texture.

        Returns
        ---------
        color : (4,) uint8
          RGBA color
        """
        return self._data.get("baseColorFactor")

    @baseColorFactor.setter
    def baseColorFactor(self, value):
        if value is None:
            # passing none effectively removes value
            self._data.pop("baseColorFactor", None)
        else:
            # non-None values must be RGBA color
            self._data["baseColorFactor"] = color.to_rgba(value)

    @property
    def normalTexture(self):
        """
        The normal map texture.

        Returns
        ----------
        image : PIL.Image
          Normal texture.
        """
        return self._data.get("normalTexture")

    @normalTexture.setter
    def normalTexture(self, value):
        if value is None:
            # passing none effectively removes value
            self._data.pop("normalTexture", None)
        else:
            self._data["normalTexture"] = value

    @property
    def emissiveTexture(self):
        """
        The emissive texture.

        Returns
        ----------
        image : PIL.Image
          Emissive texture.
        """
        return self._data.get("emissiveTexture")

    @emissiveTexture.setter
    def emissiveTexture(self, value):
        if value is None:
            # passing none effectively removes value
            self._data.pop("emissiveTexture", None)
        else:
            self._data["emissiveTexture"] = value

    @property
    def occlusionTexture(self):
        """
        The occlusion texture.

        Returns
        ----------
        image : PIL.Image
          Occlusion texture.
        """
        return self._data.get("occlusionTexture")

    @occlusionTexture.setter
    def occlusionTexture(self, value):
        if value is None:
            # passing none effectively removes value
            self._data.pop("occlusionTexture", None)
        else:
            self._data["occlusionTexture"] = value

    @property
    def baseColorTexture(self):
        """
        The base color texture image.

        Returns
        ----------
        image : PIL.Image
          Color texture.
        """
        return self._data.get("baseColorTexture")

    @baseColorTexture.setter
    def baseColorTexture(self, value):
        if value is None:
            # passing none effectively removes value
            self._data.pop("baseColorTexture", None)
        else:
            # non-None values must be RGBA color
            self._data["baseColorTexture"] = value

    @property
    def metallicRoughnessTexture(self):
        """
        The metallic-roughness texture.

        Returns
        ----------
        image : PIL.Image
          Metallic-roughness texture.
        """
        return self._data.get("metallicRoughnessTexture")

    @metallicRoughnessTexture.setter
    def metallicRoughnessTexture(self, value):
        if value is None:
            # passing none effectively removes value
            self._data.pop("metallicRoughnessTexture", None)
        else:
            self._data["metallicRoughnessTexture"] = value

    @property
    def name(self):
        return self._data.get("name")

    @name.setter
    def name(self, value):
        if value is None:
            # passing none effectively removes value
            self._data.pop("name", None)
        else:
            self._data["name"] = value

    def copy(self):
        # doing a straight deepcopy fails due to PIL images
        kwargs = {}
        # collect stored values as kwargs
        for k, v in self._data.items():
            if v is None:
                continue
            if hasattr(v, "copy"):
                # use an objects explicit copy if available
                kwargs[k] = v.copy()
            else:
                # otherwise just hope deepcopy does something
                kwargs[k] = copy.deepcopy(v)
        return PBRMaterial(**kwargs)

    def to_color(self, uv):
        """
        Get the rough color at a list of specified UV
        coordinates.

        Parameters
        -------------
        uv : (n, 2) float
          UV coordinates on the material

        Returns
        -------------
        colors
        """
        colors = color.uv_to_color(uv=uv, image=self.baseColorTexture)
        if colors is None and self.baseColorFactor is not None:
            colors = self.baseColorFactor.copy()
        return colors

    def to_simple(self):
        """
        Get a copy of the current PBR material as
        a simple material.

        Returns
        ------------
        simple : SimpleMaterial
          Contains material information in a simple manner
        """

        return SimpleMaterial(image=self.baseColorTexture, diffuse=self.baseColorFactor)

    @property
    def main_color(self):
        # will return default color if None
        result = color.to_rgba(self.baseColorFactor)
        return result

    def __hash__(self):
        """
        Provide a hash of the material so we can detect
        duplicate materials.

        Returns
        ------------
        hash : int
          Hash of image and parameters
        """
        return hash(bytes().join(np.asanyarray(v).tobytes() for v in self._data.values() if v is not None))


def empty_material(color=None):
    """
    Return an empty material set to a single color

    Parameters
    -----------
    color : None or (3,) uint8
      RGB color

    Returns
    -------------
    material : SimpleMaterial
      Image is a a one pixel RGB
    """
    try:
        from PIL import Image
    except BaseException as E:
        return exceptions.ExceptionWrapper(E)

    final = np.array([255, 255, 255, 255], dtype=np.uint8)
    if np.shape(color) in ((3,), (4,)):
        final[: len(color)] = color

    # create a one pixel RGB image
    image = Image.fromarray(final.reshape((1, 1, 4)).astype(np.uint8))
    return SimpleMaterial(image=image)


def pack(materials, uvs, deduplicate=True):
    """
    Pack multiple materials with texture into a single material.

    UV coordinates outside of the 0.0-1.0 range will be coerced
    into this range using a "wrap" behavior (i.e. modulus).

    Parameters
    -----------
    materials : (n,) Material
      List of multiple materials
    uvs : (n, m, 2) float
      Original UV coordinates

    Returns
    ------------
    material : SimpleMaterial
      Combined material.
    uv : (p, 2) float
      Combined UV coordinates in the 0.0-1.0 range.
    """

    from PIL import Image
    from ..path import packing
    import collections

    def material_to_img(mat):
        """
        Logic for extracting a simple image from each material.
        """
        # extract an image for each material
        img = None
        if isinstance(mat, PBRMaterial):
            if mat.baseColorTexture is not None:
                img = mat.baseColorTexture
            elif mat.baseColorFactor is not None:
                c = color.to_rgba(mat.baseColorFactor)
                assert c.shape == (4,)
                assert c.dtype == np.uint8
                img = Image.fromarray(c.reshape((1, 1, -1)))
        elif getattr(mat, "image", None) is not None:
            img = mat.image
        elif np.shape(getattr(mat, "diffuse", [])) == (4,):
            # return a one pixel image
            img = Image.fromarray(np.reshape(color.to_rgba(mat.diffuse), (1, 1, 4)).astype(np.uint8))

        if img is None:
            # return a one pixel image
            img = Image.fromarray(np.reshape([100, 100, 100, 255], (1, 1, 4)).astype(np.uint8))
        # make sure we're always returning in RGBA mode
        return img.convert("RGBA")

    if deduplicate:
        # start by collecting a list of indexes for each material hash
        unique_idx = collections.defaultdict(list)
        [unique_idx[hash(m)].append(i) for i, m in enumerate(materials)]
        # now we only need the indexes and don't care about the hashes
        mat_idx = list(unique_idx.values())
    else:
        # otherwise just use all the indexes
        mat_idx = np.arange(len(materials)).reshape((-1, 1))

    assert set(np.concatenate(mat_idx).ravel()) == set(range(len(uvs)))
    assert len(uvs) == len(materials)

    # collect the images from the materials
    images = [material_to_img(materials[g[0]]) for g in mat_idx]

    # pack the multiple images into a single large image
    final, offsets = packing.images(images, power_resize=True)

    # the size of the final texture image
    final_size = np.array(final.size, dtype=np.float64)
    # collect scaled new UV coordinates by material index
    new_uv = {}
    for group, img, off in zip(mat_idx, images, offsets):
        # how big was the original image
        scale = img.size / final_size
        # what is the offset in fractions of final image
        xy_off = off / final_size
        # scale and translate each of the new UV coordinates
        # also make sure they are in 0.0-1.0 using modulus (i.e. wrap)
        new_uv.update({g: ((uvs[g] % 1.0) * scale) + xy_off for g in group})

    # stack the new UV coordinates in the original order
    stacked = np.vstack([new_uv[i] for i in range(len(uvs))])

    # check to make sure the packed result image matches
    # the original input image exactly in unit tests
    if tol.strict:
        # get the pixel color from the original image
        check = []
        for uv, mat in zip(uvs, materials):
            # get the image from the material and whether or not
            # it had to fill in with default dataa
            img = material_to_img(mat)
            current = color.uv_to_color(image=img, uv=(uv % 1))
            check.append(current)

        check_flat = np.vstack(check)
        # get the pixel color from the packed image
        compare = color.uv_to_color(uv=stacked, image=final)
        # should be exactly identical
        # note this is only true for simple colors
        # interpolation on complicated stuff can break this
        assert (compare == check_flat).all()

    return SimpleMaterial(image=final), stacked
